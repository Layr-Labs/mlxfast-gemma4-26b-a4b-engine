import Foundation
import MLXFastCore
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Coverage for the v1.2 BATCHED (cohort) free-run surface — request-form
// validation, the serial cohort assembly whose counters benchd's consistency
// QUADRUPLE cross-checks, the wire plumbing of the additive fields, and the
// async stream-collector bridging. Everything here is model-free: the CBv2
// engine itself only runs with real weights, so the driver's engine half is
// box-gated and everything AROUND it is pinned down here.

private let gateOn = RuntimeWorkerRequestContext(advertisesSpeculativeProtocol: true)
private let cohortOpen = RuntimeWorkerRequestContext(
    advertisesSpeculativeProtocol: true, cohortBatchSize: 8)
private let registry = RuntimeWorkerSpecRegistry.serialOnlyWorker

private func request(_ json: String) throws -> RuntimeWorkerRequest {
    try JSONDecoder().decode(RuntimeWorkerRequest.self, from: Data(json.utf8))
}

private func rejection(
    _ json: String,
    context: RuntimeWorkerRequestContext = gateOn
) -> String? {
    do {
        _ = try validateGenericWorkerRequest(
            try request(json), context: context, specRegistry: registry)
        return nil
    } catch {
        return "\(error)"
    }
}

private func accepted(
    _ json: String,
    context: RuntimeWorkerRequestContext = gateOn
) throws -> RuntimeWorkerValidatedRequest {
    try validateGenericWorkerRequest(
        try request(json), context: context, specRegistry: registry)
}

/// Representative synthetic `decode_ns_by_stream` input for the assembler
/// tests below — the assembler only validates shape (length == batchSize)
/// and packages the vector; the REAL per-slot elapsed-ns computation is
/// covered separately by the `commitTimestampNs` / collector tests further
/// down. Distinct, strictly-increasing values so a test that mis-threads
/// slot order would be caught.
private func fakeDecodeNs(_ count: Int) -> [UInt64] {
    (0..<count).map { UInt64(($0 + 1) * 1_000_000) }
}

/// A conformant batched begin at width B (benchd session field names).
private func beginJSON(batchSize: Int, streams: Int? = nil) -> String {
    let seeds = (0..<(streams ?? batchSize))
        .map { "[\($0 * 10 + 1),\($0 * 10 + 2)]" }
        .joined(separator: ",")
    return #"{"id":8,"kind":"free_decode_begin","spec":{"mode":"serial"},"#
        + #""seed_tokens_by_stream":[\#(seeds)],"batch_size":\#(batchSize)}"#
}

// MARK: - Capability constants

@Test
func cohortCapabilityMatchesBenchdAndIsAdvertisedBesideV11() {
    // Wire value matches benchd's CAPABILITY_BATCHED_FREE_RUN_DECODE, and the
    // v1.1 capability survives beside it (a batched engine still serves B=1
    // single-stream legs).
    #expect(runtimeWorkerBatchedFreeRunDecodeCapability == "batched_free_run_decode")
    #expect(runtimeWorkerMaxCohortBatchSize == 8)
    #expect(runtimeWorkerPerStreamTimingCapability == "per_stream_timing")
    #expect(
        runtimeWorkerAdvertisedCapabilities == [
            "free_run_decode", "batched_free_run_decode", "per_stream_timing",
        ])
}

/// The UNGATED advertisement pin. `cohort_reference_replay` was made ungated
/// (the verb is dispatchable on a plain worker, and benchd's (b) admission
/// oracle spawns PLAIN, so a gated cap made it fail closed and reject every
/// candidate) — but nothing pinned the resulting hello, so BOTH failure
/// directions were open: the cap silently disappearing again (benchd's oracle
/// rejects everything once more) and the speculative caps silently LEAKING onto
/// a gate-off hello (advertising verbs that are not dispatchable there, which is
/// a false acknowledgment — the same posture the cross-kind field guards take).
/// Driven through `runtimeWorkerHelloCapabilities`, the real emitter the hello
/// calls, so this cannot pass against a re-typed literal that has drifted.
@Test
func helloAdvertisesReplayUngatedAndTheSpeculativeCapsOnlyWhenGatedOn() {
    #expect(runtimeWorkerCohortReferenceReplayCapability == "cohort_reference_replay")
    // Gate OFF (no --speculative-protocol): EXACTLY the replay capability.
    #expect(
        runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol: false)
            == ["cohort_reference_replay"])
    // Gate ON: the same ungated capability PLUS the three speculative ones.
    #expect(
        runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol: true)
            == [
                "cohort_reference_replay", "free_run_decode",
                "batched_free_run_decode", "per_stream_timing",
            ])
    // Stated structurally as well as by value, so a capability added to either
    // constant later has to land on the correct side of the gate to pass: the
    // gate-on list is the ungated head followed by the advertised list, and the
    // gate-off list is that head alone.
    #expect(
        runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol: true)
            == [runtimeWorkerCohortReferenceReplayCapability]
                + runtimeWorkerAdvertisedCapabilities)
    // No speculative capability leaks onto the gate-off hello.
    let gateOff = runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol: false)
    for capability in runtimeWorkerAdvertisedCapabilities {
        #expect(!gateOff.contains(capability))
    }
}

// MARK: - Cohort begin validation

@Test
func cohortBeginAcceptsConformantForm() throws {
    let validated = try accepted(beginJSON(batchSize: 2))
    #expect(validated.cohortBegin != nil)
    #expect(validated.cohortBegin?.batchSize == 2)
    #expect(validated.cohortBegin?.seedTokensByStream == [[1, 2], [11, 12]])
    // The one cohort spec resolves exactly as the v1.1 form's does.
    #expect(validated.effectiveSpec == .serial())
}

@Test
func cohortBeginAtWidthOneIsStillTheCohortForm() throws {
    // B=1 selects the COHORT form (benchd's B=1 equivalence gate drives the
    // NEW verbs at width 1) — it must not be silently narrowed to v1.1.
    let validated = try accepted(beginJSON(batchSize: 1))
    #expect(validated.cohortBegin?.batchSize == 1)
}

@Test
func cohortBeginRequiresExplicitBatchSize() throws {
    let message = rejection(
        #"{"id":8,"kind":"free_decode_begin","seed_tokens_by_stream":[[1],[2]]}"#)
    #expect(message?.contains("explicit") == true)
    #expect(message?.contains("batch_size") == true)
}

@Test
func cohortBeginRequiresSeedStreams() throws {
    let message = rejection(
        #"{"id":8,"kind":"free_decode_begin","batch_size":2}"#)
    #expect(message?.contains("seed_tokens_by_stream") == true)
}

@Test
func cohortBeginRefusesBothSeedForms() throws {
    let message = rejection(
        #"{"id":8,"kind":"free_decode_begin","seed_tokens":[1],"#
            + #""seed_tokens_by_stream":[[1],[2]],"batch_size":2}"#)
    #expect(message?.contains("never both") == true)
}

@Test
func cohortBeginBoundsTheWidthAtTheSingleConstant() throws {
    // Width 0 and width 9 both die naming the 1...8 ceiling; 8 is accepted.
    for bad in [0, 9] {
        let message = rejection(beginJSON(batchSize: bad, streams: max(bad, 1)))
        #expect(
            message?.contains("1...\(runtimeWorkerMaxCohortBatchSize)") == true,
            "width \(bad) must die naming the ceiling")
    }
    #expect(rejection(beginJSON(batchSize: 8)) == nil)
}

@Test
func cohortBeginRefusesWidthStreamMismatch() throws {
    // B is explicit and never inferred: a declared width that disagrees with
    // the carried stream count is refused, not reconciled.
    let message = rejection(beginJSON(batchSize: 3, streams: 2))
    #expect(message?.contains("declared batch_size 3") == true)
    #expect(message?.contains("2 seed streams") == true)
}

@Test
func cohortBeginRefusesEmptyAndRaggedSeeds() throws {
    let empty = rejection(
        #"{"id":8,"kind":"free_decode_begin","#
            + #""seed_tokens_by_stream":[[1,2],[]],"batch_size":2}"#)
    #expect(empty?.contains("must not be empty") == true)
    let ragged = rejection(
        #"{"id":8,"kind":"free_decode_begin","#
            + #""seed_tokens_by_stream":[[1,2],[3]],"batch_size":2}"#)
    #expect(ragged?.contains("rectangular") == true)
}

@Test
func cohortBeginRefusedWhileCohortPhaseOpen() throws {
    let message = rejection(beginJSON(batchSize: 2), context: cohortOpen)
    #expect(message?.contains("already open") == true)
}

@Test
func cohortFieldsAreRejectedCrossKind() throws {
    // Same false-acknowledgment posture as the cross-kind spec guard.
    #expect(
        rejection(#"{"id":1,"kind":"prefill","prompt_tokens":[1],"batch_size":2}"#)?
            .contains("batch_size is valid only on") == true)
    #expect(
        rejection(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1],"#
                + #""seed_tokens_by_stream":[[1]]}"#)?
            .contains("seed_tokens_by_stream is valid only on") == true)
    #expect(
        rejection(
            #"{"id":1,"kind":"free_decode_run","count":4,"#
                + #""seed_tokens_by_stream":[[1]]}"#)?
            .contains("seed_tokens_by_stream is valid only on") == true)
}

@Test
func cohortSurfaceIsSpawnGated() throws {
    // Guard 4 gates the free-run KINDS, and with them the whole cohort form:
    // gate-off, a batched begin dies on the spawn gate before any cohort
    // validation runs.
    let gateOff = RuntimeWorkerRequestContext()
    let message = rejection(
        #"{"id":8,"kind":"free_decode_begin","#
            + #""seed_tokens_by_stream":[[1],[2]],"batch_size":2}"#,
        context: gateOff)
    #expect(message?.contains("belongs to the v1.1 speculative surface") == true)
}

@Test
func cohortBeginRefusesNonSerialSpecFailClosed() throws {
    // SERIAL path only in this increment: every non-serial mode dies at spec
    // resolution (dflash capability-absent, dspark stub, mtp not a spelling
    // this engine's vocabulary contains) — never a silent serial downgrade.
    let dflash = rejection(
        #"{"id":8,"kind":"free_decode_begin","spec":{"mode":"dflash","#
            + #""dflash":{"depth":2}},"seed_tokens_by_stream":[[1],[2]],"#
            + #""batch_size":2}"#)
    #expect(dflash != nil)
    let mtp = rejection(
        #"{"id":8,"kind":"free_decode_begin","spec":{"mode":"mtp"},"#
            + #""seed_tokens_by_stream":[[1],[2]],"batch_size":2}"#)
    #expect(mtp != nil)
}

// MARK: - Cohort run validation

@Test
func cohortRunAcceptsSealedWidthAndBoundsCount() throws {
    let validated = try accepted(
        #"{"id":9,"kind":"free_decode_run","count":128,"batch_size":8}"#,
        context: cohortOpen)
    #expect(validated.cohortRunBatchSize == 8)
    #expect(validated.freeRunCount == 128)
    // The shared v1.1 count bound applies unchanged to the cohort form.
    let over = rejection(
        #"{"id":9,"kind":"free_decode_run","count":1537,"batch_size":8}"#,
        context: cohortOpen)
    #expect(over?.contains("count must be in") == true)
}

@Test
func cohortRunBeforeCohortBeginIsRefused() throws {
    let message = rejection(
        #"{"id":9,"kind":"free_decode_run","count":4,"batch_size":8}"#)
    #expect(message?.contains("before batched") == true)
}

@Test
func cohortRunWidthMustMatchTheSealedWidth() throws {
    let message = rejection(
        #"{"id":9,"kind":"free_decode_run","count":4,"batch_size":4}"#,
        context: cohortOpen)
    #expect(message?.contains("does not match") == true)
    #expect(message?.contains("sealed") == true)
}

@Test
func v11RunIsRefusedWhileCohortPhaseOpen() throws {
    // The two windows must never cross: an open cohort refuses the
    // single-stream run form.
    let message = rejection(
        #"{"id":9,"kind":"free_decode_run","count":4}"#, context: cohortOpen)
    #expect(message?.contains("cohort phase is open") == true)
}

@Test
func v11FormsRemainByteForByteAccepted() throws {
    // The single-stream v1.1 verbs validate exactly as before when no cohort
    // field rides: same acceptance, same resolved shapes.
    let begin = try accepted(
        #"{"id":8,"kind":"free_decode_begin","seed_tokens":[1,2],"#
            + #""spec":{"mode":"serial"}}"#)
    #expect(begin.cohortBegin == nil)
    #expect(begin.effectiveSpec == .serial())
    let run = try accepted(
        #"{"id":9,"kind":"free_decode_run","count":128}"#,
        context: RuntimeWorkerRequestContext(
            hasDecodeRoute: true, advertisesSpeculativeProtocol: true))
    #expect(run.cohortRunBatchSize == nil)
    #expect(run.freeRunCount == 128)
}

// MARK: - Serial cohort assembly (the QUADRUPLE's engine-side counters)

@Test
func serialCohortAssemblyProducesTheStructuralSerialShape() throws {
    // B=3, N=4: streams collected as seed + 4 committed (+ possible overshoot,
    // trimmed below).
    let streams = [
        [100, 1, 2, 3, 4],
        [200, 5, 6, 7, 8],
        [300, 9, 10, 11, 12],
    ]
    let result = try assembleSerialCohortFreeRun(
        streamsWithSeed: streams, batchSize: 3, targetN: 4,
        decodeNsByStream: fakeDecodeNs(3))
    // The rectangle drops each slot's seed (the begin already returned it —
    // the §2.2 begin/run seam) and is exactly B x N.
    #expect(result.tokensByStream == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    // Serial regime, structurally: R == N single-width rounds, every row's
    // natural walk == the committed width, the full cohort active every round.
    #expect(result.acceptanceLengths == [1, 1, 1, 1])
    #expect(result.naturalAcceptedByStream == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    #expect(result.activeStreamsByRound == [3, 3, 3, 3])
    #expect(result.rounds == 4)
    // Cohort sums: nothing drafted, nothing accepted-from-a-draft, B*N
    // committed; completed_work is the SCALAR R+1.
    #expect(result.draftedTotal == 0)
    #expect(result.acceptedTotal == 0)
    #expect(result.committedTotal == 12)
    #expect(result.depthClampReasons.isEmpty)
    #expect(result.completedWork == 5)
    // Per-stream timing instrumentation: the assembler packages the passed
    // vector through untouched, in slot order.
    #expect(result.decodeNsByStream == fakeDecodeNs(3))
}

@Test
func serialCohortAssemblyTrimsOvershootAndRefusesShortStreams() throws {
    // A slot that raced past N before the cancel landed is trimmed to N.
    let trimmed = try assembleSerialCohortFreeRun(
        streamsWithSeed: [[100, 1, 2, 3, 4, 5, 6]], batchSize: 1, targetN: 4,
        decodeNsByStream: fakeDecodeNs(1))
    #expect(trimmed.tokensByStream == [[1, 2, 3, 4]])
    // A short slot is a broken cohort: refused with the slot named.
    #expect {
        try assembleSerialCohortFreeRun(
            streamsWithSeed: [[100, 1, 2], [200, 5, 6, 7]],
            batchSize: 2, targetN: 3, decodeNsByStream: fakeDecodeNs(2))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .streamTokenCount(slot: 0, expected: 3, got: 2)
    }
    // A wrong stream count likewise.
    #expect {
        try assembleSerialCohortFreeRun(
            streamsWithSeed: [[100, 1]], batchSize: 2, targetN: 1,
            decodeNsByStream: fakeDecodeNs(2))
    } throws: { error in
        error as? RuntimeWorkerCohortError == .streamCount(expected: 2, got: 1)
    }
}

@Test
func serialCohortAssemblyRefusesADecodeNsByStreamLengthMismatch() throws {
    // A driver wiring bug — the decode-ns vector doesn't cover the cohort
    // width — is refused rather than silently padded/truncated.
    #expect {
        try assembleSerialCohortFreeRun(
            streamsWithSeed: [[100, 1], [200, 2]], batchSize: 2, targetN: 1,
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .decodeNsByStreamCount(expected: 2, got: 1)
    }
}

@Test
func serialCohortAssemblyAtWidthOneMatchesTheV11SerialBuilder() throws {
    // benchd's B=1 equivalence gate runs the NEW verbs at width 1 and expects
    // the same observable behavior as the v1.1 route. Engine-side, the
    // cohort's per-round counters at B=1 must equal what the v1.1 serial
    // free-run builder assembles for the same committed stream — same
    // histogram, same totals, same completed_work.
    let n = 6
    let committed = Array(1...n)
    var builder = RuntimeWorkerFreeRunBuilder(targetN: n)
    for token in committed {
        builder.addRound(committedTokens: [token], drafted: 0, accepted: 0)
    }
    let v11 = try builder.finish()

    let cohort = try assembleSerialCohortFreeRun(
        streamsWithSeed: [[999] + committed], batchSize: 1, targetN: n,
        decodeNsByStream: fakeDecodeNs(1))
    #expect(cohort.tokensByStream == [v11.tokens])
    #expect(cohort.acceptanceLengths == v11.acceptanceLengths)
    #expect(cohort.draftedTotal == v11.draftedTotal)
    #expect(cohort.acceptedTotal == v11.acceptedTotal)
    #expect(cohort.committedTotal == v11.committedTotal)
    #expect(cohort.rounds == v11.rounds)
    #expect(cohort.completedWork == v11.completedWork)
}

// MARK: - MTP cohort assembly (pure — real-engine coverage is
// RuntimeWorkerMTPRoundExecutionTests.swift's `cohortMTPRunMatchesPlainDecode
// AndSatisfiesTheQuadruple`)

private func mtpMetrics(
    drafted: Int = 0, accepted: Int = 0,
    skippedRows: [String: Int] = [:], controllerFallbacks: [String: Int] = [:],
    roundAudits: [CBv2MTPRoundAuditRecord] = []
) -> CBv2MTPMetrics {
    var metrics = CBv2MTPMetrics()
    metrics.draftedTokens = drafted
    metrics.acceptedTokens = accepted
    metrics.skippedRows = skippedRows
    metrics.controllerFallbacks = controllerFallbacks
    metrics.roundAudits = roundAudits
    return metrics
}

/// One synthetic verify-round finalize record for slot `slot`, committing
/// `confirmed` tokens and ending at cumulative committed count `cumulative`
/// (seed included) over a `seedTokenCount`-token prompt.
private func mtpAudit(
    slot: Int, confirmed: Int, cumulative: Int, seedTokenCount: Int, k: Int = 2
) -> CBv2MTPRoundAuditRecord {
    CBv2MTPRoundAuditRecord(
        requestID: UInt64(slot), k: k,
        draftTokens: Array(repeating: 0, count: k),
        targetTokens: Array(repeating: 0, count: k + 1),
        accepted: confirmed - 1, confirmed: confirmed,
        rejected: (1 + k) - confirmed,
        tokensCountAfter: seedTokenCount + cumulative,
        numComputedAfter: seedTokenCount + cumulative - 1,
        generatedAfter: cumulative,
        finishReason: nil)
}

private let mtpFixtureSeedTokenCount = 4

@Test
func mtpCohortAssemblyBuildsTheCommonWidthFromSlotZeroAndTrimsOvershoot() throws {
    // B=2, N=5: slot 0's post-seed rounds are [3, 3] (sums to 6, one past N —
    // the SAME final-round overshoot trim `RuntimeWorkerFreeRunBuilder` uses).
    // Slot 1 agrees round-for-round. Every multi-token chunk is backed by its
    // engine finalize record (the reconciliation contract).
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2, 3, 4, 5, 6], chunkSizes: [1, 3, 3], finished: nil),
        (tokens: [200, 11, 12, 13, 14, 15, 16], chunkSizes: [1, 3, 3], finished: nil),
    ]
    let audits = [
        mtpAudit(slot: 0, confirmed: 3, cumulative: 4, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 1, confirmed: 3, cumulative: 4, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 0, confirmed: 3, cumulative: 7, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 1, confirmed: 3, cumulative: 7, seedTokenCount: mtpFixtureSeedTokenCount),
    ]
    let result = try assembleMTPCohortFreeRun(
        perSlot: perSlot, batchSize: 2, targetN: 5,
        seedTokenCount: mtpFixtureSeedTokenCount,
        baselineMetrics: mtpMetrics(drafted: 1, accepted: 1),
        finalMetrics: mtpMetrics(drafted: 9, accepted: 7, roundAudits: audits),
        decodeNsByStream: fakeDecodeNs(2))
    #expect(result.tokensByStream == [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
    // Round 0 committed 3, round 1 clamped to the remaining 2 (5 - 3).
    #expect(result.acceptanceLengths == [3, 2])
    #expect(result.rounds == 2)
    #expect(result.committedTotal == 10)
    // draftedTotal/acceptedTotal are the FINAL-minus-BASELINE diff, not the
    // raw final snapshot.
    #expect(result.draftedTotal == 8)
    #expect(result.acceptedTotal == 6)
    #expect(result.completedWork == 3)
    // The documented floor: every row's natural walk == the committed width.
    #expect(result.naturalAcceptedByStream == [[3, 2], [3, 2]])
    #expect(result.activeStreamsByRound == [2, 2])
    #expect(result.decodeNsByStream == fakeDecodeNs(2))
}

@Test
func mtpCohortAssemblyAcceptsLegallyOffsetRoundHistories() throws {
    // The false refusal the acceptance-rule audit exposed, in miniature:
    // slot 0 commits [plain-1, verify-3], slot 1 commits [verify-2, verify-2]
    // — same total (4), different segmentation, every multi-token chunk
    // backed by its own engine finalize record. Previously refused as
    // `mtpRoundHistoryDisagreement`; now ACCEPTED, with slot 0's profile as
    // the representative audit histogram.
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2, 3, 4], chunkSizes: [1, 1, 3], finished: nil),
        (tokens: [200, 11, 12, 13, 14], chunkSizes: [1, 2, 2], finished: nil),
    ]
    let audits = [
        mtpAudit(slot: 1, confirmed: 2, cumulative: 3, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 0, confirmed: 3, cumulative: 5, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 1, confirmed: 2, cumulative: 5, seedTokenCount: mtpFixtureSeedTokenCount),
    ]
    let result = try assembleMTPCohortFreeRun(
        perSlot: perSlot, batchSize: 2, targetN: 4,
        seedTokenCount: mtpFixtureSeedTokenCount,
        baselineMetrics: nil,
        finalMetrics: mtpMetrics(drafted: 4, accepted: 3, roundAudits: audits),
        decodeNsByStream: fakeDecodeNs(2))
    #expect(result.tokensByStream == [[1, 2, 3, 4], [11, 12, 13, 14]])
    #expect(result.acceptanceLengths == [1, 3])
    #expect(result.committedTotal == 8)
    #expect(result.completedWork == 3)
}

@Test
func mtpCohortAssemblyRefusesAMultiTokenChunkWithoutAVerifyAudit() throws {
    // Slot 1's 3-wide chunk claims a speculative commit the engine never
    // finalized — a corrupt assembly, refused. (This is also what now
    // catches the OLD disagreement fixture: an unbacked segmentation.)
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2, 3, 4], chunkSizes: [1, 2, 2], finished: nil),
        (tokens: [200, 11, 12, 13, 14], chunkSizes: [1, 1, 3], finished: nil),
    ]
    let audits = [
        mtpAudit(slot: 0, confirmed: 2, cumulative: 3, seedTokenCount: mtpFixtureSeedTokenCount),
        mtpAudit(slot: 0, confirmed: 2, cumulative: 5, seedTokenCount: mtpFixtureSeedTokenCount),
    ]
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 2, targetN: 4,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil,
            finalMetrics: mtpMetrics(drafted: 2, accepted: 2, roundAudits: audits),
            decodeNsByStream: fakeDecodeNs(2))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .mtpChunkWithoutVerifyAudit(slot: 1, cumulative: 5, width: 3)
    }
}

@Test
func mtpCohortAssemblyRefusesAnAuditWidthDisagreement() throws {
    // The engine's finalize record says the round committed 2; the stream
    // carries 3 at that position. Both cannot be right — refused.
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2, 3, 4], chunkSizes: [1, 1, 3], finished: nil)
    ]
    let audits = [
        mtpAudit(slot: 0, confirmed: 2, cumulative: 5, seedTokenCount: mtpFixtureSeedTokenCount)
    ]
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 1, targetN: 4,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil,
            finalMetrics: mtpMetrics(drafted: 2, accepted: 1, roundAudits: audits),
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .mtpAuditWidthDisagreement(
                slot: 0, cumulative: 5, auditConfirmed: 2, chunkWidth: 3)
    }
}

@Test
func mtpCohortAssemblyRefusesATruncatedAuditRing() throws {
    // At the retention cap the oldest records were dropped, so coverage of
    // the window can no longer be proven — fail closed.
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2], chunkSizes: [1, 1, 1], finished: nil)
    ]
    let full = (0 ..< CBv2MTPRoundAuditRecord.retainedRecordCap).map { index in
        mtpAudit(
            slot: 7, confirmed: 1, cumulative: index + 2,
            seedTokenCount: mtpFixtureSeedTokenCount)
    }
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 1, targetN: 2,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil,
            finalMetrics: mtpMetrics(roundAudits: full),
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .mtpRoundAuditsTruncated(count: CBv2MTPRoundAuditRecord.retainedRecordCap)
    }
}

@Test
func mtpCohortAssemblyRefusesAMalformedSeedChunk() throws {
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        // First delta committed 2 tokens, not the expected single seed token.
        (tokens: [100, 101, 1, 2, 3], chunkSizes: [2, 3], finished: nil)
    ]
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 1, targetN: 3,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil, finalMetrics: mtpMetrics(),
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError == .mtpSeedChunkMalformed(slot: 0, got: 2)
    }
}

@Test
func mtpCohortAssemblyRefusesWhenMetricsAreMissing() throws {
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2], chunkSizes: [1, 2], finished: nil)
    ]
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 1, targetN: 2,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil, finalMetrics: nil,
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError == .mtpMetricsMissing
    }
}

@Test
func mtpCohortAssemblyRefusesADecodeNsByStreamLengthMismatch() throws {
    let perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = [
        (tokens: [100, 1, 2], chunkSizes: [1, 2], finished: nil),
        (tokens: [200, 11, 12], chunkSizes: [1, 2], finished: nil),
    ]
    #expect {
        try assembleMTPCohortFreeRun(
            perSlot: perSlot, batchSize: 2, targetN: 2,
            seedTokenCount: mtpFixtureSeedTokenCount,
            baselineMetrics: nil, finalMetrics: mtpMetrics(),
            decodeNsByStream: fakeDecodeNs(1))
    } throws: { error in
        error as? RuntimeWorkerCohortError
            == .decodeNsByStreamCount(expected: 2, got: 1)
    }
}

@Test
func mtpDepthClampReasonsDiffsAgainstBaselineAndMergesBothHistograms() {
    let baseline = mtpMetrics(
        skippedRows: ["kv_headroom": 1],
        controllerFallbacks: ["warmup": 2])
    let final = mtpMetrics(
        skippedRows: ["kv_headroom": 3, "batch_gate": 1],
        controllerFallbacks: ["warmup": 2, "tail_depth": 4])
    let merged = mtpDepthClampReasons(baseline: baseline, final: final)
    // kv_headroom: 3 - 1 = 2 (diffed); batch_gate: 1 - 0 = 1 (new key);
    // warmup: 2 - 2 = 0 (dropped, no delta); tail_depth: 4 - 0 = 4 (new key).
    #expect(merged == ["kv_headroom": 2, "batch_gate": 1, "tail_depth": 4])
}

@Test
func mtpDepthClampReasonsWithNoBaselineUsesTheFinalSnapshotDirectly() {
    let final = mtpMetrics(
        skippedRows: ["carry_invalid": 1], controllerFallbacks: ["exploration": 2])
    #expect(
        mtpDepthClampReasons(baseline: nil, final: final)
            == ["carry_invalid": 1, "exploration": 2])
}

// MARK: - Wire plumbing (additive fields on the closed envelopes)

@Test
func requestDecodesTheCohortFieldsAndOmitsThemWhenAbsent() throws {
    // The exact lines benchd's session serializes for the cohort verbs.
    let begin = try request(
        #"{"id":8,"kind":"free_decode_begin","spec":{"mode":"serial"},"#
            + #""seed_tokens_by_stream":[[1,2],[3,4]],"batch_size":2}"#)
    #expect(begin.seedTokensByStream == [[1, 2], [3, 4]])
    #expect(begin.batchSize == 2)
    let run = try request(
        #"{"id":9,"kind":"free_decode_run","count":128,"batch_size":8}"#)
    #expect(run.rowCount == 128)
    #expect(run.batchSize == 8)
    // A v1.1 line decodes to nil cohort fields, and a request built without
    // them never emits the keys (wire-additive both directions).
    let v11 = try request(#"{"id":9,"kind":"free_decode_run","count":8}"#)
    #expect(v11.seedTokensByStream == nil)
    #expect(v11.batchSize == nil)
    let encoded = String(
        decoding: try JSONEncoder().encode(
            RuntimeWorkerRequest(id: 1, kind: "free_decode_run", rowCount: 8)),
        as: UTF8.self)
    #expect(!encoded.contains("batch_size"))
    #expect(!encoded.contains("seed_tokens_by_stream"))
}

@Test
func requestEnvelopeStaysClosedOverUnknownFields() throws {
    // deny-unknown-fields is unchanged by the additive extension.
    #expect(throws: DecodingError.self) {
        _ = try request(
            #"{"id":9,"kind":"free_decode_run","count":8,"batch_sizes":8}"#)
    }
}

@Test
func responseCarriesTheCohortFieldsWithBenchdWireNames() throws {
    let response = RuntimeWorkerResponse(
        id: 9,
        nonce: "n",
        ok: true,
        acceptanceLengths: [1, 1],
        draftedTotal: 0,
        acceptedTotal: 0,
        committedTotal: 4,
        maxBatchSize: 8,
        seedTokenByStream: [10, 11],
        effectiveBatchSize: 2,
        tokensByStream: [[700, 701], [800, 801]],
        naturalAcceptedByStream: [[1, 1], [1, 1]],
        rounds: 2,
        activeStreamsByRound: [2, 2],
        depthClampReasons: ["tail_depth": 3],
        prefillNsByStream: [1_000_000, 1_200_000],
        decodeNsByStream: [5_000_000, 5_400_000]
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let encoded = String(decoding: try encoder.encode(response), as: UTF8.self)
    // Exactly benchd's WorkerResponse field names (bench-protocol v1.2).
    for key in [
        #""max_batch_size":8"#,
        #""seed_token_by_stream":[10,11]"#,
        #""effective_batch_size":2"#,
        #""tokens_by_stream":[[700,701],[800,801]]"#,
        #""natural_accepted_by_stream":[[1,1],[1,1]]"#,
        #""rounds":2"#,
        #""active_streams_by_round":[2,2]"#,
        #""depth_clamp_reasons":{"tail_depth":3}"#,
        #""prefill_ns_by_stream":[1000000,1200000]"#,
        #""decode_ns_by_stream":[5000000,5400000]"#,
    ] {
        #expect(encoded.contains(key), "missing \(key) in \(encoded)")
    }
    // Round-trip through the engine's own closed-envelope decoder.
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self, from: Data(encoded.utf8))
    #expect(decoded.tokensByStream == [[700, 701], [800, 801]])
    #expect(decoded.depthClampReasons == ["tail_depth": 3])
    #expect(decoded.prefillNsByStream == [1_000_000, 1_200_000])
    #expect(decoded.decodeNsByStream == [5_000_000, 5_400_000])
    // A minimal response never emits a v1.2 key.
    let minimal = String(
        decoding: try encoder.encode(RuntimeWorkerResponse(id: 0, ok: true)),
        as: UTF8.self)
    for key in [
        "max_batch_size", "seed_token_by_stream", "effective_batch_size",
        "tokens_by_stream", "natural_accepted_by_stream", "\"rounds\"",
        "active_streams_by_round", "depth_clamp_reasons",
        "prefill_ns_by_stream", "decode_ns_by_stream",
    ] {
        #expect(!minimal.contains(key), "minimal response leaked \(key)")
    }
    // The minimal response round-trips to nil for both new fields — "absent
    // when [the per-stream-timing capability is] off" is the same
    // nil-on-decode shape every other v1.2 field already has.
    let decodedMinimal = try JSONDecoder().decode(
        RuntimeWorkerResponse.self, from: Data(minimal.utf8))
    #expect(decodedMinimal.prefillNsByStream == nil)
    #expect(decodedMinimal.decodeNsByStream == nil)
}

// MARK: - Per-stream timing instrumentation (pure helper + collector)

@Test
func commitTimestampNsWalksCumulativeChunkSizesToTheLandingAppend() {
    // Serial regime: one token per append. Cumulative count 1 (the seed) is
    // append 0; cumulative count 4 is append 3.
    let serialChunks = [1, 1, 1, 1, 1]
    let serialTimestamps: [UInt64] = [100, 200, 300, 400, 500]
    #expect(
        commitTimestampNs(
            chunkSizes: serialChunks, commitTimestampsNs: serialTimestamps,
            atCumulativeCount: 1) == 100)
    #expect(
        commitTimestampNs(
            chunkSizes: serialChunks, commitTimestampsNs: serialTimestamps,
            atCumulativeCount: 4) == 400)
    // MTP regime: variable per-round widths. Seed (count 1) lands at append
    // 0; cumulative count 4 (seed + 3 decode tokens) is reached mid-round-1
    // (1 + 3 == 4), so it lands at append 1's timestamp, NOT append 2's —
    // the walk stops at the FIRST append that reaches the target, never
    // overshoots to a later one.
    let mtpChunks = [1, 3, 4]
    let mtpTimestamps: [UInt64] = [10, 20, 30]
    #expect(
        commitTimestampNs(
            chunkSizes: mtpChunks, commitTimestampsNs: mtpTimestamps,
            atCumulativeCount: 1) == 10)
    #expect(
        commitTimestampNs(
            chunkSizes: mtpChunks, commitTimestampsNs: mtpTimestamps,
            atCumulativeCount: 4) == 20)
    #expect(
        commitTimestampNs(
            chunkSizes: mtpChunks, commitTimestampsNs: mtpTimestamps,
            atCumulativeCount: 8) == 30)
    // Never reached: the histories fell short (a wiring bug the driver
    // refuses on, `RuntimeWorkerCohortError.commitTimestampMissing`).
    #expect(
        commitTimestampNs(
            chunkSizes: mtpChunks, commitTimestampsNs: mtpTimestamps,
            atCumulativeCount: 9) == nil)
    // A timestamps history shorter than the chunk-size history it should be
    // 1:1 with is also refused (defensive — the collector always grows both
    // under the same lock, so this is a synthetic-only case).
    #expect(
        commitTimestampNs(
            chunkSizes: [1, 1], commitTimestampsNs: [10], atCumulativeCount: 2) == nil)
}

@Test
func collectorRecordsARealCommitTimestampPerAppendInLandingOrder() throws {
    // The collector's own commit-timestamp history — not the order a caller
    // happens to poll slots in — is what the driver samples (this file's
    // header note on `commitTimestampsNs`). Three appends land in order;
    // each one's recorded timestamp is non-decreasing (monotone clock), and
    // the history is 1:1 with chunkSizes.
    let collector = RuntimeWorkerCohortStreamCollector(slot: 0)
    collector.append([5])
    collector.append([6, 7])
    collector.append([8])
    _ = try collector.waitForTokenCount(4)
    let (chunkSizes, commitTimestampsNs) = collector.commitHistorySnapshot()
    #expect(chunkSizes == [1, 2, 1])
    #expect(commitTimestampsNs.count == 3)
    // Monotone/ordering sanity: successive real clock samples never go
    // backwards.
    #expect(commitTimestampsNs[0] <= commitTimestampsNs[1])
    #expect(commitTimestampsNs[1] <= commitTimestampsNs[2])
    // And the pure helper, fed this REAL history, lands on the right append
    // for both sample points the driver actually uses: cumulative 1 (seed)
    // is append 0, cumulative 4 (seed + 3) is append 2 (1 + 2 + 1 == 4).
    #expect(
        commitTimestampNs(
            chunkSizes: chunkSizes, commitTimestampsNs: commitTimestampsNs,
            atCumulativeCount: 1) == commitTimestampsNs[0])
    #expect(
        commitTimestampNs(
            chunkSizes: chunkSizes, commitTimestampsNs: commitTimestampsNs,
            atCumulativeCount: 4) == commitTimestampsNs[2])
}

// MARK: - Stream collector (async bridging, no engine)

private func makeStream(
    _ build: @escaping (AsyncStream<CBv2Event>.Continuation) -> Void
) -> AsyncStream<CBv2Event> {
    AsyncStream<CBv2Event> { continuation in
        build(continuation)
    }
}

private let collectorUsage = CBv2Usage(promptTokens: 2, completionTokens: 3)

@Test
func collectorGathersTokensAcrossDeltasAndReturnsPrefixes() throws {
    let collector = RuntimeWorkerCohortStreamCollector(slot: 0)
    collector.consume(
        makeStream { continuation in
            continuation.yield(.delta(text: "", tokens: [5], logprobs: nil))
            continuation.yield(.delta(text: "", tokens: [6, 7], logprobs: nil))
            continuation.yield(
                .finished(reason: .length, usage: collectorUsage))
            continuation.finish()
        })
    // The seed wait and the run wait are prefix reads of one append-only log.
    #expect(try collector.waitForTokenCount(1) == [5])
    #expect(try collector.waitForTokenCount(3) == [5, 6, 7])
}

@Test
func collectorThrowsEndedEarlyWithSlotAndReason() throws {
    let collector = RuntimeWorkerCohortStreamCollector(slot: 3)
    collector.consume(
        makeStream { continuation in
            continuation.yield(.delta(text: "", tokens: [1], logprobs: nil))
            continuation.yield(
                .finished(reason: .stop, usage: collectorUsage))
            continuation.finish()
        })
    #expect {
        try collector.waitForTokenCount(4)
    } throws: { error in
        guard case let .streamEndedEarly(slot, committed, target, _)? =
            error as? RuntimeWorkerCohortError
        else { return false }
        return slot == 3 && committed == 1 && target == 4
    }
}

@Test
func collectorUnblocksAWaiterWhenTokensArrive() throws {
    // The protocol thread blocks in waitForTokenCount while the consumer task
    // appends — deliver the tokens from another thread after the wait starts.
    let collector = RuntimeWorkerCohortStreamCollector(slot: 0)
    DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(50)) {
        collector.append([42, 43])
    }
    #expect(try collector.waitForTokenCount(2) == [42, 43])
}

// MARK: - Contiguous KV budget

@Test
func cohortKVBudgetScalesWithWidthAndChargesWindowedRings() {
    let full = CBv2LayerKind(
        attention: .full, headDim: 256, kvHeads: 4, queryHeads: 8)
    let windowed = CBv2LayerKind(
        attention: .slidingWindow(512), headDim: 256, kvHeads: 4, queryHeads: 8)
    let shared = CBv2LayerKind(
        attention: .slidingWindow(512), sharesKVWithLayer: 0,
        headDim: 256, kvHeads: 4, queryHeads: 8)
    let kinds = [full, windowed, shared]
    let b1 = cohortContiguousKVBytesBudget(
        layerKinds: kinds, batchSize: 1, maxSequenceLength: 2048)
    let b8 = cohortContiguousKVBytesBudget(
        layerKinds: kinds, batchSize: 8, maxSequenceLength: 2048)
    #expect(b1 > 0)
    #expect(b8 == b1 * 8)
    // Per stream: full layer 2048 tokens + windowed ring 512 tokens, K and V,
    // 4 heads x 256 dims x 2 bytes fp16, x2 headroom; the KV-shared layer
    // owns no storage.
    let perToken = 2 * 4 * 256 * 2
    #expect(b1 == (2048 + 512) * perToken * 2)
}
