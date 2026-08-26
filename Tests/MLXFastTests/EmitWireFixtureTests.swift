// Emits the engine's canonical wire responses through the worker's OWN type and
// encoder, so the cross-repo crosscheck in benchd consumes CAPTURED real-encoder
// bytes (sha-pinned), never a hand-written literal. Run:
//   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test \
//     --filter emitEngineWireFixture
// then the printed path/sha is placed into benchd's fixtures and pinned both ends.
import Foundation
import CryptoKit
import MLXFastCore
import MLXLMCommon
import Testing
@testable import MLXFastRuntimeWorkerSupport

private typealias WorkerResponse = MLXFastRuntimeWorkerSupport.RuntimeWorkerResponse

// Sha of the canonical (.sortedKeys) real-encoder fixture; benchd pins the same value.
//
// REPINNED 2026-08-25 (ungated `cohort_reference_replay` advertisement): the
// gate-on hello (line 5, `batchedHello`) now leads with the UNGATED
// `cohort_reference_replay` capability, so its `capabilities` array grows from
// 3 entries to 4 — `["cohort_reference_replay", "free_run_decode",
// "batched_free_run_decode", "per_stream_timing"]`. That is the ONLY field that
// moves: 3005 -> 3031 bytes is exactly the 26 added bytes of the quoted string
// plus its comma, line count stays 11, and every other line (1-4, 6-11) and
// every other field on line 5 is BYTE-IDENTICAL to the prior repin (e79fd829...).
//
// This repin is also a FIX, not just an update. The capability list here used to
// read `runtimeWorkerAdvertisedCapabilities` directly, while the real hello
// prepended the ungated capability — so when that ungating landed, this fixture
// silently stopped mirroring the real emitter WITHOUT the sha moving, and the
// pin stayed green over a stale surface. The line now calls
// `runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol:)`, the same
// function the hello itself calls, so a future capability change moves these
// bytes instead of quietly diverging from them.
//
// CROSS-REPO: benchd pins this same constant
// (`benchd/crates/bench-runner/src/wire_crosscheck.rs`) and both sides must
// agree byte-for-byte (port-notes 9.3). The benchd-side repin to this value, and
// re-placing the captured fixture there, is a SEPARATE change in the benchd
// repository — it is not in this branch.
//
// REPINNED 2026-08-23 (per-stream timing instrumentation, spec step 1): the
// `per_stream_timing` hello capability joins `runtimeWorkerAdvertisedCapabilities`
// (line 5's `batchedHello` — capabilities list grows from 2 to 3 entries), and
// the batched cohort begin/run lines (6/7 serial, 10/11 mtp) each gain one new
// additive field: `prefill_ns_by_stream` on the begin, `decode_ns_by_stream` on
// the run. Both are derived through the REAL `commitTimestampNs` pure helper
// (RuntimeWorkerCohortSupport) over representative synthetic per-slot commit
// histories (this fixture generator is model-free by design), not hand-typed
// ns deltas — same discipline the prior repins use for every other cohort
// counter. Every other line (1-4, 8, 9) and every other field on 5-7/10-11 is
// BYTE-IDENTICAL to the prior repin (718799e...); only the SHA and per-field
// diffs above move. Line count stays 11.
private let ENGINE_WIRE_V1_SHA256 = "4d7e2657b801d23eb20b3d9aecbcd2a543ce83b56eea326552515ed1fe323a7f"

@Test
func emitEngineWireFixture() throws {
    let registry = RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable: true)

    // 1) gated-on generic hello — carries the fields benchd used to REJECT
    //    (spec_modes / capabilities / head_provenance) plus the v1 identity.
    let hello = WorkerResponse(
        id: 0,
        nonce: "session-nonce",
        ok: true,
        expertStats: .zero,
        specModes: registry.advertisedModeStrings,
        // Non-nil, exercising benchd's `deny_unknown_fields` decoder against
        // every field the wire may legally carry. As of the MTP arm this is
        // also an honest shape: a worker that loaded an assistant head
        // produces exactly this (sha256/bytes/file_count) via
        // computeGemma4AssistantHeadProvenance.
        headProvenance: RuntimeWorkerHeadProvenance(
            sha256: String(repeating: "a", count: 64),
            bytes: 849_398_784,
            fileCount: 2
        ),
        capabilities: [runtimeWorkerFreeRunDecodeCapability],
        protocolVersion: runtimeWorkerProtocolVersion,
        backend: runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )

    // 2) phase_diagnostics — the fatal case: mlx_* memory ints ALWAYS emitted,
    //    which broke every phase close under benchd's deny_unknown_fields.
    let phase = WorkerResponse(
        id: 3,
        nonce: "session-nonce",
        ok: true,
        expertStats: .zero,
        peakRamGB: 1.25,
        mlxActiveMemoryBytes: 1,
        mlxCacheMemoryBytes: 0,
        mlxPeakMemoryBytes: 2,
        completedWork: 3,  // R+1 for the R=2 free-run below (seed forward + 2 rounds)
        cacheMemory: 0
    )

    // 3) free_decode_begin — the seed forward that OPENS the free-run phase. benchd verifies this
    //    `seed_token` against `expected_decode_seed_token` (PROTOCOL-v1.1 §2.2) and it establishes
    //    the last-committed state (§2.1). It is pinned in the captured bytes so the begin/run SEAM
    //    is a cross-repo fact and not a convention either side can drift on its own (#109 W3
    //    finding 6).
    let freeBegin = WorkerResponse(
        id: 1,
        nonce: "session-nonce",
        ok: true,
        seedToken: 699,
        effectiveSpec: .serial()
    )

    // 4) free_decode_run — the scored path's AUDIT counters + the §2.6 triple. §2.2 matches
    //    `tokens[i]` against `expected_decode_tokens[i]`, the N tokens AFTER the seed — so
    //    `tokens[0]` is 700 and NOT the 699 the begin above already returned.
    let freeRun = WorkerResponse(
        id: 2,
        nonce: "session-nonce",
        ok: true,
        tokens: [700, 701, 702, 703],
        acceptanceLengths: [3, 1],
        draftedTotal: 4,
        acceptedTotal: 2,
        committedTotal: 4
    )

    // 5) v1.2 BATCHED gate-on hello — the cohort form's capability surface,
    //    mirroring the worker's REAL hello assembly (Gemma4RuntimeWorker gate-on
    //    branch): the advertised capability LIST (v1.1 free-run plus its v1.2
    //    batched form) and the cohort-width ceiling benchd uses to refuse an
    //    over-wide cohort pre-GPU. Both values are the worker's own constants,
    //    not literals. `head_provenance` is ABSENT, mirroring the real hello on
    //    this engine (always null — no head is loadable); line 1 above already
    //    pins the non-nil decode path.
    //
    //    The capability list is taken from `runtimeWorkerHelloCapabilities`, the
    //    SAME function the real hello calls, rather than from
    //    `runtimeWorkerAdvertisedCapabilities` directly. That direct use is what
    //    let the ungating change slip past this fixture silently: the real hello
    //    gained a prepended `cohort_reference_replay` while this line kept
    //    emitting the three speculative caps alone, so the captured "real
    //    encoder" bytes stopped mirroring the real emitter WITHOUT moving the
    //    sha — a green pin over a stale surface. Driving the emitter closes that
    //    hole for every future capability change.
    let batchedHello = WorkerResponse(
        id: 0,
        nonce: "session-nonce",
        ok: true,
        expertStats: .zero,
        specModes: registry.advertisedModeStrings,
        capabilities: runtimeWorkerHelloCapabilities(
            advertisesSpeculativeProtocol: true),
        maxBatchSize: runtimeWorkerMaxCohortBatchSize,
        protocolVersion: runtimeWorkerProtocolVersion,
        backend: runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )

    // 6/7) the batched begin/run pair, assembled through the REAL serial-cohort
    //    driver path (`assembleSerialCohortFreeRun`, RuntimeWorkerCohortSupport)
    //    rather than hand-built counters, so the cohort AUDIT vectors in the
    //    captured bytes are the worker's own assembly. Small closed cohort
    //    B=2, N=3: each slot's collected stream INCLUDES its leading seed token
    //    (the one the batched begin already returned), and the N committed
    //    tokens come AFTER it — the same §2.2 begin/run seam as the
    //    single-stream lines above, per slot.
    let streamsWithSeed = [[800, 801, 802, 803], [900, 901, 902, 903]]
    let cohortB = 2
    let cohortN = 3

    // Per-stream timing instrumentation (spec step 1): synthetic per-slot
    // commit histories fed through the REAL pure helper
    // (`commitTimestampNs`, RuntimeWorkerCohortSupport) the driver itself
    // uses — not hand-typed ns deltas. A serial cohort appends exactly one
    // token per commit, so each slot's chunk-size history is `[1,1,1,1]`
    // (seed + 3 decode tokens, matching `streamsWithSeed`'s 4 entries per
    // slot); the two slots' timestamps are staggered to look like two
    // concurrently-running (not lockstep-identical) streams.
    let cohortPrefillPhaseStartNs: UInt64 = 1_000_000_000
    let cohortDecodePhaseStartNs: UInt64 = 2_000_000_000
    let cohortChunkSizes: [[Int]] = [[1, 1, 1, 1], [1, 1, 1, 1]]
    let cohortCommitTimestampsNs: [[UInt64]] = [
        [1_000_050_000, 2_000_100_000, 2_000_210_000, 2_000_320_000],
        [1_000_070_000, 2_000_120_000, 2_000_230_000, 2_000_340_000],
    ]
    let cohortPrefillNsByStream = try zip(cohortChunkSizes, cohortCommitTimestampsNs).map {
        chunkSizes, timestamps -> UInt64 in
        guard
            let ts = commitTimestampNs(
                chunkSizes: chunkSizes, commitTimestampsNs: timestamps, atCumulativeCount: 1)
        else { throw MLXFastError.invalidInput("fixture: unreachable cumulative count 1") }
        return ts - cohortPrefillPhaseStartNs
    }
    let cohortDecodeNsByStream = try zip(cohortChunkSizes, cohortCommitTimestampsNs).map {
        chunkSizes, timestamps -> UInt64 in
        guard
            let ts = commitTimestampNs(
                chunkSizes: chunkSizes, commitTimestampsNs: timestamps,
                atCumulativeCount: cohortN + 1)
        else {
            throw MLXFastError.invalidInput(
                "fixture: unreachable cumulative count \(cohortN + 1)")
        }
        return ts - cohortDecodePhaseStartNs
    }
    let cohortResult = try assembleSerialCohortFreeRun(
        streamsWithSeed: streamsWithSeed,
        batchSize: cohortB,
        targetN: cohortN,
        decodeNsByStream: cohortDecodeNsByStream)

    // 6) batched free_decode_begin — mirrors `handleCohortFreeDecodeBegin`'s
    //    response: the serial spec echo, the B seed-forward argmaxes in SLOT
    //    ORDER, the never-ignored width echo, and (new) the per-slot
    //    cohort-prefill elapsed-ns vector.
    let cohortBegin = WorkerResponse(
        id: 4,
        nonce: "session-nonce",
        ok: true,
        effectiveSpec: .serial(),
        seedTokenByStream: streamsWithSeed.map { $0[0] },
        effectiveBatchSize: cohortB,
        prefillNsByStream: cohortPrefillNsByStream
    )

    // 7) batched free_decode_run — mirrors `handleCohortFreeDecodeRun`'s
    //    response, every counter from the real assembly: the B x N committed
    //    rectangle, the single-vector common-width histogram (sum == N), the
    //    B x R natural walks, the closed-cohort active histogram, the serial
    //    totals (drafted == accepted == 0, committed == B * N), the sealed
    //    EMPTY depth-clamp histogram, and (new) the per-slot decode-phase
    //    elapsed-ns vector straight from the real assembly.
    let cohortRun = WorkerResponse(
        id: 5,
        nonce: "session-nonce",
        ok: true,
        acceptanceLengths: cohortResult.acceptanceLengths,
        draftedTotal: cohortResult.draftedTotal,
        acceptedTotal: cohortResult.acceptedTotal,
        committedTotal: cohortResult.committedTotal,
        effectiveBatchSize: cohortB,
        tokensByStream: cohortResult.tokensByStream,
        naturalAcceptedByStream: cohortResult.naturalAcceptedByStream,
        rounds: cohortResult.rounds,
        activeStreamsByRound: cohortResult.activeStreamsByRound,
        depthClampReasons: cohortResult.depthClampReasons,
        decodeNsByStream: cohortResult.decodeNsByStream
    )

    // 8) the cohort window's phase_diagnostics — same discipline as line 2: the
    //    cohort's phase-close barrier counter is pinned in captured bytes so
    //    benchd's cohort QUADRUPLE (`verify_cohort_consistency`) consumes a
    //    captured `completed_work`, not a convention. SCALAR `R + 1` (a round
    //    is one engine forward regardless of B) — from the real assembly's
    //    computed property, not a literal.
    let cohortPhase = WorkerResponse(
        id: 6,
        nonce: "session-nonce",
        ok: true,
        expertStats: .zero,
        peakRamGB: 1.25,
        mlxActiveMemoryBytes: 1,
        mlxCacheMemoryBytes: 0,
        mlxPeakMemoryBytes: 2,
        completedWork: cohortResult.completedWork,
        cacheMemory: 0
    )

    // 9) mtp-mode decode_begin echo — the new line this repin adds. Exercises
    //    the `{"mtp":{"depth":...}}` effective-spec sub-object through the
    //    REAL registry resolution path (not a hand-built literal): a worker
    //    that can run mtp, asked for it with no explicit depth, echoes the
    //    pinned ceiling (`Gemma4MTPEnvelope.maxDraftTokens`, MTPEnvelope.swift).
    let mtpDecodeBegin = WorkerResponse(
        id: 7,
        nonce: "session-nonce",
        ok: true,
        seedToken: 1200,
        effectiveSpec: try registry.resolveEffectiveSpec(
            RuntimeWorkerSpecRequest(mode: .mtp))
    )

    // 10/11) the batched MTP begin/run pair — the SAME B=2, N=3 shape as the
    //    serial cohort pair (6/7) above, but spec `mtp` and assembled through
    //    the REAL `assembleMTPCohortFreeRun` over a representative per-slot
    //    round history: both slots' post-seed chunk sequence is [1, 2] (a
    //    non-drafting round of width 1 then a one-draft-accepted round of
    //    width 2), matching the closed-cohort D4 shape (no stop tokens,
    //    every round B-wide). This fixture generator is model-free by
    //    design (see the file header), so the per-slot histories and metrics
    //    snapshots are representative synthetic inputs FED THROUGH the real
    //    assembler — the same discipline line 6/7's `streamsWithSeed` uses
    //    for `assembleSerialCohortFreeRun`.
    let mtpStreamsWithSeed = [[800, 801, 802, 803], [900, 901, 902, 903]]
    let mtpCohortB = 2
    let mtpCohortN = 3
    let mtpPerSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] =
        mtpStreamsWithSeed.map { ($0, [1, 1, 2], nil) }

    // Per-stream timing instrumentation: same discipline as the serial pair
    // above, fed through the REAL `commitTimestampNs` helper — but over the
    // MTP regime's variable per-round chunk sizes (`[1, 1, 2]`: seed, then a
    // width-1 round, then a width-2 round; sum 4 == seed + mtpCohortN).
    let mtpPrefillPhaseStartNs: UInt64 = 3_000_000_000
    let mtpDecodePhaseStartNs: UInt64 = 4_000_000_000
    let mtpChunkSizes: [[Int]] = [[1, 1, 2], [1, 1, 2]]
    let mtpCommitTimestampsNs: [[UInt64]] = [
        [3_000_060_000, 4_000_150_000, 4_000_410_000],
        [3_000_080_000, 4_000_170_000, 4_000_430_000],
    ]
    let mtpPrefillNsByStream = try zip(mtpChunkSizes, mtpCommitTimestampsNs).map {
        chunkSizes, timestamps -> UInt64 in
        guard
            let ts = commitTimestampNs(
                chunkSizes: chunkSizes, commitTimestampsNs: timestamps, atCumulativeCount: 1)
        else { throw MLXFastError.invalidInput("fixture: unreachable cumulative count 1") }
        return ts - mtpPrefillPhaseStartNs
    }
    let mtpDecodeNsByStream = try zip(mtpChunkSizes, mtpCommitTimestampsNs).map {
        chunkSizes, timestamps -> UInt64 in
        guard
            let ts = commitTimestampNs(
                chunkSizes: chunkSizes, commitTimestampsNs: timestamps,
                atCumulativeCount: mtpCohortN + 1)
        else {
            throw MLXFastError.invalidInput(
                "fixture: unreachable cumulative count \(mtpCohortN + 1)")
        }
        return ts - mtpDecodePhaseStartNs
    }
    let mtpCohortSeedTokenCount = 4
    let mtpCohortResult = try assembleMTPCohortFreeRun(
        perSlot: mtpPerSlot,
        batchSize: mtpCohortB,
        targetN: mtpCohortN,
        seedTokenCount: mtpCohortSeedTokenCount,
        baselineMetrics: nil,
        finalMetrics: {
            var metrics = CBv2MTPMetrics()
            metrics.draftedTokens = 5
            metrics.acceptedTokens = 3
            metrics.controllerFallbacks = ["tail_depth": 1]
            // Reconciliation contract: every multi-token chunk in the
            // per-slot histories above ([1, 2] per slot — the width-2 round
            // ends at cumulative committed count 4, seed included) is backed
            // by its engine finalize record. INPUT to the real assembler
            // only; the emitted fixture bytes are unchanged.
            metrics.roundAudits = (0 ..< 2).map { slot in
                CBv2MTPRoundAuditRecord(
                    requestID: UInt64(slot), k: 1,
                    draftTokens: [0], targetTokens: [0, 0],
                    accepted: 1, confirmed: 2, rejected: 0,
                    tokensCountAfter: mtpCohortSeedTokenCount + 4,
                    numComputedAfter: mtpCohortSeedTokenCount + 3,
                    generatedAfter: 4,
                    finishReason: nil)
            }
            return metrics
        }(),
        decodeNsByStream: mtpDecodeNsByStream)

    // 10) batched free_decode_begin (mtp) — mirrors `handleCohortFreeDecodeBegin`'s
    //    response for a resolved-mtp cohort: the mtp spec echo (through the
    //    REAL registry resolution path, not a literal), the B seed-forward
    //    argmaxes, the width echo, and (new) the per-slot cohort-prefill
    //    elapsed-ns vector.
    let mtpCohortBegin = WorkerResponse(
        id: 8,
        nonce: "session-nonce",
        ok: true,
        effectiveSpec: try registry.resolveEffectiveSpec(
            RuntimeWorkerSpecRequest(mode: .mtp)),
        seedTokenByStream: mtpStreamsWithSeed.map { $0[0] },
        effectiveBatchSize: mtpCohortB,
        prefillNsByStream: mtpPrefillNsByStream
    )

    // 11) batched free_decode_run (mtp) — mirrors `handleCohortFreeDecodeRun`'s
    //    response for the mtp route: every counter from the REAL
    //    `assembleMTPCohortFreeRun` assembly above, including the non-empty
    //    `depth_clamp_reasons` histogram a drafting cohort can genuinely
    //    report (the serial pair's is always sealed empty — line 7 above —
    //    because a non-speculative cohort clamps nothing), and (new) the
    //    per-slot decode-phase elapsed-ns vector.
    let mtpCohortRun = WorkerResponse(
        id: 9,
        nonce: "session-nonce",
        ok: true,
        acceptanceLengths: mtpCohortResult.acceptanceLengths,
        draftedTotal: mtpCohortResult.draftedTotal,
        acceptedTotal: mtpCohortResult.acceptedTotal,
        committedTotal: mtpCohortResult.committedTotal,
        effectiveBatchSize: mtpCohortB,
        tokensByStream: mtpCohortResult.tokensByStream,
        naturalAcceptedByStream: mtpCohortResult.naturalAcceptedByStream,
        rounds: mtpCohortResult.rounds,
        activeStreamsByRound: mtpCohortResult.activeStreamsByRound,
        depthClampReasons: mtpCohortResult.depthClampReasons,
        decodeNsByStream: mtpCohortResult.decodeNsByStream
    )

    // .sortedKeys canonicalizes key order at every level (the response carries an
    // unordered expert_stats dict) so the fixture is byte-stable and sha-pinnable;
    // field names/types/values are the worker's real Codable output, untrimmed.
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
    let lines = try [
        hello, phase, freeBegin, freeRun,
        batchedHello, cohortBegin, cohortRun, cohortPhase,
        mtpDecodeBegin, mtpCohortBegin, mtpCohortRun,
    ].map { try encoder.encode($0) }
    var blob = Data()
    for line in lines { blob.append(line); blob.append(0x0A) }

    // Machine-independent destination: a hardcoded scratchpad path is one
    // developer's sandbox and fails everywhere else (and in CI). The emitted
    // path is printed to STDERR below — that is how the operator finds the file
    // to copy into benchd's fixtures.
    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("engine-wire-v1.jsonl")
    try blob.write(to: outURL)

    let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
    FileHandle.standardError.write(
        "EMIT_WIRE_FIXTURE path=\(outURL.path) bytes=\(blob.count) sha256=\(digest)\n"
            .data(using: .utf8)!
    )
    // Pin: the engine encoder must keep emitting exactly this canonical field set.
    // Regenerate + repin (both repos) intentionally if the wire surface changes.
    #expect(digest == ENGINE_WIRE_V1_SHA256)
    #expect(lines.count == 11)
    // The SEAM, asserted on the emitted bytes: the run's first committed token is NOT the token
    // the begin already returned (#109 W3 finding 6, PROTOCOL-v1.1 §2.2).
    #expect(freeBegin.seedToken == 699)
    #expect(freeRun.tokens?.first == 700)
    #expect(freeRun.tokens?.contains(freeBegin.seedToken!) == false)
    // The COHORT seam, per slot: each slot's committed row starts AFTER that
    // slot's seed, and no row re-emits its seed (the batched generalization of
    // the v1.1 seam above).
    #expect(cohortBegin.seedTokenByStream == [800, 900])
    for (slot, row) in (cohortRun.tokensByStream ?? []).enumerated() {
        #expect(row.first == streamsWithSeed[slot][1])
        #expect(!row.contains(streamsWithSeed[slot][0]))
    }
    // And the cohort counters satisfy the QUADRUPLE's scalar equations on the
    // emitted values: committed == B * N, sum(common widths) == N, and the
    // phase line's barrier is the SCALAR R + 1.
    #expect(cohortRun.committedTotal == cohortB * cohortN)
    #expect(cohortRun.acceptanceLengths?.reduce(0, +) == cohortN)
    #expect(cohortRun.draftedTotal == 0 && cohortRun.acceptedTotal == 0)
    #expect(cohortPhase.completedWork == cohortResult.rounds + 1)
    // Per-stream timing instrumentation: the gate-on hello advertises the
    // capability alongside the batched form it rides on, and both new
    // vectors are B-long, in slot order, and match the values the real
    // `commitTimestampNs` helper derived above.
    #expect(batchedHello.capabilities?.contains(runtimeWorkerPerStreamTimingCapability) == true)
    // ...and the gate-on hello leads with the UNGATED reference-replay capability
    // (benchd's (b) admission oracle reads it off a PLAIN hello; it rides on the
    // gate-on hello too). Asserted on the emitted line so the captured bytes,
    // not just the constant, carry it.
    #expect(batchedHello.capabilities?.first == runtimeWorkerCohortReferenceReplayCapability)
    #expect(
        batchedHello.capabilities
            == runtimeWorkerHelloCapabilities(advertisesSpeculativeProtocol: true))
    #expect(cohortBegin.prefillNsByStream == cohortPrefillNsByStream)
    #expect(cohortBegin.prefillNsByStream?.count == cohortB)
    #expect(cohortRun.decodeNsByStream == cohortResult.decodeNsByStream)
    #expect(cohortRun.decodeNsByStream?.count == cohortB)
    // The mtp echo: spec_modes advertises mtp again, and the new line's
    // effective_spec names mode "mtp" with depth clamped to the pinned
    // ceiling (no depth was requested).
    #expect(registry.advertisedModeStrings == ["serial", "mtp"])
    #expect(mtpDecodeBegin.effectiveSpec?.mode == "mtp")
    #expect(mtpDecodeBegin.effectiveSpec?.mtp?.depth == Gemma4MTPEnvelope.maxDraftTokens)
    // The batched MTP pair: same seam discipline as the serial cohort pair,
    // plus the QUADRUPLE's scalar equations, plus a REAL (non-empty)
    // depth-clamp histogram — the thing a non-speculative cohort can never
    // report.
    #expect(mtpCohortBegin.effectiveSpec?.mode == "mtp")
    #expect(mtpCohortBegin.seedTokenByStream == [800, 900])
    for (slot, row) in (mtpCohortRun.tokensByStream ?? []).enumerated() {
        #expect(row.first == mtpStreamsWithSeed[slot][1])
        #expect(!row.contains(mtpStreamsWithSeed[slot][0]))
    }
    #expect(mtpCohortRun.committedTotal == mtpCohortB * mtpCohortN)
    #expect(mtpCohortRun.acceptanceLengths?.reduce(0, +) == mtpCohortN)
    #expect(mtpCohortRun.draftedTotal == 5 && mtpCohortRun.acceptedTotal == 3)
    #expect(mtpCohortRun.depthClampReasons == ["tail_depth": 1])
    // Per-stream timing instrumentation, mtp leg: same shape as the serial
    // pair above.
    #expect(mtpCohortBegin.prefillNsByStream == mtpPrefillNsByStream)
    #expect(mtpCohortBegin.prefillNsByStream?.count == mtpCohortB)
    #expect(mtpCohortRun.decodeNsByStream == mtpCohortResult.decodeNsByStream)
    #expect(mtpCohortRun.decodeNsByStream?.count == mtpCohortB)
}
