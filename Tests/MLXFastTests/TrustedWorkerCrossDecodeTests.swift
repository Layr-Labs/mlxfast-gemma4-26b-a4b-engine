import Foundation
import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastRuntimeWorkerSupport
import Testing

// The missing kit test (fix #5): the trusted native CLI's OWN response decoder
// must parse EVERY line the participant worker emits — in BOTH the gated-off (v1)
// surface AND the gated-on (v1.1) surface. This is the regression guard for the
// spawn gate: the un-gated new hello fields (spec_modes / head_provenance /
// capabilities) were REJECTED by this decoder, which broke every worker-spawning
// verb of mlxfast-swift at the hello. Byte-for-byte: a worker response is built
// with the worker's OWN type, encoded with the worker's OWN encoder settings, and
// handed to the TRUSTED decoder.

// Both modules define `RuntimeWorkerResponse`; disambiguate by module.
private typealias WorkerResponse = MLXFastRuntimeWorkerSupport.RuntimeWorkerResponse
private typealias TrustedResponse = MLXFastHarness.RuntimeWorkerResponse

/// Encode a worker response exactly as the worker does (`.withoutEscapingSlashes`),
/// then decode the resulting line with the TRUSTED native-client decoder. Throws
/// iff the trusted decoder rejects the line — which is the regression this guards.
private func crossDecode(_ response: WorkerResponse) throws -> TrustedResponse {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    let line = try encoder.encode(response)
    return try JSONDecoder().decode(TrustedResponse.self, from: line)
}

// MARK: - The decoder is a SUPERSET by construction, not by example

@Test
func trustedDecoderKeySetEqualsTheWorkerKeySet() {
    // The per-response tests below prove specific lines parse. This proves the
    // property they sample: every wire key the worker can emit is a key the
    // trusted decoder declares, and vice versa — so a field added to one side
    // and forgotten on the other fails HERE rather than at a hello on the box.
    #expect(
        Set(WorkerResponse.CodingKeys.allCases.map(\.rawValue))
            == Set(TrustedResponse.CodingKeys.allCases.map(\.rawValue)))
}

/// The SAME property, one level down, for the `cohort_reference_replay` report.
///
/// The test above compares only the OUTER response key set, where the whole
/// report is a single key (`cohort_reference_replay`) — so a field added to the
/// report on one side and forgotten on the other passed it untouched. The report
/// is a second pair of twins (`Sources/MLXFastTrustedHarness/
/// Gemma4RuntimeCohortReferenceReplayReport.swift` mirrors the worker-support
/// definitions, and that file's own header claims this parity test guards it),
/// so the guard has to reach every nested level to actually be that guard.
@Test
func trustedReplayReportKeySetsEqualTheWorkerReplayReportKeySets() {
    #expect(
        Set(MLXFastRuntimeWorkerSupport.CohortReferenceReplayReport.CodingKeys
            .allCases.map(\.rawValue))
            == Set(MLXFastHarness.CohortReferenceReplayReport.CodingKeys
                .allCases.map(\.rawValue)))
    #expect(
        Set(MLXFastRuntimeWorkerSupport.CohortReferenceReplayStreamReport.CodingKeys
            .allCases.map(\.rawValue))
            == Set(MLXFastHarness.CohortReferenceReplayStreamReport.CodingKeys
                .allCases.map(\.rawValue)))
    #expect(
        Set(MLXFastRuntimeWorkerSupport.CohortReferenceReplayPosition.CodingKeys
            .allCases.map(\.rawValue))
            == Set(MLXFastHarness.CohortReferenceReplayPosition.CodingKeys
                .allCases.map(\.rawValue)))
    // Stated positively as well, so a field DROPPED from both twins at once —
    // which set equality alone would happily accept — still reds here.
    #expect(
        Set(MLXFastHarness.CohortReferenceReplayReport.CodingKeys.allCases
            .map(\.rawValue))
            == ["logit_provenance", "logit_topk", "rel_envelope", "replay_width",
                "streams"])
}

/// The report crosses the wire intact, stamp included: built with the WORKER's
/// type and encoder, decoded by the TRUSTED mirror. The key-set test above
/// compares declarations; this one proves the bytes actually round-trip, which
/// is what the trusted parent does with a pinned reference worker's reply.
@Test
func trustedDecoderParsesACohortReferenceReplayReportIncludingTheWidthStamp() throws {
    func report(width: String) -> MLXFastRuntimeWorkerSupport.CohortReferenceReplayReport {
        MLXFastRuntimeWorkerSupport.CohortReferenceReplayReport(
            logitProvenance: "post_softcap",
            logitTopK: 2,
            relEnvelope: 0.05,
            replayWidth: width,
            streams: [
                MLXFastRuntimeWorkerSupport.CohortReferenceReplayStreamReport(
                    slot: 0,
                    positions: [
                        MLXFastRuntimeWorkerSupport.CohortReferenceReplayPosition(
                            committedToken: 5,
                            sequentialArgmax: 5,
                            rankedTokens: [5, 9],
                            rankedLogits: [3.5, 2.0],
                            rankedRelativeGaps: [0, 0.4285714285714286],
                            committedTokenLogit: 3.5,
                            committedRelativeGap: 0,
                            withinEnvelopeDepth: 1)
                    ])
            ])
    }
    // Both legal widths survive the crossing.
    for width in ["cohort", "canonical"] {
        let decoded = try crossDecode(
            WorkerResponse(id: 1, nonce: "n", ok: true, cohortReferenceReplay: report(width: width)))
        let crossed = try #require(decoded.cohortReferenceReplay)
        #expect(crossed.replayWidth == width)
        #expect(crossed.logitProvenance == "post_softcap")
        #expect(crossed.streams.first?.positions.first?.committedToken == 5)
    }
}

// MARK: - Gated-off: the v1 surface must parse (unchanged native path)

@Test
func trustedDecoderParsesGatedOffGenericHello() throws {
    // What runWorker emits WITHOUT --speculative-protocol: id/nonce/ok +
    // expert_stats + the three ungated v1 identity fields, and NO v1.1 fields.
    let hello = WorkerResponse(
        id: 0,
        nonce: "session-nonce",
        ok: true,
        expertStats: .zero,
        protocolVersion: MLXFastRuntimeWorkerSupport.runtimeWorkerProtocolVersion,
        backend: MLXFastRuntimeWorkerSupport.runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )
    let decoded = try crossDecode(hello)
    #expect(decoded.id == 0)
    #expect(decoded.ok)
    #expect(decoded.protocolVersion == 1)
    #expect(decoded.backend == "mlx")
    #expect(decoded.device == "Apple Mock GPU")
    // The v1.1 surface is absent on a gated-off hello.
    #expect(decoded.specModes == nil)
    #expect(decoded.capabilities == nil)
    #expect(decoded.headProvenance == nil)
}

@Test
func trustedDecoderParsesGatedOffMTPHelloAndBeginAndRound() throws {
    // Native mtp-runtime-worker gated-off: hello carries no spec_modes /
    // head_provenance / effective_spec; the round carries the MTP ledger fields.
    let hello = WorkerResponse(
        id: 0,
        nonce: "n",
        ok: true,
        protocolVersion: MLXFastRuntimeWorkerSupport.runtimeWorkerProtocolVersion,
        backend: MLXFastRuntimeWorkerSupport.runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )
    #expect(try crossDecode(hello).protocolVersion == 1)

    let begin = WorkerResponse(id: 1, nonce: "n", ok: true, seedToken: 42)
    let decodedBegin = try crossDecode(begin)
    #expect(decodedBegin.seedToken == 42)
    #expect(decodedBegin.effectiveSpec == nil)

    let round = WorkerResponse(
        id: 2,
        nonce: "n",
        ok: true,
        tokens: [10, 11, 12],
        declaredRows: 3,
        perRowTop2Tokens: [[1, 2], [3, 4], [5, 6]],
        perRowTop2Logits: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        draftTokens: [11, 12],
        acceptedDraftCount: 2,
        rejectedDraftCount: 0,
        targetCacheOffset: 515
    )
    let decodedRound = try crossDecode(round)
    #expect(decodedRound.tokens == [10, 11, 12])
    #expect(decodedRound.declaredRows == 3)
    #expect(decodedRound.targetCacheOffset == 515)
}

@Test
func trustedDecoderParsesGatedOffDecodeAndPhaseDiagnostics() throws {
    let decodeBegin = WorkerResponse(id: 1, nonce: "n", ok: true, seedToken: 7)
    #expect(try crossDecode(decodeBegin).seedToken == 7)

    let decodeStep = WorkerResponse(id: 2, nonce: "n", ok: true, token: 99)
    #expect(try crossDecode(decodeStep).token == 99)

    // phase_diagnostics carries completed_work / cache_memory — fields the trusted
    // decoder previously did not know (a gated-off native phase would still send
    // them). They must parse.
    let phase = WorkerResponse(
        id: 3,
        nonce: "n",
        ok: true,
        expertStats: .zero,
        peakRamGB: 1.25,
        mlxActiveMemoryBytes: 1,
        mlxCacheMemoryBytes: 0,
        mlxPeakMemoryBytes: 2,
        completedWork: 130,
        cacheMemory: 0
    )
    let decodedPhase = try crossDecode(phase)
    #expect(decodedPhase.completedWork == 130)
    #expect(decodedPhase.cacheMemory == 0)

    let failure = WorkerResponse(id: 4, nonce: "n", ok: false, error: "boom")
    let decodedFailure = try crossDecode(failure)
    #expect(!decodedFailure.ok)
    #expect(decodedFailure.error == "boom")
}

// MARK: - Gated-on: the v1.1 surface must ALSO parse (superset decoder)

@Test
func trustedDecoderParsesGatedOnGenericHello() throws {
    // What runWorker emits WITH --speculative-protocol v1.1: the v1 surface PLUS
    // spec_modes / capabilities / head_provenance. These are the exact fields the
    // trusted decoder used to reject, breaking every worker-spawning verb.
    let registry = RuntimeWorkerSpecRegistry.serialOnlyWorker
    let hello = WorkerResponse(
        id: 0,
        nonce: "n",
        ok: true,
        expertStats: .zero,
        specModes: registry.advertisedModeStrings,
        headProvenance: RuntimeWorkerHeadProvenance(
            sha256: String(repeating: "a", count: 64),
            bytes: 849_398_784,
            fileCount: 2
        ),
        capabilities: [runtimeWorkerFreeRunDecodeCapability],
        protocolVersion: MLXFastRuntimeWorkerSupport.runtimeWorkerProtocolVersion,
        backend: MLXFastRuntimeWorkerSupport.runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )
    let decoded = try crossDecode(hello)
    #expect(decoded.protocolVersion == 1)
    #expect(decoded.specModes == ["serial"])
    #expect(decoded.capabilities == ["free_run_decode"])
    #expect(decoded.headProvenance?.sha256 == String(repeating: "a", count: 64))
    #expect(decoded.headProvenance?.bytes == 849_398_784)
    #expect(decoded.headProvenance?.fileCount == 2)
}

/// A hello that advertises spec_modes and head_provenance but NO capabilities.
/// The shape used to belong to the native MTP worker; it is kept because the
/// property under test is the trusted decoder's tolerance of an ABSENT
/// `capabilities` array alongside present v1.1 objects, which is independent of
/// which worker emits it.
@Test
func trustedDecoderParsesAGatedOnHelloWithoutCapabilities() throws {
    let registry = RuntimeWorkerSpecRegistry.serialOnlyWorker
    let hello = WorkerResponse(
        id: 0,
        nonce: "n",
        ok: true,
        specModes: registry.advertisedModeStrings,
        headProvenance: RuntimeWorkerHeadProvenance(
            sha256: String(repeating: "b", count: 64),
            bytes: 10,
            fileCount: 1
        ),
        protocolVersion: MLXFastRuntimeWorkerSupport.runtimeWorkerProtocolVersion,
        backend: MLXFastRuntimeWorkerSupport.runtimeWorkerBackendLabel,
        device: "Apple Mock GPU"
    )
    let decoded = try crossDecode(hello)
    #expect(decoded.specModes == ["serial"])
    #expect(decoded.capabilities == nil)
    #expect(decoded.headProvenance?.fileCount == 1)
}

@Test
func trustedDecoderParsesGatedOnEffectiveSpecEchoes() throws {
    // decode_begin / free_decode_begin gated-on serial echo.
    let serialEcho = WorkerResponse(
        id: 1,
        nonce: "n",
        ok: true,
        seedToken: 42,
        effectiveSpec: .serial()
    )
    let decodedSerial = try crossDecode(serialEcho)
    #expect(decodedSerial.effectiveSpec?.mode == "serial")
    #expect(decodedSerial.effectiveSpec?.mtp == nil)
}

/// The trusted decoder's mirror (`TrustedWorkerEffectiveSpec`) still declares an
/// `mtp` block even though this engine can no longer PRODUCE one, so this half
/// of the cross-decode is driven from raw bytes rather than through
/// `crossDecode`. That is the point: the trusted decoder is deliberately a
/// SUPERSET of what any one worker emits, and its job at a hello is to not
/// choke on a field it does not consume. Narrowing it to serial-only would make
/// the arm's return a decoder break instead of an additive change.
@Test
func trustedDecoderStillToleratesAnMTPEffectiveSpecItCannotProduce() throws {
    let line = Data(#"{"id":1,"nonce":"n","ok":true,"seed_token":42,"effective_spec":{"mode":"mtp","mtp":{"depth":3}}}"#.utf8)
    let decoded = try JSONDecoder().decode(TrustedResponse.self, from: line)
    #expect(decoded.effectiveSpec?.mode == "mtp")
    #expect(decoded.effectiveSpec?.mtp?.depth == 3)
}

@Test
func trustedDecoderParsesGatedOnFreeDecodeRun() throws {
    // free_decode_run AUDIT counters (acceptance_lengths + the three totals).
    let run = WorkerResponse(
        id: 2,
        nonce: "n",
        ok: true,
        tokens: [1, 2, 3, 4],
        acceptanceLengths: [1, 2, 1],
        draftedTotal: 3,
        acceptedTotal: 2,
        committedTotal: 4
    )
    let decoded = try crossDecode(run)
    #expect(decoded.tokens == [1, 2, 3, 4])
    #expect(decoded.acceptanceLengths == [1, 2, 1])
    #expect(decoded.draftedTotal == 3)
    #expect(decoded.acceptedTotal == 2)
    #expect(decoded.committedTotal == 4)
}
