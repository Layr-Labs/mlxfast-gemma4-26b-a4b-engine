import Foundation
import MLX
import MLXFastCore
import MLXLLM
@testable import MLXLMCommon
import MLXRandom
import MLXSpeculative
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Conformance for the per-module speculative-config surface (David-ruled
// 2026-08-19): spec parse per mode, effective_spec echo, spec_modes advertising
// runnable-only, and the fail-closed negatives — dspark stub, capability-absent
// dflash, fullwidth-hex, cross-module keys, unknown fields.
//
// The `mtp` cases here were DELETED, not disabled, with the harness-port
// increment (2026-08-22) and RESTORED, re-derived rather than un-deleted, with
// the MTP arm (2026-08-23, Gemma 4 26B A4B): `RuntimeWorkerMTPBlock`,
// `EffectiveMTP`, and `Gemma4MTPEnvelope.resolveDepth` are new types/functions,
// not the Qwen-era originals. `RuntimeWorkerSpecRegistry.serialOnlyWorker` (the
// no-head-loaded worker) still refuses `mtp` at RESOLUTION — see
// `mtpRealModuleIsRegisteredButNotRunnableWithoutAHead` below — which is the
// distinction that outlived the harness-port increment's stricter
// refused-at-PARSE posture: `mtp` is a real module again (parses, validates,
// echoes), so a headless worker's refusal now reads as "not runnable HERE"
// (dflash's shape), not "not a mode this engine knows".

private func decodeSpec(_ json: String) throws -> RuntimeWorkerSpecRequest {
    try JSONDecoder().decode(
        RuntimeWorkerSpecRequest.self, from: Data(json.utf8))
}

private func encodeEffective(_ spec: RuntimeWorkerEffectiveSpec) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(spec), as: UTF8.self)
}

// MARK: - Parse per mode

@Test
func specParsesSerial() throws {
    let spec = try decodeSpec(#"{"mode":"serial"}"#)
    #expect(spec.mode == .serial)
    #expect(spec.dflash == nil)
    #expect(spec.dspark == nil)
}

@Test
func specParsesMTP() throws {
    let spec = try decodeSpec(#"{"mode":"mtp","mtp":{"depth":2}}"#)
    #expect(spec.mode == .mtp)
    #expect(spec.mtp?.depth == 2)
    #expect(spec.dflash == nil)
    #expect(spec.dspark == nil)
}

@Test
func specParsesMTPWithNoBlockLeavesDepthNil() throws {
    let spec = try decodeSpec(#"{"mode":"mtp"}"#)
    #expect(spec.mode == .mtp)
    #expect(spec.mtp == nil)
}

/// Cross-module rejection extends to `mtp`: a spec declaring `mtp` may not
/// also carry a `dflash`/`dspark` block, and a spec declaring another mode
/// may not carry an `mtp` block — the SAME generic guard `RuntimeWorkerSpecRequest.init(from:)`
/// already applies to every other mode, exercised here because `mtp` is the
/// newest module in the union and the guard is generic code any module could
/// silently stop reaching.
@Test
func mtpBlockIsRejectedUnderAnotherMode() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"serial","mtp":{"depth":1}}"#)
    }
}

@Test
func mtpModeRejectsAForeignBlock() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"mtp","dflash":{"depth":1}}"#)
    }
}

/// `mtp` is a REAL module again (`RuntimeWorkerSpecModuleTable`), so it is
/// well-formed on the wire everywhere — but `RuntimeWorkerSpecRegistry.serialOnlyWorker`
/// (a worker that loaded no assistant head) still refuses it at RESOLUTION,
/// the same "not runnable on this engine" shape `dflash` already produces on
/// this engine (capability-absent real module), never a silent downgrade to
/// serial.
@Test
func mtpRealModuleIsRegisteredButNotRunnableWithoutAHead() throws {
    let spec = try decodeSpec(#"{"mode":"mtp","mtp":{"depth":2}}"#)
    #expect(throws: (any Error).self) {
        _ = try RuntimeWorkerSpecRegistry.serialOnlyWorker.resolveEffectiveSpec(spec)
    }
}

/// `RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable: true)` — the shape a
/// worker with a loaded head actually runs with — resolves `mtp`, clamps
/// depth into `Gemma4MTPEnvelope`'s pinned range, and echoes
/// `{"mtp":{"depth":...}}`. A `nil` request depth resolves to the pinned
/// ceiling (3); an over-large request clamps down to it, never up.
@Test
func mtpResolvesAndEchoesTheClampedDepthOnAHeadCapableWorker() throws {
    let registry = RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable: true)
    #expect(registry.advertisedModeStrings.contains("mtp"))

    let withNoDepth = try decodeSpec(#"{"mode":"mtp"}"#)
    let effectiveNoDepth = try registry.resolveEffectiveSpec(withNoDepth)
    #expect(effectiveNoDepth.mode == "mtp")
    #expect(effectiveNoDepth.mtp?.depth == Gemma4MTPEnvelope.maxDraftTokens)

    let withHugeDepth = try decodeSpec(#"{"mode":"mtp","mtp":{"depth":999}}"#)
    let effectiveHuge = try registry.resolveEffectiveSpec(withHugeDepth)
    #expect(effectiveHuge.mtp?.depth == Gemma4MTPEnvelope.maxDraftTokens)

    let withZeroDepth = try decodeSpec(#"{"mode":"mtp","mtp":{"depth":0}}"#)
    let effectiveZero = try registry.resolveEffectiveSpec(withZeroDepth)
    #expect(effectiveZero.mtp?.depth == 0)
}

/// A worker with no request spec at all still defaults to serial even when it
/// CAN run mtp — MTP is opt-in per request, never a silent default (the
/// serial CONTROL leg's contract must not change shape depending on worker
/// capability).
@Test
func mtpCapableWorkerStillDefaultsToSerialWithNoSpec() throws {
    let registry = RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable: true)
    let effective = try registry.resolveEffectiveSpec(nil)
    #expect(effective.mode == "serial")
}

@Test
func specParsesDFlashWithDraft() throws {
    let sha = String(repeating: "a", count: 64)
    let spec = try decodeSpec(
        #"{"mode":"dflash","dflash":{"depth":4,"draft":{"artifact":"hf:x@1","sha256":"\#(sha)"}}}"#
    )
    #expect(spec.mode == .dflash)
    #expect(spec.dflash?.depth == 4)
    #expect(spec.dflash?.draft?.artifact == "hf:x@1")
    #expect(spec.dflash?.draft?.sha256 == sha)
}

@Test
func specParsesDSparkReservedEmptyBlock() throws {
    // The stub still parses+validates its (currently empty) block.
    let spec = try decodeSpec(#"{"mode":"dspark","dspark":{}}"#)
    #expect(spec.mode == .dspark)
    #expect(spec.dspark != nil)
}

@Test
func specParsesInsideAFullWorkerRequest() throws {
    let sha = String(repeating: "c", count: 64)
    let request = try JSONDecoder().decode(
        RuntimeWorkerRequest.self,
        from: Data(
            #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2],"spec":{"mode":"dflash","dflash":{"depth":5,"draft":{"artifact":"hf:x@1","sha256":"\#(sha)"}}}}"#
                .utf8)
    )
    #expect(request.spec?.mode == .dflash)
    #expect(request.spec?.dflash?.depth == 5)
}

// MARK: - Effective echo

@Test
func serialEchoIsModeOnly() throws {
    let effective = try RuntimeWorkerSpecRegistry.serialOnlyWorker
        .resolveEffectiveSpec(decodeSpec(#"{"mode":"serial"}"#))
    #expect(effective.mode == "serial")
    #expect(try encodeEffective(effective) == #"{"mode":"serial"}"#)
}

@Test
func absentSpecResolvesToTheEngineDefault() throws {
    // v1 callers send no spec; the engine default echoes serial and never
    // throws.
    let effective = try RuntimeWorkerSpecRegistry.serialOnlyWorker
        .resolveEffectiveSpec(nil)
    #expect(effective.mode == "serial")
    #expect(try encodeEffective(effective) == #"{"mode":"serial"}"#)
}

// MARK: - spec_modes advertises runnable-only

@Test
func specModesAdvertisesRunnableOnly() {
    let modes = RuntimeWorkerSpecRegistry.serialOnlyWorker.advertisedModeStrings
    #expect(modes == ["serial"])
    // Stubs and capability-absent real modes are never advertised.
    #expect(!modes.contains("dspark"))
    #expect(!modes.contains("dflash"))
}

// MARK: - Fail-closed negatives

@Test
func dsparkStubFailsClosedWithNotImplemented() throws {
    let spec = try decodeSpec(#"{"mode":"dspark"}"#)
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try RuntimeWorkerSpecRegistry.serialOnlyWorker
                .resolveEffectiveSpec(spec)
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    #expect(caught?.contains("not implemented on this engine") == true)
}

@Test
func dflashRealModuleIsNotRunnableOnThisEngine() throws {
    let sha = String(repeating: "b", count: 64)
    let spec = try decodeSpec(
        #"{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(sha)"}}}"#)
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try RuntimeWorkerSpecRegistry.serialOnlyWorker
                .resolveEffectiveSpec(spec)
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    // Distinct from the stub error: dflash is real, merely capability-absent.
    #expect(caught?.contains("not runnable on this engine") == true)
}

@Test
func fullwidthHexSHA256IsRejectedAtParse() {
    // U+FF10.. fullwidth digits: 64 of them look like hex but are not ASCII.
    let fullwidth = String(repeating: "\u{FF10}", count: 64)
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(
            #"{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(fullwidth)"}}}"#)
    }
}

@Test
func shortSHA256IsRejected() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(
            #"{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"abc"}}}"#)
    }
}

@Test
func crossModuleKeyIsRejected() {
    // serial mode carrying a dflash block is config drift, not an ignored field.
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"serial","dflash":{"depth":2}}"#)
    }
}

@Test
func unknownTopLevelFieldIsRejected() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"serial","bogus":1}"#)
    }
}

@Test
func unknownFieldInsideAModuleBlockIsRejected() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"dflash","dflash":{"depth":2,"bogus":1}}"#)
    }
}

@Test
func unknownModeIsRejected() {
    #expect(throws: (any Error).self) {
        _ = try decodeSpec(#"{"mode":"teleport"}"#)
    }
}

// MARK: - Head provenance seal
//
// The WIRE SHAPE is pinned here; the tree-digest COMPUTATION
// (`computeGemma4AssistantHeadProvenance`, Gemma4A4BAssistantHead.swift) is
// MLX-adjacent (imports MLXLLM/MLXLMCommon for the drafter/target types its
// caller binds against) and is exercised by real-weights box-gated tests, not
// here. This engine sends null when no head loaded and the harness's own
// recomputed digest once one does (2026-08-23) — no longer unconditionally
// null.

@Test
func headProvenanceSerializesWithSnakeCaseFileCount() throws {
    let provenance = RuntimeWorkerHeadProvenance(
        sha256: String(repeating: "0", count: 64), bytes: 10, fileCount: 3)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(decoding: try encoder.encode(provenance), as: UTF8.self)
    #expect(json.contains("\"file_count\":3"))
    #expect(json.contains("\"bytes\":10"))
}

// MARK: - Gemma4MTPEnvelope (the envelope-knob trust boundary)
//
// Declaration + seal + refusal, exercised directly. `requirePinned(rectangularCap:queryBlockPin:)`
// takes the two pins as parameters (defaulted to the real constants) precisely
// so these refusal branches are reachable from a test without mutating
// production `static let`s.

@Test
func envelopeRefusesWhenTheRectangularCapIsUnset() {
    for zeroOrNegative in [0, -1] {
        #expect(throws: Gemma4MTPEnvelope.EnvelopeError.rectangularCapUnset) {
            try Gemma4MTPEnvelope.requirePinned(
                rectangularCap: zeroOrNegative, queryBlockPin: 128)
        }
    }
}

@Test
func envelopeRefusesWhenTheQueryBlockPinIsUnset() {
    #expect(throws: Gemma4MTPEnvelope.EnvelopeError.queryBlockPinUnset) {
        try Gemma4MTPEnvelope.requirePinned(rectangularCap: 32, queryBlockPin: nil)
    }
    for zeroOrNegative in [0, -1] {
        #expect(throws: Gemma4MTPEnvelope.EnvelopeError.queryBlockPinUnset) {
            try Gemma4MTPEnvelope.requirePinned(
                rectangularCap: 32, queryBlockPin: zeroOrNegative)
        }
    }
}

@Test
func envelopeAcceptsBothPinsExplicitlySet() throws {
    try Gemma4MTPEnvelope.requirePinned(rectangularCap: 32, queryBlockPin: 128)
    // The real production constants must themselves pass — a test asserting
    // the refusal machinery works is not a substitute for asserting the
    // shipped values are actually valid.
    try Gemma4MTPEnvelope.requirePinned()
}

/// The relationship `MTPEnvelope.swift` documents as load-bearing, checked as
/// arithmetic rather than trusted as prose: at this track's pinned B=8,
/// `maxAutomaticRectangularTokens / 8 - 1` must equal `maxDraftTokens` exactly
/// — the vendored planner's own `CBv2MTPRoundDriver.maximumAutomaticDepth`
/// formula (`maxWidth = cap / plannedDecodeRows; depth = min(maxDraftTokens,
/// max(0, maxWidth - 1))`), reproduced here so raising one constant without
/// the other fails a test instead of silently starving the widened depths.
///
/// This is the FAST, GPU-free half of the check — pure arithmetic, no model.
/// `maximumAutomaticDepthMatchesTheRealVendoredPlannerAtEveryPlannedRowCount`
/// below is the review-ledgered follow-up: it calls the REAL
/// `CBv2MTPRoundDriver.maximumAutomaticDepth` (not a re-implementation of its
/// formula) through a real, tiny fixture driver, so the two tests
/// cross-verify each other rather than one merely restating the other's
/// assumption.
@Test
func rectangularCapAndDraftDepthAreTheSameEnvelopeOnTwoAxes() {
    let plannedDecodeRows = 8  // this track's pinned B
    let maxWidth = Gemma4MTPEnvelope.maxAutomaticRectangularTokens / plannedDecodeRows
    let vendoredPlannerDepth = max(0, maxWidth - 1)
    #expect(vendoredPlannerDepth == Gemma4MTPEnvelope.maxDraftTokens)
    #expect(Gemma4MTPEnvelope.maxDraftTokens == 3)
    #expect(Gemma4MTPEnvelope.maxAutomaticRectangularTokens == 32)
}

/// The REAL vendored function, not a re-implementation of its formula:
/// `CBv2MTPRoundDriver.maximumAutomaticDepth(plannedDecodeRows:)`, called on a
/// driver built via `CBv2MTPRoundDriver.build(model:drafter:config:)` from a
/// tiny random-init Gemma 4 fixture (the same weight-free fixture shape
/// `RuntimeWorkerMTPRoundExecutionTests.swift` and the vendored
/// `CBv2MTPRoundSmokeTests` use — no real weights, no network, no box),
/// CONFIGURED WITH THIS TRACK'S REAL PRODUCTION PINS
/// (`Gemma4MTPEnvelope.maxDraftTokens` / `.maxAutomaticRectangularTokens`).
/// Checks the exact three cases the round-execution mission named: at cap 32
/// / maxDraftTokens 3, planned decode rows 8, 2, and 1 all resolve to exactly
/// depth 3 — `maxDraftTokens` is the binding constraint at every one of this
/// track's admitted batch widths (`32 / rows - 1 >= 3` for rows in 1...8), not
/// merely at the pinned B=8 the arithmetic test above checks alone.
///
/// Requires `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` (constructing `Gemma4TextModel`
/// allocates real MLXArrays) and, once per checkout,
/// `tools/build-mlx-metallib.sh --all-build-roots` — this repo's standing
/// convention for any MLX-touching test (see RuntimeWorkerSupportTests.swift).
@Test
func maximumAutomaticDepthMatchesTheRealVendoredPlannerAtEveryPlannedRowCount() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    let vocabSize = 32
    let hiddenSize = 16
    let targetJSON = """
        {"model_type":"gemma4_text","hidden_size":\(hiddenSize),"num_hidden_layers":4,
         "intermediate_size":32,"num_attention_heads":2,"head_dim":8,"global_head_dim":8,
         "num_key_value_heads":1,"num_kv_shared_layers":0,
         "layer_types":["sliding_attention","full_attention","full_attention","sliding_attention"],
         "sliding_window":8,"final_logit_softcapping":30.0,"tie_word_embeddings":true,
         "vocab_size":\(vocabSize),"vocab_size_per_layer_input":\(vocabSize),"rms_norm_eps":1e-6,
         "hidden_size_per_layer_input":0,"use_double_wide_mlp":false}
        """
    let targetConfig = try JSONDecoder.json5().decode(
        Gemma4TextConfiguration.self, from: Data(targetJSON.utf8))
    let drafterJSON = """
        {"model_type":"gemma4_assistant","backbone_hidden_size":\(hiddenSize),
         "use_ordered_embeddings":false,"num_centroids":4,"centroid_intermediate_top_k":2,
         "text_config":{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":1,
         "intermediate_size":16,"num_attention_heads":2,"head_dim":8,"global_head_dim":8,
         "num_key_value_heads":1,"num_kv_shared_layers":0,
         "layer_types":["sliding_attention"],"sliding_window":8,
         "final_logit_softcapping":null,"tie_word_embeddings":true,"vocab_size":\(vocabSize),
         "vocab_size_per_layer_input":\(vocabSize),"rms_norm_eps":1e-6,
         "hidden_size_per_layer_input":0,"use_double_wide_mlp":false}}
        """
    let drafterConfig = try JSONDecoder.json5().decode(
        Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8))
    MLXRandom.seed(0x0EED_0BE7)
    let target = Gemma4TextModel(targetConfig)
    let drafter = try Gemma4AssistantDraftModel(config: drafterConfig)
    eval(target, drafter)
    let mtpDrafter = try Gemma4CBv2MTPDrafter(drafter: drafter, target: target)
    let config = CBv2MTPConfig(
        enabled: true,
        maxDraftTokens: Gemma4MTPEnvelope.maxDraftTokens,
        maxSpeculativeBatch: 8,
        verificationMode: .automatic,
        maxAutomaticRectangularTokens: Gemma4MTPEnvelope.maxAutomaticRectangularTokens)
    let driver = try #require(
        CBv2MTPRoundDriver.build(
            model: CBv2SteppableLanguageModelAdapter(target), drafter: mtpDrafter, config: config),
        """
        the fixture drafter/target must satisfy the same capture-layer/target-identity \
        contract the real assistant head does, or this test proves nothing
        """)
    for plannedDecodeRows in [8, 2, 1] {
        #expect(
            driver.maximumAutomaticDepth(plannedDecodeRows: plannedDecodeRows) == 3,
            "plannedDecodeRows=\(plannedDecodeRows)")
    }
}

@Test
func resolveDepthClampsIntoZeroToMaxDraftTokens() {
    #expect(Gemma4MTPEnvelope.resolveDepth(nil) == Gemma4MTPEnvelope.maxDraftTokens)
    #expect(Gemma4MTPEnvelope.resolveDepth(-5) == 0)
    #expect(Gemma4MTPEnvelope.resolveDepth(0) == 0)
    #expect(Gemma4MTPEnvelope.resolveDepth(2) == 2)
    #expect(Gemma4MTPEnvelope.resolveDepth(999) == Gemma4MTPEnvelope.maxDraftTokens)
}

// MARK: - DFlash arm: capability, echo, and the broken-head refusal
//
// (gemma4-dflash-real-loader lane, 2026-08-25 — the real `DFlashDraftModel`
// loader. The registry now carries the DFlash capability as a VALUE, which
// is what lets the echo state the depth that will run and the digest of the
// drafter that will run it.)

private func dflashWorker(
    maxDepth: Int = 15,
    sha256: String = String(repeating: "c", count: 64)
) -> RuntimeWorkerSpecRegistry {
    RuntimeWorkerSpecRegistry.gemma4Worker(
        mtpAvailable: false,
        dflash: .available(maxDepth: maxDepth, provenanceSHA256: sha256))
}

@Test
func dflashIsAdvertisedOnlyWhenADrafterIsBound() {
    #expect(dflashWorker().advertisedModeStrings == ["serial", "dflash"])
    #expect(
        RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable: false, dflash: .absent)
            .advertisedModeStrings == ["serial"])
    #expect(
        RuntimeWorkerSpecRegistry.gemma4Worker(
            mtpAvailable: false, dflash: .broken(reason: "bad config")
        ).advertisedModeStrings == ["serial"])
}

/// A PRESENT-but-unloadable `dflash-head/` is capability-absent at startup
/// (fail-soft: the worker must still reach hello) and refuses HERE, naming
/// the underlying load failure rather than the generic shape.
@Test
func brokenDFlashHeadRefusesAtResolutionNamingTheLoadFailure() throws {
    let reason =
        "invalid value at 'dflash_config': DFlash target layer id 44 is outside 0..<30."
    let registry = RuntimeWorkerSpecRegistry.gemma4Worker(
        mtpAvailable: false, dflash: .broken(reason: reason))
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try registry.resolveEffectiveSpec(try decodeSpec(#"{"mode":"dflash"}"#))
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    #expect(caught?.contains("dflash-head/ IS staged but could not be loaded") == true)
    #expect(caught?.contains("target layer id 44 is outside") == true)
}

/// The depth echo is the depth the round loop runs — clamped by the BOUND
/// drafter's block, never an unclamped module default. (#38 echoed
/// `experimentalDFlashMaxBlockSize` = 16 while executing an envelope-clamped
/// 3.)
@Test
func dflashDepthEchoIsClampedByTheBoundDrafter() throws {
    let registry = dflashWorker(maxDepth: 15)
    let unspecified = try registry.resolveEffectiveSpec(
        try decodeSpec(#"{"mode":"dflash"}"#))
    #expect(unspecified.dflash?.depth == 15)

    let overLarge = try registry.resolveEffectiveSpec(
        try decodeSpec(#"{"mode":"dflash","dflash":{"depth":64}}"#))
    #expect(overLarge.dflash?.depth == 15)

    let narrow = try dflashWorker(maxDepth: 3).resolveEffectiveSpec(
        try decodeSpec(#"{"mode":"dflash","dflash":{"depth":9}}"#))
    #expect(narrow.dflash?.depth == 3)
    #expect(narrow.mode == "dflash")
}

/// The echoed draft identity is the harness's own digest of the bound
/// drafter, never a caller-supplied string; a caller that declares a
/// different digest is refused rather than echoed back.
@Test
func dflashDraftEchoIsTheBoundDrafterDigest() throws {
    let bound = String(repeating: "c", count: 64)
    let registry = dflashWorker(sha256: bound)

    let noDeclaration = try registry.resolveEffectiveSpec(
        try decodeSpec(#"{"mode":"dflash"}"#))
    #expect(noDeclaration.dflash?.draft?.sha256 == bound)
    #expect(noDeclaration.dflash?.draft?.artifact == gemma4DFlashHeadDirectoryName)

    let matching = try registry.resolveEffectiveSpec(
        try decodeSpec(
            #"{"mode":"dflash","dflash":{"draft":{"artifact":"whatever","sha256":"\#(bound)"}}}"#))
    #expect(matching.dflash?.draft?.sha256 == bound)
    // The caller's artifact LABEL never survives either — the echo names the
    // directory the drafter was actually loaded from.
    #expect(matching.dflash?.draft?.artifact == gemma4DFlashHeadDirectoryName)

    let other = String(repeating: "d", count: 64)
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try registry.resolveEffectiveSpec(
                try decodeSpec(
                    #"{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(other)"}}}"#))
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    #expect(caught?.contains("refusing rather than") == true)
}

/// The DFlash load-failure renderer must surface the DecodingError's own
/// context message. `DecodingError.localizedDescription` is the useless
/// "The data couldn't be read because it isn't in the correct format."
@Test
func dflashLoadFailureDescriptionSurfacesTheDecodingErrorDetail() throws {
    let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "block_size": 4,
            "num_target_layers": 3,
            "layer_types": ["full_attention", "full_attention"],
            "dflash_config": { "target_layer_ids": [0, 7], "mask_token_id": 4 }
        }
        """
    var described: String?
    do {
        _ = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
    } catch {
        described = describeGemma4DFlashLoadFailure(error)
        // Precondition: the raw localizedDescription is the useless one.
        #expect(error.localizedDescription.contains("target layer id") == false)
    }
    #expect(described?.contains("DFlash target layer id 7 is outside 0..<3") == true)
    #expect(described?.contains("dflash_config") == true)
}
