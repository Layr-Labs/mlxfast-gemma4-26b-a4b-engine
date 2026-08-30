import Foundation
import MLXFastCore
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Coverage for the six wire guards of the generic benchd-facing worker
// (review finding (2) on PR #5). Every one of these used to live inline in
// `handleWorkerRequest`, whose signature demands a 21.6 GB weight cache — so
// the port-requirement probes below had only ever been run by hand against a
// live worker on the box, never by a test. They are pure decisions over decoded
// wire data, and `validateGenericWorkerRequest` is where they live now.
//
// The probe inputs are the ones from the live-box probe run: fullwidth
// U+FF10/U+FF41, an embedded NUL, a non-serial TF spec, a cross-kind spec,
// count 0 / 1537, and a post-poison request.
//
// The `mtp` probes went with the MTP arm (2026-08-22, Gemma 4 26B A4B harness
// port): the hostile-depth inputs (1e20, Int.max, 0, -3) probed
// `resolveMTPDepth`, which no longer exists, and the declared-ceiling section
// probed `enforceQwenMTPDeclaredSpec`, likewise gone. They are DELETED rather
// than disabled because a `.disabled` reason cannot name a symbol that is not
// there. The hostile-sha probes survive unchanged — they were always the
// dflash block's, and that block is still parsed and validated.

private let gateOn = RuntimeWorkerRequestContext(advertisesSpeculativeProtocol: true)
private let registry = RuntimeWorkerSpecRegistry.serialOnlyWorker

private func request(_ json: String) throws -> RuntimeWorkerRequest {
    try JSONDecoder().decode(RuntimeWorkerRequest.self, from: Data(json.utf8))
}

/// Validate and return the thrown error's message, or nil when it succeeded.
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

/// Validate a request that must be ACCEPTED, returning what it resolved to.
private func accepted(
    _ json: String,
    context: RuntimeWorkerRequestContext = gateOn
) throws -> RuntimeWorkerValidatedRequest {
    try validateGenericWorkerRequest(
        try request(json), context: context, specRegistry: registry)
}

// MARK: - Guard 1: poison is sticky and serves nothing

@Test
func poisonedSessionRejectsEveryFurtherRequest() throws {
    let poisoned = RuntimeWorkerRequestContext(
        poisoned: true, hasDecodeRoute: true, advertisesSpeculativeProtocol: true)
    // Post-poison probe: a request that would otherwise be perfectly valid.
    for json in [
        #"{"id":9,"kind":"free_decode_run","count":8}"#,
        #"{"id":9,"kind":"decode_begin","seed_tokens":[1,2]}"#,
        #"{"id":9,"kind":"phase_diagnostics"}"#,
    ] {
        #expect(
            rejection(json, context: poisoned)
                == "the generic decode session is poisoned after an earlier failure")
    }
    // The same requests pass on a clean session.
    #expect(rejection(#"{"id":9,"kind":"phase_diagnostics"}"#) == nil)
}

// MARK: - Guard 2: the spawn gate

@Test
func specRequiresTheSpawnGate() throws {
    let gateOff = RuntimeWorkerRequestContext(advertisesSpeculativeProtocol: false)
    let message = rejection(
        #"{"id":1,"kind":"decode_begin","seed_tokens":[1],"spec":{"mode":"serial"}}"#,
        context: gateOff)
    #expect(message?.contains("did not advertise the speculative protocol at spawn") == true)
    #expect(message?.contains(runtimeWorkerSpeculativeProtocolFlag) == true)
    // Gated off, a request WITHOUT a spec is the untouched v1 path.
    #expect(rejection(#"{"id":1,"kind":"decode_begin","seed_tokens":[1]}"#, context: gateOff) == nil)
}

// MARK: - Guard 3: cross-kind spec rejection

@Test
func specIsRejectedOnEveryKindButTheTwoDecodeBegins() throws {
    for kind in [
        "correctness", "correctness_begin", "correctness_step", "prefill",
        "decode_step", "free_decode_run", "free_decode_run_timed",
        "free_decode_finalize", "phase_diagnostics",
    ] {
        let message = rejection(
            #"{"id":1,"kind":"\#(kind)","spec":{"mode":"serial"}}"#)
        #expect(
            message
                == "spec is valid only on decode_begin / free_decode_begin; "
                    + "request kind '\(kind)' must not carry one")
    }
    // The two kinds that DO carry one resolve it.
    #expect(
        try accepted(
            #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1],"spec":{"mode":"serial"}}"#
        ).effectiveSpec == .serial())
    #expect(
        try accepted(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1],"spec":{"mode":"serial"}}"#
        ).effectiveSpec == .serial())
}

// MARK: - Guard 4: the free-run kinds are gated at spawn too (#10 item 3)

@Test
func freeRunKindsAreRefusedWhenTheSpawnGateIsOff() throws {
    // #10 item 3 adjudication: the spawn gate governs SERVING the free-run
    // kinds, not only advertising `free_run_decode` on the hello. Gated off,
    // both kinds are refused — a worker must not serve a kind its own hello
    // never offered.
    let gateOff = RuntimeWorkerRequestContext(
        hasDecodeRoute: true, advertisesSpeculativeProtocol: false)
    for json in [
        #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2]}"#,
        #"{"id":2,"kind":"free_decode_run","count":8}"#,
        #"{"id":3,"kind":"free_decode_run_timed","count":8}"#,
        #"{"id":4,"kind":"free_decode_finalize"}"#,
    ] {
        let message = rejection(json, context: gateOff)
        #expect(message?.contains("belongs to the v1.1 speculative surface") == true)
        #expect(message?.contains(runtimeWorkerSpeculativeProtocolFlag) == true)
        #expect(message?.contains(runtimeWorkerFreeRunDecodeCapability) == true)
    }
    // The gate runs BEFORE the per-kind checks, so a gated-off request reports
    // the gate rather than a missing seed or an unsealed route.
    #expect(
        rejection(#"{"id":1,"kind":"free_decode_begin"}"#, context: gateOff)?
            .contains("belongs to the v1.1 speculative surface") == true)
    #expect(
        rejection(#"{"id":2,"kind":"free_decode_run","count":8}"#)
            == "runtime worker free_decode_run before free_decode_begin")

    // Gated ON, both kinds are served — this guard adds nothing to the path
    // benchd actually drives (a free-run leg is spawned WITH the flag).
    let gateOnBegun = RuntimeWorkerRequestContext(
        hasDecodeRoute: true, advertisesSpeculativeProtocol: true)
    #expect(rejection(#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2]}"#) == nil)
    #expect(
        try accepted(#"{"id":2,"kind":"free_decode_run","count":8}"#, context: gateOnBegun)
            .freeRunCount == 8)

    // The gate is scoped to the free-run kinds: the teacher-forced kinds stay
    // exactly as unchanged as the gated-off v1 path requires.
    for json in [
        #"{"id":1,"kind":"decode_begin","seed_tokens":[1,2]}"#,
        #"{"id":2,"kind":"decode_step","token":7}"#,
        #"{"id":3,"kind":"phase_diagnostics"}"#,
    ] {
        #expect(rejection(json, context: gateOff) == nil)
    }
}

@Test
func deferredFreeRunRequiresExactlyOneUntimedFinalize() throws {
    let open = RuntimeWorkerRequestContext(
        hasDecodeRoute: true, advertisesSpeculativeProtocol: true)
    #expect(
        try accepted(
            #"{"id":2,"kind":"free_decode_run_timed","count":8}"#,
            context: open
        ).freeRunCount == 8)
    #expect(
        rejection(#"{"id":3,"kind":"free_decode_finalize"}"#, context: open)
            == "runtime worker free_decode_finalize without a pending timed run")

    let pending = RuntimeWorkerRequestContext(
        hasDecodeRoute: true,
        advertisesSpeculativeProtocol: true,
        hasPendingFreeRunFinalize: true)
    #expect(rejection(#"{"id":3,"kind":"free_decode_finalize"}"#, context: pending) == nil)
    #expect(
        rejection(
            #"{"id":4,"kind":"free_decode_finalize","count":1}"#,
            context: pending)
            == "runtime worker free_decode_finalize must not carry count")
}

@Test
func pendingFinalizeBlocksEveryOtherRecognizedVerbAndUnknownKind() throws {
    let pending = RuntimeWorkerRequestContext(
        hasDecodeRoute: true,
        advertisesSpeculativeProtocol: true,
        hasPendingFreeRunFinalize: true)
    let blockedKinds = [
        "correctness",
        "correctness_begin",
        "correctness_step",
        "prefill",
        "decode_begin",
        "decode_step",
        "free_decode_begin",
        "free_decode_run",
        "free_decode_run_timed",
        "record_reference_begin",
        "record_reference_run",
        "cohort_reference_replay",
        "phase_diagnostics",
        "representative_unknown_kind",
    ]

    for (id, kind) in blockedKinds.enumerated() {
        #expect(
            rejection(
                #"{"id":\#(id),"kind":"\#(kind)"}"#,
                context: pending)
                == "runtime worker request '\(kind)' refused while "
                    + "free_decode_finalize is pending")
    }
}

// MARK: - Guard 5: a teacher-forced decode_begin is serial-only
//
// NARROWED, not weakened. This guard used to be demonstrated with an `mtp`
// spec, which was the only non-serial mode this engine could RUN. With that arm
// gone, every non-serial mode is refused one step earlier — at spec RESOLUTION,
// as not-runnable / not-implemented — so no input reaches the guard's own
// branch today. What is still assertable, and what this now asserts, is that a
// teacher-forced begin admits serial and nothing else; the branch itself is
// re-covered the moment a second runnable mode returns.
@Test
func teacherForcedDecodeBeginAdmitsSerialAndRefusesEveryOtherMode() throws {
    #expect(
        try accepted(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1,2],"spec":{"mode":"serial"}}"#
        ).effectiveSpec == .serial())
    // dspark (stub) and dflash (capability-absent) are both refused on a
    // teacher-forced begin, each with its own distinct reason.
    #expect(
        rejection(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1,2],"spec":{"mode":"dspark"}}"#
        )?.contains("not implemented on this engine") == true)
    let sha = String(repeating: "a", count: 64)
    #expect(
        rejection(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1,2],"spec":{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(sha)"}}}}"#
        )?.contains("not runnable on this engine") == true)
}

// MARK: - Guard 6: free_decode_run.count bound (1...1536)

@Test
func freeDecodeRunCountIsBoundedEngineSide() throws {
    let begun = RuntimeWorkerRequestContext(
        hasDecodeRoute: true, advertisesSpeculativeProtocol: true)
    let ceiling = MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
    #expect(ceiling == 1_536)
    let expected = "runtime worker free_decode_run count must be in 1...\(ceiling)"

    // count 0, one past the ceiling, negative, and absent.
    for json in [
        #"{"id":2,"kind":"free_decode_run","count":0}"#,
        #"{"id":2,"kind":"free_decode_run","count":1537}"#,
        #"{"id":2,"kind":"free_decode_run","count":-1}"#,
        #"{"id":2,"kind":"free_decode_run"}"#,
    ] {
        #expect(rejection(json, context: begun) == expected)
    }
    // Both ends of the legal range are accepted.
    #expect(try accepted(#"{"id":2,"kind":"free_decode_run","count":1}"#, context: begun).freeRunCount == 1)
    #expect(
        try accepted(#"{"id":2,"kind":"free_decode_run","count":1536}"#, context: begun)
            .freeRunCount == 1_536)
}

@Test
func freeDecodeRunBeforeBeginIsRejectedBeforeTheCountIsEvenRead() {
    // No route sealed: sequencing fails first, so an in-range count does not
    // rescue an out-of-order request.
    #expect(
        rejection(#"{"id":2,"kind":"free_decode_run","count":8}"#)
            == "runtime worker free_decode_run before free_decode_begin")
}

// MARK: - Spec resolution: the hostile-sha probe inputs

@Test
func hostileDraftSHAsAreRejectedAtParseTime() throws {
    // Fullwidth digit U+FF10 and fullwidth letter U+FF41 in an otherwise
    // 64-character sha: rejected, so a head identity can never be coarsened.
    let fullwidthDigit = "\u{FF10}" + String(repeating: "a", count: 63)
    let fullwidthLetter = "\u{FF41}" + String(repeating: "0", count: 63)
    for sha in [fullwidthDigit, fullwidthLetter] {
        let message = rejection(
            #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1],"spec":{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(sha)"}}}}"#
        )
        #expect(message?.contains("non-ASCII or non-hex character") == true)
    }
    // An embedded NUL is likewise a non-hex scalar (U+0000), not a terminator.
    // Escaped on the wire (\u0000) so this is a well-formed JSON string that
    // decodes to a 64-scalar value whose last scalar is NUL.
    let withNUL = String(repeating: "a", count: 63) + #"\u0000"#
    let nulMessage = rejection(
        #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1],"spec":{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(withNUL)"}}}}"#
    )
    #expect(nulMessage?.contains("non-ASCII or non-hex character") == true)
    #expect(nulMessage?.contains("U+0000") == true)
}

@Test
func unrunnableAndStubModesFailClosedOnFreeDecodeBegin() throws {
    let sha = String(repeating: "a", count: 64)
    // dflash is a REAL module that is capability-absent on this engine.
    #expect(
        rejection(
            #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1],"spec":{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(sha)"}}}}"#
        )?.contains("not runnable on this engine") == true)
    // dspark is a STUB — a different error on purpose.
    #expect(
        rejection(
            #"{"id":1,"kind":"free_decode_begin","seed_tokens":[1],"spec":{"mode":"dspark"}}"#
        )?.contains("not implemented on this engine") == true)
}

// MARK: - No-spec and missing-field paths

@Test
func absentSpecOnADecodeBeginIsThePlainV1Path() throws {
    // A no-spec decode_begin resolves to nil (plain serial), NOT to the
    // registry default: the generic worker's v1 contract is unchanged.
    #expect(try accepted(#"{"id":1,"kind":"decode_begin","seed_tokens":[1,2]}"#).effectiveSpec == nil)
    #expect(
        try accepted(#"{"id":1,"kind":"free_decode_begin","seed_tokens":[1,2]}"#)
            .effectiveSpec == nil)
}

@Test
func decodeBeginKindsRequireSeedTokens() {
    #expect(
        rejection(#"{"id":1,"kind":"decode_begin"}"#)
            == "runtime worker decode_begin request missing seed_tokens")
    #expect(
        rejection(#"{"id":1,"kind":"free_decode_begin"}"#)
            == "runtime worker free_decode_begin request missing seed_tokens")
}

@Test
func traceDiagnosticsAreCorrectnessOnlyAndMustBeWellFormed() {
    // Valid only on the correctness kinds.
    #expect(
        rejection(#"{"id":1,"kind":"prefill","top_k":4,"expected_token":7}"#)
            == "runtime worker trace diagnostics are valid only for correctness requests")
    // Present but malformed: non-positive top_k, or an out-of-vocab token.
    let malformed =
        "runtime worker trace diagnostics require positive top_k and a valid expected_token"
    #expect(rejection(#"{"id":1,"kind":"correctness_begin","top_k":0,"expected_token":7}"#) == malformed)
    #expect(rejection(#"{"id":1,"kind":"correctness_step","top_k":4,"expected_token":-1}"#) == malformed)
    #expect(
        rejection(
            #"{"id":1,"kind":"correctness_step","top_k":4,"expected_token":\#(MLXFastConstants.vocabSize)}"#
        ) == malformed)
    // Well-formed and on the right kind: accepted.
    #expect(rejection(#"{"id":1,"kind":"correctness_begin","top_k":4,"expected_token":7}"#) == nil)
}

// MARK: - Recording verbs (record_reference_begin / record_reference_run)

// The CBv2-backed reference-tape recorder (record-reference-tape
// --recording-backend cbv2) drives two trusted-CLI-only kinds against the
// participant worker so the pool tapes are produced by the SAME width-1 CBv2
// engine session the post-#29 free-run legs run (port-notes 5.1,
// within-backend). These guards are the recording verbs' wire sequencing and
// bounds, pure and model-free like every other guard in this file.
//
// FAILING FIRST (recorder increment): red until validateGenericWorkerRequest
// learns the two kinds — today an unknown kind sails through validation and
// only dies at the handler's default arm, so the sequencing/bounds contracts
// below do not exist yet.

@Test
func recordReferenceRunBeforeBeginIsRefused() {
    // No recording session is open on a fresh context; the run must not
    // validate (the handler would have no session to drain).
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_run","count":8}"#,
            context: RuntimeWorkerRequestContext())
            == "runtime worker record_reference_run before record_reference_begin")
}

@Test
func recordReferenceBeginRequiresSeedTokens() {
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_begin"}"#,
            context: RuntimeWorkerRequestContext())
            == "runtime worker record_reference_begin request missing seed_tokens")
}

@Test
func recordingVerbsRejectARidingSpec() {
    // Guard 3 (cross-kind spec) already covers every non-decode-begin kind;
    // pin that the recording verbs stay inside that posture rather than
    // growing a spec surface of their own.
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_begin","seed_tokens":[1],"spec":{"mode":"serial"}}"#)
            == "spec is valid only on decode_begin / free_decode_begin; request "
            + "kind 'record_reference_begin' must not carry one")
}

@Test
func recordReferenceRunCountBoundIsEnforced() throws {
    let recording = RuntimeWorkerRequestContext(hasRecordingSession: true)
    let bound = MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
    let message = "runtime worker record_reference_run count must be in 1...\(bound)"
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_run","count":0}"#,
            context: recording) == message)
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_run","count":\#(bound + 1)}"#,
            context: recording) == message)
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_run"}"#,
            context: recording) == message)
    // In bounds: accepted, and the validated count is what the handler runs.
    let validated = try accepted(
        #"{"id":1,"kind":"record_reference_run","count":129}"#,
        context: recording)
    #expect(validated.recordingRunCount == 129)
    #expect(validated.freeRunCount == nil)
}

@Test
func recordReferenceBeginRefusesWhileARecordingSessionIsOpen() {
    // A silently replaced session would leak a live engine; the second
    // begin must be refused until record_reference_run consumes the first.
    let recording = RuntimeWorkerRequestContext(hasRecordingSession: true)
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_begin","seed_tokens":[1,2]}"#,
            context: recording)
            == "runtime worker record_reference_begin while a recording "
            + "session is already open; consume it with record_reference_run "
            + "first")
    // And accepted on a clean session, without the v1.1 spawn gate: the
    // recording verbs are the trusted operator CLI's, which spawns the
    // worker without --speculative-protocol exactly as it does for the
    // legacy recorder's correctness verbs.
    #expect(
        rejection(
            #"{"id":1,"kind":"record_reference_begin","seed_tokens":[1,2]}"#,
            context: RuntimeWorkerRequestContext()) == nil)
}
