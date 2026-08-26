import Foundation
import MLXFastCore

// Pure, model-free WIRE VALIDATION for the generic benchd-facing worker.
//
// Every guard here used to live inline at the top of (and inside the switch of)
// `Gemma4Runtime.handleWorkerRequest`, whose signature demands a
// `Gemma4A4BRuntimeWeightCache` — a ~21.6 GB resident model. That made the guards
// unreachable from a unit test: the only way to exercise them was to load the
// model, which no test does. They are decisions over already-decoded wire data
// and need nothing else, so they live here instead and `handleWorkerRequest`
// calls this once, up front.
//
// The six guards, in evaluation order:
//
//   1. POISON      — a session that already failed serves nothing more.
//   2. SPAWN GATE  — a `spec` may ride only when the worker advertised the v1.1
//                    speculative surface at spawn; otherwise the surface does
//                    not exist and a spec is a protocol violation.
//   3. CROSS-KIND  — a `spec` rides ONLY on the two decode-begin kinds. Anywhere
//                    else it is a field the worker honors nowhere, and silently
//                    ignoring it would be a false acknowledgment.
//   4. FREE-RUN    — the `free_decode_*` kinds are v1.1 surface too, so the
//      GATE         spawn gate governs ACCEPTING them and not only advertising
//                    them. Gated off, the worker serves no kind its own hello
//                    did not offer.
//   5. TF-SERIAL   — a teacher-forced `decode_begin` runs one single-token
//                    serial forward per step and CANNOT execute speculation, so
//                    a non-serial spec is rejected rather than echoed-then-
//                    ignored.
//   6. COUNT BOUND — `free_decode_run.count` is unbounded on the wire; bound it
//                    engine-side at the configured decode ceiling.
//
// Spec RESOLUTION (`resolveEffectiveSpec`) runs here too, so a bad or
// unrunnable mode fails closed BEFORE the handler touches the allocator or the
// model — the same "resolve before any expensive work" ordering the worker
// already used.
//
// Trace diagnostics (`top_k` / `expected_token`) are validated here as well:
// they are the same shape of pure pre-switch check and were sitting in the same
// block.

/// Session facts the validator needs. Scalars only — deliberately NOT
/// `RuntimeWorkerState`, so a test can express "poisoned session" or "no prior
/// free_decode_begin" without constructing KV caches.
struct RuntimeWorkerRequestContext: Equatable {
    /// `state.poisoned`: an earlier request in this session failed.
    let poisoned: Bool
    /// Whether a `(free_)decode_begin` has sealed a route in this session
    /// (`state.decodeRoute != nil`). Only `free_decode_run` reads it.
    let hasDecodeRoute: Bool
    /// Whether the worker advertised the v1.1 speculative surface at spawn.
    let advertisesSpeculativeProtocol: Bool
    /// v1.2 (COHORT): the width a batched `free_decode_begin` sealed in this
    /// session (`state.cohortSession?.batchSize`), or `nil` when no cohort
    /// phase is open. Read by the batched `free_decode_run` form (the sealed
    /// width is never re-negotiated) and by a second batched begin (refused).
    let cohortBatchSize: Int?
    /// RECORDING (trusted-CLI-only): whether a `record_reference_begin`
    /// opened a recording session that has not been consumed
    /// (`state.recordingSession != nil`). Read by `record_reference_run`
    /// (sequencing) and by a second `record_reference_begin` (refused — a
    /// silently replaced session would leak a live engine).
    let hasRecordingSession: Bool

    init(
        poisoned: Bool = false,
        hasDecodeRoute: Bool = false,
        advertisesSpeculativeProtocol: Bool = false,
        cohortBatchSize: Int? = nil,
        hasRecordingSession: Bool = false
    ) {
        self.poisoned = poisoned
        self.hasDecodeRoute = hasDecodeRoute
        self.advertisesSpeculativeProtocol = advertisesSpeculativeProtocol
        self.cohortBatchSize = cohortBatchSize
        self.hasRecordingSession = hasRecordingSession
    }
}

/// What validation produced for the handler to execute. Everything here is
/// already checked; the handler does no further wire validation.
struct RuntimeWorkerValidatedRequest: Equatable {
    /// The resolved `effective_spec` for the two decode-begin kinds. `nil` on
    /// every other kind, and on a decode-begin that carried no spec (the v1
    /// plain-serial path).
    let effectiveSpec: RuntimeWorkerEffectiveSpec?
    /// The bounds-checked `free_decode_run.count`. `nil` on every other kind.
    let freeRunCount: Int?
    /// v1.2 (COHORT): the validated cohort form of a `free_decode_begin`
    /// (explicit width + shape-checked per-slot seeds). `nil` on the v1.1
    /// single-stream form and every other kind.
    let cohortBegin: RuntimeWorkerValidatedCohortBegin?
    /// v1.2 (COHORT): the validated width of a batched `free_decode_run`
    /// (present iff the request carried `batch_size`, already checked equal to
    /// the width the batched begin sealed). `nil` on the v1.1 form.
    let cohortRunBatchSize: Int?
    /// RECORDING: the bounds-checked `record_reference_run.count`. `nil` on
    /// every other kind.
    let recordingRunCount: Int?
    /// PR-1 fidelity-gate: the validated `cohort_reference_replay` payload
    /// (shape-checked per-stream seeds + committed journals, bounded
    /// characterization params). `nil` on every other kind.
    let cohortReferenceReplay: RuntimeWorkerValidatedCohortReferenceReplay?

    init(
        effectiveSpec: RuntimeWorkerEffectiveSpec? = nil,
        freeRunCount: Int? = nil,
        cohortBegin: RuntimeWorkerValidatedCohortBegin? = nil,
        cohortRunBatchSize: Int? = nil,
        recordingRunCount: Int? = nil,
        cohortReferenceReplay: RuntimeWorkerValidatedCohortReferenceReplay? = nil
    ) {
        self.effectiveSpec = effectiveSpec
        self.freeRunCount = freeRunCount
        self.cohortBegin = cohortBegin
        self.cohortRunBatchSize = cohortRunBatchSize
        self.recordingRunCount = recordingRunCount
        self.cohortReferenceReplay = cohortReferenceReplay
    }
}

/// Run every pre-execution wire guard for one generic-worker request.
///
/// Throws `MLXFastError.invalidInput` with the exact operator-facing message
/// for the first guard that fails; returns the resolved spec / bounded count
/// the handler needs otherwise. Pure: no model, no allocator, no I/O.
func validateGenericWorkerRequest(
    _ request: RuntimeWorkerRequest,
    context: RuntimeWorkerRequestContext,
    specRegistry: RuntimeWorkerSpecRegistry
) throws -> RuntimeWorkerValidatedRequest {
    // 1. Fail closed on a poisoned session: an earlier request in this session
    //    errored, so its phase state is discarded and unusable.
    guard !context.poisoned else {
        throw MLXFastError.invalidInput(
            "the generic decode session is poisoned after an earlier failure")
    }
    // 2. A `spec` may ride only when this worker advertised the v1.1 surface at
    //    spawn. Gated off, the speculative surface does not exist, so a spec on
    //    ANY kind is a protocol violation — reject rather than silently ignore.
    if !context.advertisesSpeculativeProtocol, request.spec != nil {
        throw MLXFastError.invalidInput(
            "a spec was sent but this worker did not advertise the "
                + "speculative protocol at spawn "
                + "(\(runtimeWorkerSpeculativeProtocolFlag))")
    }
    // 3. The spec rides ONLY on the two decode-begin kinds. On every other kind
    //    it is a cross-kind field the worker honors nowhere, so reject it rather
    //    than silently ignore it (a silently-ignored spec is a false
    //    acknowledgment of a mode the worker will not run).
    if request.spec != nil,
        request.kind != "decode_begin",
        request.kind != "free_decode_begin"
    {
        throw MLXFastError.invalidInput(
            "spec is valid only on decode_begin / free_decode_begin; request "
                + "kind '\(request.kind)' must not carry one")
    }
    // 4. The free-run kinds are v1.1 surface, so the spawn gate governs whether
    //    they are SERVED, not merely whether `free_run_decode` appears on the
    //    hello. Gated off, a `free_decode_*` request asks for a mode this
    //    process never offered, and answering it would be the same false
    //    acknowledgment guard 2 refuses for a spec.
    //
    //    This cannot regress the measured path: a free-run leg is spawned WITH
    //    the flag by construction (benchd's `leg_spawn_args` pushes it for a
    //    free-run regime, and `validate_spawn_argv` fences the argv), and
    //    benchd's session refuses to issue the kinds at all unless the hello
    //    advertised the capability (`Session::require_free_run_capability`,
    //    a hard protocol error and never a silent fallback). A gate-off
    //    `free_decode_*` request is therefore out of contract on the caller's
    //    side before it reaches here; this is where that is said out loud.
    if !context.advertisesSpeculativeProtocol,
        request.kind == "free_decode_begin" || request.kind == "free_decode_run"
    {
        throw MLXFastError.invalidInput(
            "\(request.kind) belongs to the v1.1 speculative surface, which "
                + "this worker did not advertise at spawn "
                + "(\(runtimeWorkerSpeculativeProtocolFlag)); the free-run "
                + "kinds are served only when "
                + "'\(runtimeWorkerFreeRunDecodeCapability)' is advertised")
    }
    // 4b. v1.2 (COHORT) — the cohort request fields ride only on the kinds
    //     that honor them, same false-acknowledgment posture as guard 3. (The
    //     free-run kinds themselves are already spawn-gated by guard 4, which
    //     therefore gates the whole cohort surface too.)
    try rejectCohortFieldsOnWrongKinds(request)
    // 4c. PR-1 fidelity-gate — the cohort reference-replay request fields ride
    //     ONLY on `cohort_reference_replay`, same false-acknowledgment posture.
    try rejectCohortReferenceReplayFieldsOnWrongKinds(request)
    // Trace diagnostics belong to the correctness kinds only, and must be
    // internally well-formed when present.
    let carriesTraceDiagnostics =
        request.topK != nil || request.expectedToken != nil
    if carriesTraceDiagnostics {
        guard request.kind == "correctness_begin"
            || request.kind == "correctness_step"
        else {
            throw MLXFastError.invalidInput(
                "runtime worker trace diagnostics are valid only for correctness requests"
            )
        }
        guard let topK = request.topK, topK > 0,
              let expectedToken = request.expectedToken,
              expectedToken >= 0,
              expectedToken < MLXFastConstants.vocabSize
        else {
            throw MLXFastError.invalidInput(
                "runtime worker trace diagnostics require positive top_k and a valid expected_token"
            )
        }
    }

    switch request.kind {
    case "decode_begin":
        guard request.seedTokens != nil else {
            throw MLXFastError.invalidInput(
                "runtime worker decode_begin request missing seed_tokens")
        }
        let effective = try resolveGenericDecodeSpec(
            request.spec, specRegistry: specRegistry)
        // 5. A teacher-forced window runs one single-token serial forward per
        //    step — it CANNOT execute speculation. Reject any non-serial spec
        //    here rather than echo a speculative mode and then silently run
        //    serial (a false
        //    acknowledgment). Speculative acceptance is measured on the SEPARATE
        //    free-run path (free_decode_begin), the only kind that actually runs
        //    the block session.
        if let effective,
            effective.mode != RuntimeWorkerDecodeRoute.serial.rawValue
        {
            throw MLXFastError.invalidInput(
                "teacher-forced decode_begin cannot execute speculation; spec "
                    + "mode '\(effective.mode)' is runnable only on "
                    + "free_decode_begin. TF legs are serial-only.")
        }
        return RuntimeWorkerValidatedRequest(effectiveSpec: effective)

    case "free_decode_begin":
        // v1.2 (COHORT): presence of either cohort field selects the cohort
        // form — validated as such, never silently narrowed to single-stream.
        // The v1.1 branch below is byte-for-byte the pre-cohort behavior.
        if runtimeWorkerRequestSelectsCohortBegin(request) {
            return RuntimeWorkerValidatedRequest(
                effectiveSpec: try resolveGenericDecodeSpec(
                    request.spec, specRegistry: specRegistry),
                cohortBegin: try validateCohortFreeDecodeBegin(
                    request, context: context))
        }
        guard request.seedTokens != nil else {
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_begin request missing seed_tokens")
        }
        return RuntimeWorkerValidatedRequest(
            effectiveSpec: try resolveGenericDecodeSpec(
                request.spec, specRegistry: specRegistry))

    case "free_decode_run":
        // v1.2 (COHORT): a present `batch_size` selects the cohort form, which
        // sequences against the batched begin's sealed width instead of the
        // single-stream route. An open cohort phase conversely refuses the
        // v1.1 form — the two windows must never cross.
        if request.batchSize != nil {
            let batchSize = try validateCohortFreeDecodeRun(
                request, context: context)
            guard let n = request.rowCount, n > 0,
                  n <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_run count must be in "
                        + "1...\(MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens)")
            }
            return RuntimeWorkerValidatedRequest(
                freeRunCount: n, cohortRunBatchSize: batchSize)
        }
        guard context.cohortBatchSize == nil else {
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_run without batch_size while a "
                    + "cohort phase is open; the batched window's run must "
                    + "carry the sealed batch_size")
        }
        guard context.hasDecodeRoute else {
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_run before free_decode_begin")
        }
        // 6. Bound N engine-side. The wire field is unbounded, so an absurd
        //    count could pin the worker in an unbounded free-run loop; cap it at
        //    the same configured decode ceiling the block round paths use.
        guard let n = request.rowCount, n > 0,
              n <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_run count must be in "
                    + "1...\(MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens)")
        }
        return RuntimeWorkerValidatedRequest(freeRunCount: n)

    case "record_reference_begin":
        // RECORDING (trusted-CLI-only; the CBv2-backed reference-tape
        // recorder). Deliberately NOT behind the v1.1 spawn gate: the gate
        // governs the benchd-facing speculative surface a hello negotiates,
        // while the recording verbs are the trusted operator CLI's — that
        // CLI spawns the worker itself, without the flag, exactly as it does
        // for the correctness verbs the legacy recorder drives. No spec ever
        // rides them (guard 3 above), so nothing speculative is reachable
        // through this surface.
        guard request.seedTokens != nil else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_begin request missing seed_tokens")
        }
        guard !context.hasRecordingSession else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_begin while a recording "
                    + "session is already open; consume it with "
                    + "record_reference_run first")
        }
        return RuntimeWorkerValidatedRequest()

    case "record_reference_run":
        guard context.hasRecordingSession else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_run before record_reference_begin")
        }
        // Same engine-side count bound as free_decode_run (guard 6): the wire
        // field is unbounded and an absurd count would pin the worker in an
        // unbounded free-run loop.
        guard let n = request.rowCount, n > 0,
              n <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_run count must be in "
                    + "1...\(MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens)")
        }
        return RuntimeWorkerValidatedRequest(recordingRunCount: n)

    case "cohort_reference_replay":
        // REFERENCE-REPLAY ORACLE (trusted-CLI-only; MEASUREMENT mode).
        // Deliberately NOT behind the v1.1 spawn gate, same trusted-operator
        // posture as the recording verbs: the trusted CLI spawns this worker
        // itself (from the pinned baseline tree over organizer weights), after
        // the timed window with the candidate torn down. No spec ever rides it
        // (guard 3 above rejects a cross-kind spec). The pure shape/bounds check
        // lives in `validateCohortReferenceReplay`; the verb renders no verdict.
        return RuntimeWorkerValidatedRequest(
            cohortReferenceReplay: try validateCohortReferenceReplay(request))

    default:
        return RuntimeWorkerValidatedRequest()
    }
}

/// Resolve a generic decode request's `spec` into the effective echo, or `nil`
/// for a v1 no-spec (plain serial) decode. Resolution fails closed for a stub
/// (dspark), a capability-absent real module, or a malformed block. Which
/// modes are runnable is the REGISTRY's per-worker fact — `serial` always,
/// `mtp` with an assistant head staged, `dflash` with a real
/// `DFlashDraftModel` bound from `dflash-head/` — and every mode outside that
/// set is refused here rather than silently downgraded.
func resolveGenericDecodeSpec(
    _ spec: RuntimeWorkerSpecRequest?,
    specRegistry: RuntimeWorkerSpecRegistry
) throws -> RuntimeWorkerEffectiveSpec? {
    guard let spec else { return nil }
    return try specRegistry.resolveEffectiveSpec(spec)
}
