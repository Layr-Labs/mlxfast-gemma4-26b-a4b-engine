import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Metal

// Gemma4Runtime is split across Gemma4Runtime*.swift for auditability.
// Generated split; behavior identical to the original single file.

// MARK: - Phase-0 hello identity (protocol_version / backend / device)

/// The engine's implemented protocol version, emitted on EVERY hello
/// (`protocol_version`). Stays `1` across the v1.1 additive extension
/// (PROTOCOL.md §"Phase-0 hello fields" / §"v1.1 additive extension"): v1.1 is
/// advertised by capability flags, not a version bump. benchd's session setup
/// REFUSES a hello whose `protocol_version` is absent or != 1, so this rides on
/// every hello regardless of whether the speculative surface is advertised.
let runtimeWorkerProtocolVersion = 1

/// The compute backend label on the hello (`backend`). Apple-Silicon MLX/Metal.
let runtimeWorkerBackendLabel = "mlx"

/// The device identity label on the hello (`device`). Read once, pre-hello,
/// outside every timed window. Purely a label so cross-backend scores are marked
/// not-comparable; benchd does not constrain its value. Falls back to a stable
/// constant when no Metal device is enumerable.
func runtimeWorkerDeviceLabel() -> String {
    MTLCreateSystemDefaultDevice()?.name ?? "apple-metal"
}

extension Gemma4Runtime {
    public static func runWorker(
        weightsPath: String,
        mtpHeadPath: String? = nil,
        dflashHeadPath: String? = nil,
        advertisesSpeculativeProtocol: Bool = false
    ) throws {
        // The worker holds the ~21.6 GB model for its whole lifetime, so it must
        // never outlive the harness parent that spawned it. Reading protocol
        // stdin already ends the worker on parent death (pipe EOF), but only
        // while the worker is blocked reading -- NOT during the minutes-long
        // model load below, or while a forward is in flight. Start the orphan
        // self-reaper first so a parent that dies during those windows cannot
        // leave a resident-model orphan that out-of-memories the next run.
        startRuntimeWorkerOrphanReaper()
        // Move the protocol away from fd 0/1 before any editable model code
        // runs. Startup validation and model construction may log or otherwise
        // use standard I/O; none of that may be confused with protocol traffic.
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        try validateRuntimeWorkerPinnedConfiguration(weightsPath: weightsPath)
        let config = try Gemma4A4BConfig.load(from: weightsPath)
        let loader = try Gemma4A4BWeightLoader(weightsPath: weightsPath)
        // Validate transformed-weight structure HERE, inside the sandboxed worker,
        // rather than in the trusted parent. These checks execute editable
        // MLXFastModel code (DenseTensorStore / Gemma4A4BWeightLoader); the parent
        // used to run the equivalent via BenchmarkPreflight.check, which meant
        // submitted code ran in the unsandboxed process that authors score.json.
        // Failing here throws before the protocol hello below, so the parent's
        // worker client sees the worker fail to start and records a failed
        // benchmark -- same coverage, no submitted code in the score-writing
        // parent.
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)
        // Constructing the weight cache loads the whole 4-bit Qwen 3.6 text
        // tower and runs its constructor-time kernel warmup, all before the
        // protocol hello -- outside every scored window.
        let weightCache = Gemma4A4BRuntimeWeightCache(loader: loader, config: config)
        // THE TARGET QUANTIZATION BIND (David ruling 2026-08-26). Everything
        // checked so far -- `validateRuntimeWorkerPinnedConfiguration` above,
        // and the transformed-weight metadata -- reads the DISK. The step that
        // turns those declarations into live `Quantized` modules is
        // `Gemma4A4BRuntimeWeights.quantizeWithPerPathWidths`, which is
        // participant-editable, so until this call the target's frozen format
        // was enforced against a declaration nobody checked the runtime
        // against. This reads the loaded modules themselves. See
        // `Gemma4TargetQuantizationBind.swift`.
        //
        // HERE, and not later: `Gemma4A4BRuntimeWeightCache.libraryModel` is a
        // `let` built once in the cache's initializer, so every later
        // `requireLibraryModel()` in this worker returns the instance verified
        // on this line. Throwing here fails the worker START, before the
        // protocol hello and before any GPU work, on BOTH legs -- benchd spawns
        // one worker per leg -- so a refusal is fatal to the run rather than a
        // per-request error the harness could retry past.
        //
        // "IS A `let`" IS ITSELF AN EDITABLE FACT, so it is CHECKED rather than
        // relied on. `requireLibraryModel()` lives in `Sources/MLXFastModel`,
        // and the worker calls it many times after this point. A candidate
        // could turn `libraryModel` into a computed property that hands back a
        // freshly re-quantized model on the second call, which would make the
        // verification above a snapshot of an instance nothing else ever uses.
        // Two calls, one identity comparison, and that whole class of
        // substitution is refused at startup.
        //
        // WHAT THIS DOES NOT COVER, stated plainly rather than implied: the
        // SAME instance can still be mutated in place later, by editable
        // request-path code calling `Module.update(modules:)` on it. Closing
        // that needs the verified instance threaded through
        // `handleWorkerRequest` (whose signature takes the cache, not a model)
        // instead of re-fetched, which is a refactor and not this change.
        //
        // The layer count comes from `MLXFastConstants`, NOT from `config`:
        // `config` is loaded by editable code, and an expected set built from a
        // candidate-supplied layer count would collapse to empty on a candidate
        // that reported zero layers. The trusted config gate above already
        // pinned the on-disk `num_hidden_layers` to this same constant, so the
        // two agree by the time this runs.
        //
        // NO `#if` GUARD. `MLXFAST_TRUSTED_HARNESS` is defined only on the
        // `MLXFastHarness` target (path `Sources/MLXFastTrustedHarness`,
        // Package.swift:104-107). This file is the scored worker's target, so a
        // conditional here would compile the gate out of the binary that
        // actually runs.
        let verifiedTarget = try weightCache.requireLibraryModel()
        try validateLoadedTargetQuantization(
            model: verifiedTarget,
            numHiddenLayers: MLXFastConstants.numHiddenLayers)
        guard try weightCache.requireLibraryModel() === verifiedTarget else {
            throw MLXFastError.invalidInput(
                "target quantization is frozen: the runtime weight cache returned a "
                    + "different target instance on a second call, so the verified model "
                    + "is not the model this worker would run"
            )
        }

        // Generic benchd-facing spec surface. benchd drives GENERIC kinds only
        // (decode_begin/decode_step + free_decode_begin/free_decode_run) and
        // selects the mode with a per-request `spec`; this worker resolves the
        // effective spec, echoes it, and routes by mode.
        //
        // MTP RETURNED 2026-08-23 (Gemma 4 26B A4B MTP arm): runnability is
        // decided ONCE here, by whether a staged assistant head loads and
        // binds to THIS worker's target instance. `mtpAvailable` is fixed for
        // the worker's whole lifetime — the hello's `spec_modes` and every
        // later resolution must agree with what was decided before the first
        // byte was read (RuntimeWorkerSpecRegistry.gemma4Worker's own doc
        // comment). The staging directory is `mtpHeadPath` — the `--mtp-head`
        // argv value, restored 2026-08-25 to close the benchd spawn-contract
        // mismatch (benchd passes it on EVERY measure-job leg) — when given,
        // else the CWD `./mtp-head/` default the native trusted CLI uses. On
        // the default channel a directory that is simply absent is the normal
        // no-head case, not an error; an EXPLICIT `--mtp-head` directory is a
        // declaration and refuses at startup unless it loads, exactly like a
        // default-channel directory that exists but is broken (a
        // present-but-broken head is a refusal, never a silent downgrade to
        // serial-only — the same DECIDE-2 posture `Gemma4MTPHeadDeclaration`
        // states for a broken manifest — see
        // `resolveGemma4AssistantHeadStaging`).
        // `headLoad` is kept alive for the worker's whole lifetime
        // (this function's scope does not return until the process exits) so
        // `headLoad?.drafter` can be bound into a real CBv2 MTP round loop
        // per request (2026-08-23 round-execution increment — see
        // `handleWorkerRequest`'s `mtpDrafter` parameter and
        // `Gemma4RuntimeMTPDriver.swift` / `Gemma4RuntimeCohortDriver.swift`'s
        // `runMTP`).
        let headLoad = try loadGemma4AssistantHeadIfStaged(
            explicitDirectoryPath: mtpHeadPath,
            target: try weightCache.requireLibraryModel())
        // DFLASH ARM (2026-08-25): a REAL `DFlashDraftModel` bound to this
        // worker's Gemma 4 target from the CWD `dflash-head/` default.
        // Runnability is decided ONCE here and stays fixed for the worker's
        // lifetime, so the hello's advertised `dflash` and every later
        // resolution agree.
        //
        // FAIL-SOFT (audit-required). This load NEVER throws — see
        // `loadGemma4DFlashHeadIfStaged`. An absent `dflash-head/` is silent
        // capability-absence. A PRESENT-but-unloadable one is
        // capability-absence PLUS the named stderr warning below, and the
        // worker still reaches hello with serial + mtp intact: a broken
        // DFlash head must not take the serial CONTROL leg down with it. The
        // present-but-broken REFUSAL still happens — at DFlash spec
        // resolution, where it can name the actual load failure — never as a
        // pre-hello death.
        //
        // PER-LEG STAGING (David ruling 2026-08-26). `dflashHeadPath` is the
        // `--dflash-head` argv value, and it is what makes the DFlash arm
        // scoreable: benchd spawns both legs with no `current_dir`, so the
        // former CWD-only channel had BOTH legs resolving the SAME
        // `./dflash-head/` — the candidate's drafter resident on the scored
        // DENOMINATOR leg. With the flag, the serial control loads the PINNED
        // drafter and the candidate its own.
        //
        // The one place this load CAN now throw is a declared-but-unstageable
        // `--dflash-head`, and that exception to fail-soft is deliberate:
        // fail-soft exists so a BROKEN drafter cannot take the serial control
        // leg down, whereas a MISWIRED per-leg staging is precisely what
        // endangers the serial control leg. Falling back to the CWD default
        // there would silently restore the cross-leg collapse.
        let dflashOutcome = try loadGemma4DFlashHeadIfStaged(
            explicitDirectoryPath: dflashHeadPath,
            target: try weightCache.requireLibraryModel())
        let dflashLoad: Gemma4DFlashHeadLoadResult?
        let dflashCapability: RuntimeWorkerSpecRegistry.DFlashCapability
        switch dflashOutcome {
        case .absent:
            dflashLoad = nil
            dflashCapability = .absent
        case .loaded(let load):
            dflashLoad = load
            dflashCapability = .available(
                maxDepth: load.maxDepth, provenanceSHA256: load.provenance.sha256)
        case .incompatible(let directory, let reason):
            dflashLoad = nil
            dflashCapability = .broken(reason: reason)
            FileHandle.standardError.write(
                Data(
                    ("[runtime-worker] dflash-head-unloadable: a DFlash head is "
                        + "staged at \(directory.path) but could not be loaded — "
                        + "\(reason). The dflash arm is capability-absent for this "
                        + "worker; serial and mtp are unaffected and a "
                        + "{\"mode\":\"dflash\"} request will be refused by name.\n")
                        .utf8))
        }
        let specRegistry = RuntimeWorkerSpecRegistry.gemma4Worker(
            mtpAvailable: headLoad != nil,
            dflash: dflashCapability)
        // The free-run leg's stop-token set, read from the backbone's own
        // config.json / generation_config.json. It stays a first-class value
        // rather than an inline read because the moment a second leg exists
        // again, BOTH legs of a paired measurement have to stop on exactly the
        // same set -- one side stopping early while the other runs on puts the
        // difference straight into the ratio the two legs exist to produce.
        let stopTokens = resolveRuntimeWorkerStopTokens(
            directory: URL(fileURLWithPath: weightsPath))

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        // expertStats is always the zero struct for this RAM-resident dense
        // runtime (no expert-streaming machinery); kept in the protocol hello
        // so the schema/field shape stays unchanged from earlier submissions.
        // spec_modes / capabilities / head_provenance advertise the generic
        // benchd-facing v1.1 surface: which modes are runnable, that the free-run
        // timed decode is available, and which head (if any) this worker loaded.
        // They are GATED at spawn (fix #2): emitted only when benchd opted in with
        // --speculative-protocol, so the native trusted CLI (which never passes it)
        // sees only the v1 hello its decoder accepts. protocol_version/backend/
        // device are ungated and ride on every hello.
        let advertise = advertisesSpeculativeProtocol
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true,
            expertStats: expertStats(from: weightCache),
            specModes: advertise ? specRegistry.advertisedModeStrings : nil,
            // The harness's OWN recomputed tree digest over whatever loaded
            // (never the manifest's declared value — see
            // RuntimeWorkerHeadProvenance and computeGemma4AssistantHeadProvenance).
            // nil is still an honest "this run drafted with nothing" when no
            // head staged; non-nil once one does.
            //
            // ARM DISCRIMINATION. This is a hello field — emitted once, before
            // any spec is chosen — so it cannot name the arm a later run will
            // use. It therefore means exactly one thing: the MTP head's
            // digest, the pinned-head provenance benchd already reads. A
            // DFlash-only worker still fills it (a digest of the only head
            // there is), but a worker with BOTH heads staged does NOT let the
            // DFlash digest masquerade as the MTP one and vice versa: the
            // per-RUN, arm-specific statement is `effective_spec.dflash.draft`
            // on the decode-begin echo, which carries the harness's recomputed
            // digest of the drafter that arm actually bound
            // (`RuntimeWorkerSpecRegistry.effectiveFor`). #38 collapsed the
            // two into this one slot with `??`, so a dflash run on a
            // both-staged worker reported the MTP head's digest as its
            // drafter identity.
            headProvenance: headLoad?.provenance ?? dflashLoad?.provenance,
            // v1.1 free-run plus its v1.2 batched (cohort) form, with the
            // cohort-width ceiling benchd uses to refuse an over-wide cohort
            // pre-GPU. Both gated at spawn like the rest of the speculative
            // surface (the native trusted CLI's v1 hello decoder never sees
            // them).
            // `cohort_reference_replay` is advertised UNGATED (the verb is not
            // behind the speculative spawn gate); the speculative caps ride only
            // on a gate-on hello. benchd's (b) oracle spawns PLAIN and needs the
            // replay cap visible, so it is always present. Assembled by
            // `runtimeWorkerHelloCapabilities` (RuntimeWorkerCohortSupport) —
            // the one place that list is built, which the gate pin and the
            // captured wire fixture both drive.
            capabilities: runtimeWorkerHelloCapabilities(
                advertisesSpeculativeProtocol: advertise),
            maxBatchSize: advertise
                ? runtimeWorkerMaxCohortBatchSize : nil,
            protocolVersion: runtimeWorkerProtocolVersion,
            backend: runtimeWorkerBackendLabel,
            device: runtimeWorkerDeviceLabel()
        )))
        var state = RuntimeWorkerState()

        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                do {
                    response = try handleWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        weightCache: weightCache,
                        // The instance the startup bind accepted. Carried in so
                        // the pre-measure re-check can compare the model a
                        // measured window is about to run against the model
                        // that was verified, rather than re-deriving "the
                        // target" from the editable cache accessor.
                        verifiedTarget: verifiedTarget,
                        specRegistry: specRegistry,
                        stopTokens: stopTokens,
                        advertisesSpeculativeProtocol: advertise,
                        // `headLoad?.drafter` is the SAME bound
                        // `Gemma4CBv2MTPDrafter` `RuntimeWorkerSpecRegistry
                        // .gemma4Worker(mtpAvailable:)` decided runnability
                        // from above — round execution reads it now instead
                        // of only keeping it alive unused (see this
                        // function's header note on `headLoad`).
                        mtpDrafter: headLoad?.drafter,
                        dflashDrafter: dflashLoad?.drafter,
                        state: &state
                    )
                } catch {
                    // fix #4: any handler error poisons the session (fail-closed),
                    // so a half-advanced KV/recurrent cache can never be reused by
                    // a later request in this phase.
                    state.poisoned = true
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(id: -1, nonce: sessionNonce, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            try protocolIO.writeLine(data)
        }
    }

    /// One-shot structural validation used by the trusted `preflight` command.
    /// Protocol stdout is isolated before any editable model code runs so
    /// participant logging cannot forge the JSON result.
    public static func runPreflightWorker(weightsPath: String) throws {
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        let response: RuntimeWorkerPreflightResponse
        do {
            try validateRuntimeWorkerPinnedConfiguration(weightsPath: weightsPath)
            let config = try Gemma4A4BConfig.load(from: weightsPath)
            let loader = try Gemma4A4BWeightLoader(weightsPath: weightsPath)
            try loader.denseStore.validateReadableByteRanges()
            try loader.validateRequiredMetadata(config: config)
            response = RuntimeWorkerPreflightResponse(ok: true)
        } catch {
            response = RuntimeWorkerPreflightResponse(
                ok: false,
                error: "\(error)"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        try protocolIO.writeLine(try encoder.encode(response))
        if let error = response.error {
            throw MLXFastError.invalidInput(
                "participant worker preflight failed: \(error)"
            )
        }
    }

    /// Poll interval for the worker's orphan self-reaper. Coarse on purpose:
    /// the check is two syscalls, and a couple of seconds of residual
    /// residency after a dead parent is harmless.
    static let runtimeWorkerOrphanPollSeconds = 2.0

    /// Background self-reaper: exit the worker promptly once the spawning
    /// parent is gone, instead of relying solely on protocol-stdin EOF (which
    /// the worker only observes while blocked reading between requests).
    ///
    /// The worker is always spawned by the harness's RuntimeWorkerClient, so
    /// on macOS its parent dying re-parents it to launchd and `getppid()`
    /// becomes 1 -- an unambiguous "the harness that owns me is dead" signal
    /// that cannot fire in any healthy run (local or ranked). Exiting frees
    /// the ~21.6 GB model residency so the next run cannot double-load into an
    /// out-of-memory. The seams exist for tests only; production callers use
    /// the defaults.
    @discardableResult
    static func startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: Double = Gemma4Runtime.runtimeWorkerOrphanPollSeconds,
        isOrphaned: @escaping @Sendable () -> Bool = { getppid() == 1 },
        onOrphaned: @escaping @Sendable () -> Void = {
            fputs(
                "mlxfast-swift: runtime worker parent exited; shutting down to release model memory\n",
                stderr
            )
            exit(1)
        }
    ) -> Thread {
        let thread = Thread {
            while !Thread.current.isCancelled {
                if isOrphaned() {
                    onOrphaned()
                    return
                }
                Thread.sleep(forTimeInterval: pollIntervalSeconds)
            }
        }
        thread.name = "mlxfast.worker-orphan-reaper"
        thread.start()
        return thread
    }

    /// Trusted allocator state applied at the START of every new worker forward
    /// sequence, after the parent has already started the phase timer.
    /// Submitted MLXFastModel code runs during worker initialization and may
    /// change the process-global MLX cache policy, so the trusted request
    /// handler re-normalizes the allocator at the sequence boundary.
    ///
    /// Scope: the boundary only. This is NOT an enforced cap for the rest of
    /// the phase -- editable code may change `Memory.cacheLimit` again inside
    /// the charged window, and any allocation that follows is charged like all
    /// other work. The substantive defense is `Memory.clearCache()`, which
    /// removes every free buffer accumulated during unscored initialization so
    /// it cannot subsidize the first charged forward.
    static let trustedRuntimeWorkerPhaseStartCacheLimitBytes = 6 << 30

    static func resetRuntimeWorkerAllocatorForPhaseStart() throws {
        Memory.cacheLimit = trustedRuntimeWorkerPhaseStartCacheLimitBytes
        Memory.clearCache()
        let remainingCacheBytes = Memory.cacheMemory
        // The pinned MLX clearCache contract synchronously deallocates every
        // cached (free) buffer under its evaluation lock. Live model weights
        // and KV state are active memory, not cacheMemory, so exact zero is the
        // safe fail-closed postcondition rather than a tolerance. This also
        // makes "no MLX allocator activity in flight across a request
        // boundary" part of the submission contract: background work that
        // repopulates the cache here fails the run closed.
        guard remainingCacheBytes == 0 else {
            throw MLXFastError.invalidInput(
                "runtime worker failed to clear the MLX allocator cache at phase start"
            )
        }
    }

    /// One forward through the RAM-resident Qwen 3.6 text tower. It is an
    /// INSTANCE model whose per-layer `[KVCache]`
    /// stack both stores K/V and supplies RoPE positions, so the model
    /// takes no explicit offset. `positionOffset` is kept as the caller's
    /// statement of where the sequence should be and is validated against
    /// the cache offsets, preserving the old adapter's fail-loudly contract
    /// for a stale or reused cache. Returns `[1, 1, vocab]` LAST-token
    /// logits (Qwen applies no final softcap and no embedding scaling).
    static func gemma4Logits(
        inputIDs: MLXArray,
        model: Gemma4TextModel,
        cache: [KVCache],
        positionOffset: Int
    ) throws -> MLXArray {
        try verifyQwenCachePosition(positionOffset: positionOffset, cache: cache)
        return model(inputIDs, cache: cache)
    }

    /// Validate the caller's expected position against the HYBRID cache stack.
    ///
    /// RE-DERIVED FOR GEMMA 4 26B A4B (2026-08-22), and the previous rule was
    /// not merely stale -- it described a tower that is not here. The Qwen form
    /// of this check demanded that the non-full layers carry a RECURRENT cache
    /// pinned at offset 0, because Qwen's 48 gated-delta layers hold recurrent
    /// state with no per-position addressing. Gemma 4 has NO recurrent layers
    /// at all. `Gemma4TextModel.newCache(parameters: nil)` returns
    /// `StandardKVCache` on the 5 full-attention layers (5/11/17/23/29) and
    /// `RotatingKVCache(maxSize: slidingWindow)` on the other 25, and BOTH
    /// advance `offset` by the number of positions written -- the ring lives in
    /// `RotatingKVCache.idx`, not in `offset`. Left as it was, this gate would
    /// have accepted the first prefill (everything at 0) and then refused the
    /// very next forward on real weights, which is the same failure the Qwen
    /// re-derivation was written to fix, in the other direction.
    ///
    /// So the invariant is once again LOCKSTEP, as it was for Laguna, plus a
    /// topology check that the two cache classes sit where the layer schedule
    /// says:
    ///
    /// - one cache per layer, at the pinned 30-layer count;
    /// - every cache offset equals the caller's expected position;
    /// - the global layers carry an unbounded contiguous cache and the sliding
    ///   layers a windowed one.
    ///
    /// The class check is the trusted side of `Gemma4A4BKVBackend`'s contiguous
    /// pin. It is deliberately NOT delegated to it: that type lives in
    /// participant-editable `MLXFastModel`, and this is the trusted parent's
    /// own fail-loudly check on a stale, reused or re-backed cache.
    static func verifyQwenCachePosition(
        positionOffset: Int,
        cache: [KVCache]
    ) throws {
        guard positionOffset >= 0 else {
            throw MLXFastError.invalidInput("Gemma 4 position offset must be non-negative")
        }
        guard cache.count == MLXFastConstants.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "Gemma 4 model returned \(cache.count) layer caches, expected one per layer "
                    + "(\(MLXFastConstants.numHiddenLayers))"
            )
        }

        let interval = MLXFastConstants.fullAttentionInterval
        for (layerIndex, layerCache) in cache.enumerated() {
            let isGlobal = layerIndex % interval == interval - 1
            // Topology: a global layer's cache is unbounded, a sliding layer's
            // is windowed at exactly the pinned window. `maxSize` distinguishes
            // them without naming a concrete class, so a vendored rename cannot
            // silently turn this into a no-op.
            if isGlobal {
                guard layerCache.maxSize == nil else {
                    throw MLXFastError.invalidInput(
                        "Gemma 4 full-attention layer \(layerIndex) must carry an "
                            + "unbounded KV cache, got one capped at "
                            + "\(layerCache.maxSize.map(String.init) ?? "nil")"
                    )
                }
            } else {
                guard layerCache.maxSize == MLXFastConstants.slidingWindow else {
                    throw MLXFastError.invalidInput(
                        "Gemma 4 sliding-attention layer \(layerIndex) must carry a "
                            + "KV cache windowed at \(MLXFastConstants.slidingWindow), "
                            + "got \(layerCache.maxSize.map(String.init) ?? "unbounded")"
                    )
                }
            }
            // Lockstep: both cache classes count every position written, so a
            // layer that disagrees is a stale or partially-advanced stack.
            guard layerCache.offset == positionOffset else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 position offset \(positionOffset) does not match KV cache "
                        + "offset \(layerCache.offset) at layer \(layerIndex)"
                )
            }
        }
    }

    /// Force-evaluate the per-layer KV state so the seed prefill's cache
    /// writes are complete before decode steps are timed against it.
    static func materializeQwenCacheState(_ cache: [KVCache]) {
        eval(cache)
    }

    static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        weightCache: Gemma4A4BRuntimeWeightCache,
        verifiedTarget: Module,
        specRegistry: RuntimeWorkerSpecRegistry,
        stopTokens: Set<Int>,
        advertisesSpeculativeProtocol: Bool,
        mtpDrafter: Gemma4CBv2MTPDrafter?,
        dflashDrafter: DFlashDraftModel? = nil,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        // EVERY pre-execution wire guard — poison, the spawn gate, the
        // cross-kind spec rejection, trace diagnostics, spec resolution, the
        // teacher-forced non-serial rejection, and the free_decode_run count
        // bound — is decided here, by a pure function that needs no model. See
        // RuntimeWorkerRequestValidation.swift; it is unit-tested directly,
        // which the inline versions of these guards could not be (this
        // signature requires a 21.6 GB weight cache).
        let validated = try validateGenericWorkerRequest(
            request,
            context: RuntimeWorkerRequestContext(
                poisoned: state.poisoned,
                hasDecodeRoute: state.decodeRoute != nil,
                advertisesSpeculativeProtocol: advertisesSpeculativeProtocol,
                cohortBatchSize: state.cohortSession?.batchSize,
                hasRecordingSession: state.recordingSession != nil
            ),
            specRegistry: specRegistry
        )
        switch request.kind {
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let tokens = try generateGreedyCached(
                promptTokens: promptTokens,
                steps: steps,
                weightCache: weightCache
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                tokens: tokens,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_begin":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing prompt_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            let logits = try gemma4Logits(
                inputIDs: inputIDsArray(promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            let token = try Gemma4Correctness.greedyToken(from: logits)
            let diagnostics = try correctnessLogitDiagnostics(
                from: logits,
                topK: request.topK
                    ?? MLXFastConstants.correctnessTopLogits,
                expectedToken: request.expectedToken
            )
            state.correctnessCache = cache
            state.correctnessPromptTokenCount = promptTokens.count
            state.correctnessStep = 0
            // Timed step (benchd `is_timed_step`): count toward the phase barrier.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: diagnostics.topLogits,
                expectedTokenLogit: diagnostics.expectedTokenLogit,
                expectedTokenRank: diagnostics.expectedTokenRank,
                topLogitMargin: diagnostics.topLogitMargin,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_step":
            guard let previousToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing token")
            }
            guard let cache = state.correctnessCache else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness step before begin")
            }
            let logits = try gemma4Logits(
                inputIDs: inputIDsArray([previousToken]),
                model: try weightCache.requireLibraryModel(),
                cache: cache,
                positionOffset: state.correctnessPromptTokenCount + state.correctnessStep
            )
            let token = try Gemma4Correctness.greedyToken(from: logits)
            let diagnostics = try correctnessLogitDiagnostics(
                from: logits,
                topK: request.topK
                    ?? MLXFastConstants.correctnessTopLogits,
                expectedToken: request.expectedToken
            )
            state.correctnessStep += 1
            // Timed step (benchd `is_timed_step`): count toward the phase barrier.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: diagnostics.topLogits,
                expectedTokenLogit: diagnostics.expectedTokenLogit,
                expectedTokenRank: diagnostics.expectedTokenRank,
                topLogitMargin: diagnostics.topLogitMargin,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "prefill":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker prefill request missing prompt_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            // PRE-MEASURE RE-CHECK. Deliberately AFTER `newCache` -- which is
            // editable model code -- and immediately before the forward, so no
            // editable statement sits between the check and the measured work.
            // The instance checked is `model`, the same object handed to
            // `gemma4Logits` on the next line.
            try revalidateTargetForMeasuredWindow(
                phase: "prefill",
                verifiedTarget: verifiedTarget,
                currentTarget: model,
                numHiddenLayers: MLXFastConstants.numHiddenLayers)
            let logits = try gemma4Logits(
                inputIDs: inputIDsArray(promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            eval(logits)
            let token = try Gemma4Correctness.greedyToken(from: logits)
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "decode_begin":
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode_begin request missing seed_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            // Generic benchd-facing decode window. A `spec` selects the mode and
            // is resolved + echoed as `effective_spec`; an absent spec is the v1
            // plain (serial) decode, unchanged. The teacher-forced timed window
            // runs one single-token forward per step for EVERY mode (so the
            // phase-close barrier `completed_work == issued_steps` holds); MTP's
            // speculative acceptance is measured on the SEPARATE free-run path
            // (free_decode_run), which is benchd's scored MTP series. The route is
            // sealed here so decode_step honors the same mode.
            //
            // The spec was resolved and the non-serial rejection applied by
            // validateGenericWorkerRequest above.
            let effective = validated.effectiveSpec
            state.declaredSpec = effective
            // Exactly one whole-prompt (seed) forward runs here, with NO preceding
            // warmup pass. The decode measurement deliberately charges this seed
            // prefill to the decode phase (see measureWorkerDecode). A second,
            // identical whole-prompt forward -- the warmup this used to run before
            // the seed -- let submitted model code memoize one pass and serve the
            // other from that memo (both had the same tokens at offset 0), so two
            // charged forwards collapsed into one and inflated decode_speedup with
            // no real speedup. The trusted harness cannot force editable code to
            // recompute a forward it issues, so the only robust defense is to never
            // issue two identical forwards in the timed window: with a single seed
            // forward there is no identical predecessor to reuse, and the 128
            // single-token decode steps are input-dependent and cannot be
            // precomputed. Prefill/decode/correctness each run in their own worker
            // process, so no model-owned memo persists across phases; the trusted
            // reset above separately removes allocator free-buffer state.
            // PRE-MEASURE RE-CHECK. The model is hoisted out of the call so
            // the instance that is CHECKED is provably the instance that is
            // RUN: `plainSeedForward` receives this exact object, and nothing
            // stands between the two lines.
            let decodeModel = try weightCache.requireLibraryModel()
            try revalidateTargetForMeasuredWindow(
                phase: "decode_begin",
                verifiedTarget: verifiedTarget,
                currentTarget: decodeModel,
                numHiddenLayers: MLXFastConstants.numHiddenLayers)
            let seedToken = try plainSeedForward(
                seedTokens: seedTokens,
                model: decodeModel,
                state: &state)
            // Timed step (benchd `is_timed_step`): count toward the phase barrier.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken,
                // v1.1 echo — gated at spawn (fix #2). Off ⇒ no spec was accepted
                // above, so this is nil; on ⇒ echo what the engine will run.
                effectiveSpec: advertisesSpeculativeProtocol ? effective : nil
            )

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            // decode_step invokes only the same editable entry points the
            // correctness path invokes (the Qwen model forward /
            // greedyToken); it must never call an editable hook that is
            // unique to the scored decode path. The former editable
            // decode-delay knob (removed) was exactly such a phase oracle:
            // because submitted model code is editable, the mere fact that it
            // was invoked ONLY on the timed decode path told the submission
            // "I am being scored now", which lets it serve a slow/correct
            // path while checked and a cheap path while timed. Keep
            // trusted->editable calls phase-agnostic.
            let token = try plainDecodeStep(
                inputToken: inputToken,
                model: try weightCache.requireLibraryModel(),
                state: &state)
            // Timed step (benchd `is_timed_step`): count toward the phase barrier.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "free_decode_begin":
            // PRE-MEASURE RE-CHECK, at the TOP of the case and ahead of the
            // cohort branch below, because for the SCORED regime this verb IS
            // the measured prefill window: benchd's `prefill_elapsed_seconds`
            // brackets `free_decode_begin` (bench-runner
            // timing.rs@dc7712ca:761-767). Everything after this line -- the
            // cohort seed prefill, the single-stream seed forward -- is the
            // measured work itself, so this is the last point at which a check
            // is outside it.
            try revalidateTargetForMeasuredWindow(
                phase: "free_decode_begin",
                verifiedTarget: verifiedTarget,
                currentTarget: try weightCache.requireLibraryModel(),
                numHiddenLayers: MLXFastConstants.numHiddenLayers)
            // v1.2 (COHORT): the batched form of the same verb — B seed
            // prefills through the vendored CBv2 engine, all admitted before
            // any consumer starts (closed cohort). Selected by the validated
            // cohort fields; the v1.1 single-stream path below is untouched.
            if let cohortBegin = validated.cohortBegin {
                return try handleCohortFreeDecodeBegin(
                    request,
                    cohort: cohortBegin,
                    effectiveSpec: validated.effectiveSpec,
                    sessionNonce: sessionNonce,
                    weightCache: weightCache,
                    mtpDrafter: mtpDrafter,
                    dflashDrafter: dflashDrafter,
                    state: &state
                )
            }
            // v1.1 oracle-verified free-run timed decode — the seed forward that
            // opens the phase. Same seed contract as decode_begin; the spec is
            // resolved + echoed and the route sealed for the following
            // free_decode_run. This seed forward is the free-run phase's first
            // completed_work unit (the phase closes at R + 1).
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_begin request missing seed_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let freeEffective = validated.effectiveSpec
            let freeRoute: RuntimeWorkerDecodeRoute = try freeEffective.map {
                try runtimeWorkerDecodeRoute(
                    forEffectiveMode: $0.mode,
                    // The fence, restated at the route: `dflash` routes only
                    // on a worker that actually bound a drafter. Resolution
                    // already refused it otherwise, so this is defence in
                    // depth against a resolution/route disagreement rather
                    // than the primary gate.
                    dflashAvailable: dflashDrafter != nil)
            } ?? .serial
            state.decodeRoute = freeRoute
            state.declaredSpec = freeEffective
            // LEG-IMPLEMENTATION IDENTITY (2026-08-25, exactness round two):
            // BOTH v1.1 free-run legs open the SAME width-1 CBv2 engine
            // session — the serial leg drafter-less, the mtp leg bound to
            // the loaded assistant head. The serial leg's former executor
            // (`plainSeedForward`/`plainDecodeStep`, the legacy
            // `model.newCache` single-token loop) is the implementation the
            // teacher-forced correctness verbs — and therefore the pinned
            // reference tapes — run; the mtp leg's stream was computed by
            // the CBv2 engine. A paired token-exactness gate over streams
            // from two different implementations of the same model is
            // unsound at near-tie argmaxes on the production tuple
            // (port-notes 3.1, within-backend rule 5.1) — the sealed
            // 2026-08-25 box evidence (deterministic prompt-specific
            // divergences, byte-identical across two verify-strategy
            // builds) is that unsoundness observed. The teacher-forced
            // decode verbs (`decode_begin`/`decode_step`) deliberately KEEP
            // the legacy loop: they are tape-consistent by construction and
            // benchd's teacher-forced timed mode feeds the tape's own
            // tokens, so implementation identity with the free-run legs is
            // not required there.
            let freeSeedToken = try openSingleStreamFreeRunSession(
                route: freeRoute,
                seedTokens: seedTokens,
                model: try weightCache.requireLibraryModel(),
                mtpDrafter: mtpDrafter,
                dflashDrafter: dflashDrafter,
                // Route-selected: the mtp block carries mtp's depth, the
                // dflash block dflash's. The unused arm is nil, and the
                // session opener reads only the one matching its route.
                requestedDepth: freeRoute == .dflash
                    ? freeEffective?.dflash?.depth
                    : freeEffective?.mtp?.depth,
                stopTokens: stopTokens,
                state: &state)
            state.freeRunLastToken = freeSeedToken
            // The seed forward is the free-run phase's first completed_work unit.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: freeSeedToken,
                // v1.1 echo — gated at spawn (fix #2).
                effectiveSpec: advertisesSpeculativeProtocol
                    ? freeEffective : nil
            )

        case "free_decode_run":
            // v1.1: free-run the engine's own loop until N committed tokens, then
            // return tokens[] + the AUDIT counters (acceptance_lengths / totals).
            // The wire field is `count` (decoded as rowCount). Every verify round
            // adds one completed_work unit, so the phase closes at R + 1.
            // Sequencing and the count bound were checked by
            // validateGenericWorkerRequest above; both are non-nil here.
            //
            // v1.2 (COHORT): the batched form (validated cohortRunBatchSize)
            // free-runs the WHOLE cohort to N committed tokens per stream and
            // returns the B x N rectangle plus the cohort AUDIT counters.
            if let cohortBatchSize = validated.cohortRunBatchSize,
                let cohortN = validated.freeRunCount
            {
                return try handleCohortFreeDecodeRun(
                    request,
                    batchSize: cohortBatchSize,
                    targetN: cohortN,
                    sessionNonce: sessionNonce,
                    state: &state
                )
            }
            guard let route = state.decodeRoute, let n = validated.freeRunCount
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_run before free_decode_begin")
            }
            // Capture the session BEFORE the run consumes it: the session
            // diagnostics (executor identity, sealed-config echo, strategy
            // counts, per-verify-round acceptance/rollback audits) must be
            // emitted on FAILURE paths too — the divergent legs are exactly
            // the ones whose observability matters.
            let diagnosticsSession = state.freeRunSession
            let result: RuntimeWorkerFreeRunResult
            do {
                result = try runFreeDecode(
                    targetN: n,
                    route: route,
                    state: &state
                )
            } catch {
                emitFreeRunSessionDiagnostics(
                    route: route, session: diagnosticsSession,
                    state: state, result: nil)
                throw error
            }
            emitFreeRunSessionDiagnostics(
                route: route, session: diagnosticsSession,
                state: state, result: result)
            // R verify rounds → R completed_work units (the seed added one already).
            state.completedWork += result.rounds
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                tokens: result.tokens,
                acceptanceLengths: result.acceptanceLengths,
                draftedTotal: result.draftedTotal,
                acceptedTotal: result.acceptedTotal,
                committedTotal: result.committedTotal
            )

        case "record_reference_begin":
            // RECORDING SURFACE (trusted-CLI-only; `record-reference-tape
            // --recording-backend cbv2`). Open the SAME width-1 CBv2 engine
            // session the v1.1 free-run legs run — the shared begin executor
            // below, `openSingleStreamFreeRunSession`, in its pure serial
            // configuration (no drafter, no head, no spec) — so the pool
            // tapes are PRODUCED by the implementation that is oracle-checked
            // against them (port-notes 5.1, within-backend; the leg-identity
            // fix's own "re-record the tapes through the leg implementation"
            // consequence). The one recording-only difference is the
            // engine-side top-2 readout (`recordingTopLogprobs: 2`), which is
            // observability over the RAW logits and cannot move the argmax —
            // see the session init's doc note and the recording
            // executor-identity test.
            //
            // benchd never issues this kind (its verb set is pinned in its
            // own session code); it exists for the trusted operator CLI,
            // which spawns this worker itself. Sequencing and the seed guard
            // were checked by validateGenericWorkerRequest above.
            guard let recordSeedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_begin request missing seed_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let recordSeedToken = try openSingleStreamFreeRunSession(
                route: .serial,
                seedTokens: recordSeedTokens,
                model: try weightCache.requireLibraryModel(),
                mtpDrafter: nil,
                requestedDepth: nil,
                stopTokens: stopTokens,
                state: &state,
                recordingTopLogprobs: 2)
            // The shared begin executor stores its session in the free-run
            // slot; MOVE it to the recording slot so a recording window and a
            // benchd free-run window can never consume each other's session
            // (free_decode_run's guard keys on decodeRoute + freeRunSession,
            // both untouched here).
            state.recordingSession = state.freeRunSession
            state.freeRunSession = nil
            // Same accounting shape as the free-run begin (the seed forward
            // is one completed unit); the recorder never reads the phase
            // barrier, but the counter stays honest for phase_diagnostics.
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: recordSeedToken
            )

        case "record_reference_run":
            // RECORDING SURFACE: drain the recording session to N committed
            // rows and return the chain PLUS the per-row top-2 readout the
            // tape document carries (existing wire fields
            // `per_row_top2_tokens` / `per_row_top2_logits`; no new response
            // schema). Single-shot like free_decode_run.
            guard let recordingSession = state.recordingSession,
                let recordingN = validated.recordingRunCount
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_run has no open recording "
                        + "session (record_reference_begin did not open one, or "
                        + "it was already consumed — validation drift)")
            }
            state.recordingSession = nil
            let recordingResult: RuntimeWorkerFreeRunResult
            do {
                recordingResult = try recordingSession.run(targetN: recordingN)
            } catch {
                // Executor-identity evidence on failure paths too — the same
                // session-diagnostics line the free-run legs emit (route
                // serial, executor cbv2-width1-engine): this stderr line is
                // the tape-generation session's backend provenance.
                emitFreeRunSessionDiagnostics(
                    route: .serial, session: recordingSession,
                    state: state, result: nil)
                throw error
            }
            emitFreeRunSessionDiagnostics(
                route: .serial, session: recordingSession,
                state: state, result: recordingResult)
            let recordingRows = try assembleRecordingTopTwoRows(
                tokens: recordingResult.tokens,
                tokenLogprobs: recordingSession.tokenLogprobsSnapshot())
            state.completedWork += recordingResult.rounds
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                tokens: recordingResult.tokens,
                perRowTop2Tokens: recordingRows.top2Tokens,
                perRowTop2Logits: recordingRows.top2Logits
            )

        case "cohort_reference_replay":
            // MEASUREMENT-MODE REFERENCE-REPLAY ORACLE (trusted-CLI-only; fidelity
            // gate PR-1). The PINNED reference (this worker's own
            // organizer-transformed weights) replays each stream's committed
            // journal teacher-forced at the request's `replay_width` — `cohort`
            // (batch-B, the scored candidate's geometry, David-ruled default) or
            // `canonical` (per-stream width-1 diagnostic) — and returns the per
            // stream x position readout. Trusted-side invariant: the ranked logits
            // are the reference's own (`requireLibraryModel()`), never the
            // candidate's; nothing candidate-authored enters the readout (the
            // request carries only token ids), and the trusted parent runs this
            // AFTER candidate teardown so the two weight sets never coexist.
            // Renders NO admit/reject verdict — no scored path, no armed anything,
            // `official_scoring_enabled` stays false. Consumed later by Phase-2
            // calibration and the admission-ladder PR. Sequencing and bounds were
            // checked by validateGenericWorkerRequest above.
            guard let validatedReplay = validated.cohortReferenceReplay else {
                throw MLXFastError.invalidInput(
                    "runtime worker cohort_reference_replay reached the handler "
                        + "without a validated payload (validation drift)")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let replayReport = try cohortReferenceReplayReport(
                model: try weightCache.requireLibraryModel(),
                seedsByStream: validatedReplay.seedsByStream,
                committedByStream: validatedReplay.committedByStream,
                logitTopK: validatedReplay.logitTopK,
                relEnvelope: validatedReplay.relEnvelope,
                replayWidth: validatedReplay.replayWidth)
            state.completedWork += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                cohortReferenceReplay: replayReport)

        case "phase_diagnostics":
            // v1.2 (COHORT): a cohort session that is somehow still live at the
            // phase boundary (the run path already shut its engine down on the
            // normal path) is shut down NOW, before the allocator drain below —
            // a running engine loop would repopulate the cache the barrier
            // asserts empty, and its released KV must land in the cache before
            // clearCache() runs.
            if let cohortSession = state.cohortSession {
                cohortSession.shutdownBlocking()
                state.cohortSession = nil
                state.cohortSessionIsMTP = false
            }
            // v1.1 (SINGLE-STREAM): same belt-and-suspenders teardown for
            // a session `free_decode_run` did not consume.
            if let freeRunSession = state.freeRunSession {
                freeRunSession.shutdownBlocking()
                state.freeRunSession = nil
            }
            // RECORDING: same teardown for a session `record_reference_run`
            // did not consume.
            if let recordingSession = state.recordingSession {
                recordingSession.shutdownBlocking()
                state.recordingSession = nil
            }
            // Engine shutdown drains its serial queue, but it is not a Metal
            // completion fence. CBv2 submits through MLX's process-global GPU
            // stream, so retire that stream before snapshotting allocator state
            // or clearing free buffers. Otherwise a late command-buffer
            // retirement can repopulate cacheMemory after clearCache(), which
            // violates the exact-zero phase barrier even though the session is
            // already gone.
            Stream.gpu.synchronize()
            let peakRamGB = peakResidentMemoryGB()
            let stats = expertStats(from: weightCache)
            let mlxActiveMemoryBytes = Memory.activeMemory
            let mlxCacheMemoryBytes = Memory.cacheMemory
            let mlxPeakMemoryBytes = Memory.peakMemory
            Memory.clearCache()
            // benchd's phase-close barrier reads completed_work (== issued timed
            // steps, or R + 1 for a free-run phase) and asserts the allocator
            // drain via cache_memory (0 after clearCache). Report both, then reset
            // the phase counter across the boundary (parity with benchd's runner).
            let completedWork = state.completedWork
            let drainedCacheMemory = Memory.cacheMemory
            state.completedWork = 0
            state.decodeRoute = nil
            state.declaredSpec = nil
            state.freeRunLastToken = nil
            state.freeRunSeedTokenCount = nil
            state.freeRunConfigEcho = nil
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                expertStats: stats,
                peakRamGB: peakRamGB,
                mlxActiveMemoryBytes: mlxActiveMemoryBytes,
                mlxCacheMemoryBytes: mlxCacheMemoryBytes,
                mlxPeakMemoryBytes: mlxPeakMemoryBytes,
                completedWork: completedWork,
                cacheMemory: drainedCacheMemory
            )

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
    }


    /// The single seed (whole-prompt) forward that opens a plain decode / free-run
    /// window: one Qwen forward at offset 0, materialized so the KV writes are
    /// complete before any following timed step. Stores the decode cache/offset on
    /// `state` and returns the seed token. Shared by decode_begin and the serial
    /// free_decode_begin.
    static func plainSeedForward(
        seedTokens: [Int],
        model: Gemma4TextModel,
        state: inout RuntimeWorkerState
    ) throws -> Int {
        let cache = model.newCache(parameters: nil)
        let logits = try gemma4Logits(
            inputIDs: inputIDsArray(seedTokens),
            model: model,
            cache: cache,
            positionOffset: 0
        )
        let token = try Gemma4Correctness.greedyToken(from: logits)
        materializeQwenCacheState(cache)
        state.decodeCache = cache
        state.decodeSeedTokenCount = seedTokens.count
        state.decodeStep = 0
        return token
    }

    /// One plain single-token decode forward from `inputToken`, advancing exactly
    /// one KV position. Shared by decode_step and the serial free-run loop. Invokes
    /// only the same editable entry points the correctness path uses (phase-agnostic).
    static func plainDecodeStep(
        inputToken: Int,
        model: Gemma4TextModel,
        state: inout RuntimeWorkerState
    ) throws -> Int {
        guard let cache = state.decodeCache else {
            throw MLXFastError.invalidInput("runtime worker decode_step before decode_begin")
        }
        let logits = try gemma4Logits(
            inputIDs: inputIDsArray([inputToken]),
            model: model,
            cache: cache,
            positionOffset: state.decodeSeedTokenCount + state.decodeStep
        )
        let token = try Gemma4Correctness.greedyToken(from: logits)
        state.decodeStep += 1
        return token
    }

    /// Free-run the engine's own decode loop until N committed tokens, assembling
    /// the AUDIT counters benchd's §2.6 consistency triple cross-checks.
    ///
    ///   * serial — a drafter-less width-1 engine session: every round is one
    ///     plain forward committing exactly one token, so acceptance_lengths
    ///     is `[1]*N` and drafted/accepted are structural zeros (honest
    ///     serial control).
    ///   * mtp — the drafter-bound session's REAL draft/verify rounds.
    ///
    /// The route emits what a round PRODUCED, never what it wrote. The seed token
    /// `free_decode_begin` already returned is NOT re-emitted here — PROTOCOL-v1.1
    /// §2.2 verifies it separately against `expected_decode_seed_token` and starts
    /// `expected_decode_tokens[0]` after it (#109 W3 finding 6).
    ///
    /// The final round is clamped so the histogram sums to exactly N.
    ///
    /// EOS SYMMETRY. A leg that commits a stop token before it has committed N
    /// tokens throws `RuntimeWorkerFreeRunError.stopTokenBeforeTarget`, naming
    /// the leg, the token and the committed position it landed at. Paired legs
    /// must fail symmetrically: one side stopping early while the other runs on
    /// puts the difference straight into the ratio those two legs exist to
    /// produce — both legs now detect it through the same engine finalize
    /// path over the same stop-token set.
    ///
    /// A stop token committed AT position N is not an error — the leg delivered
    /// exactly what was asked for, and nothing was truncated.
    static func runFreeDecode(
        targetN: Int,
        route: RuntimeWorkerDecodeRoute,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerFreeRunResult {
        // BOTH routes drain the width-1 CBv2 engine session
        // `free_decode_begin` opened (`RuntimeWorkerFreeRunSession`,
        // Gemma4RuntimeMTPDriver.swift) — the serial leg's session is
        // drafter-less plain decode, the mtp leg's runs REAL draft/verify
        // rounds through the vendored `CBv2MTPRoundDriver`. One decode
        // implementation computes both legs' committed streams
        // (leg-implementation identity — see the free_decode_begin arm's
        // header note). The session assembles and validates its own
        // `RuntimeWorkerFreeRunResult` (acceptance histogram, clamping, the
        // symmetric early-EOS verdict) because it observes rounds as
        // delivered chunks, not as a loop this function drives step by
        // step.
        // The DFLASH arm drains its own round loop (no CBv2 engine); the
        // assembled `RuntimeWorkerFreeRunResult` — acceptance histogram,
        // clamping, the symmetric early-EOS verdict — comes off the SAME
        // `RuntimeWorkerFreeRunBuilder` the other two legs use, so the wire
        // shape and the §2.6 consistency triple are identical.
        if route == .dflash {
            guard let dflashSession = state.dflashFreeRunSession else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_run dflash route has no matching "
                        + "session (free_decode_begin did not open one, or it was "
                        + "already consumed — validation drift)")
            }
            state.dflashFreeRunSession = nil
            return try dflashSession.run(targetN: targetN)
        }
        guard let session = state.freeRunSession else {
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_run \(route.rawValue) route has no "
                    + "matching session (free_decode_begin did not open one, or "
                    + "it was already consumed — validation drift)")
        }
        let result = try session.run(targetN: targetN)
        state.freeRunSession = nil
        return result
    }

    /// The one place a v1.1 (single-stream) free-run phase opens its engine:
    /// a width-1 CBv2 cohort engine over the resident model — drafter-less
    /// for the serial leg, bound to the loaded assistant head (with the
    /// envelope-sealed config) for the mtp leg — wrapped in a
    /// `RuntimeWorkerFreeRunSession` stored on `state`. Returns the
    /// session's seed token (the prefill-bonus argmax `free_decode_begin`
    /// puts on the wire). Extracted from the begin arm so tests drive the
    /// EXACT executor the wire dispatches to.
    static func openSingleStreamFreeRunSession(
        route: RuntimeWorkerDecodeRoute,
        seedTokens: [Int],
        model: Gemma4TextModel,
        mtpDrafter: Gemma4CBv2MTPDrafter?,
        dflashDrafter: DFlashDraftModel? = nil,
        requestedDepth: Int?,
        stopTokens: Set<Int>,
        state: inout RuntimeWorkerState,
        recordingTopLogprobs: Int = 0
    ) throws -> Int {
        let maxTokens =
            MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens + 1
        // The DFLASH arm does not open a CBv2 engine at all (see
        // `Gemma4DFlashFreeRunSession.swift` for the three structural reasons
        // the CBv2 drafter seam cannot carry a `DFlashDraftModel`), so it
        // returns from its own branch rather than falling through to the
        // shared `state.freeRunSession` assignment below.
        if case .dflash = route {
            guard let dflashDrafter else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_begin resolved dflash mode but "
                        + "this worker has no bound DFlash drafter; spec "
                        + "resolution should have refused dflash for a worker "
                        + "with no staged dflash-head (wiring bug)")
            }
            // The depth the ECHO already committed to. `requestedDepth` here
            // is `effective_spec.dflash.depth` — the value
            // `RuntimeWorkerSpecRegistry.resolveDFlashDepth` clamped against
            // this drafter's own block ceiling — so the round loop runs
            // exactly what the caller was told it would run. A nil can only
            // reach here from a non-wire caller (tests), which gets the same
            // clamp against the drafter's own ceiling rather than a
            // CBv2 envelope constant.
            let depth =
                requestedDepth
                ?? gemma4DFlashMaxDepth(for: dflashDrafter)
            let dflashSession = try RuntimeWorkerDFlashFreeRunSession(
                target: model,
                drafter: dflashDrafter,
                seedTokens: seedTokens,
                depth: depth,
                stopTokens: stopTokens)
            state.dflashFreeRunSession = dflashSession
            state.freeRunSeedTokenCount = seedTokens.count
            state.freeRunConfigEcho = (
                source: "DFlashDraftModel.draftBlock(blockSize: depth + 1)",
                mode: "dflash_greedy_block",
                cap: depth + 1,
                depth: depth
            )
            return dflashSession.seedToken
        }
        let session: RuntimeWorkerFreeRunSession
        switch route {
        case .serial:
            let engine = try makeCohortEngine(
                model: model,
                batchSize: 1,
                seedTokenCount: seedTokens.count,
                maxTokensPerStream: maxTokens)
            session = try RuntimeWorkerFreeRunSession(
                engine: engine,
                mode: .serial,
                seedTokens: seedTokens,
                maxTokens: maxTokens,
                stopTokens: stopTokens,
                // 0 on every wire leg; > 0 only for the recording verbs
                // (`record_reference_begin`), whose per-row top-2 readout
                // rides the session's collector — see the session init's
                // doc note for why the readout cannot move the argmax.
                recordingTopLogprobs: recordingTopLogprobs)
            state.freeRunConfigEcho = nil
        case .mtp:
            guard let mtpDrafter else {
                throw MLXFastError.invalidInput(
                    "runtime worker free_decode_begin resolved mtp mode but "
                        + "this worker has no bound assistant-head drafter; "
                        + "spec resolution should have refused mtp for a "
                        + "headless worker (wiring bug)")
            }
            let depth = requestedDepth ?? Gemma4MTPEnvelope.maxDraftTokens
            let mtpConfig = try Gemma4MTPEnvelope.resolveConfig(depth: depth)
            let engine = try makeCohortEngine(
                model: model,
                batchSize: 1,
                seedTokenCount: seedTokens.count,
                maxTokensPerStream: maxTokens,
                mtpDrafter: mtpDrafter,
                mtpConfig: mtpConfig)
            try requireMTPActive(engine)
            session = try RuntimeWorkerFreeRunSession(
                engine: engine,
                mode: .mtp,
                seedTokens: seedTokens,
                maxTokens: maxTokens,
                stopTokens: stopTokens)
            // Observability (exactness round two): echo what was SEALED, at
            // the seam it was sealed from, so the on-box session log
            // observes the causal config instead of assuming it.
            state.freeRunConfigEcho = (
                source: "Gemma4MTPEnvelope.resolveConfig(depth:)",
                mode: mtpConfig.verificationMode.rawValue,
                cap: mtpConfig.maxAutomaticRectangularTokens,
                depth: depth
            )
        case .dflash:
            // Unreachable: handled by the early return above, which is where
            // it must live because the dflash arm runs no CBv2 engine.
            throw MLXFastError.invalidInput(
                "runtime worker free_decode_begin reached the CBv2 session "
                    + "switch on the dflash route (wiring bug)")
        }
        state.freeRunSession = session
        state.freeRunSeedTokenCount = seedTokens.count
        return session.seedToken
    }

    /// Recording surface: turn the session collector's per-token top-logprob
    /// readouts into the per-row top-2 arrays the tape document carries.
    /// `tokens` is the run result's committed chain (post-seed, clamped to
    /// N); `tokenLogprobs` is the collector snapshot, 1:1 with the FULL
    /// committed order whose index 0 is the seed token — so row `i` of the
    /// chain reads `tokenLogprobs[i + 1]`.
    ///
    /// Pure and model-free (unit-tested directly). Fail-closed: a row with a
    /// missing readout, fewer than 2 alternatives, or a readout naming a
    /// different token than the one the engine committed is a wiring fault
    /// and throws — the recorder must never write a tape whose diagnostics
    /// describe some other forward. Semantic tape invariants (top-2[0] ==
    /// argmax, ordering, finiteness) are re-enforced by the TRUSTED
    /// recorder on the other side of the wire; the checks here just fail
    /// early with a message that names the row.
    static func assembleRecordingTopTwoRows(
        tokens: [Int],
        tokenLogprobs: [CBv2TokenLogprob?]
    ) throws -> (top2Tokens: [[Int]], top2Logits: [[Double]]) {
        guard tokenLogprobs.count >= tokens.count + 1 else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_run collected "
                    + "\(tokenLogprobs.count) top-logprob readouts for "
                    + "\(tokens.count) committed rows plus the seed; the "
                    + "recording session did not observe every forward")
        }
        var top2Tokens: [[Int]] = []
        var top2Logits: [[Double]] = []
        top2Tokens.reserveCapacity(tokens.count)
        top2Logits.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() {
            // + 1: index 0 of the collector order is the seed token, whose
            // readout the tape does not carry (rows start after the seed).
            guard let readout = tokenLogprobs[index + 1] else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_run rows[\(index)] has no "
                        + "top-logprob readout; the engine reported none for "
                        + "this forward")
            }
            guard readout.token == token else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_run rows[\(index)] "
                        + "readout describes token \(readout.token) but the "
                        + "engine committed \(token) (collector misalignment)")
            }
            guard readout.topLogprobs.count >= 2 else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_run rows[\(index)] "
                        + "carries \(readout.topLogprobs.count) top logprobs; "
                        + "a reference tape row needs the top 2")
            }
            let top = Array(readout.topLogprobs.prefix(2))
            top2Tokens.append([top[0].token, top[1].token])
            top2Logits.append([Double(top[0].logprob), Double(top[1].logprob)])
        }
        return (top2Tokens, top2Logits)
    }

    /// Assemble and emit one free-run leg's session diagnostics to stderr
    /// (RuntimeWorkerFreeRunDiagnostics.swift — observability, never the
    /// wire). Called on success AND failure paths of `free_decode_run`;
    /// `result` is nil when the run threw (the audit records and strategy
    /// counters drained into `session.finalMetrics` before the throw, so a
    /// failed leg still reports its rounds).
    static func emitFreeRunSessionDiagnostics(
        route: RuntimeWorkerDecodeRoute,
        session: RuntimeWorkerFreeRunSession?,
        state: RuntimeWorkerState,
        result: RuntimeWorkerFreeRunResult?
    ) {
        let diagnostics = makeFreeRunSessionDiagnostics(
            route: route, session: session, state: state, result: result)
        diagnostics.emitToStandardError()
        // Sidecar transport (round three): stderr does not survive into the
        // sealed evidence on success paths, so the same diagnostics also
        // land as one JSON line in the operator-configured sidecar file —
        // see RuntimeWorkerFreeRunDiagnostics.swift for the channel
        // contract. A no-op when the environment variable is absent.
        diagnostics.writeSidecar()
    }

    /// Pure assembly half of the emission above, split out so tests can pin
    /// the REAL wire arm's reported values (executor identity above all)
    /// without scraping stderr.
    static func makeFreeRunSessionDiagnostics(
        route: RuntimeWorkerDecodeRoute,
        session: RuntimeWorkerFreeRunSession?,
        state: RuntimeWorkerState,
        result: RuntimeWorkerFreeRunResult?
    ) -> RuntimeWorkerFreeRunSessionDiagnostics {
        let metrics = session?.finalMetrics
        let executor: String
        switch route {
        case .serial:
            // Engine-backed since the leg-implementation-identity fix
            // (2026-08-25); a nil session here would mean the legacy
            // single-token loop computed the leg — the asymmetry the fix
            // removed — and the diagnostics say so rather than assume.
            executor =
                session == nil
                ? RuntimeWorkerFreeRunExecutor.legacySingleTokenLoop
                : RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine
        case .mtp, .dflash:
            // Both speculative arms run the drafter-bound CBv2 engine — same
            // executor identity; the route field already names which arm.
            executor = RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine
        }
        return RuntimeWorkerFreeRunSessionDiagnostics(
            route: route,
            executor: executor,
            seedTokenCount: state.freeRunSeedTokenCount ?? -1,
            committedTotal: result?.committedTotal ?? -1,
            rounds: result?.rounds ?? -1,
            draftedTotal: result?.draftedTotal ?? metrics.map {
                Swift.max(0, $0.draftedTokens)
            } ?? -1,
            acceptedTotal: result?.acceptedTotal ?? metrics.map {
                Swift.max(0, $0.acceptedTokens)
            } ?? -1,
            configSource: state.freeRunConfigEcho?.source,
            verificationMode: state.freeRunConfigEcho?.mode,
            rectangularCap: state.freeRunConfigEcho?.cap,
            requestedDepth: state.freeRunConfigEcho?.depth,
            serialVerifyRounds: metrics?.serialVerificationRounds,
            rectangularVerifyRounds: metrics?.rectangularVerificationRounds,
            seedSteps: metrics?.seedSteps,
            roundAudits: metrics?.roundAudits ?? []
        )
    }

}

func validateRuntimeWorkerPinnedConfiguration(weightsPath: String) throws {
    let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
    let values = try path.resourceValues(
        forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]
    )
    let maximumConfigByteCount = 1 * 1024 * 1024
    guard values.isRegularFile == true,
          values.isSymbolicLink != true,
          let byteCount = values.fileSize,
          byteCount > 0,
          byteCount <= maximumConfigByteCount
    else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json must be a non-symlink regular file no larger than \(maximumConfigByteCount) bytes"
        )
    }
    try validateRuntimeWorkerPinnedConfigurationData(Data(contentsOf: path))
}

/// Accept exactly the transformed `mlx-community/gemma-4-26B-A4B-it-qat-4bit`
/// artifact this track targets -- pinned revision
/// `0e3cbab38ce568cf6e23543010d08d03b731910c`, see
/// `docs/gemma4-port-notes.md` section 1 for the pin and the geometry table
/// this gate restates.
///
/// SINGLE-SOURCED with `Gemma4A4BConfig`
/// (`Sources/MLXFastModel/Gemma4A4BConfig.swift`): that type's `load(data:)`
/// already performs the exact key-set check (every required key present and
/// non-null, every forbidden key absent, no unknown key -- including the
/// 120-entry per-tensor quantization-override table), the frozen invariant
/// check (`model_type == "gemma4_text"`, the 30-layer six-group RoPE/layer
/// schedule, the affine group-64 quantization fallback, ...), and the
/// structural sanity check. This function calls directly into it instead of
/// re-encoding a second, hand-maintained key/value list, which is what let
/// the previous (Qwen-era) version of this gate silently drift out of sync
/// with the artifact it was actually validating: it still enforced the
/// `qwen3_5_text` schema, so a real transformed Gemma 4 config failed here
/// before weight loading even began --
/// missing=["attn_output_gate","full_attention_interval","hidden_act",
/// "linear_conv_kernel_dim","linear_key_head_dim","linear_num_key_heads",
/// "linear_num_value_heads","linear_value_head_dim","mamba_ssm_dtype",
/// "mtp_num_hidden_layers","mtp_use_dedicated_embeddings",
/// "partial_rotary_factor"],
/// unexpected=["attention_k_eq_v","enable_moe_block",
/// "final_logit_softcapping","global_head_dim","hidden_activation",
/// "hidden_size_per_layer_input","moe_intermediate_size","num_experts",
/// "num_global_key_value_heads","num_kv_shared_layers","sliding_window",
/// "top_k_experts","use_bidirectional_attention","use_double_wide_mlp",
/// "vocab_size_per_layer_input"] -- exactly the box-observed refusal.
/// Routing both checks through one loader is what keeps that from
/// recurring: there is only one place this target's config contract is
/// written down.
///
/// SINGLE-SOURCED here (the participant worker target links
/// `MLXFastModel`), but NOT in the `MLXFastTrustedHarness` twin: the trusted
/// `mlxfast-swift` target deliberately does not depend on `MLXFastModel` (it
/// links no MLX, model, or kernel code -- see `Package.swift`'s
/// `MLXFastHarness` target, path `Sources/MLXFastTrustedHarness`), so
/// `Gemma4A4BConfig` is out of scope there. That twin re-derives the same
/// contract locally instead, sharing only the static key manifest
/// (`MLXFastCore.Gemma4A4BConfigKeys`) and the geometry
/// (`MLXFastConstants`) that both sides can reach; the two are kept in
/// lockstep by `gemma4A4BTrustedGateAgreesWithConfigLoaderAcrossFixtures` in
/// `Tests/MLXFastTests/Model/Gemma4A4BRuntimeWorkerGateLockstepTests.swift`
/// rather than by sharing one function body.
func validateRuntimeWorkerPinnedConfigurationData(_ data: Data) throws {
    _ = try Gemma4A4BConfig.load(data: data)
}

struct RuntimeWorkerRequest: Codable {
    let id: Int
    let kind: String
    let promptTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let steps: Int?
    let topK: Int?
    let expectedToken: Int?
    let maxBlockSize: Int?
    // Reference-side only (dflash_reference_rows). The REFERENCE worker -- built
    // from the pinned baseline tree and loading organizer weights -- is the only
    // party that receives these. The candidate worker never sees this kind, so
    // there is no in-band verify opcode for a submission to detect.
    let prefixTokens: [Int]?
    let startOffset: Int?
    let rowCount: Int?
    let declaredBlockWidth: Int?
    // How much of `prefixTokens` is the SEED, i.e. the only span the candidate
    // ever bulk-prefills. The reference needs it to build the width-1 frame the
    // way the candidate does -- one bulk forward over the seed, then one
    // single-token forward per position after it. Without it the reference can
    // only guess, and guessing `prefixTokens[0 ..< start_offset]` is the frame
    // bug this field exists to remove.
    let seedTokenCount: Int?
    // Reference-side only (dflash_reference_rows). The candidate's ACTUAL verify
    // block for the round being replayed: `[bonus, d0, ..., d_{K-2}]`, built by
    // the trusted parent from its own committed token plus the drafts the round
    // journalled. It is what lets the reference reach the REJECTED tail rows --
    // the emitted context stops describing the verify input at the first
    // rejection, so without this the tail is compared to nothing.
    let verifyBlockTokens: [Int]?
    // Optional per-module speculative configuration on a `decode_begin`. A
    // tagged union `{ "mode": … }` the benchd-facing worker parses, resolves and
    // echoes as `effective_spec`. Absent = engine default (v1 callers such as
    // the mtp-timed driver send none and are unchanged). Wire-additive; consumed
    // only by the MTP-track begin, rejected as a cross-kind field elsewhere.
    let spec: RuntimeWorkerSpecRequest?
    // v1.2 (additive, COHORT — batched free-run): per-stream seed token IDs for
    // a batched `free_decode_begin`, B inner arrays in SLOT ORDER. The
    // single-stream v1.1 form keeps `seed_tokens`; a request carries one or the
    // other, never both (validated fail-closed).
    let seedTokensByStream: [[Int]]?
    // v1.2 (additive, COHORT): the EXPLICIT cohort width B, carried on both
    // batched free-run kinds. Never inferred from an array length alone; echoed
    // back as `effective_batch_size` (never-ignored on benchd's side). Presence
    // selects the cohort form of the verb.
    let batchSize: Int?
    // PR-1 fidelity-gate (additive, COHORT REFERENCE-REPLAY ORACLE;
    // trusted-CLI-only). `cohort_reference_replay` carries B streams' seeds
    // (`replay_seeds_by_stream`) and the candidate's committed journals
    // (`committed_by_stream`); the pinned reference replays each teacher-forced
    // at canonical width-1 and reports its own per-position readouts. A DEDICATED
    // seed field (not the free-run cohort's `seed_tokens_by_stream`) keeps this
    // trusted-CLI reference surface uncoupled from the benchd-facing free-run
    // cohort guard. `logit_top_k` / `rel_envelope` are MEASUREMENT
    // characterization params (default 16 / 0.05, swept by calibration), NOT
    // admission constants — the verb renders no verdict. All four ride ONLY on
    // `cohort_reference_replay` (rejected cross-kind).
    let replaySeedsByStream: [[Int]]?
    let committedByStream: [[Int]]?
    let logitTopK: Int?
    let relEnvelope: Double?
    // The reference's replay WIDTH for `cohort_reference_replay`:
    // `"cohort"` (batch-B, the scored candidate's geometry, David-ruled default)
    // or `"canonical"` (per-stream width-1 diagnostic). Absent ⇒ the default
    // (`cohortReferenceReplayDefaultWidth`, currently `cohort`). Rides ONLY on
    // `cohort_reference_replay` (rejected cross-kind). The reference replays at
    // the candidate's cohort width so the divergence comparison is like-for-like
    // and the false batch-geometry divergence (port-notes 3.1) is priced out.
    let replayWidth: String?

    init(
        id: Int,
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil,
        topK: Int? = nil,
        expectedToken: Int? = nil,
        maxBlockSize: Int? = nil,
        prefixTokens: [Int]? = nil,
        startOffset: Int? = nil,
        rowCount: Int? = nil,
        declaredBlockWidth: Int? = nil,
        seedTokenCount: Int? = nil,
        verifyBlockTokens: [Int]? = nil,
        spec: RuntimeWorkerSpecRequest? = nil,
        seedTokensByStream: [[Int]]? = nil,
        batchSize: Int? = nil,
        replaySeedsByStream: [[Int]]? = nil,
        committedByStream: [[Int]]? = nil,
        logitTopK: Int? = nil,
        relEnvelope: Double? = nil,
        replayWidth: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.promptTokens = promptTokens
        self.token = token
        self.seedTokens = seedTokens
        self.steps = steps
        self.topK = topK
        self.expectedToken = expectedToken
        self.maxBlockSize = maxBlockSize
        self.prefixTokens = prefixTokens
        self.startOffset = startOffset
        self.rowCount = rowCount
        self.declaredBlockWidth = declaredBlockWidth
        self.seedTokenCount = seedTokenCount
        self.verifyBlockTokens = verifyBlockTokens
        self.spec = spec
        self.seedTokensByStream = seedTokensByStream
        self.batchSize = batchSize
        self.replaySeedsByStream = replaySeedsByStream
        self.committedByStream = committedByStream
        self.logitTopK = logitTopK
        self.relEnvelope = relEnvelope
        self.replayWidth = replayWidth
    }

    init(from decoder: Swift.Decoder) throws {
        let wireContainer = try decoder.container(
            keyedBy: RuntimeWorkerWireCodingKey.self
        )
        let allowedKeys = Set(CodingKeys.allCases.map(\.rawValue))
        if let unknownKey = wireContainer.allKeys.first(
            where: { !allowedKeys.contains($0.stringValue) }
        ) {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath + [unknownKey],
                    debugDescription:
                        "runtime worker request contains unknown field \(unknownKey.stringValue)"
                )
            )
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        promptTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .promptTokens
        )
        token = try container.decodeIfPresent(Int.self, forKey: .token)
        seedTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .seedTokens
        )
        steps = try container.decodeIfPresent(Int.self, forKey: .steps)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        expectedToken = try container.decodeIfPresent(
            Int.self,
            forKey: .expectedToken
        )
        maxBlockSize = try container.decodeIfPresent(
            Int.self,
            forKey: .maxBlockSize
        )
        prefixTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .prefixTokens
        )
        startOffset = try container.decodeIfPresent(Int.self, forKey: .startOffset)
        rowCount = try container.decodeIfPresent(Int.self, forKey: .rowCount)
        declaredBlockWidth = try container.decodeIfPresent(
            Int.self,
            forKey: .declaredBlockWidth
        )
        seedTokenCount = try container.decodeIfPresent(
            Int.self,
            forKey: .seedTokenCount
        )
        verifyBlockTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .verifyBlockTokens
        )
        spec = try container.decodeIfPresent(
            RuntimeWorkerSpecRequest.self,
            forKey: .spec
        )
        seedTokensByStream = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .seedTokensByStream
        )
        batchSize = try container.decodeIfPresent(Int.self, forKey: .batchSize)
        replaySeedsByStream = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .replaySeedsByStream
        )
        committedByStream = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .committedByStream
        )
        logitTopK = try container.decodeIfPresent(Int.self, forKey: .logitTopK)
        relEnvelope = try container.decodeIfPresent(
            Double.self,
            forKey: .relEnvelope
        )
        replayWidth = try container.decodeIfPresent(
            String.self,
            forKey: .replayWidth
        )
    }

    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(kind, forKey: .kind)
        try container.encodeIfPresent(promptTokens, forKey: .promptTokens)
        try container.encodeIfPresent(token, forKey: .token)
        try container.encodeIfPresent(seedTokens, forKey: .seedTokens)
        try container.encodeIfPresent(steps, forKey: .steps)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(expectedToken, forKey: .expectedToken)
        try container.encodeIfPresent(maxBlockSize, forKey: .maxBlockSize)
        try container.encodeIfPresent(prefixTokens, forKey: .prefixTokens)
        try container.encodeIfPresent(startOffset, forKey: .startOffset)
        try container.encodeIfPresent(rowCount, forKey: .rowCount)
        try container.encodeIfPresent(
            declaredBlockWidth,
            forKey: .declaredBlockWidth
        )
        try container.encodeIfPresent(seedTokenCount, forKey: .seedTokenCount)
        try container.encodeIfPresent(
            verifyBlockTokens,
            forKey: .verifyBlockTokens
        )
        try container.encodeIfPresent(spec, forKey: .spec)
        try container.encodeIfPresent(
            seedTokensByStream,
            forKey: .seedTokensByStream
        )
        try container.encodeIfPresent(batchSize, forKey: .batchSize)
        try container.encodeIfPresent(
            replaySeedsByStream,
            forKey: .replaySeedsByStream
        )
        try container.encodeIfPresent(
            committedByStream,
            forKey: .committedByStream
        )
        try container.encodeIfPresent(logitTopK, forKey: .logitTopK)
        try container.encodeIfPresent(relEnvelope, forKey: .relEnvelope)
        try container.encodeIfPresent(replayWidth, forKey: .replayWidth)
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
        case topK = "top_k"
        case expectedToken = "expected_token"
        // DFlash/MTP block decode: the parent-chosen block width for this
        // round. Deliberately the ONLY block-shaped field on the request --
        // the worker is never told the remaining or total decode length,
        // which would let it special-case the tail of the scored window.
        case maxBlockSize = "max_block_size"
        case prefixTokens = "prefix_tokens"
        case startOffset = "start_offset"
        case rowCount = "count"
        case declaredBlockWidth = "declared_block_width"
        case seedTokenCount = "seed_token_count"
        case verifyBlockTokens = "verify_block_tokens"
        case spec
        case seedTokensByStream = "seed_tokens_by_stream"
        case batchSize = "batch_size"
        case replaySeedsByStream = "replay_seeds_by_stream"
        case committedByStream = "committed_by_stream"
        case logitTopK = "logit_top_k"
        case relEnvelope = "rel_envelope"
        case replayWidth = "replay_width"
    }
}

private struct RuntimeWorkerWireCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

struct RuntimeWorkerState {
    var correctnessCache: [KVCache]?
    var correctnessPromptTokenCount = 0
    var correctnessStep = 0
    var decodeCache: [KVCache]?
    var decodeSeedTokenCount = 0
    var decodeStep = 0
    // Generic benchd-facing dispatch bookkeeping.
    //
    // `completedWork` is the phase-close barrier counter: bumped on every timed
    // forward benchd counts (decode_begin / decode_step / correctness_begin /
    // correctness_step, and — for a free-run phase — the free_decode_begin seed
    // plus each free_decode_run verify round), read out and reset on
    // phase_diagnostics. It equals benchd's `issued_steps` for a teacher-forced
    // phase and `R + 1` for a free-run phase.
    var completedWork = 0
    /// The decode route sealed on the current (free_)decode_begin from the
    /// resolved effective_spec, honored by the matching decode_step / run.
    var decodeRoute: RuntimeWorkerDecodeRoute?
    /// The effective spec sealed on the current (free_)decode_begin, echoed to
    /// benchd.
    var declaredSpec: RuntimeWorkerEffectiveSpec?
    /// The most recently committed token in a free-run window, fed as the next
    /// serial forward's input.
    var freeRunLastToken: Int?
    /// v1.2 (COHORT): the live batched free-run session sealed by a batched
    /// `free_decode_begin` — the CBv2 engine plus its per-slot stream
    /// collectors. `nil` outside a cohort phase; cleared by the batched run on
    /// completion and (belt-and-suspenders) by `phase_diagnostics`.
    var cohortSession: RuntimeWorkerCohortSession?
    /// Whether `cohortSession` was opened with a bound MTP drafter (`true`)
    /// or is the plain target-only serial cohort engine (`false`). Decides
    /// which assembler `handleCohortFreeDecodeRun` calls; reset alongside
    /// `cohortSession` on completion and at `phase_diagnostics`.
    var cohortSessionIsMTP = false
    /// v1.1 (SINGLE-STREAM): the live engine-backed free-run session sealed
    /// by a non-cohort `free_decode_begin` — the width-1 CBv2 engine plus
    /// its one-slot collector. `nil` outside an engine-backed free-run
    /// phase; cleared by `free_decode_run` on completion and
    /// (belt-and-suspenders) by `phase_diagnostics`.
    var freeRunSession: RuntimeWorkerFreeRunSession?
    /// v1.1 (SINGLE-STREAM, DFLASH ARM): the live DFlash round-loop session
    /// sealed by a `free_decode_begin` whose route resolved to `.dflash`.
    /// Its own slot rather than a case of `freeRunSession` because the arm
    /// runs no CBv2 engine at all (target + drafter + two plain `[KVCache]`
    /// arrays — see `Gemma4DFlashFreeRunSession.swift`). Exactly one of the
    /// two slots is ever non-nil for a given phase; both are cleared the
    /// same way.
    var dflashFreeRunSession: RuntimeWorkerDFlashFreeRunSession?
    /// RECORDING (trusted-CLI-only): the live engine-backed session opened
    /// by `record_reference_begin` — the SAME width-1 CBv2 session type as
    /// `freeRunSession`, held in its own slot so a recording window and a
    /// benchd free-run window can never consume each other's session.
    /// Cleared by `record_reference_run` on consumption and
    /// (belt-and-suspenders) by `phase_diagnostics`.
    var recordingSession: RuntimeWorkerFreeRunSession?
    /// v1.1 observability: the seed prompt length of the open free-run
    /// phase, echoed in the session diagnostics so audit records' absolute
    /// token counts convert to benchd step indices.
    var freeRunSeedTokenCount: Int?
    /// v1.1 observability (mtp leg): what `free_decode_begin` sealed —
    /// config source, verification mode, rectangular cap, resolved depth —
    /// echoed at `free_decode_run` so the session log OBSERVES the sealed
    /// strategy rather than assuming it.
    var freeRunConfigEcho: (source: String, mode: String, cap: Int, depth: Int)?
    /// Fail-closed latch (fix #4): once any request in this session errors, the
    /// session is poisoned and every subsequent request is rejected. Matches the
    /// PROTOCOL.md's session-discard-on-error invariant — an
    /// errored forward may have advanced lazy KV/recurrent cache metadata, so a
    /// half-advanced cache must never be reused.
    var poisoned = false
}

struct RuntimeWorkerPreflightResponse: Codable, Equatable {
    let ok: Bool
    let error: String?

    init(ok: Bool, error: String? = nil) {
        self.ok = ok
        self.error = error
    }
}

struct RuntimeWorkerResponse: Codable {
    let id: Int
    let nonce: String?
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let expectedTokenLogit: Double?
    let expectedTokenRank: Int?
    let topLogitMargin: Double?
    let seedToken: Int?
    let tokens: [Int]?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?
    let mlxActiveMemoryBytes: Int?
    let mlxCacheMemoryBytes: Int?
    let mlxPeakMemoryBytes: Int?
    let targetVerificationMode: String?
    let exactPairSegmentCount: Int?
    let exactPairRollbackRowCount: Int?
    let serialVerificationRowCount: Int?
    // Criterion E work-binding diagnostics (DFlash block decode). The primary
    // token predicate cannot price a verifier that is cheap at confident steps,
    // so the parent also binds emitted tokens to executed target compute:
    // `declaredRows` is what the worker claims it pushed through the target this
    // round, and the per-row readouts are checked against the pinned reference
    // for EVERY declared row including rejected ones. The hidden digest forces
    // the trunk; the top-2 logit VALUES force the lm_head per row.
    let declaredRows: Int?
    let perRowHiddenDigest: [String]?
    let perRowTop2Tokens: [[Int]]?
    let perRowTop2Logits: [[Double]]?
    // The round's drafter proposals, `declaredRows - 1` of them, in verify-input
    // order. The parent binds them to the emitted tokens with no reference at all
    // (an accepted draft IS the emitted token at that index) and hands them back
    // to the reference so it can replay the round's real verify block and price
    // the REJECTED tail rows. Before this field the tail carried readouts nothing
    // compared to anything.
    let draftTokens: [Int]?
    let acceptedDraftCount: Int?
    let rejectedDraftCount: Int?
    let rollbackRoundCount: Int?
    let targetCacheOffset: Int?
    let kvDigest: String?
    let kvVacancyDigest: String?
    // Reference-side verdicts (dflash_reference_rows). Parallel arrays, one
    // entry per requested row: the width-1 argmax, the block-frame argmax, and
    // the top-2 ids/VALUES that Amendment 1 makes the cross-build work binder.
    let referenceK1Argmax: [Int]?
    let referenceBlockArgmax: [Int]?
    let referenceTop2Tokens: [[Int]]?
    let referenceTop2Logits: [[Double]]?
    // Amendment 16: the reference's own top-1 logit per row, plus the token the
    // row PREDICTS and that token's reference logit. Those last two make the
    // near-tie test a question about the emitted token itself rather than about
    // membership in a fixed-size shortlist, which a three-way tie cannot
    // express. They cover a PREFIX of the requested rows -- a request whose
    // context stops at its final row has no next token there -- so a consumer
    // must bound-check instead of assuming one entry per row.
    let referenceTop1Logits: [Double]?
    let referenceEmittedTokens: [Int]?
    let referenceEmittedTokenLogits: [Double]?
    // Every block width the reference replayed for these rows, ascending, and
    // the argmax each one produced. One request covers all of them so the
    // reference can branch each frame off the SAME continuous cache without
    // rewinding, which is what keeps the pass O(n) instead of O(n^2).
    let referenceFrameWidths: [Int]?
    let referenceFrameArgmax: [[Int]]?
    // Per-row top-2 ids/VALUES for EVERY row of the candidate's own verify block,
    // when the request supplied one. This is the only reference readout that
    // reaches a REJECTED row: the emitted-context frames stop matching the
    // candidate's verify input at the first rejection, so before this the
    // rejected tail's readouts were length-checked and discarded.
    let referenceVerifyTop2Tokens: [[Int]]?
    let referenceVerifyTop2Logits: [[Double]]?
    // Per-module speculative-config surface (David-ruled 2026-08-19). None of
    // these three time or score anything; they let benchd measure any mode and
    // seal what the engine acknowledged.
    //   * `effectiveSpec` — the module-parsed, default-filled block the engine
    //     will actually run, echoed on the `decode_begin` response. Provenance
    //     seals ONLY this echo; a leg whose echo diverges from the request
    //     rejects (benchd side).
    //   * `specModes` — hello capability list: the modes THIS engine can run,
    //     runnable-only (a stub like dspark is never advertised).
    //   * `headProvenance` — sha256(full-width 64-hex)/bytes/file_count of the
    //     head tree the worker loaded, so head identity stays verifiable.
    let effectiveSpec: RuntimeWorkerEffectiveSpec?
    let specModes: [String]?
    let headProvenance: RuntimeWorkerHeadProvenance?
    // Protocol v1.1 (additive) — the generic benchd-facing timed-decode surface.
    // benchd drives the generic `decode_begin`/`decode_step` and
    // `free_decode_begin`/`free_decode_run` kinds against this worker and consumes
    // exactly these fields (wire names match benchd's `WorkerResponse`):
    //   * `capabilities` — hello only; `["free_run_decode"]` advertises the
    //     oracle-verified free-run timed-decode mode.
    //   * `completedWork` / `cacheMemory` — phase_diagnostics only; the phase-close
    //     barrier (completed_work == issued timed steps, or R+1 for a free-run
    //     phase) and the allocator-drain assertion (cache_memory == 0).
    //   * `acceptanceLengths` / `draftedTotal` / `acceptedTotal` / `committedTotal`
    //     — the `free_decode_run` AUDIT counters; `acceptanceLengths` is the
    //     per-round committed histogram whose sum == N and whose length is R.
    let capabilities: [String]?
    let completedWork: Int?
    let cacheMemory: Int?
    let acceptanceLengths: [Int]?
    let draftedTotal: Int?
    let acceptedTotal: Int?
    let committedTotal: Int?
    // Protocol v1.2 (additive, COHORT — batched free-run). The cohort form of
    // the free-run verbs, gated by the `batched_free_run_decode` capability
    // (wire names match benchd's `WorkerResponse` on the cohort branch):
    //   * `maxBatchSize` — hello only (gate-on); the engine's cohort-width
    //     ceiling, so benchd can refuse an over-wide cohort pre-GPU.
    //   * `seedTokenByStream` — batched free_decode_begin only; the B seed
    //     forwards' argmaxes, one per slot in SLOT ORDER, each oracle-checked.
    //   * `effectiveBatchSize` — never-ignored echo of the requested width; a
    //     divergent (or missing) echo discards the leg benchd-side.
    //   * `tokensByStream` — batched free_decode_run only; the B x N committed
    //     rectangle, every token exact-matched against its slot's golden.
    //   * `naturalAcceptedByStream` / `rounds` / `activeStreamsByRound` /
    //     `depthClampReasons` — the cohort AUDIT counters benchd's consistency
    //     QUADRUPLE cross-checks; never scored.
    let maxBatchSize: Int?
    let seedTokenByStream: [Int]?
    let effectiveBatchSize: Int?
    let tokensByStream: [[Int]]?
    let naturalAcceptedByStream: [[Int]]?
    let rounds: Int?
    let activeStreamsByRound: [Int]?
    let depthClampReasons: [String: Int]?
    // Per-stream timing instrumentation (per-stream-instrumentation-spec.md
    // step 1), gated behind the `per_stream_timing` hello capability
    // (always advertised alongside `batched_free_run_decode` for this
    // increment — see `runtimeWorkerAdvertisedCapabilities`):
    //   * `prefillNsByStream` — batched `free_decode_begin` only; per-slot
    //     monotonic ns from cohort-prefill start to that slot's seed
    //     commit, SLOT ORDER, same length as `seedTokenByStream`.
    //   * `decodeNsByStream` — batched `free_decode_run` only; per-slot
    //     monotonic ns from decode-phase start to that slot's final-token
    //     commit, SLOT ORDER, same length as `tokensByStream`.
    // Raw engine clock reads only — nothing summed, ratioed, or converted to
    // seconds. Untrusted for scoring until benchd's attestation admits them
    // (engine-reported-time-untrusted doctrine); this increment is
    // report-only on the benchd side.
    let prefillNsByStream: [UInt64]?
    let decodeNsByStream: [UInt64]?
    // PR-1 fidelity-gate (additive, trusted-CLI-only). The `cohort_reference_replay`
    // MEASUREMENT report: per stream x position the pinned reference's own
    // sequential argmax, top-K post-softcap logits/tokens, the committed token's
    // relative gap, and an informational within-envelope depth. Renders NO
    // admit/reject verdict; benchd never issues this verb, so it never appears in
    // the pinned engine-wire fixture. Consumed later by Phase-2 calibration and
    // the admission-ladder PR.
    let cohortReferenceReplay: CohortReferenceReplayReport?
    // Phase-0 hello identity (v1), emitted on EVERY hello and meaningful only
    // there: the engine's protocol version (always 1), the compute backend, and
    // the device label. Ungated — benchd requires `protocol_version` on the hello
    // and the native trusted decoder now accepts all three — so they ride even
    // when the speculative v1.1 surface is gated off at spawn.
    let protocolVersion: Int?
    let backend: String?
    let device: String?

    init(
        id: Int,
        nonce: String? = nil,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        expectedTokenLogit: Double? = nil,
        expectedTokenRank: Int? = nil,
        topLogitMargin: Double? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil,
        mlxActiveMemoryBytes: Int? = nil,
        mlxCacheMemoryBytes: Int? = nil,
        mlxPeakMemoryBytes: Int? = nil,
        targetVerificationMode: String? = nil,
        exactPairSegmentCount: Int? = nil,
        exactPairRollbackRowCount: Int? = nil,
        serialVerificationRowCount: Int? = nil,
        declaredRows: Int? = nil,
        perRowHiddenDigest: [String]? = nil,
        perRowTop2Tokens: [[Int]]? = nil,
        perRowTop2Logits: [[Double]]? = nil,
        draftTokens: [Int]? = nil,
        acceptedDraftCount: Int? = nil,
        rejectedDraftCount: Int? = nil,
        rollbackRoundCount: Int? = nil,
        targetCacheOffset: Int? = nil,
        kvDigest: String? = nil,
        kvVacancyDigest: String? = nil,
        referenceK1Argmax: [Int]? = nil,
        referenceBlockArgmax: [Int]? = nil,
        referenceTop2Tokens: [[Int]]? = nil,
        referenceTop2Logits: [[Double]]? = nil,
        referenceTop1Logits: [Double]? = nil,
        referenceEmittedTokens: [Int]? = nil,
        referenceEmittedTokenLogits: [Double]? = nil,
        referenceFrameWidths: [Int]? = nil,
        referenceFrameArgmax: [[Int]]? = nil,
        referenceVerifyTop2Tokens: [[Int]]? = nil,
        referenceVerifyTop2Logits: [[Double]]? = nil,
        effectiveSpec: RuntimeWorkerEffectiveSpec? = nil,
        specModes: [String]? = nil,
        headProvenance: RuntimeWorkerHeadProvenance? = nil,
        capabilities: [String]? = nil,
        completedWork: Int? = nil,
        cacheMemory: Int? = nil,
        acceptanceLengths: [Int]? = nil,
        draftedTotal: Int? = nil,
        acceptedTotal: Int? = nil,
        committedTotal: Int? = nil,
        maxBatchSize: Int? = nil,
        seedTokenByStream: [Int]? = nil,
        effectiveBatchSize: Int? = nil,
        tokensByStream: [[Int]]? = nil,
        naturalAcceptedByStream: [[Int]]? = nil,
        rounds: Int? = nil,
        activeStreamsByRound: [Int]? = nil,
        depthClampReasons: [String: Int]? = nil,
        prefillNsByStream: [UInt64]? = nil,
        decodeNsByStream: [UInt64]? = nil,
        cohortReferenceReplay: CohortReferenceReplayReport? = nil,
        protocolVersion: Int? = nil,
        backend: String? = nil,
        device: String? = nil
    ) {
        self.id = id
        self.nonce = nonce
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.expectedTokenLogit = expectedTokenLogit
        self.expectedTokenRank = expectedTokenRank
        self.topLogitMargin = topLogitMargin
        self.seedToken = seedToken
        self.tokens = tokens
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
        self.mlxCacheMemoryBytes = mlxCacheMemoryBytes
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
        self.targetVerificationMode = targetVerificationMode
        self.exactPairSegmentCount = exactPairSegmentCount
        self.exactPairRollbackRowCount = exactPairRollbackRowCount
        self.serialVerificationRowCount = serialVerificationRowCount
        self.declaredRows = declaredRows
        self.perRowHiddenDigest = perRowHiddenDigest
        self.perRowTop2Tokens = perRowTop2Tokens
        self.perRowTop2Logits = perRowTop2Logits
        self.draftTokens = draftTokens
        self.acceptedDraftCount = acceptedDraftCount
        self.rejectedDraftCount = rejectedDraftCount
        self.rollbackRoundCount = rollbackRoundCount
        self.targetCacheOffset = targetCacheOffset
        self.kvDigest = kvDigest
        self.kvVacancyDigest = kvVacancyDigest
        self.referenceK1Argmax = referenceK1Argmax
        self.referenceBlockArgmax = referenceBlockArgmax
        self.referenceTop2Tokens = referenceTop2Tokens
        self.referenceTop2Logits = referenceTop2Logits
        self.referenceTop1Logits = referenceTop1Logits
        self.referenceEmittedTokens = referenceEmittedTokens
        self.referenceEmittedTokenLogits = referenceEmittedTokenLogits
        self.referenceFrameWidths = referenceFrameWidths
        self.referenceFrameArgmax = referenceFrameArgmax
        self.referenceVerifyTop2Tokens = referenceVerifyTop2Tokens
        self.referenceVerifyTop2Logits = referenceVerifyTop2Logits
        self.effectiveSpec = effectiveSpec
        self.specModes = specModes
        self.headProvenance = headProvenance
        self.capabilities = capabilities
        self.completedWork = completedWork
        self.cacheMemory = cacheMemory
        self.acceptanceLengths = acceptanceLengths
        self.draftedTotal = draftedTotal
        self.acceptedTotal = acceptedTotal
        self.committedTotal = committedTotal
        self.maxBatchSize = maxBatchSize
        self.seedTokenByStream = seedTokenByStream
        self.effectiveBatchSize = effectiveBatchSize
        self.tokensByStream = tokensByStream
        self.naturalAcceptedByStream = naturalAcceptedByStream
        self.rounds = rounds
        self.activeStreamsByRound = activeStreamsByRound
        self.depthClampReasons = depthClampReasons
        self.prefillNsByStream = prefillNsByStream
        self.decodeNsByStream = decodeNsByStream
        self.cohortReferenceReplay = cohortReferenceReplay
        self.protocolVersion = protocolVersion
        self.backend = backend
        self.device = device
    }

    init(from decoder: Swift.Decoder) throws {
        let wireContainer = try decoder.container(
            keyedBy: RuntimeWorkerWireCodingKey.self
        )
        let allowedKeys = Set(CodingKeys.allCases.map(\.rawValue))
        if let unknownKey = wireContainer.allKeys.first(
            where: { !allowedKeys.contains($0.stringValue) }
        ) {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath + [unknownKey],
                    debugDescription:
                        "runtime worker response contains unknown field "
                        + unknownKey.stringValue
                )
            )
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        nonce = try container.decodeIfPresent(String.self, forKey: .nonce)
        ok = try container.decode(Bool.self, forKey: .ok)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        token = try container.decodeIfPresent(Int.self, forKey: .token)
        topLogits = try container.decodeIfPresent(
            [CorrectnessTraceLogit].self,
            forKey: .topLogits
        )
        expectedTokenLogit = try container.decodeIfPresent(
            Double.self,
            forKey: .expectedTokenLogit
        )
        expectedTokenRank = try container.decodeIfPresent(
            Int.self,
            forKey: .expectedTokenRank
        )
        topLogitMargin = try container.decodeIfPresent(
            Double.self,
            forKey: .topLogitMargin
        )
        seedToken = try container.decodeIfPresent(Int.self, forKey: .seedToken)
        tokens = try container.decodeIfPresent([Int].self, forKey: .tokens)
        expertStats = try container.decodeIfPresent(
            ExpertStreamingStats.self,
            forKey: .expertStats
        )
        peakRamGB = try container.decodeIfPresent(
            Double.self,
            forKey: .peakRamGB
        )
        mlxActiveMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxActiveMemoryBytes
        )
        mlxCacheMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxCacheMemoryBytes
        )
        mlxPeakMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxPeakMemoryBytes
        )
        targetVerificationMode = try container.decodeIfPresent(
            String.self,
            forKey: .targetVerificationMode
        )
        exactPairSegmentCount = try container.decodeIfPresent(
            Int.self,
            forKey: .exactPairSegmentCount
        )
        exactPairRollbackRowCount = try container.decodeIfPresent(
            Int.self,
            forKey: .exactPairRollbackRowCount
        )
        serialVerificationRowCount = try container.decodeIfPresent(
            Int.self,
            forKey: .serialVerificationRowCount
        )
        declaredRows = try container.decodeIfPresent(
            Int.self,
            forKey: .declaredRows
        )
        perRowHiddenDigest = try container.decodeIfPresent(
            [String].self,
            forKey: .perRowHiddenDigest
        )
        perRowTop2Tokens = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .perRowTop2Tokens
        )
        perRowTop2Logits = try container.decodeIfPresent(
            [[Double]].self,
            forKey: .perRowTop2Logits
        )
        draftTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .draftTokens
        )
        acceptedDraftCount = try container.decodeIfPresent(
            Int.self,
            forKey: .acceptedDraftCount
        )
        rejectedDraftCount = try container.decodeIfPresent(
            Int.self,
            forKey: .rejectedDraftCount
        )
        rollbackRoundCount = try container.decodeIfPresent(
            Int.self,
            forKey: .rollbackRoundCount
        )
        targetCacheOffset = try container.decodeIfPresent(
            Int.self,
            forKey: .targetCacheOffset
        )
        kvDigest = try container.decodeIfPresent(
            String.self,
            forKey: .kvDigest
        )
        kvVacancyDigest = try container.decodeIfPresent(
            String.self,
            forKey: .kvVacancyDigest
        )
        referenceK1Argmax = try container.decodeIfPresent(
            [Int].self,
            forKey: .referenceK1Argmax
        )
        referenceBlockArgmax = try container.decodeIfPresent(
            [Int].self,
            forKey: .referenceBlockArgmax
        )
        referenceTop2Tokens = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .referenceTop2Tokens
        )
        referenceTop2Logits = try container.decodeIfPresent(
            [[Double]].self,
            forKey: .referenceTop2Logits
        )
        referenceTop1Logits = try container.decodeIfPresent(
            [Double].self,
            forKey: .referenceTop1Logits
        )
        referenceEmittedTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .referenceEmittedTokens
        )
        referenceEmittedTokenLogits = try container.decodeIfPresent(
            [Double].self,
            forKey: .referenceEmittedTokenLogits
        )
        referenceFrameWidths = try container.decodeIfPresent(
            [Int].self,
            forKey: .referenceFrameWidths
        )
        referenceFrameArgmax = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .referenceFrameArgmax
        )
        referenceVerifyTop2Tokens = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .referenceVerifyTop2Tokens
        )
        referenceVerifyTop2Logits = try container.decodeIfPresent(
            [[Double]].self,
            forKey: .referenceVerifyTop2Logits
        )
        effectiveSpec = try container.decodeIfPresent(
            RuntimeWorkerEffectiveSpec.self,
            forKey: .effectiveSpec
        )
        specModes = try container.decodeIfPresent(
            [String].self,
            forKey: .specModes
        )
        headProvenance = try container.decodeIfPresent(
            RuntimeWorkerHeadProvenance.self,
            forKey: .headProvenance
        )
        capabilities = try container.decodeIfPresent(
            [String].self,
            forKey: .capabilities
        )
        completedWork = try container.decodeIfPresent(
            Int.self,
            forKey: .completedWork
        )
        cacheMemory = try container.decodeIfPresent(
            Int.self,
            forKey: .cacheMemory
        )
        acceptanceLengths = try container.decodeIfPresent(
            [Int].self,
            forKey: .acceptanceLengths
        )
        draftedTotal = try container.decodeIfPresent(
            Int.self,
            forKey: .draftedTotal
        )
        acceptedTotal = try container.decodeIfPresent(
            Int.self,
            forKey: .acceptedTotal
        )
        committedTotal = try container.decodeIfPresent(
            Int.self,
            forKey: .committedTotal
        )
        maxBatchSize = try container.decodeIfPresent(
            Int.self,
            forKey: .maxBatchSize
        )
        seedTokenByStream = try container.decodeIfPresent(
            [Int].self,
            forKey: .seedTokenByStream
        )
        effectiveBatchSize = try container.decodeIfPresent(
            Int.self,
            forKey: .effectiveBatchSize
        )
        tokensByStream = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .tokensByStream
        )
        naturalAcceptedByStream = try container.decodeIfPresent(
            [[Int]].self,
            forKey: .naturalAcceptedByStream
        )
        rounds = try container.decodeIfPresent(Int.self, forKey: .rounds)
        activeStreamsByRound = try container.decodeIfPresent(
            [Int].self,
            forKey: .activeStreamsByRound
        )
        depthClampReasons = try container.decodeIfPresent(
            [String: Int].self,
            forKey: .depthClampReasons
        )
        prefillNsByStream = try container.decodeIfPresent(
            [UInt64].self,
            forKey: .prefillNsByStream
        )
        decodeNsByStream = try container.decodeIfPresent(
            [UInt64].self,
            forKey: .decodeNsByStream
        )
        cohortReferenceReplay = try container.decodeIfPresent(
            CohortReferenceReplayReport.self,
            forKey: .cohortReferenceReplay
        )
        protocolVersion = try container.decodeIfPresent(
            Int.self,
            forKey: .protocolVersion
        )
        backend = try container.decodeIfPresent(String.self, forKey: .backend)
        device = try container.decodeIfPresent(String.self, forKey: .device)
    }

    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(nonce, forKey: .nonce)
        try container.encode(ok, forKey: .ok)
        try container.encodeIfPresent(error, forKey: .error)
        try container.encodeIfPresent(token, forKey: .token)
        try container.encodeIfPresent(topLogits, forKey: .topLogits)
        try container.encodeIfPresent(
            expectedTokenLogit,
            forKey: .expectedTokenLogit
        )
        try container.encodeIfPresent(
            expectedTokenRank,
            forKey: .expectedTokenRank
        )
        try container.encodeIfPresent(
            topLogitMargin,
            forKey: .topLogitMargin
        )
        try container.encodeIfPresent(seedToken, forKey: .seedToken)
        try container.encodeIfPresent(tokens, forKey: .tokens)
        try container.encodeIfPresent(expertStats, forKey: .expertStats)
        try container.encodeIfPresent(peakRamGB, forKey: .peakRamGB)
        try container.encodeIfPresent(
            mlxActiveMemoryBytes,
            forKey: .mlxActiveMemoryBytes
        )
        try container.encodeIfPresent(
            mlxCacheMemoryBytes,
            forKey: .mlxCacheMemoryBytes
        )
        try container.encodeIfPresent(
            mlxPeakMemoryBytes,
            forKey: .mlxPeakMemoryBytes
        )
        try container.encodeIfPresent(
            targetVerificationMode,
            forKey: .targetVerificationMode
        )
        try container.encodeIfPresent(
            exactPairSegmentCount,
            forKey: .exactPairSegmentCount
        )
        try container.encodeIfPresent(
            exactPairRollbackRowCount,
            forKey: .exactPairRollbackRowCount
        )
        try container.encodeIfPresent(
            serialVerificationRowCount,
            forKey: .serialVerificationRowCount
        )
        try container.encodeIfPresent(declaredRows, forKey: .declaredRows)
        try container.encodeIfPresent(
            perRowHiddenDigest,
            forKey: .perRowHiddenDigest
        )
        try container.encodeIfPresent(
            perRowTop2Tokens,
            forKey: .perRowTop2Tokens
        )
        try container.encodeIfPresent(
            perRowTop2Logits,
            forKey: .perRowTop2Logits
        )
        try container.encodeIfPresent(draftTokens, forKey: .draftTokens)
        try container.encodeIfPresent(
            acceptedDraftCount,
            forKey: .acceptedDraftCount
        )
        try container.encodeIfPresent(
            rejectedDraftCount,
            forKey: .rejectedDraftCount
        )
        try container.encodeIfPresent(
            rollbackRoundCount,
            forKey: .rollbackRoundCount
        )
        try container.encodeIfPresent(
            targetCacheOffset,
            forKey: .targetCacheOffset
        )
        try container.encodeIfPresent(kvDigest, forKey: .kvDigest)
        try container.encodeIfPresent(
            kvVacancyDigest,
            forKey: .kvVacancyDigest
        )
        try container.encodeIfPresent(
            referenceK1Argmax,
            forKey: .referenceK1Argmax
        )
        try container.encodeIfPresent(
            referenceBlockArgmax,
            forKey: .referenceBlockArgmax
        )
        try container.encodeIfPresent(
            referenceTop2Tokens,
            forKey: .referenceTop2Tokens
        )
        try container.encodeIfPresent(
            referenceTop2Logits,
            forKey: .referenceTop2Logits
        )
        try container.encodeIfPresent(
            referenceTop1Logits,
            forKey: .referenceTop1Logits
        )
        try container.encodeIfPresent(
            referenceEmittedTokens,
            forKey: .referenceEmittedTokens
        )
        try container.encodeIfPresent(
            referenceEmittedTokenLogits,
            forKey: .referenceEmittedTokenLogits
        )
        try container.encodeIfPresent(
            referenceFrameWidths,
            forKey: .referenceFrameWidths
        )
        try container.encodeIfPresent(
            referenceFrameArgmax,
            forKey: .referenceFrameArgmax
        )
        try container.encodeIfPresent(
            referenceVerifyTop2Tokens,
            forKey: .referenceVerifyTop2Tokens
        )
        try container.encodeIfPresent(
            referenceVerifyTop2Logits,
            forKey: .referenceVerifyTop2Logits
        )
        try container.encodeIfPresent(effectiveSpec, forKey: .effectiveSpec)
        try container.encodeIfPresent(specModes, forKey: .specModes)
        try container.encodeIfPresent(headProvenance, forKey: .headProvenance)
        try container.encodeIfPresent(capabilities, forKey: .capabilities)
        try container.encodeIfPresent(completedWork, forKey: .completedWork)
        try container.encodeIfPresent(cacheMemory, forKey: .cacheMemory)
        try container.encodeIfPresent(
            acceptanceLengths,
            forKey: .acceptanceLengths
        )
        try container.encodeIfPresent(draftedTotal, forKey: .draftedTotal)
        try container.encodeIfPresent(acceptedTotal, forKey: .acceptedTotal)
        try container.encodeIfPresent(committedTotal, forKey: .committedTotal)
        try container.encodeIfPresent(maxBatchSize, forKey: .maxBatchSize)
        try container.encodeIfPresent(
            seedTokenByStream,
            forKey: .seedTokenByStream
        )
        try container.encodeIfPresent(
            effectiveBatchSize,
            forKey: .effectiveBatchSize
        )
        try container.encodeIfPresent(tokensByStream, forKey: .tokensByStream)
        try container.encodeIfPresent(
            naturalAcceptedByStream,
            forKey: .naturalAcceptedByStream
        )
        try container.encodeIfPresent(rounds, forKey: .rounds)
        try container.encodeIfPresent(
            activeStreamsByRound,
            forKey: .activeStreamsByRound
        )
        try container.encodeIfPresent(
            depthClampReasons,
            forKey: .depthClampReasons
        )
        try container.encodeIfPresent(
            prefillNsByStream,
            forKey: .prefillNsByStream
        )
        try container.encodeIfPresent(
            decodeNsByStream,
            forKey: .decodeNsByStream
        )
        try container.encodeIfPresent(
            cohortReferenceReplay,
            forKey: .cohortReferenceReplay
        )
        try container.encodeIfPresent(protocolVersion, forKey: .protocolVersion)
        try container.encodeIfPresent(backend, forKey: .backend)
        try container.encodeIfPresent(device, forKey: .device)
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case expectedTokenLogit = "expected_token_logit"
        case expectedTokenRank = "expected_token_rank"
        case topLogitMargin = "top_logit_margin"
        case seedToken = "seed_token"
        case tokens
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
        case mlxActiveMemoryBytes = "mlx_active_memory_bytes"
        case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case targetVerificationMode = "target_verification_mode"
        case exactPairSegmentCount = "exact_pair_segment_count"
        case exactPairRollbackRowCount = "exact_pair_rollback_row_count"
        case serialVerificationRowCount = "serial_verification_row_count"
        case declaredRows = "declared_rows"
        case perRowHiddenDigest = "per_row_hidden_digest"
        case perRowTop2Tokens = "per_row_top2_tokens"
        case perRowTop2Logits = "per_row_top2_logits"
        case draftTokens = "draft_tokens"
        case acceptedDraftCount = "accepted_draft_count"
        case rejectedDraftCount = "rejected_draft_count"
        case rollbackRoundCount = "rollback_round_count"
        case targetCacheOffset = "target_cache_offset"
        case kvDigest = "kv_digest"
        case kvVacancyDigest = "kv_vacancy_digest"
        case referenceK1Argmax = "reference_k1_argmax"
        case referenceBlockArgmax = "reference_block_argmax"
        case referenceTop2Tokens = "reference_top2_tokens"
        case referenceTop2Logits = "reference_top2_logits"
        case referenceTop1Logits = "reference_top1_logits"
        case referenceEmittedTokens = "reference_emitted_tokens"
        case referenceEmittedTokenLogits = "reference_emitted_token_logits"
        case referenceFrameWidths = "reference_frame_widths"
        case referenceFrameArgmax = "reference_frame_argmax"
        case referenceVerifyTop2Tokens = "reference_verify_top2_tokens"
        case referenceVerifyTop2Logits = "reference_verify_top2_logits"
        case effectiveSpec = "effective_spec"
        case specModes = "spec_modes"
        case headProvenance = "head_provenance"
        case capabilities
        case completedWork = "completed_work"
        case cacheMemory = "cache_memory"
        case acceptanceLengths = "acceptance_lengths"
        case draftedTotal = "drafted_total"
        case acceptedTotal = "accepted_total"
        case committedTotal = "committed_total"
        case maxBatchSize = "max_batch_size"
        case seedTokenByStream = "seed_token_by_stream"
        case effectiveBatchSize = "effective_batch_size"
        case tokensByStream = "tokens_by_stream"
        case naturalAcceptedByStream = "natural_accepted_by_stream"
        case rounds
        case activeStreamsByRound = "active_streams_by_round"
        case depthClampReasons = "depth_clamp_reasons"
        case prefillNsByStream = "prefill_ns_by_stream"
        case decodeNsByStream = "decode_ns_by_stream"
        case cohortReferenceReplay = "cohort_reference_replay"
        case protocolVersion = "protocol_version"
        case backend
        case device
    }
}

final class BufferedFileLineReader {
    static let defaultMaximumLineByteCount = 4 * 1024 * 1024

    private let handle: FileHandle
    private let maximumLineByteCount: Int
    private var buffer = Data()

    init(
        handle: FileHandle,
        maximumLineByteCount: Int = BufferedFileLineReader.defaultMaximumLineByteCount
    ) {
        self.handle = handle
        self.maximumLineByteCount = maximumLineByteCount
    }

    func readLine() throws -> Data? {
        guard maximumLineByteCount > 0 else {
            throw MLXFastError.invalidInput("runtime worker protocol line limit must be positive")
        }
        while true {
            if let newlineIndex = buffer.firstIndex(of: 0x0a) {
                let lineByteCount = buffer.distance(from: buffer.startIndex, to: newlineIndex)
                guard lineByteCount <= maximumLineByteCount else {
                    throw MLXFastError.invalidInput(
                        "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                    )
                }
                let line = Data(buffer.prefix(lineByteCount))
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                return line
            }
            guard buffer.count <= maximumLineByteCount else {
                throw MLXFastError.invalidInput(
                    "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                )
            }
            let chunk = handle.availableData
            if chunk.isEmpty {
                guard !buffer.isEmpty else {
                    return nil
                }
                guard buffer.count <= maximumLineByteCount else {
                    throw MLXFastError.invalidInput(
                        "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                    )
                }
                defer { buffer.removeAll(keepingCapacity: true) }
                return buffer
            }
            buffer.append(chunk)
        }
    }
}

final class RuntimeWorkerProtocolIO {
    private let input: BufferedFileLineReader
    private let output: FileHandle

    private init(inputDescriptor: Int32, outputDescriptor: Int32) {
        self.input = BufferedFileLineReader(
            handle: FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        )
        self.output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
    }

    static func isolatingStandardIO() throws -> RuntimeWorkerProtocolIO {
        let descriptors = try duplicateRuntimeWorkerProtocolDescriptors(
            inputDescriptor: STDIN_FILENO,
            outputDescriptor: STDOUT_FILENO
        )
        let inputFD = descriptors.input
        let outputFD = descriptors.output
        do {
            try redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY, label: "stdin")
            try redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY, label: "stdout")
        } catch {
            close(inputFD)
            close(outputFD)
            throw error
        }
        return RuntimeWorkerProtocolIO(inputDescriptor: inputFD, outputDescriptor: outputFD)
    }

    func readLine() throws -> String? {
        guard let data = try input.readLine() else {
            return nil
        }
        guard let line = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput("runtime worker received non-UTF8 protocol input")
        }
        return line
    }

    func writeLine(_ data: Data) throws {
        guard data.count <= BufferedFileLineReader.defaultMaximumLineByteCount else {
            throw MLXFastError.invalidInput(
                "runtime worker protocol response exceeds "
                    + "\(BufferedFileLineReader.defaultMaximumLineByteCount) bytes"
            )
        }
        try output.write(contentsOf: data)
        try output.write(contentsOf: Data([0x0a]))
    }
}

func duplicateRuntimeWorkerProtocolDescriptors(
    inputDescriptor: Int32,
    outputDescriptor: Int32,
    duplicate: (_ descriptor: Int32, _ label: String) throws -> Int32 = duplicatePrivateDescriptor,
    closeDescriptor: (_ descriptor: Int32) -> Void = { _ = Darwin.close($0) }
) throws -> (input: Int32, output: Int32) {
    let input = try duplicate(inputDescriptor, "stdin")
    do {
        let output = try duplicate(outputDescriptor, "stdout")
        return (input: input, output: output)
    } catch {
        closeDescriptor(input)
        throw error
    }
}

func duplicatePrivateDescriptor(_ descriptor: Int32, label: String) throws -> Int32 {
    // F_DUPFD requires its lower bound to be below RLIMIT_NOFILE. Standard
    // macOS launchd jobs may inherit a soft limit of 256, so a randomized
    // 64...512 bound makes worker startup fail nondeterministically.
    let lowerBound = STDERR_FILENO + 1
    let duplicatedFD = fcntl(descriptor, F_DUPFD_CLOEXEC, lowerBound)
    guard duplicatedFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to duplicate \(label) for protocol I/O")
    }
    return duplicatedFD
}

func redirectDescriptorToDevNull(_ descriptor: Int32, flags: Int32, label: String) throws {
    let devNullFD = open("/dev/null", flags)
    guard devNullFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to open /dev/null for \(label) redirection")
    }
    defer {
        close(devNullFD)
    }
    guard dup2(devNullFD, descriptor) >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to redirect \(label) away from protocol I/O")
    }
}

/// Continuously drains a runtime worker's stderr pipe on a background thread,
/// forwarding each completed line to `emit` (prefixed and token-redacted) and
/// keeping a capped raw tail for the exit diagnostic. Local modes attach this
/// so participants' debug prints in model code show up live during the edit
/// loop; it also means a chatty worker can no longer fill the undrained pipe
/// buffer and stall the run. Official runs attach the same drain with a no-op
/// emitter, so submitted output is consumed but never forwarded to CI logs.
final class WorkerStderrDrain: @unchecked Sendable {
    private let handle: FileHandle
    private let emit: (String) -> Void
    private let lock = NSLock()
    private var tail = Data()
    private var pendingLine = Data()
    private var pendingLineWasTruncated = false
    private let finished = DispatchSemaphore(value: 0)
    static let tailByteLimit = 64 * 1024
    static let forwardedLinePrefix = "mlxfast-worker: "
    static let truncatedLine = "[worker stderr line exceeded 65536 bytes]"

    init(
        handle: FileHandle,
        emit: ((String) -> Void)? = nil
    ) {
        self.handle = handle
        self.emit = emit ?? { line in
            fputs(line, stderr)
            fflush(stderr)
        }
        let thread = Thread { [self] in
            drainToEOF()
            finished.signal()
        }
        thread.name = "mlxfast.worker-stderr-drain"
        thread.start()
    }

    /// Blocks until the reader thread hits EOF (the worker exited and all
    /// output was ingested) or the timeout passes, then returns the raw tail
    /// for the exit diagnostic (which applies its own sanitization).
    func drainedOutput(timeoutSeconds: Double) -> String {
        if finished.wait(timeout: .now() + timeoutSeconds) == .success {
            // Re-signal so later calls (or repeated diagnostics) do not block.
            finished.signal()
        }
        lock.lock()
        defer {
            lock.unlock()
        }
        var data = tail
        if pendingLineWasTruncated {
            data.append(Data(Self.truncatedLine.utf8))
        } else {
            data.append(pendingLine)
        }
        if data.count > Self.tailByteLimit {
            data = Data(data.suffix(Self.tailByteLimit))
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func drainToEOF() {
        while true {
            let chunk = handle.readData(ofLength: 8192)
            if chunk.isEmpty {
                break
            }
            ingest(chunk)
        }
        flushPendingLine()
    }

    private func ingest(_ chunk: Data) {
        var completedLines: [String] = []
        lock.lock()
        pendingLine.append(chunk)
        while let newlineIndex = pendingLine.firstIndex(of: 0x0a) {
            let lineLength = pendingLine.distance(from: pendingLine.startIndex, to: newlineIndex)
            let lineWasTruncated = pendingLineWasTruncated || lineLength > Self.tailByteLimit
            let lineData = lineWasTruncated
                ? Data(Self.truncatedLine.utf8)
                : Data(pendingLine.prefix(lineLength))
            pendingLine = Data(pendingLine.dropFirst(lineLength + 1))
            pendingLineWasTruncated = false
            appendToTailLocked(lineData + Data([0x0a]))
            completedLines.append(String(decoding: lineData, as: UTF8.self))
        }
        if pendingLine.count > Self.tailByteLimit {
            pendingLine.removeAll(keepingCapacity: true)
            pendingLineWasTruncated = true
        }
        lock.unlock()
        for line in completedLines {
            emitLine(line)
        }
    }

    private func flushPendingLine() {
        lock.lock()
        let remainder = pendingLineWasTruncated
            ? Data(Self.truncatedLine.utf8)
            : pendingLine
        pendingLine = Data()
        pendingLineWasTruncated = false
        if !remainder.isEmpty {
            appendToTailLocked(remainder + Data([0x0a]))
        }
        lock.unlock()
        if !remainder.isEmpty {
            emitLine(String(decoding: remainder, as: UTF8.self))
        }
    }

    private func appendToTailLocked(_ data: Data) {
        tail.append(data)
        if tail.count > Self.tailByteLimit {
            tail = Data(tail.suffix(Self.tailByteLimit))
        }
    }

    private func emitLine(_ line: String) {
        emit("\(Self.forwardedLinePrefix)\(redactedWorkerStderrLine(line))\n")
    }
}

/// Per-line redaction for forwarded worker stderr: worker output comes from
/// submitted model code that has seen the (possibly private) golden, so lines
/// that look like token comparisons are collapsed exactly like the shared
/// error-path redaction.
func redactedWorkerStderrLine(_ line: String) -> String {
    if line.range(of: "expected", options: .caseInsensitive) != nil
        || line.range(of: "actual", options: .caseInsensitive) != nil
    {
        return "token-validation-failed"
    }
    return line
}

final class RuntimeWorkerWatchdog: @unchecked Sendable {
    private let process: Process
    private let timer: DispatchSourceTimer
    private let lock = NSLock()
    private var active = true
    private var fired = false

    init(process: Process, timeoutSeconds: Double, terminationGraceSeconds: Double) {
        self.process = process
        self.timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        timer.schedule(deadline: .now() + timeoutSeconds)
        timer.setEventHandler { [weak self] in
            self?.fire(terminationGraceSeconds: terminationGraceSeconds)
        }
        timer.resume()
    }

    @discardableResult
    func cancelAndReturnDidFire() -> Bool {
        lock.lock()
        active = false
        let result = fired
        lock.unlock()
        timer.cancel()
        return result
    }

    private func fire(terminationGraceSeconds: Double) {
        lock.lock()
        guard active else {
            lock.unlock()
            return
        }
        active = false
        fired = true
        lock.unlock()

        if process.isRunning {
            process.terminate()
        }
        DispatchQueue.global(qos: .userInitiated).asyncAfter(
            deadline: .now() + max(terminationGraceSeconds, 0)
        ) { [process] in
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }
    }
}

@discardableResult
func stopRuntimeWorkerProcess(
    _ process: Process,
    timeoutSeconds: Double,
    pollIntervalMicroseconds: useconds_t = 10_000
) -> Bool {
    guard process.isRunning else {
        return true
    }
    process.terminate()
    let boundedTimeout = timeoutSeconds.isFinite
        ? min(max(timeoutSeconds, 0), 24 * 60 * 60)
        : 0
    let now = DispatchTime.now().uptimeNanoseconds
    let timeoutNanoseconds = UInt64(boundedTimeout * 1_000_000_000)
    let (deadline, overflow) = now.addingReportingOverflow(timeoutNanoseconds)
    let resolvedDeadline = overflow ? UInt64.max : deadline
    while process.isRunning, DispatchTime.now().uptimeNanoseconds < resolvedDeadline {
        usleep(pollIntervalMicroseconds)
    }
    if process.isRunning {
        kill(process.processIdentifier, SIGKILL)
    }
    let killDeadline = DispatchTime.now().uptimeNanoseconds + 2_000_000_000
    while process.isRunning, DispatchTime.now().uptimeNanoseconds < killDeadline {
        usleep(pollIntervalMicroseconds)
    }
    return !process.isRunning
}

extension Gemma4Runtime {
    public static func runPreflightWithWorker(
        weightsPath: String,
        worker options: RuntimeWorkerOptions
    ) throws {
        guard options.requestTimeoutSeconds.isFinite,
              options.requestTimeoutSeconds > 0,
              options.shutdownTimeoutSeconds.isFinite,
              options.shutdownTimeoutSeconds >= 0,
              options.terminationGraceSeconds.isFinite,
              options.terminationGraceSeconds >= 0
        else {
            throw MLXFastError.invalidInput(
                "runtime worker preflight timeouts must be valid"
            )
        }

        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        let workerArguments = [
            "preflight",
            "--weights",
            weightsPath,
        ]
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
            ] + workerArguments
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = workerArguments
        }
        process.environment = sanitizedRuntimeWorkerEnvironment(
            ProcessInfo.processInfo.environment
        )
        process.standardInput = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        let stderrDrain = WorkerStderrDrain(
            handle: stderr.fileHandleForReading,
            emit: options.forwardsWorkerStderr ? nil : { _ in }
        )
        let watchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: options.requestTimeoutSeconds,
            terminationGraceSeconds: options.terminationGraceSeconds
        )
        let output = BufferedFileLineReader(
            handle: stdout.fileHandleForReading
        )
        do {
            guard let data = try output.readLine() else {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight closed stdout without a response"
                )
            }
            let response = try JSONDecoder().decode(
                RuntimeWorkerPreflightResponse.self,
                from: data
            )
            process.waitUntilExit()
            if watchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight timed out"
                )
            }
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds
                    + options.terminationGraceSeconds + 1
            )
            guard response.ok, process.terminationStatus == 0 else {
                throw MLXFastError.invalidInput(
                    response.error
                        ?? "runtime worker preflight exited with status "
                        + "\(process.terminationStatus)"
                )
            }
        } catch {
            let timedOut = watchdog.cancelAndReturnDidFire()
            if process.isRunning {
                _ = stopRuntimeWorkerProcess(
                    process,
                    timeoutSeconds: options.shutdownTimeoutSeconds
                )
            }
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds
                    + options.terminationGraceSeconds + 1
            )
            if timedOut {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight timed out"
                )
            }
            throw error
        }
    }
}

final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: BufferedFileLineReader
    private let stderrDrain: WorkerStderrDrain
    private let requestTimeoutSeconds: Double
    private let shutdownTimeoutSeconds: Double
    private let terminationGraceSeconds: Double
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var sessionNonce = ""
    private var nextID = 1
    private var closed = false

    init(
        options: RuntimeWorkerOptions,
        weightsPath: String,
        dflashDrafterPath: String? = nil
    ) throws {
        guard options.helloTimeoutSeconds.isFinite,
              options.helloTimeoutSeconds > 0,
              options.requestTimeoutSeconds.isFinite,
              options.requestTimeoutSeconds > 0,
              options.shutdownTimeoutSeconds.isFinite,
              options.shutdownTimeoutSeconds >= 0,
              options.terminationGraceSeconds.isFinite,
              options.terminationGraceSeconds >= 0
        else {
            throw MLXFastError.invalidInput("runtime worker timeouts must be positive")
        }
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        // A drafter path switches the worker to the DFlash block-decode
        // subcommand. The DFlash worker serves only dflash_* kinds, so a serial
        // request cannot be smuggled into a block-decode session or vice versa.
        let workerArguments: [String]
        if let dflashDrafterPath {
            workerArguments = [
                "dflash-runtime-worker",
                "--weights",
                weightsPath,
                "--drafter",
                dflashDrafterPath,
            ]
        } else {
            workerArguments = [
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        }
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
            ] + workerArguments
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = workerArguments
        }
        process.environment = sanitizedRuntimeWorkerEnvironment(ProcessInfo.processInfo.environment)
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        self.process = process
        self.input = stdin.fileHandleForWriting
        self.output = BufferedFileLineReader(handle: stdout.fileHandleForReading)
        self.requestTimeoutSeconds = options.requestTimeoutSeconds
        self.shutdownTimeoutSeconds = options.shutdownTimeoutSeconds
        self.terminationGraceSeconds = options.terminationGraceSeconds
        // Always consume the pipe. Official runs use a no-op emitter so worker
        // output cannot reach logs, while local modes retain live forwarding.
        self.stderrDrain = WorkerStderrDrain(
            handle: stderr.fileHandleForReading,
            emit: options.forwardsWorkerStderr ? nil : { _ in }
        )
        let helloWatchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: options.helloTimeoutSeconds,
            terminationGraceSeconds: options.terminationGraceSeconds
        )
        let hello: RuntimeWorkerResponse
        do {
            hello = try readResponseLine(validateNonce: false)
            if helloWatchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput("runtime worker timed out waiting for protocol hello")
            }
            guard hello.id == 0, hello.ok, let nonce = hello.nonce, !nonce.isEmpty else {
                throw MLXFastError.invalidInput("runtime worker did not return a valid protocol hello")
            }
            self.sessionNonce = nonce
        } catch {
            let helloTimedOut = helloWatchdog.cancelAndReturnDidFire()
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: options.shutdownTimeoutSeconds)
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds + options.terminationGraceSeconds + 1
            )
            if helloTimedOut {
                throw MLXFastError.invalidInput("runtime worker timed out waiting for protocol hello")
            }
            throw error
        }
    }

    deinit {
        close()
    }

    func close() {
        guard !closed else {
            return
        }
        closed = true
        try? input.close()
        if process.isRunning {
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: shutdownTimeoutSeconds)
        }
        _ = stderrDrain.drainedOutput(timeoutSeconds: shutdownTimeoutSeconds + terminationGraceSeconds + 1)
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness",
            promptTokens: promptTokens,
            steps: steps
        )
    }

    func beginTeacherForcedCorrectness(
        promptTokens: [Int],
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_begin",
            promptTokens: promptTokens,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    func teacherForcedCorrectnessStep(
        previousToken: Int,
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_step",
            token: previousToken,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "prefill",
            promptTokens: promptTokens
        )
    }

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_begin",
            seedTokens: seedTokens
        )
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_step",
            token: inputToken
        )
    }

    func phaseDiagnostics() throws -> RuntimeWorkerResponse {
        try send(kind: "phase_diagnostics")
    }

    // --- DFlash block decode (laguna-xs-2.1-dflash-v1) --------------------
    // The parent chooses every block width and never tells the worker how much
    // of the decode window remains.

    /// Untimed phase start: allocator clear plus working-set re-touch. Issued
    /// before the parent's clock starts, because neither step sees the seed.
    func warmDFlashDecode() throws -> RuntimeWorkerResponse {
        try send(kind: "dflash_decode_warm")
    }

    func beginDFlashDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "dflash_decode_begin",
            seedTokens: seedTokens
        )
    }

    func dflashDecodeBlock(
        previousCommittedToken: Int,
        maxBlockSize: Int
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "dflash_decode_block",
            token: previousCommittedToken,
            maxBlockSize: maxBlockSize
        )
    }

    func dflashPhaseDiagnostics() throws -> RuntimeWorkerResponse {
        try send(kind: "dflash_phase_diagnostics")
    }

    /// Reference-side seed prefill (contract layer L1).
    ///
    /// Establishes the run's seed token in the candidate's own frame -- one bulk
    /// forward over the whole seed -- and leaves the reference's continuous
    /// width-1 frame positioned at the end of the seed, so the first row request
    /// is a plain continuation rather than a rebuild.
    func dflashReferencePrefill(
        seedTokens: [Int]
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "dflash_reference_prefill",
            seedTokens: seedTokens
        )
    }

    /// Reference-side row request (contract layer L1). Only ever sent to a
    /// worker spawned from the PINNED BASELINE tree over organizer weights, and
    /// only after the timed window: the candidate is torn down first, so this
    /// kind never reaches submitted code.
    ///
    /// `rowCount` is how many positions the width-1 frame walks; `widestFrame`
    /// is the widest block frame to replay for those same positions. The
    /// reference answers every width in `rowCount ... widestFrame` from one
    /// request because each is a branch off the same continuous cache -- asking
    /// for them separately would rewind the walk and force a re-prefill.
    ///
    /// `verifyBlockTokens`, when supplied, is the candidate's own verify input
    /// for this round -- `[bonus] + journalled drafts` -- and the reference
    /// replays it as a further branch off the same cache so the parent can price
    /// the rejected tail. It is built by the parent from its committed chain plus
    /// the round journal, so a worker cannot choose the bonus row.
    func dflashReferenceRows(
        prefixTokens: [Int],
        seedTokenCount: Int,
        startOffset: Int,
        rowCount: Int,
        widestFrame: Int,
        verifyBlockTokens: [Int]? = nil
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "dflash_reference_rows",
            prefixTokens: prefixTokens,
            startOffset: startOffset,
            rowCount: rowCount,
            declaredBlockWidth: widestFrame,
            seedTokenCount: seedTokenCount,
            verifyBlockTokens: verifyBlockTokens
        )
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil,
        topK: Int? = nil,
        expectedToken: Int? = nil,
        maxBlockSize: Int? = nil,
        prefixTokens: [Int]? = nil,
        startOffset: Int? = nil,
        rowCount: Int? = nil,
        declaredBlockWidth: Int? = nil,
        seedTokenCount: Int? = nil,
        verifyBlockTokens: [Int]? = nil
    ) throws -> RuntimeWorkerResponse {
        guard process.isRunning else {
            throw MLXFastError.invalidInput("runtime worker exited before request \(kind): \(workerExitDiagnostic())")
        }
        let id = nextID
        nextID += 1
        let request = RuntimeWorkerRequest(
            id: id,
            kind: kind,
            promptTokens: promptTokens,
            token: token,
            seedTokens: seedTokens,
            steps: steps,
            topK: topK,
            expectedToken: expectedToken,
            maxBlockSize: maxBlockSize,
            prefixTokens: prefixTokens,
            startOffset: startOffset,
            rowCount: rowCount,
            declaredBlockWidth: declaredBlockWidth,
            seedTokenCount: seedTokenCount,
            verifyBlockTokens: verifyBlockTokens
        )
        var data = try encoder.encode(request)
        guard data.count <= BufferedFileLineReader.defaultMaximumLineByteCount else {
            throw MLXFastError.invalidInput(
                "runtime worker protocol request exceeds "
                    + "\(BufferedFileLineReader.defaultMaximumLineByteCount) bytes"
            )
        }
        data.append(0x0a)
        let watchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: requestTimeoutSeconds,
            terminationGraceSeconds: terminationGraceSeconds
        )
        let response: RuntimeWorkerResponse
        do {
            try input.write(contentsOf: data)
            response = try readResponseLine(validateNonce: true)
        } catch {
            if watchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput("runtime worker timed out handling request \(kind)")
            }
            throw error
        }
        guard !watchdog.cancelAndReturnDidFire() else {
            throw MLXFastError.invalidInput("runtime worker timed out handling request \(kind)")
        }
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine(validateNonce: Bool) throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            let response = try decoder.decode(RuntimeWorkerResponse.self, from: data)
            if validateNonce, response.nonce != sessionNonce {
                throw MLXFastError.invalidInput("runtime worker returned a response with an invalid nonce")
            }
            return response
        }
    }

    private func readWorkerOutputLine() throws -> Data {
        guard let data = try output.readLine() else {
            throw MLXFastError.invalidInput(
                "runtime worker closed stdout before returning a response: \(workerExitDiagnostic())"
            )
        }
        return data
    }

    private func workerExitDiagnostic() -> String {
        if process.isRunning {
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: shutdownTimeoutSeconds)
        }
        let stderr = stderrDrain.drainedOutput(
            timeoutSeconds: shutdownTimeoutSeconds + terminationGraceSeconds + 1
        )
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = sanitizeWorkerDiagnostic(trimmed)
        let status = process.isRunning ? "timeout" : String(process.terminationStatus)
        if redacted.isEmpty {
            return "exit_status=\(status)"
        }
        return "exit_status=\(status) stderr=\(redacted)"
    }

    private func sanitizeWorkerDiagnostic(_ value: String) -> String {
        let singleLine = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        if singleLine.range(of: "expected", options: .caseInsensitive) != nil
            || singleLine.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return singleLine
    }
}

/// Runtime-worker child environment policy: STRICT ALLOWLIST, not a denylist.
///
/// The runtime worker is the only process that executes submitted model code,
/// and submitted code can read its whole environment via
/// `ProcessInfo.processInfo.environment`. The ranked pipeline runs that same
/// code in two separate passes with different harness environments -- the
/// unscored correctness/gates pass and the scored timed pass -- so ANY
/// inherited variable whose value differs between the passes is a phase
/// oracle: a submission could serve correct-but-slow behavior while its
/// tokens are checked and a cheaper path while its speed is measured,
/// inflating the paired score without a real optimization.
///
/// A remove-by-default denylist structurally cannot close that class: every
/// new harness/CI/workflow variable reopens it by default (MLXFAST_NOTE,
/// MLXFAST_SCORE_PATH, MLXFAST_INTEGRITY_PATH, the semantic-GPQA knobs,
/// BENCH_GOLDEN_PATH, and GIT_CONFIG_* all leaked through the previous
/// denylist, and the first three differ between the gates and timed passes).
/// So this filter starts from an EMPTY environment and copies in only the
/// names below, which makes the child environment byte-identical across
/// phases by construction -- the phase-isolation property is tested directly
/// by `runtimeWorkerEnvironmentIsIdenticalAcrossPipelinePhases`.
///
/// Keep-set rationale (everything else is dropped):
/// - Exact POSIX/login/session basics (`PATH`, `HOME`, `TMPDIR`, ...): the
///   dynamic loader, Foundation, and Metal's shader-cache paths rely on
///   them; their values are fixed per host/user, never per phase.
/// - `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`: constant "1" wherever trusted
///   scripts set them; they only ever remove (network) work.
/// - `LC_`/`DYLD_`/`MTL_`/`METAL_` prefixes: locale, dynamic-loader, and
///   Metal-framework configuration families. System-level, operator-owned,
///   phase-independent; dropping loader/Metal config could break how the
///   worker loads MLX and its metallib.
/// - `MLX_` prefix: MLX core tuning knobs read by mlx::core (e.g.
///   MLX_DISABLE_COMPILE, MLX_MAX_OPS_PER_BUFFER, MLX_RESOURCE_LIMIT) and by
///   the mlx-swift-lm fork (MLX_COMPILED_DECODE). Note "MLX_" does NOT match
///   harness "MLXFAST_*" names -- harness variables stay excluded.
/// - `DARKBLOOM_` prefix: model-runtime opt-ins read only by model-side code
///   and the mlx-swift-lm fork. The ranked workflow never sets them (absent
///   in BOTH ranked phases); they exist for operator/participant tuning on
///   local machines and must keep reaching the worker there.
/// - `MLXFAST_USE_RUNTIME_WORKER` is force-set to "0" so the child can never
///   recursively spawn another worker.
///
/// `SSH_AUTH_SOCK` is deliberately NOT allowlisted. The worker needs no SSH
/// agent (weights arrive via argv, dependencies are pre-resolved, network is
/// denied by its Seatbelt profile), and forwarding a live agent socket hands
/// submitted model code a usable authentication channel in any context where
/// the parent process happens to hold one. It is harmless on the ranked box
/// (no agent is present and PF blocks egress) but has no legitimate use here,
/// so it is dropped like every other non-essential name.
///
/// Maintainer contract: do NOT regress this to keep-by-default, do NOT add a
/// broad `MLXFAST_` (or `BENCH_`) allowance, do NOT re-add `SSH_AUTH_SOCK` (or
/// any other credential/agent socket), and never allowlist a name whose value
/// trusted code could set differently between the gates and timed passes. The
/// worker itself needs no MLXFAST_* configuration: its weights path arrives
/// via argv (`runtime-worker --weights ...`).
func sanitizedRuntimeWorkerEnvironment(_ environment: [String: String]) -> [String: String] {
    let allowedExactKeys: Set<String> = [
        "HF_HUB_OFFLINE",
        "HOME",
        "LANG",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TRANSFORMERS_OFFLINE",
        "USER",
        // macOS per-user default text encoding consulted by CoreFoundation.
        "__CF_USER_TEXT_ENCODING",
    ]
    let allowedPrefixes = [
        "DARKBLOOM_",
        "DYLD_",
        "LC_",
        "METAL_",
        "MLX_",
        "MTL_",
    ]
    var sanitized: [String: String] = [:]
    for (key, value) in environment
        where allowedExactKeys.contains(key)
        || allowedPrefixes.contains(where: { key.hasPrefix($0) })
    {
        sanitized[key] = value
    }
    sanitized["MLXFAST_USE_RUNTIME_WORKER"] = "0"
    return sanitized
}

func generateRuntimeWorkerNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeMutableBytes { buffer in
        if let baseAddress = buffer.baseAddress {
            arc4random_buf(baseAddress, buffer.count)
        }
    }
    return bytes
        .map { String(format: "%02x", $0) }
        .joined()
}

func runtimeWorkerLineLooksLikeJSONResponse(_ data: Data) -> Bool {
    for byte in data where byte != 0x20 && byte != 0x09 && byte != 0x0d {
        return byte == 0x7b
    }
    return false
}
