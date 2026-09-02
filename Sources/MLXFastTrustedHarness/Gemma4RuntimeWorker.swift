import Darwin
import Foundation
#if !MLXFAST_TRUSTED_HARNESS
import Metal
import MLX
#endif
import MLXFastCore
#if !MLXFAST_TRUSTED_HARNESS
import MLXFastModel
import MLXLLM
import MLXLMCommon
#endif

// Gemma4Runtime is split across Gemma4Runtime*.swift for auditability.
// Generated split; behavior identical to the original single file.

// MARK: - Phase-0 hello identity (protocol_version / backend / device)

// Defined UNGUARDED so the trusted target -- which links no MLX and compiles the
// worker bodies out -- still resolves these symbols. The trusted DFlash worker
// twin references them from `experimentalDFlashWorkerHello`, and while that body
// sits inside `#if !MLXFAST_TRUSTED_HARNESS` (so the reference never compiles in
// the trusted target today), leaving the symbols undefined here is a latent
// unlinked reference the compile-out merely masks. The two `let`s mirror the
// participant copy in Sources/MLXFastHarness byte-for-byte; the device label
// diverges only in that the trusted target links no Metal (see its body).

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
    #if !MLXFAST_TRUSTED_HARNESS
    MTLCreateSystemDefaultDevice()?.name ?? "apple-metal"
    #else
    // The trusted target links no Metal and never reaches this call: every
    // reference to it lives in a worker body compiled out under the same guard.
    // Resolve the symbol to the documented fallback so the reference links
    // rather than dangling, without pulling Metal into the trusted module.
    "apple-metal"
    #endif
}

#if !MLXFAST_TRUSTED_HARNESS
extension Gemma4Runtime {
    public static func runWorker(weightsPath: String) throws {
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
        _ = try weightCache.requireLibraryModelAtDrainFencedBoundary()
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        // expertStats is always the zero struct for this RAM-resident dense
        // runtime (no expert-streaming machinery); kept in the protocol hello
        // so the schema/field shape stays unchanged from earlier submissions.
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true,
            expertStats: expertStats(from: weightCache)
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
                        state: &state
                    )
                } catch {
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
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
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
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            _ = try weightCache.requireLibraryModel()
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
            let model = try weightCache.requireLibraryModel()
            try resetRuntimeWorkerAllocatorForPhaseStart()
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
            let model = try weightCache.requireLibraryModel()
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let cache = model.newCache(parameters: nil)
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
            let model = try weightCache.requireLibraryModel()
            try resetRuntimeWorkerAllocatorForPhaseStart()
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
            let cache = model.newCache(parameters: nil)
            let logits = try gemma4Logits(
                inputIDs: inputIDsArray(seedTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            let token = try Gemma4Correctness.greedyToken(from: logits)
            let seedToken = token
            materializeQwenCacheState(cache)
            state.decodeCache = cache
            state.decodeSeedTokenCount = seedTokens.count
            state.decodeStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            guard let cache = state.decodeCache else {
                throw MLXFastError.invalidInput("runtime worker decode_step before decode_begin")
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
            let logits = try gemma4Logits(
                inputIDs: inputIDsArray([inputToken]),
                model: try weightCache.requireLibraryModel(),
                cache: cache,
                positionOffset: state.decodeSeedTokenCount + state.decodeStep
            )
            let token = try Gemma4Correctness.greedyToken(from: logits)
            state.decodeStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "phase_diagnostics":
            let peakRamGB = peakResidentMemoryGB()
            let stats = expertStats(from: weightCache)
            let mlxActiveMemoryBytes = Memory.activeMemory
            let mlxCacheMemoryBytes = Memory.cacheMemory
            let mlxPeakMemoryBytes = Memory.peakMemory
            Memory.clearCache()
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                expertStats: stats,
                peakRamGB: peakRamGB,
                mlxActiveMemoryBytes: mlxActiveMemoryBytes,
                mlxCacheMemoryBytes: mlxCacheMemoryBytes,
                mlxPeakMemoryBytes: mlxPeakMemoryBytes
            )

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
    }

}
#endif

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
/// NOT single-sourced with `Gemma4A4BConfig`
/// (`Sources/MLXFastModel/Gemma4A4BConfig.swift`), unlike the
/// `MLXFastHarness` (participant worker) twin of this function: THIS target
/// (`MLXFastHarness` in `Package.swift`, path `Sources/MLXFastTrustedHarness`)
/// deliberately does not depend on `MLXFastModel` -- the trusted
/// `mlxfast-swift` binary links no MLX, model, or kernel code, so
/// `Gemma4A4BConfig` is out of scope here by construction, not by oversight.
/// This re-derives the same contract locally instead, sharing only what both
/// sides CAN reach without crossing that boundary:
/// `MLXFastCore.Gemma4A4BConfigKeys` for the key-set manifest (required,
/// forbidden, and the quantization-override family names) and
/// `MLXFastCore.MLXFastConstants` for the frozen geometry -- the same split
/// `Gemma4A4BConfig` itself uses. The two validators are kept in lockstep by
/// `gemma4A4BTrustedGateAgreesWithConfigLoaderAcrossFixtures` in
/// `Tests/MLXFastTests/Model/Gemma4A4BRuntimeWorkerGateLockstepTests.swift`,
/// which runs both against the same fixture set and asserts the same
/// accept/reject outcome, rather than by one function body.
///
/// This is what let the previous (Qwen-era) version of this gate silently
/// drift out of sync with the artifact it was actually validating: it still
/// enforced the `qwen3_5_text` schema, so a real transformed Gemma 4 config
/// failed here before weight loading even began --
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
func validateRuntimeWorkerPinnedConfigurationData(_ data: Data) throws {
    let json: Any
    do {
        json = try JSONSerialization.jsonObject(with: data)
    } catch {
        throw MLXFastError.invalidInput("runtime worker config.json must contain valid JSON")
    }
    guard let root = json as? [String: Any] else {
        throw MLXFastError.invalidInput("runtime worker config.json must be a JSON object")
    }

    try gemma4TrustedGateRequireExactKeys(root)

    var errors: [String] = []
    func expect<T: Equatable>(_ name: String, _ actual: T, _ expected: T) {
        if actual != expected {
            errors.append("\(name)=\(actual) expected \(expected)")
        }
    }

    do {
        let numHiddenLayers = try gemma4TrustedGateInt("num_hidden_layers", in: root)
        expect("num_hidden_layers", numHiddenLayers, MLXFastConstants.numHiddenLayers)
        expect("model_type", try gemma4TrustedGateString("model_type", in: root), "gemma4_text")
        expect("vocab_size", try gemma4TrustedGateInt("vocab_size", in: root), MLXFastConstants.vocabSize)
        expect(
            "vocab_size_per_layer_input",
            try gemma4TrustedGateInt("vocab_size_per_layer_input", in: root), 262_144)
        expect("hidden_size", try gemma4TrustedGateInt("hidden_size", in: root), MLXFastConstants.hiddenSize)
        expect(
            "hidden_size_per_layer_input",
            try gemma4TrustedGateInt("hidden_size_per_layer_input", in: root), 0)
        expect(
            "intermediate_size", try gemma4TrustedGateInt("intermediate_size", in: root),
            MLXFastConstants.intermediateSize)
        expect(
            "num_attention_heads", try gemma4TrustedGateInt("num_attention_heads", in: root),
            MLXFastConstants.attentionHeads)
        expect(
            "num_key_value_heads", try gemma4TrustedGateInt("num_key_value_heads", in: root),
            MLXFastConstants.numKeyValueHeads)
        expect(
            "num_global_key_value_heads",
            try gemma4TrustedGateInt("num_global_key_value_heads", in: root),
            MLXFastConstants.numGlobalKeyValueHeads)
        expect("num_kv_shared_layers", try gemma4TrustedGateInt("num_kv_shared_layers", in: root), 0)
        expect("head_dim", try gemma4TrustedGateInt("head_dim", in: root), MLXFastConstants.headDim)
        expect(
            "global_head_dim", try gemma4TrustedGateInt("global_head_dim", in: root),
            MLXFastConstants.globalHeadDim)
        expect(
            "sliding_window", try gemma4TrustedGateInt("sliding_window", in: root),
            MLXFastConstants.slidingWindow)
        expect("rms_norm_eps", try gemma4TrustedGateDouble("rms_norm_eps", in: root), 1e-6)
        expect(
            "hidden_activation", try gemma4TrustedGateString("hidden_activation", in: root),
            "gelu_pytorch_tanh")
        expect(
            "max_position_embeddings", try gemma4TrustedGateInt("max_position_embeddings", in: root),
            262_144)
        expect("attention_bias", try gemma4TrustedGateBool("attention_bias", in: root), false)
        expect("attention_dropout", try gemma4TrustedGateDouble("attention_dropout", in: root), 0)
        expect("attention_k_eq_v", try gemma4TrustedGateBool("attention_k_eq_v", in: root), true)
        expect(
            "final_logit_softcapping", try gemma4TrustedGateDouble("final_logit_softcapping", in: root),
            MLXFastConstants.finalLogitSoftcapping)
        expect(
            "tie_word_embeddings", try gemma4TrustedGateBool("tie_word_embeddings", in: root),
            MLXFastConstants.tieWordEmbeddings)
        expect("enable_moe_block", try gemma4TrustedGateBool("enable_moe_block", in: root), true)
        expect("num_experts", try gemma4TrustedGateInt("num_experts", in: root), MLXFastConstants.numExperts)
        expect(
            "top_k_experts", try gemma4TrustedGateInt("top_k_experts", in: root),
            MLXFastConstants.numExpertsPerToken)
        expect(
            "moe_intermediate_size", try gemma4TrustedGateInt("moe_intermediate_size", in: root),
            MLXFastConstants.moeIntermediateSize)
        expect("use_double_wide_mlp", try gemma4TrustedGateBool("use_double_wide_mlp", in: root), false)
        expect(
            "use_bidirectional_attention",
            try gemma4TrustedGateString("use_bidirectional_attention", in: root), "vision")
        expect("dtype", try gemma4TrustedGateString("dtype", in: root), "bfloat16")
        expect("use_cache", try gemma4TrustedGateBool("use_cache", in: root), true)
        expect("bos_token_id", try gemma4TrustedGateInt("bos_token_id", in: root), 2)
        expect("eos_token_id", try gemma4TrustedGateInt("eos_token_id", in: root), 1)
        expect("pad_token_id", try gemma4TrustedGateInt("pad_token_id", in: root), 0)

        let layerTypes = try gemma4TrustedGateStringArray("layer_types", in: root)
        let interval = MLXFastConstants.fullAttentionInterval
        let expectedLayerTypes = (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % interval == interval - 1 ? "full_attention" : "sliding_attention"
        }
        if layerTypes != expectedLayerTypes {
            errors.append(
                "layer_types does not match the pinned \(interval)-layer repeat schedule"
            )
        }

        let rope = try gemma4TrustedGateObject("rope_parameters", in: root)
        guard Set(rope.keys) == ["sliding_attention", "full_attention"] else {
            errors.append(
                "rope_parameters key set is \(Set(rope.keys).sorted()), expected "
                    + "[full_attention, sliding_attention]"
            )
            throw MLXFastError.invalidInput("rope_parameters key set mismatch")
        }
        let sliding = try gemma4TrustedGateObject("sliding_attention", in: rope)
        guard Set(sliding.keys) == ["rope_theta", "rope_type"] else {
            errors.append(
                "rope_parameters.sliding_attention key set is \(Set(sliding.keys).sorted())"
            )
            throw MLXFastError.invalidInput("sliding_attention key set mismatch")
        }
        expect("sliding_attention.rope_theta", try gemma4TrustedGateDouble("rope_theta", in: sliding), 10_000)
        expect("sliding_attention.rope_type", try gemma4TrustedGateString("rope_type", in: sliding), "default")

        let full = try gemma4TrustedGateObject("full_attention", in: rope)
        guard Set(full.keys) == ["rope_theta", "rope_type", "partial_rotary_factor"] else {
            errors.append("rope_parameters.full_attention key set is \(Set(full.keys).sorted())")
            throw MLXFastError.invalidInput("full_attention key set mismatch")
        }
        expect(
            "full_attention.rope_theta", try gemma4TrustedGateDouble("rope_theta", in: full), 1_000_000)
        expect(
            "full_attention.rope_type", try gemma4TrustedGateString("rope_type", in: full), "proportional")
        expect(
            "full_attention.partial_rotary_factor",
            try gemma4TrustedGateDouble("partial_rotary_factor", in: full), 0.25)

        let quantization = try gemma4TrustedGateObject("quantization", in: root)
        expect("quantization.group_size", try gemma4TrustedGateInt("group_size", in: quantization), 64)
        expect("quantization.bits", try gemma4TrustedGateInt("bits", in: quantization), 4)
        expect("quantization.mode", try gemma4TrustedGateString("mode", in: quantization), "affine")

        // The 120-entry per-tensor override table (port notes section 1.3):
        // four projection families on every layer, each promoted to 8 bits.
        // The expected key set is pinned by CONSTRUCTION (same discipline as
        // `Gemma4A4BConfig.quantizationOverrideErrors`), not by count.
        var expectedOverrides: Set<String> = []
        for layer in 0..<numHiddenLayers {
            for family in Gemma4A4BConfigKeys.quantizationOverrideFamilies {
                // Mirrors `Gemma4A4BWeightNames.modelPrefix` +
                // `Gemma4A4BWeightNames.layer(_:_:)`
                // (`Sources/MLXFastModel/Gemma4A4BWeights.swift`), which this
                // target cannot import; kept in lockstep by the test named
                // in this function's doc comment.
                expectedOverrides.insert("language_model.model.layers.\(layer).\(family)")
            }
        }
        let scalarKeys: Set<String> = ["group_size", "bits", "mode"]
        let actualOverrides = Set(quantization.keys).subtracting(scalarKeys)
        let missingOverrides = expectedOverrides.subtracting(actualOverrides).sorted()
        let unexpectedOverrides = actualOverrides.subtracting(expectedOverrides).sorted()
        if !missingOverrides.isEmpty {
            errors.append(
                "quantization is missing \(missingOverrides.count) expected per-tensor "
                    + "override(s), first: \(missingOverrides[0])"
            )
        }
        if !unexpectedOverrides.isEmpty {
            errors.append(
                "quantization carries \(unexpectedOverrides.count) unexpected per-tensor "
                    + "override(s), first: \(unexpectedOverrides[0])"
            )
        }
        for key in expectedOverrides.intersection(actualOverrides).sorted() {
            guard let entry = quantization[key] as? [String: Any],
                  Set(entry.keys) == ["group_size", "bits"],
                  try gemma4TrustedGateInt("bits", in: entry) == 8,
                  try gemma4TrustedGateInt("group_size", in: entry) == 64
            else {
                errors.append("quantization override \(key) does not match the pinned bits=8 group_size=64 shape")
                break
            }
        }
    } catch let error as MLXFastError {
        // A structural parse failure (wrong JSON kind, non-finite number, ...)
        // is itself a rejection reason; fold it in with whatever field-level
        // errors were already collected instead of losing it.
        errors.append(error.description)
    }

    guard errors.isEmpty else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json does not match the pinned Gemma 4 26B A4B "
                + "gemma4_text architecture: " + errors.joined(separator: "; ")
        )
    }
}

private func gemma4TrustedGateRequiredValue(_ key: String, in root: [String: Any]) throws -> Any {
    guard let value = root[key] else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) is required")
    }
    guard !(value is NSNull) else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must not be null")
    }
    return value
}

private func gemma4TrustedGateInt(_ key: String, in root: [String: Any]) throws -> Int {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          let integer = Int(number.stringValue)
    else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json field \(key) must be a finite integer in Int range"
        )
    }
    return integer
}

private func gemma4TrustedGateDouble(_ key: String, in root: [String: Any]) throws -> Double {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a finite number")
    }
    let result = number.doubleValue
    guard result.isFinite else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a finite number")
    }
    return result
}

private func gemma4TrustedGateBool(_ key: String, in root: [String: Any]) throws -> Bool {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let number = value as? NSNumber,
          CFGetTypeID(number) == CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a boolean")
    }
    return number.boolValue
}

private func gemma4TrustedGateString(_ key: String, in root: [String: Any]) throws -> String {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let result = value as? String else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a string")
    }
    return result
}

private func gemma4TrustedGateObject(_ key: String, in root: [String: Any]) throws -> [String: Any] {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let result = value as? [String: Any] else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a JSON object")
    }
    return result
}

private func gemma4TrustedGateStringArray(_ key: String, in root: [String: Any]) throws -> [String] {
    let value = try gemma4TrustedGateRequiredValue(key, in: root)
    guard let result = value as? [String] else {
        throw MLXFastError.invalidInput("runtime worker config.json field \(key) must be a string array")
    }
    return result
}

private func gemma4TrustedGateRequireExactKeys(_ root: [String: Any]) throws {
    var errors: [String] = []
    for key in Gemma4A4BConfigKeys.required.sorted() {
        guard let value = root[key] else {
            errors.append("missing required key \(key)")
            continue
        }
        if value is NSNull {
            errors.append("required key \(key) must not be null")
        }
    }
    for key in Gemma4A4BConfigKeys.forbidden where root[key] != nil && !(root[key] is NSNull) {
        errors.append("forbidden key \(key) is present and non-null")
    }
    let known = Gemma4A4BConfigKeys.required
        .union(Gemma4A4BConfigKeys.forbidden)
        .union([Gemma4A4BConfigKeys.quantizationKey])
    for key in root.keys.sorted() where !known.contains(key) {
        errors.append("unexpected key \(key)")
    }
    guard errors.isEmpty else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json key-set check failed: " + errors.joined(separator: ", ")
        )
    }
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
        verifyBlockTokens: [Int]? = nil
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

#if !MLXFAST_TRUSTED_HARNESS
    struct RuntimeWorkerState {
        var correctnessCache: [KVCache]?
        var correctnessPromptTokenCount = 0
        var correctnessStep = 0
        var decodeCache: [KVCache]?
        var decodeSeedTokenCount = 0
        var decodeStep = 0
    }
#endif

struct RuntimeWorkerPreflightResponse: Codable, Equatable {
    let ok: Bool
    let error: String?

    init(ok: Bool, error: String? = nil) {
        self.ok = ok
        self.error = error
    }
}

// Trusted-side mirrors of the two OBJECT fields the worker emits on the v1.1
// speculative surface. The trusted native CLI never consumes these — it decodes
// them only so its strict unknown-field decoder does not REJECT a gated-on line
// (the cross-decode regression guard). Wire shapes match the worker's
// RuntimeWorkerHeadProvenance / RuntimeWorkerEffectiveSpec byte-for-byte.

struct TrustedWorkerHeadProvenance: Codable, Equatable {
    let sha256: String
    let bytes: Int
    let fileCount: Int

    enum CodingKeys: String, CodingKey {
        case sha256
        case bytes
        case fileCount = "file_count"
    }
}

struct TrustedWorkerEffectiveSpec: Codable, Equatable {
    let mode: String
    let mtp: EffectiveMTP?
    let dflash: EffectiveDFlash?

    struct EffectiveMTP: Codable, Equatable {
        let depth: Int
    }

    struct EffectiveDFlash: Codable, Equatable {
        let depth: Int
        let draft: EffectiveDraft?

        struct EffectiveDraft: Codable, Equatable {
            let artifact: String
            let sha256: String
        }
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
    // Phase-0 hello identity (v1). The worker emits these on EVERY hello and
    // benchd requires `protocol_version`; the trusted native client accepts them
    // (and ignores them — it validates only id/ok/nonce) so its worker-spawning
    // verbs decode the hello instead of rejecting an unknown field.
    let protocolVersion: Int?
    let backend: String?
    let device: String?
    // v1.1 speculative surface (gated on at spawn). The trusted native CLI never
    // spawns a gated-on worker in production, but decoding these keeps its strict
    // decoder a proper SUPERSET so it parses a gated-on line instead of rejecting
    // it — the cross-decode regression guard for the spawn gate (fix #5).
    let specModes: [String]?
    let capabilities: [String]?
    let headProvenance: TrustedWorkerHeadProvenance?
    let effectiveSpec: TrustedWorkerEffectiveSpec?
    let completedWork: Int?
    let cacheMemory: Int?
    let acceptanceLengths: [Int]?
    let draftedTotal: Int?
    let acceptedTotal: Int?
    let committedTotal: Int?
    let physicalVerifierWidth: Int?
    // v1.2 batched (cohort) free-run surface. Same superset rationale as the
    // v1.1 block above: the trusted native CLI never issues the batched verbs,
    // but its strict decoder must keep parsing every line the worker can emit
    // (the gate-on hello now carries `max_batch_size` beside `capabilities`),
    // so the cross-decode key-set guard holds by construction.
    let maxBatchSize: Int?
    let seedTokenByStream: [Int]?
    let effectiveBatchSize: Int?
    let tokensByStream: [[Int]]?
    let naturalAcceptedByStream: [[Int]]?
    let rounds: Int?
    let activeStreamsByRound: [Int]?
    let depthClampReasons: [String: Int]?
    // Per-stream timing instrumentation (spec step 1). Same superset
    // rationale as the rest of the v1.2 block above: the trusted native CLI
    // never issues the batched verbs, but its strict decoder must keep
    // parsing every line the worker can emit.
    let prefillNsByStream: [UInt64]?
    let decodeNsByStream: [UInt64]?
    // PR-1 fidelity-gate (trusted-side DECODE mirror). The trusted parent parses
    // the `cohort_reference_replay` measurement report a pinned reference worker
    // returns. benchd never issues this verb; the trusted decoder must still
    // parse every line the worker can emit (cross-decode parity).
    let cohortReferenceReplay: CohortReferenceReplayReport?

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
        protocolVersion: Int? = nil,
        backend: String? = nil,
        device: String? = nil,
        specModes: [String]? = nil,
        capabilities: [String]? = nil,
        headProvenance: TrustedWorkerHeadProvenance? = nil,
        effectiveSpec: TrustedWorkerEffectiveSpec? = nil,
        completedWork: Int? = nil,
        cacheMemory: Int? = nil,
        acceptanceLengths: [Int]? = nil,
        draftedTotal: Int? = nil,
        acceptedTotal: Int? = nil,
        committedTotal: Int? = nil,
        physicalVerifierWidth: Int? = nil,
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
        cohortReferenceReplay: CohortReferenceReplayReport? = nil
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
        self.protocolVersion = protocolVersion
        self.backend = backend
        self.device = device
        self.specModes = specModes
        self.capabilities = capabilities
        self.headProvenance = headProvenance
        self.effectiveSpec = effectiveSpec
        self.completedWork = completedWork
        self.cacheMemory = cacheMemory
        self.acceptanceLengths = acceptanceLengths
        self.draftedTotal = draftedTotal
        self.acceptedTotal = acceptedTotal
        self.committedTotal = committedTotal
        self.physicalVerifierWidth = physicalVerifierWidth
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
        protocolVersion = try container.decodeIfPresent(
            Int.self,
            forKey: .protocolVersion
        )
        backend = try container.decodeIfPresent(String.self, forKey: .backend)
        device = try container.decodeIfPresent(String.self, forKey: .device)
        specModes = try container.decodeIfPresent(
            [String].self,
            forKey: .specModes
        )
        capabilities = try container.decodeIfPresent(
            [String].self,
            forKey: .capabilities
        )
        headProvenance = try container.decodeIfPresent(
            TrustedWorkerHeadProvenance.self,
            forKey: .headProvenance
        )
        effectiveSpec = try container.decodeIfPresent(
            TrustedWorkerEffectiveSpec.self,
            forKey: .effectiveSpec
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
        physicalVerifierWidth = try container.decodeIfPresent(
            Int.self,
            forKey: .physicalVerifierWidth
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
        try container.encodeIfPresent(protocolVersion, forKey: .protocolVersion)
        try container.encodeIfPresent(backend, forKey: .backend)
        try container.encodeIfPresent(device, forKey: .device)
        try container.encodeIfPresent(specModes, forKey: .specModes)
        try container.encodeIfPresent(capabilities, forKey: .capabilities)
        try container.encodeIfPresent(headProvenance, forKey: .headProvenance)
        try container.encodeIfPresent(effectiveSpec, forKey: .effectiveSpec)
        try container.encodeIfPresent(completedWork, forKey: .completedWork)
        try container.encodeIfPresent(cacheMemory, forKey: .cacheMemory)
        try container.encodeIfPresent(
            acceptanceLengths,
            forKey: .acceptanceLengths
        )
        try container.encodeIfPresent(draftedTotal, forKey: .draftedTotal)
        try container.encodeIfPresent(acceptedTotal, forKey: .acceptedTotal)
        try container.encodeIfPresent(committedTotal, forKey: .committedTotal)
        try container.encodeIfPresent(
            physicalVerifierWidth,
            forKey: .physicalVerifierWidth
        )
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
        case protocolVersion = "protocol_version"
        case backend
        case device
        case specModes = "spec_modes"
        case capabilities
        case headProvenance = "head_provenance"
        case effectiveSpec = "effective_spec"
        case completedWork = "completed_work"
        case cacheMemory = "cache_memory"
        case acceptanceLengths = "acceptance_lengths"
        case draftedTotal = "drafted_total"
        case acceptedTotal = "accepted_total"
        case committedTotal = "committed_total"
        case physicalVerifierWidth = "physical_verifier_width"
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

/// Reassert the actual runtime-worker executable as the final Seatbelt exec
/// rule. Operator-provided profiles can outlive a binary-layout change; a
/// stale allow would otherwise make sandbox-exec die before the protocol hello.
/// Strip every earlier exec exception, then append a deny plus one literal
/// allow so the resulting profile admits exactly this worker.
func runtimeWorkerSandboxProfile(
    rebinding profilePath: String,
    toExecutableAt executablePath: String
) throws -> String {
    let source = try String(contentsOfFile: profilePath, encoding: .utf8)
    var retainedLines: [Substring] = []
    for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("(allow process-exec") {
            guard trimmed.hasSuffix(")") else {
                throw MLXFastError.invalidInput(
                    "runtime worker sandbox profile has an unsupported multiline process-exec allow"
                )
            }
            continue
        }
        retainedLines.append(line)
    }
    let sourceWithoutExecAllows = retainedLines.joined(separator: "\n")
    let resolvedExecutablePath = URL(fileURLWithPath: executablePath)
        .standardizedFileURL
        .resolvingSymlinksInPath()
        .path
    let escapedExecutablePath = resolvedExecutablePath
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
    let separator = sourceWithoutExecAllows.hasSuffix("\n") ? "" : "\n"
    let rebound = sourceWithoutExecAllows + separator + """
    ;; Trusted-harness executable binding (must remain the final exec rules).
    (deny process-exec*)
    (allow process-exec (literal "\(escapedExecutablePath)"))
    """
    let outputURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(
            "mlxfast-runtime-worker-bound-\(UUID().uuidString).sb"
        )
    try rebound.write(to: outputURL, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o400],
        ofItemAtPath: outputURL.path
    )
    return outputURL.path
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
        // BELT, not the trust boundary (bench#143 wire b): a redundant in-process
        // re-verification of the worker executable against its env-seam sha pin
        // before spawn. The AUTHORITATIVE binding is the bench gate's
        // WP_ENGINE_BIN_SHA256 seal (window-preflight.sh), which run-paired-window.sh
        // now forces to run; this engine-side check is competitor-editable and only
        // narrows the window in honest/misconfigured setups. See RuntimeWorkerExecutablePin.
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: options.executablePath
        )
        if let configuredSandboxProfilePath = options.sandboxProfilePath {
            let sandboxProfilePath = try runtimeWorkerSandboxProfile(
                rebinding: configuredSandboxProfilePath,
                toExecutableAt: options.executablePath
            )
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
        // BELT, not the trust boundary (bench#143 wire b): a redundant in-process
        // re-verification of the worker executable against its env-seam sha pin
        // before spawn. The AUTHORITATIVE binding is the bench gate's
        // WP_ENGINE_BIN_SHA256 seal (window-preflight.sh), which run-paired-window.sh
        // now forces to run; this engine-side check is competitor-editable and only
        // narrows the window in honest/misconfigured setups. See RuntimeWorkerExecutablePin.
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: options.executablePath
        )
        if let configuredSandboxProfilePath = options.sandboxProfilePath {
            let sandboxProfilePath = try runtimeWorkerSandboxProfile(
                rebinding: configuredSandboxProfilePath,
                toExecutableAt: options.executablePath
            )
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

    // --- CBv2 reference-tape recording (record-reference-tape) ------------
    // Trusted-CLI-only verbs: the recorder's CBv2 backend opens the SAME
    // width-1 CBv2 engine session the v1.1 free-run legs run (drafter-less
    // serial configuration) and drains it with a per-row top-2 readout, so
    // the pool tapes are produced by the implementation that is
    // oracle-checked against them (port-notes 5.1, within-backend). benchd
    // never issues these kinds; both ride existing request fields
    // (seed_tokens / count) and existing response fields (seed_token, tokens,
    // per_row_top2_tokens / per_row_top2_logits).

    /// Open one fresh CBv2 recording pass: full seed prefill through the
    /// engine, returns the prefill-bonus (seed) argmax.
    func beginRecordReference(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "record_reference_begin",
            seedTokens: seedTokens
        )
    }

    /// Free-run the open recording pass to `rowCount` committed rows and
    /// return the chain plus the per-row top-2 readout.
    func runRecordReference(rowCount: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "record_reference_run",
            rowCount: rowCount
        )
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
