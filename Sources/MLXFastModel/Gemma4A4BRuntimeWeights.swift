import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

/// KV cache backend for this track.
///
/// RULED 2026-08-22: contiguous, pinned EXPLICITLY on both legs (the serial
/// control and the speculative leg), with refusal rather than degradation.
/// Darkbloom's production decision is contiguous — the v0.8.1 capacity revert
/// made it the serving arm, with paged explicit-only — and the parity target
/// is darkbloom's serving arm.
///
/// The parity claim this supports is WITHIN-BACKEND: the speculative leg must
/// be token-exact against a serial control running the same backend. That
/// claim is only meaningful if both legs provably ran the same backend, which
/// is why `.auto` is not a legal value here rather than merely discouraged —
/// an auto-resolved leg makes the claim unfalsifiable even when it happens to
/// resolve correctly.
public enum Gemma4A4BKVBackend: String, Equatable, Sendable {
    case contiguous

    /// The pinned backend for this track. There is deliberately no `auto` and
    /// no `paged` case: a value that cannot be spelled cannot be selected by
    /// accident, and adding one is a pinned-identity change.
    public static let pinned: Gemma4A4BKVBackend = .contiguous

    /// Environment name a caller may use to ASSERT the backend. It cannot
    /// select a different one — an unrecognised or non-matching value is a
    /// refusal, never a silent fallback. This exists so a box session can
    /// prove which backend a leg ran rather than infer it.
    public static let environmentName = "MLXFAST_GEMMA4_KV_BACKEND"

    /// Resolve and verify the backend for one leg.
    ///
    /// Refuse-not-degrade: the only accepted values are absent (meaning "take
    /// the pin") or the exact pinned spelling. Anything else throws, including
    /// a value this type could otherwise represent, because the failure this
    /// guards against is a requested backend silently resolving to a different
    /// one — the exact mechanism that changed the answer in the darkbloom
    /// adoption-exactness report without being visible in the result.
    public static func resolve(
        requested: String?
    ) throws -> Gemma4A4BKVBackend {
        guard let requested, !requested.isEmpty else { return pinned }
        guard requested == pinned.rawValue else {
            throw MLXFastError.invalidInput(
                "\(environmentName)=\(requested) is not runnable on this "
                    + "engine: the KV backend is pinned to "
                    + "\(pinned.rawValue) on both legs and this engine "
                    + "refuses rather than degrading to another backend. See "
                    + "docs/gemma4-port-notes.md section 5."
            )
        }
        return pinned
    }

    /// Verify that a constructed cache stack is the pinned backend.
    ///
    /// `Gemma4TextModel.newCache` builds `StandardKVCache` on the global
    /// layers and `RotatingKVCache` on the sliding ones — both contiguous by
    /// construction. This asserts that, so a vendored change that swapped in a
    /// paged implementation is caught at startup rather than by a parity
    /// mismatch much later.
    public static func validateContiguous(caches: [any KVCache]) throws {
        for (index, cache) in caches.enumerated() {
            let isContiguous =
                cache is StandardKVCache || cache is RotatingKVCache
            guard isContiguous else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 26B A4B KV backend check failed: cache for layer "
                        + "\(index) is \(type(of: cache)), which is not one of "
                        + "the pinned contiguous cache types "
                        + "(StandardKVCache / RotatingKVCache)"
                )
            }
        }
    }
}

/// Eagerly loads exactly one RAM-resident Gemma 4 26B A4B execution backend.
///
/// The scored path is the VENDORED `Gemma4TextModel` reached through this
/// cache, not a re-implemented forward pass. This type owns the checkpoint →
/// model wiring: shard load, prefix strip, sanitize, per-path quantization,
/// strict parameter update, and constructor-time warmup.
public final class Gemma4A4BRuntimeWeightCache {
    public let loader: Gemma4A4BWeightLoader
    public let config: Gemma4A4BConfig
    public let libraryModel: Gemma4TextModel?
    public let loadError: Error?

    public init(loader: Gemma4A4BWeightLoader, config: Gemma4A4BConfig) {
        self.loader = loader
        self.config = config

        let startupEnvironment = ProcessInfo.processInfo.environment
        if config.numHiddenLayers >= 16,
           RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
               physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
               requestedProfile:
                   startupEnvironment[
                       RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
                   ],
               killSwitchValue:
                   startupEnvironment["DARKBLOOM_QWEN_MTP_POST_WIRE_COMMAND_BUFFER"]
           )
        {
            setenv("MLX_MAX_MB_PER_BUFFER", "512", 1)
            setenv("MLX_MAX_OPS_PER_BUFFER", "512", 1)
            Memory.cacheLimit = 32 << 30
        }

        do {
            let model = try Self.loadLibraryModel(
                denseStore: loader.denseStore, config: config)
            libraryModel = model
            loadError = nil
            Self.warmLibraryModel(model, config: config)
        } catch {
            libraryModel = nil
            loadError = error
        }
    }

    public func requireLibraryModel() throws -> Gemma4TextModel {
        // Fence for fast-ack engine shutdowns: a previous phase's detached
        // drain (already GPU-complete in practice — the inter-phase gap is
        // orders of magnitude longer than a drain) must be fully retired
        // before this phase builds a new engine on the same allocator.
        // Normally returns immediately with nothing registered.
        CBv2DetachedDrainRegistry.joinAll(timeout: 5)
        guard let libraryModel else {
            throw loadError
                ?? MLXFastError.invalidInput(
                    "Gemma 4 26B A4B text model was not loaded")
        }
        return libraryModel
    }

    /// Startup readiness check used by the sandboxed worker.
    public func validateSelectedBackend() throws {
        let model = try requireLibraryModel()
        try Gemma4A4BKVBackend.validateContiguous(
            caches: model.newCache(parameters: nil))
    }

    /// Preserve the proven load sequence:
    /// shard load -> prefix strip -> sanitize -> per-path affine quantize ->
    /// strict parameter update -> eval.
    private static func loadLibraryModel(
        denseStore: DenseTensorStore,
        config: Gemma4A4BConfig
    ) throws -> Gemma4TextModel {
        let directory = URL(fileURLWithPath: denseStore.weightsPath)
        let configData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let textConfiguration = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: configData)
        let model = Gemma4TextModel(textConfiguration)

        var weights: [String: MLXArray] = [:]
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let discoveredShards = entries
            .filter { $0.pathExtension == "safetensors" }
            .map(\.lastPathComponent)
        let shardNames = try validateRuntimeShardInventory(
            referencedShards: denseStore.shardNames,
            discoveredShards: discoveredShards
        )

        var nameTracker = RuntimeWeightNameTracker()
        for shardName in shardNames {
            let shard = directory.appendingPathComponent(shardName)
            let expectedNames = denseStore.tensorNames(inShard: shardName)
            for (key, value) in try loadArrays(url: shard) {
                let renamed = try nameTracker.register(
                    originalName: key,
                    shardName: shardName,
                    expectedNames: expectedNames
                )
                weights[renamed] = value
            }
        }
        try nameTracker.validateComplete(
            expectedNames: Set(denseStore.tensorNames))

        let sanitized = model.sanitize(weights: weights)
        // GATEUP-FUSE-PREFILL: the sanitize pass re-binds the routed-expert
        // gate/up planes as slices of a concatenated storage; the raw loaded
        // planes are then reachable only through that concatenation, and
        // dropping this dictionary lets the evaluation below retire each
        // layer's raw planes as soon as its storage is built instead of
        // holding every layer's until this function returns.
        weights.removeAll()
        try quantizeWithPerPathWidths(
            model: model, sanitized: sanitized, config: config)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized),
            verify: [.all]
        )
        eval(model)
        return model
    }

    /// Apply the checkpoint's MIXED-PRECISION quantization contract.
    ///
    /// This is the code half of the config's override table, and getting it
    /// wrong is the top silent-divergence risk on this target. The Qwen-era
    /// loader passed ONE `(groupSize, bits, mode)` triple for every path with
    /// a `.scales` sibling. On this checkpoint that quantizes 120 tensors at
    /// 4 bits that were written at 8 — right names, right shapes, wrong
    /// numerics, and nothing downstream notices.
    ///
    /// NAME-SPACE MAPPING, which is the subtle part. `quantize(model:)` reports
    /// RUNTIME module paths (`model.layers.0.mlp.gate_proj`), because
    /// `RuntimeWeightNameTracker` strips the `language_model.` prefix on load.
    /// The config's override keys are CHECKPOINT paths
    /// (`language_model.model.layers.0.mlp.gate_proj`). Looking a runtime path
    /// up directly would miss every override and fail open — silently, and in
    /// exactly the direction that produces the bug above — so the prefix is
    /// restored before lookup, and the restoration is verified below rather
    /// than assumed.
    ///
    /// Internal rather than private: `Gemma4A4BUpstreamEquivalence` (the
    /// runtime-vs-vendored gate, docs/gemma4-port-notes.md section 6.4) calls
    /// this directly so its "runtime" leg is this exact scored-path code, not
    /// a re-derived copy that could drift from it.
    static func quantizeWithPerPathWidths(
        model: Gemma4TextModel,
        sanitized: [String: MLXArray],
        config: Gemma4A4BConfig
    ) throws {
        let mode: QuantizationMode
        switch config.quantization.mode {
        case "affine":
            mode = .affine
        default:
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization mode "
                    + "\(config.quantization.mode) is unsupported"
            )
        }

        // Fail closed if the prefix mapping stops matching the checkpoint's
        // own naming: every override key must be reachable from some runtime
        // path, i.e. must carry the prefix this mapping removes.
        let prefix = "\(RuntimeWeightNameTracker.languageModelPrefix)"
        let unreachable = config.quantization.overrides.keys
            .filter { !$0.hasPrefix(prefix) }
            .sorted()
        guard unreachable.isEmpty else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization override key(s) do not carry the "
                    + "\(prefix) prefix and could never be matched against a "
                    + "runtime module path: \(unreachable.prefix(3).joined(separator: ", "))"
            )
        }

        var appliedOverrides = 0
        quantize(model: model) { path, _ in
            guard sanitized["\(path).scales"] != nil else { return nil }
            let checkpointPath = prefix + path
            let spec = config.quantization.spec(forPath: checkpointPath)
            if config.quantization.overrides[checkpointPath] != nil {
                appliedOverrides += 1
            }
            return (
                groupSize: spec.groupSize, bits: spec.bits, mode: mode
            )
        }

        // Every declared override must have been applied to a real module. A
        // count short of the table means the config named a path the model
        // does not have, which is a contract break, not a tolerable no-op.
        guard appliedOverrides == config.quantization.overrides.count else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B quantization applied \(appliedOverrides) of "
                    + "\(config.quantization.overrides.count) declared "
                    + "per-tensor overrides; every declared override must "
                    + "match a quantized module"
            )
        }
    }

    /// Prompt-independent constructor warmup using Gemma's BOS token.
    ///
    /// One prefill-shaped forward and one single-token decode step against a
    /// throwaway cache, evaluated and discarded. Inputs are constant BOS
    /// tokens, so this is prompt-independent and cannot affect model output.
    private static func warmLibraryModel(
        _ model: Gemma4TextModel, config: Gemma4A4BConfig
    ) {
        let bosToken = Int32(config.bosTokenId)
        let caches = model.newCache(parameters: nil)
        let prefillTokens = MLXArray(
            Array(repeating: bosToken, count: MLXFastConstants.correctnessPromptTokens),
            [1, MLXFastConstants.correctnessPromptTokens]
        )
        eval(model(prefillTokens, cache: caches))
        eval(model(MLXArray([bosToken], [1, 1]), cache: caches))
        warmCohortShapes(model, config: config)
        // Retire the warm's own buffers HERE, unscored: the trusted
        // phase-start clearCache runs inside the charged window, so any
        // free buffers the warm leaves behind would be deallocated on the
        // measured clock.
        Memory.clearCache()
    }

    /// Prompt-independent constructor warmup at the scored B=8 cohort shapes.
    ///
    /// The legacy warm above only touches the single-stream `[1, 1024]` and
    /// `[1, 1]` paths. The measured cohort windows run through the CBv2
    /// engine at batch 8, whose kernel set is gated on width-8 geometry and
    /// therefore never compiles from the legacy warm: the M=8 ordinary
    /// QMV row-pair (attention projections and the tied lm_head), the
    /// 64-assignment expert gather pair, the 64-key route sort and the fused
    /// weighted expert-unsort, the batched two-pass sliding decode attention
    /// at a saturated 1024 window, and the packed `[8, 1024]` prefill
    /// attention. On a cold per-process pipeline cache each of those
    /// first-compiles inside a parent-clocked measured window.
    ///
    /// This warm drives one real CBv2 engine round-trip at exactly those
    /// shapes — an 8-stream BOS-only 1024-token seed prefill plus a few
    /// `[8, 1]` decode rounds — then shuts the engine down and discards every
    /// output. Inputs are constant BOS tokens against throwaway caches, so
    /// the warm is prompt-independent and cannot affect model output; the
    /// only durable effect is process-global compiled-pipeline state. It is
    /// best-effort and deadline-guarded: no failure here may fail model load.
    private static func warmCohortShapes(
        _ model: Gemma4TextModel, config: Gemma4A4BConfig
    ) {
        let batch = 8
        let seedCount = MLXFastConstants.correctnessPromptTokens
        // Seed argmax + three decode rounds: round one crosses the ring
        // wrap/saturation transition, the later rounds compile the
        // steady-state batched decode path. The env override exists for
        // local profiling only (a longer warm makes this a B=8 step driver
        // for CBV2_STEP_PROFILE); the default is what ships.
        let warmCompletionTokens = min(
            128,
            max(
                1,
                ProcessInfo.processInfo.environment["MLXFAST_WARM_COHORT_TOKENS"]
                    .flatMap(Int.init) ?? 4))
        do {
            let caches = try model.newCacheV2 { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }
            // Admission-ledger ceiling only (no allocation): sized well above
            // the ~1 GiB the 8-row warm cohort can retain.
            let backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 4 << 30))
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds,
                backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: batch,
                    maxBatchedTokensPerStep: Swift.max(2048, batch * seedCount),
                    prefillChunkSize: Swift.max(512, seedCount),
                    maxWaiting: batch,
                    enablePrefixCache: false))
            let seeds = Array(
                repeating: Int(config.bosTokenId), count: seedCount)
            let drained = DispatchSemaphore(value: 0)
            let consumer = Task {
                await withTaskGroup(of: Void.self) { group in
                    for slot in 0..<batch {
                        guard
                            let stream = try? engine.submit(
                                CBv2Request(
                                    id: CBv2RequestID(UInt64(slot)),
                                    promptTokens: seeds,
                                    sampling: CBv2SamplingParams(temperature: 0),
                                    maxTokens: warmCompletionTokens,
                                    stopTokens: [],
                                    prefixCacheEnabled: false))
                        else { continue }
                        group.addTask {
                            var tokens: [Int] = []
                            for await event in stream {
                                if case .delta(_, let ids, _) = event {
                                    tokens.append(contentsOf: ids)
                                }
                                if case .finished = event { break }
                            }
                            // Local diagnostics only: BOS-seeded greedy decode
                            // is deterministic, so these ids are an integrated
                            // B=8 bit-exactness smoke across kernel variants.
                            if ProcessInfo.processInfo
                                .environment["MLXFAST_WARM_COHORT_DEBUG_TOKENS"]
                                != nil
                            {
                                FileHandle.standardError.write(
                                    Data(
                                        "[warm] slot \(slot) tokens \(tokens)\n"
                                            .utf8))
                            }
                        }
                    }
                    await group.waitForAll()
                }
                drained.signal()
            }
            // Deadline so a wedged warm can never wedge worker startup.
            if drained.wait(timeout: .now() + 120) == .timedOut {
                consumer.cancel()
                for slot in 0..<batch {
                    engine.cancel(CBv2RequestID(UInt64(slot)))
                }
            }
            let stopped = DispatchSemaphore(value: 0)
            Task {
                // Fully retire the warm engine before the worker continues:
                // the warm is unscored, so the fast-ack path buys nothing
                // here and a lingering drain would overlap real work.
                await engine.shutdownSynchronously()
                stopped.signal()
            }
            _ = stopped.wait(timeout: .now() + 30)
            if CBv2StepProfiler.enabled {
                // Local diagnostics only (armed by CBV2_STEP_PROFILE): the
                // worker's stdout is the protocol channel, so the table goes
                // to stderr like the startup memory-profile announcement.
                FileHandle.standardError.write(
                    Data(
                        ("[warmCohortShapes] step profile over "
                            + "\(warmCompletionTokens) warm tokens\n"
                            + CBv2StepProfiler.summaryTable() + "\n").utf8))
            }
        } catch {
            // Best-effort: a cohort-warm refusal must never fail model load;
            // the measured run then simply pays the first-touch compiles it
            // pays today.
        }
    }
}
