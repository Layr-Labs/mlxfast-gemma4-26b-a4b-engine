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

/// Keep an MTP rectangle on the same eight-row QMV geometry as serial decode.
/// The leading gap makes MLX treat draft positions as matrix batches instead
/// of flattening them into one 16/24/32-row QMM.
private func gemma4B8PositionBatches(_ x: MLXArray) -> MLXArray? {
    guard x.ndim == 3, x.dim(0) == 8,
        (2...4).contains(x.dim(1))
    else { return nil }

    let positions = x.dim(1)
    let inputDimensions = x.dim(2)
    let paddedPositionMajor = padded(
        x.swappedAxes(0, 1), widths: [0, [0, 1], 0])
    return asStrided(
        paddedPositionMajor,
        [positions, 8, inputDimensions],
        strides: [9 * inputDimensions, inputDimensions, 1])
}

private func gemma4B8RectangularQuantizedMM(
    _ x: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray? {
    guard let positionBatches = gemma4B8PositionBatches(x) else { return nil }
    return quantizedMM(
        positionBatches,
        weight,
        scales: scales,
        biases: biases,
        transpose: true,
        groupSize: groupSize,
        bits: bits,
        mode: mode
    ).swappedAxes(0, 1)
}

private let gemma4B8QuantizedArgmaxPartials = MLXFast.metalKernel(
    name: "gemma4_b8_q4_argmax_partials_v1",
    inputNames: ["hidden", "weight", "scales", "biases"],
    outputNames: ["partial_scores", "partial_ids"],
    source: """
        constexpr uint VALUES_PER_THREAD = 8;
        constexpr uint K_BLOCK = VALUES_PER_THREAD * 32;
        constexpr uint STREAMS_PER_GROUP = 4;
        constexpr uint OUTPUTS_PER_STEP = 4;

        const uint lane = thread_position_in_threadgroup.x;
        const uint vocab_group = threadgroup_position_in_grid.x;
        const uint stream_group = threadgroup_position_in_grid.y;
        const uint stream_base = stream_group * STREAMS_PER_GROUP;
        const uint output_base = vocab_group * TILE;

        float best_scores[STREAMS_PER_GROUP];
        uint best_ids[STREAMS_PER_GROUP];
        for (uint stream = 0; stream < STREAMS_PER_GROUP; ++stream) {
            best_scores[stream] = -INFINITY;
            best_ids[stream] = 0xffffffffu;
        }

        for (uint output_offset = 0; output_offset < TILE;
             output_offset += OUTPUTS_PER_STEP) {
            float result[OUTPUTS_PER_STEP][STREAMS_PER_GROUP];
            for (uint row = 0; row < OUTPUTS_PER_STEP; ++row) {
                for (uint stream = 0; stream < STREAMS_PER_GROUP; ++stream) {
                    result[row][stream] = 0.0f;
                }
            }

            for (uint k = 0; k < K; k += K_BLOCK) {
                uint packed[OUTPUTS_PER_STEP];
                float local_scales[OUTPUTS_PER_STEP];
                float local_biases[OUTPUTS_PER_STEP];
                for (uint row = 0; row < OUTPUTS_PER_STEP; ++row) {
                    const uint output_row = output_base + output_offset + row;
                    packed[row] = weight[
                        output_row * (K / 8) + (k / 8) + lane];
                    const uint group_index = output_row * GROUPS
                        + (k / 64) + (lane / 8);
                    local_scales[row] = float(scales[group_index]);
                    local_biases[row] = float(biases[group_index]);
                }

                for (uint stream = 0; stream < STREAMS_PER_GROUP; ++stream) {
                    const uint hidden_base = (stream_base + stream) * K
                        + k + lane * VALUES_PER_THREAD;
                    float x[VALUES_PER_THREAD];
                    float sum = 0.0f;
                    for (uint i = 0; i < VALUES_PER_THREAD; i += 4) {
                        const float x0 = float(hidden[hidden_base + i]);
                        const float x1 = float(hidden[hidden_base + i + 1]);
                        const float x2 = float(hidden[hidden_base + i + 2]);
                        const float x3 = float(hidden[hidden_base + i + 3]);
                        sum += x0 + x1 + x2 + x3;
                        x[i] = x0;
                        x[i + 1] = x1 / 16.0f;
                        x[i + 2] = x2 / 256.0f;
                        x[i + 3] = x3 / 4096.0f;
                    }

                    for (uint row = 0; row < OUTPUTS_PER_STEP; ++row) {
                        const uint low = packed[row] & 0xffffu;
                        const uint high = packed[row] >> 16;
                        float accum =
                            x[0] * float(low & 0x000fu)
                            + x[1] * float(low & 0x00f0u)
                            + x[2] * float(low & 0x0f00u)
                            + x[3] * float(low & 0xf000u);
                        accum +=
                            x[4] * float(high & 0x000fu)
                            + x[5] * float(high & 0x00f0u)
                            + x[6] * float(high & 0x0f00u)
                            + x[7] * float(high & 0xf000u);
                        result[row][stream] +=
                            local_scales[row] * accum + sum * local_biases[row];
                    }
                }
            }

            for (uint row = 0; row < OUTPUTS_PER_STEP; ++row) {
                const uint token = output_base + output_offset + row;
                for (uint stream = 0; stream < STREAMS_PER_GROUP; ++stream) {
                    const float reduced = simd_sum(result[row][stream]);
                    if (lane == 0) {
                        const float score = float(static_cast<T>(reduced));
                        if (score > best_scores[stream]
                            || (score == best_scores[stream]
                                && token < best_ids[stream])) {
                            best_scores[stream] = score;
                            best_ids[stream] = token;
                        }
                    }
                }
            }
        }

        if (lane == 0) {
            const uint partial_base =
                (stream_group * VOCAB_GROUPS + vocab_group)
                * STREAMS_PER_GROUP;
            for (uint stream = 0; stream < STREAMS_PER_GROUP; ++stream) {
                partial_scores[partial_base + stream] = best_scores[stream];
                partial_ids[partial_base + stream] = best_ids[stream];
            }
        }
    """,
    ensureRowContiguous: true
)

private let gemma4B8QuantizedArgmaxReduce = MLXFast.metalKernel(
    name: "gemma4_b8_q4_argmax_reduce_v1",
    inputNames: ["partial_scores", "partial_ids"],
    outputNames: ["token_ids"],
    source: """
        const uint lane = thread_position_in_threadgroup.x;
        const uint stream = threadgroup_position_in_grid.x;
        const uint stream_group = stream / STREAMS_PER_GROUP;
        const uint stream_lane = stream % STREAMS_PER_GROUP;

        float best_score = -INFINITY;
        uint best_id = 0xffffffffu;
        for (uint group = lane; group < VOCAB_GROUPS; group += 256) {
            const uint index =
                (stream_group * VOCAB_GROUPS + group) * STREAMS_PER_GROUP
                + stream_lane;
            const float score = partial_scores[index];
            const uint token = partial_ids[index];
            if (score > best_score || (score == best_score && token < best_id)) {
                best_score = score;
                best_id = token;
            }
        }

        threadgroup float local_scores[256];
        threadgroup uint local_ids[256];
        local_scores[lane] = best_score;
        local_ids[lane] = best_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                const float other_score = local_scores[lane + stride];
                const uint other_id = local_ids[lane + stride];
                if (other_score > local_scores[lane]
                    || (other_score == local_scores[lane]
                        && other_id < local_ids[lane])) {
                    local_scores[lane] = other_score;
                    local_ids[lane] = other_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            token_ids[stream] = local_ids[0];
        }
    """,
    ensureRowContiguous: true
)

private final class Gemma4B8QuantizedLinear: QuantizedLinear,
    Gemma4B8PreparedQuantizedLinear
{
    init(
        _ linear: Linear,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        super.init(
            weight: linear.weight,
            bias: linear.bias,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let positionBatches = gemma4B8PositionBatches(x)
        else { return super.callAsFunction(x) }
        return gemma4ProjectPreparedB8(positionBatches)
    }

    func gemma4ProjectPreparedB8(_ positionBatches: MLXArray) -> MLXArray {
        var result = quantizedMM(
            positionBatches,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        ).swappedAxes(0, 1)
        if let bias {
            result = result + bias
        }
        return result
    }
}

private final class Gemma4B8QuantizedEmbedding: QuantizedEmbedding,
    Gemma4B8QuantizedArgmaxHead
{
    init(
        _ embedding: Embedding,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        super.init(
            weight: embedding.weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        gemma4B8RectangularQuantizedMM(
            x,
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode) ?? super.asLinear(x)
    }

    func gemma4ArgmaxB8(_ hidden: MLXArray) -> MLXArray? {
        guard hidden.ndim == 3, hidden.dim(0) == 8,
            (2...4).contains(hidden.dim(1)),
            hidden.dtype == .bfloat16,
            groupSize == 64, bits == 4, mode == .affine,
            let biases
        else { return nil }

        let streams = hidden.dim(0) * hidden.dim(1)
        let dimensions = hidden.dim(2)
        let vocabulary = shape.0
        let tile = 64
        guard dimensions == shape.1, vocabulary.isMultiple(of: tile) else {
            return nil
        }

        let vocabularyGroups = vocabulary / tile
        let streamGroups = streams / 4
        let partials = gemma4B8QuantizedArgmaxPartials(
            [hidden, weight, scales, biases],
            template: [
                ("T", hidden.dtype),
                ("K", dimensions),
                ("GROUPS", dimensions / groupSize),
                ("TILE", tile),
                ("VOCAB_GROUPS", vocabularyGroups),
            ],
            grid: (vocabularyGroups * 32, streamGroups, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [streamGroups, vocabularyGroups, 4],
                [streamGroups, vocabularyGroups, 4],
            ],
            outputDTypes: [.float32, .uint32]
        )
        let ids = gemma4B8QuantizedArgmaxReduce(
            partials,
            template: [
                ("VOCAB_GROUPS", vocabularyGroups),
                ("STREAMS_PER_GROUP", 4),
            ],
            grid: (streams * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[streams]],
            outputDTypes: [.uint32]
        )[0]
        return ids.reshaped(hidden.dim(0), hidden.dim(1)).asType(.int32)
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
        quantize(
            model: model,
            filter: { path, _ in
                guard sanitized["\(path).scales"] != nil else { return nil }
                let checkpointPath = prefix + path
                let spec = config.quantization.spec(forPath: checkpointPath)
                if config.quantization.overrides[checkpointPath] != nil {
                    appliedOverrides += 1
                }
                return (
                    groupSize: spec.groupSize, bits: spec.bits, mode: mode
                )
            },
            apply: { module, groupSize, bits, mode in
                if let embedding = module as? Embedding {
                    return Gemma4B8QuantizedEmbedding(
                        embedding,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode)
                }
                if let linear = module as? Linear {
                    return Gemma4B8QuantizedLinear(
                        linear,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode)
                }
                return quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode)
            })

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
