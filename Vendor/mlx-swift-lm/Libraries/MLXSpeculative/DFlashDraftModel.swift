// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private let dflashDrafterFuseQKV: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_DFLASH_DRAFTER_FUSE_QKV"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    return true
}()

private let dflashDrafterFuseMLPGateUp: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_DFLASH_DRAFTER_FUSE_MLP_GATE_UP"]?.lowercased() {
    case "0", "false", "no", "off":
        return false
    default:
        return true
    }
}()

private let dflashCompiledSiluGateUp: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    silu(gate) * up
}

private final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    private var fusedGateUpProj: Linear?
    private var fusedGateUpUnavailable = false

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if dflashDrafterFuseMLPGateUp, let fused = fusedGateUp() {
            let parts = fused(x).split(parts: 2, axis: -1)
            return down(dflashCompiledSiluGateUp(parts[0], parts[1]))
        }

        return down(dflashCompiledSiluGateUp(gate(x), up(x)))
    }

    private func fusedGateUp() -> Linear? {
        if let fusedGateUpProj {
            return fusedGateUpProj
        }
        guard !fusedGateUpUnavailable,
            !(gate is QuantizedLinear),
            !(up is QuantizedLinear),
            gate.bias == nil,
            up.bias == nil,
            gate.weight.shape == up.weight.shape
        else {
            fusedGateUpUnavailable = true
            return nil
        }

        let fused = Linear(
            weight: concatenated([gate.weight, up.weight], axis: 0),
            bias: nil
        )
        fusedGateUpProj = fused
        return fused
    }
}

// PORT NOTE (engine, 2026-08-25). The fork's left-padding helpers
// (`dFlashDrafterLeftPadding` / `dFlashDrafterLeftPaddingMask`) and
// `makeBatchedCache(leftPadding:)` are NOT ported. They exist solely to
// serve `BatchKVCache` / `BatchRotatingKVCache`, the v1 batched KV stack
// upstream deleted at ffede00 and which this engine's vendored
// `MLXLMCommon` therefore does not carry (see docs/gemma4-port-notes.md).
// Every cache type this tree can hand the drafter — `StandardKVCache`,
// `RotatingKVCache` — reports no left padding, so the fork's helpers would
// return `nil` on every call here and the mask branches they gate are
// unreachable. Dropping them is behavior-preserving for the caches that
// exist; a future batched DFlash cohort has to bring a batched cache type
// with it and restore this pair alongside it.

private final class DFlashAttention: Module {
    let config: DFlashConfiguration
    let layerType: DFlashLayerType
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private var fusedProposalQKVProj: Linear?
    private var fusedContextKVProj: Linear?
    private var fusedProposalQKVUnavailable = false
    private var fusedContextKVUnavailable = false

    init(_ config: DFlashConfiguration, layerIdx: Int) {
        self.config = config
        self.layerType = config.layerTypes[layerIdx]
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) throws -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var context = context
        var contextLength = context.dim(1)

        if layerType == .slidingAttention {
            guard let slidingWindow = config.slidingWindow else {
                throw DFlashError.missingSlidingWindow
            }
            let keepContext = slidingWindow - 1
            if contextLength > keepContext {
                let skip = contextLength - keepContext
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                if let cache = cache as? BaseKVCache {
                    cache.offset += skip
                }
            }
        }

        let batchBaseOffset = (cache as? BatchPositionedKVCache)?.batchOffset
        let baseOffset = cache.offset

        let qDim = config.attentionHeads * config.headDim
        let kvDim = config.kvHeads * config.headDim

        var queries: MLXArray
        var proposalKeys: MLXArray
        var proposalValues: MLXArray
        if dflashDrafterFuseQKV, let fusedProposalQKV = fusedProposalQKV() {
            let parts = fusedProposalQKV(x).split(indices: [qDim, qDim + kvDim], axis: -1)
            queries = parts[0]
            proposalKeys = parts[1]
            proposalValues = parts[2]
        } else {
            queries = qProj(x)
            proposalKeys = kProj(x)
            proposalValues = vProj(x)
        }

        var contextKeys: MLXArray
        var contextValues: MLXArray
        if dflashDrafterFuseQKV, let fusedContextKV = fusedContextKV() {
            let parts = fusedContextKV(context).split(indices: [kvDim], axis: -1)
            contextKeys = parts[0]
            contextValues = parts[1]
        } else {
            contextKeys = kProj(context)
            contextValues = vProj(context)
        }

        queries =
            qNorm(queries.reshaped(B, L, config.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        contextKeys =
            kNorm(contextKeys.reshaped(B, contextLength, config.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        contextValues =
            contextValues.reshaped(B, contextLength, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)
        proposalKeys =
            kNorm(proposalKeys.reshaped(B, L, config.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        proposalValues =
            proposalValues.reshaped(B, L, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)

        if let batchBaseOffset {
            let proposalOffset = batchBaseOffset + Int32(contextLength)
            queries = rope(queries, offset: proposalOffset)
            contextKeys = rope(contextKeys, offset: batchBaseOffset)
            proposalKeys = rope(proposalKeys, offset: proposalOffset)
        } else {
            queries = rope(queries, offset: baseOffset + contextLength)
            contextKeys = rope(contextKeys, offset: baseOffset)
            proposalKeys = rope(proposalKeys, offset: baseOffset + contextLength)
        }

        let (cachedKeys, cachedValues) = cache.update(
            keys: contextKeys, values: contextValues)
        let cachedLength = cachedKeys.dim(2)
        let keys = concatenated([cachedKeys, proposalKeys], axis: 2)
        let values = concatenated([cachedValues, proposalValues], axis: 2)

        // No left padding is reachable in this tree — see the port note above
        // `DFlashDraftModel`'s cache helpers — so the fork's `hasLeftPadding`
        // branches collapse to their unpadded arms.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if layerType == .slidingAttention {
            let slidingWindow = config.slidingWindow!
            mask =
                cachedLength + L <= slidingWindow
                ? .causal
                : .array(
                    createCausalMask(
                        n: L,
                        offset: cachedLength,
                        windowSize: slidingWindow,
                        leftPadding: nil))
        } else {
            mask = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    private func fusedProposalQKV() -> Linear? {
        if let fusedProposalQKVProj {
            return fusedProposalQKVProj
        }
        guard !fusedProposalQKVUnavailable,
            !(qProj is QuantizedLinear),
            !(kProj is QuantizedLinear),
            !(vProj is QuantizedLinear),
            qProj.bias == nil,
            kProj.bias == nil,
            vProj.bias == nil,
            qProj.weight.shape.dropFirst() == kProj.weight.shape.dropFirst(),
            qProj.weight.shape.dropFirst() == vProj.weight.shape.dropFirst()
        else {
            fusedProposalQKVUnavailable = true
            return nil
        }

        let fused = Linear(
            weight: concatenated([qProj.weight, kProj.weight, vProj.weight], axis: 0),
            bias: nil
        )
        fusedProposalQKVProj = fused
        return fused
    }

    private func fusedContextKV() -> Linear? {
        if let fusedContextKVProj {
            return fusedContextKVProj
        }
        guard !fusedContextKVUnavailable,
            !(kProj is QuantizedLinear),
            !(vProj is QuantizedLinear),
            kProj.bias == nil,
            vProj.bias == nil,
            kProj.weight.shape.dropFirst() == vProj.weight.shape.dropFirst()
        else {
            fusedContextKVUnavailable = true
            return nil
        }

        let fused = Linear(
            weight: concatenated([kProj.weight, vProj.weight], axis: 0),
            bias: nil
        )
        fusedContextKVProj = fused
        return fused
    }
}

private final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DFlashConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = DFlashAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = DFlashMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) throws -> MLXArray {
        let h = try x + selfAttn(inputLayerNorm(x), context: context, rope: rope, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class DFlashDraftModel: Module, @unchecked Sendable {
    /// PARTICIPANT DRAFT-DEPTH LEVER (editable). The DFlash draft depth this
    /// submission runs. It is the number of speculative tokens the drafter
    /// proposes per round. The block that it emits is `depth + 1`. The extra
    /// column is a bonus token. This depth is FIXED for the run. Block diffusion
    /// drafts a whole block at once, so unlike MTP this depth does not adapt each
    /// round. `gemma4DFlashMaxDepth` clamps it to the drafter ceiling
    /// (`recommendedBlockSize - 1`) and the engine ceiling
    /// (`experimentalDFlashMaxBlockSize - 1`, which is 15). A value above a
    /// ceiling clamps to the ceiling. The engine does not refuse it. Default 1.
    ///
    /// UNIFORM WITH MTP. `CBv2MTPRoundDriver.submissionDraftDepth` is the MTP
    /// counterpart. It has the same name, the same meaning, and the same default
    /// 1. The per-arm behaviour differs. MTP adapts up to its ceiling each round.
    /// DFlash proposes a fixed block of this size. The constant you edit is the
    /// same on both arms.
    public static let submissionDraftDepth = 1

    public let config: DFlashConfiguration

    @ModuleInfo(key: "fc") public var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") public var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") private var layers: [DFlashDecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    private let rope: RoPELayer
    private var targetEmbed: ((MLXArray) -> MLXArray)?
    private var targetLMHead: ((MLXArray) -> MLXArray)?
    private var boundTargetID: ObjectIdentifier?

    public init(config: DFlashConfiguration) {
        self.config = config
        _contextProjection.wrappedValue = Linear(
            config.targetHiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlashDecoderLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    public func bind(target: any DFlashTargetModel) throws {
        let newID = ObjectIdentifier(target as AnyObject)
        if let existing = boundTargetID {
            if existing == newID { return }
            throw DFlashError.rebindForbidden
        }

        try validateCompatibility(with: target)
        self.targetEmbed = { [target] tokens in
            target.embedTokensForDFlash(tokens)
        }
        self.targetLMHead = { [target] hidden in
            target.logitsForDFlashHidden(hidden)
        }
        self.boundTargetID = newID
    }

    public func unbind() {
        self.targetEmbed = nil
        self.targetLMHead = nil
        self.boundTargetID = nil
    }

    public func makeCache() throws -> [KVCache] {
        var caches = [KVCache]()
        caches.reserveCapacity(config.layerTypes.count)
        for layerType in config.layerTypes {
            switch layerType {
            case .fullAttention:
                caches.append(StandardKVCache())
            case .slidingAttention:
                guard let slidingWindow = config.slidingWindow else {
                    throw DFlashError.missingSlidingWindow
                }
                caches.append(RotatingKVCache(maxSize: slidingWindow - 1, keep: 0))
            }
        }
        return caches
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int = 0
    ) throws -> MLXArray {
        guard let targetEmbed, let targetLMHead else {
            throw DFlashError.drafterNotBound
        }
        guard cache.count == layers.count else {
            throw DFlashError.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard logitsStart >= 0 else {
            throw DFlashError.invalidLogitsStart(logitsStart)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlashError.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize,
                actual: targetHidden.dim(-1)
            )
        }

        var h = targetEmbed(inputs)
        let context = hiddenNorm(contextProjection(targetHidden))

        for (i, layer) in layers.enumerated() {
            h = try layer(h, context: context, rope: rope, cache: cache[i])
        }

        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }

        var logits = targetLMHead(norm(h))
        if let cap = config.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    public func draftBlock(
        bonus: Int,
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int
    ) throws -> MLXArray {
        guard blockSize >= 2 else {
            throw DFlashError.invalidBlockSize(blockSize)
        }
        let maskValues = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let block = MLXArray([Int32(bonus)] + maskValues)[.newAxis, .ellipsis]
        let logits = try self(
            block,
            targetHidden: targetHidden,
            cache: cache,
            logitsStart: 1
        )
        return logits.argMax(axis: -1)
    }

    public func draftBlock(
        bonus: [Int],
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int
    ) throws -> MLXArray {
        guard blockSize >= 2 else {
            throw DFlashError.invalidBlockSize(blockSize)
        }
        let batchSize = bonus.count
        let maskValues = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let rows = bonus.flatMap { rowBonus in
            [Int32(rowBonus)] + maskValues
        }
        let block = MLXArray(rows, [batchSize, blockSize])
        let logits = try self(
            block,
            targetHidden: targetHidden,
            cache: cache,
            logitsStart: 1
        )
        return logits.argMax(axis: -1)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.hasPrefix("embed_tokens.") && !key.hasPrefix("lm_head.")
        }
    }

    public static func loadConfiguration(from directory: URL) throws -> DFlashConfiguration {
        try DFlashConfigurationDocument.read(from: directory).config
    }

    public static func load(from directory: URL) async throws -> DFlashDraftModel {
        try await load(from: directory, bindTo: nil)
    }

    public static func load(
        from directory: URL,
        bindTo target: (any DFlashTargetModel)?
    ) async throws -> DFlashDraftModel {
        let document = try DFlashConfigurationDocument.read(from: directory)

        // THE STRUCTURAL REFUSALS COME FIRST, BEFORE ANY MLX WORK.
        // `loadWeights` refuses a directory with no `config.json`, one that
        // cannot be enumerated, one with no `*.safetensors` at all, and a
        // duplicate tensor key. Constructing `DFlashDraftModel` allocates MLX
        // arrays and therefore initializes an MLX stream, and on a machine with
        // no usable Metal library that initialization ABORTS THE PROCESS rather
        // than throwing -- so a checkpoint that was never going to load used to
        // take the whole process down instead of returning its own named error.
        // Reading the weights first makes the refusal reachable everywhere,
        // which is better behaviour for a participant pointing the loader at a
        // half-staged directory as well as for a test host without a metallib.
        //
        // Nothing downstream depends on the old order: `sanitize` needs the
        // drafter, and it still runs after both.
        let weights = try loadWeights(from: directory)

        let drafter = DFlashDraftModel(config: document.config)
        if let target {
            try drafter.bind(target: target)
        }

        let sanitized = drafter.sanitize(weights: weights)
        try applyDeclaredQuantization(
            document.quantization, to: drafter, weights: sanitized)
        let params = ModuleParameters.unflattened(sanitized)
        try drafter.update(parameters: params, verify: [.all])
        eval(drafter)
        return drafter
    }

    /// Quantize the drafter's modules to match a quantized checkpoint, BEFORE
    /// its weights are bound.
    ///
    /// This is the DFlash half of the participant contract's head-requant
    /// promise (docs/participant-contract.md §3.4) and it mirrors the MTP head
    /// loader's flow (`Gemma4AssistantDraftModel.load(from:)` in
    /// Libraries/MLXLLM/Models/Gemma4MTP.swift): a module is quantized iff the
    /// checkpoint carries its `.scales` tensor, the geometry comes from the
    /// head's OWN `config.json`, and the quantize step runs against the
    /// SANITIZED weights so the borrowed `embed_tokens.`/`lm_head.` tensors a
    /// DFlash drafter never owns cannot decide anything here.
    ///
    /// Both mismatch directions fail closed, because either one would make the
    /// worker report a drafter it did not run: quantized weights with no
    /// declaration cannot bind at full precision, and a declaration whose
    /// checkpoint carries no packed tensor cannot quietly load as fp16.
    private static func applyDeclaredQuantization(
        _ quantization: BaseConfiguration.PerLayerQuantization?,
        to drafter: DFlashDraftModel,
        weights: [String: MLXArray]
    ) throws {
        let scalesSuffix = ".scales"
        let quantizedPaths = Set(
            weights.keys.compactMap { key in
                key.hasSuffix(scalesSuffix)
                    ? String(key.dropLast(scalesSuffix.count))
                    : nil
            })

        guard let quantization else {
            if let path = quantizedPaths.sorted().first {
                throw DFlashError.quantizedWeightsWithoutDeclaration(path)
            }
            return
        }
        guard !quantizedPaths.isEmpty else {
            throw DFlashError.declaredQuantizationWithoutQuantizedWeights
        }

        quantize(model: drafter) { path, _ in
            quantizedPaths.contains(path)
                ? quantization.quantization(layer: path)?.asTuple
                : nil
        }
    }

    public static func load(
        from downloader: any Downloader,
        id: String,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DFlashDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory)
    }

    public static func load(
        from downloader: any Downloader,
        id: String,
        bindTo target: any DFlashTargetModel,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DFlashDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory, bindTo: target)
    }

    private static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashError.missingConfig(directory.path)
        }

        var weights = [String: MLXArray]()
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        else {
            throw DFlashError.unreadableDirectory(directory.path)
        }

        let urls = enumerator.allObjects
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }
        guard !urls.isEmpty else {
            throw DFlashError.noSafetensorsFound(directory.path)
        }

        for url in urls where url.pathExtension == "safetensors" {
            let (shardWeights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shardWeights {
                guard weights[key] == nil else {
                    throw DFlashError.duplicateWeightKey(key)
                }
                weights[key] = value
            }
        }

        return weights
    }

    private func validateCompatibility(with target: any DFlashTargetModel) throws {
        if config.hiddenSize != target.dFlashHiddenSize {
            throw DFlashError.incompatibleDrafter(
                field: "hiddenSize",
                drafter: "\(config.hiddenSize)",
                target: "\(target.dFlashHiddenSize)"
            )
        }
        if config.vocabularySize != target.dFlashVocabularySize {
            throw DFlashError.incompatibleDrafter(
                field: "vocabSize",
                drafter: "\(config.vocabularySize)",
                target: "\(target.dFlashVocabularySize)"
            )
        }
        if config.numTargetLayers != target.dFlashLayerCount {
            throw DFlashError.incompatibleDrafter(
                field: "numTargetLayers",
                drafter: "\(config.numTargetLayers)",
                target: "\(target.dFlashLayerCount)"
            )
        }
        try DFlashTargetValidation.validateTargetLayerIds(
            config.targetLayerIds, layerCount: target.dFlashLayerCount)
    }
}
