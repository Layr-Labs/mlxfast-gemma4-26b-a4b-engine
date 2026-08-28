// Copyright  2026 Apple Inc.

// Gemma 4 Multi-Token Prediction (MTP) speculative decoding.
// Single-file consolidation of drafter, sampling, round-loop, and token iterator support.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

// MARK: - Errors

/// Errors thrown by the Gemma 4 MTP drafter pipeline.
public enum Gemma4MTPError: LocalizedError, Sendable, Equatable {
    /// An assistant config is malformed or outside the supported, bounded
    /// geometry. `field` and `reason` are stable diagnostics for callers.
    case invalidConfiguration(field: String, reason: String)

    /// The target passed to `generateGemma4MTP` wasn't a `Gemma4TextModel`.
    /// Associated value is the actual type name.
    case unsupportedTarget(String)

    /// `bind(target:)` was called a second time with a different target.
    /// The drafter's weights encode assumptions about one specific target;
    /// rebinding to a different target is not supported.
    case rebindForbidden

    /// A drafter/target compatibility check failed at bind time. `field`
    /// names the mismatched field; `drafter` and `target` are the observed
    /// values (stringified).
    case incompatibleDrafter(field: String, drafter: String, target: String)

    /// `generateGemma4MTP` was called with a `blockSize` outside the
    /// supported range (typically 2-16).
    case invalidBlockSize(Int)

    /// A drafter forward was attempted before `bind(target:)`.
    case drafterNotBound

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let field, let reason):
            return "Invalid Gemma4 assistant configuration at \(field): \(reason)."
        case .unsupportedTarget(let typeName):
            return "Gemma4 MTP requires a Gemma4TextModel target; got \(typeName)."
        case .rebindForbidden:
            return
                "Gemma4AssistantDraftModel cannot be rebound to a different target. "
                + "Construct a new drafter instead."
        case .incompatibleDrafter(let field, let drafter, let target):
            return
                "Drafter/target mismatch on \(field): drafter=\(drafter), target=\(target)."
        case .invalidBlockSize(let n):
            return "Invalid blockSize \(n): must be between 2 and 16 inclusive."
        case .drafterNotBound:
            return
                "Gemma4AssistantDraftModel.bind(target:) must be called before "
                + "any forward pass."
        }
    }
}

// MARK: - Shared KV Types

/// Public namespace for Gemma 4 types that need cross-module visibility.
public enum Gemma4 {

    /// Position offset for RoPE, either a single scalar (standard decode)
    /// or a per-row `MLXArray` (continuous-batching / drafter paths).
    ///
    /// `@unchecked Sendable` because `MLXArray` is not `Sendable`; callers must
    /// treat these values as immutable snapshots (they are only ever read, never
    /// mutated in place).
    public enum PositionOffset: @unchecked Sendable {
        case scalar(Int)
        case batch(MLXArray)
        /// Graph-tracked offset from `CompilableKVCache.offsetArray`.
        /// Must stay as an `MLXArray` so `compile()` can trace through
        /// the RoPE computation without forcing a host readback.
        case graphArray(MLXArray)
    }
}

/// Immutable snapshot of per-layer-type K/V captured from a target
/// `Gemma4TextModel` during `forwardForMTP`. Consumed by the Gemma 4 MTP
/// drafter; every drafter layer reads from one of these two slots rather
/// than projecting its own K/V.
public struct Gemma4SharedKV: @unchecked Sendable {
    /// K/V from the target's last non-shared full-attention layer.
    /// Shape: `[B, nGlobalKVHeads, T, globalHeadDim]`.
    public let fullAttention: (MLXArray, MLXArray)
    /// K/V from the target's last non-shared sliding-attention layer.
    /// Shape: `[B, nKVHeads, T, headDim]`.
    public let slidingAttention: (MLXArray, MLXArray)

    public init(
        fullAttention: (MLXArray, MLXArray),
        slidingAttention: (MLXArray, MLXArray)
    ) {
        self.fullAttention = fullAttention
        self.slidingAttention = slidingAttention
    }

    /// Trim the tail of both K/V tensors by `rejected` time positions. Used
    /// by the MTP round-loop to match the post-rollback target cache length.
    /// If `rejected >= T`, clamps to a 1-slot tail so the drafter always has
    /// at least one K/V position to attend to.
    public static func sliceTail(
        from shared: Gemma4SharedKV, rejected: Int
    ) -> Gemma4SharedKV {
        func slice(_ kv: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
            let T = kv.0.dim(2)
            let valid = max(1, T - max(0, rejected))
            if valid >= T { return kv }
            let k = kv.0[.ellipsis, ..<valid, 0...]
            let v = kv.1[.ellipsis, ..<valid, 0...]
            return (k, v)
        }
        return Gemma4SharedKV(
            fullAttention: slice(shared.fullAttention),
            slidingAttention: slice(shared.slidingAttention)
        )
    }

    /// Zero per-row tail positions of the shared K/V. For each row `b`,
    /// slots `[keepLengths[b], T)` in both K and V tensors (for both
    /// layer types) are set to 0.
    ///
    /// Used by the B>1 MTP round loop to match the per-row target cache
    /// zeroing performed by `rollbackSpeculativeCache(.perRow:)`. Rows
    /// that accepted less than max have their tail K/V zeroed so the
    /// drafter doesn't attend to post-rollback "stale" positions.
    ///
    /// This differs from `sliceTail` which uniformly trims the T axis.
    /// `zeroTailPerRow` preserves the T axis length (so all rows share
    /// the same tensor shape) but zeros the invalid slots - the
    /// SDPA/attention computation sees zeros, which contribute nothing
    /// to the softmax when multiplied by queries.
    ///
    /// - Parameter keepLengths: int array of shape `[B]`. For row `b`,
    ///   slots `[0, keepLengths[b])` are preserved; slots
    ///   `[keepLengths[b], T)` are zeroed.
    public static func zeroTailPerRow(
        from shared: Gemma4SharedKV, keepLengths: MLXArray
    ) -> Gemma4SharedKV {
        func zero(_ kv: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
            let T = kv.0.dim(2)
            let positions = MLXArray(Int32(0) ..< Int32(T))
                .reshaped([1, 1, T, 1])                        // [1, 1, T, 1]
            let keep = keepLengths.asType(.int32)
                .reshaped([-1, 1, 1, 1])                       // [B, 1, 1, 1]
            let keepMask = positions .< keep                   // [B, 1, T, 1]
            let maskK = keepMask.asType(kv.0.dtype)
            let maskV = keepMask.asType(kv.1.dtype)
            return (kv.0 * maskK, kv.1 * maskV)
        }
        return Gemma4SharedKV(
            fullAttention: zero(shared.fullAttention),
            slidingAttention: zero(shared.slidingAttention)
        )
    }
}

/// Mutable sink for the shared-KV capture hook in
/// `Gemma4TextModelInner.callAsFunction`. A reference type so the trunk can
/// write into it without a return-value contortion; cleared by the caller
/// between forwards.
public final class Gemma4SharedKVCapture: @unchecked Sendable {
    public var fullAttention: (MLXArray, MLXArray)? = nil
    public var slidingAttention: (MLXArray, MLXArray)? = nil

    public init() {}

    /// Snapshot into an immutable `Gemma4SharedKV`. Throws via
    /// `fatalError` if either slot is missing - the capture hook is
    /// expected to populate both.
    public func snapshot() -> Gemma4SharedKV {
        guard let full = fullAttention else {
            fatalError("Gemma4SharedKVCapture: fullAttention was not populated")
        }
        guard let sliding = slidingAttention else {
            fatalError("Gemma4SharedKVCapture: slidingAttention was not populated")
        }
        return Gemma4SharedKV(fullAttention: full, slidingAttention: sliding)
    }
}

/// Result of `Gemma4TextModel.forwardForMTP`. Carries both the LM head
/// output and the pre-head trunk hidden, plus the shared-KV snapshot the
/// drafter needs for its next round.
public struct Gemma4MTPForward: @unchecked Sendable {
    /// `[B, L, vocab]` - softcap applied.
    public let logits: MLXArray
    /// `[B, L, hidden_size]` - last decoder-layer output BEFORE the final
    /// `model.norm`. This matches HF's `hidden_states` capture point and is
    /// what the MTP drafter's `pre_projection` was trained against.
    public let lastHidden: MLXArray
    /// Per-layer-type K/V from the last non-shared layers of the target.
    public let capturedSharedKV: Gemma4SharedKV

    public init(
        logits: MLXArray, lastHidden: MLXArray, capturedSharedKV: Gemma4SharedKV
    ) {
        self.logits = logits
        self.lastHidden = lastHidden
        self.capturedSharedKV = capturedSharedKV
    }
}

/// Count of accepted speculative tokens per round, either scalar (B=1)
/// or per-row (`[B]` int32 array, B>1).
public enum Gemma4AcceptCount: @unchecked Sendable {
    case scalar(Int)
    case perRow(MLXArray)

    /// Max accepted across all rows (== the scalar value in the scalar case).
    public func maxAccepted() -> Int {
        switch self {
        case .scalar(let n): return n
        case .perRow(let arr): return Int(arr.max().item(Int32.self))
        }
    }
}

// MARK: - Automatic Policy

/// Gemma 4 target family inferred from the loaded text-model config.
public enum Gemma4MTPModelFamily: Sendable, Equatable {
    case e2b
    case e4b
    case moeA4B
    case unknownDense
    case unknownMoE
}

/// Execution strategy for a requested MTP batch.
public enum Gemma4MTPBatchStrategy: Sendable, Equatable {
    /// Run each request as a single stream (`B=1`) using this block size.
    case singleStream(blockSize: Int)
    /// Run the requests through the B>1 round loop using this block size.
    case batched(blockSize: Int)

    public var blockSize: Int {
        switch self {
        case .singleStream(let blockSize), .batched(let blockSize):
            return blockSize
        }
    }

    public var usesBatchedRoundLoop: Bool {
        switch self {
        case .batched: return true
        case .singleStream: return false
        }
    }
}

/// Single-stream runtime action selected by the automatic MTP policy.
public enum Gemma4MTPSingleStreamAction: Sendable, Equatable {
    /// Run an MTP round with this block size.
    case mtp(blockSize: Int)
    /// Emit the next token from the target model only.
    case targetOnly

    public var blockSize: Int? {
        switch self {
        case .mtp(let blockSize): blockSize
        case .targetOnly: nil
        }
    }
}

/// Model-aware defaults for Gemma 4 MTP on Apple Silicon.
///
/// The policy is intentionally conservative: only model families with
/// measured B>1 wins opt into the batched round loop. Unknown dense and
/// MoE variants fall back to the B=1 path until benchmarked.
public struct Gemma4MTPAutomaticPolicy: Sendable, Equatable {
    public let family: Gemma4MTPModelFamily
    public let singleStreamBlockSize: Int
    public let batchedBlockSize: Int?

    public var supportsBatchedMTP: Bool {
        batchedBlockSize != nil
    }

    /// Whether the public automatic generation entry point should use MTP
    /// for this family when the caller does not force a block size.
    public var enablesAutomaticMTP: Bool {
        family != .e4b
    }

    public func strategy(forBatchSize batchSize: Int) -> Gemma4MTPBatchStrategy {
        precondition(batchSize >= 1, "batchSize must be at least 1")
        if batchSize == 1 {
            return .singleStream(blockSize: singleStreamBlockSize)
        }
        if let batchedBlockSize {
            return .batched(blockSize: batchedBlockSize)
        }
        return .singleStream(blockSize: singleStreamBlockSize)
    }

    public func singleStreamAction(
        generatedTokens: Int,
        maxTokens: Int?
    ) -> Gemma4MTPSingleStreamAction {
        precondition(generatedTokens >= 0, "generatedTokens must be non-negative")

        if !enablesAutomaticMTP {
            return .targetOnly
        }

        if family == .e2b && singleStreamBlockSize == 3 && batchedBlockSize == nil {
            if let maxTokens, maxTokens <= 16 {
                return .mtp(blockSize: 4)
            }
            if generatedTokens < 256 {
                return .targetOnly
            }
            return .mtp(blockSize: 4)
        }

        return .mtp(blockSize: singleStreamBlockSize)
    }

    public static func automatic(for target: Gemma4TextModel) -> Self {
        automatic(for: target.configuration)
    }

    /// Resolve the automatic policy for any Gemma 4 MTP target tower
    /// (text-only or VLM), using its resolved text configuration.
    public static func automatic(forTarget target: any Gemma4MTPTarget) -> Self {
        automatic(for: target.mtpConfiguration)
    }

    public static func automatic(for config: Gemma4TextConfiguration) -> Self {
        if config.enableMoeBlock {
            if config.hiddenSize == 2816 && config.numHiddenLayers == 30 {
                return Self(
                    family: .moeA4B,
                    singleStreamBlockSize: 3,
                    batchedBlockSize: nil)
            }
            return Self(
                family: .unknownMoE,
                singleStreamBlockSize: 3,
                batchedBlockSize: nil)
        }

        if config.hiddenSize == 1536 && config.numHiddenLayers == 35 {
            if config.quantizationBits == 4 {
                return Self(
                    family: .e2b,
                    singleStreamBlockSize: 3,
                    batchedBlockSize: nil)
            }
            return Self(
                family: .e2b,
                singleStreamBlockSize: 5,
                batchedBlockSize: 3)
        }

        if config.hiddenSize == 2560 && config.numHiddenLayers == 42 {
            return Self(
                family: .e4b,
                singleStreamBlockSize: 3,
                batchedBlockSize: nil)
        }

        return Self(
            family: .unknownDense,
            singleStreamBlockSize: 4,
            batchedBlockSize: nil)
    }
}

// MARK: - Gemma4TextModel MTP Extensions

extension Gemma4TextModel {

    /// Forward pass tailored for MTP speculative decoding.
    ///
    /// Returns both the LM head output (logits) and the pre-head trunk
    /// hidden, plus a snapshot of the shared-KV tensors consumed by the
    /// drafter on its next round.
    public func forwardForMTP(
        _ tokens: MLXArray, cache: [KVCache]
    ) -> Gemma4MTPForward {
        let capture = Gemma4SharedKVCapture()
        let fullIdx = model.lastFullAttentionNonSharedIdx
        let slidingIdx = model.lastSlidingAttentionNonSharedIdx
        let (postNorm, preNorm) = model.callCapturingPreNorm(
            tokens, cache: cache
        ) { idx, kv in
            if idx == fullIdx {
                capture.fullAttention = kv
            } else if idx == slidingIdx {
                capture.slidingAttention = kv
            }
        }
        let logits = applyLMHead(postNorm)
        return Gemma4MTPForward(
            logits: logits,
            lastHidden: preNorm,
            capturedSharedKV: capture.snapshot()
        )
    }

    /// Rewind target KV caches after a speculative-decoding round.
    ///
    /// Uniformly trims rejected suffix positions. In the `.perRow` case,
    /// batched caches are additionally zeroed per row so rows that accepted
    /// fewer than the batch max cannot attend to stale post-rollback K/V.
    public func rollbackSpeculativeCache(
        _ caches: [KVCache],
        accepted: Gemma4AcceptCount,
        blockSize: Int
    ) {
        let maxAccepted = accepted.maxAccepted()
        let trim = Swift.max(0, blockSize - maxAccepted - 1)

        for cache in caches {
            guard cache.isTrimmable else { continue }
            if trim > 0 {
                _ = cache.trim(trim)
            }
        }

        // Per-row tail zeroing died with the legacy batched caches
        // (`BatchKVCache`, v0.7.5 one-engine). The uniform batch-max trim
        // above remains exact for B=1; per-row ragged rollback returns with
        // the CBv2-MTP integration, engine-side over CBv2 sequence KV.
        _ = accepted
    }
}

// MARK: - Drafter Masks

public enum Gemma4DrafterMaskBuilder {

    /// Full-attention mask: always `.none`. SDPA handles bidirectional
    /// attention without an explicit mask.
    public static func bidirectionalFull(
        queryLen: Int, kvLen: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        _ = (queryLen, kvLen, dtype)
        return .none
    }

    /// Bidirectional sliding-window mask.
    ///
    /// For each query position `q in [queryOffset, queryOffset + queryLen)`,
    /// allow attention to KV positions `k in (q - window, q + window)`.
    /// Returns `.none` when the whole KV fits inside every query's window.
    /// Otherwise returns a materialized `.array(...)` additive mask with
    /// `-inf` outside the window and `0` inside.
    public static func bidirectionalSWA(
        queryLen: Int, queryOffset: Int, kvLen: Int,
        window: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // No masking needed: the entire KV fits in the bidirectional window
        // of every query.
        if kvLen <= window && queryOffset + queryLen <= kvLen + window {
            return .none
        }

        let qIdx = MLXArray(Int32(queryOffset) ..< Int32(queryOffset + queryLen))
            .reshaped([queryLen, 1])
        let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
            .reshaped([1, kvLen])
        let dist = qIdx - kIdx
        let inside = MLX.logicalAnd(
            dist .> Int32(-window),
            dist .< Int32(window))

        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        let bias = MLX.where(inside, zero, negInf)
        // SDPA additive mask shape: broadcastable to (B, H, L, S). Add
        // leading singleton axes for batch and head.
        return .array(bias.reshaped([1, 1, queryLen, kvLen]))
    }
}

// MARK: - Masked Embedder

public final class Gemma4MaskedEmbedder: Module, @unchecked Sendable {
    @ModuleInfo(key: "centroids") public var centroids: Linear
    @ParameterInfo(key: "token_ordering") public var tokenOrdering: MLXArray

    public let hiddenSize: Int
    public let numCentroids: Int
    public let topK: Int
    public let vocabSize: Int
    public let vocabSizePerCentroid: Int

    public init(
        hiddenSize: Int,
        numCentroids: Int,
        topK: Int,
        vocabSize: Int
    ) {
        precondition(
            vocabSize % numCentroids == 0,
            "vocabSize must be divisible by numCentroids")
        self.hiddenSize = hiddenSize
        self.numCentroids = numCentroids
        self.topK = topK
        self.vocabSize = vocabSize
        self.vocabSizePerCentroid = vocabSize / numCentroids

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        // token_ordering arrives as int64 in some checkpoints; the drafter
        // sanitize step casts to int32. Default here is zeros.
        self._tokenOrdering.wrappedValue =
            MLXArray.zeros([vocabSize], type: Int32.self)
        super.init()
    }

    /// Compute sparse logits over the full vocab.
    ///
    /// - Parameters:
    ///   - hiddenStates: `[B, L, hidden]`.
    ///   - lmHeadWeight: `[vocabSize, hidden]` - typically tied to the
    ///     drafter's `embed_tokens.weight`.
    /// - Returns: `[B, L, vocabSize]` with non-selected positions set to
    ///   `min(selected) - 1`.
    public func callAsFunction(
        hiddenStates: MLXArray, lmHeadWeight: MLXArray
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        // Cluster scores -> top-K cluster indices.
        let centroidLogits = centroids(hiddenStates)  // [B, L, numCentroids]

        // Negate-and-partition-from-front idiom: indices of the top-K
        // scoring entries land at positions [0, topK). Matches the
        // pattern in Libraries/MLXLMCommon/Evaluate.swift:288.
        let topKIndices = argPartition(-centroidLogits, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK
        ]  // [B, L, topK]

        // Reshape token_ordering to [numCentroids, vocabSizePerCentroid].
        let ordering =
            tokenOrdering.reshaped([numCentroids, vocabSizePerCentroid])

        // Gather canonical token IDs for each selected cluster.
        // Result shape: [B, L, topK, vocabSizePerCentroid].
        let selectedCanonical = ordering[topKIndices]

        // Gather embedding rows for the selected tokens.
        // [B, L, topK * vocabSizePerCentroid, hidden]
        let flatIdx = selectedCanonical.reshaped([-1])
        let selectedEmb = lmHeadWeight[flatIdx].reshaped([
            B, L, topK * vocabSizePerCentroid, hiddenSize,
        ])

        // selected_logits = hidden @ selectedEmb.T
        // [B, L, 1, hidden] @ [B, L, hidden, topK * vsc] -> [B, L, 1, topK * vsc]
        let selectedLogits = matmul(
            hiddenStates.expandedDimensions(axis: -2),
            selectedEmb.swappedAxes(-1, -2)
        ).squeezed(axis: -2)  // [B, L, topK * vsc]

        // Keep the sentinel device-resident. A host scalar extraction here
        // would add one synchronization per draft token and violate CBv2's
        // single finalize boundary.
        let sentinelValue = selectedLogits.min() - 1.0

        // Full-vocab output tensor filled with the sentinel.
        let out = MLX.full(
            [B, L, vocabSize],
            values: sentinelValue,
            dtype: hiddenStates.dtype)

        // Scatter selected logits into their canonical positions.
        let scatterIdx = selectedCanonical.reshaped([
            B, L, topK * vocabSizePerCentroid,
        ])
        return putAlong(out, scatterIdx, values: selectedLogits, axis: -1)
    }

    // MARK: - Test support

    /// Set centroids weight + tokenOrdering for unit tests.
    /// Not part of the public surface.
    internal func setTestWeights(
        centroidsWeight: MLXArray, tokenOrdering: MLXArray
    ) {
        var params = ModuleParameters()
        params["centroids"] = .dictionary([
            "weight": .value(centroidsWeight)
        ])
        params["token_ordering"] = .value(tokenOrdering)
        self.update(parameters: params)
    }

    /// Set only the centroids weight.
    internal func setTestCentroidWeight(_ w: MLXArray) {
        var params = ModuleParameters()
        params["centroids"] = .dictionary([
            "weight": .value(w)
        ])
        self.update(parameters: params)
    }
}

// MARK: - Speculative Walk

/// Accept/reject walker for speculative decoding. Pure Swift - no MLX
/// dependency so it can be exhaustively unit-tested.
public enum Gemma4SpeculativeWalk {

    /// Single-row greedy accept-prefix walker.
    ///
    /// The speculative-decoding contract: `main` contains `k + 1` tokens
    /// (one verify per draft plus one "bonus" token that the target sampled
    /// from the position after the last draft). `draft` contains the `k`
    /// tokens the drafter proposed. Walk left-to-right, accept while
    /// `draft[i] == main[i]`, and always emit one "correction" or "bonus"
    /// token from `main` past the accepted prefix.
    ///
    /// - Returns: `(acceptedCount, emittedTokens)` where
    ///   `emittedTokens.count == acceptedCount + 1`.
    public static func single(draft: [Int], main: [Int]) -> (Int, [Int]) {
        // If draft is empty, just emit main[0] (the bonus).
        guard !draft.isEmpty else {
            precondition(!main.isEmpty, "main must contain at least the bonus")
            return (0, [main[0]])
        }
        precondition(
            main.count >= draft.count + 1,
            "main must have at least draft.count + 1 tokens (drafts + bonus)"
        )
        var accepted = 0
        for i in 0 ..< draft.count {
            if main[i] != draft[i] { break }
            accepted += 1
        }
        // Emit main[0...accepted] inclusive - that's accepted + 1 tokens.
        return (accepted, Array(main[0 ... accepted]))
    }

    /// Multi-row greedy accept-prefix walker with per-row emit budgets.
    ///
    /// - Parameters:
    ///   - draft: per-row draft tokens, length `[B][k]`.
    ///   - main: per-row verify tokens, length `[B][k+1]`.
    ///   - budgets: per-row max emit count. A row emits at most
    ///     `budgets[i]` tokens; the accept count is capped accordingly so
    ///     the `emitted.count == accepted + 1` invariant holds per row.
    /// - Returns: `(acceptedPerRow, emittedPerRow)`.
    public static func batched(
        draft: [[Int]], main: [[Int]], budgets: [Int]
    ) -> ([Int], [[Int]]) {
        precondition(
            draft.count == main.count && draft.count == budgets.count,
            "batched: all inputs must have the same outer length B"
        )
        var acceptedOut: [Int] = []
        var emittedOut: [[Int]] = []
        acceptedOut.reserveCapacity(draft.count)
        emittedOut.reserveCapacity(draft.count)
        for i in 0 ..< draft.count {
            var (a, e) = single(draft: draft[i], main: main[i])
            let budget = Swift.max(0, budgets[i])
            if e.count > budget {
                e = Array(e.prefix(budget))
                a = Swift.max(0, e.count - 1)
            }
            acceptedOut.append(a)
            emittedOut.append(e)
        }
        return (acceptedOut, emittedOut)
    }
}

// MARK: - Speculative Sampling

/// Result of one round of rejection-based speculative sampling.
public struct Gemma4SpeculativeSampleResult: Sendable {
    /// Number of drafter tokens accepted (0...K).
    public let accepted: Int
    /// Tokens emitted by this round: first `accepted` drafter tokens,
    /// followed by one corrective / bonus token. Always `accepted + 1`
    /// entries.
    public let emitted: [Int]

    public init(accepted: Int, emitted: [Int]) {
        self.accepted = accepted
        self.emitted = emitted
    }
}

/// Convert logits into the same sampling distribution used by
/// `GenerateParameters.sampler()`: compute log-probabilities, apply
/// topP/minP/topK filters, then apply temperature for the final categorical
/// distribution. The mask is applied in log-space before softmax so masked
/// positions land at 0 in `probs`.
///
/// IMPORTANT: The SAME filter parameters must be used for both drafter
/// and target in `gemma4SpeculativeSampleRound`. Applying topP/topK to only
/// one side would make the accept ratio `p(c)/q(c)` ill-defined (e.g.
/// if `c` is in p's nucleus but not q's, `q(c) = 0`).
private func softmaxWithFilters(
    _ logits: MLXArray,
    temperature: Float,
    topP: Float,
    topK: Int,
    minP: Float
) -> MLXArray {
    precondition(temperature > 0, "temperature must be > 0 for stochastic sampling")
    var logprobs = logSoftmax(logits.asType(.float32), axis: -1)
    // Order matches mlx_lm's TopPSampler: topP -> minP -> topK.
    if topP > 0 && topP < 1 {
        logprobs = applyTopP(logprobs, p: topP)
    }
    if minP > 0 {
        logprobs = applyMinP(logprobs, minP: minP)
    }
    if topK > 0 {
        logprobs = applyTopK(logprobs, k: topK)
    }
    return softmax(logprobs / temperature, axis: -1)
}

/// Public helper: sample one token from `logits` along the last axis
/// with temperature + optional topP / topK / minP filters. Returns the
/// sampled token index. Used by the MTP token iterator for the initial
/// bonus and for drafter draft steps so that the same filter masks are
/// applied everywhere along the generation path.
public func gemma4SampleLogitsWithFilters(
    _ logits: MLXArray,
    temperature: Float,
    topP: Float,
    topK: Int,
    minP: Float,
    rngKey: MLXArray
) -> MLXArray {
    let probs = softmaxWithFilters(
        logits, temperature: temperature,
        topP: topP, topK: topK, minP: minP)
    return sampleFromDistribution(probs, key: rngKey)
}

/// Mask all but top-K log-probabilities to -inf along the last axis.
private func applyTopK(_ logprobs: MLXArray, k: Int) -> MLXArray {
    let vocab = logprobs.shape.last ?? 0
    if k <= 0 || k >= vocab { return logprobs }
    let sortedIdx = argSort(-logprobs, axis: -1)
    let topIdx = sortedIdx[.ellipsis, 0 ..< k]
    let topVals = takeAlong(logprobs, topIdx, axis: -1)
    let threshold = topVals.min(axes: [-1], keepDims: true)
    let negInf = MLXArray(-Float.infinity)
    return MLX.where(logprobs .>= threshold, logprobs, negInf)
}

/// Top-P (nucleus): mask tokens outside the smallest set whose
/// cumulative softmax mass is at least `p`.
private func applyTopP(_ logprobs: MLXArray, p: Float) -> MLXArray {
    let sortedIdxAsc = argSort(logprobs, axis: -1)
    let sortedLogprobs = takeAlong(logprobs, sortedIdxAsc, axis: -1)
    let sortedProbs = exp(sortedLogprobs)
    let cumProbs = sortedProbs.cumsum(axis: -1)
    let keepMask = cumProbs .> MLXArray(1.0 - p)
    let negInf = MLXArray(-Float.infinity)
    let maskedSorted = MLX.where(keepMask, sortedLogprobs, negInf)
    let inverseIdx = argSort(sortedIdxAsc, axis: -1)
    return takeAlong(maskedSorted, inverseIdx, axis: -1)
}

/// Min-P: mask tokens whose probability is below `minP * max_prob`.
/// Computed in log-space: `logp >= max_logp + log(minP)`.
private func applyMinP(_ logprobs: MLXArray, minP: Float) -> MLXArray {
    let maxLogprob = logprobs.max(axis: -1, keepDims: true)
    let threshold = maxLogprob + MLXArray(log(minP))
    let negInf = MLXArray(-Float.infinity)
    return MLX.where(logprobs .>= threshold, logprobs, negInf)
}

/// Sample one token from a distribution `[vocab]` (or `[..., vocab]`)
/// along the last axis. Returns an `[...]` int32 array.
private func sampleFromDistribution(_ probs: MLXArray, key: MLXArray) -> MLXArray {
    // MLX has `MLXRandom.categorical(logits)` but we have probabilities.
    // Use cumulative sum + uniform random number comparison.
    let u = MLXRandom.uniform(low: Float(0), high: Float(1), key: key)
    let cdf = probs.cumsum(axis: -1)
    // count positions where cdf < u; that's the sampled index.
    let lt = cdf .< u
    return lt.sum(axis: -1).asType(.int32)
}

/// Run rejection-based speculative sampling for one round.
///
/// - Parameters:
///   - draftLogits: [K, vocab] - drafter logits at each of the K proposed
///     positions, *pre-temperature* (raw drafter output).
///   - draftTokens: [K] - tokens the drafter sampled.
///   - verifyLogits: [bs=K+1, vocab] - target logits at each verify
///     position, *pre-temperature* (raw target output).
///   - temperature: sampling temperature (> 0).
///   - topP: nucleus mass (1.0 = disabled). Applied identically to p and q.
///   - topK: top-K filter (0 = disabled). Applied identically to p and q.
///   - minP: min-P filter (0 = disabled). Applied identically to p and q.
///   - rngKey: MLX random key for the uniform draws.
/// - Returns: `(accepted, emitted)` - number of drafter tokens accepted
///   and the list of emitted tokens for this round.
public func gemma4SpeculativeSampleRound(
    draftLogits: MLXArray,
    draftTokens: [Int],
    verifyLogits: MLXArray,
    temperature: Float,
    topP: Float = 1.0,
    topK: Int = 0,
    minP: Float = 0.0,
    rngKey: MLXArray
) -> Gemma4SpeculativeSampleResult {
    precondition(temperature > 0, "stochastic path requires temperature > 0")
    let K = draftTokens.count
    precondition(verifyLogits.dim(0) == K + 1,
        "verifyLogits must have K+1 positions")
    precondition(draftLogits.dim(0) == K,
        "draftLogits must have K positions")

    // Apply temperature + identical filters on both p and q, once.
    let q = softmaxWithFilters(
        draftLogits, temperature: temperature,
        topP: topP, topK: topK, minP: minP)  // [K, vocab]
    let p = softmaxWithFilters(
        verifyLogits, temperature: temperature,
        topP: topP, topK: topK, minP: minP)  // [K+1, vocab]

    // Split the rng key into 2K subkeys (K for accept r_i, K for resample
    // + final sample).
    let keys = MLXRandom.split(key: rngKey, into: 2 * (K + 1))

    var accepted = 0
    var emitted: [Int] = []
    for i in 0 ..< K {
        let c = draftTokens[i]
        let pi = p[i]  // [vocab]
        let qi = q[i]  // [vocab]
        let pc = pi[c].item(Float.self)
        let qc = qi[c].item(Float.self)
        let alpha = min(Float(1), qc > 0 ? pc / qc : 0)
        let r = MLXRandom.uniform(
            low: Float(0), high: Float(1), key: keys[i]).item(Float.self)
        if r < alpha {
            emitted.append(c)
            accepted += 1
        } else {
            // Reject. Resample from normalize(max(pi - qi, 0)).
            let diff = MLX.maximum(pi - qi, MLXArray(Float(0)))
            let s = diff.sum().item(Float.self)
            let resampleDist: MLXArray
            if s > 0 {
                resampleDist = diff / s
            } else {
                // Numerical edge case: p == q exactly. Fall back to p
                // which in theory should sum to 1 already.
                resampleDist = pi
            }
            let nextTok = sampleFromDistribution(
                resampleDist, key: keys[K + i])
            emitted.append(Int(nextTok.item(Int32.self)))
            return Gemma4SpeculativeSampleResult(accepted: accepted, emitted: emitted)
        }
    }
    // All K accepted - sample one bonus from p_K.
    let bonus = sampleFromDistribution(p[K], key: keys[2 * K])
    emitted.append(Int(bonus.item(Int32.self)))
    return Gemma4SpeculativeSampleResult(accepted: accepted, emitted: emitted)
}

// MARK: - Assistant Configuration

/// Configuration for the Gemma 4 Multi-Token Prediction "assistant" drafter.
///
/// Mirrors the HF `Gemma4AssistantConfig` schema (flattened top-level +
/// nested `text_config`). Some fields are drafter-specific
/// (`backboneHiddenSize`, `useOrderedEmbeddings`, `numCentroids`,
/// `centroidIntermediateTopK`, `blockSize`); the nested `textConfig`
/// reuses the same `Gemma4TextConfiguration` that the target model uses,
/// with a post-init clamp that forces `numKvSharedLayers =
/// numHiddenLayers` when the checkpoint omits it (matching HF semantics
/// for drafter checkpoints).
public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public var modelType: String
    public var backboneHiddenSize: Int
    public var useOrderedEmbeddings: Bool
    public var numCentroids: Int
    public var centroidIntermediateTopK: Int
    public var blockSize: Int
    public var textConfig: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case blockSize = "block_size"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        do {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decode(String.self, forKey: .modelType)
            self.backboneHiddenSize = try container.decode(
                Int.self, forKey: .backboneHiddenSize)
            self.useOrderedEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
            self.numCentroids =
                try container.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
            self.centroidIntermediateTopK =
                try container.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
            // block_size isn't in the HF config file - it's a runtime hyperparam
            // passed into the round loop. Default 4 matches Google's published
            // recommendation.
            self.blockSize =
                try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 4

            let preflight = try container.decode(
                Gemma4AssistantTextConfigurationPreflight.self,
                forKey: .textConfig)
            try Gemma4AssistantConfigurationValidator.validatePreflight(preflight)
            var textConfig = try container.decode(
                Gemma4TextConfiguration.self, forKey: .textConfig)

            // HF post-init: drafter checkpoints require every layer to consume
            // the target's shared K/V. If the checkpoint omits num_kv_shared_layers
            // (decoder default 20, typically > num_hidden_layers=4 for drafters)
            // or sets it to 0, clamp it to num_hidden_layers so every drafter
            // layer is KV-shared.
            if textConfig.numKvSharedLayers == 0
                || textConfig.numKvSharedLayers > textConfig.numHiddenLayers
            {
                textConfig.numKvSharedLayers = textConfig.numHiddenLayers
            }
            self.textConfig = textConfig
            try Gemma4AssistantConfigurationValidator.validate(self)
        } catch let error as Gemma4MTPError {
            throw error
        } catch {
            throw Gemma4MTPError.invalidConfiguration(
                field: "config.json",
                reason: "could not be decoded")
        }
    }
}

// MARK: - Assistant Draft Model

internal struct Gemma4DrafterMasks {
    let full: MLXFast.ScaledDotProductAttentionMaskMode
    let sliding: MLXFast.ScaledDotProductAttentionMaskMode
}

public final class Gemma4AssistantDraftModel: Module, @unchecked Sendable {
    public let config: Gemma4AssistantConfiguration

    @ModuleInfo public var model: Gemma4TextModelInner
    @ModuleInfo(key: "pre_projection") public var preProjection: Linear
    @ModuleInfo(key: "post_projection") public var postProjection: Linear
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") public var maskedEmbedder: Gemma4MaskedEmbedder?

    // Set by bind(target:); drafter is not usable before bind().
    internal var targetEmbed: ((MLXArray) -> MLXArray)?
    internal var boundTargetID: ObjectIdentifier?

    public init(config: Gemma4AssistantConfiguration) throws {
        let geometry = try Gemma4AssistantConfigurationValidator.validate(config)
        self.config = config

        let textCfg = config.textConfig
        self._model.wrappedValue =
            Gemma4TextModelInner(textCfg, forceSharedKV: true)

        // pre: [2 * backbone -> drafter_hidden]
        self._preProjection.wrappedValue = Linear(
            geometry.preProjectionInputSize, textCfg.hiddenSize, bias: false)
        // post: [drafter_hidden -> backbone]
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)

        if !textCfg.tieWordEmbeddings {
            // Explicit LM head instantiated; Gemma4MaskedEmbedder stays nil.
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabSize, bias: false)
        }

        if config.useOrderedEmbeddings {
            self._maskedEmbedder.wrappedValue = Gemma4MaskedEmbedder(
                hiddenSize: textCfg.hiddenSize,
                numCentroids: config.numCentroids,
                topK: config.centroidIntermediateTopK,
                vocabSize: textCfg.vocabSize
            )
        }

        super.init()
    }

    /// Bind the drafter to a Gemma 4 target. Captures a closure into the
    /// target's scaled embedding lookup + runs compatibility validation.
    /// Idempotent on the same target; throws `.rebindForbidden` if called
    /// with a different target.
    public func bind(target: any Gemma4MTPTarget) throws {
        let newID = ObjectIdentifier(target)
        if let existing = boundTargetID {
            if existing == newID { return }  // idempotent same-target
            throw Gemma4MTPError.rebindForbidden
        }
        try validateCompatibility(with: target)
        self.targetEmbed = { [target] tokens in
            target.embedTokensForDrafter(tokens)
        }
        self.boundTargetID = newID
    }

    /// Clear the binding. Subsequent forward passes will throw
    /// `.drafterNotBound`.
    public func unbind() {
        self.targetEmbed = nil
        self.boundTargetID = nil
    }

    // MARK: - Forward pass

    /// Drafter forward pass.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `[B, 1, 2 * backboneHiddenSize]` - the drafter-step
    ///     input, which the round-loop constructs as
    ///     `concat([target_embed(last_token), last_hidden], axis: -1)`.
    ///   - sharedKV: K/V snapshot from the target's last non-shared
    ///     full-attention and sliding-attention layers. Every drafter layer
    ///     reads from the appropriate slot by its `layerType`.
    ///   - positionOffset: absolute position of the bonus token; held
    ///     constant across all drafter steps within a block.
    /// - Returns: `(lastHidden: [B, 1, backboneHiddenSize], logits: [B, 1, vocabSize])`.
    ///
    /// Softcap is **not** applied (drafter configs have
    /// `final_logit_softcapping: null`).
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let h = preProjection(inputsEmbeds)
        let masks = makeMasks(
            queryLen: h.dim(1),
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            dtype: h.dtype)
        return forwardProjected(
            h,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
    }

    internal func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        masks: Gemma4DrafterMasks
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let h = preProjection(inputsEmbeds)
        return forwardProjected(
            h,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
    }

    internal func makeMasks(
        queryLen: Int,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        dtype: DType
    ) -> Gemma4DrafterMasks {
        let textCfg = config.textConfig
        let queryOffset: Int
        switch positionOffset {
        case .scalar(let v): queryOffset = v
        case .batch(let arr): queryOffset = Int(arr[0].item(Int32.self))
        case .graphArray(let arr): queryOffset = Int(arr[0].item(Int32.self))
        }

        let fullKVLen = sharedKV.fullAttention.0.dim(2)
        let slidingKVLen = sharedKV.slidingAttention.0.dim(2)

        let fullMask = Gemma4DrafterMaskBuilder.bidirectionalFull(
            queryLen: queryLen, kvLen: fullKVLen, dtype: dtype)
        let slidingMask = Gemma4DrafterMaskBuilder.bidirectionalSWA(
            queryLen: queryLen, queryOffset: queryOffset,
            kvLen: slidingKVLen, window: textCfg.slidingWindow, dtype: dtype)
        return Gemma4DrafterMasks(full: fullMask, sliding: slidingMask)
    }

    private func forwardProjected(
        _ projected: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        masks: Gemma4DrafterMasks
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig
        var h = projected
        // Run each drafter layer with the appropriate shared-KV + mask.
        for (i, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[i]
            let kv: (MLXArray, MLXArray)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layerType {
            case "full_attention":
                kv = sharedKV.fullAttention
                mask = masks.full
            case "sliding_attention":
                kv = sharedKV.slidingAttention
                mask = masks.sliding
            default:
                preconditionFailure(
                    "Gemma4AssistantDraftModel: unexpected layerType '\(layerType)' at "
                    + "layer \(i). Compat validation should have rejected this."
                )
            }
            let (out, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                positionOffset: positionOffset
            )
            h = out
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)
        let logits = applyLMHead(h)
        return (lastHidden, logits)
    }

    /// Run preProjection → drafter layers → `model.norm` → `postProjection`
    /// and return the pre-LM-head hidden. The drafter's `forwardProjected`
    /// always applies the LM head; this variant stops before it so the CBv2
    /// drafter can batch the LM head across the k<=3 speculative envelope
    /// (DLMB). Shape: `[B, 1, hidden]`.
    ///
    /// The drafter layers consume the shared KV (no cache writes) and the
    /// per-step Q-only SDPA runs once per call — same path as
    /// `forwardProjected`, just no LM head at the tail.
    internal func forwardProjectedNoHead(
        _ projected: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset,
        masks: Gemma4DrafterMasks
    ) -> (preHeadHidden: MLXArray, lastHidden: MLXArray) {
        let textCfg = config.textConfig
        var h = projected
        for (i, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[i]
            let kv: (MLXArray, MLXArray)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layerType {
            case "full_attention":
                kv = sharedKV.fullAttention
                mask = masks.full
            case "sliding_attention":
                kv = sharedKV.slidingAttention
                mask = masks.sliding
            default:
                preconditionFailure(
                    "Gemma4AssistantDraftModel: unexpected layerType '\(layerType)' at "
                    + "layer \(i). Compat validation should have rejected this."
                )
            }
            let (out, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                positionOffset: positionOffset
            )
            h = out
        }
        h = model.norm(h)
        let lastHidden = postProjection(h)
        // `h` here is the pre-LM-head hidden; the caller batched-projects it.
        return (preHeadHidden: h, lastHidden: lastHidden)
    }

    /// Apply the LM head to a batched `[N, hidden]` hidden-states tensor in
    /// one dispatch. The LM head dispatch mirrors `applyLMHead`: masked
    /// centroid, explicit `lm_head`, or tied to `embed_tokens`. No softcap.
    ///
    /// `applyLMHead` is private; this is the public seam the CBv2 drafter
    /// uses to batch the projection across k positions. For the
    /// masked-centroid path the embedder is naturally `[B, L, hidden] → [B,
    /// L, vocab]`, so calling it with `[N, hidden]` (reshaped to `[N, 1,
    /// hidden]`) and reshaping the output back to `[N, vocab]` is
    /// semantically identical to k separate calls — just one matmul
    /// dispatch instead of k.
    internal func batchedApplyLMHead(_ hidden: MLXArray) -> MLXArray {
        let shape = hidden.shape
        precondition(shape.count == 2,
            "Gemma4AssistantDraftModel.batchedApplyLMHead: expected [N, hidden], got \(shape)")
        let hidden3D = hidden.reshaped([shape[0], 1, shape[1]])
        let logits3D: MLXArray
        if let maskedEmbedder {
            logits3D = maskedEmbedder(
                hiddenStates: hidden3D,
                lmHeadWeight: model.embedTokens.weight)
        } else if let lmHead {
            logits3D = lmHead(hidden3D)
        } else {
            logits3D = model.embedTokens.asLinear(hidden3D)
        }
        let vocab = logits3D.dim(-1)
        return logits3D.reshaped([shape[0], vocab])
    }

    /// Dispatch the LM head: masked-centroid if `useOrderedEmbeddings`,
    /// tied otherwise (unless an explicit `lm_head` is present).
    /// No softcap.
    private func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        if let maskedEmbedder {
            return maskedEmbedder(
                hiddenStates: hidden,
                lmHeadWeight: model.embedTokens.weight
            )
        }
        if let lmHead {
            return lmHead(hidden)
        }
        // Tied: project hidden through the (transposed) token embedding.
        return model.embedTokens.asLinear(hidden)
    }

    // MARK: - Weight sanitization

    /// Sanitize a raw drafter checkpoint dictionary into the form the
    /// drafter's modules expect.
    ///
    /// Rules:
    /// - Cast `masked_embedding.token_ordering` to int32 (HF ships it as int64).
    /// - Drop `lm_head.weight` when the drafter config has
    ///   `tieWordEmbeddings == true` (the LM head is tied to `embed_tokens`;
    ///   a present `lm_head.weight` in the checkpoint is redundant).
    /// - Throw on unexpected `k_proj` / `v_proj` / `k_norm` / `v_norm`
    ///   weights: every drafter layer is kv-shared by construction, so
    ///   those modules aren't instantiated. A stray weight indicates a
    ///   checkpoint/config mismatch and should surface loudly rather than
    ///   be silently dropped.
    ///
    /// - Throws: `Gemma4MTPError.incompatibleDrafter(field:...)` with a
    ///   descriptive field name when an unexpected K/V weight is present.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        let forbiddenSubstrings = [
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".self_attn.k_norm",
            ".self_attn.v_norm",
        ]

        for (key, value) in weights {
            // Fail loud on forbidden K/V weights.
            for substring in forbiddenSubstrings {
                if key.contains(substring) {
                    throw Gemma4MTPError.incompatibleDrafter(
                        field: key,
                        drafter: "(unexpected - drafter layers are kv-shared)",
                        target: "(should not be present)"
                    )
                }
            }

            // Drop lm_head.weight when tied.
            if key == "lm_head.weight" && config.textConfig.tieWordEmbeddings {
                continue
            }

            // Cast masked_embedding.token_ordering to int32.
            if key == "masked_embedding.token_ordering" {
                out[key] = value.asType(.int32)
                continue
            }

            out[key] = value
        }

        return out
    }

    // MARK: - Compatibility validation

    /// Fail-fast on every drafter/target mismatch with the field name in
    /// the error. Called once at bind time.
    private func validateCompatibility(with target: any Gemma4MTPTarget) throws {
        try Gemma4MTPCompatibilityValidator.validate(
            drafter: config,
            target: target.mtpConfiguration)
    }

    // MARK: - Loading

    /// Load a Gemma 4 MTP drafter from a local directory containing
    /// `config.json` and one or more `*.safetensors` files.
    ///
    /// - Parameter directory: local directory containing the drafter
    ///   checkpoint (e.g. pre-downloaded via `huggingface-cli`).
    /// - Returns: a ready-to-use drafter with weights loaded and evaluated.
    /// - Throws: file I/O errors, JSON decoding errors, or
    ///   `Gemma4MTPError.incompatibleDrafter` from `sanitize`.
    public static func load(from directory: URL) async throws -> Gemma4AssistantDraftModel {
        let configURL = directory.appending(component: "config.json")
        let document = try Gemma4AssistantConfigurationDocument.read(from: configURL)
        let config = document.config

        // Construct the drafter (random init).
        let drafter = try Gemma4AssistantDraftModel(config: config)

        // Collect all safetensors files in the directory. `resolvingSymlinksInPath`
        // ensures a symlinked directory (e.g. pointing into the HF cache) is
        // followed — `contentsOfDirectory` over the resolved path lists the
        // shards, and `loadArraysAndMetadata` follows the per-file blob symlinks.
        var weights = [String: MLXArray]()
        let scanDir = directory.resolvingSymlinksInPath()
        let shardURLs = (try? FileManager.default.contentsOfDirectory(
            at: scanDir, includingPropertiesForKeys: nil)) ?? []
        for url in shardURLs where url.pathExtension == "safetensors" {
            let (shardWeights, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in shardWeights {
                weights[k] = v
            }
        }

        // Run drafter sanitize (throws on unexpected K/V weights).
        let sanitized = try drafter.sanitize(weights: weights)

        // Quantize matching modules before applying weights when the checkpoint
        // is quantized (4-bit QAT drafter). A module is quantized iff its
        // `.scales` tensor is present. Mirrors `loadWeights` in MLXLMCommon.
        if let plq = document.baseConfiguration.perLayerQuantization {
            quantize(model: drafter) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? plq.quantization(layer: path)?.asTuple : nil
            }
        }

        // Apply weights.
        let params = ModuleParameters.unflattened(sanitized)
        try drafter.update(parameters: params, verify: [.all])
        eval(drafter)

        return drafter
    }

    /// Load a Gemma 4 MTP drafter from a remote model ID via a `Downloader`.
    ///
    /// Downloads `config.json` and `*.safetensors` files (tokenizer files are
    /// skipped - drafters reuse the target's tokenizer at generation time).
    ///
    /// - Parameters:
    ///   - downloader: any `Downloader` (e.g. `HubClient`).
    ///   - id: the model identifier (e.g. `"mlx-community/gemma-4-E4B-it-assistant-bf16"`).
    ///   - revision: the revision to download (defaults to the downloader's
    ///     default, typically `"main"`).
    ///   - useLatest: whether to bypass the downloader's cache.
    ///   - progressHandler: optional progress callback.
    /// - Returns: a ready-to-use drafter.
    public static func load(
        from downloader: any Downloader,
        id: String,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> Gemma4AssistantDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory)
    }
}

// MARK: - Greedy Round

internal struct Gemma4MTPGreedyRoundResult {
    let accepted: Int
    let tokens: [Int]
    let bonus: Int
    let hidden: MLXArray
    let sharedKV: Gemma4SharedKV
}

/// Run one single-batch greedy MTP round.
///
/// The caller owns remaining-token truncation. `blockSize` is the actual
/// per-round size (`bonus + drafts`), not necessarily the user's maximum
/// configured block size.
internal func runGemma4MTPGreedyRound(
    target: any Gemma4MTPTarget,
    drafter: Gemma4AssistantDraftModel,
    cache: [KVCache],
    bonus: Int,
    hidden: MLXArray,
    sharedKV: Gemma4SharedKV,
    blockSize: Int
) -> Gemma4MTPGreedyRoundResult {
    let k = blockSize - 1
    let driveOffset = cache[0].offset
    let positionOffset = Gemma4.PositionOffset.scalar(driveOffset)
    let masks = drafter.makeMasks(
        queryLen: 1,
        sharedKV: sharedKV,
        positionOffset: positionOffset,
        dtype: hidden.dtype)

    var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    var h = hidden
    var draftPerStep: [MLXArray] = []
    draftPerStep.reserveCapacity(k)

    for _ in 0 ..< k {
        let tokEmbed = target.embedTokensForDrafter(tok)
        let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
        let (newH, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
        let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
        let sampled2d = sampled[.newAxis, .ellipsis]
        draftPerStep.append(sampled2d)
        tok = sampled2d
        h = newH
    }

    let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput: MLXArray =
        draftPerStep.isEmpty
        ? bonusCol
        : concatenated([bonusCol] + draftPerStep, axis: 1)
    let verifyOut = target.forwardForMTP(verifyInput, cache: cache)
    let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
    let draftConcat: MLXArray =
        draftPerStep.isEmpty
        ? MLXArray.zeros([1, 0], dtype: .int32)
        : concatenated(draftPerStep, axis: 1)

    eval(mainTokens, draftConcat)
    let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
    let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self)
                                 .map { Int($0) }
    let (accepted, newTokens) = Gemma4SpeculativeWalk.single(
        draft: draftTokens, main: mainInts)

    if accepted < k {
        target.rollbackSpeculativeCache(
            cache, accepted: .scalar(accepted), blockSize: blockSize)
    }

    let rejected = k - accepted
    return Gemma4MTPGreedyRoundResult(
        accepted: accepted,
        tokens: newTokens,
        bonus: newTokens.last!,
        hidden: verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...],
        sharedKV: Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: rejected)
    )
}

// MARK: - Round Loops

/// Run the Gemma 4 MTP round loop for a single request (B=1).
///
/// - Parameters:
///   - target: the Gemma 4 target model. Must have been prefilled (i.e.
///     `forwardForMTP(promptTokens, cache:)` called once by the caller
///     before invoking this function) so `firstBonus`, `firstHidden`, and
///     `firstSharedKV` are available.
///   - drafter: the bound Gemma 4 MTP drafter. `bind(target:)` will be
///     called internally; if already bound to `target`, it's idempotent.
///   - targetCache: the KV caches from the prefill forward. Will be
///     advanced and rewound by this function.
///   - firstBonus: the token sampled from the last prefill position
///     (greedy argmax). This is the initial bonus - we yield it
///     immediately as the first stream element.
///   - firstHidden: the last-position hidden from the prefill forward,
///     shape `[B=1, L=1, hiddenSize]`.
///   - firstSharedKV: the shared-KV snapshot from the prefill forward.
///   - maxTokens: maximum total tokens to emit (including firstBonus).
///   - blockSize: speculative block size. Typical range 2-8.
/// - Returns: an `AsyncStream<Generation>` emitting each generated token
///   as `.chunk("<int>")`. The stream finishes when `maxTokens` is
///   reached or the loop terminates for any other reason.
/// - Throws: `Gemma4MTPError.invalidBlockSize` for blockSize < 2 or > 16.
///
/// Tokens are yielded as `.chunk("<int>")` strings; a real tokenizer
/// integration layer (see `generateGemma4MTP` in Task 22) wraps this with
/// `tokenizer.decode(tokenIds:)`.
public func runGemma4MTPRounds(
    target: any Gemma4MTPTarget,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: Int,
    firstHidden: MLXArray,
    firstSharedKV: Gemma4SharedKV,
    maxTokens: Int,
    blockSize: Int
) throws -> AsyncStream<Generation> {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    return AsyncStream<Generation> { continuation in
        // First yield: the prefill bonus.
        continuation.yield(.chunk("\(firstBonus)"))
        var emitted = 1

        // Mutable state.
        var bonus = firstBonus
        var hidden = firstHidden
        var sharedKV = firstSharedKV

        while emitted < maxTokens {
            let remaining = maxTokens - emitted
            let bs = min(blockSize, remaining + 1)
            if bs <= 1 { break }
            let round = runGemma4MTPGreedyRound(
                target: target,
                drafter: drafter,
                cache: targetCache,
                bonus: bonus,
                hidden: hidden,
                sharedKV: sharedKV,
                blockSize: bs)

            // --- Emit ---
            var stopEarly = false
            for t in round.tokens {
                continuation.yield(.chunk("\(t)"))
                emitted += 1
                if emitted >= maxTokens {
                    stopEarly = true
                    break
                }
            }
            if stopEarly {
                continuation.finish()
                return
            }

            // --- Update state ---
            hidden = round.hidden
            bonus = round.bonus
            sharedKV = round.sharedKV

            if emitted % 256 == 0 {
                MLX.Memory.clearCache()
            }
        }

        continuation.finish()
    }
}

// MARK: - Batched (B > 1) output type

/// Per-step output of the B>1 MTP round loop. Each `slots[i]` maps to
/// the original batch row `i`.
public struct BatchedGeneration: Sendable {
    public let slots: [Slot]

    public struct Slot: Sendable {
        /// Original index in the input batch.
        public let row: Int
        /// Token emitted this step; nil when the row is finished or the
        /// step didn't produce output for this row.
        public let token: Int?
        /// Non-nil on the step where the row finishes.
        public let finishReason: FinishReason?

        public init(row: Int, token: Int?, finishReason: FinishReason?) {
            self.row = row
            self.token = token
            self.finishReason = finishReason
        }
    }

    public enum FinishReason: Sendable {
        case stop
        case eos
        case length
    }

    public init(slots: [Slot]) {
        self.slots = slots
    }
}

// MARK: - Reusable batched MTP round state

/// One round of batched Gemma 4 MTP, factored out of `runGemma4MTPRoundsBatched`
/// so both the standalone stream API and the continuous-batching engine can
/// drive it. Holds the per-round *carry* state (per-row pending tokens, the
/// target hidden seed, and the shared-KV snapshot) for the currently active
/// rows. The caller owns the target KV cache, the per-row budgets/stop logic,
/// emission, and (on a row finishing) compaction.
///
/// Confirmed-prefix invariant: after every `step`, the target cache holds
/// exactly each row's confirmed prefix and `pendings[b]` carries the
/// confirmed-but-trimmed tail (≥ 1 token, last = the row's newest bonus).
/// See `runGemma4MTPRoundsBatched` for the full rationale.
public final class Gemma4MTPBatchState: @unchecked Sendable {
    public let target: Gemma4TextModel
    public let drafter: Gemma4AssistantDraftModel
    public let blockSize: Int

    /// Per active-row confirmed-but-not-cached tokens (last = newest bonus).
    public private(set) var pendings: [[Int]]
    /// Target hidden seed for the drafter chain, `[B, 1, H]`.
    public private(set) var hidden: MLXArray
    /// Shared-KV snapshot the drafter attends to.
    public private(set) var sharedKV: Gemma4SharedKV

    /// Number of currently active rows.
    public var activeCount: Int { pendings.count }

    // Optional per-phase profiling (enabled by the standalone API).
    public var profile = false
    public private(set) var rounds = 0
    private var draftMs = 0.0, verifyMs = 0.0, walkMs = 0.0, updateMs = 0.0
    private var emitTotal = 0, tInTotal = 0, draftStepsTotal = 0

    /// - Parameters carry the active-row seed produced by a prefill
    ///   `forwardForMTP`: `firstBonus[b]` is the token sampled at row b's last
    ///   prompt position, `firstHidden` is `[B,1,H]`, `firstSharedKV` the
    ///   captured snapshot. The caller must have already compacted these to the
    ///   active set (rows that finished at prefill removed).
    public init(
        target: Gemma4TextModel,
        drafter: Gemma4AssistantDraftModel,
        blockSize: Int,
        firstBonus: [Int],
        firstHidden: MLXArray,
        firstSharedKV: Gemma4SharedKV
    ) {
        self.target = target
        self.drafter = drafter
        self.blockSize = blockSize
        self.pendings = firstBonus.map { [$0] }
        self.hidden = firstHidden
        self.sharedKV = firstSharedKV
    }

    /// Run exactly one MTP round over the active rows.
    ///
    /// - Parameter targetCache: the target's per-layer batched KV caches
    ///   (owned by the caller); trimmed in place by the round's rollback.
    /// - Parameter maxNewPerRow: per active-row cap on tokens emitted this
    ///   round (the row's remaining budget). Every entry must be ≥ 1.
    /// - Returns: per active-row emitted tokens, each `1...maxNewPerRow[b]`
    ///   (accepted drafts + one correction/bonus). Index aligns with the
    ///   current active set; `pendings`/`hidden`/`sharedKV` are advanced.
    public func step(targetCache: [KVCache], maxNewPerRow: [Int]) -> [[Int]] {
        let B = pendings.count
        precondition(maxNewPerRow.count == B, "maxNewPerRow must match active rows")
        let minRemaining = maxNewPerRow.min() ?? 0
        let bs = Swift.min(blockSize, minRemaining + 1)
        precondition(bs >= 2, "Gemma4MTPBatchState.step requires bs >= 2; guard in caller")

        // --- Round geometry (fixed verify width = bs) ---
        let pendCounts = pendings.map(\.count)
        let maxPend = pendCounts.max() ?? 1
        let tIn = Swift.max(bs, maxPend)
        let freshCounts = pendCounts.map { tIn - $0 }
        let draftSteps = freshCounts.max() ?? 0
        let tDraft0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // --- Draft (lockstep) ---
        let physicalLen = targetCache[0].offset
        let positionOffset: Gemma4.PositionOffset
        if let positioned = targetCache[0] as? BatchPositionedKVCache {
            let pendOffsets = MLXArray(pendCounts.map { Int32($0 - 1) })
            positionOffset = .batch(positioned.batchOffset + pendOffsets)
        } else {
            positionOffset = .scalar(physicalLen + maxPend - 1)
        }
        let masks = drafter.makeMasks(
            queryLen: 1, sharedKV: sharedKV,
            positionOffset: .scalar(physicalLen + maxPend - 1), dtype: hidden.dtype)
        var tok = MLXArray(pendings.map { Int32($0.last!) }, [B, 1])
        var h = hidden
        var draftPerStep: [MLXArray] = []
        draftPerStep.reserveCapacity(draftSteps)
        for _ in 0 ..< draftSteps {
            let tokEmbed = target.embedTokensForDrafter(tok)
            let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
            let (newH, logits) = drafter(
                inputsEmbeds: inputsEmbeds, sharedKV: sharedKV,
                positionOffset: positionOffset, masks: masks)
            let sampled2d = logits.squeezed(axis: 1).argMax(axis: -1).reshaped([B, 1])
            draftPerStep.append(sampled2d)
            tok = sampled2d
            h = newH
        }

        // --- Verify input assembly (GPU-side) ---
        let pendPadded: [Int32] = pendings.flatMap { row -> [Int32] in
            row.map { Int32($0) } + Array(repeating: Int32(0), count: tIn - row.count)
        }
        let pendTensor = MLXArray(pendPadded, [B, tIn])
        let verifyInput: MLXArray
        let draftConcat: MLXArray
        if draftSteps > 0 {
            draftConcat = concatenated(draftPerStep, axis: 1)
            let pendCountCol = MLXArray(pendCounts.map { Int32($0) }, [B, 1])
            let positions = MLXArray(Array(0 ..< Int32(tIn)), [1, tIn])
            let draftIdx = clip(positions - pendCountCol, min: 0, max: Int32(draftSteps - 1))
            let shiftedDrafts = takeAlong(draftConcat, draftIdx, axis: -1)
            verifyInput = MLX.where(positions .< pendCountCol, pendTensor, shiftedDrafts)
        } else {
            draftConcat = MLXArray.zeros([B, 0], dtype: .int32)
            verifyInput = pendTensor
        }

        var tVerify0 = 0.0
        if profile {
            eval(verifyInput)
            let now = CFAbsoluteTimeGetCurrent()
            draftMs += (now - tDraft0) * 1000
            tVerify0 = now
        }

        let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
        let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
        eval(mainTokens, draftConcat)
        var tWalk0 = 0.0
        if profile {
            let now = CFAbsoluteTimeGetCurrent()
            verifyMs += (now - tVerify0) * 1000
            tWalk0 = now
        }
        let mainFlat = mainTokens.asArray(Int32.self)
        let draftFlat = draftConcat.asArray(Int32.self)

        // --- Walk (per row, capped by maxNewPerRow) ---
        var accepted: [Int] = []
        var newTokensPerRow: [[Int]] = []
        accepted.reserveCapacity(B)
        newTokensPerRow.reserveCapacity(B)
        for bi in 0 ..< B {
            let p = pendCounts[bi]
            let kRow = freshCounts[bi]
            let mStart = bi * tIn + (p - 1)
            let mainRow = mainFlat[mStart ..< mStart + kRow + 1].map { Int($0) }
            let dStart = bi * draftSteps
            let draftRow = draftSteps > 0
                ? draftFlat[dStart ..< dStart + kRow].map { Int($0) }
                : []
            var (a, e) = Gemma4SpeculativeWalk.single(draft: draftRow, main: mainRow)
            let budget = Swift.max(0, maxNewPerRow[bi])
            if e.count > budget {
                e = Array(e.prefix(budget))
                a = Swift.max(0, e.count - 1)
            }
            accepted.append(a)
            newTokensPerRow.append(e)
        }

        if profile {
            let now = CFAbsoluteTimeGetCurrent()
            walkMs += (now - tWalk0) * 1000
            rounds += 1
            emitTotal += newTokensPerRow.reduce(0) { $0 + $1.count }
            tInTotal += tIn
            draftStepsTotal += draftSteps
        }
        let tUpdate0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // --- Rewind (uniform, exact) ---
        let confirmedCounts = (0 ..< B).map { pendCounts[$0] + accepted[$0] }
        let maxRejected = (0 ..< B).map { tIn - confirmedCounts[$0] }.max() ?? 0
        if maxRejected > 0 {
            for cache in targetCache { _ = cache.trim(maxRejected) }
        }

        // --- Advance carry ---
        let hiddenDim = verifyOut.lastHidden.dim(2)
        let seedIdx = MLX.broadcast(
            MLXArray(confirmedCounts.map { Int32($0 - 1) }, [B, 1, 1]),
            to: [B, 1, hiddenDim])
        hidden = MLX.takeAlong(verifyOut.lastHidden, seedIdx, axis: 1)

        let cachedNow = tIn - maxRejected
        for bi in 0 ..< B {
            guard let bonus = newTokensPerRow[bi].last else { continue }
            let confirmedInput = pendings[bi] + Array(newTokensPerRow[bi].prefix(accepted[bi]))
            pendings[bi] = Array(confirmedInput[cachedNow...]) + [bonus]
        }
        sharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: maxRejected)
        if profile { updateMs += (CFAbsoluteTimeGetCurrent() - tUpdate0) * 1000 }

        return newTokensPerRow
    }

    /// Shrink the carry to keep only the active-row indices in `keepLocal`
    /// (after some rows finished). The caller must separately filter the
    /// target cache by the same indices.
    public func compactCarry(keepLocal: [Int]) {
        pendings = keepLocal.map { pendings[$0] }
        let keepIdx = MLXArray(keepLocal.map { Int32($0) }, [keepLocal.count])
        hidden = MLX.take(hidden, keepIdx, axis: 0)
        sharedKV = Gemma4SharedKV(
            fullAttention: (
                MLX.take(sharedKV.fullAttention.0, keepIdx, axis: 0),
                MLX.take(sharedKV.fullAttention.1, keepIdx, axis: 0)),
            slidingAttention: (
                MLX.take(sharedKV.slidingAttention.0, keepIdx, axis: 0),
                MLX.take(sharedKV.slidingAttention.1, keepIdx, axis: 0)))
    }

    /// Emit the accumulated per-phase profile (no-op unless `profile`).
    public func printProfile() {
        guard profile, rounds > 0 else { return }
        let n = Double(rounds)
        print(String(
            format: "[MTP-B prof] rounds=%d emit/round=%.2f tIn=%.2f draftSteps=%.2f "
                + "| draft=%.1fms verify=%.1fms walk=%.1fms update=%.1fms per round",
            rounds, Double(emitTotal) / n, Double(tInTotal) / n,
            Double(draftStepsTotal) / n,
            draftMs / n, verifyMs / n, walkMs / n, updateMs / n))
    }
}

// MARK: - Token Iterator

/// Single-batch (B=1) greedy MTP token iterator. Conforms to
/// ``TokenIteratorProtocol`` so callers can use the existing
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:wiredMemoryTicket:)``
/// entry point.
///
/// Usage:
/// ```swift
/// let iter = try Gemma4MTPTokenIterator(
///     input: lmInput,
///     target: targetGemma4TextModel,
///     drafter: drafter,
///     parameters: GenerateParameters(maxTokens: 512, temperature: 0),
///     blockSize: 4)
/// let (stream, _) = generateTask(
///     promptTokenCount: lmInput.text.tokens.size,
///     modelConfiguration: modelContext.configuration,
///     tokenizer: modelContext.tokenizer,
///     iterator: iter)
/// for await gen in stream { ... }
/// ```
///
/// Supports greedy (`temperature=0`) and B=1 stochastic sampling via
/// rejection-based speculative sampling.
public struct Gemma4MTPTokenIterator: TokenIteratorProtocol {

    // Kept alive by the iterator so the GPU ops finish cleanly; not
    // inspected from the outside.
    private let target: any Gemma4MTPTarget
    private let drafter: Gemma4AssistantDraftModel
    private var cache: [KVCache]
    private let configuredBlockSize: Int
    private let automaticPolicy: Gemma4MTPAutomaticPolicy?
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private var rngKey: MLXArray

    // MTP state - mutated each round.
    private var bonus: Int
    private var hidden: MLXArray
    private var sharedKV: Gemma4SharedKV

    // Token stream state.
    private var pendingTokens: [Int] = []
    private var pendingIndex: Int = 0

    // TokenIteratorProtocol conformance.
    public var tokenCount: Int = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize - runs the prompt prefill and stages the first bonus.
    /// The first token emitted by the iterator is the bonus sampled from
    /// the last prefill position.
    ///
    /// - Parameters:
    ///   - input: prepared LM input. `input.text.tokens` is used for
    ///     prefill.
    ///   - target: the Gemma 4 target model.
    ///   - drafter: the MTP drafter. Will be bound to `target` on first
    ///     call (idempotent if already bound).
    ///   - cache: optional pre-allocated cache. If nil, allocated via
    ///     `target.newCache(parameters:)`.
    ///   - parameters: generation parameters. `maxTokens`, `temperature`,
    ///     `topP`, `topK`, `minP` are honored. At `temperature == 0` the
    ///     iterator uses the greedy walker (exact match with baseline).
    ///     At `temperature > 0` it uses rejection-based speculative
    ///     sampling with the same filter masks applied to both drafter
    ///     and target distributions. Emitted tokens match the distribution
    ///     that pure target sampling with the same filters would produce.
    ///   - blockSize: fixed speculative block size (2-16). Pass `nil` to use
    ///     the model-aware automatic policy, which may adapt the B=1 runtime
    ///     action as generation length grows.
    ///   - rngSeed: seed for rejection sampling (only used when
    ///     `temperature > 0`). Default 0 which seeds from the current MLX
    ///     random state.
    /// - Throws: `Gemma4MTPError.invalidBlockSize` / bind errors.
    public init(
        input: LMInput,
        target: any Gemma4MTPTarget,
        drafter: Gemma4AssistantDraftModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int? = nil,
        rngSeed: UInt64 = 0
    ) throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(forTarget: target)
        let resolvedBlockSize = blockSize
            ?? policy.strategy(forBatchSize: 1).blockSize
        guard resolvedBlockSize >= 2 && resolvedBlockSize <= 16 else {
            throw Gemma4MTPError.invalidBlockSize(resolvedBlockSize)
        }
        try drafter.bind(target: target)

        self.target = target
        self.drafter = drafter
        self.cache = cache ?? target.mtpNewCache(parameters: parameters)
        self.configuredBlockSize = resolvedBlockSize
        self.automaticPolicy = blockSize == nil ? policy : nil
        self.maxTokens = parameters.maxTokens
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.rngKey = rngSeed == 0
            ? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1e6))
            : MLXRandom.key(rngSeed)

        // Prefill.
        let prefillStart = Date()
        var promptTokens = input.text.tokens
        if promptTokens.ndim == 1 {
            promptTokens = promptTokens[.newAxis, .ellipsis]
        }
        let prefillOut = target.forwardForMTP(promptTokens, cache: self.cache)
        let lastLogits = prefillOut.logits[0..., -1, 0...]
        let firstBonus: Int
        if parameters.temperature == 0 {
            let firstBonusArr = lastLogits.asType(.float32).argMax(axis: -1)
            eval(firstBonusArr)
            firstBonus = Int(firstBonusArr.item(Int32.self))
        } else {
            // Sample the first bonus from the target's temperature-scaled
            // softmax with the same filters the rest of the round loop
            // will use. Split the rng key so the round loop gets fresh
            // keys each call.
            let keys = MLXRandom.split(key: self.rngKey, into: 2)
            let idx = gemma4SampleLogitsWithFilters(
                lastLogits,
                temperature: parameters.temperature,
                topP: parameters.topP, topK: parameters.topK, minP: parameters.minP,
                rngKey: keys[0])
            eval(idx)
            firstBonus = Int(idx.item(Int32.self))
            self.rngKey = keys[1]
        }
        self.bonus = firstBonus
        self.hidden = prefillOut.lastHidden[
            0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
        self.sharedKV = prefillOut.capturedSharedKV
        self.promptPrefillTime = -prefillStart.timeIntervalSinceNow

        // First emitted token is the prefill bonus.
        self.pendingTokens.append(self.bonus)
    }

    /// Initialize from a caller-supplied prefill instead of running a
    /// text-token prefill. Used by the multimodal VLM path: the caller merges
    /// image/video features into the prompt embeddings, runs one
    /// `forwardForMTP`-style multimodal pass to advance `cache` and capture the
    /// seed `Gemma4MTPForward`, then hands it here. The subsequent decode
    /// rounds are text-only and identical to the text path. The first emitted
    /// token is the bonus sampled from the prefill's last position.
    ///
    /// - Parameters:
    ///   - prefill: seed produced by the caller's multimodal forward (logits,
    ///     pre-norm last hidden, shared-KV snapshot).
    ///   - cache: the caches the caller already advanced over the prompt.
    public init(
        prefill: Gemma4MTPForward,
        cache: [KVCache],
        target: any Gemma4MTPTarget,
        drafter: Gemma4AssistantDraftModel,
        parameters: GenerateParameters,
        blockSize: Int? = nil,
        rngSeed: UInt64 = 0
    ) throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(forTarget: target)
        let resolvedBlockSize = blockSize
            ?? policy.strategy(forBatchSize: 1).blockSize
        guard resolvedBlockSize >= 2 && resolvedBlockSize <= 16 else {
            throw Gemma4MTPError.invalidBlockSize(resolvedBlockSize)
        }
        try drafter.bind(target: target)

        self.target = target
        self.drafter = drafter
        self.cache = cache
        self.configuredBlockSize = resolvedBlockSize
        self.automaticPolicy = blockSize == nil ? policy : nil
        self.maxTokens = parameters.maxTokens
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.rngKey = rngSeed == 0
            ? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1e6))
            : MLXRandom.key(rngSeed)

        let lastLogits = prefill.logits[0..., -1, 0...]
        let firstBonus: Int
        if parameters.temperature == 0 {
            let arr = lastLogits.asType(.float32).argMax(axis: -1)
            eval(arr)
            firstBonus = Int(arr.item(Int32.self))
        } else {
            let keys = MLXRandom.split(key: self.rngKey, into: 2)
            let idx = gemma4SampleLogitsWithFilters(
                lastLogits,
                temperature: parameters.temperature,
                topP: parameters.topP, topK: parameters.topK, minP: parameters.minP,
                rngKey: keys[0])
            eval(idx)
            firstBonus = Int(idx.item(Int32.self))
            self.rngKey = keys[1]
        }
        self.bonus = firstBonus
        self.hidden = prefill.lastHidden[
            0..., -1 ..< prefill.lastHidden.dim(1), 0...]
        self.sharedKV = prefill.capturedSharedKV
        self.promptPrefillTime = 0

        // First emitted token is the prefill bonus.
        self.pendingTokens.append(firstBonus)
    }

    private mutating func nextTargetOnlyToken() -> Int? {
        let input = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        let out = target.forwardForMTP(input, cache: cache)
        let logits = out.logits[0..., -1, 0...]
        let nextArr: MLXArray
        if temperature == 0 {
            nextArr = logits.asType(.float32).argMax(axis: -1)
        } else {
            let keys = MLXRandom.split(key: rngKey, into: 2)
            nextArr = gemma4SampleLogitsWithFilters(
                logits,
                temperature: temperature,
                topP: topP, topK: topK, minP: minP,
                rngKey: keys[0])
            rngKey = keys[1]
        }

        eval(nextArr)
        let next = Int(nextArr.item(Int32.self))
        hidden = out.lastHidden[0..., -1 ..< out.lastHidden.dim(1), 0...]
        sharedKV = out.capturedSharedKV
        bonus = next
        tokenCount += 1
        return next
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain pending tokens before the next round.
        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return t
        }

        // All drained - run one MTP round to refill.
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        let action = automaticPolicy?.singleStreamAction(
            generatedTokens: tokenCount, maxTokens: maxTokens)
            ?? .mtp(blockSize: configuredBlockSize)
        if action == .targetOnly {
            return nextTargetOnlyToken()
        }

        guard case .mtp(let roundBlockSize) = action else { return nil }
        let remaining = (maxTokens.map { $0 - tokenCount }) ?? roundBlockSize
        guard remaining > 0 else { return nil }
        let bs = Swift.min(roundBlockSize, remaining + 1)
        if bs <= 1 { return nil }
        let k = bs - 1

        if temperature == 0 {
            let round = runGemma4MTPGreedyRound(
                target: target,
                drafter: drafter,
                cache: cache,
                bonus: bonus,
                hidden: hidden,
                sharedKV: sharedKV,
                blockSize: bs)
            hidden = round.hidden
            bonus = round.bonus
            sharedKV = round.sharedKV
            pendingTokens.append(contentsOf: round.tokens)
            if pendingTokens.isEmpty { return nil }
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return t
        }

        // Draft k tokens, GPU-resident.
        // At temperature > 0 we also need to retain per-step drafter
        // logits so the walker can compute q(c_i) for rejection sampling.
        let driveOffset = cache[0].offset
        let positionOffset = Gemma4.PositionOffset.scalar(driveOffset)
        let masks = drafter.makeMasks(
            queryLen: 1,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            dtype: hidden.dtype)
        var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        var h = hidden
        var draftPerStep: [MLXArray] = []
        var draftLogitsPerStep: [MLXArray] = []  // only populated when temperature > 0
        draftPerStep.reserveCapacity(k)
        if temperature > 0 {
            draftLogitsPerStep.reserveCapacity(k)
        }
        for _ in 0 ..< k {
            let tokEmbed = target.embedTokensForDrafter(tok)
            let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
            let (newH, logits) = drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionOffset: positionOffset,
                masks: masks)
            let stepLogits = logits.squeezed(axis: 1)  // [1, vocab]
            let sampled: MLXArray
            if temperature > 0 {
                draftLogitsPerStep.append(stepLogits)
                let keys = MLXRandom.split(key: rngKey, into: 2)
                rngKey = keys[0]
                sampled = gemma4SampleLogitsWithFilters(
                    stepLogits,
                    temperature: temperature,
                    topP: topP, topK: topK, minP: minP,
                    rngKey: keys[1])
            } else {
                sampled = stepLogits.argMax(axis: -1)
            }
            let sampled2d = sampled[.newAxis, .ellipsis]
            draftPerStep.append(sampled2d)
            tok = sampled2d
            h = newH
        }

        // Verify.
        let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        let verifyInput: MLXArray =
            draftPerStep.isEmpty
            ? bonusCol
            : concatenated([bonusCol] + draftPerStep, axis: 1)
        let verifyOut = target.forwardForMTP(verifyInput, cache: cache)
        // fp32 argmax at verify to avoid bf16 near-uniform-tail argmax
        // flips that would otherwise diverge from baseline.
        let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
        let draftConcat: MLXArray =
            draftPerStep.isEmpty
            ? MLXArray.zeros([1, 0], dtype: .int32)
            : concatenated(draftPerStep, axis: 1)
        eval(mainTokens, draftConcat)
        let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
        let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self)
                                     .map { Int($0) }

        // Walk: greedy or rejection-based depending on temperature.
        let accepted: Int
        let newTokens: [Int]
        if temperature > 0 && k > 0 {
            // Gather verify logits at positions 0..k (bs=k+1 positions).
            let verifyLogits = verifyOut.logits.squeezed(axis: 0)  // [bs, vocab]
            // Stack drafter logits into [k, vocab].
            let draftLogits = concatenated(draftLogitsPerStep, axis: 0)
            let keys = MLXRandom.split(key: rngKey, into: 2)
            rngKey = keys[0]
            let result = gemma4SpeculativeSampleRound(
                draftLogits: draftLogits,
                draftTokens: draftTokens,
                verifyLogits: verifyLogits,
                temperature: temperature,
                topP: topP, topK: topK, minP: minP,
                rngKey: keys[1])
            accepted = result.accepted
            newTokens = result.emitted
        } else {
            var a = 0
            while a < k && draftTokens[a] == mainInts[a] {
                a += 1
            }
            accepted = a
            newTokens = Array(mainInts[0 ..< a + 1])
        }

        // Rewind.
        if accepted < k {
            target.rollbackSpeculativeCache(
                cache, accepted: .scalar(accepted), blockSize: bs)
        }

        // Update state.
        hidden = verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...]
        bonus = newTokens.last!
        let rejected = k - accepted
        sharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: rejected)

        pendingTokens.append(contentsOf: newTokens)
        if pendingTokens.isEmpty { return nil }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}

// MARK: - Generation API

/// Generate text from a Gemma 4 target using a Gemma 4 MTP drafter.
///
/// Internally wraps ``Gemma4MTPTokenIterator`` and a tokenizer-decoding
/// loop, producing an `AsyncStream<Generation>` that yields decoded text
/// chunks and a terminal `.info(...)` with counts.
///
/// - Parameters:
///   - input: prepared language-model input. `input.text.tokens` is used
///     for prefill; images/videos are ignored (MTP is text-only).
///   - parameters: generation parameters. `maxTokens` caps output length
///     (defaults to 1024 if nil). `temperature > 0` enables stochastic
///     rejection-based speculative sampling; `temperature == 0` is
///     greedy and produces byte-identical output to no-drafter baseline.
///   - target: `ModelContext` whose `model` is a `Gemma4TextModel` (or a
///     `Gemma4Model` VLM wrapper whose text portion is `Gemma4TextModel`).
///     Throws `unsupportedTarget` otherwise.
///   - drafter: the loaded Gemma 4 MTP drafter.
///   - blockSize: fixed speculative block size (2-16). Pass `nil` to use
///     the model-aware automatic policy, which may adapt the B=1 runtime
///     action as generation length grows.
///   - rngSeed: seed for stochastic sampling (only consulted when
///     `parameters.temperature > 0`). Default 0 -> seeds from the system
///     clock.
/// - Returns: an `AsyncStream<Generation>` yielding `.chunk(String)`
///   (decoded token text) and one terminal `.info(...)`.
/// - Throws: `Gemma4MTPError.unsupportedTarget`, `.invalidBlockSize`,
///   or any error thrown by `drafter.bind(target:)`.
public func generateGemma4MTP(
    input: LMInput,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int? = nil,
    rngSeed: UInt64 = 0
) throws -> AsyncStream<Generation> {
    let gemma4: Gemma4TextModel
    if let t = target.model as? Gemma4TextModel {
        gemma4 = t
    } else if let wrapper = target.model as? Gemma4Model {
        gemma4 = wrapper.textModel
    } else {
        throw Gemma4MTPError.unsupportedTarget(
            String(describing: type(of: target.model)))
    }

    let tokenizer = target.tokenizer
    let eosIds = target.configuration.eosTokenIds
    var params = parameters
    if params.maxTokens == nil {
        params.maxTokens = 1024
    }
    let promptTokenCount = input.text.tokens.size

    let policy = Gemma4MTPAutomaticPolicy.automatic(for: gemma4)
    if blockSize == nil && !policy.enablesAutomaticMTP {
        return try generate(input: input, parameters: params, context: target)
    }

    // Build the iterator outside the stream closure so init errors can
    // throw synchronously. Prefill runs inside init.
    let iter = try Gemma4MTPTokenIterator(
        input: input,
        target: gemma4,
        drafter: drafter,
        parameters: params,
        blockSize: blockSize,
        rngSeed: rngSeed
    )
    let prefillElapsed = iter.promptPrefillTime

    // Wrap the iterator in a Sendable box so we can move it into the
    // stream closure without tripping strict concurrency (the iterator
    // holds MLXArray state which is `@unchecked Sendable`-compatible but
    // isn't annotated yet).
    let boxed = IteratorBox(iter: iter)

    return AsyncStream<Generation> { continuation in
        let task = Task {
            let generateStart = Date()
            var tokenCount = 0
            var stopReason: GenerateStopReason = .length
            // Streaming detokenizer: buffers a token whose decoded segment ends
            // mid-codepoint (U+FFFD) and emits only complete text. Byte-level
            // BPE splits many characters (CJK, emoji, accented Latin) across
            // multiple tokens, so the previous per-token `decode([tok])` emitted
            // replacement characters / split glyphs for any non-ASCII output.
            // Matches the continuous-batching engine's detokenization.
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            while let tok = boxed.next() {
                tokenCount += 1
                detokenizer.append(token: tok)
                if let chunk = detokenizer.next(), !chunk.isEmpty {
                    continuation.yield(.chunk(chunk))
                }
                if eosIds.contains(tok) {
                    stopReason = .stop
                    break
                }
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }
            }
            let elapsed = Date().timeIntervalSince(generateStart)
            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: prefillElapsed,
                generationTime: elapsed,
                stopReason: stopReason
            )
            continuation.yield(.info(info))
            continuation.finish()
        }
        continuation.onTermination = { _ in
            task.cancel()
        }
    }
}

/// Box wrapping a `Gemma4MTPTokenIterator` so it can be captured by a
/// `Sendable` closure. The iterator itself holds `MLXArray` state which
/// is conceptually reference-counted GPU memory - safe to move across
/// tasks, but the Swift compiler needs an explicit `@unchecked`.
private final class IteratorBox: @unchecked Sendable {
    private var iter: Gemma4MTPTokenIterator
    init(iter: Gemma4MTPTokenIterator) {
        self.iter = iter
    }
    func next() -> Int? {
        iter.next()
    }
}
