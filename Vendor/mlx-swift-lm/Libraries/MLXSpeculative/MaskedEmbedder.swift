// Copyright © 2026 Apple Inc.
//
// Centroid-routed sparse LM head for Gemma 4 E2B / E4B drafters.
//
// Mirrors HF `Gemma4AssistantMaskedEmbedder`. The drafter scores
// `numCentroids` (typ. 2048) token clusters, materializes the top-K
// (typ. 32) clusters' tokens (~4096 of 262144 vocab), and scatters those
// logits back into a full-vocab tensor. Non-selected positions are filled
// with a sentinel `min(selected_logits) - 1` so they lose any argmax /
// sampling competition.
//
// Reference:
//   mlx_vlm/speculative/drafters/gemma4_assistant/masked_embedder.py
//   in Blaizzy/mlx-vlm#1112 (merged 244f4bb).

import Foundation
import MLX
import MLXNN

public final class MaskedEmbedder: Module, @unchecked Sendable {
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
    ///   - lmHeadWeight: `[vocabSize, hidden]` — typically tied to the
    ///     drafter's `embed_tokens.weight`.
    /// - Returns: `[B, L, vocabSize]` with non-selected positions set to
    ///   `min(selected) - 1`.
    public func callAsFunction(
        hiddenStates: MLXArray, lmHeadWeight: MLXArray
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        // Cluster scores → top-K cluster indices.
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
        // [B, L, 1, hidden] @ [B, L, hidden, topK * vsc] → [B, L, 1, topK * vsc]
        let selectedLogits = matmul(
            hiddenStates.expandedDimensions(axis: -2),
            selectedEmb.swappedAxes(-1, -2)
        ).squeezed(axis: -2)  // [B, L, topK * vsc]

        // Scalar fill is faster than an explicit on-device broadcast on
        // M3 for this shape; the only CPU sync in the sparse head is this
        // scalar sentinel extraction.
        let sentinelValue = selectedLogits.min().item(Float.self) - 1.0

        // Full-vocab output tensor filled with the sentinel.
        let out = MLX.full(
            [B, L, vocabSize],
            values: MLXArray(sentinelValue),
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
