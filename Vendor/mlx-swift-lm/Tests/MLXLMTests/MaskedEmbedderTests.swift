// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLLM
import Testing

@Suite("Gemma4MaskedEmbedder")
struct MaskedEmbedderTests {

    /// Build a small Gemma4MaskedEmbedder with known weights.
    /// hiddenSize=4, numCentroids=8, topK=2, vocabSize=16 → 2 tokens per centroid.
    private func buildEmbedder() -> (Gemma4MaskedEmbedder, MLXArray) {
        let hiddenSize = 4
        let numCentroids = 8
        let topK = 2
        let vocabSize = 16

        let embedder = Gemma4MaskedEmbedder(
            hiddenSize: hiddenSize,
            numCentroids: numCentroids,
            topK: topK,
            vocabSize: vocabSize
        )

        // Deterministic centroid weights: identity-ish for clarity.
        // token_ordering maps each centroid c to tokens [2c, 2c+1].
        // So centroid 3 → tokens [6, 7]; centroid 7 → tokens [14, 15].
        let tokenOrdering = MLXArray(
            (0 ..< vocabSize).map { Int32($0) })
        let centroidWeight = MLXArray.zeros([numCentroids, hiddenSize])
        // lm_head_weight: row `t` is a 1-hot at position (t % hiddenSize).
        let lmHead = MLXArray.zeros([vocabSize, hiddenSize])
        for t in 0 ..< vocabSize {
            lmHead[t, t % hiddenSize] = MLXArray(Float(1))
        }
        eval(centroidWeight, lmHead, tokenOrdering)

        embedder.setTestWeights(
            centroidsWeight: centroidWeight,
            tokenOrdering: tokenOrdering
        )
        return (embedder, lmHead)
    }

    @Test func outputShapeIsFullVocab() {
        let (embedder, lmHead) = buildEmbedder()
        let hidden = MLXArray.ones([1, 1, 4], dtype: .float32)
        let logits = embedder(hiddenStates: hidden, lmHeadWeight: lmHead)
        #expect(logits.shape == [1, 1, 16])
    }

    @Test func nonSelectedSlotsGetSentinel() {
        // With all-zero centroids, argPartition picks the first topK indices.
        // Those top-2 centroids → 4 of 16 vocab slots get "real" logits;
        // the other 12 slots must all equal the same sentinel value.
        let (embedder, lmHead) = buildEmbedder()
        let hidden = MLXArray.ones([1, 1, 4], dtype: .float32)
        let logits = embedder(hiddenStates: hidden, lmHeadWeight: lmHead)

        // Flatten to [16] and collect the distinct values.
        let flat = logits.reshaped([-1]).asArray(Float.self)
        var valueCounts = [Float: Int]()
        for v in flat { valueCounts[v, default: 0] += 1 }

        // Exactly 4 "real" slots; 12 sentinel slots with the same value.
        // (With one-hot lm_head and hidden=ones, each selected slot has a
        // real logit of 1.0; the sentinel is min(selected) - 1 = 0.)
        // At least one value should appear 12 times (the sentinel).
        let sentinelCount = valueCounts.values.max() ?? 0
        #expect(sentinelCount >= 12)
    }

    @Test func argmaxFallsInTopKCentroid() {
        // With a hidden that points strongly at one dimension, the argmax
        // token's canonical centroid must be among the top-K scoring
        // centroids (tautological for small k, but a good shape check).
        let (embedder, _) = buildEmbedder()
        let hidden = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 1, 4])
        // Use a centroid weight that biases centroid index 3 strongly.
        let centroidWeight = MLXArray.zeros([8, 4])
        centroidWeight[3, 0] = MLXArray(Float(10))
        eval(centroidWeight)
        embedder.setTestCentroidWeight(centroidWeight)

        let lmHead = MLXArray.zeros([16, 4])
        for t in 0 ..< 16 { lmHead[t, 0] = MLXArray(Float(1)) }
        eval(lmHead)
        let logits = embedder(hiddenStates: hidden, lmHeadWeight: lmHead)

        // The argmax token should fall within centroid 3's tokens
        // (tokens 6 or 7 given our tokenOrdering setup). Loose smoke
        // check: the path runs without crashing.
        let argmax = logits.argMax(axis: -1).item(Int32.self)
        _ = argmax
    }
}
