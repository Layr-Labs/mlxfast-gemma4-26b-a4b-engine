// SamplerV2.swift
//
// Vectorized token selection for ContinuousBatchingV2 (workstream E).
// Consumes the transformed [B, vocab] logits produced by
// `LogitsPipelineV2.process` and returns one token per row [B].
//
//  - All-greedy fast path: an exact tiled argmax for large vocabularies,
//    with the stock `argMax` fallback, when every row is greedy
//    (temperature < 1e-5).
//  - Mixed batches: Gumbel-max via the exponential-noise race
//    (`argmax(probs / e)`, e ~ Exp(1)) so there is no multinomial and no
//    host sync. Greedy and sampled picks are merged with a per-row
//    `where(greedy, argmax, sampled)`.
//  - Per-row keyed RNG: each row's noise is generated from its own key,
//    derived ONLY from (seed, requestID, per-request step index). A row's
//    random stream therefore never depends on its batchmates or on when it
//    joined the batch — the batch-composition-invariance requirement
//    (research report 12, item 5). Reproducibility is best-effort under
//    batching, per the contract: fixed (seed, requestID) reproduces the
//    same stream across runs given the same per-row logits.
//
// Threading: `sample` builds graph nodes only (no eval, no `.item()`).
// `setRows` runs at batch-membership change; `commit` is O(B) host counter
// bookkeeping (no device work).

import Foundation
import MLX
import MLXRandom

public final class SamplerV2 {

    public static let greedyEpsilon: Float = LogitsPipelineV2.greedyEpsilon

    private let fallbackSeed: UInt64

    private struct RowState {
        var id: CBv2RequestID
        var seed: UInt64
        var greedy: Bool
        var step: UInt64
    }

    private var rows: [RowState] = []
    private var allGreedy = true
    private var greedyMask: MLXArray?

    public init(fallbackSeed: UInt64? = nil) {
        self.fallbackSeed = fallbackSeed ?? UInt64.random(in: .min ... .max)
    }

    // MARK: Membership change

    public func setRows(_ rows: [CBv2SamplerRow]) {
        self.rows = rows.map { row in
            RowState(
                id: row.id,
                seed: row.params.seed ?? fallbackSeed,
                greedy: row.params.temperature < Self.greedyEpsilon,
                step: UInt64(row.outputTokens.count)
            )
        }
        allGreedy = self.rows.allSatisfy(\.greedy)
        greedyMask = self.rows.isEmpty ? nil : MLXArray(self.rows.map(\.greedy))
    }

    // MARK: Step path

    public func sample(from logits: MLXArray) -> MLXArray {
        precondition(
            logits.dim(0) == rows.count,
            "logits rows (\(logits.dim(0))) != configured rows (\(rows.count)) — call setRows")

        if allGreedy {
            return CBv2ParallelArgMaxV1.apply(logits)
                ?? argMax(logits, axis: -1).asType(.int32)
        }
        let greedyTokens = argMax(logits, axis: -1).asType(.int32)

        let vocab = logits.dim(-1)
        let probs = softmax(logits, axis: -1)
        let noise = exponentialNoise(vocab: vocab)
        let sampledTokens = argMax(probs / noise, axis: -1).asType(.int32)

        guard let greedyMask else { return sampledTokens }
        return which(greedyMask, greedyTokens, sampledTokens)
    }

    public func commit() {
        for i in rows.indices {
            rows[i].step &+= 1
        }
    }

    // MARK: - Per-row keyed noise

    private func exponentialNoise(vocab: Int) -> MLXArray {
        var perRow = [MLXArray]()
        perRow.reserveCapacity(rows.count)
        for row in rows {
            if row.greedy {
                perRow.append(MLXArray.full([1, vocab], values: MLXArray(Float(1))))
                continue
            }
            let key = MLXRandom.key(Self.mix(seed: row.seed, id: row.id.raw, step: row.step))
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [1, vocab], type: Float.self, key: key)
            let e = maximum(-log(1 - u), MLXArray(Float(1e-20)))
            perRow.append(e)
        }
        return concatenated(perRow, axis: 0)
    }

    static func mix(seed: UInt64, id: UInt64, step: UInt64) -> UInt64 {
        func splitmix(_ value: UInt64) -> UInt64 {
            var z = value &+ 0x9E37_79B9_7F4A_7C15
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        return splitmix(splitmix(splitmix(seed) ^ id) ^ step)
    }
}
