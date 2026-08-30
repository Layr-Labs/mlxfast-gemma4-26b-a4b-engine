// LogitslessGreedyHeadV2.swift
//
// LGH-001: seams for a tied head that returns exact greedy token ids
// instead of materializing a full logits plane. Non-conforming models and
// samplers retain the existing logits-then-sample path. Production default
// is ON; DARKBLOOM_GEMMA4_LOGITSLESS_HEAD=0 restores the stock head.

import MLX

/// Model forward refinement for an exact fused top-1 decode head.
public protocol CBv2ArgmaxDecodeForwardable: AnyObject {
    /// Pure host admission predicate evaluated before the forward graph is built.
    func cbv2AdmitsArgmaxDecode(_ tokens: MLXArray) -> Bool

    /// Returns `[B]` int32 ids exactly matching raw-BF16-logits `argMax`.
    func cbv2DecodeArgmax(_ tokens: MLXArray, caches: [KVCache]) -> MLXArray
}

/// Steppable-model refinement used by the batching engine.
public protocol CBv2ArgmaxDecodeSteppableModel: CBv2SteppableModel {
    func admitsArgmaxDecode(tokens: MLXArray) -> Bool
    func decodeArgmax(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}

/// Sampler refinement for a step whose only legal operation is greedy argmax.
public protocol CBv2FusedGreedySampler: CBv2StepSampler {
    /// Must reject every value-sensitive option and every constrained row.
    func admitsFusedGreedy(
        params: [CBv2SamplingParams], hasTokenConstraints: Bool
    ) -> Bool

    /// Invalidates sampler state after a fused step bypasses `sample`.
    func noteFusedGreedySample()
}
