// LogitslessGreedyHeadV2.swift
//
// LGH-001 --- the seam that lets the tied LM head return TOKENS instead of
// logits on the chained pure-decode step.
//
// On that step the `[8, 262144]` logits tensor has exactly one consumer:
// `argMax`. The softcap in front of it is strictly increasing, so it cannot
// move the argmax, and every other pipeline transform is inactive when each
// row is plain greedy. Materialising the plane therefore costs one 4 MB store,
// one softcap pass that reads it and writes 8 MB, and one argmax pass that
// reads those 8 MB back --- all of it on the step's dependent chain, all of it
// to recover eight integers the head kernel already knows.
//
// Both protocols below are opt-in refinements. A model or sampler that does
// not conform (or that answers `false`) keeps the established
// logits-then-sample path byte for byte.

import Foundation
import MLX

/// Models whose decode forward can end in a fused top-1 selection.
///
/// `cbv2AdmitsArgmaxDecode` is a pure host predicate over the step geometry,
/// answered BEFORE the forward graph is built so the engine can pick the seam.
/// `cbv2DecodeArgmax` is total once it holds: a conformer that cannot fuse its
/// head after the trunk must still return the argmax of the logits it would
/// otherwise have produced.
public protocol CBv2ArgmaxDecodeForwardable: AnyObject {
    func cbv2AdmitsArgmaxDecode(_ tokens: MLXArray) -> Bool
    /// `[B]` int32 token ids --- the same array `argMax(logits, axis: -1)
    /// .asType(.int32)` would have produced for this step.
    func cbv2DecodeArgmax(_ tokens: MLXArray, caches: [KVCache]) -> MLXArray
}

/// Steppable-model refinement of the above, answered at runtime by the
/// adapter exactly like the MTP and prefill capabilities.
public protocol CBv2ArgmaxDecodeSteppableModel: CBv2SteppableModel {
    func admitsArgmaxDecode(tokens: MLXArray) -> Bool
    func decodeArgmax(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}

/// Samplers that can hand a whole step to a fused greedy head.
///
/// `admitsFusedGreedy` must answer true ONLY when this sampler's own `sample`
/// would collapse to `argMax(logits, axis: -1).asType(.int32)` for every row:
/// no bias, no penalties, no temperature, no top-k/p/min-p, no logprobs.
/// `noteFusedGreedySample` then reports the step the sampler did not see, so
/// its per-row state can be rebuilt from confirmed history before any later
/// step that does go through `sample`.
public protocol CBv2FusedGreedySampler: CBv2StepSampler {
    func admitsFusedGreedy(params: [CBv2SamplingParams]) -> Bool
    func noteFusedGreedySample()
}
