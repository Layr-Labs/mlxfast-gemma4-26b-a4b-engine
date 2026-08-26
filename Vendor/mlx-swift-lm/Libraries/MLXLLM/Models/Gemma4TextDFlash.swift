//
//  Gemma4TextDFlash.swift
//  mlx-swift-lm
//
//  DFlash target conformance for `Gemma4TextModel`.
//
//  PORT NOTE (engine, 2026-08-25, gemma4-dflash-real-loader lane). The
//  upstream fork carries these members INSIDE `Gemma4Text.swift` on
//  `origin/dflash-framework-updates` (@ d41c3003). This engine's vendored
//  `Gemma4Text.swift` descends from mlx-swift-lm main @ ed55bee plus local
//  edits and has diverged materially from that branch (CBv2 prefill
//  specializations, PLE/shared-KV handling, the weighted-unsort pair), so
//  wholesale-replacing it would drag in an unrelated model rewrite. The
//  DFlash pieces are grafted instead: `Gemma4Text.swift` gained only the
//  trunk's optional `dFlashHiddenCapture` observer, the `forceArrayMask`
//  pass-through, `callCapturingDFlashHiddenStates`, and `applyRawLMHead`.
//  Everything else DFlash-shaped lives here.
//
//  DELIBERATELY NOT PORTED from the branch: the verify-path FUSIONS and
//  their selectors (`DFlashVerifyQuantizedLinear` call sites, sequential /
//  mixed / auto verify, the trunk timing split, `dFlashLayerComponents`).
//  Those are throughput optimizations layered on the branch's own attention
//  and MoE code, which this tree does not share; each would need its own
//  numerical certification here. The conformance below is the plain,
//  correct path: one trunk pass, capture the taps, apply the head.

import Foundation
import MLX
import MLXLMCommon

/// Collects the target's post-layer hidden states at the DFlash tap layers,
/// in the order `dflash_config.target_layer_ids` lists them (which is the
/// order `DFlashTargetForward` concatenates them in, and therefore the order
/// the drafter's `fc` projection was trained against — NOT sorted order).
final class Gemma4DFlashHiddenCapture {
    private let positionsByLayer: [Int?]
    private var hiddenStates: [MLXArray?]

    init(layerIds: [Int], layerCount: Int) {
        var positions = [Int?](repeating: nil, count: layerCount)
        for (position, layerId) in layerIds.enumerated() {
            positions[layerId] = position
        }
        self.positionsByLayer = positions
        self.hiddenStates = [MLXArray?](repeating: nil, count: layerIds.count)
    }

    @inline(__always)
    func capture(_ hidden: MLXArray, layer: Int) {
        if let position = positionsByLayer[layer] {
            hiddenStates[position] = hidden
        }
    }

    /// Every slot is filled because `validateTargetLayerIds` already proved
    /// each id is in `0..<layerCount` and the trunk visits every layer.
    func orderedHiddenStates() -> [MLXArray] {
        hiddenStates.map { $0! }
    }
}

/// Force an array attention mask for a multi-token DFlash verify forward over
/// a non-empty cache. Default ON, matching the fork's
/// `MLX_GEMMA4_DFLASH_ARRAY_VERIFY_MASK` default: a `[1, 1+k]` verify at a
/// non-zero cache offset is exactly the shape where the symbolic causal fast
/// path and the materialized mask have been observed to disagree. The env
/// var is kept so a box can A/B it without a rebuild.
private let gemma4DFlashArrayVerifyMask: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "MLX_GEMMA4_DFLASH_ARRAY_VERIFY_MASK"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

extension Gemma4TextModel: DFlashTargetModel {
    public var dFlashVocabularySize: Int { vocabularySize }
    public var dFlashHiddenSize: Int { configuration.hiddenSize }
    public var dFlashLayerCount: Int { configuration.numHiddenLayers }

    public func forwardForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        let forward = try model.callCapturingDFlashHiddenStates(
            inputs,
            cache: cache,
            targetLayerIds: targetLayerIds,
            forceArrayMask: gemma4DFlashShouldForceArrayMask(inputs, cache: cache))
        return DFlashTargetForward(
            logits: applyLMHead(forward.postNorm),
            hiddenStates: forward.hiddenStates
        )
    }

    /// Greedy verify: the argmax is taken from the SOFTCAPPED logits, the
    /// same values ordinary decode ranks, so a DFlash-verified token is the
    /// token plain decode would have produced at that position.
    public func forwardGreedyTokensForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        let forward = try model.callCapturingDFlashHiddenStates(
            inputs,
            cache: cache,
            targetLayerIds: targetLayerIds,
            forceArrayMask: gemma4DFlashShouldForceArrayMask(inputs, cache: cache))
        let targetHidden =
            forward.hiddenStates.count == 1
            ? forward.hiddenStates[0]
            : concatenated(forward.hiddenStates, axis: -1)
        return DFlashGreedyTargetForward(
            tokens: applyLMHead(forward.postNorm).argMax(axis: -1),
            targetHidden: targetHidden
        )
    }

    public func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        embedTokensForDrafter(tokens)
    }

    public func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        applyRawLMHead(hidden)
    }
}

private func gemma4DFlashShouldForceArrayMask(
    _ inputs: MLXArray, cache: [KVCache]?
) -> Bool {
    gemma4DFlashArrayVerifyMask
        && inputs.ndim >= 2
        && inputs.dim(1) > 1
        && (cache?.first?.offset ?? 0) > 0
}
