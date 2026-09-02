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
//  The fork's model-specific verify modules still cannot be transplanted
//  directly into this materially different trunk. Small physical rows instead
//  reuse this tree's already installed and numerically certified B1/C2-C4/C8/C16
//  verifier table. It owns the same projection sites the reference DFlash
//  implementation specializes, while preserving this target's arithmetic,
//  mixed-precision storage, shared-KV handling, and fixed-width route contract.

import Foundation
import MLX
import MLXLMCommon

/// Fixed-width greedy target forward bound once to one persistent cache bank.
/// Validation, verifier lookup, target-tap routing, and compile state capture
/// all finish before this value is published to the measured DFlash loop.
public struct Gemma4DFlashGreedyForwardBinding {
    public let columns: Int
    fileprivate let hiddenStateCount: Int
    fileprivate let body: (MLXArray) -> [MLXArray]

    public func callAsFunction(_ input: MLXArray) -> DFlashGreedyTargetForward {
        let outputs = body(input)
        return DFlashGreedyTargetForward(
            tokens: outputs[0],
            hiddenStates: Array(outputs[1 ..< (hiddenStateCount + 1)]))
    }
}

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

extension Gemma4TextModel: DFlashTargetModel, DFlashTargetCacheRollbackProvider {
    public var dFlashVocabularySize: Int { vocabularySize }
    public var dFlashHiddenSize: Int { configuration.hiddenSize }
    public var dFlashLayerCount: Int { configuration.numHiddenLayers }

    public func forwardForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        let verifier = dFlashInstalledVerifier(for: inputs)
        let forward = try model.callCapturingDFlashHiddenStates(
            inputs,
            cache: cache,
            targetLayerIds: targetLayerIds,
            forceArrayMask: gemma4DFlashShouldForceArrayMask(inputs, cache: cache),
            verifier: verifier)
        return DFlashTargetForward(
            logits: applyLMHead(forward.postNorm, verifier: verifier),
            hiddenStates: forward.hiddenStates
        )
    }

    /// Greedy verify: Gemma's final tanh softcap is strictly monotonic, so the
    /// order-only route can take argmax from the raw head without changing the
    /// token ordinary softcapped greedy decode would have produced.
    public func forwardGreedyTokensForDFlash(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        let verifier = dFlashInstalledVerifier(for: inputs)
        let forward = try model.callCapturingDFlashHiddenStates(
            inputs,
            cache: cache,
            targetLayerIds: targetLayerIds,
            forceArrayMask: gemma4DFlashShouldForceArrayMask(inputs, cache: cache),
            verifier: verifier)
        let tokens = CBv2OrderOnlyLogits.withGreedyOrderOnly {
            applyLMHead(forward.postNorm, verifier: verifier).argMax(axis: -1)
        }
        return DFlashGreedyTargetForward(
            tokens: tokens,
            hiddenStates: forward.hiddenStates
        )
    }

    public func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        embedTokensForDrafter(tokens)
    }

    public func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        applyRawLMHead(hidden)
    }

    /// Bind a whole fixed-width verifier graph, including cache writes, to a
    /// persistent compile state. C2-C4/C8/C16 use the installed exact
    /// projection table; other adaptive widths compile the established stock
    /// projections over the same fixed caches.
    public func bindCompiledDFlashGreedyForward(
        cache: [KVCache],
        targetLayerIds: [Int],
        columns: Int
    ) throws -> Gemma4DFlashGreedyForwardBinding {
        try DFlashTargetValidation.validateTargetLayerIds(
            targetLayerIds, layerCount: configuration.numHiddenLayers)
        let verifier = cbv2MTPVerifierContext(batch: 1, columns: columns)
        let certifiedShape = CBv2Gemma4MTPVerifierShape(
            batch: 1, columns: columns)
        if CBv2Gemma4MTPVerifierRoute.production.supports(certifiedShape),
            verifier == nil
        {
            preconditionFailure(
                "compiled DFlash verifier requires an installed B1/C\(columns) route")
        }
        precondition(
            cache.count == configuration.numHiddenLayers
                && cache.allSatisfy { $0 is CompilableKVCache },
            "compiled DFlash verifier requires one fixed cache per target layer")

        let hiddenStateCount = targetLayerIds.count
        let stateArrays = cache.flatMap { $0.innerState() }
        let stateCount = stateArrays.count
        let resultCount = hiddenStateCount + 1
        let compiled = compile(shapeless: false) { [self] arrays in
            let savedState = stateArrays.map { $0._copyContextInternal() }
            for (state, tracer) in zip(stateArrays, arrays.dropFirst()) {
                state._updateInternal(tracer)
            }
            let forward = model.callCapturingValidatedDFlashHiddenStates(
                arrays[0],
                cache: cache,
                targetLayerIds: targetLayerIds,
                forceArrayMask: true,
                verifier: verifier)
            let tokens = applyLMHead(
                forward.postNorm, verifier: verifier
            ).argMax(axis: -1)
            let updatedState = cache.flatMap { $0.innerState() }.map {
                $0._copyContextInternal()
            }
            for (state, saved) in zip(stateArrays, savedState) {
                state._updateInternal(saved)
            }
            return [tokens] + forward.hiddenStates + updatedState
        }
        return Gemma4DFlashGreedyForwardBinding(
            columns: columns,
            hiddenStateCount: hiddenStateCount,
            body: { input in
                CBv2OrderOnlyLogits.withGreedyOrderOnly {
                    let outputs = compiled([input] + stateArrays)
                    for (state, updated) in zip(
                        stateArrays, outputs.suffix(stateCount))
                    {
                        state._updateInternal(updated)
                    }
                    return Array(outputs.prefix(resultCount))
                }
            })
    }

    /// Bind one construction-certified CBv2 target forward. The immutable
    /// model verifier context, target tap topology, and concrete cache stack
    /// are captured before the D15 cache transaction is published. Decode
    /// subsequently calls this closure directly after selecting only by the
    /// actual physical column count.
    public func bindCertifiedDFlashGreedyForward(
        cache: [KVCache],
        targetLayerIds: [Int],
        columns: Int
    ) throws -> Gemma4DFlashGreedyForwardBinding {
        try DFlashTargetValidation.validateTargetLayerIds(
            targetLayerIds, layerCount: configuration.numHiddenLayers)
        guard let verifier = cbv2MTPVerifierContext(
            batch: 1, columns: columns)
        else {
            throw Gemma4MTPVerifierInstallationError.incompatibleModel(
                "DFlash B1/C\(columns) verifier context")
        }
        guard cache.count == configuration.numHiddenLayers,
            let cacheStack = CBv2CertifiedContiguousLayerCacheStack(cache)
        else {
            throw Gemma4MTPVerifierInstallationError.incompatibleModel(
                "DFlash unified physical-B1 contiguous cache stack")
        }

        let hiddenStateCount = targetLayerIds.count
        return Gemma4DFlashGreedyForwardBinding(
            columns: columns,
            hiddenStateCount: hiddenStateCount,
            body: { [self] input in
                CBv2OrderOnlyLogits.withGreedyOrderOnly {
                    let outputs = certifiedDFlashGreedyVerifier(
                        input,
                        cacheStack: cacheStack,
                        targetLayerIds: targetLayerIds,
                        verifier: verifier)
                    return outputs
                }
            })
    }

    /// Construction-bound noncompiled twin for widths outside the installed
    /// C2-C4/C8/C16 table. It shares the persistent fixed cache state and skips the
    /// immutable target-layer validation already performed here.
    public func bindDirectDFlashGreedyForward(
        cache: [KVCache],
        targetLayerIds: [Int],
        columns: Int
    ) throws -> Gemma4DFlashGreedyForwardBinding {
        try DFlashTargetValidation.validateTargetLayerIds(
            targetLayerIds, layerCount: configuration.numHiddenLayers)
        precondition(
            cache.count == configuration.numHiddenLayers
                && cache.allSatisfy { $0 is CompilableKVCache },
            "direct DFlash verifier requires one fixed cache per target layer")
        let hiddenStateCount = targetLayerIds.count
        return Gemma4DFlashGreedyForwardBinding(
            columns: columns,
            hiddenStateCount: hiddenStateCount,
            body: { [self] input in
                CBv2OrderOnlyLogits.withGreedyOrderOnly {
                    let forward = model.callCapturingValidatedDFlashHiddenStates(
                        input,
                        cache: cache,
                        targetLayerIds: targetLayerIds,
                        forceArrayMask: true,
                        verifier: nil)
                    let tokens = applyLMHead(forward.postNorm).argMax(axis: -1)
                    return [tokens] + forward.hiddenStates
                }
            })
    }

    /// The session constructor proves every installed target-cache entry has
    /// a direct recent-tail rollback route. No per-round snapshot is needed.
    public func makeDFlashCacheRollbackState(
        cache: [KVCache]
    ) -> (any DFlashTargetRollbackState)? {
        nil
    }

    /// Direct Gemma 4 rollback matching the reference DFlash implementation.
    /// Saturated sliding-window rings restore temporal order before dropping
    /// rejected rows; full-attention caches use their ordinary tail trim.
    public func rollbackDFlashCache(
        _ cache: inout [KVCache],
        state: (any DFlashTargetRollbackState)?,
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        let acceptedHidden =
            verifiedTargetHidden[0..., 0 ..< acceptedTokenCount + 1, 0...]
        guard rejectedTokenCount > 0 else { return acceptedHidden }
        cache.forEach { _ = $0.trimRecent(rejectedTokenCount) }
        return acceptedHidden
    }

    /// Route only on the physical B1/C2-C4/C8/C16 shape that genuinely changes per
    /// round. Model topology, quantization, storage layout, and every projection
    /// binding were already validated atomically by `installCBv2MTPVerifier()`
    /// during target construction; the measured path reads the immutable table.
    private func dFlashInstalledVerifier(
        for inputs: MLXArray
    ) -> Gemma4MTPVerifierContext? {
        cbv2MTPVerifierContext(batch: inputs.dim(0), columns: inputs.dim(1))
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
