// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Abstraction over a Gemma 4 text tower that can drive MTP speculative
/// decoding.
///
/// ``Gemma4TextModel`` is the canonical target for text-only loads and for
/// MLXVLM Gemma 4: the VLM owns and exposes this exact object as `textModel`.
/// Single-stream and CBv2 MTP therefore bind to the same text architecture,
/// weights, hidden capture, and cache identity used by direct VLM forwards.
public protocol Gemma4MTPTarget: AnyObject {

    /// The resolved text configuration, used for drafter-compatibility
    /// validation and the automatic block-size policy.
    var mtpConfiguration: Gemma4TextConfiguration { get }

    /// Allocate a fresh set of KV caches (one per non-shared layer) matching
    /// the tower's layer layout.
    func mtpNewCache(parameters: GenerateParameters?) -> [any KVCache]

    /// Scaled input embedding for `tokens` (`embed(tokens) * sqrt(hidden)`),
    /// the "target embedding" the drafter concatenates with the last hidden
    /// to build its per-step input.
    func embedTokensForDrafter(_ tokens: MLXArray) -> MLXArray

    /// Forward pass tailored for MTP: returns logits, the pre-norm last
    /// hidden, and the shared-KV snapshot the drafter consumes next round.
    func forwardForMTP(_ tokens: MLXArray, cache: [KVCache]) -> Gemma4MTPForward

    /// Rewind KV caches after a speculative round (uniform suffix trim, plus
    /// per-row zeroing for batched caches).
    func rollbackSpeculativeCache(
        _ caches: [KVCache], accepted: Gemma4AcceptCount, blockSize: Int)
}

// MARK: - Shared text-tower conformance

extension Gemma4TextModel: Gemma4MTPTarget {
    public var mtpConfiguration: Gemma4TextConfiguration { configuration }

    public func mtpNewCache(parameters: GenerateParameters?) -> [any KVCache] {
        newCache(parameters: parameters)
    }

    // `embedTokensForDrafter`, `forwardForMTP`, and `rollbackSpeculativeCache`
    // are declared directly on `Gemma4TextModel` with matching signatures, so
    // they satisfy the protocol without further work.
}

// MARK: - Construction-bound CBv2 verifier

/// Runtime module paths promoted to eight-bit in the shipped target.  The set
/// is derived from the certified layer count so a missing or extra promoted
/// projection prevents installation instead of becoming a decode-time check.
private func gemma4MTPVerifierOverridePaths() -> Set<String> {
    var paths: Set<String> = []
    for layer in 0..<30 {
        for family in ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "router.proj"] {
            paths.insert("model.layers.\(layer).\(family)")
        }
    }
    return paths
}

/// Certify the materialized target once, while the adapter is being built.
/// The installed route consequently owns only quantized affine leaves with the
/// exact width table it was designed around; no metadata inspection or stock
/// fallback remains in the measured forward.
private func gemma4CanInstallBatch8Depth1Verifier(_ target: Gemma4TextModel) -> Bool {
    guard gemma4SupportsBatch8Depth1MTPVerifier(target.configuration) else { return false }

    let overrides = gemma4MTPVerifierOverridePaths()
    var foundOverrides: Set<String> = []
    var foundEmbedding = false
    var foundQuantizedLinear = false
    var foundQuantizedExperts = false

    for (path, module) in target.leafModules().flattened() {
        if module is Linear && !(module is QuantizedLinear) { return false }
        if module is SwitchLinear && !(module is QuantizedSwitchLinear) { return false }
        if module is Embedding && !(module is QuantizedEmbedding) { return false }

        guard let quantized = module as? Quantized else { continue }
        let isOverride = overrides.contains(path)
        let expectedBits = isOverride ? 8 : 4
        guard quantized.bits == expectedBits,
            quantized.groupSize == 64,
            quantized.mode == .affine
        else { return false }

        let parameters = Dictionary(
            module.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        guard parameters["weight"] != nil,
            parameters["scales"] != nil,
            parameters["biases"] != nil
        else { return false }

        if isOverride { foundOverrides.insert(path) }
        if path == "model.embed_tokens", module is QuantizedEmbedding {
            foundEmbedding = true
        }
        if module is QuantizedLinear { foundQuantizedLinear = true }
        if module is QuantizedSwitchLinear { foundQuantizedExperts = true }
    }

    return foundOverrides == overrides
        && foundEmbedding
        && foundQuantizedLinear
        && foundQuantizedExperts
}

/// Replace all dense quantized leaves as one module-tree update after the
/// complete inventory has succeeded. A failed candidate leaves the target
/// untouched; the enabled M16 call therefore needs no eligibility fallback.
private func gemma4InstallMTPVerifyLinears(_ target: Gemma4TextModel) -> Bool {
    var updates: [(String, Module)] = []
    for (path, module) in target.leafModules().flattened() {
        guard let linear = module as? QuantizedLinear else { continue }
        guard let installed = Gemma4MTPVerifyQuantizedLinear.install(linear) else {
            return false
        }
        updates.append((path, installed))
    }
    guard !updates.isEmpty else { return false }
    target.update(modules: ModuleChildren.unflattened(updates))
    return true
}

private final class Gemma4Batch8Depth1Verifier: CBv2MTPInstalledVerifier {
    let geometry = CBv2MTPVerificationGeometry(batchSize: 8, draftDepth: 1)
    unowned let target: Gemma4TextModel
    let tiedHead: Gemma4MTPVerifyTiedHead

    init(target: Gemma4TextModel, tiedHead: Gemma4MTPVerifyTiedHead) {
        self.target = target
        self.tiedHead = tiedHead
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [KVCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(tokens.ndim == 2 && tokens.shape == [8, 2])
        let hidden = target.cbv2MTPVerifyHiddenStates(tokens, caches: caches)
        let logits = target.cbv2MTPVerifySoftcap(tiedHead(hidden.postNorm))
        return (logits, hidden.preNorm)
    }
}

extension Gemma4TextModel: CBv2MTPVerifierInstallable {
    public func installCBv2MTPVerifier() -> (any CBv2MTPInstalledVerifier)? {
        guard gemma4CanInstallBatch8Depth1Verifier(self),
            let tiedHead = makeMTPVerifyTiedHead(),
            gemma4InstallMTPVerifyLinears(self)
        else { return nil }
        return Gemma4Batch8Depth1Verifier(target: self, tiedHead: tiedHead)
    }
}
