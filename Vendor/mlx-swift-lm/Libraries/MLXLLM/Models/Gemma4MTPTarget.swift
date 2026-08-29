// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

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
