// PrefillOutputV2.swift
//
// The prompt-side output contract for ContinuousBatchingV2.
//
// CBv2 consumes exactly ONE position from a prompt chunk: the frontier
// chunk's last hidden row (which samples the first generated token).
// Intermediate chunks contribute nothing but their KV writes. The default
// `CBv2SteppableModel.forward` nevertheless projects EVERY prompt position
// through the vocabulary head — for Gemma 4 that is a [B, chunk, 262144]
// tensor built and then discarded, per chunk, per request.
//
// This file adds an OPT-IN refinement. Models that can honor it return only
// what the engine needs; models that cannot are untouched and the engine
// slices their full logits exactly as before (`EngineLoopV2.prefillOutput`).
//
// Invariants a conforming model MUST preserve, for every requirement:
//  - process EVERY input token,
//  - perform EVERY K/V write and advance every cache offset,
//  - keep attention, masks, positions, and multimodal spans identical to
//    `forward`.
// Only the UNUSED vocabulary projection (and, for models that specialize
// further, unused token-local work on discarded rows) may be skipped.
//
// Decode is explicitly out of scope: `decodeLogits`, the compiled [B, 1]
// path, and MTP target verification never route through this seam.

import Foundation
import MLX

public enum CBv2PrefillRequirement: Sendable, Equatable {
    case evaluationOnly
    case lastPositionLogits
}

public protocol CBv2PrefillSteppableModel: CBv2SteppableModel {
    func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray
}

public protocol CBv2PackedPrefillSteppableModel: CBv2PrefillSteppableModel {
    var supportsPackedPrefill: Bool { get }
    var supportsPackedMultimodalPrefill: Bool { get }
}

extension CBv2PackedPrefillSteppableModel {
    public var supportsPackedMultimodalPrefill: Bool { false }
}

public protocol CBv2LanguageModelPrefillForwardable {
    func cbv2Prefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray

    var cbv2SupportsPackedPrefill: Bool { get }

    var cbv2SupportsPackedMultimodalPrefill: Bool { get }
}

extension CBv2LanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var cbv2SupportsPackedMultimodalPrefill: Bool { false }
}
