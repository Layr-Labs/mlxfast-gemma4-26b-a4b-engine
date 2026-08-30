// SteppableAdapterV2.swift
//
// Bridges WS-F's model v2 branches into the engine's `CBv2SteppableModel`
// seam. Gemma 4 and GPT-OSS detect CBv2 layer caches through their existing
// `callAsFunction(_:cache:)` entry points (the caches conform to the legacy
// `KVCache` protocol with trapping legacy methods), so the adapter is a
// thin cast-and-forward: no model file changes, no mask construction, no
// padding — the v2 branch inside the model owns the dispatch.
//
// Cache construction stays with `model.newCacheV2(makeLayerCache:)` (the
// single entry point — GPT-OSS primes its sinks-activation probe there);
// wrap the result in a `CBv2LayerCacheBank` for the engine.

import Foundation
import MLX

/// `CBv2SteppableModel` over any `LanguageModel` whose forward path
/// understands `CBv2AttendingLayerCache` (Gemma 4, GPT-OSS, test fixtures).
public final class CBv2SteppableLanguageModelAdapter: CBv2SteppableModel {

    private let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }

    public func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        return model(tokens, cache: asKVCaches(caches))
    }

    private func asKVCaches(_ caches: [CBv2AttendingLayerCache]) -> [KVCache] {
        caches.map { cache -> KVCache in
            guard let kv = cache as? KVCache else {
                fatalError(
                    "CBv2 layer cache \(type(of: cache)) must conform to KVCache to drive "
                        + "\(type(of: model)) through callAsFunction(_:cache:)")
            }
            return kv
        }
    }
}

// MARK: - Prompt-output narrowing (prefill only)

/// Answered at RUNTIME like the multimodal/MTP capabilities: only models
/// conforming to `CBv2LanguageModelPrefillForwardable` (Gemma4TextModel) can
/// narrow their prompt output. Everything else keeps the full-logits
/// `forward` contract and is sliced by the engine, so this conformance can
/// never change what a non-conforming model computes.
extension CBv2SteppableLanguageModelAdapter: CBv2PackedPrefillSteppableModel {

    public var supportsPackedPrefill: Bool {
        (model as? CBv2LanguageModelPrefillForwardable)?.cbv2SupportsPackedPrefill ?? false
    }

    public var supportsPackedMultimodalPrefill: Bool {
        (model as? CBv2LanguageModelPrefillForwardable)?
            .cbv2SupportsPackedMultimodalPrefill ?? false
    }

    public func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        guard let prefillable = model as? CBv2LanguageModelPrefillForwardable else {
            // Fail SAFE, not fatal: reproduce `forward` + the engine's own
            // slicing. `EngineLoopV2.prefillOutput` only routes here after a
            // successful cast, so this is belt-and-braces for direct callers.
            let logits: MLXArray
            if let inputEmbeddings {
                logits = forward(
                    tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
            } else {
                logits = forward(tokens: tokens, caches: caches)
            }
            switch requirement {
            case .evaluationOnly: return logits[0..., -1, 0 ..< 1]
            case .lastPositionLogits: return logits[0..., -1, 0...]
            }
        }
        return prefillable.cbv2Prefill(
            tokens,
            inputEmbedding: inputEmbeddings,
            cache: asKVCaches(caches),
            requirement: requirement)
    }
}

// MARK: - Multimodal (vision prefill)

/// The adapter answers the multimodal capability at RUNTIME: it can wrap any
/// `LanguageModel`, and only models conforming to `CBv2EmbeddingForwardable`
/// (Gemma4TextModel) can prefill from spliced embeddings. Conformance alone
/// is structural, not capability — a Gemma4TextModel loaded from a TEXT-ONLY
/// config (`use_bidirectional_attention` nil/non-`vision`) can execute the
/// embedding forward but was never trained for the bidirectional span masks
/// CBv2 applies, so the capability check also consults the model-level
/// `supportsVisionSpanPrefill` flag (PR#63 review). Requests against
/// non-conforming models or unsupported configs are rejected at submit
/// (`CBv2MultimodalError.unsupportedModel`), so the trapping guards below
/// are unreachable in a correctly gated engine.
extension CBv2SteppableLanguageModelAdapter: CBv2MultimodalSteppableModel {

    public var supportsMultimodalPrefill: Bool {
        (model as? CBv2EmbeddingForwardable)?.supportsVisionSpanPrefill ?? false
    }

    public func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        guard let embeddable = model as? CBv2EmbeddingForwardable else {
            preconditionFailure(
                "CBv2 multimodal: \(type(of: model)) is not CBv2EmbeddingForwardable — submit gating failed"
            )
        }
        return embeddable.scaledInputEmbeddings(tokens)
    }

    public func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        guard let embeddable = model as? CBv2EmbeddingForwardable else {
            preconditionFailure(
                "CBv2 multimodal: \(type(of: model)) is not CBv2EmbeddingForwardable — submit gating failed"
            )
        }
        return embeddable.embeddingForward(
            tokens, inputEmbedding: inputEmbeddings, cache: asKVCaches(caches))
    }
}

// MARK: - MTP (speculative decoding)

/// Answered at RUNTIME like the multimodal capability: the adapter wraps any
/// `LanguageModel`, and only `CBv2MTPForwardable` conformers (Gemma4TextModel)
/// can drive MTP rounds. The engine gates speculation on
/// `mtpCaptureLayers != nil` before ever calling `forwardWithHidden`, so the
/// trapping guard below is unreachable in a correctly gated engine.
extension CBv2SteppableLanguageModelAdapter: CBv2MTPSteppableModel {

    public var mtpCaptureLayers: CBv2MTPCaptureLayers? {
        (model as? CBv2MTPForwardable)?.cbv2MTPCaptureLayers
    }

    public var mtpTargetIdentity: ObjectIdentifier? {
        guard let target = model as? any CBv2MTPForwardable else { return nil }
        return ObjectIdentifier(target)
    }

    public func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        guard let forwardable = model as? CBv2MTPForwardable else {
            preconditionFailure(
                "CBv2 MTP: \(type(of: model)) is not CBv2MTPForwardable — engine gating failed")
        }
        return forwardable.cbv2ForwardWithHidden(tokens, caches: asKVCaches(caches))
    }
}
