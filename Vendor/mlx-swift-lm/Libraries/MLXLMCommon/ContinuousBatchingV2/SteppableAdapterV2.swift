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

public protocol CBv2LanguageModelDecodeOutputCoversCacheMutations: AnyObject {}

private let cbv2CompactDecodeRootMarksArmed =
    ProcessInfo.processInfo.environment["MLXFAST_ENGAGE_MARKS"] != nil

public final class CBv2SteppableLanguageModelAdapter: CBv2SteppableModel {

    private let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }

    public func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        return model(tokens, cache: asKVCaches(caches))
    }

    public func compactDecodeEvaluationRoots(
        forwardOutput: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> [MLXArray]? {
        guard model is any CBv2LanguageModelDecodeOutputCoversCacheMutations else {
            return nil
        }
        let contiguous = caches.compactMap { $0 as? CBv2LayerCache }
        guard !contiguous.isEmpty,
            contiguous.count == caches.count,
            contiguous.allSatisfy({ $0.kind.sharesKVWithLayer == nil }),
            let rowCount = contiguous.first?.rows.count,
            rowCount > 0,
            contiguous.allSatisfy({ $0.rows.count == rowCount }),
            contiguous.allSatisfy({ cache in
                cache.rows.allSatisfy {
                    $0 is any CBv2DecodeRootCompactionCapableSequenceKV
                }
            }),
            let stateIdentity = contiguous[0].unifiedPositionStateIdentity,
            let offsets = contiguous[0].unifiedPositionOffsets,
            contiguous.dropFirst().allSatisfy({
                $0.unifiedPositionStateIdentity == stateIdentity
            })
        else { return nil }

        var roots = [forwardOutput, offsets]
        roots.reserveCapacity(2 + contiguous.count)
        roots.append(contentsOf: contiguous.map(\.decodeRingWriteFenceEvaluationRoot))
        if cbv2CompactDecodeRootMarksArmed {
            CBv2EngageMark.once(
                "compact-decode-roots rows=\(rowCount) layers=\(contiguous.count) "
                    + "roots=\(roots.count)")
        }
        return roots
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

// MARK: - Logitsless greedy head (LGH-001)

/// Answered at RUNTIME like the multimodal/MTP capabilities: only
/// `CBv2ArgmaxDecodeForwardable` conformers can end a decode step in a fused
/// top-1, and `admitsArgmaxDecode` gates every call, so a non-conforming model
/// keeps the full-logits `forward` contract untouched.
extension CBv2SteppableLanguageModelAdapter: CBv2ArgmaxDecodeSteppableModel {

    public func admitsArgmaxDecode(tokens: MLXArray) -> Bool {
        (model as? CBv2ArgmaxDecodeForwardable)?.cbv2AdmitsArgmaxDecode(tokens) ?? false
    }

    public func decodeArgmax(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        guard let forwardable = model as? CBv2ArgmaxDecodeForwardable else {
            preconditionFailure(
                "CBv2 argmax decode: \(type(of: model)) is not CBv2ArgmaxDecodeForwardable "
                    + "— engine gating failed")
        }
        return forwardable.cbv2DecodeArgmax(tokens, caches: asKVCaches(caches))
    }
}
