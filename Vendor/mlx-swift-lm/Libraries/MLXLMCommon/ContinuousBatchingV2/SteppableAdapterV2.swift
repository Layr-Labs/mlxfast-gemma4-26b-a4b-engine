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

/// Model-level proof that an ordinary CBv2 decode output transitively
/// consumes every K/V mutation performed by that forward. This is deliberately
/// narrower than `LanguageModel`: an adapter cannot infer the dependency from
/// a cache type alone.
public protocol CBv2LanguageModelDecodeOutputCoversCacheMutations: AnyObject {}

/// Local engagement diagnostic for the decode-root compaction, armed only by
/// `MLXFAST_ENGAGE_MARKS` (the ranked runner sets no environment). Reading one
/// static Bool keeps the disarmed path allocation-free.
private let cbv2CompactDecodeRootMarksArmed =
    ProcessInfo.processInfo.environment["MLXFAST_ENGAGE_MARKS"] != nil

/// `CBv2SteppableModel` over any `LanguageModel` whose forward path
/// understands `CBv2AttendingLayerCache` (Gemma 4, GPT-OSS, test fixtures).
public final class CBv2SteppableLanguageModelAdapter: CBv2SteppableModel {

    private let model: any LanguageModel
    private let supportsCompactDecodeRoots: Bool

    private struct KVCachesTopology {
        let identities: [ObjectIdentifier]
        let values: [KVCache]

        func matches(_ caches: [CBv2AttendingLayerCache]) -> Bool {
            guard identities.count == caches.count else { return false }
            for index in caches.indices {
                if identities[index] != ObjectIdentifier(caches[index]) { return false }
            }
            return true
        }
    }

    private struct CompactDecodeTopology {
        let identities: [ObjectIdentifier]
        let revisions: [UInt64]
        let layers: [CBv2LayerCache]
        let rowCount: Int

        func matches(_ caches: [CBv2AttendingLayerCache]) -> Bool {
            guard identities.count == caches.count else { return false }
            for index in caches.indices {
                if identities[index] != ObjectIdentifier(caches[index])
                    || revisions[index] != layers[index].decodeRootTopologyRevision
                {
                    return false
                }
            }
            return true
        }
    }

    private var kvCachesTopology: KVCachesTopology?
    private var compactDecodeTopology: CompactDecodeTopology?

    public init(_ model: any LanguageModel) {
        self.model = model
        self.supportsCompactDecodeRoots =
            model is any CBv2LanguageModelDecodeOutputCoversCacheMutations
    }

    public func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        return model(tokens, cache: asKVCaches(caches))
    }

    /// Initial fail-closed decode compaction: only an affirming model over an
    /// all-owning, all-contiguous bank with one shared position state. The
    /// output root forces every K/V mutation; the post-forward position root
    /// collapses the next-step offset chain; per-layer fences conservatively
    /// retain explicit ordering for fused in-place sliding-ring writes.
    public func compactDecodeEvaluationRoots(
        forwardOutput: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> [MLXArray]? {
        guard supportsCompactDecodeRoots else { return nil }

        let contiguous: [CBv2LayerCache]
        let rowCount: Int
        if let cached = compactDecodeTopology, cached.matches(caches) {
            contiguous = cached.layers
            rowCount = cached.rowCount
        } else {
            let validated = caches.compactMap { $0 as? CBv2LayerCache }
            guard !validated.isEmpty,
                validated.count == caches.count,
                validated.allSatisfy({ $0.kind.sharesKVWithLayer == nil }),
                let validatedRowCount = validated.first?.rows.count,
                validatedRowCount > 0,
                validated.allSatisfy({ $0.rows.count == validatedRowCount }),
                validated.allSatisfy({ cache in
                    cache.rows.allSatisfy {
                        $0 is any CBv2DecodeRootCompactionCapableSequenceKV
                    }
                }),
                let stateIdentity = validated[0].unifiedPositionStateIdentity,
                validated.dropFirst().allSatisfy({
                    $0.unifiedPositionStateIdentity == stateIdentity
                })
            else {
                compactDecodeTopology = nil
                return nil
            }
            contiguous = validated
            rowCount = validatedRowCount
            compactDecodeTopology = CompactDecodeTopology(
                identities: caches.map(ObjectIdentifier.init),
                revisions: validated.map(\.decodeRootTopologyRevision),
                layers: validated,
                rowCount: validatedRowCount)
        }

        guard let offsets = contiguous[0].unifiedPositionOffsets else { return nil }

        var roots = [forwardOutput, offsets]
        roots.reserveCapacity(2 + contiguous.count)
        for cache in contiguous {
            roots.append(cache.decodeRingWriteFenceEvaluationRoot)
        }
        if cbv2CompactDecodeRootMarksArmed {
            CBv2EngageMark.once(
                "compact-decode-roots rows=\(rowCount) layers=\(contiguous.count) "
                    + "roots=\(roots.count)")
        }
        return roots
    }

    private func asKVCaches(_ caches: [CBv2AttendingLayerCache]) -> [KVCache] {
        if let cached = kvCachesTopology, cached.matches(caches) {
            return cached.values
        }

        var identities: [ObjectIdentifier] = []
        var values: [KVCache] = []
        identities.reserveCapacity(caches.count)
        values.reserveCapacity(caches.count)
        for cache in caches {
            guard let kv = cache as? KVCache else {
                fatalError(
                    "CBv2 layer cache \(type(of: cache)) must conform to KVCache to drive "
                        + "\(type(of: model)) through callAsFunction(_:cache:)")
            }
            identities.append(ObjectIdentifier(cache))
            values.append(kv)
        }
        kvCachesTopology = KVCachesTopology(identities: identities, values: values)
        return values
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
