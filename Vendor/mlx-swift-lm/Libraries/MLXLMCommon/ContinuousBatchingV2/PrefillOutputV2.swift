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

/// What a prompt chunk's forward actually has to return.
///
/// - `evaluationOnly`: an intermediate chunk. The engine only needs a small
///   nonempty tensor whose graph transitively depends on the full trunk, so
///   that evaluating it commits the chunk's KV writes. No vocabulary
///   projection is required.
/// - `lastPositionLogits`: the chunk reaches the prompt frontier. The engine
///   needs `[B, vocab]` for the final position only — never the full
///   `[B, L, vocab]`.
public enum CBv2PrefillRequirement: Sendable, Equatable {
    case evaluationOnly
    case lastPositionLogits
}

/// Opt-in prompt-forward refinement for steppable models.
///
/// `inputEmbeddings` is non-nil only for a multimodal chunk whose image
/// soft-token embeddings were spliced over the text embeddings; conformers
/// must then take the same embedding path `CBv2MultimodalSteppableModel
/// .forward(tokens:inputEmbeddings:caches:)` would take.
public protocol CBv2PrefillSteppableModel: CBv2SteppableModel {
    func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray
}

/// Opt-in refinement for models whose prompt forward keeps per-row
/// semantics when several EQUAL-LENGTH chunks are executed as one
/// rectangular `[B, L]` pass. This makes the transformer traversal
/// layer-major across those rows: each layer's weights are read once for
/// the whole cohort instead of once per row.
///
/// This is a claim about the MODEL only. The engine additionally requires
/// the cache provider to vouch that its layer caches keep independent rows
/// (`CBv2LayerCacheProvider.supportsPackedPrefill`). Packing rows that splice
/// image embeddings and carry row-local span masks requires the stronger,
/// separately fail-closed `supportsPackedMultimodalPrefill` claim.
public protocol CBv2PackedPrefillSteppableModel: CBv2PrefillSteppableModel {
    var supportsPackedPrefill: Bool { get }
    var supportsPackedMultimodalPrefill: Bool { get }
}

extension CBv2PackedPrefillSteppableModel {
    public var supportsPackedMultimodalPrefill: Bool { false }
}

/// Model-level (KVCache-shaped) twin of `CBv2PrefillSteppableModel`, for
/// `LanguageModel` conformers reached through
/// `CBv2SteppableLanguageModelAdapter` — the same indirection
/// `CBv2EmbeddingForwardable` uses for multimodal prefill.
public protocol CBv2LanguageModelPrefillForwardable {
    func cbv2Prefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray

    /// Whether this model's prompt forward is safe to run as a rectangular
    /// `[B, L]` cohort of independent rows. Fail-closed default: false.
    var cbv2SupportsPackedPrefill: Bool { get }

    /// Stronger claim for rectangular embedding-forward rows with one
    /// optional vision span-mask context per row. Fail-closed default: false.
    var cbv2SupportsPackedMultimodalPrefill: Bool { get }
}

extension CBv2LanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var cbv2SupportsPackedMultimodalPrefill: Bool { false }
}

public enum CBv2PrefillLogitDistributionGuard {
    /// Asserts that prefill logit tensors remain full-rank, unsuppressed, and bit-exact.
    public static func assertPrefillLogitIntegrity(logits: MLXArray) {
        guard !CBv2OrderOnlyLogits.engaged else {
            fatalError("CBv2OrderOnlyLogits must not suppress full logit tensors during evaluation or reference replay")
        }
        precondition(logits.ndim >= 2, "Prefill logits must have at least [B, vocab] rank")
    }
}

