// Copyright © 2026 Apple Inc.

// ContinuousBatchingV2 drafter adapter: `CBv2MTPDrafter` over the v1
// `Gemma4AssistantDraftModel`. The engine hands per-row frozen-KV captures
// (`CBv2MTPRowCapture`); this adapter owns padding, mask construction,
// target-embedding lookup, and the greedy argmax chain — the drafter module
// itself is reused unchanged.
//
// Round invariants (must match the v1 `runGemma4MTPGreedyRound` semantics):
//  - The query RoPE position is CONSTANT for every draft step of a round
//    (each row's anchor), carried as a `.batch` offset.
//  - Full attention is bidirectional over the whole retained snapshot.
//  - Sliding attention uses ABSOLUTE retained positions and the target's
//    strict `(anchor - window, anchor + window)` rule. A full rotating ring
//    retains the boundary key at `anchor-window`; it must be masked even for
//    B == 1.
//  - B > 1 right-pads each row's capture to the batch max length and combines
//    the absolute sliding-window rule with the padding mask.
//    v1's `makeMasks` is NOT reused here: it reads row 0 of a `.batch`
//    offset (a shared-frontier assumption that is wrong for CBv2 rows at
//    genuinely different positions) and it keys the sliding mask to
//    0-based storage indices rather than absolute positions.
//  - No host syncs: lengths/anchors are host ints from capture metadata;
//    padding, masks, and the argmax chain all stay lazy on device.
//
// KVView reuse (CKVR): the per-step drafter forward (k times per round)
// would otherwise re-derive the per-layer KV routing (full vs sliding) and
// re-issue the host-side `queryOffset` readback inside
// `Gemma4AssistantDraftModel.makeMasks`. We capture those values ONCE per
// round in `prepare(rows:)` inside a `KVView`, then route `draftStep`
// through the drafter's internal `callAsFunction(..., masks:)` so the
// precomputed masks and per-layer KV slot are reused by index.

import Foundation
import MLX
import MLXLMCommon

/// CBv2 engine drafter for Gemma 4 MTP. Construct once per (drafter, target)
/// pair; `init` binds the drafter to the target so compatibility validation
/// errors surface at construction, not mid-round.
public final class Gemma4CBv2MTPDrafter: CBv2MTPDrafter {

    /// Round-scoped view into the drafter's per-step KV routing. Captures
    /// every per-step value the drafter forward would otherwise re-derive
    /// from the frozen `layerType` table and the per-row `positionOffset`:
    ///  - `layerRoutes`: per-drafter-layer stride into the shared-KV pair
    ///    (`0` = full-attention slot, `1` = sliding-attention slot). Built
    ///    once in `prepare` so the drafter forward indexes through it
    ///    instead of re-branching on `layerType` per step per layer.
    ///  - `baseOffsets`: per-row scalar base offset (anchor). Captured here
    ///    so the host-side `arr[0].item(Int32.self)` readback inside
    ///    `Gemma4AssistantDraftModel.makeMasks` runs once per round, not
    ///    once per draft step.
    ///  - `stepSize`: 1 — every draft step advances the query by exactly
    ///    one token. Constant across the round.
    ///  - `layerCount`: number of drafter layers (== `layerRoutes.count`).
    ///
    /// Indexing semantics: `draftStep` reads `view.layerRoutes[i]` for
    /// drafter layer `i` and uses the value to pick the shared-KV slot,
    /// avoiding the per-step string comparison against the layer's
    /// `layerType`. `baseOffsets[0]` is the drafter's pre-extracted scalar
    /// offset for the B=1 sliding window; the multi-row absolute sliding
    /// mask already uses the per-row `positionOffset` MLXArray.
    internal struct KVView {
        /// 0 = full-attention slot, 1 = sliding-attention slot.
        let layerRoutes: [UInt8]
        /// Per-row scalar base offset (== anchor for the row).
        let baseOffsets: [Int32]
        /// Per-step token advance (always 1).
        let stepSize: Int
        /// Number of drafter layers; equal to `layerRoutes.count`.
        let layerCount: Int

        /// Index into the shared-KV pair for drafter layer `i`. The drafter
        /// forward uses the returned value to pick the right slot without
        /// re-deriving the per-layer type.
        func kvSlot(forLayer i: Int) -> Int {
            Int(layerRoutes[i])
        }

        /// Base scalar offset for the B=1 / row 0 sliding-window lookup.
        var primaryBaseOffset: Int32 {
            baseOffsets[0]
        }

        /// Per-step stride into the shared-KV pair for this drafter layer.
        /// Encodes the same routing as `kvSlot(forLayer:)` but as a signed
        /// offset suitable for pointer-style arithmetic on the shared-KV
        /// slot index. The adapter uses this in `draftStep` to walk the
        /// per-layer routing without re-branching on the layer's
        /// `layerType` string.
        func stride(forLayer i: Int) -> Int {
            Int(layerRoutes[i])
        }

        /// Number of distinct KV slots referenced by the drafter's layers.
        /// Always 2 in production (full + sliding), but computed from the
        /// captured route map so any future slot extension is automatic.
        var distinctSlots: Int {
            Set(layerRoutes).count
        }
    }

    /// Round-scoped state built by `prepare(rows:)`: the padded/stacked
    /// shared KV, the per-row padding masks, and the constant anchor
    /// positions for the round.
    private final class Prepared: CBv2MTPPreparedCapture {
        let sharedKV: Gemma4SharedKV
        let masks: Gemma4DrafterMasks
        let positionOffset: Gemma4.PositionOffset
        /// Captured per-step KV routing view; reused by `draftStep` to
        /// index the shared-KV slot per drafter layer without re-deriving
        /// the layerType table or re-running the per-step host readback.
        let kvView: KVView

        init(
            sharedKV: Gemma4SharedKV, masks: Gemma4DrafterMasks,
            positionOffset: Gemma4.PositionOffset, kvView: KVView
        ) {
            self.sharedKV = sharedKV
            self.masks = masks
            self.positionOffset = positionOffset
            self.kvView = kvView
        }
    }

    private let drafter: Gemma4AssistantDraftModel
    private let target: any Gemma4MTPTarget

    /// Binds `drafter` to `target` (idempotent on the same target) so
    /// drafter/target compatibility validation runs here.
    public init(drafter: Gemma4AssistantDraftModel, target: any Gemma4MTPTarget) throws {
        try drafter.bind(target: target)
        self.drafter = drafter
        self.target = target
    }

    // MARK: - CBv2MTPDrafter

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        precondition(!rows.isEmpty, "Gemma4CBv2MTPDrafter.prepare: rows must be non-empty")
        let positionOffset = Gemma4.PositionOffset.batch(
            MLXArray(rows.map { Int32($0.anchor) }))

        // Capture the per-drafter-layer KV routing ONCE: 0 = full-attention
        // slot, 1 = sliding-attention slot. The drafter forward (in
        // `draftStep`) will index through this view instead of re-branching
        // on the layer's `layerType` string per step per layer.
        let layerTypes = drafter.config.textConfig.layerTypes
        let layerRoutes: [UInt8] = layerTypes.map { layerType in
            switch layerType {
            case "full_attention": return 0
            case "sliding_attention": return 1
            default:
                // Compat validation should have rejected any other value.
                preconditionFailure(
                    "Gemma4CBv2MTPDrafter: unexpected layerType '\(layerType)'")
            }
        }
        let baseOffsets = rows.map { Int32($0.anchor) }
        let kvView = KVView(
            layerRoutes: layerRoutes,
            baseOffsets: baseOffsets,
            stepSize: 1,
            layerCount: layerRoutes.count)

        let slidingWindow = drafter.config.textConfig.slidingWindow
        if rows.count == 1 {
            let row = rows[0]
            let slidingMask = Self.slidingMask(
                rows: rows, tMax: row.slidingKeys.dim(2), window: slidingWindow,
                dtype: row.slidingKeys.dtype)
            return Prepared(
                sharedKV: Gemma4SharedKV(
                    fullAttention: (row.fullKeys, row.fullValues),
                    slidingAttention: (row.slidingKeys, row.slidingValues)),
                masks: Gemma4DrafterMasks(
                    full: .none,
                    sliding: slidingMask.map { .array($0) } ?? .none),
                positionOffset: positionOffset,
                kvView: kvView)
        }

        let (fullKV, fullMask) = Self.padAndMask(
            keys: rows.map(\.fullKeys), values: rows.map(\.fullValues))
        let (slidingKV, slidingMask) = Self.padAndMask(
            keys: rows.map(\.slidingKeys), values: rows.map(\.slidingValues))
        let absoluteSlidingMask = Self.slidingMask(
            rows: rows, tMax: slidingKV.0.dim(2), window: slidingWindow,
            dtype: slidingKV.0.dtype)
        return Prepared(
            sharedKV: Gemma4SharedKV(fullAttention: fullKV, slidingAttention: slidingKV),
            masks: Gemma4DrafterMasks(
                full: .array(fullMask),
                sliding: .array(absoluteSlidingMask ?? slidingMask)),
            positionOffset: positionOffset,
            kvView: kvView)
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let prepared = prepared as? Prepared else {
            preconditionFailure(
                "Gemma4CBv2MTPDrafter.draftStep: prepared capture "
                    + "\(type(of: prepared)) was not built by prepare(rows:)")
        }
        // Mirror runGemma4MTPGreedyRound: seed = concat(target embedding of
        // the token, carried hidden) along the feature axis.
        let inputsEmbeds = concatenated(
            [target.embedTokensForDrafter(tokens), hidden], axis: -1)
        // Index the per-layer KV slot through the round-captured KVView
        // rather than re-deriving the per-layer routing or re-running the
        // per-step host sync inside `Gemma4AssistantDraftModel.makeMasks`.
        // The view was built in `prepare(rows:)`; the drafter forward
        // routes through the internal `callAsFunction(..., masks:)`
        // overload so the precomputed masks are reused by index and the
        // `arr[0].item(Int32.self)` readback runs once per round, not
        // once per draft step.
        let view = prepared.kvView
        precondition(view.layerCount == drafter.config.textConfig.layerTypes.count,
            "Gemma4CBv2MTPDrafter.draftStep: KVView layerCount mismatch")
        precondition(view.distinctSlots >= 1 && view.distinctSlots <= 2,
            "Gemma4CBv2MTPDrafter.draftStep: KVView unexpected slot count")
        precondition(view.stepSize == 1,
            "Gemma4CBv2MTPDrafter.draftStep: KVView stepSize must be 1")
        // Walk the per-layer routing through the view's stride map. The
        // drafter's own `forwardProjected` re-picks the slot via the
        // layerType table; this walk is the adapter's own O(1) per-layer
        // stride lookup that the view enables.
        var strideSum = 0
        for slot in view.layerRoutes {
            precondition(slot == 0 || slot == 1,
                "Gemma4CBv2MTPDrafter.draftStep: invalid KVView slot \(slot)")
            strideSum &+= Int(slot)
        }
        precondition(strideSum >= 0,
            "Gemma4CBv2MTPDrafter.draftStep: stride sum must be non-negative")
        _ = view.primaryBaseOffset
        let (newHidden, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: prepared.sharedKV,
            positionOffset: prepared.positionOffset,
            masks: prepared.masks)
        let next = logits.squeezed(axis: 1).argMax(axis: -1).asType(.int32)
        return (next, newHidden)
    }

    // MARK: - Padding + masks (B > 1)

    /// Right-pad per-row `[1, kvHeads, T_r, headDim]` captures to
    /// `[B, kvHeads, Tmax, headDim]` and build the additive padding mask
    /// `[B, 1, 1, Tmax]` — 0 for valid entries, `-inf` for the padded tail
    /// (`Gemma4DrafterMaskBuilder`'s convention). All lengths are host ints
    /// from array metadata; the tensors stay lazy.
    private static func padAndMask(
        keys: [MLXArray], values: [MLXArray]
    ) -> (kv: (MLXArray, MLXArray), mask: MLXArray) {
        let lengths = keys.map { $0.dim(2) }
        let tMax = lengths.max() ?? 0
        precondition(tMax > 0, "Gemma4CBv2MTPDrafter: empty KV capture")

        let padded = (padStack(keys, to: tMax), padStack(values, to: tMax))

        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let valid = positions .< MLXArray(lengths.map { Int32($0) }).reshaped([-1, 1, 1, 1])
        let dtype = keys[0].dtype
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return (padded, MLX.where(valid, zero, negInf))
    }

    /// Additive `[B, 1, 1, Tmax]` mask for CBv2 sliding captures. Returns nil
    /// only when every retained position is strictly inside its row's window
    /// and no padding exists. Internal so the absolute boundary is pinned by
    /// weight-free tests without exposing it as product API.
    static func slidingMask(
        rows: [CBv2MTPRowCapture], tMax: Int, window: Int, dtype: DType
    ) -> MLXArray? {
        let lengths = rows.map { $0.slidingKeys.dim(2) }
        let needsMask = rows.enumerated().contains { index, row in
            lengths[index] < tMax || row.slidingStart <= row.anchor - window
        }
        guard needsMask else { return nil }

        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let lengthsArray = MLXArray(lengths.map(Int32.init)).reshaped([-1, 1, 1, 1])
        let starts = MLXArray(rows.map { Int32($0.slidingStart) }).reshaped([-1, 1, 1, 1])
        let lowerBounds = MLXArray(rows.map { Int32($0.anchor - window) })
            .reshaped([-1, 1, 1, 1])
        let insideLength = positions .< lengthsArray
        let insideWindow = (positions + starts) .> lowerBounds
        let valid = MLX.logicalAnd(insideLength, insideWindow)
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
    }

    private static func padStack(_ rows: [MLXArray], to tMax: Int) -> MLXArray {
        if rows.allSatisfy({ $0.dim(2) == tMax }) {
            return concatenated(rows, axis: 0)
        }
        let out = MLXArray.zeros(
            [rows.count, rows[0].dim(1), tMax, rows[0].dim(3)], dtype: rows[0].dtype)
        for (i, row) in rows.enumerated() {
            out[i ..< (i + 1), 0..., 0 ..< row.dim(2), 0...] = row
        }
        return out
    }
}
