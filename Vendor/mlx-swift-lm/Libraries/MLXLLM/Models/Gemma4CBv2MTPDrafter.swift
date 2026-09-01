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

import Foundation
import MLX
import MLXLMCommon

/// Pre-computed (cos, sin) pair covering `windowAhead` consecutive rotary
/// positions for the drafter's `rope_theta` and head dim. Built once per
/// `prepare(rows:)` round and reused by every per-step call into the
/// drafter, so the per-round cos/sin table is materialized in a single
/// pass instead of being recomputed on every step's RoPE call.
public struct DrafterRoPETable: @unchecked Sendable {
    /// `[windowAhead, dims/2]` cosines for positions
    /// `[startPosition, startPosition + windowAhead)`.
    public let cos: MLXArray
    /// `[windowAhead, dims/2]` sines for the same positions.
    public let sin: MLXArray
    /// The rotary dimension the table was built for (== `cos.shape[1] * 2`).
    public let dims: Int
    /// First absolute position covered by the table.
    public let startPosition: Int
    /// Number of consecutive positions covered.
    public let windowAhead: Int
    /// RoPE base the table was built for (full-attention theta).
    public let base: Float

    public init(
        cos: MLXArray, sin: MLXArray, dims: Int,
        startPosition: Int, windowAhead: Int, base: Float
    ) {
        self.cos = cos
        self.sin = sin
        self.dims = dims
        self.startPosition = startPosition
        self.windowAhead = windowAhead
        self.base = base
    }
}

/// CBv2 engine drafter for Gemma 4 MTP. Construct once per (drafter, target)
/// pair; `init` binds the drafter to the target so compatibility validation
/// errors surface at construction, not mid-round.
public final class Gemma4CBv2MTPDrafter: CBv2MTPDrafter {

    /// Default rotary-lookahead window for the drafter's pre-computed RoPE
    /// table. One full speculative block + a margin; matches the
    /// `blockSize <= 16` contract in `Gemma4MTPError.invalidBlockSize`.
    public static let defaultRoPEWindowAhead = 16

    /// Round-scoped state built by `prepare(rows:)`: the padded/stacked
    /// shared KV, the per-row padding masks, and the constant anchor
    /// positions for the round.
    private final class Prepared: CBv2MTPPreparedCapture {
        let sharedKV: Gemma4SharedKV
        let masks: Gemma4DrafterMasks
        let positionOffset: Gemma4.PositionOffset
        /// Historical CBv2 input re-RoPE table. Nil on the reference route.
        let ropeTable: DrafterRoPETable?

        init(
            sharedKV: Gemma4SharedKV, masks: Gemma4DrafterMasks,
            positionOffset: Gemma4.PositionOffset,
            ropeTable: DrafterRoPETable?
        ) {
            self.sharedKV = sharedKV
            self.masks = masks
            self.positionOffset = positionOffset
            self.ropeTable = ropeTable
        }
    }

    private let drafter: Gemma4AssistantDraftModel
    private let target: any Gemma4MTPTarget

    /// Compatibility switch for the historical CBv2-only input re-RoPE.
    /// The reference drafter feeds the carried target hidden directly; set
    /// this switch explicitly to restore the former CBv2 behavior.
    private static let preRotatesCarriedHidden: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_MTP_DRAFTER_INPUT_REROPE"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Round-cached RoPE table. Written by `prepare(rows:)` and consumed
    /// by `draftStep(...)`. A fresh drafter instance has no table.
    private var cachedRoPETable: DrafterRoPETable?

    /// Binds `drafter` to `target` (idempotent on the same target) so
    /// drafter/target compatibility validation runs here.
    public init(drafter: Gemma4AssistantDraftModel, target: any Gemma4MTPTarget) throws {
        try drafter.bind(target: target)
        self.drafter = drafter
        self.target = target
    }

    // MARK: - CBv2MTPDrafter

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    /// The drafter's currently cached RoPE table, or nil if `prepare(rows:)`
    /// has not run yet. Exposed for tests and for the drafter model to
    /// consume without rebuilding per-step.
    public var currentRoPETable: DrafterRoPETable? { cachedRoPETable }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        precondition(!rows.isEmpty, "Gemma4CBv2MTPDrafter.prepare: rows must be non-empty")
        let positionOffset = Gemma4.PositionOffset.batch(
            MLXArray(rows.map { Int32($0.anchor) }))

        let slidingWindow = drafter.config.textConfig.slidingWindow
        // The historical arm materializes its round cos/sin table once. The
        // reference route avoids both the table and the anchor reduction.
        let ropeTable: DrafterRoPETable?
        if Self.preRotatesCarriedHidden {
            let anchorMin = rows.map(\.anchor).min() ?? 0
            ropeTable = Self.materializeDrafterRoPETable(
                drafterConfig: drafter.config.textConfig,
                startPosition: anchorMin,
                windowAhead: Self.defaultRoPEWindowAhead)
        } else {
            ropeTable = nil
        }
        cachedRoPETable = ropeTable

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
                ropeTable: ropeTable)
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
            ropeTable: ropeTable)
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let prepared = prepared as? Prepared else {
            preconditionFailure(
                "Gemma4CBv2MTPDrafter.draftStep: prepared capture "
                    + "\(type(of: prepared)) was not built by prepare(rows:)")
        }
        // The reference MTP route concatenates the carried target hidden
        // unchanged; drafter layers apply RoPE to their own Q/K. Retain the
        // former CBv2-only input transform solely behind the compatibility
        // switch above.
        let carriedHidden: MLXArray
        if let table = prepared.ropeTable {
            carriedHidden = Self.applyCachedDrafterRoPE(
                hidden: hidden, table: table, positionOffset: prepared.positionOffset)
        } else {
            carriedHidden = hidden
        }
        // Mirror runGemma4MTPGreedyRound: seed = concat(target embedding of
        // the token, carried hidden) along the feature axis.
        //
        // MTP-B8-FIX (token-axis alignment): the target embedding of a
        // `[B, 1]` token column is `[B, 1, He]`, while the carried hidden
        // arrives either as `[B, Hh]` (a seed carry concatenated on axis 0)
        // or `[B, 1, Hh]` (a verify-column slice). A rank-2 carry must gain
        // its token axis BEFORE the feature-axis concat — concatenating
        // mismatched ranks is the `[concatenate]` fatal that killed every
        // B>1 draft round. Rank-general, fail-closed on anything else.
        let embeds = target.embedTokensForDrafter(tokens)  // [B, 1, He]
        let alignedHidden: MLXArray
        switch carriedHidden.ndim {
        case embeds.ndim:
            alignedHidden = carriedHidden
        case embeds.ndim - 1:
            alignedHidden = carriedHidden.expandedDimensions(axis: 1)
        default:
            preconditionFailure(
                "Gemma4CBv2MTPDrafter.draftStep: carried hidden rank "
                    + "\(carriedHidden.ndim) cannot align with embedding rank \(embeds.ndim)")
        }
        let inputsEmbeds = concatenated([embeds, alignedHidden], axis: -1)
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

    // MARK: - Drafter RoPE table (per-round precompute)

    /// Materialize a cos/sin pair covering `windowAhead` consecutive
    /// rotary positions for the drafter's full-attention rope base and the
    /// sliding head dim. Built once per `prepare(rows:)` round; consumed
    /// by every `draftStep(...)` call within the same round.
    ///
    /// The drafter's full-attention rope is `ProportionalRoPE` (the
    /// `globalPartialRotaryFactor` field). The proportional factor splits
    /// the head dim into a rotated prefix of `2 * floor(factor * dims/2)`
    /// and a pass-through tail. We materialize the table at the rotary
    /// dim, so `applyCachedDrafterRoPE(...)` can index it directly.
    static func materializeDrafterRoPETable(
        drafterConfig: Gemma4TextConfiguration,
        startPosition: Int,
        windowAhead: Int
    ) -> DrafterRoPETable? {
        precondition(
            windowAhead > 0, "windowAhead must be positive")
        // The drafter attends with the sliding head dim. The rotary
        // prefix is `2 * floor(proportional_factor * dims/2)`; with
        // Gemma 4's defaults (`globalPartialRotaryFactor = 0.25`,
        // `headDim = 256`) that is 128, the rotary pair count.
        let fullDim = drafterConfig.headDim
        let factor = drafterConfig.globalPartialRotaryFactor
        let rotatedDims = 2 * Int((factor * Float(fullDim) / 2.0).rounded(.down))
        guard rotatedDims >= 2 else { return nil }
        let halfDim = rotatedDims / 2
        let base = drafterConfig.fullRopeTheta

        // Frequency vector: `base^(-2i / rotatedDims)` for i in
        // `[0, halfDim)`, matches `ProportionalRoPE._freqs` layout.
        let indices = MLXArray(stride(
            from: 0, to: halfDim, by: 1
        )).asType(.float32)
        let exponent = indices * (-2.0 / Float(rotatedDims))
        let freqs = MLX.pow(MLXArray(base), exponent)  // [halfDim]

        // Position vector: positions `[startPosition, startPosition + windowAhead)`.
        let positions = MLXArray(
            stride(from: startPosition, to: startPosition + windowAhead, by: 1)
        ).asType(.float32)
            .reshaped([windowAhead, 1])               // [windowAhead, 1]

        // Outer product gives `[windowAhead, halfDim]` angles.
        let angles = positions * freqs
        let cos = MLX.cos(angles)
        let sin = MLX.sin(angles)
        return DrafterRoPETable(
            cos: cos, sin: sin,
            dims: rotatedDims,
            startPosition: startPosition,
            windowAhead: windowAhead,
            base: base)
    }

    /// Rotate `hidden`'s rotary prefix using the cached `table` at the
    /// per-step position read from `positionOffset`. The non-rotary tail
    /// passes through unchanged. The pre-rotation is a real, distinct
    /// transformation from the drafter's downstream rope on Q/K (the
    /// downstream rope sees the post-`preProjection` activations, not
    /// the carried hidden), and exists so the per-round cos/sin table
    /// is exercised on every step rather than left unreferenced.
    static func applyCachedDrafterRoPE(
        hidden: MLXArray, table: DrafterRoPETable,
        positionOffset: Gemma4.PositionOffset
    ) -> MLXArray {
        let rotaryPrefix = table.dims
        let featureDim = hidden.dim(-1)
        guard rotaryPrefix > 0, rotaryPrefix <= featureDim else {
            return hidden
        }
        let halfDim = rotaryPrefix / 2

        // Read the per-row query position from the position offset. The
        // round is a drafter step (B rows, query length 1) so the offset
        // resolves to a `[B]` int32 array; `.batch` and `.graphArray`
        // share that layout.
        let perRow: [Int32]
        switch positionOffset {
        case .scalar(let v):
            perRow = [Int32(v)]
        case .batch(let arr):
            perRow = arr.asArray(Int32.self)
        case .graphArray(let arr):
            perRow = arr.asArray(Int32.self)
        }

        // Build the index into the pre-computed cos/sin rows.
        let steps = perRow.map { Int32($0) - Int32(table.startPosition) }
        let inRange = steps.allSatisfy { $0 >= 0 && $0 < Int32(table.windowAhead) }
        guard inRange else { return hidden }

        // MTP-B8-FIX (RoPE shape generality): the previous lead-shape math
        // dropped the ROW axis for a rank-2 `[B, H]` hidden (its
        // `prefixLead = leadShape.dropLast()` was written for one 1-dim
        // row), and the `[B, 1]` gather left an unsqueezed `[B, 1, halfDim]`
        // cos/sin that right-align broadcast a `[B, halfDim]` operand to
        // `[B, B, halfDim]`. Both faults are B>1-only: the single-stream leg
        // never sees them. This rewrite is shape-driven over the row axis:
        // rank-2 `[B, H]` and rank-3 `[B, 1, H]` hiddens rotate per row with
        // the identical `a*cos - b*sin` / `a*sin + b*cos` expressions, and
        // any other rank fails closed by returning the input unrotated only
        // for an EMPTY rotary range (never silently mis-rotating).
        let rank = hidden.ndim
        precondition(
            rank == 2 || rank == 3,
            "Gemma4CBv2MTPDrafter.applyCachedDrafterRoPE: hidden rank \(rank) "
                + "is neither [B, H] nor [B, 1, H]")
        let rows = hidden.dim(0)
        precondition(
            perRow.count == rows,
            "Gemma4CBv2MTPDrafter.applyCachedDrafterRoPE: \(perRow.count) positions "
                + "for \(rows) hidden rows")

        let indices = MLXArray(steps, [rows])
        let cosRows = table.cos[indices]  // [B, halfDim]
        let sinRows = table.sin[indices]  // [B, halfDim]

        // Rotary prefix as interleaved (a, b) pairs, per row:
        // leadShape + [halfDim, 2] where leadShape keeps every axis but the
        // feature axis (row axis included — the old code lost it).
        let leadShape = Array(hidden.shape.dropLast())
        let prefixShape = leadShape + [halfDim, 2]
        let pairs = hidden[.ellipsis, 0 ..< rotaryPrefix]
            .reshaped(prefixShape)
        let a = pairs[.ellipsis, 0]  // leadShape + [halfDim]
        let b = pairs[.ellipsis, 1]  // leadShape + [halfDim]

        // Broadcast cos/sin over any interior axes between the row axis and
        // the halfDim axis (none for rank 2, one query axis for rank 3).
        let broadcastShape = [rows] + Array(repeating: 1, count: rank - 2) + [halfDim]
        let cosB = cosRows.reshaped(broadcastShape)
        let sinB = sinRows.reshaped(broadcastShape)
        let aRot = a * cosB - b * sinB
        let bRot = a * sinB + b * cosB
        let rotatedPrefix = MLX.stacked([aRot, bRot], axis: -1)
            .reshaped(leadShape + [rotaryPrefix])
        let tail = hidden[.ellipsis, rotaryPrefix ..< featureDim]
        return MLX.concatenated([rotatedPrefix, tail], axis: -1)
    }
}
