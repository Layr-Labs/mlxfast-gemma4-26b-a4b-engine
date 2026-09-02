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
        /// Pre-computed drafter cos/sin covering this round's
        /// `[anchorMin, anchorMin + windowAhead)`. May be nil if the
        /// drafter's geometry made the table empty.
        let ropeTable: DrafterRoPETable?
        /// The target's q4 sliding mirrors when the sliding layers attend
        /// them in place; nil keeps the round on the prepared copies.
        let mirror: Gemma4DrafterMirrorAttention.Context?

        init(
            sharedKV: Gemma4SharedKV, masks: Gemma4DrafterMasks,
            positionOffset: Gemma4.PositionOffset,
            ropeTable: DrafterRoPETable?,
            mirror: Gemma4DrafterMirrorAttention.Context? = nil
        ) {
            self.sharedKV = sharedKV
            self.masks = masks
            self.positionOffset = positionOffset
            self.ropeTable = ropeTable
            self.mirror = mirror
        }
    }

    private let drafter: Gemma4AssistantDraftModel
    private let target: any Gemma4MTPTarget

    /// Round-cached RoPE table. Written by `prepare(rows:)` and consumed
    /// by `draftStep(...)`. A fresh drafter instance has no table.
    private var cachedRoPETable: DrafterRoPETable?

    /// Binds `drafter` to `target` (idempotent on the same target) so
    /// drafter/target compatibility validation runs here.
    public init(drafter: Gemma4AssistantDraftModel, target: any Gemma4MTPTarget) throws {
        try drafter.bind(target: target)
        self.drafter = drafter
        self.target = target
        if let model = target as? Gemma4TextModel {
            warmSpeculativeCohort(model: model)
        }
    }

    /// Compile every kernel a chained wide-verify round dispatches BEFORE
    /// the first scored window: the ranked box has a cold per-process
    /// pipeline cache and a first-use compile inside the decode window costs
    /// hundreds of milliseconds. Runs a few speculative rounds of an
    /// eight-row cohort over BOS seeds through the same engine the scored
    /// cohort uses. Kill switch `DARKBLOOM_CBV2_MTP_WARM=0`.
    private func warmSpeculativeCohort(model: Gemma4TextModel) {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_WARM"],
            ["0", "false", "no", "off"].contains(raw.lowercased())
        {
            return
        }
        guard CBv2MTPSpeculationPolicy.speculationEnabled else { return }
        let batch = 8
        let seedCount = 1024
        let depth = CBv2MTPSpeculationPolicy.draftDepth
        let warmTokens = 4 * (depth + 1)
        do {
            let caches = try model.newCacheV2 { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 4 << 30))
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds,
                backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: batch,
                    maxBatchedTokensPerStep: Swift.max(2048, batch * seedCount),
                    prefillChunkSize: Swift.max(512, seedCount),
                    maxWaiting: batch,
                    enablePrefixCache: false),
                mtpDrafter: self,
                mtpConfig: CBv2MTPConfig(
                    enabled: true,
                    maxDraftTokens: depth,
                    maxSpeculativeBatch: batch,
                    fixedDraftTokens: depth,
                    verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 32))
            // Rows differ so the accept walk goes ragged and the per-row
            // (ragged) kernels compile too, not only the uniform ones.
            // The eight rows stay alive together (the quant-authoritative
            // decode road serves exactly eight) and are cancelled together
            // once every row has produced enough rounds.
            let drained = DispatchSemaphore(value: 0)
            let progress = WarmProgress(slots: batch, target: warmTokens)
            let consumer = Task {
                await withTaskGroup(of: Void.self) { group in
                    for slot in 0 ..< batch {
                        var seeds = Array(repeating: 2, count: seedCount)
                        for i in stride(from: 1, to: seedCount, by: 1) {
                            seeds[i] = 1000 + ((i * 7919 + slot * 104729) % 20000)
                        }
                        guard let stream = try? engine.submit(
                            CBv2Request(
                                id: CBv2RequestID(UInt64(slot)),
                                promptTokens: seeds,
                                sampling: CBv2SamplingParams(temperature: 0),
                                maxTokens: 4096,
                                stopTokens: [],
                                prefixCacheEnabled: false))
                        else { continue }
                        group.addTask {
                            for await event in stream {
                                if case .delta(_, let ids, _) = event {
                                    if progress.record(slot: slot, count: ids.count) {
                                        for other in 0 ..< batch {
                                            engine.cancel(CBv2RequestID(UInt64(other)))
                                        }
                                    }
                                }
                                if case .finished = event { break }
                            }
                        }
                    }
                    await group.waitForAll()
                }
                drained.signal()
            }
            if drained.wait(timeout: .now() + 120) == .timedOut {
                consumer.cancel()
                for slot in 0 ..< batch { engine.cancel(CBv2RequestID(UInt64(slot))) }
            }
            let stopped = DispatchSemaphore(value: 0)
            Task {
                await engine.shutdownSynchronously()
                stopped.signal()
            }
            _ = stopped.wait(timeout: .now() + 30)
            Memory.clearCache()
            CBv2EngageMark.once("mtp-warm-cohort")
        } catch {
        }
    }

    // MARK: - CBv2MTPDrafter

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    /// The drafter's currently cached RoPE table, or nil if `prepare(rows:)`
    /// has not run yet. Exposed for tests and for the drafter model to
    /// consume without rebuilding per-step.
    public var currentRoPETable: DrafterRoPETable? { cachedRoPETable }

    public func prepare(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture?
    ) -> CBv2MTPPreparedCapture {
        guard let cohort else { return prepare(rows: rows) }
        precondition(
            rows.count == cohort.anchors.dim(0),
            "Gemma4CBv2MTPDrafter.prepare: \(rows.count) rows but \(cohort.anchors.dim(0)) anchors")
        return prepareDevice(rows: rows, cohort: cohort)
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        precondition(!rows.isEmpty, "Gemma4CBv2MTPDrafter.prepare: rows must be non-empty")
        let positionOffset = Gemma4.PositionOffset.batch(
            MLXArray(rows.map { Int32($0.anchor) }))

        let slidingWindow = drafter.config.textConfig.slidingWindow
        // Materialize the round's cos/sin table once. Anchor the table at
        // the minimum anchor across the batch so a per-row query position
        // anywhere inside `[anchorMin, anchorMin + windowAhead)` indexes
        // into the table with a non-negative step.
        let anchorMin = rows.map(\.anchor).min() ?? 0
        let ropeTable = Self.materializeDrafterRoPETable(
            drafterConfig: drafter.config.textConfig,
            startPosition: anchorMin,
            windowAhead: Self.defaultRoPEWindowAhead)
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
        // If the round has a cached RoPE table, pre-rotate the drafter's
        // carried hidden along its rotary prefix using the cached
        // (cos, sin) for the per-step query position. The drafter's
        // `preProjection` is a Linear, so pre-rotation of the input is
        // a real, distinct transformation from the drafter's downstream
        // RoPE on Q/K. The pre-rotation is intentionally cheap (one
        // per-round table lookup + an `a*cos-b*sin` slice) and amortizes
        // the per-step `MLXFast.RoPE` frequency compute into the single
        // `prepare(rows:)` materialization: the downstream rope module
        // no longer pays the table-build cost on every draft step.
        // B > 1 shape discipline: the target's pre-norm hidden reaches this
        // seam as `[B, 1, H]` on the stock trunk but as `[B, H]` from the
        // promoted fused decode tail; the token embedding below is always
        // `[B, 1, He]`. Normalize once so the RoPE prefix math and the
        // feature-axis concat see one rank.
        let hidden = hidden.ndim == 2 ? hidden.expandedDimensions(axis: 1) : hidden
        let rotatedHidden: MLXArray
        if let table = prepared.ropeTable {
            rotatedHidden = Self.applyCachedDrafterRoPE(
                hidden: hidden, table: table, positionOffset: prepared.positionOffset)
        } else {
            rotatedHidden = hidden
        }
        // Mirror runGemma4MTPGreedyRound: seed = concat(target embedding of
        // the token, carried hidden) along the feature axis.
        let inputsEmbeds = concatenated(
            [target.embedTokensForDrafter(tokens), rotatedHidden], axis: -1)
        let (newHidden, logits) = Gemma4DrafterMirrorAttention.with(prepared.mirror) {
            drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: prepared.sharedKV,
                positionOffset: prepared.positionOffset,
                masks: prepared.masks)
        }
        let next = logits.squeezed(axis: 1).argMax(axis: -1).asType(.int32)
        return (next, newHidden)
    }

    /// Per-slot token counts for the warm cohort; `record` returns true
    /// exactly once, when every slot has reached the target.
    private final class WarmProgress: @unchecked Sendable {
        private let lock = NSLock()
        private var counts: [Int]
        private let target: Int
        private var fired = false
        init(slots: Int, target: Int) {
            counts = Array(repeating: 0, count: slots)
            self.target = target
        }
        func record(slot: Int, count: Int) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            counts[slot] += count
            guard !fired, counts.allSatisfy({ $0 >= target }) else { return false }
            fired = true
            return true
        }
    }

    // MARK: - Device geometry (chained rounds)

    /// Chained rounds carry device geometry (host counters lag the device
    /// truth): every anchor, length and mask stays on device and the hidden
    /// pre-rotation (which reads anchors back to the host) is skipped.
    private func prepareDevice(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture
    ) -> CBv2MTPPreparedCapture {
        cachedRoPETable = nil
        let fullKV = cohort.pooledFull.map { ($0.keys, $0.values) }
            ?? Self.padStack(rows.map(\.fullKeys), rows.map(\.fullValues))
        let fullMask = Self.lengthMask(
            tMax: fullKV.0.dim(2), lengths: cohort.fullLengths, dtype: fullKV.0.dtype)

        // The stacked copies and their mask stay lazy: a round that attends
        // the mirrors never evaluates them, and a refused geometry still has
        // the stock road to run.
        let slidingWindow = drafter.config.textConfig.slidingWindow
        let slidingKV = Self.padStack(rows.map(\.slidingKeys), rows.map(\.slidingValues))
        let slidingMask = Self.slidingMaskDevice(
            tMax: slidingKV.0.dim(2), lengths: MLXArray(rows.map { Int32($0.slidingKeys.dim(2)) }),
            starts: cohort.slidingStarts, anchors: cohort.anchors,
            window: slidingWindow, dtype: slidingKV.0.dtype)
        return Prepared(
            sharedKV: Gemma4SharedKV(fullAttention: fullKV, slidingAttention: slidingKV),
            masks: Gemma4DrafterMasks(full: .array(fullMask), sliding: .array(slidingMask)),
            positionOffset: .batch(cohort.anchors),
            ropeTable: nil,
            mirror: Gemma4DrafterMirrorAttention.context(
                rows: rows, cohort: cohort, window: slidingWindow))
    }

    /// Additive `[B, 1, 1, tMax]` mask: 0 where `position < lengths[row]`.
    private static func lengthMask(tMax: Int, lengths: MLXArray, dtype: DType) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let valid = positions .< lengths.reshaped([-1, 1, 1, 1])
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
    }

    /// Device twin of `slidingMask`: inside the retained length AND strictly
    /// inside the target's `(anchor - window, anchor + window)` rule.
    private static func slidingMaskDevice(
        tMax: Int, lengths: MLXArray, starts: MLXArray, anchors: MLXArray,
        window: Int, dtype: DType
    ) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let insideLength = positions .< lengths.reshaped([-1, 1, 1, 1])
        let insideWindow =
            (positions + starts.reshaped([-1, 1, 1, 1])) .> (anchors - Int32(window)).reshaped([-1, 1, 1, 1])
        let valid = MLX.logicalAnd(insideLength, insideWindow)
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
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

        let padded = padStack(keys, values)

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

    private static func padStack(_ keys: [MLXArray], _ values: [MLXArray]) -> (MLXArray, MLXArray) {
        let tMax = keys.map { $0.dim(2) }.max() ?? 0
        return (padStack(keys, to: tMax), padStack(values, to: tMax))
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

        // Shape-generic rotation over the ROW axis. `hidden` is `[B, 1, H]`
        // (or `[B, H]`); every leading axis is kept as-is, the rotary prefix
        // is viewed as `[..., halfDim, 2]` pairs, and the per-row (cos, sin)
        // rows broadcast along the leading axes after the row axis. The
        // previous form dropped the row axis's neighbour and produced a
        // rank-2 prefix beside a rank-3 tail, which the concat below refuses
        // for every B.
        let indices = MLXArray(steps, [perRow.count])
        let leadShape = Array(hidden.shape.dropLast())  // [B] or [B, 1]
        let broadcastShape =
            [perRow.count] + Array(repeating: 1, count: max(0, leadShape.count - 1))
            + [halfDim]
        let cosB = table.cos[indices].reshaped(broadcastShape)  // [B, (1,) halfDim]
        let sinB = table.sin[indices].reshaped(broadcastShape)

        let prefix = hidden[.ellipsis, 0 ..< rotaryPrefix]
        let pairs = prefix.reshaped(leadShape + [halfDim, 2])
        let a = pairs[.ellipsis, 0]  // [..., halfDim]
        let b = pairs[.ellipsis, 1]
        let aRot = a * cosB - b * sinB
        let bRot = a * sinB + b * cosB
        let rotatedPrefix = MLX.stacked([aRot, bRot], axis: -1)
            .reshaped(leadShape + [rotaryPrefix])
        let tail = hidden[.ellipsis, rotaryPrefix ..< featureDim]
        return MLX.concatenated([rotatedPrefix, tail], axis: -1)
    }
}

/// Chained rounds: the drafter's sliding layers attend the target's q4
/// sliding mirror in place instead of running stock SDPA over dequantized,
/// stacked and masked copies. Bound around the drafter forward by
/// `Gemma4CBv2MTPDrafter.draftStep`; read by the shared-KV sliding path of
/// `Gemma4Attention`, which falls back to the prepared copies whenever the
/// kernel refuses the geometry.
enum Gemma4DrafterMirrorAttention {
    struct Context {
        let mirrors: [MLXArray]
        /// `[8]` uint32 boundary slots (anchor mod window), skipped like the
        /// sliding mask skips them.
        let slotBases: MLXArray
        let fence: MLXArray
        let window: Int
    }

    /// Kill switch: `DARKBLOOM_CBV2_MTP_MIRROR_ATTENTION=0` keeps every round
    /// on the dequantized road.
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_MIRROR_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private final class Binding {
        let context: Context
        init(_ context: Context) { self.context = context }
    }

    private static let bindingKey = "Gemma4DrafterMirrorAttention.context"

    /// Bound per thread: a round's graph is built synchronously on one
    /// thread, and other forwards (other suites of one test process, another
    /// engine) must never observe this round's mirrors.
    private(set) static var current: Context? {
        get { (Thread.current.threadDictionary[bindingKey] as? Binding)?.context }
        set { Thread.current.threadDictionary[bindingKey] = newValue.map(Binding.init) }
    }

    /// nil unless every row of an eight-row cohort carries its mirror and the
    /// cohort carries the slot bases and fence; the dequantized road applies.
    static func context(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture, window: Int
    ) -> Context? {
        guard enabled, rows.count == 8,
            let slotBases = cohort.slidingMirrorSlotBases,
            let fence = cohort.slidingMirrorFence
        else { return nil }
        let mirrors = rows.compactMap(\.slidingMirror)
        guard mirrors.count == rows.count else { return nil }
        return Context(
            mirrors: mirrors, slotBases: slotBases.asType(.uint32), fence: fence, window: window)
    }

    static func with<T>(_ context: Context?, _ body: () -> T) -> T {
        let previous = current
        current = context
        defer { current = previous }
        return body()
    }

    /// nil outside a mirror round, on a full-attention layer, or when the
    /// kernel refuses the geometry: the caller then runs its stock SDPA over
    /// the prepared copies, and the refusal leaves a mark so the slower road
    /// stays visible.
    static func attend(queries: MLXArray, isSliding: Bool) -> MLXArray? {
        guard isSliding, let current else { return nil }
        guard let output = CBv2MTPMirrorAttention.attend(
                queries: queries, mirrors: current.mirrors, slotBases: current.slotBases,
                fence: current.fence, window: current.window)
        else {
            CBv2EngageMark.once("mtp-drafter-q4-mirror-attention-fallback")
            return nil
        }
        return output
    }
}
