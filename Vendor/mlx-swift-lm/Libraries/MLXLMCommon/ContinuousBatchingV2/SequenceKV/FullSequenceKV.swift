// FullSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for FULL attention layers.
//
// One instance owns the K/V for ONE sequence at ONE layer. There is no shared
// batch frontier, no left padding, and no batch-wide trim: joining or leaving
// a batch never touches this object (batch membership is just list membership
// in `CBv2LayerCache.rows`).

import Foundation
import MLX
import MLXFast

/// Counters for the v2 core runtime's own host-interaction points.
///
/// The engine step loop must never force a host sync (`.item()`, `asArray`,
/// blocking `eval`) and must never rebuild per-row metadata arrays from host
/// integers outside membership changes. Any CBv2 core code that *does* touch
/// the host goes through these counters so tests can assert the step loop is
/// clean (see `CBv2CoreTests`). This deliberately does not instrument MLX
/// itself — only our own sync points.
public enum CBv2CoreInstrumentation {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _hostSyncs = 0
    nonisolated(unsafe) private static var _positionOffsetsHostRebuilds = 0

    /// Number of host syncs performed by CBv2 core code.
    public static var hostSyncs: Int {
        lock.lock()
        defer { lock.unlock() }
        return _hostSyncs
    }

    /// Number of times a layer cache rebuilt `positionOffsets` from host
    /// integers. Must only ever increase on batch membership changes.
    public static var positionOffsetsHostRebuilds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _positionOffsetsHostRebuilds
    }

    static func recordHostSync() {
        lock.lock()
        defer { lock.unlock() }
        _hostSyncs += 1
    }

    static func recordPositionOffsetsHostRebuild() {
        lock.lock()
        defer { lock.unlock() }
        _positionOffsetsHostRebuilds += 1
    }
}

/// Internal hook so `CBv2LayerCache` can hand a row's lazily-mutated storage
/// arrays to the engine loop's `asyncEval` (graph/metadata hygiene: no
/// unconsumed lazy chain may grow O(steps) — DAR-325).
protocol CBv2InnerStateProviding {
    func cbv2InnerState() -> [MLXArray]
}

/// ATT-008: shared batch-wide K/V storage for a lockstep decode cohort of
/// full-attention rows.
///
/// One pool owns `[rowCount, kvHeads, capacity, headDim]` K and V buffers;
/// row `i` of the pool holds exactly the bytes row `i`'s private buffers held
/// before migration (the migration is a one-time `concatenated` bit-copy).
/// What the pool buys is DISPATCH SHAPE, not different numerics:
///
/// - a lockstep decode append becomes ONE `[B, kvHeads, 1, headDim]` slice
///   assignment per K and V instead of `B` per-row assignments, and
/// - all `B` rows' attention views become ONE strided
///   `[B, kvHeads, offset, headDim]` view, so the whole cohort can ride a
///   single batched attention call instead of `B` row-local calls.
///
/// Batching the call does not change any row's arithmetic: for the D=512
/// decode shapes this feeds (M=1 matmuls and softmax), every MLX kernel
/// selection — the gemv/gemv_t configuration, the `gemv_al` alignment gate
/// (which requires `batch_size_out == 1` and so never fires for either
/// dispatch), the softmax variant and threadgroup size, and the
/// `check_transpose` no-copy branches — is a pure function of
/// (M, N, K, dtype, last-two-dim strides). The batch extent only scales
/// `grid.z` / the row count, so each row's per-output add order is the stock
/// per-row order BY CONSTRUCTION. Verified bit-exact (uint16) against the
/// per-row chain at the production geometry, kL ∈ {1024, 1025, 1100, 1152}
/// and across simulated append steps.
///
/// A pooled row remains a fully functional `CBv2SequenceKV`: `update`,
/// `snapshot`, `rollback` and `cbv2InnerState` route through the pool with
/// unchanged semantics, so every non-batched code path (per-row decode
/// fallback, prefill continuations, drain steps with fewer rows) keeps
/// working on pooled storage and stays bit-identical to the unpooled layout.
final class CBv2FullDecodeCohortPool {
    let rowCount: Int
    let kvHeads: Int
    let headDim: Int

    private(set) var keys: MLXArray
    private(set) var values: MLXArray
    private(set) var capacity: Int

    /// Hard ceiling on growth: the largest `maxLength` of the migrated rows.
    private let capacityLimit: Int

    init(
        keys: MLXArray, values: MLXArray, capacity: Int,
        rowCount: Int, kvHeads: Int, headDim: Int, capacityLimit: Int
    ) {
        self.keys = keys
        self.values = values
        self.capacity = capacity
        self.rowCount = rowCount
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacityLimit = capacityLimit
    }

    var nbytes: Int { keys.nbytes + values.nbytes }

    /// One row's append through the pool — the pooled twin of the private
    /// buffer's slice assignment (identical values into identical slots).
    func rowAppend(
        index: Int, keys newKeys: MLXArray, values newValues: MLXArray,
        at offset: Int, count n: Int
    ) {
        ensureCapacity(offset + n)
        keys[index ..< (index + 1), 0..., offset ..< (offset + n), 0...] = newKeys
        values[index ..< (index + 1), 0..., offset ..< (offset + n), 0...] = newValues
    }

    /// Lockstep decode append: every row writes the SAME slot, so all
    /// `rowCount` rows commit with one slice assignment per K and V.
    func batchAppend(keys newKeys: MLXArray, values newValues: MLXArray, at offset: Int) {
        ensureCapacity(offset + 1)
        keys[0..., 0..., offset ..< (offset + 1), 0...] = newKeys
        values[0..., 0..., offset ..< (offset + 1), 0...] = newValues
    }

    /// Zero-copy temporal-order views of one row — shape
    /// `[1, kvHeads, offset, headDim]`, stride-identical to the view the
    /// row's private buffer used to return.
    func rowViews(index: Int, upTo offset: Int) -> (MLXArray, MLXArray) {
        (
            keys[index ..< (index + 1), 0..., ..<offset, 0...],
            values[index ..< (index + 1), 0..., ..<offset, 0...]
        )
    }

    /// Zero-copy batch-wide views `[rowCount, kvHeads, offset, headDim]` for
    /// the single batched attention call.
    func batchViews(upTo offset: Int) -> (MLXArray, MLXArray) {
        (keys[0..., 0..., ..<offset, 0...], values[0..., 0..., ..<offset, 0...])
    }

    private func ensureCapacity(_ needed: Int) {
        guard needed > capacity else { return }
        precondition(
            needed <= capacityLimit,
            "CBv2FullDecodeCohortPool: append past capacity limit (\(needed) > \(capacityLimit)) — admission bug"
        )
        // Same doubling policy as the private buffers.
        let newCapacity = min(capacityLimit, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys, MLXArray.zeros([rowCount, kvHeads, growth, headDim], dtype: keys.dtype)],
            axis: 2)
        values = concatenated(
            [values, MLXArray.zeros([rowCount, kvHeads, growth, headDim], dtype: values.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}

/// `CBv2SequenceKV` for full (non-windowed) attention.
///
/// Storage is one contiguous `[1, kvHeads, capacity, headDim]` buffer per
/// K and V, grown by doubling (initial capacity = promptLength + 256, capped
/// at `maxLength`). Appends are slice assignments — `mlx_slice_update`
/// donates the input buffer when refcount permits, so an append is O(n), not
/// O(cache). `update` returns temporal-order zero-copy strided views
/// `[..., 0..<retained, :]`; MLX SDPA accepts strided K/V.
///
/// A row may be MIGRATED into a `CBv2FullDecodeCohortPool` (ATT-008, see
/// `cohortPool(binding:)`): its bytes move once into the pool's batch axis
/// and every accessor then routes through the pool with identical semantics
/// and identical returned-view strides.
public final class CBv2FullSequenceKV: CBv2DecodeRootCompactionCapableSequenceKV,
    CBv2InnerStateProviding
{

    /// Extra slots allocated beyond the prompt so the first decode steps
    /// don't immediately grow the buffer.
    static let initialSlack = 256

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    /// Hard cap on this sequence's length; growth beyond it is an engine
    /// admission bug and traps.
    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    private var capacity: Int

    /// KVQ-D512-VAL: 4-bit affine mirror of the VALUE plane only, in groups
    /// of 64, maintained ALONGSIDE the bf16 buffer (which stays the source of
    /// truth for every logical view, snapshot, borrow and rollback path).
    /// Only the B=8 D=512 full-attention decode AV kernel reads it; cutting
    /// its 1,024-byte rows to 288 is the entire point.
    ///
    /// Layout `[kvHeads, capacity, 72]` uint32, slot-for-slot with the bf16
    /// plane: words 0..<64 hold eight 4-bit values each, words 64..<72 hold
    /// one fp16 (scale, bias) pair per 64-element group, scale in the low
    /// half. Same packing the sliding-window mirror uses at D=256.
    private var valueMirror: MLXArray?

    /// Number of leading tokens whose mirror rows are known to match the
    /// bf16 plane. Any write that does not also pack leaves this behind
    /// `absoluteOffset`, and every reader is fail-closed on the difference.
    private var mirrorValidCount: Int = 0

    /// 64 payload words + 8 group words for D = 512.
    static let mirrorRowWords = 512 / 8 + 512 / 64

    /// `MLX_KV_D512_VMIRROR=0` never allocates the mirror, so the decode
    /// chain takes the established bf16 road in the same binary. `MLX_`
    /// prefix: the worker env sanitizer only passes that namespace through.
    static let valueMirrorEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_D512_VMIRROR"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `MLX_KV_D512_VMIRROR_SIM=1` (default OFF, local only): overwrite the
    /// bf16 value plane with its quantize/dequantize round trip after every
    /// write, so the SINGLE-STREAM path — the one a local `--local-iterate`
    /// golden run exercises — sees the values the B=8 quantized AV kernel
    /// would reconstruct. That turns the local teacher-forced golden into a
    /// real drift measurement for this mechanism. Never set on the box.
    static let valueMirrorSimulate: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KV_D512_VMIRROR_SIM"] ?? "")
                .lowercased())
    }()

    private var mirrorEligible: Bool {
        Self.valueMirrorEnabled && headDim == 512 && cohortPool == nil
    }

    /// The packed value mirror, or nil when it is not caught up with the
    /// bf16 plane. Fail-closed: a caller that gets nil must read bf16.
    var cbv2QuantValuePlane: MLXArray? {
        guard cohortPool == nil, mirrorValidCount == absoluteOffset,
            let valueMirror
        else { return nil }
        return valueMirror
    }

    /// ATT-008 cohort pooling (nil until `cohortPool(binding:)` migrates this
    /// row). While bound, `keys`/`values` are nil and the pool's row
    /// `cohortIndex` is the storage.
    private(set) var cohortPool: CBv2FullDecodeCohortPool?
    private(set) var cohortIndex: Int = -1

    /// - Parameters:
    ///   - promptLength: expected prompt length, used to size the initial
    ///     allocation (`promptLength + 256`, capped at `maxLength`).
    ///   - maxLength: maximum total tokens this sequence may ever hold.
    ///   - kvHeads/headDim: from the layer's `CBv2LayerKind`; validated
    ///     against the arrays passed to `update`.
    public init(promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int) {
        precondition(maxLength > 0, "CBv2FullSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2FullSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = min(maxLength, max(1, promptLength + Self.initialSlack))
    }

    public var byteCount: Int {
        if let pool = cohortPool {
            // This row's share of the pooled allocation; summing every bound
            // row reproduces the pool total, so the backend ledger stays
            // truthful after migration.
            return pool.nbytes / pool.rowCount
        }
        // KVQ-D512-VAL: the mirror is real resident memory and must be
        // visible to the engine's capacity accounting.
        return (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (valueMirror?.nbytes ?? 0)
    }

    /// WRITE-016-D512: the `update()` bookkeeping advance without the two
    /// slice assignments, for a token whose K/V bytes were already stored in
    /// place by the fused QK dispatch. The caller (the fused wrapper) gates
    /// capacity >= the new length before the store, so this never needs
    /// `ensureCapacity`; a step that would grow the buffer falls back to the
    /// append path instead.
    public func advanceAfterFusedAppend() {
        precondition(
            cohortPool == nil && keys != nil && values != nil,
            "CBv2FullSequenceKV: fused append advance requires private storage")
        precondition(
            absoluteOffset + 1 <= maxLength,
            "CBv2FullSequenceKV: fused append past maxLength — admission bug")
        absoluteOffset += 1
    }

    /// KVQ-D512-VAL twin of `advanceAfterFusedAppend` for the store dispatch
    /// that ALSO packed the new token's mirror row in the same kernel. The
    /// caller only reaches here after `cbv2QuantValuePlane` handed it a
    /// caught-up mirror, so the guard below should never fire; it retires the
    /// mirror rather than trapping, because a stale mirror only ever needs to
    /// cost the cohort its quantized road, never the run.
    public func advanceAfterFusedAppendWithMirror() {
        let caughtUp = valueMirror != nil && mirrorValidCount == absoluteOffset
        advanceAfterFusedAppend()
        if caughtUp {
            mirrorValidCount = absoluteOffset
        } else {
            valueMirror = nil
            mirrorValidCount = 0
        }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2FullSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2FullSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2FullSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        if let pool = cohortPool {
            // Pooled twin of the private-buffer append below: same values
            // into the same slots, same returned-view strides.
            pool.rowAppend(
                index: cohortIndex, keys: newKeys, values: newValues,
                at: absoluteOffset, count: n)
            absoluteOffset += n
            return pool.rowViews(index: cohortIndex, upTo: absoluteOffset)
        }

        ensureCapacity(absoluteOffset + n, keyTemplate: newKeys, valueTemplate: newValues)

        keys![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newKeys
        values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newValues
        packValueMirror(newValues, at: absoluteOffset, count: n)
        if Self.valueMirrorSimulate, mirrorEligible, newValues.dim(3) == 512 {
            Self.noteSimulationOnce()
            values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] =
                Self.simulateValueRoundTrip(newValues)
        }
        absoluteOffset += n

        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    /// Confirm this row's slot of a pool-level `batchAppend` (the batched
    /// decode path commits all rows' K/V in one slice assignment, then bumps
    /// each row's offset here instead of calling `update`).
    func confirmPooledBatchAppend(_ n: Int) {
        precondition(cohortPool != nil, "CBv2FullSequenceKV: batch append without a pool")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )
        absoluteOffset += n
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        if let pool = cohortPool {
            let (poolKeys, poolValues) = pool.rowViews(
                index: cohortIndex, upTo: absoluteOffset)
            return (poolKeys, poolValues, absoluteOffset)
        }
        guard let keys, let values else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...],
            absoluteOffset
        )
    }

    /// Plain rollback is already value-exact (see `rollback`: the offset
    /// decrement makes the un-confirmed tail structurally unreachable and
    /// the confirmed prefix is untouched), so speculative begin/commit are
    /// the contract's default no-ops.
    public var supportsSpeculativeWrites: Bool { true }

    /// Rollback the last `n` tokens (speculative rejection). The un-confirmed
    /// tail is structurally unreachable afterwards: every view this class
    /// hands out is sliced to `..<absoluteOffset`, and the tail slots are
    /// overwritten by the next `update` before they can ever be exposed —
    /// so no zeroing pass is needed.
    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2FullSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2FullSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
        // The rolled-back mirror slots still hold the rejected token's bytes;
        // they are structurally unreachable for the same reason the bf16 tail
        // is, and the next append repacks them.
        mirrorValidCount = min(mirrorValidCount, absoluteOffset)
    }

    func cbv2InnerState() -> [MLXArray] {
        if let pool = cohortPool {
            return [pool.keys, pool.values]
        }
        return [keys, values].compactMap { $0 }
    }

    // MARK: - ATT-008 cohort pooling

    /// Resolve (or form) the shared decode pool for `rows`, or nil when the
    /// rows are not poolable — in which case the caller keeps the per-row
    /// path, which remains correct on pooled and unpooled rows alike.
    ///
    /// Resolution: every row already bound to ONE pool with
    /// `cohortIndex == position` returns that pool. Formation: every row
    /// unpooled with identical geometry (kvHeads, headDim, capacity, dtype,
    /// buffer shape) migrates once — a `concatenated` bit-copy of each row's
    /// committed buffer into the pool's batch axis — and the private buffers
    /// are released. Any mix fails closed.
    static func cohortPool(binding rows: [CBv2FullSequenceKV])
        -> CBv2FullDecodeCohortPool?
    {
        guard !rows.isEmpty else { return nil }

        if let pool = rows[0].cohortPool {
            guard pool.rowCount == rows.count else { return nil }
            for (index, row) in rows.enumerated() {
                guard row.cohortPool === pool, row.cohortIndex == index else {
                    return nil
                }
            }
            return pool
        }

        let head = rows[0]
        guard let headKeys = head.keys, let headValues = head.values else { return nil }
        // Pool growth allocates `[rowCount, kvHeads, growth, headDim]` blocks,
        // so every row's buffer must match the class geometry EXACTLY (not
        // merely each other).
        let expectedShape = [1, head.kvHeads, head.capacity, head.headDim]
        for row in rows {
            guard row.cohortPool == nil,
                row.kvHeads == head.kvHeads,
                row.headDim == head.headDim,
                row.capacity == head.capacity,
                let rowKeys = row.keys, let rowValues = row.values,
                rowKeys.dtype == headKeys.dtype,
                rowValues.dtype == headValues.dtype,
                rowKeys.shape == expectedShape,
                rowValues.shape == expectedShape
            else { return nil }
        }

        let pool = CBv2FullDecodeCohortPool(
            keys: concatenated(rows.map { $0.keys! }, axis: 0),
            values: concatenated(rows.map { $0.values! }, axis: 0),
            capacity: head.capacity,
            rowCount: rows.count,
            kvHeads: head.kvHeads,
            headDim: head.headDim,
            capacityLimit: rows.map(\.maxLength).max()!)
        for (index, row) in rows.enumerated() {
            row.cohortPool = pool
            row.cohortIndex = index
            row.keys = nil
            row.values = nil
            // KVQ-D512-VAL: nothing maintains a mirror through the pool, so
            // migration retires it and the row rejoins the bf16 road.
            row.valueMirror = nil
            row.mirrorValidCount = 0
        }
        return pool
    }

    // MARK: - Private

    private func ensureCapacity(_ needed: Int, keyTemplate: MLXArray, valueTemplate: MLXArray) {
        if keys == nil {
            capacity = min(maxLength, max(capacity, needed))
            keys = MLXArray.zeros(
                [1, kvHeads, capacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, capacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
            if mirrorEligible, valueTemplate.dim(3) == 512 {
                valueMirror = MLXArray.zeros(
                    [kvHeads, capacity, Self.mirrorRowWords], dtype: .uint32)
                mirrorValidCount = 0
            }
            return
        }
        guard needed > capacity else { return }

        // Grow by doubling, capped at maxLength. The concat copies the old
        // buffer once per doubling — amortized O(1) per appended token.
        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys!, MLXArray.zeros([1, kvHeads, growth, keys!.dim(3)], dtype: keys!.dtype)],
            axis: 2)
        values = concatenated(
            [values!, MLXArray.zeros([1, kvHeads, growth, values!.dim(3)], dtype: values!.dtype)],
            axis: 2)
        if let mirror = valueMirror {
            valueMirror = concatenated(
                [
                    mirror,
                    MLXArray.zeros(
                        [kvHeads, growth, Self.mirrorRowWords], dtype: .uint32),
                ],
                axis: 1)
        }
        capacity = newCapacity
    }

    // MARK: - KVQ-D512-VAL packing

    /// Pack `[1, kvHeads, n, 512]` bf16 values into the mirror slots
    /// `[offset, offset + n)`. One dispatch per update, whatever `n` is.
    private func packValueMirror(_ newValues: MLXArray, at offset: Int, count n: Int) {
        guard valueMirror != nil, newValues.dim(3) == 512,
            mirrorValidCount == offset
        else { return }
        let rows = kvHeads * n
        let packed = Self.valuePackKernel(
            [newValues.reshaped([rows, 512])],
            template: [("T", newValues.dtype)],
            grid: (64, 1, rows),
            threadGroup: (64, 1, 1),
            outputShapes: [[rows, Self.mirrorRowWords]],
            outputDTypes: [.uint32]
        )[0]
        valueMirror![0..., offset ..< (offset + n), 0...] =
            packed.reshaped([kvHeads, n, Self.mirrorRowWords])
        mirrorValidCount = offset + n
    }

    nonisolated(unsafe) private static var simulationLogged = false
    private static let simulationLogLock = NSLock()

    /// One line to stderr the first time the simulation actually perturbs a
    /// value plane. A silent probe proves nothing.
    private static func noteSimulationOnce() {
        simulationLogLock.lock()
        let fresh = !simulationLogged
        simulationLogged = true
        simulationLogLock.unlock()
        if fresh {
            FileHandle.standardError.write(
                Data("[kvq-d512-val] simulation active\n".utf8))
        }
    }

    /// Host expression for the mirror's quantize/dequantize round trip, used
    /// only by the local simulation flag. It is the same expression the
    /// packer kernel is verified against, so the values it writes are the
    /// ones the AV kernel reconstructs, up to the bf16 store.
    private static func simulateValueRoundTrip(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let grouped = x.asType(.float32)
            .reshaped([shape[0], shape[1], shape[2], 8, 64])
        let mn = grouped.min(axis: -1, keepDims: true)
        let mx = grouped.max(axis: -1, keepDims: true)
        let scale = maximum((mx - mn) / 15, MLXArray(Float(1e-6)))
            .asType(.float16).asType(.float32)
        let bias = mn.asType(.float16).asType(.float32)
        let q = clip(round((grouped - bias) / scale), min: 0, max: 15)
        return (q * scale + bias).reshaped(shape).asType(x.dtype)
    }

    /// One threadgroup of 64 lanes per (head, token) row. A lane owns eight
    /// consecutive elements, so it lies wholly inside one 64-element group
    /// and the eight lanes of a group are contiguous and aligned — an xor
    /// butterfly over 1, 2, 4 reduces exactly them. The fp16 rounding of
    /// (scale, bias) happens BEFORE quantization, so the reader reconstructs
    /// with the identical pair the mirror stores.
    ///
    /// This is the D = 512 transcription of the promoted sliding-window
    /// packer `cbv2_kvq4g64_pack_d256_v1`, deliberately duplicated rather
    /// than shared: that packer is a live path and re-composing its source
    /// string would change its text, and a Swift-hosted kernel whose text
    /// changes without its name serves a stale cached body.
    private static let valuePackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kvq4g64_pack_d512_v1",
        inputNames: ["x"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 512;
            constexpr int lanes = 64;
            constexpr int per_lane = D / lanes;           // 8 values
            constexpr int group_size = 64;
            constexpr int payload_words = D / 8;          // 64 (8 nibbles each)

            const int row = int(threadgroup_position_in_grid.z);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* xr = x + size_t(row) * D;
            device uint32_t* out =
                packed_w + size_t(row) * (payload_words + D / group_size);

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = 0; i < per_lane; ++i) {
                const float v = float(xr[lane * per_lane + i]);
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
            for (uint m = 1; m < 8; m <<= 1) {
                vmin = min(vmin, simd_shuffle_xor(vmin, m));
                vmax = max(vmax, simd_shuffle_xor(vmax, m));
            }

            const half hs = half(max((vmax - vmin) / 15.0f, 1e-6f));
            const half hb = half(vmin);
            const float s = float(hs);
            const float b = float(hb);

            uint32_t word = 0u;
            for (int i = 0; i < per_lane; ++i) {
                const float q = metal::rint((float(xr[lane * per_lane + i]) - b) / s);
                word |= uint32_t(clamp(q, 0.0f, 15.0f)) << (4 * i);
            }
            out[lane] = word;
            if (lane % 8 == 0) {
                out[payload_words + lane / 8] =
                    uint32_t(as_type<ushort>(hs)) | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """
    )
}
