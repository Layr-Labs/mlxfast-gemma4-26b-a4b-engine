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

    /// D512Q4 packer. One threadgroup of 32 lanes per (plane, row, head):
    /// each lane owns 16 consecutive elements of the 512-wide vector, which
    /// spans exactly one quarter of a 64-element group boundary, so the
    /// group reduction is a butterfly over the four lanes sharing a group.
    ///
    /// Row layout matches the sliding mirror: `D/8` payload words holding
    /// eight 4-bit values each, then `D/64` tail words holding that group's
    /// fp16 scale in the low half and bias in the high half.
    private static let packPairKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_d512_q4g64_pack_pair_v1",
        inputNames: ["keys", "values"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 512;
            constexpr int simd_width = 32;
            constexpr int per_lane = D / simd_width;      // 16 values
            constexpr int group_size = 64;
            constexpr int lanes_per_group = group_size / per_lane;   // 4
            constexpr int payload_words = D / 8;          // 64
            constexpr int row_words = payload_words + D / group_size; // 72

            const int slot = int(threadgroup_position_in_grid.x);
            const int plane = slot / (ROWS * HEADS);
            const int rest = slot % (ROWS * HEADS);
            const int row = rest / HEADS;
            const int head = rest % HEADS;
            const int lane = int(thread_position_in_threadgroup.x);

            const device T* src =
                (plane == 0 ? keys : values) + (size_t(row) * HEADS + head) * D;
            device uint32_t* out = packed_w + size_t(slot) * row_words;

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = 0; i < per_lane; ++i) {
                const float v = float(src[lane * per_lane + i]);
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
            // Four lanes share a 64-element group and are contiguous and
            // aligned, so xor over 1 and 2 reduces exactly them.
            for (uint m = 1; m < lanes_per_group; m <<= 1) {
                vmin = min(vmin, simd_shuffle_xor(vmin, m));
                vmax = max(vmax, simd_shuffle_xor(vmax, m));
            }

            const half hs = half(max((vmax - vmin) / 15.0f, 1e-6f));
            const half hb = half(vmin);
            const float s = float(hs);
            const float b = float(hb);

            // Each lane writes its own two payload words (16 values).
            for (int w = 0; w < per_lane / 8; ++w) {
                uint32_t word = 0u;
                for (int i = 0; i < 8; ++i) {
                    const float q = metal::rint(
                        (float(src[lane * per_lane + w * 8 + i]) - b) / s);
                    word |= uint32_t(clamp(q, 0.0f, 15.0f)) << (4 * i);
                }
                out[lane * (per_lane / 8) + w] = word;
            }
            if (lane % lanes_per_group == 0) {
                out[payload_words + lane / lanes_per_group] =
                    uint32_t(as_type<ushort>(hs)) | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """
    )

    /// `[rows, heads, 1, D]` keys and values -> packed
    /// `[2, rows, heads, 1, D/8 + D/64]` uint32.
    static func packPairQ4(
        keys: MLXArray, values: MLXArray, rowCount: Int, kvHeads: Int, headDim: Int
    ) -> MLXArray {
        let slots = 2 * rowCount * kvHeads
        let words = headDim / 8 + headDim / 64
        return packPairKernel(
            [keys, values],
            template: [("T", keys.dtype), ("ROWS", rowCount), ("HEADS", kvHeads)],
            grid: (slots * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[2, rowCount, kvHeads, 1, words]],
            outputDTypes: [.uint32]
        )[0]
    }

    /// `MLX_D512_QUANT=0` disables the full-attention mirror wholesale.
    static let quantEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_D512_QUANT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

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

    /// D512Q4: 4-bit affine mirror of this row's full-attention K/V, on the
    /// same contract the sliding-window mirror uses (64-element groups, one
    /// fp16 scale/bias pair each, eight values to a word). The bf16 buffers
    /// stay the source of truth for every path except the batched decode
    /// read. Layout `[2, kvHeads, capacity, headDim/8 + headDim/64]` uint32:
    /// 72 words, 288 bytes, against 1024 bytes of bf16.
    ///
    /// The mirror lives on the ROW, not on the ATT-008 pool: the promoted
    /// WRITE-022-D512 road requires `dim(0) == 1` buffers, so the hot decode
    /// path runs on private per-row storage and never sees the pool.
    private var quantMirror: MLXArray?

    /// `MLX_D512_QUANT=0` disables the full-attention mirror wholesale.
    static let quantEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_D512_QUANT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// The mirror for this row, or nil when the quantized road cannot serve
    /// it. Callers treat nil as all-or-nothing across the cohort.
    var decodeQuantMirror: MLXArray? {
        guard Self.quantEnabled, headDim == 512, let quantMirror else { return nil }
        return quantMirror
    }

    /// Pack `count` tokens into the mirror at `offset`, one pack dispatch for
    /// the whole chunk. Prefill and decode both come through here, so the
    /// mirror covers exactly the positions the decode read attends.
    func appendQuantMirror(keys newKeys: MLXArray, values newValues: MLXArray, at offset: Int, count n: Int) {
        guard Self.quantEnabled, headDim == 512, quantMirror != nil else { return }
        // packPairQ4 wants [rows, heads, 1, D] and emits [2, rows, heads, 1, W];
        // here the "rows" axis carries the chunk's tokens instead.
        // [1, heads, n, D] -> [n, heads, 1, D]
        let k = newKeys.transposed(0, 2, 1, 3).reshaped([n, kvHeads, 1, headDim])
        let v = newValues.transposed(0, 2, 1, 3).reshaped([n, kvHeads, 1, headDim])
        let packed = CBv2FullDecodeCohortPool.packPairQ4(
            keys: k, values: v, rowCount: n, kvHeads: kvHeads, headDim: headDim)
        // packed: [2, n, heads, 1, W] -> mirror slice [2, heads, n, W]
        let words = headDim / 8 + headDim / 64
        quantMirror![0..., 0..., offset ..< (offset + n), 0...] =
            packed.reshaped([2, n, kvHeads, words]).transposed(0, 2, 1, 3)
        CBv2EngageMark.once("d512q4prefill")
    }

    /// Allocate (or reallocate after a growth) the mirror. A growth drops the
    /// old mirror instead of copying it; the read road refuses while the
    /// mirror does not cover the attended prefix, so a dropped mirror
    /// degrades to the bf16 road rather than serving stale bytes.
    func allocateQuantMirror() {
        guard Self.quantEnabled, headDim == 512, keys != nil else { return }
        quantMirror = MLXArray.zeros(
            [2, kvHeads, capacity, headDim / 8 + headDim / 64], dtype: .uint32)
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
        return (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
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
        // D512Q4: the mirror has to cover the PROMPT too. Two ranked runs
        // died because it did not: the decode read walked every committed
        // position while the mirror only ever held decode tokens, so the
        // prompt's 1024 positions dequantized from zeros and attention was
        // garbage from the first step. The local free-run divergence at
        // position 0 was that symptom, not a near-tie flip.
        appendQuantMirror(keys: newKeys, values: newValues, at: absoluteOffset, count: n)
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
            allocateQuantMirror()
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
        // D512Q4: grow the mirror the same way the bf16 buffers grow, by
        // concatenation. Reallocating it empty here wiped every packed
        // position, and since capacity starts at exactly the prompt length
        // the very first decode step triggers the growth — so the mirror lost
        // the whole prompt one step in. That is what killed dbf08f01, and it
        // is the same failure as never packing prefill, wearing a different
        // hat.
        if let mirror = quantMirror {
            let words = headDim / 8 + headDim / 64
            quantMirror = concatenated(
                [mirror, MLXArray.zeros([2, kvHeads, growth, words], dtype: .uint32)],
                axis: 2)
        }
        capacity = newCapacity
        if quantMirror == nil { allocateQuantMirror() }
    }
}
