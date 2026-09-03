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

/// Auxiliary quantized storage for the five Gemma full-attention D=512
/// layers. The native bf16 arrays remain authoritative; this storage is only
/// a bandwidth-saving reader mirror for the exact B=8 decode kernels.
enum CBv2D512FullKVQuant {
    enum Format: Equatable {
        case q8
        case q4
        case nf8

        var bits: Int {
            switch self {
            case .q8, .nf8: return 8
            case .q4: return 4
            }
        }

        var payloadWords: Int { 512 / (32 / bits) }
        var rowWords: Int { payloadWords + 8 }

        var engageMark: String {
            switch self {
            case .q8: return "d512-fullkv-q8"
            case .q4: return "d512-fullkv-q4"
            case .nf8: return "d512-fullkv-nf8"
            }
        }

        var name: String {
            switch self {
            case .q8: return "q8"
            case .q4: return "q4"
            case .nf8: return "nf8"
            }
        }
    }

    enum Planes: Equatable {
        case kv
        case k

        var count: Int { self == .kv ? 2 : 1 }
    }

    private static func isFalse(_ raw: String?) -> Bool {
        guard let raw else { return false }
        return ["0", "false", "no", "off"].contains(raw.lowercased())
    }

    static let format: Format? = {
        switch (ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_FULL_KV_QUANT_FORMAT"] ?? "nf8").lowercased()
        {
        case "q8": return .q8
        case "q4": return .q4
        case "nf8": return .nf8
        default: return nil
        }
    }()

    /// KV is the default because the measured NF8 reader passed the drift
    /// gate and was faster than the K-only alternative. Invalid values fail
    /// closed instead of silently selecting KV.
    static let planes: Planes? = {
        switch (ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_FULL_KV_QUANT_PLANES"] ?? "kv").lowercased()
        {
        case "kv": return .kv
        case "k": return .k
        default: return nil
        }
    }()

    /// Unset is ON, matching the other tree switches. Invalid format is a
    /// fail-closed bf16 configuration rather than an accidental NF8 choice.
    static let enabled: Bool = !isFalse(
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_D512_FULL_KV_QUANT"]
    ) && format != nil && planes != nil

    /// NF8 uses the inverse of a symmetric mu-law compander. Uniform code
    /// indices therefore put more code points near the centre of [0, 1],
    /// where normalized V has most of its mass, while retaining exact 0/1
    /// endpoints for each affine group.
    private static let nf8Codebook: [Float] = {
        let mu = 1.0
        return (0 ..< 256).map { index in
            let u = 2.0 * Double(index) / 255.0 - 1.0
            let sign = u < 0 ? -1.0 : 1.0
            let expanded = sign * (Foundation.pow(1.0 + mu, abs(u)) - 1.0) / mu
            return Float(0.5 + 0.5 * expanded)
        }
    }()

    static func nf8Value(_ code: UInt32) -> Float {
        nf8Codebook[Int(code)]
    }

    /// Header storage is constant memory for all NF8 readers. The packer
    /// calls the same monotone table through binary search (nine comparisons).
    static let nf8LUTHeader: String = {
        let values = nf8Codebook.map { String(format: "%.9ff", $0) }
            .joined(separator: ", ")
        return """
            constant float NF8_LUT[256] = {\(values)};
            inline uint nf8_nearest(float value) {
                value = clamp(value, 0.0f, 1.0f);
                int lo = 0;
                int hi = 255;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    if (NF8_LUT[mid] < value) lo = mid;
                    else hi = mid;
                }
                return uint(value - NF8_LUT[lo] <= NF8_LUT[hi] - value
                    ? lo : hi);
            }
        """
    }()

    static let verify: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment[
                "DARKBLOOM_GEMMA4_D512_FULL_KV_QUANT_VERIFY"] ?? "")
                .lowercased())
    }()

    /// Local-only isolation switch. The quantized store still writes native
    /// K/V rows and the mirror, but the D512 attention owner keeps reading the
    /// native rows. This is intentionally default-off and is only for
    /// diagnosing store-vs-reader drift.
    static let storeOnlyDiagnostic: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment[
                "DARKBLOOM_GEMMA4_D512_FULL_KV_QUANT_STORE_ONLY"] ?? "")
                .lowercased())
    }()

}

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

    /// Layout `[plane, kvHead, position, rowWords]` in uint32 words. Plane 0
    /// is K (normed + RoPE), and plane 1 is V (normed only) when `planes=kv`.
    /// The two native bf16 arrays above remain the authoritative contract.
    private var quantMirror: MLXArray?

    private var quantEligible: Bool {
        CBv2D512FullKVQuant.enabled && kvHeads == 2 && headDim == 512
    }

    private static let q8PackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_fullkv_q8g64_pack_d512_v1",
        inputNames: ["keys", "values"],
        outputNames: ["packed"],
        source: """
            constexpr int D = 512;
            constexpr int GROUP_SIZE = 64;
            constexpr int GROUPS = D / GROUP_SIZE;
            constexpr int PAYLOAD_WORDS = D / 4;
            constexpr int ROW_WORDS = PAYLOAD_WORDS + GROUPS;

            const int row = int(threadgroup_position_in_grid.x);
            const int plane = row / (HEADS * N);
            const int local = row - plane * (HEADS * N);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* src = (plane == 0 ? keys : values) + local * D;
            device uint32_t* dst = packed + row * ROW_WORDS;

            float x[16];
            for (int i = 0; i < 16; ++i) {
                x[i] = static_cast<float>(src[lane * 16 + i]);
            }
            float lo = 3.402823466e+38F;
            float hi = -3.402823466e+38F;
            for (int i = 0; i < 16; ++i) {
                lo = min(lo, x[i]);
                hi = max(hi, x[i]);
            }
            const int group = lane / 4;
            for (uint mask = 1; mask < 4; mask <<= 1) {
                lo = min(lo, simd_shuffle_xor(lo, mask));
                hi = max(hi, simd_shuffle_xor(hi, mask));
            }
            const half hs = half(max((hi - lo) / 255.0f, 1.0e-6f));
            const half hb = half(lo);
            const float scale = static_cast<float>(hs);
            const float bias = static_cast<float>(hb);

            for (int word = 0; word < 4; ++word) {
                uint32_t packedWord = 0u;
                for (int element = 0; element < 4; ++element) {
                    const float q = metal::rint(
                        (x[word * 4 + element] - bias) / scale);
                    packedWord |= uint32_t(clamp(q, 0.0f, 255.0f))
                        << (8 * element);
                }
                dst[group * 16 + (lane & 3) * 4 + word] = packedWord;
            }
            if ((lane & 3) == 0) {
                dst[PAYLOAD_WORDS + group] =
                    uint32_t(as_type<ushort>(hs))
                    | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """,
        ensureRowContiguous: true
    )

    private static let q4PackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_fullkv_q4g64_pack_d512_v1",
        inputNames: ["keys", "values"],
        outputNames: ["packed"],
        source: """
            constexpr int D = 512;
            constexpr int GROUP_SIZE = 64;
            constexpr int GROUPS = D / GROUP_SIZE;
            constexpr int PAYLOAD_WORDS = D / 8;
            constexpr int ROW_WORDS = PAYLOAD_WORDS + GROUPS;

            const int row = int(threadgroup_position_in_grid.x);
            const int plane = row / (HEADS * N);
            const int local = row - plane * (HEADS * N);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* src = (plane == 0 ? keys : values) + local * D;
            device uint32_t* dst = packed + row * ROW_WORDS;

            float x[16];
            for (int i = 0; i < 16; ++i) {
                x[i] = static_cast<float>(src[lane * 16 + i]);
            }
            float lo = 3.402823466e+38F;
            float hi = -3.402823466e+38F;
            for (int i = 0; i < 16; ++i) {
                lo = min(lo, x[i]);
                hi = max(hi, x[i]);
            }
            const int group = lane / 4;
            for (uint mask = 1; mask < 4; mask <<= 1) {
                lo = min(lo, simd_shuffle_xor(lo, mask));
                hi = max(hi, simd_shuffle_xor(hi, mask));
            }
            const half hs = half(max((hi - lo) / 15.0f, 1.0e-6f));
            const half hb = half(lo);
            const float scale = static_cast<float>(hs);
            const float bias = static_cast<float>(hb);

            for (int word = 0; word < 2; ++word) {
                uint32_t packedWord = 0u;
                for (int element = 0; element < 8; ++element) {
                    const float q = metal::rint(
                        (x[word * 8 + element] - bias) / scale);
                    packedWord |= uint32_t(clamp(q, 0.0f, 15.0f))
                        << (4 * element);
                }
                dst[group * 8 + (lane & 3) * 2 + word] = packedWord;
            }
            if ((lane & 3) == 0) {
                dst[PAYLOAD_WORDS + group] =
                    uint32_t(as_type<ushort>(hs))
                    | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """,
        ensureRowContiguous: true
    )

    private static let nf8PackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_fullkv_nf8g64_pack_d512_v1",
        inputNames: ["keys", "values"],
        outputNames: ["packed"],
        source: """
            constexpr int D = 512;
            constexpr int GROUPS = D / 64;
            constexpr int PAYLOAD_WORDS = D / 4;
            constexpr int ROW_WORDS = PAYLOAD_WORDS + GROUPS;

            const int row = int(threadgroup_position_in_grid.x);
            const int plane = row / (HEADS * N);
            const int local = row - plane * (HEADS * N);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* src = (plane == 0 ? keys : values) + local * D;
            device uint32_t* dst = packed + row * ROW_WORDS;

            float x[16];
            for (int i = 0; i < 16; ++i) {
                x[i] = static_cast<float>(src[lane * 16 + i]);
            }
            float lo = 3.402823466e+38F;
            float hi = -3.402823466e+38F;
            for (int i = 0; i < 16; ++i) {
                lo = min(lo, x[i]);
                hi = max(hi, x[i]);
            }
            const int group = lane / 4;
            for (uint mask = 1; mask < 4; mask <<= 1) {
                lo = min(lo, simd_shuffle_xor(lo, mask));
                hi = max(hi, simd_shuffle_xor(hi, mask));
            }
            const half hs = half(max((hi - lo) / 255.0f, 1.0e-6f));
            const half hb = half(lo);
            const float scale = static_cast<float>(hs);
            const float bias = static_cast<float>(hb);

            for (int word = 0; word < 4; ++word) {
                uint32_t packedWord = 0u;
                for (int element = 0; element < 4; ++element) {
                    const float normalized = clamp(
                        (x[word * 4 + element] - bias) / (scale * 255.0f),
                        0.0f, 1.0f);
                    const uint32_t code = nf8_nearest(normalized);
                    packedWord |= code << (8 * element);
                }
                dst[group * 16 + (lane & 3) * 4 + word] = packedWord;
            }
            if ((lane & 3) == 0) {
                dst[PAYLOAD_WORDS + group] =
                    uint32_t(as_type<ushort>(hs))
                    | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """,
        header: CBv2D512FullKVQuant.nf8LUTHeader,
        ensureRowContiguous: true
    )

    private static func packPairChunk(keys: MLXArray, values: MLXArray) -> MLXArray? {
        guard CBv2D512FullKVQuant.enabled,
            let format = CBv2D512FullKVQuant.format,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            keys.ndim == 4,
            values.shape == keys.shape,
            keys.dim(0) == 1,
            keys.dim(1) == 2,
            keys.dim(3) == 512,
            keys.dim(2) > 0
        else { return nil }

        let kernel: MLXFast.MLXFastKernel
        switch format {
        case .q8: kernel = q8PackKernel
        case .q4: kernel = q4PackKernel
        case .nf8: kernel = nf8PackKernel
        }
        let planeCount = CBv2D512FullKVQuant.planes!.count
        let heads = keys.dim(1)
        let n = keys.dim(2)
        let words = format.rowWords
        let packed = kernel(
            [keys, values],
            template: [("T", keys.dtype), ("HEADS", heads), ("N", n)],
            grid: (planeCount * heads * n * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[planeCount, heads, n, words]],
            outputDTypes: [.uint32]
        )[0]
        CBv2EngageMark.once("d512-fullkv-pack")
        return packed
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
            + (quantMirror?.nbytes ?? 0)
    }

    /// Typed native storage accessor for the D512 attention owner. Keeping
    /// this separate from `cbv2InnerState()` prevents the auxiliary mirror
    /// from changing the native snapshot/cache-shape contract.
    func d512NativeBuffers() -> (keys: MLXArray, values: MLXArray)? {
        guard let keys, let values, cohortPool == nil else { return nil }
        return (keys, values)
    }

    /// Typed accessor for the optional D512 mirror. A nil result is the
    /// normal fail-closed answer for q-off, invalid format, frozen replay,
    /// pooled rows, or any row whose pack path was unable to maintain it.
    func d512QuantMirror() -> MLXArray? {
        guard quantEligible, cohortPool == nil else { return nil }
        return quantMirror
    }

    /// Local-only mirror diagnostic. It samples the seed rows and the first
    /// eight appended rows per D512 layer; the production path never calls
    /// this method unless the explicit VERIFY switch is armed.
    func verifyD512QuantMirrorNewest(position: Int) {
        guard CBv2D512FullKVQuant.verify,
            let native = d512NativeBuffers(), let mirror = quantMirror,
            position >= 0, position < native.keys.dim(2)
        else { return }
        Self.verifyLock.lock()
        let first = !Self.verifiedD512Rows.contains(ObjectIdentifier(self))
        if first { Self.verifiedD512Rows.insert(ObjectIdentifier(self)) }
        Self.verifyLock.unlock()
        guard let format = CBv2D512FullKVQuant.format else { return }

        if first {
            verifyD512QuantMirrorRows(
                native: native, mirror: mirror, format: format,
                positions: [0, 127, 255, 383, 511, 639, 767, 895, 1023],
                label: "seed")
        }
        // A short diagnostic run samples the first eight decode rows. The
        // reader's conversion boundary is exercised by converting the
        // reconstructed fp32 values back to bf16 before measuring error.
        if position >= 1024 && position < 1032 {
            verifyD512QuantMirrorRows(
                native: native, mirror: mirror, format: format,
                positions: [position], label: "step")
        }
    }

    private func verifyD512QuantMirrorRows(
        native: (keys: MLXArray, values: MLXArray), mirror: MLXArray,
        format: CBv2D512FullKVQuant.Format, positions: [Int], label: String
    ) {
        let validPositions = positions.filter {
            $0 >= 0 && $0 < native.keys.dim(2)
        }
        guard !validPositions.isEmpty else { return }

        let valuesPerWord = 32 / format.bits
        let qMax = Float((1 << format.bits) - 1)
        for plane in 0 ..< mirror.dim(0) {
            let source = plane == 0 ? native.keys : native.values
            var squaredError: Float = 0
            var uniformQ8SquaredError: Float = 0
            var squaredSource: Float = 0
            var maxError: Float = 0
            var groupsOverScaleHalf = 0
            var groupsOverScaleHalfFP32 = 0
            var groupCount = 0
            var maxBiasMinusMin: Float = 0
            var maxScaleMinusIdeal: Float = 0
            var maxIdealScale: Float = 0
            var maxStoredScale: Float = 0

            for position in validPositions {
                for head in 0 ..< kvHeads {
                    let sourceRow = source[0, head, position]
                        .asType(.float32).asArray(Float.self)
                    let packed = mirror[plane, head, position]
                        .asArray(UInt32.self)
                    var reconstructed = Array(repeating: Float(0), count: headDim)
                    var uniformQ8 = format == .nf8
                        ? Array(repeating: Float(0), count: headDim)
                        : []

                    for group in 0 ..< (headDim / 64) {
                        let metadata = packed[format.payloadWords + group]
                        let scale = Float(Float16(
                            bitPattern: UInt16(metadata & 0xffff)))
                        let bias = Float(Float16(
                            bitPattern: UInt16(metadata >> 16)))
                        let start = group * 64
                        let end = start + 64
                        var sourceMin = Float.greatestFiniteMagnitude
                        var sourceMax = -Float.greatestFiniteMagnitude
                        for element in start ..< end {
                            sourceMin = min(sourceMin, sourceRow[element])
                            sourceMax = max(sourceMax, sourceRow[element])
                            let word = packed[element / valuesPerWord]
                            let shift = (element % valuesPerWord) * format.bits
                            let code = (word >> UInt32(shift))
                                & UInt32((1 << format.bits) - 1)
                            reconstructed[element] = format == .nf8
                                ? CBv2D512FullKVQuant.nf8Value(code)
                                    * scale * qMax + bias
                                : Float(code) * scale + bias
                            squaredSource += sourceRow[element] * sourceRow[element]
                        }
                        let idealScale = max(
                            (sourceMax - sourceMin) / qMax, 1.0e-6)
                        if format == .nf8 {
                            let uniformScale = Float(Float16(idealScale))
                            let uniformBias = Float(Float16(sourceMin))
                            for element in start ..< end {
                                let code = min(max(
                                    ((sourceRow[element] - uniformBias)
                                        / uniformScale).rounded(),
                                    0), qMax)
                                uniformQ8[element] = code * uniformScale
                                    + uniformBias
                            }
                        }
                        maxBiasMinusMin = max(
                            maxBiasMinusMin, abs(bias - sourceMin))
                        maxScaleMinusIdeal = max(
                            maxScaleMinusIdeal, abs(scale - idealScale))
                        maxIdealScale = max(maxIdealScale, idealScale)
                        maxStoredScale = max(maxStoredScale, scale)
                    }

                    // This is the reader's boundary: q*scale+bias is formed
                    // in fp32 and then cast to the bf16 operand type.
                    let reconstructedBF16 = MLXArray(reconstructed)
                        .asType(.bfloat16).asType(.float32)
                        .asArray(Float.self)
                    if format == .nf8 {
                        let uniformQ8BF16 = MLXArray(uniformQ8)
                            .asType(.bfloat16).asType(.float32)
                            .asArray(Float.self)
                        for element in 0 ..< headDim {
                            let error = uniformQ8BF16[element]
                                - sourceRow[element]
                            uniformQ8SquaredError += error * error
                        }
                    }
                    for element in 0 ..< headDim {
                        let error = reconstructedBF16[element] - sourceRow[element]
                        squaredError += error * error
                        maxError = max(maxError, abs(error))
                    }

                    for group in 0 ..< (headDim / 64) {
                        let metadata = packed[format.payloadWords + group]
                        let scale = Float(Float16(
                            bitPattern: UInt16(metadata & 0xffff)))
                        var groupMaxError: Float = 0
                        var groupMaxErrorFP32: Float = 0
                        let start = group * 64
                        for element in start ..< (start + 64) {
                            groupMaxErrorFP32 = max(
                                groupMaxErrorFP32,
                                abs(reconstructed[element] - sourceRow[element]))
                            groupMaxError = max(
                                groupMaxError,
                                abs(reconstructedBF16[element] - sourceRow[element]))
                        }
                        if groupMaxErrorFP32 > scale / 2 {
                            groupsOverScaleHalfFP32 += 1
                        }
                        if groupMaxError > scale / 2 {
                            groupsOverScaleHalf += 1
                        }
                        groupCount += 1
                    }
                }
            }

            let elementCount = Float(validPositions.count * kvHeads * headDim)
            let sourceRMS = sqrt(squaredSource / elementCount)
            let errorRMS = sqrt(squaredError / elementCount)
            let uniformQ8RMS = format == .nf8
                ? sqrt(uniformQ8SquaredError / elementCount) : 0
            let fraction = Float(groupsOverScaleHalf) / Float(groupCount)
            let line = String(
                format: "[d512-fullkv-verify] format=%@ planes=%d sample=%@ positions=%@ "
                    + "plane=%@ rmsError=%.9g maxAbsError=%.9g sourceRMS=%.9g "
                    + "uniformQ8RMS=%.9g "
                    + "groupsOverScaleHalfFP32=%d/%d fractionFP32=%.9g "
                    + "groupsOverScaleHalfBF16=%d/%d fractionBF16=%.9g "
                    + "maxAbsBiasMinusSourceMin=%.9g maxAbsScaleMinusIdeal=%.9g "
                    + "maxIdealScale=%.9g maxStoredScale=%.9g reconstruction=bf16\n",
                format.name, mirror.dim(0), label,
                validPositions.map(String.init).joined(separator: ","),
                plane == 0 ? "K" : "V", errorRMS, maxError, sourceRMS,
                uniformQ8RMS,
                groupsOverScaleHalfFP32, groupCount,
                Float(groupsOverScaleHalfFP32) / Float(groupCount),
                groupsOverScaleHalf, groupCount, fraction,
                maxBiasMinusMin, maxScaleMinusIdeal,
                maxIdealScale, maxStoredScale)
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    private static let verifyLock = NSLock()
    nonisolated(unsafe) private static var verifiedD512Rows = Set<ObjectIdentifier>()

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

        // Prefill/continuation writes keep the auxiliary mirror in lockstep.
        // The exact B=8 fused decode path writes its one new row in the
        // norm/RoPE store kernel instead, so it never comes through here.
        if quantMirror != nil, !writeQuantMirror(
            keys: newKeys, values: newValues, at: absoluteOffset)
        {
            // Native storage remains valid; a failed pack must make every
            // later D512 reader fall back rather than consume stale bytes.
            quantMirror = nil
        }

        keys![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newKeys
        values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newValues
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

    /// PREFILL-FULLKV-ADOPT: true when this row has no committed K/V of any
    /// kind — no private buffers, no pool, offset 0. That is exactly the
    /// state whose first `update` would take `ensureCapacity`'s
    /// fresh-allocation branch, so it is also exactly the state whose
    /// committed prefix after any first update is the incoming chunk alone.
    var canAdoptFreshChunk: Bool {
        cohortPool == nil && keys == nil && values == nil && absoluteOffset == 0
    }

    /// PREFILL-FULLKV-ADOPT: a strictly fresh row's first update, restructured
    /// to ADOPT the chunk tensor as the K/V storage instead of allocating a
    /// zero-filled capacity buffer and copying the chunk into it (the full-KV
    /// twin of `CBv2WindowedSequenceKV`'s FRESH-RING-ADOPT). A fresh row's
    /// committed prefix is the chunk and nothing else, so the adopted buffer
    /// holds byte-for-byte the values the incumbent path writes; the
    /// differences are all in unobservable or self-correcting territory:
    ///
    /// - `capacity` is `n`, the adopted buffer's true extent (it MUST track
    ///   the buffer: `ensureCapacity` skips growth on `needed <= capacity`,
    ///   and slice appends index into the real dims). The incumbent path's
    ///   `promptLength + initialSlack` headroom is gone, so the FIRST later
    ///   append — append `n'` makes `offset + n' > capacity == offset` always
    ///   — grows by the same doubling reallocation `ensureCapacity` always
    ///   performs, into a fresh PRIVATE buffer. The growth mechanism, policy
    ///   and cap are unchanged; only its first firing moves earlier.
    /// - Because `capacity == absoluteOffset` for as long as the adopted
    ///   buffer lives, the adopted storage is read-only by construction:
    ///   every plain append takes the growth branch (write lands in the new
    ///   private buffer), and every fused in-place writer (WRITE-022 /
    ///   WRITE-016-D512) is admitted only on `dim(2) >= keyLength` headroom
    ///   this buffer never has — the aliased chunk bytes can never be
    ///   mutated through the row.
    /// - `byteCount` reports the adopted buffers' `nbytes` — truthful for
    ///   what is actually held (the caller's rectangle, kept alive by the
    ///   eight row slices) and never smaller than what admission charged:
    ///   the contiguous backend bills `max(byteCount, reservation)`.
    ///
    /// Return views, `absoluteOffset`, `retainedCount`, `snapshot`,
    /// `rollback` and `cbv2InnerState` (still exactly two arrays) are the
    /// incumbent path's, with identical values.
    func adoptFreshChunk(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        precondition(
            canAdoptFreshChunk,
            "CBv2FullSequenceKV: fresh-chunk adoption requires a strictly fresh row")
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
        keys = newKeys
        values = newValues
        capacity = n
        absoluteOffset = n
        if quantEligible {
            quantMirror = Self.packPairChunk(keys: newKeys, values: newValues)
        }
        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
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
        // Keep the published/cache state exactly native: snapshots, prefix
        // cache shape checks, rollback, and model-facing state all continue
        // to see the authoritative bf16 K/V pair. The decode owner gets the
        // optional mirror through d512QuantMirror() instead.
        guard let keys, let values else { return [] }
        return [keys, values]
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

        // The pooled layout has no mirror plane. Refuse the migration while
        // the D512 feature is armed so a later exact fast-path miss cannot
        // silently reinterpret the pool's batch axis as a private mirror.
        guard !CBv2D512FullKVQuant.enabled
            || !rows.contains(where: { $0.kvHeads == 2 && $0.headDim == 512 })
        else { return nil }

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
            if quantEligible,
                keyTemplate.dtype == .bfloat16,
                valueTemplate.dtype == .bfloat16,
                keyTemplate.dim(3) == headDim,
                valueTemplate.dim(3) == headDim
            {
                quantMirror = MLXArray.zeros(
                    [CBv2D512FullKVQuant.planes!.count, kvHeads, capacity,
                        CBv2D512FullKVQuant.format!.rowWords],
                    dtype: .uint32)
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
        if let quantMirror, let format = CBv2D512FullKVQuant.format {
            self.quantMirror = concatenated(
                [
                    quantMirror,
                    MLXArray.zeros(
                        [CBv2D512FullKVQuant.planes!.count, kvHeads, growth,
                            format.rowWords], dtype: .uint32),
                ],
                axis: 2)
        }
        capacity = newCapacity
    }

    private func writeQuantMirror(
        keys newKeys: MLXArray, values newValues: MLXArray, at offset: Int
    ) -> Bool {
        guard let quantMirror,
            let packed = Self.packPairChunk(keys: newKeys, values: newValues),
            offset >= 0,
            offset + newKeys.dim(2) <= capacity
        else { return false }
        quantMirror[.ellipsis, offset ..< (offset + newKeys.dim(2)), 0...] = packed
        return true
    }
}
