// WindowedSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for SLIDING-WINDOW attention.
//
// The window is enforced by STORAGE EVICTION keyed to absolute positions —
// never by masks over shared buffers (report 10 §4 invariant 6). Storage is
// a ring of exactly `window` slots; the token at absolute position `p` lives
// in physical slot `p % window`, so eviction is simply "newer tokens
// overwrite slots window positions behind them". The RECENT end is always
// kept. `absoluteOffset` keeps counting past the window.

import Foundation
import MLX
import MLXFast

public final class CBv2WindowedSequenceKV: CBv2DecodeRootCompactionCapableSequenceKV,
    CBv2InnerStateProviding
{

    public let window: Int

    public private(set) var absoluteOffset: Int

    private var oldestValidPosition: Int

    public var retainedCount: Int { absoluteOffset - oldestValidPosition }

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    private var quantMirror: MLXArray?

    private(set) var bf16RingStale = false

    static let quantEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let quantSimulate: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KV_QUANT_SIM"] ?? "")
                .lowercased())
    }()

    static let q4BF16RingElideEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_Q4_BF16_ELIDE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private var quantEligible: Bool {
        Self.quantEnabled && headDim == 256 && window > 0
            && (window & (window - 1)) == 0
    }

    private var borrowableChunkViews: (keys: MLXArray, values: MLXArray)?

    private var speculativeWriteArmed = false

    private var staged: (keys: MLXArray, values: MLXArray, basePosition: Int)?

    public init(window: Int, kvHeads: Int, headDim: Int, initialOffset: Int = 0) {
        precondition(window > 0, "CBv2WindowedSequenceKV: window must be > 0")
        precondition(initialOffset >= 0, "CBv2WindowedSequenceKV: negative initialOffset")
        self.window = window
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.absoluteOffset = initialOffset
        self.oldestValidPosition = initialOffset
    }

    public var byteCount: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (staged.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
            + (quantMirror?.nbytes ?? 0)
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2WindowedSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2WindowedSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2WindowedSequenceKV: keys/values token count mismatch")
        precondition(n > 0, "CBv2WindowedSequenceKV: empty update")

        if speculativeWriteArmed {
            return stageSpeculativeUpdate(newKeys: newKeys, newValues: newValues, count: n)
        }
        precondition(
            !bf16RingStale,
            "CBv2WindowedSequenceKV: update after fused quant elided writes — "
                + "BF16 ring is stale; set MLX_KV_Q4_BF16_ELIDE=0 to keep it authoritative"
        )

        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)

        if n == 1 {
            writeDecodeToken(keys: newKeys, values: newValues)
            return (
                temporalOrder(keys!, from: oldestValidPosition, to: absoluteOffset),
                temporalOrder(values!, from: oldestValidPosition, to: absoluteOffset)
            )
        }

        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        var kParts = ringSlices(keys!, from: historyFrom, to: absoluteOffset)
        var vParts = ringSlices(values!, from: historyFrom, to: absoluteOffset)
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        let writeCount = min(n, window)
        let firstWritten = absoluteOffset + n - writeCount
        let kTail = writeCount == n ? newKeys : newKeys[.ellipsis, (n - writeCount)..., 0...]
        let vTail = writeCount == n ? newValues : newValues[.ellipsis, (n - writeCount)..., 0...]
        writeRing(keys!, tokens: kTail, firstPosition: firstWritten, mirrorPlane: 0)
        writeRing(values!, tokens: vTail, firstPosition: firstWritten, mirrorPlane: 1)

        absoluteOffset += n
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)

        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    func decodeRingWrite(keys newKeys: MLXArray, values newValues: MLXArray) {
        precondition(staged == nil && newKeys.dim(2) == 1 && newValues.dim(2) == 1)
        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)
        writeDecodeToken(keys: newKeys, values: newValues)
    }

    func decodeRingWriteBF16Only(keys newKeys: MLXArray, values newValues: MLXArray) {
        precondition(
            staged == nil && newKeys.dim(2) == 1 && newValues.dim(2) == 1
                && keys != nil && values != nil && retainedCount == window)
        borrowableChunkViews = nil
        writeRing(keys!, tokens: newKeys, firstPosition: absoluteOffset)
        writeRing(values!, tokens: newValues, firstPosition: absoluteOffset)
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    func advanceDecodeRingAfterQuantWrite() {
        precondition(
            staged == nil && keys != nil && retainedCount == window,
            "CBv2WindowedSequenceKV: fused quant advance outside a full-ring decode step")
        borrowableChunkViews = nil
        bf16RingStale = true
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    var decodeRingView: (keys: MLXArray, values: MLXArray, start: Int)? {
        guard staged == nil, let keys, let values, retainedCount == window else { return nil }
        return (keys, values, oldestValidPosition % window)
    }

    var decodeRingQuantView: MLXArray? {
        guard staged == nil, quantMirror != nil, retainedCount == window
        else { return nil }
        return quantMirror
    }

    var decodeRingQuantViewBeforeWrite: (mirror: MLXArray, start: Int)? {
        guard staged == nil, let quantMirror, retainedCount == window else { return nil }
        return (quantMirror, (oldestValidPosition + 1) % window)
    }

    var decodeRingViewBeforeWrite: (keys: MLXArray, values: MLXArray, start: Int)? {
        guard staged == nil, !bf16RingStale, let keys, let values, retainedCount == window else { return nil }
        return (keys, values, (oldestValidPosition + 1) % window)
    }

    func advanceDecodeRingAfterFusedWrite() {
        precondition(
            staged == nil && keys != nil && retainedCount == window,
            "CBv2WindowedSequenceKV: fused ring advance outside a full-ring decode step")
        borrowableChunkViews = nil
        diagnosticFusedDispatch()
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    // MARK: - Speculative (MTP) staging

    public var supportsSpeculativeWrites: Bool { true }

    public func beginSpeculativeWrite() {
        precondition(
            !speculativeWriteArmed,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite while already armed")
        precondition(
            staged == nil,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite with a staged update pending — commit first"
        )
        speculativeWriteArmed = true
    }

    private func stageSpeculativeUpdate(
        newKeys: MLXArray, newValues: MLXArray, count n: Int
    ) -> (MLXArray, MLXArray) {
        let logical = snapshot()
        let historyCount = min(logical.keys.dim(2), window - 1)
        let historyFrom = logical.keys.dim(2) - historyCount
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if historyCount > 0 {
            kParts.append(logical.keys[.ellipsis, historyFrom..., 0...])
            vParts.append(logical.values[.ellipsis, historyFrom..., 0...])
        }
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        if let existing = staged {
            let confirmed = absoluteOffset - existing.basePosition
            let existingKeys = existing.keys[.ellipsis, ..<confirmed, 0...]
            let existingValues = existing.values[.ellipsis, ..<confirmed, 0...]
            staged = (
                concatenated([existingKeys, newKeys], axis: 2),
                concatenated([existingValues, newValues], axis: 2),
                existing.basePosition)
        } else {
            staged = (newKeys, newValues, absoluteOffset)
        }
        absoluteOffset += n
        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    public func commitSpeculativeWrite() {
        speculativeWriteArmed = false
        guard let staged else { return }
        self.staged = nil
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            allocateIfNeeded(keyTemplate: staged.keys, valueTemplate: staged.values)
            let writeCount = min(confirmed, window)
            let skip = confirmed - writeCount
            writeRing(
                keys!, tokens: staged.keys[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount, mirrorPlane: 0)
            writeRing(
                values!, tokens: staged.values[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount, mirrorPlane: 1)
        }
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
        borrowableChunkViews = nil
    }

    public func borrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if let views = borrowableChunkViews { return views }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    func decodeBorrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if staged != nil {
            precondition(
                borrowableChunkViews != nil,
                "CBv2WindowedSequenceKV: staged decode borrow after rollback — commit first")
            let views = borrowableChunkViews!
            precondition(
                views.keys.dim(2) <= window && views.values.dim(2) <= window,
                "CBv2WindowedSequenceKV: staged decode borrow exceeds window \(window)")
            return views
        }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        precondition(
            !bf16RingStale,
            "CBv2WindowedSequenceKV: snapshot after fused quant elided writes — "
                + "BF16 ring is stale; set MLX_KV_Q4_BF16_ELIDE=0 to keep it authoritative"
        )
        if let staged {
            return stagedSnapshot(staged)
        }
        guard let keys, let values, retainedCount > 0 else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            temporalOrder(keys, from: oldestValidPosition, to: absoluteOffset),
            temporalOrder(values, from: oldestValidPosition, to: absoluteOffset),
            absoluteOffset
        )
    }

    private func stagedSnapshot(
        _ staged: (keys: MLXArray, values: MLXArray, basePosition: Int)
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if let keys, let values, staged.basePosition > oldestValidPosition {
            kParts = ringSlices(keys, from: oldestValidPosition, to: staged.basePosition)
            vParts = ringSlices(values, from: oldestValidPosition, to: staged.basePosition)
        }
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            kParts.append(staged.keys[.ellipsis, ..<confirmed, 0...])
            vParts.append(staged.values[.ellipsis, ..<confirmed, 0...])
        }
        guard !kParts.isEmpty else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2),
            vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2),
            absoluteOffset
        )
    }

    public func fastForward(to offset: Int) {
        precondition(
            keys == nil && absoluteOffset == oldestValidPosition,
            "CBv2WindowedSequenceKV.fastForward requires a fresh state")
        precondition(
            !speculativeWriteArmed && staged == nil,
            "CBv2WindowedSequenceKV.fastForward with a speculative write pending")
        precondition(offset >= absoluteOffset, "fastForward cannot move backwards")
        absoluteOffset = offset
        oldestValidPosition = offset
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2WindowedSequenceKV.rollback: negative n")
        if let staged {
            precondition(
                n <= absoluteOffset - staged.basePosition,
                "CBv2WindowedSequenceKV.rollback(\(n)) exceeds staged range "
                    + "\(absoluteOffset - staged.basePosition)")
            absoluteOffset -= n
            borrowableChunkViews = nil
            return
        }
        precondition(
            n <= retainedCount,
            "CBv2WindowedSequenceKV.rollback(\(n)) exceeds retained \(retainedCount)")
        absoluteOffset -= n
        borrowableChunkViews = nil
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values, quantMirror].compactMap { $0 }
    }

    // MARK: - MTP mirror road (see MTP/CBv2MTPMirrorOps.swift)

    var mtpMirrorRoadAvailable: Bool {
        staged == nil && !speculativeWriteArmed && quantMirror != nil
            && retainedCount == window
    }

    var mtpQuantMirror: MLXArray? { quantMirror }

    /// True when the bf16 ring is stale (the mirror is authoritative), so a
    /// verify round must take the mirror road or not speculate at all.
    var mtpNeedsMirrorRoad: Bool { bf16RingStale }

    var mtpSlotBase: Int { absoluteOffset % window }

    func mtpRollbackMirrorRoad(_ n: Int) {
        precondition(n >= 0, "CBv2WindowedSequenceKV.mtpRollbackMirrorRoad: negative n")
        precondition(
            staged == nil && !speculativeWriteArmed,
            "CBv2WindowedSequenceKV.mtpRollbackMirrorRoad on a staged row")
        precondition(
            n <= retainedCount,
            "CBv2WindowedSequenceKV.mtpRollbackMirrorRoad(\(n)) exceeds retained \(retainedCount)")
        absoluteOffset -= n
        oldestValidPosition -= n
        borrowableChunkViews = nil
    }

    func mtpDequantizedRetainedViews(fence: MLXArray) -> (keys: MLXArray, values: MLXArray)? {
        guard let quantMirror, staged == nil, retainedCount > 0, headDim == 256 else { return nil }
        return CBv2MTPMirrorOps.dequantize(
            mirror: quantMirror, start: oldestValidPosition % window,
            count: retainedCount, kvHeads: kvHeads, headDim: headDim, window: window,
            fence: fence)
    }

    /// Chained rounds: the same view with the first retained slot taken from
    /// a `[1]` int32 device value (the row's device position mod window).
    func mtpDequantizedRetainedViews(
        startDevice: MLXArray, fence: MLXArray
    ) -> (keys: MLXArray, values: MLXArray)? {
        guard let quantMirror, staged == nil, retainedCount == window, headDim == 256 else { return nil }
        return CBv2MTPMirrorOps.dequantize(
            mirror: quantMirror, start: startDevice,
            count: window, kvHeads: kvHeads, headDim: headDim, window: window,
            fence: fence)
    }

    // MARK: - Ring geometry

    private func writeDecodeToken(keys newKeys: MLXArray, values newValues: MLXArray) {
        borrowableChunkViews = nil
        let paired = writePairedMirror(
            keys: newKeys, values: newValues, firstPosition: absoluteOffset)
        writeRing(
            keys!, tokens: newKeys, firstPosition: absoluteOffset,
            mirrorPlane: paired ? nil : 0)
        writeRing(
            values!, tokens: newValues, firstPosition: absoluteOffset,
            mirrorPlane: paired ? nil : 1)
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    private func ringSlices(_ array: MLXArray, from: Int, to: Int) -> [MLXArray] {
        guard to > from else { return [] }
        let count = to - from
        precondition(count <= window, "ring range exceeds window")
        let start = from % window
        if start + count <= window {
            return [array[.ellipsis, start ..< (start + count), 0...]]
        }
        return [
            array[.ellipsis, start ..< window, 0...],
            array[.ellipsis, 0 ..< (start + count - window), 0...],
        ]
    }

    private func temporalOrder(_ array: MLXArray, from: Int, to: Int) -> MLXArray {
        let slices = ringSlices(array, from: from, to: to)
        switch slices.count {
        case 0:
            return array[.ellipsis, 0 ..< 0, 0...]
        case 1:
            return slices[0]
        default:
            return concatenated(slices, axis: 2)
        }
    }

    private func writeRing(
        _ buffer: MLXArray, tokens: MLXArray, firstPosition: Int, mirrorPlane: Int? = nil
    ) {
        var tokens = tokens
        let n = tokens.dim(2)
        precondition(n <= window, "writeRing: more tokens than slots")
        if let plane = mirrorPlane, quantMirror != nil {
            writeMirror(plane: plane, tokens: tokens, firstPosition: firstPosition)
            if Self.quantSimulate {
                tokens = Self.quantRoundTrip(tokens)
            }
        }
        let start = firstPosition % window
        if start + n <= window {
            buffer[.ellipsis, start ..< (start + n), 0...] = tokens
        } else {
            let first = window - start
            buffer[.ellipsis, start ..< window, 0...] = tokens[.ellipsis, ..<first, 0...]
            buffer[.ellipsis, 0 ..< (n - first), 0...] = tokens[.ellipsis, first..., 0...]
        }
    }

    // MARK: - KVQ-001 quantized mirror

    private static func quantParams(_ f: MLXArray) -> (scale: MLXArray, bias: MLXArray) {
        let mn = f.min(axis: -1, keepDims: true)
        let mx = f.max(axis: -1, keepDims: true)
        let scale = maximum((mx - mn) / 255, MLXArray(Float(1e-6)))
            .asType(.float16).asType(.float32)
        let bias = mn.asType(.float16).asType(.float32)
        return (scale, bias)
    }

    private static func quantPack(_ x: MLXArray) -> MLXArray {
        let f = x[0].asType(.float32)
        let heads = f.dim(0), n = f.dim(1), d = f.dim(2)
        let groups = d / 64
        let grouped = f.reshaped([heads, n, groups, 64])
        let mn = grouped.min(axis: -1, keepDims: true)
        let mx = grouped.max(axis: -1, keepDims: true)
        let scale = maximum((mx - mn) / 15, MLXArray(Float(1e-6)))
            .asType(.float16).asType(.float32)
        let bias = mn.asType(.float16).asType(.float32)
        let q = clip(round((grouped - bias) / scale), min: 0, max: 15)
            .asType(.uint32)
            .reshaped([heads, n, d / 8, 8])
        var payload = MLXArray.zeros([heads, n, d / 8], dtype: .uint32)
        for i in 0 ..< 8 {
            payload = payload + (q[.ellipsis, i] << MLXArray(Int32(4 * i)))
        }
        let pair = concatenated(
            [scale.asType(.float16), bias.asType(.float16)], axis: -1)
        let tail = pair.view(dtype: .uint32).reshaped([heads, n, groups])
        return concatenated([payload.asType(.uint32), tail], axis: -1)
    }

    static let gpuPackEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_GPUPACK"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let pairedMirrorWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_PAIRWRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let q4FusedMirrorWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_Q4_FUSED_WRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let gpuPackCheck: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KV_QUANT_PACK_CHECK"] ?? "")
                .lowercased())
    }()

    private static let quantPackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kvq4g64_pack_d256_v1",
        inputNames: ["x"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 256;
            constexpr int simd_width = 32;
            constexpr int per_lane = D / simd_width;      // 8 values
            constexpr int group_size = 64;
            constexpr int payload_words = D / 8;          // 32 (8 nibbles each)

            const int row = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* xr = x + row * D;
            device uint32_t* out = packed_w + row * (payload_words + D / group_size);

            // A lane owns 8 consecutive elements, so it lies wholly inside one
            // 64-element group; the eight lanes of a group are contiguous and
            // aligned, so an xor butterfly over 1,2,4 reduces exactly them.
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

    private static let quantPackPairKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kvq4g64_pack_pair_d256_v1",
        inputNames: ["keys", "values"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 256;
            constexpr int simd_width = 32;
            constexpr int per_lane = D / simd_width;      // 8 values
            constexpr int group_size = 64;
            constexpr int payload_words = D / 8;          // 32
            constexpr int row_words = payload_words + D / group_size;

            const int row = int(threadgroup_position_in_grid.x);
            const int plane = row / HEADS;
            const int head = row % HEADS;
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* src = (plane == 0 ? keys : values) + head * D;
            device uint32_t* out = packed_w + row * row_words;

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = 0; i < per_lane; ++i) {
                const float v = float(src[lane * per_lane + i]);
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
                const float q = metal::rint((float(src[lane * per_lane + i]) - b) / s);
                word |= uint32_t(clamp(q, 0.0f, 15.0f)) << (4 * i);
            }
            out[lane] = word;
            if (lane % 8 == 0) {
                out[payload_words + lane / 8] =
                    uint32_t(as_type<ushort>(hs)) | (uint32_t(as_type<ushort>(hb)) << 16);
            }
        """
    )

    // MARK: - KVQ-DIAG: compute-and-discard probe

    static let diagEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    private static let diagBlocks = 256
    private static let diagStride = 64
    nonisolated(unsafe) private static var diagCounter = 0
    private static let diagLock = NSLock()
    static let selfTestArmed: Bool = ["1","true","yes","on"].contains(
        (ProcessInfo.processInfo.environment["MLX_KVQ4_SELFTEST"] ?? "").lowercased())
    nonisolated(unsafe) private static var shapeLogged = false
    nonisolated(unsafe) private static var mirrorChecked = false
    nonisolated(unsafe) private static var diagDispatched = false
    nonisolated(unsafe) private static var diagRejected = false

    private static func diagDispatchOnce() {
        diagLock.lock(); let fresh = !diagDispatched; diagDispatched = true; diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] dispatched\n".utf8))
        }
    }

    private static func diagRejectOnce(_ why: String) {
        diagLock.lock(); let fresh = !diagRejected; diagRejected = true; diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] skipped: \(why)\n".utf8))
        }
    }

private static let diagQuantReadKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_ring_2pass_a_q8_d256_g2_diag_b\(diagBlocks)_v1",
        inputNames: [
            "queries",
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "starts",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int row_stride = D + 4;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            constexpr int row_words = row_stride / 4;
            const device uint32_t* mirror_w = m0;
            switch (batch_index) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }
            const uint start = starts[batch_index];

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            int slot = int((start + block) % N);
            device T* partial = partials
                + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
            device float* sum_out = sums + batch_head * BLOCKS + block;
            device float* max_out = maxs + batch_head * BLOCKS + block;

            thread float q[values_per_lane];
            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const device uint32_t* krow_w =
                    mirror_w + (kv_head * N + slot) * row_words;
                const device uint32_t* vrow_w =
                    mirror_w + ((KV_HEADS + kv_head) * N + slot) * row_words;
                const uint32_t ktw = krow_w[D / 4];
                const uint32_t vtw = vrow_w[D / 4];
                const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                const float kb = float(as_type<half>(ushort(ktw >> 16)));
                const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                const float vb = float(as_type<half>(ushort(vtw >> 16)));
                const uint32_t kw0 = krow_w[lane * 2];
                const uint32_t kw1 = krow_w[lane * 2 + 1];
                const uint32_t vw0 = vrow_w[lane * 2];
                const uint32_t vw1 = vrow_w[lane * 2 + 1];
                float score = 0.0f;
                for (int element = 0; element < 4; ++element) {
                    score += q[element]
                        * fma(float((kw0 >> (8 * element)) & 0xffu), ks, kb);
                }
                for (int element = 0; element < 4; ++element) {
                    score += q[4 + element]
                        * fma(float((kw1 >> (8 * element)) & 0xffu), ks, kb);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < 4; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor
                            * fma(float((vw0 >> (8 * element)) & 0xffu), vs, vb);
                    accumulator[4 + element] = accumulator[4 + element] * old_factor
                        + score_factor
                            * fma(float((vw1 >> (8 * element)) & 0xffu), vs, vb);
                }

                slot += BLOCKS;
                if (slot >= N) slot -= N;
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

    private func diagnosticQuantDispatch() {
        guard Self.diagEnabled, let quantMirror else { return }
        guard retainedCount == window, headDim == 256, kvHeads == 8, window == 1024
        else {
            Self.diagRejectOnce(
                "retained=\(retainedCount) window=\(window) headDim=\(headDim) kvHeads=\(kvHeads)")
            return
        }
        Self.diagLock.lock()
        Self.diagCounter += 1
        let fire = Self.diagCounter % Self.diagStride == 0
        Self.diagLock.unlock()
        guard fire else { return }
        let batch = 8
        let queryHeads = 16
        let queries = MLXArray.zeros(
            [batch, queryHeads, 1, headDim], dtype: .bfloat16)
        let starts = MLXArray(
            Array(repeating: UInt32(oldestValidPosition % window), count: batch),
            [batch])
        let mirrors = Array(repeating: quantMirror, count: batch)
        let out = Self.diagQuantReadKernel(
            [queries] + mirrors + [starts],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", window),
                ("GQA", queryHeads / kvHeads),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", Self.diagBlocks),
            ],
            grid: (kvHeads * 32, batch * (queryHeads / kvHeads), Self.diagBlocks),
            threadGroup: (32, queryHeads / kvHeads, 1),
            outputShapes: [
                [batch, queryHeads, 1, Self.diagBlocks, headDim],
                [batch, queryHeads, 1, Self.diagBlocks],
                [batch, queryHeads, 1, Self.diagBlocks],
            ],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        asyncEval(out[1])
        Self.diagDispatchOnce()
    }

private static let diagFusedKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ringwrite_sdpa_2pass_a_q8_d256_g2_diagfused_b\(diagBlocks)_v1",
        inputNames: [
            "queries",
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "starts", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["partials", "sums", "maxs", "fence"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr uint ring_mask = uint(N - 1);
            constexpr int row_stride = D + 4;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            constexpr int row_words = row_stride / 4;
            const device uint32_t* mirror_w = m0;
            switch (batch_index) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            const device uint32_t* mkeys_w =
                mirror_w + kv_head * N * row_words;
            const device uint32_t* mvalues_w =
                mirror_w + (KV_HEADS + kv_head) * N * row_words;
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const uint ring_start = starts[batch_index];
            const uint write_slot = (ring_start + ring_mask) & ring_mask;
            if (block == 0 && query_head_in_group == 0) {
                float kmn = 3.402823466e+38F;
                float kmx = -3.402823466e+38F;
                float vmn = 3.402823466e+38F;
                float vmx = -3.402823466e+38F;
                for (int element = 0; element < values_per_lane; ++element) {
                    const float kx = float(new_key[element]);
                    const float vx = float(new_value[element]);
                    kmn = min(kmn, kx);
                    kmx = max(kmx, kx);
                    vmn = min(vmn, vx);
                    vmx = max(vmx, vx);
                }
                kmn = simd_min(kmn);
                kmx = simd_max(kmx);
                vmn = simd_min(vmn);
                vmx = simd_max(vmx);
                const half khs = half(max((kmx - kmn) / 255.0f, 1e-6f));
                const half khb = half(kmn);
                const half vhs = half(max((vmx - vmn) / 255.0f, 1e-6f));
                const half vhb = half(vmn);
                const float kqs = float(khs);
                const float kqb = float(khb);
                const float vqs = float(vhs);
                const float vqb = float(vhb);
                device uint32_t* mk_w = const_cast<device uint32_t*>(mkeys_w)
                    + write_slot * row_words;
                device uint32_t* mv_w = const_cast<device uint32_t*>(mvalues_w)
                    + write_slot * row_words;
                uint32_t kws[2] = {0u, 0u};
                uint32_t vws[2] = {0u, 0u};
                for (int element = 0; element < values_per_lane; ++element) {
                    const float kq = clamp(
                        rint((float(new_key[element]) - kqb) / kqs), 0.0f, 255.0f);
                    const float vq = clamp(
                        rint((float(new_value[element]) - vqb) / vqs), 0.0f, 255.0f);
                    kws[element / 4] |= uint32_t(kq) << (8 * (element % 4));
                    vws[element / 4] |= uint32_t(vq) << (8 * (element % 4));
                }
                mk_w[lane * 2] = kws[0];
                mk_w[lane * 2 + 1] = kws[1];
                mv_w[lane * 2] = vws[0];
                mv_w[lane * 2 + 1] = vws[1];
                if (lane == 0) {
                    mk_w[D / 4] = uint32_t(as_type<ushort>(khs))
                        | (uint32_t(as_type<ushort>(khb)) << 16);
                    mv_w[D / 4] = uint32_t(as_type<ushort>(vhs))
                        | (uint32_t(as_type<ushort>(vhb)) << 16);
                }
            }
            if (batch_index == 0 && kv_head == 0 && block == 0
                && query_head_in_group == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }

            device T* partial = partials
                + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
            device float* sum_out = sums + batch_head * BLOCKS + block;
            device float* max_out = maxs + batch_head * BLOCKS + block;

            thread float q[values_per_lane];
            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            uint slot = (ring_start + uint(block)) & ring_mask;
            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const bool current = token == N - 1;
                float score = 0.0f;
                if (current) {
                    for (int element = 0; element < values_per_lane; ++element) {
                        score += q[element] * float(new_key[element]);
                    }
                } else {
                    const device uint32_t* krow_w = mkeys_w + slot * row_words;
                    const uint32_t ktw = krow_w[D / 4];
                    const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                    const float kb = float(as_type<half>(ushort(ktw >> 16)));
                    const uint32_t kw0 = krow_w[lane * 2];
                    const uint32_t kw1 = krow_w[lane * 2 + 1];
                for (int element = 0; element < 4; ++element) {
                        score += q[element]
                            * fma(float((kw0 >> (8 * element)) & 0xffu), ks, kb);
                    }
                        for (int element = 0; element < 4; ++element) {
                        score += q[4 + element]
                            * fma(float((kw1 >> (8 * element)) & 0xffu), ks, kb);
                    }
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                if (current) {
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] = accumulator[element] * old_factor
                            + score_factor * float(new_value[element]);
                    }
                } else {
                    const device uint32_t* vrow_w = mvalues_w + slot * row_words;
                    const uint32_t vtw = vrow_w[D / 4];
                    const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                    const float vb = float(as_type<half>(ushort(vtw >> 16)));
                    const uint32_t vw0 = vrow_w[lane * 2];
                    const uint32_t vw1 = vrow_w[lane * 2 + 1];
                    for (int element = 0; element < 4; ++element) {
                        accumulator[element] = accumulator[element] * old_factor
                            + score_factor
                                * fma(float((vw0 >> (8 * element)) & 0xffu), vs, vb);
                        accumulator[4 + element] = accumulator[4 + element] * old_factor
                            + score_factor
                                * fma(float((vw1 >> (8 * element)) & 0xffu), vs, vb);
                    }
                }

                slot = (slot + uint(BLOCKS)) & ring_mask;
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

private static let diagBF16WriteKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ring_bf16_write_d256_diag_v1",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "starts", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["fence"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr uint ring_mask = uint(N - 1);

            const int group = int(threadgroup_position_in_grid.x);
            const int batch_index = group / KV_HEADS;
            const int kv_head = group % KV_HEADS;
            const int lane = int(thread_index_in_simdgroup);

            const device T* keys = k0;
            const device T* values = v0;
            switch (batch_index) {
                case 1: keys = k1; values = v1; break;
                case 2: keys = k2; values = v2; break;
                case 3: keys = k3; values = v3; break;
                case 4: keys = k4; values = v4; break;
                case 5: keys = k5; values = v5; break;
                case 6: keys = k6; values = v6; break;
                case 7: keys = k7; values = v7; break;
                default: break;
            }

            const uint ring_start = starts[batch_index];
            const uint write_slot = (ring_start + ring_mask) & ring_mask;
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            device T* write_key = const_cast<device T*>(keys)
                + kv_head * N * D + write_slot * D + lane * values_per_lane;
            device T* write_value = const_cast<device T*>(values)
                + kv_head * N * D + write_slot * D + lane * values_per_lane;
            for (int element = 0; element < values_per_lane; ++element) {
                write_key[element] = new_key[element];
                write_value[element] = new_value[element];
            }
            if (group == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """,
        ensureRowContiguous: true
    )

    nonisolated(unsafe) private static var diagScratch:
        (mirrors: [MLXArray], keys: [MLXArray], values: [MLXArray])?

    private static func diagScratchBuffers(
        kvHeads: Int, window: Int, headDim: Int
    ) -> (mirrors: [MLXArray], keys: [MLXArray], values: [MLXArray]) {
        if let diagScratch { return diagScratch }
        let made = (
            mirrors: (0 ..< 8).map { _ in
                MLXArray.zeros(
                    [2, kvHeads, window, headDim / 8 + headDim / 64], dtype: .uint32)
            },
            keys: (0 ..< 8).map { _ in
                MLXArray.zeros([1, kvHeads, window, headDim], dtype: .bfloat16)
            },
            values: (0 ..< 8).map { _ in
                MLXArray.zeros([1, kvHeads, window, headDim], dtype: .bfloat16)
            }
        )
        diagScratch = made
        return made
    }

    private func diagnosticFusedDispatch() {
        guard Self.diagFusedEnabled, let quantMirror, retainedCount == window,
            headDim == 256, kvHeads == 8, window == 1024
        else { return }
        Self.diagLock.lock()
        Self.diagCounter += 1
        let fire = Self.diagCounter % Self.diagStride == 0
        Self.diagLock.unlock()
        guard fire else { return }
        let batch = 8
        let queryHeads = 16
        let gqa = queryHeads / kvHeads
        let scratch = Self.diagScratchBuffers(
            kvHeads: kvHeads, window: window, headDim: headDim)
        let queries = MLXArray.zeros(
            [batch, queryHeads, 1, headDim], dtype: .bfloat16)
        let starts = MLXArray(
            Array(repeating: UInt32(oldestValidPosition % window), count: batch),
            [batch])
        let newKV = MLXArray.zeros([batch, kvHeads, 1, headDim], dtype: .bfloat16)
        let fenceIn = MLXArray.zeros([1], dtype: .int32)
        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype),
            ("D", headDim),
            ("N", window),
            ("GQA", gqa),
            ("KV_HEADS", kvHeads),
            ("BLOCKS", Self.diagBlocks),
        ]
        let liveMirrors = Self.diagLiveWrite
            ? Array(repeating: quantMirror, count: batch) : scratch.mirrors
        let fused = Self.diagFusedKernel(
            [queries] + liveMirrors + [starts, newKV, newKV, fenceIn],
            template: template,
            grid: (kvHeads * 32, batch * gqa, Self.diagBlocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [
                [batch, queryHeads, 1, Self.diagBlocks, headDim],
                [batch, queryHeads, 1, Self.diagBlocks],
                [batch, queryHeads, 1, Self.diagBlocks],
                [1],
            ],
            outputDTypes: [.bfloat16, .float32, .float32, .int32]
        )
        let companion = Self.diagBF16WriteKernel(
            scratch.keys + scratch.values + [starts, newKV, newKV, fused[3]],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", window),
                ("KV_HEADS", kvHeads),
            ],
            grid: (batch * kvHeads * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32]
        )
        asyncEval(fused[1], companion[0])
        Self.diagFusedDispatchOnce()
    }

    static let diagFusedEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG_FUSED"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    static let diagLiveWrite: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG_LIVE"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    nonisolated(unsafe) private static var diagFusedDispatched = false

    private static func diagFusedDispatchOnce() {
        diagLock.lock()
        let fresh = !diagFusedDispatched
        diagFusedDispatched = true
        diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] fused dispatched\n".utf8))
        }
    }

    public static func selfTestKVQ4() {
        guard ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KVQ4_SELFTEST"] ?? "").lowercased())
        else { return }
        let d = 256
        let ramp = ((MLXArray(0 ..< Int32(d)).asType(.float32) - 128.0) / 37.0)
            .reshaped([1, 1, 1, d]).asType(.bfloat16)
        let packed = quantPackGPU(ramp)                  // [1, 1, 36] uint32
        let words = packed.reshaped([36])
        var recon = [Float](repeating: 0, count: d)
        let host = words.asArray(UInt32.self)
        for e in 0 ..< d {
            let word = host[e / 8]
            let nib = Float((word >> UInt32(4 * (e % 8))) & 0xF)
            let tail = host[32 + e / 64]
            let s = Float(Float16(bitPattern: UInt16(tail & 0xffff)))
            let b = Float(Float16(bitPattern: UInt16(tail >> 16)))
            recon[e] = nib * s + b
        }
        let orig = ramp.reshaped([d]).asType(.float32).asArray(Float.self)
        var maxErr: Float = 0
        for e in 0 ..< d { maxErr = Swift.max(maxErr, abs(recon[e] - orig[e])) }
        FileHandle.standardError.write(Data(
            "[kvq4-selftest] max abs err \(maxErr) orig[0..3]=\(orig[0..<4]) recon[0..3]=\(recon[0..<4])\n".utf8))

        let rows = 8 * 1024
        let tiled = broadcast(packed.reshaped([1, 1, 36]), to: [8, 1024, 36])
        let mirror = concatenated([tiled, tiled], axis: 0)
            .reshaped([2, 8, 1024, 36])
        var oneHot = [Float](repeating: 0, count: 8 * 16 * 256)
        for h in 0 ..< (8 * 16) { oneHot[h * 256] = 1.0 }
        let q = MLXArray(oneHot, [8, 16, 1, 256]).asType(.bfloat16)
        _ = rows
        CBv2RaggedTwoPassDecodeAttentionV1.selfTestReadKernel(mirror: mirror, queries: q)
        FileHandle.standardError.write(Data(
            "[kvq4-kernel] expected score per token = recon[0] = \(recon[0])\n".utf8))
    }

    public static func warmPackPipeline() {
        selfTestKVQ4()
        guard quantEnabled, gpuPackEnabled else { return }
        let dummy = MLXArray.zeros([1, 8, 1, 256], dtype: .float16)
        eval(quantPackGPU(dummy))
        guard pairedMirrorWriteEnabled else { return }
        let pair = MLXArray.zeros([1, 8, 1, 256], dtype: .bfloat16)
        eval(quantPackPairGPU(keys: pair, values: pair))
    }

    private static func quantPackGPU(_ x: MLXArray) -> MLXArray {
        let kvHeads = x.dim(1)
        let n = x.dim(2)
        let rows = kvHeads * n
        return quantPackKernel(
            [x],
            template: [("T", x.dtype)],
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[kvHeads, n, x.dim(3) / 8 + x.dim(3) / 64]],
            outputDTypes: [.uint32]
        )[0]
    }

    static func quantRoundTrip(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let d = f.dim(-1)
        let lead = Array(f.shape.dropLast())
        let grouped = f.reshaped(lead + [d / 64, 64])
        let mn = grouped.min(axis: -1, keepDims: true)
        let mx = grouped.max(axis: -1, keepDims: true)
        let scale = maximum((mx - mn) / 15, MLXArray(Float(1e-6)))
            .asType(.float16).asType(.float32)
        let bias = mn.asType(.float16).asType(.float32)
        let q = clip(round((grouped - bias) / scale), min: 0, max: 15)
        return (q * scale + bias).reshaped(f.shape).asType(x.dtype)
    }

    static func quantPackPairGPU(keys: MLXArray, values: MLXArray) -> MLXArray {
        let heads = keys.dim(1)
        let headDim = keys.dim(3)
        return quantPackPairKernel(
            [keys, values],
            template: [("T", keys.dtype), ("HEADS", heads)],
            grid: (2 * heads * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[2, heads, 1, headDim / 8 + headDim / 64]],
            outputDTypes: [.uint32]
        )[0]
    }

    private func writePairedMirror(
        keys newKeys: MLXArray, values newValues: MLXArray, firstPosition: Int
    ) -> Bool {
        guard Self.pairedMirrorWriteEnabled,
            let quantMirror,
            Self.gpuPackEnabled, !Self.gpuPackCheck, !Self.quantSimulate,
            headDim == 256,
            newKeys.dtype == newValues.dtype,
            newKeys.shape == [1, kvHeads, 1, headDim],
            newValues.shape == [1, kvHeads, 1, headDim]
        else { return false }
        let packed = Self.quantPackPairGPU(keys: newKeys, values: newValues)
        let slot = firstPosition % window
        quantMirror[0..., 0..., slot ..< (slot + 1), 0...] = packed
        return true
    }

    private func writeMirror(plane: Int, tokens: MLXArray, firstPosition: Int) {
        guard let quantMirror else { return }
        let packedFlat: MLXArray
        if Self.gpuPackCheck {
            let gpu = Self.quantPackGPU(tokens)
            let host = Self.quantPack(tokens)
            let mismatches = (gpu .!= host).sum().item(Int.self)
            if mismatches != 0 {
                FileHandle.standardError.write(
                    "[kvq-gpupack] BYTE MISMATCH: \(mismatches) bytes differ\n"
                        .data(using: .utf8)!)
            }
            packedFlat = host.view(dtype: .uint8)
        } else if Self.gpuPackEnabled {
            packedFlat = Self.quantPackGPU(tokens).view(dtype: .uint8)
        } else {
            packedFlat = Self.quantPack(tokens).view(dtype: .uint8)
        }
        let packed = packedFlat.view(dtype: .uint32).expandedDimensions(axis: 0)
        let n = tokens.dim(2)
        let start = firstPosition % window
        if Self.selfTestArmed {
            Self.diagLock.lock()
            let fresh = !Self.mirrorChecked
            Self.mirrorChecked = true
            Self.diagLock.unlock()
            if fresh {
                let row = packed[0, 0, 0].asArray(UInt32.self)
                let src = tokens[0, 0, 0].asType(.float32).asArray(Float.self)
                var maxErr: Float = 0
                for e in 0 ..< min(256, src.count) {
                    let nib = Float((row[e / 8] >> UInt32(4 * (e % 8))) & 0xF)
                    let tail = row[32 + e / 64]
                    let s = Float(Float16(bitPattern: UInt16(tail & 0xffff)))
                    let b = Float(Float16(bitPattern: UInt16(tail >> 16)))
                    maxErr = Swift.max(maxErr, abs(nib * s + b - src[e]))
                }
                FileHandle.standardError.write(Data(
                    "[kvq4-insitu] plane=\(plane) n=\(n) rowWords=\(row.count) maxErr=\(maxErr) src[0..3]=\(src[0..<4])\n".utf8))
            }
        }
        if start + n <= window {
            quantMirror[plane ..< (plane + 1), 0..., start ..< (start + n), 0...] = packed
        } else {
            let first = window - start
            quantMirror[plane ..< (plane + 1), 0..., start ..< window, 0...] =
                packed[.ellipsis, ..<first, 0...]
            quantMirror[plane ..< (plane + 1), 0..., 0 ..< (n - first), 0...] =
                packed[.ellipsis, first..., 0...]
        }
    }

    private func allocateIfNeeded(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        guard keys == nil else { return }
        keys = MLXArray.zeros(
            [1, kvHeads, window, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
        values = MLXArray.zeros(
            [1, kvHeads, window, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
        if quantEligible, keyTemplate.dtype == .bfloat16,
            keyTemplate.dim(3) == headDim, valueTemplate.dim(3) == headDim
        {
            Self.selfTestKVQ4()
            quantMirror = MLXArray.zeros(
                [2, kvHeads, window, headDim / 8 + headDim / 64], dtype: .uint32)
        }
    }
}
