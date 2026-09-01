// CBv2MTPMirrorOps.swift
//
// MTP verify rounds on the production q4 MIRROR road.
//
// The promoted B=8 decode path is quant-authoritative on the 25 sliding
// layers: the fused pass-A kernel packs each step's K/V straight into the
// q4g64 mirror and the BF16 ring is left stale (`bf16RingStale`). The
// vendored MTP verify path predates that road: it stages speculative K/V
// through `snapshot()` (BF16 views), so on the promoted tree an MTP round
// either traps on the stale ring or falls onto the generic BF16 attention
// road whose bytes differ from what serial decode reads.
//
// This file gives the round the SAME road serial decode runs. Every verify
// column is an ordinary `[B, 1]` decode forward: the fused pass-A writes the
// column's K/V into the live mirror slot in place, exactly like a committed
// token. What makes that safe to roll back is an UNDO LOG:
//
//   * Before the round, the `k` mirror slots that columns 1...k will
//     overwrite (positions p+1...p+k, aliasing positions p+1-W...p+k-W) are
//     copied out (`undoCapture`). The copy is threaded into the layer's
//     write fence so it is ordered ahead of the first in-place store.
//   * At finalize, the rejected suffix's slots are restored from the log by
//     one fenced dispatch per layer (`restore`), and the row's counters
//     rewind. Column 0 (the seed token) is always committed and never
//     restored; the first rejected column's slot is the one the next round's
//     seed overwrites before anything reads it, so only the columns AFTER it
//     need restoring.
//
// The drafter's frozen sliding capture is served by dequantizing the mirror
// (`dequantize`) — the same bytes the target attends — instead of the stale
// BF16 ring.
//
// Kill switch: `DARKBLOOM_CBV2_MTP_MIRROR_ROAD=0` keeps the vendored staging
// path (which on this tree requires `MLX_KV_Q4_BF16_ELIDE=0`).

import Foundation
import MLX
import MLXFast

/// WIDE VERIFY (opt-in, `DARKBLOOM_CBV2_MTP_WIDE_VERIFY=1`): the mirror-road
/// round scores all 1+k columns in ONE `[B, 1+k]` target forward instead of
/// 1+k `[B, 1]` forwards. Attention is still serialized per column through
/// the exact decode road (`CBv2AttentionV1.updateAndAttend`); every
/// token-local plane sees `B * (1+k)` rows at once, so a weight plane is read
/// once per round instead of once per column. Kernel hosts consult
/// `CBv2MTPWideVerifyContext.active` to admit the widened rectangle; a host
/// that has no widened body slices the rectangle into `[B, 1]` tiles through
/// its exact narrow kernel, so every row's arithmetic is the narrow kernel's.
public enum CBv2MTPWideVerifyContext {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_WIDE_VERIFY"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Wide verify expert plane: `DARKBLOOM_CBV2_MTP_EXPERT_MERGE=1` runs ONE
    /// sorted gather over every rectangle row (a distinct expert's weights
    /// read once per round); unset keeps the eight-row tiles that replicate
    /// the promoted 64-assignment gather exactly.
    public static let mergedExpertGather: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_EXPERT_MERGE"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Set on the engine thread for the duration of a wide verify forward's
    /// graph construction. Kernel hosts read it to distinguish a verify
    /// rectangle from a prompt chunk of the same shape class.
    nonisolated(unsafe) public private(set) static var active = false
    /// The rectangle's column count while active (1 + k).
    nonisolated(unsafe) public private(set) static var columns = 1

    /// Fallback widening for a host whose kernel body is pinned to eight
    /// rows: run `body` over each eight-row tile of `x` (axis 0) and
    /// concatenate. Every row's arithmetic is the narrow kernel's; only the
    /// weight plane is re-read per tile. nil unless a wide verify is active
    /// and `x` is a strict multiple of the tile, or when any tile refuses.
    public static func rowTiles(
        _ x: MLXArray, tile: Int = 8, _ body: (MLXArray) -> MLXArray?
    ) -> MLXArray? {
        guard active, x.ndim >= 1, x.dim(0) > tile, x.dim(0) % tile == 0 else { return nil }
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(x.dim(0) / tile)
        for start in stride(from: 0, to: x.dim(0), by: tile) {
            guard let output = body(x[start ..< (start + tile)]) else { return nil }
            outputs.append(output)
        }
        return concatenated(outputs, axis: 0)
    }

    public static func with<T>(columns count: Int, _ body: () -> T) -> T {
        precondition(!active, "CBv2MTPWideVerifyContext: nested wide verify")
        active = true
        columns = count
        defer {
            active = false
            columns = 1
        }
        return body()
    }
}

/// One layer's mirror-road bookkeeping for an in-flight round: the eight
/// windowed rows in batch order, their pre-round slot bases, and the undo
/// log the finalize-time restore reads.
struct CBv2MTPMirrorRestoreLayer {
    let cache: CBv2LayerCache
    let rows: [CBv2WindowedSequenceKV]
    let mirrors: [MLXArray]
    let slotBases: [Int]
    let undo: MLXArray
    let depth: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
}

enum CBv2MTPMirrorOps {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_MIRROR_ROAD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Words per (plane, head, slot) row of the mirror: 32 payload words of
    /// eight 4-bit codes plus one fp16 (scale, bias) word per 64-group.
    static func rowWords(headDim: Int) -> Int { headDim / 8 + headDim / 64 }

    // MARK: - Dequantized capture (drafter frozen KV)

    private static let dequantKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_mtp_kvq4_mirror_dequant_v1",
        inputNames: ["mirror", "params", "write_fence"],
        outputNames: ["keys", "values"],
        source: """
            constexpr int payload_words = D / 8;
            constexpr int row_words = payload_words + D / 64;
            const int lane = int(thread_position_in_grid.x);
            const int head = int(thread_position_in_grid.y);
            const int t = int(thread_position_in_grid.z);
            const uint start = params[0];
            const uint count = params[1];
            // `write_fence` is consumed for ordering only: the read must
            // follow the previous round's restore of this mirror.
            (void)write_fence[0];
            if (uint(t) >= count) return;
            const int slot = int((start + uint(t)) % uint(N));
            const device uint32_t* krow = mirror + (head * N + slot) * row_words;
            const device uint32_t* vrow = mirror + ((KV_HEADS + head) * N + slot) * row_words;
            const int group = lane / 8;
            const uint32_t ktw = krow[payload_words + group];
            const uint32_t vtw = vrow[payload_words + group];
            const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
            const float kb = float(as_type<half>(ushort(ktw >> 16)));
            const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
            const float vb = float(as_type<half>(ushort(vtw >> 16)));
            const uint32_t kw = krow[lane];
            const uint32_t vw = vrow[lane];
            device T* kout = keys + (head * int(count) + t) * D + lane * 8;
            device T* vout = values + (head * int(count) + t) * D + lane * 8;
            #pragma unroll
            for (int element = 0; element < 8; ++element) {
                kout[element] = T(fma(float((kw >> (4 * element)) & 0xfu), ks, kb));
                vout[element] = T(fma(float((vw >> (4 * element)) & 0xfu), vs, vb));
            }
        """,
        ensureRowContiguous: true
    )

    /// Temporal-order `[1, kvHeads, count, headDim]` bf16 K and V of the
    /// retained ring `[start, start + count)` (slot indices, modulo the
    /// window), dequantized from the q4 mirror.
    static func dequantize(
        mirror: MLXArray, start: Int, count: Int,
        kvHeads: Int, headDim: Int, window: Int, fence: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        precondition(count > 0 && count <= window)
        precondition(headDim == 256, "mirror dequant expects D=256 (eight codes per lane)")
        let params = MLXArray([UInt32(start), UInt32(count)])
        let outputs = dequantKernel(
            [mirror, params, fence],
            template: [
                ("T", DType.bfloat16),
                ("D", headDim),
                ("N", window),
                ("KV_HEADS", kvHeads),
            ],
            grid: (32, kvHeads, count),
            threadGroup: (32, 1, 1),
            outputShapes: [[1, kvHeads, count, headDim], [1, kvHeads, count, headDim]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }

    // MARK: - Undo log

    private static let captureKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_mtp_kvq4_mirror_capture_v1",
        inputNames: [
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "params", "write_fence",
        ],
        outputNames: ["undo", "fence"],
        source: """
            constexpr int row_words = D / 8 + D / 64;
            const int word = int(thread_position_in_grid.x);
            const int plane_head = int(thread_position_in_grid.y);
            const int rc = int(thread_position_in_grid.z);
            const int k = int(params[ROWS]);
            const int row = rc / k;
            const int c = rc % k + 1;
            if (word == 0 && plane_head == 0 && rc == 0) {
                fence[0] = write_fence[0] + 1;
            }
            if (word >= row_words) return;
            const uint base = params[row];
            const int slot = int((base + uint(c)) % uint(N));
            const device uint32_t* mirror_w = m0;
            switch (row) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }
            undo[((row * (2 * KV_HEADS) + plane_head) * k + (c - 1)) * row_words + word] =
                mirror_w[(plane_head * N + slot) * row_words + word];
        """,
        ensureRowContiguous: true
    )

    /// Copy out, for one layer's eight rows, the mirror rows columns `1...k`
    /// of the round will overwrite: `[8, 2 * kvHeads, k, rowWords]` uint32.
    /// Ordered after `fence` (the layer's write fence before the round); the
    /// returned fence is what column 0's pass-A must consume so no in-place
    /// store of the round can overtake the capture.
    static func capture(
        mirrors: [MLXArray], slotBases: [Int], depth k: Int,
        kvHeads: Int, headDim: Int, window: Int, fence: MLXArray
    ) -> (undo: MLXArray, fence: MLXArray) {
        precondition(mirrors.count == 8 && slotBases.count == 8 && k > 0)
        var params = slotBases.map { UInt32($0) }
        params.append(UInt32(k))
        let words = rowWords(headDim: headDim)
        let threads = ((words + 31) / 32) * 32
        let outputs = captureKernel(
            mirrors + [MLXArray(params), fence],
            template: [
                ("D", headDim),
                ("N", window),
                ("KV_HEADS", kvHeads),
                ("ROWS", 8),
            ],
            grid: (threads, 2 * kvHeads, 8 * k),
            threadGroup: (threads, 1, 1),
            outputShapes: [[8, 2 * kvHeads, k, words], [1]],
            outputDTypes: [.uint32, .int32]
        )
        return (outputs[0], outputs[1])
    }

    // MARK: - Restore

    private static let restoreKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_mtp_kvq4_mirror_restore_v1",
        inputNames: [
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "undo", "params", "write_fence",
        ],
        outputNames: ["fence"],
        source: """
            constexpr int row_words = D / 8 + D / 64;
            const int word = int(thread_position_in_grid.x);
            const int plane_head = int(thread_position_in_grid.y);
            const int rc = int(thread_position_in_grid.z);
            const int k = int(params[2 * ROWS]);
            const int row = rc / k;
            const int c = rc % k + 1;
            if (word == 0 && plane_head == 0 && rc == 0) {
                fence[0] = write_fence[0] + 1;
            }
            if (word >= row_words) return;
            const uint first_restored = params[ROWS + row];
            if (uint(c) < first_restored) return;
            const uint base = params[row];
            const int slot = int((base + uint(c)) % uint(N));
            const device uint32_t* mirror_w = m0;
            switch (row) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }
            device uint32_t* dst = const_cast<device uint32_t*>(mirror_w)
                + (plane_head * N + slot) * row_words + word;
            dst[0] = undo[((row * (2 * KV_HEADS) + plane_head) * k + (c - 1)) * row_words + word];
        """,
        ensureRowContiguous: true
    )

    /// Restore the rejected columns' mirror slots in place for one layer's
    /// eight rows, ordered after `fence` (the layer's write fence at the end
    /// of the round). `firstRestored[r]` is the first column index whose
    /// slot must be restored for row `r` (`k + 1` restores nothing).
    /// Returns the fence the layer's next write must consume.
    static func restore(
        mirrors: [MLXArray], undo: MLXArray, slotBases: [Int], firstRestored: [Int],
        depth k: Int, kvHeads: Int, headDim: Int, window: Int, fence: MLXArray
    ) -> MLXArray {
        precondition(mirrors.count == 8 && slotBases.count == 8 && firstRestored.count == 8)
        precondition(k > 0)
        var params = slotBases.map { UInt32($0) }
        params.append(contentsOf: firstRestored.map { UInt32($0) })
        params.append(UInt32(k))
        let words = rowWords(headDim: headDim)
        let threads = ((words + 31) / 32) * 32
        let outputs = restoreKernel(
            mirrors + [undo, MLXArray(params), fence],
            template: [
                ("D", headDim),
                ("N", window),
                ("KV_HEADS", kvHeads),
                ("ROWS", 8),
            ],
            grid: (threads, 2 * kvHeads, 8 * k),
            threadGroup: (threads, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32]
        )
        CBv2EngageMark.once("mtp-mirror-restore")
        return outputs[0]
    }
}
