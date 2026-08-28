// Gemma4MMAQuantizedGEMV.swift
//
// M=8 dense quantized GEMV on the simdgroup matrix units.
// Shipped for the tied LM head only (N >= 8192): at that width the
// matrix units beat stock ordinary/quad QMV (local 1.75x isolated, 2.32 ->
// 1.32 ms). Small-N (q/k/v/o, dense MLP) measured 0.7-0.9x stock -- those
// stay on the ordinary affine_qmv pair/triple/quad path.
//
// WHY. The stock `affine_qmv` family computes `x * (packed & mask)` per
// (row, weight) in scalar float32; at M=8 that is ~2 ALU ops per MAC and the
// projections become ALU-bound (~90M MACs each), far under memory bandwidth.
// The matrix units do the MACs at a fraction of the cost.
//
// EXACT WEIGHTS. The 4-bit / 8-bit codes are fed to the MMA as bfloat
// INTEGERS (0...15 / 0...255 are exact in bfloat16); the affine transform is
// applied per 64-wide group in float32 afterwards:
//   sum_k x_k * (s_g * q_k + b_g)  ==  s_g * (sum_k x_k * q_k) + b_g * (sum_k x_k)
// so no target weight is re-represented or rounded. The only difference from
// the stock kernel is the float32 summation order inside a group (tree vs
// sequential) and across groups. The bf16 output rounding is the same
// `static_cast<T>(float)` as stock.
//
// Gated fail-closed to M == 8, bf16, K % 256 == 0, N >= 8192, N % 16 == 0,
// group 64, bits 4 or 8; anything else returns nil and the caller uses
// the stock op. `DARKBLOOM_GEMMA4_MMA_QMV=0` disables it.

import Foundation
import MLX
import MLXFast

public enum Gemma4MMAQuantizedGEMV {
    public static let rows = 8
    static let columnsPerSimdgroup = 16
    static let groupSize = 64

    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_MMA_QMV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    // One simdgroup per 16 output columns. Lane l dequantizes columns
    // (l / 4) + {0, 8} for k-range [16 * (l % 4), +16) of the current 64-group
    // into a threadgroup tile laid out [column][k], then eight 8x8 MMAs per
    // column block accumulate x(8x64) . q(64x8) in float32. Per group the
    // affine epilogue folds scale/bias with the group's x-sums, which come
    // from the same A tiles against a ones tile.
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine_qmv_m8_v3",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int NPS = 16;            // columns per threadgroup
            constexpr int GS = 64;             // affine group size
            constexpr int KG = K / GS;         // groups along K
            constexpr int KSPLIT = 4;          // simdgroups per threadgroup, each a K range
            constexpr int GPS = KG / KSPLIT;   // groups per simdgroup (K % 256 == 0 guaranteed by the host)
            threadgroup bfloat16_t qtiles[KSPLIT * NPS * GS];
            threadgroup float partials[KSPLIT * 32 * 4];
            threadgroup float sb_tiles[KSPLIT * NPS * GPS * 2];   // [sg][col][group][scale, bias]

            const int tg = int(threadgroup_position_in_grid.x);
            const int sgi = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);
            const int n0 = tg * NPS;
            const int lane_col = lane / 4;          // 0..7
            const int lane_kc = (lane % 4) * 16;    // 0,16,32,48
            threadgroup bfloat16_t* qtile = qtiles + sgi * NPS * GS;

            // lane -> (fm, fn): element coordinates this lane holds in an 8x8 fragment
            const int qid = lane / 4;
            const int fm = (qid & 4) + ((lane / 2) % 4);
            const int fn = (qid & 2) * 2 + (lane % 2) * 2;
            // the four lanes sharing row fm are {lane, lane^1, lane^8, lane^9};
            // this lane owns quads [4*qq, 4*qq+4) of the row's 16 quads per group
            const int qq = (lane & 1) + 2 * ((lane >> 3) & 1);

            float acc0 = 0.0f, acc1 = 0.0f;   // columns n0 + fn, +1
            float acc2 = 0.0f, acc3 = 0.0f;   // columns n0 + 8 + fn, +1

            const int wrow_u32 = (BITS == 4) ? (K / 8) : (K / 4);
            const int srow = KG;
            const int g_begin = sgi * GPS;
            const int g_end = g_begin + GPS;
            threadgroup float* sb = sb_tiles + sgi * NPS * GPS * 2;
            for (int i = lane; i < NPS * GPS; i += 32) {
                const int col = i / GPS;
                const int gg = i % GPS;
                const int n = n0 + col;
                sb[i * 2] = float(scales[n * srow + g_begin + gg]);
                sb[i * 2 + 1] = float(biases[n * srow + g_begin + gg]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int g = g_begin; g < g_end; ++g) {
                // ---- dequantize this group's 16 x 64 codes into qtile as exact integers
                for (int cb = 0; cb < 2; ++cb) {
                    const int n = n0 + cb * 8 + lane_col;
                    threadgroup bfloat16_t* dst = qtile + (cb * 8 + lane_col) * GS + lane_kc;
                    if (BITS == 4) {
                        const device uint32_t* wp = w + n * wrow_u32 + g * 8 + (lane_kc / 8);
                        const uint32_t p0 = wp[0];
                        const uint32_t p1 = wp[1];
                        for (int i = 0; i < 8; ++i) {
                            dst[i] = bfloat16_t(float((p0 >> (4 * i)) & 0xfu));
                            dst[8 + i] = bfloat16_t(float((p1 >> (4 * i)) & 0xfu));
                        }
                    } else {
                        const device uint32_t* wp = w + n * wrow_u32 + g * 16 + (lane_kc / 4);
                        for (int j = 0; j < 4; ++j) {
                            const uint32_t p = wp[j];
                            for (int i = 0; i < 4; ++i) {
                                dst[4 * j + i] = bfloat16_t(float((p >> (8 * i)) & 0xffu));
                            }
                        }
                    }
                }

                // ---- the group's activation sum for row fm, in the stock kernel's
                // arithmetic: 4-bit sums each quad in bf16 (T + T) before widening,
                // 8-bit widens every value first.
                float xs = 0.0f;
                {
                    const device T* xq = x + fm * K + g * GS + qq * 16;
                    if (BITS == 4) {
                        for (int i = 0; i < 16; i += 4) {
                            xs += float(xq[i] + xq[i + 1] + xq[i + 2] + xq[i + 3]);
                        }
                    } else {
                        for (int i = 0; i < 16; ++i) {
                            xs += float(xq[i]);
                        }
                    }
                }
                xs += simd_shuffle_xor(xs, 1);
                xs += simd_shuffle_xor(xs, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // ---- x(8 x 64) . q(64 x 8) for both column blocks
                simdgroup_matrix<float, 8, 8> c0 = simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> c1 = simdgroup_matrix<float, 8, 8>(0.0f);
                for (int kk = 0; kk < GS; kk += 8) {
                    simdgroup_matrix<bfloat16_t, 8, 8> a;
                    simdgroup_load(a, x + g * GS + kk, K);
                    simdgroup_matrix<bfloat16_t, 8, 8> b0;
                    simdgroup_matrix<bfloat16_t, 8, 8> b1;
                    simdgroup_load(b0, qtile + kk, GS, ulong2(0, 0), true);
                    simdgroup_load(b1, qtile + 8 * GS + kk, GS, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(c0, a, b0, c0);
                    simdgroup_multiply_accumulate(c1, a, b1, c1);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // ---- affine epilogue: acc += s * (x.q) + b * (sum x)
                {
                    const int gg = g - g_begin;
                    const threadgroup float* e0 = sb + ((fn) * GPS + gg) * 2;
                    const threadgroup float* e1 = sb + ((fn + 1) * GPS + gg) * 2;
                    const threadgroup float* e2 = sb + ((8 + fn) * GPS + gg) * 2;
                    const threadgroup float* e3 = sb + ((8 + fn + 1) * GPS + gg) * 2;
                    acc0 += e0[0] * c0.thread_elements()[0] + e0[1] * xs;
                    acc1 += e1[0] * c0.thread_elements()[1] + e1[1] * xs;
                    acc2 += e2[0] * c1.thread_elements()[0] + e2[1] * xs;
                    acc3 += e3[0] * c1.thread_elements()[1] + e3[1] * xs;
                }
            }

            // ---- combine the K-split partials in a fixed order and store
            threadgroup float* mine = partials + (sgi * 32 + lane) * 4;
            mine[0] = acc0; mine[1] = acc1; mine[2] = acc2; mine[3] = acc3;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgi == 0) {
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                for (int s = 0; s < KSPLIT; ++s) {
                    const threadgroup float* part = partials + (s * 32 + lane) * 4;
                    o0 += part[0]; o1 += part[1]; o2 += part[2]; o3 += part[3];
                }
                device T* yr = y + fm * N;
                yr[n0 + fn] = T(o0);
                yr[n0 + fn + 1] = T(o1);
                yr[n0 + 8 + fn] = T(o2);
                yr[n0 + 8 + fn + 1] = T(o3);
            }
        """,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    /// x: [8, K] (or any shape with 8 rows total) bf16; w: [N, K*bits/32] uint32;
    /// scales/biases: [N, K/64] bf16. Returns [8, N] bf16 or nil when ineligible.
    public static func apply(
        x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?, bits: Int, groupSize: Int
    ) -> MLXArray? {
        guard enabled, let biases,
            groupSize == 64, bits == 4 || bits == 8,
            x.dtype == .bfloat16, scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            weight.dtype == .uint32, weight.ndim == 2, scales.ndim == 2, biases.ndim == 2
        else { return nil }
        let K = x.dim(-1)
        guard x.size == rows * K, K % (groupSize * 4) == 0 else { return nil }
        let N = weight.dim(0)
        guard N >= 8192, N % columnsPerSimdgroup == 0,
            weight.dim(1) == K * bits / 32,
            scales.shape == [N, K / groupSize], biases.shape == [N, K / groupSize]
        else { return nil }
        let x2 = x.ndim == 2 ? x : x.reshaped([rows, K])
        let out = kernel(
            [x2, weight, scales, biases],
            template: [("T", x.dtype), ("BITS", bits), ("K", K), ("N", N)],
            grid: (N / columnsPerSimdgroup * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[rows, N]],
            outputDTypes: [x.dtype]
        )[0]
        return out
    }
}
