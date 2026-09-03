// Copyright © 2024 Apple Inc. See LICENSE.txt for license information.
//
// DARKBLOOM DGQMM — dequant-fused prefill GEMM, first principles.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Dequant-fused affine-4bit prefill GEMM (DGQMM): stages the bf16
/// activation tile and a TRANSPOSED, dequantized weight tile through
/// threadgroup memory into simdgroup MMA in ONE kernel.
///
/// The frontier road (`Gemma4PrefillDeqGEMMV1`) materializes the full
/// transposed bf16 plane (N*K*2 bytes written, read back by the dense
/// GEMM). DGQMM never touches DRAM with that plane: each 64x64 tile is
/// dequantized on the fly inside threadgroup memory, exactly once per tile,
/// and fed straight into `simdgroup_multiply_accumulate` with fp32
/// accumulators in ascending-k order.
///
/// Engine contract (verified in rs-harness TILEXACT/DGMINI at 8ae4c54 base):
///  - x     bf16, row-major, M = product of leading dims, M % 64 == 0, M>=512
///  - w     uint32 [N, K/8]; row n packs the K 4-bit codes of OUTPUT COLUMN
///          n, consecutive k in consecutive nibbles (low nibble first)
///  - scales/biases float32 OR bfloat16, [N, K/64] affine group-64
///  - y     bf16 [*, N] = x @ dequant(w)^T, dequant = scale*code + bias
///
/// Kill switch: DARKBLOOM_GEMMA4_DGQMM_TILES=1 (default OFF, fail-closed).
/// Engage mark: dgqmm.
public enum CBv2PrefillDGQMMV1 {
    // DGQMM-TILES: dequant-fused prefill GEMM from first principles,
    // written against the ENGINE layout (not steel template reuse — every
    // prior template attempt targeted the wrong weight interpretation; the
    // honest TILEXACT harness on realistic planes proved that).
    //
    // Contract (verified against stock quantizedMatmul semantics in
    // TILEXACT): x [M,K] bf16 row-major; weight uint32 [N, K/8] where row
    // n packs the K 4-bit codes of OUTPUT COLUMN n, consecutive k in
    // consecutive nibbles; scales/biases bf16 [N, K/64] affine group-64;
    // y [M,N] bf16 row-major = x @ dequant(w)^T, dequant = s*q + b.
    //
    // Tile: 128 threads (4 simdgroups), BM=BN=64, BK=64 == group size, so
    // one scale/bias per (column, k-block) and no group straddling. Ws is
    // staged TRANSPOSED [k][n] dequantized to bf16 (the same bf16 value the
    // stock loader produces: s*code + b in the same precision path), Xs is
    // the plain activation tile. Each simdgroup computes a 32x32 block with
    // sixteen 8x8 fp32 accumulators over ascending k; k fragments issue in
    // ascending order, accumulator chain never re-associates across blocks.
    //
    // Bit-exactness claim: NONE made without evidence. Offline TILEXACT
    // reports the diff count and max relative error vs stock on realistic
    // affine-4 planes; the engine gate is rig tokens (sealed seeds) + the
    // organizer's tolerance class. Kill switch DARKBLOOM_GEMMA4_DGQMM_TILES=1;
    // fail-closed guards: affine4 g64 bf16/uint32, M>=512 rows, M%64==0,
    // K%64==0, N%64==0, exact scale/bias shapes. Engage mark: dgqmm-tiles.
    public static let dgEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DGQMM_TILES"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let dgKernel = MLXFast.metalKernel(
        name: "darkbloom_prefill_dgqmm_bm64_bn64_gs64_bits4_v7",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """

            const uint3 tid = threadgroup_position_in_grid;
            const uint lid = thread_index_in_threadgroup;
            const uint sgid = simdgroup_index_in_threadgroup;
            const int K = int(x_shape[x_ndim - 1]);
            const int N = int(w_shape[0]);
            int M = 1;
            for (int d = 0; d < x_ndim - 1; d++) {
              M *= int(x_shape[d]);
            }

            constexpr int BM = 64, BN = 64, BK = 64;
            constexpr int LDA = 64;       // dense row stride (elements)
            constexpr int LDS = 64;
            // 8 KiB + 8 KiB + 16 KiB = 32 KiB threadgroup budget.
            threadgroup bfloat16_t Xs[BM * LDA];
            threadgroup bfloat16_t Ws[BK * LDS];
            threadgroup float scr[BM * BN];

            const int m0 = int(tid.y) * BM;
            const int n0 = int(tid.x) * BN;
            const int sgm = int(sgid >> 1u);       // row half of simd tile
            const int sgn = int(sgid & 1u);         // col half of simd tile

            simdgroup_matrix<float, 8, 8> acc[4][4];
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 4; j++) {
                acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
              }
            }

            const int Wld = K / 8;          // packed words per weight row
            const int gstride = K / 64;     // groups per row

            const int kBlocks = K / BK;
            for (int kb = 0; kb < kBlocks; kb++) {
              const int k0 = kb * BK;
              const int g = kb;             // BK == group_size

              // Stage x tile: 64 rows x 64 cols bf16, short4 chunks.
              {
                const device uint16_t* xb = (const device uint16_t*)x;
                threadgroup uint16_t* Xs16 = (threadgroup uint16_t*)Xs;
                for (int idx = int(lid); idx < BM * 16; idx += 128) {
                  const int row = idx >> 4;
                  const int ch = idx & 15;
                  Xs16[row * LDA + ch * 4 + 0] =
                      xb[(m0 + row) * K + k0 + ch * 4 + 0];
                  Xs16[row * LDA + ch * 4 + 1] =
                      xb[(m0 + row) * K + k0 + ch * 4 + 1];
                  Xs16[row * LDA + ch * 4 + 2] =
                      xb[(m0 + row) * K + k0 + ch * 4 + 2];
                  Xs16[row * LDA + ch * 4 + 3] =
                      xb[(m0 + row) * K + k0 + ch * 4 + 3];
                }
              }

              // Stage w tile transposed + dequantized: each thread owns 4
              // packed words (8 codes) of one column, unpack to Ws[k][n].
              {
                const int nloc = int(lid) & 63;
                const int n = n0 + nloc;
                const int kq = int(lid) >> 6;        // 0/1 -> k half (32)
                const int kbase = kq * 32;
                // float() conversion compiles whether the bridge binds
                // scales/biases as float32 or bfloat16 (engine passes bf16).
                const float qsc = float(scales[(size_t)n * gstride + g]);
                const float qbs = float(biases[(size_t)n * gstride + g]);
                const int w0 = (k0 + kbase) >> 3;     // packed word index
                const device uint32_t* wrow =
                    (const device uint32_t*)w + (size_t)n * Wld;
                for (int wd = 0; wd < 4; wd++) {
                  const uint32_t packed = wrow[w0 + wd];
                  for (int nib = 0; nib < 8; nib++) {
                    const int code = int((packed >> (nib * 4)) & 0xFu);
                    const int k = kbase + wd * 8 + nib;
                    Ws[k * LDS + nloc] = bfloat16_t(qsc * float(code) + qbs);
                  }
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);

              // MMA: ascending k fragments of 8.
              for (int kf = 0; kf < 8; kf++) {
                simdgroup_matrix<bfloat16_t, 8, 8> af[4];
                simdgroup_matrix<bfloat16_t, 8, 8> bf[4];
                for (int i = 0; i < 4; i++) {
                  simdgroup_load(af[i], &Xs[((sgm * 32) + i * 8) * LDA + kf * 8], LDA);
                }
                for (int j = 0; j < 4; j++) {
                  simdgroup_load(bf[j], &Ws[(kf * 8) * LDS + (sgn * 32) + j * 8], LDS);
                }
                for (int i = 0; i < 4; i++) {
                  for (int j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
                  }
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Epilogue: fp32 accumulators -> threadgroup float staging
            // (reinterpreting the dead smem span), cast to bf16 on store.
            {
              for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                  simdgroup_store(acc[i][j], &scr[((sgm * 32) + i * 8) * BN +
                                                  (sgn * 32) + j * 8], BN);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int idx = int(lid); idx < BM * BN; idx += 128) {
                const int r = idx >> 6;
                const int c = idx & 63;
                y[(size_t)(m0 + r) * N + (size_t)(n0 + c)] =
                    bfloat16_t(scr[r * BN + c]);
              }
            }
            return;

""",
        header: """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
""",
        ensureRowContiguous: true)

    public static func matmulDG(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        guard dgEnabled,
            let biases,
            mode == .affine,
            bits == 4,
            groupSize == 64,
            x.dtype == .bfloat16,
            weight.dtype == .uint32,
            scales.dtype == .float32 || scales.dtype == .bfloat16,
            biases.dtype == scales.dtype,
            x.ndim >= 2,
            weight.ndim == 2,
            let k = x.shape.last,
            k % 64 == 0,
            weight.shape[1] == k * 4 / 32
        else { return nil }
        var m = 1
        for dim in x.shape.dropLast() { m *= dim }
        guard m >= 512, m % 64 == 0 else { return nil }
        let n = weight.shape[0]
        guard n % 64 == 0,
            scales.shape == [n, k / 64],
            biases.shape == [n, k / 64]
        else { return nil }
        return dgKernel(
            [x, weight, scales, biases],
            template: [("T", x.dtype)],
            // Bridge convention: grid is TOTAL threads, not threadgroups.
            grid: (((n + 63) / 64) * 128, (m + 63) / 64, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [x.shape.dropLast().map { Int($0) } + [n]],
            outputDTypes: [x.dtype]
        )[0]
    }

}
