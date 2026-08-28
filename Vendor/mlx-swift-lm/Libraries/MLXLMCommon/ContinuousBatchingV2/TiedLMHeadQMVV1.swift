// LMH-001: tight-grid dispatch for the tied lm_head ordinary QMV at batch eight.
//
// The vendored MLX host launches ordinary QMV as
//
//     MTL::Size group_dims(bk, 2, 1);                     // (32, 2, 1)
//     MTL::Size grid_dims(M, (N + bn - 1) / bn, B);       // x = M
//     compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
//
// (`backend/metal/quantized.cpp`), so the x extent of the grid is the cohort row
// count, eight. The promoted large-N tier in `quantized.h` claims four cohort
// rows per threadgroup and returns from the rest:
//
//     const int first_m = int(tid.x) * 4;
//     if (first_m >= 8) { return; }
//
// Two x-groups do the work; six are launched and retire immediately. On the tied
// lm_head that is the largest grid of the decode step -- N = 262144 gives
// N / 8 = 32768 y-groups, so 8 * 32768 = 262144 threadgroups are launched and
// 196608 of them exist only to hit that early return.
//
// The host grid is not editable. This file instead dispatches the same
// computation from a custom kernel whose own grid has x extent two, so only the
// groups that were already doing the work are launched. `first_m` is kept as
// `tid.x * 4`, so with tid.x in {0, 1} the two surviving groups claim rows 0-3
// and 4-7 exactly as before: the same threadgroup does the same rows with the
// same pointers.
//
// The body below mirrors the current `qmv_affine4_g64_quad_stream_impl`: one
// activation buffer walks four input rows while their packed weights and affine
// metadata stay resident. The load, nibble-dot, affine-close, reduction, and
// BF16-store expressions retain the production order. Preprocessor expansion
// fixes the four row applications at compile time, avoiding the register and
// control-flow cost measured for a dynamic input-row loop.
//
// `DARKBLOOM_CBV2_TIED_LMHEAD_QMV=0` restores the stock path inside the same
// executable.

import Foundation
import MLX
import MLXFast

public enum CBv2TiedLMHeadQMVV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_TIED_LMHEAD_QMV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Pinned to the ruled decode cohort and this checkpoint's tower.
    private static let batch = 8
    private static let groupSize = 64
    private static let bits = 4
    private static let rowsPerGroup = 4
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let outputsPerGroup = 8
    /// `values_per_thread * SIMD` in the kernel; the tail block needs one more.
    private static let values_per_thread_block = 256

    private static let unrolledStreamKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_unrolled_stream_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int SIMD = 32;
            constexpr int num_simdgroups = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int values_per_thread = 8;
            constexpr int block_size = values_per_thread * SIMD;
            constexpr int bytes_per_thread = 4;
            constexpr int uint16_per_thread = 2;
            constexpr int scale_step_per_thread = 8;

            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int first_m = int(tid.x) * 4;

            const device T* x0 = x + (first_m + 0) * K;
            const device T* x1 = x + (first_m + 1) * K;
            const device T* x2 = x + (first_m + 2) * K;
            const device T* x3 = x + (first_m + 3) * K;
            device T* y0 = y + (first_m + 0) * OUTN;
            device T* y1 = y + (first_m + 1) * OUTN;
            device T* y2 = y + (first_m + 2) * OUTN;
            device T* y3 = y + (first_m + 3) * OUTN;

            const int in_vec_size_w = K / 2;
            const int in_vec_size_g = K / 64;
            const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
            const device uint8_t* ws = (const device uint8_t*)w
                + out_row * in_vec_size_w + simd_lid * bytes_per_thread;
            scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            x0 += simd_lid * values_per_thread;
            x1 += simd_lid * values_per_thread;
            x2 += simd_lid * values_per_thread;
            x3 += simd_lid * values_per_thread;
            y0 += out_row;
            y1 += out_row;
            y2 += out_row;
            y3 += out_row;

            thread float x_thread[values_per_thread];
            thread uint16_t packed[results_per_simdgroup][uint16_per_thread];
            thread float scale_local[results_per_simdgroup];
            thread float bias_local[results_per_simdgroup];
            thread float result0[results_per_simdgroup] = {0};
            thread float result1[results_per_simdgroup] = {0};
            thread float result2[results_per_simdgroup] = {0};
            thread float result3[results_per_simdgroup] = {0};

            #define LOAD_AND_DOT(INPUT, RESULT) \
                do { \
                    float sum = 0; \
                    for (int vi = 0; vi < values_per_thread; vi += 4) { \
                        sum += INPUT[vi] + INPUT[vi + 1] \
                            + INPUT[vi + 2] + INPUT[vi + 3]; \
                        x_thread[vi] = INPUT[vi]; \
                        x_thread[vi + 1] = INPUT[vi + 1] / 16.0f; \
                        x_thread[vi + 2] = INPUT[vi + 2] / 256.0f; \
                        x_thread[vi + 3] = INPUT[vi + 3] / 4096.0f; \
                    } \
                    for (int row = 0; row < results_per_simdgroup; ++row) { \
                        float accum = 0; \
                        for (int wi = 0; wi < uint16_per_thread; ++wi) { \
                            const uint16_t code = packed[row][wi]; \
                            accum += \
                                (x_thread[4 * wi] * (code & 0x000f) + \
                                 x_thread[4 * wi + 1] * (code & 0x00f0) + \
                                 x_thread[4 * wi + 2] * (code & 0x0f00) + \
                                 x_thread[4 * wi + 3] * (code & 0xf000)); \
                        } \
                        RESULT[row] += scale_local[row] * accum \
                            + sum * bias_local[row]; \
                    } \
                } while (0)

            #define LOAD_SAFE_AND_DOT(INPUT, RESULT) \
                do { \
                    float sum = 0; \
                    int vi = 0; \
                    for (; vi < remaining; vi += 4) { \
                        sum += INPUT[vi] + INPUT[vi + 1] \
                            + INPUT[vi + 2] + INPUT[vi + 3]; \
                        x_thread[vi] = INPUT[vi]; \
                        x_thread[vi + 1] = INPUT[vi + 1] / 16.0f; \
                        x_thread[vi + 2] = INPUT[vi + 2] / 256.0f; \
                        x_thread[vi + 3] = INPUT[vi + 3] / 4096.0f; \
                    } \
                    for (; vi < values_per_thread; ++vi) x_thread[vi] = 0; \
                    for (int row = 0; row < results_per_simdgroup; ++row) { \
                        float accum = 0; \
                        for (int wi = 0; wi < uint16_per_thread; ++wi) { \
                            const uint16_t code = packed[row][wi]; \
                            accum += \
                                (x_thread[4 * wi] * (code & 0x000f) + \
                                 x_thread[4 * wi + 1] * (code & 0x00f0) + \
                                 x_thread[4 * wi + 2] * (code & 0x0f00) + \
                                 x_thread[4 * wi + 3] * (code & 0xf000)); \
                        } \
                        RESULT[row] += scale_local[row] * accum \
                            + sum * bias_local[row]; \
                    } \
                } while (0)

            int k = 0;
            for (; k < K - block_size; k += block_size) {
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint16_t* wl =
                        (const device uint16_t*)(ws + row * in_vec_size_w);
                    for (int i = 0; i < uint16_per_thread; ++i) packed[row][i] = wl[i];
                    scale_local[row] = scales[row * in_vec_size_g];
                    bias_local[row] = biases[row * in_vec_size_g];
                }
                LOAD_AND_DOT(x0, result0);
                LOAD_AND_DOT(x1, result1);
                LOAD_AND_DOT(x2, result2);
                LOAD_AND_DOT(x3, result3);
                ws += block_size / 2;
                scales += block_size / 64;
                biases += block_size / 64;
                x0 += block_size;
                x1 += block_size;
                x2 += block_size;
                x3 += block_size;
            }

            const int remaining = clamp(
                static_cast<int>(K - k - simd_lid * values_per_thread),
                0, values_per_thread);
            if (remaining > 0) {
                for (int row = 0; row < results_per_simdgroup; ++row) {
                    const device uint16_t* wl =
                        (const device uint16_t*)(ws + row * in_vec_size_w);
                    for (int i = 0; i < uint16_per_thread; ++i) packed[row][i] = wl[i];
                    scale_local[row] = scales[row * in_vec_size_g];
                    bias_local[row] = biases[row * in_vec_size_g];
                }
                LOAD_SAFE_AND_DOT(x0, result0);
                LOAD_SAFE_AND_DOT(x1, result1);
                LOAD_SAFE_AND_DOT(x2, result2);
                LOAD_SAFE_AND_DOT(x3, result3);
            }

            #undef LOAD_AND_DOT
            #undef LOAD_SAFE_AND_DOT

            for (int row = 0; row < results_per_simdgroup; ++row) {
                result0[row] = simd_sum(result0[row]);
                result1[row] = simd_sum(result1[row]);
                result2[row] = simd_sum(result2[row]);
                result3[row] = simd_sum(result3[row]);
                if (simd_lid == 0) {
                    y0[row] = static_cast<T>(result0[row]);
                    y1[row] = static_cast<T>(result1[row]);
                    y2[row] = static_cast<T>(result2[row]);
                    y3[row] = static_cast<T>(result3[row]);
                }
            }
            """,
        ensureRowContiguous: true
    )

    /// Returns `nil` unless every pin holds; the caller then keeps the stock path.
    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        inDim: Int,
        outDim: Int
    ) -> MLXArray? {
        // Every dimension is validated against every other, so the gate is a
        // full shape pin at runtime even though the tower's hidden size is not
        // written as a literal here.
        guard enabled,
            let biases,
            x.dtype == .bfloat16,
            scales.dtype == x.dtype,
            biases.dtype == x.dtype,
            weight.dtype == .uint32,
            x.ndim == 3,
            x.dim(0) == batch,
            x.dim(1) == 1,
            x.dim(2) == inDim,
            outDim >= 8,
            outDim % outputsPerGroup == 0,
            inDim % groupSize == 0,
            inDim >= 2 * values_per_thread_block,
            weight.ndim == 2,
            weight.dim(0) == outDim,
            weight.dim(1) == inDim * bits / 32,
            scales.ndim == 2,
            scales.dim(0) == outDim,
            scales.dim(1) == inDim / groupSize,
            biases.shape == scales.shape
        else { return nil }

        let xGroups = batch / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        return unrolledStreamKernel(
            [x, weight, scales, biases],
            template: [
                ("T", x.dtype),
                ("K", inDim),
                ("OUTN", outDim),
            ],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, 1, outDim]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
