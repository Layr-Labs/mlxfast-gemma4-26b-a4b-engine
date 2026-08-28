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
// The body below is the promoted `qmv_affine4_g64_quad_impl` with `load_vector`,
// `load_vector_safe` and `qdot_affine4_quad` expanded in place. Every expression
// is copied verbatim -- in particular the `T` operands are left as `T`, because
// `bfloat16_t` is Metal's native `bfloat` and `x[i] + x[i + 1]` rounds in bf16;
// rewriting those additions in float would change results. Only pointer
// derivation and the grid differ.
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

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_quad_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int SIMD = 32;
            constexpr int num_simdgroups = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int values_per_thread = 8;
            constexpr int block_size = values_per_thread * SIMD;
            constexpr int bytes_per_thread = 4;
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

            const device uint8_t* ws = (const device uint8_t*)w;
            thread float x0_thread[values_per_thread];
            thread float x1_thread[values_per_thread];
            thread float x2_thread[values_per_thread];
            thread float x3_thread[values_per_thread];
            thread float result0[results_per_simdgroup] = {0};
            thread float result1[results_per_simdgroup] = {0};
            thread float result2[results_per_simdgroup] = {0};
            thread float result3[results_per_simdgroup] = {0};

            const int in_vec_size_w = K / 2;
            const int in_vec_size_g = K / 64;
            const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup)
                + int(simd_gid) * results_per_simdgroup;

            ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
            const device T* sc = scales + out_row * in_vec_size_g
                + simd_lid / scale_step_per_thread;
            const device T* bi = biases + out_row * in_vec_size_g
                + simd_lid / scale_step_per_thread;
            x0 += simd_lid * values_per_thread;
            x1 += simd_lid * values_per_thread;
            x2 += simd_lid * values_per_thread;
            x3 += simd_lid * values_per_thread;
            y0 += out_row;
            y1 += out_row;
            y2 += out_row;
            y3 += out_row;

            int k = 0;
            for (; k < K - block_size; k += block_size) {
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for (int i = 0; i < values_per_thread; i += 4) {
                    sum0 += x0[i] + x0[i + 1] + x0[i + 2] + x0[i + 3];
                    x0_thread[i] = x0[i];
                    x0_thread[i + 1] = x0[i + 1] / 16.0f;
                    x0_thread[i + 2] = x0[i + 2] / 256.0f;
                    x0_thread[i + 3] = x0[i + 3] / 4096.0f;
                    sum1 += x1[i] + x1[i + 1] + x1[i + 2] + x1[i + 3];
                    x1_thread[i] = x1[i];
                    x1_thread[i + 1] = x1[i + 1] / 16.0f;
                    x1_thread[i + 2] = x1[i + 2] / 256.0f;
                    x1_thread[i + 3] = x1[i + 3] / 4096.0f;
                    sum2 += x2[i] + x2[i + 1] + x2[i + 2] + x2[i + 3];
                    x2_thread[i] = x2[i];
                    x2_thread[i + 1] = x2[i + 1] / 16.0f;
                    x2_thread[i + 2] = x2[i + 2] / 256.0f;
                    x2_thread[i + 3] = x2[i + 3] / 4096.0f;
                    sum3 += x3[i] + x3[i + 1] + x3[i + 2] + x3[i + 3];
                    x3_thread[i] = x3[i];
                    x3_thread[i + 1] = x3[i + 1] / 16.0f;
                    x3_thread[i + 2] = x3[i + 2] / 256.0f;
                    x3_thread[i + 3] = x3[i + 3] / 4096.0f;
                }

                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t* wl = ws + row * in_vec_size_w;
                    const device T* sl = sc + row * in_vec_size_g;
                    const device T* bl = bi + row * in_vec_size_g;
                    const float scale = sl[0];
                    const float bias = bl[0];
                    float accum0 = 0;
                    float accum1 = 0;
                    float accum2 = 0;
                    float accum3 = 0;
                    const device uint16_t* wp = (const device uint16_t*)wl;
                    for (int i = 0; i < (values_per_thread / 4); i++) {
                        const uint16_t packed = wp[i];
                        accum0 +=
                            (x0_thread[4 * i] * (packed & 0x000f) +
                             x0_thread[4 * i + 1] * (packed & 0x00f0) +
                             x0_thread[4 * i + 2] * (packed & 0x0f00) +
                             x0_thread[4 * i + 3] * (packed & 0xf000));
                        accum1 +=
                            (x1_thread[4 * i] * (packed & 0x000f) +
                             x1_thread[4 * i + 1] * (packed & 0x00f0) +
                             x1_thread[4 * i + 2] * (packed & 0x0f00) +
                             x1_thread[4 * i + 3] * (packed & 0xf000));
                        accum2 +=
                            (x2_thread[4 * i] * (packed & 0x000f) +
                             x2_thread[4 * i + 1] * (packed & 0x00f0) +
                             x2_thread[4 * i + 2] * (packed & 0x0f00) +
                             x2_thread[4 * i + 3] * (packed & 0xf000));
                        accum3 +=
                            (x3_thread[4 * i] * (packed & 0x000f) +
                             x3_thread[4 * i + 1] * (packed & 0x00f0) +
                             x3_thread[4 * i + 2] * (packed & 0x0f00) +
                             x3_thread[4 * i + 3] * (packed & 0xf000));
                    }
                    // Kept as a separate assignment then accumulate, exactly
                    // as `qdot_affine4_quad` writes its out params before the
                    // caller does `result[row] += dot`. Folding the two into
                    // one expression lets the compiler contract differently.
                    float dot0;
                    float dot1;
                    float dot2;
                    float dot3;
                    dot0 = scale * accum0 + sum0 * bias;
                    dot1 = scale * accum1 + sum1 * bias;
                    dot2 = scale * accum2 + sum2 * bias;
                    dot3 = scale * accum3 + sum3 * bias;
                    result0[row] += dot0;
                    result1[row] += dot1;
                    result2[row] += dot2;
                    result3[row] += dot3;
                }

                ws += block_size / 2;
                sc += block_size / 64;
                bi += block_size / 64;
                x0 += block_size;
                x1 += block_size;
                x2 += block_size;
                x3 += block_size;
            }

            const int remaining = clamp(
                static_cast<int>(K - k - simd_lid * values_per_thread),
                0,
                values_per_thread);
            if (remaining > 0) {
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for (int i = 0; i < remaining; i += 4) {
                    sum0 += x0[i] + x0[i + 1] + x0[i + 2] + x0[i + 3];
                    x0_thread[i] = x0[i];
                    x0_thread[i + 1] = x0[i + 1] / 16.0f;
                    x0_thread[i + 2] = x0[i + 2] / 256.0f;
                    x0_thread[i + 3] = x0[i + 3] / 4096.0f;
                    sum1 += x1[i] + x1[i + 1] + x1[i + 2] + x1[i + 3];
                    x1_thread[i] = x1[i];
                    x1_thread[i + 1] = x1[i + 1] / 16.0f;
                    x1_thread[i + 2] = x1[i + 2] / 256.0f;
                    x1_thread[i + 3] = x1[i + 3] / 4096.0f;
                    sum2 += x2[i] + x2[i + 1] + x2[i + 2] + x2[i + 3];
                    x2_thread[i] = x2[i];
                    x2_thread[i + 1] = x2[i + 1] / 16.0f;
                    x2_thread[i + 2] = x2[i + 2] / 256.0f;
                    x2_thread[i + 3] = x2[i + 3] / 4096.0f;
                    sum3 += x3[i] + x3[i + 1] + x3[i + 2] + x3[i + 3];
                    x3_thread[i] = x3[i];
                    x3_thread[i + 1] = x3[i + 1] / 16.0f;
                    x3_thread[i + 2] = x3[i + 2] / 256.0f;
                    x3_thread[i + 3] = x3[i + 3] / 4096.0f;
                }
                for (int i = remaining; i < values_per_thread; i++) {
                    x0_thread[i] = 0;
                    x1_thread[i] = 0;
                    x2_thread[i] = 0;
                    x3_thread[i] = 0;
                }

                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t* wl = ws + row * in_vec_size_w;
                    const device T* sl = sc + row * in_vec_size_g;
                    const device T* bl = bi + row * in_vec_size_g;
                    const float scale = sl[0];
                    const float bias = bl[0];
                    float accum0 = 0;
                    float accum1 = 0;
                    float accum2 = 0;
                    float accum3 = 0;
                    const device uint16_t* wp = (const device uint16_t*)wl;
                    for (int i = 0; i < (values_per_thread / 4); i++) {
                        const uint16_t packed = wp[i];
                        accum0 +=
                            (x0_thread[4 * i] * (packed & 0x000f) +
                             x0_thread[4 * i + 1] * (packed & 0x00f0) +
                             x0_thread[4 * i + 2] * (packed & 0x0f00) +
                             x0_thread[4 * i + 3] * (packed & 0xf000));
                        accum1 +=
                            (x1_thread[4 * i] * (packed & 0x000f) +
                             x1_thread[4 * i + 1] * (packed & 0x00f0) +
                             x1_thread[4 * i + 2] * (packed & 0x0f00) +
                             x1_thread[4 * i + 3] * (packed & 0xf000));
                        accum2 +=
                            (x2_thread[4 * i] * (packed & 0x000f) +
                             x2_thread[4 * i + 1] * (packed & 0x00f0) +
                             x2_thread[4 * i + 2] * (packed & 0x0f00) +
                             x2_thread[4 * i + 3] * (packed & 0xf000));
                        accum3 +=
                            (x3_thread[4 * i] * (packed & 0x000f) +
                             x3_thread[4 * i + 1] * (packed & 0x00f0) +
                             x3_thread[4 * i + 2] * (packed & 0x0f00) +
                             x3_thread[4 * i + 3] * (packed & 0xf000));
                    }
                    // Kept as a separate assignment then accumulate, exactly
                    // as `qdot_affine4_quad` writes its out params before the
                    // caller does `result[row] += dot`. Folding the two into
                    // one expression lets the compiler contract differently.
                    float dot0;
                    float dot1;
                    float dot2;
                    float dot3;
                    dot0 = scale * accum0 + sum0 * bias;
                    dot1 = scale * accum1 + sum1 * bias;
                    dot2 = scale * accum2 + sum2 * bias;
                    dot3 = scale * accum3 + sum3 * bias;
                    result0[row] += dot0;
                    result1[row] += dot1;
                    result2[row] += dot2;
                    result3[row] += dot3;
                }
            }

            for (int row = 0; row < results_per_simdgroup; row++) {
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
        return kernel(
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
