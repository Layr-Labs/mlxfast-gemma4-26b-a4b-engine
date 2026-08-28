// QMV8-PAIR-001: tight-grid dispatch for the ordinary affine-8 QMV pair tier.
//
// Third member of the same family as `CBv2TiedLMHeadQMVV1` (affine-4 quad tier)
// and `CBv2OrdinaryQMVPairV1` (affine-4 pair tier). `affine_qmv`'s `bits == 8`
// branch claims two cohort rows per threadgroup:
//
//     const int first_m = int(tid.x) * 2;
//     if (first_m >= 8) { return; }
//
// The host launches x = M = 8 threadgroups either way, so four of every eight
// retire at that `return`. This dispatches with an x extent of four; `first_m`
// stays `tid.x * 2`, so the four surviving groups claim rows {0,1}, {2,3},
// {4,5}, {6,7} exactly as before.
//
// This checkpoint's 8-bit tensors are the four per-tensor override families --
// `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `router.proj`, thirty each.
//
// The body is `qmv_affine8_g64_pair_impl` with `load_vector`,
// `load_vector_safe` and `qdot_affine8_pair` expanded in place, every
// expression verbatim. Note the geometry differs from the 4-bit impls:
// values_per_thread 4, block_size 128, in_vec_size_w == in_vec_size (one byte
// per value), scale_step_per_thread 16.
//
// `DARKBLOOM_CBV2_ORDINARY_QMV8_PAIR=0` restores the stock path.

import Foundation
import MLX
import MLXFast

public enum CBv2OrdinaryQMV8PairV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_ORDINARY_QMV8_PAIR"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let groupSize = 64
    private static let bits = 8
    private static let rowsPerGroup = 2
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let outputsPerGroup = 8
    private static let minInDim = 256

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_ordinary_qmv_affine8_g64_pair_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int SIMD = 32;
            constexpr int num_simdgroups = 2;
            constexpr int results_per_simdgroup = 4;
            constexpr int values_per_thread = 4;
            constexpr int block_size = values_per_thread * SIMD;
            constexpr int bytes_per_thread = 4;
            constexpr int scale_step_per_thread = 16;

            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int first_m = int(tid.x) * 2;

            const device T* x0 = x + (first_m + 0) * K;
            const device T* x1 = x + (first_m + 1) * K;
            device T* y0 = y + (first_m + 0) * OUTN;
            device T* y1 = y + (first_m + 1) * OUTN;

            const device uint8_t* ws = (const device uint8_t*)w;
            thread float x0_thread[values_per_thread];
            thread float x1_thread[values_per_thread];
            thread float result0[results_per_simdgroup] = {0};
            thread float result1[results_per_simdgroup] = {0};

            const int in_vec_size_w = K;
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
            y0 += out_row;
            y1 += out_row;

            int k = 0;
            for (; k < K - block_size; k += block_size) {
                float sum0 = 0;
                float sum1 = 0;
                for (int i = 0; i < values_per_thread; i++) {
                    sum0 += x0[i];
                    x0_thread[i] = x0[i];
                    sum1 += x1[i];
                    x1_thread[i] = x1[i];
                }

                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t* wl = ws + row * in_vec_size_w;
                    const device T* sl = sc + row * in_vec_size_g;
                    const device T* bl = bi + row * in_vec_size_g;
                    const float scale = sl[0];
                    const float bias = bl[0];
                    float accum0 = 0;
                    float accum1 = 0;
                    for (int i = 0; i < values_per_thread; i++) {
                        const uint8_t packed = wl[i];
                        accum0 += x0_thread[i] * packed;
                        accum1 += x1_thread[i] * packed;
                    }
                    float dot0;
                    float dot1;
                    dot0 = scale * accum0 + sum0 * bias;
                    dot1 = scale * accum1 + sum1 * bias;
                    result0[row] += dot0;
                    result1[row] += dot1;
                }

                ws += block_size;
                sc += block_size / 64;
                bi += block_size / 64;
                x0 += block_size;
                x1 += block_size;
            }

            const int remaining = clamp(
                static_cast<int>(K - k - simd_lid * values_per_thread),
                0,
                values_per_thread);
            if (remaining > 0) {
                float sum0 = 0;
                float sum1 = 0;
                for (int i = 0; i < remaining; i++) {
                    sum0 += x0[i];
                    x0_thread[i] = x0[i];
                    sum1 += x1[i];
                    x1_thread[i] = x1[i];
                }
                for (int i = remaining; i < values_per_thread; i++) {
                    x0_thread[i] = 0;
                    x1_thread[i] = 0;
                }

                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t* wl = ws + row * in_vec_size_w;
                    const device T* sl = sc + row * in_vec_size_g;
                    const device T* bl = bi + row * in_vec_size_g;
                    const float scale = sl[0];
                    const float bias = bl[0];
                    float accum0 = 0;
                    float accum1 = 0;
                    for (int i = 0; i < values_per_thread; i++) {
                        const uint8_t packed = wl[i];
                        accum0 += x0_thread[i] * packed;
                        accum1 += x1_thread[i] * packed;
                    }
                    float dot0;
                    float dot1;
                    dot0 = scale * accum0 + sum0 * bias;
                    dot1 = scale * accum1 + sum1 * bias;
                    result0[row] += dot0;
                    result1[row] += dot1;
                }
            }

            for (int row = 0; row < results_per_simdgroup; row++) {
                result0[row] = simd_sum(result0[row]);
                result1[row] = simd_sum(result1[row]);
                if (simd_lid == 0) {
                    y0[row] = static_cast<T>(result0[row]);
                    y1[row] = static_cast<T>(result1[row]);
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
            inDim >= minInDim,
            // `qmv()` routes K % 512 == 0 to `affine_qmv_fast`, which has no
            // pair tier and no idle groups to remove.
            inDim % 512 != 0,
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
