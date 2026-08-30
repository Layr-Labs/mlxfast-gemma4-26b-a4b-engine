// Copyright © 2026 Apple Inc.

import MLX
import MLXFast

/// Construction-bound affine QMV shared by two to four verifier positions at
/// physical batch one.
///
/// Each output row retains the ordinary single-row QMV lane arithmetic and
/// `simd_sum` reduction.  The only sharing is that one lane loads a packed
/// weight packet and its affine metadata once, then applies that immutable
/// packet independently to each verifier column.
public enum Gemma4B1MTPQuantizedProjection {
    public typealias Projection = (MLXArray) -> MLXArray

    public static func bind(
        columns: Int,
        inDim: Int,
        outDim: Int,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> Projection? {
        guard (2...4).contains(columns),
            inDim > 0,
            outDim > 0,
            inDim.isMultiple(of: 64),
            groupSize == 64,
            bits == 4 || bits == 8,
            let biases,
            weight.dtype == .uint32,
            scales.dtype == .bfloat16,
            biases.dtype == .bfloat16,
            weight.shape == [outDim, inDim * bits / 32],
            scales.shape == [outDim, inDim / 64],
            biases.shape == scales.shape
        else { return nil }

        let kernel = bits == 4 ? affine4Kernel : affine8Kernel
        return { x in
            precondition(x.shape == [1, columns, inDim])
            let flat = x.reshaped([columns, inDim])
            return kernel(
                [flat, weight, scales, biases],
                template: [("T", DType.bfloat16), ("COLUMNS", columns)],
                grid: (32, outDim, 1),
                threadGroup: (32, 1, 1),
                outputShapes: [[columns, outDim]],
                outputDTypes: [.bfloat16]
            )[0].reshaped([1, columns, outDim])
        }
    }

    // Arithmetic provenance: qmv_impl<T, 64, 4> in pinned quantized.h,
    // specifically its values-per-thread/block geometry, load_vector affine-4
    // transform, qdot affine-4 expression, K loop, tail, and simd_sum close.
    private static let affine4Kernel = MLXFast.metalKernel(
        name: "cbv2_gemma4_b1_mtp_affine4_g64_exact_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int values_per_thread = 8;
            constexpr int block_size = values_per_thread * 32;
            constexpr int scale_step_per_thread = 64 / values_per_thread;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int out_row = int(threadgroup_position_in_grid.y);
            const int lane = int(thread_index_in_simdgroup);
            const int in_vec_size_w = in_vec_size / 2;
            const int in_vec_size_g = in_vec_size / 64;

            thread float x_thread[COLUMNS][values_per_thread];
            thread float result[COLUMNS] = {0};

            int k = 0;
            for (; k <= in_vec_size - block_size; k += block_size) {
                const device ushort* wl =
                    reinterpret_cast<const device ushort*>(
                        reinterpret_cast<const device uchar*>(w) +
                        out_row * in_vec_size_w + k / 2 + lane * 4);
                const ushort packed0 = wl[0];
                const ushort packed1 = wl[1];
                const int group_index =
                    out_row * in_vec_size_g + k / 64 +
                    lane / scale_step_per_thread;
                const float s = scales[group_index];
                const float b = biases[group_index];

                for (int column = 0; column < COLUMNS; column++) {
                    const device T* xm =
                        x + column * in_vec_size + k +
                        lane * values_per_thread;
                    float sum = 0;
                    sum += xm[0] + xm[1] + xm[2] + xm[3];
                    x_thread[column][0] = xm[0];
                    x_thread[column][1] = xm[1] / 16.0f;
                    x_thread[column][2] = xm[2] / 256.0f;
                    x_thread[column][3] = xm[3] / 4096.0f;
                    sum += xm[4] + xm[5] + xm[6] + xm[7];
                    x_thread[column][4] = xm[4];
                    x_thread[column][5] = xm[5] / 16.0f;
                    x_thread[column][6] = xm[6] / 256.0f;
                    x_thread[column][7] = xm[7] / 4096.0f;

                    float accum = 0;
                    accum +=
                        (x_thread[column][0] * (packed0 & 0x000f) +
                         x_thread[column][1] * (packed0 & 0x00f0) +
                         x_thread[column][2] * (packed0 & 0x0f00) +
                         x_thread[column][3] * (packed0 & 0xf000));
                    accum +=
                        (x_thread[column][4] * (packed1 & 0x000f) +
                         x_thread[column][5] * (packed1 & 0x00f0) +
                         x_thread[column][6] * (packed1 & 0x0f00) +
                         x_thread[column][7] * (packed1 & 0xf000));
                    result[column] += s * accum + sum * b;
                }
            }

            const int tail_values = in_vec_size - k;
            if (tail_values > 0 &&
                lane < tail_values / values_per_thread) {
                const device ushort* wl =
                    reinterpret_cast<const device ushort*>(
                        reinterpret_cast<const device uchar*>(w) +
                        out_row * in_vec_size_w + k / 2 + lane * 4);
                const ushort packed0 = wl[0];
                const ushort packed1 = wl[1];
                const int group_index =
                    out_row * in_vec_size_g + k / 64 +
                    lane / scale_step_per_thread;
                const float s = scales[group_index];
                const float b = biases[group_index];

                for (int column = 0; column < COLUMNS; column++) {
                    const device T* xm =
                        x + column * in_vec_size + k +
                        lane * values_per_thread;
                    float sum = 0;
                    sum += xm[0] + xm[1] + xm[2] + xm[3];
                    x_thread[column][0] = xm[0];
                    x_thread[column][1] = xm[1] / 16.0f;
                    x_thread[column][2] = xm[2] / 256.0f;
                    x_thread[column][3] = xm[3] / 4096.0f;
                    sum += xm[4] + xm[5] + xm[6] + xm[7];
                    x_thread[column][4] = xm[4];
                    x_thread[column][5] = xm[5] / 16.0f;
                    x_thread[column][6] = xm[6] / 256.0f;
                    x_thread[column][7] = xm[7] / 4096.0f;

                    float accum = 0;
                    accum +=
                        (x_thread[column][0] * (packed0 & 0x000f) +
                         x_thread[column][1] * (packed0 & 0x00f0) +
                         x_thread[column][2] * (packed0 & 0x0f00) +
                         x_thread[column][3] * (packed0 & 0xf000));
                    accum +=
                        (x_thread[column][4] * (packed1 & 0x000f) +
                         x_thread[column][5] * (packed1 & 0x00f0) +
                         x_thread[column][6] * (packed1 & 0x0f00) +
                         x_thread[column][7] * (packed1 & 0xf000));
                    result[column] += s * accum + sum * b;
                }
            }

            for (int column = 0; column < COLUMNS; column++) {
                result[column] = simd_sum(result[column]);
                if (lane == 0) {
                    y[column * out_vec_size + out_row] =
                        static_cast<T>(result[column]);
                }
            }
            """,
        ensureRowContiguous: true)

    // Arithmetic provenance: the corresponding qmv_impl<T, 64, 8> path.  Its
    // lane packet is four bytes rather than affine-4's two ushort packets.
    private static let affine8Kernel = MLXFast.metalKernel(
        name: "cbv2_gemma4_b1_mtp_affine8_g64_exact_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            constexpr int values_per_thread = 4;
            constexpr int block_size = values_per_thread * 32;
            constexpr int scale_step_per_thread = 64 / values_per_thread;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int out_row = int(threadgroup_position_in_grid.y);
            const int lane = int(thread_index_in_simdgroup);
            const int in_vec_size_w = in_vec_size;
            const int in_vec_size_g = in_vec_size / 64;

            thread float x_thread[COLUMNS][values_per_thread];
            thread float result[COLUMNS] = {0};

            int k = 0;
            for (; k <= in_vec_size - block_size; k += block_size) {
                const device uchar* wl =
                    reinterpret_cast<const device uchar*>(w) +
                    out_row * in_vec_size_w + k + lane * 4;
                const uchar packed0 = wl[0];
                const uchar packed1 = wl[1];
                const uchar packed2 = wl[2];
                const uchar packed3 = wl[3];
                const int group_index =
                    out_row * in_vec_size_g + k / 64 +
                    lane / scale_step_per_thread;
                const float s = scales[group_index];
                const float b = biases[group_index];

                for (int column = 0; column < COLUMNS; column++) {
                    const device T* xm =
                        x + column * in_vec_size + k +
                        lane * values_per_thread;
                    float sum = 0;
                    sum += xm[0];
                    x_thread[column][0] = xm[0];
                    sum += xm[1];
                    x_thread[column][1] = xm[1];
                    sum += xm[2];
                    x_thread[column][2] = xm[2];
                    sum += xm[3];
                    x_thread[column][3] = xm[3];

                    float accum = 0;
                    accum += x_thread[column][0] * packed0;
                    accum += x_thread[column][1] * packed1;
                    accum += x_thread[column][2] * packed2;
                    accum += x_thread[column][3] * packed3;
                    result[column] += s * accum + sum * b;
                }
            }

            const int tail_values = in_vec_size - k;
            if (tail_values > 0 &&
                lane < tail_values / values_per_thread) {
                const device uchar* wl =
                    reinterpret_cast<const device uchar*>(w) +
                    out_row * in_vec_size_w + k + lane * 4;
                const uchar packed0 = wl[0];
                const uchar packed1 = wl[1];
                const uchar packed2 = wl[2];
                const uchar packed3 = wl[3];
                const int group_index =
                    out_row * in_vec_size_g + k / 64 +
                    lane / scale_step_per_thread;
                const float s = scales[group_index];
                const float b = biases[group_index];

                for (int column = 0; column < COLUMNS; column++) {
                    const device T* xm =
                        x + column * in_vec_size + k +
                        lane * values_per_thread;
                    float sum = 0;
                    sum += xm[0];
                    x_thread[column][0] = xm[0];
                    sum += xm[1];
                    x_thread[column][1] = xm[1];
                    sum += xm[2];
                    x_thread[column][2] = xm[2];
                    sum += xm[3];
                    x_thread[column][3] = xm[3];

                    float accum = 0;
                    accum += x_thread[column][0] * packed0;
                    accum += x_thread[column][1] * packed1;
                    accum += x_thread[column][2] * packed2;
                    accum += x_thread[column][3] * packed3;
                    result[column] += s * accum + sum * b;
                }
            }

            for (int column = 0; column < COLUMNS; column++) {
                result[column] = simd_sum(result[column]);
                if (lane == 0) {
                    y[column * out_vec_size + out_row] =
                        static_cast<T>(result[column]);
                }
            }
            """,
        ensureRowContiguous: true)
}
