// Tight-grid dispatch for the accepted affine4/affine8 g64 four-row QMV.
//
// Ordinary QMV launches eight x-threadgroups for the B=8 cohort, while the
// stream helper claims four rows per group and returns from groups 2...7. This
// custom dispatch runs the same helper body with exactly two x-threadgroups.

import Foundation
import MLX
import MLXFast

public enum CBv2QMVQuadStreamTightGridV1 {
    private static let enabled: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_CBV2_QMV_TIGHT_GRID"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let groupSize = 64
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let rowsPerXGroup = 4
    private static let outputsPerThreadgroup = 8
    private static let supportedAffine4Outputs = [1024, 2048, 4096, 8192]

    private static let header = """
        template <typename T, typename U, int values_per_thread, int bits>
        inline U cbv2_qmv4_load_vector(
            const device T* x,
            thread U* x_thread) {
          U sum = 0;
          for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U cbv2_qmv4_load_vector_safe(
            const device T* x,
            thread U* x_thread,
            int N) {
          U sum = 0;
          for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }
          return sum;
        }

        template <typename U, int values_per_thread>
        inline U cbv2_qdot_affine4_registered(
            const thread uint16_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          for (int i = 0; i < (values_per_thread / 4); i++) {
            accum +=
                (x_thread[4 * i] * (w[i] & 0x000f) +
                 x_thread[4 * i + 1] * (w[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (w[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (w[i] & 0xf000));
          }
          return scale * accum + sum * bias;
        }

        template <typename T>
        inline void cbv2_qmv_affine4_g64_quad_stream(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            const device T* x2,
            const device T* x3,
            device T* y0,
            device T* y1,
            device T* y2,
            device T* y3,
            const int in_vec_size,
            uint3 tid,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int values_per_thread = 8;
          constexpr int block_size = values_per_thread * 32;
          constexpr int bytes_per_thread = 4;
          constexpr int uint16_per_thread = bytes_per_thread / 2;
          constexpr int scale_step_per_thread = 8;

          const device uint8_t* ws = (const device uint8_t*)w;
          thread float x_thread[values_per_thread];
          thread uint16_t packed[results_per_simdgroup][uint16_per_thread];
          thread float scale_local[results_per_simdgroup];
          thread float bias_local[results_per_simdgroup];
          thread float result0[results_per_simdgroup] = {0};
          thread float result1[results_per_simdgroup] = {0};
          thread float result2[results_per_simdgroup] = {0};
          thread float result3[results_per_simdgroup] = {0};

          const int in_vec_size_w = in_vec_size / 2;
          const int in_vec_size_g = in_vec_size / 64;
          const int out_row = tid.y *
              (num_simdgroups * results_per_simdgroup) +
              simd_gid * results_per_simdgroup;

          ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
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

          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            for (int row = 0; row < results_per_simdgroup; row++) {
              const device uint16_t* wl =
                  (const device uint16_t*)(ws + row * in_vec_size_w);
              for (int i = 0; i < uint16_per_thread; i++) {
                packed[row][i] = wl[i];
              }
              scale_local[row] = scales[row * in_vec_size_g];
              bias_local[row] = biases[row * in_vec_size_g];
            }

            float sum = cbv2_qmv4_load_vector<T, float, values_per_thread, 4>(
                x0, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result0[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector<T, float, values_per_thread, 4>(
                x1, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result1[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector<T, float, values_per_thread, 4>(
                x2, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result2[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector<T, float, values_per_thread, 4>(
                x3, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result3[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }

            ws += block_size / 2;
            scales += block_size / 64;
            biases += block_size / 64;
            x0 += block_size;
            x1 += block_size;
            x2 += block_size;
            x3 += block_size;
          }

          const int remaining = clamp(
              static_cast<int>(
                  in_vec_size - k - simd_lid * values_per_thread),
              0,
              values_per_thread);
          if (remaining > 0) {
            for (int row = 0; row < results_per_simdgroup; row++) {
              const device uint16_t* wl =
                  (const device uint16_t*)(ws + row * in_vec_size_w);
              for (int i = 0; i < uint16_per_thread; i++) {
                packed[row][i] = wl[i];
              }
              scale_local[row] = scales[row * in_vec_size_g];
              bias_local[row] = biases[row * in_vec_size_g];
            }

            float sum = cbv2_qmv4_load_vector_safe<
                T, float, values_per_thread, 4>(x0, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result0[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector_safe<
                T, float, values_per_thread, 4>(x1, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result1[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector_safe<
                T, float, values_per_thread, 4>(x2, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result2[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv4_load_vector_safe<
                T, float, values_per_thread, 4>(x3, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result3[row] += cbv2_qdot_affine4_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
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
        }
        """

    private static let affine8Header = """
        template <typename T, typename U, int values_per_thread>
        inline U cbv2_qmv8_load_vector(
            const device T* x,
            thread U* x_thread) {
          U sum = 0;
          for (int i = 0; i < values_per_thread; i++) {
            sum += x[i];
            x_thread[i] = x[i];
          }
          return sum;
        }

        template <typename U, int values_per_thread>
        inline U cbv2_qdot_affine8_registered(
            const thread uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          for (int i = 0; i < values_per_thread; i++) {
            accum += x_thread[i] * w[i];
          }
          return scale * accum + sum * bias;
        }

        template <typename T>
        inline void cbv2_qmv_affine8_g64_quad_stream(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            const device T* x2,
            const device T* x3,
            device T* y0,
            device T* y1,
            device T* y2,
            device T* y3,
            const int in_vec_size,
            uint3 tid,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int values_per_thread = 4;
          constexpr int block_size = values_per_thread * 32;
          constexpr int bytes_per_thread = 4;
          constexpr int scale_step_per_thread = 16;

          const device uint8_t* ws = (const device uint8_t*)w;
          thread float x_thread[values_per_thread];
          thread uint8_t packed[results_per_simdgroup][bytes_per_thread];
          thread float scale_local[results_per_simdgroup];
          thread float bias_local[results_per_simdgroup];
          thread float result0[results_per_simdgroup] = {0};
          thread float result1[results_per_simdgroup] = {0};
          thread float result2[results_per_simdgroup] = {0};
          thread float result3[results_per_simdgroup] = {0};

          const int in_vec_size_w = in_vec_size;
          const int in_vec_size_g = in_vec_size / 64;
          const int out_row = tid.y *
              (num_simdgroups * results_per_simdgroup) +
              simd_gid * results_per_simdgroup;

          ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
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

          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            for (int row = 0; row < results_per_simdgroup; row++) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              for (int i = 0; i < bytes_per_thread; i++) {
                packed[row][i] = wl[i];
              }
              scale_local[row] = scales[row * in_vec_size_g];
              bias_local[row] = biases[row * in_vec_size_g];
            }

            float sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x0, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result0[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x1, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result1[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x2, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result2[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x3, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result3[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }

            ws += block_size;
            scales += block_size / 64;
            biases += block_size / 64;
            x0 += block_size;
            x1 += block_size;
            x2 += block_size;
            x3 += block_size;
          }

          // Preserve the accepted aligned-tail specialization from the base.
          // Both supported K values are g64-aligned, so every active lane owns
          // one complete four-value packet.
          const uint active_tail_lanes =
              uint((in_vec_size - k) / values_per_thread);
          if (simd_lid < active_tail_lanes) {
            for (int row = 0; row < results_per_simdgroup; row++) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              for (int i = 0; i < bytes_per_thread; i++) {
                packed[row][i] = wl[i];
              }
              scale_local[row] = scales[row * in_vec_size_g];
              bias_local[row] = biases[row * in_vec_size_g];
            }

            float sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x0, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result0[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x1, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result1[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x2, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result2[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
            }
            sum = cbv2_qmv8_load_vector<
                T, float, values_per_thread>(x3, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              result3[row] += cbv2_qdot_affine8_registered<
                  float, values_per_thread>(
                      packed[row], x_thread, scale_local[row],
                      bias_local[row], sum);
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
        }
        """

    private static let affine4Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_attention_qmv_affine4_g64_quad_stream_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const int first_m = int(threadgroup_position_in_grid.x) * 4;
            cbv2_qmv_affine4_g64_quad_stream<T>(
                (const device uint32_t*)w,
                scales,
                biases,
                x + (first_m + 0) * K,
                x + (first_m + 1) * K,
                x + (first_m + 2) * K,
                x + (first_m + 3) * K,
                y + (first_m + 0) * N,
                y + (first_m + 1) * N,
                y + (first_m + 2) * N,
                y + (first_m + 3) * N,
                K,
                threadgroup_position_in_grid,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: header,
        ensureRowContiguous: true
    )

    private static let affine8Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_mlp_qmv_affine8_g64_quad_stream_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const int first_m = int(threadgroup_position_in_grid.x) * 4;
            cbv2_qmv_affine8_g64_quad_stream<T>(
                (const device uint32_t*)w,
                scales,
                biases,
                x + (first_m + 0) * K,
                x + (first_m + 1) * K,
                x + (first_m + 2) * K,
                x + (first_m + 3) * K,
                y + (first_m + 0) * N,
                y + (first_m + 1) * N,
                y + (first_m + 2) * N,
                y + (first_m + 3) * N,
                K,
                threadgroup_position_in_grid,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: affine8Header,
        ensureRowContiguous: true
    )

    /// Returns nil unless the tensor and frozen affine quantization match one
    /// of the exact B=8 singleton projection contracts served by the accepted
    /// four-row stream helpers.
    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        bits: Int,
        inputDimension: Int,
        outputDimension: Int
    ) -> MLXArray? {
        let supportedGeometry: Bool
        switch bits {
        case 4:
            supportedGeometry =
                inputDimension == 2816
                && supportedAffine4Outputs.contains(outputDimension)
        case 8:
            supportedGeometry =
                (inputDimension == 2816 && outputDimension == 2112)
                || (inputDimension == 2112 && outputDimension == 2816)
        default:
            supportedGeometry = false
        }
        guard enabled,
            let biases,
            supportedGeometry,
            x.dtype == .bfloat16,
            x.shape == [batch, 1, inputDimension],
            weight.dtype == .uint32,
            weight.shape == [outputDimension, inputDimension * bits / 32],
            scales.dtype == x.dtype,
            scales.shape == [outputDimension, inputDimension / groupSize],
            biases.dtype == x.dtype,
            biases.shape == scales.shape
        else { return nil }

        let xGroups = batch / rowsPerXGroup
        let yGroups = outputDimension / outputsPerThreadgroup
        let selectedKernel = bits == 4 ? affine4Kernel : affine8Kernel
        CBv2EngageMark.once(bits == 4 ? "qmv4-tight2" : "qmv8-tight2")
        return selectedKernel(
            [x, weight, scales, biases],
            template: [
                ("T", x.dtype),
                ("K", inputDimension),
                ("N", outputDimension),
            ],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, 1, outputDimension]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
