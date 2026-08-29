// DMLP-001: tight-grid dispatch for Gemma 4's dense affine-8 MLP QMV at
// batch eight and decode length one.
//
// The vendored MLX host launches ordinary QMV with an x grid extent equal to
// the input-row count, M = 8. IPG3 (not a 4-row clamp) claims three then two
// rows per group:
//
//     first_m = threadgroup_position_in_grid.x * 3
//
// Groups 0 and 1 run qmv_affine8_g64_triple_stream_impl over rows 0..2 and
// 3..5; group 2 runs qmv_affine8_g64_pair_impl over the TAIL=2 remainder
// (rows 6..7). Host grid.x is 3, not batch/4. The y grid, threadgroup shape,
// row ownership, pointers, arithmetic, K loop, g64-aligned tail and simd
// reductions stay the promoted affine-8 helpers.
//
// Across one batch-eight decode step the family has 60 gate/up projections at
// K=2816,N=2112 and 30 down projections at K=2112,N=2816.
//
// `DARKBLOOM_CBV2_DENSE_MLP_QMV=0` restores the stock QuantizedLinear path in
// the same executable. `DARKBLOOM_CBV2_DENSE_MLP_XSUM=0` keeps the tight grid
// but disables the activation-sum table. Every non-production dtype, mode,
// shape or quantization geometry fails closed to the prior path. Selected
// arrays are normalized to row-contiguous layout by `MLXFast.metalKernel`.

import Foundation
import MLX
import MLXFast

public enum CBv2DenseMLPQMVV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DENSE_MLP_QMV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    public static let activationSumsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DENSE_MLP_XSUM"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let sequence = 1
    private static let groupSize = 64
    private static let bits = 8
    private static let rowsPerGroup = 3
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let outputsPerGroup = 8
    private static let valuesPerLane = 4
    private static let kBlock = simdWidth * valuesPerLane

    /// An opaque table tied to one activation tensor. The initializer is not
    /// public: production tables can only come from `activationSums(for:)`,
    /// whose exact B8/L1/K2816 gate also pins the table geometry.
    public struct ActivationSums {
        fileprivate let values: MLXArray
    }

    /// The affine-8 helpers below are the promoted pair + triple-stream
    /// bodies. IPG=3 on M=8 is 3+3+2: two triples then pair TAIL=2. Reuses
    /// `load_vector` from the tied-head replica, then the affine-8 registered
    /// / pair dots from `quantized.h`.
    ///
    /// The helper's `in_vec_size` is a value rather than a `constant` address-
    /// space reference because custom kernels receive shape metadata instead
    /// of MLX's private scalar constant buffer. The value is read at runtime
    /// from `x_shape`; it is deliberately not a template constant, preserving
    /// the incumbent runtime K loop for both live shapes.
    private static let kernelHeader = CBv2TiedLMHeadQMVV1.kernelHeader + """

template <typename U, int values_per_thread>
inline U qdot_affine8_registered(
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

template <typename U, int values_per_thread>
inline void qdot_affine8_pair(
    const device uint8_t* w,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  U accum0 = 0;
  U accum1 = 0;
  for (int i = 0; i < values_per_thread; i++) {
    const uint8_t packed = w[i];
    accum0 += x0[i] * packed;
    accum1 += x1[i] * packed;
  }
  out0 = scale * accum0 + sum0 * bias;
  out1 = scale * accum1 + sum1 * bias;
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_pair_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    device T* y0,
    device T* y1,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 4;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread float x1_thread[values_per_thread];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    float sum0 = load_vector<T, float, values_per_thread, 8>(x0, x0_thread);
    float sum1 = load_vector<T, float, values_per_thread, 8>(x1, x1_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine8_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    float sum0 = load_vector_safe<T, float, values_per_thread, 8>(
        x0, x0_thread, remaining);
    float sum1 = load_vector_safe<T, float, values_per_thread, 8>(
        x1, x1_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine8_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
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
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_triple_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    device T* y0,
    device T* y1,
    device T* y2,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 4;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
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

  const int in_vec_size_w = in_vec_size;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  x2 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
  }

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

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
    }
  }
}
"""

    /// DMLP-002 keeps DMLP-001's kernel text intact and derives xsum bodies
    /// which replace only the four-value activation sum. The x values are
    /// still loaded into the same register array, in the same order, and every
    /// qdot, K accumulation and simd reduction remains text-for-text identical.
    /// first_m uses IPG=3 so the table index matches the 3+3+2 host grid.
    private static let activationSumKernelHeader: String = {
        var result = kernelHeader

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            template <typename T, const int group_size, const int bits>
            METAL_FUNC void qmv_affine8_g64_pair_impl(
            """,
            with: """
            template <typename T, typename U, int values_per_thread>
            inline void load_affine8_values(
                const device T* x,
                thread U* x_thread) {
              for (int i = 0; i < values_per_thread; i++) {
                x_thread[i] = x[i];
              }
            }

            template <typename T, const int group_size, const int bits>
            METAL_FUNC void qmv_affine8_g64_pair_xsum_impl(
            """
        )
        replaceOnce(
            """
            METAL_FUNC void qmv_affine8_g64_triple_stream_impl(
            """,
            with: """
            METAL_FUNC void qmv_affine8_g64_triple_stream_xsum_impl(
            """
        )
        replaceOnce(
            """
                const device T* biases,
                const device T* x0,
                const device T* x1,
                device T* y0,
                device T* y1,
            """,
            with: """
                const device T* biases,
                const device float* x_sums,
                const device T* x0,
                const device T* x1,
                device T* y0,
                device T* y1,
            """
        )
        replaceOnce(
            """
                const device T* biases,
                const device T* x0,
                const device T* x1,
                const device T* x2,
                device T* y0,
                device T* y1,
                device T* y2,
            """,
            with: """
                const device T* biases,
                const device float* x_sums,
                const device T* x0,
                const device T* x1,
                const device T* x2,
                device T* y0,
                device T* y1,
                device T* y2,
            """
        )
        replaceOnce(
            """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;

              ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
              scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              x0 += simd_lid * values_per_thread;
              x1 += simd_lid * values_per_thread;
              y0 += out_row;
              y1 += out_row;
            """,
            with: """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
              const int first_m = int(tid.x) * 3;

              ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
              scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              x0 += simd_lid * values_per_thread;
              x1 += simd_lid * values_per_thread;
              y0 += out_row;
              y1 += out_row;
            """
        )
        replaceOnce(
            """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;

              ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
              scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              x0 += simd_lid * values_per_thread;
              x1 += simd_lid * values_per_thread;
              x2 += simd_lid * values_per_thread;
              y0 += out_row;
              y1 += out_row;
              y2 += out_row;
            """,
            with: """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
              const int first_m = int(tid.x) * 3;

              ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
              scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
              x0 += simd_lid * values_per_thread;
              x1 += simd_lid * values_per_thread;
              x2 += simd_lid * values_per_thread;
              y0 += out_row;
              y1 += out_row;
              y2 += out_row;
            """
        )

        replaceOnce(
            "float sum0 = load_vector<T, float, values_per_thread, 8>(x0, x0_thread);",
            """
            load_affine8_values<T, float, values_per_thread>(x0, x0_thread);
                float sum0 = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m];
            """
        )
        replaceOnce(
            "float sum1 = load_vector<T, float, values_per_thread, 8>(x1, x1_thread);",
            """
            load_affine8_values<T, float, values_per_thread>(x1, x1_thread);
                float sum1 = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 1];
            """
        )

        let tripleLoads: [(String, String)] = [
            (
                "float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);",
                """
                load_affine8_values<T, float, values_per_thread>(x0, x_thread);
                float sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m];
                """),
            (
                "sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);",
                """
                load_affine8_values<T, float, values_per_thread>(x1, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 1];
                """),
            (
                "sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);",
                """
                load_affine8_values<T, float, values_per_thread>(x2, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 2];
                """),
        ]
        for (old, new) in tripleLoads {
            precondition(result.components(separatedBy: old).count == 3)
            result = result.replacingOccurrences(of: old, with: new)
        }
        return result
    }()

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_ipg3_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            constexpr int M = 8;
            constexpr int IPG = 3;
            constexpr int TAIL = M % IPG;
            const int first_m = int(tid.x) * IPG;
            if (first_m >= M) {
                return;
            }
            if (TAIL == 0 || M - first_m >= IPG) {
                qmv_affine8_g64_triple_stream_impl<T, 64, 8>(
                    w,
                    scales,
                    biases,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    x + (first_m + 2) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    y + (first_m + 2) * out_vec_size,
                    in_vec_size,
                    tid,
                    simd_gid,
                    simd_lid);
            } else {
                qmv_affine8_g64_pair_impl<T, 64, 8>(
                    w,
                    scales,
                    biases,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    in_vec_size,
                    tid,
                    simd_gid,
                    simd_lid);
            }
            return;
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// One exact float sum for each `(K/128 block, simd lane, cohort row)`.
    /// The expression is the bits==8 arm of stock `load_vector`: a float zero
    /// followed by four ascending `sum += bf16` operations. Row is the unit-
    /// stride dimension so one table serves both gate and up projections.
    private static let activationSumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_affine8_xsum_v1",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: """
            const uint lane = thread_position_in_grid.x;
            const uint k_block = thread_position_in_grid.y;
            const uint row = thread_position_in_grid.z;
            const int in_vec_size = x_shape[x_ndim - 1];
            const device T* xp =
                x + row * in_vec_size + k_block * 128 + lane * 4;
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                sum += xp[i];
            }
            xSums[(k_block * 32 + lane) * 8 + row] = sum;
            """,
        ensureRowContiguous: true
    )

    private static let activationSumQMVKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_ipg3_xsum_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            constexpr int M = 8;
            constexpr int IPG = 3;
            constexpr int TAIL = M % IPG;
            const int first_m = int(tid.x) * IPG;
            if (first_m >= M) {
                return;
            }
            if (TAIL == 0 || M - first_m >= IPG) {
                qmv_affine8_g64_triple_stream_xsum_impl<T, 64, 8>(
                    w,
                    scales,
                    biases,
                    xSums,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    x + (first_m + 2) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    y + (first_m + 2) * out_vec_size,
                    in_vec_size,
                    tid,
                    simd_gid,
                    simd_lid);
            } else {
                qmv_affine8_g64_pair_xsum_impl<T, 64, 8>(
                    w,
                    scales,
                    biases,
                    xSums,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    in_vec_size,
                    tid,
                    simd_gid,
                    simd_lid);
            }
            return;
            """,
        header: activationSumKernelHeader,
        ensureRowContiguous: true
    )

    @inline(__always)
    private static func liveShape(inDim: Int, outDim: Int) -> Bool {
        (inDim == 2816 && outDim == 2112)
            || (inDim == 2112 && outDim == 2816)
    }

    /// Builds the shared gate/up table only for the exact dense decode input.
    /// Returning an opaque value prevents callers from fabricating a table with
    /// a plausible shape; all other inputs keep DMLP-001 unchanged.
    public static func activationSums(for x: MLXArray) -> ActivationSums? {
        guard enabled,
            activationSumsEnabled,
            x.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) == batch,
            x.dim(1) == sequence,
            x.dim(2) == 2816,
            x.size == batch * sequence * 2816
        else { return nil }
        let blocks = 2816 / kBlock
        let values = activationSumKernel(
            [x],
            template: [("T", x.dtype)],
            grid: (simdWidth, blocks, batch),
            threadGroup: (simdWidth, 1, 1),
            outputShapes: [[blocks * simdWidth * batch]],
            outputDTypes: [.float32]
        )[0]
        return ActivationSums(values: values)
    }

    /// Returns `nil` unless every production pin holds. The caller then invokes
    /// the original QuantizedLinear unchanged.
    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        activationSums: ActivationSums? = nil
    ) -> MLXArray? {
        guard enabled,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let biases,
            x.dtype == .bfloat16,
            scales.dtype == x.dtype,
            biases.dtype == x.dtype,
            weight.dtype == .uint32,
            x.ndim == 3,
            x.dim(0) == batch,
            x.dim(1) == sequence
        else { return nil }

        let inDim = x.dim(2)
        guard weight.ndim == 2 else { return nil }
        let outDim = weight.dim(0)
        guard liveShape(inDim: inDim, outDim: outDim),
            inDim % Self.groupSize == 0,
            outDim % outputsPerGroup == 0,
            x.size == batch * sequence * inDim,
            weight.dim(1) == inDim * Self.bits / 32,
            scales.shape == [outDim, inDim / Self.groupSize],
            biases.shape == scales.shape
        else { return nil }

        // IPG3: 3 x-groups covering 3+3+2, not grid.x = batch/4.
        let xGroups = (batch + rowsPerGroup - 1) / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        let useActivationSums = activationSums != nil
            && inDim == 2816
            && outDim == 2112
        let selected = useActivationSums ? activationSumQMVKernel : kernel
        let inputs = useActivationSums
            ? [x, weight, scales, biases, activationSums!.values]
            : [x, weight, scales, biases]
        return selected(
            inputs,
            template: [("T", x.dtype)],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, outDim]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
