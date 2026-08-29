// DMLP-001: tight-grid dispatch for Gemma 4's dense affine-8 MLP QMV at
// batch eight and decode length one.
//
// The vendored MLX host launches ordinary QMV with an x grid extent equal to
// the input-row count, M = 8. The promoted affine-8 wide-N tier then claims
// four rows per group:
//
//     first_m = threadgroup_position_in_grid.x * 4
//
// Only x-groups zero and one can work; groups two through seven return before
// any buffer access. This candidate owns the launch for exactly the two dense
// MLP geometries in the pinned Gemma tower and dispatches those same two useful
// groups directly. The y grid, threadgroup shape, row ownership, pointers,
// arithmetic, K loop, g64-aligned tail and simd reductions are unchanged.
//
// Across one batch-eight decode step the family has 60 gate/up projections at
// K=2816,N=2112 and 30 down projections at K=2112,N=2816. The stock grids
// submit 211,200 threadgroups; 158,400 of them are guaranteed early returns.
// The tight grids submit the 52,800 groups that already did all useful work.
//
// `DARKBLOOM_CBV2_DENSE_MLP_QMV=0` restores the stock QuantizedLinear path in
// the same executable. `DARKBLOOM_CBV2_DENSE_MLP_XSUM=0` keeps the tight grid
// but disables the activation-sum table. Every non-production dtype, mode,
// shape or quantization geometry fails closed to the prior path. Selected
// arrays are normalized to row-contiguous layout by `MLXFast.metalKernel`.
// `DARKBLOOM_CBV2_DENSE_MLP_MMA8=0` keeps the promoted tight-grid road while
// disabling the matrix-unit experiment layered above it.

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

    public static let matrixUnitEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DENSE_MLP_MMA8"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let sequence = 1
    private static let groupSize = 64
    private static let bits = 8
    private static let rowsPerGroup = 4
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

    /// The affine-8 helper below is the current promoted
    /// `qmv_affine8_g64_quad_stream_impl` body. It reuses the exact
    /// `load_vector` definition and Metal constants already carried by the
    /// promoted tied-head replica, then appends only the affine-8 registered
    /// dot product and quad-stream helper from `quantized.h`.
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

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_quad_stream_impl(
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
  thread float result3[results_per_simdgroup] = {0};

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
  x3 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;
  y3 += out_row;

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
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
    x3 += block_size;
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
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
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

    /// DMLP-MMA8-001: the promoted dense helper already owns the exact B8/L1
    /// affine8/group64 shapes. This alternative keeps those host guards but
    /// elects one threadgroup per eight output rows and computes all eight
    /// cohort columns with `simdgroup_float8x8`. Each g64 group is closed
    /// independently before the next group, matching the architecture of the
    /// officially promoted affine4 Q/K/V matrix-unit tier.
    private static let matrixUnitKernelHeader = kernelHeader + """

struct dmlp_mma8_coord {
  short fm;
  short fn;
};

inline dmlp_mma8_coord dmlp_mma8_lane(uint lane) {
  const short qid = short(lane / 4);
  return {
      short((qid & 4) + short((lane / 2) % 4)),
      short((qid & 2) * 2 + short(lane % 2) * 2)};
}

template <typename T>
inline float dmlp_mma8_lo(uint u) {
  return float(as_type<T>(ushort(u & 0xFFFFu)));
}

template <typename T>
inline float dmlp_mma8_hi(uint u) {
  return float(as_type<T>(ushort(u >> 16)));
}

template <typename T>
inline float dmlp_mma8_runsum4(uint4 r) {
  thread T xt[8];
  xt[0] = as_type<T>(ushort(r.x & 0xFFFFu));
  xt[1] = as_type<T>(ushort(r.x >> 16));
  xt[2] = as_type<T>(ushort(r.y & 0xFFFFu));
  xt[3] = as_type<T>(ushort(r.y >> 16));
  xt[4] = as_type<T>(ushort(r.z & 0xFFFFu));
  xt[5] = as_type<T>(ushort(r.z >> 16));
  xt[6] = as_type<T>(ushort(r.w & 0xFFFFu));
  xt[7] = as_type<T>(ushort(r.w >> 16));
  float sum = 0.0f;
  sum += xt[0] + xt[1] + xt[2] + xt[3];
  sum += xt[4] + xt[5] + xt[6] + xt[7];
  return sum;
}

template <typename T, int KS>
METAL_FUNC void dmlp_qmv_mma8_affine8_g64_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  const int G = K / 64;
  const int gh = (G + 1) / 2;
  const int g_begin = (KS == 2 && simd_gid == 1) ? gh : 0;
  const int g_end = (KS == 2 && simd_gid == 0) ? gh : G;
  const dmlp_mma8_coord c = dmlp_mma8_lane(simd_lid);

  const device uint8_t* wrow =
      (const device uint8_t*)w + (n0 + c.fm) * K + 8 * c.fn;
  const device T* srow = scales + (n0 + c.fm) * G;
  const device T* brow = biases + (n0 + c.fm) * G;
  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

  for (int g = g_begin; g < g_end; ++g) {
    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));
    float2 rs = float2(
        dmlp_mma8_runsum4<T>(r0), dmlp_mma8_runsum4<T>(r1));
    rs += simd_shuffle_xor(rs, 2u);
    rs += simd_shuffle_xor(rs, 4u);
    rs += simd_shuffle_xor(rs, 16u);

    B0.thread_elements()[0] = dmlp_mma8_lo<T>(r0.x);
    B0.thread_elements()[1] = dmlp_mma8_lo<T>(r1.x);
    B1.thread_elements()[0] = dmlp_mma8_hi<T>(r0.x);
    B1.thread_elements()[1] = dmlp_mma8_hi<T>(r1.x);
    B2.thread_elements()[0] = dmlp_mma8_lo<T>(r0.y);
    B2.thread_elements()[1] = dmlp_mma8_lo<T>(r1.y);
    B3.thread_elements()[0] = dmlp_mma8_hi<T>(r0.y);
    B3.thread_elements()[1] = dmlp_mma8_hi<T>(r1.y);
    B4.thread_elements()[0] = dmlp_mma8_lo<T>(r0.z);
    B4.thread_elements()[1] = dmlp_mma8_lo<T>(r1.z);
    B5.thread_elements()[0] = dmlp_mma8_hi<T>(r0.z);
    B5.thread_elements()[1] = dmlp_mma8_hi<T>(r1.z);
    B6.thread_elements()[0] = dmlp_mma8_lo<T>(r0.w);
    B6.thread_elements()[1] = dmlp_mma8_lo<T>(r1.w);
    B7.thread_elements()[0] = dmlp_mma8_hi<T>(r0.w);
    B7.thread_elements()[1] = dmlp_mma8_hi<T>(r1.w);

    const uint4 wv = *((const device uint4*)(wrow + 64 * g));
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    A.thread_elements()[0] = float(wv.x & 0xFFu);
    A.thread_elements()[1] = float(wv.z & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B0, C);
    A.thread_elements()[0] = float((wv.x >> 8) & 0xFFu);
    A.thread_elements()[1] = float((wv.z >> 8) & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B1, C);
    A.thread_elements()[0] = float((wv.x >> 16) & 0xFFu);
    A.thread_elements()[1] = float((wv.z >> 16) & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B2, C);
    A.thread_elements()[0] = float(wv.x >> 24);
    A.thread_elements()[1] = float(wv.z >> 24);
    simdgroup_multiply_accumulate(C, A, B3, C);
    A.thread_elements()[0] = float(wv.y & 0xFFu);
    A.thread_elements()[1] = float(wv.w & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B4, C);
    A.thread_elements()[0] = float((wv.y >> 8) & 0xFFu);
    A.thread_elements()[1] = float((wv.w >> 8) & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B5, C);
    A.thread_elements()[0] = float((wv.y >> 16) & 0xFFu);
    A.thread_elements()[1] = float((wv.w >> 16) & 0xFFu);
    simdgroup_multiply_accumulate(C, A, B6, C);
    A.thread_elements()[0] = float(wv.y >> 24);
    A.thread_elements()[1] = float(wv.w >> 24);
    simdgroup_multiply_accumulate(C, A, B7, C);

    acc0 += s * C.thread_elements()[0] + rs.x * b;
    acc1 += s * C.thread_elements()[1] + rs.y * b;
  }

  if (KS == 2) {
    if (simd_gid == 1) {
      red[simd_lid] = float2(acc0, acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
    const float2 other = red[simd_lid];
    acc0 = acc0 + other.x;
    acc1 = acc1 + other.y;
  }

  y[c.fn * N + n0 + c.fm] = static_cast<T>(acc0);
  y[(c.fn + 1) * N + n0 + c.fm] = static_cast<T>(acc1);
}

"""

    /// DMLP-002 keeps DMLP-001's kernel text intact and derives a second body
    /// which replaces only the four-value activation sum. The x values are
    /// still loaded into the same register array, in the same order, and every
    /// qdot, K accumulation and simd reduction remains text-for-text identical.
    private static let activationSumKernelHeader: String = {
        var result = kernelHeader

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        func replaceLast(_ old: String, with new: String) {
            guard let range = result.range(of: old, options: .backwards) else {
                preconditionFailure("DMLP-002 kernel transform marker is missing")
            }
            result.replaceSubrange(range, with: new)
        }

        // The tied-head prefix also contains a `(biases, x0)` signature and
        // the same out-row formula. DMLP's affine8 helper is appended after
        // that prefix, so replace the final occurrence in each case.
        replaceLast(
            """
            template <typename T, const int group_size, const int bits>
            METAL_FUNC void qmv_affine8_g64_quad_stream_impl(
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
            METAL_FUNC void qmv_affine8_g64_quad_stream_xsum_impl(
            """
        )
        replaceLast(
            """
                const device T* biases,
                const device T* x0,
            """,
            with: """
                const device T* biases,
                const device float* x_sums,
                const device T* x0,
            """
        )
        replaceLast(
            """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
            """,
            with: """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
              const int first_m = int(tid.x) * 4;
            """
        )

        let loads: [(String, String)] = [
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
            (
                "sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);",
                """
                load_affine8_values<T, float, values_per_thread>(x3, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 3];
                """),
        ]
        for (old, new) in loads {
            precondition(result.components(separatedBy: old).count == 3)
            result = result.replacingOccurrences(of: old, with: new)
        }
        return result
    }()

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int first_m = int(tid.x) * 4;
            if (first_m >= 8) {
                return;
            }
            qmv_affine8_g64_quad_stream_impl<T, 64, 8>(
                w,
                scales,
                biases,
                x + first_m * in_vec_size,
                x + (first_m + 1) * in_vec_size,
                x + (first_m + 2) * in_vec_size,
                x + (first_m + 3) * in_vec_size,
                y + first_m * out_vec_size,
                y + (first_m + 1) * out_vec_size,
                y + (first_m + 2) * out_vec_size,
                y + (first_m + 3) * out_vec_size,
                in_vec_size,
                tid,
                simd_gid,
                simd_lid);
            return;
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    private static let matrixUnitKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_mma8_affine8_g64_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int n0 = int(tid.y) * 8;
            threadgroup float2 red[32];
            dmlp_qmv_mma8_affine8_g64_impl<T, 2>(
                w, scales, biases, x, y,
                in_vec_size, out_vec_size, n0, red, simd_gid, simd_lid);
            return;
            """,
        header: matrixUnitKernelHeader,
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
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_xsum_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int first_m = int(tid.x) * 4;
            if (first_m >= 8) {
                return;
            }
            qmv_affine8_g64_quad_stream_xsum_impl<T, 64, 8>(
                w,
                scales,
                biases,
                xSums,
                x + first_m * in_vec_size,
                x + (first_m + 1) * in_vec_size,
                x + (first_m + 2) * in_vec_size,
                x + (first_m + 3) * in_vec_size,
                y + first_m * out_vec_size,
                y + (first_m + 1) * out_vec_size,
                y + (first_m + 2) * out_vec_size,
                y + (first_m + 3) * out_vec_size,
                in_vec_size,
                tid,
                simd_gid,
                simd_lid);
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
            !matrixUnitEnabled,
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

        let xGroups = batch / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        if matrixUnitEnabled {
            return matrixUnitKernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, yGroups * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, sequence, outDim]],
                outputDTypes: [x.dtype]
            )[0]
        }
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
