// Exact tight-grid affine4/g64 QMV for Gemma 4's B=8, L=1 attention output
// projection. Sliding layers are K=4096,N=2816; full layers K=8192,N=2816.
// Both enter MLX's `affine_qmv_fast` because K % 512 == 0.
//
// At N < 4096 the promoted fast kernel pairs two input rows per x-group, but
// the frozen host still launches M=8 x-groups. Groups 0...3 own rows 0...7;
// groups 4...7 return immediately. This replica launches only the four useful
// groups. Every x load, activation sum, qdot, K accumulation and simd_sum
// retains the `qmv_fast_crossrow_affine4_g64<T,8>` order.

import Foundation
import MLX
import MLXFast

public enum CBv2AttentionOQMVV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ATTN_O_QMV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// MMA-O8: route the same o_proj shapes through a matrix-unit kernel that
    /// serves all eight cohort rows from one weight fetch, exactly like the
    /// promoted `gemma4_qmv_mma8_affine4_g64_impl` Q/K/V tier. The tight-grid
    /// pair kernel below streams the weight plane four times (2 rows per
    /// x-group); this streams it once. Kill switch preserves the pair path.
    public static let mma8Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ATTN_O_MMA8"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let sequence = 1
    private static let outputWidth = 2816
    private static let groupSize = 64
    private static let bits = 4
    private static let rowsPerGroup = 2
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let outputsPerGroup = 8
    private static let kernelHeader = CBv2TiedLMHeadQMVV1.kernelHeader + """

inline float2 attention_o_qdot_affine4_loaded_pair(
    const thread uint16_t* ws,
    const thread float* x0,
    const thread float* x1,
    float scale,
    float bias,
    float2 sum) {
  float2 accum = 0;
  for (int i = 0; i < 4; i++) {
    accum +=
        (float2(x0[4 * i], x1[4 * i]) * (ws[i] & 0x000f) +
         float2(x0[4 * i + 1], x1[4 * i + 1]) * (ws[i] & 0x00f0) +
         float2(x0[4 * i + 2], x1[4 * i + 2]) * (ws[i] & 0x0f00) +
         float2(x0[4 * i + 3], x1[4 * i + 3]) * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <typename T>
METAL_FUNC void attention_o_qmv_fast_crossrow_affine4_g64_tight(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;

  const int first_m = int(tid.x) * 2;
  if (first_m >= 8) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  thread float2 pair_result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    pair_result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint16_t packed[rows_per_simd][4];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];

    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint8_t* wb =
          reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 2 + simd_lid * bytes_per_lane;
      const device uint16_t* ws =
          reinterpret_cast<const device uint16_t*>(wb);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index =
          row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    thread float x1[values_per_thread];
    const device T* xm0 =
        x + first_m * in_vec_size + k + simd_lid * values_per_thread;
    const device T* xm1 = xm0 + in_vec_size;
    const float sum0 =
        load_vector<T, float, values_per_thread, 4>(xm0, x0);
    const float sum1 =
        load_vector<T, float, values_per_thread, 4>(xm1, x1);
    for (int r = 0; r < rows_per_simd; r++) {
      pair_result[r] += attention_o_qdot_affine4_loaded_pair(
          packed[r], x0, x1, scale_local[r], bias_local[r],
          float2(sum0, sum1));
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced0 = simd_sum(pair_result[r].x);
    const float reduced1 = simd_sum(pair_result[r].y);
    if (simd_lid == 0) {
      y[first_m * out_vec_size + out_row + r] = static_cast<T>(reduced0);
      y[(first_m + 1) * out_vec_size + out_row + r] =
          static_cast<T>(reduced1);
    }
  }
}
"""

    /// Textual twin of the promoted Q/K/V tier's
    /// `gemma4_qmv_mma8_affine4_g64_impl<T, 2>` (mlx-generated/quantized.cpp),
    /// hosted as a tight-grid Swift kernel so o_proj (K % 512 == 0, which the
    /// frozen MLX host routes to `affine_qmv_fast`, away from the tier) can
    /// reach the same one-weight-fetch-serves-all-eight-rows arithmetic. Only
    /// the two C macros are joined to single lines; every load, lane
    /// assignment, MMA step, run-sum tree, and the KS=2 threadgroup close keep
    /// the donor's text, so the accumulation order is the tier's own.
    private static let mma8KernelHeader = """
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

struct mma8_coord {
  short fm;
  short fn;
};

inline mma8_coord mma8_lane(uint lane) {
  const short qid = short(lane / 4);
  return {
      short((qid & 4) + short((lane / 2) % 4)),
      short((qid & 2) * 2 + short(lane % 2) * 2)};
}

template <typename T, bool TWO_BYTE = (sizeof(T) == 2)>
struct mma8_u16 {
  static inline T cast(ushort u) {
    return T(0);
  }
};

template <typename T>
struct mma8_u16<T, true> {
  static inline T cast(ushort u) {
    return as_type<T>(u);
  }
};

template <typename T>
inline float mma8_lo(uint u) {
  return float(mma8_u16<T>::cast(ushort(u & 0xFFFFu)));
}

template <typename T>
inline float mma8_hi(uint u) {
  return float(mma8_u16<T>::cast(ushort(u >> 16)));
}

template <typename T>
inline float mma8_runsum4(uint4 r) {
  thread T xt[8];
  xt[0] = mma8_u16<T>::cast(ushort(r.x & 0xFFFFu));
  xt[1] = mma8_u16<T>::cast(ushort(r.x >> 16));
  xt[2] = mma8_u16<T>::cast(ushort(r.y & 0xFFFFu));
  xt[3] = mma8_u16<T>::cast(ushort(r.y >> 16));
  xt[4] = mma8_u16<T>::cast(ushort(r.z & 0xFFFFu));
  xt[5] = mma8_u16<T>::cast(ushort(r.z >> 16));
  xt[6] = mma8_u16<T>::cast(ushort(r.w & 0xFFFFu));
  xt[7] = mma8_u16<T>::cast(ushort(r.w >> 16));
  float sum = 0;
  sum += xt[0] + xt[1] + xt[2] + xt[3];
  sum += xt[4] + xt[5] + xt[6] + xt[7];
  return sum;
}

#define MMA8_SETB(BB, W, HI) BB.thread_elements()[0] = mma8_##HI<T>(r0.W); BB.thread_elements()[1] = mma8_##HI<T>(r1.W);

#define MMA8_STEP(BB, J) A.thread_elements()[0] = float(extract_bits(wv.x, 4 * (J), 4)); A.thread_elements()[1] = float(extract_bits(wv.y, 4 * (J), 4)); simdgroup_multiply_accumulate(C, A, BB, C);

template <typename T, int KS, int KFIX>
METAL_FUNC void attention_o_qmv_mma8_affine4_g64_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow =
      (const device uint8_t*)w + (n0 + c.fm) * (K / 2) + 4 * c.fn;
  const device T* srow = scales + (n0 + c.fm) * G;
  const device T* brow = biases + (n0 + c.fm) * G;
  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

  // ATTN-O-MMA8-CARRY2-023: two-stage weight operand register carry for the
  // attention output projection matrix-unit tier. Memory loads for packed
  // weight codes are scheduled two groups ahead (wv_next2), hiding device
  // memory latency while preserving exact arithmetic order and bitwise identity.
  uint2 wv_next = *((const device uint2*)(wrow + 32 * g0));
  uint2 wv_next2 =
      *((const device uint2*)(wrow + 32 * (g0 + min(1, nGroups - 1))));
  T s_next = srow[g0];
  T b_next = brow[g0];

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;
    const uint2 wv = wv_next;
    const float s = float(s_next);
    const float b = float(b_next);
    const int g_next = g0 + min(gi + 1, nGroups - 1);
    const int g_next2 = g0 + min(gi + 2, nGroups - 1);
    wv_next = wv_next2;
    wv_next2 = *((const device uint2*)(wrow + 32 * g_next2));
    s_next = srow[g_next];
    b_next = brow[g_next];
    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

    float2 rs = float2(mma8_runsum4<T>(r0), mma8_runsum4<T>(r1));
    rs += simd_shuffle_xor(rs, 2u);
    rs += simd_shuffle_xor(rs, 4u);
    rs += simd_shuffle_xor(rs, 16u);

    MMA8_SETB(B0, x, lo)
    MMA8_SETB(B1, x, hi)
    MMA8_SETB(B2, y, lo)
    MMA8_SETB(B3, y, hi)
    MMA8_SETB(B4, z, lo)
    MMA8_SETB(B5, z, hi)
    MMA8_SETB(B6, w, lo)
    MMA8_SETB(B7, w, hi)

    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    MMA8_STEP(B0, 0)
    MMA8_STEP(B1, 1)
    MMA8_STEP(B2, 2)
    MMA8_STEP(B3, 3)
    MMA8_STEP(B4, 4)
    MMA8_STEP(B5, 5)
    MMA8_STEP(B6, 6)
    MMA8_STEP(B7, 7)

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

    private static let mma8KernelK4096 = MLXFast.metalKernel(
        name: "cbv2_b8_l1_attention_o_mma8_affine4_g64_k4096_carry2_v4",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            attention_o_qmv_mma8_affine4_g64_impl<T, 2, 4096>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let mma8KernelK8192 = MLXFast.metalKernel(
        name: "cbv2_b8_l1_attention_o_mma8_affine4_g64_k8192_carry2_v4",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            attention_o_qmv_mma8_affine4_g64_impl<T, 2, 8192>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let qmvKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_attention_o_affine4_g64_tight_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            attention_o_qmv_fast_crossrow_affine4_g64_tight<T>(
                w, scales, biases, x, y,
                x_shape[x_ndim - 1], w_shape[0], tid,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: kernelHeader,
        ensureRowContiguous: true)

    @inline(__always)
    private static func liveInputWidth(_ width: Int) -> Bool {
        width == 4096 || width == 8192
    }

    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
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
            x.dim(1) == sequence,
            weight.ndim == 2
        else { return nil }

        let inDim = x.dim(2)
        guard liveInputWidth(inDim),
            x.size == batch * sequence * inDim,
            weight.shape == [outputWidth, inDim * Self.bits / 32],
            scales.shape == [outputWidth, inDim / Self.groupSize],
            biases.shape == scales.shape
        else { return nil }

        if mma8Enabled {
            // One threadgroup per 8-column output tile; all eight cohort rows
            // are served from a single weight fetch, so the o_proj plane is
            // streamed once per round instead of four times. Grid is in
            // threads: (32, 2, 1) threads per group, N/8 groups along y.
            let yTiles = outputWidth / outputsPerGroup
            let kernel = inDim == 8192 ? mma8KernelK8192 : mma8KernelK4096
            return kernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, yTiles * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, sequence, outputWidth]],
                outputDTypes: [x.dtype]
            )[0]
        }

        let xGroups = batch / rowsPerGroup
        let yGroups = outputWidth / outputsPerGroup
        return qmvKernel(
            [x, weight, scales, biases],
            template: [("T", x.dtype)],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, outputWidth]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
