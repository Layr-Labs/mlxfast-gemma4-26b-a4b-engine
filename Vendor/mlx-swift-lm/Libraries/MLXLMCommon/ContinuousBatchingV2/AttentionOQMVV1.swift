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
  float2 accum =
      (float2(x0[0], x1[0]) * (ws[0] & 0x000f) +
       float2(x0[1], x1[1]) * (ws[0] & 0x00f0) +
       float2(x0[2], x1[2]) * (ws[0] & 0x0f00) +
       float2(x0[3], x1[3]) * (ws[0] & 0xf000)) +
      (float2(x0[4], x1[4]) * (ws[1] & 0x000f) +
       float2(x0[5], x1[5]) * (ws[1] & 0x00f0) +
       float2(x0[6], x1[6]) * (ws[1] & 0x0f00) +
       float2(x0[7], x1[7]) * (ws[1] & 0xf000)) +
      (float2(x0[8], x1[8]) * (ws[2] & 0x000f) +
       float2(x0[9], x1[9]) * (ws[2] & 0x00f0) +
       float2(x0[10], x1[10]) * (ws[2] & 0x0f00) +
       float2(x0[11], x1[11]) * (ws[2] & 0xf000)) +
      (float2(x0[12], x1[12]) * (ws[3] & 0x000f) +
       float2(x0[13], x1[13]) * (ws[3] & 0x00f0) +
       float2(x0[14], x1[14]) * (ws[3] & 0x0f00) +
       float2(x0[15], x1[15]) * (ws[3] & 0xf000));
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
