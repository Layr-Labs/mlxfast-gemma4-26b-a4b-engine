// LMH-001v2: tight-grid dispatch for the tied lm_head ordinary QMV at batch
// eight, carrying the promoted quad-STREAM body verbatim.
//
// The vendored MLX host launches ordinary QMV as
//
//     MTL::Size group_dims(bk, 2, 1);                     // (32, 2, 1)
//     MTL::Size grid_dims(M, (N + bn - 1) / bn, B);       // x = M
//     compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
//
// (`backend/metal/quantized.cpp`), so the x extent of the grid is the cohort
// row count, eight. The promoted wide-N tier in `quantized.h`
// (`qmv_affine4_g64_quad_stream_impl`) claims four cohort rows per threadgroup
// and returns from the rest:
//
//     const int first_m = int(tid.x) * 4;
//     if (first_m >= 8) { return; }
//
// Two x-groups do the work; six are launched and retire immediately. On the
// tied lm_head that is the largest grid of the decode step -- N = 262144 gives
// N / 8 = 32768 y-groups, so 8 * 32768 = 262144 threadgroups are launched and
// 196608 of them exist only to hit that early return.
//
// The host grid is not editable. This file instead dispatches the same
// computation from a custom kernel whose own grid has x extent two, so only
// the groups that were already doing the work are launched. `first_m` is kept
// as `tid.x * 4`, so with tid.x in {0, 1} the two surviving groups claim rows
// 0-3 and 4-7 exactly as before: the same threadgroup does the same rows with
// the same pointers.
//
// LINEAGE. The tight-grid dispatch is josusanmartin's LMH-001 (submission
// `2de922a4`; its isolated ancestor holds an official box receipt of +0.216%
// composite / +0.383% decode on the `3ab9bd4` parent, fidelity passed). That
// original embedded the then-promoted register-resident quad body. The current
// promoted tier is the register-STREAMED quad (`49e4becd`), measured faster
// than the retired quad on the box precisely because only one row of x is
// live at a time; shipping the old body at two groups would trade the grid
// saving back for the slower body. The header below therefore carries the
// PROMOTED helpers VERBATIM from `quantized.h` -- `load_vector`,
// `load_vector_safe`, `qdot_affine4_registered`, and
// `qmv_affine4_g64_quad_stream_impl` -- and the kernel body reproduces the
// stock `[[kernel]] affine_qmv` wide-N call block, so the compiler sees the
// same code shape the JIT library compiles and makes the same contraction
// decisions. (A hand-inlined expansion of the same arithmetic measured 1-4
// ulp off the library kernel under this compiler; verbatim inclusion is what
// the prior parity receipts on this track used, and the local runtime parity
// test pins it bitwise.) Only the grid differs from the stock road.
//
// LMH-002 (this submission). The header's short walks whose trip counts are
// already compile-time constants -- the affine-4 registered dot's lane walk
// (values_per_thread / 4), load_vector's fixed-count value walk, the
// four-row cohort walks (results_per_simdgroup), the packed uint16 walk
// (uint16_per_thread), and the final four-row SIMD reduction -- now carry
// `#pragma clang loop unroll(full)`. The K traversal is NOT annotated: its
// trip count comes from the live input shape. No address, predicate,
// accumulation order, or arithmetic expression is changed, so the bitwise
// parity this file pins is unaffected; only loop-control codegen differs.
// The kernel name is versioned to `..._unroll_v3` so the named MLX kernel
// cache cannot satisfy the new request with the previously compiled body.
// Mechanism inherited from fkiene's promoted `22154b54` (DenseMLPQMVV1).
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

    static let kernelHeader = """
#define METAL_FUNC inline
constant constexpr const int SIMD_SIZE = 32;


template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread>
inline U qdot_affine4_registered(
    const thread uint16_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  #pragma clang loop unroll(full)
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (w[i] & 0x000f) +
         x_thread[4 * i + 1] * (w[i] & 0x00f0) +
         x_thread[4 * i + 2] * (w[i] & 0x0f00) +
         x_thread[4 * i + 3] * (w[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_quad_stream_impl(
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
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
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
  for (; k < in_vec_size - block_size; k += block_size) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 4>(x0, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x1, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x2, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x3, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
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
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum =
        load_vector_safe<T, float, values_per_thread, 4>(x0, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x1, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x2, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x3, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  #pragma clang loop unroll(full)
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

    // MTP-PV: the sixteen-row verify additions live in their OWN header
    // string so the shared `kernelHeader` stays byte-identical —
    // `CBv2DenseMLPQMVV1.activationSumKernelHeader` splices that string with
    // `replaceOnce` anchors asserted to occur exactly once, and any impl
    // text appended there trips those preconditions at one-time init
    // (root-caused from a live signal-5 crash report; the oct-r1 smoke's
    // four identical candidate-leg deaths were this, not the kernels).
    static let verifier16KernelHeader = kernelHeader + """

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

// MTP-PV OCT-STREAM (4-bit): the promoted quad-stream body extended to
// EIGHT streamed activation rows against one register-held weight packet.
// Each row's load/qdot/accumulate/simd_sum/cast sequence is the quad body's
// verbatim (rows grouped 0-3 / 4-7 exactly as two quad calls would group
// them), so per-row arithmetic is bit-identical to the certified M8 launch;
// only weight-packet OWNERSHIP widens from four rows to eight. Registers:
// one x row live at a time (streamed), 32 accumulators.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_oct_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    const device T* x4,
    const device T* x5,
    const device T* x6,
    const device T* x7,
    device T* y0,
    device T* y1,
    device T* y2,
    device T* y3,
    device T* y4,
    device T* y5,
    device T* y6,
    device T* y7,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
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
  thread float result4[results_per_simdgroup] = {0};
  thread float result5[results_per_simdgroup] = {0};
  thread float result6[results_per_simdgroup] = {0};
  thread float result7[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
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
  x4 += simd_lid * values_per_thread;
  x5 += simd_lid * values_per_thread;
  x6 += simd_lid * values_per_thread;
  x7 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;
  y3 += out_row;
  y4 += out_row;
  y5 += out_row;
  y6 += out_row;
  y7 += out_row;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 4>(x0, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x1, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x2, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x3, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x4, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result4[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x5, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result5[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x6, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result6[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x7, x_thread);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result7[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
    x3 += block_size;
    x4 += block_size;
    x5 += block_size;
    x6 += block_size;
    x7 += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum =
        load_vector_safe<T, float, values_per_thread, 4>(x0, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x1, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x2, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x3, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x4, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result4[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x5, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result5[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x6, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result6[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x7, x_thread, remaining);
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result7[row] += qdot_affine4_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  #pragma clang loop unroll(full)
  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    result3[row] = simd_sum(result3[row]);
    result4[row] = simd_sum(result4[row]);
    result5[row] = simd_sum(result5[row]);
    result6[row] = simd_sum(result6[row]);
    result7[row] = simd_sum(result7[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
      y3[row] = static_cast<T>(result3[row]);
      y4[row] = static_cast<T>(result4[row]);
      y5[row] = static_cast<T>(result5[row]);
      y6[row] = static_cast<T>(result6[row]);
      y7[row] = static_cast<T>(result7[row]);
    }
  }
}

// MTP-PV HEXA-STREAM (4-bit): all sixteen activation rows streamed against
// one register-held weight packet of TWO output rows, so a dispatch reads
// every weight byte exactly once (the oct form reads them twice — once per
// row-octet threadgroup on the x axis). Per-row arithmetic and per-quad
// grouping order are unchanged: rows walk 0..15 in the same qdot chains,
// accumulators stay one float per (x row, out row). 32 accumulators, one
// x row live at a time.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_hexa16_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int stream_rows = 16;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int uint16_per_thread = bytes_per_thread / 2;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread uint16_t packed[results_per_simdgroup][uint16_per_thread];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result[stream_rows][results_per_simdgroup] = {{0}};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xl = x + simd_lid * values_per_thread;
  y += out_row;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < stream_rows; m++) {
      const float sum = load_vector<T, float, values_per_thread, 4>(
          xl + m * in_vec_size, x_thread);
      #pragma clang loop unroll(full)
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine4_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    xl += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * in_vec_size_w);
      #pragma clang loop unroll(full)
      for (int i = 0; i < uint16_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < stream_rows; m++) {
      const float sum = load_vector_safe<T, float, values_per_thread, 4>(
          xl + m * in_vec_size, x_thread, remaining);
      #pragma clang loop unroll(full)
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine4_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }
  }

  #pragma clang loop unroll(full)
  for (int m = 0; m < stream_rows; m++) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[m][row] = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
      }
    }
  }
}

// MTP-PV HEXA-STREAM (8-bit): the same single-weight-read sixteen-row form
// over the affine-8 body constants.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_hexa16_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int stream_rows = 16;
  constexpr int values_per_thread = 4;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread uint8_t packed[results_per_simdgroup][bytes_per_thread];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result[stream_rows][results_per_simdgroup] = {{0}};

  const int in_vec_size_w = in_vec_size;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xl = x + simd_lid * values_per_thread;
  y += out_row;

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

    for (int m = 0; m < stream_rows; m++) {
      const float sum = load_vector<T, float, values_per_thread, 8>(
          xl + m * in_vec_size, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine8_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    xl += block_size;
  }

  const uint active_tail_lanes = uint((in_vec_size - k) / values_per_thread);
  if (simd_lid < active_tail_lanes) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    for (int m = 0; m < stream_rows; m++) {
      const float sum = load_vector<T, float, values_per_thread, 8>(
          xl + m * in_vec_size, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine8_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }
  }

  for (int m = 0; m < stream_rows; m++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[m][row] = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
      }
    }
  }
}

// MTP-PV OCT-STREAM (8-bit): the same eight-stream extension over the
// affine-8 quad-stream body (values_per_thread 4, byte-packed weights).
// Serves the thirty layers' eight-bit promoted projections
// (mlp.{down,gate,up}, router) on the verify plane.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_oct_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    const device T* x4,
    const device T* x5,
    const device T* x6,
    const device T* x7,
    device T* y0,
    device T* y1,
    device T* y2,
    device T* y3,
    device T* y4,
    device T* y5,
    device T* y6,
    device T* y7,
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
  thread float result4[results_per_simdgroup] = {0};
  thread float result5[results_per_simdgroup] = {0};
  thread float result6[results_per_simdgroup] = {0};
  thread float result7[results_per_simdgroup] = {0};

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
  x4 += simd_lid * values_per_thread;
  x5 += simd_lid * values_per_thread;
  x6 += simd_lid * values_per_thread;
  x7 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;
  y3 += out_row;
  y4 += out_row;
  y5 += out_row;
  y6 += out_row;
  y7 += out_row;

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
    sum = load_vector<T, float, values_per_thread, 8>(x4, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result4[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x5, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result5[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x6, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result6[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x7, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result7[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
    x3 += block_size;
    x4 += block_size;
    x5 += block_size;
    x6 += block_size;
    x7 += block_size;
  }

  const uint active_tail_lanes = uint((in_vec_size - k) / values_per_thread);
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
    sum = load_vector<T, float, values_per_thread, 8>(x4, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result4[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x5, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result5[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x6, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result6[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x7, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result7[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    result3[row] = simd_sum(result3[row]);
    result4[row] = simd_sum(result4[row]);
    result5[row] = simd_sum(result5[row]);
    result6[row] = simd_sum(result6[row]);
    result7[row] = simd_sum(result7[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
      y3[row] = static_cast<T>(result3[row]);
      y4[row] = static_cast<T>(result4[row]);
      y5[row] = static_cast<T>(result5[row]);
      y6[row] = static_cast<T>(result6[row]);
      y7[row] = static_cast<T>(result7[row]);
    }
  }
}
"""

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_quad_stream_unroll_v3",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = K;
            const int out_vec_size = OUTN;

            const int first_m = int(tid.x) * 4;
            if (first_m >= 8) {
                return;
            }
            qmv_affine4_g64_quad_stream_impl<T, 64, 4>(
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

        return matmulM8(
            x: x, weight: weight, scales: scales, biases: biases,
            inDim: inDim, outDim: outDim)
    }

    /// The certified `[8, 1, K]` head launch, extracted verbatim from
    /// `matmul` so the fixed verifier can reuse the identical arithmetic.
    private static func matmulM8(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        inDim: Int,
        outDim: Int
    ) -> MLXArray {
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

    /// Fixed B8 verifier head. Each target column retains the certified M8
    /// arithmetic; only the number of independently scored columns varies.
    /// Callers must install the route only while `enabled` holds — the kill
    /// switch is honored at verifier installation, not per call.
    public static func matmulVerifierColumnsM8(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        inDim: Int,
        outDim: Int
    ) -> MLXArray {
        precondition(enabled, "verifier head installed with the tied-head kernel disabled")
        precondition(x.dim(0) == batch && x.dim(1) > 0 && x.dim(2) == inDim)
        func column(_ index: Int) -> MLXArray {
            let input = x[0..., index ..< (index + 1), 0...]
            return matmulM8(
                x: input, weight: weight, scales: scales, biases: biases,
                inDim: inDim, outDim: outDim)
        }

        return concatenated((0..<x.dim(1)).map(column), axis: 1)
    }

    // MARK: - MTP-PV OCT-STREAM sixteen-row verify launch

    private static let octKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_mtpverify16_qmv_affine_g64_oct_stream_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = K;
            const int out_vec_size = OUTN;

            const int first_m = int(tid.x) * 8;
            if (first_m >= 16) {
                return;
            }
            if (BITS == 4) {
                qmv_affine4_g64_oct_stream_impl<T, 64, 4>(
                    w, scales, biases,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    x + (first_m + 2) * in_vec_size,
                    x + (first_m + 3) * in_vec_size,
                    x + (first_m + 4) * in_vec_size,
                    x + (first_m + 5) * in_vec_size,
                    x + (first_m + 6) * in_vec_size,
                    x + (first_m + 7) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    y + (first_m + 2) * out_vec_size,
                    y + (first_m + 3) * out_vec_size,
                    y + (first_m + 4) * out_vec_size,
                    y + (first_m + 5) * out_vec_size,
                    y + (first_m + 6) * out_vec_size,
                    y + (first_m + 7) * out_vec_size,
                    in_vec_size, tid, simd_gid, simd_lid);
            } else {
                qmv_affine8_g64_oct_stream_impl<T, 64, 8>(
                    w, scales, biases,
                    x + first_m * in_vec_size,
                    x + (first_m + 1) * in_vec_size,
                    x + (first_m + 2) * in_vec_size,
                    x + (first_m + 3) * in_vec_size,
                    x + (first_m + 4) * in_vec_size,
                    x + (first_m + 5) * in_vec_size,
                    x + (first_m + 6) * in_vec_size,
                    x + (first_m + 7) * in_vec_size,
                    y + first_m * out_vec_size,
                    y + (first_m + 1) * out_vec_size,
                    y + (first_m + 2) * out_vec_size,
                    y + (first_m + 3) * out_vec_size,
                    y + (first_m + 4) * out_vec_size,
                    y + (first_m + 5) * out_vec_size,
                    y + (first_m + 6) * out_vec_size,
                    y + (first_m + 7) * out_vec_size,
                    in_vec_size, tid, simd_gid, simd_lid);
            }
            return;
            """,
        header: verifier16KernelHeader,
        ensureRowContiguous: true
    )

    private static let hexaKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_mtpverify16_qmv_affine_g64_hexa16_stream_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            if (BITS == 4) {
                qmv_affine4_g64_hexa16_stream_impl<T, 64, 4>(
                    w, scales, biases, x, y, K, OUTN, tid, simd_gid, simd_lid);
            } else {
                qmv_affine8_g64_hexa16_stream_impl<T, 64, 8>(
                    w, scales, biases, x, y, K, OUTN, tid, simd_gid, simd_lid);
            }
            return;
            """,
        header: verifier16KernelHeader,
        ensureRowContiguous: true
    )

    /// `DARKBLOOM_GEMMA4_MTP_VERIFY_HEXA=0` keeps the sixteen-row verify on
    /// the oct-stream form (weights read twice per call) instead of the
    /// hexa16 form (weights read once). Default ON.
    public static let verifierHexaEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MTP_VERIFY_HEXA"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Sixteen-row stream QMV master switch — DEFAULT OFF, measured
    /// negative: on M4 the stock road's [8, K] calls take the simdgroup-
    /// matrix QMM path, which already reads weights once per eight-row call
    /// and outruns the scalar stream bodies (paired A/B receipts: stock
    /// 0.8496, oct 0.7115, hexa 0.7740 on the same baseline). Kept for
    /// M5-side re-evaluation via `DARKBLOOM_GEMMA4_MTP_VERIFY16=1`.
    public static let verifier16Enabled: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment[
                "DARKBLOOM_GEMMA4_MTP_VERIFY16"] ?? "")
                .lowercased())
    }()

    /// Sixteen-row verify-plane QMV. Hexa16 form: ONE dispatch reading every
    /// weight byte exactly once for all sixteen rows. Oct form (fallback or
    /// kill switch): weights read once per row-octet (twice per call). Both
    /// keep the promoted stream bodies' per-row arithmetic; the per-column
    /// road read weights four times per verify round. Returns nil unless
    /// every pin holds; callers keep their established road then.
    public static func matmulVerifier16(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        inDim: Int,
        outDim: Int,
        bits: Int
    ) -> MLXArray? {
        guard enabled,
            bits == 4 || bits == 8,
            x.dtype == .bfloat16,
            scales.dtype == x.dtype,
            biases.dtype == x.dtype,
            weight.dtype == .uint32,
            x.ndim == 2,
            x.dim(0) == 16,
            x.dim(1) == inDim,
            outDim >= outputsPerGroup,
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

        if verifierHexaEnabled, outDim % 4 == 0 {
            // hexa16: 2 output rows per simdgroup -> outDim/4 threadgroups.
            let yGroupsHexa = outDim / 4
            return hexaKernel(
                [x, weight, scales, biases],
                template: [
                    ("T", x.dtype),
                    ("K", inDim),
                    ("OUTN", outDim),
                    ("BITS", bits),
                ],
                grid: (simdWidth, yGroupsHexa * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[16, outDim]],
                outputDTypes: [x.dtype]
            )[0]
        }
        let yGroups = outDim / outputsPerGroup
        return octKernel(
            [x, weight, scales, biases],
            template: [
                ("T", x.dtype),
                ("K", inDim),
                ("OUTN", outDim),
                ("BITS", bits),
            ],
            grid: (2 * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[16, outDim]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
