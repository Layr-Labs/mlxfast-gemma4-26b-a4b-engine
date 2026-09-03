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

    /// Packed-word loads are aligned by the fixed 4-byte lane stride.
    private static let packed32Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_TIED_LMHEAD_PACKED32"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// EXP-008: Apple matrix coprocessor MMA-8 acceleration for the tied LM head.
    /// Uses simdgroup_float8x8 matrix multiplication units (16-column tiles, TILES=2).
    public static let mma8Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_TIED_LMHEAD_MMA8"]
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
    private static let tilesPerGroup = 2
    private static let mmaOutputsPerGroup = 8 * tilesPerGroup // 16
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

    private static let packed32KernelHeader: String = {
        var result = kernelHeader
        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }
        replaceOnce(
            """
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
            """,
            with: """
            template <typename U, int values_per_thread>
            inline U qdot_affine4_registered_word(
                uint packed_word,
                const thread U* x_thread,
                U scale,
                U bias,
                U sum) {
              static_assert(values_per_thread == 8, "Word load expects eight 4-bit values");
              const uint packed0 = packed_word & 0xffffu;
              const uint packed1 = packed_word >> 16;
              U accum =
                  (x_thread[0] * (packed0 & 0x000f) +
                   x_thread[1] * (packed0 & 0x00f0) +
                   x_thread[2] * (packed0 & 0x0f00) +
                   x_thread[3] * (packed0 & 0xf000));
              accum +=
                  (x_thread[4] * (packed1 & 0x000f) +
                   x_thread[5] * (packed1 & 0x00f0) +
                   x_thread[6] * (packed1 & 0x0f00) +
                   x_thread[7] * (packed1 & 0xf000));
              return scale * accum + sum * bias;
            }
            """
        )
        replaceOnce(
            "thread uint16_t packed[results_per_simdgroup][uint16_per_thread];",
            with: "thread uint packed[results_per_simdgroup];"
        )
        let loadBlock = """
              const device uint16_t* wl =
                  (const device uint16_t*)(ws + row * in_vec_size_w);
              #pragma clang loop unroll(full)
              for (int i = 0; i < uint16_per_thread; i++) {
                packed[row][i] = wl[i];
              }
            """
        precondition(result.components(separatedBy: loadBlock).count == 3)
        result = result.replacingOccurrences(
            of: loadBlock,
            with: """
              const device uint* wl =
                  (const device uint*)(ws + row * in_vec_size_w);
              packed[row] = *wl;
            """
        )
        result = result.replacingOccurrences(
            of: "qdot_affine4_registered<float, values_per_thread>(",
            with: "qdot_affine4_registered_word<float, values_per_thread>(")
        return result
    }()

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

    template <typename T, int KS, int TILES, int KFIX>
    METAL_FUNC void tied_lmhead_mma8_affine4_g64_mt(
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

      const device uint8_t* wrow[TILES];
      const device T* srow[TILES];
      const device T* brow[TILES];
      thread float acc0[TILES];
      thread float acc1[TILES];
    #pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        const int nt = n0 + t * 8;
        wrow[t] = (const device uint8_t*)w + (nt + c.fm) * (K / 2) + 4 * c.fn;
        srow[t] = scales + (nt + c.fm) * G;
        brow[t] = biases + (nt + c.fm) * G;
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
      }

      const device T* x0 = x + c.fn * K + 8 * c.fm;
      const device T* x1 = x0 + K;

      simdgroup_float8x8 A;
      simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

      uint2 wv_next[TILES];
      uint2 wv_next2[TILES];
      T s_next[TILES];
      T b_next[TILES];
    #pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        wv_next[t] = *((const device uint2*)(wrow[t] + 32 * g0));
        wv_next2[t] =
            *((const device uint2*)(wrow[t] + 32 * (g0 + min(1, nGroups - 1))));
        s_next[t] = srow[t][g0];
        b_next[t] = brow[t][g0];
      }

    #pragma unroll
      for (int gi = 0; gi < nGroups; ++gi) {
        const int g = g0 + gi;

        uint2 wv_cur[TILES];
        float s_cur[TILES];
        float b_cur[TILES];
    #pragma clang loop unroll(full)
        for (int t = 0; t < TILES; ++t) {
          wv_cur[t] = wv_next[t];
          s_cur[t] = float(s_next[t]);
          b_cur[t] = float(b_next[t]);
        }
        const int g_next = g0 + min(gi + 1, nGroups - 1);
        const int g_next2 = g0 + min(gi + 2, nGroups - 1);
    #pragma clang loop unroll(full)
        for (int t = 0; t < TILES; ++t) {
          wv_next[t] = wv_next2[t];
          wv_next2[t] = *((const device uint2*)(wrow[t] + 32 * g_next2));
          s_next[t] = srow[t][g_next];
          b_next[t] = brow[t][g_next];
        }

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

    #pragma clang loop unroll(full)
        for (int t = 0; t < TILES; ++t) {
          const uint2 wv = wv_cur[t];
          const float s = s_cur[t];
          const float b = b_cur[t];

          simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
          MMA8_STEP(B0, 0)
          MMA8_STEP(B1, 1)
          MMA8_STEP(B2, 2)
          MMA8_STEP(B3, 3)
          MMA8_STEP(B4, 4)
          MMA8_STEP(B5, 5)
          MMA8_STEP(B6, 6)
          MMA8_STEP(B7, 7)

          acc0[t] += s * C.thread_elements()[0] + rs.x * b;
          acc1[t] += s * C.thread_elements()[1] + rs.y * b;
        }
      }

      if (KS == 2) {
        if (simd_gid == 1) {
    #pragma clang loop unroll(full)
          for (int t = 0; t < TILES; ++t) {
            red[t * 32 + simd_lid] = float2(acc0[t], acc1[t]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 1) {
          return;
        }
    #pragma clang loop unroll(full)
        for (int t = 0; t < TILES; ++t) {
          const float2 other = red[t * 32 + simd_lid];
          acc0[t] = acc0[t] + other.x;
          acc1[t] = acc1[t] + other.y;
        }
      }

    #pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        const int nt = n0 + t * 8;
        y[c.fn * N + nt + c.fm] = static_cast<T>(acc0[t]);
        y[(c.fn + 1) * N + nt + c.fm] = static_cast<T>(acc1[t]);
      }
    }
    """

    private static let mma8Kernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_mma8_affine4_g64_mt2_k2816_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            tied_lmhead_mma8_affine4_g64_mt<T, 2, 2, 2816>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true
    )

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

    private static let packed32Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_quad_stream_packed32_v1",
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
        header: packed32KernelHeader,
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

        if mma8Enabled, inDim == 2816, outDim == 262144 {
            let yTiles = outDim / mmaOutputsPerGroup // 262144 / 16 = 16384
            CBv2EngageMark.once("tied-lmhead-mma8-mt2")
            return mma8Kernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, yTiles * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, 1, outDim]],
                outputDTypes: [x.dtype]
            )[0]
        }

        let xGroups = batch / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        let selectedKernel: MLXFast.MLXFastKernel
        if packed32Enabled {
            CBv2EngageMark.once("tied-lmhead-packed32")
            selectedKernel = packed32Kernel
        } else {
            selectedKernel = kernel
        }
        return selectedKernel(
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
