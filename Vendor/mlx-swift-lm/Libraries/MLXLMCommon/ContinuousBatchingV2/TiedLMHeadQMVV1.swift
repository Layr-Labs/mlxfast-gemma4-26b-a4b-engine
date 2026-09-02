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
//
// LMH-OCTO-001 (this submission). The promoted quad body claims FOUR cohort
// rows per threadgroup, so the batch-eight cohort needs two x-groups and each
// of them walks the SAME tied-head rows: for one `out_row` quad, the packed
// weight uint16 pair, the group scale and the group bias are loaded once by
// the group that owns cohort rows 0-3 and loaded a second time by the group
// that owns rows 4-7. On the tied lm_head that duplicated stream is the widest
// one in the decode step -- 262144 output rows against a 2560-wide input --
// and it buys nothing: the eight cohort rows share one right-hand side.
//
// The eight-stream body below claims all eight rows in one threadgroup. The
// weight / scale / bias staging block is unchanged and now feeds eight
// activation streams instead of four, so per K block the same sixteen device
// loads serve twice the arithmetic and the x grid extent falls from two
// threadgroups to one (32768 threadgroups instead of 65536, none of them
// retiring on an early return).
//
// Exactness. `out_row` is derived from `tid.y` and `simd_gid` exactly as
// before, so a given threadgroup still owns the same four output rows and the
// same pointers. Each `result<n>[row]` accumulates the identical terms in the
// identical K order through the identical `qdot_affine4_registered` call with
// the identical `packed` / `scale_local` / `bias_local` / `sum` operands; the
// per-row `load_vector` sums, the runtime-bounded `load_vector_safe` tail, the
// `simd_sum` reduction and the store positions are untouched. No address,
// predicate, accumulation order or arithmetic expression differs from the quad
// body, so the emitted logits are bitwise equal to the promoted path's.
//
// `kernelHeader` is left byte-identical because `AttentionOQMVV1` and
// `DenseMLPQMVV1` splice it; the new body lives in a file-local header string
// and a separately named kernel, so neither sibling's compiled source moves.
//
// `DARKBLOOM_CBV2_TIED_LMHEAD_OCTO=0` restores the four-row quad dispatch, and
// `DARKBLOOM_CBV2_TIED_LMHEAD_QMV=0` still restores the stock host road.

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

    /// LMH-OCTO-001 kill switch. `0`/`false`/`no`/`off` restores the promoted
    /// four-row quad dispatch inside the same executable.
    private static let octoStreamEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_TIED_LMHEAD_OCTO"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `kernelHeader` stays byte-identical -- `AttentionOQMVV1` and
    /// `DenseMLPQMVV1` splice it -- so the eight-stream body is appended into a
    /// file-local header instead of edited in place.
    private static let octoKernelHeader = kernelHeader + """

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_octo_stream_impl(
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

"""

    private static let octoKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_octo_stream_ey_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = K;
            const int out_vec_size = OUTN;

            qmv_affine4_g64_octo_stream_impl<T, 64, 4>(
                w,
                scales,
                biases,
                x + 0 * in_vec_size,
                x + 1 * in_vec_size,
                x + 2 * in_vec_size,
                x + 3 * in_vec_size,
                x + 4 * in_vec_size,
                x + 5 * in_vec_size,
                x + 6 * in_vec_size,
                x + 7 * in_vec_size,
                y + 0 * out_vec_size,
                y + 1 * out_vec_size,
                y + 2 * out_vec_size,
                y + 3 * out_vec_size,
                y + 4 * out_vec_size,
                y + 5 * out_vec_size,
                y + 6 * out_vec_size,
                y + 7 * out_vec_size,
                in_vec_size,
                tid,
                simd_gid,
                simd_lid);
            return;
            """,
        header: octoKernelHeader,
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

        let yGroups = outDim / outputsPerGroup
        if octoStreamEnabled {
            // One x-group. The eight cohort rows are streamed against a single
            // weight / scale / bias load stream, so the second group that used
            // to re-issue that stream for rows 4-7 is not launched at all.
            return octoKernel(
                [x, weight, scales, biases],
                template: [
                    ("T", x.dtype),
                    ("K", inDim),
                    ("OUTN", outDim),
                ],
                grid: (simdWidth, yGroups * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, 1, outDim]],
                outputDTypes: [x.dtype]
            )[0]
        }

        let xGroups = batch / rowsPerGroup
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
