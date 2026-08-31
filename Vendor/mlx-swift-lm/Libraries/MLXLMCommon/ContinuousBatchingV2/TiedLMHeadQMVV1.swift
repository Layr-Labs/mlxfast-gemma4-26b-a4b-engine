// LMH-OCTO-001: one pass over the tied lm_head weight matrix instead of two.
//
// The vendored MLX host launches ordinary QMV as
//
//     MTL::Size group_dims(bk, 2, 1);                     // (32, 2, 1)
//     MTL::Size grid_dims(M, (N + bn - 1) / bn, B);       // x = M
//     compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
//
// (`backend/metal/quantized.cpp`), so the x extent of the grid is the cohort
// row count, eight, and each x-group re-reads the same weight rows for its own
// cohort row. The stock body reads the matrix EIGHT times per decode step.
// The promoted quad-STREAM body in `quantized.h` claims four cohort rows per
// threadgroup and returns from the rest, which cuts that to TWO passes; this
// file's tight-grid dispatch (josusanmartin's LMH-001, submission `2de922a4`,
// official receipt +0.216% composite / +0.383% decode on the `3ab9bd4` parent)
// stops launching the six groups that only hit the early return, but the two
// surviving x-groups still stream the whole matrix each.
//
// On the tied lm_head that second pass is the largest single read in the
// decode step. N = 262144 rows of K = 2816 affine-4 weights is 352 MiB of
// packed words plus 44 MiB of group scales and biases; at two passes the
// kernel pulls 792 MiB to produce 2.1 M logits, an arithmetic intensity of
// about a third of an operation per byte. It is the most bandwidth-skewed op
// in the tower.
//
// THIS CHANGE. The body becomes an octo-stream: `results_per_simdgroup` drops
// from four to two and the cohort loop widens from four rows to all eight, so
// one simdgroup holds two output rows' packed words and drives every cohort
// row against them before moving to the next K block. The grid's x extent
// becomes one. Each weight word, each group scale and each group bias is then
// fetched exactly once per decode step rather than twice, and the resident
// working set per thread goes DOWN: `result` is still sixteen floats (eight
// streams by two rows, where it was four by four), while `packed` halves from
// [4][2] to [2][2] and `scale_local`/`bias_local` halve from four entries to
// two. Nothing is spent to buy the saving.
//
// What it costs is the query side. `load_vector` now runs eight times per K
// block instead of four, because each of the eight cohort rows must be widened
// and run-summed against the two resident weight rows. That input rectangle is
// eight rows of 2816 bfloat16, 45 KiB in total, and every threadgroup reads
// the same 45 KiB, so those loads come out of cache rather than DRAM. The
// multiply-accumulate count is unchanged: sixteen `qdot_affine4_registered`
// calls per block either way, because the product of streams and rows is
// sixteen in both shapes.
//
// EXACTNESS. Every output element is still produced by the same arithmetic in
// the same order. `load_vector` sees the same eight bfloat16 words at the same
// addresses; `qdot_affine4_registered` sees the same packed word, the same
// pre-scaled query values, the same scale, bias and run sum; the K traversal
// visits the same blocks in the same ascending order into the same float
// accumulator; the same `simd_sum` reduces it; the same bfloat16 conversion
// stores it. Only the assignment of (output row, cohort row) pairs to
// threadgroups changes, and no accumulator is shared between pairs. The kernel
// name is versioned to `..._octo_stream_v4` so the named MLX kernel cache
// cannot answer the new request with the previously compiled quad body.
//
// LINEAGE. The header below still carries the PROMOTED helpers VERBATIM from
// `quantized.h` -- `load_vector`, `load_vector_safe` and
// `qdot_affine4_registered`, each with the `#pragma clang loop unroll(full)`
// annotations on their compile-time walks that `df8d0ef` promoted at +1.11%.
// A hand-inlined expansion of the same arithmetic measured 1-4 ulp off the
// library kernel under this compiler; verbatim inclusion is what the prior
// parity receipts on this track used, and the local runtime parity test pins
// it bitwise.
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
    private static let simdWidth = 32
    private static let simdGroups = 2
    /// `num_simdgroups * results_per_simdgroup` in the octo body.
    private static let outputsPerGroup = 4
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
METAL_FUNC void qmv_affine4_g64_octo_stream_impl(
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
  constexpr int streams = 8;
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
  thread float result[streams][results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
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
    for (int m = 0; m < streams; m++) {
      const float sum = load_vector<T, float, values_per_thread, 4>(
          x + m * in_vec_size, x_thread);
      #pragma clang loop unroll(full)
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine4_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x += block_size;
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
    for (int m = 0; m < streams; m++) {
      const float sum = load_vector_safe<T, float, values_per_thread, 4>(
          x + m * in_vec_size, x_thread, remaining);
      #pragma clang loop unroll(full)
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[m][row] += qdot_affine4_registered<float, values_per_thread>(
            packed[row], x_thread, scale_local[row], bias_local[row], sum);
      }
    }
  }

  #pragma clang loop unroll(full)
  for (int row = 0; row < results_per_simdgroup; row++) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < streams; m++) {
      result[m][row] = simd_sum(result[m][row]);
    }
    if (simd_lid == 0) {
      #pragma clang loop unroll(full)
      for (int m = 0; m < streams; m++) {
        y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
      }
    }
  }
}
"""

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_tied_lmhead_qmv_affine4_g64_octo_stream_v4",
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
                x,
                y,
                in_vec_size,
                out_vec_size,
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

        let yGroups = outDim / outputsPerGroup
        return kernel(
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
}
