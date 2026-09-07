// Copyright © 2026. Canonical Apple Metal helper notices retained below.
import Foundation
import MLX

/// Keep the incumbent span-four down-QMV arithmetic and omit its empty y groups.
public enum Gemma4DownTightGridV1 {
    static let enabled: Bool = {
        #if os(macOS)
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_DOWN_TIGHT_GRID"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
        #else
        return false
        #endif
    }()

    /// Number of consecutive output-row groups one threadgroup walks.
    ///
    /// The dispatch covers a fixed 352 row groups of eight rows each, split
    /// `352 / span` ways. `span` therefore trades threadgroup count against
    /// how many times the per-threadgroup route election runs. The election is
    /// one `rhs_indices` word plus, when the prefix-bound tag is absent, a
    /// backward walk over at most 64 words; the walk it guards is a full
    /// `704 x 8` affine-4 accumulation per row group, so the election is a
    /// small fixed preamble either way.
    ///
    /// Span eight was measured on the ranked box and lost, which says this
    /// dispatch is short enough that occupancy, not preamble, is what binds
    /// it. Span two is the other direction: 176 threadgroups per layer instead
    /// of 88, same arithmetic, same order.
    static let tileSpan: Int = {
        #if os(macOS)
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_DOWN_TILE_SPAN2"]
        else { return 2 }
        return ["0", "false", "no", "off"].contains(raw.lowercased()) ? 4 : 2
        #else
        return 4
        #endif
    }()

    /// Bound to the immutable sanitized checkpoint, like the fused gate/up storage.
    public final class Storage {
        private let weight: MLXArray
        private let scales: MLXArray
        private let biases: MLXArray

        public init?(weight: MLXArray, scales: MLXArray, biases: MLXArray) {
            guard weight.dtype == .uint32 && weight.shape == [128, 2816, 88],
                scales.dtype == .bfloat16 && scales.shape == [128, 2816, 11],
                biases.dtype == .bfloat16 && biases.shape == scales.shape
            else { return nil }
            self.weight = weight
            self.scales = scales
            self.biases = biases
        }

        static func admits(x: MLXArray, indices: MLXArray) -> Bool {
            x.dtype == .bfloat16 && x.shape == [64, 1, 704]
                && indices.dtype == .uint32 && indices.shape == [64]
        }

        /// The caller supplies the incumbent identity LHS and sorted RHS keys.
        func call(
            x: MLXArray, lhsIndices: MLXArray, indices: MLXArray,
            taggedRoute: Bool = false
        ) -> MLXArray {
            call(x: x, lhsIndices: lhsIndices, indices: indices, span: tileSpan,
                taggedRoute: taggedRoute)
        }

        func call(
            x: MLXArray, lhsIndices: MLXArray, indices: MLXArray, span: Int,
            taggedRoute: Bool = false
        ) -> MLXArray {
            (taggedRoute ? kernelTagged : kernel)(
                [weight, scales, biases, x, lhsIndices, indices],
                template: [("T", DType.bfloat16), ("SPAN", span)],
                grid: (32, (352 / span) * 2, 64), threadGroup: (32, 2, 1),
                outputShapes: [[64, 1, 2816]], outputDTypes: [.bfloat16]
            )[0]
        }
    }

    /// DOWN-TAGGED-ROUTE. Companion to the gate/up variant: when the route
    /// producer emits prefix-bounds tagged words, the raw-key fallbacks here
    /// -- a backward scan for `run_offset` and a forward peek for `has_pair`,
    /// both data-dependent reads of `rhs_indices` -- can never execute, yet
    /// they inline into the tile helper and are charged to every threadgroup.
    /// Selected per call from the producer's own contract; distinct kernel
    /// name; output bit-identical.
    private static func makeKernel(tagged: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
        name: "gemma4_b8_down_qmv_span4_tight_zorder_v1"
            + (tagged ? "_tagged_v1" : ""),
        inputNames: ["w", "scales", "biases", "x", "lhs_indices", "rhs_indices"],
        outputNames: ["y"],
        source: #"""
uint3 tid = threadgroup_position_in_grid;
const uint linear = tid.y + tid.z * (352 / SPAN);
tid.y = (linear / 64) * SPAN;
tid.z = linear % 64;
// Preserve assignment-fast enumeration with 352 / SPAN surviving y groups.
// Map compact y-group g to the old survivor SPAN*g. The helper, its pair
// elections, per-tile walk, qdot chains, SIMD reductions and stores are verbatim.
gather_qmv_gemma4_down_tile<T, 64, 4, SPAN>(
    w, scales, biases, x, lhs_indices, rhs_indices, y,
    gemma4_tight_down_K, gemma4_tight_down_N, 1, 1,
    704, 2816 * 704 / 8, 2816 * 704 / 64, 2816 * 704 / 64,
    tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""#,
        header: "#define DOWN_TAGGED_ROUTE \(tagged ? 1 : 0)\n" + #"""
// Copyright © 2023-2024 Apple Inc. Canonical helper bodies verified byte-identical to 093e716.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
static constant constexpr const int SIMD_SIZE=32;
static constant constexpr const int QUAD_SIZE=4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
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
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
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
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
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

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread>
inline void qdot_affine4_pair_word(
    uint packedWord,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  static_assert(values_per_thread == 8, "Word load expects eight 4-bit values");
  const uint packed0 = packedWord & 0xffffu;
  const uint packed1 = packedWord >> 16;
  U accum0 =
      (x0[0] * (packed0 & 0x000f) +
       x0[1] * (packed0 & 0x00f0) +
       x0[2] * (packed0 & 0x0f00) +
       x0[3] * (packed0 & 0xf000));
  U accum1 =
      (x1[0] * (packed0 & 0x000f) +
       x1[1] * (packed0 & 0x00f0) +
       x1[2] * (packed0 & 0x0f00) +
       x1[3] * (packed0 & 0xf000));
  accum0 +=
      (x0[4] * (packed1 & 0x000f) +
       x0[5] * (packed1 & 0x00f0) +
       x0[6] * (packed1 & 0x0f00) +
       x0[7] * (packed1 & 0xf000));
  accum1 +=
      (x1[4] * (packed1 & 0x000f) +
       x1[5] * (packed1 & 0x00f0) +
       x1[6] * (packed1 & 0x0f00) +
       x1[7] * (packed1 & 0xf000));
  out0 = scale * accum0 + sum0 * bias;
  out1 = scale * accum1 + sum1 * bias;
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k <= in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k <= in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int tail_values = static_cast<int>(in_vec_size - k);
    if (tail_values > 0) {
      // Affine callers keep K a whole number of quantization groups and k
      // advances by whole blocks, so the tail is a whole number of
      // values_per_thread lane packets: routed-expert down_proj K=704 leaves
      // 192 values = 24 complete packets, dense down_proj K=2112 (8-bit)
      // leaves 64 = 16.  Active lanes run the fixed unrolled loader and qdot;
      // the dynamic safe-tail remains only for a genuinely partial packet,
      // which no affine caller presents.
      if (tail_values % values_per_thread == 0) {
        const uint active_tail_lanes = uint(tail_values / values_per_thread);
        if (simd_lid < active_tail_lanes) {
          U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device T* sl = scales + row * in_vec_size_g;
            const device T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] +=
                qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
          }
        }
      } else {
        const int remaining = clamp(
            static_cast<int>(tail_values - simd_lid * values_per_thread),
            0,
            values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe<T, U, values_per_thread, bits>(
              x, x_thread, remaining);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device T* sl = scales + row * in_vec_size_g;
            const device T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] += qdot_safe<U, values_per_thread, bits>(
                wl, x_thread, s, b, sum, remaining);
          }
        }
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_pair_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    device T* y0,
    device T* y1,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread float x1_thread[values_per_thread];
  thread uint packed[results_per_simdgroup];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
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
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 = load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
    float sum1 = load_vector<T, float, values_per_thread, 4>(x1, x1_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      float dot0;
      float dot1;
      qdot_affine4_pair_word<float, values_per_thread>(
          packed[row], x0_thread, x1_thread, scale_local[row], bias_local[row], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
  }

  // Every Gemma 4 caller entering this specialized g64 path has K aligned to
  // 64.  The final block therefore contains an integral number of complete
  // eight-value lane packets (32 lanes for K=2816, 24 for expert down_proj
  // K=704); no active lane needs the generic dynamic safe-tail loops.
  const uint active_tail_lanes =
      uint((in_vec_size - k) / values_per_thread);
  if (simd_lid < active_tail_lanes) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 =
        load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
    float sum1 =
        load_vector<T, float, values_per_thread, 4>(x1, x1_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      float dot0;
      float dot1;
      qdot_affine4_pair_word<float, values_per_thread>(
          packed[row], x0_thread, x1_thread, scale_local[row], bias_local[row], sum0, sum1, dot0, dot1);
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

template <typename T, int group_size, int bits, int span>
METAL_FUNC void gather_qmv_gemma4_down_tile(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const uint lhs_stride,
    const uint rhs_stride,
    const int64_t x_stride,
    const int64_t w_stride,
    const int64_t s_stride,
    const int64_t b_stride,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int gemma4_down_tile_span = span; // dispatch-selected: 4 or 2
  if (tid.y % uint(gemma4_down_tile_span) != 0u) {
    return;
  }
  const uint assignment = tid.z;
  const uint32_t route_word = rhs_indices[assignment * rhs_stride];
#if DOWN_TAGGED_ROUTE
  const uint32_t expert = route_word & 0xffu;
  const uint run_offset = (route_word >> 8) & 0x3fu;
#else
  const bool expert_prefix_bounds = (route_word & 0x80000000u) != 0u;
  const uint32_t expert =
      expert_prefix_bounds ? (route_word & 0xffu) : route_word;
  uint run_offset = 0;
  if (expert_prefix_bounds) {
    run_offset = (route_word >> 8) & 0x3fu;
  } else {
    for (uint prior = assignment; prior > 0; --prior) {
      if (rhs_indices[(prior - 1) * rhs_stride] != expert) {
        break;
      }
      run_offset++;
    }
  }
#endif
  // Odd positions are produced by the immediately preceding pair leader.
  if ((run_offset & 1) != 0) {
    return;
  }
  const device uint32_t* tile_w = w + expert * w_stride;
  const device T* tile_scales = scales + expert * s_stride;
  const device T* tile_biases = biases + expert * b_stride;
  const device T* tile_x0 =
      x + lhs_indices[assignment * lhs_stride] * x_stride;
  device T* tile_y0 = y + assignment * out_vec_size;
#if DOWN_TAGGED_ROUTE
  const bool has_pair = (((route_word >> 14) & 0x3fu) + 1u) > 1u;
#else
  const bool has_pair = expert_prefix_bounds
      ? (((route_word >> 14) & 0x3fu) + 1u) > 1u
      : assignment + 1 < 64 &&
          rhs_indices[(assignment + 1) * rhs_stride] == expert;
#endif
  if (has_pair) {
    const device T* tile_x1 =
        x + lhs_indices[(assignment + 1) * lhs_stride] * x_stride;
    device T* tile_y1 = y + (assignment + 1) * out_vec_size;
    for (int t = 0; t < gemma4_down_tile_span; t++) {
      uint3 tile_tid = tid;
      tile_tid.y = tid.y + uint(t);
      qmv_affine4_g64_pair_impl<T, group_size, bits>(
          tile_w,
          tile_scales,
          tile_biases,
          tile_x0,
          tile_x1,
          tile_y0,
          tile_y1,
          in_vec_size,
          tile_tid,
          simd_gid,
          simd_lid);
    }
    return;
  }
  for (int t = 0; t < gemma4_down_tile_span; t++) {
    uint3 tile_tid = tid;
    tile_tid.y = tid.y + uint(t);
    qmv_impl<T, group_size, bits>(
        tile_w,
        tile_scales,
        tile_biases,
        tile_x0,
        tile_y0,
        in_vec_size,
        out_vec_size,
        tile_tid,
        simd_gid,
        simd_lid);
  }
}

constant int gemma4_tight_down_K=704;
constant int gemma4_tight_down_N=2816;
"""#,
        ensureRowContiguous: true)
    }

    private static let kernel: MLXFast.MLXFastKernel = makeKernel(tagged: false)
    private static let kernelTagged: MLXFast.MLXFastKernel = makeKernel(tagged: true)
}
