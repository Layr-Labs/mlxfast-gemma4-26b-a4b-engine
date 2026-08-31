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
//
// MMA-MLP-001-DOWN adds a second body under the same dispatch and the same
// guards: an 8-bit affine g64 simdgroup-matrix (`simdgroup_float8x8`) tier that
// serves all eight cohort rows from ONE weight fetch, engaged on the DOWN plane
// (K = 2112, N = 2816) only. `DARKBLOOM_GEMMA4_MLP_MMA8_DOWN=0` restores
// DMLP-001 for it byte for byte. The gate/up plane keeps the tight kernel and
// its activation-sum table exactly as on the tip; its arm exists but is opt-in
// (`DARKBLOOM_GEMMA4_MLP_MMA8_GATEUP=1`) -- see the switch doc comment for why
// the two planes are not shipped together.

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

    /// MMA-MLP-001 arms, one per dense MLP plane. The two planes are separate
    /// because they behave differently, not for convenience.
    ///
    /// DOWN (K = 2112, N = 2816) is the shipped arm and defaults ON.
    /// `DARKBLOOM_GEMMA4_MLP_MMA8_DOWN=0` restores DMLP-001 for it byte for
    /// byte, so a ranked rejection can be bisected post hoc without a rebuild
    /// pair.
    ///
    /// GATE/UP (K = 2816, N = 2112) defaults OFF and is OPT-IN, which is the
    /// opposite polarity of every other switch in this family and is deliberate.
    /// Both arms are the same one-ulp class against the per-row M = 1 road, but
    /// on the local eight-tape cohort probe -- packed admission, deterministic
    /// draw, so the only variable is the arm -- the gate/up arm moves one stream
    /// from zero mismatches to a divergence at row 1, while the down arm
    /// introduces no new early divergence at all. A first-row flip is a token
    /// the trusted oracle did not rank near the top, i.e. exactly the class the
    /// cohort tolerance gate refuses outright rather than pricing against its
    /// budget. Until that arm has a ranked result of its own it stays behind an
    /// explicit `DARKBLOOM_GEMMA4_MLP_MMA8_GATEUP=1`, and the gate/up plane runs
    /// the promoted DMLP-001/DMLP-002 tight kernel exactly as on the tip.
    public static let mma8GateUpEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_MMA8_GATEUP"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    public static let mma8DownEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_MMA8_DOWN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Reuse each down-plane lane's exact affine bias sum across output
    /// tiles. Disabling this restores the original per-tile MMA8 reduction.
    private static let mma8DownLaneSumsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_MMA8_DOWN_LANE_SUMS"]
        else { return false }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Down-only compile-time K/group walk; the odd 33-group split remains
    /// 17+16. The lane-sum opt-in and gate/up paths retain their old kernels.
    private static let mma8DownStaticKEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_DOWN_STATIC_K"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Share each x-side K-group build across two adjacent down-projection
    /// output tiles. The body pins K = 2112 the way the static-K kernel does,
    /// so it engages only while that switch is on; explicit false here
    /// restores the promoted single-tile body.
    private static let mma8DownMultiTileEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_MMA8_DOWN_MULTITILE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Mirrors the `TILES` argument and the `tid.y * 16` tile stride baked into
    /// the multi-tile kernel source; the two only ever move together.
    private static let mma8DownTilesPerGroup = 2

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

        /// ZIP-ROUTER-001 ordering handle. Read-only view of the table's own
        /// output array so a caller can name this dispatch as an
        /// `MLX.depends` edge when it interleaves the dense chain with an
        /// independent chain. `Depends` emits no dispatch and its output
        /// aliases its input's buffer, so handing the array out cannot change
        /// what any kernel reads or writes.
        public var dependencyHandle: MLXArray { values }
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
  #pragma unroll
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
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      #pragma unroll
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    #pragma unroll
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
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      #pragma unroll
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    #pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  #pragma unroll
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
              #pragma unroll
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

    /// MMA-MLP-001. Verbatim transcription of the `kGemma4QmvMma8Affine8`
    /// tier body from `zarar/t6-mma-s2` (quantized.h / quantized.cpp twins,
    /// `gemma4_qmv_mma8_affine8_g64_impl`), hosted here the way
    /// `CBv2AttentionOQMVV1` hosts the 4-bit o_proj MMA body. Only the
    /// preprocessor line continuations are collapsed onto one line, which the
    /// preprocessor does anyway and which a Swift multiline literal requires.
    /// The arithmetic is untouched: one fp32 `simdgroup_float8x8` chain forms
    /// all 64 products of a g64 group in ascending k, `mma8_runsum8`
    /// reproduces `load_vector`'s bits == 8 sequential fp32 `sum` bit for bit,
    /// and the close is the reference's own `s * C + rs * b`. KS = 2 splits
    /// the K/64 groups across the two simdgroups; an odd count (down_proj,
    /// G = 33) gives the extra group to simdgroup 0, deterministically.
    private static let mma8KernelHeader = """
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

struct mma8_coord {
  short fm;
  short fn;
};

// steel/gemm/mma.h's `get_coord` arithmetic, reproduced locally so the same
// text compiles wherever this body is pasted: lane (fm, fn) owns elements
// (fm, fn) and (fm, fn + 1) of every 8x8 operand.
inline mma8_coord mma8_lane(uint lane) {
  const short qid = short(lane / 4);
  return {
      short((qid & 4) + short((lane / 2) % 4)),
      short((qid & 2) * 2 + short(lane % 2) * 2)};
}

// The x-side loads pull sixteen bytes at a time and split them into eight
// 16-bit lanes, which only makes sense for a 2-byte T. `affine_qmv` is also
// instantiated for `float`; the tier gate carries `sizeof(T) == 2` so the
// float instantiation never runs this body, and this primary template is what
// lets it still compile.
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

// Widening a 16-bit float to fp32 is exact, so these two reproduce the
// reference's own operand values bit for bit.
template <typename T>
inline float mma8_lo(uint u) {
  return float(mma8_u16<T>::cast(ushort(u & 0xFFFFu)));
}

template <typename T>
inline float mma8_hi(uint u) {
  return float(mma8_u16<T>::cast(ushort(u >> 16)));
}

// Textual twin of `load_vector<T, float, 8, 4>`'s `sum` on the same aligned
// 8-run that the reference lane owns: the parenthesised 4-tuple is evaluated
// on T exactly as in the reference, then the two trees are added in fp32. The
// bias term of the affine form therefore reuses the reference's own
// elementary values, not a re-derived sum.
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

// The 8-bit twin of the run sum. `load_vector`'s bits == 8 branch is NOT the
// bits == 4 branch: it accumulates `sum += x[i]` one element at a time in the
// ACCUMULATOR type (U = float), and it stores `x_thread[i] = x[i]` with no
// pre-scaling, because the 8-bit `qdot` arm multiplies by the whole byte code
// 0..255 rather than by a masked-and-shifted nibble. The 8-bit reference lane
// carries values_per_thread = 4 (block 128, scale_step_per_thread = 16), so
// its `sum` is the fp32 sequential sum of ONE aligned 4-run. This helper
// therefore rebuilds the TWO reference 4-run sums that tile the aligned 8-run
// this lane loads, each bit for bit, and adds them; the bias term of the
// affine close is then built out of the reference's own `sum` values.
template <typename T>
inline float mma8_runsum8(uint4 r) {
  thread T xt[8];
  xt[0] = mma8_u16<T>::cast(ushort(r.x & 0xFFFFu));
  xt[1] = mma8_u16<T>::cast(ushort(r.x >> 16));
  xt[2] = mma8_u16<T>::cast(ushort(r.y & 0xFFFFu));
  xt[3] = mma8_u16<T>::cast(ushort(r.y >> 16));
  xt[4] = mma8_u16<T>::cast(ushort(r.z & 0xFFFFu));
  xt[5] = mma8_u16<T>::cast(ushort(r.z >> 16));
  xt[6] = mma8_u16<T>::cast(ushort(r.w & 0xFFFFu));
  xt[7] = mma8_u16<T>::cast(ushort(r.w >> 16));
  float s0 = 0;
  s0 += xt[0];
  s0 += xt[1];
  s0 += xt[2];
  s0 += xt[3];
  float s1 = 0;
  s1 += xt[4];
  s1 += xt[5];
  s1 += xt[6];
  s1 += xt[7];
  return s0 + s1;
}

// The 8-bit A fill. A lane (fm, fn) owns A elements (fm, fn) and (fm, fn + 1),
// i.e. the codes at k = 64 g + 8 fn + j and k = 64 g + 8 (fn + 1) + j, which
// are byte j and byte 8 + j of the sixteen consecutive weight bytes this lane
// loads as one `uint4`. Byte j lives in word j / 4 at bit 8 * (j % 4), so the
// two components and the shift are all the macro needs. The code is the whole
// byte 0..255 and x is unscaled, exactly as the 8-bit `qdot` arm forms its
// product; a bf16 x carries 8 significant bits and the code 8, so the product
// needs at most 16 and is exact in fp32.
#define MMA8_STEP8(BB, WLO, WHI, SH) A.thread_elements()[0] = float(extract_bits(wv.WLO, (SH), 8)); A.thread_elements()[1] = float(extract_bits(wv.WHI, (SH), 8)); simdgroup_multiply_accumulate(C, A, BB, C);

// x is [8, K] with K % 64 == 0, w is packed [N, K / 4] uint32 (one byte per
// code, so the row stride is K bytes and K % 64 == 0 keeps every `uint4` load
// 16-byte aligned), scales and biases are [N, K / 64], y is [8, N]. `n0` is
// the first of the eight output rows this threadgroup owns. KS = 2 splits the
// K / 64 groups between the two simdgroups of the host's (32, 2, 1)
// threadgroup; an odd group count (down_proj: K = 2112 -> G = 33 -> 17 + 16)
// gives the extra group to simdgroup 0, which is deterministic and independent
// of scheduling. `red` is 32 float2 of threadgroup memory for the KS = 2 close.
template <typename T, int KS>
METAL_FUNC void gemma4_qmv_mma8_affine8_g64_impl(
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
  const mma8_coord c = mma8_lane(simd_lid);

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

    // Each B lane owns the two 8-runs whose run sums the C lane (fm, fn)
    // needs; three xor-butterfly steps over the fm lane bits broadcast
    // RS[g][fn] and RS[g][fn + 1] to all eight lanes of the fn column group.
    float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));
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

    const uint4 wv = *((const device uint4*)(wrow + 64 * g));
    const float s = float(srow[g]);
    const float b = float(brow[g]);

    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    MMA8_STEP8(B0, x, z, 0)
    MMA8_STEP8(B1, x, z, 8)
    MMA8_STEP8(B2, x, z, 16)
    MMA8_STEP8(B3, x, z, 24)
    MMA8_STEP8(B4, y, w, 0)
    MMA8_STEP8(B5, y, w, 8)
    MMA8_STEP8(B6, y, w, 16)
    MMA8_STEP8(B7, y, w, 24)

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

    private static let mma8Kernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_affine8_g64_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            gemma4_qmv_mma8_affine8_g64_impl<T, 2>(
                w, scales, biases, x, y,
                x_shape[x_ndim - 1], w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    // DMLP-STATIC-K-018: the frontier's attention kernels expose their K walk
    // to the compiler. Down has G=33, so its final second-SIMD iteration must
    // skip before any load/MMA, retaining the original 17+16 split and close.
    private static let mma8DownStaticKHeader: String = {
        var result = mma8KernelHeader
        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }
        replaceOnce(
            "gemma4_qmv_mma8_affine8_g64_impl(",
            with: "gemma4_qmv_mma8_affine8_g64_down_k2112_impl(")
        replaceOnce(
            """
                device T* y,
                const int K,
                const int N,
            """,
            with: """
                device T* y,
                const int N,
            """)
        replaceOnce(
            """
              const int G = K / 64;
              const int gh = (G + 1) / 2;
              const int g_begin = (KS == 2 && simd_gid == 1) ? gh : 0;
              const int g_end = (KS == 2 && simd_gid == 0) ? gh : G;
            """,
            with: """
              constexpr int K = 2112;
              constexpr int G = K / 64;
              constexpr int gh = (G + 1) / 2;
              constexpr int nGroups = (KS == 2) ? gh : G;
              const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
            """)
        replaceOnce(
            "  for (int g = g_begin; g < g_end; ++g) {",
            with: """
              #pragma unroll
              for (int gi = 0; gi < nGroups; ++gi) {
                const int g = g0 + gi;
                if (g >= G) continue;
            """)
        return result
    }()

    private static let mma8DownStaticKKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_affine8_g64_down_k2112_unroll_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            gemma4_qmv_mma8_affine8_g64_down_k2112_impl<T, 2>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: mma8DownStaticKHeader,
        ensureRowContiguous: true)

    private static let mma8DownMultiTileHeader = mma8KernelHeader + """

template <typename T, int KS, int TILES, int KFIX>
METAL_FUNC void gemma4_qmv_mma8_affine8_g64_down_mt(
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
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    wrow[t] = (const device uint8_t*)w + (nt + c.fm) * K + 8 * c.fn;
    srow[t] = scales + (nt + c.fm) * G;
    brow[t] = biases + (nt + c.fm) * G;
  }
  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  thread float acc0[TILES];
  thread float acc1[TILES];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

#pragma clang loop unroll_count(4)
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;
    if (g >= G) continue;
    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

    float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));
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
      const uint4 wv = *((const device uint4*)(wrow[t] + 64 * g));
      const float s = float(srow[t][g]);
      const float b = float(brow[t][g]);

      simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
      MMA8_STEP8(B0, x, z, 0)
      MMA8_STEP8(B1, x, z, 8)
      MMA8_STEP8(B2, x, z, 16)
      MMA8_STEP8(B3, x, z, 24)
      MMA8_STEP8(B4, y, w, 0)
      MMA8_STEP8(B5, y, w, 8)
      MMA8_STEP8(B6, y, w, 16)
      MMA8_STEP8(B7, y, w, 24)

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

    private static let mma8DownMultiTileKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_affine8_g64_down_mt2_k2112_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            gemma4_qmv_mma8_affine8_g64_down_mt<T, 2, 2, 2112>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8DownMultiTileHeader,
        ensureRowContiguous: true)

    /// One complete SIMD group per g64 group. Keep all lane results rather
    /// than canonicalizing row sums: the consumer reads its own lane's tree.
    private static let mma8DownLaneSumKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_down_lane_sums_v1",
        inputNames: ["x"],
        outputNames: ["laneSums"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int g = int(threadgroup_position_in_grid.y);
            const int K = x_shape[x_ndim - 1];
            const mma8_coord c = mma8_lane(lane);
            const device T* x0 = x + c.fn * K + 8 * c.fm;
            const device T* x1 = x0 + K;
            const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
            const uint4 r1 = *((const device uint4*)(x1 + 64 * g));
            float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));
            rs += simd_shuffle_xor(rs, 2u);
            rs += simd_shuffle_xor(rs, 4u);
            rs += simd_shuffle_xor(rs, 16u);
            ((device float2*)laneSums)[g * 32 + lane] = rs;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let mma8DownLaneSumHeader: String = {
        var result = mma8KernelHeader
        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }
        replaceOnce(
            "gemma4_qmv_mma8_affine8_g64_impl(",
            with: "gemma4_qmv_mma8_affine8_g64_lane_sums_impl(")
        replaceOnce(
            """
                const device T* x,
                device T* y,
                const int K,
            """,
            with: """
                const device T* x,
                const device float2* laneSums,
                device T* y,
                const int K,
            """)
        replaceOnce(
            """
                // Each B lane owns the two 8-runs whose run sums the C lane (fm, fn)
                // needs; three xor-butterfly steps over the fm lane bits broadcast
                // RS[g][fn] and RS[g][fn + 1] to all eight lanes of the fn column group.
                float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));
                rs += simd_shuffle_xor(rs, 2u);
                rs += simd_shuffle_xor(rs, 4u);
                rs += simd_shuffle_xor(rs, 16u);
            """,
            with: """
                // Producer retained this exact lane's original reduction tree.
                const float2 rs = laneSums[g * 32 + simd_lid];
            """)
        return result
    }()

    private static let mma8DownLaneSumQMVKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_affine8_g64_down_lane_sums_v1",
        inputNames: ["x", "w", "scales", "biases", "laneSums"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            gemma4_qmv_mma8_affine8_g64_lane_sums_impl<T, 2>(
                w, scales, biases, x, (const device float2*)laneSums, y,
                x_shape[x_ndim - 1], w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8DownLaneSumHeader,
        ensureRowContiguous: true)

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_v2_unroll",
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

    /// One exact float sum for each `(K/128 block, simd lane, cohort row)`.
    /// The expression is the bits==8 arm of stock `load_vector`: a float zero
    /// followed by four ascending `sum += bf16` operations. Row is the unit-
    /// stride dimension so one table serves both gate and up projections.
    private static let activationSumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_affine8_xsum_v2_unroll",
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
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum += xp[i];
            }
            xSums[(k_block * 32 + lane) * 8 + row] = sum;
            """,
        ensureRowContiguous: true
    )

    private static let activationSumQMVKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_xsum_v2_unroll",
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

    /// Adopts an exact producer-emitted table for the same pinned decode
    /// geometry as `activationSums(for:)`. The layer-glue producer writes the
    /// table while its BF16 dense input is still resident in registers, using
    /// the identical four-value accumulation order as the standalone kernel.
    public static func activationSums(
        produced values: MLXArray, for x: MLXArray
    ) -> ActivationSums? {
        let blocks = 2816 / kBlock
        guard enabled,
            activationSumsEnabled,
            x.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) == batch,
            x.dim(1) == sequence,
            x.dim(2) == 2816,
            x.size == batch * sequence * 2816,
            values.dtype == .float32,
            values.ndim == 1,
            values.size == blocks * simdWidth * batch
        else { return nil }
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

        // MMA-MLP-001: matrix-unit body for the dense MLP planes, DOWN plane
        // only by default. One threadgroup per 8-column output tile serves all
        // eight cohort rows from a single weight fetch; the DMLP-001 quad-stream
        // body below streams the same plane twice (four rows per x-group), so
        // the plane is read once per round instead of twice. `liveShape` above
        // already pins the pair, so naming gate/up here also names down.
        //
        // The MMA close builds its own `rs` with `mma8_runsum8`, so this path
        // never consumes DMLP-002's xSums table. With the gate/up arm off the
        // table is still built and consumed for the gate/up plane exactly as on
        // the tip -- the fall-through below is the promoted path unchanged.
        let isGateUp = inDim == 2816 && outDim == 2112
        if isGateUp ? mma8GateUpEnabled : mma8DownEnabled {
            let yTiles = outDim / outputsPerGroup
            if !isGateUp && mma8DownLaneSumsEnabled {
                let groups = inDim / Self.groupSize
                let laneSums = mma8DownLaneSumKernel(
                    [x],
                    template: [("T", x.dtype)],
                    grid: (simdWidth, groups, 1),
                    threadGroup: (simdWidth, 1, 1),
                    outputShapes: [[groups, simdWidth, 2]],
                    outputDTypes: [.float32]
                )[0]
                return mma8DownLaneSumQMVKernel(
                    [x, weight, scales, biases, laneSums],
                    template: [("T", x.dtype)],
                    grid: (simdWidth, yTiles * simdGroups, 1),
                    threadGroup: (simdWidth, simdGroups, 1),
                    outputShapes: [[batch, sequence, outDim]],
                    outputDTypes: [x.dtype]
                )[0]
            }
            if !isGateUp, mma8DownStaticKEnabled,
                mma8DownMultiTileEnabled,
                yTiles % mma8DownTilesPerGroup == 0 {
                CBv2EngageMark.once("dense-mlp-mma8-down-multitile")
                let tileGroups = yTiles / mma8DownTilesPerGroup
                return mma8DownMultiTileKernel(
                    [x, weight, scales, biases],
                    template: [("T", x.dtype)],
                    grid: (simdWidth, tileGroups * simdGroups, 1),
                    threadGroup: (simdWidth, simdGroups, 1),
                    outputShapes: [[batch, sequence, outDim]],
                    outputDTypes: [x.dtype]
                )[0]
            }
            let selectedMMA = !isGateUp && mma8DownStaticKEnabled
                ? mma8DownStaticKKernel : mma8Kernel
            return selectedMMA(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, yTiles * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, sequence, outDim]],
                outputDTypes: [x.dtype]
            )[0]
        }

        let xGroups = batch / rowsPerGroup
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
