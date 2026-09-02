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

    /// W4-LOAD (engage mark `mlp-w4-load`). The four packed affine-8 codes a
    /// lane owns for one output row are four CONTIGUOUS bytes at a 4-byte
    /// aligned address, so they can be fetched as ONE aligned 32-bit load
    /// instead of four separate 1-byte loads. The codes, their order and every
    /// arithmetic statement that consumes them are unchanged, so the arm is
    /// bit-identical to the promoted bodies by construction.
    ///
    /// Alignment holds for every address this arm forms. The quad-stream body
    /// advances `ws` by `out_row * in_vec_size_w + simd_lid * bytes_per_thread`
    /// once and by `block_size` per K block, and the fetch adds
    /// `row * in_vec_size_w`, so
    ///
    ///     addr - base = (out_row + row) * in_vec_size_w
    ///                 + bytes_per_thread * simd_lid
    ///                 + block_size * n
    ///
    /// with the body's own constants `bytes_per_thread == 4` and
    /// `block_size == 128`, both multiples of 4, and `in_vec_size_w == inDim`
    /// where `liveShape` admits only 2816 (= 4 * 704) and 2112 (= 4 * 528).
    /// Every term is a multiple of 4. `weight.dtype == .uint32` is required by
    /// `matmul`, so the buffer base is itself at least 4-byte aligned. The tail
    /// block forms the same addresses under a lane predicate, which restricts
    /// WHICH lanes load and never the address. The load reads exactly the bytes
    /// `wl[0 ..< 4]` the incumbent loop read, so no new byte is touched and the
    /// bounds are unchanged.
    ///
    /// `DARKBLOOM_GEMMA4_MLP_W4_LOAD=0` restores the promoted quad-stream and
    /// activation-sum bodies byte for byte in the same binary.
    public static let w4Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MLP_W4_LOAD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// DMLP-ROWS16. With speculation on, one decode round runs a single
    /// rectangular verify forward on `[B, 1 + k]` instead of the `[B, 1]`
    /// step. At the shipped depth (k = 1) that is `[8, 2]` -- SIXTEEN flat
    /// rows through the same dense MLP planes. The promoted bodies decline
    /// the rectangle on `x.dim(1) == sequence` alone and it falls to generic
    /// MLX, so the whole verify forward costs 82 ms against the 21 ms the
    /// eight-row step costs.
    ///
    /// This arm admits the rectangle to the SAME kernels at sixteen rows.
    /// Every row keeps its own accumulators and its own K walk, textually the
    /// walk the eight-row body runs, so each flat row's output is bit-identical
    /// to what the eight-row step produces for that row -- the property that
    /// lets a verify column cost zero token budget. The weight planes are read
    /// once for sixteen rows instead of once for eight.
    ///
    /// `DARKBLOOM_GEMMA4_DMLP_ROWS16=0` restores the incumbent decline for the
    /// sixteen-row shape in the same executable. The eight-row path does not
    /// read this switch, takes kernels whose names and source text this ticket
    /// does not touch, and is byte-identical to the base.
    public static let rows16Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DMLP_ROWS16"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// DMLP-ROWS16 receipt. `DARKBLOOM_GEMMA4_DMLP_ROWS16_XCHECK=1` re-runs
    /// every sixteen-row dense MLP call through the eight-row path twice --
    /// once on the rectangle's column-0 rows, once on its column-1 rows -- and
    /// reports the count of bf16 bit patterns that differ. Synchronous and
    /// diagnostic only; it adds dispatches, so it never shares a run with a
    /// timing measurement.
    private static let rows16CrossCheckEnabled: Bool =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_DMLP_ROWS16_XCHECK"] == "1"

    nonisolated(unsafe) private static var rows16CrossCheckCalls = 0

    /// Stable FNV-1a fingerprint of a kernel's own header and source text,
    /// emitted from INSIDE the closure that builds the text rather than beside
    /// the branch that selects it, so the mark is evidence that this exact text
    /// was handed to the Metal compiler. Armed by `MLXFAST_ENGAGE_MARKS`, inert
    /// otherwise.
    private static func sourceFingerprint(_ text: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        return String(hash, radix: 16)
    }

    @inline(__always)
    private static func markKernelSource(
        _ tag: String, header: String, source: String
    ) -> String {
        CBv2EngageMark.once(
            "\(tag) fp=\(sourceFingerprint(header + "\u{0}" + source))")
        return source
    }

    private static let batch = 8
    private static let sequence = 1
    /// The verify rectangle's column count at the shipped draft depth.
    private static let verifyColumns = 2
    private static let rows16 = 16
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
        /// Flat row count the table was built for: 8 for the decode step, 16
        /// for the `[8, 2]` verify rectangle. The table's row stride is this
        /// value, so a consumer that reads it with the wrong stride reads the
        /// wrong lane; `matmul` refuses a table whose count is not its own.
        fileprivate let rows: Int

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

    // BFILL: each B operand is filled immediately before the step that
    // consumes it instead of all eight ahead of the run, so one B is live
    // at a time rather than eight. `r0`/`r1` are not written between here
    // and the last step, and the accumulation order into `C` is unchanged.
    const uint4 wv = *((const device uint4*)(wrow + 64 * g));
    const float s = float(srow[g]);
    const float b = float(brow[g]);

    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    MMA8_SETB(B0, x, lo)
    MMA8_STEP8(B0, x, z, 0)
    MMA8_SETB(B1, x, hi)
    MMA8_STEP8(B1, x, z, 8)
    MMA8_SETB(B2, y, lo)
    MMA8_STEP8(B2, x, z, 16)
    MMA8_SETB(B3, y, hi)
    MMA8_STEP8(B3, x, z, 24)
    MMA8_SETB(B4, z, lo)
    MMA8_STEP8(B4, y, w, 0)
    MMA8_SETB(B5, z, hi)
    MMA8_STEP8(B5, y, w, 8)
    MMA8_SETB(B6, w, lo)
    MMA8_STEP8(B6, y, w, 16)
    MMA8_SETB(B7, w, hi)
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
        // DMLP-DOWN-CARRY-019: the weight-operand register carry the frontier
        // applied to the two attention matrix-unit tiers, on the dense down
        // plane. Every address here is a function of the group index alone, so
        // one group's codes, scale and bias stay resident in registers while
        // the next group's are read a whole iteration ahead of their use.
        // `g_next` is clamped to the last valid group, so the final look-ahead
        // re-reads a group already read and its value is discarded.
        //
        // The read stays at the statement it already occupied. Moving it to the
        // top of the body instead perturbs the close's floating-point
        // contraction and stops the output being bit-identical; keeping it here
        // leaves the emitted arithmetic word for word what the tip emits.
        //
        // DMLP-DOWN-CARRY2-021: the codes go TWO groups deep. One iteration of
        // this body is eight matrix-unit steps and a three-step shuffle chain,
        // which is well short of a device load's latency, so a single group of
        // look-ahead leaves the second half of that latency uncovered. The
        // scale and the bias stay one group ahead: they are two bytes each
        // against the codes' sixteen, they sit in the same cache lines as their
        // neighbours, and a second copy of them would cost registers for no
        // stream. `g` is monotone here and the loop count is a constant, so the
        // hand-off is register renaming in a fully unrolled body, not a copy.
        replaceOnce(
            """
              simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;
            """,
            with: """
              simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

              uint4 wv_next = *((const device uint4*)(wrow + 64 * g0));
              uint4 wv_next2 =
                  *((const device uint4*)(wrow + 64 * min(g0 + 1, G - 1)));
              T s_next = srow[g0];
              T b_next = brow[g0];
            """)
        replaceOnce(
            """
                const uint4 wv = *((const device uint4*)(wrow + 64 * g));
                const float s = float(srow[g]);
                const float b = float(brow[g]);
            """,
            with: """
                const uint4 wv = wv_next;
                const float s = float(s_next);
                const float b = float(b_next);
                const int g_next = min(g + 1, G - 1);
                wv_next = wv_next2;
                wv_next2 = *((const device uint4*)(wrow + 64 * min(g + 2, G - 1)));
                s_next = srow[g_next];
                b_next = brow[g_next];
            """)
        return result
    }()

    private static let mma8DownStaticKKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_mma8_affine8_g64_down_k2112_carry2_bfill_v4",
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

    /// The two quad-stream launch bodies, lifted out of the four
    /// `MLXFast.metalKernel` calls that already carried them so the sixteen-row
    /// twins can be DERIVED from them by one checked substitution instead of
    /// transcribed. The text is character for character the promoted text; the
    /// four eight-row kernels below still receive exactly the string they
    /// received before, under exactly the names they had.
    private static let quadStreamSource: String = """
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
            """

    private static let quadStreamXSumSource: String = """
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
            """

    /// DMLP-ROWS16: the sixteen-row launch body. One substitution -- the
    /// x-group bound -- and nothing else. `first_m` is still `tid.x * 4`, the
    /// four rows an x-group owns are still four adjacent flat rows at the same
    /// stride, and the impl call, its operands, its `in_vec_size` and its
    /// simd reductions are the promoted text. Groups 2 and 3 (`first_m` 8 and
    /// 12) did nothing but return on the eight-row grid; here they carry the
    /// rectangle's second half.
    private static func rows16Source(_ source: String) -> String {
        let old = "if (first_m >= 8) {"
        precondition(
            source.components(separatedBy: old).count == 2,
            "DMLP-ROWS16 quad-stream x-group bound is missing")
        return source.replacingOccurrences(of: old, with: "if (first_m >= 16) {")
    }

    /// DMLP-ROWS16: the activation-sum table's ROW STRIDE is the cohort row
    /// count, so a sixteen-row table is read with a stride of sixteen. The
    /// lane, the K block and the row a term is read from are unchanged, so the
    /// float this body consumes for a given row is the same float the eight-row
    /// body consumes for that row; only the address it sits at moves.
    private static func rows16XSumHeader(_ header: String) -> String {
        let old = "+ int(simd_lid)) * 8 + first_m"
        precondition(
            header.components(separatedBy: old).count == 9,
            "DMLP-ROWS16 xsum row stride markers are missing")
        return header.replacingOccurrences(
            of: old, with: "+ int(simd_lid)) * 16 + first_m")
    }

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_v2_unroll",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad", header: kernelHeader, source: quadStreamSource),
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// The promoted producer body, named so its sixteen-row twin can be
    /// derived from it by one checked substitution. Character for character
    /// the text the promoted kernel already carried.
    private static let activationSumSource: String = """
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
            """

    private static func rows16XSumProducerSource(_ source: String) -> String {
        let old = "xSums[(k_block * 32 + lane) * 8 + row] = sum;"
        precondition(
            source.components(separatedBy: old).count == 2,
            "DMLP-ROWS16 xsum producer store is missing")
        return source.replacingOccurrences(
            of: old, with: "xSums[(k_block * 32 + lane) * 16 + row] = sum;")
    }

    /// One exact float sum for each `(K/128 block, simd lane, cohort row)`.
    /// The expression is the bits==8 arm of stock `load_vector`: a float zero
    /// followed by four ascending `sum += bf16` operations. Row is the unit-
    /// stride dimension so one table serves both gate and up projections.
    private static let activationSumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_affine8_xsum_v2_unroll",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: markKernelSource(
            "dmlp-fp-xsum-producer", header: "", source: activationSumSource),
        ensureRowContiguous: true
    )

    /// DMLP-ROWS16 producer. The sum a row's lane accumulates is formed by the
    /// same four ascending `sum += bf16` operations over the same four
    /// activation values, so the float stored for a given row is bit-identical
    /// to the eight-row table's. Only the table's ROW STRIDE moves, from eight
    /// to sixteen, because sixteen rows now share one table.
    private static let activationSumRows16Kernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_b8_l2_dense_mlp_affine8_xsum_rows16_v1",
            inputNames: ["x"],
            outputNames: ["xSums"],
            source: markKernelSource(
                "dmlp-fp-xsum-producer-rows16", header: "",
                source: rows16XSumProducerSource(activationSumSource)),
            ensureRowContiguous: true
        )

    private static let activationSumQMVKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_xsum_v2_unroll",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad-xsum", header: activationSumKernelHeader, source: quadStreamXSumSource),
        header: activationSumKernelHeader,
        ensureRowContiguous: true
    )

    /// The `uchar4` twin of the promoted registered dot product. The
    /// accumulation text is identical; only the container of the four
    /// already-loaded weight codes changes, from four separate `thread uint8_t`
    /// to one `uchar4` that a single 32-bit device load fills.
    private static let wideDotHelper = """

template <typename U, int values_per_thread>
inline U qdot_affine8_registered_v4(
    const thread uchar4 w,
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

"""

    /// Rewrite a promoted quad-stream header into its W4 twin. Three
    /// single-purpose substitutions with checked occurrence counts, so any
    /// future drift in the donor body fails construction instead of silently
    /// transforming an unintended text.
    private static func w4Header(_ source: String) -> String {
        var text = source
        func replaceExactly(_ old: String, _ new: String, _ count: Int) {
            precondition(text.components(separatedBy: old).count == count + 1)
            text = text.replacingOccurrences(of: old, with: new)
        }
        replaceExactly(
            "qdot_affine8_registered<", "qdot_affine8_registered_v4<", 8)
        replaceExactly(
            "  thread uint8_t packed[results_per_simdgroup][bytes_per_thread];",
            "  thread uchar4 packed[results_per_simdgroup];",
            1)
        replaceExactly(
            "      const device uint8_t* wl = ws + row * in_vec_size_w;\n"
                + "      #pragma unroll\n"
                + "      for (int i = 0; i < bytes_per_thread; i++) {\n"
                + "        packed[row][i] = wl[i];\n"
                + "      }",
            "      packed[row] =\n"
                + "          *((const device uchar4*)(ws + row * in_vec_size_w));",
            2)
        replaceExactly(
            "template <typename U, int values_per_thread>\n"
                + "inline U qdot_affine8_registered(",
            wideDotHelper
                + "template <typename U, int values_per_thread>\n"
                + "inline U qdot_affine8_registered(",
            1)
        return text
    }

    private static let w4KernelHeader: String = w4Header(kernelHeader)

    private static let w4ActivationSumKernelHeader: String =
        w4Header(activationSumKernelHeader)

    /// The W4 twins. Source text is the promoted kernels' own, character for
    /// character; only the name and the header differ, so the grid, threadgroup
    /// shape, dispatch count, operands and bytes moved are all unchanged.
    private static let w4Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_w4_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad-w4", header: w4KernelHeader, source: quadStreamSource),
        header: w4KernelHeader,
        ensureRowContiguous: true
    )

    private static let w4ActivationSumQMVKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_dense_mlp_qmv_affine8_g64_quad_stream_xsum_w4_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad-xsum-w4", header: w4ActivationSumKernelHeader, source: quadStreamXSumSource),
        header: w4ActivationSumKernelHeader,
        ensureRowContiguous: true
    )


    // MARK: - DMLP-ROWS16 sixteen-row twins
    //
    // Four gate/up launches and one down launch, one per configuration the
    // eight-row road can be in. Each is a SEPARATE kernel with its own name, so
    // a name-keyed compiled-kernel cache can never serve one row count's body
    // to the other, and the eight-row kernels above keep their names and their
    // source text unchanged.

    private static let rows16XSumKernelHeader: String =
        rows16XSumHeader(activationSumKernelHeader)

    private static let w4Rows16XSumKernelHeader: String =
        rows16XSumHeader(w4ActivationSumKernelHeader)

    private static let rows16Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l2_dense_mlp_qmv_affine8_g64_quad_stream_rows16_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad-rows16", header: kernelHeader,
            source: rows16Source(quadStreamSource)),
        header: kernelHeader,
        ensureRowContiguous: true
    )

    private static let rows16ActivationSumQMVKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_b8_l2_dense_mlp_qmv_affine8_g64_quad_stream_xsum_rows16_v1",
            inputNames: ["x", "w", "scales", "biases", "xSums"],
            outputNames: ["y"],
            source: markKernelSource(
                "dmlp-fp-quad-xsum-rows16", header: rows16XSumKernelHeader,
                source: rows16Source(quadStreamXSumSource)),
            header: rows16XSumKernelHeader,
            ensureRowContiguous: true
        )

    private static let w4Rows16Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l2_dense_mlp_qmv_affine8_g64_quad_stream_w4_rows16_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-quad-w4-rows16", header: w4KernelHeader,
            source: rows16Source(quadStreamSource)),
        header: w4KernelHeader,
        ensureRowContiguous: true
    )

    private static let w4Rows16ActivationSumQMVKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_b8_l2_dense_mlp_qmv_affine8_g64_quad_stream_xsum_w4_rows16_v1",
            inputNames: ["x", "w", "scales", "biases", "xSums"],
            outputNames: ["y"],
            source: markKernelSource(
                "dmlp-fp-quad-xsum-w4-rows16", header: w4Rows16XSumKernelHeader,
                source: rows16Source(quadStreamXSumSource)),
            header: w4Rows16XSumKernelHeader,
            ensureRowContiguous: true
        )

    /// DMLP-ROWS16-DOWN. The down plane's matrix-unit body carries the cohort
    /// rows on the FRAGMENT COLUMN axis: lane `(fm, fn)` reads x rows `fn` and
    /// `fn + 1`, so one `simdgroup_float8x8` chain serves exactly eight rows.
    /// Sixteen rows are therefore TWO M tiles, and this body runs both from one
    /// weight fetch: the second tile's x pointers are the first tile's plus
    /// `8 * K`, and everything else about a tile -- its own `C`, its own run
    /// sums, its own pair of accumulators, its own half of the threadgroup
    /// reduction slab and its own close -- is private to it.
    ///
    /// Bit-exactness. Tile 0's statements are the donor's, character for
    /// character; tile 1's are the same statements over the same-shaped
    /// operands, and the two macros it expands through differ from the donor's
    /// only in taking the activation registers and the accumulator matrix as
    /// parameters instead of naming them. Both tiles walk the identical `g`
    /// range in the identical order, consume the same `wv`, `s` and `b` a
    /// register carry already produced, and close with the identical
    /// `acc += s * C.thread_elements()[i] + rs[i] * b`. Every output word is
    /// therefore the float sum the eight-row kernel accumulates for that row,
    /// in the same order, and the plane is streamed once instead of twice.
    private static let mma8DownStaticKRows16Header: String = {
        let marker =
            "template <typename T, int KS>\n"
            + "METAL_FUNC void gemma4_qmv_mma8_affine8_g64_down_k2112_impl("
        guard let range = mma8DownStaticKHeader.range(of: marker) else {
            preconditionFailure("DMLP-ROWS16 down donor body is missing")
        }
        var body = String(mma8DownStaticKHeader[range.lowerBound...])
        func replaceExactly(_ old: String, _ new: String, _ count: Int) {
            precondition(
                body.components(separatedBy: old).count == count + 1,
                "DMLP-ROWS16 down donor drift")
            body = body.replacingOccurrences(of: old, with: new)
        }
        replaceExactly(
            "gemma4_qmv_mma8_affine8_g64_down_k2112_impl(",
            "gemma4_qmv_mma8_affine8_g64_down_k2112_m16_impl(", 1)
        replaceExactly(
            "  const device T* x0 = x + c.fn * K + 8 * c.fm;\n"
                + "  const device T* x1 = x0 + K;\n",
            "  const device T* x0 = x + c.fn * K + 8 * c.fm;\n"
                + "  const device T* x1 = x0 + K;\n"
                + "  const device T* x0b = x0 + 8 * K;\n"
                + "  const device T* x1b = x1 + 8 * K;\n", 1)
        replaceExactly(
            "  float acc0 = 0.0f;\n  float acc1 = 0.0f;\n",
            "  float acc0 = 0.0f;\n  float acc1 = 0.0f;\n"
                + "  float acc0b = 0.0f;\n  float acc1b = 0.0f;\n", 1)
        replaceExactly(
            "    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));\n"
                + "    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));\n",
            "    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));\n"
                + "    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));\n"
                + "    const uint4 r0b = *((const device uint4*)(x0b + 64 * g));\n"
                + "    const uint4 r1b = *((const device uint4*)(x1b + 64 * g));\n",
            1)
        replaceExactly(
            "    float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));\n"
                + "    rs += simd_shuffle_xor(rs, 2u);\n"
                + "    rs += simd_shuffle_xor(rs, 4u);\n"
                + "    rs += simd_shuffle_xor(rs, 16u);\n",
            "    float2 rs = float2(mma8_runsum8<T>(r0), mma8_runsum8<T>(r1));\n"
                + "    rs += simd_shuffle_xor(rs, 2u);\n"
                + "    rs += simd_shuffle_xor(rs, 4u);\n"
                + "    rs += simd_shuffle_xor(rs, 16u);\n"
                + "    float2 rsb = float2(mma8_runsum8<T>(r0b), mma8_runsum8<T>(r1b));\n"
                + "    rsb += simd_shuffle_xor(rsb, 2u);\n"
                + "    rsb += simd_shuffle_xor(rsb, 4u);\n"
                + "    rsb += simd_shuffle_xor(rsb, 16u);\n", 1)
        replaceExactly(
            "    acc0 += s * C.thread_elements()[0] + rs.x * b;\n"
                + "    acc1 += s * C.thread_elements()[1] + rs.y * b;\n",
            "    acc0 += s * C.thread_elements()[0] + rs.x * b;\n"
                + "    acc1 += s * C.thread_elements()[1] + rs.y * b;\n"
                + "\n"
                + "    simdgroup_float8x8 Cb = simdgroup_float8x8(0.0f);\n"
                + "    MMA8_SETB_M(B0, r0b, r1b, x, lo)\n"
                + "    MMA8_STEP8_M(B0, Cb, x, z, 0)\n"
                + "    MMA8_SETB_M(B1, r0b, r1b, x, hi)\n"
                + "    MMA8_STEP8_M(B1, Cb, x, z, 8)\n"
                + "    MMA8_SETB_M(B2, r0b, r1b, y, lo)\n"
                + "    MMA8_STEP8_M(B2, Cb, x, z, 16)\n"
                + "    MMA8_SETB_M(B3, r0b, r1b, y, hi)\n"
                + "    MMA8_STEP8_M(B3, Cb, x, z, 24)\n"
                + "    MMA8_SETB_M(B4, r0b, r1b, z, lo)\n"
                + "    MMA8_STEP8_M(B4, Cb, y, w, 0)\n"
                + "    MMA8_SETB_M(B5, r0b, r1b, z, hi)\n"
                + "    MMA8_STEP8_M(B5, Cb, y, w, 8)\n"
                + "    MMA8_SETB_M(B6, r0b, r1b, w, lo)\n"
                + "    MMA8_STEP8_M(B6, Cb, y, w, 16)\n"
                + "    MMA8_SETB_M(B7, r0b, r1b, w, hi)\n"
                + "    MMA8_STEP8_M(B7, Cb, y, w, 24)\n"
                + "\n"
                + "    acc0b += s * Cb.thread_elements()[0] + rsb.x * b;\n"
                + "    acc1b += s * Cb.thread_elements()[1] + rsb.y * b;\n", 1)
        replaceExactly(
            "      red[simd_lid] = float2(acc0, acc1);\n",
            "      red[simd_lid] = float2(acc0, acc1);\n"
                + "      red[32 + simd_lid] = float2(acc0b, acc1b);\n", 1)
        replaceExactly(
            "    const float2 other = red[simd_lid];\n"
                + "    acc0 = acc0 + other.x;\n"
                + "    acc1 = acc1 + other.y;\n",
            "    const float2 other = red[simd_lid];\n"
                + "    acc0 = acc0 + other.x;\n"
                + "    acc1 = acc1 + other.y;\n"
                + "    const float2 otherb = red[32 + simd_lid];\n"
                + "    acc0b = acc0b + otherb.x;\n"
                + "    acc1b = acc1b + otherb.y;\n", 1)
        replaceExactly(
            "  y[c.fn * N + n0 + c.fm] = static_cast<T>(acc0);\n"
                + "  y[(c.fn + 1) * N + n0 + c.fm] = static_cast<T>(acc1);\n",
            "  y[c.fn * N + n0 + c.fm] = static_cast<T>(acc0);\n"
                + "  y[(c.fn + 1) * N + n0 + c.fm] = static_cast<T>(acc1);\n"
                + "  y[(8 + c.fn) * N + n0 + c.fm] = static_cast<T>(acc0b);\n"
                + "  y[(9 + c.fn) * N + n0 + c.fm] = static_cast<T>(acc1b);\n", 1)
        // The donor's own two macros with the activation registers and the
        // accumulator matrix as parameters. Expanded, tile 1's statements are
        // the donor's statements with `r0`/`r1`/`C` renamed.
        let macros = """

#define MMA8_SETB_M(BB, RA, RB, W, HI) BB.thread_elements()[0] = mma8_##HI<T>(RA.W); BB.thread_elements()[1] = mma8_##HI<T>(RB.W);
#define MMA8_STEP8_M(BB, CC, WLO, WHI, SH) A.thread_elements()[0] = float(extract_bits(wv.WLO, (SH), 8)); A.thread_elements()[1] = float(extract_bits(wv.WHI, (SH), 8)); simdgroup_multiply_accumulate(CC, A, BB, CC);

"""
        return mma8DownStaticKHeader + macros + body
    }()

    private static let mma8DownStaticKRows16Kernel = MLXFast.metalKernel(
        name: "cbv2_b8_l2_dense_mlp_mma8_affine8_g64_down_k2112_carry2_bfill_rows16_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: markKernelSource(
            "dmlp-fp-down-rows16", header: mma8DownStaticKRows16Header,
            source: """
                const uint3 tid = threadgroup_position_in_grid;
                threadgroup float2 red[64];
                gemma4_qmv_mma8_affine8_g64_down_k2112_m16_impl<T, 2>(
                    w, scales, biases, x, y,
                    w_shape[0], int(tid.y) * 8, red,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
                """),
        header: mma8DownStaticKRows16Header,
        ensureRowContiguous: true)

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
            x.dim(2) == 2816,
            let rows = flatRows(x)
        else { return nil }
        guard x.size == rows * 2816 else { return nil }
        let blocks = 2816 / kBlock
        let values = (rows == batch ? activationSumKernel : activationSumRows16Kernel)(
            [x],
            template: [("T", x.dtype)],
            grid: (simdWidth, blocks, rows),
            threadGroup: (simdWidth, 1, 1),
            outputShapes: [[blocks * simdWidth * rows]],
            outputDTypes: [.float32]
        )[0]
        return ActivationSums(values: values, rows: rows)
    }

    /// The flat row count this file will serve for `x`: `batch` for the decode
    /// step, `batch * verifyColumns` for the MTP verify rectangle, `nil` for
    /// every other length. The rectangle is admitted only while DMLP-ROWS16 is
    /// on; with the switch off this returns `nil` for it and the caller keeps
    /// the incumbent decline.
    @inline(__always)
    private static func flatRows(_ x: MLXArray) -> Int? {
        if x.dim(1) == sequence { return batch }
        if rows16Enabled && x.dim(1) == verifyColumns { return batch * verifyColumns }
        return nil
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
        return ActivationSums(values: values, rows: batch)
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
            let rows = flatRows(x)
        else { return nil }
        let columns = x.dim(1)

        let inDim = x.dim(2)
        guard weight.ndim == 2 else { return nil }
        let outDim = weight.dim(0)
        guard liveShape(inDim: inDim, outDim: outDim),
            inDim % Self.groupSize == 0,
            outDim % outputsPerGroup == 0,
            x.size == rows * inDim,
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
            if rows != batch {
                // DMLP-ROWS16-DOWN. Only the SHIPPED matrix-unit arm has a
                // sixteen-row twin: the down plane's static-K carry body. The
                // opt-in gate/up arm and the opt-in lane-sum arm have no
                // sixteen-row body, so the rectangle declines here and keeps
                // the stock path for them, exactly as it did before this
                // ticket. Neither is the scored configuration.
                guard !isGateUp, !mma8DownLaneSumsEnabled, mma8DownStaticKEnabled
                else { return nil }
                CBv2EngageMark.once("dmlp-rows16-down")
                let wide = mma8DownStaticKRows16Kernel(
                    [x, weight, scales, biases],
                    template: [("T", x.dtype)],
                    grid: (simdWidth, yTiles * simdGroups, 1),
                    threadGroup: (simdWidth, simdGroups, 1),
                    outputShapes: [[batch, columns, outDim]],
                    outputDTypes: [x.dtype]
                )[0]
                if rows16CrossCheckEnabled {
                    rows16CrossCheck(
                        wide: wide, x: x, weight: weight, scales: scales,
                        biases: biases, inDim: inDim, outDim: outDim,
                        plane: "down-mma8")
                }
                return wide
            }
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

        let xGroups = rows / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        // A table built for one row count is read with that count as its row
        // stride, so a table whose count is not this call's is refused and the
        // call keeps the sum-free body.
        let useActivationSums = activationSums?.rows == rows
            && inDim == 2816
            && outDim == 2112
        if w4Enabled {
            CBv2EngageMark.once("mlp-w4-load")
        }
        let wide = rows != batch
        if wide {
            CBv2EngageMark.once(
                useActivationSums ? "dmlp-rows16-xsum" : "dmlp-rows16")
        }
        let selected: MLXFast.MLXFastKernel
        if useActivationSums {
            if wide {
                selected = w4Enabled
                    ? w4Rows16ActivationSumQMVKernel : rows16ActivationSumQMVKernel
            } else {
                selected = w4Enabled ? w4ActivationSumQMVKernel : activationSumQMVKernel
            }
        } else {
            if wide {
                selected = w4Enabled ? w4Rows16Kernel : rows16Kernel
            } else {
                selected = w4Enabled ? w4Kernel : kernel
            }
        }
        let inputs = useActivationSums
            ? [x, weight, scales, biases, activationSums!.values]
            : [x, weight, scales, biases]
        let y = selected(
            inputs,
            template: [("T", x.dtype)],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, columns, outDim]],
            outputDTypes: [x.dtype]
        )[0]
        if wide && rows16CrossCheckEnabled {
            rows16CrossCheck(
                wide: y, x: x, weight: weight, scales: scales, biases: biases,
                inDim: inDim, outDim: outDim,
                plane: isGateUp ? "gateup-quad" : "down-quad")
        }
        return y
    }

    /// DMLP-ROWS16 receipt. Re-run this sixteen-row call through the eight-row
    /// road twice -- once on the rectangle's column-0 rows (flat rows 0, 2, ...,
    /// 14), once on its column-1 rows -- and report how many bf16 BIT PATTERNS
    /// of the sixteen-row output differ from the eight-row output for the same
    /// rows. The inner calls carry `x.dim(1) == 1`, so they take the untouched
    /// eight-row kernels and cannot re-enter this check. Synchronous and
    /// diagnostic; it adds dispatches and never shares a run with a timing
    /// measurement.
    private static func rows16CrossCheck(
        wide: MLXArray,
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        inDim: Int,
        outDim: Int,
        plane: String
    ) {
        let flatX = x.reshaped([rows16, inDim])
        let flatY = wide.reshaped([rows16, outDim])
        var report =
            "DMLP-ROWS16 xcheck call=\(rows16CrossCheckCalls) plane=\(plane)"
        for (label, start) in [("col0", 0), ("col1", 1)] {
            let index = MLXArray((0 ..< batch).map { Int32(start + 2 * $0) })
            let narrowX = flatX[index].reshaped([batch, sequence, inDim])
            guard let narrowY = matmul(
                x: narrowX,
                weight: weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                activationSums: activationSums(for: narrowX))
            else {
                report += " \(label)_UNAVAILABLE"
                continue
            }
            let wideBits = flatY[index].view(dtype: .uint16)
            let narrowBits = narrowY.reshaped([batch, outDim]).view(dtype: .uint16)
            let mismatches = MLX.sum(MLX.notEqual(wideBits, narrowBits)).item(Int32.self)
            report += " \(label)_mismatch=\(mismatches)/\(narrowBits.size)"
        }
        rows16CrossCheckCalls += 1
        FileHandle.standardError.write(Data((report + "\n").utf8))
    }
}
