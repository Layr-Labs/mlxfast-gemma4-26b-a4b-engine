import Foundation
import MLX
import MLXFast

/// MOE-DOWN-OCTET-001: the routed-expert DOWN projection at the batch-8
/// decode geometry, launched from Swift with a grid the frozen host dispatch
/// cannot express.
///
/// Geometry (Gemma 4 26B A4B, exact-match gated below): 64 sorted
/// assignments (8 streams x top-8), `x` [64, 1, 704] bf16 activations in
/// assignment order, `w` [128, 2816, 88] uint32 affine-4 g64, `scales` /
/// `biases` [128, 2816, 11] bf16, sorted raw expert keys [64] uint32.
///
/// What changes against the incumbent `affine_gather_qmv` down-tile arm
/// (`quantized.h`, `gather_qmv_gemma4_down_tile`):
///
/// 1. GRID. The host launches (1, 352, 64) threadgroups and the tile arm
///    returns from three of every four before reading anything. Here the grid
///    is (1, 88, 64): one threadgroup per (span-4 tile group, assignment), so
///    the 16,896 exiting groups per layer never get scheduled.
/// 2. OCTET REUSE. The incumbent pair arm serves at most two same-expert
///    assignments per weight stream; a run of three or four pays the expert's
///    down plane twice. The leader of a run here walks up to eight input rows
///    through ONE weight fetch, so every expert's down plane is read exactly
///    once per layer per round.
/// 3. PIPELINED LOADS. The 12-step (4 tiles x 3 K-blocks) walk keeps two
///    register sets for packed words, scales and biases and issues step i+1's
///    loads before step i's arithmetic. Loads-only rescheduling.
///
/// Exactness. Every (assignment, output row) keeps its own fp32 accumulator,
/// the same 32-lane x 8-value K decomposition, the same block order
/// (256, 256, 192-tail on lanes 0..23), the same `qdot` expression
/// (`qdot_affine4_registered_word`, the per-stream arm of
/// `qdot_affine4_pair_word` verbatim), the same `simd_sum`, and the same bf16
/// store as the incumbent pair / stock arms. Only which threadgroup performs
/// the work, how many x rows share a packed-word load, and when loads are
/// issued differ. Verified uint16-exact against `gatherQuantizedMM` on the
/// production geometry for run lengths 1 through 8
/// (`Gemma4MoEDownGatherOctetParityTests`).
///
/// Kill switch: `DARKBLOOM_GEMMA4_MOE_DOWN_OCTET=0` returns the plane to the
/// incumbent gather. Engage mark: `moe-down-octet`.
public enum Gemma4MoEDownGatherOctetV1 {
    public static let kernelName = "cbv2_b8_moe_down_gather_octet_q4g64_span4_v1"

    static let assignments = 64
    static let inVec = 704
    static let outVec = 2816
    static let experts = 128
    static let span = 4
    static let tileGroups = outVec / (span * 8)  // 88

    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MOE_DOWN_OCTET"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let kernelHeader = """
        // ---- Verbatim arithmetic from quantized.h (bits == 4 arms) ----
        template <typename T>
        inline float gemma4_octet_load_x8(const device T* x, thread float* x_thread) {
          float sum = 0;
          for (int i = 0; i < 8; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float gemma4_octet_qdot_word(
            uint packed_word,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const uint packed0 = packed_word & 0xffffu;
          const uint packed1 = packed_word >> 16;
          float accum =
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

        // One run leader: walks `span` consecutive 8-row output tiles for up to
        // eight same-expert input rows through one weight stream per tile.
        template <typename T>
        METAL_FUNC void gemma4_moe_down_gather_octet_impl(
            const device uint8_t* ws,
            const device T* scales,
            const device T* biases,
            const device T* x,
            device T* y,
            const uint run_len,
            const uint tile_group,
            const uint simd_gid,
            const uint simd_lid) {
          constexpr int span = 4;
          constexpr int rows_per_simd = 4;
          constexpr int steps = 3;
          constexpr int in_vec_size_w = 352;
          constexpr int in_vec_size_g = 11;
          constexpr int in_vec_size = 704;
          constexpr int out_vec_size = 2816;
          constexpr int tail_lanes = 24;
          constexpr int max_streams = 8;

          const int lane_w = int(simd_lid) * 4;
          const int lane_g = int(simd_lid) / 8;
          const int lane_x = int(simd_lid) * 8;
          const int row0 = int(tile_group) * (span * 8) + int(simd_gid) * rows_per_simd;

          thread float x_thread[8];
          thread float result[max_streams][rows_per_simd];
          thread uint packed_a[rows_per_simd];
          thread float scale_a[rows_per_simd];
          thread float bias_a[rows_per_simd];
          thread uint packed_b[rows_per_simd];
          thread float scale_b[rows_per_simd];
          thread float bias_b[rows_per_simd];

          {
            const device uint8_t* wrow = ws + row0 * in_vec_size_w + lane_w;
            const device T* srow = scales + row0 * in_vec_size_g + lane_g;
            const device T* brow = biases + row0 * in_vec_size_g + lane_g;
            #pragma clang loop unroll(full)
            for (int r = 0; r < rows_per_simd; ++r) {
              packed_a[r] = *((const device uint*)(wrow + r * in_vec_size_w));
              scale_a[r] = srow[r * in_vec_size_g];
              bias_a[r] = brow[r * in_vec_size_g];
            }
          }

          #pragma clang loop unroll(full)
          for (int t = 0; t < span; ++t) {
            #pragma clang loop unroll(full)
            for (int s = 0; s < max_streams; ++s) {
              #pragma clang loop unroll(full)
              for (int r = 0; r < rows_per_simd; ++r) {
                result[s][r] = 0.0f;
              }
            }
            const int tile_row = row0 + t * 8;
            #pragma clang loop unroll(full)
            for (int blk = 0; blk < steps; ++blk) {
              const int step = t * steps + blk;
              const bool use_a = (step & 1) == 0;
              const int next = step + 1;
              if (next < span * steps) {
                const int nt = next / steps;
                const int nb = next - nt * steps;
                const bool next_active = (nb < 2) || (int(simd_lid) < tail_lanes);
                if (next_active) {
                  const int nrow = row0 + nt * 8;
                  const device uint8_t* wrow = ws + nrow * in_vec_size_w + nb * 128 + lane_w;
                  const device T* srow = scales + nrow * in_vec_size_g + nb * 4 + lane_g;
                  const device T* brow = biases + nrow * in_vec_size_g + nb * 4 + lane_g;
                  if (use_a) {
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < rows_per_simd; ++r) {
                      packed_b[r] = *((const device uint*)(wrow + r * in_vec_size_w));
                      scale_b[r] = srow[r * in_vec_size_g];
                      bias_b[r] = brow[r * in_vec_size_g];
                    }
                  } else {
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < rows_per_simd; ++r) {
                      packed_a[r] = *((const device uint*)(wrow + r * in_vec_size_w));
                      scale_a[r] = srow[r * in_vec_size_g];
                      bias_a[r] = brow[r * in_vec_size_g];
                    }
                  }
                }
              }
              const bool active = (blk < 2) || (int(simd_lid) < tail_lanes);
              if (active) {
                const device T* xblk = x + blk * 256 + lane_x;
                #pragma clang loop unroll(full)
                for (int s = 0; s < max_streams; ++s) {
                  if (uint(s) < run_len) {
                    const float sum = gemma4_octet_load_x8<T>(xblk + s * in_vec_size, x_thread);
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < rows_per_simd; ++r) {
                      const uint pw = use_a ? packed_a[r] : packed_b[r];
                      const float sc = use_a ? scale_a[r] : scale_b[r];
                      const float bs = use_a ? bias_a[r] : bias_b[r];
                      result[s][r] += gemma4_octet_qdot_word(pw, x_thread, sc, bs, sum);
                    }
                  }
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int s = 0; s < max_streams; ++s) {
              if (uint(s) < run_len) {
                device T* yrow = y + s * out_vec_size + tile_row;
                #pragma clang loop unroll(full)
                for (int r = 0; r < rows_per_simd; ++r) {
                  const float v = simd_sum(result[s][r]);
                  if (simd_lid == 0) {
                    yrow[r] = static_cast<T>(v);
                  }
                }
              }
            }
          }
        }
        """

    static let kernelSource = """
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const uint assignment = tid.z;
        const uint expert = keys[assignment];
        // Position inside the sorted same-expert run. Followers of an octet
        // leader return; every eighth position starts a fresh octet so a run
        // longer than eight (impossible at top-8 over 8 rows, tolerated anyway)
        // is still fully written.
        uint run_offset = 0u;
        for (uint prior = assignment; prior > 0u; --prior) {
          if (keys[prior - 1u] != expert) {
            break;
          }
          run_offset += 1u;
        }
        if ((run_offset & 7u) != 0u) {
          return;
        }
        uint run_len = 1u;
        while (run_len < 8u && assignment + run_len < 64u &&
               keys[assignment + run_len] == expert) {
          run_len += 1u;
        }
        const device uint8_t* ws =
            ((const device uint8_t*)w) + ulong(expert) * ulong(2816 * 352);
        const device T* es = scales + ulong(expert) * ulong(2816 * 11);
        const device T* eb = biases + ulong(expert) * ulong(2816 * 11);
        gemma4_moe_down_gather_octet_impl<T>(
            ws,
            es,
            eb,
            x + ulong(assignment) * 704ul,
            y + ulong(assignment) * 2816ul,
            run_len,
            tid.y,
            simd_gid,
            simd_lid);
        """

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: kernelName,
        inputNames: ["x", "w", "scales", "biases", "keys"],
        outputNames: ["y"],
        source: kernelSource,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// Returns `nil` unless every production pin holds; the caller then issues
    /// the incumbent gather unchanged.
    public static func matmul(
        activated x: MLXArray,
        sortedKeys keys: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        initNaN: Bool = false
    ) -> MLXArray? {
        guard enabled,
            x.dtype == .bfloat16, x.ndim == 3,
            x.dim(0) == assignments, x.dim(1) == 1, x.dim(2) == inVec,
            keys.dtype == .uint32, keys.ndim == 1, keys.size == assignments,
            weight.dtype == .uint32, weight.shape == [experts, outVec, inVec / 8],
            scales.dtype == .bfloat16, scales.shape == [experts, outVec, inVec / 64],
            biases.dtype == .bfloat16, biases.shape == [experts, outVec, inVec / 64]
        else { return nil }
        CBv2EngageMark.once("moe-down-octet")
        let x2 = x.reshaped([assignments, inVec])
        let y = kernel(
            [x2, weight, scales, biases, keys],
            template: [("T", DType.bfloat16)],
            grid: (64, tileGroups, assignments),
            threadGroup: (64, 1, 1),
            outputShapes: [[assignments, outVec]],
            outputDTypes: [.bfloat16],
            initValue: initNaN ? Float.nan : nil
        )[0]
        return y.reshaped([assignments, 1, outVec])
    }
}
