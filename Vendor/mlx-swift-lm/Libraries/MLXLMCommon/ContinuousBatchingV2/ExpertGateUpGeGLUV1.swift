// ExpertGateUpGeGLUV1.swift
//
// EXPERT-GATEUP-GEGLU. At the batch-eight decode cell the routed-expert road
// runs three dispatches before the down projection: the gate gather, the up
// gather, and the GeGLU product. The two gathers are independent and the
// concurrent encoder already overlaps them, but the product is a genuine
// read-after-write stage — it cannot start until both have landed and the down
// gather cannot start until it has — so it is a global buffer barrier wrapped
// around a kernel that moves only 270 KB.
//
// This kernel produces the activation directly. It gathers over the PAIRED
// gate|up right-hand side that `SwitchGateUpFusedStorage` already builds and
// keeps resident for the prompt pass (`[128, 1408, 352]`, alternating 16-column
// gate and up blocks), so one threadgroup owns a gate band and the matching up
// band, and closes the product in its own epilogue. The two `[64, 1, 704]`
// projection planes are never written and never read back.
//
// WHY THE PRODUCER AND NOT THE CONSUMER. Folding the same product into the DOWN
// gather's activation load was built and measured first, and refuted: the down
// projection reads each activation value once per output tile (352 of them), so
// the tape would run 352 times per element instead of once. That arm measured
// 39% slower on an isolated probe. A producer emits each element exactly once,
// which is why the fold belongs here.
//
// EXACTNESS. Each of the two contraction chains is the shipped
// `qmv_affine4_g64_pair_impl` / `qmv_impl` arithmetic: the same lane -> K
// mapping, the same `load_vector` bits == 4 transform, the same eight-term
// `qdot` in the same 4 + 4 grouping, the same per-(output row, assignment)
// accumulator, the same `scale * accum + sum * bias` close and the same
// `simd_sum`. The chains are independent, so interleaving them adds no term to
// either. Both accumulators are rounded with the same `static_cast<T>` the
// split gathers' stores apply, so the operands entering the tape are the words
// the incumbent wrote, and the tape is the compiled decode product's own.
//
// The incumbent gathers elect a leader at every even `run_offset` and serve the
// pair from one weight stream; that election is reproduced here, and it is
// worth roughly a quarter of the gate/up weight traffic at this cohort's
// routing. Every arm of that family produces the same words as stock
// `qmv_impl`, so reproducing one arm is bit-identical to whichever arm the
// incumbent selected.

import Foundation
import MLX
import MLXFast

public enum CBv2ExpertGateUpGeGLUV1 {

    /// Kill switch. Off returns nil, so the kernel is never built, no dispatch
    /// presents its geometry, and the incumbent two-gathers-plus-product road
    /// runs byte for byte.
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_EXPERT_GATEUP_GEGLU"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    private static let experts = 128
    private static let assignments = 64
    private static let rows = 8
    private static let inDim = 2816  // K
    private static let hidden = 704  // activated width
    private static let pairedOut = 1408
    private static let groupSize = 64
    private static let bits = 4
    private static let simdWidth = 32
    private static let simdGroups = 2
    /// One threadgroup per eight-column tile of the ACTIVATED output.
    private static let tiles = 88  // 704 / 8

    private static let header = """
        #include <metal_stdlib>
        using namespace metal;

        #ifndef METAL_FUNC
        #define METAL_FUNC inline
        #endif

        // The compiled decode GeGLU tape (`gemma4SafeGeluProductShaped`).
        template <typename T>
        inline T gemma4_egu_tape(T gate, T up) {
          const T cubic_0 = static_cast<T>(static_cast<T>(0.044715f) * gate);
          const T cubic_1 = static_cast<T>(cubic_0 * gate);
          const T cubic_2 = static_cast<T>(cubic_1 * gate);
          const T inner = static_cast<T>(gate + cubic_2);
          const T scaled =
              static_cast<T>(static_cast<T>(0.7978845608028654f) * inner);
          const T curved = metal::precise::tanh(scaled);
          const T shifted = static_cast<T>(static_cast<T>(1.0f) + curved);
          const T half_gate = static_cast<T>(static_cast<T>(0.5f) * gate);
          const T gelu = static_cast<T>(half_gate * shifted);
          return static_cast<T>(gelu * up);
        }

        // `load_vector<T, float, 8, 4>`'s bits == 4 arm, verbatim.
        template <typename T>
        inline float gemma4_egu_load_vector4(
            const device T* x, thread float* x_thread) {
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

        // `qdot_affine4_registered_word`, verbatim.
        template <typename U>
        inline U gemma4_egu_qdot_word(
            uint packedWord, const thread U* x, U scale, U bias, U sum) {
          const uint packed0 = packedWord & 0xffffu;
          const uint packed1 = packedWord >> 16;
          U accum =
              (x[0] * (packed0 & 0x000f) +
               x[1] * (packed0 & 0x00f0) +
               x[2] * (packed0 & 0x0f00) +
               x[3] * (packed0 & 0xf000));
          accum +=
              (x[4] * (packed1 & 0x000f) +
               x[5] * (packed1 & 0x00f0) +
               x[6] * (packed1 & 0x0f00) +
               x[7] * (packed1 & 0xf000));
          return scale * accum + sum * bias;
        }

        // One assignment: the gate band and the matching up band walked
        // together, product closed per output column.
        template <typename T>
        METAL_FUNC void gemma4_egu_single(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            device T* y0,
            const int gate_row,
            const int out_col,
            uint simd_gid, uint simd_lid) {
          constexpr int results_per_simdgroup = 4;
          constexpr int values_per_thread = 8;
          constexpr int block_size = values_per_thread * 32;
          constexpr int bytes_per_thread = 4;
          constexpr int scale_step_per_thread = 8;
          constexpr int K = 2816;
          constexpr int in_vec_size_w = K / 2;
          constexpr int in_vec_size_g = K / 64;
          constexpr int UP_OFFSET = 16;

          const int gr = gate_row + simd_gid * results_per_simdgroup;
          const int ur = gr + UP_OFFSET;
          const device uint8_t* wg =
              (const device uint8_t*)w + gr * in_vec_size_w
              + simd_lid * bytes_per_thread;
          const device uint8_t* wu =
              (const device uint8_t*)w + ur * in_vec_size_w
              + simd_lid * bytes_per_thread;
          const device T* sg =
              scales + gr * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* bg =
              biases + gr * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* su =
              scales + ur * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* bu =
              biases + ur * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* x = x0 + simd_lid * values_per_thread;

          thread float x_thread[values_per_thread];
          thread float gacc[results_per_simdgroup] = {0};
          thread float uacc[results_per_simdgroup] = {0};

          for (int k = 0; k <= K - block_size; k += block_size) {
            float sum = gemma4_egu_load_vector4<T>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              const uint pg = *((const device uint*)(wg + row * in_vec_size_w));
              const uint pu = *((const device uint*)(wu + row * in_vec_size_w));
              gacc[row] += gemma4_egu_qdot_word<float>(
                  pg, x_thread, float(sg[row * in_vec_size_g]),
                  float(bg[row * in_vec_size_g]), sum);
              uacc[row] += gemma4_egu_qdot_word<float>(
                  pu, x_thread, float(su[row * in_vec_size_g]),
                  float(bu[row * in_vec_size_g]), sum);
            }
            wg += block_size / 2;
            wu += block_size / 2;
            sg += block_size / 64;
            bg += block_size / 64;
            su += block_size / 64;
            bu += block_size / 64;
            x += block_size;
          }

          for (int row = 0; row < results_per_simdgroup; row++) {
            const float g = simd_sum(gacc[row]);
            const float u = simd_sum(uacc[row]);
            if (simd_lid == 0) {
              y0[out_col + simd_gid * results_per_simdgroup + row] =
                  gemma4_egu_tape<T>(static_cast<T>(g), static_cast<T>(u));
            }
          }
        }

        // Two assignments sharing one weight stream, each with its own
        // accumulators and its own K chain.
        template <typename T>
        METAL_FUNC void gemma4_egu_pair(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            device T* y0,
            device T* y1,
            const int gate_row,
            const int out_col,
            uint simd_gid, uint simd_lid) {
          constexpr int results_per_simdgroup = 4;
          constexpr int values_per_thread = 8;
          constexpr int block_size = values_per_thread * 32;
          constexpr int bytes_per_thread = 4;
          constexpr int scale_step_per_thread = 8;
          constexpr int K = 2816;
          constexpr int in_vec_size_w = K / 2;
          constexpr int in_vec_size_g = K / 64;
          constexpr int UP_OFFSET = 16;

          const int gr = gate_row + simd_gid * results_per_simdgroup;
          const int ur = gr + UP_OFFSET;
          const device uint8_t* wg =
              (const device uint8_t*)w + gr * in_vec_size_w
              + simd_lid * bytes_per_thread;
          const device uint8_t* wu =
              (const device uint8_t*)w + ur * in_vec_size_w
              + simd_lid * bytes_per_thread;
          const device T* sg =
              scales + gr * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* bg =
              biases + gr * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* su =
              scales + ur * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* bu =
              biases + ur * in_vec_size_g + simd_lid / scale_step_per_thread;
          const device T* xa = x0 + simd_lid * values_per_thread;
          const device T* xb = x1 + simd_lid * values_per_thread;

          thread float xa_thread[values_per_thread];
          thread float xb_thread[values_per_thread];
          thread float gacc0[results_per_simdgroup] = {0};
          thread float uacc0[results_per_simdgroup] = {0};
          thread float gacc1[results_per_simdgroup] = {0};
          thread float uacc1[results_per_simdgroup] = {0};

          for (int k = 0; k <= K - block_size; k += block_size) {
            float sum0 = gemma4_egu_load_vector4<T>(xa, xa_thread);
            float sum1 = gemma4_egu_load_vector4<T>(xb, xb_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              const uint pg = *((const device uint*)(wg + row * in_vec_size_w));
              const uint pu = *((const device uint*)(wu + row * in_vec_size_w));
              const float sgv = float(sg[row * in_vec_size_g]);
              const float bgv = float(bg[row * in_vec_size_g]);
              const float suv = float(su[row * in_vec_size_g]);
              const float buv = float(bu[row * in_vec_size_g]);
              gacc0[row] += gemma4_egu_qdot_word<float>(
                  pg, xa_thread, sgv, bgv, sum0);
              gacc1[row] += gemma4_egu_qdot_word<float>(
                  pg, xb_thread, sgv, bgv, sum1);
              uacc0[row] += gemma4_egu_qdot_word<float>(
                  pu, xa_thread, suv, buv, sum0);
              uacc1[row] += gemma4_egu_qdot_word<float>(
                  pu, xb_thread, suv, buv, sum1);
            }
            wg += block_size / 2;
            wu += block_size / 2;
            sg += block_size / 64;
            bg += block_size / 64;
            su += block_size / 64;
            bu += block_size / 64;
            xa += block_size;
            xb += block_size;
          }

          for (int row = 0; row < results_per_simdgroup; row++) {
            const float g0 = simd_sum(gacc0[row]);
            const float u0 = simd_sum(uacc0[row]);
            const float g1 = simd_sum(gacc1[row]);
            const float u1 = simd_sum(uacc1[row]);
            if (simd_lid == 0) {
              const int c = out_col + simd_gid * results_per_simdgroup + row;
              y0[c] = gemma4_egu_tape<T>(
                  static_cast<T>(g0), static_cast<T>(u0));
              y1[c] = gemma4_egu_tape<T>(
                  static_cast<T>(g1), static_cast<T>(u1));
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_b8_expert_gateup_geglu_affine4_g64_k2816_v1",
        inputNames: ["x", "w", "scales", "biases", "lhsIndices", "rhsIndices"],
        outputNames: ["y"],
        source: """
            constexpr int K = 2816;
            constexpr int H = 704;
            constexpr int W_EXPERT = 1408 * (K / 8);
            constexpr int G_EXPERT = 1408 * (K / 64);

            const uint3 tid = threadgroup_position_in_grid;
            const uint assignment = tid.z;
            const uint32_t route_word = rhsIndices[assignment];
            const bool tagged = (route_word & 0x80000000u) != 0u;
            const uint32_t expert = tagged ? (route_word & 0xffu) : route_word;
            uint run_offset = 0;
            if (tagged) {
                run_offset = (route_word >> 8) & 0x3fu;
            } else {
                for (uint prior = assignment; prior > 0; --prior) {
                    if (rhsIndices[prior - 1] != expert) { break; }
                    run_offset++;
                }
            }
            if ((run_offset & 1) != 0) {
                return;
            }
            // Tile j owns activated columns 8j..8j+7. Those live in the paired
            // plane's gate band at rows 32*(j/2) + 8*(j%2), with the matching
            // up band 16 rows later.
            const int j = int(tid.y);
            const int gate_row = 32 * (j / 2) + 8 * (j % 2);
            const int out_col = 8 * j;

            const device uint32_t* tw = w + expert * W_EXPERT;
            const device T* ts = scales + expert * G_EXPERT;
            const device T* tb = biases + expert * G_EXPERT;
            const device T* x0 = x + lhsIndices[assignment] * K;
            device T* y0 = y + assignment * H;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const bool has_pair = tagged
                ? ((((route_word >> 14) & 0x3fu) + 1u) > 1u)
                : (assignment + 1 < 64 && rhsIndices[assignment + 1] == expert);
            if (has_pair) {
                const device T* x1 = x + lhsIndices[assignment + 1] * K;
                device T* y1 = y + (assignment + 1) * H;
                gemma4_egu_pair<T>(
                    tw, ts, tb, x0, x1, y0, y1, gate_row, out_col,
                    simd_gid, simd_lid);
                return;
            }
            gemma4_egu_single<T>(
                tw, ts, tb, x0, y0, gate_row, out_col, simd_gid, simd_lid);
            """,
        header: header,
        ensureRowContiguous: true)

    /// The routed-expert gate and up projections with the GeGLU closed in the
    /// projection's own epilogue: `[64, 1, 704]` activated rows, no split
    /// projection planes and no product dispatch. Returns nil for anything that
    /// is not the pinned decode geometry.
    public static func apply(
        x: MLXArray,
        pairedWeight: MLXArray,
        pairedScales: MLXArray,
        pairedBiases: MLXArray,
        lhsIndices: MLXArray?,
        rhsIndices: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        guard enabled,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let lhsIndices,
            x.dtype == .bfloat16,
            pairedScales.dtype == .bfloat16,
            pairedBiases.dtype == .bfloat16,
            pairedWeight.dtype == .uint32,
            lhsIndices.dtype == .uint32,
            rhsIndices.dtype == .uint32,
            x.ndim == 3, x.dim(0) == rows, x.dim(1) == 1, x.dim(2) == inDim,
            pairedWeight.shape == [experts, pairedOut, inDim * Self.bits / 32],
            pairedScales.shape == [experts, pairedOut, inDim / Self.groupSize],
            pairedBiases.shape == pairedScales.shape,
            lhsIndices.ndim == 1, lhsIndices.size == assignments,
            rhsIndices.ndim == 1, rhsIndices.size == assignments
        else { return nil }

        CBv2EngageMark.once("expert-gateup-geglu")
        return kernel(
            [x, pairedWeight, pairedScales, pairedBiases, lhsIndices, rhsIndices],
            template: [("T", x.dtype)],
            grid: (simdWidth, tiles * simdGroups, assignments),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[assignments, 1, hidden]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
