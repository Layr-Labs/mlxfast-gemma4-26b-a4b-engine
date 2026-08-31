// PREFILL-GLUE-001: single-dispatch fusion of the MoE decoder layer's serial
// norm/residual glue on the PREFILL plane.
//
// Every MoE layer runs a strictly serial chain of full-width elementwise and
// row-reduction passes between its matmuls (`Gemma4DecoderLayer`):
//
//     postAttn  = postAttentionLayernorm(attnOut)      //  \  2 dispatches
//     out       = residual + postAttn                  //  /
//     h1        = preFeedforwardLayernorm(out)         //  \  2 dispatches over
//     h2        = preFeedforwardLayernorm2(out)        //  /  the SAME input
//     ...
//     h1        = postFeedforwardLayernorm1(h1)        //  \
//     h2        = postFeedforwardLayernorm2(h2)        //   |
//     out       = h1 + h2                              //   >  5 dispatches
//     out       = postFeedforwardLayernorm(out)        //   |
//     out       = residual2 + out                      //  /
//
// At DECODE those nine dispatches move ~45 KB each and are pure launch latency,
// which the concurrent encoder largely hides behind the expert branch. At
// PREFILL the same nine run on `[1, chunk, 2816]`: at chunk 1024 that is 5.8 MB
// per tensor, so the chain is bandwidth-bound, sits on the dependent critical
// path, and hides behind nothing.
//
// Measured on this machine at the production prefill row width, one MoE layer's
// worth of the stock chain: 243 us at chunk 512, 494 us at chunk 1024, 989 us at
// chunk 2048 -- about 300 GB/s, i.e. saturated. Multiplied by 30 MoE layers that
// is 14.8 ms per 1024-token chunk of pure glue.
//
// This file collapses the nine into three. Nothing here is a new algorithm: the
// arithmetic is `rms_single_row` reproduced verbatim, and the only thing removed
// is the round trip to device memory between consecutive passes.
//
// NUMERICS. Each kernel replicates `rms_single_row`
// (`backend/metal/kernels/rms_norm.metal`) exactly at this row width: 704
// threads x N_READS = 4, float square accumulation in thread-read order,
// `simd_sum`, a 32-slot cross-simd combine with the unused slots zeroed, and
// `metal::precise::rsqrt(acc / 2816 + eps)`.
//
// The one trap is intermediate rounding. MSL bfloat arithmetic promotes to
// float, but the stock chain stores each norm result to bf16 memory BEFORE the
// next op reads it. A naively fused expression would carry an unrounded float
// product into the following add and drift near-tie argmaxes. Every point where
// the stock graph materialises a bf16 array is therefore an explicit
// `static_cast<T>` here, so the fused kernels round in exactly the same places
// the stock dispatches do.
//
// SCOPE. These engage only on the prefill plane: `[B, chunk >= 2, 2816]`
// bfloat16 with `eps == 1e-6`. B is 8 on the scored cohort, because CBv2
// coalesces equal-length prompt chunks into one layer-major `[B, chunk]`
// forward, and 1 on a single-stream local run. The batch-eight DECODE plane
// (`[8, 1, 2816]`) is excluded by `chunk >= 2`, as are CBv2 tail-narrowed
// single-row prompts, MTP rectangles and every other dtype or geometry, all of
// which fall through to the untouched stock chain. Kill switch:
// `DARKBLOOM_GEMMA4_PREFILL_GLUE=0`.

import Foundation
import MLX
import MLXFast

public enum Gemma4PrefillGlueV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_GLUE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let expertTailFusionEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_EXPERT_TAIL_FUSION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// This checkpoint's hidden size, and the `rms_single_row` launch geometry
    /// the stock host derives from it (`RMS_N_READS` 4, so 2816 / 4 = 704
    /// threads, 22 simdgroups, one threadgroup per row).
    static let axis = 2816
    private static let nReads = 4
    private static let threadsPerRow = 704
    private static let eps: Float = 1e-6

    /// `rms_single_row`'s reduction, reproduced so the fused kernels see the
    /// same accumulation order, the same cross-simd combine and the same
    /// `precise::rsqrt` the stock kernel uses.
    static let kernelHeader = """
        constant constexpr const int GLUE_AXIS = 2816;
        constant constexpr const int GLUE_NREADS = 4;
        // Pinned; `planeRows` refuses any other eps.
        constant constexpr const float GLUE_EPS = 1e-6f;

        inline float glue_inv_rms(
            thread const float* xv,
            threadgroup float* local_sums,
            threadgroup float* local_inv,
            uint simd_lane_id,
            uint simd_group_id,
            float eps) {
          float acc = 0;
          for (int i = 0; i < GLUE_NREADS; i++) {
            acc += xv[i] * xv[i];
          }
          acc = simd_sum(acc);
          if (simd_group_id == 0) {
            local_sums[simd_lane_id] = 0;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lane_id == 0) {
            local_sums[simd_group_id] = acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group_id == 0) {
            acc = simd_sum(local_sums[simd_lane_id]);
            if (simd_lane_id == 0) {
              local_inv[0] = metal::precise::rsqrt(acc / GLUE_AXIS + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          return local_inv[0];
        }

        /// Both reductions of the tail in one pass, so `h1` and `h2` are each
        /// read from device memory exactly once.
        inline void glue_inv_rms2(
            thread const float* av,
            thread const float* bv,
            threadgroup float* local_sums_a,
            threadgroup float* local_sums_b,
            threadgroup float* local_inv2,
            uint simd_lane_id,
            uint simd_group_id,
            float eps,
            thread float& inv_a,
            thread float& inv_b) {
          float acc_a = 0;
          float acc_b = 0;
          for (int i = 0; i < GLUE_NREADS; i++) {
            acc_a += av[i] * av[i];
            acc_b += bv[i] * bv[i];
          }
          acc_a = simd_sum(acc_a);
          acc_b = simd_sum(acc_b);
          if (simd_group_id == 0) {
            local_sums_a[simd_lane_id] = 0;
            local_sums_b[simd_lane_id] = 0;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_lane_id == 0) {
            local_sums_a[simd_group_id] = acc_a;
            local_sums_b[simd_group_id] = acc_b;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group_id == 0) {
            acc_a = simd_sum(local_sums_a[simd_lane_id]);
            acc_b = simd_sum(local_sums_b[simd_lane_id]);
            if (simd_lane_id == 0) {
              local_inv2[0] = metal::precise::rsqrt(acc_a / GLUE_AXIS + eps);
              local_inv2[1] = metal::precise::rsqrt(acc_b / GLUE_AXIS + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          inv_a = local_inv2[0];
          inv_b = local_inv2[1];
        }
        """

    // MARK: - norm + residual (2 dispatches -> 1)

    private static let normResidualKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_glue_norm_residual_2816_v1",
        inputNames: ["x", "w", "res"],
        outputNames: ["out"],
        source: """
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float xv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                xv[i] = static_cast<float>(x[base + i]);
            }

            const float inv = glue_inv_rms(
                xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                // The stock pair stores `w * T(x*inv)` to bf16, then reads it
                // back for the add. Round in the same place.
                const T normed = static_cast<T>(w[j] * static_cast<T>(xv[i] * inv));
                out[base + i] = res[base + i] + normed;
            }
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// `residual + rmsNorm(x, weight)`. Returns `nil` off the prefill plane.
    public static func normResidual(
        x: MLXArray, weight: MLXArray, residual: MLXArray, eps epsIn: Float
    ) -> MLXArray? {
        guard let rows = planeRows(x, weight: weight, eps: epsIn),
            residual.shape == x.shape,
            residual.dtype == x.dtype
        else { return nil }

        return normResidualKernel(
            [x, weight, residual],
            template: [("T", x.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
    }

    // MARK: - dual pre-norm (2 dispatches -> 1)

    private static let dualPreNormKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_glue_dual_prenorm_2816_v1",
        inputNames: ["x", "w1", "w2"],
        outputNames: ["out1", "out2"],
        source: """
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float xv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                xv[i] = static_cast<float>(x[base + i]);
            }

            // One sum-of-squares serves both weights: the two stock kernels
            // reduce the identical input and differ only in the weight vector.
            const float inv = glue_inv_rms(
                xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T scaled = static_cast<T>(xv[i] * inv);
                out1[base + i] = w1[j] * scaled;
                out2[base + i] = w2[j] * scaled;
            }
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// `(rmsNorm(x, w1), rmsNorm(x, w2))`. Returns `nil` off the prefill plane.
    public static func dualPreNorm(
        x: MLXArray, w1: MLXArray, w2: MLXArray, eps epsIn: Float
    ) -> (MLXArray, MLXArray)? {
        guard let rows = planeRows(x, weight: w1, eps: epsIn),
            w2.shape == w1.shape,
            w2.dtype == w1.dtype
        else { return nil }

        let outs = dualPreNormKernel(
            [x, w1, w2],
            template: [("T", x.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, x.dtype]
        )
        return (outs[0], outs[1])
    }

    // MARK: - branch tail (5 dispatches -> 1)

    private static let tailKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_glue_tail_2816_v1",
        inputNames: ["h1", "h2", "w1", "w2", "w3", "res2"],
        outputNames: ["out"],
        source: """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                av[i] = static_cast<float>(h1[base + i]);
                bv[i] = static_cast<float>(h2[base + i]);
            }

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            // The branch sum stays in registers. The stock graph writes it to
            // bf16 between the norms and the final norm, so round it here.
            float tv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed = static_cast<T>(w3[j] * static_cast<T>(tv[i] * inv_t));
                out[base + i] = res2[base + i] + normed;
            }
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// `res2 + rmsNorm(rmsNorm(h1, w1) + rmsNorm(h2, w2), w3)`.
    /// Returns `nil` off the prefill plane.
    public static func branchTail(
        h1: MLXArray, h2: MLXArray, w1: MLXArray, w2: MLXArray, w3: MLXArray,
        residual2: MLXArray, eps epsIn: Float
    ) -> MLXArray? {
        guard let rows = planeRows(h1, weight: w1, eps: epsIn),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual2.shape == h1.shape, residual2.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype
        else { return nil }

        return tailKernel(
            [h1, h2, w1, w2, w3, residual2],
            template: [("T", h1.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [h1.shape],
            outputDTypes: [h1.dtype]
        )[0]
    }

    // MARK: - branch tail, chained (7 dispatches -> 1)

    /// The tail above stops at the stored `out` row, leaving two more
    /// full-width serial passes behind it: the terminal layer-scalar multiply,
    /// and the NEXT layer's `inputLayernorm(out)`. On the decode cohort those
    /// two are already folded in (`Gemma4FusedLayerGlue.tailChained`), but that
    /// gate pins `dim(1) == 1`, so on the prefill plane both still run as
    /// standalone dispatches over `[B, chunk, 2816]`.
    ///
    /// The threadgroup already holds the finished row in registers when it
    /// stores `out`, so both cost one extra in-kernel reduction rather than a
    /// re-read of the row plus two launches.
    private static let tailChainKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_glue_tail_chain_2816_v1",
        inputNames: ["h1", "h2", "w1", "w2", "w3", "res2", "s", "wn"],
        outputNames: ["out", "normed"],
        source: """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                av[i] = static_cast<float>(h1[base + i]);
                bv[i] = static_cast<float>(h2[base + i]);
            }

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            float tv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            // The stock graph stores the residual sum to bf16, then the scalar
            // multiply reads it back and stores again. Both roundings are
            // explicit here, so `out` is the same array either way.
            const T scalar = s[0];
            float ov[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed3 = static_cast<T>(w3[j] * static_cast<T>(tv[i] * inv_t));
                const T summed = static_cast<T>(res2[base + i] + normed3);
                const T scaled = static_cast<T>(summed * scalar);
                out[base + i] = scaled;
                ov[i] = static_cast<float>(scaled);
            }

            // The next layer's input norm, over exactly the bf16 values just
            // stored to `out`.
            const float inv_n = glue_inv_rms(
                ov, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                normed[base + i] = wn[j] * static_cast<T>(ov[i] * inv_n);
            }
            """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    /// `branchTail`, plus the terminal `* layerScalar`, plus the next layer's
    /// `rmsNorm(out, nextInputNormWeight)`. Returns `nil` off the prefill plane.
    public static func branchTailChained(
        h1: MLXArray, h2: MLXArray, w1: MLXArray, w2: MLXArray, w3: MLXArray,
        residual2: MLXArray, layerScalar: MLXArray, nextInputNormWeight: MLXArray,
        eps epsIn: Float
    ) -> (out: MLXArray, normedNext: MLXArray)? {
        guard let rows = planeRows(h1, weight: w1, eps: epsIn),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual2.shape == h1.shape, residual2.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype,
            layerScalar.size == 1, layerScalar.dtype == h1.dtype,
            nextInputNormWeight.shape == w1.shape,
            nextInputNormWeight.dtype == w1.dtype
        else { return nil }

        let outs = tailChainKernel(
            [h1, h2, w1, w2, w3, residual2, layerScalar, nextInputNormWeight],
            template: [("T", h1.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [h1.shape, h1.shape],
            outputDTypes: [h1.dtype, h1.dtype]
        )
        return (outs[0], outs[1])
    }

    // MARK: - sorted expert reduction + chained branch tail (2 dispatches -> 1)

    /// The sorted expert reducer and chained prefill tail both traverse the
    /// same `[tokens, hidden]` expert result. Produce each reduced expert value
    /// in the tail thread that consumes it, removing the intermediate tensor.
    private static let expertTailChainKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_expert_unsort_tail_chain_2816_v3",
        inputNames: [
            "sorted", "inverse_order", "route_weights", "h1",
            "w1", "w2", "w3", "res2", "s", "wn",
        ],
        outputNames: ["out", "normed"],
        source: """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint assignment_base = row * 8;

            static_assert(
                GLUE_NREADS == 4, "the vector expert gather owns four features");

            // `GLUE_AXIS / GLUE_NREADS` is 704 with no remainder and both
            // `row * GLUE_AXIS + lid * GLUE_NREADS` and `lid * GLUE_NREADS`
            // are whole multiples of four elements, so every four-feature
            // packet this thread owns is one eight-byte aligned vector.
            // `ensureRowContiguous` makes each buffer start an allocation
            // base, so the vector index of the row element is
            // `row * 704 + lid` and of a length-axis weight vector is `lid`.
            // Same bytes, same order, one load instead of four.
            const size_t row_vec =
                size_t(row) * (GLUE_AXIS / GLUE_NREADS) + lid;

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            {
                const vec<T, 4> h1v = ((const device vec<T, 4>*)h1)[row_vec];
                av[0] = static_cast<float>(h1v.x);
                av[1] = static_cast<float>(h1v.y);
                av[2] = static_cast<float>(h1v.z);
                av[3] = static_cast<float>(h1v.w);
            }

            // `inverse_order` and `route_weights` are indexed by the row and the
            // slot alone, so the slot walk sits above the feature walk and each
            // pair is read once per thread rather than once per feature. The
            // four features a thread owns are adjacent and eight-byte aligned,
            // so the sorted row arrives in one vector load. Slot order and the
            // per-slot bfloat16 rounding of the running sum are unchanged.
            const device vec<T, 4>* sorted4 = (const device vec<T, 4>*)sorted;
            vec<T, 4> accumulator = vec<T, 4>((T)0, (T)0, (T)0, (T)0);
            for (uint slot = 0; slot < 8; ++slot) {
                const uint assignment = assignment_base + slot;
                const uint sorted_row = (uint)inverse_order[assignment];
                const float route = (float)route_weights[assignment];
                const vec<T, 4> sv =
                    sorted4[size_t(sorted_row) * (GLUE_AXIS / GLUE_NREADS) + lid];
                const vec<T, 4> weighted = vec<T, 4>(
                    (T)((float)sv.x * route),
                    (T)((float)sv.y * route),
                    (T)((float)sv.z * route),
                    (T)((float)sv.w * route));
                accumulator = accumulator + weighted;
            }
            bv[0] = static_cast<float>(accumulator.x);
            bv[1] = static_cast<float>(accumulator.y);
            bv[2] = static_cast<float>(accumulator.z);
            bv[3] = static_cast<float>(accumulator.w);

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            const vec<T, 4> w1v = ((const device vec<T, 4>*)w1)[lid];
            const vec<T, 4> w2v = ((const device vec<T, 4>*)w2)[lid];
            const T w1e[GLUE_NREADS] = { w1v.x, w1v.y, w1v.z, w1v.w };
            const T w2e[GLUE_NREADS] = { w2v.x, w2v.y, w2v.z, w2v.w };

            float tv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                const T n1 = static_cast<T>(
                    w1e[i] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(
                    w2e[i] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            const T scalar = s[0];
            const vec<T, 4> w3v = ((const device vec<T, 4>*)w3)[lid];
            const vec<T, 4> resv = ((const device vec<T, 4>*)res2)[row_vec];
            const T w3e[GLUE_NREADS] = { w3v.x, w3v.y, w3v.z, w3v.w };
            const T rese[GLUE_NREADS] = { resv.x, resv.y, resv.z, resv.w };

            T outv[GLUE_NREADS];
            float ov[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                const T normed3 = static_cast<T>(
                    w3e[i] * static_cast<T>(tv[i] * inv_t));
                const T summed = static_cast<T>(rese[i] + normed3);
                const T scaled = static_cast<T>(summed * scalar);
                outv[i] = scaled;
                ov[i] = static_cast<float>(scaled);
            }
            ((device vec<T, 4>*)out)[row_vec] =
                vec<T, 4>(outv[0], outv[1], outv[2], outv[3]);

            const float inv_n = glue_inv_rms(
                ov, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            const vec<T, 4> wnv = ((const device vec<T, 4>*)wn)[lid];
            const T wne[GLUE_NREADS] = { wnv.x, wnv.y, wnv.z, wnv.w };
            T nv[GLUE_NREADS];
            for (int i = 0; i < GLUE_NREADS; i++) {
                nv[i] = wne[i] * static_cast<T>(ov[i] * inv_n);
            }
            ((device vec<T, 4>*)normed)[row_vec] =
                vec<T, 4>(nv[0], nv[1], nv[2], nv[3]);
        """,
        header: kernelHeader,
        ensureRowContiguous: true
    )

    public static func branchTailChainedUnsort(
        h1: MLXArray,
        expert: WeightedExpertUnsortCarrier,
        w1: MLXArray,
        w2: MLXArray,
        w3: MLXArray,
        residual2: MLXArray,
        layerScalar: MLXArray,
        nextInputNormWeight: MLXArray,
        eps epsIn: Float
    ) -> (out: MLXArray, normedNext: MLXArray)? {
        guard expertTailFusionEnabled,
            let rows = planeRows(h1, weight: w1, eps: epsIn),
            expert.sortedOutputs.ndim == 2,
            expert.sortedOutputs.shape == [rows * 8, axis],
            expert.sortedOutputs.dtype == h1.dtype,
            expert.inverseOrder.ndim == 1,
            expert.inverseOrder.size == rows * 8,
            expert.inverseOrder.dtype == .uint32,
            expert.weights.ndim == 2,
            expert.weights.shape == [rows, 8],
            expert.weights.dtype == h1.dtype,
            residual2.shape == h1.shape,
            residual2.dtype == h1.dtype,
            w2.shape == w1.shape,
            w2.dtype == w1.dtype,
            w3.shape == w1.shape,
            w3.dtype == w1.dtype,
            layerScalar.size == 1,
            layerScalar.dtype == h1.dtype,
            nextInputNormWeight.shape == w1.shape,
            nextInputNormWeight.dtype == w1.dtype
        else { return nil }

        CBv2EngageMark.once("prefill-expert-tail-fuse")
        let outputs = expertTailChainKernel(
            [
                expert.sortedOutputs, expert.inverseOrder, expert.weights, h1,
                w1, w2, w3, residual2, layerScalar, nextInputNormWeight,
            ],
            template: [("T", h1.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [h1.shape, h1.shape],
            outputDTypes: [h1.dtype, h1.dtype]
        )
        return (outputs[0], outputs[1])
    }

    // MARK: - plane gate

    /// The prefill plane. `dim(1) >= 2` is the load-bearing condition: it keeps
    /// this off the batch-eight decode cohort (`[8, 1, 2816]`) and off any
    /// single-row rectangle, so the decode chain is untouched.
    ///
    /// The leading dimension is deliberately NOT pinned to 1. CBv2 coalesces
    /// equal-length prompt chunks into one layer-major `[B, chunk]` forward
    /// when the model and the cache provider both prove rectangular per-row
    /// semantics (`EngineLoopV2`, the `packedPrefillSupported` branch), and
    /// `Gemma4TextModel.cbv2SupportsPackedPrefill` is `true`. So the scored
    /// eight-stream cohort presents `[8, chunk, 2816]` here, not
    /// `[1, chunk, 2816]`. A single-stream local run presents the latter, which
    /// makes this the exact shape a local-only check cannot see.
    ///
    /// Rows are the product of the two leading dimensions. Every input is
    /// `ensureRowContiguous`, and each row is reduced independently, so the
    /// kernels do not care how those rows are grouped.
    private static func planeRows(_ x: MLXArray, weight: MLXArray, eps epsIn: Float) -> Int? {
        guard enabled,
            epsIn == eps,
            x.dtype == .bfloat16,
            weight.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) >= 1,
            x.dim(1) >= 2,
            x.dim(2) == axis,
            weight.ndim == 1,
            weight.dim(0) == axis
        else { return nil }
        return x.dim(0) * x.dim(1)
    }
}
