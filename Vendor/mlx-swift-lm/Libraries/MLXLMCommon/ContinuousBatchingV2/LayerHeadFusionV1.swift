// HEAD-001: one dispatch for the decoder layer's head, and one reduction
// instead of two over the same row.
//
// The head is four graph operations per layer:
//
//     let postAttn = postAttentionLayernorm(attnOut)   // MLXFast.rmsNorm
//     var out      = residual + postAttn               // residual
//     var h1       = preFeedforwardLayernorm(out)      // MLXFast.rmsNorm
//     var h2       = preFeedforwardLayernorm2(out)     // MLXFast.rmsNorm
//
// Two things are wasted here, and only one of them is a dispatch count.
//
// `preFeedforwardLayernorm` and `preFeedforwardLayernorm2` are both
// `RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)` and are both applied to the
// SAME `out` row. They therefore compute the SAME mean of squares and differ
// only in the learned scale they multiply through. One of those two reductions
// is duplicate arithmetic, on every layer of every decode step. This kernel
// computes it once.
//
// `postAttn` is also materialized — 8 x 2816 bf16 written and immediately read
// by the residual add and by nothing else. This kernel keeps it in registers.
//
// So four dispatches become one, three reductions become two, and one of the
// four written tensors disappears. `out` is still written because the router and
// the second residual both read it.
//
// Rounding points, reproduced exactly, following `rms_single_row` in
// `rms_norm.metal`: float squares per thread, `simd_sum`, per-simdgroup partials
// over a zero-initialised array, `simd_sum` again, then
// `metal::precise::rsqrt(acc / axis_size + eps)`, and a write of
// `w[i] * static_cast<T>(x[i] * inv)` with the cast to T BEFORE the weight
// multiply. The residual add rounds in T. `out` is read back from its own output
// buffer for the second stage rather than recomputed, so the value that feeds
// the second reduction is the same rounded value the graph would have produced.
//
// `DARKBLOOM_CBV2_LAYER_HEAD=0` restores the four-operation path.

import Foundation
import MLX
import MLXFast

public enum CBv2LayerHeadFusionV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_LAYER_HEAD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let reads = 4
    private static let simdWidth = 32

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_layer_head_rms_residual_dualrms_v1",
        inputNames: ["attn", "wPost", "residual", "wPre1", "wPre2", "epsPost", "epsPre"],
        outputNames: ["out", "h1", "h2"],
        source: """
            constexpr uint reads = 4;
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;

            const uint base = row * D + lid * reads;
            const device T* a  = attn + base;
            const device T* r  = residual + base;
            const device T* wp = wPost + lid * reads;
            const device T* w1 = wPre1 + lid * reads;
            const device T* w2 = wPre2 + lid * reads;
            device T* o  = out + base;
            device T* o1 = h1 + base;
            device T* o2 = h2 + base;

            threadgroup float local_sums[32];
            threadgroup float local_inv;

            // Reduction 1: over the attention output.
            float acc = 0.0f;
            for (uint i = 0; i < reads; ++i) {
                const float xi = a[i];
                acc += xi * xi;
            }
            acc = simd_sum(acc);
            if (simd_group == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd_group] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) local_inv = metal::precise::rsqrt(acc / float(D) + epsPost[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv1 = local_inv;

            // post-attention norm, residual, and the second reduction, in one
            // pass. `postAttn` never reaches device memory.
            float acc2 = 0.0f;
            for (uint i = 0; i < reads; ++i) {
                const T postAttn = wp[i] * static_cast<T>(a[i] * inv1);
                const T s = r[i] + postAttn;
                o[i] = s;
                const float xi = s;
                acc2 += xi * xi;
            }
            acc2 = simd_sum(acc2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd_group] = acc2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc2 = simd_sum(local_sums[lane]);
                if (lane == 0) local_inv = metal::precise::rsqrt(acc2 / float(D) + epsPre[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv2 = local_inv;

            // Both pre-feed-forward norms share that one reduction.
            for (uint i = 0; i < reads; ++i) {
                const T s = o[i];
                const T normed = static_cast<T>(s * inv2);
                o1[i] = w1[i] * normed;
                o2[i] = w2[i] * normed;
            }
            """,
        ensureRowContiguous: true
    )

    /// Returns `nil` unless every pin holds; the caller keeps the four-op path.
    public static func apply(
        attn: MLXArray,
        postWeight: MLXArray,
        residual: MLXArray,
        preWeight1: MLXArray,
        preWeight2: MLXArray,
        postEps: Float,
        preEps1: Float,
        preEps2: Float
    ) -> (out: MLXArray, h1: MLXArray, h2: MLXArray)? {
        // The two pre-feed-forward norms share ONE reduction here, which is
        // only correct if they share an epsilon. Both are constructed with
        // `config.rmsNormEps` on this checkpoint, but the gate refuses rather
        // than assumes: a checkpoint that gave them different epsilons would
        // otherwise be silently mis-normalized.
        guard enabled,
            preEps1 == preEps2,
            attn.dtype == .bfloat16,
            residual.dtype == attn.dtype,
            postWeight.dtype == attn.dtype,
            preWeight1.dtype == attn.dtype,
            preWeight2.dtype == attn.dtype,
            attn.ndim == 3,
            attn.dim(1) == 1,
            attn.shape == residual.shape,
            postWeight.ndim == 1, postWeight.dim(0) == attn.dim(2),
            preWeight1.ndim == 1, preWeight1.dim(0) == attn.dim(2),
            preWeight2.ndim == 1, preWeight2.dim(0) == attn.dim(2)
        else { return nil }

        let rows = attn.dim(0)
        let dim = attn.dim(2)
        guard dim % (reads * simdWidth) == 0 else { return nil }
        let threads = dim / reads
        guard threads <= 1024, threads / simdWidth <= 32 else { return nil }

        let r = kernel(
            [
                attn, postWeight, residual, preWeight1, preWeight2,
                MLXArray([postEps]), MLXArray([preEps1]),
            ],
            template: [("T", attn.dtype), ("D", dim)],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [attn.shape, attn.shape, attn.shape],
            outputDTypes: [attn.dtype, attn.dtype, attn.dtype]
        )
        return (r[0], r[1], r[2])
    }
}
