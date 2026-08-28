// TAIL-001: one dispatch for the decoder layer's post-feed-forward tail.
//
// The tail is four separate graph operations per layer:
//
//     out = h1 + h2                          // dense + sparse branch sum
//     out = postFeedforwardLayernorm(out)    // MLXFast.rmsNorm
//     out = residual2 + out                  // residual
//     out = out * layerScalar                // learned per-layer scalar
//
// On the scored B=8 decode path each is a Metal dispatch over a single
// [8, 1, 2816] rectangle, so 30 layers spend 120 dispatches and write three
// intermediate tensors that nothing else reads. This kernel performs the same
// four steps, in the same order, with the same rounding points, and writes only
// the final tensor.
//
// The PLE gating block that sits between the residual add and the scalar
// multiply is skipped whenever `per_layer_input_gate` is absent, which is the
// case on this checkpoint: `hiddenSizePerLayerInput` is zero, so the modules are
// never constructed and the branch is unreachable. The gate below still refuses
// to engage unless the caller states the block was skipped, so a checkpoint that
// does construct PLE keeps the established path.
//
// Rounding points, reproduced exactly:
//   1. `h1 + h2` rounds in T (bfloat16_t is Metal's native bfloat).
//   2. `rmsNorm` accumulates float squares, reduces per simdgroup then across
//      simdgroups, takes `precise::rsqrt(acc / axis + eps)`, and writes
//      `w[i] * static_cast<T>(x[i] * inv)` — the cast to T happens BEFORE the
//      weight multiply, exactly as `rms_single_row` in `rms_norm.metal` does.
//   3. the residual add rounds in T.
//   4. the scalar multiply rounds in T, and the gate requires the scalar to
//      share the activation dtype so no promotion can change the result type.
//
// `DARKBLOOM_CBV2_BRANCH_TAIL=0` restores the four-operation path.

import Foundation
import MLX
import MLXFast

public enum CBv2BranchTailFusionV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_BRANCH_TAIL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let reads = 4
    private static let simdWidth = 32

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_branch_tail_rms_residual_scale_v1",
        inputNames: ["h1", "h2", "w", "residual", "scale", "epsv"],
        outputNames: ["out"],
        source: """
            constexpr uint reads = 4;
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;

            const uint base = row * D + lid * reads;
            const device T* a = h1 + base;
            const device T* b = h2 + base;
            const device T* r = residual + base;
            const device T* wp = w + lid * reads;
            device T* o = out + base;

            // Step 1 and the reduction half of step 2. `s` is recomputed in the
            // write loop rather than staged, so the value that feeds the sum and
            // the value that is normalized come from the same expression.
            float acc = 0.0f;
            for (uint i = 0; i < reads; ++i) {
                const T s = a[i] + b[i];
                const float xi = s;
                acc += xi * xi;
            }
            acc = simd_sum(acc);

            threadgroup float local_sums[32];
            threadgroup float local_inv;
            if (simd_group == 0) local_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[simd_group] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) {
                    local_inv = metal::precise::rsqrt(acc / float(D) + epsv[0]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const T sc = scale[0];
            for (uint i = 0; i < reads; ++i) {
                const T s = a[i] + b[i];
                const T normed = wp[i] * static_cast<T>(s * local_inv);
                const T added = r[i] + normed;
                o[i] = added * sc;
            }
            """,
        ensureRowContiguous: true
    )

    /// Returns `nil` unless every pin holds; the caller keeps the four-op path.
    ///
    /// - Parameter pleSkipped: the caller's statement that the PLE gating block
    ///   between the residual add and the scalar multiply is not taken.
    public static func apply(
        h1: MLXArray,
        h2: MLXArray,
        weight: MLXArray,
        eps: Float,
        residual: MLXArray,
        layerScalar: MLXArray,
        pleSkipped: Bool
    ) -> MLXArray? {
        guard enabled, pleSkipped,
            h1.dtype == .bfloat16,
            h2.dtype == h1.dtype,
            weight.dtype == h1.dtype,
            residual.dtype == h1.dtype,
            layerScalar.dtype == h1.dtype,
            // MLX emits `const constant T&` (not a pointer) for a rank-0 input,
            // so `scale[0]` would not compile; require a real 1-D array.
            layerScalar.ndim >= 1,
            layerScalar.size == 1,
            h1.ndim == 3,
            h1.dim(1) == 1,
            h1.shape == h2.shape,
            h1.shape == residual.shape,
            weight.ndim == 1,
            weight.dim(0) == h1.dim(2)
        else { return nil }

        let rows = h1.dim(0)
        let dim = h1.dim(2)
        // One threadgroup per row; `reads` elements per thread.
        guard dim % (reads * simdWidth) == 0 else { return nil }
        let threads = dim / reads
        guard threads <= 1024, threads / simdWidth <= 32 else { return nil }

        return kernel(
            [h1, h2, weight, residual, layerScalar, MLXArray([eps])],
            template: [("T", h1.dtype), ("D", dim)],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [h1.shape],
            outputDTypes: [h1.dtype]
        )[0]
    }
}
