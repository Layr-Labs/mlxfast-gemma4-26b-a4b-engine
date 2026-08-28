// Gemma4ScaledTailV1.swift
//
// Exact batch-8 Gemma 4 decode tail with the checkpoint's per-layer scalar.
//
// GLUE-001 already fuses the MoE post-branch normalizations, their sum, the
// final normalization, and the residual add.  On a text tower with no PLE,
// the very next operation is the per-layer scalar multiply.  Keeping that
// multiply in the same Metal dispatch removes one serial elementwise launch
// per decoder layer while retaining the stock bfloat16 store/load boundary as
// the explicit `summed` local below.

import Foundation
import MLX
import MLXFast

public enum CBv2Gemma4ScaledTailV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_FUSED_SCALED_TAIL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let rows = 8
    private static let axis = 2_816
    private static let eps: Float = 1e-6
    private static let threadgroupThreads = 704

    private static func rmsReduce(_ source: String, into slot: String) -> String {
        """
            {
                float acc = 0;
                for (int i = 0; i < 4; i++) {
                    float xi = (float)\(source)[base + i];
                    acc += xi * xi;
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        \(slot) = metal::precise::rsqrt(acc / 2816.0f + 1e-06f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """
    }

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_gemma4_scaled_tail_2816_bf16_v1",
        inputNames: ["a", "b", "res", "w1", "w2", "w3", "layer_scalar"],
        outputNames: ["out"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(rmsReduce("b", into: "local_inv[1]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            for (int i = 0; i < 4; i++) {
                const T normed = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                // The established fused tail stores this residual sum as
                // bfloat16 before the following binary multiply reloads it.
                // An explicit T local preserves that exact rounding boundary.
                const T summed = res[base + i] + normed;
                out[base + i] = summed * layer_scalar[0];
            }
        """,
        ensureRowContiguous: true
    )

    public static func tail(
        mlpOut: MLXArray, expertOut: MLXArray, residual: MLXArray,
        w1: MLXArray, w2: MLXArray, w3: MLXArray,
        layerScalar: MLXArray, eps: Float
    ) -> MLXArray? {
        guard enabled,
            eps == Self.eps,
            mlpOut.shape == [rows, 1, axis], mlpOut.dtype == .bfloat16,
            expertOut.shape == mlpOut.shape, expertOut.dtype == .bfloat16,
            residual.shape == mlpOut.shape, residual.dtype == .bfloat16,
            w1.shape == [axis], w1.dtype == .bfloat16,
            w2.shape == [axis], w2.dtype == .bfloat16,
            w3.shape == [axis], w3.dtype == .bfloat16,
            layerScalar.shape == [1], layerScalar.dtype == .bfloat16
        else { return nil }

        CBv2EngageMark.once("scaled-tail")
        return kernel(
            [mlpOut, expertOut, residual, w1, w2, w3, layerScalar],
            template: [("T", mlpOut.dtype)],
            grid: (rows * threadgroupThreads, 1, 1),
            threadGroup: (threadgroupThreads, 1, 1),
            outputShapes: [[rows, 1, axis]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
