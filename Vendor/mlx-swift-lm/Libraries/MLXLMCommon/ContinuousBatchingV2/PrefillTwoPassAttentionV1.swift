// PREFILL-FLASH-001: Two-pass prefill SDPA eliminating scores matrix.
//
// Motivation (benbuschmann census e2ecd8c, 8/28 9:43 PM):
// - ALL prefill attention uses composed fallback (head_dim 256/512 unsupported by fused SDPA)
// - Composed path materializes full [B, H, L, kL] scores matrix:
//     QK write -> softmax read+write -> AV read = ~23 GB/prefill
// - Two-pass kernel eliminates scores matrix entirely
// - Prior D=512 two-pass online-softmax (receipt 9b8358a) PASSED THE TOKEN GATE
//
// Design:
//   Pass A (QK + max/sumexp):
//     - QK matmul with causal mask applied in epilogue
//     - Per-row reduction: compute max and sumexp WITHOUT materializing full scores
//     - Store only [B, H, L, 2] partials (max, sumexp)
//   
//   Pass B (AV with renormalized scores):
//     - Recompute QK from cached Q, K
//     - Compute exp(s-max)/sumexp on the fly
//     - AV matmul produces output directly
//
// Target: 25 sliding layers (head_dim=256). Full layers (head_dim=512) use ComposedPrefillSDPAV1.
//
// Exactness: NOT bit-exact. Uses 10% token-divergence tolerance.
//
// Kill switch: DARKBLOOM_CBV2_PREFILL_FLASH_2PASS=0

import Foundation
import MLX
import MLXFast

/// Two-pass prefill SDPA: eliminates scores matrix materialization.
///
/// For production prefill geometry (8 streams, 1024 tokens, 25 sliding layers
/// at head_dim=256, 5 full layers at head_dim=512):
///
///   Composed fallback: materializes [B, H, L, kL] scores (~23 GB/prefill)
///   Two-pass: eliminates scores matrix entirely
///
/// The scores matrix is the single largest non-GEMM item in prefill.
/// Eliminating it is the key to prefill performance gains.
enum CBv2PrefillTwoPassAttentionV1 {

    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_FLASH_2PASS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Pass A kernel: computes QK + per-row max/sumexp
    private static let passAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_prefill_flash_2pass_a_bf16_d256_g2_v1",
        inputNames: ["queries", "keys"],
        outputNames: ["maxes", "sumexps"],
        source: """
            constexpr int DIM = 256;
            
            const int b = int(threadgroup_position_in_grid.x);
            const int h = int(threadgroup_position_in_grid.y);
            const int q_row = int(threadgroup_position_in_grid.z);
            
            if (q_row >= L) return;
            
            const device bfloat16_t* q_row_ptr = queries + (b * L + q_row) * DIM;
            
            float maxval = -1.0 / 0.0;
            
            for (int k_pos = 0; k_pos < kL; k_pos++) {
                if (k_pos >= q_row) continue;
                
                const device bfloat16_t* k_row_ptr = keys + (b * kL + k_pos) * DIM;
                
                float score = 0.0;
                for (int d = 0; d < DIM; d++) {
                    score += float(q_row_ptr[d]) * float(k_row_ptr[d]);
                }
                
                if (score > maxval) {
                    maxval = score;
                }
            }
            
            float sumexp_val = 0.0;
            for (int k_pos = 0; k_pos < kL; k_pos++) {
                if (k_pos >= q_row) continue;
                
                const device bfloat16_t* k_row_ptr = keys + (b * kL + k_pos) * DIM;
                
                float score = 0.0;
                for (int d = 0; d < DIM; d++) {
                    score += float(q_row_ptr[d]) * float(k_row_ptr[d]);
                }
                
                sumexp_val += exp(score - maxval);
            }
            
            maxes[b * L + q_row] = maxval;
            sumexps[b * L + q_row] = sumexp_val;
        """
    )

    /// Pass B kernel: computes AV with renormalized scores
    private static let passBKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_prefill_flash_2pass_b_bf16_d256_g2_v1",
        inputNames: ["queries", "keys", "values", "maxes", "sumexps"],
        outputNames: ["output"],
        source: """
            constexpr int DIM = 256;
            
            const int b = int(threadgroup_position_in_grid.x);
            const int h = int(threadgroup_position_in_grid.y);
            const int q_row = int(threadgroup_position_in_grid.z);
            
            if (q_row >= L) return;
            
            float maxval = maxes[b * L + q_row];
            float sumexp_val = sumexps[b * L + q_row];
            
            const device bfloat16_t* q_row_ptr = queries + (b * L + q_row) * DIM;
            device bfloat16_t* out_row_ptr = output + (b * L + q_row) * DIM;
            
            for (int d = 0; d < DIM; d++) {
                out_row_ptr[d] = bfloat16_t(0.0);
            }
            
            for (int k_pos = 0; k_pos < kL; k_pos++) {
                if (k_pos >= q_row) continue;
                
                const device bfloat16_t* k_row_ptr = keys + (b * kL + k_pos) * DIM;
                const device bfloat16_t* v_row_ptr = values + (b * kL + k_pos) * DIM;
                
                float score = 0.0;
                for (int d = 0; d < DIM; d++) {
                    score += float(q_row_ptr[d]) * float(k_row_ptr[d]);
                }
                
                float prob = exp(score - maxval) / sumexp_val;
                
                for (int d = 0; d < DIM; d++) {
                    out_row_ptr[d] += bfloat16_t(prob * float(v_row_ptr[d]));
                }
            }
        """
    )

    /// Two-pass attention for sliding layers.
    ///
    /// Returns nil to fall back to ComposedPrefillSDPAV1.
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, L: Int, kL: Int, window: Int?,
        bidirectional: Bool, sinks: MLXArray?
    ) -> MLXArray? {
        guard enabled else { return nil }

        // Guard: only for sliding layers (head_dim=256)
        let queryDim = queries.dim(3)
        guard queryDim == 256 else { return nil }

        // Guard: causal mask only
        guard sinks == nil, !bidirectional else { return nil }

        // Guard: production geometry
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }
        guard queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16 else { return nil }

        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let nKVHeads = keys.dim(1)
        let valueDim = values.dim(3)

        guard nKVHeads > 0, values.dim(1) == nKVHeads, nQHeads % nKVHeads == 0 else { return nil }
        guard queries.dim(2) == L, keys.dim(2) == kL, values.dim(2) == kL else { return nil }
        guard keys.dim(0) == B, values.dim(0) == B else { return nil }
        guard keys.dim(3) == queryDim else { return nil }

        // Only for causal mask (no array mask)
        if let window, kL > window { return nil }
        guard kL >= L else { return nil }

        // GQA handling
        let nRepeats = nQHeads / nKVHeads
        var q = queries
        var k = keys
        var v = values

        if nRepeats > 1 {
            q = queries.reshaped([B, nKVHeads, nRepeats, L, queryDim])
            k = keys.expandedDimensions(axis: 2)
            v = values.expandedDimensions(axis: 2)
        }

        // PREFILL-FLASH-2PASS: Two-pass attention
        // Pass A: Compute QK + per-row max/sumexp
        let passA = passAKernel(
            [q, k],
            template: [("L", L), ("kL", kL)],
            grid: (B, nKVHeads * nRepeats, L),
            threadGroup: (1, 1, 1),
            outputShapes: [[B, nKVHeads * nRepeats, L], [B, nKVHeads * nRepeats, L]],
            outputDTypes: [.float32, .float32])

        let maxes = passA[0]
        let sumexps = passA[1]

        // Pass B: Compute AV with renormalized scores
        let passB = passBKernel(
            [q, k, v, maxes, sumexps],
            template: [("L", L), ("kL", kL)],
            grid: (B, nKVHeads * nRepeats, L),
            threadGroup: (1, 1, 1),
            outputShapes: [[B, nKVHeads * nRepeats, L, valueDim]],
            outputDTypes: [.bfloat16])

        let output = passB[0]

        // Reshape back if needed
        if nRepeats > 1 {
            return output.reshaped([B, nQHeads, L, valueDim])
        }
        return output
    }
}
