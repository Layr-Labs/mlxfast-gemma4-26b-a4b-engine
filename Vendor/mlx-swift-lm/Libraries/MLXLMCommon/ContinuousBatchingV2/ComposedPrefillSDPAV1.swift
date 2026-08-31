// ComposedPrefillSDPAV1.swift
//
// PREFILL-QSCALE-ELIDE. Gemma 4 attention runs `scale == 1.0` exactly
// (`Gemma4Attention.init`: the query pre-attention scalar is folded into
// `q_norm`, so the attention call carries the identity). MLX's SDPA
// UNCONDITIONALLY materializes `scale * q` as the first op of its unfused
// fallback (`fast.cpp`, `auto q = multiply(array(scale, ...), inputs[0])`),
// even when `scale` is one — a full read+write of the query rectangle per
// call that produces a bit-for-bit copy of its input.
//
// Gemma 4 always takes that fallback on the prompt plane: MLX's fused
// full-attention kernel supports head_dim 64/80/128 and Gemma 4 attends at
// 256 (sliding) / 512 (full), and the vector kernel needs `L <= 8`, so
// `ScaledDotProductAttention::use_fallback` is true for every q-block of a
// prompt chunk. The fallback is then a PLAIN OP GRAPH — fast.cpp calls it
// directly, no primitive, no fused kernel — so it can be transcribed here
// op-for-op and the identity multiply simply left out.
//
// At the ranked prefill geometry (8 rows x 1024 tokens, q-blocks of 128,
// 25 sliding layers at [8, 16, 128, 256] + 5 full layers at
// [8, 16, 128, 512]) that is 232 dispatches and 1.11 x 10^9 bf16 elements of
// round-tripped identity per prefill step -- ~4.4 GB of traffic, the single
// largest non-GEMM item inside composed attention in the E4 prefill census.
//
// EXACTNESS. Every surviving op is the verbatim fast.cpp fallback:
//   n_repeats > 1  ->  q.unflatten(1, [kv, rep]); k, v expand_dims(2)
//   scores = matmul(q, swapaxes(k, -1, -2))
//   causal mask   = greater_equal(arange(kL-L, kL)[:, None], arange(0, kL))
//   scores = where(mask, scores, finfo(bf16).min)      // 0xFF7F
//   scores = softmax(scores, axis: -1, precise: true)
//   out    = matmul(scores, v);  flatten(out, 1, 2) when n_repeats > 1
// The ONLY difference is the deleted `q * 1.0`. bf16 multiplication by one
// returns its operand's bit pattern for every finite input, denormals and
// signed zeros included (the product is computed in fp32 and rounded back:
// `1.0f * float(x)` is `float(x)`, which round-trips to `x`), so the deleted
// op is the identity on the queries and every downstream value -- scores,
// probabilities, output -- is unchanged bit-for-bit.
//
// FAIL-CLOSED. Anything outside the exact regime above returns nil and the
// caller takes the established `MLXFast.scaledDotProductAttention` call:
// scale != 1, sinks, softcap, bidirectional spans, mixed dtypes, an array
// mask (a windowed layer whose returned history exceeds its window), a head
// dim MLX's fused kernels DO support, or `L <= 8` (decode, and every MTP
// verify width, stay on the stock path by construction).
//
// Kill switch: `DARKBLOOM_CBV2_PREFILL_SDPA_COMPOSE=0`.

import Foundation
import MLX

enum CBv2ComposedPrefillSDPAV1 {

    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_SDPA_COMPOSE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// PREFILL-MASK-FUSE. Applies the causal mask in the QK^T GEMM's own
    /// epilogue (`addmm`) instead of as a separate `where` pass over the
    /// score matrix. Kill switch: `DARKBLOOM_CBV2_PREFILL_MASK_FUSE=0`.
    static let maskFuseEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_MASK_FUSE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Lowest finite bfloat16 (bits 0xFF7F), i.e. `finfo(bfloat16).min` --
    /// the value MLX's fallback substitutes for masked score entries. The
    /// fp32 bit pattern 0xFF7F0000 is exactly this number, so the conversion
    /// to bf16 is exact under any rounding mode.
    ///
    /// Built ONCE. `MLXArray(Float, dtype: .bfloat16)` is not a host-side
    /// constructor -- mlx-swift routes it through `mlx_astype`, i.e. a real
    /// one-element `copy` DISPATCH. Rebuilding it per call would add exactly
    /// as many dispatches per prefill step as this rider deletes; the C++
    /// fallback pays nothing for the same constant because `array(double,
    /// bfloat16)` is a host construction there. A constant scalar is safe to
    /// share across graphs: it is an input, never a mutated output.
    nonisolated(unsafe) private static let bfloat16LowestScalar: MLXArray =
        MLXArray(Float(bitPattern: 0xFF7F_0000), dtype: .bfloat16)

    /// bfloat16 NEGATIVE zero (bits 0x8000) -- the additive identity the
    /// fused-mask bias carries on every UNMASKED score.
    ///
    /// `-0.0` and not `+0.0`: IEEE-754 round-to-nearest makes `x + (-0.0)`
    /// return `x` for EVERY float, signed zeros included (`(+0) + (-0) = +0`,
    /// `(-0) + (-0) = -0`), whereas `x + (+0.0)` maps `-0.0` to `+0.0` and
    /// would flip one bit of a score that happened to be a negative zero.
    /// With `-0.0` the GEMM epilogue is the exact identity on the fp32
    /// accumulator, so an unmasked entry rounds to the same bfloat16 word the
    /// plain `matmul` would have stored.
    nonisolated(unsafe) private static let bfloat16NegativeZeroScalar: MLXArray =
        MLXArray(Float(bitPattern: 0x8000_0000), dtype: .bfloat16)

    /// Causal masks, memoized on `(L, kL)`.
    ///
    /// The mask is a PURE FUNCTION of the two block lengths -- MLX's fallback
    /// rebuilds `arange(kL - L, kL)`, `arange(0, kL)` and their comparison on
    /// every one of the 232 attention calls a prefill step makes, and there
    /// are only eight distinct `(L, kL)` pairs in a 1024-token chunk. Every
    /// entry is a constant read-only input; nothing writes to a cached array,
    /// so sharing one across graphs and across steps is safe. Bounded: the
    /// table is dropped wholesale if it ever exceeds `maxCachedMasks`, so an
    /// unusual chunk geometry cannot grow it without limit.
    private static let maxCachedMasks = 64
    nonisolated(unsafe) private static var maskCache: [Int: MLXArray] = [:]
    private static let maskCacheLock = NSLock()

    private static func causalMask(L: Int, kL: Int) -> MLXArray {
        let key = L &* 1_000_003 &+ kL
        maskCacheLock.lock()
        if let hit = maskCache[key] {
            maskCacheLock.unlock()
            return hit
        }
        maskCacheLock.unlock()
        // fast.cpp: q_idx = arange(kL - L, kL)[:, None]; k_idx = arange(0, kL)[None, :]
        let qIndices = MLXArray(Int32(kL - L) ..< Int32(kL)).expandedDimensions(axis: 1)
        let kIndices = MLXArray(Int32(0) ..< Int32(kL)).expandedDimensions(axis: 0)
        let mask = qIndices .>= kIndices
        eval(mask)
        maskCacheLock.lock()
        if maskCache.count >= maxCachedMasks { maskCache.removeAll(keepingCapacity: true) }
        maskCache[key] = mask
        maskCacheLock.unlock()
        return mask
    }

    /// Causal mask BIAS, memoized on `(L, kL)` exactly like the boolean mask:
    /// `-0.0` where the mask admits the key, `finfo(bfloat16).min` (0xFF7F)
    /// where it does not. Same purity argument -- a read-only constant that is
    /// a pure function of the two block lengths, eight distinct values per
    /// 1024-token chunk -- and the same bounded table.
    nonisolated(unsafe) private static var maskBiasCache: [Int: MLXArray] = [:]

    private static func causalMaskBias(L: Int, kL: Int) -> MLXArray {
        let key = L &* 1_000_003 &+ kL
        maskCacheLock.lock()
        if let hit = maskBiasCache[key] {
            maskCacheLock.unlock()
            return hit
        }
        maskCacheLock.unlock()
        let bias = MLX.where(
            causalMask(L: L, kL: kL),
            bfloat16NegativeZeroScalar,
            bfloat16LowestScalar)
        eval(bias)
        maskCacheLock.lock()
        if maskBiasCache.count >= maxCachedMasks {
            maskBiasCache.removeAll(keepingCapacity: true)
        }
        maskBiasCache[key] = bias
        maskCacheLock.unlock()
        return bias
    }

    /// Head dims for which MLX has a fused kernel; those calls must keep
    /// taking it, because the fused kernel is NOT the fallback graph.
    @inline(__always)
    private static func mlxHasFusedKernel(queryDim: Int, valueDim: Int, L: Int) -> Bool {
        guard queryDim == valueDim else { return false }
        if L > 8 { return queryDim == 64 || queryDim == 80 || queryDim == 128 }
        return queryDim == 64 || queryDim == 96 || queryDim == 128 || queryDim == 256
    }

    /// GQA plane for a whole prompt chunk: `unflatten(queries, 1, [kv, rep])`
    /// hoisted OUT of the q-block loop. MLX has no `unflatten` binding in
    /// Swift, and `reshape` on the STRIDED per-block query view would force a
    /// contiguous copy -- exactly the pass this rider exists to delete. On the
    /// row-contiguous chunk the same reshape is a pure view (`prepare_reshape`
    /// takes the `row_contiguous` branch and shares the buffer), so splitting
    /// the head axis once up front and slicing the block out of the 5-D view
    /// leaves nothing to materialize: `matmul` accepts the result directly
    /// (last axis stride 1, second-to-last stride == head dim, batch strides
    /// carried by `steel_matmul`'s batch descriptor).
    /// nil when this attention call cannot take the composed path at all.
    static func queryPlane(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray? {
        guard enabled, scale == 1.0, sinks == nil, softcap == nil else { return nil }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }
        guard queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            values.dtype == .bfloat16
        else { return nil }
        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let nKVHeads = keys.dim(1)
        let queryDim = queries.dim(3)
        guard nKVHeads > 0, values.dim(1) == nKVHeads, nQHeads % nKVHeads == 0 else {
            return nil
        }
        let nRepeats = nQHeads / nKVHeads
        guard nRepeats > 1 else { return nil }
        guard !mlxHasFusedKernel(
            queryDim: queryDim, valueDim: values.dim(3), L: queries.dim(2))
        else { return nil }
        return queries.reshaped([B, nKVHeads, nRepeats, queries.dim(2), queryDim])
    }

    /// Single-pass FlashAttention-2 online softmax kernel for Apple Silicon.
    /// Fuses QK^T dot product, causal mask, online softmax normalization,
    /// and P*V value accumulation into an 8-warp block-cooperative kernel,
    /// completely eliminating all intermediate DRAM score materializations and
    /// sharing K/V loads across queries in threadgroup SRAM.
    static let flashAttentionEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_FLASH_ATTN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let flashAttentionKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_fused_prefill_flash_sdpa_v3",
        inputNames: ["queries", "keys", "values"],
        outputNames: ["output"],
        source: """
            constexpr int simd_width = 32;
            constexpr int num_warps = 8;
            constexpr int values_per_lane = D / simd_width;
            constexpr int vecs_per_lane = values_per_lane / 4;
            constexpr float neg_infinity = -3.402823466e+38F;

            typedef vec<T, 4> T4;
            typedef vec<float, 4> float4;

            const int q_row_base = int(threadgroup_position_in_grid.y) * num_warps;
            const int local_y = int(thread_position_in_threadgroup.y);
            const int lane = int(thread_position_in_threadgroup.x);
            const int local_id = local_y * simd_width + lane;

            const int q_row = q_row_base + local_y;
            const bool valid_q = (q_row < L);

            const int batch_head = int(threadgroup_position_in_grid.z);
            const int q_head = batch_head % N_Q_HEADS;
            const int batch_idx = batch_head / N_Q_HEADS;
            const int kv_head = q_head / GQA;

            const int q_abs_pos = (KL - L) + q_row;
            const int q_head_stride = L * D;
            const int k_head_stride = KL * D;
            const int v_head_stride = KL * D;
            const int o_head_stride = L * D;

            float4 q_reg[vecs_per_lane];
            if (valid_q) {
                const device T* q_lane = queries
                    + batch_idx * (N_Q_HEADS * q_head_stride)
                    + q_head * q_head_stride
                    + q_row * D + lane * values_per_lane;
                #pragma clang loop unroll(full)
                for (int e = 0; e < vecs_per_lane; ++e) {
                    const device T4* q_vec = (const device T4*)(q_lane + e * 4);
                    T4 val = *q_vec;
                    q_reg[e] = {float(val[0]), float(val[1]), float(val[2]), float(val[3])};
                }
            }

            float4 O_acc[vecs_per_lane];
            #pragma clang loop unroll(full)
            for (int e = 0; e < vecs_per_lane; ++e) {
                O_acc[e] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            float max_score = neg_infinity;
            float sum_exp = 0.0f;

            const int k_end = valid_q ? (q_abs_pos + 1) : 0;

            const int max_q_row = min(L - 1, q_row_base + num_warps - 1);
            const int max_q_abs_pos = (KL - L) + max_q_row;
            const int group_k_end = max_q_abs_pos + 1;

            const device T* k_base = keys + batch_idx * (N_KV_HEADS * k_head_stride) + kv_head * k_head_stride;
            const device T* v_base = values + batch_idx * (N_KV_HEADS * v_head_stride) + kv_head * v_head_stride;

            threadgroup T4 k_shared[128];
            threadgroup T4 v_shared[128];

            const int num_vecs = D / 4;

            for (int k_idx = 0; k_idx < group_k_end; ++k_idx) {
                for (int i = local_id; i < num_vecs; i += num_warps * simd_width) {
                    const device T4* k_vec_ptr = (const device T4*)(k_base + k_idx * D + i * 4);
                    const device T4* v_vec_ptr = (const device T4*)(v_base + k_idx * D + i * 4);
                    k_shared[i] = *k_vec_ptr;
                    v_shared[i] = *v_vec_ptr;
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (valid_q && k_idx < k_end) {
                    float dot_val = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < vecs_per_lane; ++e) {
                        int shared_idx = lane * vecs_per_lane + e;
                        T4 k_val = k_shared[shared_idx];
                        float4 k_f4 = {float(k_val[0]), float(k_val[1]), float(k_val[2]), float(k_val[3])};
                        dot_val += q_reg[e][0] * k_f4[0] + q_reg[e][1] * k_f4[1] + q_reg[e][2] * k_f4[2] + q_reg[e][3] * k_f4[3];
                    }
                    dot_val = simd_sum(dot_val);

                    const float new_max = max(max_score, dot_val);
                    const float alpha = (max_score == neg_infinity) ? 0.0f : fast::exp(max_score - new_max);
                    const float score_factor = fast::exp(dot_val - new_max);
                    max_score = new_max;
                    sum_exp = sum_exp * alpha + score_factor;

                    #pragma clang loop unroll(full)
                    for (int e = 0; e < vecs_per_lane; ++e) {
                        int shared_idx = lane * vecs_per_lane + e;
                        T4 v_val = v_shared[shared_idx];
                        float4 v_f4 = {float(v_val[0]), float(v_val[1]), float(v_val[2]), float(v_val[3])};
                        O_acc[e][0] = O_acc[e][0] * alpha + score_factor * v_f4[0];
                        O_acc[e][1] = O_acc[e][1] * alpha + score_factor * v_f4[1];
                        O_acc[e][2] = O_acc[e][2] * alpha + score_factor * v_f4[2];
                        O_acc[e][3] = O_acc[e][3] * alpha + score_factor * v_f4[3];
                    }
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (valid_q) {
                const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                device T* out_lane = output
                    + batch_idx * (N_Q_HEADS * o_head_stride)
                    + q_head * o_head_stride
                    + q_row * D + lane * values_per_lane;
                
                #pragma clang loop unroll(full)
                for (int e = 0; e < vecs_per_lane; ++e) {
                    device T4* out_vec = (device T4*)(out_lane + e * 4);
                    T4 val = {
                        T(O_acc[e][0] * inv_sum),
                        T(O_acc[e][1] * inv_sum),
                        T(O_acc[e][2] * inv_sum),
                        T(O_acc[e][3] * inv_sum)
                    };
                    *out_vec = val;
                }
            }
        """,
        ensureRowContiguous: true
    )

    /// The fast.cpp SDPA fallback, minus the identity query scale.
    /// Returns nil when this call is not provably that graph.
    /// `queryPlaneSlice` is this block's view of `queryPlane(...)` when the
    /// caller could hoist it; nil means reshape here instead.
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, L: Int, kL: Int, window: Int?,
        bidirectional: Bool, sinks: MLXArray?,
        queryPlaneSlice: MLXArray? = nil
    ) -> MLXArray? {
        guard enabled, scale == 1.0, sinks == nil, !bidirectional else { return nil }
        // Decode (L == 1) and every MTP verify width (L in 2...8) keep the
        // stock path: this rider is the prompt plane only.
        guard L > 8 else { return nil }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }
        guard queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            values.dtype == .bfloat16
        else { return nil }
        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let queryDim = queries.dim(3)
        let valueDim = values.dim(3)
        guard queries.dim(2) == L, keys.dim(2) == kL, values.dim(2) == kL else { return nil }
        guard keys.dim(0) == B, values.dim(0) == B else { return nil }
        guard keys.dim(3) == queryDim, valueDim == queryDim else { return nil }
        guard !mlxHasFusedKernel(queryDim: queryDim, valueDim: valueDim, L: L) else { return nil }
        // Symbolic `.causal` only: an array mask (kL > window) keeps the
        // stock call, whose fallback reshapes the broadcast mask.
        if let window, kL > window { return nil }
        guard kL >= L else { return nil }
        let nKVHeads = keys.dim(1)
        guard nKVHeads > 0, values.dim(1) == nKVHeads, nQHeads % nKVHeads == 0 else {
            return nil
        }
        let nRepeats = nQHeads / nKVHeads

        if flashAttentionEnabled, (queryDim == 256 || queryDim == 512) {
            CBv2EngageMark.once("prefill-flash-sdpa")
            let numGroupsY = (L + 7) / 8
            return flashAttentionKernel(
                [queries, keys, values],
                template: [
                    ("T", queries.dtype),
                    ("D", queryDim),
                    ("GQA", nRepeats),
                    ("N_Q_HEADS", nQHeads),
                    ("N_KV_HEADS", nKVHeads),
                    ("L", L),
                    ("KL", kL),
                ],
                grid: (32, numGroupsY * 8, B * nQHeads),
                threadGroup: (32, 8, 1),
                outputShapes: [[B, nQHeads, L, queryDim]],
                outputDTypes: [queries.dtype]
            )[0]
        }

        var q = queries
        var k = keys
        var v = values
        if nRepeats > 1 {
            guard let plane = queryPlaneSlice,
                plane.ndim == 5, plane.dim(0) == B, plane.dim(1) == nKVHeads,
                plane.dim(2) == nRepeats, plane.dim(3) == L, plane.dim(4) == queryDim,
                plane.dtype == queries.dtype
            else { return nil }
            q = plane
            k = expandedDimensions(k, axis: 2)
            v = expandedDimensions(v, axis: 2)
        }

        CBv2EngageMark.once("prefill-sdpa-compose")
        var scores: MLXArray
        if maskFuseEnabled {
            CBv2EngageMark.once("prefill-mask-fuse")
            scores = addMM(causalMaskBias(L: L, kL: kL), q, k.swappedAxes(-1, -2))
        } else {
            scores = matmul(q, k.swappedAxes(-1, -2))
            scores = MLX.where(causalMask(L: L, kL: kL), scores, bfloat16LowestScalar)
        }

        scores = MLX.softmax(scores, axis: -1, precise: true)
        var output = matmul(scores, v)
        if nRepeats > 1 {
            output = output.reshaped([B, nQHeads, L, valueDim])
        }
        return output
    }
}
