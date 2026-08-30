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
import MLXFast

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

    /// PREFILL-SOFTMAX-SHAPE. The score rectangle's row length is a
    /// COMPILE-TIME constant in this kernel instead of a `constant int&`
    /// argument. Kill switch: `DARKBLOOM_CBV2_PREFILL_SOFTMAX_SHAPE=0`.
    static let shapedSoftmaxEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_SOFTMAX_SHAPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Row lengths this rider specialises. `kL % 128 == 0` is what makes the
    /// specialisation exact: `softmax.cpp` sizes the threadgroup at
    /// `32 * ceil(ceil(kL/4)/32)`, which for a multiple of 128 is EXACTLY
    /// `kL/4` threads, so every thread's `lid*4 + 4 <= axis_size` test is true
    /// by construction and stock never enters its guarded arm. `kL <= 1024`
    /// bounds the family at eight distinct kernels -- the ranked prefill
    /// geometry is 8 rows x 1024 tokens in q-blocks of 128, and the sliding
    /// window is 1024, so those eight cover every scored call. A longer
    /// full-attention history keeps the stock call rather than adding another
    /// JIT compile to a measured window.
    private static let maxShapedSoftmaxAxis = 1024

    nonisolated(unsafe) private static var softmaxKernelCache:
        [Int: MLXFast.MLXFastKernel] = [:]
    private static let softmaxKernelLock = NSLock()

    /// `block_softmax_precise_bfloat16` with `axis_size` promoted from a
    /// runtime argument to a literal, and the prologue that exists only to
    /// initialise the reduction pads replaced by lane-bounded reads.
    ///
    /// Two mechanical differences from `softmax.h`, both exact:
    ///
    ///  1. `N` and `NSG = N/128` are literals. The load and store guards
    ///     `lid * 4 + 4 <= axis_size` fold to `true` (the caller dispatches
    ///     exactly `N/4` threads, so the largest `lid*4 + 4` is `N`), the
    ///     guarded arms become dead code, and `gid * size_t(axis_size)`
    ///     becomes a shift. Same values, same order, one branch fewer per
    ///     thread on each of the two passes over the row.
    ///
    ///  2. Stock opens with simdgroup 0 writing `Limits<float>::min` and `0`
    ///     across all 32 entries of both reduction scratch arrays, then a
    ///     threadgroup barrier, purely so that the final cross-simdgroup
    ///     reduction reads sane values from the `32 - NSG` lanes no simdgroup
    ///     wrote. With `NSG` a literal those lanes can supply their own
    ///     identity instead: `simd_lane_id < NSG ? local_max[simd_lane_id]
    ///     : -INFINITY` and `... : 0.0f`. `Limits<float>::min` IS `-INFINITY`
    ///     (`utils.h`, `instantiate_float_limit`), so the values entering
    ///     `simd_max` and `simd_sum` are bit-identical to stock's in
    ///     bit-identical lanes and the xor butterflies see the same tree. That
    ///     deletes two threadgroup stores and one full threadgroup barrier per
    ///     row.
    ///
    /// The pad identities are exact, not approximate: `max(x, -INFINITY) == x`
    /// for every `x` including NaN-free score maxima, and `x + 0.0f == x` for
    /// every `x` other than `-0.0f`. A simdgroup's `normalizer` partial is a
    /// sum of `fast::exp` results, each `>= +0.0`, so it is never `-0.0`.
    private static func shapedSoftmaxKernel(axisSize N: Int) -> MLXFast.MLXFastKernel {
        softmaxKernelLock.lock()
        if let hit = softmaxKernelCache[N] {
            softmaxKernelLock.unlock()
            return hit
        }
        softmaxKernelLock.unlock()
        let kernel = MLXFast.metalKernel(
            name: "cbv2_prefill_block_softmax_precise_n\(N)_v1",
            inputNames: ["scores"],
            outputNames: ["probs"],
            source: """
                constexpr int N = \(N);
                constexpr int NSG = \(N / 128);

                const uint gid = threadgroup_position_in_grid.x;
                const int lid = int(thread_position_in_threadgroup.x);
                const int simd_lane_id = int(thread_index_in_simdgroup);
                const int simd_group_id = int(simdgroup_index_in_threadgroup);

                threadgroup float local_max[32];
                threadgroup float local_normalizer[32];

                float ld[4];
                const device T* in = scores + gid * size_t(N) + lid * 4;
                for (int i = 0; i < 4; i++) {
                    ld[i] = static_cast<float>(in[i]);
                }

                float maxval = -3.402823466e+38F;
                for (int i = 0; i < 4; i++) {
                    maxval = (maxval < ld[i]) ? ld[i] : maxval;
                }
                maxval = simd_max(maxval);
                if (simd_lane_id == 0) {
                    local_max[simd_group_id] = maxval;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    maxval = simd_max(
                        (simd_lane_id < NSG) ? local_max[simd_lane_id] : -INFINITY);
                    if (simd_lane_id == 0) {
                        local_max[0] = maxval;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maxval = local_max[0];

                float normalizer = 0.0f;
                for (int i = 0; i < 4; i++) {
                    float exp_x = fast::exp(ld[i] - maxval);
                    ld[i] = exp_x;
                    normalizer += exp_x;
                }
                normalizer = simd_sum(normalizer);
                if (simd_lane_id == 0) {
                    local_normalizer[simd_group_id] = normalizer;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    normalizer = simd_sum(
                        (simd_lane_id < NSG) ? local_normalizer[simd_lane_id] : 0.0f);
                    if (simd_lane_id == 0) {
                        local_normalizer[0] = normalizer;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                normalizer = 1 / local_normalizer[0];

                device T* out_row = probs + gid * size_t(N) + lid * 4;
                for (int i = 0; i < 4; i++) {
                    out_row[i] = static_cast<T>(ld[i] * normalizer);
                }
                """,
            ensureRowContiguous: true
        )
        softmaxKernelLock.lock()
        softmaxKernelCache[N] = kernel
        softmaxKernelLock.unlock()
        return kernel
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
        guard keys.dim(3) == queryDim else { return nil }
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

        var q = queries
        var k = keys
        var v = values
        if nRepeats > 1 {
            // STRICT: without the hoisted plane the GQA split would have to
            // `reshape` a strided q-block, which copies -- trading the deleted
            // identity multiply for a copy of the same size instead of
            // deleting it. Refuse rather than break even.
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
        // PREFILL-MASK-FUSE: the causal mask is applied by the QK^T GEMM's
        // OWN epilogue instead of by a second full pass over the score
        // rectangle. `steel_gemm_fused`'s `use_out_source` epilogue is
        // `TransformAdd::apply(acc, C) = float(acc) + float(C)` evaluated on
        // the fp32 accumulator BEFORE the single bfloat16 store, so:
        //   * unmasked (bias -0.0): `acc + (-0.0) == acc` for every fp32
        //     value, signed zeros included, and the store rounds exactly as
        //     the plain matmul's `TransformNone` store does;
        //   * masked (bias 0xFF7F = -3.3895e38): ulp(3.39e38) is 2^104, so
        //     `acc + bias` rounds to `bias` for any score with |acc| < 2^103
        //     -- every attention score this model produces by many orders of
        //     magnitude -- and the store yields exactly 0xFF7F, the same word
        //     `where(mask, scores, finfo(bf16).min)` writes.
        // `c` rides the GEMM as a broadcast (batch strides 0, ldc = kL): MLX
        // passes its strides straight through to the kernel, so nothing is
        // materialized and the [L, kL] bias is read once per output tile out
        // of cache instead of the whole rectangle being read and rewritten.
        var scores: MLXArray
        if maskFuseEnabled {
            CBv2EngageMark.once("prefill-mask-fuse")
            scores = addMM(causalMaskBias(L: L, kL: kL), q, k.swappedAxes(-1, -2))
        } else {
            scores = matmul(q, k.swappedAxes(-1, -2))
            scores = MLX.where(causalMask(L: L, kL: kL), scores, bfloat16LowestScalar)
        }

        // PREFILL-SOFTMAX-SHAPE: the row length is a literal in the kernel
        // rather than a `constant int&`, and the reduction pads carry their
        // own identity instead of being written by a prologue. Only for row
        // lengths where stock's threadgroup is exactly `kL/4` threads, which
        // is what makes the collapsed guards the arm stock already takes.
        if shapedSoftmaxEnabled, scores.dtype == .bfloat16,
            kL % 128 == 0, kL <= maxShapedSoftmaxAxis, scores.size % kL == 0
        {
            CBv2EngageMark.once("prefill-softmax-shape")
            let threads = kL / 4
            let rows = scores.size / kL
            scores = shapedSoftmaxKernel(axisSize: kL)(
                [scores],
                template: [("T", scores.dtype)],
                grid: (threads * rows, 1, 1),
                threadGroup: (threads, 1, 1),
                outputShapes: [scores.shape],
                outputDTypes: [scores.dtype]
            )[0]
        } else {
            scores = MLX.softmax(scores, axis: -1, precise: true)
        }
        var output = matmul(scores, v)
        if nRepeats > 1 {
            output = output.reshaped([B, nQHeads, L, valueDim])
        }
        return output
    }
}
