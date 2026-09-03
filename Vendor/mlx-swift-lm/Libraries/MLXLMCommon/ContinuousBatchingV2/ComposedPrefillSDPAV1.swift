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

    nonisolated(unsafe) private static let standard1024Mask: MLXArray = {
        let qIndices = MLXArray(Int32(0) ..< Int32(1024)).expandedDimensions(axis: 1)
        let kIndices = MLXArray(Int32(0) ..< Int32(1024)).expandedDimensions(axis: 0)
        let mask = qIndices .>= kIndices
        eval(mask)
        return mask
    }()

    nonisolated(unsafe) private static let standard1024MaskBias: MLXArray = {
        let qIndices = MLXArray(Int32(0) ..< Int32(1024)).expandedDimensions(axis: 1)
        let kIndices = MLXArray(Int32(0) ..< Int32(1025)).expandedDimensions(axis: 0)
        let padded = MLX.where(
            qIndices .>= kIndices,
            bfloat16NegativeZeroScalar,
            bfloat16LowestScalar)
        eval(padded)
        let bias = padded[0..., 0 ..< 1024]
        eval(bias)
        return bias
    }()

    private static func causalMask(L: Int, kL: Int) -> MLXArray {
        if L == 1024 && kL == 1024 {
            return standard1024Mask
        }
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

    /// CAUSAL-CLOAD (concept receipt: solver i34-9, submission d0ccbe3c).
    /// When enabled, the cached bias is built over `kL + 1` key columns and
    /// the array handed to `addMM` is the leading-`kL`-column view of that
    /// storage. The logical operand is unchanged — still exactly `[L, kL]`,
    /// still bfloat16 negative zero where a score is admitted and the lowest
    /// finite bfloat16 where it is masked — only its row stride changes,
    /// from `kL` to `kL + 1`. That stride is the signature the fused Steel
    /// GEMM recognizes to SYNTHESIZE this operand in its epilogue instead of
    /// loading it; any kernel branch that does not synthesize still loads
    /// correct values through the same stride. The padding column is filled
    /// by the same comparison rule as every other column and never read.
    /// Kill switch: `DARKBLOOM_CBV2_PREFILL_MASK_SYNTH=0` (unpadded bias,
    /// stride kL — the kernel signature then never fires).
    static let maskSynthEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_MASK_SYNTH"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Causal mask BIAS, memoized on `(L, kL)` exactly like the boolean mask:
    /// `-0.0` where the mask admits the key, `finfo(bfloat16).min` (0xFF7F)
    /// where it does not. Same purity argument -- a read-only constant that is
    /// a pure function of the two block lengths, eight distinct values per
    /// 1024-token chunk -- and the same bounded table.
    nonisolated(unsafe) private static var maskBiasCache: [Int: MLXArray] = [:]

    private static func causalMaskBias(L: Int, kL: Int) -> MLXArray {
        if maskSynthEnabled && L == 1024 && kL == 1024 {
            return standard1024MaskBias
        }
        let key = L &* 1_000_003 &+ kL
        maskCacheLock.lock()
        if let hit = maskBiasCache[key] {
            maskCacheLock.unlock()
            return hit
        }
        maskCacheLock.unlock()
        let bias: MLXArray
        if maskSynthEnabled {
            // Padded storage, sliced view: the comparison is expressed
            // directly over query and key index vectors (not the shared
            // boolean-mask helper) so the inert padding column is produced
            // by the same rule as every real column.
            let qIndices = MLXArray(Int32(kL - L) ..< Int32(kL))
                .expandedDimensions(axis: 1)
            let kIndices = MLXArray(Int32(0) ..< Int32(kL + 1))
                .expandedDimensions(axis: 0)
            let padded = MLX.where(
                qIndices .>= kIndices,
                bfloat16NegativeZeroScalar,
                bfloat16LowestScalar)
            eval(padded)
            bias = padded[0..., 0 ..< kL]
            CBv2EngageMark.once("prefill-mask-synth-bias")
        } else {
            bias = MLX.where(
                causalMask(L: L, kL: kL),
                bfloat16NegativeZeroScalar,
                bfloat16LowestScalar)
        }
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

        var output: MLXArray
        if let fused = CBv2PrefillAttnTrafficV1.attend(scores: scores, values: v) {
            // PREFILL-ATTN-TRAFFIC (at1): the softmax is applied by the P.V
            // GEMM's own A loader from a per-row statistics pre-pass; the
            // probability rectangle is never written or read.
            output = fused
        } else {
            if let vecScores = CBv2PrefillSoftmaxVecV1.apply(scores) {
                scores = vecScores
            } else {
                scores = MLX.softmax(scores, axis: -1, precise: true)
            }
            output = matmul(scores, v)
        }
        if nRepeats > 1 {
            output = output.reshaped([B, nQHeads, L, valueDim])
        }
        return output
    }
}

/// PREFILL-SOFTMAX-VEC. A vectorized, bit-exact transcription of
/// `softmax_single_row<bfloat16_t, float, N_READS=4>`
/// (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/softmax.h`)
/// for the `MLX.softmax(scores, axis: -1, precise: true)` call in
/// `CBv2ComposedPrefillSDPAV1.attend`. MLX's own dispatch
/// (`.../backend/metal/softmax.cpp:53,64-68`) selects this exact kernel --
/// the non-looped "block" path -- whenever `axis_size <= 4096`, sizing the
/// threadgroup at `32 * ceil(ceil(axis_size / 4) / 32)`; this transcription
/// mirrors both the kernel body and that sizing rule
/// (`RaggedTwoPassDecodeAttentionV1.swift`'s `softmaxKernel` already hosts a
/// scalar, non-vectorized transcription of the same stock kernel for the
/// D=512 decode chain -- same values, same barrier count as the stock
/// kernel. This is the vectorized, fewer-barrier sibling for prefill.).
///
/// Two changes from the stock kernel. Neither touches a value.
///
/// 1. VECTOR LOADS/STORES, gated on `kL % 4 == 0` at the host (`apply`
///    below; kL % 4 != 0 keeps the stock call). Once every row's length is
///    a multiple of 4, every row starts at an offset that is itself a
///    multiple of 4 elements from the row-contiguous buffer's base (each
///    row is `axis_size` elements after the previous one), so
///    `lid * N_READS + N_READS <= axis_size` becomes EXACTLY the same
///    predicate as `lid * N_READS < axis_size` -- the in-bounds boundary
///    itself always sits on a 4-element multiple, so no lane can ever be
///    PARTIALLY in bounds. That collapses the stock kernel's per-element
///    tail ternary (`(lid*N_READS+i < axis_size) ? AccT(in[i]) :
///    Limits<AccT>::min`, evaluated once per i) to one per-lane branch
///    (`row_valid`) that is provably equivalent for every lane: true
///    reproduces the stock in-bounds branch's four scalar loads verbatim,
///    as one `vec<T, 4>` load of the same four addresses in the same
///    order; false reproduces the stock tail branch's four
///    `Limits<AccT>::min` (`-INFINITY` for `AccT = float`) fills verbatim,
///    since with kL % 4 == 0 the tail branch is never PARTIALLY true. The
///    store side is the same argument in reverse. Only the load/store
///    WIDTH changes -- a promoted-kernel load-sharing move, not a value
///    change.
///
/// 2. BARRIER COUNT: 5 becomes 2. The stock kernel zero-fills all 32
///    `local_max`/`local_normalizer` slots (`Limits<AccT>::min` / `0`)
///    behind a barrier so slots beyond the live simdgroup count hold a
///    defined identity, then lets ONLY simdgroup 0 read all 32 slots and
///    publish the combined result back to slot 0 behind two more barriers
///    (one per array) for every other simdgroup to read. This kernel
///    instead has EVERY simdgroup read the same 32 conceptual values
///    directly: the just-written partial in slots `[0, num_simdgroups)`,
///    and the literal identity substituted in a REGISTER (never read from
///    threadgroup memory, so the stale contents of an unfilled slot are
///    never touched) for the rest -- and each simdgroup runs its own
///    `simd_max` / `simd_sum` over that set. Those two intrinsics are the
///    SAME instruction sequence over the SAME per-lane operands regardless
///    of which physical simdgroup issues them -- the reduction hierarchy
///    is a pure function of lane index, not of which simdgroup executes
///    it, the same "promoted-kernel invariant" this file's neighbor
///    `RaggedTwoPassDecodeAttentionV1.swift` already relies on for its
///    XFOLD butterfly ("the merge hierarchy is therefore the SAME for
///    every lane ... only the left/right order at each node varies with
///    the lane, and float addition is commutative, so every sum is
///    bit-identical") -- so every simdgroup lands on the bit-identical
///    scalar the stock kernel's slot-0 broadcast would have handed it,
///    without needing to broadcast anything.
///
///    That removal deletes: the zero-fill write and its barrier (nothing
///    ever reads an unfilled slot, so nothing needs the identity written
///    there first); and both publish-back round trips and their barriers
///    (every simdgroup already independently holds the combined answer,
///    so nothing needs to read a value another simdgroup broadcast). The
///    two barriers that survive are the ones ordering "every simdgroup's
///    own partial write to `local_max` / `local_normalizer` is visible
///    threadgroup-wide" before "every simdgroup reads that array" -- for
///    each array, that write/read pair is the one true cross-simdgroup
///    dependency in this kernel, and no barrier can be proven to order it
///    away.
///
/// Fails closed (keeps the stock `MLX.softmax` call, byte-preserved) on:
/// env kill-switch (default ON), a non-bfloat16 row, `kL % 4 != 0`, or
/// `kL` outside `(0, 4096]` (the block/non-looped kernel-selection window;
/// `kL > 4096` is `softmax_looped`, a different kernel this file does not
/// transcribe).
enum CBv2PrefillSoftmaxVecV1 {

    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PREFILL_SOFTMAX_VEC"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// softmax.cpp's SOFTMAX_LOOPED_LIMIT. Only `axis_size <= 4096` takes
    /// the non-looped "block" kernel this file transcribes.
    private static let maxKeyLength = 4096

    /// Verbatim transcription of `softmax_single_row<bfloat16_t, float, 4>`
    /// (kernels/softmax.h) -- see the enum doc comment above for the two
    /// load/store and barrier changes and why neither touches a value.
    /// params[0] = axis_size (kL); params[1] = num_simdgroups
    /// (threadgroup size / 32, computed on the host from the SAME sizing
    /// rule softmax.cpp uses to pick the threadgroup size, so it always
    /// equals what this dispatch's own threadGroup.x implies).
    private static let source = """
        const int axis_size = int(params[0]);
        const int num_simdgroups = int(params[1]);

        const int gid = int(threadgroup_position_in_grid.x);
        const int lid = int(thread_position_in_threadgroup.x);
        const int simd_lane_id = int(thread_index_in_simdgroup);
        const int simd_group_id = int(simdgroup_index_in_threadgroup);

        threadgroup float local_max[32];
        threadgroup float local_normalizer[32];

        typedef vec<T, 4> T4;

        float ld[4];
        const int base = lid * 4;
        const bool row_valid = base < axis_size;
        const device T* row_in = scores + size_t(gid) * axis_size;
        if (row_valid) {
            T4 raw = *reinterpret_cast<const device T4*>(row_in + base);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ld[i] = static_cast<float>(raw[i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ld[i] = -INFINITY;
            }
        }

        // MAX. Phase-2 write is UNCHANGED from the stock kernel: only
        // simdgroup_id == 0's lane owns slot 0, only simdgroup_id == 1's
        // lane owns slot 1, and so on -- exactly the stock
        // `local_max[simd_group_id] = maxval` write.
        float maxval = -3.402823466e+38F;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            maxval = (maxval < ld[i]) ? ld[i] : maxval;
        }
        maxval = simd_max(maxval);
        if (simd_lane_id == 0) {
            local_max[simd_group_id] = maxval;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Every simdgroup reads the SAME 32 conceptual values the
            // stock kernel's simdgroup-0-only combine read: the real
            // partial in slots < num_simdgroups (just published above,
            // visible threadgroup-wide after the barrier), and the
            // literal max-identity (-INFINITY, i.e. Limits<float>::min)
            // in the rest -- substituted here in a register instead of
            // read back from a zero-filled slot, since no thread ever
            // wrote a real value there. Same simd_max instruction, same
            // per-lane operand set as the stock combine: bit-identical
            // result in every simdgroup, no publish-back needed.
            float slot = (simd_lane_id < num_simdgroups)
                ? local_max[simd_lane_id] : -INFINITY;
            maxval = simd_max(slot);
        }

        float normalizer = 0.0f;
        #pragma unroll
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
        {
            // Same argument as the max combine above, for the sum: the
            // sum-identity is 0.0f (Limits<float> has no analogue here --
            // the stock kernel zero-fills `local_normalizer` directly).
            float slot = (simd_lane_id < num_simdgroups)
                ? local_normalizer[simd_lane_id] : 0.0f;
            normalizer = simd_sum(slot);
        }
        normalizer = 1.0f / normalizer;

        if (row_valid) {
            device T* row_out = probs + size_t(gid) * axis_size;
            T4 result;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                result[i] = static_cast<T>(ld[i] * normalizer);
            }
            *reinterpret_cast<device T4*>(row_out + base) = result;
        }
        """

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_prefill_sdpa_softmax_vec_bf16_v1",
        inputNames: ["scores", "params"],
        outputNames: ["probs"],
        source: source,
        ensureRowContiguous: true
    )

    /// PROMPT-GLUE2 (pg2): the transcription above with `RPT` rows per
    /// threadgroup. One row per threadgroup leaves 16384 threadgroups of
    /// 32..256 threads per query block and the dispatch bound by
    /// threadgroup residency rather than by its bytes. Here `RPT` rows
    /// share a threadgroup of `RPT * threadgroupSize` threads: thread
    /// `tid` serves row `row_slot = tid / row_threads` with the incumbent's
    /// `lid`, `simd_lane_id` and row-local `simd_group_id` (row_threads is
    /// a multiple of 32, so a row's simdgroups are whole simdgroups), and
    /// each row reduces through its own slot of the shared arrays. The
    /// per-lane loads, the max and sum trees, the exp and the store are
    /// the incumbent's text over the same lanes and the same operands; the
    /// two barriers order the same write/read pairs, now for every row of
    /// the threadgroup at once. A trailing partial threadgroup (when the
    /// row count is not a multiple of `RPT`) holds whole rows, since the
    /// grid is a multiple of the row width.
    /// Internal (not private): `CBv2PrefillAttnTrafficV1` derives its
    /// row-statistics twin from this exact text.
    static let sourcePg2 = """
        const int axis_size = int(params[0]);
        const int num_simdgroups = int(params[1]);

        // PROMPT-GLUE2 (pg2): RPT rows share one threadgroup. `row_slot` is
        // this thread's row within the threadgroup; `gid`, `lid`, `simd_lane_id`
        // and `simd_group_id` are the incumbent's values for that row (row_threads
        // is a multiple of 32, so a row's simdgroups are whole simdgroups), and
        // the body below is the incumbent's text over the same lanes and the same
        // operands, indexing the row's own slot of the shared arrays.
        const int row_threads = num_simdgroups * 32;
        const int tid = int(thread_position_in_threadgroup.x);
        const int row_slot = tid / row_threads;
        const int gid = int(threadgroup_position_in_grid.x) * RPT + row_slot;
        const int lid = tid - row_slot * row_threads;
        const int simd_lane_id = int(thread_index_in_simdgroup);
        const int simd_group_id = int(simdgroup_index_in_threadgroup) - row_slot * num_simdgroups;

        threadgroup float local_max[RPT][32];
        threadgroup float local_normalizer[RPT][32];

        typedef vec<T, 4> T4;

        float ld[4];
        const int base = lid * 4;
        const bool row_valid = base < axis_size;
        const device T* row_in = scores + size_t(gid) * axis_size;
        if (row_valid) {
            T4 raw = *reinterpret_cast<const device T4*>(row_in + base);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ld[i] = static_cast<float>(raw[i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ld[i] = -INFINITY;
            }
        }

        // MAX. Phase-2 write is UNCHANGED from the stock kernel: only
        // simdgroup_id == 0's lane owns slot 0, only simdgroup_id == 1's
        // lane owns slot 1, and so on -- exactly the stock
        // `local_max[row_slot][simd_group_id] = maxval` write.
        float maxval = -3.402823466e+38F;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            maxval = (maxval < ld[i]) ? ld[i] : maxval;
        }
        maxval = simd_max(maxval);
        if (simd_lane_id == 0) {
            local_max[row_slot][simd_group_id] = maxval;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Every simdgroup reads the SAME 32 conceptual values the
            // stock kernel's simdgroup-0-only combine read: the real
            // partial in slots < num_simdgroups (just published above,
            // visible threadgroup-wide after the barrier), and the
            // literal max-identity (-INFINITY, i.e. Limits<float>::min)
            // in the rest -- substituted here in a register instead of
            // read back from a zero-filled slot, since no thread ever
            // wrote a real value there. Same simd_max instruction, same
            // per-lane operand set as the stock combine: bit-identical
            // result in every simdgroup, no publish-back needed.
            float slot = (simd_lane_id < num_simdgroups)
                ? local_max[row_slot][simd_lane_id] : -INFINITY;
            maxval = simd_max(slot);
        }

        float normalizer = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        if (simd_lane_id == 0) {
            local_normalizer[row_slot][simd_group_id] = normalizer;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Same argument as the max combine above, for the sum: the
            // sum-identity is 0.0f (Limits<float> has no analogue here --
            // the stock kernel zero-fills `local_normalizer` directly).
            float slot = (simd_lane_id < num_simdgroups)
                ? local_normalizer[row_slot][simd_lane_id] : 0.0f;
            normalizer = simd_sum(slot);
        }
        normalizer = 1.0f / normalizer;

        if (row_valid) {
            device T* row_out = probs + size_t(gid) * axis_size;
            T4 result;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                result[i] = static_cast<T>(ld[i] * normalizer);
            }
            *reinterpret_cast<device T4*>(row_out + base) = result;
        }
        """

    private static let kernelPg2: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_prefill_sdpa_softmax_vec_bf16_pg2",
        inputNames: ["scores", "params"],
        outputNames: ["probs"],
        source: sourcePg2,
        ensureRowContiguous: true
    )

    /// Rows per threadgroup for the pg2 twin, by key length: the measured
    /// optimum of the ranked geometry (any value is exact); other lengths
    /// take about 640 threads per threadgroup.
    static func rowsPerThreadgroup(axisSize: Int, threadgroupSize: Int) -> Int {
        let table: [Int: Int] = [
            1024: 2, 896: 2, 768: 3, 640: 4, 512: 6, 384: 6, 256: 10, 128: 16,
        ]
        if let rows = table[axisSize] { return rows }
        return max(1, min(640 / threadgroupSize, 1024 / threadgroupSize))
    }

    nonisolated(unsafe) private static let precomputedParams: [Int: MLXArray] = {
        var table: [Int: MLXArray] = [:]
        for axis in [128, 256, 384, 512, 640, 768, 896, 1024] {
            let tg = ((axis + 3) / 4 + 31) / 32 * 32
            let numSimdgroups = tg / 32
            let arr = MLXArray([UInt32(axis), UInt32(numSimdgroups)])
            eval(arr)
            table[axis] = arr
        }
        return table
    }()

    @inline(__always)
    private static func getParams(axisSize: Int, numSimdgroups: Int) -> MLXArray {
        if let hit = precomputedParams[axisSize] {
            return hit
        }
        return MLXArray([UInt32(axisSize), UInt32(numSimdgroups)])
    }

    /// Runs the vectorized softmax, or returns nil to keep the caller on
    /// the stock `MLX.softmax(scores, axis: -1, precise: true)` call.
    /// `scores` may be any contiguous rank; it is treated as a flat
    /// `[n_rows, axis_size]` exactly like MLX's own C++ dispatch
    /// (`n_rows = in.data_size() / axis_size`), and the output keeps
    /// `scores`'s own shape.
    static func apply(_ scores: MLXArray) -> MLXArray? {
        guard enabled, scores.dtype == .bfloat16, scores.ndim >= 1 else { return nil }
        let axisSize = scores.dim(scores.ndim - 1)
        guard axisSize > 0, axisSize % 4 == 0, axisSize <= maxKeyLength else { return nil }
        let totalElements = scores.shape.reduce(1, *)
        guard totalElements > 0, totalElements % axisSize == 0 else { return nil }
        let nRows = totalElements / axisSize
        // softmax.cpp:64-68: 32 * ceil(ceil(axis_size / 4) / 32).
        let threadgroupSize = ((axisSize + 3) / 4 + 31) / 32 * 32
        guard threadgroupSize > 0, threadgroupSize <= 1024 else { return nil }
        let numSimdgroups = threadgroupSize / 32
        let paramsArray = getParams(axisSize: axisSize, numSimdgroups: numSimdgroups)

        // PROMPT-GLUE2 (pg2): prompt-width score rectangles take the
        // rows-per-threadgroup twin; the incumbent computes the identical
        // words for every other rectangle.
        if Gemma4PromptGlue2V1.enabled, nRows >= Gemma4PromptGlue2V1.minRows {
            let rows = rowsPerThreadgroup(axisSize: axisSize, threadgroupSize: threadgroupSize)
            if rows > 1, rows * threadgroupSize <= 1024 {
                CBv2EngageMark.once("prefill-softmax-vec")
                Gemma4PromptGlue2V1.mark()
                let probs = kernelPg2(
                    [scores, paramsArray],
                    template: [("T", scores.dtype), ("RPT", rows)],
                    grid: (threadgroupSize * nRows, 1, 1),
                    threadGroup: (threadgroupSize * rows, 1, 1),
                    outputShapes: [scores.shape],
                    outputDTypes: [.bfloat16]
                )[0]
                if Gemma4PromptGlue2V1.xcheck {
                    let reference = kernel(
                        [scores, paramsArray],
                        template: [("T", scores.dtype)],
                        grid: (threadgroupSize * nRows, 1, 1),
                        threadGroup: (threadgroupSize, 1, 1),
                        outputShapes: [scores.shape],
                        outputDTypes: [.bfloat16]
                    )[0]
                    Gemma4PromptGlue2V1.report(
                        probs, reference: reference,
                        site: "softmax kL=\(axisSize) rows/tg=\(rows)")
                }
                return probs
            }
        }

        CBv2EngageMark.once("prefill-softmax-vec")
        return kernel(
            [scores, paramsArray],
            template: [("T", scores.dtype)],
            grid: (threadgroupSize * nRows, 1, 1),
            threadGroup: (threadgroupSize, 1, 1),
            outputShapes: [scores.shape],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

/// PREFILL-ATTN-TRAFFIC (at1). The composed prompt attention's probability
/// rectangle `P = softmax(S)` is never materialized. Today each query block
/// runs QK^T -> S (bf16), softmax reads S and writes P, and the P.V GEMM reads
/// P: four passes over the score rectangle, ~17.5 GB per ranked prefill. Here
/// the softmax kernel is split in two:
///
///   1. `cbv2_prefill_sdpa_softmax_stats_bf16_at1`: the pg2 softmax
///      transcription, text for text, with the probability store replaced by
///      a store of the row's `maxval` and `normalizer` (`1.0f / sum`) -- the
///      two scalars every lane of the row holds after the same two
///      reduction trees -- as fp32 bit patterns in four bf16 words per row
///      (`[.., L, 4]`, 8 bytes a row against 2 kL bytes of probabilities).
///   2. The P.V product is issued as `addMM(stats[.., 0..<1], S, V,
///      alpha: 1, beta: -0.0)`: the out-source operand is the column view of
///      the carrier, broadcast over the output columns (row stride 4,
///      column stride 0). The steel fused GEMM twins (non-nax and nax, the
///      `.h` sources and their mlx-generated JIT strings) recognize exactly
///      that signature on a bf16 NN addmm and, instead of adding the
///      operand in the epilogue, have their A loader stage
///      `T(fast::exp(float(s) - maxval) * normalizer)` for every score it
///      loads, then store the accumulator as the plain `matmul(P, V)` does.
///
/// EXACTNESS. The prompt softmax kernel writes, per element,
/// `static_cast<T>(fast::exp(float(raw) - maxval) * normalizer)` with
/// `normalizer = 1.0f / sum`; the loader evaluates the same five operations
/// (bf16->fp32 widen, fp32 subtract, `fast::exp`, fp32 multiply, fp32->bf16
/// round) on the same three operands, in the same kernel environment (MLX's
/// utils preamble, fast-math off, the same `bfloat16_t`), so every staged P
/// word is the word the softmax kernel would have stored. `maxval` and
/// `normalizer` are transported bit for bit (the carrier holds their fp32
/// patterns, never a rounding). The GEMM then consumes identical P words
/// through identical tiles, K order and accumulators (the loaded bytes, the
/// threadgroup layout and the MMA are untouched), and its epilogue is the
/// plain store, so the output rectangle is bit-identical to
/// `matmul(softmax(S), V)`. Rows always own their diagonal score, so every
/// row's statistics are finite.
///
/// Prompt width only (`minRows` token rows and up, the composed path's own
/// `L > 8` guard beneath it); tiles are MN-aligned for every steel tile by
/// the `L % 128 == 0 && D % 128 == 0` guard, and the K tail (nax `bk` = 256
/// against kL = 128..1024) is served by the loader's bound-checked twin.
/// Kill switch: `DARKBLOOM_GEMMA4_PREFILL_ATTN_TRAFFIC=0` (incumbent softmax
/// + matmul dispatches, byte for byte). Engage mark: `prefill-attn-traffic`.
/// `DARKBLOOM_GEMMA4_PREFILL_ATTN_TRAFFIC_XCHECK=1` evaluates the incumbent
/// pair beside every fused call and counts differing output words
/// (diagnostic only; forces evaluation).
enum CBv2PrefillAttnTrafficV1 {

    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_ATTN_TRAFFIC"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_ATTN_TRAFFIC_XCHECK"]
        == "1"

    /// softmax.cpp's SOFTMAX_LOOPED_LIMIT: the transcribed block kernel.
    private static let maxKeyLength = 4096

    /// The out-source `beta` the steel GEMM twins recognize: fp32 negative
    /// zero (bits 0x80000000). It makes MLX select the `do_axpby` epilogue
    /// (`beta != 1`), which the twins replace by the loader transform on this
    /// signature; no other addmm passes a negative-zero beta.
    private static let loaderBeta: Float = Float(bitPattern: 0x8000_0000)

    /// The pg2 softmax text with its probability store replaced by the row
    /// statistics store. Everything above the store -- loads, the max and sum
    /// trees, `fast::exp`, `1.0f / normalizer` -- is the pg2 kernel's own.
    /// nil (twin disabled, incumbent pair kept) if the donor text ever
    /// drifts: a soft gate, never a trap inside a lazy static.
    private static let statsSource: String? = {
        let text = CBv2PrefillSoftmaxVecV1.sourcePg2
        let store = [
            "if (row_valid) {",
            "    device T* row_out = probs + size_t(gid) * axis_size;",
            "    T4 result;",
            "    #pragma unroll",
            "    for (int i = 0; i < 4; i++) {",
            "        result[i] = static_cast<T>(ld[i] * normalizer);",
            "    }",
            "    *reinterpret_cast<device T4*>(row_out + base) = result;",
            "}",
        ].joined(separator: "\n")
        let statsStore = [
            "// PREFILL-ATTN-TRAFFIC (at1): the row's statistics instead of its",
            "// probabilities -- the maxval and normalizer every lane of the row",
            "// holds here, as fp32 bit patterns in the row's four bf16 words.",
            "if (lid == 0) {",
            "    *reinterpret_cast<device uint2*>(stats + size_t(gid) * 4) =",
            "        uint2(as_type<uint>(maxval), as_type<uint>(normalizer));",
            "}",
        ].joined(separator: "\n")
        guard text.components(separatedBy: store).count == 2 else {
            FileHandle.standardError.write(
                Data("[prefill-attn-traffic] pg2 softmax text drifted; twin disabled\n".utf8))
            return nil
        }
        return text.replacingOccurrences(of: store, with: statsStore)
    }()

    private static let statsKernel: MLXFast.MLXFastKernel? = statsSource.map { source in
        MLXFast.metalKernel(
            name: "cbv2_prefill_sdpa_softmax_stats_bf16_at1",
            inputNames: ["scores", "params"],
            outputNames: ["stats"],
            source: source,
            ensureRowContiguous: true
        )
    }

    /// `matmul(softmax(scores, axis: -1, precise: true), values)` with the
    /// probabilities never materialized, or nil to keep the incumbent pair.
    /// `scores` is the row-contiguous `[.., L, kL]` score rectangle of one
    /// query block, `values` the `[.., kL, D]` operand the incumbent
    /// `matmul` takes.
    static func attend(scores: MLXArray, values: MLXArray) -> MLXArray? {
        guard enabled, CBv2PrefillSoftmaxVecV1.enabled, let statsKernel else { return nil }
        guard scores.dtype == .bfloat16, values.dtype == .bfloat16 else { return nil }
        guard scores.ndim >= 2, values.ndim == scores.ndim else { return nil }
        let axisSize = scores.dim(scores.ndim - 1)
        guard axisSize > 0, axisSize % 4 == 0, axisSize <= maxKeyLength else { return nil }
        let L = scores.dim(scores.ndim - 2)
        let D = values.dim(values.ndim - 1)
        guard values.dim(values.ndim - 2) == axisSize else { return nil }
        // Every steel tile (bm, bn in 32/64/128) divides these, so the GEMM
        // takes its MN-aligned path, the only one carrying the loader twin.
        guard L % 128 == 0, D % 128 == 0 else { return nil }
        let totalElements = scores.shape.reduce(1, *)
        guard totalElements > 0, totalElements % axisSize == 0 else { return nil }
        let nRows = totalElements / axisSize
        guard nRows >= Gemma4PromptGlue2V1.minRows else { return nil }
        // softmax.cpp:64-68: 32 * ceil(ceil(axis_size / 4) / 32).
        let threadgroupSize = ((axisSize + 3) / 4 + 31) / 32 * 32
        guard threadgroupSize > 0, threadgroupSize <= 1024 else { return nil }
        let numSimdgroups = threadgroupSize / 32
        let rows = CBv2PrefillSoftmaxVecV1.rowsPerThreadgroup(
            axisSize: axisSize, threadgroupSize: threadgroupSize)
        guard rows >= 1, rows * threadgroupSize <= 1024 else { return nil }
        let paramsArray = MLXArray([UInt32(axisSize), UInt32(numSimdgroups)])
        var statsShape = scores.shape
        statsShape[statsShape.count - 1] = 4

        CBv2EngageMark.once("prefill-attn-traffic")
        let stats = statsKernel(
            [scores, paramsArray],
            template: [("T", scores.dtype), ("RPT", rows)],
            grid: (threadgroupSize * nRows, 1, 1),
            threadGroup: (threadgroupSize * rows, 1, 1),
            outputShapes: [statsShape],
            outputDTypes: [.bfloat16]
        )[0]
        // The column view of the carrier: row stride 4, column stride 0 once
        // addmm broadcasts it over the D output columns -- the signature.
        let carrier = stats[.ellipsis, 0 ..< 1]
        let output = addMM(carrier, scores, values, alpha: 1.0, beta: loaderBeta)

        if xcheck {
            let probabilities =
                CBv2PrefillSoftmaxVecV1.apply(scores)
                ?? MLX.softmax(scores, axis: -1, precise: true)
            let reference = matmul(probabilities, values)
            report(
                output, reference: reference,
                site: "P.V kL=\(axisSize) D=\(D) rows=\(nRows)")
        }
        return output
    }

    // MARK: - diagnostics (never on a timed run)

    /// Counts words that differ between the fused output and the incumbent
    /// pair's, evaluating both.
    private static func report(_ candidate: MLXArray, reference: MLXArray, site: String) {
        guard candidate.shape == reference.shape, candidate.dtype == reference.dtype else {
            FileHandle.standardError.write(
                Data(
                    ("[xcheck] prefill-attn-traffic \(site): shape/dtype mismatch "
                        + "\(candidate.shape) \(candidate.dtype) vs "
                        + "\(reference.shape) \(reference.dtype)\n").utf8))
            return
        }
        let differs = candidate.view(dtype: .uint16) .!= reference.view(dtype: .uint16)
        let differing = MLX.sum(differs, stream: .default)
        eval(candidate, reference, differing)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prefill-attn-traffic \(site) shape \(candidate.shape) "
                    + "words \(candidate.size) differing \(differing.item(Int32.self))\n").utf8))
    }
}
