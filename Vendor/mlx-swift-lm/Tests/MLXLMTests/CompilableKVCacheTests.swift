// Tests for the compiled-decode foundation: CompilableKVCache (overflow-bin,
// compile-traceable KV cache) and DynamicSlice. (The legacy CompiledDecode
// forward-compilation tests died with the v1 engine; CBv2CompiledDecodeTests
// covers the v2 compiled path.)
//
// These pin two properties the future compiled-decode path depends on:
//   1. Numerical equivalence — the overflow-bin full buffer + array mask is
//      equivalent to KVCacheSimple's dynamically-sized [..offset] slice.
//   2. Compile-traceability — a decode step (model forward + cache write) can be
//      captured by `compile(inputs:outputs:)` and re-invoked, with the MLXArray
//      `offsetArray` advancing across compiled calls (proving DynamicSlice +
//      `_updateInternal` thread through the tracer).

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("CompilableKVCacheTests")
struct CompilableKVCacheTests {

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private func maskArray(
        _ mode: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        if case .array(let m) = mode { return m }
        return nil
    }

    // MARK: - 1. Overflow-bin equivalence

    /// The valid region of CompilableKVCache's full buffer matches the slice
    /// KVCacheSimple returns, step-for-step, for a prefill + several decodes.
    @Test func overflowBinValidRegionMatchesSimpleCache() {
        let (B, H, D, maxLen) = (1, 2, 4, 16)
        let ck = CompilableKVCache(maxLength: maxLen)
        let simple = KVCacheSimple()

        var offset = 0
        // 3-token prefill, then three single-token decodes.
        for (idx, s) in [3, 1, 1, 1].enumerated() {
            // Distinct value per step so concatenation order is verifiable.
            let k = MLXArray.ones([B, H, s, D]) * Float(idx + 1)
            let v = MLXArray.ones([B, H, s, D]) * Float(-(idx + 1))

            let (ckK, ckV) = ck.update(keys: k, values: v)
            let (skK, skV) = simple.update(keys: k, values: v)
            offset += s

            // Full buffer keeps a constant shape across steps.
            #expect(ckK.shape == [B, H, maxLen, D])
            #expect(ckV.shape == [B, H, maxLen, D])

            let ckValidK = ckK[.ellipsis, ..<offset, 0...]
            let ckValidV = ckV[.ellipsis, ..<offset, 0...]
            #expect(maxAbsDiff(ckValidK, skK) < 1e-6, "keys mismatch at step \(idx)")
            #expect(maxAbsDiff(ckValidV, skV) < 1e-6, "values mismatch at step \(idx)")
            #expect(ck.offset == simple.offset)
        }
    }

    // MARK: - 2. Mask semantics

    /// makeMask marks exactly the valid (causal) positions. For the last query
    /// row, the number of attended keys equals the post-update offset (full
    /// attention) or `min(offset, window)` (sliding window).
    @Test func makeMaskCausalAndWindowedCounts() {
        let (B, H, D, maxLen) = (1, 1, 2, 32)

        // Full attention: drive to offset O, then a decode-step mask (n=1) at
        // pre-update offset O should attend to O+1 positions [0...O].
        let ck = CompilableKVCache(maxLength: maxLen)
        for o in 0 ..< 5 {
            let mode = ck.makeMask(n: 1, windowSize: nil, returnArray: true)
            let mask = try! #require(maskArray(mode))
            #expect(mask.shape == [1, maxLen])
            // pre-update offset == o, so trues = o + 1
            #expect(Int(mask.asType(.int32).sum().item(Int32.self)) == o + 1)
            // advance one token
            _ = ck.update(
                keys: MLXArray.ones([B, H, 1, D]), values: MLXArray.ones([B, H, 1, D]))
        }

        // Sliding window: window w caps attended keys at min(offset+1, w).
        let w = 3
        let cw = CompilableKVCache(maxLength: maxLen)
        for o in 0 ..< 6 {
            let mode = cw.makeMask(n: 1, windowSize: w, returnArray: true)
            let mask = try! #require(maskArray(mode))
            let expected = Swift.min(o + 1, w)
            #expect(
                Int(mask.asType(.int32).sum().item(Int32.self)) == expected,
                "windowed trues at offset \(o)")
            _ = cw.update(
                keys: MLXArray.ones([B, H, 1, D]), values: MLXArray.ones([B, H, 1, D]))
        }
    }

    // MARK: - 3. from: conversion

    /// CompilableKVCache(from: KVCacheSimple) preserves the prefilled KV data and
    /// offset, copying it into the fixed-size buffer at position 0.
    @Test func convertFromSimpleCachePreservesState() {
        let (B, H, D) = (1, 2, 4)
        let simple = KVCacheSimple()
        let k = MLXArray.ones([B, H, 5, D]) * 7
        let v = MLXArray.ones([B, H, 5, D]) * 9
        _ = simple.update(keys: k, values: v)

        let ck = CompilableKVCache(from: simple, maxLength: 16)
        #expect(ck.offset == 5)
        let validK = ck.keys![.ellipsis, ..<5, 0...]
        #expect(maxAbsDiff(validK, k) < 1e-6)
    }

}
