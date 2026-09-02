// CompilableKVCache: Fixed-size KV cache using the Overflow Bin pattern.
//
// Ported from osaurus-ai/vmlx-swift-lm (Libraries/MLXLMCommon/CompilableKVCache.swift).
// This is the foundation type for whole-step compiled decode: it is the only
// full-attention cache whose `update()` is graph-traceable, so a decode step
// (model forward + cache write) can be captured by `compile(inputs:outputs:)`.
//
// Standard KVCacheSimple returns keys[..<offset] — a dynamically sized slice that
// changes every decode step. This prevents compile() from tracing through the cache
// because DynamicSlice requires static slice_size.
//
// CompilableKVCache solves this by:
// 1. Pre-allocating a fixed-size buffer [B, H, maxLength, D]
// 2. Writing new keys/values via dynamicSliceUpdate (compile-traceable writes)
// 3. Returning the FULL buffer from update() — constant shape every step
// 4. Generating a boolean attention mask in makeMask() that marks active positions
//
// The attention kernel handles the masking, computing only on valid positions.
// This trades marginal redundant compute (masked zeros) for enabling compile()
// to fuse hundreds of FFI crossings into a single compiled call.
//
// Usage:
//   // After prefill with standard cache, convert for compiled decode:
//   let compilableCache = standardCache.map { c in
//       CompilableKVCache(from: c, maxLength: 2048)
//   }
//
// NOTE (Darkbloom): this type is presently UNWIRED — it ships as the verified
// foundation for a future compiled-decode path. See `CompiledDecode.swift` for
// the `compileDecodeForward` wrapper and the port plan for batched integration
// into `GenerationBatch`.

import Foundation
import MLX
import MLXNN

public class CompilableKVCache: BaseKVCache {

    public var keys: MLXArray?
    public var values: MLXArray?

    public var offsetArray: MLXArray

    public let maxLength: Int

    public var step: Int

    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxLength))

    public init(maxLength: Int = 4096, step: Int = 256) {
        self.maxLength = maxLength
        self.step = step
        self.offsetArray = MLXArray([Int32(0)])
        super.init()
    }

    public static func promote(from cache: KVCacheSimple, maxLength: Int) -> CompilableKVCache {
        return CompilableKVCache(from: cache, maxLength: maxLength)
    }

    public convenience init(from cache: KVCache, maxLength: Int = 4096) {
        self.init(maxLength: maxLength)

        let existingState = cache.state
        if existingState.count >= 2 {
            let existingKeys = existingState[0]  // [B, H, seqLen, D]
            let existingValues = existingState[1]

            let seqLen = existingKeys.dim(2)
            let B = existingKeys.dim(0)
            let H = existingKeys.dim(1)
            let kD = existingKeys.dim(3)
            let vD = existingValues.dim(3)

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: existingKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: existingValues.dtype)

            self.keys![.ellipsis, ..<seqLen, 0...] = existingKeys
            self.values![.ellipsis, ..<seqLen, 0...] = existingValues

            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }

    // MARK: - KVCache protocol

    public override var offset: Int {
        get {
            offsetArray[0].item(Int.self)
        }
        set {
            offsetArray = MLXArray([Int32(newValue)])
        }
    }

    public override func innerState() -> [MLXArray] {
        if let keys, let values {
            return [keys, values, offsetArray]
        }
        return [offsetArray]
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let nTokens = newKeys.dim(2)

        if self.keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValues.dtype)
        }

        let prev = offsetArray
        let newOffset = prev + MLXArray([Int32(nTokens)])

        self.keys!._updateInternal(
            dynamicSliceUpdate(self.keys!, update: newKeys, start: prev, axes: [2]))
        self.values!._updateInternal(
            dynamicSliceUpdate(self.values!, update: newValues, start: prev, axes: [2]))

        self.offsetArray._updateInternal(newOffset)

        return (self.keys!, self.values!)
    }

    // MARK: - Mask (Overflow Bin)

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let currentOffsetArr = offsetArray  // MLXArray [1] int32

        let linds: MLXArray
        if n == 1 {
            linds = currentOffsetArr.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + currentOffsetArr).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxLength)

        var mask = linds .>= rinds

        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        return .array(mask)
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let off: Int = offsetArray[0].item(Int.self)
            if off == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<off, 0...],
                    values[.ellipsis, ..<off, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else { return }
            let seqLen = newValue[0].dim(2)
            let B = newValue[0].dim(0)
            let H = newValue[0].dim(1)
            let kD = newValue[0].dim(3)
            let vD = newValue[1].dim(3)

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newValue[0].dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValue[1].dtype)
            self.keys![.ellipsis, ..<seqLen, 0...] = newValue[0]
            self.values![.ellipsis, ..<seqLen, 0...] = newValue[1]
            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let current: Int = offsetArray[0].item(Int.self)
        let trimmed = min(current, n)
        offsetArray = MLXArray([Int32(current - trimmed)])
        super.offset = current - trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let c = CompilableKVCache(maxLength: maxLength, step: step)
        c.keys = keys
        c.values = values
        c.offsetArray = offsetArray
        return c
    }

    // MARK: - Debug

    public var debugDescription: String {
        "CompilableKVCache(offset=\(offset), maxLength=\(maxLength), "
            + "shape=\(keys?.shape.description ?? "nil"))"
    }
}
