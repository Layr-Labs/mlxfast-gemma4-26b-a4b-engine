// CompilableRotatingKVCache: compile-traceable rotating KV cache.
//
// Ported from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/CompilableRotatingKVCache.swift).
//
// Standard RotatingKVCache breaks compile() in two ways:
//   1. Buffer growth via concat rebinds the keys property — the trace
//      captures the original object and rebinding loses it.
//   2. Ring-buffer wrap via Swift Int reset doesn't match the trace's
//      assumed linear layout.
//
// This subclass fixes both by:
//   - Pre-allocating the unified buffer to maxCacheSize at promotion time.
//     No concat-growth during decode.
//   - Tracking the write index as idxArray (MLXArray [1] int32). Wrap
//     arithmetic uses MLXArray ops (where_ + modulo) so the compile tracer
//     can follow.
//   - Returning the FULL [B, H, maxCacheSize, D] buffer from update().
//     The attention mask restricts valid positions.
//
// Initialise via init(from:) on an existing RotatingKVCache populated by
// prefill, or via the static promote(from:maxLength:) helper.

import Foundation
import MLX
import MLXNN

public final class CompilableRotatingKVCache: RotatingKVCache, @unchecked Sendable {

    public var idxArray: MLXArray

    public var offsetArray: MLXArray

    private lazy var maskRinds: MLXArray = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    // MARK: - Init

    public override init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.idxArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        super.init(maxSize: maxSize, keep: keep, step: step)
    }

    public convenience init(from rotating: RotatingKVCache) {
        self.init(
            maxSize: rotating.maxCacheSize,
            keep: rotating.keep,
            step: rotating.step
        )

        self.idx = rotating.idx
        self.offset = rotating.offset

        if let srcK = rotating.keys, let srcV = rotating.values {
            let B = srcK.dim(0)
            let H = srcK.dim(1)
            let kD = srcK.dim(3)
            let vD = srcV.dim(3)
            let curLen = srcK.dim(2)

            if curLen < maxCacheSize {
                let padLen = maxCacheSize - curLen
                let padK = MLXArray.zeros([B, H, padLen, kD], dtype: srcK.dtype)
                let padV = MLXArray.zeros([B, H, padLen, vD], dtype: srcV.dtype)
                self.keys = concatenated([srcK, padK], axis: 2)
                self.values = concatenated([srcV, padV], axis: 2)
            } else {
                self.keys = srcK
                self.values = srcV
            }
        }

        self.idxArray = MLXArray([Int32(self.idx)])
        self.offsetArray = MLXArray([Int32(self.offset)])
    }

    public static func promote(from cache: RotatingKVCache, maxLength: Int) -> CompilableRotatingKVCache {
        return CompilableRotatingKVCache(from: cache)
    }

    // MARK: - Overridden update

    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        if keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            keys = MLXArray.zeros([B, H, maxCacheSize, kD], dtype: newKeys.dtype)
            values = MLXArray.zeros([B, H, maxCacheSize, vD], dtype: newValues.dtype)
        }

        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: idxArray, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: idxArray, axes: [2]))

        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = idxArray + advance
        let maxSz = MLXArray([Int32(maxCacheSize)])
        let keepArr = MLXArray([Int32(keep)])
        let cycleLen = maxSz - keepArr  // number of rotating slots

        let rotatedIdx: MLXArray
        if keep > 0 {
            rotatedIdx = keepArr + ((advancedIdx - keepArr) % cycleLen)
        } else {
            rotatedIdx = advancedIdx % maxSz
        }
        let newIdx = MLX.`where`(advancedIdx .< maxSz, advancedIdx, rotatedIdx)

        idxArray._updateInternal(newIdx)
        offsetArray._updateInternal(offsetArray + advance)

        return (keys!, values!)
    }

    // MARK: - makeMask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, maxCacheSize)
        let causal = linds .>= rinds

        let maxSzArr = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrueMask = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSzArr, allTrueMask, causal)

        if let windowSize {
            let tokenInds = (rinds - idxArray + MLXArray(Int32(maxCacheSize))) % Int32(maxCacheSize)
            let windowFilter = tokenInds .>= Int32(maxCacheSize - windowSize)
            mask = mask & windowFilter
        }

        return .array(mask)
    }

    // MARK: - innerState

    public override func innerState() -> [MLXArray] {
        var state = [MLXArray]()
        if let k = keys { state.append(k) }
        if let v = values { state.append(v) }
        state.append(idxArray)
        state.append(offsetArray)
        return state
    }
}
