// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Compiled-in RMSNorm→RoPE fuse + SWKV mask-trim knobs.
/// `fusedRmsRopeTile` matches `rms_single_row` SIMD_SIZE; `fusedRmsRopeUnroll`
/// matches `RMS_N_READS` / `rope_impl` N=4. Not env-gated.
public enum Gemma4FusedRmsRope {
    public static let fusedRmsRopeEnabled = true
    public static let fusedRmsRopeTile = 32
    public static let fusedRmsRopeUnroll = 4
    public static let attentionTrimMask = true
    public static let swkvTrimRadiusP = 1024
}

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - graphOffsetArray Helper

/// Returns a graph-visible cache offset when the cache exposes one.
///
/// `KVCache.offset` is an `Int` API for compatibility with upstream model
/// ports. Compile-safe caches keep the offset as an `MLXArray` so the value
/// can flow through `compile()` without an `.item()` readback. Model-side
/// helpers that need positions outside `applyRotaryPosition` should call this
/// before falling back to `cache.offset`.
public func graphOffsetArray(for cache: KVCache?) -> MLXArray? {
    // Snapshot with `+ 0` so cache.update() advancing offsetArray
    // doesn't shift the caller's RoPE position. Without this, the query
    // gets a position one step ahead of the keys in compiled decode.
    if let compilableRot = cache as? CompilableRotatingKVCache {
        return compilableRot.offsetArray + 0
    }
    if let compilable = cache as? CompilableKVCache {
        return compilable.offsetArray + 0
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return batchCache.batchOffset + 0
    }
    return nil
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// When the cache exposes an `offsetArray` (e.g. `CompilableKVCache`), the
/// offset is passed as an `MLXArray` so the compile tracer can track it
/// through the graph without triggering a synchronous GPU readback. For all
/// other cache types, the standard `Int`-based offset path is used.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines scalar or per-sequence offset), or `nil`
///     for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let offsetArray = graphOffsetArray(for: cache) {
        return applyRotaryPosition(rope, to: x, offsetArray: offsetArray)
    }
    return applyRotaryPosition(rope, to: x, offset: cache?.offset ?? 0)
}

/// Scalar-offset twin used by `gemma4ApplyRotaryPosition` (.scalar).
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, offset: Int) -> MLXArray {
    rope(x, offset: offset)
}

/// Array-offset twin used by `gemma4ApplyRotaryPosition` (.batch / .graphArray)
/// so CompilableRotatingKVCache.offsetArray stays graph-visible.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, offsetArray: MLXArray)
    -> MLXArray
{
    rope(x, offset: offsetArray)
}
