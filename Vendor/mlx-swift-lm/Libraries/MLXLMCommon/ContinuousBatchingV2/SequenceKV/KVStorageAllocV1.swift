// KVStorageAllocV1.swift
//
// Uninitialized allocation for CBv2 sequence-KV storage buffers.
//
// The contiguous KV backends size their storage up front (the windowed ring
// at exactly `window` slots, the full buffer at `promptLength + slack`) and
// then only ever READ positions that have been written: every accessor is
// keyed to `[oldestValidPosition, absoluteOffset)` (WindowedSequenceKV ring
// slices; FullSequenceKV `..<absoluteOffset` views), the batch-wide ring
// attention kernels engage only on a FULL ring and walk exactly N retained
// slots, and the not-full case attends bounded temporal-order views instead
// of raw storage. The `MLXArray.zeros` fill that backs each allocation is
// therefore dead work: no unwritten slot's value is ever observed. At the
// cohort geometry that fill is not small — the 25 sliding layers alone
// zero-fill 2 × [1, 8, window, 256] bf16 per row before the first prompt
// chunk can commit.
//
// `storage(shape:dtype:)` returns a buffer of the requested shape whose
// bytes are simply not initialized: a custom Metal kernel with an EMPTY body
// (outputs of a custom kernel are plain `allocator::malloc` storage unless an
// init value is requested; the kernel writes nothing, so the allocation is
// the entire product). Shape, dtype, contiguity, and donation eligibility of
// the returned array match the zeros array it replaces, so every downstream
// slice assignment and kernel decision is unchanged — only the fill traffic
// disappears. Values are unchanged wherever values are defined: written
// slots hold exactly the bytes the incumbent path writes, and unwritten
// slots remain unobservable by the read-bounds contract above.
//
// Fail-closed: the kill switch (or any dtype outside the KV bf16/fp16
// family) returns the literal `MLXArray.zeros` allocation, byte-identical to
// the established behavior.

import Foundation
import MLX
import MLXFast

enum CBv2KVStorageAllocV1 {

    /// Kill switch. `0`/`false`/`no`/`off` restores the established
    /// `MLXArray.zeros` allocation at every call site.
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_KV_ALLOC_NOFILL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Intentionally empty body: a custom kernel's outputs are allocated
    /// without an initializing fill, so dispatching this kernel produces an
    /// uninitialized buffer of the requested output shape at the cost of one
    /// single-thread no-op launch (the zeros path it replaces launches a full
    /// fill over the same bytes).
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kv_storage_alloc_nofill_v1",
        inputNames: ["seed"],
        outputNames: ["storage"],
        source: """
            // No output writes on purpose: the output allocation is the
            // product. The seed read keeps the body a plain non-empty
            // function; its value is unused.
            (void)seed[0];
        """,
        ensureRowContiguous: true
    )

    /// Tiny anchor input; custom kernels take at least one input array.
    /// `nonisolated(unsafe)` matches the tree's established pattern for
    /// static `MLXArray` holders (ComposedPrefillSDPAV1, PagedAttentionKernel).
    nonisolated(unsafe) private static let seed = MLXArray.zeros([1], dtype: .int32)

    /// An allocation of `shape`/`dtype` with UNINITIALIZED contents, for
    /// storage whose readers are bounded to written positions. Falls back to
    /// `MLXArray.zeros` when disarmed or when the dtype is not a KV dtype.
    static func storage(shape: [Int], dtype: DType) -> MLXArray {
        guard enabled, dtype == .bfloat16 || dtype == .float16 else {
            return MLXArray.zeros(shape, dtype: dtype)
        }
        CBv2EngageMark.once("kv-alloc-nofill")
        return kernel(
            [seed],
            grid: (1, 1, 1),
            threadGroup: (1, 1, 1),
            outputShapes: [shape],
            outputDTypes: [dtype]
        )[0]
    }
}
