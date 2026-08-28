// PagedAttentionKernel.swift
//
// Swift dispatch wrapper for the CBv2 paged-attention decode kernels
// (`pagedattention.metal`, shipped as a bundle resource). Decode is a
// two-pass flash-decoding dispatch: pass A computes unnormalized partials
// over PTOK-token partitions (one threadgroup per (sequence, kv head,
// partition)); pass B merges the partials per (sequence, query head) with
// sinks folded into the denominator. See the .metal header for the full
// contract and attribution.
//
// Kernel instances are JIT-compiled by MLX and cached both by MLX (per
// kernel name) and here (per configuration). Every distinct configuration
// gets a distinct kernel NAME: MLX's custom-kernel library cache is keyed
// by name and re-JITs whenever the generated source for a name changes, so
// sharing one name across dtypes/head-dims would thrash the pipeline cache.
//
// PERFORMANCE NOTE (2026-07, kernel-opt track): the original real-model
// ~100x slowdown (GPT-OSS-20B at ~2 s/token, 30 s step-watchdog kills —
// benchmarks/reports/gptoss-20b-mxfp4q8-main.md) was NOT kernel math: MLX
// retains kernel-INPUT data handles until the command buffer completes, so
// per-layer slab slice-updates failed donation and copied the full multi-
// GiB slab (~370 GiB of copies per token). Fixed by writing the slabs IN
// PLACE (fused decode write in pass A + `bulkWrite` for chunks) with a
// per-group write-fence chain for ordering; measured 25.5 s -> 13.6 ms per
// 24-layer step at the 16 GiB pool (docs/engine-v2/kernel-research.md §3).

import Foundation
import MLX
import MLXFast

/// Configuration key for one compiled paged-attention kernel variant.
struct PagedAttentionKernelKey: Hashable {
    enum Pass: Hashable {
        case part
        case merge
        /// Bulk in-place KV write (prefill chunks / prefix adoption).
        case write
    }
    var pass: Pass
    var dtype: DType
    var headDim: Int
    var pageSize: Int
    var gqa: Int
    var simdgroups: Int
    var hasSinks: Bool
    var hasSoftcap: Bool
    /// Tokens per flash-decoding partition (the shader's PTOK). Adaptive —
    /// see `PagedAttentionKernel.partitionTokens(for:)` — so it MUST be part
    /// of the variant identity: MLX's custom-kernel library cache is keyed
    /// by kernel NAME and re-JITs whenever a name's generated source
    /// changes, so two PTOKs sharing a name would thrash the pipeline cache
    /// on every batch-shape change.
    var partitionTokens: Int
    /// Pass A only: fuse the decode KV write into the kernel.
    var hasWrite: Bool = false

    var kernelName: String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        let p: String
        switch pass {
        case .part: p = "part"
        case .merge: p = "merge"
        case .write: p = "write"
        }
        return "cbv2_paged_\(p)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)"
            + "_n\(simdgroups)_t\(partitionTokens)"
            + "_sink\(hasSinks ? 1 : 0)_cap\(hasSoftcap ? 1 : 0)"
            + (hasWrite ? "_w1" : "")
    }
}

enum PagedAttentionMSL {
    /// Bodies of the auto-generated kernel functions. They reference the
    /// thread attributes and `_shape` helpers by name so MLX includes them
    /// in the generated signature (MLX scans the body source for tokens).
    ///
    /// Two pass-A variants: the decode path fuses the per-row KV write
    /// (HAS_WRITE=true, `knew`/`vnew` inputs); the KV-borrowing path
    /// dispatches the write-free variant, where `q` stands in for the
    /// never-read `knew`/`vnew` parameters (dead code under
    /// HAS_WRITE=false).
    static let partBody: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            // grid z extent == the partial buffers' partition capacity
            // (output `_shape` params are not injected by MLX, only inputs').
            const int maxpart = threadgroups_per_grid.z;
            // Advance the slab group's write-fence chain: this dispatch
            // wrote the step's K/V tiles in place, and later slab readers
            // consume the fence to order after it.
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                && thread_position_in_grid.z == 0) {
                fence[0] = wfence[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_impl<T, D, S, GQA, HPT, NSG, PTOK, true, HAS_SOFTCAP>(
                q, knew, vnew, kcache, vcache, tables, seqinfo, params,
                kvh, maxp, maxpart, q_smem, red_smem, partials, meta,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """

    static let partBodyNoWrite: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            // grid z extent == the partial buffers' partition capacity
            // (output `_shape` params are not injected by MLX, only inputs').
            const int maxpart = threadgroups_per_grid.z;
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_impl<T, D, S, GQA, HPT, NSG, PTOK, false, HAS_SOFTCAP>(
                q, q, q, kcache, vcache, tables, seqinfo, params,
                kvh, maxp, maxpart, q_smem, red_smem, partials, meta,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """

    /// Bulk in-place KV write: scatters `[KVH, N, D]` tiles into the slabs
    /// and emits the next fence in the group's write chain.
    static let writeBody: String = """
            const int kvh = ktile_shape[0];
            const int n = ktile_shape[1];
            const int d = ktile_shape[2];
            cbv2::paged_kv_write_impl<T, S>(
                ktile, vtile, slots, prev, kcache, vcache,
                kvh, n, d, fence,
                thread_position_in_grid);
        """

    static let mergeBody: String = """
            const int heads = partials_shape[1];
            const int maxpart = partials_shape[2];
            cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS>(
                partials, meta, seqinfo, sinks, heads, maxpart, out,
                threadgroup_position_in_grid,
                thread_index_in_simdgroup);
        """
}

/// Batched single-token (decode) paged attention.
public enum PagedAttentionKernel {

    /// Head dims with a validated shared-memory/register budget. The
    /// threadgroup-memory budget additionally depends on the GQA factor,
    /// but the head split (`headsPerThreadgroup`) keeps every supported
    /// head dim within it at any fleet GQA — Gemma-4 global layers
    /// (headDim 512, GQA 8) run split 2-heads-per-threadgroup.
    public static let supportedHeadDims: Set<Int> = [64, 128, 256, 512]

    /// Tokens per flash-decoding partition — the MAXIMUM, and the value the
    /// sizer returns whenever the GPU is already saturated. Must be a
    /// multiple of the page size; the pool refuses a page size that does
    /// not divide it (`PagedKVPool.init`).
    ///
    /// 256 keeps the partial buffers small, but it is the wrong choice at
    /// short context and small batch: see `partitionTokensForDispatch`.
    public static let partitionTokens = 256

    /// The ONLY partition lengths the sizer may return, ascending.
    ///
    /// The sizer is deliberately quantised to a three-rung ladder rather
    /// than returning any page multiple in `[64, 256]`. PTOK is a kernel
    /// TEMPLATE parameter, so every distinct value is a distinct compiled
    /// variant with its own name and its own JIT
    /// (`PagedAttentionKernelKey`). A continuous sizer would mint a new
    /// variant per context bucket per layer shape and defeat
    /// `PagedAttentionKernelSmoke.runtimeSmoke`, whose whole job is that
    /// traffic never pays compilation. Three rungs bound the decode variant
    /// set at 3x per shape, and the occupancy win is coarse — 4x more
    /// threadgroups — so nothing is lost by not resolving 80 from 96.
    ///
    /// The bottom rung is the floor: below it the partial buffers and the
    /// merge pass grow faster than the extra occupancy pays back, and a
    /// threadgroup has too few tokens to amortise staging its query into
    /// threadgroup memory. The top rung is `partitionTokens`.
    ///
    /// Every rung must divide evenly into pages for the page sizes the pool
    /// admits; rungs that do not are filtered per dispatch.
    public static let partitionTokenLadder = [64, 128, 256]

    /// Floor for the adaptive sizer — the ladder's bottom rung. DERIVED, so
    /// that changing the ladder cannot leave a stale floor behind.
    public static var minPartitionTokens: Int { partitionTokenLadder[0] }

    /// Pass-A threadgroups the sizer aims to launch.
    ///
    /// WS-6.4. Pass A launches `kvHeads * (gqa / hpt) * batch *
    /// ceil(len / PTOK)` threadgroups. At GPT-OSS decode shapes (8 kv
    /// heads, GQA 8, hpt 8 so no split) with B == 1 and a 512-token
    /// context that is `8 * 1 * 2 == 16` threadgroups — on a 40-core GPU,
    /// under half the machine, and it is why paged trails contiguous at
    /// B == 1 (88.5 vs 101.8 tok/s) while matching it at B >= 4. Sizing
    /// PTOK down multiplies the partition count until the machine is full.
    ///
    /// MEASURED, M4 Max (40-core GPU), GPT-OSS decode shape, us/dispatch,
    /// `partitionSizingBenchmark` with `TARGET=0` as the baseline column:
    ///
    ///     B=1 ctx  256   369 / 385  ->  303 / 310 / 348   (PTOK 64)
    ///     B=1 ctx  512   378 / 394  ->  306 / 331 / 336   (PTOK 64)
    ///     B=1 ctx 1024   381 / 384  ->  292 / 349 / 393   (PTOK 64)
    ///     B=2 ctx  512   378 / 375  ->  318              (PTOK 64)
    ///
    /// 128 is chosen to be the LEAST aggressive value that captures this.
    /// It moves PTOK only for B == 1 and B == 2 at short context — every
    /// other operating point, including B == 1 at ctx 4096 and all of
    /// B >= 4, still dispatches at PTOK 256 and is bit-identical to the
    /// pre-WS-6.4 behaviour.
    ///
    /// Larger targets looked better still in one sweep (`TARGET=512` took
    /// B=1 ctx 4096 from 511 to 304 us by dropping to PTOK 64), but that
    /// did NOT reproduce: repeat runs of the identical configuration
    /// measured 511 and 1386 us. The box was running ~30 concurrent build
    /// and test processes, so anything outside the short-context regime is
    /// below the noise floor and is deliberately NOT claimed. Retune on a
    /// quiet machine via the env knob before widening the default.
    ///
    /// `DARKBLOOM_CBV2_PAGED_PTOK_TARGET=0` disables adaptation and pins
    /// PTOK to `partitionTokens` — the pre-WS-6.4 behaviour, and the kill
    /// switch. Values above `partitionTargetLimit` are CLAMPED, not
    /// rejected.
    public static let partitionTargetThreadgroups: Int = partitionTarget(
        environment: ProcessInfo.processInfo.environment)

    /// Environment variable backing `partitionTargetThreadgroups`.
    static let partitionTargetEnvironmentKey = "DARKBLOOM_CBV2_PAGED_PTOK_TARGET"

    /// Target used when the knob is unset or unparseable.
    ///
    /// DEFAULT 0 (adaptation OFF) as of v0.8.0. The sizer derives PTOK from
    /// `batch` and the batch-wide `maxAttendLength`, so a row's partition
    /// count — and therefore its online-softmax summation order — moved with
    /// its BATCHMATES. That contradicts design goal 1 in
    /// `pagedattention.metal`, which promises a row is "bit-identical
    /// regardless of batchmates" and predicates that proof on a FIXED PTOK.
    /// Measured: a 1024-token row on the GPT-OSS shape takes PTOK 64 alone,
    /// 128 at B=2, 256 at B=4 — and one 2048-token batchmate moves it 64->256
    /// on its own. Summation-order changes flip tokens at argmax near-ties
    /// (the same effect the query-block knob demonstrated at the v0.8.0
    /// gate), so this was observable nondeterminism under concurrent load.
    ///
    /// Set the env knob to restore adaptation; recovering the WS-6.4 B=1
    /// occupancy win invariantly needs per-row sizing (`batch: 1`, each row's
    /// OWN attended length) plus dispatch bucketed by rung, because PTOK is a
    /// kernel template parameter and one dispatch carries one value.
    static let partitionTargetDefault = 0

    /// Ceiling on the operator-settable threadgroup target.
    ///
    /// The target is a THREADGROUP COUNT — "how much of the machine one
    /// decode dispatch should fill". The largest Apple GPU in the fleet has
    /// 80 cores, so 4,096 is ~50x saturation, and past it the knob stops
    /// expressing anything new: with a target this high the sizer already
    /// returns the ladder floor for every context a served model can hold
    /// (the next rung up needs `128 * target / perPartition` attended
    /// tokens — over 65,000 even at the least favourable shape). Clamping
    /// therefore costs no reachable behaviour and buys two things:
    ///
    ///  - `partitionTokensForDispatch` can no longer be handed a target
    ///    whose ceiling division overflows. This is an operator-facing kill
    ///    switch; `DARKBLOOM_CBV2_PAGED_PTOK_TARGET=9223372036854775807`
    ///    used to parse as a valid non-negative tuning value and then trap
    ///    the daemon on its first paged decode.
    ///  - `PagedAttentionKernelSmoke.smokeAttendLengths` can sweep the WHOLE
    ///    reachable partition range instead of a prefix of it, so no rung
    ///    the sizer can select is left to JIT on live traffic.
    public static let partitionTargetLimit = 4096

    /// `partitionTargetThreadgroups` as a function of an environment, so the
    /// knob's own parse — including a hostile value — is testable without a
    /// second process.
    static func partitionTarget(environment: [String: String]) -> Int {
        guard let raw = environment[partitionTargetEnvironmentKey] else {
            return partitionTargetDefault
        }
        if let value = Int(raw), value >= 0 {
            return min(value, partitionTargetLimit)
        }
        // All-digit values too large for `Int` are an over-range tuning
        // setting, not a typo: clamp them exactly as `Int.max` clamps,
        // rather than silently reverting to the default.
        let digits = raw.hasPrefix("+") ? raw.dropFirst() : raw[...]
        if !digits.isEmpty, digits.allSatisfy({ $0.isASCII && $0.isNumber }) {
            return partitionTargetLimit
        }
        return partitionTargetDefault
    }

    /// Tokens per partition for one decode dispatch: a rung of
    /// `partitionTokenLadder` that divides evenly into `pageSize` pages, so
    /// `PTOK % pageSize == 0` holds by construction — the invariant `decode`
    /// preconditions and `PagedKVPool.init` guards.
    ///
    /// A pure function of the dispatch shape, with no device state: the same
    /// shape always maps to the same rung, so the kernel-variant cache sees
    /// a small stable set of names rather than one per step.
    ///
    /// - Parameters:
    ///   - maxAttendLength: longest attended range over the batch's rows.
    ///   - batch: rows in this dispatch.
    ///   - kvHeads: KV heads of the layer.
    ///   - headSplits: threadgroups per kv head (`gqa / headsPerThreadgroup`).
    public static func partitionTokensForDispatch(
        maxAttendLength: Int, batch: Int, kvHeads: Int, headSplits: Int, pageSize: Int
    ) -> Int {
        partitionTokensForDispatch(
            maxAttendLength: maxAttendLength, batch: batch, kvHeads: kvHeads,
            headSplits: headSplits, pageSize: pageSize,
            target: partitionTargetThreadgroups)
    }

    /// `partitionTokensForDispatch` with the threadgroup target injected —
    /// the seam the smoke and its tests use to reason about a target other
    /// than this process's.
    static func partitionTokensForDispatch(
        maxAttendLength: Int, batch: Int, kvHeads: Int, headSplits: Int, pageSize: Int,
        target: Int
    ) -> Int {
        // Rungs this page size can express. A page larger than a rung cannot
        // be subdivided into it; the pool refuses a page size that does not
        // divide `partitionTokens`, but `decode` is callable directly.
        let rungs = partitionTokenLadder.filter {
            $0 % pageSize == 0 && $0 <= partitionTokens
        }
        guard let smallest = rungs.first, target > 0 else {
            return partitionTokens
        }
        let largest = rungs[rungs.count - 1]

        // Threadgroups already launched per partition: everything except the
        // partition count is fixed by the layer and the batch.
        let perPartition = max(1, kvHeads * headSplits * batch)
        let wantedPartitions = ceilingDivide(target, perPartition)
        guard wantedPartitions > 1 else { return largest }

        // Largest rung that still yields `wantedPartitions` partitions —
        // the smallest partial buffers that fill the machine. When even the
        // smallest rung cannot (the context is simply too short), take it:
        // that is the floor, not a target miss to correct further.
        let idealTokens = ceilingDivide(maxAttendLength, wantedPartitions)
        return rungs.last { $0 <= idealTokens } ?? smallest
    }

    /// `ceil(lhs / rhs)` without the `lhs + rhs - 1` overflow. Both operands
    /// can come from outside the process (`maxAttendLength` from a caller,
    /// the target from an operator's environment), and an overflow trap on
    /// this path takes the daemon down on its first paged decode.
    private static func ceilingDivide(_ lhs: Int, _ rhs: Int) -> Int {
        precondition(rhs > 0, "ceilingDivide by \(rhs)")
        let quotient = lhs / rhs
        // `quotient + 1` cannot overflow: reaching `Int.max` needs rhs == 1,
        // and then the remainder is 0 and this returns early.
        return lhs % rhs == 0 ? quotient : quotient + 1
    }

    // MARK: - Threadgroup-memory budget (single source of truth)
    //
    // These constants mirror the part kernel's threadgroup allocations —
    // `PagedAttentionMSL.partBody` above plus the RSTRIDE layout in
    // pagedattention.metal (which carries a matching keep-in-sync comment):
    //
    //   threadgroup float q_smem[HPT * D];               // staged queries
    //   threadgroup float red_smem[NSG * HPT * (D + 2)]; // merge records:
    //                                                    // acc[D], m, l
    //
    // HPT is the HEAD SPLIT (`headsPerThreadgroup`), NOT the full GQA
    // factor — sizing these by GQA is exactly the bug that made Gemma-4
    // global layers (d512, GQA 8) a 32,832 B process fatal. Both buffers
    // are float32 regardless of the slab dtype T (K/V rows are converted
    // to float on load), the merge pass and the bulk-write kernel allocate
    // NO threadgroup memory, and neither the HAS_SOFTCAP, HAS_SINKS nor
    // HAS_WRITE variant adds any — so the budget is a function of
    // (headDim, gqa, simdgroups) alone. `CBv2PagedEligibilityTests`
    // asserts the generated bodies still match this model.

    /// Metal's per-threadgroup memory cap (`setThreadgroupMemoryLength`
    /// limit on Apple GPUs). A dispatch over this limit is an UNCATCHABLE
    /// process fatal ("Threadgroup memory size (...) exceeds the maximum
    /// (32768)"), not a thrown error — eligibility must be refused
    /// statically, before any dispatch.
    public static let threadgroupMemoryLimit = 32 * 1024

    /// Per-(simdgroup, query head) merge record trailer: the shader's
    /// RSTRIDE is `D + 2` floats — acc[D] plus (m, l).
    public static let mergeRecordMetaFloats = 2

    /// Candidate simdgroup counts for the part kernel, largest first.
    static let simdgroupCandidates = [8, 4, 2, 1]

    /// Per-thread float32 budget for the pass-A value accumulator
    /// (`acc[HPT][D/32]`): caps the head split so registers never explode
    /// at large head dims (32 floats == the level the fleet's validated
    /// d64/GQA-8 shape already runs at).
    static let maxAccumulatorFloatsPerThread = 32

    /// Query heads per pass-A threadgroup (the kernel's HPT template
    /// parameter): the largest divisor of `gqa` whose per-thread value
    /// accumulator stays within `maxAccumulatorFloatsPerThread`. A kv
    /// head's GQA query heads split across `gqa / hpt` threadgroups; the
    /// split is per-head, so no head's arithmetic changes. d64/d128 fleet
    /// shapes keep hpt == gqa (identical to the unsplit kernel); d512 at
    /// GQA 8 (Gemma-4 global layers) runs hpt == 2.
    public static func headsPerThreadgroup(headDim: Int, gqa: Int) -> Int {
        let cap = max(1, maxAccumulatorFloatsPerThread * 32 / headDim)
        var hpt = min(gqa, cap)
        while gqa % hpt != 0 { hpt -= 1 }
        return hpt
    }

    /// Exact threadgroup bytes the part kernel allocates for one
    /// threadgroup: (q_smem + red_smem) float32 elements, sized by the
    /// HEAD-SPLIT group (`headsPerThreadgroup`), not the full GQA factor.
    public static func partThreadgroupBytes(
        headDim: Int, gqa: Int, simdgroups: Int
    ) -> Int {
        let hpt = headsPerThreadgroup(headDim: headDim, gqa: gqa)
        let qSmem = hpt * headDim
        let redSmem = simdgroups * hpt * (headDim + mergeRecordMetaFloats)
        return (qSmem + redSmem) * MemoryLayout<Float32>.size
    }

    /// Largest simdgroup count whose staging + merge buffers fit the Metal
    /// threadgroup-memory cap, or nil when even NSG=1 exceeds it — the
    /// configuration is statically ineligible for the paged kernels.
    static func simdgroupsPerThreadgroup(headDim: Int, gqa: Int) -> Int? {
        simdgroupCandidates.first {
            partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: $0)
                <= threadgroupMemoryLimit
        }
    }

    /// Static eligibility of one attention shape for the paged decode
    /// kernels: nil when eligible, else a human-readable reason. Callers
    /// building paged state (`PagedKVBackend.init`) MUST refuse ineligible
    /// shapes before any dispatch (see `threadgroupMemoryLimit`).
    public static func ineligibilityReason(headDim: Int, gqa: Int) -> String? {
        guard supportedHeadDims.contains(headDim) else {
            return "paged kernel does not support headDim \(headDim); "
                + "supported: \(supportedHeadDims.sorted())"
        }
        guard gqa >= 1 else {
            return "invalid GQA factor \(gqa)"
        }
        guard simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) != nil else {
            let bytes = partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: 1)
            return "headDim \(headDim) with GQA \(gqa) needs \(bytes) B of "
                + "threadgroup memory even at 1 simdgroup, over the "
                + "\(threadgroupMemoryLimit) B Metal limit — the paged part "
                + "kernel would trap at first dispatch"
        }
        return nil
    }

    private final class KernelCache: @unchecked Sendable {
        private var kernels: [PagedAttentionKernelKey: MLXFast.MLXFastKernel] = [:]
        private let lock = NSLock()

        func kernel(
            for key: PagedAttentionKernelKey,
            source: String
        ) -> MLXFast.MLXFastKernel {
            lock.lock()
            defer { lock.unlock() }
            if let k = kernels[key] { return k }
            let k: MLXFast.MLXFastKernel
            switch key.pass {
            case .part where key.hasWrite:
                // `wfence` is the slab group's write fence: the graph edge
                // (and the encoder barrier it induces) orders this dispatch
                // after every prior in-place write of the group; the
                // `fence` output advances the chain for later readers.
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: [
                        "q", "knew", "vnew", "kcache", "vcache", "tables", "seqinfo",
                        "params", "wfence",
                    ],
                    outputNames: ["partials", "meta", "fence"],
                    source: PagedAttentionMSL.partBody,
                    header: source,
                    ensureRowContiguous: true
                )
            case .part:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: [
                        "q", "kcache", "vcache", "tables", "seqinfo", "params", "wfence",
                    ],
                    outputNames: ["partials", "meta"],
                    source: PagedAttentionMSL.partBodyNoWrite,
                    header: source,
                    ensureRowContiguous: true
                )
            case .merge:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["partials", "meta", "seqinfo", "sinks"],
                    outputNames: ["out"],
                    source: PagedAttentionMSL.mergeBody,
                    header: source,
                    ensureRowContiguous: true
                )
            case .write:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["ktile", "vtile", "slots", "prev", "kcache", "vcache"],
                    outputNames: ["fence"],
                    source: PagedAttentionMSL.writeBody,
                    header: source,
                    ensureRowContiguous: true
                )
            }
            kernels[key] = k
            return k
        }
    }

    private static let cache = KernelCache()

    static func kernel(
        for key: PagedAttentionKernelKey,
        source: String
    ) -> MLXFast.MLXFastKernel {
        cache.kernel(for: key, source: source)
    }

    /// Shared dummy sinks (the generated signature keeps a `device` input
    /// even when HAS_SINKS is false). Read-only after creation; MLXArray is
    /// not Sendable but this one is never mutated (engine-thread discipline).
    nonisolated(unsafe) private static let zeroSinks: MLXArray = {
        let z = MLXArray.zeros([8], dtype: .float32)
        eval(z)
        return z
    }()

    /// One row of the `[B, 8]` int32 `seqinfo` argument `decode` takes.
    ///
    /// The layout — `{attendStart, attendLen, tableLen, writePage,
    /// writeSlot, 0, 0, 0}` — used to exist only as prose in `decode`'s
    /// parameter list while five call sites hand-packed it, and it had
    /// already diverged: `PagedDecodeProfiler` fed `row.table.count` as
    /// `tableLen` where the production path feeds
    /// `PagedSequenceKV.decodeTableLength`. Those are the same number only
    /// while a row's physical table happens to be as long as its ring, which
    /// the profiler's gpt-oss shapes make true and a partially-allocated
    /// ring (a prefix-adopted windowed row) makes false — and then the
    /// shader's `table[logicalPage % tableLen]` wraps at the wrong length and
    /// aliases the wrong physical pages.
    ///
    /// So the layout is a type, next to its only consumer.
    struct SeqInfoRow: Equatable {
        /// First absolute position this row may attend.
        var attendStart: Int
        /// Positions attended, INCLUDING the newly written one.
        var attendLength: Int
        /// Divisor for `table[logicalPage % tableLen]`. For a row, this is
        /// `PagedSequenceKV.decodeTableLength` — never `table.count`.
        var tableLength: Int
        /// Fused-write destination. Both zero for a dispatch that does not
        /// write (KV-borrowing layers, attention-only probes).
        var writePage: Int32 = 0
        var writeSlot: Int = 0

        var packed: [Int32] {
            [
                Int32(attendStart), Int32(attendLength), Int32(tableLength),
                writePage, Int32(writeSlot), 0, 0, 0,
            ]
        }
    }

    /// Pack rows into the `[B, 8]` int32 array `decode` takes, and report the
    /// `maxAttendLength` that must accompany it — the two always travel
    /// together, and computing the max separately is its own drift risk.
    static func seqinfo(_ rows: [SeqInfoRow]) -> (array: MLXArray, maxAttendLength: Int) {
        precondition(!rows.isEmpty, "[PagedAttentionKernel] seqinfo needs at least one row")
        var flat = [Int32]()
        flat.reserveCapacity(rows.count * 8)
        var maxAttendLength = 1
        for row in rows {
            flat.append(contentsOf: row.packed)
            maxAttendLength = max(maxAttendLength, row.attendLength)
        }
        return (MLXArray(flat, [rows.count, 8]), maxAttendLength)
    }

    /// Dispatch decode attention for `B` rows.
    ///
    /// - Parameters:
    ///   - queries: `[B, queryHeads, 1, headDim]` or `[B, queryHeads, headDim]`,
    ///     any float dtype (converted to the slab dtype if needed).
    ///   - newKeys/newValues: this step's K/V tiles `[B, kvHeads, headDim]`
    ///     (any float dtype). When non-nil the kernel writes them IN PLACE
    ///     into the slabs at each row's `{writePage, writeSlot}` (seqinfo
    ///     fields 3/4) before attending — the fused decode write. Pass nil
    ///     for KV-borrowing dispatches (the rows were written by their
    ///     owning layer).
    ///   - kSlab/vSlab: pool slabs `[P, kvHeads, pageSize, headDim]`.
    ///   - tables: `[B, maxPages]` int32, `maxPages >= 8`.
    ///   - seqinfo: `[B, 8]` int32 rows — build it with `SeqInfoRow` and
    ///     `PagedAttentionKernel.seqinfo(_:)` rather than packing the layout
    ///     by hand.
    ///   - maxAttendLength: max over rows of the attended length (host-side
    ///     Swift Int — sizes the partial buffers, never a device sync).
    ///   - sinks: optional per-query-head sink logits `[queryHeads]`.
    ///   - params: `[8]` float32 `{softcap, scale, 0…}` (cache it per layer —
    ///     it is constant across steps).
    ///   - softcap: whether params[0] is an active softcap.
    ///   - writeFence: the slab group's write fence (`[1]` int32,
    ///     `PagedKVGroup.writeFence`). The graph edge orders this dispatch
    ///     after every prior in-place write of the group (see
    ///     pagedattention.metal, "In-place slab writes").
    /// - Returns: attention `[B, queryHeads, headDim]` in the slab dtype,
    ///   plus — when the fused write ran — the group's NEXT write fence,
    ///   which the caller MUST store back into the group.
    public static func decode(
        queries: MLXArray,
        newKeys: MLXArray? = nil,
        newValues: MLXArray? = nil,
        kSlab: MLXArray,
        vSlab: MLXArray,
        tables: MLXArray,
        seqinfo: MLXArray,
        maxAttendLength: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        pageSize: Int,
        writeFence: MLXArray,
        kernelSource: String,
        stream: StreamOrDevice = .default
    ) -> (out: MLXArray, nextWriteFence: MLXArray?) {
        var q = queries
        if q.ndim == 4 {
            precondition(q.dim(2) == 1, "decode kernel requires L == 1")
            q = q.squeezed(axis: 2)
        }
        precondition(q.ndim == 3, "queries must be [B, QH, D]")
        precondition(
            (newKeys == nil) == (newValues == nil),
            "newKeys/newValues must be passed together")

        let dtype = kSlab.dtype
        let b = q.dim(0)
        let queryHeads = q.dim(1)
        let headDim = q.dim(2)
        let kvHeads = kSlab.dim(1)
        precondition(kSlab.dim(3) == headDim && vSlab.dim(3) == headDim)
        precondition(kSlab.dim(2) == pageSize)
        precondition(queryHeads % kvHeads == 0, "GQA requires QH % KVH == 0")
        precondition(supportedHeadDims.contains(headDim), "unsupported head dim \(headDim)")
        precondition(tables.dim(0) == b && seqinfo.dim(0) == b)
        precondition(tables.dim(1) >= 8, "pad tables to >= 8 columns for a stable signature")
        precondition(pageSize > 0)
        precondition(maxAttendLength >= 1)

        let gqa = queryHeads / kvHeads
        guard let nsg = simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) else {
            preconditionFailure(
                "paged decode dispatched for a statically ineligible shape: "
                    + (ineligibilityReason(headDim: headDim, gqa: gqa)
                        ?? "headDim \(headDim), GQA \(gqa)"))
        }

        if q.dtype != dtype { q = q.asType(dtype) }

        let hasWrite = newKeys != nil
        var partInputs: [MLXArray] = [q]
        if let newKeys, let newValues {
            precondition(
                newKeys.ndim == 3 && newKeys.dim(0) == b && newKeys.dim(1) == kvHeads
                    && newKeys.dim(2) == headDim,
                "newKeys must be [B, KVH, D]")
            partInputs.append(newKeys.dtype == dtype ? newKeys : newKeys.asType(dtype))
            partInputs.append(
                newValues.dtype == dtype ? newValues : newValues.asType(dtype))
        }
        partInputs.append(contentsOf: [kSlab, vSlab, tables, seqinfo, params, writeFence])

        let hpt = headsPerThreadgroup(headDim: headDim, gqa: gqa)
        let splits = gqa / hpt
        // WS-6.4: partition length adapts to context and batch so a short
        // B == 1 decode still fills the GPU. Every value it can return is a
        // page multiple, which is the invariant the shader relies on.
        let ptok = partitionTokensForDispatch(
            maxAttendLength: maxAttendLength, batch: b, kvHeads: kvHeads,
            headSplits: splits, pageSize: pageSize)
        precondition(ptok % pageSize == 0, "PTOK must be a page multiple")
        // `ceilingDivide`, not `(a + b - 1) / b`: `maxAttendLength` is
        // caller-supplied, and this is the site that motivated the helper.
        let maxParts = ceilingDivide(maxAttendLength, ptok)
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: hasWrite)
        let tg = 32 * nsg
        let partOut = kernel(for: partKey, source: kernelSource)(
            partInputs,
            template: [
                ("T", dtype),
                ("D", headDim),
                ("S", pageSize),
                ("GQA", gqa),
                ("HPT", hpt),
                ("NSG", nsg),
                ("PTOK", ptok),
                ("HAS_SOFTCAP", softcap),
            ],
            grid: (kvHeads * splits * tg, b, maxParts),
            threadGroup: (tg, 1, 1),
            outputShapes: hasWrite
                ? [[b, queryHeads, maxParts, headDim], [b, queryHeads, maxParts, 2], [1]]
                : [[b, queryHeads, maxParts, headDim], [b, queryHeads, maxParts, 2]],
            outputDTypes: hasWrite ? [.float32, .float32, .int32] : [.float32, .float32],
            stream: stream
        )

        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false,
            partitionTokens: ptok)
        let outputs = kernel(for: mergeKey, source: kernelSource)(
            [partOut[0], partOut[1], seqinfo, sinks ?? zeroSinks],
            template: [
                ("T", dtype),
                ("D", headDim),
                ("PTOK", ptok),
                ("HAS_SINKS", sinks != nil),
            ],
            grid: (queryHeads * 32, b, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, queryHeads, headDim]],
            outputDTypes: [dtype],
            stream: stream
        )
        return (outputs[0], hasWrite ? partOut[2] : nil)
    }

    /// Bulk in-place KV write (prefill chunks / prefix adoption): scatter
    /// `keys`/`values` `[KVH, N, D]` (slab dtype) into the slabs at the
    /// host-computed physical `slots` (`[N]` int32, `page * pageSize +
    /// slot` per token) and return the next fence in the group's write
    /// chain. Callers MUST route every subsequent read of the group's
    /// slabs through the returned fence (see pagedattention.metal,
    /// "In-place slab writes") — `PagedKVPool` owns that discipline.
    public static func bulkWrite(
        kSlab: MLXArray,
        vSlab: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        slots: MLXArray,
        prevFence: MLXArray,
        pageSize: Int,
        kernelSource: String,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let dtype = kSlab.dtype
        precondition(keys.ndim == 3 && values.ndim == 3, "tiles must be [KVH, N, D]")
        precondition(keys.dtype == dtype && values.dtype == dtype, "tiles in slab dtype")
        let kvHeads = keys.dim(0)
        let n = keys.dim(1)
        let headDim = keys.dim(2)
        precondition(kSlab.dim(1) == kvHeads && kSlab.dim(3) == headDim)
        precondition(kSlab.dim(2) == pageSize)
        precondition(slots.dim(0) >= max(n, 8) && n > 0, "pad slots to >= 8 entries")

        // The write kernel has no PTOK template parameter; 0 keeps its
        // variant identity independent of the decode partition sizing.
        let key = PagedAttentionKernelKey(
            pass: .write, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: 0,
            simdgroups: 0, hasSinks: false, hasSoftcap: false, partitionTokens: 0)
        let outputs = kernel(for: key, source: kernelSource)(
            [keys, values, slots, prevFence, kSlab, vSlab],
            template: [
                ("T", dtype),
                ("S", pageSize),
            ],
            grid: (headDim, kvHeads, n),
            threadGroup: (min(headDim, 256), 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32],
            stream: stream
        )
        return outputs[0]
    }
}
