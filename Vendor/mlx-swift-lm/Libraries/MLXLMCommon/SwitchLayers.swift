import Foundation
import MLX
import MLXNN

/// Identity gather table for the sorted 64-assignment decode geometry.
nonisolated(unsafe) private let switchDownIdentity64 = MLXArray((0..<64).map { UInt32($0) })

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}
/// Effective-selection count for the direct sorted-expert reduction. Benchmark
/// callers arm this after warmup and snapshot it only after the engine is idle.
/// The unarmed hot path reads one plain Bool and performs no atomic operation,
/// locking, allocation, or clock access.
public struct WeightedExpertUnsortStats: Sendable, Equatable {
    public let effectiveCalls: Int
}

/// Benchmark-facing requested/effective contract for one measured scope.
public struct WeightedExpertUnsortProvenance: Sendable, Equatable {
    public let requested: Bool
    public let effectiveCalls: Int

    public var engaged: Bool { effectiveCalls > 0 }
    public var missingExpectedEngagement: Bool { requested && !engaged }
}

private final class WeightedExpertUnsortProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var effectiveCalls = 0
    // Benchmark boundaries guarantee no engine work is in flight while this
    // plain flag changes. Concurrent recorders only read it while armed.
    private var enabled = false

    @inline(__always)
    func recordEffective() {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
        // Defensively close a recorder/snapshot lock handoff. The idle-boundary
        // contract prevents a concurrent unsynchronized flag mutation.
        guard enabled else { return }
        effectiveCalls += 1
    }

    func snapshot() -> WeightedExpertUnsortStats {
        lock.lock()
        enabled = false
        defer { lock.unlock() }
        return WeightedExpertUnsortStats(effectiveCalls: effectiveCalls)
    }

    func reset() {
        lock.lock()
        effectiveCalls = 0
        enabled = true
        lock.unlock()
    }
}

private let weightedExpertUnsortProbe = WeightedExpertUnsortProbe()

/// Process-wide provenance snapshot for the weighted expert unsort experiment.
public func weightedExpertUnsortStats() -> WeightedExpertUnsortStats {
    weightedExpertUnsortProbe.snapshot()
}

/// Disarm and snapshot one benchmark scope with its resolved request state.
public func weightedExpertUnsortProvenance(
    requested: Bool
) -> WeightedExpertUnsortProvenance {
    let stats = weightedExpertUnsortStats()
    return WeightedExpertUnsortProvenance(
        requested: requested,
        effectiveCalls: stats.effectiveCalls)
}

/// Reset the provenance counters before a benchmark cell.
public func resetWeightedExpertUnsortStats() {
    weightedExpertUnsortProbe.reset()
}

/// Fused inverse-permutation + weighted reduction for the sorted MoE prefill path.
///
/// `SwitchGLU` sorts expert assignments before its gathered matrix multiplies.
/// The regular path restores `[tokens, topK, hidden]` and then reduces it with
/// ``weightedExpertSum``. This kernel reads those sorted rows through the inverse
/// permutation and writes `[tokens, hidden]` directly, avoiding that full
/// `[tokens, topK, hidden]` intermediate.
private let weightedExpertUnsortKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "weighted_expert_unsort_vec8_v3",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
        typedef vec<T, 8> T8;
        // One lane owns eight consecutive features (128-bit load/store), so the
        // grid is an eighth as wide and each row read and store are one
        // eight-wide vector. The hidden extent is `threads_per_grid.x * 8u`,
        // which is 2816.
        uint oct = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        const uint hidden = threads_per_grid.x * 8u;

        T8 accumulator = T8((T)0);
        const uint assignment_base = token * (uint)K;
        for (uint slot = 0; slot < (uint)K; ++slot) {
            const uint assignment = assignment_base + slot;
            const uint sorted_row = (uint)inverse_order[assignment];
            const device T8* row = reinterpret_cast<const device T8*>(
                sorted_outputs + sorted_row * hidden);
            const T8 source = row[oct];
            const float weight = (float)weights[assignment];
            // Preserve the legacy bfloat16 multiply-then-reduce rounding.
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; ++j) {
                const T weighted = (T)((float)source[j] * weight);
                accumulator[j] = accumulator[j] + weighted;
            }
        }
        reinterpret_cast<device T8*>(output + token * hidden)[oct] =
            accumulator;
    """,
    ensureRowContiguous: true
)

/// Consume production-shaped sorted Gemma 4 expert rows through their inverse
/// permutation and reduce original top-K slots into `[tokens, hidden]`.
///
/// This primitive deliberately accepts only the production logical layout:
/// bfloat16 `[tokens * 8, 2816]`, uint32 inverse order, and bfloat16
/// `[tokens, 8]`. Callers must use the legacy scatter + weighted sum for every
/// other dtype, shape, or layout.
public func weightedExpertUnsort(
    sortedOutputs: MLXArray,
    inverseOrder: MLXArray,
    weights: MLXArray
) -> MLXArray {
    precondition(
        sortedOutputs.ndim == 2 && sortedOutputs.dim(1) == 2816
            && sortedOutputs.dtype == .bfloat16,
        "weightedExpertUnsort outputs must be bfloat16 [assignments, 2816]")
    precondition(
        inverseOrder.ndim == 1 && inverseOrder.dtype == .uint32,
        "weightedExpertUnsort inverse order must be flat uint32")
    precondition(
        weights.ndim == 2 && weights.dim(1) == 8 && weights.size >= 64
            && weights.dtype == .bfloat16,
        "weightedExpertUnsort weights must be sorted-prefill bfloat16 [tokens, 8]")
    precondition(
        sortedOutputs.dim(0) == weights.size && inverseOrder.size == weights.size,
        "weightedExpertUnsort assignment counts must match")

    let tokens = weights.dim(0)
    weightedExpertUnsortProbe.recordEffective()
    return weightedExpertUnsortKernel(
        [sortedOutputs, inverseOrder, weights],
        template: [
            ("T", sortedOutputs.dtype),
            ("K", 8),
        ],
        grid: (352, tokens, 1),
        threadGroup: (32, 4, 1),
        outputShapes: [[tokens, 2816]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Exact sorted expert rows whose ordered top-K reduction is intentionally
/// deferred to a downstream fused consumer.
///
/// Keeping this carrier explicit prevents generic callers from mistaking
/// `[assignments, hidden]` for the already-reduced `[tokens, hidden]` result.
public struct DeferredWeightedExpertRows {
    public let sortedOutputs: MLXArray
    public let inverseOrder: MLXArray
    public let weights: MLXArray

    init(sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray) {
        self.sortedOutputs = sortedOutputs
        self.inverseOrder = inverseOrder
        self.weights = weights
    }
}

/// Materialize a deferred carrier through the established reduction. Used only
/// when a downstream fused consumer declines after the producer was selected.
public func resolveDeferredWeightedExpertRows(
    _ rows: DeferredWeightedExpertRows
) -> MLXArray {
    weightedExpertUnsort(
        sortedOutputs: rows.sortedOutputs,
        inverseOrder: rows.inverseOrder,
        weights: rows.weights)
}


// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// GELU-FUSE: the SAME body, compiled WITHOUT `shapeless`, for the routed
/// expert's pinned decode signatures only. Shapeless tracing adds broadcast
/// nodes on every binary op that a shape-specialised trace omits on equal
/// shapes; those nodes push this expression past MLX's fusion depth limit and
/// split it into two Metal kernels with a materialised intermediate. The
/// shape-specialised trace fits and emits one.
private let compiledGeGLUShaped: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(body)
    }
    return body
}()

private let switchGeluShapedFuseEnabled: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_GELU_SHAPED_FUSE"] != "0"

/// Admit only the routed-expert decode rectangles: `[64, 1, N]` / `[64, N]`,
/// both operands bfloat16 with identical shapes. Prefill's per-prompt row
/// counts stay on the shapeless closure so the compiler cache cannot grow.
@inline(__always)
private func geGLUClaimsPinnedDecode(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard switchGeluShapedFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape
    else { return false }
    let s = gate.shape
    if s.count == 3, s[0] == 64, s[1] == 1 { return true }
    if s.count == 2, s[0] == 64 { return true }
    return false
}

/// GELU-FUSE-PREFILL: a bounded set of additionally admitted shapes.
///
/// GELU-FUSE left prefill on the shapeless closure for one stated reason — a
/// shape-specialised compile adds a compiler-cache entry per distinct input
/// shape, the lookup is a linear scan, and prefill row counts vary per prompt,
/// so an unbounded admission would keep growing the scan the decode hot path
/// walks. That reason is about the *number* of entries, not about prefill, so a
/// hard cap answers it directly: at most ``shapedGeluPrefillShapeCap`` distinct
/// rectangles are ever admitted, and the cap+1st falls open to the shapeless
/// closure forever after.
///
/// The decode signatures are matched before this is consulted, so the decode
/// plane never takes the lock and never sees a behaviour change.
public final class ShapedGeluPrefillShapes: @unchecked Sendable {
    private let lock = NSLock()
    private var shapes: [[Int]] = []
    private let cap: Int

    public init(cap: Int) { self.cap = cap }

    @inline(__always)
    public func admits(_ shape: [Int]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if shapes.contains(shape) { return true }
        guard shapes.count < cap else { return false }
        shapes.append(shape)
        return true
    }
}

/// Four is one more than the distinct rectangles a cohort prefill produces (the
/// full batched step, a short final chunk, and the single-stream local verb).
public let shapedGeluPrefillShapeCap = 4

/// Smallest rectangle worth a cache entry. The prefill routed-expert plane is
/// 65,536 rows; every speculative verify width is at most 256, so nothing in
/// production lands near this floor from either side.
public let shapedGeluPrefillMinElements = 1 << 20

private let switchGeluPrefillFuseEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GELU_SHAPED_FUSE_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let switchGeluPrefillShapes = ShapedGeluPrefillShapes(
    cap: shapedGeluPrefillShapeCap)

@inline(__always)
private func geGLUClaimsPrefill(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard switchGeluPrefillFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape,
        gate.size >= shapedGeluPrefillMinElements,
        switchGeluPrefillShapes.admits(gate.shape)
    else { return false }
    CBv2EngageMark.once("gelu-shaped-prefill-experts")
    return true
}

@inline(__always)
private func geGLUProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    if geGLUClaimsPinnedDecode(gate, up) || geGLUClaimsPrefill(gate, up) {
        return compiledGeGLUShaped(gate, up)
    }
    return compiledGeGLU(gate, up)
}
// MARK: - ROUTE-SIMD-RANK-64: exact decode route table

/// Stable rank sort for the exact B=8/top-K=8 decode plane. One SIMDgroup
/// loads the 64 keys once (two keys per lane), broadcasts them to every lane,
/// and directly emits the three routing products consumed downstream.
private let routeSimdRank64Kernel: MLXFast.MLXFastKernel =
    MLXFast.metalKernel(
        name: "mlx_lm_route_simd_rank_scatter_m8_u32_n64_unroll_v2",
        inputNames: ["indices"],
        outputNames: ["row_order", "sorted_keys", "inverse_order"],
        source: """
            const uint assignment = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint key = (uint)indices[assignment];
            const uint key_low = (uint)indices[lane];
            const uint key_high = (uint)indices[32u + lane];
            uint rank = 0;
            #pragma clang loop unroll(full)
            for (uint source = 0; source < 32; ++source) {
                const uint other_low = simd_broadcast(key_low, ushort(source));
                rank += (other_low < key)
                    || (other_low == key && source < assignment);
                const uint other_high = simd_broadcast(key_high, ushort(source));
                const uint high_assignment = 32u + source;
                rank += (other_high < key)
                    || (other_high == key && high_assignment < assignment);
            }
            row_order[rank] = assignment >> 3;
            sorted_keys[rank] = key;
            inverse_order[assignment] = rank;
        """,
        ensureRowContiguous: true
    )

/// EXPERT-PREFIX-BOUNDS-001 carrier for the exact decode route table. The
/// gathered-QMV host ABI has no spare buffer, so each sorted rhs-index word
/// carries its expert plus the within-run bounds that the gather kernel needs:
///
///   bits  0...7   expert id
///   bits  8...13  assignments before this one in its expert run
///   bits 14...19  assignments remaining in the run, including this one, minus 1
///   bits 20...30  zero
///   bit      31   prefix-bounds-valid tag
///
/// The tag makes every other producer fail closed to the incumbent raw-index
/// path. The exact Gemma projection gate below also excludes linear biases,
/// whose generic `bias[indices]` lookup must continue to receive raw indices.
private let routeSimdRank64PrefixBoundsKernel: MLXFast.MLXFastKernel =
    MLXFast.metalKernel(
        name: "mlx_lm_route_simd_rank_bounds_scatter_m8_u32_n64_unroll_v2",
        inputNames: ["indices"],
        outputNames: ["row_order", "sorted_keys", "inverse_order"],
        source: """
            const uint assignment = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint key = (uint)indices[assignment];
            const uint key_low = (uint)indices[lane];
            const uint key_high = (uint)indices[32u + lane];
            uint rank = 0;
            uint run_offset = 0;
            uint run_length = 0;
            #pragma clang loop unroll(full)
            for (uint source = 0; source < 32; ++source) {
                const uint other_low = simd_broadcast(key_low, ushort(source));
                rank += (other_low < key)
                    || (other_low == key && source < assignment);
                run_offset += other_low == key && source < assignment;
                run_length += other_low == key;
                const uint other_high = simd_broadcast(key_high, ushort(source));
                const uint high_assignment = 32u + source;
                rank += (other_high < key)
                    || (other_high == key && high_assignment < assignment);
                run_offset += other_high == key && high_assignment < assignment;
                run_length += other_high == key;
            }
            const uint run_remaining = run_length - run_offset;
            row_order[rank] = assignment >> 3;
            sorted_keys[rank] = 0x80000000u | key
                | (run_offset << 8) | ((run_remaining - 1) << 14);
            inverse_order[assignment] = rank;
        """,
        ensureRowContiguous: true
    )

private let routeSimdRank64Enabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_ROUTE_SIMD_RANK"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

// MARK: - ROUTE-CSORT-64: fused counting-sort route table (donor port)

/// Stable counting sort for the flattened B=8 decode route table (64 uint32
/// expert keys), `argSort`-identical by construction — a port of the proven
/// production fused-scatter v4 kernel from the sibling challenge tree
/// (mlxfast-challenge SwitchLayers.swift `routeCountingSortFused`), retiled
/// from 128 to 64 keys per tile so the exact decode geometry (n = 64
/// assignments → one 256-thread threadgroup) is accepted, and RENAMED
/// (`_t64_v1`) so the Metal pipeline cannot collide with the donor's `_v4`
/// instance if both ever share a metallib cache.
///
/// Exactness (donor stability lemma, re-verified here by parity harness):
/// the vendored merge sort is stable at every stage (thread sort swaps only
/// on strictly-less, the merge prefers A on ties), so `argSort`'s tie order
/// is input order; a stable counting sort reproduces the exact permutation
/// for EVERY input, not just tested ones. At the write point where the
/// scatter emits `order[off] = idx`, every downstream index product of the
/// sorted-MoE chain is already known — `idx / m` is the gathered row
/// (`order.floorDivide(m)`), the tested key IS `indices[order[off]]`, and
/// `off` is the inverse permutation entry for `idx` — so ONE dispatch
/// replaces the 4-kernel `argSort` → `floorDivide` → take → `argSort` chain
/// with byte-identical integer outputs. Counters are commutative integer
/// atomics (any accumulation order produces identical tables); the per-key
/// walk of each tile slice is in input order, so no write order ever
/// depends on scheduling.
///
/// The 256-entry counter table requires keys < 256; callers guarantee this
/// via the `numExperts` guard on `gatherSortIndices`. DEFAULT ON here (the
/// donor tree's own default); `DARKBLOOM_ROUTE_COUNTING_SORT` set to
/// `0`/`false`/`no`/`off` is the kill switch back to the established
/// `argSort` chain. Engage mark: `route-csort64`.
private let routeCountingSort64Enabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_ROUTE_COUNTING_SORT"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let expertPrefixBoundsEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_EXPERT_PREFIX_BOUNDS"]
    else { return false }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let routeSortTile64 = 64
private let routeFusedScatterTopK = 8
/// Key-space bound of the fused scatter's 256-entry counter table.
let routeCountingSortKeyBound = 256

private let routeFusedScatterKernelT64: MLXFast.MLXFastKernel = {
    let m = routeFusedScatterTopK
    return MLXFast.metalKernel(
        name: "mlx_lm_route_csort_scatter_fused_m\(m)_u32_t64_v1",
        inputNames: ["keys"],
        outputNames: ["row_order", "sorted_keys", "inverse_order"],
        source: """
            constexpr uint TILE = \(routeSortTile64);
            constexpr uint M = \(m);
            uint t = threadgroup_position_in_grid.x;
            uint k = thread_position_in_threadgroup.x;
            uint simd_id = k / 32;
            uint lane = k % 32;
            uint n = keys_shape[0];
            // In-threadgroup histograms replace both the standalone hist
            // dispatch and the scan dispatch: one cooperative pass counts
            // every key (totals) and every key in earlier tiles (before),
            // then a simd exclusive prefix over the 256 totals yields the
            // base table. Counts and sums are commutative integer adds, so
            // any accumulation order produces the byte-identical tables.
            threadgroup atomic_uint tg_total[256];
            threadgroup atomic_uint tg_before[256];
            atomic_store_explicit(&tg_total[k], 0u, memory_order_relaxed);
            atomic_store_explicit(&tg_before[k], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Split at the before-limit boundary so the tail segment
            // carries no branch; identical counters, identical adds.
            uint before_limit = t * TILE;
            uint idx = k;
            for (; idx < before_limit; idx += 256) {
                uint key = keys[idx];
                atomic_fetch_add_explicit(
                    &tg_total[key], 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &tg_before[key], 1u, memory_order_relaxed);
            }
            for (; idx < n; idx += 256) {
                atomic_fetch_add_explicit(
                    &tg_total[keys[idx]], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint total = atomic_load_explicit(&tg_total[k], memory_order_relaxed);
            uint lane_excl = simd_prefix_exclusive_sum(total);
            threadgroup uint simd_totals[8];
            if (lane == 31) {
                simd_totals[simd_id] = lane_excl + total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint simd_base = 0;
            for (uint s = 0; s < simd_id; ++s) {
                simd_base += simd_totals[s];
            }
            // Rank base for key k in tile t: global base + earlier tiles.
            uint off = simd_base + lane_excl +
                atomic_load_explicit(&tg_before[k], memory_order_relaxed);
            // Walk this tile's slice in input order: stability by
            // construction, exactly the stock scatter's write order.
            if (total > 0) {
                for (uint i = 0; i < TILE; ++i) {
                    uint idx = t * TILE + i;
                    if (keys[idx] == k) {
                        row_order[off] = idx / M;
                        sorted_keys[off] = k;
                        inverse_order[idx] = off;
                        ++off;
                    }
                }
            }
            """,
        ensureRowContiguous: false
    )
}()

/// Counting-sort twin that emits the tagged per-assignment bounds word while
/// it already owns the exact stable run offsets. This adds no dispatch and no
/// route-table load: `total`, the expert base, and `off` are live sort state.
private let routeFusedScatterPrefixBoundsKernelT64: MLXFast.MLXFastKernel = {
    let m = routeFusedScatterTopK
    return MLXFast.metalKernel(
        name: "mlx_lm_route_csort_bounds_scatter_fused_m\(m)_u32_t64_v1",
        inputNames: ["keys"],
        outputNames: ["row_order", "sorted_keys", "inverse_order"],
        source: """
            constexpr uint TILE = \(routeSortTile64);
            constexpr uint M = \(m);
            uint t = threadgroup_position_in_grid.x;
            uint k = thread_position_in_threadgroup.x;
            uint simd_id = k / 32;
            uint lane = k % 32;
            uint n = keys_shape[0];
            threadgroup atomic_uint tg_total[256];
            threadgroup atomic_uint tg_before[256];
            atomic_store_explicit(&tg_total[k], 0u, memory_order_relaxed);
            atomic_store_explicit(&tg_before[k], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before_limit = t * TILE;
            uint idx = k;
            for (; idx < before_limit; idx += 256) {
                uint key = keys[idx];
                atomic_fetch_add_explicit(
                    &tg_total[key], 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &tg_before[key], 1u, memory_order_relaxed);
            }
            for (; idx < n; idx += 256) {
                atomic_fetch_add_explicit(
                    &tg_total[keys[idx]], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint total = atomic_load_explicit(&tg_total[k], memory_order_relaxed);
            uint lane_excl = simd_prefix_exclusive_sum(total);
            threadgroup uint simd_totals[8];
            if (lane == 31) {
                simd_totals[simd_id] = lane_excl + total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint simd_base = 0;
            for (uint s = 0; s < simd_id; ++s) {
                simd_base += simd_totals[s];
            }
            const uint expert_base = simd_base + lane_excl;
            uint off = expert_base
                + atomic_load_explicit(&tg_before[k], memory_order_relaxed);
            if (total > 0) {
                for (uint i = 0; i < TILE; ++i) {
                    uint idx = t * TILE + i;
                    if (keys[idx] == k) {
                        const uint run_offset = off - expert_base;
                        const uint run_remaining = total - run_offset;
                        row_order[off] = idx / M;
                        sorted_keys[off] = 0x80000000u | k
                            | (run_offset << 8) | ((run_remaining - 1) << 14);
                        inverse_order[idx] = off;
                        ++off;
                    }
                }
            }
            """,
        ensureRowContiguous: false
    )
}()

private func routeCountingSortFusedT64(
    _ indices: MLXArray, m: Int, expertPrefixBounds: Bool = false
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray)? {
    let n = indices.size
    guard routeCountingSort64Enabled,
        indices.dtype == .uint32,
        n > 0, n % routeSortTile64 == 0,
        m == routeFusedScatterTopK
    else { return nil }
    CBv2EngageMark.once("route-csort64")
    let tiles = n / routeSortTile64
    let emitExpertPrefixBounds = expertPrefixBounds && n == routeSortTile64
    if emitExpertPrefixBounds {
        CBv2EngageMark.once("expert-prefix-bounds")
    }
    let kernel = emitExpertPrefixBounds
        ? routeFusedScatterPrefixBoundsKernelT64 : routeFusedScatterKernelT64
    let outputs = kernel(
        [indices],
        grid: (tiles * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - PREFILL-CSORT-128 (general-geometry exact stable counting sort)

/// Exact stable counting sort for the GENERAL MoE route geometry — the
/// prefill/verification tables that ROUTE-CSORT-64 refuses (it is retiled for
/// the n = 64 eight-row decode cohort and pays an O(n) rescan per tile).
///
/// Where the census puts it: in the packed 8x1024 prefill window MLX's generic
/// `argSort` over the flattened route table (`partition_mbsort` +
/// `merge_mbsort`, ~10 dispatches per layer x 30 layers, twice — once for
/// `order`, once for the inverse) costs 392.6 ms of 5508 ms on the M4 (7.2%).
/// Sorts are latency/memory bound, so they do not shrink with the ranked box's
/// NAX GEMM speedup: the same census projects the ROUTE bucket to 19-36% of the
/// sealed 1.254 s M5 prefill window. This lane deletes the sort, not shrinks it.
///
/// Three dispatches, no comparisons:
///   1. `_hist_v1`    — one threadgroup per 256-key block builds a 256-entry
///                      threadgroup histogram (commutative integer atomics) and
///                      writes it to `H[block][key]`.
///   2. `_scan_v1`    — ONE threadgroup: thread `e` sums `H[.][e]` over blocks
///                      to get `total[e]`, a simd exclusive prefix over the 256
///                      totals gives the global bin base `base[e]`, and a second
///                      pass writes `O[block][e] = base[e] + sum_{b<block} H[b][e]`.
///   3. `_scatter_v1` — one threadgroup per block stages the block's 256 keys in
///                      threadgroup memory; thread `k` counts how many earlier
///                      keys IN ITS OWN BLOCK carry its key (`rank`) and lands at
///                      `pos = O[block][key] + rank`.
///
/// Exactness (why this is `argSort`-identical, not merely equal on tests): for
/// the key at flat index `idx` in block `b`, `O[b][key] + rank` is by
/// construction `#{keys with a smaller expert} + #{equal keys at a smaller flat
/// index}`, which is exactly the rank of `idx` under a STABLE sort by key. The
/// vendored merge argsort is stable at every stage (thread sort swaps only on
/// strictly-less, the merge prefers A on ties), so its tie order is input order
/// too — the two permutations agree for EVERY input, not just tested ones.
/// At the single write point every downstream index product is already known:
/// `idx / m` is `order.floorDivide(m)`, the key IS `indices[order]`, and `pos`
/// is the inverse-permutation entry for `idx` (`argSort(order)`), so three
/// dispatches replace `argSort` -> `floorDivide` -> take -> `argSort` with
/// byte-identical integer outputs and every consumer (the `gather_qmm`
/// `rhsIndices`/`lhsIndices`, `weightedExpertUnsort`, `scatterUnsort`) is
/// untouched.
///
/// The counter table is a fixed 256 entries wide regardless of `numExperts`, so
/// no expert count is baked into any kernel: bins above `numExperts` simply hold
/// zero and contribute nothing to the bases. Callers must still prove keys are
/// below that width via the `numExperts` guard (`routeCountingSortKeyBound`).
///
/// Every kernel indexes its inputs linearly, so all three ask MLX for
/// `ensureRowContiguous` — free for the contiguous route tables production
/// actually hands us (MLX skips the copy when the flag is already set) and a
/// hard guarantee for anything else that ever reaches this helper.
///
/// Kill switch: `DARKBLOOM_ROUTE_CSORT_PREFILL` set to `0`/`false`/`no`/`off`
/// restores the `argSort` chain. Engage mark: `route-csort-prefill`.
private let routeCsortPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_ROUTE_CSORT_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// Keys per histogram/scatter block.
private let routeCsortPrefillBlock = 256
/// Counter-table width; must equal ``routeCountingSortKeyBound`` and the 256
/// threads per threadgroup the three kernels launch with.
private let routeCsortPrefillWidth = 256
/// Largest `n` accepted. Positions, block offsets and grid extents are uint32 /
/// Int32 on the Metal side; this bound keeps every one of them representable
/// with room to spare and is ~4000x the largest production route table.
private let routeCsortPrefillMaxKeys = 1 << 28

/// One-shot stderr note of the geometries that reach the route sort. Off unless
/// `DARKBLOOM_ROUTE_CSORT_DEBUG` is set; the hot path reads one static Bool.
private let routeCsortDebugShapes: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTE_CSORT_DEBUG"] != nil

private final class RouteCsortShapeLog: @unchecked Sendable {
    private let lock = NSLock()
    private var seen = Set<String>()

    @inline(__always)
    func note(_ make: () -> String) {
        guard routeCsortDebugShapes else { return }
        let key = make()
        lock.lock()
        let fresh = seen.insert(key).inserted
        lock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[route-csort] \(key)\n".utf8))
        }
    }
}

private let routeCsortShapeLog = RouteCsortShapeLog()

private let routeCsortPrefillHistKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_hist_v1",
    inputNames: ["keys"],
    outputNames: ["block_hist"],
    source: """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        threadgroup atomic_uint tg_count[WIDTH];
        atomic_store_explicit(&tg_count[k], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint idx = b * BLOCK + k;
        if (idx < n) {
            atomic_fetch_add_explicit(
                &tg_count[keys[idx]], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Integer adds commute, so the table is identical for every
        // interleaving the hardware picks.
        block_hist[b * WIDTH + k] =
            atomic_load_explicit(&tg_count[k], memory_order_relaxed);
        """,
    ensureRowContiguous: true
)

private let routeCsortPrefillScanKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_scan_v3",
    inputNames: ["block_hist"],
    outputNames: ["block_offset", "expert_offsets", "tile_offsets"],
    source: """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint e = thread_position_in_threadgroup.x;
        uint simd_id = e / 32;
        uint lane = e % 32;
        uint nblocks = (uint)block_hist_shape[0];
        uint total = 0u;
        // Admission proves every key is below NE, so columns at or above it are
        // zero in every block and cannot contribute to the total. Skipping the
        // accumulation retires whole SIMD groups at once when the counter table
        // is wider than the model's expert count.
        if (e < (uint)NE) {
            for (uint b = 0; b < nblocks; ++b) {
                total += block_hist[b * WIDTH + e];
            }
        }
        // Global bin base: exclusive prefix over the 256 expert totals.
        uint lane_excl = simd_prefix_exclusive_sum(total);
        threadgroup uint simd_totals[8];
        if (lane == 31) {
            simd_totals[simd_id] = lane_excl + total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            running += simd_totals[s];
        }
        running += lane_excl;

        // The NAX expert kernel consumes the same exclusive expert prefix and
        // an exclusive prefix of eight-row tiles. Expert `NE` is the terminal
        // entry, so the 129-entry tables describe empty experts as well.
        uint tile_count = (total + 7u) / 8u;
        uint tile_lane_excl = simd_prefix_exclusive_sum(tile_count);
        threadgroup uint simd_tile_totals[8];
        if (lane == 31) {
            simd_tile_totals[simd_id] = tile_lane_excl + tile_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            tile_running += simd_tile_totals[s];
        }
        tile_running += tile_lane_excl;
        if (e <= (uint)NE) {
            expert_offsets[e] = running;
            tile_offsets[e] = tile_running;
        }

        // Exclusive scan over blocks for this expert, offset by the bin base.
        // Column `e` of `block_offset` is read by the scatter only as
        // `block_offset[b * WIDTH + key]` for a key that occurs in block `b`,
        // so a column whose global total is zero is never read and need not be
        // written. The counter table is 256 wide while the model routes 128
        // experts, so at minimum half the columns are unconditionally dead.
        if (total > 0u) {
            for (uint b = 0; b < nblocks; ++b) {
                block_offset[b * WIDTH + e] = running;
                running += block_hist[b * WIDTH + e];
            }
        }
        """,
    ensureRowContiguous: true
)

/// PROMPT-GLUE2 (pg2): `mlx_lm_route_csort128_scan_v3` from one 1024-thread
/// threadgroup, eight block ranges wide. The incumbent's one threadgroup of
/// 256 threads walks every block twice in sequence per expert column; here
/// the block loop is split into eight ranges of `nblocks / 8`, one part per
/// 128-column slice of the threadgroup. Every quantity is an unsigned
/// integer sum, so the range partials combined in range order are the
/// incumbent's totals word for word, the expert prefix is the same
/// `simd_prefix_exclusive_sum` over the same lanes (each part's simdgroups
/// hold the same totals; part 0 publishes the simdgroup totals), and each
/// part's running offset starts at the bin base plus the earlier ranges'
/// counts, which is exactly the incumbent's running value at that block.
/// Columns with a zero total are left unwritten, as the incumbent leaves
/// them. Admitted for 128 experts, the 256-wide table and a block count
/// divisible by eight.
///
/// NAX extension: emits the same `expert_offsets` / `tile_offsets` exclusive
/// prefixes as `routeCsortPrefillScanKernel` (129 entries, terminal included)
/// so the fused NAX MoE consumer sees identical tables whichever scan ran.
/// The 8x128 layout has no `e == NE` thread, so part 0 writes columns
/// `e < NE` plus the terminal entry from the last column.
private let routeCsortPrefillScanKernelPg2: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_scan_pg2",
    inputNames: ["block_hist"],
    outputNames: ["block_offset", "expert_offsets", "tile_offsets"],
    source: """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        constexpr uint PARTS = 8;
        constexpr uint COLS = (uint)NE;
        // PROMPT-GLUE2 (pg2): 1024 threads = PARTS block ranges x COLS expert
        // columns. Every sum below is an unsigned integer sum, so the split of
        // the block loop into PARTS ranges combined in range order yields the
        // incumbent's totals and running offsets word for word.
        const uint t = thread_position_in_threadgroup.x;
        const uint e = t % COLS;
        const uint part = t / COLS;
        const uint nblocks = (uint)block_hist_shape[0];
        const uint per = nblocks / PARTS;
        const uint b0 = part * per;
        threadgroup uint part_sum[PARTS][COLS];
        threadgroup uint simd_totals[COLS / 32];
        uint partial = 0u;
        for (uint b = b0; b < b0 + per; ++b) {
            partial += block_hist[b * WIDTH + e];
        }
        part_sum[part][e] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total = 0u;
        for (uint p = 0; p < PARTS; ++p) {
            total += part_sum[p][e];
        }
        // Global bin base: exclusive prefix over the expert totals. Each part's
        // simdgroups hold the same totals in the same lanes, so every part
        // computes the same prefix; part 0 publishes the simdgroup totals.
        const uint lane = e % 32;
        const uint simd_id = e / 32;
        const uint lane_excl = simd_prefix_exclusive_sum(total);
        if (part == 0 && lane == 31) {
            simd_totals[simd_id] = lane_excl + total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            running += simd_totals[s];
        }
        running += lane_excl;
        // NAX prefixes: the same exclusive expert base and exclusive
        // eight-row-tile base the incumbent v3 scan publishes. `running` here
        // is still the bin base (earlier parts are added below for the block
        // scan only), and `total` is identical across parts, so every part
        // computes the same bases; part 0 publishes them once.
        const uint expert_base = running;
        uint tile_count = (total + 7u) / 8u;
        uint tile_lane_excl = simd_prefix_exclusive_sum(tile_count);
        threadgroup uint simd_tile_totals[COLS / 32];
        if (part == 0 && lane == 31) {
            simd_tile_totals[simd_id] = tile_lane_excl + tile_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_base = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            tile_base += simd_tile_totals[s];
        }
        tile_base += tile_lane_excl;
        if (part == 0 && e < (uint)NE) {
            expert_offsets[e] = expert_base;
            tile_offsets[e] = tile_base;
        }
        if (part == 0 && e == COLS - 1) {
            expert_offsets[NE] = expert_base + total;
            tile_offsets[NE] = tile_base + tile_count;
        }
        for (uint p = 0; p < part; ++p) {
            running += part_sum[p][e];
        }
        // Exclusive scan over this part's blocks for the column, offset by the
        // bin base plus the earlier parts' counts; columns with a zero total are
        // never read by the scatter and are left unwritten, as the incumbent
        // leaves them.
        if (total > 0u) {
            for (uint b = b0; b < b0 + per; ++b) {
                block_offset[b * WIDTH + e] = running;
                running += block_hist[b * WIDTH + e];
            }
        }
        """,
    ensureRowContiguous: true
)

private let routeCsortPrefillScatterKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort128_scatter_v1",
    inputNames: ["keys", "block_offset"],
    outputNames: ["row_order", "sorted_keys", "inverse_order"],
    source: """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        uint idx = b * BLOCK + k;
        // Tail block: the sentinel is outside the proven key space (keys are
        // below the 256-wide counter table), so it can never tie a real key.
        uint key = (idx < n) ? keys[idx] : 0xffffffffu;
        threadgroup uint tg_keys[BLOCK];
        tg_keys[k] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < n) {
            // Stable local rank: earlier keys in this block only. Read in
            // index order from threadgroup memory, so no write position ever
            // depends on scheduling.
            uint rank = 0u;
            for (uint j = 0; j < k; ++j) {
                rank += (tg_keys[j] == key) ? 1u : 0u;
            }
            uint pos = block_offset[b * WIDTH + key] + rank;
            row_order[pos] = idx / (uint)M;
            sorted_keys[pos] = key;
            inverse_order[idx] = pos;
        }
        """,
    ensureRowContiguous: true
)

/// Exact stable counting sort of a flat uint32 route table. Returns nil (fail
/// closed onto `argSort`) unless every precondition of the kernels holds.
private struct RouteCsortPrefillResult {
    let rowOrder: MLXArray
    let sortedKeys: MLXArray
    let inverseOrder: MLXArray
    let expertOffsets: MLXArray
    let tileOffsets: MLXArray
}

private func routeCountingSortPrefill(
    _ indices: MLXArray, m: Int, numExperts: Int
) -> RouteCsortPrefillResult? {
    let n = indices.size
    guard routeCsortPrefillEnabled,
        indices.dtype == .uint32,
        indices.ndim == 1,
        numExperts > 0,
        numExperts <= routeCsortPrefillWidth,
        numExperts <= routeCountingSortKeyBound,
        m >= 1,
        n > routeSortTile64,
        n <= routeCsortPrefillMaxKeys
    else { return nil }
    CBv2EngageMark.once("route-csort-prefill")
    let blocks = (n + routeCsortPrefillBlock - 1) / routeCsortPrefillBlock
    let width = routeCsortPrefillWidth
    let hist = routeCsortPrefillHistKernel(
        [indices],
        grid: (blocks * width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[blocks, width]],
        outputDTypes: [.uint32]
    )[0]
    let scanOutputs: [MLXArray]
    // PROMPT-GLUE2 (pg2): the prompt plane's key table takes the eight-part
    // scan; every other table keeps the incumbent dispatch. Both scans emit
    // the full NAX triple (block offsets plus the 129-entry expert/tile
    // prefixes), so the scatter always reads scanOutputs[0] and the fused NAX
    // MoE consumer always reads scanOutputs[1]/[2] from whichever ran.
    if Gemma4PromptGlue2V1.enabled, numExperts == 128, width == 256,
        blocks >= 8, blocks % 8 == 0, n / m >= Gemma4PromptGlue2V1.minRows
    {
        scanOutputs = routeCsortPrefillScanKernelPg2(
            [hist],
            template: [("NE", numExperts)],
            grid: (1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[blocks, width], [numExperts + 1], [numExperts + 1]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
        if Gemma4PromptGlue2V1.xcheck {
            let reference = routeCsortPrefillScanKernel(
                [hist],
                template: [("NE", numExperts)],
                grid: (width, 1, 1),
                threadGroup: (width, 1, 1),
                outputShapes: [[blocks, width], [numExperts + 1], [numExperts + 1]],
                outputDTypes: [.uint32, .uint32, .uint32]
            )
            // Only columns with a nonzero total are written by either kernel.
            let written = hist.sum(axis: 0) .> UInt32(0)
            Gemma4PromptGlue2V1.report(
                scanOutputs[0], reference: reference[0], site: "route-csort scan",
                mask: written)
        }
        Gemma4PromptGlue2V1.mark()
    } else {
        scanOutputs = routeCsortPrefillScanKernel(
            [hist],
            template: [("NE", numExperts)],
            grid: (width, 1, 1),
            threadGroup: (width, 1, 1),
            outputShapes: [[blocks, width], [numExperts + 1], [numExperts + 1]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
    }
    let outputs = routeCsortPrefillScatterKernel(
        [indices, scanOutputs[0]],
        template: [("M", m)],
        grid: (blocks * width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return RouteCsortPrefillResult(
        rowOrder: outputs[0], sortedKeys: outputs[1], inverseOrder: outputs[2],
        expertOffsets: scanOutputs[1], tileOffsets: scanOutputs[2])
}

/// GLUE-FOLD carrier: the exact decode route table (`row_order`,
/// `sorted_keys`, `inverse_order`, each `[64]` uint32) computed upstream by a
/// producer that already holds the router scores, so `projectExperts` never
/// issues its standalone route-table dispatch. The arrays must be exactly what
/// `gatherSortIndices` would have produced for the same `[8, 8]` indices --
/// raw (untagged) sorted expert keys included -- and any shape, dtype or
/// switch-state mismatch declines the carrier and re-issues the incumbent
/// chain unchanged.
public struct SwitchRouteTable {
    public let rowOrder: MLXArray
    public let sortedKeys: MLXArray
    public let inverseOrder: MLXArray

    public init(rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray) {
        self.rowOrder = rowOrder
        self.sortedKeys = sortedKeys
        self.inverseOrder = inverseOrder
    }
}

/// `numExperts` is the exclusive upper bound of the index key space. Callers
/// that know it (SwitchGLU) pass it so PREFILL-CSORT-128 can prove its 256-entry
/// counter table covers every key; the default (`Int.max`) fails closed onto the
/// established `argSort` chain, which is what the generic MoE models that share
/// this helper (GPTOSS, NemotronH) keep getting.
public func gatherSort(
    x: MLXArray, indices: MLXArray, numExperts: Int = Int.max
) -> (MLXArray, MLXArray, MLXArray) {
    if routeSimdRank64Enabled,
        (indices.shape == [8, 8] || (indices.ndim == 1 && indices.size == 64)),
        indices.dtype == .uint32
    {
        let flat = indices.flattened()
        let outputs = routeSimdRank64Kernel(
            [flat],
            grid: (64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[64], [64], [64]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
        return (
            x.flattened(start: 0, end: -3)[outputs[0]],
            outputs[1],
            outputs[2]
        )
    }
    let m = indices.dim(-1)
    let indices = indices.flattened()
    routeCsortShapeLog.note {
        "gatherSort n=\(indices.size) m=\(m) E=\(numExperts) "
            + "dtype=\(indices.dtype)"
    }
    // PREFILL-CSORT-128: three dispatches with byte-identical outputs.
    if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
        return (
            x.flattened(start: 0, end: -3)[fused.rowOrder],
            fused.sortedKeys,
            fused.inverseOrder
        )
    }
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

/// PRENORM-GATHER: the sort of `gatherSort` without its gather. Returns the
/// token row of every sorted position, the sorted expert keys and the inverse
/// order, derived exactly as `gatherSort` derives them (the PREFILL-CSORT-128
/// kernels when they admit, the `argSort` chain otherwise), so a producer that
/// knows the inverse order can emit the sorted plane itself and the standalone
/// gather of `x` is never issued.
public func gatherSortOrder(
    indices: MLXArray, numExperts: Int = Int.max
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    routeCsortShapeLog.note {
        "gatherSortOrder n=\(indices.size) m=\(m) E=\(numExperts) "
            + "dtype=\(indices.dtype)"
    }
    if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
        return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
    }
    let order = argSort(indices)
    let inverseOrder = argSort(order)
    return (order.floorDivide(m), indices[order], inverseOrder)
}

/// PRENORM-GATHER: a producer that emits the sorted expert plane
/// `[assignments, 1, inputDims]` directly from the sort's inverse order, so
/// `SwitchGLU` never issues its standalone gather of the activations it was
/// handed. Returning `nil`, or a plane of any other shape or dtype, selects
/// that gather.
public typealias SwitchSortedPlaneProducer = (_ inverseOrder: MLXArray) -> MLXArray?

/// `numExperts` is the exclusive upper bound of the index key space; callers
/// that know it (SwitchGLU) pass it so the counting-sort fast path can prove
/// its 256-entry counter table covers every key. The default (`Int.max`)
/// fails closed onto the established `argSort` chain.
public func gatherSortIndices(
    indices: MLXArray, numExperts: Int = Int.max,
    expertPrefixBounds: Bool = false
) -> (MLXArray, MLXArray, MLXArray) {
    if routeSimdRank64Enabled,
        (indices.shape == [8, 8] || (indices.ndim == 1 && indices.size == 64)),
        indices.dtype == .uint32
    {
        if expertPrefixBounds {
            CBv2EngageMark.once("expert-prefix-bounds")
        }
        let flat = indices.flattened()
        let kernel = expertPrefixBounds
            ? routeSimdRank64PrefixBoundsKernel : routeSimdRank64Kernel
        let outputs = kernel(
            [flat],
            grid: (64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[64], [64], [64]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
        return (outputs[0], outputs[1], outputs[2])
    }
    let m = indices.dim(-1)
    let indices = indices.flattened()
    routeCsortShapeLog.note {
        "gatherSortIndices n=\(indices.size) m=\(m) E=\(numExperts) "
            + "dtype=\(indices.dtype)"
    }
    if numExperts <= routeCountingSortKeyBound {
        // PREFILL-CSORT-128 owns everything wider than the retiled decode
        // cohort; ROUTE-CSORT-64 keeps the exact n = 64 geometry it was built
        // for (one threadgroup, no histogram/scan round trip).
        if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
            return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
        }
        // ROUTE-CSORT-64: one fused dispatch with byte-identical outputs
        // (default ON; see routeCountingSort64Enabled above).
        if let fused = routeCountingSortFusedT64(
            indices, m: m, expertPrefixBounds: expertPrefixBounds)
        {
            return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
        }
    }
    let order = argSort(indices)
    return (order.floorDivide(m), indices[order], argSort(order))
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - GATEUP-GEGLU-PREFILL: close the fused expert GEMM in its store epilogue

/// GATEUP-GEGLU-PREFILL. On the sorted routed-expert prefill plane the gate and
/// up projections are two `gather_qmm_rhs` dispatches (N = 704 each) over the
/// same gathered activations `[rows * topK, 1, 2816]` and the same sorted
/// expert keys; the activations (369 MB for the packed 8 x 1024 cohort) are
/// therefore streamed from DRAM twice per layer. This arm issues one gather
/// over a paired gate/up right-hand side and closes the rounded GeGLU directly
/// from its MMA fragments. The activation is read once and both the second
/// gather and the standalone shaped-GeGLU dispatch disappear.
///
/// Storage. Adjacent 16-column gate/up blocks let both the NAX 64-column tile
/// and portable 32-column tile own complete pairs without cross-threadgroup
/// exchange. Decode and speculative verification still require contiguous
/// split matrices, so the loaded gate/up arrays remain their primary storage
/// and the paired prefill plane is one additional load-time copy. It contains
/// the same frozen quantized bytes; no weight is re-quantized or represented in
/// another numerical format.
///
/// Exactness: every output column of the gathered quantized GEMM owns an
/// independent K-chain -- the tile pipeline (bm/bn/bk, K-step order, per-group
/// affine dequant `scale * nibble + bias`) is unchanged. The epilogue rounds
/// each accumulator to bfloat16 at the same boundary as the two stock GEMM
/// outputs, then reproduces every bfloat16 temporary in `compiledGeGLU` before
/// storing the compact 704-column plane.
///
/// Routing. The fused right-hand side is dispatched exactly where the host
/// would select the sorted right-hand-side kernel: a sorted plane without
/// left-hand indices of at least `max(16, 4 * experts)` rows (512 here). The
/// eight-row decode cohort carries left-hand indices and never qualifies;
/// smaller sorted rectangles (speculative verification) keep the split views
/// and their gathered vector kernels. The `gate_up_proj` module slot is
/// deliberately NOT populated: the direct sorted reduction requires it to be
/// nil, so the storage lives in a private, reflection-inert member.
///
/// Kill switch: `DARKBLOOM_GEMMA4_PREFILL_GATEUP_FUSE` set to
/// `0`/`false`/`no`/`off` leaves the split arrays as loaded and restores the
/// two split gathers. Engage marks: `prefill-gateup-fuse` and
/// `prefill-gateup-gelu-epilogue`.
public let switchGateUpFusePrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_GATEUP_FUSE"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// The fused gate/up affine 4-bit right-hand side of one expert layer plus the
/// split projection arrays used outside the large sorted prefill path. A
/// plain final class (not a `Module`, not an `MLXArray` tuple) so the module
/// reflection that enumerates parameters treats it as an opaque value: it is
/// never a parameter in its own right, never quantized again, never saved and
/// never updated. Built once at load, retained for the life of the layer.
public final class SwitchGateUpFusedStorage {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    public let gateWeight: MLXArray
    public let gateScales: MLXArray
    public let gateBiases: MLXArray
    public let upWeight: MLXArray
    public let upScales: MLXArray
    public let upBiases: MLXArray
    public let hiddenDims: Int

    /// Exact production geometry only: two packed affine 4-bit / group-64
    /// planes of 2816 -> 704 over 128 experts, `[128, 704, 352]` uint32 with
    /// `[128, 704, 44]` bfloat16 scales and biases. Any other pair returns nil
    /// and leaves the split storage exactly as loaded.
    public init?(
        gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray,
        upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray
    ) {
        guard gateWeight.shape == [128, 704, 352]
            && gateScales.shape == [128, 704, 44]
            && gateBiases.shape == [128, 704, 44]
            && upWeight.shape == [128, 704, 352]
            && upScales.shape == [128, 704, 44]
            && upBiases.shape == [128, 704, 44]
            && gateWeight.dtype == .uint32 && upWeight.dtype == .uint32
            && gateScales.dtype == .bfloat16 && upScales.dtype == .bfloat16
            && gateBiases.dtype == .bfloat16 && upBiases.dtype == .bfloat16
        else { return nil }
        let n = 704
        // Pair one 16-column gate block with the matching up block. Both the
        // NAX 64-column tile and portable 32-column tile then own complete
        // gate/up pairs and can close GeGLU without cross-group exchange.
        func paired16(_ gate: MLXArray, _ up: MLXArray, tail: Int) -> MLXArray {
            let gateBlocks = gate.reshaped(128, n / 16, 16, tail)
            let upBlocks = up.reshaped(128, n / 16, 16, tail)
            return MLX.stacked([gateBlocks, upBlocks], axis: 2)
                .reshaped(128, n * 2, tail)
        }
        self.weight = paired16(gateWeight, upWeight, tail: 352)
        self.scales = paired16(gateScales, upScales, tail: 44)
        self.biases = paired16(gateBiases, upBiases, tail: 44)
        self.gateWeight = gateWeight
        self.gateScales = gateScales
        self.gateBiases = gateBiases
        self.upWeight = upWeight
        self.upScales = upScales
        self.upBiases = upBiases
        self.hiddenDims = n
    }
}

// MARK: - FLASH-MOE-PREFILL: transient tiled expert MLP

/// Experimental and intentionally opt-in. The NAX kernel targets the exact
/// Gemma 4 expert contract; the scalar kernel below remains its correctness
/// fallback. Both consume the sorted expert plane and route products already
/// produced above and never create a second regrouping structure.
private let switchFlashMoEPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_FLASH_MOE"]
    else { return false }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// One group owns one expert and walks that expert's contiguous sorted run.
/// The 2816-element accumulator is threadgroup-local; the 64-element gate,
/// up, and GeGLU tiles never leave threadgroup memory.
private let switchFlashMoEScalarKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_flash_moe_q4_g64_t64_v1",
    inputNames: [
        "x", "indices", "gate_up_weight", "gate_up_scales", "gate_up_biases",
        "down_weight", "down_scales", "down_biases",
    ],
    outputNames: ["out"],
    source: """
        const uint tid = thread_position_in_threadgroup.x;
        const uint expert = threadgroup_position_in_grid.x;
        const uint assignment_count = indices_shape[0];

        threadgroup uint bounds[2];
        threadgroup T gate_tile[64];
        threadgroup T up_tile[64];
        threadgroup T h_tile[64];
        threadgroup float y_accumulator[2816];

        // `indices` is the existing stable expert-major route table. Since it
        // is sorted, one cursor identifies this expert's complete run without
        // building another boundary/index buffer.
        uint cursor = 0u;
        if (tid == 0u) {
            while (cursor < assignment_count && indices[cursor] < expert) {
                cursor++;
            }
            const uint begin = cursor;
            while (cursor < assignment_count && indices[cursor] == expert) {
                cursor++;
            }
            bounds[0] = begin;
            bounds[1] = cursor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint begin = bounds[0];
        const uint end = bounds[1];
        for (uint assignment = begin; assignment < end; assignment++) {
            // Each lane owns 44 output elements. This accumulator is reset for
            // one sorted assignment and retained across all 11 H tiles.
            for (uint n = tid; n < 2816u; n += 64u) {
                y_accumulator[n] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tile_base = 0u; tile_base < 704u; tile_base += 64u) {
                const uint j = tile_base + tid;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (uint k = 0u; k < 2816u; k++) {
                    const float value = static_cast<float>(x[
                        size_t(assignment) * 2816u + k]);
                    gate_sum += value * flash_moe_q4<T>(
                        gate_up_weight, gate_up_scales, gate_up_biases,
                        expert, j, k, 1408u, 352u, 44u);
                    up_sum += value * flash_moe_q4<T>(
                        gate_up_weight, gate_up_scales, gate_up_biases,
                        expert, 704u + j, k, 1408u, 352u, 44u);
                }

                // Match the two gathered QMM outputs' BF16 materialization
                // before applying the existing tanh-approximate GeGLU math.
                gate_tile[tid] = static_cast<T>(gate_sum);
                up_tile[tid] = static_cast<T>(up_sum);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const float gate = static_cast<float>(gate_tile[tid]);
                const float up = static_cast<float>(up_tile[tid]);
                const float gelu = 0.5f * gate * (1.0f + tanh(
                    sqrt(2.0f / 3.14159265358979323846f)
                        * (gate + 0.044715f * gate * gate * gate)));
                h_tile[tid] = static_cast<T>(gelu * up);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint n = tid; n < 2816u; n += 64u) {
                    float accumulator = y_accumulator[n];
                    for (uint local_j = 0u; local_j < 64u; local_j++) {
                        accumulator += static_cast<float>(h_tile[local_j])
                            * flash_moe_q4<T>(
                                down_weight, down_scales, down_biases,
                                expert, n, tile_base + local_j,
                                2816u, 88u, 11u);
                    }
                    y_accumulator[n] = accumulator;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint n = tid; n < 2816u; n += 64u) {
                out[size_t(assignment) * 2816u + n] =
                    static_cast<T>(y_accumulator[n]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
    header: """
        template <typename T>
        inline float flash_moe_q4(
            const device uint* weight,
            const device T* scales,
            const device T* biases,
            uint expert,
            uint output_row,
            uint input_index,
            uint output_rows,
            uint packed_columns,
            uint group_columns) {
            const size_t row = size_t(expert) * output_rows + output_row;
            const uint word = weight[row * packed_columns + (input_index >> 3u)];
            const uint q = (word >> ((input_index & 7u) * 4u)) & 0x0fu;
            const size_t group = row * group_columns + (input_index >> 6u);
            return static_cast<float>(scales[group]) * static_cast<float>(q)
                + static_cast<float>(biases[group]);
        }
    """,
    ensureRowContiguous: true
)

/// The custom-kernel compiler does not have the vendored MLX include path, so
/// this is the small fixed-shape subset of the shipped NAX primitives needed by
/// the Gemma M=8 route. The fragment coordinates and MPP descriptor are kept
/// identical to `steel/gemm/nax.h`; only the generic surface is narrowed to the
/// affine 4-bit, group-64 loader used here.
private let switchFlashMoENAXHeader = """
    #include <metal_simdgroup>
    #include <metal_simdgroup_matrix>
    #include <metal_stdlib>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

    using namespace metal;

    namespace mlx {
    namespace steel {

    struct NAXFrag {
        static constant constexpr const short kFragRows = 16;
        static constant constexpr const short kFragCols = 16;
        static constant constexpr const short kElemsPerFrag = 8;
        static constant constexpr const short kElemRows = 2;
        static constant constexpr const short kElemCols = 4;
        static constant constexpr const short kElemRowsJump = 8;

        template <typename U>
        using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

        static inline short2 get_coord() {
            const ushort simd_lane_id =
                __metal_get_thread_index_in_simdgroup(ushort());
            const short qid = simd_lane_id >> 2;
            const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
            const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
            return short2{fn, fm};
        }

        template <typename T, typename P>
        static inline void load(
            thread dtype_frag_t<T>& dst, P src, int str_x, int str_y,
            short off_x, short off_y) {
            const short2 sc = get_coord();
            src += sc.y * str_x + sc.x * str_y;
            for (short i = 0; i < kElemRows; ++i) {
                for (short j = 0; j < kElemCols; ++j) {
                    dst[i * kElemCols + j] = static_cast<T>(src[
                        (off_x + i * kElemRowsJump) * str_x
                            + (off_y + j) * str_y]);
                }
            }
        }

        template <typename T, typename P>
        static inline void load_rows(
            thread dtype_frag_t<T>& dst, P src, int str_x, int str_y,
            short lim_x, short off_x, short off_y) {
            const short2 sc = get_coord();
            src += sc.y * str_x + sc.x * str_y;
            const short rows = lim_x - sc.y;
            for (short i = 0; i < kElemRows; ++i) {
                const short r = off_x + i * kElemRowsJump;
                for (short j = 0; j < kElemCols; ++j) {
                    dst[i * kElemCols + j] = r < rows
                        ? static_cast<T>(src[r * str_x + (off_y + j) * str_y])
                        : T(0);
                }
            }
        }

        template <typename C, typename U>
        static inline void store_safe(
            const thread dtype_frag_t<C>& src, threadgroup U* dst, int str_x,
            int str_y, short lim_x, short lim_y, short off_x, short off_y) {
            const short2 sc = get_coord();
            dst += sc.y * str_x + sc.x * str_y;
            for (short i = 0; i < kElemRows; ++i) {
                const short r = off_x + i * kElemRowsJump;
                for (short j = 0; j < kElemCols; ++j) {
                    const short c = off_y + j;
                    if (r < lim_x - sc.y && c < lim_y - sc.x) {
                        dst[r * str_x + c * str_y] = static_cast<U>(
                            src[i * kElemCols + j]);
                    }
                }
            }
        }

        template <typename C, typename U>
        static inline void store_safe(
            const thread dtype_frag_t<C>& src, device U* dst, int str_x,
            int str_y, short lim_x, short lim_y, short off_x, short off_y) {
            const short2 sc = get_coord();
            dst += sc.y * str_x + sc.x * str_y;
            for (short i = 0; i < kElemRows; ++i) {
                const short r = off_x + i * kElemRowsJump;
                for (short j = 0; j < kElemCols; ++j) {
                    const short c = off_y + j;
                    if (r < lim_x - sc.y && c < lim_y - sc.x) {
                        dst[r * str_x + c * str_y] = static_cast<U>(
                            src[i * kElemCols + j]);
                    }
                }
            }
        }

        template <typename CType, typename AType, typename BType>
        static inline void mma(
            thread dtype_frag_t<CType>& C0, thread dtype_frag_t<CType>& C1,
            const thread dtype_frag_t<AType>& A,
            const thread dtype_frag_t<BType>& B0,
            const thread dtype_frag_t<BType>& B1) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, 16, false, true, true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
            auto ct_a = gemm_op.template get_left_input_cooperative_tensor<
                AType, BType, CType>();
            auto ct_b = gemm_op.template get_right_input_cooperative_tensor<
                AType, BType, CType>();
            auto ct_c = gemm_op.template get_destination_cooperative_tensor<
                decltype(ct_a), decltype(ct_b), CType>();
            for (short i = 0; i < kElemsPerFrag; ++i) {
                ct_a[i] = A[i];
                ct_b[i] = B0[i];
                ct_b[kElemsPerFrag + i] = B1[i];
                ct_c[i] = C0[i];
                ct_c[kElemsPerFrag + i] = C1[i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; ++i) {
                C0[i] = ct_c[i];
                C1[i] = ct_c[kElemsPerFrag + i];
            }
        }
    };

    template <typename T, short TILE_ROWS, short TILE_COLS>
    struct NAXTile {
        using NAXFrag_t = NAXFrag;
        using frag_type = typename NAXFrag_t::template dtype_frag_t<T>;
        static constant constexpr const short kTileRows = TILE_ROWS;
        static constant constexpr const short kTileCols = TILE_COLS;
        static constant constexpr const short kNumFrags = TILE_ROWS * TILE_COLS;
        frag_type val_frags[kNumFrags];

        inline void clear() {
            for (short i = 0; i < kNumFrags; ++i) {
                val_frags[i] = frag_type(0);
            }
        }

        inline thread frag_type& frag_at(short i, short j) {
            return val_frags[i * kTileCols + j];
        }

        inline const thread frag_type& frag_at(short i, short j) const {
            return val_frags[i * kTileCols + j];
        }

        template <bool transpose>
        inline thread frag_type& frag_at(
            short i, short j, metal::bool_constant<transpose>) {
            if constexpr (transpose) {
                return frag_at(j, i);
            }
            return frag_at(i, j);
        }

        template <typename U>
        inline void load(const device U* src, int ld) {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::load(
                        frag_at(i, j), src, ld, 1, i * 16, j * 16);
                }
            }
        }

        template <typename U>
        inline void load_rows(const device U* src, int ld, short rows) {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::load_rows(
                        frag_at(i, j), src, ld, 1, rows, i * 16, j * 16);
                }
            }
        }

        template <typename U>
        inline void load_rows(const threadgroup U* src, int ld, short rows) {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::load_rows(
                        frag_at(i, j), src, ld, 1, rows, i * 16, j * 16);
                }
            }
        }

        template <typename U>
        inline void load(const threadgroup U* src, int ld) {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::load(
                        frag_at(i, j), src, ld, 1, i * 16, j * 16);
                }
            }
        }

        template <typename U>
        inline void store_safe(
            threadgroup U* dst, int ld, short2 limits) const {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::store_safe(
                        frag_at(i, j), dst, ld, 1, limits.y, limits.x,
                        i * 16, j * 16);
                }
            }
        }

        template <typename U>
        inline void store_safe(device U* dst, int ld, short2 limits) const {
            for (short i = 0; i < kTileRows; ++i) {
                for (short j = 0; j < kTileCols; ++j) {
                    NAXFrag_t::store_safe(
                        frag_at(i, j), dst, ld, 1, limits.y, limits.x,
                        i * 16, j * 16);
                }
            }
        }
    };

    template <class CTile, class ATile, class BTile>
    inline void tile_matmad_nax(
        thread CTile& C, thread ATile& A, thread BTile& B) {
        static_assert(CTile::kTileRows == ATile::kTileRows);
        static_assert(CTile::kTileCols == BTile::kTileCols);
        static_assert(ATile::kTileCols == BTile::kTileRows);
        for (short mm = 0; mm < CTile::kTileRows; ++mm) {
            for (short nn = 0; nn < CTile::kTileCols; nn += 2) {
                for (short kk = 0; kk < ATile::kTileCols; ++kk) {
                    NAXFrag::mma(
                        C.frag_at(mm, nn), C.frag_at(mm, nn + 1),
                        A.frag_at(mm, kk), B.frag_at(kk, nn),
                        B.frag_at(kk, nn + 1));
                }
            }
        }
    }

    template <
        typename T, short BROWS, short BCOLS, short DST_LD,
        short REDUCTION_DIM, short TGP_SIZE, short GROUP_SIZE, short BITS>
    struct QuantizedBlockLoader {
        static_assert(BROWS == 64 && BCOLS == 64 && DST_LD == 72);
        static_assert(REDUCTION_DIM == 1 && TGP_SIZE == 64);
        static_assert(GROUP_SIZE == 64 && BITS == 4);

        const device uint8_t* src;
        const device T* scales;
        const device T* biases;
        threadgroup T* dst;
        const int src_ld;

        QuantizedBlockLoader(
            const device uint8_t* src_, const device T* scales_,
            const device T* biases_, int src_ld_, threadgroup T* dst_,
            uint simd_gid, uint simd_lid)
            : src(src_ + (simd_gid * 32u + simd_lid) * (src_ld_ / 2)),
              scales(scales_ + (simd_gid * 32u + simd_lid) * (src_ld_ / 64)),
              biases(biases_ + (simd_gid * 32u + simd_lid) * (src_ld_ / 64)),
              dst(dst_ + (simd_gid * 32u + simd_lid) * DST_LD),
              src_ld(src_ld_) {}

        inline void load_unsafe() const {
            const T scale = scales[0];
            const T bias = biases[0];
            for (short i = 0; i < 32; ++i) {
                const uint8_t packed = src[i];
                dst[2 * i] = static_cast<T>(scale * (packed & 0x0fu) + bias);
                dst[2 * i + 1] = static_cast<T>(
                    scale * ((packed >> 4) & 0x0fu) + bias);
            }
        }

        inline void next() {
            src += 32;
            scales += 1;
            biases += 1;
        }
    };

    template <typename T, typename InPtr, typename OutPtr>
    inline void flash_moe_qmm_t_nax(
        const device uint32_t* weights, const device T* scales,
        const device T* biases, InPtr input, OutPtr output,
        threadgroup T* ws, uint expert, uint weight_row, int weight_rows,
        int K, int output_ld, short valid_rows, uint simd_gid,
        uint simd_lid) {
        constexpr int BM = 16;
        constexpr int BN = 64;
        constexpr int BK = 64;
        constexpr int WM = 1;
        constexpr int WN = 2;
        constexpr int BK_PADDED = 72;
        const int K_PACKED_BYTES = K / 2;
        const int K_GROUPS = K / 64;

        const device uint8_t* weight_bytes =
            reinterpret_cast<const device uint8_t*>(weights)
            + (size_t(expert) * weight_rows + weight_row) * K_PACKED_BYTES;
        const device T* scale_rows =
            scales + (size_t(expert) * weight_rows + weight_row) * K_GROUPS;
        const device T* bias_rows =
            biases + (size_t(expert) * weight_rows + weight_row) * K_GROUPS;
        QuantizedBlockLoader<
            T, BN, BK, BK_PADDED, 1, WM * WN * 32, 64, 4>
            loader(weight_bytes, scale_rows, bias_rows, K, ws,
                simd_gid, simd_lid);

        constexpr short TM = BM / WM / 16;
        constexpr short TN = BN / WN / 16;
        const short tn = static_cast<short>(simd_gid % WN) * (BN / WN);
        NAXTile<float, TM, TN> d_tile;
        d_tile.clear();

        for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < BK; kk += 32) {
                NAXTile<T, TM, 2> a_tile;
                NAXTile<T, 2, TN> b_tile;
                a_tile.load_rows(input + k + kk, K, valid_rows);
                b_tile.load(ws + tn * BK_PADDED + kk, BK_PADDED);
                tile_matmad_nax(d_tile, a_tile, b_tile);
            }
            loader.next();
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        d_tile.store_safe(output + tn, output_ld, short2(BN / WN, valid_rows));
    }

    } // namespace steel
    } // namespace mlx

    """

private let switchFlashMoENAXKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_flash_moe_q4_g64_nax_m8_v1",
    inputNames: [
        "x", "gate_up_weight", "gate_up_scales", "gate_up_biases",
        "down_weight", "down_scales", "down_biases", "expert_offsets",
        "tile_offsets",
    ],
    outputNames: ["out"],
    source: """
        const uint tid = thread_index_in_threadgroup;
        const uint group = threadgroup_position_in_grid.x;
        threadgroup uint route[3];
        if (tid == 0u) {
            if (group >= tile_offsets[128]) {
                route[2] = 0u;
            } else {
                uint lo = 0u;
                uint hi = 128u;
                while (lo + 1u < hi) {
                    const uint mid = (lo + hi) / 2u;
                    if (tile_offsets[mid] <= group) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                const uint begin = expert_offsets[lo]
                    + (group - tile_offsets[lo]) * 8u;
                route[0] = lo;
                route[1] = begin;
                route[2] = min(8u, expert_offsets[lo + 1u] - begin);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (route[2] == 0u) {
            return;
        }

        const uint expert = route[0];
        const uint assignment = route[1];
        const short valid_rows = static_cast<short>(route[2]);
        threadgroup bfloat16_t h_local[8 * 704];
        threadgroup bfloat16_t ws[64 * 72];

        for (uint tile = 0u; tile < 11u; ++tile) {
            const uint base = tile * 64u;
            mlx::steel::flash_moe_qmm_t_nax(
                gate_up_weight, gate_up_scales, gate_up_biases,
                x + assignment * 2816u, h_local + base, ws,
                expert, base, 1408, 2816, 704, valid_rows,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            mlx::steel::flash_moe_qmm_t_nax(
                gate_up_weight, gate_up_scales, gate_up_biases,
                x + assignment * 2816u, ws, ws,
                expert, 704u + base, 1408, 2816, 64, valid_rows,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < uint(valid_rows) * 64u; i += 64u) {
                const uint row = i / 64u;
                const uint col = i % 64u;
                const float gate = static_cast<float>(h_local[row * 704u + base + col]);
                const float up = static_cast<float>(ws[row * 64u + col]);
                const float gate2 = gate * gate;
                const float gelu = 0.5f * gate * (1.0f + tanh(
                    sqrt(2.0f / 3.14159265358979323846f)
                        * (gate + 0.044715f * gate * gate2)));
                h_local[row * 704u + base + col] = static_cast<bfloat16_t>(gelu * up);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint tile = 0u; tile < 44u; ++tile) {
            const uint base = tile * 64u;
            mlx::steel::flash_moe_qmm_t_nax(
                down_weight, down_scales, down_biases, h_local,
                out + assignment * 2816u + base, ws,
                expert, base, 2816, 704, 2816, valid_rows,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
    header: switchFlashMoENAXHeader,
    ensureRowContiguous: true
)

// MARK: - NAX-DOWN-PREFILL: specialized expert down-projection only

/// NAX-DOWN-PREFILL. Replaces ONLY the large expert down projection
/// (`[M, 704] x [704, 2816] -> [M, 2816]`) on the sorted routed-expert
/// prefill plane with a specialized NAX kernel. Routing, sorting, the gate
/// and up projections, and GeGLU stay exactly as dispatched today; the
/// incumbent `downProj` gathered-QMM runs whenever this arm declines.
///
/// Tiling mirrors the fused kernel's down phase verbatim: one threadgroup
/// per eight-row expert tile (physical BM = 16, logical rows <= 8 masked via
/// `load_rows`/`store_safe`), BN = 64, BK = 64, WM = 1, WN = 2 (64 threads,
/// 2 SIMDgroups). The K loop (704, 11 steps) and the per-element MMA
/// accumulation order are unchanged from the incumbent gathered-QMM path,
/// and dequantization is the same affine group-64 nibble formula evaluated
/// in bfloat16, so the stored bf16 outputs match the incumbent's rounding
/// boundary.
///
/// Routing comes from the already-computed PREFILL-CSORT-128 scan
/// (`expert_offsets`/`tile_offsets`, 129 entries); no second sort or
/// regrouping is introduced. Decode (lhs-index route) and speculative
/// verification (no scan metadata, small rows) never qualify.
///
/// Kill switch: `DARKBLOOM_GEMMA4_PREFILL_NAX_DOWN=1` arms it; default OFF.
/// Engage mark: `prefill-nax-down`.
private let switchNaxDownPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_NAX_DOWN"]
    else { return false }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// One threadgroup owns one eight-row expert tile (found by binary search
/// over `tile_offsets`, exactly as the fused NAX MoE kernel routes) and
/// streams all 44 BN = 64 down-projection tiles for those rows. `h` is the
/// already-sorted GeGLU plane `[rows, 1, 704]`; `out` is `[rows, 1, 2816]`.
/// Threadgroup memory is only the dequant staging `ws` (64 x 72 bf16) plus
/// the 3-word route triple: 4608 * 2 + 12 = 9228 bytes, ~28% of the 32 KiB
/// budget the repository's attention kernels treat as the limit.
private let switchNaxDownPrefillKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_nax_down_m8_v1",
    inputNames: [
        "h", "down_weight", "down_scales", "down_biases", "expert_offsets",
        "tile_offsets",
    ],
    outputNames: ["out"],
    source: """
        const uint tid = thread_index_in_threadgroup;
        const uint group = threadgroup_position_in_grid.x;
        threadgroup uint route[3];
        if (tid == 0u) {
            if (group >= tile_offsets[128]) {
                route[2] = 0u;
            } else {
                uint lo = 0u;
                uint hi = 128u;
                while (lo + 1u < hi) {
                    const uint mid = (lo + hi) / 2u;
                    if (tile_offsets[mid] <= group) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                const uint begin = expert_offsets[lo]
                    + (group - tile_offsets[lo]) * 8u;
                route[0] = lo;
                route[1] = begin;
                route[2] = min(8u, expert_offsets[lo + 1u] - begin);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (route[2] == 0u) {
            return;
        }

        const uint expert = route[0];
        const uint assignment = route[1];
        const short valid_rows = static_cast<short>(route[2]);
        threadgroup bfloat16_t ws[64 * 72];

        for (uint tile = 0u; tile < 44u; ++tile) {
            const uint base = tile * 64u;
            mlx::steel::flash_moe_qmm_t_nax(
                down_weight, down_scales, down_biases, h + assignment * 704u,
                out + assignment * 2816u + base, ws,
                expert, base, 2816, 704, 2816, valid_rows,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """,
    header: switchFlashMoENAXHeader,
    ensureRowContiguous: true
)

/// Exact production down-projection contract only. Returns nil (fail closed
/// onto the incumbent `downProj` gathered-QMM) unless every geometric,
/// dtype, quantization and routing assumption holds.
private func switchNaxDownPrefill(
    activated: MLXArray,
    idx: MLXArray,
    route: RouteCsortPrefillResult?,
    down: QuantizedSwitchLinear?
) -> MLXArray? {
    guard switchNaxDownPrefillEnabled,
        let route,
        let down,
        // Sorted prefill plane only: `[rows, 1, 704]` bf16 with the sorted
        // expert key per row.
        activated.ndim == 3,
        activated.dim(1) == 1,
        activated.dim(2) == 704,
        activated.dtype == .bfloat16,
        // Mirror the host's sorted right-hand-side floor (at least
        // `4 * experts` rows): prefill rows qualify, the 64-assignment
        // decode cohort and small speculative rectangles do not.
        activated.dim(0) >= 512,
        idx.ndim == 1,
        idx.dtype == .uint32,
        idx.size == activated.dim(0),
        // Frozen target contract: affine Q4 group-64, no bias, exact packed
        // shapes. Never re-quantized, never re-represented.
        down.inputDims == 704,
        down.outputDims == 2816,
        down.numExperts == 128,
        down.groupSize == 64,
        down.bits == 4,
        down.mode == .affine,
        down.bias == nil,
        down.weight.shape == [128, 2816, 88],
        down.scales.shape == [128, 2816, 11],
        down.weight.dtype == .uint32,
        down.scales.dtype == .bfloat16
    else { return nil }
    guard let downBiases = down.biases,
        downBiases.shape == [128, 2816, 11],
        downBiases.dtype == .bfloat16,
        route.expertOffsets.shape == [129],
        route.tileOffsets.shape == [129],
        route.expertOffsets.dtype == .uint32,
        route.tileOffsets.dtype == .uint32
    else { return nil }
    CBv2EngageMark.once("prefill-nax-down")
    let rows = activated.dim(0)
    let groups = (rows + 7) / 8 + 128 - 1
    return switchNaxDownPrefillKernel(
        [
            activated, down.weight, down.scales, downBiases,
            route.expertOffsets, route.tileOffsets,
        ],
        grid: (groups * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[rows, 1, 2816]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - SwitchGLU

/// Semantic profile required by the exact Gemma direct-reduction experiment.
/// Generic SwitchGLU instances never infer production eligibility from a
/// one-point activation probe.
public enum SwitchGLUWeightedReductionProfile: Sendable {
    case generic
    case gemma4ProductionGeGLU
}

/// Inputs retained from the direct sorted expert reduction so a downstream
/// prefill kernel can consume the sorted rows without materializing the
/// intermediate `[tokens, hidden]` reduction.
public struct WeightedExpertUnsortCarrier {
    let sortedOutputs: MLXArray
    let inverseOrder: MLXArray
    let weights: MLXArray
}

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let weightedReductionProfile: SwitchGLUWeightedReductionProfile

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    /// GATEUP-FUSE-PREFILL: the primary gate|up storage bound at load (nil
    /// when the layer is not the exact production geometry or the arm is
    /// off), and its once-resolved dispatch contract.
    private var fusedGateUpStorage: SwitchGateUpFusedStorage?
    private var fusedGateUpResolved = false
    private var fusedGateUpContract: (groupSize: Int, bits: Int, mode: QuantizationMode)?

    /// Bind the concatenated gate|up storage whose slices the split
    /// projections were (or will be) loaded with. Load-time only.
    public func bindFusedGateUpStorage(_ storage: SwitchGateUpFusedStorage) {
        fusedGateUpStorage = storage
        fusedGateUpResolved = false
        fusedGateUpContract = nil
    }

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false,
        weightedReductionProfile: SwitchGLUWeightedReductionProfile = .generic
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        self.weightedReductionProfile = weightedReductionProfile
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2,
                numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false,
        weightedReductionProfile: SwitchGLUWeightedReductionProfile = .generic
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        self.weightedReductionProfile = weightedReductionProfile
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2,
                numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// GATEUP-FUSE-PREFILL: resolve (once) the quantization contract the
    /// fused storage is dispatched with. It is read from the bound split
    /// projections, which must be the exact production contract; anything
    /// else resolves to nil once and the split views are used forever after.
    private func fusedGateUpDispatch()
        -> (storage: SwitchGateUpFusedStorage, groupSize: Int, bits: Int, mode: QuantizationMode)?
    {
        guard let storage = fusedGateUpStorage else { return nil }
        if !fusedGateUpResolved {
            fusedGateUpResolved = true
            if gateUpProj == nil, hiddenDims == storage.hiddenDims,
                let gate = gateProj as? QuantizedSwitchLinear,
                let up = upProj as? QuantizedSwitchLinear,
                gate.groupSize == 64 && gate.bits == 4
                    && gate.mode == .affine && gate.bias == nil
                    && up.groupSize == 64 && up.bits == 4
                    && up.mode == .affine && up.bias == nil,
                gate.weight.shape == [128, 704, 352]
                    && up.weight.shape == [128, 704, 352]
                    && storage.weight.shape == [128, 1408, 352]
                    && storage.scales.shape == [128, 1408, 44]
                    && storage.biases.shape == [128, 1408, 44]
                    && storage.weight.dtype == .uint32
                    && storage.scales.dtype == .bfloat16
                    && storage.biases.dtype == .bfloat16
            {
                fusedGateUpContract = (gate.groupSize, gate.bits, gate.mode)
            }
        }
        guard let contract = fusedGateUpContract else { return nil }
        return (storage, contract.groupSize, contract.bits, contract.mode)
    }

    private func projectExperts(
        _ x: MLXArray, _ indices: MLXArray,
        sortedPlane: SwitchSortedPlaneProducer? = nil,
        routeTable: SwitchRouteTable? = nil
    ) -> (output: MLXArray, inverseOrder: MLXArray?, sorted: Bool) {
        let useLhsIndices =
            indices.size == 64 && indices.ndim == 2 && indices.shape == [8, 8]
            && x.ndim == 2 && x.shape == [8, inputDims]
        let useExpertPrefixBounds =
            expertPrefixBoundsEnabled && useLhsIndices
            && indices.dtype == .uint32 && x.dtype == .bfloat16
            && expertPrefixBoundsProjectionsEligible
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64

        var idx = indices
        // ROUTE-LAZY-INVERSE-ORDER: the sentinel `MLXArray()` this variable
        // used to hold is dead at every site -- under `doSort` a real producer
        // overwrites it, and without it the optional return already carries
        // nil. Holding the optional avoids one throwaway C++ array handle per
        // layer per forward.
        var inverseOrder: MLXArray?
        var lhsIndices: MLXArray?
        var flashRouteMetadata: RouteCsortPrefillResult?
        if doSort {
            if useLhsIndices {
                x = x.flattened(start: 0, end: -3)
                // GLUE-FOLD: an upstream producer already emitted the exact
                // route table beside the top-8 selection; consume it and the
                // standalone `mlx_lm_route_simd_rank_scatter` dispatch never
                // enters the chain. Any mismatch -- including the raw-key
                // contract (prefix-bounds tagging wants tagged keys) or a
                // disabled incumbent rank path -- re-issues the incumbent
                // chain, which produces byte-identical arrays.
                if let table = routeTable,
                    !useExpertPrefixBounds,
                    routeSimdRank64Enabled,
                    numExperts == 128,
                    table.rowOrder.dtype == .uint32,
                    table.rowOrder.ndim == 1, table.rowOrder.size == 64,
                    table.sortedKeys.dtype == .uint32,
                    table.sortedKeys.ndim == 1, table.sortedKeys.size == 64,
                    table.inverseOrder.dtype == .uint32,
                    table.inverseOrder.ndim == 1, table.inverseOrder.size == 64
                {
                    (lhsIndices, idx, inverseOrder) = (
                        table.rowOrder, table.sortedKeys, table.inverseOrder
                    )
                } else {
                    (lhsIndices, idx, inverseOrder) = gatherSortIndices(
                        indices: indices, numExperts: numExperts,
                        expertPrefixBounds: useExpertPrefixBounds)
                }
            } else if let sortedPlane {
                // PRENORM-GATHER: the producer writes the sorted plane from
                // the inverse order; `x` is only read if it declines.
                // FLASH-MOE-PREFILL and NAX-DOWN-PREFILL also need the scan's
                // expert/tile prefixes. Reuse that result rather than sorting
                // once for the public route tuple and again for the fused kernel.
                if switchFlashMoEPrefillEnabled || switchNaxDownPrefillEnabled,
                    let route = routeCountingSortPrefill(
                        indices.flattened(), m: indices.dim(-1), numExperts: numExperts)
                {
                    if let plane = sortedPlane(route.inverseOrder),
                        plane.ndim == 3, plane.dim(0) == indices.size,
                        plane.dim(1) == 1, plane.dim(2) == inputDims,
                        plane.dtype == x.dtype
                    {
                        x = plane
                    } else {
                        x = x.flattened(start: 0, end: -3)[route.rowOrder]
                    }
                    idx = route.sortedKeys
                    inverseOrder = route.inverseOrder
                    flashRouteMetadata = route
                } else {
                    let order = gatherSortOrder(indices: indices, numExperts: numExperts)
                    if let plane = sortedPlane(order.inverseOrder),
                        plane.ndim == 3, plane.dim(0) == indices.size,
                        plane.dim(1) == 1, plane.dim(2) == inputDims,
                        plane.dtype == x.dtype
                    {
                        x = plane
                    } else {
                        x = x.flattened(start: 0, end: -3)[order.rowOrder]
                    }
                    idx = order.sortedKeys
                    inverseOrder = order.inverseOrder
                }
            } else {
                (x, idx, inverseOrder) = gatherSort(
                    x: x, indices: indices, numExperts: numExperts)
            }
        }

        // FLASH-MOE-PREFILL: consume the already sorted expert plane directly.
        // The 3-D sorted plane is only supplied by the prefill pre-norm
        // producer; decode uses the lhs-index route-table form and therefore
        // cannot enter this branch. Any failed storage or metadata check falls
        // through to the existing gate/up -> GeGLU -> down sequence below.
        let flashMoEAdmission = switchFlashMoEPrefillEnabled
            && sortedPlane != nil
            && doSort
            && !useLhsIndices
            && lhsIndices == nil
            && inputDims == 2816
            && hiddenDims == 704
            && numExperts == 128
            && indices.ndim == 2
            && indices.dim(1) == 8
            && indices.dtype == .uint32
            && x.ndim == 3
            && x.dim(0) == indices.size
            && x.dim(1) == 1
            && x.dim(2) == 2816
            && x.dtype == .bfloat16
            && idx.ndim == 1
            && idx.size == x.dim(0)
            && idx.dtype == .uint32

        if flashMoEAdmission {
            let down = downProj as? QuantizedSwitchLinear
            if let fused = fusedGateUpDispatch(),
                let down,
                let downBiases = down.biases,
                down.inputDims == 704,
                down.outputDims == 2816,
                down.numExperts == 128,
                down.groupSize == 64,
                down.bits == 4,
                down.mode == .affine,
                down.bias == nil,
                down.weight.shape == [128, 2816, 88],
                down.scales.shape == [128, 2816, 11],
                downBiases.shape == [128, 2816, 11],
                down.weight.dtype == .uint32,
                down.scales.dtype == .bfloat16,
                downBiases.dtype == .bfloat16
            {
                if let route = flashRouteMetadata,
                    route.expertOffsets.shape == [numExperts + 1],
                    route.tileOffsets.shape == [numExperts + 1],
                    route.expertOffsets.dtype == .uint32,
                    route.tileOffsets.dtype == .uint32
                {
                    CBv2EngageMark.once("prefill-flash-moe")
                    let groups = (x.dim(0) + 7) / 8 + numExperts - 1
                    let output = switchFlashMoENAXKernel(
                        [
                            x, fused.storage.weight, fused.storage.scales,
                            fused.storage.biases, down.weight, down.scales, downBiases,
                            route.expertOffsets, route.tileOffsets,
                        ],
                        grid: (groups * 64, 1, 1),
                        threadGroup: (64, 1, 1),
                        outputShapes: [x.shape],
                        outputDTypes: [x.dtype]
                    )[0]
                    return (output, inverseOrder, true)
                }

                // Keep the original scalar prototype available when the route
                // scan cannot provide its internal metadata.
                CBv2EngageMark.once("prefill-flash-moe-scalar-fallback")
                let output = switchFlashMoEScalarKernel(
                    [
                        x, idx, fused.storage.weight, fused.storage.scales,
                        fused.storage.biases, down.weight, down.scales, downBiases,
                    ],
                    template: [("T", x.dtype)],
                    grid: (numExperts, 1, 1),
                    threadGroup: (64, 1, 1),
                    outputShapes: [x.shape],
                    outputDTypes: [x.dtype]
                )[0]
                return (output, inverseOrder, true)
            }
            CBv2EngageMark.once("prefill-flash-moe-fallback")
        }

        let xGate: MLXArray
        let xUp: MLXArray
        // PROMPT-GLUE (pg1): the routed-expert GeLU product computed straight
        // off the fused gate|up plane, in place of the strided-view closure.
        var promptActivated: MLXArray? = nil
        if let gateUpProj {
            let xGateUp = gateUpProj(
                x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            guard let gateProj, let upProj else {
                preconditionFailure("SwitchGLU requires gate_up_proj or gate_proj/up_proj")
            }
            // GATEUP-FUSE-PREFILL: the sorted right-hand-side plane (the
            // production prefill) reads its gathered activations once through
            // one gather over the concatenated gate|up storage. Same kernel
            // pipeline, same per-column K-chains; the halves are views. The
            // admission mirrors the host's sorted right-hand-side selection
            // exactly, so the split views never meet that kernel.
            if doSort, !useLhsIndices, lhsIndices == nil,
                x.ndim == 3, x.dim(-2) == 1, x.dim(-1) == inputDims,
                x.dim(0) >= 16, x.dim(0) / numExperts >= 4,
                x.dtype == .bfloat16,
                let fused = fusedGateUpDispatch()
            {
                CBv2EngageMark.once("prefill-gateup-fuse")
                let xGateUp = MLX.gatherQuantizedMM(
                    x,
                    fused.storage.weight,
                    scales: fused.storage.scales,
                    biases: fused.storage.biases,
                    lhsIndices: nil,
                    rhsIndices: idx,
                    transpose: true,
                    groupSize: fused.groupSize,
                    bits: fused.bits,
                    mode: fused.mode,
                    sortedIndices: true
                )
                // The specialized gathered-QMM epilogue stores the compact
                // [rows, 704] GeGLU plane in the first physical half of the
                // ordinary [rows, 1, 1408] output allocation.
                let activated = xGateUp.flattened()[..<(xGateUp.size / 2)]
                    .reshaped(x.dim(0), 1, hiddenDims)
                CBv2EngageMark.once("prefill-gateup-gelu-epilogue")
                // NAX-DOWN-PREFILL: the sorted prefill plane's down
                // projection only. Declines for decode, verification, and
                // any off-contract geometry, leaving the incumbent below.
                if let naxDown = switchNaxDownPrefill(
                    activated: activated, idx: idx, route: flashRouteMetadata,
                    down: downProj as? QuantizedSwitchLinear)
                {
                    return (naxDown, inverseOrder, true)
                }
                let downLhs: MLXArray? =
                    (idx.ndim == 1 && idx.size == 64) ? switchDownIdentity64 : nil
                x = downProj(
                    activated, idx, lhsIndices: downLhs, sortedIndices: true)
                return (x, inverseOrder, true)
            } else {
                xUp = upProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
                xGate = gateProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            }
        }

        let activated: MLXArray
        if let promptActivated {
            activated = promptActivated
        } else if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = geGLUProduct(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        // NAX-DOWN-PREFILL: the sorted prefill plane's down projection only.
        // Declines (nil) for decode, verification, and any off-contract
        // geometry, preserving the established output graph.
        if let naxDown = switchNaxDownPrefill(
            activated: activated, idx: idx, route: flashRouteMetadata,
            down: downProj as? QuantizedSwitchLinear)
        {
            return (naxDown, inverseOrder, true)
        }
        // DOWN-LHS-IDENTITY: at the sorted [64] geometry the down projection
        // gathers activation row `assignment` for assignment `assignment`;
        // hand it that identity table instead of leaving `lhsIndices` nil,
        // which otherwise materializes the same arange(64) on every call.
        let downLhs: MLXArray? =
            (doSort && idx.ndim == 1 && idx.size == 64) ? switchDownIdentity64 : nil
        x = downProj(activated, idx, lhsIndices: downLhs, sortedIndices: doSort)
        // Under `doSort` a producer above always assigned `inverseOrder`;
        // otherwise it is still nil, which is exactly what the old
        // `doSort ? inverseOrder : nil` produced.
        return (x, inverseOrder, doSort)
    }

    /// Cached eligibility: the projection tensors are bound at load time and
    /// the tag consumers defensively decode tagged words even on fallback
    /// paths, so evaluating the geometry once per module (not per forward,
    /// where its ~15 shape/dtype bridge reads per layer-round are measurable
    /// host cost) is safe.
    private lazy var expertPrefixBoundsProjectionsEligible: Bool =
        supportsExpertPrefixBoundsProjections()

    /// Exact production geometry whose three quantized gathers understand the
    /// tagged rhs-index carrier. Every other SwitchGLU, quantization, bias, or
    /// projection shape retains raw expert indices and the incumbent kernels.
    private func supportsExpertPrefixBoundsProjections() -> Bool {
        guard inputDims == 2816, hiddenDims == 704, numExperts == 128,
            gateUpProj == nil,
            let gate = gateProj as? QuantizedSwitchLinear,
            let up = upProj as? QuantizedSwitchLinear,
            let down = downProj as? QuantizedSwitchLinear
        else { return false }

        guard gate.inputDims == 2816 && gate.outputDims == 704
            && gate.numExperts == 128
            && gate.groupSize == 64 && gate.bits == 4
            && gate.mode == .affine && gate.bias == nil
            && up.inputDims == 2816 && up.outputDims == 704
            && up.numExperts == 128
            && up.groupSize == 64 && up.bits == 4
            && up.mode == .affine && up.bias == nil
        else { return false }

        guard down.inputDims == 704 && down.outputDims == 2816
            && down.numExperts == 128
            && down.groupSize == 64 && down.bits == 4
            && down.mode == .affine && down.bias == nil
        else { return false }

        guard let gateBiases = gate.biases, let upBiases = up.biases,
            let downBiases = down.biases,
            gate.weight.shape == [128, 704, 352]
                && gate.scales.shape == [128, 704, 44]
                && gateBiases.shape == [128, 704, 44]
                && up.weight.shape == [128, 704, 352]
                && up.scales.shape == [128, 704, 44]
                && upBiases.shape == [128, 704, 44]
                && down.weight.shape == [128, 2816, 88]
                && down.scales.shape == [128, 2816, 11]
                && downBiases.shape == [128, 2816, 11]
        else { return false }

        return gate.weight.dtype == .uint32
            && gate.scales.dtype == .bfloat16
            && gateBiases.dtype == .bfloat16
            && up.weight.dtype == .uint32
            && up.scales.dtype == .bfloat16
            && upBiases.dtype == .bfloat16
            && down.weight.dtype == .uint32
            && down.scales.dtype == .bfloat16
            && downBiases.dtype == .bfloat16
    }

    private func legacyWeightedReduction(
        _ projected: (output: MLXArray, inverseOrder: MLXArray?, sorted: Bool),
        indices: MLXArray,
        weights: MLXArray
    ) -> MLXArray {
        var output = projected.output
        if let inverseOrder = projected.inverseOrder {
            output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
        }
        return weightedExpertSum(MLX.squeezed(output, axis: -2), weights)
    }

    private func supportsWeightedExpertUnsort(
        _ x: MLXArray, _ indices: MLXArray, weights: MLXArray
    ) -> Bool {
        // Exact Gemma 4 26B-A4B production contract. The explicit semantic
        // profile keeps generic SwitchGLU/custom activations on the established
        // implementation.
        guard weightedReductionProfile == .gemma4ProductionGeGLU else { return false }
        return inputDims == 2816
            && hiddenDims == 704
            && numExperts == 128
            && gateUpProj == nil
            && activationProduct == nil
            && isGeluActivation
            && x.ndim == 2
            && x.dim(1) == 2816
            && x.dtype == .bfloat16
            && indices.ndim == 2
            && indices.dim(0) == x.dim(0)
            && indices.dim(1) == 8
            && indices.dtype == .uint32
            && weights.ndim == 2
            && weights.shape == indices.shape
            && weights.dtype == .bfloat16
            && indices.size >= 64
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> MLXArray {
        var projected = projectExperts(x, indices, sortedPlane: sortedPlane)
        if let inverseOrder = projected.inverseOrder {
            projected.output = scatterUnsort(
                x: projected.output, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(projected.output, axis: -2)
    }

    /// Preserve the promoted gathered down projection and defer only its
    /// inverse-permutation + weighted top-K reduction to a downstream consumer.
    ///
    /// This is decode-only and exact-geometry-only. Returning nil leaves
    /// ``callAndWeightedReduce`` as the complete established fallback.
    public func callAndDeferWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true,
        routeTable: SwitchRouteTable? = nil
    ) -> DeferredWeightedExpertRows? {
        let isEightRowDecode =
            !isProductionPrefill && x.dim(0) == 8 && indices.size == 64
        guard fuseSortedReduction && isEightRowDecode,
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else { return nil }

        let projected = projectExperts(x, indices, routeTable: routeTable)
        guard projected.sorted,
            let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == 2816,
            projected.output.dtype == .bfloat16
        else { return nil }

        weightedExpertUnsortProbe.recordEffective()
        return DeferredWeightedExpertRows(
            sortedOutputs: MLX.squeezed(projected.output, axis: -2),
            inverseOrder: inverseOrder,
            weights: weights)
    }

    /// Always-called expert projection + weighted reduction entry point.
    ///
    /// When the experiment is enabled, the exact sorted production Gemma
    /// prefill contract and the exact eight-row decode cohort reduce directly
    /// to `[tokens, hidden]`. Smaller decode cohorts, rectangular speculative
    /// verification, generic/custom-activation, dtype/layout, and near-geometry
    /// calls retain scatter/unsort followed by ``weightedExpertSum``.
    public func callAndWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true
    ) -> MLXArray {
        callAndWeightedReduceWithUnsortCarrier(
            x,
            indices,
            weights: weights,
            fuseSortedReduction: fuseSortedReduction,
            isProductionPrefill: isProductionPrefill
        ).output
    }

    /// The direct reduction plus its already-sorted inputs. Generic and decode
    /// paths return no carrier and preserve the established output graph.
    public func callAndWeightedReduceWithUnsortCarrier(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> (output: MLXArray, carrier: WeightedExpertUnsortCarrier?) {
        // At B=8 decode there are exactly 64 assignments (8 rows x top-k 8),
        // which is the sorting threshold and the minimum geometry accepted by
        // weightedExpertUnsort. Keep the decode gate exact so MTP rectangles
        // and smaller serving cohorts remain on their established reduction.
        let isEightRowDecode =
            !isProductionPrefill && x.dim(0) == 8 && indices.size == 64
        guard fuseSortedReduction && (isProductionPrefill || isEightRowDecode),
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else {
            return (
                weightedExpertSum(
                    callAsFunction(x, indices, sortedPlane: sortedPlane), weights),
                nil
            )
        }

        let projected = projectExperts(x, indices, sortedPlane: sortedPlane)
        guard projected.sorted,
            let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == 2816,
            projected.output.dtype == .bfloat16
        else {
            return (
                legacyWeightedReduction(projected, indices: indices, weights: weights),
                nil
            )
        }

        let sortedOutputs = MLX.squeezed(projected.output, axis: -2)
        let output = weightedExpertUnsort(
            sortedOutputs: sortedOutputs,
            inverseOrder: inverseOrder,
            weights: weights)
        let carrier =
            isProductionPrefill
            ? WeightedExpertUnsortCarrier(
                sortedOutputs: sortedOutputs,
                inverseOrder: inverseOrder,
                weights: weights)
            : nil
        return (output, carrier)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, lhsIndices: MLXArray? = nil,
        sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(
            x, weightT, lhsIndices: lhsIndices, rhsIndices: indices,
            sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, lhsIndices: MLXArray? = nil,
        sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            lhsIndices: lhsIndices,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
