import Foundation
import MLX
import MLXNN

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
    outputNames: ["block_offset"],
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
private func routeCountingSortPrefill(
    _ indices: MLXArray, m: Int, numExperts: Int
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray)? {
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
    let offsets = routeCsortPrefillScanKernel(
        [hist],
        template: [("NE", numExperts)],
        grid: (width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[blocks, width]],
        outputDTypes: [.uint32]
    )[0]
    let outputs = routeCsortPrefillScatterKernel(
        [indices, offsets],
        template: [("M", m)],
        grid: (blocks * width, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return (outputs[0], outputs[1], outputs[2])
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

// MARK: - GATEUP-FUSE-PREFILL: one gathered gate|up GEMM on the sorted prefill plane

/// GATEUP-FUSE-PREFILL. On the sorted routed-expert prefill plane the gate and
/// up projections are two `gather_qmm_rhs` dispatches (N = 704 each) over the
/// same gathered activations `[rows * topK, 1, 2816]` and the same sorted
/// expert keys; the activations (369 MB for the packed 8 x 1024 cohort) are
/// therefore streamed from DRAM twice per layer. This arm issues ONE
/// `gather_qmm_rhs` over a concatenated right-hand side (N = 1408: the gate
/// columns followed by the up columns) and hands the shaped GeGLU the two
/// column halves as strided views, so the activations are read once and one
/// dispatch per layer disappears.
///
/// Storage. The concatenated `[experts, 1408, packed-in]` weight and its
/// `[experts, 1408, groups]` scales and biases are the PRIMARY per-layer
/// storage, built once at load (``SwitchGateUpFusedStorage``, bound from the
/// model's sanitize pass). The module's `gate_proj` / `up_proj` parameters
/// become zero-copy slices of that storage: rows `0..<704` and `704..<1408`
/// of the output axis, each a contiguous `[704, packed-in]` matrix per
/// expert with an expert stride of 1408 rows. The gathered decode and
/// verification kernels address experts through the batch strides they are
/// handed and require only the per-expert matrix to be contiguous, so they
/// read the identical bytes through the views without a copy. Net resident
/// weight memory therefore equals the split layout's exactly; nothing is
/// duplicated.
///
/// Exactness: every output column of the gathered quantized GEMM owns an
/// independent K-chain -- the tile pipeline (bm/bn/bk, K-step order, per-group
/// affine dequant `scale * nibble + bias`) is fixed by (K, group size, bits)
/// and never by N -- and the concatenated weight, scales and biases are a pure
/// copy of the split ones along the output axis. 1408 keeps every alignment
/// predicate (32 and 64) that 704 satisfies, so the same pipeline is
/// selected. The consumer receives the identical gate / up bytes through views
/// with the identical shapes it saw before.
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
/// two split gathers. Engage mark: `prefill-gateup-fuse`.
public let switchGateUpFusePrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_GATEUP_FUSE"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// The concatenated `[gate ; up]` affine 4-bit right-hand side of one expert
/// layer plus the two zero-copy views the split projections are bound to. A
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
        // Output axis (axis 1 of [experts, out, packed-in]): gate rows
        // 0..<704 followed by up rows 704..<1408, for weight, scales and
        // biases alike. Pure copies of the loaded planes; the split
        // parameters below are slices sharing this storage, not copies.
        let weight = concatenated([gateWeight, upWeight], axis: 1)
        let scales = concatenated([gateScales, upScales], axis: 1)
        let biases = concatenated([gateBiases, upBiases], axis: 1)
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.gateWeight = weight[0..., ..<n]
        self.gateScales = scales[0..., ..<n]
        self.gateBiases = biases[0..., ..<n]
        self.upWeight = weight[0..., n...]
        self.upScales = scales[0..., n...]
        self.upBiases = biases[0..., n...]
        self.hiddenDims = n
    }
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
        let doSort = indices.size >= 64
        var x = (doSort && useLhsIndices)
            ? x.expandedDimensions(axis: 1)
            : MLX.expandedDimensions(x, axes: [-2, -3])

        var idx = indices
        var inverseOrder = MLXArray()
        var lhsIndices: MLXArray?
        if doSort {
            if useLhsIndices {
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
            } else {
                (x, idx, inverseOrder) = gatherSort(
                    x: x, indices: indices, numExperts: numExperts)
            }
        }

        let xGate: MLXArray
        let xUp: MLXArray
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
                xGate = xGateUp[.ellipsis, ..<hiddenDims]
                xUp = xGateUp[.ellipsis, hiddenDims...]
            } else {
                xUp = upProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
                xGate = gateProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            }
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = geGLUProduct(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)
        return (x, doSort ? inverseOrder : nil, doSort)
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
