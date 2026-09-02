import Foundation
import MLX
import MLXNN

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}
public struct WeightedExpertUnsortStats: Sendable, Equatable {
    public let effectiveCalls: Int
}

public struct WeightedExpertUnsortProvenance: Sendable, Equatable {
    public let requested: Bool
    public let effectiveCalls: Int

    public var engaged: Bool { effectiveCalls > 0 }
    public var missingExpectedEngagement: Bool { requested && !engaged }
}

private final class WeightedExpertUnsortProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var effectiveCalls = 0
    private var enabled = false

    @inline(__always)
    func recordEffective() {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
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

public func weightedExpertUnsortStats() -> WeightedExpertUnsortStats {
    weightedExpertUnsortProbe.snapshot()
}

public func weightedExpertUnsortProvenance(
    requested: Bool
) -> WeightedExpertUnsortProvenance {
    let stats = weightedExpertUnsortStats()
    return WeightedExpertUnsortProvenance(
        requested: requested,
        effectiveCalls: stats.effectiveCalls)
}

public func resetWeightedExpertUnsortStats() {
    weightedExpertUnsortProbe.reset()
}

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

public func resolveDeferredWeightedExpertRows(
    _ rows: DeferredWeightedExpertRows
) -> MLXArray {
    weightedExpertUnsort(
        sortedOutputs: rows.sortedOutputs,
        inverseOrder: rows.inverseOrder,
        weights: rows.weights)
}

// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

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

@inline(__always)
private func geGLUClaimsPinnedDecode(_ gate: MLXArray, _ up: MLXArray) -> Bool {
    guard switchGeluShapedFuseEnabled,
        gate.dtype == .bfloat16, up.dtype == .bfloat16,
        gate.shape == up.shape
    else { return false }
    let s = gate.shape
    if s.count == 3, s[0] == 64, s[1] == 1 { return true }
    if s.count == 2, s[0] == 64 { return true }
    if CBv2MTPWideVerifyContext.active {
        let rows = 64 * CBv2MTPWideVerifyContext.columns
        if s.count == 3, s[0] == rows, s[1] == 1 { return true }
        if s.count == 2, s[0] == rows { return true }
    }
    return false
}

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

public let shapedGeluPrefillShapeCap = 4

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

// MARK: - ROUTE-RANK-WIDE: exact wide-verify route table (128..512 assignments)

public enum SwitchRouteRankWide {
    nonisolated(unsafe) public static var enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_ROUTE_RANK_WIDE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    static let maxAssignments = 512

    static func admits(_ n: Int) -> Bool {
        enabled && n > routeSortTile64 && n <= maxAssignments && n % routeSortTile64 == 0
    }
}

private let routeRankWideKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_rank_wide_u32_v1",
    inputNames: ["indices"],
    outputNames: ["row_order", "sorted_keys", "inverse_order"],
    source: """
        const uint i = thread_position_in_grid.x;
        threadgroup uint tg_keys[N];
        const uint key = (uint)indices[i];
        tg_keys[i] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The stable counting sort's write position, computed directly:
        // keys below, then equal keys at a lower input index.
        uint rank = 0;
        uint before = 0;
        uint total = 0;
        for (uint j = 0; j < N; ++j) {
            const uint other = tg_keys[j];
            const bool same = other == key;
            rank += (other < key) || (same && j < i);
            before += same && j < i;
            total += same;
        }
        row_order[rank] = i / M;
        sorted_keys[rank] = BOUNDS
            ? (0x80000000u | key | (before << 8) | ((total - before - 1) << 14))
            : key;
        inverse_order[i] = rank;
    """,
    ensureRowContiguous: true
)

private func routeRankWide(
    _ indices: MLXArray, m: Int, expertPrefixBounds: Bool
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray)? {
    let n = indices.size
    guard SwitchRouteRankWide.admits(n), indices.dtype == .uint32 else { return nil }
    CBv2EngageMark.once("route-rank-wide")
    let outputs = routeRankWideKernel(
        [indices],
        template: [("N", n), ("M", m), ("BOUNDS", expertPrefixBounds)],
        grid: (n, 1, 1),
        threadGroup: (n, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - ROUTE-CSORT-64: fused counting-sort route table (donor port)

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

private let routeCsortPrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_ROUTE_CSORT_PREFILL"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let routeCsortPrefillBlock = 256
private let routeCsortPrefillWidth = 256
private let routeCsortPrefillMaxKeys = 1 << 28

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
    name: "mlx_lm_route_csort128_scan_v1",
    inputNames: ["block_hist"],
    outputNames: ["block_offset"],
    source: """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint e = thread_position_in_threadgroup.x;
        uint simd_id = e / 32;
        uint lane = e % 32;
        uint nblocks = (uint)block_hist_shape[0];
        uint total = 0u;
        for (uint b = 0; b < nblocks; ++b) {
            total += block_hist[b * WIDTH + e];
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
        for (uint b = 0; b < nblocks; ++b) {
            block_offset[b * WIDTH + e] = running;
            running += block_hist[b * WIDTH + e];
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

public typealias SwitchSortedPlaneProducer = (_ inverseOrder: MLXArray) -> MLXArray?

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
    if indices.ndim == 2, numExperts <= routeCountingSortKeyBound,
        let fused = routeRankWide(
            indices, m: indices.dim(-1), expertPrefixBounds: expertPrefixBounds)
    {
        return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
    }
    let m = indices.dim(-1)
    let indices = indices.flattened()
    routeCsortShapeLog.note {
        "gatherSortIndices n=\(indices.size) m=\(m) E=\(numExperts) "
            + "dtype=\(indices.dtype)"
    }
    if numExperts <= routeCountingSortKeyBound {
        if let fused = routeCountingSortPrefill(indices, m: m, numExperts: numExperts) {
            return (fused.rowOrder, fused.sortedKeys, fused.inverseOrder)
        }
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

public let switchGateUpFusePrefillEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_GEMMA4_PREFILL_GATEUP_FUSE"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

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

public enum SwitchGLUWeightedReductionProfile: Sendable {
    case generic
    case gemma4ProductionGeGLU
}

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
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let weightedReductionProfile: SwitchGLUWeightedReductionProfile

    let isSiluActivation: Bool
    let isGeluActivation: Bool

    private var fusedGateUpStorage: SwitchGateUpFusedStorage?
    private var fusedGateUpResolved = false
    private var fusedGateUpContract: (groupSize: Int, bits: Int, mode: QuantizationMode)?

    public func bindFusedGateUpStorage(_ storage: SwitchGateUpFusedStorage) {
        fusedGateUpStorage = storage
        fusedGateUpResolved = false
        fusedGateUpContract = nil
    }

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
            indices.ndim == 2 && indices.dim(1) == 8 && x.ndim == 2
            && x.shape == [indices.dim(0), inputDims]
            && (indices.size == 64
                || (CBv2MTPWideVerifyContext.active
                    && SwitchRouteRankWide.admits(indices.size)))
        let useExpertPrefixBounds =
            expertPrefixBoundsEnabled && useLhsIndices
            && indices.dtype == .uint32 && x.dtype == .bfloat16
            && expertPrefixBoundsProjectionsEligible
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()
        var lhsIndices: MLXArray?
        if doSort {
            if useLhsIndices {
                x = x.flattened(start: 0, end: -3)
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

    private lazy var expertPrefixBoundsProjectionsEligible: Bool =
        supportsExpertPrefixBoundsProjections()

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
        let isWideVerifyRows =
            !isProductionPrefill && CBv2MTPWideVerifyContext.active
            && x.dim(0) > 8 && x.dim(0) % 8 == 0 && indices.size == 8 * x.dim(0)
        guard fuseSortedReduction && (isEightRowDecode || isWideVerifyRows),
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else { return nil }

        if isWideVerifyRows, !CBv2MTPWideVerifyContext.mergedExpertGather {
            var sortedTiles: [MLXArray] = []
            var inverseTiles: [MLXArray] = []
            var weightTiles: [MLXArray] = []
            let rows = x.dim(0)
            for (tileIndex, start) in stride(from: 0, to: rows, by: 8).enumerated() {
                guard let tile = callAndDeferWeightedReduce(
                    x[start ..< (start + 8)], indices[start ..< (start + 8)],
                    weights: weights[start ..< (start + 8)],
                    fuseSortedReduction: fuseSortedReduction,
                    isProductionPrefill: isProductionPrefill, routeTable: nil)
                else { return nil }
                sortedTiles.append(tile.sortedOutputs)
                inverseTiles.append(tile.inverseOrder + UInt32(64 * tileIndex))
                weightTiles.append(tile.weights)
            }
            CBv2EngageMark.once("mtp-wide-verify-expert-tiles")
            return DeferredWeightedExpertRows(
                sortedOutputs: concatenated(sortedTiles, axis: 0),
                inverseOrder: concatenated(inverseTiles, axis: 0),
                weights: concatenated(weightTiles, axis: 0))
        }

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

    public func callAndWeightedReduceWithUnsortCarrier(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool,
        isProductionPrefill: Bool = true,
        sortedPlane: SwitchSortedPlaneProducer? = nil
    ) -> (output: MLXArray, carrier: WeightedExpertUnsortCarrier?) {
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
