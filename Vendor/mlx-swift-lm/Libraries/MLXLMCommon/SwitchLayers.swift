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
    name: "weighted_expert_unsort",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
        uint feature = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;

        T accumulator = (T)0;
        const uint assignment_base = token * (uint)K;
        for (uint slot = 0; slot < (uint)K; ++slot) {
            const uint assignment = assignment_base + slot;
            const uint sorted_row = (uint)inverse_order[assignment];
            // Preserve the legacy bfloat16 multiply-then-reduce rounding.
            const T weighted = (T)(
                (float)sorted_outputs[sorted_row * threads_per_grid.x + feature]
                * (float)weights[assignment]);
            accumulator = accumulator + weighted;
        }
        output[token * threads_per_grid.x + feature] = accumulator;
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
        grid: (2816, tokens, 1),
        threadGroup: (64, 4, 1),
        outputShapes: [[tokens, 2816]],
        outputDTypes: [.bfloat16]
    )[0]
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

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

/// Single-dispatch stable rank-sort for the exact B=8 decode assignment plane
/// (64 keys over the 128-expert alphabet). Each thread computes its
/// assignment's stable rank directly — strictly-smaller keys plus earlier
/// equal keys — which is the definition the two-argSort pipeline below
/// realizes, so all three outputs are bit-identical to it (stability
/// included). One 64-thread dispatch replaces two bitonic argSort pipelines,
/// a floor-divide, and a gather on every MoE layer of every decode step.
private let gemma4GatherSort64Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "gemma4_gather_sort_64",
    inputNames: ["indices"],
    outputNames: ["lhs_indices", "sorted_indices", "inverse_order"],
    source: """
        const uint assignment = thread_position_in_grid.x;
        const uint key = (uint)indices[assignment];
        uint rank = 0;
        for (uint other = 0; other < 64; ++other) {
            const uint other_key = (uint)indices[other];
            rank += (uint)((other_key < key)
                || (other_key == key && other < assignment));
        }
        lhs_indices[rank] = assignment / 8;
        sorted_indices[rank] = key;
        inverse_order[assignment] = rank;
    """,
    ensureRowContiguous: true
)

public func gatherSortIndices(indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    // Exact ranked decode plane only: [8, 8] uint32. Every other shape,
    // dtype, or width keeps the established sort pipeline.
    if indices.ndim == 2, indices.shape == [8, 8], indices.dtype == .uint32 {
        let outs = gemma4GatherSort64Kernel(
            [indices],
            grid: (64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[64], [64], [64]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
        return (outs[0], outs[1], outs[2])
    }
    let indices = indices.flattened()
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

// MARK: - SwitchGLU

/// Semantic profile required by the exact Gemma direct-reduction experiment.
/// Generic SwitchGLU instances never infer production eligibility from a
/// one-point activation probe.
public enum SwitchGLUWeightedReductionProfile: Sendable {
    case generic
    case gemma4ProductionGeGLU
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

    private func projectExperts(
        _ x: MLXArray, _ indices: MLXArray
    ) -> (output: MLXArray, inverseOrder: MLXArray?, sorted: Bool) {
        let useLhsIndices =
            indices.size == 64 && indices.ndim == 2 && indices.shape == [8, 8]
            && x.ndim == 2 && x.shape == [8, inputDims]
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()
        var lhsIndices: MLXArray?
        if doSort {
            if useLhsIndices {
                x = x.flattened(start: 0, end: -3)
                (lhsIndices, idx, inverseOrder) = gatherSortIndices(indices: indices)
            } else {
                (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
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
            xUp = upProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
            xGate = gateProj(x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = compiledGeGLU(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)
        return (x, doSort ? inverseOrder : nil, doSort)
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

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var projected = projectExperts(x, indices)
        if let inverseOrder = projected.inverseOrder {
            projected.output = scatterUnsort(
                x: projected.output, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(projected.output, axis: -2)
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
        // At B=8 decode there are exactly 64 assignments (8 rows x top-k 8),
        // which is the sorting threshold and the minimum geometry accepted by
        // weightedExpertUnsort.  Keep the decode gate exact so MTP rectangles
        // and smaller serving cohorts remain on their established reduction.
        let isEightRowDecode =
            !isProductionPrefill && x.dim(0) == 8 && indices.size == 64
        guard fuseSortedReduction && (isProductionPrefill || isEightRowDecode),
            supportsWeightedExpertUnsort(x, indices, weights: weights)
        else {
            return weightedExpertSum(callAsFunction(x, indices), weights)
        }

        let projected = projectExperts(x, indices)
        guard projected.sorted,
            let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == 2816,
            projected.output.dtype == .bfloat16
        else {
            return legacyWeightedReduction(projected, indices: indices, weights: weights)
        }

        return weightedExpertUnsort(
            sortedOutputs: MLX.squeezed(projected.output, axis: -2),
            inverseOrder: inverseOrder,
            weights: weights)
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
