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

/// Process-start selector for the exact Gemma 4 routed gate/up fusion.
/// Default-on keeps the optimized production path active; `0` provides a
/// same-binary control without changing any fallback graph.
private let gemma4MoEGateUpFusionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXFAST_GEMMA4_MOE_GATEUP_FUSION"] != "0"
    && MLXHardwareInfo.isCompiledDecodeSupported
    && Device.defaultDevice().deviceType == .gpu

/// Exact-shape sequential gate projection + up projection + GeGLU kernel for
/// Gemma 4's batch-8 routed experts.  It consumes the existing Q4/g64 affine
/// arrays directly: no weight repacking, re-quantization, or representation
/// change is involved.
private let gemma4MoEGateUpFusionKernel = MLXFast.metalKernel(
    name: "gemma4_moe_gate_up_geglu_fusion",
    inputNames: [
        "gate_weights", "gate_scales", "gate_biases",
        "up_weights", "up_scales", "up_biases",
        "x", "lhs_indices", "rhs_indices",
    ],
    outputNames: ["activated"],
    source: """
        const uint assignment = threadgroup_position_in_grid.z;
        const uint expert = rhs_indices[assignment];

        uint run_offset = 0;
        for (uint prior = assignment; prior > 0; --prior) {
            if (rhs_indices[prior - 1] != expert) {
                break;
            }
            run_offset++;
        }

        // Match affine_gather_qmv's sorted-run pairing exactly.  The preceding
        // even-position threadgroup produces odd members of an expert run.
        if ((run_offset & 1u) != 0u) {
            return;
        }

        const bool has_pair =
            assignment + 1u < 64u && rhs_indices[assignment + 1u] == expert;
        const uint out_row = threadgroup_position_in_grid.y * 8u
            + simdgroup_index_in_threadgroup * 4u;
        const uint expert_weight_offset = expert * 704u * 352u;
        const uint expert_parameter_offset = expert * 704u * 44u;

        const device uint32_t* gate_w = gate_weights + expert_weight_offset;
        const device T* gate_s = gate_scales + expert_parameter_offset;
        const device T* gate_b = gate_biases + expert_parameter_offset;
        const device uint32_t* up_w = up_weights + expert_weight_offset;
        const device T* up_s = up_scales + expert_parameter_offset;
        const device T* up_b = up_biases + expert_parameter_offset;

        const uint lhs0 = lhs_indices[assignment];
        const device T* x0 = x + lhs0 * 2816u;

        thread T gate0[4];
        thread T gate1[4];
        thread T up0[4];
        thread T up1[4];

        if (has_pair) {
            const uint lhs1 = lhs_indices[assignment + 1u];
            const device T* x1 = x + lhs1 * 2816u;
            gemma4_project_affine4_pair<T>(
                gate_w, gate_s, gate_b, x0, x1, out_row,
                thread_index_in_simdgroup, gate0, gate1);
            gemma4_project_affine4_pair<T>(
                up_w, up_s, up_b, x0, x1, out_row,
                thread_index_in_simdgroup, up0, up1);

            if (thread_index_in_simdgroup == 0u) {
                for (uint row = 0; row < 4u; ++row) {
                    activated[assignment * 704u + out_row + row] =
                        gemma4_exact_geglu<T>(gate0[row], up0[row]);
                    activated[(assignment + 1u) * 704u + out_row + row] =
                        gemma4_exact_geglu<T>(gate1[row], up1[row]);
                }
            }
        } else {
            gemma4_project_affine4_single<T>(
                gate_w, gate_s, gate_b, x0, out_row,
                thread_index_in_simdgroup, gate0);
            gemma4_project_affine4_single<T>(
                up_w, up_s, up_b, x0, out_row,
                thread_index_in_simdgroup, up0);

            if (thread_index_in_simdgroup == 0u) {
                for (uint row = 0; row < 4u; ++row) {
                    activated[assignment * 704u + out_row + row] =
                        gemma4_exact_geglu<T>(gate0[row], up0[row]);
                }
            }
        }
    """,
    header: """
        #include <metal_simdgroup>
        #include <metal_stdlib>

        using namespace metal;

        template <typename T>
        inline float gemma4_load_affine4_x(
            const device T* x,
            thread float* x_thread) {
            float sum = 0.0f;
            for (int i = 0; i < 8; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                x_thread[i] = x[i];
                x_thread[i + 1] = x[i + 1] / 16.0f;
                x_thread[i + 2] = x[i + 2] / 256.0f;
                x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
            return sum;
        }

        inline float gemma4_affine4_dot_single(
            const device uint8_t* weights,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
            float accum = 0.0f;
            const device uint16_t* packed =
                (const device uint16_t*)weights;
            for (int i = 0; i < 2; ++i) {
                accum +=
                    (x_thread[4 * i] * (packed[i] & 0x000f) +
                     x_thread[4 * i + 1] * (packed[i] & 0x00f0) +
                     x_thread[4 * i + 2] * (packed[i] & 0x0f00) +
                     x_thread[4 * i + 3] * (packed[i] & 0xf000));
            }
            return scale * accum + sum * bias;
        }

        inline void gemma4_affine4_dot_pair(
            const device uint8_t* weights,
            const thread float* x0,
            const thread float* x1,
            float scale,
            float bias,
            float sum0,
            float sum1,
            thread float& out0,
            thread float& out1) {
            float accum0 = 0.0f;
            float accum1 = 0.0f;
            const device uint16_t* packed =
                (const device uint16_t*)weights;
            for (int i = 0; i < 2; ++i) {
                const uint16_t value = packed[i];
                accum0 +=
                    (x0[4 * i] * (value & 0x000f) +
                     x0[4 * i + 1] * (value & 0x00f0) +
                     x0[4 * i + 2] * (value & 0x0f00) +
                     x0[4 * i + 3] * (value & 0xf000));
                accum1 +=
                    (x1[4 * i] * (value & 0x000f) +
                     x1[4 * i + 1] * (value & 0x00f0) +
                     x1[4 * i + 2] * (value & 0x0f00) +
                     x1[4 * i + 3] * (value & 0xf000));
            }
            out0 = scale * accum0 + sum0 * bias;
            out1 = scale * accum1 + sum1 * bias;
        }

        template <typename T>
        inline void gemma4_project_affine4_pair(
            const device uint32_t* weights,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            uint out_row,
            uint simd_lane,
            thread T* output0,
            thread T* output1) {
            constexpr uint packed_row_bytes = 1408u;
            constexpr uint parameter_row = 44u;
            constexpr uint block = 256u;

            const device uint8_t* weight_bytes =
                (const device uint8_t*)weights
                + out_row * packed_row_bytes + simd_lane * 4u;
            const device T* scale =
                scales + out_row * parameter_row + simd_lane / 8u;
            const device T* bias =
                biases + out_row * parameter_row + simd_lane / 8u;
            x0 += simd_lane * 8u;
            x1 += simd_lane * 8u;

            thread float result0[4] = {0.0f};
            thread float result1[4] = {0.0f};
            thread float x0_thread[8];
            thread float x1_thread[8];

            for (uint k = 0u; k < 2816u; k += block) {
                const float sum0 = gemma4_load_affine4_x<T>(x0 + k, x0_thread);
                const float sum1 = gemma4_load_affine4_x<T>(x1 + k, x1_thread);
                for (uint row = 0u; row < 4u; ++row) {
                    float dot0;
                    float dot1;
                    gemma4_affine4_dot_pair(
                        weight_bytes + row * packed_row_bytes + k / 2u,
                        x0_thread, x1_thread,
                        scale[row * parameter_row + k / 64u],
                        bias[row * parameter_row + k / 64u],
                        sum0, sum1, dot0, dot1);
                    result0[row] += dot0;
                    result1[row] += dot1;
                }
            }

            for (uint row = 0u; row < 4u; ++row) {
                result0[row] = simd_sum(result0[row]);
                result1[row] = simd_sum(result1[row]);
                if (simd_lane == 0u) {
                    output0[row] = static_cast<T>(result0[row]);
                    output1[row] = static_cast<T>(result1[row]);
                }
            }
        }

        template <typename T>
        inline void gemma4_project_affine4_single(
            const device uint32_t* weights,
            const device T* scales,
            const device T* biases,
            const device T* x,
            uint out_row,
            uint simd_lane,
            thread T* output) {
            constexpr uint packed_row_bytes = 1408u;
            constexpr uint parameter_row = 44u;
            constexpr uint block = 256u;

            const device uint8_t* weight_bytes =
                (const device uint8_t*)weights
                + out_row * packed_row_bytes + simd_lane * 4u;
            const device T* scale =
                scales + out_row * parameter_row + simd_lane / 8u;
            const device T* bias =
                biases + out_row * parameter_row + simd_lane / 8u;
            x += simd_lane * 8u;

            thread float result[4] = {0.0f};
            thread float x_thread[8];

            for (uint k = 0u; k < 2816u; k += block) {
                const float sum = gemma4_load_affine4_x<T>(x + k, x_thread);
                for (uint row = 0u; row < 4u; ++row) {
                    result[row] += gemma4_affine4_dot_single(
                        weight_bytes + row * packed_row_bytes + k / 2u,
                        x_thread,
                        scale[row * parameter_row + k / 64u],
                        bias[row * parameter_row + k / 64u],
                        sum);
                }
            }

            for (uint row = 0u; row < 4u; ++row) {
                result[row] = simd_sum(result[row]);
                if (simd_lane == 0u) {
                    output[row] = static_cast<T>(result[row]);
                }
            }
        }

        // Preserve the compiled Gemma graph's bfloat16 boundary after every
        // primitive, including the two projection outputs.
        template <typename T>
        inline T gemma4_exact_geglu(T gate, T up) {
            T left = static_cast<T>(0.5f) * gate;
            T cubic = static_cast<T>(0.044715f) * gate;
            cubic = cubic * gate;
            cubic = cubic * gate;
            T inner = gate + cubic;
            inner = static_cast<T>(0.7978845608028654f) * inner;
            inner = static_cast<T>(metal::precise::tanh(inner));
            inner = static_cast<T>(1.0f) + inner;
            T activated = left * inner;
            return static_cast<T>(activated * up);
        }
    """,
    ensureRowContiguous: true
)

/// Internal so the opt-in exact-shape test exercises the production kernel,
/// rather than a copied stand-in. Callers must establish the eligibility
/// contract before invoking it.
func gemma4MoEGateUpFusion(
    gateWeights: MLXArray,
    gateScales: MLXArray,
    gateBiases: MLXArray,
    upWeights: MLXArray,
    upScales: MLXArray,
    upBiases: MLXArray,
    x: MLXArray,
    lhsIndices: MLXArray,
    rhsIndices: MLXArray
) -> MLXArray {
    gemma4MoEGateUpFusionKernel(
        [
            gateWeights, gateScales, gateBiases,
            upWeights, upScales, upBiases,
            x, lhsIndices, rhsIndices,
        ],
        template: [("T", DType.bfloat16)],
        grid: (64, 88, 64),
        threadGroup: (64, 1, 1),
        outputShapes: [[64, 1, 704]],
        outputDTypes: [.bfloat16]
    )[0]
}

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

public func gatherSortIndices(indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
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

    /// Return the fused activation only for the exact Gemma 4 batch-8 decode
    /// contract. Every other semantic profile, shape, dtype, quantization, or
    /// module configuration falls through to the established projection graph.
    private func fusedGemma4GateUpActivation(
        _ x: MLXArray,
        _ indices: MLXArray,
        lhsIndices: MLXArray?,
        sortedIndices: Bool
    ) -> MLXArray? {
        guard gemma4MoEGateUpFusionEnabled,
            weightedReductionProfile == .gemma4ProductionGeGLU,
            inputDims == 2816,
            hiddenDims == 704,
            numExperts == 128,
            gateUpProj == nil,
            activationProduct == nil,
            isGeluActivation,
            sortedIndices,
            x.shape == [8, 1, 2816],
            x.dtype == .bfloat16,
            indices.shape == [64],
            indices.dtype == .uint32,
            let lhsIndices,
            lhsIndices.shape == [64],
            lhsIndices.dtype == .uint32,
            let gate = gateProj as? QuantizedSwitchLinear,
            let up = upProj as? QuantizedSwitchLinear,
            gate.inputDims == 2816,
            gate.outputDims == 704,
            gate.numExperts == 128,
            up.inputDims == 2816,
            up.outputDims == 704,
            up.numExperts == 128,
            gate.groupSize == 64,
            gate.bits == 4,
            gate.mode == .affine,
            up.groupSize == 64,
            up.bits == 4,
            up.mode == .affine,
            gate.bias == nil,
            up.bias == nil,
            gate.weight.shape == [128, 704, 352],
            gate.weight.dtype == .uint32,
            gate.scales.shape == [128, 704, 44],
            gate.scales.dtype == .bfloat16,
            let gateBiases = gate.biases,
            gateBiases.shape == [128, 704, 44],
            gateBiases.dtype == .bfloat16,
            up.weight.shape == [128, 704, 352],
            up.weight.dtype == .uint32,
            up.scales.shape == [128, 704, 44],
            up.scales.dtype == .bfloat16,
            let upBiases = up.biases,
            upBiases.shape == [128, 704, 44],
            upBiases.dtype == .bfloat16
        else { return nil }

        // MLXFast's ensureRowContiguous contract preserves the narrow kernel's
        // flat-addressing requirement without materializing or re-representing
        // an already contiguous production array.
        return gemma4MoEGateUpFusion(
            gateWeights: gate.weight,
            gateScales: gate.scales,
            gateBiases: gateBiases,
            upWeights: up.weight,
            upScales: up.scales,
            upBiases: upBiases,
            x: x,
            lhsIndices: lhsIndices,
            rhsIndices: indices)
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

        if let activated = fusedGemma4GateUpActivation(
            x, idx, lhsIndices: lhsIndices, sortedIndices: doSort)
        {
            x = downProj(activated, idx, sortedIndices: doSort)
            return (x, inverseOrder, true)
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
