// GEMMA-DOWN-001: decode-only fusion of the routed expert down projection
// with the established inverse-permutation + weighted expert reduction.
//
// The production Gemma 4 decode cohort has eight tokens, eight selected
// experts per token, and therefore 64 sorted expert assignments.  The stock
// graph computes an affine-4/group-64 QMV for every assignment, writes a
// [64, 2816] BF16 tensor, reads that tensor back through the inverse sorting
// permutation, multiplies by BF16 router weights, and reduces slots 0...7.
// This kernel keeps those 64 projected rows in threadgroup memory one
// eight-feature tile at a time and writes only the final [8, 2816] result.
//
// Arithmetic boundaries are intentionally unchanged:
//
//   * each affine QMV uses stock `load_vector`/`qdot` operation order, FP32
//     accumulation, and `simd_sum`;
//   * each projected expert element is cast to BF16 before routing;
//   * inverse-order lookup restores the original top-k slots;
//   * router multiplication is rounded to BF16;
//   * slots are accumulated sequentially from 0 through 7 in BF16.
//
// Sorted duplicate-expert runs retain the promoted gather-QMV pairing rule:
// adjacent rows are paired from the start of each run and share each packed
// weight load; an odd final row uses the scalar stock qdot.  The grid has one
// 256-thread group per eight output features, so no global [64, 2816]
// intermediate is materialized.

import Foundation
import MLX
import MLXFast

/// Pure, value-only contract used to pin the optimized dispatch in tests.
/// Runtime array dtype/layout checks are additive and remain fail-closed.
public struct Gemma4RoutedDownReduceContract: Sendable, Equatable {
    public let productionGeGLUProfile: Bool
    public let hasSeparateGateAndUp: Bool
    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let batchSize: Int
    public let sequenceLength: Int
    public let topK: Int
    public let quantizationBits: Int
    public let quantizationGroupSize: Int
    public let affineQuantization: Bool
    public let hasProjectionBias: Bool
    public let hasAffineBiases: Bool

    public init(
        productionGeGLUProfile: Bool,
        hasSeparateGateAndUp: Bool,
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        batchSize: Int,
        sequenceLength: Int,
        topK: Int,
        quantizationBits: Int,
        quantizationGroupSize: Int,
        affineQuantization: Bool,
        hasProjectionBias: Bool,
        hasAffineBiases: Bool
    ) {
        self.productionGeGLUProfile = productionGeGLUProfile
        self.hasSeparateGateAndUp = hasSeparateGateAndUp
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.topK = topK
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.affineQuantization = affineQuantization
        self.hasProjectionBias = hasProjectionBias
        self.hasAffineBiases = hasAffineBiases
    }
}

/// Exact production geometry/representation selector.  This is deliberately
/// pure so near-geometry and representation fallbacks can be exhaustively
/// unit-tested without allocating the production expert bank.
public func gemma4RoutedDownReduceEligible(
    _ contract: Gemma4RoutedDownReduceContract
) -> Bool {
    contract.productionGeGLUProfile
        && contract.hasSeparateGateAndUp
        && contract.inputDims == 2816
        && contract.hiddenDims == 704
        && contract.numExperts == 128
        && contract.batchSize == 8
        && contract.sequenceLength == 1
        && contract.topK == 8
        && contract.quantizationBits == 4
        && contract.quantizationGroupSize == 64
        && contract.affineQuantization
        && !contract.hasProjectionBias
        && contract.hasAffineBiases
}

/// Default-on request flag.  Setting
/// `DARKBLOOM_GEMMA4_ROUTED_DOWN_REDUCE=0` (or false/no/off) restores the
/// established gathered down-QMV followed by weighted expert unsort.
public func gemma4RoutedDownReduceFlag(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

internal let gemma4RoutedDownReduceRequested = gemma4RoutedDownReduceFlag(
    ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_ROUTED_DOWN_REDUCE"])

private let gemma4RoutedDownReduceHeader = """
#define SIMD_SIZE 32

// Verbatim bits == 4 behavior of MLX quantized.h's load_vector.  In
// particular, `sum` receives the source-T four-term expression before its
// conversion to float, matching the affine bias term used by stock QMV.
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 4, "Gemma routed-down supports affine 4-bit only");
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

// Verbatim bits == 4 arm and close of quantized.h's qdot.
template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 4, "Gemma routed-down supports affine 4-bit only");
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// Verbatim promoted duplicate-run pair arithmetic from quantized.h.  Each
// result retains its independent scalar accumulation and K-loop order.
template <typename U, int values_per_thread>
inline void qdot_affine4_pair(
    const device uint8_t* w,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  U accum0 = 0;
  U accum1 = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    const uint16_t packed = ws[i];
    accum0 +=
        (x0[4 * i] * (packed & 0x000f) +
         x0[4 * i + 1] * (packed & 0x00f0) +
         x0[4 * i + 2] * (packed & 0x0f00) +
         x0[4 * i + 3] * (packed & 0xf000));
    accum1 +=
        (x1[4 * i] * (packed & 0x000f) +
         x1[4 * i + 1] * (packed & 0x00f0) +
         x1[4 * i + 2] * (packed & 0x0f00) +
         x1[4 * i + 3] * (packed & 0xf000));
  }
  out0 = scale * accum0 + sum0 * bias;
  out1 = scale * accum1 + sum1 * bias;
}
"""

private let gemma4RoutedDownReduceKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "gemma4_affine4_g64_routed_down_reduce_b8_v1",
    inputNames: [
        "activated", "down_weight", "down_scales", "down_biases",
        "sorted_indices", "inverse_order", "router_weights",
    ],
    outputNames: ["output"],
    source: """
        constexpr uint simd_width = 32;
        constexpr uint simdgroups = 8;
        constexpr int values_per_thread = 8;
        constexpr int block_size = values_per_thread * int(simd_width);
        constexpr uint outputs_per_tile = 8;
        constexpr uint packed_row_bytes = K / 2;
        constexpr uint affine_groups_per_row = K / 64;
        constexpr uint packed_expert_bytes = OUTN * packed_row_bytes;
        constexpr uint affine_expert_values = OUTN * affine_groups_per_row;

        const uint lid = thread_position_in_threadgroup.x;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint output_base = threadgroup_position_in_grid.x * outputs_per_tile;

        // Split each sorted equal-expert run into adjacent pairs followed by
        // an optional singleton, exactly like affine_gather_qmv's promoted
        // run-offset rule.  One thread builds at most 64 tiny task records.
        threadgroup ushort task_starts[64];
        threadgroup uchar task_is_pair[64];
        threadgroup uint task_count;
        threadgroup T projected[64 * outputs_per_tile];
        if (lid == 0) {
            uint assignment = 0;
            uint count = 0;
            while (assignment < ASSIGNMENTS) {
                const bool pair =
                    assignment + 1 < ASSIGNMENTS &&
                    sorted_indices[assignment] == sorted_indices[assignment + 1];
                task_starts[count] = ushort(assignment);
                task_is_pair[count] = pair ? uchar(1) : uchar(0);
                assignment += pair ? 2 : 1;
                count++;
            }
            task_count = count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Eight simdgroups drain the pair/single task stream.  A simdgroup
        // computes all eight rows of this output tile; stock uses two
        // simdgroups by four rows, but each row keeps the same independent
        // qdot/K-loop/simd_sum sequence.
        for (uint task = simd_gid; task < task_count; task += simdgroups) {
            const uint assignment0 = uint(task_starts[task]);
            const bool pair = task_is_pair[task] != 0;
            // Keep every formed device pointer inside the activation plane.
            // A singleton never reads x1, but using its own row avoids forming
            // an out-of-range pointer when the last task starts at row 63.
            const uint assignment1 = pair ? assignment0 + 1 : assignment0;
            const uint expert = uint(sorted_indices[assignment0]);

            const device T* x0 =
                activated + assignment0 * K + lane * values_per_thread;
            const device T* x1 =
                activated + assignment1 * K + lane * values_per_thread;
            const device uint8_t* weights =
                (const device uint8_t*)down_weight +
                expert * packed_expert_bytes +
                output_base * packed_row_bytes + lane * 4;
            const device T* scales =
                down_scales + expert * affine_expert_values +
                output_base * affine_groups_per_row + lane / 8;
            const device T* biases =
                down_biases + expert * affine_expert_values +
                output_base * affine_groups_per_row + lane / 8;

            thread float x0_thread[values_per_thread];
            thread float x1_thread[values_per_thread];
            thread float result0[outputs_per_tile] = {0};
            thread float result1[outputs_per_tile] = {0};

            int k = 0;
            for (; k <= K - block_size; k += block_size) {
                const float sum0 =
                    load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
                if (pair) {
                    const float sum1 =
                        load_vector<T, float, values_per_thread, 4>(x1, x1_thread);
                    for (uint row = 0; row < outputs_per_tile; ++row) {
                        const device uint8_t* row_weights =
                            weights + row * packed_row_bytes;
                        const device T* row_scales =
                            scales + row * affine_groups_per_row;
                        const device T* row_biases =
                            biases + row * affine_groups_per_row;
                        float dot0;
                        float dot1;
                        qdot_affine4_pair<float, values_per_thread>(
                            row_weights, x0_thread, x1_thread,
                            row_scales[0], row_biases[0], sum0, sum1,
                            dot0, dot1);
                        result0[row] += dot0;
                        result1[row] += dot1;
                    }
                } else {
                    for (uint row = 0; row < outputs_per_tile; ++row) {
                        const device uint8_t* row_weights =
                            weights + row * packed_row_bytes;
                        const device T* row_scales =
                            scales + row * affine_groups_per_row;
                        const device T* row_biases =
                            biases + row * affine_groups_per_row;
                        result0[row] += qdot<float, values_per_thread, 4>(
                            row_weights, x0_thread,
                            row_scales[0], row_biases[0], sum0);
                    }
                }
                weights += block_size / 2;
                scales += block_size / 64;
                biases += block_size / 64;
                x0 += block_size;
                x1 += block_size;
            }

            // K is group-64 aligned.  Every active tail lane owns one whole
            // eight-value packet, exactly the promoted gather-QMV tail.
            const uint active_tail_lanes = uint((K - k) / values_per_thread);
            if (lane < active_tail_lanes) {
                const float sum0 =
                    load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
                if (pair) {
                    const float sum1 =
                        load_vector<T, float, values_per_thread, 4>(x1, x1_thread);
                    for (uint row = 0; row < outputs_per_tile; ++row) {
                        const device uint8_t* row_weights =
                            weights + row * packed_row_bytes;
                        const device T* row_scales =
                            scales + row * affine_groups_per_row;
                        const device T* row_biases =
                            biases + row * affine_groups_per_row;
                        float dot0;
                        float dot1;
                        qdot_affine4_pair<float, values_per_thread>(
                            row_weights, x0_thread, x1_thread,
                            row_scales[0], row_biases[0], sum0, sum1,
                            dot0, dot1);
                        result0[row] += dot0;
                        result1[row] += dot1;
                    }
                } else {
                    for (uint row = 0; row < outputs_per_tile; ++row) {
                        const device uint8_t* row_weights =
                            weights + row * packed_row_bytes;
                        const device T* row_scales =
                            scales + row * affine_groups_per_row;
                        const device T* row_biases =
                            biases + row * affine_groups_per_row;
                        result0[row] += qdot<float, values_per_thread, 4>(
                            row_weights, x0_thread,
                            row_scales[0], row_biases[0], sum0);
                    }
                }
            }

            for (uint row = 0; row < outputs_per_tile; ++row) {
                result0[row] = simd_sum(result0[row]);
                if (pair) {
                    result1[row] = simd_sum(result1[row]);
                }
                if (lane == 0) {
                    projected[assignment0 * outputs_per_tile + row] =
                        static_cast<T>(result0[row]);
                    if (pair) {
                        projected[assignment1 * outputs_per_tile + row] =
                            static_cast<T>(result1[row]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One simdgroup owns one original token.  Its first eight lanes own
        // the tile's eight output features and visit original slots 0...7.
        if (simd_gid < TOKENS && lane < outputs_per_tile) {
            T accumulator = static_cast<T>(0);
            const uint assignment_base = simd_gid * TOPK;
            for (uint slot = 0; slot < TOPK; ++slot) {
                const uint assignment = assignment_base + slot;
                const uint sorted_row = uint(inverse_order[assignment]);
                const T weighted = static_cast<T>(
                    static_cast<float>(
                        projected[sorted_row * outputs_per_tile + lane]) *
                    static_cast<float>(router_weights[assignment]));
                accumulator = accumulator + weighted;
            }
            output[simd_gid * OUTN + output_base + lane] = accumulator;
        }
        """,
    header: gemma4RoutedDownReduceHeader,
    ensureRowContiguous: true
)

/// Generic kernel seam used by the exact production wrapper and focused
/// arithmetic tests.  `SwitchGLU` exposes it to inference only through the
/// production contract above.
public func gemma4RoutedDownReduceAffine4(
    activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    downBiases: MLXArray,
    sortedIndices: MLXArray,
    inverseOrder: MLXArray,
    routerWeights: MLXArray,
    tokenCount: Int,
    topK: Int,
    inputWidth: Int,
    outputWidth: Int
) -> MLXArray {
    let assignments = tokenCount * topK
    precondition(tokenCount > 0 && tokenCount <= 8)
    precondition(topK > 0 && topK <= 8)
    precondition(assignments <= 64)
    precondition(inputWidth >= 64 && inputWidth.isMultiple(of: 64))
    precondition(outputWidth > 0 && outputWidth.isMultiple(of: 8))
    precondition(
        activated.dtype == .bfloat16 && activated.size == assignments * inputWidth)
    precondition(
        downWeight.dtype == .uint32 && downWeight.ndim == 3
            && downWeight.dim(1) == outputWidth
            && downWeight.dim(2) == inputWidth * 4 / 32)
    precondition(
        downScales.dtype == .bfloat16 && downScales.ndim == 3
            && downScales.dim(0) == downWeight.dim(0)
            && downScales.dim(1) == outputWidth
            && downScales.dim(2) == inputWidth / 64)
    precondition(downBiases.dtype == .bfloat16 && downBiases.shape == downScales.shape)
    precondition(
        sortedIndices.dtype == .uint32 && sortedIndices.size == assignments)
    precondition(inverseOrder.dtype == .uint32 && inverseOrder.size == assignments)
    precondition(
        routerWeights.dtype == .bfloat16 && routerWeights.size == assignments)

    return gemma4RoutedDownReduceKernel(
        [
            activated.reshaped(assignments, inputWidth),
            downWeight, downScales, downBiases,
            sortedIndices.flattened(), inverseOrder.flattened(),
            routerWeights.reshaped(tokenCount, topK),
        ],
        template: [
            ("T", activated.dtype),
            ("K", inputWidth),
            ("OUTN", outputWidth),
            ("TOKENS", tokenCount),
            ("TOPK", topK),
            ("ASSIGNMENTS", assignments),
        ],
        grid: ((outputWidth / 8) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[tokenCount, outputWidth]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Exact production primitive.  `SwitchGLU` performs the semantic/profile and
/// module-type gate before calling this array-level shape/dtype gate.
public func gemma4RoutedDownReduce(
    activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    downBiases: MLXArray,
    sortedIndices: MLXArray,
    inverseOrder: MLXArray,
    routerWeights: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16 && activated.size == 64 * 704)
    precondition(downWeight.shape == [128, 2816, 88] && downWeight.dtype == .uint32)
    precondition(downScales.shape == [128, 2816, 11] && downScales.dtype == .bfloat16)
    precondition(downBiases.shape == downScales.shape && downBiases.dtype == .bfloat16)
    precondition(sortedIndices.dtype == .uint32 && sortedIndices.size == 64)
    precondition(inverseOrder.dtype == .uint32 && inverseOrder.size == 64)
    precondition(routerWeights.shape == [8, 8] && routerWeights.dtype == .bfloat16)

    return gemma4RoutedDownReduceAffine4(
        activated: activated,
        downWeight: downWeight,
        downScales: downScales,
        downBiases: downBiases,
        sortedIndices: sortedIndices,
        inverseOrder: inverseOrder,
        routerWeights: routerWeights,
        tokenCount: 8,
        topK: 8,
        inputWidth: 704,
        outputWidth: 2816)
}
