// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFast

/// DMLP-MMA-001 --- one eight-row matrix-unit pass for Gemma 4's dense
/// affine-8 gate and up projections.
///
/// The production decode cell has two independent `[2112, 2816]` affine-8,
/// group-64 banks and one `[8, 1, 2816]` BF16 activation.  The established
/// quad-stream QMV reads each packed bank twice, once for cohort rows 0...3
/// and once for rows 4...7.  This kernel presents the eight cohort rows as an
/// 8-wide matrix tile, so every packed byte in each bank is read once.  Gate
/// and up are two planes of one launch and share one exact activation-sum
/// prepass.
///
/// This is deliberately a production-cell specialization, not a general
/// QuantizedLinear replacement.  Any mismatch in shape, layout metadata,
/// dtype, quantization mode, module bias, or environment gate returns `nil`;
/// the caller then retains the exact tight-grid + shared-xsum implementation.
public enum CBv2DenseMLPGateUpMMAV1 {
    private static let batch = 8
    private static let sequence = 1
    private static let inputWidth = 2816
    private static let outputWidth = 2112
    private static let groupSize = 64
    private static let bits = 8
    private static let outputsPerThreadgroup = 32
    private static let threadsPerThreadgroup = 128

    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DENSE_MLP_MMA"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }()

    public struct Outputs {
        public let gate: MLXArray
        public let up: MLXArray
    }

    /// Stock affine-8's bias term uses a float accumulator fed by four
    /// ascending BF16 values.  Compute the corresponding sum once per
    /// `(cohort row, affine group)`; both projection banks consume this table.
    private static let activationSumKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_b8_l1_dense_gateup_mma_affine8_xsum_v1",
            inputNames: ["x"],
            outputNames: ["xSums"],
            source: """
                constexpr uint M_ROWS = 8;
                constexpr uint GROUP = 64;
                constexpr uint N_GROUPS = K / GROUP;
                const uint cell = thread_position_in_grid.x;
                if (cell >= M_ROWS * N_GROUPS) return;

                const device T* xp =
                    x + (cell / N_GROUPS) * K
                      + (cell % N_GROUPS) * GROUP;
                float sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < GROUP; i += 4) {
                    sum += xp[i + 0];
                    sum += xp[i + 1];
                    sum += xp[i + 2];
                    sum += xp[i + 3];
                }
                xSums[cell] = sum;
                """,
            ensureRowContiguous: true)

    /// Each threadgroup owns 32 outputs from one bank (four simdgroups, eight
    /// output rows apiece).  A simdgroup accumulates the transposed product
    /// `W[8, K] * X^T[K, 8]`, so its result fragment carries all eight cohort
    /// rows for each packed weight row.  BF16 operands are lossless here:
    /// activations are already BF16 and every affine-8 code in 0...255 is
    /// exactly representable in BF16.  Accumulation and scale/bias closure are
    /// FP32; only the final outputs narrow to BF16, matching QuantizedLinear.
    private static let projectionKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_b8_l1_dense_gateup_mma_affine8_g64_v1",
            inputNames: [
                "x",
                "gateW", "gateScales", "gateBiases",
                "upW", "upScales", "upBiases",
                "xSums",
            ],
            outputNames: ["gateOut", "upOut"],
            source: """
                constexpr uint M_ROWS = 8;
                constexpr uint GROUP = 64;
                constexpr uint N_SG = 4;
                constexpr uint N_PSG = 8;
                constexpr uint X_STRIDE = 9;
                constexpr uint N_GROUPS = K / GROUP;
                constexpr uint W_ROW_U32 = K / 4;
                constexpr uint G_ROW = K / GROUP;

                const uint lid = thread_position_in_threadgroup.x;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint lane = thread_index_in_simdgroup;
                const uint tg = threadgroup_position_in_grid.x;
                const bool useUp = threadgroup_position_in_grid.y != 0;

                threadgroup T Xs[GROUP * X_STRIDE];

                const uint n0 = tg * (N_SG * N_PSG);
                const uint sgN0 = n0 + sg * N_PSG;

                const device uint32_t* selectedW = useUp ? upW : gateW;
                const device T* selectedScales =
                    useUp ? upScales : gateScales;
                const device T* selectedBiases =
                    useUp ? upBiases : gateBiases;
                device T* selectedOut = useUp ? upOut : gateOut;

                // The simdgroup matrix fragment exposes two adjacent
                // row-major elements per lane.  `fragmentRow` selects one of
                // eight output rows and `fragmentCol` one of four adjacent
                // activation pairs (0/1, 2/3, 4/5, 6/7).
                const uint fragmentRow =
                    ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
                const uint fragmentCol =
                    ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
                const uint wordSide = fragmentCol >> 2;
                const uint byteInWord = fragmentCol & 3u;

                const device uint32_t* fragmentWRow =
                    selectedW + (sgN0 + fragmentRow) * W_ROW_U32;
                const device T* scaleRow =
                    selectedScales
                    + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
                const device T* biasRow =
                    selectedBiases
                    + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;

                simdgroup_matrix<float, 8, 8> acc =
                    simdgroup_matrix<float, 8, 8>(0.0f);

                for (uint g = 0; g < N_GROUPS; ++g) {
                    // One lane needs one of the two uint32 words for each
                    // eight-value K slice.  Adjacent fragment lanes request
                    // the same word, so the device coalescer merges them; the
                    // whole 64-byte affine group is fetched once per output
                    // row, not once per cohort half.
                    uint packedWords[8];
                    #pragma clang loop unroll(full)
                    for (uint t = 0; t < 8; ++t) {
                        packedWords[t] = fragmentWRow[
                            g * (GROUP / 4) + t * 2 + wordSide];
                    }

                    const float laneScale =
                        lane < N_PSG ? float(scaleRow[g]) : 0.0f;
                    const float laneBias =
                        lane < N_PSG ? float(biasRow[g]) : 0.0f;

                    // Start the read before the barrier protecting Xs, then
                    // publish the four register-held values after it.
                    const uint activationJ = lid / 2;
                    const uint activationM0 = (lid % 2) * 4;
                    const device T* activationPtr =
                        x + activationM0 * K + g * GROUP + activationJ;
                    const T activation0 = activationPtr[0 * K];
                    const T activation1 = activationPtr[1 * K];
                    const T activation2 = activationPtr[2 * K];
                    const T activation3 = activationPtr[3 * K];

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    threadgroup T* activationDst =
                        Xs + activationJ * X_STRIDE + activationM0;
                    activationDst[0] = activation0;
                    activationDst[1] = activation1;
                    activationDst[2] = activation2;
                    activationDst[3] = activation3;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<float, 8, 8> groupAcc =
                        simdgroup_matrix<float, 8, 8>(0.0f);
                    #pragma clang loop unroll(full)
                    for (uint t = 0; t < 8; ++t) {
                        simdgroup_matrix<T, 8, 8> Wm;
                        simdgroup_matrix<T, 8, 8> Xm;
                        const uint packed = packedWords[t];
                        Wm.thread_elements()[0] = T(float(
                            (packed >> (8 * byteInWord)) & 0xFFu));
                        Wm.thread_elements()[1] = T(float(
                            (packed >> (8 * (byteInWord + 1))) & 0xFFu));
                        simdgroup_load(
                            Xm, Xs + t * 8 * X_STRIDE, X_STRIDE);
                        simdgroup_multiply_accumulate(
                            groupAcc, Wm, Xm, groupAcc);
                    }

                    const float rowScale =
                        simd_shuffle(laneScale, ushort(fragmentRow));
                    const float rowBias =
                        simd_shuffle(laneBias, ushort(fragmentRow));
                    acc.thread_elements()[0] = metal::fma(
                        rowScale, groupAcc.thread_elements()[0],
                        acc.thread_elements()[0]);
                    acc.thread_elements()[1] = metal::fma(
                        rowScale, groupAcc.thread_elements()[1],
                        acc.thread_elements()[1]);
                    acc.thread_elements()[0] = metal::fma(
                        rowBias,
                        xSums[fragmentCol * N_GROUPS + g],
                        acc.thread_elements()[0]);
                    acc.thread_elements()[1] = metal::fma(
                        rowBias,
                        xSums[(fragmentCol + 1) * N_GROUPS + g],
                        acc.thread_elements()[1]);
                }

                const uint outputN = sgN0 + fragmentRow;
                selectedOut[fragmentCol * N + outputN] =
                    T(acc.thread_elements()[0]);
                selectedOut[(fragmentCol + 1) * N + outputN] =
                    T(acc.thread_elements()[1]);
                """,
            header: "#include <metal_simdgroup_matrix>\n",
            ensureRowContiguous: true)

    /// Returns both projections, or `nil` unless this is exactly the ranked
    /// Gemma 4 dense decode cell.
    public static func apply(
        x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        gateBiases: MLXArray?,
        gateGroupSize: Int,
        gateBits: Int,
        gateMode: QuantizationMode,
        upWeight: MLXArray,
        upScales: MLXArray,
        upBiases: MLXArray?,
        upGroupSize: Int,
        upBits: Int,
        upMode: QuantizationMode
    ) -> Outputs? {
        guard enabled,
            gateGroupSize == groupSize,
            upGroupSize == groupSize,
            gateBits == bits,
            upBits == bits,
            gateMode == .affine,
            upMode == .affine,
            let gateBiases,
            let upBiases,
            x.dtype == .bfloat16,
            gateWeight.dtype == .uint32,
            upWeight.dtype == .uint32,
            gateScales.dtype == .bfloat16,
            gateBiases.dtype == .bfloat16,
            upScales.dtype == .bfloat16,
            upBiases.dtype == .bfloat16,
            x.ndim == 3,
            x.shape == [batch, sequence, inputWidth],
            x.size == batch * sequence * inputWidth,
            gateWeight.shape == [outputWidth, inputWidth * bits / 32],
            upWeight.shape == gateWeight.shape,
            gateScales.shape == [outputWidth, inputWidth / groupSize],
            gateBiases.shape == gateScales.shape,
            upScales.shape == gateScales.shape,
            upBiases.shape == gateScales.shape
        else { return nil }

        let flatX = x.reshaped([batch, inputWidth])
        let sumCells = batch * (inputWidth / groupSize)
        let sumThreads = 128
        let sumThreadgroups = (sumCells + sumThreads - 1) / sumThreads
        let xSums = activationSumKernel(
            [flatX],
            template: [("T", x.dtype), ("K", inputWidth)],
            grid: (sumThreadgroups * sumThreads, 1, 1),
            threadGroup: (sumThreads, 1, 1),
            outputShapes: [[sumCells]],
            outputDTypes: [.float32]
        )[0]

        let threadgroupsPerPlane = outputWidth / outputsPerThreadgroup
        let projected = projectionKernel(
            [
                flatX,
                gateWeight, gateScales, gateBiases,
                upWeight, upScales, upBiases,
                xSums,
            ],
            template: [
                ("T", x.dtype),
                ("K", inputWidth),
                ("N", outputWidth),
            ],
            grid: (
                threadgroupsPerPlane * threadsPerThreadgroup,
                2,
                1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [
                [batch, sequence, outputWidth],
                [batch, sequence, outputWidth],
            ],
            outputDTypes: [x.dtype, x.dtype]
        )
        CBv2EngageMark.once("gemma4-dense-gateup-mma-v1")
        return Outputs(gate: projected[0], up: projected[1])
    }
}
