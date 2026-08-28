// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFast

/// Generation-17-only packed affine-Q4 projection for Gemma 4's B8 decode
/// attention inputs. The original `uint32` weight storage is presented to MPP
/// as a non-owning `uint4b_format` tensor; no expanded or cached weight plane is
/// constructed.
///
/// This helper deliberately exposes a two-stage API. `prepare` computes the
/// affine input-group sums once for one layer input, and the Q/K/V projections
/// reuse that same prepared result. Every unsupported device, dtype, shape, or
/// quantization returns `nil`, leaving the caller on `QuantizedLinear`.
public enum Gemma4PackedInlineQ4MPP {
    private static let rows = 8
    private static let inputWidth = 2816
    private static let groupSize = 64
    private static let groups = inputWidth / groupSize
    private static let outputTile = 32
    private static let threadsPerThreadgroup = 32

    /// Process-level rollback switch. The candidate is enabled by default only
    /// on eligible hardware; any explicit false spelling restores stock QMV.
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PACKED_MPP_Q4"
        ] else { return true }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "no", "off": return false
        default: return true
        }
    }()

    /// Mirror vendored MLX's NAX availability floor for the ranked Mac: macOS
    /// 26.2 and Apple GPU generation 17 or newer. Parsing the numeric suffix
    /// handles names such as `applegpu_g17s` without confusing the trailing
    /// architecture letter for the generation.
    private static let hardwareEligible: Bool = {
        #if os(macOS)
        guard #available(macOS 26.2, *) else { return false }
        let architecture = MLX.GPU.deviceInfo().architecture.lowercased()
        guard let marker = architecture.range(of: "applegpu_g") else { return false }
        var generation = 0
        var sawDigit = false
        for byte in architecture[marker.upperBound...].utf8 {
            guard byte >= 48 && byte <= 57 else { break }
            sawDigit = true
            generation = generation * 10 + Int(byte - 48)
        }
        let minimumGeneration = architecture.last == "p" ? 18 : 17
        return sawDigit && generation >= minimumGeneration
        #else
        return false
        #endif
    }()

    /// Opaque layer-local activation state shared by Q/K/V.
    public struct PreparedInput {
        fileprivate let flatX: MLXArray
        fileprivate let xSums: MLXArray
    }

    /// Preserve stock affine-QMV's activation-dtype quirk inside each four-term
    /// addend. Unlike stock's lane-local bias products plus final `simd_sum`,
    /// MPP serializes the eight addends into one g64 sum, so this does not claim
    /// stock reduction-order identity.
    private static let xSumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_packed_mpp_q4_xsum_m8_k2816_v1",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: """
            constexpr uint M_ROWS = 8;
            constexpr uint K = 2816;
            constexpr uint GROUP = 64;
            constexpr uint N_GROUPS = K / GROUP;

            const uint cell = thread_position_in_grid.x;
            if (cell >= M_ROWS * N_GROUPS) return;

            const device bfloat* xp =
                x + (cell / N_GROUPS) * K + (cell % N_GROUPS) * GROUP;
            float sum = 0.0f;
            for (uint chunk = 0; chunk < GROUP / 8; ++chunk) {
                const uint i = chunk * 8;
                sum += xp[i + 0] + xp[i + 1] + xp[i + 2] + xp[i + 3];
                sum += xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
            }
            xSums[cell] = sum;
            """,
        ensureRowContiguous: true
    )

    /// One SIMD owns one N32 output tile and all eight cohort rows. `B` spans
    /// the complete logical KxN plane, so its first-dimension stride is the
    /// frozen target's exact K2816 row (1,408 packed bytes). Each group slice
    /// is a non-owning view into that original pointer.
    private static let projectionKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_packed_mpp_affine4_m8_k2816_n32_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: """
            constexpr uint M_ROWS = 8;
            constexpr uint K = 2816;
            constexpr uint GROUP = 64;
            constexpr uint N_TILE = 32;
            constexpr uint GROUPS = K / GROUP;

            const uint tile = threadgroup_position_in_grid.x;
            const uint n0 = tile * N_TILE;

            device bfloat* xBase = const_cast<device bfloat*>(x);
            device uchar* wBase = reinterpret_cast<device uchar*>(
                const_cast<device uint32_t*>(w));

            extents<int32_t, K, M_ROWS> aShape;
            extents<int32_t, K, N> bShape;
            tensor<device bfloat, extents<int32_t, K, M_ROWS>, tensor_inline>
                A(xBase, aShape);
            tensor<device uint4b_format, extents<int32_t, K, N>, tensor_inline>
                B(wBase, bShape);

            constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
                M_ROWS, N_TILE, GROUP, false, true, false);
            mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> op;

            auto a0 = A.slice(0, 0);
            auto b0 = B.slice(0, n0);
            auto accum = op.get_destination_cooperative_tensor<
                decltype(a0), decltype(b0), float>();
            #pragma unroll
            for (uint16_t e = 0; e < accum.get_capacity(); ++e) {
                accum[e] = 0.0f;
            }

            for (uint group = 0; group < GROUPS; ++group) {
                auto aTile = A.slice(group * GROUP, 0);
                auto bTile = B.slice(group * GROUP, n0);
                auto qdot = op.get_destination_cooperative_tensor<
                    decltype(aTile), decltype(bTile), float>();
                #pragma unroll
                for (uint16_t e = 0; e < qdot.get_capacity(); ++e) {
                    qdot[e] = 0.0f;
                }
                op.run(aTile, bTile, qdot);

                #pragma unroll
                for (uint16_t e = 0; e < qdot.get_capacity(); ++e) {
                    const auto indices = qdot.get_multidimensional_index(e);
                    const uint n = n0 + uint(indices[0]);
                    const uint row = uint(indices[1]);
                    const uint affine = n * GROUPS + group;
                    const float term = float(scales[affine]) * qdot[e]
                        + xSums[row * GROUPS + group] * float(biases[affine]);
                    accum[e] += term;
                }
            }

            #pragma unroll
            for (uint16_t e = 0; e < accum.get_capacity(); ++e) {
                const auto indices = accum.get_multidimensional_index(e);
                const uint n = n0 + uint(indices[0]);
                const uint row = uint(indices[1]);
                out[row * N + n] = bfloat(accum[e]);
            }
            """,
        header: "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>\n",
        ensureRowContiguous: true
    )

    /// Prepare one exact CBv2 B8 decode layer input. The hardware gate comes
    /// before either kernel property is referenced, so generation-15 hosts do
    /// not construct or evaluate MPP state.
    public static func prepare(x: MLXArray) -> PreparedInput? {
        guard enabled, hardwareEligible else { return nil }
        guard x.dtype == .bfloat16 else { return nil }
        guard x.ndim == 3, x.dim(0) == rows, x.dim(1) == 1,
            x.dim(2) == inputWidth, x.size == rows * inputWidth
        else { return nil }

        let flatX = x.reshaped([rows, inputWidth])
        let sumCells = rows * groups
        let sumThreads = 128
        let sumThreadgroups = (sumCells + sumThreads - 1) / sumThreads
        let xSums = xSumKernel(
            [flatX],
            grid: (sumThreadgroups * sumThreads, 1, 1),
            threadGroup: (sumThreads, 1, 1),
            outputShapes: [[sumCells]],
            outputDTypes: [.float32]
        )[0]
        return PreparedInput(flatX: flatX, xSums: xSums)
    }

    /// Apply one shape-eligible attention projection, returning `[8, N]`.
    /// The affine close is deliberately inside the group loop and the final
    /// output crosses one BF16 store boundary.
    public static func apply(
        prepared: PreparedInput,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        guard enabled, hardwareEligible else { return nil }
        guard let biases else { return nil }
        guard groupSize == Self.groupSize, bits == 4 else { return nil }
        guard weight.dtype == .uint32, scales.dtype == .bfloat16,
            biases.dtype == .bfloat16
        else { return nil }
        guard weight.ndim == 2, scales.ndim == 2, biases.ndim == 2 else {
            return nil
        }

        let n = weight.dim(0)
        switch n {
        case 1024, 2048, 4096, 8192: break
        default: return nil
        }
        guard weight.dim(1) == inputWidth * bits / 32 else { return nil }
        guard scales.dim(0) == n, biases.dim(0) == n,
            scales.dim(1) == groups, biases.dim(1) == groups
        else { return nil }
        guard n % outputTile == 0 else { return nil }

        return projectionKernel(
            [prepared.flatX, weight, scales, biases, prepared.xSums],
            template: [("N", n)],
            grid: ((n / outputTile) * threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
