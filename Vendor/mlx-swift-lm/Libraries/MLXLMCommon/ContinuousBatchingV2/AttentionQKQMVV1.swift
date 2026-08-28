// Exact B=8, L=1 attention Q/K affine-4 QMV with one shared activation-sum
// table. The output-row arithmetic is the promoted quad-stream body; only the
// grid and the source of its affine activation sum change.

import Foundation
import MLX
import MLXFast

public enum CBv2AttentionQKQMVV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ATTN_QK_QMV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    public static let activationSumsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_ATTN_QK_XSUM"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let sequence = 1
    private static let inputWidth = 2816
    private static let groupSize = 64
    private static let bits = 4
    private static let rowsPerGroup = 4
    private static let simdWidth = 32
    private static let simdGroups = 4
    private static let resultsPerSIMDGroup = 4
    private static let outputsPerGroup = simdGroups * resultsPerSIMDGroup
    private static let valuesPerLane = 8
    private static let kBlock = simdWidth * valuesPerLane

    /// Opaque by construction: a consumer can only receive a table produced
    /// from an exact production-shape activation tensor.
    public struct ActivationSums {
        fileprivate let values: MLXArray
    }

    /// Derive the candidate from the currently promoted tied-head replica.
    /// The string surgery is deliberately guarded by exact occurrence counts:
    /// a future helper drift fails at process start instead of silently
    /// compiling a near-match.
    private static let kernelHeader: String = {
        var result = CBv2TiedLMHeadQMVV1.kernelHeader

        func replaceOnce(_ old: String, with new: String) {
            precondition(
                result.components(separatedBy: old).count == 2,
                "attention Q/K kernel transform marker drifted")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            "constexpr int num_simdgroups = 2;",
            with: "constexpr int num_simdgroups = 4;")

        replaceOnce(
            """
            template <typename T, const int group_size, const int bits>
            METAL_FUNC void qmv_affine4_g64_quad_stream_impl(
            """,
            with: """
            template <typename T, typename U, int values_per_thread>
            inline void load_affine4_values(
                const device T* x,
                thread U* x_thread) {
              for (int i = 0; i < values_per_thread; i += 4) {
                x_thread[i] = x[i];
                x_thread[i + 1] = x[i + 1] / 16.0f;
                x_thread[i + 2] = x[i + 2] / 256.0f;
                x_thread[i + 3] = x[i + 3] / 4096.0f;
              }
            }

            template <typename T, const int group_size, const int bits>
            METAL_FUNC void qmv_affine4_g64_quad_stream_xsum_impl(
            """)

        replaceOnce(
            """
                const device T* biases,
                const device T* x0,
            """,
            with: """
                const device T* biases,
                const device float* x_sums,
                const device T* x0,
            """)

        replaceOnce(
            """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
            """,
            with: """
              const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                  simd_gid * results_per_simdgroup;
              const int first_m = int(tid.x) * 4;
            """)

        let fullLoads: [(String, String)] = [
            (
                "float sum = load_vector<T, float, values_per_thread, 4>(x0, x_thread);",
                """
                load_affine4_values<T, float, values_per_thread>(x0, x_thread);
                float sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m];
                """),
            (
                "sum = load_vector<T, float, values_per_thread, 4>(x1, x_thread);",
                """
                load_affine4_values<T, float, values_per_thread>(x1, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 1];
                """),
            (
                "sum = load_vector<T, float, values_per_thread, 4>(x2, x_thread);",
                """
                load_affine4_values<T, float, values_per_thread>(x2, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 2];
                """),
            (
                "sum = load_vector<T, float, values_per_thread, 4>(x3, x_thread);",
                """
                load_affine4_values<T, float, values_per_thread>(x3, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 3];
                """),
        ]
        for (old, new) in fullLoads {
            replaceOnce(old, with: new)
        }

        let tailLoads: [(String, String)] = [
            (
                """
                float sum =
                        load_vector_safe<T, float, values_per_thread, 4>(x0, x_thread, remaining);
                """,
                """
                load_affine4_values<T, float, values_per_thread>(x0, x_thread);
                float sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m];
                """),
            (
                """
                sum =
                        load_vector_safe<T, float, values_per_thread, 4>(x1, x_thread, remaining);
                """,
                """
                load_affine4_values<T, float, values_per_thread>(x1, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 1];
                """),
            (
                """
                sum =
                        load_vector_safe<T, float, values_per_thread, 4>(x2, x_thread, remaining);
                """,
                """
                load_affine4_values<T, float, values_per_thread>(x2, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 2];
                """),
            (
                """
                sum =
                        load_vector_safe<T, float, values_per_thread, 4>(x3, x_thread, remaining);
                """,
                """
                load_affine4_values<T, float, values_per_thread>(x3, x_thread);
                sum = x_sums[
                    ((k / block_size) * SIMD_SIZE + int(simd_lid)) * 8 + first_m + 3];
                """),
        ]
        for (old, new) in tailLoads {
            replaceOnce(old, with: new)
        }

        return result
    }()

    /// One float per (256-value K block, SIMD lane, cohort row). The two
    /// four-term additions and the float accumulation reproduce affine-4
    /// `load_vector` exactly; no output-row or projection value is involved.
    private static let activationSumKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_attention_qk_affine4_xsum_v1",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: """
            const uint lane = thread_position_in_grid.x;
            const uint k_block = thread_position_in_grid.y;
            const uint row = thread_position_in_grid.z;
            const int in_vec_size = x_shape[x_ndim - 1];
            const device T* xp =
                x + row * in_vec_size + k_block * 256 + lane * 8;
            float sum = 0.0f;
            sum += xp[0] + xp[1] + xp[2] + xp[3];
            sum += xp[4] + xp[5] + xp[6] + xp[7];
            xSums[(k_block * 32 + lane) * 8 + row] = sum;
            """,
        ensureRowContiguous: true)

    private static let qmvKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_attention_qk_affine4_g64_xsum_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            const int in_vec_size = x_shape[x_ndim - 1];
            const int out_vec_size = w_shape[0];
            const int first_m = int(tid.x) * 4;
            if (first_m >= 8) {
                return;
            }
            qmv_affine4_g64_quad_stream_xsum_impl<T, 64, 4>(
                w,
                scales,
                biases,
                xSums,
                x + first_m * in_vec_size,
                x + (first_m + 1) * in_vec_size,
                x + (first_m + 2) * in_vec_size,
                x + (first_m + 3) * in_vec_size,
                y + first_m * out_vec_size,
                y + (first_m + 1) * out_vec_size,
                y + (first_m + 2) * out_vec_size,
                y + (first_m + 3) * out_vec_size,
                in_vec_size,
                tid,
                simd_gid,
                simd_lid);
            return;
            """,
        header: kernelHeader,
        ensureRowContiguous: true)

    @inline(__always)
    private static func liveOutputWidth(_ width: Int) -> Bool {
        width == 1024 || width == 2048 || width == 4096 || width == 8192
    }

    public static func activationSums(for x: MLXArray) -> ActivationSums? {
        guard enabled,
            activationSumsEnabled,
            x.dtype == .bfloat16,
            x.shape == [batch, sequence, inputWidth],
            x.size == batch * sequence * inputWidth
        else { return nil }

        let blocks = inputWidth / kBlock
        let values = activationSumKernel(
            [x],
            template: [("T", x.dtype)],
            grid: (simdWidth, blocks, batch),
            threadGroup: (simdWidth, 1, 1),
            outputShapes: [[blocks * simdWidth * batch]],
            outputDTypes: [.float32]
        )[0]
        return ActivationSums(values: values)
    }

    /// Returns nil on representation or geometry drift; selected arrays are
    /// normalized to row-contiguous layout by `MLXFast.metalKernel`.
    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        activationSums: ActivationSums?
    ) -> MLXArray? {
        guard enabled,
            activationSumsEnabled,
            let activationSums,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let biases,
            x.dtype == .bfloat16,
            scales.dtype == x.dtype,
            biases.dtype == x.dtype,
            weight.dtype == .uint32,
            x.shape == [batch, sequence, inputWidth],
            x.size == batch * sequence * inputWidth,
            weight.ndim == 2
        else { return nil }

        let outDim = weight.dim(0)
        guard liveOutputWidth(outDim),
            outDim.isMultiple(of: outputsPerGroup),
            weight.shape == [outDim, inputWidth * Self.bits / 32],
            scales.shape == [outDim, inputWidth / Self.groupSize],
            biases.shape == scales.shape
        else { return nil }

        let xGroups = batch / rowsPerGroup
        let yGroups = outDim / outputsPerGroup
        return qmvKernel(
            [x, weight, scales, biases, activationSums.values],
            template: [("T", x.dtype)],
            grid: (xGroups * simdWidth, yGroups * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, outDim]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
