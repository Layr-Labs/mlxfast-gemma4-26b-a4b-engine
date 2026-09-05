// M5 tensor-operation path for the existing affine-4 batch-eight head.
// Original packed codes stay in place. Each 64-value group's raw product
// receives its original scale; the native BF16 quad-sum supplies its bias.
// This preserves the incumbent group boundaries and BF16 output rounding.
// Matrix reduction order is device-defined: official token validation remains
// necessary even after synthetic comparisons pass on another GPU generation.

import Foundation
import MLX
import MLXFast
#if canImport(Metal)
import Metal
#endif

internal enum Gemma4TensorHeadV1 {
    private static let enabled: Bool = {
        #if canImport(Metal)
        // Integer tensor operands arrived in 26.4. Earlier systems retain
        // the existing SIMD-matrix implementation without compiling this shader.
        guard #available(macOS 26.4, iOS 26.4, visionOS 26.4, *) else { return false }
        let setting = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_TENSOR_HEAD"]?.trimmingCharacters(in: .whitespaces).lowercased()
        if let setting, ["0", "false", "no", "off"].contains(setting) { return false }
        // Explicit enable supports comparative testing of the portable MPP
        // fallback. It is slower on M4 in our measurements, so older GPUs are
        // never enabled by default.
        if let setting, ["1", "true", "yes", "on"].contains(setting) { return true }
        return MTLCreateSystemDefaultDevice()?.supportsFamily(.apple10) == true
        #else
        return false
        #endif
    }()

    private static let simdgroups = 4
    private static let tileWidth = 32

    private static func admits(
        x: MLXArray, w: MLXArray, scales: MLXArray,
        biases: MLXArray, sums: MLXArray
    ) -> Bool {
        guard enabled, StreamOrDevice.default == .gpu,
            x.dtype == .bfloat16, x.shape == [8, 2816],
            w.dtype == .uint32, w.ndim == 2,
            w.dim(0) >= 8192, w.dim(0) % (simdgroups * tileWidth) == 0,
            w.dim(1) == 352,
            scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            scales.shape == [w.dim(0), 44], biases.shape == scales.shape,
            sums.dtype == .float32, sums.shape == [352]
        else { return false }
        return true
    }

    static func projection(
        x: MLXArray, w: MLXArray, scales: MLXArray,
        biases: MLXArray, sums: MLXArray
    ) -> MLXArray? {
        guard admits(x: x, w: w, scales: scales, biases: biases, sums: sums) else {
            return nil
        }
        CBv2EngageMark.once("tensor-head-u4")
        return logitsKernel(
            [x, w, scales, biases, sums],
            template: [("T", x.dtype), ("K", 2816), ("N", w.dim(0)),
                ("NT", tileWidth), ("SG", simdgroups)],
            grid: (w.dim(0), 1, 1), threadGroup: (simdgroups * 32, 1, 1),
            outputShapes: [[8, w.dim(0)]], outputDTypes: [.bfloat16], stream: .gpu
        )[0]
    }

    static func partialArgmax(
        x: MLXArray, w: MLXArray, scales: MLXArray,
        biases: MLXArray, sums: MLXArray
    ) -> (values: MLXArray, indices: MLXArray, tiles: Int)? {
        guard admits(x: x, w: w, scales: scales, biases: biases, sums: sums) else {
            return nil
        }
        CBv2EngageMark.once("tensor-head-u4-argmax")
        let tiles = w.dim(0) / (simdgroups * tileWidth)
        let result = argmaxKernel(
            [x, w, scales, biases, sums],
            template: [("T", x.dtype), ("K", 2816), ("N", w.dim(0)),
                ("NT", tileWidth), ("SG", simdgroups)],
            grid: (w.dim(0), 1, 1), threadGroup: (simdgroups * 32, 1, 1),
            outputShapes: [[8 * tiles], [8 * tiles]],
            outputDTypes: [.float32, .uint32], stream: .gpu)
        return (result[0], result[1], tiles)
    }

    // tensor_inline's current MPP destination helper does not strip const
    // from element types. These views are used only as read operands. No
    // input buffer is written, copied, unpacked, or persistently reformatted.
    // Accumulator layouts are opaque: register transfers require every lane's
    // capacity, validity, and logical coordinates to match. Each SIMD group
    // has its own shared-memory fallback when layouts differ.
    private static let productSource =
        """
        const uint sg = thread_index_in_threadgroup / 32;
        const uint tile = threadgroup_position_in_grid.x * SG + sg;
        constexpr int GROUPS = K / 64;
        constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(8, NT, 64, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        constexpr auto db = mpp::tensor_ops::matmul2d_descriptor(8, NT, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<d, metal::execution_simdgroup> op;
        mpp::tensor_ops::matmul2d<db, metal::execution_simdgroup> biasOp;
        using E = metal::dextents<int, 2>;
        metal::tensor<device T, E, metal::tensor_inline> X(const_cast<device T*>(x), E(K, 8));
        metal::tensor<device metal::uint4b_format, E, metal::tensor_inline> W(reinterpret_cast<device uchar*>(const_cast<device uint*>(w)), E(K, N));
        metal::tensor<device float, E, metal::tensor_inline> XS(const_cast<device float*>(sums), E(8, 8), metal::array<int, 2>{1, GROUPS});
        metal::tensor<device T, E, metal::tensor_inline> BS(const_cast<device T*>(b) + tile * NT * GROUPS, E(8, NT), metal::array<int, 2>{1, GROUPS});
        auto group = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
        auto acc = op.template get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
        auto biasAcc = biasOp.template get_destination_cooperative_tensor<decltype(XS), decltype(BS), float>();
        threadgroup float bridgeStorage[SG * 8 * NT];
        threadgroup float* bridge = bridgeStorage + sg * 8 * NT;
        bool sameLayout = acc.get_capacity() == biasAcc.get_capacity();
        for (uint i = 0; i<min(acc.get_capacity(), biasAcc.get_capacity()); ++i) {
            const bool av = acc.is_valid_element(i), bv = biasAcc.is_valid_element(i);
            sameLayout = sameLayout && av == bv;
            if (av && bv) {
                auto ai = acc.get_multidimensional_index(i), bi = biasAcc.get_multidimensional_index(i);
                sameLayout = sameLayout && ai[0] == bi[0] && ai[1] == bi[1];
            }
        }
        sameLayout = simd_all(sameLayout);

        #pragma unroll
        for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int block = 0; block<GROUPS; block += 8) {
            for (int gg = 0; gg<8 && block + gg<GROUPS; ++gg) {
                const int g = block + gg;
                #pragma unroll
                for (uint i = 0; i<group.get_capacity(); ++i) if (group.is_valid_element(i)) group[i] = 0.0f;
                auto gx = X.slice(g * 64, 0);
                auto gw = W.slice(g * 64, tile * NT);
                op.run(gx, gw, group);
                #pragma unroll
                for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
                    auto ij = acc.get_multidimensional_index(i);
                    acc[i] = metal::fma(float(s[(tile * NT + ij[0]) * GROUPS + g]), group[i], acc[i]);
                }
            }
            const int bg = min(8, GROUPS-block);
            metal::tensor<device float, E, metal::tensor_inline> Af(const_cast<device float*>(sums) + block, E(bg, 8), metal::array<int, 2>{1, GROUPS});
            metal::tensor<device T, E, metal::tensor_inline> Bf(const_cast<device T*>(b) + tile * NT * GROUPS + block, E(bg, NT), metal::array<int, 2>{1, GROUPS});
            if (sameLayout) {
                #pragma unroll
                for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) biasAcc[i] = acc[i];
                biasOp.run(Af, Bf, biasAcc);
                #pragma unroll
                for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = biasAcc[i];
            } else {
                #pragma unroll
                for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
                    auto ij = acc.get_multidimensional_index(i);
                    bridge[ij[1] * NT + ij[0]] = acc[i];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (uint i = 0; i<biasAcc.get_capacity(); ++i) if (biasAcc.is_valid_element(i)) {
                    auto ij = biasAcc.get_multidimensional_index(i);
                    biasAcc[i] = bridge[ij[1] * NT + ij[0]];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                biasOp.run(Af, Bf, biasAcc);
                #pragma unroll
                for (uint i = 0; i<biasAcc.get_capacity(); ++i) if (biasAcc.is_valid_element(i)) {
                    auto ij = biasAcc.get_multidimensional_index(i);
                    bridge[ij[1] * NT + ij[0]] = biasAcc[i];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
                    auto ij = acc.get_multidimensional_index(i);
                    acc[i] = bridge[ij[1] * NT + ij[0]];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        """

    private static let logitsSource = productSource + "\n" +
        """
        #pragma unroll
        for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
            auto ij = acc.get_multidimensional_index(i);
            if (ij[1]<8) y[ij[1] * N + tile * NT + ij[0]] = T(acc[i]);
        }
        """

    private static let argmaxSource = productSource + "\n" +
        """
        #pragma unroll
        for (uint i = 0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
            auto ij = acc.get_multidimensional_index(i);
            bridge[ij[1] * NT + ij[0]] = float(T(acc[i]));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float localValues[SG * 8];
        threadgroup uint localIDs[SG * 8];
        const uint lane = thread_index_in_simdgroup;
        #pragma unroll
        for (uint row = 0; row<8; ++row) {
            const float v = bridge[row * NT + lane];
            const float best = simd_max(v);
            const uint id = simd_min(v == best ? tile * NT + lane : 0xffffffffu);
            if (lane == 0) {
                localValues[sg * 8 + row] = best;
                localIDs[sg * 8 + row] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint row = thread_index_in_threadgroup;
        if (row<8) {
            float v = localValues[row];
            uint id = localIDs[row];
            #pragma unroll
            for (uint j = 1; j<SG; ++j) {
                const float vv = localValues[j * 8 + row];
                const uint ii = localIDs[j * 8 + row];
                if (vv>v || (vv == v && ii<id)) {
                    v = vv;
                    id = ii;
                }
            }
            const uint dest = row * (N / (NT * SG)) + threadgroup_position_in_grid.x;
            pv[dest] = v; pi[dest] = id;
        }
        """

    private static let logitsKernel = MLXFast.metalKernel(
        name: "gemma4_tensor_affine4_head_logits_v1",
        inputNames: ["x", "w", "s", "b", "sums"],
        outputNames: ["y"], source: logitsSource,
        header: "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensureRowContiguous: true)

    private static let argmaxKernel = MLXFast.metalKernel(
        name: "gemma4_tensor_affine4_head_argmax_v1",
        inputNames: ["x", "w", "s", "b", "sums"],
        outputNames: ["pv", "pi"], source: argmaxSource,
        header: "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensureRowContiguous: true)

}
