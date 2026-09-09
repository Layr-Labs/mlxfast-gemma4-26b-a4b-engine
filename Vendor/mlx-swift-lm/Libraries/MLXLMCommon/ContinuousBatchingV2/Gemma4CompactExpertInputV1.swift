import Foundation
import MLX
import MLXFast

/// Production prefill gather without the eightfold sorted activation plane.
enum Gemma4CompactExpertInputV1 {
    static let available: Bool = {
        let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_COMPACT_EXPERT_INPUT"] ?? "1"
        guard !["0", "false", "no", "off"].contains(raw.lowercased()) else { return false }
        #if canImport(Metal)
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        let override = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        let architecture = override.isEmpty ? GPU.deviceInfo().architecture : override
        guard architecture.count >= 3,
            let generation = Int(architecture.dropLast().suffix(2))
        else { return false }
        return generation >= (architecture.last == "p" ? 18 : 17)
        #else
        return false
        #endif
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "gemma4_prefill_compact_expert_input_nax_v1",
        inputNames: ["x", "w", "scales", "biases", "indices", "row_order"],
        outputNames: ["y"],
        source: """
        const int m = M, n = 1408, k = 2816;
        threadgroup T ws[64 * 72];
        compact_gather_qmm_rhs_nax<T, 64, 4, 64, 64, 64, 2, 2, true>(
            x, w, scales, biases, indices, row_order, y, m, n, k,
            threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, ws);
        """,
        header: Gemma4CompactExpertInputNAXSourceV1.header,
        ensureRowContiguous: true)

    static func apply(
        _ x: MLXArray, rowOrder: MLXArray, sortedKeys: MLXArray,
        storage: SwitchGateUpFusedStorage
    ) -> MLXArray? {
        guard available, StreamOrDevice.default == .gpu,
            x.ndim == 2, x.dim(1) == 2816, x.dtype == .bfloat16,
            rowOrder.ndim == 1, sortedKeys.shape == rowOrder.shape,
            rowOrder.dtype == .uint32, sortedKeys.dtype == .uint32,
            rowOrder.size == x.dim(0) * 8, rowOrder.size >= 512,
            rowOrder.size % 64 == 0,
            storage.weight.shape == [128, 1408, 352], storage.weight.dtype == .uint32,
            storage.scales.shape == [128, 1408, 44], storage.scales.dtype == .bfloat16,
            storage.biases.shape == storage.scales.shape, storage.biases.dtype == .bfloat16
        else { return nil }
        let m = rowOrder.size
        let result = kernel(
            [x, storage.weight, storage.scales, storage.biases, sortedKeys, rowOrder],
            template: [("T", DType.bfloat16), ("M", m)],
            grid: (22 * 32, (m / 64) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[m, 1, 1408]], outputDTypes: [.bfloat16])[0]
        CBv2EngageMark.once("prefill-compact-expert-input")
        return result.flattened()[..<(m * 704)].reshaped(m, 1, 704)
    }
}
