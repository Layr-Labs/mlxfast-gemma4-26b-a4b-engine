import Foundation
import MLX
import Testing

@Suite("DFlash reference QMV-wide route")
struct DFlashReferenceQMVWideRouteTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test
    func mlx0321AffineQMVWideKernelAndDispatchAreInstalled() throws {
        let mlxRoot = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal")
        let dispatch = try String(
            contentsOf: mlxRoot.appendingPathComponent("quantized.cpp"),
            encoding: .utf8)
        let kernel = try String(
            contentsOf: mlxRoot.appendingPathComponent("kernels/quantized.h"),
            encoding: .utf8)
        let instantiations = try String(
            contentsOf: mlxRoot.appendingPathComponent("kernels/quantized.metal"),
            encoding: .utf8)
        let generated = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"),
            encoding: .utf8)

        #expect(dispatch.contains("void qmv_wide("))
        #expect(dispatch.contains("M >= 2 && use_qmv_wide(mode, d)"))
        #expect(kernel.contains("METAL_FUNC void qmv_wide_impl("))
        #expect(kernel.contains("[[kernel]] void affine_qmv_wide("))
        #expect(instantiations.contains("affine_qmv_wide"))
        #expect(generated.contains("METAL_FUNC void qmv_wide_impl("))
        #expect(generated.contains("[[kernel]] void affine_qmv_wide("))
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [2, 3, 4])
    func affineQMVWideMatchesDequantizedReference(rows: Int) {
        let inputFeatures = 512
        let outputFeatures = 67
        let inputValues: [Float] = (0..<(rows * inputFeatures)).map { index in
            Float((index * 37 + rows * 11) % 257 - 128) / 128
        }
        let weightValues: [Float] = (0..<(outputFeatures * inputFeatures)).map { index in
            Float((index * 29 + 17) % 251 - 125) / 96
        }
        let input = MLXArray(inputValues, [rows, inputFeatures])
        let weight = MLXArray(weightValues, [outputFeatures, inputFeatures])
        let (packed, scales, biases) = quantized(
            weight, groupSize: 64, bits: 4, mode: .affine)
        let actual = quantizedMM(
            input, packed,
            scales: scales, biases: biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let referenceWeight = dequantized(
            packed, scales: scales, biases: biases,
            groupSize: 64, bits: 4, mode: .affine, dtype: .float32)
        let reference = matmul(input, referenceWeight.T)
        eval(actual, reference)

        #expect(actual.shape == reference.shape)
        let maximumError = zip(actual.asArray(Float.self), reference.asArray(Float.self))
            .map { abs($0 - $1) }
            .max() ?? .infinity
        // qmv_wide changes the reduction tree relative to dequantize + GEMM;
        // the deterministic adversarial fixture stays within 5e-3 absolute.
        #expect(maximumError < 5e-3)
    }
}
