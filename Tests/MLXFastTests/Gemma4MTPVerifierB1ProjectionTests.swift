import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 B1 MTP verifier shared projection")
struct Gemma4MTPVerifierB1ProjectionTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    private struct B1AffineProjectionFixture {
        let x: MLXArray
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let ordinaryB1: (MLXArray) -> MLXArray

        init(columns: Int, bits: Int, groupSize: Int, inDim: Int, outDim: Int) {
            x = MLXArray((0..<(columns * inDim)).map {
                Float(($0 * 37 + columns * 11) % 257 - 128) / 128
            }).reshaped([1, columns, inDim]).asType(.bfloat16)

            let packed = inDim * bits / 32
            weight = MLXArray((0..<(outDim * packed)).map {
                UInt32(truncatingIfNeeded: $0 &* 2_654_435_761)
            }).reshaped([outDim, packed])
            scales = MLXArray((0..<(outDim * inDim / groupSize)).map {
                Float(($0 % 31) + 1) / 64
            }).reshaped([outDim, inDim / groupSize]).asType(.bfloat16)
            biases = MLXArray((0..<(outDim * inDim / groupSize)).map {
                Float(($0 % 17) - 8) / 128
            }).reshaped([outDim, inDim / groupSize]).asType(.bfloat16)

            let pinnedWeight = weight
            let pinnedScales = scales
            let pinnedBiases = biases
            ordinaryB1 = { input in
                quantizedMM(
                    input, pinnedWeight,
                    scales: pinnedScales,
                    biases: pinnedBiases,
                    transpose: true,
                    groupSize: groupSize,
                    bits: bits,
                    mode: .affine)
            }
        }
    }

    private func independentColumns(
        _ x: MLXArray,
        project: (MLXArray) -> MLXArray
    ) -> MLXArray {
        concatenated((0..<x.dim(1)).map { column in
            project(x[0..., column..<(column + 1), 0...])
        }, axis: 1)
    }

    @Test
    func sharedProjectionSourceAdmitsExactlyC2C3C4C8AndC16() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                    + "Gemma4B1MTPQuantizedProjection.swift"),
            encoding: .utf8)

        #expect(source.contains("private static let certifiedColumns: Set<Int> = [2, 3, 4, 8, 16]"))
        #expect(source.contains("certifiedColumns.contains(columns)"))
        #expect(!source.contains("(2...8).contains(columns)"))
        #expect(!source.contains("(2...16).contains(columns)"))
    }

    @Test
    func installedProjectionDoesNotInspectRuntimeTopology() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "Gemma4B1MTPQuantizedProjection.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let closureStart = try #require(source.range(of: "        return { x in"))
        let closureEnd = try #require(source.range(
            of: "\n        }\n    }\n\n    // Arithmetic provenance:",
            range: closureStart.upperBound..<source.endIndex))
        let installedClosure = String(
            source[closureStart.lowerBound..<closureEnd.lowerBound])

        #expect(!installedClosure.contains("precondition"))
        #expect(!installedClosure.contains("x.shape"))
        #expect(!installedClosure.contains("x.ndim"))
        #expect(!installedClosure.contains("x.dim("))
        #expect(!installedClosure.contains("x.size"))
        #expect(!installedClosure.contains("x.dtype"))
    }

    @Test(
        .enabled(if: runtimeEnabled),
        arguments: [2, 3, 4, 8, 16], [4, 8])
    func fixedWidthProjectionMatchesIndependentB1(
        columns: Int, bits: Int
    ) throws {
        let fixture = B1AffineProjectionFixture(
            columns: columns,
            bits: bits,
            groupSize: 64,
            inDim: 2_816,
            outDim: 2_112)
        let reference = independentColumns(fixture.x, project: fixture.ordinaryB1)
        let projection = Gemma4B1MTPQuantizedProjection.bind(
            columns: columns,
            inDim: 2_816,
            outDim: 2_112,
            weight: fixture.weight,
            scales: fixture.scales,
            biases: fixture.biases,
            groupSize: 64,
            bits: bits)
        let bound = try #require(projection)
        let candidate = bound(fixture.x)

        eval(reference, candidate)
        #expect(candidate.shape == [1, columns, 2_112])
        #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
    }
}
