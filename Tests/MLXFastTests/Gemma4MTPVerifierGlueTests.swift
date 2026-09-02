import Foundation
import MLX
import MLXFast
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 B1 MTP verifier glue")
struct Gemma4MTPVerifierGlueTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    @Test
    func verifierGlueSourceAdmitsOnlyTheFixedCertifiedColumns() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                    + "Gemma4PrefillGlueV1.swift"),
            encoding: .utf8)
        #expect(source.contains("private static let verifierColumns: Set<Int> = [2, 3, 4, 8, 16]"))
        #expect(source.contains("verifierColumns.contains(columns)"))
        #expect(!source.contains("(2...8).contains(columns)"))
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [2, 3, 4, 8, 16])
    func constructionAcceptsOnlyTheFixedRectangularPlane(columns: Int) throws {
        let weight = MLXArray.ones([2_816], dtype: .bfloat16)
        let bindings = Gemma4PrefillGlueV1.bindVerifier(
            columns: columns,
            postAttentionWeight: weight,
            densePreWeight: weight,
            expertPreWeight: weight,
            densePostWeight: weight,
            expertPostWeight: weight,
            postFeedforwardWeight: weight,
            eps: 1e-6)
        #expect(bindings != nil)
        #expect(Gemma4PrefillGlueV1.bindVerifier(
            columns: 1,
            postAttentionWeight: weight,
            densePreWeight: weight,
            expertPreWeight: weight,
            densePostWeight: weight,
            expertPostWeight: weight,
            postFeedforwardWeight: weight,
            eps: 1e-6) == nil)
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [2, 3, 4, 8, 16])
    func fixedWidthGlueIsBitExactToTheMaterializedChain(columns: Int) throws {
        let shape = [1, columns, 2_816]
        func values(_ salt: Int) -> MLXArray {
            MLXArray((0..<(columns * 2_816)).map {
                Float(($0 * 37 + salt) % 257 - 128) / 128
            }).reshaped(shape).asType(.bfloat16)
        }
        func weight(_ numerator: Float) -> MLXArray {
            (MLXArray.ones([2_816], dtype: .bfloat16) * numerator)
                .asType(.bfloat16)
        }

        let postAttentionWeight = weight(0.75)
        let densePreWeight = weight(0.5)
        let expertPreWeight = weight(0.625)
        let densePostWeight = weight(0.875)
        let expertPostWeight = weight(1.0)
        let postFeedforwardWeight = weight(0.9375)
        let bindings = try #require(Gemma4PrefillGlueV1.bindVerifier(
            columns: columns,
            postAttentionWeight: postAttentionWeight,
            densePreWeight: densePreWeight,
            expertPreWeight: expertPreWeight,
            densePostWeight: densePostWeight,
            expertPostWeight: expertPostWeight,
            postFeedforwardWeight: postFeedforwardWeight,
            eps: 1e-6))

        let attention = values(3)
        let residual = values(17)
        let actualPostAttention = bindings.normResidual(attention, residual)
        let referencePostAttention = residual + MLXFast.rmsNorm(
            attention, weight: postAttentionWeight, eps: 1e-6)

        let (actualDensePre, actualExpertPre) = bindings.dualPreNorm(
            actualPostAttention)
        let referenceDensePre = MLXFast.rmsNorm(
            actualPostAttention, weight: densePreWeight, eps: 1e-6)
        let referenceExpertPre = MLXFast.rmsNorm(
            actualPostAttention, weight: expertPreWeight, eps: 1e-6)

        let dense = values(29)
        let expert = values(43)
        let actualTail = bindings.branchTail(dense, expert, actualPostAttention)
        let densePost = MLXFast.rmsNorm(
            dense, weight: densePostWeight, eps: 1e-6)
        let expertPost = MLXFast.rmsNorm(
            expert, weight: expertPostWeight, eps: 1e-6)
        let referenceTail = actualPostAttention + MLXFast.rmsNorm(
            densePost + expertPost,
            weight: postFeedforwardWeight,
            eps: 1e-6)

        eval(
            actualPostAttention, referencePostAttention,
            actualDensePre, referenceDensePre,
            actualExpertPre, referenceExpertPre,
            actualTail, referenceTail)
        for (actual, reference) in [
            (actualPostAttention, referencePostAttention),
            (actualDensePre, referenceDensePre),
            (actualExpertPre, referenceExpertPre),
            (actualTail, referenceTail),
        ] {
            #expect(allClose(actual, reference, rtol: 0, atol: 0).item(Bool.self))
        }
    }
}
