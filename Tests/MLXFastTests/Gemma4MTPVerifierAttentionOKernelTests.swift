import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 MTP verifier attention output kernel")
struct Gemma4MTPVerifierAttentionOKernelTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    private static let profileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_ATTN_O"] == "1"
    private func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    private func expectExactBF16Storage(
        _ candidate: MLXArray, _ reference: MLXArray
    ) {
        eval(candidate, reference)
        #expect(candidate.dtype == .bfloat16)
        #expect(reference.dtype == .bfloat16)
        #expect(candidate.shape == reference.shape)
        #expect(
            candidate.asData(access: .copy).data
                == reference.asData(access: .copy).data)
    }

    @Test
    func b1VerifierBinderUsesIndependentStockM1Columns() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "AttentionOQMVV1.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let binderStart = try #require(source.range(
            of: "    public static func bindB1Verifier("))
        let binderEnd = try #require(source.range(
            of: "    /// Construction-time C8/C16 attention-output binding.",
            range: binderStart.upperBound..<source.endIndex))
        let binder = String(source[binderStart.lowerBound..<binderEnd.lowerBound])
        let closureStart = try #require(binder.range(of: "        return { x in"))
        let closure = String(binder[closureStart.lowerBound..<binder.endIndex])

        #expect(binder.contains("supportsVerifierColumns(columns)"))
        #expect(binder.contains("groupSize == Self.groupSize"))
        #expect(binder.contains("bits == Self.bits"))
        #expect(binder.contains("mode == .affine"))
        #expect(binder.contains("weight.shape == [outputWidth, inDim * Self.bits / 32]"))
        #expect(binder.contains("scales.shape == [outputWidth, inDim / Self.groupSize]"))
        #expect(closure.contains("(0..<columns).map { column in"))
        #expect(closure.contains("x[0..., column..<(column + 1), 0...]"))
        #expect(closure.contains(".reshaped([1, 1, inDim])"))
        #expect(closure.contains("quantizedMM("))
        #expect(closure.contains("axis: 1"))
        #expect(!closure.contains("Gemma4B1MTPQuantizedProjection.bind"))
        #expect(!closure.contains("ProcessInfo"))
        #expect(!closure.contains("guard "))
        #expect(!closure.contains("try"))
        #expect(!closure.contains("catch"))
    }

    @Test
    func c8AndC16AttentionOHaveASeparateSharedSerialReductionBinder() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "AttentionOQMVV1.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let binderStart = try #require(source.range(
            of: "    public static func bindB1SharedSerialReductionVerifier("))
        let binderEnd = try #require(source.range(
            of: "    /// Construction-time binding for the two live verifier o_proj planes.",
            range: binderStart.upperBound..<source.endIndex))
        let binder = String(source[binderStart.lowerBound..<binderEnd.lowerBound])

        #expect(binder.contains("guard [8, 16].contains(columns)"))
        #expect(binder.contains("Gemma4B1MTPQuantizedProjection.bind("))
        #expect(!binder.contains("quantizedMM("))
        #expect(!binder.contains("ProcessInfo"))
        #expect(!binder.contains("try"))
        #expect(!binder.contains("catch"))
    }

    @Test(.enabled(if: runtimeEnabled))
    func b1VerifierAttentionOIsBitExactToIndependentB1Columns() throws {
        let n = 2816
        for k in [4096, 8192] {
            let weightValues: [UInt32] = (0..<(n * k / 8)).map { index in
                UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 73)
            }
            let scaleValues: [Float] = (0..<(n * k / 64)).map { index in
                Float(128 + (index * 17) % 59) / 128.0
            }
            let biasValues: [Float] = (0..<(n * k / 64)).map { index in
                Float((index * 3) % 23 - 11) / 128.0
            }
            let weight = MLXArray(weightValues).reshaped([n, k / 8])
            let scales = MLXArray(scaleValues).reshaped([n, k / 64]).asType(.bfloat16)
            let biases = MLXArray(biasValues).reshaped([n, k / 64]).asType(.bfloat16)

            for columns in 2...4 {
                let xValues: [Float] = (0..<(columns * k)).map { index in
                    Float((index * 37 + columns * 13) % 269 - 134) / 128.0
                }
                let x = MLXArray(xValues).reshaped([1, columns, k]).asType(.bfloat16)
                let bound = try #require(CBv2AttentionOQMVV1.bindB1Verifier(
                    columns: columns, inDim: k,
                    weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine))
                let candidate = bound(x)
                let reference = concatenated(
                    (0..<columns).map { column in
                        quantizedMM(
                            x[0..., column..<(column + 1), 0...], weight,
                            scales: scales, biases: biases, transpose: true,
                            groupSize: 64, bits: 4, mode: .affine)
                    },
                    axis: 1)

                #expect(candidate.shape == [1, columns, n])
                expectExactBF16Storage(candidate, reference)
            }
        }
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [8, 16])
    func c8AndC16SharedAttentionOAreBitExactToIndependentB1Columns(
        columns: Int
    ) throws {
        let inDim = 4096
        let outDim = 2816
        let weightValues: [UInt32] = (0..<(outDim * inDim / 8)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 109)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 17) % 59) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 3) % 23 - 11) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 8])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)
        let xValues: [Float] = (0..<(columns * inDim)).map { index in
            Float((index * 37 + columns * 13) % 269 - 134) / 128.0
        }
        let x = MLXArray(xValues).reshaped([1, columns, inDim])
            .asType(.bfloat16)
        let bound = try #require(
            CBv2AttentionOQMVV1.bindB1SharedSerialReductionVerifier(
                columns: columns, inDim: inDim,
                weight: weight, scales: scales, biases: biases,
                groupSize: 64, bits: 4, mode: .affine))
        let candidate = bound(x)
        let reference = concatenated(
            (0..<columns).map { column in
                quantizedMM(
                    x[0..., column..<(column + 1), 0...], weight,
                    scales: scales, biases: biases, transpose: true,
                    groupSize: 64, bits: 4, mode: .affine)
            },
            axis: 1)

        #expect(candidate.shape == [1, columns, outDim])
        expectExactBF16Storage(candidate, reference)
    }

    @Test(.enabled(if: runtimeEnabled))
    func verifierAttentionOIsBitExactToIndependentB8Columns() throws {
        let n = 2816
        for k in [4096, 8192] {
        let weightValues: [UInt32] = (0..<(n * k / 8)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 37)
        }
        let scaleValues: [Float] = (0..<(n * k / 64)).map { index in
            Float(128 + (index * 17) % 59) / 128.0
        }
        let biasValues: [Float] = (0..<(n * k / 64)).map { index in
            Float((index * 3) % 23 - 11) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([n, k / 8])
        let scales = MLXArray(scaleValues).reshaped([n, k / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([n, k / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * k)).map { index in
                Float((index * 37 + columns * 13) % 269 - 134) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, k]).asType(.bfloat16)
            let candidate = try #require(
                CBv2AttentionOQMVV1.matmulVerifier(
                    x: x, weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine))
            let reference = concatenated(
                (0..<columns).map { column in
                    let one = x[0..., column..<(column + 1), 0...]
                    return CBv2AttentionOQMVV1.matmul(
                        x: one, weight: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4, mode: .affine)!
                },
                axis: 1)
            eval(candidate, reference)

            #expect(candidate.shape == [8, columns, n])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
        }
        }
    }

    @Test(.enabled(if: profileEnabled))
    func profileVerifierAttentionOAtRankedDimensions() throws {
        let n = 2816
        let clock = ContinuousClock()
        func elapsed(_ operation: () throws -> MLXArray) rethrows -> Double {
            let output = try operation()
            let start = clock.now
            eval(output)
            return seconds(start.duration(to: clock.now))
        }
        func median(_ values: [Double]) -> Double {
            values.sorted()[values.count / 2]
        }

        for k in [4096, 8192] {
            let weight = MLXArray.zeros([n, k / 8], dtype: .uint32)
            let scales = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            let biases = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            eval(weight, scales, biases)

            for columns in 2...4 {
                let x = MLXArray.zeros([8, columns, k], dtype: .bfloat16)
                func candidate() throws -> MLXArray {
                    try #require(
                        CBv2AttentionOQMVV1.matmulVerifier(
                            x: x, weight: weight, scales: scales, biases: biases,
                            groupSize: 64, bits: 4, mode: .affine))
                }
                func reference() -> MLXArray {
                    concatenated(
                        (0..<columns).map { column in
                            let one = x[0..., column..<(column + 1), 0...]
                            return CBv2AttentionOQMVV1.matmul(
                                x: one, weight: weight, scales: scales, biases: biases,
                                groupSize: 64, bits: 4, mode: .affine)!
                        },
                        axis: 1)
                }

                eval(try candidate(), reference())
                var candidateSeconds: [Double] = []
                var referenceSeconds: [Double] = []
                for iteration in 0..<5 {
                    if iteration.isMultiple(of: 2) {
                        candidateSeconds.append(try elapsed(candidate))
                        referenceSeconds.append(elapsed(reference))
                    } else {
                        referenceSeconds.append(elapsed(reference))
                        candidateSeconds.append(try elapsed(candidate))
                    }
                }
                let candidateMedian = median(candidateSeconds)
                let referenceMedian = median(referenceSeconds)
                print(
                    "MTP_ATTN_O_PROFILE k=\(k) columns=\(columns) "
                        + "candidate_ms=\(candidateMedian * 1_000) "
                        + "reference_ms=\(referenceMedian * 1_000) "
                        + "speedup=\(referenceMedian / candidateMedian)")
            }
        }
    }
}
