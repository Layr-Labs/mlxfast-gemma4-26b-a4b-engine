import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 MTP verifier dense MLP kernels")
struct Gemma4MTPVerifierDenseMLPKernelTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    private static let geometryProfileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_DENSE_GEOMETRIES"] == "1"
    private static let profileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_DENSE_MLP"] == "1"
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

    @Test(
        .enabled(if: runtimeEnabled),
        arguments: [(2816, 2112), (2112, 2816)])
    func b1VerifierDenseMLPIsBitExactToIndependentB1Columns(
        inDim: Int, outDim: Int
    ) throws {
        let weightValues: [UInt32] = (0..<(outDim * inDim / 4)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 79)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 19) % 57) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 11) % 27 - 13) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 4])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(columns * inDim)).map { index in
                Float((index * 41 + columns * 17) % 271 - 135) / 128.0
            }
            let x = MLXArray(xValues).reshaped([1, columns, inDim]).asType(.bfloat16)
            let bound = try #require(CBv2DenseMLPQMVV1.bindB1Verifier(
                columns: columns, inDim: inDim, outDim: outDim,
                weight: weight, scales: scales, biases: biases,
                groupSize: 64, bits: 8, mode: .affine))
            let candidate = bound(x)
            let reference = concatenated(
                (0..<columns).map { column in
                    quantizedMM(
                        x[0..., column..<(column + 1), 0...], weight,
                        scales: scales, biases: biases, transpose: true,
                        groupSize: 64, bits: 8, mode: .affine)
                },
                axis: 1)

            #expect(candidate.shape == [1, columns, outDim])
            expectExactBF16Storage(candidate, reference)
        }
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [(2816, 2112), (2112, 2816)])
    func verifierDenseMLPIsBitExactToIndependentB8Columns(
        inDim: Int, outDim: Int
    ) throws {
        let weightValues: [UInt32] = (0..<(outDim * inDim / 4)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 43)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 19) % 57) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 11) % 27 - 13) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 4])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * inDim)).map { index in
                Float((index * 41 + columns * 17) % 271 - 135) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, inDim]).asType(.bfloat16)
            let projection: CBv2Gemma4MTPVerifierProjection =
                inDim == 2816 ? .denseGateUp : .denseDown
            let strategy = CBv2Gemma4MTPVerifierRoute.production.strategy(
                for: projection, columns: columns)
            let candidate: MLXArray
            if strategy == .combined {
                candidate = try #require(CBv2DenseMLPQMVV1.matmulVerifier(
                    x: x, weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 8, mode: .affine))
            } else {
                let bound = try #require(CBv2DenseMLPQMVV1.bindIndependentB8(
                    columns: columns, inDim: inDim, outDim: outDim,
                    weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 8, mode: .affine))
                candidate = bound(x)
            }
            let reference = concatenated(
                (0..<columns).map { column in
                    let one = x[0..., column..<(column + 1), 0...]
                    return CBv2DenseMLPQMVV1.matmul(
                        x: one, weight: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 8, mode: .affine)!
                },
                axis: 1)
            eval(candidate, reference)

            #expect(candidate.shape == [8, columns, outDim])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func routeSelectedC4GateUpUsesPreboundIndependentB8Columns() throws {
        let inDim = 2816
        let outDim = 2112
        let weight = MLXArray.ones([outDim, inDim / 4], dtype: .uint32)
        let scales = MLXArray.ones([outDim, inDim / 64], dtype: .bfloat16)
        let biases = MLXArray.zeros([outDim, inDim / 64], dtype: .bfloat16)
        let columns = 4
        let independent = try #require(CBv2DenseMLPQMVV1.bindIndependentB8(
            columns: columns, inDim: inDim, outDim: outDim,
            weight: weight, scales: scales,
            biases: biases, groupSize: 64, bits: 8, mode: .affine))
        let xValues: [Float] = (0..<(8 * columns * inDim)).map { index in
            Float((index * 47 + 31) % 281 - 140) / 128.0
        }
        let x = MLXArray(xValues).reshaped([8, columns, inDim]).asType(.bfloat16)
        let candidate = independent(x)
        let reference = concatenated(
            (0..<columns).map { column in
                CBv2DenseMLPQMVV1.matmul(
                    x: x[0..., column..<(column + 1), 0...],
                    weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 8, mode: .affine)!
            },
            axis: 1)
        eval(candidate, reference)

        #expect(candidate.shape == [8, columns, outDim])
        #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test(.enabled(if: runtimeEnabled))
    func verifierGateUpRowGeometryRaceIsBitExact() throws {
        let inDim = 2816
        let outDim = 2112
        let weightValues: [UInt32] = (0..<(outDim * inDim / 4)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 61)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 23) % 53) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 13) % 29 - 14) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 4])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * inDim)).map { index in
                Float((index * 43 + columns * 19) % 277 - 138) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, inDim]).asType(.bfloat16)
            let reference = concatenated(
                (0..<columns).map { column in
                    CBv2DenseMLPQMVV1.matmul(
                        x: x[0..., column..<(column + 1), 0...],
                        weight: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 8, mode: .affine)!
                },
                axis: 1)

            for geometry in CBv2DenseMLPQMVV1.GateUpVerifierRows.allCases {
                let bound = try #require(CBv2DenseMLPQMVV1.bindGateUpVerifierProbe(
                    columns: columns, weight: weight, scales: scales,
                    biases: biases, groupSize: 64, bits: 8, mode: .affine,
                    rowsPerSimdgroup: geometry))
                let candidate = bound(x)
                eval(candidate, reference)
                #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
            }
        }
    }

    @Test(.enabled(if: geometryProfileEnabled))
    func profileVerifierGateUpRowGeometries() throws {
        let inDim = 2816
        let outDim = 2112
        let weight = MLXArray.zeros([outDim, inDim / 4], dtype: .uint32)
        let scales = MLXArray.zeros([outDim, inDim / 64], dtype: .bfloat16)
        let biases = MLXArray.zeros([outDim, inDim / 64], dtype: .bfloat16)
        eval(weight, scales, biases)

        let clock = ContinuousClock()
        func elapsed(_ operation: () -> MLXArray) -> Double {
            let output = operation()
            let start = clock.now
            eval(output)
            return seconds(start.duration(to: clock.now))
        }
        func median(_ values: [Double]) -> Double {
            values.sorted()[values.count / 2]
        }

        for columns in 2...4 {
            let x = MLXArray.zeros([8, columns, inDim], dtype: .bfloat16)
            for geometry in CBv2DenseMLPQMVV1.GateUpVerifierRows.allCases {
                let bound = try #require(CBv2DenseMLPQMVV1.bindGateUpVerifierProbe(
                    columns: columns, weight: weight, scales: scales,
                    biases: biases, groupSize: 64, bits: 8, mode: .affine,
                    rowsPerSimdgroup: geometry))
                eval(bound(x))
                var samples: [Double] = []
                for _ in 0..<7 {
                    samples.append(elapsed { bound(x) })
                }
                print(
                    "MTP_DENSE_GATEUP_GEOMETRY columns=\(columns) "
                        + "rows=\(geometry.rawValue) ms=\(median(samples) * 1_000)")
            }
        }
    }

    @Test(.enabled(if: profileEnabled))
    func profileVerifierDenseMLPAtRankedDimensions() throws {
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

        for (inDim, outDim) in [(2816, 2112), (2112, 2816)] {
            let weight = MLXArray.zeros([outDim, inDim / 4], dtype: .uint32)
            let scales = MLXArray.zeros([outDim, inDim / 64], dtype: .bfloat16)
            let biases = MLXArray.zeros([outDim, inDim / 64], dtype: .bfloat16)
            eval(weight, scales, biases)

            for columns in 2...4 {
                let x = MLXArray.zeros([8, columns, inDim], dtype: .bfloat16)
                func candidate() throws -> MLXArray {
                    try #require(
                        CBv2DenseMLPQMVV1.matmulVerifier(
                            x: x, weight: weight, scales: scales, biases: biases,
                            groupSize: 64, bits: 8, mode: .affine))
                }
                func reference() -> MLXArray {
                    concatenated(
                        (0..<columns).map { column in
                            let one = x[0..., column..<(column + 1), 0...]
                            return CBv2DenseMLPQMVV1.matmul(
                                x: one, weight: weight, scales: scales, biases: biases,
                                groupSize: 64, bits: 8, mode: .affine)!
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
                    "MTP_DENSE_MLP_PROFILE in=\(inDim) out=\(outDim) columns=\(columns) "
                        + "candidate_ms=\(candidateMedian * 1_000) "
                        + "reference_ms=\(referenceMedian * 1_000) "
                        + "speedup=\(referenceMedian / candidateMedian)")
            }
        }
    }
}
