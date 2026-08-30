import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 MTP verifier QKV projection kernel")
struct Gemma4MTPVerifierQKVKernelTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    private static let geometryProfileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_QKV_GEOMETRIES"] == "1"
    private static let profileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_QKV"] == "1"
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

    @Test(.enabled(if: runtimeEnabled))
    func b1VerifierQKVIsBitExactToIndependentB1Columns() throws {
        let k = 2816
        for n in [1024, 2048, 4096, 8192] {
            let weightValues: [UInt32] = (0..<(n * k / 8)).map { index in
                UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 71)
            }
            let scaleValues: [Float] = (0..<(n * k / 64)).map { index in
                Float(128 + (index * 13) % 61) / 128.0
            }
            let biasValues: [Float] = (0..<(n * k / 64)).map { index in
                Float((index * 5) % 29 - 14) / 128.0
            }
            let weight = MLXArray(weightValues).reshaped([n, k / 8])
            let scales = MLXArray(scaleValues).reshaped([n, k / 64]).asType(.bfloat16)
            let biases = MLXArray(biasValues).reshaped([n, k / 64]).asType(.bfloat16)

            for columns in 2...4 {
                let xValues: [Float] = (0..<(columns * k)).map { index in
                    Float((index * 31 + columns * 11) % 263 - 131) / 128.0
                }
                let x = MLXArray(xValues).reshaped([1, columns, k]).asType(.bfloat16)
                let bound = try #require(CBv2AttentionQKVMMA8V1.bindB1Verifier(
                    columns: columns, weight: weight, scales: scales, biases: biases,
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

    @Test(.enabled(if: runtimeEnabled))
    func verifierQKVIsBitExactToIndependentB8Columns() throws {
        let k = 2816
        for n in [1024, 2048, 4096, 8192] {
        let weightValues: [UInt32] = (0..<(n * k / 8)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 29)
        }
        let scaleValues: [Float] = (0..<(n * k / 64)).map { index in
            Float(128 + (index * 13) % 61) / 128.0
        }
        let biasValues: [Float] = (0..<(n * k / 64)).map { index in
            Float((index * 5) % 29 - 14) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([n, k / 8])
        let scales = MLXArray(scaleValues).reshaped([n, k / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([n, k / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * k)).map { index in
                Float((index * 31 + columns * 11) % 263 - 131) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, k]).asType(.bfloat16)
            let candidate = try #require(
                CBv2AttentionQKVMMA8V1.matmulVerifier(
                    x: x, weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine))
            let reference = concatenated(
                (0..<columns).map { column in
                    let one = x[0..., column..<(column + 1), 0...]
                    return CBv2AttentionQKVMMA8V1.matmul(
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

    @Test(.enabled(if: runtimeEnabled))
    func verifierQKVGeometryRaceRespectsItsNumericalContract() throws {
        let k = 2816
        let n = 2048
        let weightValues: [UInt32] = (0..<(n * k / 8)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 43)
        }
        let scaleValues: [Float] = (0..<(n * k / 64)).map { index in
            Float(128 + (index * 17) % 59) / 128.0
        }
        let biasValues: [Float] = (0..<(n * k / 64)).map { index in
            Float((index * 11) % 31 - 15) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([n, k / 8])
        let scales = MLXArray(scaleValues).reshaped([n, k / 64]).asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([n, k / 64]).asType(.bfloat16)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * k)).map { index in
                Float((index * 37 + columns * 13) % 269 - 134) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, k]).asType(.bfloat16)
            let reference = concatenated(
                (0..<columns).map { column in
                    CBv2AttentionQKVMMA8V1.matmul(
                        x: x[0..., column..<(column + 1), 0...],
                        weight: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4, mode: .affine)!
                },
                axis: 1)

            for geometry in CBv2AttentionQKVMMA8V1.VerifierGeometry.allCases {
                let bound = try #require(CBv2AttentionQKVMMA8V1.bindVerifierProbe(
                    columns: columns, weight: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine,
                    geometry: geometry))
                let candidate = bound(x)
                eval(candidate, reference)
                switch geometry {
                case .ks2Tile1, .ks2Tile2:
                    #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
                case .ks1Tile1, .ks1Tile2:
                    #expect(allClose(candidate, reference, rtol: 0.02, atol: 0.125).item(Bool.self))
                    #expect(
                        all(argMax(candidate, axis: -1) .== argMax(reference, axis: -1))
                            .item(Bool.self))
                }
            }
        }
    }

    @Test(.enabled(if: geometryProfileEnabled))
    func profileVerifierQKVGeometries() throws {
        let k = 2816
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

        for n in [1024, 2048, 4096, 8192] {
            let weight = MLXArray.zeros([n, k / 8], dtype: .uint32)
            let scales = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            let biases = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            eval(weight, scales, biases)

            for columns in 2...4 {
                let x = MLXArray.zeros([8, columns, k], dtype: .bfloat16)
                for geometry in CBv2AttentionQKVMMA8V1.VerifierGeometry.allCases {
                    let bound = try #require(CBv2AttentionQKVMMA8V1.bindVerifierProbe(
                        columns: columns, weight: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4, mode: .affine,
                        geometry: geometry))
                    eval(bound(x))
                    var samples: [Double] = []
                    for _ in 0..<7 {
                        samples.append(elapsed { bound(x) })
                    }
                    print(
                        "MTP_QKV_GEOMETRY n=\(n) columns=\(columns) "
                            + "geometry=\(geometry.rawValue) ms=\(median(samples) * 1_000)")
                }
            }
        }
    }

    @Test(.enabled(if: profileEnabled))
    func profileVerifierQKVAtRankedDimensions() throws {
        let k = 2816
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

        for n in [1024, 2048, 4096, 8192] {
            let weight = MLXArray.zeros([n, k / 8], dtype: .uint32)
            let scales = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            let biases = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
            eval(weight, scales, biases)

            for columns in 2...4 {
                let x = MLXArray.zeros([8, columns, k], dtype: .bfloat16)
                func candidate() throws -> MLXArray {
                    try #require(
                        CBv2AttentionQKVMMA8V1.matmulVerifier(
                            x: x, weight: weight, scales: scales, biases: biases,
                            groupSize: 64, bits: 4, mode: .affine))
                }
                func reference() -> MLXArray {
                    concatenated(
                        (0..<columns).map { column in
                            let one = x[0..., column..<(column + 1), 0...]
                            return CBv2AttentionQKVMMA8V1.matmul(
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
                    "MTP_QKV_PROFILE n=\(n) columns=\(columns) "
                        + "candidate_ms=\(candidateMedian * 1_000) "
                        + "reference_ms=\(referenceMedian * 1_000) "
                        + "speedup=\(referenceMedian / candidateMedian)")
            }
        }
    }
}
