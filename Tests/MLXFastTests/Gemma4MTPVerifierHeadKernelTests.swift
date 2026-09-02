import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 MTP verifier tied-head kernel")
struct Gemma4MTPVerifierHeadKernelTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    private static let profileEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_PROFILE_MTP_HEAD"] == "1"

    private func deterministicHeadWeights(
        k: Int, n: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let packedColumns = k / 8
        let groups = k / 64
        let rowCodes = MLXArray((0..<n).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 0x1357_9bdf)
        }).reshaped([n, 1])
        let columnCodes = MLXArray((0..<packedColumns).map { index in
            UInt32(truncatingIfNeeded: index &* 2_246_822_519 &+ 0x2468_ace1)
        }).reshaped([1, packedColumns])
        let weight = broadcast(rowCodes, to: [n, packedColumns])
            + broadcast(columnCodes, to: [n, packedColumns])

        let rowScales = MLXArray((0..<n).map { index in
            Float(96 + (index * 17) % 61) / 128
        }).reshaped([n, 1])
        let groupScales = MLXArray((0..<groups).map { index in
            Float((index * 13) % 29) / 128
        }).reshaped([1, groups])
        let scales = (broadcast(rowScales, to: [n, groups])
            + broadcast(groupScales, to: [n, groups])).asType(.bfloat16)

        let rowBiases = MLXArray((0..<n).map { index in
            Float((index * 11) % 31 - 15) / 128
        }).reshaped([n, 1])
        let groupBiases = MLXArray((0..<groups).map { index in
            Float((index * 7) % 19 - 9) / 256
        }).reshaped([1, groups])
        let biases = (broadcast(rowBiases, to: [n, groups])
            + broadcast(groupBiases, to: [n, groups])).asType(.bfloat16)
        return (weight, scales, biases)
    }
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
    func tiedHeadAdmitsTheFixedC16SerialReductionEntrypoint() {
        for columns in [2, 3, 4, 8, 16] {
            #expect(Gemma4MMAQuantizedGEMV.supportsVerifierColumns(columns))
        }
        for columns in [1, 5, 7, 9, 15, 17, 32] {
            #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierColumns(columns))
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func b1VerifierHeadAtArtifactWidthIsBitExactToIndependentB1Columns() throws {
        let k = 2816
        let n = 262_144
        let (weight, scales, biases) = deterministicHeadWeights(k: k, n: n)

        for columns in [2, 3, 4, 8, 16] {
            let xValues: [Float] = (0..<(columns * k)).map { index in
                Float((index * 29 + columns * 7) % 257 - 128) / 128.0
            }
            let x = MLXArray(xValues).reshaped([1, columns, k]).asType(.bfloat16)
            let bound = try #require(Gemma4MMAQuantizedGEMV.bindB1Verifier(
                columns: columns, inDim: k, outDim: n,
                w: weight, scales: scales, biases: biases,
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

    @Test
    func verifierHeadSupportsOnlyB8DepthOneThroughThree() {
        #expect(Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 8, columns: 2))
        #expect(Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 8, columns: 3))
        #expect(Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 8, columns: 4))
        #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 8, columns: 1))
        #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 7, columns: 4))
        #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierShape(batch: 8, columns: 5))
    }

    @Test(.enabled(if: runtimeEnabled))
    func verifierHeadIsBitExactToIndependentB8Columns() throws {
        let k = 2816
        let n = 262_144
        let (weight, scales, biases) = deterministicHeadWeights(k: k, n: n)

        for columns in 2...4 {
            let xValues: [Float] = (0..<(8 * columns * k)).map { index in
                Float((index * 29 + columns * 7) % 257 - 128) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, k]).asType(.bfloat16)
            let strategy = CBv2Gemma4MTPVerifierRoute.production.strategy(
                for: .tiedHead, columns: columns)
            let candidate: MLXArray
            if strategy == .combined {
                candidate = try #require(Gemma4MMAQuantizedGEMV.applyVerifier(
                    x: x, w: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4))
            } else {
                let bound = try #require(Gemma4MMAQuantizedGEMV.bindIndependentB8(
                    columns: columns, inDim: k, outDim: n, w: weight,
                    scales: scales, biases: biases, groupSize: 64, bits: 4))
                candidate = bound(x)
            }
            let reference = concatenated(
                (0..<columns).map { column in
                    let one = x[0..., column..<(column + 1), 0...]
                    return Gemma4MMAQuantizedGEMV.apply(
                        x: one, w: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4)!.reshaped([8, 1, n])
                },
                axis: 1)
            eval(candidate, reference)

            #expect(candidate.shape == [8, columns, n])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func routeSelectedC2HeadUsesPreboundIndependentB8Columns() throws {
        let k = 2816
        let n = 262_144
        let (weight, scales, biases) = deterministicHeadWeights(k: k, n: n)
        let columns = 2
        let independent = try #require(Gemma4MMAQuantizedGEMV.bindIndependentB8(
            columns: columns, inDim: k, outDim: n, w: weight,
            scales: scales, biases: biases,
            groupSize: 64, bits: 4))
        let xValues: [Float] = (0..<(8 * columns * k)).map { index in
            Float((index * 31 + 19) % 277 - 138) / 128.0
        }
        let x = MLXArray(xValues).reshaped([8, columns, k]).asType(.bfloat16)
        let candidate = independent(x)
        let reference = concatenated(
            (0..<columns).map { column in
                Gemma4MMAQuantizedGEMV.apply(
                    x: x[0..., column..<(column + 1), 0...],
                    w: weight, scales: scales, biases: biases,
                    groupSize: 64, bits: 4)!.reshaped([8, 1, n])
            },
            axis: 1)
        eval(candidate, reference)

        #expect(candidate.shape == [8, columns, n])
        #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test(.enabled(if: profileEnabled))
    func profileVerifierHeadAtRankedDimensions() throws {
        let k = 2816
        let n = 262_144
        let weight = MLXArray.zeros([n, k / 8], dtype: .uint32)
        let scales = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
        let biases = MLXArray.zeros([n, k / 64], dtype: .bfloat16)
        eval(weight, scales, biases)

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

        for columns in 2...4 {
            let x = MLXArray.zeros([8, columns, k], dtype: .bfloat16)
            func candidate() throws -> MLXArray {
                try #require(
                    Gemma4MMAQuantizedGEMV.applyVerifier(
                        x: x, w: weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4))
            }
            func reference() -> MLXArray {
                concatenated(
                    (0..<columns).map { column in
                        let one = x[0..., column..<(column + 1), 0...]
                        return Gemma4MMAQuantizedGEMV.apply(
                            x: one, w: weight, scales: scales, biases: biases,
                            groupSize: 64, bits: 4)!.reshaped([8, 1, n])
                    },
                    axis: 1)
            }

            // Compile both graphs before sampling.  Alternate order so neither
            // route owns a systematic warm-cache or thermal position.
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
                "MTP_HEAD_PROFILE columns=\(columns) "
                    + "candidate_ms=\(candidateMedian * 1_000) "
                    + "reference_ms=\(referenceMedian * 1_000) "
                    + "speedup=\(referenceMedian / candidateMedian)")
        }
    }
}
