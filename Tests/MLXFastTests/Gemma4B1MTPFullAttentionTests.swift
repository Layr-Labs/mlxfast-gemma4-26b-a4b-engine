import Foundation
import MLX
import MLXFast
@testable import MLXLMCommon
import MLXRandom
import Testing

@Suite("Gemma 4 B1 exact shared-KV full attention", .serialized)
struct Gemma4B1MTPFullAttentionTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    private static let longRuntimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_LONG_RUNTIME_TESTS"] == "1"
    private static let benchmarkEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_BENCHMARKS"] == "1"

    private func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func reference(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        historyLength: Int, columns: Int
    ) -> MLXArray {
        concatenated(
            (0..<columns).map { column in
                let visible = historyLength + column + 1
                return MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., column..<(column + 1), 0...],
                    keys: keys[0..., 0..., 0..<visible, 0...],
                    values: values[0..., 0..., 0..<visible, 0...],
                    scale: 1.0,
                    mask: .none)
            },
            axis: 2)
    }

    private func check(columns: Int, keyLength: Int, seed: UInt64) throws {
        defer { Memory.clearCache() }
        let historyLength = keyLength - columns
        #expect(historyLength >= 0)
        MLXRandom.seed(seed)
        var queries = MLXRandom.normal([1, 16, columns, 512])
            .asType(.bfloat16)
        // Keep every query column distinct and include asymmetric score ranges.
        let columnBias = MLXArray((0..<columns).map { Float($0 - 1) * 0.75 })
            .reshaped([1, 1, columns, 1]).asType(.bfloat16)
        queries = queries + columnBias
        let keys = MLXRandom.normal([1, 2, keyLength, 512]).asType(.bfloat16)
        let values = MLXRandom.normal([1, 2, keyLength, 512]).asType(.bfloat16)

        let ordinary = reference(
            queries: queries, keys: keys, values: values,
            historyLength: historyLength, columns: columns)
        let attention = try #require(
            Gemma4B1MTPFullAttentionV1.bind(columns: columns))
        let candidate = attention(queries, keys, values, historyLength)
        eval(ordinary, candidate)

        #expect(candidate.shape == [1, 16, columns, 512])
        for column in 0..<columns {
            let candidateColumn = candidate[0..., 0..., column..<(column + 1), 0...]
            let ordinaryColumn = ordinary[0..., 0..., column..<(column + 1), 0...]
            let exact = all(candidateColumn .== ordinaryColumn).item(Bool.self)
            if !exact {
                let maxAbs = abs(
                    candidateColumn.asType(.float32)
                        - ordinaryColumn.asType(.float32)
                ).max().item(Float.self)
                Issue.record(
                    "shared-KV mismatch: C=\(columns), keyLength=\(keyLength), history=\(historyLength), column=\(column), maxAbs=\(maxAbs)")
                return
            }
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func candidateMatchesIndependentOrdinaryB1AcrossKernelBoundaries() throws {
        let lengths = [
            4, 1_024, 4_095, 4_096, 4_097,
            8_191, 8_192, 8_193, 16_384,
        ]
        for columns in 2...4 {
            for length in lengths where length >= columns {
                try check(
                    columns: columns, keyLength: length,
                    seed: UInt64(columns * 100_000 + length))
            }
        }
    }

    @Test(.enabled(if: longRuntimeEnabled))
    func candidateMatchesIndependentOrdinaryB1AtLongContext() throws {
        for columns in 2...4 {
            for length in [65_536, 131_072] {
                try check(
                    columns: columns, keyLength: length,
                    seed: UInt64(columns * 1_000_000 + length))
            }
        }
    }

    @Test(.enabled(if: benchmarkEnabled))
    func benchmarkSharedKVAgainstSerialWidthOneControl() throws {
        let clock = ContinuousClock()

        func elapsed(_ operation: () -> MLXArray) -> Double {
            let output = operation()
            let start = clock.now
            eval(output)
            Stream.gpu.synchronize()
            return seconds(start.duration(to: clock.now))
        }

        func mean(_ values: [Double]) -> Double {
            values.reduce(0, +) / Double(values.count)
        }

        for keyLength in [1_024, 16_384, 65_536, 131_072] {
            for columns in 2...4 {
                defer { Memory.clearCache() }
                let historyLength = keyLength - columns
                MLXRandom.seed(UInt64(columns * 1_000_000 + keyLength))
                let queries = MLXRandom.normal([1, 16, columns, 512])
                    .asType(.bfloat16)
                let keys = MLXRandom.normal([1, 2, keyLength, 512])
                    .asType(.bfloat16)
                let values = MLXRandom.normal([1, 2, keyLength, 512])
                    .asType(.bfloat16)
                eval(queries, keys, values)

                let candidate = try #require(
                    Gemma4B1MTPFullAttentionV1.bind(columns: columns))
                func serialControl() -> MLXArray {
                    reference(
                        queries: queries, keys: keys, values: values,
                        historyLength: historyLength, columns: columns)
                }
                func sharedKV() -> MLXArray {
                    candidate(queries, keys, values, historyLength)
                }

                // Compile and establish exact parity outside the timed region.
                let controlPrimer = serialControl()
                let candidatePrimer = sharedKV()
                eval(controlPrimer, candidatePrimer)
                Stream.gpu.synchronize()
                let exact = all(controlPrimer .== candidatePrimer).item(Bool.self)
                #expect(
                    exact,
                    Comment(rawValue:
                        "benchmark parity failed: keyLength=\(keyLength), C=\(columns)"))

                var serialSeconds: [Double] = []
                var candidateSeconds: [Double] = []
                for sample in 0..<3 {
                    if sample.isMultiple(of: 2) {
                        serialSeconds.append(elapsed(serialControl))
                        candidateSeconds.append(elapsed(sharedKV))
                    } else {
                        candidateSeconds.append(elapsed(sharedKV))
                        serialSeconds.append(elapsed(serialControl))
                    }
                }

                let serialMean = mean(serialSeconds)
                let candidateMean = mean(candidateSeconds)
                let serialSamples = serialSeconds.map { $0 * 1_000 }
                let candidateSamples = candidateSeconds.map { $0 * 1_000 }
                print(
                    "GEMMA4_B1_ATTENTION_AB key_length=\(keyLength) columns=\(columns) "
                        + "samples=3 exact=\(exact) "
                        + "serial_mean_ms=\(serialMean * 1_000) "
                        + "candidate_mean_ms=\(candidateMean * 1_000) "
                        + "serial_column_tps=\(Double(columns) / serialMean) "
                        + "candidate_column_tps=\(Double(columns) / candidateMean) "
                        + "speedup=\(serialMean / candidateMean) "
                        + "serial_samples_ms=\(serialSamples) "
                        + "candidate_samples_ms=\(candidateSamples)")
            }
        }
    }

    @Test
    func binderPublishesOnlyFixedVerifierWidths() {
        #expect(Gemma4B1MTPFullAttentionV1.bind(columns: 1) == nil)
        #expect(Gemma4B1MTPFullAttentionV1.bind(columns: 2) != nil)
        #expect(Gemma4B1MTPFullAttentionV1.bind(columns: 3) != nil)
        #expect(Gemma4B1MTPFullAttentionV1.bind(columns: 4) != nil)
        #expect(Gemma4B1MTPFullAttentionV1.bind(columns: 5) == nil)
    }
}
