import Foundation
import MLX
@testable import MLXLMCommon
import Testing

private final class FixedTokenConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode
    let maxTokens: Int
    let fallbackTokenID: Int
    let initialState = 0
    private let path: [Int]

    init(_ path: [Int], mode: CBv2TokenConstraintMode = .required) {
        self.path = path
        self.mode = mode
        self.maxTokens = path.count
        self.fallbackTokenID = path.last ?? 0
    }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        guard state >= 0, state < path.count, remainingTokens > 0 else { return [] }
        return [path[state]]
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        guard state >= 0, state < path.count, tokenID == path[state] else { return nil }
        return state + 1
    }
}

private final class ImpossibleTokenConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens = 4
    let fallbackTokenID = 0
    let initialState = 0
    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] { [] }
    func nextState(state: Int, tokenID: Int) -> Int? { nil }
}

private final class DenseTokenConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens = 1
    let fallbackTokenID = 0
    let initialState = 0
    private let allowed: [Int]
    private let excluded: Int

    init(vocab: Int, excluding token: Int) {
        self.allowed = (0 ..< vocab).filter { $0 != token }
        self.excluded = token
    }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        remainingTokens > 0 ? allowed : []
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        tokenID == excluded ? nil : 1
    }
}

private final class ConstraintBundleSentinel {}

private func ensureConstraintTestMetallib() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let source = environment["MLX_METALLIB_SOURCE"], !source.isEmpty,
        FileManager.default.fileExists(atPath: source),
        let executable = Bundle(for: ConstraintBundleSentinel.self).executableURL
    else { return }
    let destination = executable.deletingLastPathComponent()
        .appendingPathComponent("mlx.metallib")
    if !FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.copyItem(
            at: URL(fileURLWithPath: source), to: destination)
    }
}

@Suite("CBv2 row-local token constraints")
struct CBv2TokenConstraintSamplingTests {
    @Test("greedy sampler enforces token constraints")
    func greedySamplerEnforcesConstraint() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2GreedySampler()
        let id = CBv2RequestID(700)
        let constraint = FixedTokenConstraint([1])
        let params = [CBv2SamplingParams(temperature: 0)]
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [0],
            tokenConstraint: constraint, maxTokens: 1)

        let token = sampler.sample(
            logits: MLXArray([Float(0), 1, 2, 100], [1, 4]),
            params: params,
            requestIDs: [id],
            stepIndex: 0,
            pendingSampledTokens: nil,
            rowContext: { [row] })
            .asArray(Int32.self)

        #expect(token == [1])
    }

    @Test("dense allowlists mask their excluded token")
    func denseAllowlistMask() throws {
        try ensureConstraintTestMetallib()
        let vocab = 32
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let id = CBv2RequestID(701)
        let params = [CBv2SamplingParams(temperature: 0)]
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [0],
            tokenConstraint: DenseTokenConstraint(vocab: vocab, excluding: 31),
            maxTokens: 1)
        var logits = [Float](repeating: 0, count: vocab)
        logits[30] = 10
        logits[31] = 100

        let token = sampler.sample(
            logits: MLXArray(logits, [1, vocab]),
            params: params,
            requestIDs: [id],
            stepIndex: 0,
            pendingSampledTokens: nil,
            rowContext: { [row] })
            .asArray(Int32.self)

        #expect(token == [30])
    }

    @Test("mixed constrained and ordinary rows never share grammar state")
    func mixedRows() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2DefaultSampler(fallbackSeed: 7)
        let constrainedID = CBv2RequestID(1)
        let ordinaryID = CBv2RequestID(2)
        let constraint = FixedTokenConstraint([1, 2])
        let params = [
            CBv2SamplingParams(temperature: 0, topLogprobs: 2),
            CBv2SamplingParams(temperature: 0),
        ]
        var rows = [
            CBv2SamplerRow(
                id: constrainedID, params: params[0],
                promptTokens: [0], tokenConstraint: constraint, maxTokens: 2),
            CBv2SamplerRow(
                id: ordinaryID, params: params[1], promptTokens: [0], maxTokens: 2),
        ]

        let logits = MLXArray([
            Float(0), 1, 2, 10,
            Float(0), 1, 2, 10,
        ], [2, 4])
        let first = sampler.sample(
            logits: logits, params: params,
            requestIDs: [constrainedID, ordinaryID],
            stepIndex: 0, pendingSampledTokens: nil,
            rowContext: { rows }).asArray(Int32.self)
        #expect(first == [1, 3])
        let raw = try #require(sampler.takeStepLogprobs())
        eval(raw.evalTargets)
        let assembled = CBv2Logprobs.assemble(
            raw.gathered, sampledTokens: first.map(Int.init),
            topLogprobsPerRow: raw.topLogprobsPerRow)
        #expect(assembled[0]?.token == 1)
        #expect(assembled[0]?.topLogprobs.first?.token == 3)

        sampler.confirmSampledTokens(
            first.map(Int.init), requestIDs: [constrainedID, ordinaryID])
        rows[0].outputTokens = [1]
        rows[1].outputTokens = [3]
        let second = sampler.sample(
            logits: logits, params: params,
            requestIDs: [constrainedID, ordinaryID],
            stepIndex: 1, pendingSampledTokens: nil,
            rowContext: { rows }).asArray(Int32.self)
        #expect(second == [2, 3])
    }

    @Test("finished request id reuse rebuilds fresh constraint state")
    func requestIDReuse() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2DefaultSampler(fallbackSeed: 9)
        let id = CBv2RequestID(4)
        let params = [CBv2SamplingParams(temperature: 0)]
        let constraint = FixedTokenConstraint([1, 2])
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [0],
            tokenConstraint: constraint, maxTokens: 2)
        let logits = MLXArray([Float(0), 1, 9, 8], [1, 4])

        let first = sampler.sample(
            logits: logits, params: params, requestIDs: [id],
            stepIndex: 0, pendingSampledTokens: nil,
            rowContext: { [row] }).asArray(Int32.self)
        #expect(first == [1])
        sampler.confirmSampledTokens(first.map(Int.init), requestIDs: [id])
        sampler.requestDidFinish(id)

        let reused = sampler.sample(
            logits: logits, params: params, requestIDs: [id],
            stepIndex: 1, pendingSampledTokens: nil,
            rowContext: { [row] }).asArray(Int32.self)
        #expect(reused == [1])
    }

    @Test("B1 B2 B4 constraint masking overhead stays bounded")
    func boundedMaskingOverhead() throws {
        try ensureConstraintTestMetallib()
        let vocab = 262_144
        let steps = 16
        let warmupPairs = 2
        let samplePairs = 9
        let clock = ContinuousClock()
        for batch in [1, 2, 4] {
            let logits = MLXArray.zeros([batch, vocab], type: Float.self)
            let params = [CBv2SamplingParams](
                repeating: .init(temperature: 0), count: batch)
            let ids = (0 ..< batch).map { CBv2RequestID(UInt64(100 + $0)) }

            func measure(constrained: Bool) -> Double {
                let sampler = CBv2DefaultSampler(fallbackSeed: 1)
                let constraints = ids.map { _ in
                    FixedTokenConstraint(
                        [Int](repeating: 1, count: steps + 1))
                }
                var rows = ids.enumerated().map { index, id in
                    CBv2SamplerRow(
                        id: id, params: params[index], promptTokens: [0],
                        tokenConstraint: constrained ? constraints[index] : nil,
                        maxTokens: steps + 1)
                }
                let start = clock.now
                for step in 0 ..< steps {
                    let sampled = sampler.sample(
                        logits: logits, params: params, requestIDs: ids,
                        stepIndex: step, pendingSampledTokens: nil,
                        rowContext: { rows })
                    let host = sampled.asArray(Int32.self).map(Int.init)
                    sampler.confirmSampledTokens(host, requestIDs: ids)
                    for index in rows.indices {
                        rows[index].outputTokens.append(host[index])
                    }
                }
                return durationMilliseconds(start.duration(to: clock.now))
            }

            // Warm both paths and alternate their order so Metal JIT, allocator
            // state, and monotonic thermal drift cannot favor one path.
            for pair in 0 ..< warmupPairs {
                if pair.isMultiple(of: 2) {
                    _ = measure(constrained: false)
                    _ = measure(constrained: true)
                } else {
                    _ = measure(constrained: true)
                    _ = measure(constrained: false)
                }
            }

            var baselineSamples: [Double] = []
            var constrainedSamples: [Double] = []
            var overheadSamples: [Double] = []
            baselineSamples.reserveCapacity(samplePairs)
            constrainedSamples.reserveCapacity(samplePairs)
            overheadSamples.reserveCapacity(samplePairs)
            for pair in 0 ..< samplePairs {
                let baseline: Double
                let constrained: Double
                if pair.isMultiple(of: 2) {
                    baseline = measure(constrained: false)
                    constrained = measure(constrained: true)
                } else {
                    constrained = measure(constrained: true)
                    baseline = measure(constrained: false)
                }
                baselineSamples.append(baseline)
                constrainedSamples.append(constrained)
                overheadSamples.append((constrained - baseline) / Double(steps))
            }

            let baseline = median(baselineSamples)
            let constrained = median(constrainedSamples)
            let overhead = max(0, median(overheadSamples))
            let ratio = constrained / baseline
            print(
                "[tool-constraint-perf] B=\(batch) baseline_median_ms=\(baseline) "
                    + "constrained_median_ms=\(constrained) ratio=\(ratio) "
                    + "overhead_median_ms_per_step=\(overhead) "
                    + "paired_overheads=\(overheadSamples)")
            #expect(overhead < 5)
        }
    }

    @Test("impossible state fails closed on the stop fallback")
    func impossibleState() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let id = CBv2RequestID(77)
        let params = [CBv2SamplingParams(temperature: 0)]
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [1],
            tokenConstraint: ImpossibleTokenConstraint(), maxTokens: 4)
        let token = sampler.sample(
            logits: MLXArray([Float(0), 10, 20], [1, 3]),
            params: params, requestIDs: [id], stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { [row] })
            .asArray(Int32.self)
        #expect(token == [0])
        #expect(
            sampler.tokenConstraintFailure(for: id)
                == "tool_constraint_impossible_state")
    }

    @Test("grammar masking runs before top-k filtering")
    func constraintPrecedesTopK() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let id = CBv2RequestID(88)
        let params = [
            CBv2SamplingParams(
                temperature: 1, topP: 1, topK: 1, seed: 3),
        ]
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [0],
            tokenConstraint: FixedTokenConstraint([1]), maxTokens: 1)
        let sampled = sampler.sample(
            logits: MLXArray([Float(0), 1, 2, 100], [1, 4]),
            params: params, requestIDs: [id], stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { [row] })
            .asArray(Int32.self)
        #expect(sampled == [1])
    }

    @Test("sampling transforms cannot resurrect forbidden logits")
    func hardMaskSurvivesNegativeRepetitionPenalty() throws {
        try ensureConstraintTestMetallib()
        let sampler = CBv2DefaultSampler(fallbackSeed: 1)
        let id = CBv2RequestID(89)
        let params = [
            CBv2SamplingParams(
                temperature: 1,
                topP: 1,
                topK: 1,
                repetitionPenalty: -1,
                repetitionContextSize: 64),
        ]
        let row = CBv2SamplerRow(
            id: id, params: params[0], promptTokens: [3],
            tokenConstraint: FixedTokenConstraint([1]), maxTokens: 1)
        let sampled = sampler.sample(
            logits: MLXArray([Float(0), 1, 2, 100], [1, 4]),
            params: params, requestIDs: [id], stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { [row] })
            .asArray(Int32.self)
        #expect(sampled == [1])
    }
}

private func durationMilliseconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) * 1_000
        + Double(components.attoseconds) / 1_000_000_000_000_000
}

private func median(_ values: [Double]) -> Double {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle]
}
