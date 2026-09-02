// CBv2MTPDepthController.swift
//
// Step-global adaptive depth selection for rectangular MTP verification.
// One controller belongs to one EngineV2, so model build, assistant revision,
// chip class, and target/assistant quantization are naturally isolated by the
// loaded engine. Within that engine, learned state is keyed by the planned
// decode-row bucket and persists across requests.

import Foundation

public enum CBv2MTPDeviceGate {
    public enum Mode: String, Sendable {
        case on
        case off
        case auto
    }

    public enum WarmStep: Sendable {
        case chainedWideRound(depth: Int, rows: Int)
        case chainedPlain(rows: Int)
    }

    public struct Result: Sendable {
        public let plainMilliseconds: Double?
        public let roundMilliseconds: Double?
        public let prior: Double
        public let margin: Double
        public let speculationEnabled: Bool
        public let reason: String
    }

    private enum Phase {
        case idle
        case warmingRounds
        case measuringRounds
        case warmingPlain
        case measuringPlain
        case complete
    }

    public static let mode: Mode = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_GATE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        else { return .auto }
        return Mode(rawValue: raw) ?? .auto
    }()

    public static let prior: Double = positiveEnvironmentValue(
        "DARKBLOOM_CBV2_MTP_GATE_PRIOR", default: 2.0)
    public static let margin: Double = nonnegativeEnvironmentValue(
        "DARKBLOOM_CBV2_MTP_GATE_MARGIN", default: 0.10)

    nonisolated(unsafe) public private(set) static var speculationEnabled =
        mode != .off && legacySpeculationEnabled
    nonisolated(unsafe) public private(set) static var measurementActive = false

    private static let sampleCount = 12
    private static let warmSampleCount = 4
    private static let lock = NSLock()
    nonisolated(unsafe) private static var phase: Phase = .idle
    nonisolated(unsafe) private static var phaseSamples = 0
    nonisolated(unsafe) private static var roundSamples: [Double] = []
    nonisolated(unsafe) private static var plainSamples: [Double] = []
    nonisolated(unsafe) private static var result: Result?
    nonisolated(unsafe) private static var lineTaken = false

    private static let legacySpeculationEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_SPECULATE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    public static func beginWarmup(warmupEnabled: Bool) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if result != nil { return mode == .on && warmupEnabled }

        guard legacySpeculationEnabled else {
            speculationEnabled = false
            phase = .complete
            result = Result(
                plainMilliseconds: nil, roundMilliseconds: nil,
                prior: prior, margin: margin, speculationEnabled: false,
                reason: "legacy override off")
            return false
        }
        switch mode {
        case .on:
            speculationEnabled = true
            phase = .complete
            result = Result(
                plainMilliseconds: nil, roundMilliseconds: nil,
                prior: prior, margin: margin, speculationEnabled: true,
                reason: "override on")
            return warmupEnabled
        case .off:
            speculationEnabled = false
            phase = .complete
            result = Result(
                plainMilliseconds: nil, roundMilliseconds: nil,
                prior: prior, margin: margin, speculationEnabled: false,
                reason: "override off")
            return false
        case .auto:
            guard warmupEnabled else {
                speculationEnabled = false
                phase = .complete
                result = Result(
                    plainMilliseconds: nil, roundMilliseconds: nil,
                    prior: prior, margin: margin, speculationEnabled: false,
                    reason: "warm-up disabled")
                return false
            }
            phase = .warmingRounds
            phaseSamples = 0
            roundSamples.removeAll(keepingCapacity: true)
            plainSamples.removeAll(keepingCapacity: true)
            speculationEnabled = true
            measurementActive = true
            return true
        }
    }

    @discardableResult
    public static func recordWarmStep(_ step: WarmStep, wallTimeNanos: UInt64) -> Bool {
        guard measurementActive, wallTimeNanos > 0 else { return false }
        lock.lock()
        defer { lock.unlock() }
        guard measurementActive else { return false }
        let milliseconds = Double(wallTimeNanos) / 1_000_000

        switch (phase, step) {
        case (.warmingRounds, .chainedWideRound(depth: 2, rows: 8)):
            phaseSamples += 1
            if phaseSamples >= warmSampleCount {
                phase = .measuringRounds
                phaseSamples = 0
            }
        case (.measuringRounds, .chainedWideRound(depth: 2, rows: 8)):
            roundSamples.append(milliseconds)
            if roundSamples.count >= sampleCount {
                phase = .warmingPlain
                phaseSamples = 0
                speculationEnabled = false
                return true
            }
        case (.warmingPlain, .chainedPlain(rows: 8)):
            phaseSamples += 1
            if phaseSamples >= warmSampleCount {
                phase = .measuringPlain
                phaseSamples = 0
            }
        case (.measuringPlain, .chainedPlain(rows: 8)):
            plainSamples.append(milliseconds)
            if plainSamples.count >= sampleCount {
                let plain = median(plainSamples)
                let round = median(roundSamples)
                let enabled = shouldEnable(
                    plainMilliseconds: plain, roundMilliseconds: round,
                    prior: prior, margin: margin)
                speculationEnabled = enabled
                measurementActive = false
                phase = .complete
                result = Result(
                    plainMilliseconds: plain, roundMilliseconds: round,
                    prior: prior, margin: margin, speculationEnabled: enabled,
                    reason: "auto")
            }
        default:
            break
        }
        return false
    }

    public static func failAutomaticMeasurement(_ reason: String) {
        lock.lock()
        defer { lock.unlock() }
        guard mode == .auto, result == nil else { return }
        speculationEnabled = false
        measurementActive = false
        phase = .complete
        result = Result(
            plainMilliseconds: plainSamples.isEmpty ? nil : median(plainSamples),
            roundMilliseconds: roundSamples.isEmpty ? nil : median(roundSamples),
            prior: prior, margin: margin, speculationEnabled: false,
            reason: reason)
    }

    public static var automaticMeasurementFinished: Bool {
        guard mode == .auto else { return false }
        lock.lock()
        defer { lock.unlock() }
        return result != nil
    }

    public static func takeStderrLine() -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard !lineTaken, let result else { return nil }
        lineTaken = true
        return stderrLine(for: result)
    }

    static func stderrLine(for result: Result) -> String {
        let state = result.speculationEnabled ? "on" : "off"
        if result.reason == "auto",
            let plain = result.plainMilliseconds, let round = result.roundMilliseconds
        {
            return String(
                format: "[mtp-gate] plain %.1f ms round %.1f ms prior %.1f margin %.2f -> speculation %@\n",
                plain, round, result.prior, result.margin, state)
        }
        let plain = result.plainMilliseconds.map { String(format: "%.1f ms", $0) } ?? "n/a"
        let round = result.roundMilliseconds.map { String(format: "%.1f ms", $0) } ?? "n/a"
        return String(
            format: "[mtp-gate] plain %@ round %@ prior %.1f margin %.2f %@ -> speculation %@\n",
            plain, round, result.prior, result.margin, result.reason, state)
    }

    public static func shouldEnable(
        plainMilliseconds: Double, roundMilliseconds: Double,
        prior: Double = 2.0, margin: Double = 0.10
    ) -> Bool {
        guard plainMilliseconds.isFinite, roundMilliseconds.isFinite,
            prior.isFinite, margin.isFinite,
            plainMilliseconds > 0, roundMilliseconds > 0,
            prior > 0, margin >= 0
        else { return false }
        let acceptedWork = plainMilliseconds * prior
        let guardedRound = roundMilliseconds * (1 + margin)
        guard acceptedWork.isFinite, guardedRound.isFinite else { return false }
        return acceptedWork >= guardedRound
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func positiveEnvironmentValue(_ name: String, default fallback: Double) -> Double {
        guard let raw = ProcessInfo.processInfo.environment[name],
            let value = Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
            value.isFinite, value > 0
        else { return fallback }
        return value
    }

    private static func nonnegativeEnvironmentValue(
        _ name: String, default fallback: Double
    ) -> Double {
        guard let raw = ProcessInfo.processInfo.environment[name],
            let value = Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
            value.isFinite, value >= 0
        else { return fallback }
        return value
    }
}

struct CBv2MTPDepthDecision: Equatable {
    let depth: Int
    let decodeRowBucket: Int
    let reason: String
    let isExploration: Bool
}

struct CBv2MTPControllerSnapshot {
    let selectedDepth: Int
    let decodeRowBucket: Int
    let conditionalAcceptance: [Double]
    let costInputs: [CBv2MTPCostInput]
}

struct CBv2MTPStepMeasurement {
    let decision: CBv2MTPDepthDecision
    let actualDepth: Int
    let costEligible: Bool
    var chained: Bool
    let seedOnly: Bool
}

final class CBv2MTPDepthController {
    private static let costAlpha = 0.3
    private static let costClampFraction = 0.25
    private static let acceptanceAlpha = 0.1
    private static let acceptanceMinSamples = 10
    private static let hysteresisFraction = 0.05
    private static let baseProbeInterval = 8
    private static let maxProbeInterval = 256

    static var speculationEnabled: Bool { CBv2MTPDeviceGate.speculationEnabled }

    static let forcedDepth: Int? = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_FORCE_DEPTH"],
            let value = Int(raw.trimmingCharacters(in: .whitespaces)), value >= 0
        else { return nil }
        return value
    }()

    private struct CostState {
        var samples = 0
        var ewmaNanos = 0.0
        var totalNanos: UInt64 = 0

        mutating func observe(_ nanos: UInt64) {
            guard nanos > 0 else { return }
            let sample = Double(nanos)
            if samples == 0 {
                ewmaNanos = sample
            } else {
                let limit = SelfLimit.fraction * ewmaNanos
                let innovation = min(max(sample - ewmaNanos, -limit), limit)
                ewmaNanos += CBv2MTPDepthController.costAlpha * innovation
            }
            samples += 1
            totalNanos &+= nanos
        }

        private enum SelfLimit {
            static let fraction = CBv2MTPDepthController.costClampFraction
        }
    }

    private struct AcceptanceState {
        var rates: [Double] = [0]
        var seen: [Int] = [0]

        mutating func observe(drafted: Int, accepted: Int) {
            guard drafted > 0 else { return }
            for position in 1 ... drafted {
                if accepted < position - 1 { break }
                grow(to: position)
                let outcome = accepted >= position ? 1.0 : 0.0
                if seen[position] == 0 {
                    rates[position] = outcome
                } else {
                    rates[position] +=
                        CBv2MTPDepthController.acceptanceAlpha
                        * (outcome - rates[position])
                }
                seen[position] += 1
            }
        }

        func rate(at position: Int) -> Double {
            if position > 0, position < seen.count,
                seen[position] >= CBv2MTPDepthController.acceptanceMinSamples
            {
                return rates[position]
            }
            guard position > 1 else { return 1 }
            for prior in stride(from: position - 1, through: 1, by: -1) {
                if prior < seen.count,
                    seen[prior] >= CBv2MTPDepthController.acceptanceMinSamples
                {
                    return rates[prior]
                }
            }
            return 1
        }

        func expectedCommitted(depth: Int) -> Double {
            guard depth > 0 else { return 1 }
            var total = 1.0
            var prefixProbability = 1.0
            for position in 1 ... depth {
                prefixProbability *= rate(at: position)
                total += prefixProbability
            }
            return total
        }

        var frontier: Int {
            var result = 0
            guard seen.count > 1 else { return result }
            for position in 1 ..< seen.count {
                guard seen[position] >= CBv2MTPDepthController.acceptanceMinSamples else {
                    break
                }
                result = position
            }
            return result
        }

        var trustedRates: [Double] {
            guard frontier > 0 else { return [] }
            return (1 ... frontier).map { rates[$0] }
        }

        private mutating func grow(to position: Int) {
            while seen.count <= position {
                seen.append(0)
                rates.append(0)
            }
        }
    }

    private struct BucketState {
        var costs: [Int: CostState] = [:]
        var acceptance = AcceptanceState()
        var activeDepth = 0
        var probeInterval = CBv2MTPDepthController.baseProbeInterval
        var roundsSinceProbe = 0
    }

    let maxDepth: Int
    let fixedDepth: Int?
    private var buckets: [Int: BucketState] = [:]
    private var lastDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)

    init(maxDepth: Int, fixedDepth: Int?) {
        let resolvedMax = min(max(maxDepth, 0), CBv2MTPConfig.testedMaxDraftTokens)
        self.maxDepth = resolvedMax
        self.fixedDepth = fixedDepth.map { min(max($0, 0), resolvedMax) }
    }

    static func decodeRowBucket(_ rows: Int) -> Int {
        guard rows > 0 else { return 0 }
        var bucket = 1
        while bucket < rows { bucket *= 2 }
        return bucket
    }

    func preview(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: false)
    }

    func select(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: true)
    }

    func observeAcceptance(decodeRowBucket: Int, drafted: Int, accepted: Int) {
        guard decodeRowBucket > 0, drafted > 0 else { return }
        var state = buckets[decodeRowBucket] ?? BucketState()
        state.acceptance.observe(drafted: drafted, accepted: accepted)
        buckets[decodeRowBucket] = state
    }

    func observeCost(decodeRowBucket: Int, depth: Int, wallTimeNanos: UInt64) {
        guard decodeRowBucket > 0, depth >= 0, depth <= maxDepth, wallTimeNanos > 0 else {
            return
        }
        var state = buckets[decodeRowBucket] ?? BucketState()
        var cost = state.costs[depth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[depth] = cost
        buckets[decodeRowBucket] = state
    }

    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        guard Self.speculationEnabled else { return false }
        guard decision.depth == 0, decision.decodeRowBucket > 0 else { return false }
        return buckets[decision.decodeRowBucket]?.costs[0] == nil
    }

    @discardableResult
    func recordFinalizedStep(
        decision: CBv2MTPDepthDecision,
        actualDepth: Int,
        wallTimeNanos: UInt64,
        costEligible: Bool,
        chained: Bool,
        finalizedPlainWork: Bool,
        finalizedVerification: Bool
    ) -> Bool {
        guard decision.decodeRowBucket > 0,
            actualDepth == decision.depth,
            actualDepth >= 0,
            actualDepth <= maxDepth
        else { return false }

        if actualDepth > 0 {
            guard finalizedVerification, costEligible, wallTimeNanos > 0 else { return false }
        } else {
            guard finalizedPlainWork else { return false }
            if chained {
                var state = buckets[decision.decodeRowBucket] ?? BucketState()
                guard state.costs[0] != nil, !decision.isExploration else { return false }
                complete(decision, state: &state)
                buckets[decision.decodeRowBucket] = state
                return true
            }
            guard costEligible, wallTimeNanos > 0 else { return false }
        }

        var state = buckets[decision.decodeRowBucket] ?? BucketState()
        var cost = state.costs[actualDepth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[actualDepth] = cost
        complete(decision, state: &state)
        buckets[decision.decodeRowBucket] = state
        return true
    }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.activeDepth ?? 0
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.probeInterval ?? Self.baseProbeInterval
    }

    func snapshot() -> CBv2MTPControllerSnapshot {
        var inputs: [CBv2MTPCostInput] = []
        for bucket in buckets.keys.sorted() {
            guard let state = buckets[bucket] else { continue }
            for depth in state.costs.keys.sorted() {
                guard let cost = state.costs[depth] else { continue }
                inputs.append(
                    CBv2MTPCostInput(
                        decodeRowBucket: bucket,
                        depth: depth,
                        samples: cost.samples,
                        ewmaWallTimeNanos: UInt64(max(0, cost.ewmaNanos.rounded())),
                        totalWallTimeNanos: cost.totalNanos))
            }
        }
        let state = buckets[lastDecision.decodeRowBucket]
        return CBv2MTPControllerSnapshot(
            selectedDepth: lastDecision.depth,
            decodeRowBucket: lastDecision.decodeRowBucket,
            conditionalAcceptance: state?.acceptance.trustedRates ?? [],
            costInputs: inputs)
    }

    private func decide(
        plannedDecodeRows: Int, canSpeculate: Bool, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        let bucket = Self.decodeRowBucket(plannedDecodeRows)
        guard bucket > 0 else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: 0, reason: "no_decode_rows",
                    isExploration: false),
                mutate: mutate)
        }
        guard Self.speculationEnabled else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: bucket, reason: "policy_target_only",
                    isExploration: false),
                mutate: mutate)
        }
        guard canSpeculate, maxDepth > 0 else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: bucket,
                    reason: maxDepth == 0 ? "max_depth_zero" : "ineligible",
                    isExploration: false),
                mutate: mutate)
        }
        if let fixedDepth {
            return finish(
                CBv2MTPDepthDecision(
                    depth: fixedDepth, decodeRowBucket: bucket, reason: "fixed",
                    isExploration: false),
                mutate: mutate)
        }
        if let forced = Self.forcedDepth {
            return finish(
                CBv2MTPDepthDecision(
                    depth: min(forced, maxDepth), decodeRowBucket: bucket,
                    reason: "forced_probe", isExploration: false),
                mutate: mutate)
        }

        let state = buckets[bucket] ?? BucketState()
        let limit = min(maxDepth, state.acceptance.frontier + 1)
        let decision: CBv2MTPDepthDecision

        if state.costs[0] == nil {
            decision = CBv2MTPDepthDecision(
                depth: 0, decodeRowBucket: bucket, reason: "warmup_baseline",
                isExploration: true)
        } else if let unsampled = (0 ... limit).first(where: { state.costs[$0] == nil }) {
            decision = CBv2MTPDepthDecision(
                depth: unsampled, decodeRowBucket: bucket, reason: "explore_cost",
                isExploration: true)
        } else {
            let current = min(state.activeDepth, limit)
            let currentGoodput = goodput(depth: current, state: state)
            var best = current
            var bestGoodput = currentGoodput
            for depth in 0 ... limit {
                let candidate = goodput(depth: depth, state: state)
                if candidate > bestGoodput {
                    best = depth
                    bestGoodput = candidate
                }
            }

            var selected = current
            var reason = current == 0 ? "unprofitable" : "goodput"
            if best != current {
                if currentGoodput <= 0
                    || bestGoodput >= currentGoodput * (1 + Self.hysteresisFraction)
                {
                    selected = best
                    reason = best == 0 ? "unprofitable" : "goodput"
                } else {
                    reason = "hysteresis"
                }
            }

            var explore = false
            let nextRounds = state.roundsSinceProbe + 1
            if nextRounds >= state.probeInterval {
                let probe = min(selected + 1, limit)
                if probe > selected {
                    selected = probe
                    reason = "explore_deeper"
                    explore = true
                }
            }
            decision = CBv2MTPDepthDecision(
                depth: selected, decodeRowBucket: bucket, reason: reason,
                isExploration: explore)
        }

        return finish(decision, mutate: mutate)
    }

    private func complete(
        _ decision: CBv2MTPDepthDecision,
        state: inout BucketState
    ) {
        if decision.isExploration {
            state.roundsSinceProbe = 0
            if decision.reason == "explore_deeper" {
                state.probeInterval = min(
                    state.probeInterval * 2, Self.maxProbeInterval)
            }
            return
        }
        if decision.depth != state.activeDepth {
            state.activeDepth = decision.depth
            state.probeInterval = Self.baseProbeInterval
            state.roundsSinceProbe = 0
        } else {
            state.roundsSinceProbe += 1
        }
    }

    private func goodput(depth: Int, state: BucketState) -> Double {
        guard let cost = state.costs[depth], cost.ewmaNanos > 0 else { return 0 }
        return state.acceptance.expectedCommitted(depth: depth) / cost.ewmaNanos
    }

    private func finish(
        _ decision: CBv2MTPDepthDecision, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        if mutate { lastDecision = decision }
        return decision
    }
}
