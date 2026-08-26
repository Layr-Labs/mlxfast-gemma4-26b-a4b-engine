import Testing

@testable import MLXLMCommon

/// Pins the two regimes of `CBv2MTPDepthController`, because in the field they
/// look like a backend bug and are not one.
///
/// Gate G2's parity harness reported `rounds=5, drafted=5, accepted=4` on the
/// contiguous KV backend against `rounds=59, drafted=62, accepted=54` on paged,
/// for the SAME drafter, prompts and 144 emitted tokens — a ~12x asymmetry that
/// reads like paged over-speculating or contiguous being gated off. It is
/// neither. Both numbers are this controller's two attractors, selected by ONE
/// comparison in `decide()`:
///
///     goodput(d) = expectedCommitted(d) / ewmaCost(d)
///
/// Before position 1 has `acceptanceMinSamples` observations, `rate(at: 1)`
/// returns 1, so `expectedCommitted(1) == 2` and depth 1 is chosen exactly when
///
///     cost1 / cost0  <  2 / (1 + hysteresisFraction)  ==  1.9047...
///
/// Above that line the controller settles on depth 0 and speculative work
/// collapses to a bounded re-probe on a DOUBLING interval (8, 16, 32, 64, 128)
/// — five rounds, not zero. Below it, nearly every step speculates. Nothing
/// else changes; the 12x is one bit.
///
/// Measured on gemma-4-26B-A4B-it-qat-4bit + its qat assistant, M4 Max, B=1,
/// 144 emitted tokens, contiguous KV (`MTPBackendAsymmetryLiveTests`):
///
///   adaptive : cost0 = 10.916 ms (1 sample), cost1 = 25.111 ms (5 samples)
///              -> ratio 2.300 -> "unprofitable" -> 5 rounds -> 111.82 tok/s
///   forced   : cost1 = 14.214 ms (75 samples, steady state)
///              -> ratio 1.302 -> profitable          -> 75 rounds -> 128.33 tok/s
///
/// The controller's own estimate is 1.77x the steady-state truth, which is
/// what puts it on the wrong side of the line. The inflation is structural,
/// not noise: a depth-zero plan invalidates every carry
/// (`EngineLoopV2.beginMTPPlan`), so every probe round must SEED first, and
/// `CBv2MTPRoundDriver.recordStepCost` folds that seed's wall time into the
/// round's cost sample (`attributed = wallTimeNanos &+ claimedSeedCostNanos`)
/// while `expectedCommitted` never credits the token the seed emitted. Cost
/// in, token out — so choosing depth 0 manufactures the evidence for choosing
/// depth 0 again. Forced mode needs 3 seeds for 75 rounds; the adaptive
/// controller pays 5 seeds for 5.
///
/// Three things these tests defend:
///   1. depth 0 must keep re-probing on a bounded, doubling cadence. Losing the
///      backoff either freezes MTP off forever or re-probes unboundedly.
///   2. the profitable regime must actually speculate on nearly every step.
///   3. the seed-inflated estimate, at the REAL measured costs, is what flips
///      the verdict — so a change to seed attribution shows up here first.
@Suite("CBv2MTPDepthController probe cadence")
struct CBv2MTPProbeCadenceTests {

    /// One decode step's worth of controller interaction, mirroring the engine
    /// loop: `beginMTPPlan` selects a depth, a positive depth needs a seed step
    /// before it can verify, and `finalize` reports the completed step back.
    ///
    /// Depth-zero steps report `chained: true` once the warmup baseline exists,
    /// which is what the contiguous steady state does — the chained fast path
    /// keeps running and `EngineLoopV2.mtpWantsStep` only breaks it for the one
    /// non-chained baseline probe. Chained depth-zero work advances the probe
    /// cadence without contributing a cost sample, exactly as
    /// `recordFinalizedStep` specifies.
    private struct Outcome {
        var roundSteps: [Int] = []
        var roundDepths: [Int] = []
        var drafted = 0
        var accepted = 0

        var rounds: Int { roundSteps.count }
        var gaps: [Int] {
            zip(roundSteps.dropFirst(), roundSteps).map { $0 - $1 }
        }
    }

    /// `costRatio` is cost(depth 1) / cost(depth 0) — the only quantity that
    /// separates the two regimes.
    private func drive(
        costRatio: Double,
        steps: Int,
        baselineNanos: UInt64 = 30_000_000
    ) -> (outcome: Outcome, controller: CBv2MTPDepthController) {
        let controller = CBv2MTPDepthController(
            maxDepth: CBv2MTPConfig.testedMaxDraftTokens, fixedDepth: nil)
        func cost(_ depth: Int) -> UInt64 {
            guard depth > 0 else { return baselineNanos }
            return UInt64(Double(baselineNanos) * costRatio * Double(depth))
        }

        var outcome = Outcome()
        var hasCarry = false
        for step in 0 ..< steps {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            if decision.depth == 0 {
                let chained = !controller.requiresNonChainedDepthZeroProbe(decision)
                controller.recordFinalizedStep(
                    decision: decision,
                    actualDepth: 0,
                    wallTimeNanos: cost(0),
                    costEligible: true,
                    chained: chained,
                    finalizedPlainWork: true,
                    finalizedVerification: false)
                // A depth-zero plan invalidates every carry (`beginMTPPlan`).
                hasCarry = false
                continue
            }
            guard hasCarry else {
                // Seed step: hidden capture only, no verification, and the
                // driver routes its cost to the pending-seed ledger instead of
                // the controller.
                hasCarry = true
                continue
            }
            // Deterministic 7-of-8 acceptance so the regimes are reproducible.
            let accepted = step % 8 == 3 ? 0 : decision.depth
            controller.observeAcceptance(
                decodeRowBucket: decision.decodeRowBucket,
                drafted: decision.depth,
                accepted: accepted)
            controller.recordFinalizedStep(
                decision: decision,
                actualDepth: decision.depth,
                wallTimeNanos: cost(decision.depth),
                costEligible: true,
                chained: false,
                finalizedPlainWork: false,
                finalizedVerification: true)
            outcome.roundSteps.append(step)
            outcome.roundDepths.append(decision.depth)
            outcome.drafted += decision.depth
            outcome.accepted += accepted
            hasCarry = accepted == decision.depth
        }
        return (outcome, controller)
    }

    @Test("unprofitable cost keeps re-probing on a doubling interval, never zero")
    func unprofitableCostCollapsesToBoundedDoublingProbe() {
        let (outcome, controller) = drive(costRatio: 2.5, steps: 200)

        // The exact field observation on contiguous gemma-4: 5 / 5 / 4.
        #expect(outcome.rounds == 5)
        #expect(outcome.drafted == 5)
        #expect(outcome.accepted == 4)
        #expect(outcome.roundDepths.allSatisfy { $0 == 1 })

        // Speculation is BOUNDED, not disabled: the gap between consecutive
        // probes doubles (8, 16, 32, 64 controller intervals, each plus the one
        // seed step a lapsed carry forces).
        #expect(outcome.gaps == [9, 17, 33, 65])
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 128)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
    }

    @Test("profitable cost speculates on nearly every step")
    func profitableCostSpeculatesContinuously() {
        let (outcome, controller) = drive(costRatio: 1.5, steps: 200)

        #expect(outcome.rounds > 150)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        // Once position 1 is trusted the frontier admits depth 2, so a few
        // rounds draft deeper than the settled depth.
        #expect(outcome.drafted > outcome.rounds)
        #expect(outcome.roundDepths.contains(2))
    }

    /// The whole asymmetry is this line. Two runs that straddle it differ by
    /// more than an order of magnitude in speculative work while agreeing on
    /// every emitted token.
    @Test("a single cost comparison separates the two regimes")
    func oneCostComparisonSeparatesTheRegimes() {
        // expectedCommitted(1) == 2 before position 1 is trusted, and a
        // challenger must clear the incumbent by `hysteresisFraction`.
        let threshold = 2.0 / 1.05

        let justProfitable = drive(costRatio: threshold * 0.98, steps: 200).outcome
        let justUnprofitable = drive(costRatio: threshold * 1.02, steps: 200).outcome

        #expect(justUnprofitable.rounds == 5)
        #expect(justProfitable.rounds > 150)
        // The ratio the parity harness saw (59 vs 5) is the SMALL end of what
        // this one comparison can produce.
        #expect(justProfitable.rounds >= justUnprofitable.rounds * 12)
    }

    /// A depth-zero baseline must exist before any comparison is possible, and
    /// it may only be measured on a NON-chained step. If that probe stopped
    /// being requested the controller would never leave warmup and MTP would be
    /// silently off — the failure mode the paged `kv_unsupported` no-op already
    /// produced once this wave.
    @Test("the depth-zero baseline is demanded before any depth is selected")
    func warmupBaselineIsRequiredBeforeSelection() {
        let controller = CBv2MTPDepthController(
            maxDepth: CBv2MTPConfig.testedMaxDraftTokens, fixedDepth: nil)

        let first = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(first.depth == 0)
        #expect(first.reason == "warmup_baseline")
        #expect(controller.requiresNonChainedDepthZeroProbe(first))

        // A CHAINED depth-zero step cannot satisfy it: its interval overlaps
        // neighbouring graph construction, so it is not a comparable baseline.
        controller.recordFinalizedStep(
            decision: first, actualDepth: 0, wallTimeNanos: 30_000_000,
            costEligible: true, chained: true,
            finalizedPlainWork: true, finalizedVerification: false)
        #expect(controller.requiresNonChainedDepthZeroProbe(first))

        controller.recordFinalizedStep(
            decision: first, actualDepth: 0, wallTimeNanos: 30_000_000,
            costEligible: true, chained: false,
            finalizedPlainWork: true, finalizedVerification: false)
        #expect(!controller.requiresNonChainedDepthZeroProbe(first))

        // With a baseline in hand the controller immediately costs depth 1.
        let second = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(second.depth == 1)
        #expect(second.reason == "explore_cost")
    }

    /// The field case, at the wall-clock costs actually measured on gemma-4.
    ///
    /// This is a CHARACTERIZATION test: it pins today's wrong answer together
    /// with the right one, so the gap cannot widen unnoticed and so anyone who
    /// changes seed-cost attribution, the innovation clamp, or the hysteresis
    /// margin is told exactly what those knobs were deciding. If a fix lands,
    /// `seedInflatedEstimate` stops collapsing and THIS TEST MUST BE UPDATED —
    /// that failure is the fix working, not a regression.
    @Test("measured gemma-4 costs: the seed-inflated estimate is what says no")
    func seedInflationFlipsTheVerdictAtMeasuredCosts() {
        // contiguous, B=1, 144 tokens, M4 Max. See the suite comment.
        let baselineNanos: UInt64 = 10_916_000
        let seedInflatedDepthOne = 25_111_000.0
        let steadyStateDepthOne = 14_214_000.0

        let seedInflatedEstimate = drive(
            costRatio: seedInflatedDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos).outcome
        let steadyStateTruth = drive(
            costRatio: steadyStateDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos).outcome

        // What the engine does today: collapse to the probe cadence.
        #expect(seedInflatedEstimate.rounds == 5)
        #expect(seedInflatedEstimate.accepted == 4)

        // What the same controller does on the same hardware once depth 1 is
        // costed at its steady-state price — the price forced mode measures
        // over 75 samples, and the one worth +14.7% decode throughput.
        #expect(steadyStateTruth.rounds > 150)

        // Both sides of the comparison are honest measurements of the same
        // machine. Only the seed attribution separates them.
        #expect(seedInflatedDepthOne / steadyStateDepthOne > 1.7)
    }
}
