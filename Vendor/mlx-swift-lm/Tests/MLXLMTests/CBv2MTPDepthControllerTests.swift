import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPDepthController")
struct CBv2MTPDepthControllerTests {
    @Test func automaticVerificationCapsDepthByRectangularWork() throws {
        let model = MTPControllerTestModel()
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: MTPControllerTestDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 8,
                    fixedDraftTokens: 7, verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 8)))

        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 1) == 7)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 2) == 3)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 4) == 1)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 8) == 0)
        #expect(driver.previewDecision(plannedDecodeRows: 4, canSpeculate: true).depth == 1)
        let blocked = driver.previewDecision(plannedDecodeRows: 8, canSpeculate: true)
        #expect(blocked.depth == 0)
        #expect(blocked.reason == "automatic_rectangular_limit")
    }

    @Test func fixedOverrideIncludesZeroAndClampsToTestedMaximum() {
        let zero = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 0)
        #expect(zero.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)

        let two = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 2)
        #expect(two.select(plannedDecodeRows: 4, canSpeculate: true).depth == 2)

        let clamped = CBv2MTPDepthController(maxDepth: 99, fixedDepth: 99)
        #expect(clamped.maxDepth == 7)
        #expect(clamped.fixedDepth == 7)
    }

    @Test func decodeRowsUsePowerOfTwoBuckets() {
        #expect(CBv2MTPDepthController.decodeRowBucket(0) == 0)
        #expect(CBv2MTPDepthController.decodeRowBucket(1) == 1)
        #expect(CBv2MTPDepthController.decodeRowBucket(2) == 2)
        #expect(CBv2MTPDepthController.decodeRowBucket(3) == 4)
        #expect(CBv2MTPDepthController.decodeRowBucket(4) == 4)
        #expect(CBv2MTPDepthController.decodeRowBucket(8) == 8)
    }

    @Test func poorAcceptanceAndSteepCostSelectDepthZero() {
        let controller = CBv2MTPDepthController(maxDepth: 1, fixedDepth: nil)
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 30_000_000)
        controller.observeCost(decodeRowBucket: 1, depth: 1, wallTimeNanos: 60_000_000)
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 0)
        }

        let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(decision.depth == 0)
        #expect(decision.reason == "unprofitable")
    }

    @Test func flatCostAndStrongConditionalAcceptanceExploreDeeper() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for depth in 0 ... 4 {
            controller.observeCost(
                decodeRowBucket: 2, depth: depth,
                wallTimeNanos: UInt64(100_000_000 + depth * 1_000_000))
        }
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 2, drafted: 4, accepted: 4)
        }

        let decision = controller.select(plannedDecodeRows: 2, canSpeculate: true)
        #expect(decision.depth >= 4)
    }

    @Test func conditionalAcceptanceOnlyUpdatesReachedPositions() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for _ in 0 ..< 10 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 4, accepted: 1)
        }
        _ = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        let snapshot = controller.snapshot()
        #expect(snapshot.conditionalAcceptance.count == 2)
        #expect(snapshot.conditionalAcceptance[0] == 1)
        #expect(snapshot.conditionalAcceptance[1] == 0)
    }

    @Test func bucketLearningIsIsolated() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 10)
        controller.observeCost(decodeRowBucket: 1, depth: 1, wallTimeNanos: 12)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason != "warmup_baseline")
        #expect(controller.select(plannedDecodeRows: 3, canSpeculate: true).reason == "warmup_baseline")
    }

    @Test func oneWallCostOutlierIsClamped() throws {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for _ in 0 ..< 20 {
            controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 16_000_000)
        }
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 160_000_000)
        _ = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        let input = try #require(
            controller.snapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        // alpha 0.3 * clamp 25% permits at most a 7.5% one-sample move.
        #expect(input.ewmaWallTimeNanos <= 17_200_001)
        #expect(input.samples == 21)
    }

    @Test func previewDoesNotConsumeExplorationCadence() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        let first = controller.preview(plannedDecodeRows: 1, canSpeculate: true)
        let second = controller.preview(plannedDecodeRows: 1, canSpeculate: true)
        #expect(first == second)
        #expect(controller.snapshot().decodeRowBucket == 0)
    }

    @Test func chainedDepthZeroElapsedCannotForceFalseSpeculation() throws {
        let driver = try makeDriver(maxDepth: 1)
        let baseline = begin(driver)
        #expect(baseline.reason == "warmup_baseline")
        #expect(driver.requiresNonChainedDepthZeroProbe(baseline))
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)

        let exploration = begin(driver)
        #expect(exploration.depth == 1)
        #expect(exploration.reason == "explore_cost")
        let row = CBv2RequestID(1)
        record(
            driver, decision: exploration, actualDepth: 0,
            wallTimeNanos: 20, seedOnly: true,
            finalizedSeedIDs: [row])
        let verify = begin(driver)
        let seedCost = driver.claimPendingSeedCost(
            decodeRowBucket: 1, finalizedVerifyIDs: [row])
        #expect(seedCost == 20)
        record(
            driver, decision: verify, actualDepth: 1,
            wallTimeNanos: 190, finalizedVerification: true,
            claimedSeedCostNanos: seedCost)

        let targetOnly = begin(driver)
        #expect(targetOnly.depth == 0)
        #expect(targetOnly.reason == "unprofitable")
        record(
            driver, decision: targetOnly, actualDepth: 0,
            wallTimeNanos: 1_000_000_000, chained: true,
            finalizedPlainWork: true)

        let baselineCost = try #require(
            driver.metricsSnapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        #expect(baselineCost.samples == 1)
        #expect(baselineCost.totalWallTimeNanos == 100)
        #expect(begin(driver).depth == 0)
    }

    @Test func exploratorySeedBindsToVerifyAndActiveTransitionWaits() throws {
        let driver = try makeDriver(maxDepth: 1)
        let baseline = begin(driver)
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)

        let exploration = begin(driver)
        let row = CBv2RequestID(7)
        record(
            driver, decision: exploration, actualDepth: 0,
            wallTimeNanos: 30, seedOnly: true,
            finalizedSeedIDs: [row])
        #expect(driver.pendingSeedCostCountForTesting == 1)
        #expect(begin(driver).reason == "explore_cost")

        let verification = driver.controllerDecision
        let seedCost = driver.claimPendingSeedCost(
            decodeRowBucket: 1, finalizedVerifyIDs: [row])
        record(
            driver, decision: verification, actualDepth: 1,
            wallTimeNanos: 20, finalizedVerification: true,
            claimedSeedCostNanos: seedCost)
        let depthOne = try #require(
            driver.metricsSnapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 1
            })
        #expect(depthOne.totalWallTimeNanos == 50)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 0)

        let transition = begin(driver)
        #expect(transition.depth == 1)
        #expect(!transition.isExploration)
        record(
            driver, decision: transition, actualDepth: 0,
            wallTimeNanos: 1_000, finalizedPlainWork: true)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 0)

        let retry = begin(driver)
        #expect(retry.depth == 1)
        record(
            driver, decision: retry, actualDepth: 1,
            wallTimeNanos: 50, finalizedVerification: true)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 1)
    }

    @Test func cancellationAndCarryLossClearPendingSeedCost() throws {
        let driver = try makeDriver(maxDepth: 1)
        let baseline = begin(driver)
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)
        let exploration = begin(driver)
        let cancelled = CBv2RequestID(11)
        record(
            driver, decision: exploration, actualDepth: 0,
            wallTimeNanos: 30, seedOnly: true,
            finalizedSeedIDs: [cancelled])
        #expect(driver.pendingSeedCostCountForTesting == 1)
        driver.requestDidFinish(cancelled)
        #expect(driver.pendingSeedCostCountForTesting == 0)
        #expect(
            driver.claimPendingSeedCost(
                decodeRowBucket: 1, finalizedVerifyIDs: [CBv2RequestID(12)]) == 0)

        let retry = begin(driver)
        let invalidated = CBv2RequestID(13)
        record(
            driver, decision: retry, actualDepth: 0,
            wallTimeNanos: 40, seedOnly: true,
            finalizedSeedIDs: [invalidated])
        driver.invalidateCarry(invalidated)
        #expect(driver.pendingSeedCostCountForTesting == 0)
    }

    @Test func noVerificationDoesNotAdvanceProbeBackoff() throws {
        let driver = try makeDriver(maxDepth: 1)
        let baseline = begin(driver)
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)
        let initialExploration = begin(driver)
        record(
            driver, decision: initialExploration, actualDepth: 1,
            wallTimeNanos: 250, finalizedVerification: true)

        for _ in 0 ..< 7 {
            let targetOnly = begin(driver)
            #expect(targetOnly.depth == 0)
            record(
                driver, decision: targetOnly, actualDepth: 0,
                wallTimeNanos: 1_000_000, chained: true,
                finalizedPlainWork: true)
        }

        let probe = begin(driver)
        #expect(probe.depth == 1)
        #expect(probe.reason == "explore_deeper")
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 8)
        record(
            driver, decision: probe, actualDepth: 0,
            wallTimeNanos: 1_000, finalizedPlainWork: true)

        let unchanged = begin(driver)
        #expect(unchanged.depth == 1)
        #expect(unchanged.reason == "explore_deeper")
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 8)
        record(
            driver, decision: unchanged, actualDepth: 1,
            wallTimeNanos: 250, finalizedVerification: true)
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 16)
    }

    @Test func seedCostLedgerIsDepthAgnosticAndRejectsChangedCohorts() {
        var ledger = CBv2MTPSeedCostLedger()
        let first = CBv2RequestID(21)
        let second = CBv2RequestID(22)
        ledger.record(decodeRowBucket: 2, requestIDs: [first, second], nanos: 77)
        #expect(ledger.take(decodeRowBucket: 1, requestIDs: [first, second]) == 0)
        #expect(ledger.take(decodeRowBucket: 2, requestIDs: [first]) == 0)
        #expect(ledger.count == 0)

        // No selected-depth input exists: the same row bucket/cohort claims
        // the seed cost for whichever depth its next verification uses.
        ledger.record(decodeRowBucket: 2, requestIDs: [first, second], nanos: 88)
        #expect(ledger.take(decodeRowBucket: 2, requestIDs: [first, second]) == 88)
    }

    private func makeDriver(maxDepth: Int) throws -> CBv2MTPRoundDriver {
        let model = MTPControllerTestModel()
        return try #require(
            CBv2MTPRoundDriver.build(
                model: model,
                drafter: MTPControllerTestDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: maxDepth,
                    maxSpeculativeBatch: 1, fixedDraftTokens: nil)))
    }

    private func begin(_ driver: CBv2MTPRoundDriver) -> CBv2MTPDepthDecision {
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        return driver.controllerDecision
    }

    private func record(
        _ driver: CBv2MTPRoundDriver,
        decision: CBv2MTPDepthDecision,
        actualDepth: Int,
        wallTimeNanos: UInt64,
        costEligible: Bool = true,
        chained: Bool = false,
        seedOnly: Bool = false,
        finalizedPlainWork: Bool = false,
        finalizedSeedIDs: Set<CBv2RequestID> = [],
        finalizedVerification: Bool = false,
        claimedSeedCostNanos: UInt64 = 0
    ) {
        driver.recordStepCost(
            CBv2MTPStepMeasurement(
                decision: decision, actualDepth: actualDepth,
                costEligible: costEligible, chained: chained,
                seedOnly: seedOnly),
            wallTimeNanos: wallTimeNanos,
            finalizedPlainWork: finalizedPlainWork,
            finalizedSeedIDs: finalizedSeedIDs,
            finalizedVerification: finalizedVerification,
            claimedSeedCostNanos: claimedSeedCostNanos)
    }
}

private final class MTPControllerTestPrepared: CBv2MTPPreparedCapture {}

private final class MTPControllerTestDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?

    init(target: MTPControllerTestModel) {
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        MTPControllerTestPrepared()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens, hidden)
    }
}

private final class MTPControllerTestModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("controller tests do not execute model graphs")
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        fatalError("controller tests do not execute model graphs")
    }
}
