// CBv2MTPRoundDriver.swift
//
// ContinuousBatchingV2 — MTP (speculative decoding) round state.
//
// The driver is a passive state holder owned by `EngineLoopV2`: the drafter
// binding, per-row drafter carries, plan-scoped speculation marks, and the
// cumulative metrics. All round LOGIC (eligibility, execution, finalize)
// lives in `EngineLoopV2+MTP.swift` — it needs the loop's row state
// (`kvStates`, `multimodalByID`) and step-building helpers.
//
// Threading: everything except `metricsSnapshot()` is engine-thread
// confined. Metrics are mutated on the engine thread and snapshotted under
// a lock so the provider can poll them from any thread
// (`EngineV2.mtpMetricsSnapshot()`).

import Foundation
import MLX

// MARK: - Drafter carry

struct CBv2MTPCarry {
    let token: Int
    let hidden: MLXArray
    let tokensCount: Int
    let kvOffset: Int
}

// MARK: - In-flight round payload

final class CBv2MTPRoundInFlight {

    struct VerifyRow {
        let id: CBv2RequestID
        let storageRows: [CBv2SequenceKV]
    }

    struct Verify {
        let k: Int
        let rows: [VerifyRow]
        let acceptancePacket: MLXArray
        let lastHidden: MLXArray
        var mirrorRestore: [CBv2MTPMirrorRestoreLayer] = []
        var acceptedDevice: MLXArray? = nil
        var seedNext: MLXArray? = nil
        var carryHiddenNext: MLXArray? = nil
        var nextBase: MLXArray? = nil
        var rowStates: [[CBv2SequenceKV?]] = []
    }

    let verify: Verify?
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    let seedHidden: MLXArray?
    var finalizedSeedIDs: Set<CBv2RequestID> = []
    var finalizedVerifyIDs: Set<CBv2RequestID> = []
    var claimedSeedCostNanos: UInt64 = 0

    struct MemorySnapshot {
        let active = Memory.activeMemory
        let cache = Memory.cacheMemory
        let peak = Memory.peakMemory
    }

    var launchMemory: MemorySnapshot? = nil

    init(
        verify: Verify?,
        seedRows: [(id: CBv2RequestID, decodeIndex: Int)],
        seedHidden: MLXArray?
    ) {
        self.verify = verify
        self.seedRows = seedRows
        self.seedHidden = seedHidden
    }
}

struct CBv2MTPSeedCostLedger {
    private struct Pending {
        var nanos: UInt64
        let requestIDs: Set<CBv2RequestID>
    }

    private var byBucket: [Int: Pending] = [:]

    mutating func record(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>, nanos: UInt64
    ) {
        guard decodeRowBucket > 0, !requestIDs.isEmpty, nanos > 0 else { return }
        if var pending = byBucket[decodeRowBucket], pending.requestIDs == requestIDs {
            pending.nanos &+= nanos
            byBucket[decodeRowBucket] = pending
        } else {
            byBucket[decodeRowBucket] = Pending(nanos: nanos, requestIDs: requestIDs)
        }
    }

    mutating func take(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>
    ) -> UInt64 {
        guard let pending = byBucket[decodeRowBucket], !requestIDs.isEmpty else { return 0 }
        if pending.requestIDs == requestIDs {
            byBucket.removeValue(forKey: decodeRowBucket)
            return pending.nanos
        }
        if !pending.requestIDs.isDisjoint(with: requestIDs) {
            byBucket.removeValue(forKey: decodeRowBucket)
        }
        return 0
    }

    mutating func invalidate(_ id: CBv2RequestID) {
        let matching = byBucket.compactMap { bucket, pending in
            pending.requestIDs.contains(id) ? bucket : nil
        }
        for bucket in matching {
            byBucket.removeValue(forKey: bucket)
        }
    }

    mutating func removeAll() {
        byBucket.removeAll(keepingCapacity: false)
    }

    var count: Int { byBucket.count }
}

// MARK: - Driver

final class CBv2MTPRoundDriver {

    static let submissionDraftDepth = 2

    static func effectiveDraftCeiling(envelopeMax: Int) -> Int {
        min(envelopeMax, submissionDraftDepth)
    }

    let config: CBv2MTPConfig
    let drafter: any CBv2MTPDrafter
    let model: any CBv2MTPSteppableModel
    let captureLayers: CBv2MTPCaptureLayers
    private let depthController: CBv2MTPDepthController

    private var carries: [CBv2RequestID: CBv2MTPCarry] = [:]
    private(set) var roundMarks: [CBv2RequestID: Int] = [:]
    private(set) var seedMarks: Set<CBv2RequestID> = []
    private(set) var controllerDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var planDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var controllerMeasurementEligible = false

    private let metricsLock = NSLock()
    private var metrics = CBv2MTPMetrics()

    private var pendingSeedCosts = CBv2MTPSeedCostLedger()

    private init(
        config: CBv2MTPConfig, drafter: any CBv2MTPDrafter,
        model: any CBv2MTPSteppableModel, captureLayers: CBv2MTPCaptureLayers
    ) {
        self.config = config
        self.drafter = drafter
        self.model = model
        self.captureLayers = captureLayers
        let ceiling = Self.effectiveDraftCeiling(envelopeMax: config.maxDraftTokens)
        self.depthController = CBv2MTPDepthController(
            maxDepth: ceiling,
            fixedDepth: config.fixedDraftTokens ?? ceiling)
        self.metrics.verificationMode = config.verificationMode
        self.metrics.maxAutomaticRectangularTokens = config.maxAutomaticRectangularTokens
    }

    static func build(
        model: CBv2SteppableModel, drafter: (any CBv2MTPDrafter)?, config: CBv2MTPConfig
    ) -> CBv2MTPRoundDriver? {
        guard config.effectiveEnabled, let drafter else { return nil }
        guard let mtpModel = model as? (any CBv2MTPSteppableModel),
            let captureLayers = mtpModel.mtpCaptureLayers
        else { return nil }
        guard let modelTarget = mtpModel.mtpTargetIdentity,
            let drafterTarget = drafter.mtpTargetIdentity,
            modelTarget == drafterTarget
        else { return nil }
        return CBv2MTPRoundDriver(
            config: config, drafter: drafter, model: mtpModel, captureLayers: captureLayers)
    }

    // MARK: Plan-scoped marks

    func beginPlan(plannedDecodeRows: Int, canSpeculate: Bool) {
        if !roundMarks.isEmpty { roundMarks = [:] }
        if !seedMarks.isEmpty { seedMarks = [] }
        controllerMeasurementEligible = canSpeculate
        controllerDecision = depthController.select(
            plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate)
        planDecision = verificationLimitedDecision(
            controllerDecision, plannedDecodeRows: plannedDecodeRows)
        guard plannedDecodeRows > 0 else { return }
        metricsLock.lock()
        metrics.selectedDepth = planDecision.depth
        metrics.decodeRowBucket = planDecision.decodeRowBucket
        metrics.depthSelections[planDecision.depth, default: 0] += 1
        metrics.controllerFallbacks[planDecision.reason, default: 0] += 1
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func previewDecision(
        plannedDecodeRows: Int, canSpeculate: Bool
    ) -> CBv2MTPDepthDecision {
        verificationLimitedDecision(
            depthController.preview(
                plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate),
            plannedDecodeRows: plannedDecodeRows)
    }

    func maximumAutomaticDepth(plannedDecodeRows: Int) -> Int {
        let cap = Self.effectiveDraftCeiling(envelopeMax: config.maxDraftTokens)
        guard config.verificationMode == .automatic, plannedDecodeRows > 0 else {
            return cap
        }
        let maxWidth = config.maxAutomaticRectangularTokens / plannedDecodeRows
        return min(cap, max(0, maxWidth - 1))
    }

    private func verificationLimitedDecision(
        _ decision: CBv2MTPDepthDecision, plannedDecodeRows: Int
    ) -> CBv2MTPDepthDecision {
        let limit = maximumAutomaticDepth(plannedDecodeRows: plannedDecodeRows)
        guard decision.depth > limit else { return decision }
        return CBv2MTPDepthDecision(
            depth: limit, decodeRowBucket: decision.decodeRowBucket,
            reason: "automatic_rectangular_limit", isExploration: false)
    }

    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        depthController.requiresNonChainedDepthZeroProbe(decision)
    }

    var isTargetOnlyPolicy: Bool {
        !CBv2MTPDepthController.speculationEnabled || depthController.maxDepth == 0
    }

    var planDepth: Int { planDecision.depth }
    var planDecodeRowBucket: Int { planDecision.decodeRowBucket }

    func clampPlanDepth(to requestedDepth: Int, reason: String) {
        let newDepth = min(max(requestedDepth, 0), planDecision.depth)
        guard newDepth != planDecision.depth else { return }
        let oldDepth = planDecision.depth
        planDecision = CBv2MTPDepthDecision(
            depth: newDepth, decodeRowBucket: planDecision.decodeRowBucket,
            reason: reason, isExploration: false)
        metricsLock.lock()
        metrics.selectedDepth = newDepth
        if let count = metrics.depthSelections[oldDepth], count > 0 {
            if count == 1 {
                metrics.depthSelections.removeValue(forKey: oldDepth)
            } else {
                metrics.depthSelections[oldDepth] = count - 1
            }
        }
        metrics.depthSelections[newDepth, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func markRound(_ id: CBv2RequestID, k: Int) { roundMarks[id] = k }
    func markSeed(_ id: CBv2RequestID) { seedMarks.insert(id) }
    func roundMark(for id: CBv2RequestID) -> Int? { roundMarks[id] }
    func isSeedMarked(_ id: CBv2RequestID) -> Bool { seedMarks.contains(id) }

    var planHasMTPWork: Bool { !roundMarks.isEmpty || !seedMarks.isEmpty }

    // MARK: Carries

    enum CarryStatus {
        case valid(CBv2MTPCarry)
        case stale
        case none
    }

    func hasValidCarry(for rec: CBv2ScheduledRequest) -> Bool {
        guard let carry = carries[rec.id] else { return false }
        return carryMatches(carry, rec: rec)
    }

    func validatedCarry(for rec: CBv2ScheduledRequest) -> CarryStatus {
        guard let carry = carries[rec.id] else { return .none }
        guard carryMatches(carry, rec: rec) else {
            carries.removeValue(forKey: rec.id)
            return .stale
        }
        return .valid(carry)
    }

    private func carryMatches(_ carry: CBv2MTPCarry, rec: CBv2ScheduledRequest) -> Bool {
        rec.pendingSamples == 0
            && carry.tokensCount == rec.tokens.count
            && carry.token == rec.tokens.last
            && carry.kvOffset == rec.numComputedTokens
    }

    func consumeCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        carries.removeValue(forKey: id)
    }

    func storeCarry(
        id: CBv2RequestID, token: Int, hidden: MLXArray, tokensCount: Int, kvOffset: Int
    ) {
        carries[id] = CBv2MTPCarry(
            token: token, hidden: hidden, tokensCount: tokensCount, kvOffset: kvOffset)
    }

    func invalidateCarry(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        pendingSeedCosts.invalidate(id)
    }

    func requestDidFinish(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        roundMarks.removeValue(forKey: id)
        seedMarks.remove(id)
        pendingSeedCosts.invalidate(id)
    }

    func removeAllRequestState() {
        carries.removeAll(keepingCapacity: false)
        roundMarks.removeAll(keepingCapacity: false)
        seedMarks.removeAll(keepingCapacity: false)
        pendingSeedCosts.removeAll()
    }

    var requestStateCountForTesting: Int {
        carries.count + roundMarks.count + seedMarks.count + pendingSeedCosts.count
    }

    // MARK: Metrics (lock-protected; polled cross-thread)

    func recordSkip(_ reason: String) {
        metricsLock.lock()
        metrics.skippedRows[reason, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordControllerFallback(_ reason: String) {
        metricsLock.lock()
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordSeedSteps(_ count: Int) {
        guard count > 0 else { return }
        metricsLock.lock()
        metrics.seedSteps += count
        metricsLock.unlock()
    }

    var roundsForProfile: Int {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return metrics.rounds
    }

    func recordRound(
        drafted: Int, accepted: Int, emitted: Int,
        audit: CBv2MTPRoundAuditRecord? = nil
    ) {
        metricsLock.lock()
        metrics.rounds += 1
        metrics.draftedTokens += drafted
        metrics.acceptedTokens += accepted
        metrics.emittedTokens += emitted
        if let audit {
            metrics.roundAudits.append(audit)
            if metrics.roundAudits.count > CBv2MTPRoundAuditRecord.retainedRecordCap {
                metrics.roundAudits.removeFirst(
                    metrics.roundAudits.count - CBv2MTPRoundAuditRecord.retainedRecordCap)
            }
        }
        if metrics.perPositionAccepted.count < drafted {
            metrics.perPositionAccepted.append(
                contentsOf: Array(
                    repeating: 0, count: drafted - metrics.perPositionAccepted.count))
        }
        for position in 0 ..< accepted {
            metrics.perPositionAccepted[position] += 1
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func recordStepAcceptance(
        drafted: Int, accepted: Int, observedDrafts: Int,
        decodeRowBucket: Int
    ) {
        depthController.observeAcceptance(
            decodeRowBucket: decodeRowBucket,
            drafted: observedDrafts,
            accepted: accepted)
        metricsLock.lock()
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func claimPendingSeedCost(
        decodeRowBucket: Int, finalizedVerifyIDs: Set<CBv2RequestID>
    ) -> UInt64 {
        pendingSeedCosts.take(
            decodeRowBucket: decodeRowBucket, requestIDs: finalizedVerifyIDs)
    }

    func recordStepCost(
        _ measurement: CBv2MTPStepMeasurement,
        wallTimeNanos: UInt64,
        finalizedPlainWork: Bool,
        finalizedSeedIDs: Set<CBv2RequestID>,
        finalizedVerification: Bool,
        claimedSeedCostNanos: UInt64
    ) {
        guard wallTimeNanos > 0 else { return }
        let decision = measurement.decision
        if measurement.seedOnly, decision.depth > 0 {
            guard measurement.costEligible, !finalizedSeedIDs.isEmpty else { return }
            pendingSeedCosts.record(
                decodeRowBucket: decision.decodeRowBucket,
                requestIDs: finalizedSeedIDs,
                nanos: wallTimeNanos)
            return
        }
        let attributed = wallTimeNanos &+ claimedSeedCostNanos
        let recorded = depthController.recordFinalizedStep(
            decision: decision,
            actualDepth: measurement.actualDepth,
            wallTimeNanos: attributed,
            costEligible: measurement.costEligible,
            chained: measurement.chained,
            finalizedPlainWork: finalizedPlainWork,
            finalizedVerification: finalizedVerification)
        guard recorded else { return }
        metricsLock.lock()
        if measurement.actualDepth > 0 {
            metrics.totalRoundWallTimeNanos &+= attributed
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    var pendingSeedCostCountForTesting: Int { pendingSeedCosts.count }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        depthController.activeDepthForTesting(decodeRowBucket: decodeRowBucket)
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        depthController.probeIntervalForTesting(decodeRowBucket: decodeRowBucket)
    }

    func metricsSnapshot() -> CBv2MTPMetrics {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return metrics
    }

    func recordVerificationStrategy(rectangular: Bool) {
        metricsLock.lock()
        if rectangular {
            metrics.rectangularVerificationRounds += 1
        } else {
            metrics.serialVerificationRounds += 1
        }
        metricsLock.unlock()
    }

    private func refreshControllerMetricsLocked() {
        let snapshot = depthController.snapshot()
        metrics.conditionalAcceptance = snapshot.conditionalAcceptance
        metrics.costInputs = snapshot.costInputs
    }
}

enum CBv2MTPHiddenIndex {
    static func carryColumn(targetOutputIndex: Int, draftDepth: Int) -> Int {
        precondition(targetOutputIndex >= 0 && targetOutputIndex <= draftDepth)
        return targetOutputIndex
    }
}
