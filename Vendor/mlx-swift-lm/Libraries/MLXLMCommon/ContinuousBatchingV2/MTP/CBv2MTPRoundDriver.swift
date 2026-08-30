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

/// One row's drafter carry: the newest confirmed-but-unfed token (the round
/// seed) plus the target's pre-norm hidden at the position BEFORE it — the
/// pair the Gemma-4 drafter chains from. `tokensCount`/`kvOffset` fingerprint
/// the row state the carry was captured against; any mismatch at plan time
/// (a plain step appended a token, preemption reset progress, an id was
/// reused) invalidates the carry, costing exactly one seed step (prefer
/// simple and correct over carry salvage).
struct CBv2MTPCarry {
    let token: Int
    /// [1, 1, H], lazy slice of an already-evaluated step output.
    let hidden: MLXArray
    /// `rec.tokens.count` at capture — the carry token must still be
    /// `tokens.last` with the same count.
    let tokensCount: Int
    /// `rec.numComputedTokens` at capture (== the row's KV absoluteOffset,
    /// the round anchor).
    let kvOffset: Int
}

// MARK: - In-flight round payload

/// MTP payload riding a `CBv2InFlightStep`. Its presence marks the step as
/// an MTP round step: the chained-decode fast path must never build on top
/// of it (the chained finalize/deferred-release machinery assumes exactly
/// one sample per row), and `finalize` runs the accept-walk for the verify
/// rows at the step's one host-sync boundary.
final class CBv2MTPRoundInFlight {

    struct VerifyRow {
        let id: CBv2RequestID
        /// Storage-owning sequence states at launch, for finalize-time
        /// `rollback` + `commitSpeculativeWrite` (identical objects to
        /// `kvStates[id]` unless the row finished mid-flight, in which case
        /// the whole state is released via the deferred-release fence and
        /// this list is not touched).
        let storageRows: [CBv2SequenceKV]
    }

    struct Verify {
        /// Draft tokens per row this round (uniform across the batch).
        let k: Int
        /// Verify-batch rows, in batch row order.
        let rows: [VerifyRow]
        /// Lazy flattened int32 packet: all [B, k] draft ids followed by all
        /// [B, 1+k] target argmaxes. One `asArray` at finalize reads both,
        /// preserving the single host-sync boundary.
        let acceptancePacket: MLXArray
        /// Lazy [B, 1+k, H] pre-norm hidden — the next carry is gathered
        /// from it at the finalize sync (index = accepted position).
        let lastHidden: MLXArray
    }

    /// nil when this round only seeded (no row had a valid carry yet).
    let verify: Verify?
    /// Seed rows: (request, row index into the step's decode batch). Their
    /// bonus token rides the step's normal `sampledTokens`; the carry hidden
    /// is sliced from `seedHidden` at finalize.
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    /// Lazy [B_decode, 1, H] pre-norm hidden of the step's decode batch
    /// (non-nil iff `seedRows` is non-empty).
    let seedHidden: MLXArray?
    /// Finalization outcomes used by host-only controller attribution. These
    /// are populated at the existing host-sync boundary.
    var finalizedSeedIDs: Set<CBv2RequestID> = []
    var finalizedVerifyIDs: Set<CBv2RequestID> = []
    var claimedSeedCostNanos: UInt64 = 0

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

/// A seed step belongs to the row cohort it prepared, not to the depth that
/// happened to be selected on that plan. A later verification may run at a
/// different depth; exact-cohort matching prevents cancelled or invalidated
/// seed work from leaking into an unrelated probe.
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

    /// Claim seed cost before verify-row completion can retire request ids.
    /// An overlapping but non-identical cohort is discarded rather than
    /// partially charging work whose rectangular batch shape changed.
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

/// Engine-side MTP state: drafter binding, carries, plan marks, metrics.
final class CBv2MTPRoundDriver {

    /// PARTICIPANT DRAFT-DEPTH LEVER (editable). The maximum MTP draft depth this
    /// submission's adaptive controller uses per decode round. This is a CEILING,
    /// not a fixed depth. The controller selects a depth from 0 to this ceiling
    /// each round. A higher value widens the adaptive range. It does not force
    /// speculation. The trusted MTP envelope (`config.maxDraftTokens`, ceiling 3
    /// at batch 8) also bounds it, so the effective cap is
    /// `min(envelope, submissionDraftDepth)`. Compiled at the envelope.
    ///
    /// UNIFORM WITH DFLASH. `DFlashDraftModel.submissionDraftDepth` is the DFlash
    /// counterpart. It has the same name, the same meaning, and the same default
    /// 1. The per-arm behaviour differs. MTP adapts up to this ceiling each round.
    /// DFlash proposes a fixed block of this size, because block diffusion drafts
    /// a whole block at once. The constant you edit is the same on both arms.
    static let submissionDraftDepth = 3

    /// The effective adaptive-depth ceiling: the trusted envelope's max, bounded
    /// by the participant's `submissionDraftDepth`. Pure and static so the cap is
    /// unit-testable without a full driver. It is a CEILING the controller adapts
    /// under, never a fixed pin.
    static func effectiveDraftCeiling(envelopeMax: Int) -> Int {
        min(envelopeMax, submissionDraftDepth)
    }

    let config: CBv2MTPConfig
    let drafter: any CBv2MTPDrafter
    /// The engine's model, downcast once at build (verify forwards go
    /// through `forwardWithHidden`).
    let model: any CBv2MTPSteppableModel
    let captureLayers: CBv2MTPCaptureLayers
    private let depthController: CBv2MTPDepthController

    // Engine-thread confined.
    private var carries: [CBv2RequestID: CBv2MTPCarry] = [:]
    /// Plan-scoped: rows the planner offered a 1+k round this plan.
    private(set) var roundMarks: [CBv2RequestID: Int] = [:]
    /// Plan-scoped: eligible rows without a valid carry — their decode step
    /// this plan is a SEED step (eager forwardWithHidden, hidden captured).
    private(set) var seedMarks: Set<CBv2RequestID> = []
    /// One selection for the whole scheduler plan. Every speculating row in
    /// that plan uses this depth, so target verification stays rectangular.
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
        // The adaptive controller's CEILING: the trusted envelope cap
        // (`config.maxDraftTokens`), further bounded by the participant's
        // `submissionDraftDepth`. `fixedDepth` stays `config.fixedDraftTokens`
        // (adaptive) — submissionDraftDepth is a ceiling, NEVER a fixed pin, so
        // the controller keeps choosing 0…this per round.
        self.depthController = CBv2MTPDepthController(
            maxDepth: Self.effectiveDraftCeiling(envelopeMax: config.maxDraftTokens),
            fixedDepth: config.fixedDraftTokens)
        self.metrics.verificationMode = config.verificationMode
        self.metrics.maxAutomaticRectangularTokens = config.maxAutomaticRectangularTokens
    }

    /// Build the driver, or nil when MTP cannot activate: config off (or the
    /// `DARKBLOOM_CBV2_MTP` kill switch), no drafter, or a model that cannot
    /// drive rounds. nil ⇒ the engine is byte-identical to MTP-less builds.
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

    /// Reset speculation marks. Called immediately before every
    /// `scheduler.plan()` so marks can never leak across plans (a rolled-
    /// back plan's marks must not classify the next plan's rows).
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
        // The envelope cap, bounded by the participant's submissionDraftDepth —
        // the same ceiling the depth controller was built with above.
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

    /// True when no plan can ever carry MTP work, because the depth
    /// controller's policy is target-only or its ceiling is zero. Fixed for
    /// the driver's lifetime, so the engine loop can skip its per-step MTP
    /// bookkeeping outright instead of re-deriving a zero depth every round.
    var isTargetOnlyPolicy: Bool {
        !CBv2MTPDepthController.speculationEnabled || depthController.maxDepth == 0
    }

    var planDepth: Int { planDecision.depth }
    var planDecodeRowBucket: Int { planDecision.decodeRowBucket }

    /// A plan boundary can discover that one otherwise eligible row cannot
    /// complete the selected full round (typically a max-token tail). Mixing
    /// that row's L=1 target decode with neighbors' L=1+k verify changes the
    /// target batch shape. Clamp the whole plan to depth zero so target-only
    /// and MTP execute the same rectangular tail batch.
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

    /// True when this plan produced any MTP work (round or seed).
    var planHasMTPWork: Bool { !roundMarks.isEmpty || !seedMarks.isEmpty }

    // MARK: Carries

    enum CarryStatus {
        case valid(CBv2MTPCarry)
        /// A carry existed but no longer matches the row (plain step,
        /// preemption, id reuse) — dropped by `validatedCarry`.
        case stale
        case none
    }

    /// Pure check (no mutation) — the chained-path pre-check uses it.
    func hasValidCarry(for rec: CBv2ScheduledRequest) -> Bool {
        guard let carry = carries[rec.id] else { return false }
        return carryMatches(carry, rec: rec)
    }

    /// Validate and return the row's carry; a stale carry is removed here
    /// (invalidate-on-mismatch — one seed step re-establishes it).
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

    /// Take the row's carry for a launching round (a fresh one is stored at
    /// the round's finalize, or the row seeds again).
    func consumeCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        carries.removeValue(forKey: id)
    }

    func storeCarry(
        id: CBv2RequestID, token: Int, hidden: MLXArray, tokensCount: Int, kvOffset: Int
    ) {
        carries[id] = CBv2MTPCarry(
            token: token, hidden: hidden, tokensCount: tokensCount, kvOffset: kvOffset)
    }

    /// Preemption / membership hygiene: the structural fingerprint would
    /// catch these lazily, but dropping eagerly keeps no stale device
    /// arrays alive.
    func invalidateCarry(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        pendingSeedCosts.invalidate(id)
    }

    /// The request left the engine for good — ids are legally reusable, so
    /// every per-id trace must go (a reused id must never inherit a carry).
    func requestDidFinish(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        roundMarks.removeValue(forKey: id)
        seedMarks.remove(id)
        pendingSeedCosts.invalidate(id)
    }

    /// Drain/shutdown drops every device-resident request trace while
    /// retaining cumulative metrics/controller estimates for a final poll.
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

    /// The controller optimizes the synchronized rectangular step, so it
    /// learns the minimum accepted prefix that every participating verify
    /// row can commit together, once per step (not once per row).
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

/// Official Gemma candidate generation carries the target hidden state that
/// predicted the newest accepted/unfed token. In a verify tensor whose input
/// columns are `[seed, d1, ...]`, that is exactly column `acceptedDrafts`.
/// Kept as a pure seam so indexing is deterministic and fixture-independent.
enum CBv2MTPHiddenIndex {
    static func carryColumn(targetOutputIndex: Int, draftDepth: Int) -> Int {
        precondition(targetOutputIndex >= 0 && targetOutputIndex <= draftDepth)
        return targetOutputIndex
    }
}
