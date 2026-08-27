// EngineLoopV2+MTPPlanning.swift
//
// MTP row eligibility and step-global scheduler planning.

extension EngineLoopV2 {

    /// Per-row hard gates. Ineligible rows remain ordinary target rows and do
    /// not contribute MTP skip metrics.
    private func mtpBasicEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        let sampling = rec.request.sampling
        guard rec.request.tokenConstraint == nil,
            sampling.temperature == 0,
            sampling.topLogprobs == 0,
            sampling.logitBias.isEmpty,
            sampling.repetitionPenalty == 1,
            sampling.frequencyPenalty == 0,
            sampling.presencePenalty == 0,
            rec.request.stopStrings.isEmpty
        else { return false }
        if multimodalByID[rec.id]?.containsSpan(at: rec.tokens.count - 1) ?? false {
            return false
        }
        return true
    }

    /// Every storage-owning row must support value-exact multi-token writes
    /// and rollback. Kept internal for backend contract tests.
    static func mtpStorageEligible(_ state: [CBv2SequenceKV?]) -> Bool {
        state.allSatisfy { $0?.supportsSpeculativeWrites ?? true }
    }

    private enum CBv2MTPPlanAction {
        case round(k: Int)
        case seed
        case none
    }

    private func mtpPlanAction(
        for rec: CBv2ScheduledRequest, recordSkips: Bool
    ) -> CBv2MTPPlanAction {
        guard let mtp else { return .none }
        guard mtpBasicEligible(rec) else { return .none }
        let depth = mtp.planDepth
        guard depth > 0 else { return .none }
        guard let state = kvStates[rec.id] else { return .none }
        guard Self.mtpStorageEligible(state) else {
            if recordSkips { mtp.recordSkip("kv_unsupported") }
            return .none
        }

        // Verify writes 1+k entries at [C-1, C-1+k]. A seed only pays off
        // when a full round can follow after its one generated token.
        let remainingToLength = rec.request.maxTokens - rec.generatedTokenCount
        switch mtp.validatedCarry(for: rec) {
        case .valid:
            guard remainingToLength >= depth else {
                if recordSkips { mtp.recordSkip("max_tokens") }
                return .none
            }
            return .round(k: depth)
        case .stale:
            if recordSkips { mtp.recordSkip("carry_invalid") }
            fallthrough
        case .none:
            guard remainingToLength >= depth + 1 else { return .none }
            return .seed
        }
    }

    /// Scheduler speculation hook for one decode-ready running row.
    func mtpPlanSpeculation(for rec: CBv2ScheduledRequest) -> Int {
        guard let mtp else { return 0 }
        switch mtpPlanAction(for: rec, recordSkips: true) {
        case .round(let k):
            mtp.markRound(rec.id, k: k)
            return k
        case .seed:
            mtp.markSeed(rec.id)
            return 0
        case .none:
            return 0
        }
    }

    /// Break the chained fast path when the next step must seed or verify.
    func mtpWantsStep(ids: [CBv2RequestID]) -> Bool {
        guard let mtp else { return false }
        if mtp.isTargetOnlyPolicy { return false }
        let rows = ids.compactMap { scheduler.record(for: $0) }
        let withinBatchGate = ids.count <= mtp.config.maxSpeculativeBatch
        let canSpeculate = withinBatchGate && rows.count == ids.count
            && mtpRowsCanSpeculate(rows)
        let decision = mtp.previewDecision(
            plannedDecodeRows: ids.count, canSpeculate: canSpeculate)
        let eligible = rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
        guard !eligible.isEmpty else { return false }
        if decision.depth == 0 {
            if mtp.config.verificationMode == .automatic,
                mtp.maximumAutomaticDepth(plannedDecodeRows: ids.count) == 0
            {
                return false
            }
            return canSpeculate && mtp.requiresNonChainedDepthZeroProbe(decision)
        }
        guard canSpeculate else { return false }
        return eligible.allSatisfy { rec in
            let remaining = rec.request.maxTokens - rec.generatedTokenCount
            return mtp.hasValidCarry(for: rec)
                ? remaining >= 1
                : remaining >= 2
        }
    }

    /// Select one controller depth for all decode rows in the scheduler plan.
    /// Chunked-prefill neighbors do not change the controller batch bucket.
    func beginMTPPlan() {
        guard let mtp else { return }
        if mtp.isTargetOnlyPolicy { return }
        let rows = scheduler.running.filter {
            !$0.isPaused && !$0.cancelRequested && $0.isDecodeReady
        }
        let withinBatchGate = rows.count <= mtp.config.maxSpeculativeBatch
        let canSpeculate = withinBatchGate && mtpRowsCanSpeculate(rows)
        mtp.beginPlan(plannedDecodeRows: rows.count, canSpeculate: canSpeculate)
        let eligibleRows = rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }

        if mtp.planDepth > 0 {
            let depth = mtp.planDepth
            let tailDepth = eligibleRows.map { rec in
                let remaining = rec.request.maxTokens - rec.generatedTokenCount
                return mtp.hasValidCarry(for: rec) ? remaining : max(0, remaining - 1)
            }.min() ?? 0
            if tailDepth < depth {
                mtp.clampPlanDepth(to: tailDepth, reason: "tail_depth")
            }
        }
        if mtp.planDepth > 0 {
            let eligibleIDs = Set(eligibleRows.map(\.id))
            let stepTokens = rows.count + eligibleRows.count * mtp.planDepth
            let capacityTokens = rows.reduce(0) { total, rec in
                let count = 1 + (eligibleIDs.contains(rec.id) ? mtp.planDepth : 0)
                return total
                    + (rec.prefixReusePlan?.capacityTokensForChunk(
                        start: rec.numComputedTokens,
                        count: count) ?? count)
            }
            if stepTokens > scheduler.config.maxBatchedTokensPerStep {
                mtp.clampPlanDepth(to: 0, reason: "step_token_budget")
            } else if !(capacity?.hasHeadroom(additionalTokens: capacityTokens) ?? true) {
                mtp.clampPlanDepth(to: 0, reason: "step_kv_headroom")
            }
        }
        if mtp.planDepth > 0,
            eligibleRows.contains(where: { !mtp.hasValidCarry(for: $0) })
        {
            // Seeding is step-global. Mixing L=1 seed rows with verify rows
            // would recreate the batch-shape drift synchronized commits avoid.
            for rec in eligibleRows { mtp.invalidateCarry(rec.id) }
            mtp.recordControllerFallback("synchronized_seed")
        }

        for rec in rows {
            let storageEligible = kvStates[rec.id].map(Self.mtpStorageEligible) ?? false
            if mtpBasicEligible(rec), kvStates[rec.id] != nil, !storageEligible {
                mtp.recordSkip("kv_unsupported")
            }
            if mtp.planDepth == 0 || !mtpBasicEligible(rec) || !storageEligible {
                mtp.invalidateCarry(rec.id)
            }
        }
        if !rows.isEmpty, !withinBatchGate {
            for _ in rows { mtp.recordSkip("batch_gate") }
        }
    }

    private func mtpRowsCanSpeculate(_ rows: [CBv2ScheduledRequest]) -> Bool {
        !rows.isEmpty && rows.allSatisfy { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
    }

    /// True when this scheduler plan carries seed or verify work.
    func mtpRoundNeeded(_ plan: CBv2StepPlan) -> Bool {
        guard let mtp, mtp.planHasMTPWork else { return false }
        return plan.assignments.contains {
            mtp.roundMark(for: $0.id) != nil || mtp.isSeedMarked($0.id)
        }
    }
}
