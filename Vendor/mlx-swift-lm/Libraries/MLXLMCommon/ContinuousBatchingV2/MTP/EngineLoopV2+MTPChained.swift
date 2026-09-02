// EngineLoopV2+MTPChained.swift
//
// Chained MTP rounds: round n+1 is built and submitted on top of round n's
// LAZY device outputs (seed token, carry hidden, position base), the way the
// chained decode step builds on the previous step's lazy tokens. The host
// finalize of round n then runs one round late and only does bookkeeping.

import Foundation
import MLX

extension EngineLoopV2 {

    static let mtpChainedRoundsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_CHAIN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    func mtpChainedRoundCandidate(_ previous: CBv2InFlightStep) -> (CBv2MTPRoundInFlight.Verify, [CBv2RequestID])? {
        guard Self.mtpChainedRoundsEnabled, let mtp,
            let verify = previous.mtpRound?.verify,
            verify.seedNext != nil, verify.carryHiddenNext != nil, verify.nextBase != nil,
            verify.rowStates.count == verify.rows.count,
            previous.discard.isEmpty
        else { return nil }
        let ids = verify.rows.map(\.id)
        guard ids.count == 8, ids.count <= mtp.config.maxSpeculativeBatch else { return nil }
        let running = scheduler.running
        guard running.count == ids.count else { return nil }
        for (rec, id) in zip(running, ids) {
            guard rec.id == id, !rec.cancelRequested, !rec.isPaused,
                rec.request.tokenConstraint == nil,
                rec.request.stopTokens.isEmpty, rec.request.stopStrings.isEmpty,
                rec.request.sampling.temperature == 0,
                kvStates[id] != nil
            else { return nil }
            guard rec.request.maxTokens - rec.generatedTokenCount - rec.pendingSamples >= 1 + verify.k
            else { return nil }
        }
        guard mtp.previewDecision(plannedDecodeRows: ids.count, canSpeculate: true).depth == verify.k
        else { return nil }
        return (verify, ids)
    }

    func launchChainedMTPRound(
        previous: CBv2InFlightStep, verify: CBv2MTPRoundInFlight.Verify, ids: [CBv2RequestID],
        driver mtp: CBv2MTPRoundDriver
    ) -> CBv2InFlightStep? {
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let k = verify.k
        var advanced: [CBv2RequestID] = []
        for id in ids {
            guard scheduler.advanceComputedForChainedRound(id: id, tokens: 1 + k) else {
                for done in advanced { scheduler.rollbackComputed(id: done, tokens: 1 + k) }
                return nil
            }
            advanced.append(id)
        }
        mtp.beginPlan(plannedDecodeRows: ids.count, canSpeculate: true)
        for id in ids { mtp.markRound(id, k: k) }

        var cacheInnerState: [MLXArray] = []
        let inputs = CBv2MTPVerifyInputs(
            ids: ids,
            states: ids.map { kvStates[$0]! },
            seedColumn: verify.seedNext!.reshaped([ids.count, 1]),
            carryHidden: verify.carryHiddenNext!,
            base: verify.nextBase!)
        guard let next = mtpBuildVerifyRound(
            inputs, k: k, driver: mtp, cacheInnerState: &cacheInnerState)
        else {
            for done in advanced { scheduler.rollbackComputed(id: done, tokens: 1 + k) }
            return nil
        }
        scheduler.markPendingSamples(counts: ids.map { (id: $0, count: 1 + k) })

        var targets: [MLXArray] = [next.acceptancePacket, next.lastHidden]
        if let seed = next.seedNext { targets.append(seed) }
        if let carry = next.carryHiddenNext { targets.append(carry) }
        if let base = next.nextBase { targets.append(base) }
        if !cacheInnerState.isEmpty {
            targets.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        asyncEval(targets)
        if CBv2StepProfiler.enabled {
            let elapsed = CFAbsoluteTimeGetCurrent() - buildStart
            CBv2StepProfiler.record("v2.mtp.chained.launch", seconds: elapsed)
            FileHandle.standardError.write(
                Data(String(format: "[mtp-round] launch %.1f ms\n", elapsed * 1000).utf8))
        }
        CBv2EngageMark.once("mtp-chained-round")
        let step = CBv2InFlightStep(
            participants: Set(ids), sampledRows: [], sampledTokens: nil, evalTargets: [],
            wallStartedNanos: wallStartedNanos)
        step.launchedByChain = true
        step.mtpRound = CBv2MTPRoundInFlight(verify: next, seedRows: [], seedHidden: nil)
        return step
    }
}
