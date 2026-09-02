// EngineLoopV2+MTP.swift
//
// Orchestrates one MTP engine step. Eligibility and planning, graph
// construction, finalize-time acceptance, and controller measurement live in
// focused sibling extensions. MTP steps never use compiled decode or chain.

import Foundation
import MLX

extension EngineLoopV2 {

    func executeMTPRound(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        guard let mtp else { return executeMixed(plan) }
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds

        let demoteAllRounds = plan.assignments.contains { assignment in
            guard let k = mtp.roundMark(for: assignment.id) else { return false }
            return assignment.numTokens != 1 + k
        }
        if demoteAllRounds {
            mtp.recordControllerFallback("step_reservation_race")
        }

        let profile = CBv2StepProfiler.enabled
        let buildStart = profile ? CFAbsoluteTimeGetCurrent() : 0
        let work = mtpPrepareRoundWork(
            plan, driver: mtp, demoteAllRounds: demoteAllRounds)
        guard !work.isEmpty else {
            scheduler.rollback(plan)
            return nil
        }
        let workEnd = profile ? CFAbsoluteTimeGetCurrent() : 0

        let graph = mtpBuildRoundGraph(work, driver: mtp)
        let graphEnd = profile ? CFAbsoluteTimeGetCurrent() : 0
        scheduler.markPendingSamples(ids: graph.sampledRows)
        if let verify = graph.verify {
            scheduler.markPendingSamples(
                counts: verify.rows.map { (id: $0.id, count: 1 + verify.k) })
        }
        mtp.recordSeedSteps(graph.seedRows.count)

        let launchMemory: CBv2MTPRoundInFlight.MemorySnapshot? =
            profile ? CBv2MTPRoundInFlight.MemorySnapshot() : nil
        asyncEval(graph.asyncEvalTargets)
        if profile {
            let now = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("v2.mtp.launch.total", seconds: now - buildStart)
            if graph.verify == nil, !graph.seedRows.isEmpty {
                Self.mtpSeedProfileLine("plan-work", seconds: workEnd - buildStart)
                Self.mtpSeedProfileLine("graph-build", seconds: graphEnd - workEnd)
                Self.mtpSeedProfileLine("submit", seconds: now - graphEnd)
                Self.mtpSeedProfileLine("launch-total", seconds: now - buildStart)
            } else if graph.verify != nil {
                FileHandle.standardError.write(
                    Data(String(format: "[mtp-round] first launch %.1f ms submit %.1f ms\n",
                        (now - buildStart) * 1000, (now - graphEnd) * 1000).utf8))
            }
        }

        let step = CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: graph.sampledRows,
            sampledTokens: graph.sampledTokens,
            evalTargets: graph.prefillEvalTargets,
            wallStartedNanos: wallStartedNanos)
        step.logprobSegments = graph.logprobSegments
        if graph.verify != nil || !graph.seedRows.isEmpty {
            let round = CBv2MTPRoundInFlight(
                verify: graph.verify,
                seedRows: graph.seedRows,
                seedHidden: graph.seedHidden)
            round.launchMemory = launchMemory
            step.mtpRound = round
        }
        return step
    }
}
