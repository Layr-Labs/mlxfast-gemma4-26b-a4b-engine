// EngineLoopV2+MTP.swift
//
// Orchestrates one MTP engine step. Eligibility and planning, graph
// construction, finalize-time acceptance, and controller measurement live in
// focused sibling extensions. MTP steps never use compiled decode or chain.

import Foundation
import MLX

extension EngineLoopV2 {

    /// Execute a plan containing MTP work. Graph construction includes seed
    /// decodes, frozen-KV drafting, target-authoritative verification, ordinary
    /// decode neighbors, and per-request prefill chunks.
    func executeMTPRound(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        guard let mtp else { return executeMixed(plan) }
        // FAST-CANCEL entry gate: when every planned row already has a
        // pending caller cancel, nothing this round could draft, verify, or
        // commit is deliverable — every row is finished at the next
        // boundary. Drop the round HERE, before any carry is consumed or
        // any graph is built, mirroring the empty-work rollback path below.
        // (Deliberately not re-checked after the graph build: unlike the
        // plain chained round, the MTP build consumes per-row carries and
        // publishes KV-capture fence edges, so the only safe drop point is
        // before it starts.)
        if EngineLoopV2.fastCancelEnabled,
            pendingCancelsCover(plan.assignments.map(\.id))
        {
            CBv2EngageMark.once("fastcancel.mtp")
            scheduler.rollback(plan)
            return nil
        }
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds

        // A per-row reserve retry can demote a round after the step-global
        // preflight. Demote all round rows so the target still sees one L=1
        // batch, and refund every speculative suffix before graph build.
        let demoteAllRounds = plan.assignments.contains { assignment in
            guard let k = mtp.roundMark(for: assignment.id) else { return false }
            return assignment.numTokens != 1 + k
        }
        if demoteAllRounds {
            mtp.recordControllerFallback("step_reservation_race")
        }

        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let work = mtpPrepareRoundWork(
            plan, driver: mtp, demoteAllRounds: demoteAllRounds)
        guard !work.isEmpty else {
            // Undo optimistic scheduler advances before pending samples can
            // block waiting admission.
            scheduler.rollback(plan)
            return nil
        }

        let graph = mtpBuildRoundGraph(work, driver: mtp)
        scheduler.markPendingSamples(ids: graph.sampledRows)
        if let verify = graph.verify {
            scheduler.markPendingSamples(
                counts: verify.rows.map { (id: $0.id, count: 1 + verify.k) })
        }
        mtp.recordSeedSteps(graph.seedRows.count)

        asyncEval(graph.asyncEvalTargets)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.launch.total", seconds: CFAbsoluteTimeGetCurrent() - buildStart)
        }

        let step = CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: graph.sampledRows,
            sampledTokens: graph.sampledTokens,
            evalTargets: graph.prefillEvalTargets,
            wallStartedNanos: wallStartedNanos)
        step.logprobSegments = graph.logprobSegments
        if graph.verify != nil || !graph.seedRows.isEmpty {
            step.mtpRound = CBv2MTPRoundInFlight(
                verify: graph.verify,
                seedRows: graph.seedRows,
                seedHidden: graph.seedHidden)
        }
        return step
    }
}
