// EngineLoopV2+MTPMeasurement.swift
//
// Host-only depth-controller cost attribution.

extension EngineLoopV2 {

    func mtpMeasurement(for plan: CBv2StepPlan) -> CBv2MTPStepMeasurement? {
        guard let mtp, mtp.controllerMeasurementEligible,
            mtp.planDecision.decodeRowBucket > 0
        else { return nil }
        let pureDecode = plan.assignments.allSatisfy { assignment in
            guard let rec = scheduler.record(for: assignment.id) else { return false }
            let before = rec.numComputedTokens - assignment.numTokens
            let finalTokenIsImageSpan =
                multimodalByID[assignment.id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            return !finalTokenIsImageSpan && rec.effectiveTokenCount - before == 1
        }
        return CBv2MTPStepMeasurement(
            decision: mtp.planDecision,
            actualDepth: 0,
            costEligible: pureDecode,
            chained: false,
            seedOnly: false)
    }

    func attachMTPMeasurement(
        _ measurement: CBv2MTPStepMeasurement?, to step: CBv2InFlightStep?,
        chained: Bool
    ) {
        guard let measurement, let step else { return }
        let actualDepth = step.mtpRound?.verify?.k ?? 0
        let seedOnly = step.mtpRound != nil && step.mtpRound?.verify == nil
        step.mtpMeasurement = CBv2MTPStepMeasurement(
            decision: measurement.decision,
            actualDepth: actualDepth,
            costEligible: measurement.costEligible,
            chained: chained,
            seedOnly: seedOnly)
    }
}
