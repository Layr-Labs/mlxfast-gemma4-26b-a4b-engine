import Foundation
@testable import MLXFastCore
import Testing

// Acceptance bands: a robust reference B is calibrated once from a fleet (drop the
// single slowest, average the rest); each run's single measurement is then checked
// against B with per-axis tolerances. Prefill is +/-3% (symmetric health gate);
// decode is +1% regression / -2.5% gain (tight on regressions, per-submission gain
// capped). Tolerances derived from the gemma4 2026-08-25 calibration session --
// see the BAND DERIVATION block in Constants.swift. Numbers anchored on the real
// retiredvm fresh-VM run 28893815980 (prefills: 5 tight at ~0.0106, one spike at
// 0.01532).

private let freshVMPrefills = [
    0.010556244302734375, 0.01071322314453125, 0.010636167642578125,
    0.010626223958984375, 0.01532235343359375, 0.01049330069921875,
]

@Test
func robustReferenceDropsTheSlowestAndAverages() {
    let B = AcceptanceBand.robustReference(samples: freshVMPrefills)
    #expect(B != nil)
    // mean of the 5 after dropping the 0.01532 spike
    #expect(abs((B ?? 0) - 0.0106048) < 1e-5)
}

@Test
func robustReferenceRejectsTooFewOrInvalid() {
    #expect(AcceptanceBand.robustReference(samples: [0.0106, 0.0106]) == nil)
    #expect(AcceptanceBand.robustReference(samples: [0.0106, 0.0, 0.0106]) == nil)
    #expect(AcceptanceBand.robustReference(samples: [0.0106, .nan, 0.0106]) == nil)
}

// MARK: - Prefill band (+/-3% symmetric)

private func checkPrefill(_ value: Double, _ reference: Double) -> AcceptanceBandResult {
    AcceptanceBand.check(
        value: value, reference: reference,
        upTolerance: MLXFastConstants.prefillBandUpTolerance,
        downTolerance: MLXFastConstants.prefillBandDownTolerance, label: "prefill"
    )
}

@Test
func prefillAcceptsAtReferenceAndModestlyFaster() {
    let B = 0.0106
    #expect(checkPrefill(B, B).passed)          // exactly at B
    #expect(checkPrefill(B * 1.02, B).passed)   // +2% slow: ok
    #expect(checkPrefill(B * 0.98, B).passed)   // -2% fast: ok
}

@Test
func prefillFailsWhenMoreThan3PercentSlow() {
    let B = 0.0106
    #expect(checkPrefill(B * 1.02, B).passed)   // +2% slow: still ok
    #expect(!checkPrefill(B * 1.031, B).passed) // +3.1% slow: fail
    #expect(checkPrefill(B * 1.031, B).reason.contains("slowdown"))
    // the real 0.01532 spike vs B=0.010605 is +44% -> fail
    #expect(!checkPrefill(0.01532235343359375, 0.0106048).passed)
}

@Test
func prefillFailsWhenMoreThan3PercentFast() {
    let B = 0.0106
    #expect(!checkPrefill(B * 0.969, B).passed) // -3.1% fast: fail
    #expect(checkPrefill(B * 0.969, B).reason.contains("chunk"))
}

// MARK: - Decode band (+1% regression / -2.5% gain)

private func checkDecode(_ value: Double, _ reference: Double) -> AcceptanceBandResult {
    AcceptanceBand.check(
        value: value, reference: reference,
        upTolerance: MLXFastConstants.decodeBandUpTolerance,
        downTolerance: MLXFastConstants.decodeBandDownTolerance, label: "decode"
    )
}

@Test
func decodeAcceptsSmallRegressionAndUpTo2Point5PercentGain() {
    let B = 0.1336
    #expect(checkDecode(B, B).passed)           // exactly at B
    #expect(checkDecode(B * 1.005, B).passed)   // +0.5% slower: within the regression cap, ok
    #expect(checkDecode(B * 0.98, B).passed)    // -2% faster: within the 2.5% gain cap, ok
}

@Test
func decodeFailsRegressionBeyond1Percent() {
    let B = 0.1336
    #expect(!checkDecode(B * 1.011, B).passed)  // +1.1% slower: fail (decode is the scored axis)
    #expect(checkDecode(B * 1.011, B).reason.contains("slowdown"))
}

@Test
func decodeFailsGainBeyond2Point5Percent() {
    let B = 0.1336
    #expect(!checkDecode(B * 0.974, B).passed)  // -2.6% faster: too big for one submission
    #expect(checkDecode(B * 0.974, B).reason.contains("chunk"))
}

@Test
func checkRejectsNonFinite() {
    #expect(!checkPrefill(0.0, 0.0106).passed)
    #expect(!checkPrefill(0.0106, 0.0).passed)
    #expect(!checkPrefill(.nan, 0.0106).passed)
}

@Test
func bandTolerancesMatchConstants() {
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.03)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.03)
    #expect(MLXFastConstants.decodeBandUpTolerance == 0.01)
    #expect(MLXFastConstants.decodeBandDownTolerance == 0.025)
}
