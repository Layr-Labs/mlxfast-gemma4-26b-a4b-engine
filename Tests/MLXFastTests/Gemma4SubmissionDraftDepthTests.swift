import Testing
@testable import MLXLMCommon
import MLXSpeculative

// The unified participant draft-depth lever. Both speculative arms read ONE
// editable constant, `submissionDraftDepth` (default 1), written identically:
//   MTP:    CBv2MTPRoundDriver.submissionDraftDepth (an adaptive CEILING).
//   DFlash: DFlashDraftModel.submissionDraftDepth (a FIXED block depth).
// These tests pin the defaults and the pure clamp/ceiling logic that turns the
// constant into the operating depth. The per-arm run behaviour (adaptive vs
// fixed block) is integration-level and covered by the ranked path; here we lock
// the arithmetic seams so a default or a clamp cannot drift silently.

@Test func submissionDraftDepthDefaultsAreOneOnBothArms() {
    #expect(CBv2MTPRoundDriver.submissionDraftDepth == 1)
    #expect(DFlashDraftModel.submissionDraftDepth == 1)
}

@Test func mtpEffectiveCeilingIsEnvelopeBoundedBySubmissionDepth() {
    // The MTP cap is min(envelope, submissionDraftDepth): a ceiling the adaptive
    // controller stays under, never a fixed pin. The envelope is never exceeded.
    let d = CBv2MTPRoundDriver.submissionDraftDepth
    #expect(CBv2MTPRoundDriver.effectiveDraftCeiling(envelopeMax: 3) == min(3, d))
    #expect(CBv2MTPRoundDriver.effectiveDraftCeiling(envelopeMax: 2) == min(2, d))
    #expect(CBv2MTPRoundDriver.effectiveDraftCeiling(envelopeMax: 0) == 0)
    // The constant can only ever LOWER the envelope cap, never raise it.
    #expect(CBv2MTPRoundDriver.effectiveDraftCeiling(envelopeMax: 2) <= 2)
}

@Test func dflashClampHonorsRequestBoundedByBothCeilings() {
    // At or below the ceilings, the requested depth runs as set. Both 6 and 8.
    #expect(DFlashDraftModel.clampDepth(requested: 1, drafterCeiling: 15, engineCeiling: 15) == 1)
    #expect(DFlashDraftModel.clampDepth(requested: 6, drafterCeiling: 15, engineCeiling: 15) == 6)
    #expect(DFlashDraftModel.clampDepth(requested: 8, drafterCeiling: 15, engineCeiling: 15) == 8)
    #expect(DFlashDraftModel.clampDepth(requested: 15, drafterCeiling: 15, engineCeiling: 15) == 15)
    // Above a ceiling clamps to it. It is not refused.
    #expect(DFlashDraftModel.clampDepth(requested: 20, drafterCeiling: 15, engineCeiling: 15) == 15)
    // The tighter of the drafter and engine ceilings wins.
    #expect(DFlashDraftModel.clampDepth(requested: 20, drafterCeiling: 7, engineCeiling: 15) == 7)
    #expect(DFlashDraftModel.clampDepth(requested: 20, drafterCeiling: 15, engineCeiling: 10) == 10)
    // Floored at 1. A non-positive request never disables the arm to depth 0.
    #expect(DFlashDraftModel.clampDepth(requested: 0, drafterCeiling: 15, engineCeiling: 15) == 1)
    #expect(DFlashDraftModel.clampDepth(requested: -3, drafterCeiling: 15, engineCeiling: 15) == 1)
}
