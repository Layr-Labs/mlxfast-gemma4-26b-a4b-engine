import Foundation
import Testing

// Re-derived for the Gemma 4 26B A4B MTP arm (2026-08-23). The ORIGINAL of
// this test (deleted with the harness-port increment, 2026-08-22) compared a
// dedicated pair of files — `Sources/MLXFastHarness/Gemma4RuntimeMTPWorker.swift`
// and its `Sources/MLXFastTrustedHarness` twin — because the Qwen-era MTP arm
// shipped its own worker/driver/report file trio, each mirrored across both
// harness trees behind the `#if !MLXFAST_TRUSTED_HARNESS` guard exactly like
// the DFlash twins still do.
//
// The Gemma 4 MTP arm does NOT re-create that dedicated-file architecture —
// its spec-mode wiring folds into the existing shared surface
// (RuntimeWorkerSpecConfig.swift, Gemma4RuntimeWorker.swift), which are NOT
// full-file twins between the two harness trees (they differ by more than a
// guard wrapper: 3165 participant lines vs 2914 trusted lines, scattered
// inline `#if !MLXFAST_TRUSTED_HARNESS` blocks, hand-synced). The one new file
// this increment adds that genuinely IS MTP driver logic AND MLX-adjacent
// (imports MLXLMCommon for `CBv2MTPConfig`) is `MTPEnvelope.swift` — the
// envelope-knob declaration/seal/refusal trust boundary — and it follows the
// SAME full-file-twin convention the old MTP files and the still-current
// DFlash twins use: an identical participant copy, wrapped in the trusted
// guard, so the two files cannot silently diverge. This test is the
// re-derived tripwire for THAT pair, matching
// `dflashWorkerTwinsDifferOnlyByTheTrustedHarnessGuard`'s shape
// byte-for-byte, per the original's own stated design (it mirrors the DFlash
// twin test "so the MTP pair can no longer drift silently").
//
// Scope note: pure source-text comparison. No MLX device work, no weights, no
// hidden material — the two files are read by relative path (CWD == package
// root under `swift test`, the same invariant HarnessHashRootSetTests relies
// on).

private let mtpEnvelopeTrustedPath =
    "Sources/MLXFastTrustedHarness/MTPEnvelope.swift"
private let mtpEnvelopeParticipantPath =
    "Sources/MLXFastHarness/MTPEnvelope.swift"

private let trustedHarnessGuardOpen = "#if !MLXFAST_TRUSTED_HARNESS\n"
private let trustedHarnessGuardClose = "#endif\n"

private func sourceText(_ path: String) throws -> String {
    try String(contentsOfFile: path, encoding: .utf8)
}

// Sources/MLXFastTrustedHarness/MTPEnvelope.swift and
// Sources/MLXFastHarness/MTPEnvelope.swift are TWINS: the same MTP
// envelope-knob declaration/seal/refusal logic compiled into the trusted
// harness target and into the participant runtime-worker target. Any edit
// that lands in one and not the other silently changes what the worker
// actually pins relative to what the trusted build would see if it ever
// needed to read the same constants. The twin's ONLY legitimate difference is
// the `#if !MLXFAST_TRUSTED_HARNESS` wrapper that compiles the whole body out
// of the trusted target (which links no MLX and no MLXLMCommon).
@Test
func mtpEnvelopeTwinsDifferOnlyByTheTrustedHarnessGuard() throws {
    let trusted = try sourceText(mtpEnvelopeTrustedPath)
    let participant = try sourceText(mtpEnvelopeParticipantPath)

    // The exact byte relationship, which is stronger than a line-by-line
    // comparison: the trusted copy is the participant copy wrapped in the
    // guard, with nothing else added, removed, or reordered.
    #expect(
        trusted
            == trustedHarnessGuardOpen + participant + trustedHarnessGuardClose,
        """
        MTP envelope twins diverged beyond the #if !MLXFAST_TRUSTED_HARNESS \
        guard; edits must land in BOTH files with identical content
        """
    )
    // Stated as a line count too, because the guard delta is exactly two lines
    // and a size drift is the cheapest signal that an edit landed in one twin.
    let trustedLines = trusted.split(separator: "\n", omittingEmptySubsequences: false)
    let participantLines = participant.split(
        separator: "\n",
        omittingEmptySubsequences: false
    )
    #expect(trustedLines.count == participantLines.count + 2)
    #expect(trustedLines.first == "#if !MLXFAST_TRUSTED_HARNESS")
}
