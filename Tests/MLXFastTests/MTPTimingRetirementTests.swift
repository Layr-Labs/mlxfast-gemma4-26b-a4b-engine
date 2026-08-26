import Foundation
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Re-derived for the Gemma 4 26B A4B MTP arm (2026-08-23), renamed from
// `Gemma4MTPTimingRetirementTests` — dropped the Qwen-specific prefix the same
// way every other Qwen-era type this track carries forward was renamed
// (Qwen35Config -> Gemma4A4BConfig, etc.), since the invariant itself is
// track-agnostic.
//
// The ORIGINAL (deleted with the harness-port increment, 2026-08-22) checked
// four things specific to the Qwen-era dedicated MTP surface that does not
// exist any more: the `mtp-verify`/`mtp-timed` CLI verb pair, a wall clock in
// `Gemma4RuntimeMTPDriver.swift`, and named timing fields on `Gemma4MTPReport`.
// None of those symbols exist in the Gemma 4 architecture — the MTP arm folds
// into the shared generic spec-mode surface rather than shipping its own
// verb/driver/report trio — so this is a genuine re-derivation, not a rename:
// same INVARIANT (the engine's own MTP-relevant code never wall-clocks
// itself, and the wire carries no engine-measured MTP timing field; benchd
// owns the clock), checked against the files THIS increment actually touches.
//
// Scope: checked-in-source assertions only. No model, no GPU, no transport —
// matches the original's own stated scope.

private let mtpRelevantHarnessFiles = [
    "Sources/MLXFastHarness/MTPEnvelope.swift",
    "Sources/MLXFastHarness/Gemma4A4BAssistantHead.swift",
    "Sources/MLXFastHarness/RuntimeWorkerSpecConfig.swift",
    "Sources/MLXFastHarness/RuntimeWorkerGenericDispatch.swift",
    "Sources/MLXFastHarness/Gemma4RuntimeCohortDriver.swift",
    "Sources/MLXFastTrustedHarness/MTPEnvelope.swift",
]

// A wall clock in any of these is exactly the timing path that moved to
// benchd (Model 2: ALL measurement lives in benchd). None of the four are
// legitimate in engine-owned MTP code: `Date()` is the general-purpose
// wall-clock read this codebase's other retirement tripwires
// (`theTrustedMTPDriverTakesNoWallClock`, the Qwen-era original) already
// checked for; `ContinuousClock`/`DispatchTime`/`CFAbsoluteTimeGetCurrent`
// are its modern-Swift and Core Foundation equivalents, none of which this
// codebase's existing timing surface used, but all of which are the same
// class of regression if one appeared here.
private let forbiddenWallClockTokens = [
    "Date()",
    "ContinuousClock(",
    "DispatchTime.now(",
    "CFAbsoluteTimeGetCurrent(",
]

// SANCTIONED EXCEPTION (2026-08-23, per-stream timing instrumentation spec
// step 1, engine PR): `Gemma4RuntimeCohortDriver.swift` now reads
// `DispatchTime.now(` — one monotonic clock per phase, sampled at the
// cohort's existing per-slot commit points, to populate the additive
// `prefill_ns_by_stream` / `decode_ns_by_stream` wire fields. This does NOT
// reopen the doctrine this file polices: the driver computes nothing from
// the samples but a raw elapsed-ns subtraction (no sums, ratios, or
// seconds conversions — the spec's own "nothing else computed engine-side"
// constraint), the values are untrusted for scoring until benchd's
// attestation admits them, and every OTHER file in this list — plus every
// OTHER wall-clock token in this one — stays fully policed below.
private let sanctionedPerStreamTimingFile = "Sources/MLXFastHarness/Gemma4RuntimeCohortDriver.swift"
private let sanctionedPerStreamTimingToken = "DispatchTime.now("
// PINNED OCCURRENCE COUNT (creep-by-addition close-out, true-second review):
// a boolean "does the sanctioned token appear" check sanctions an unbounded
// NUMBER of reads once the first one is allowed through — a second,
// unsanctioned `DispatchTime.now(` added anywhere in this same file would
// not turn the tripwire red. The sanctioned sites are exactly the four
// sample points the instrumentation actually needs: the per-slot commit
// timestamp in `RuntimeWorkerCohortStreamCollector.append`, the
// cohort-prefill phase-start in `RuntimeWorkerCohortSession.init`, and the
// decode-phase-start in each of `runSerial` / `runMTP`. The test below
// asserts the EXACT count, not presence, so both a revert (count drops
// below 4) and an unsanctioned addition (count rises above 4) go red.
private let sanctionedPerStreamTimingOccurrenceCount = 4

// Named timing fields are the OTHER half of the old report-side check
// (`theMTPReportDeclaresNoTimingFields`): even without a wall clock in THIS
// engine, a field named like a timing measurement on a wire-facing or
// envelope type is the same "did the engine measure and report its own
// speed" regression benchd's clock-ownership rule exists to prevent.
private let forbiddenTimingFieldTokens = [
    "latencySeconds",
    "decodeSeconds",
    "roundRequestSeconds",
    "seedPrefillSeconds",
    "mtpElapsed",
    "mtpDurationSeconds",
]

/// Non-overlapping occurrence count of `token` in `text`. `DispatchTime.now(`
/// is a fixed literal with no internal repetition, so a plain
/// split-and-count is exact (no overlapping-match subtlety to worry about).
private func occurrenceCount(of token: String, in text: String) -> Int {
    text.components(separatedBy: token).count - 1
}

@Test
func mtpRelevantHarnessFilesTakeNoWallClock() throws {
    for path in mtpRelevantHarnessFiles {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        for token in forbiddenWallClockTokens {
            if path == sanctionedPerStreamTimingFile, token == sanctionedPerStreamTimingToken {
                // The sanctioned exception (see the constants' doc comments
                // above) is pinned by COUNT, not presence: a boolean check
                // would let an unsanctioned SECOND read of this token creep
                // in unnoticed (creep-by-addition) once the first one is
                // allowed through. Asserting the exact count catches BOTH
                // directions — a revert (count too low) and an unsanctioned
                // addition (count too high) — the same fail-loud posture
                // the rest of this file's boolean checks already have for
                // every other token.
                let count = occurrenceCount(of: token, in: text)
                #expect(
                    count == sanctionedPerStreamTimingOccurrenceCount,
                    "\(path) has \(count) occurrence(s) of the sanctioned per-stream-timing monotonic clock read (\(token)), expected exactly \(sanctionedPerStreamTimingOccurrenceCount) — either the instrumentation regressed/grew, or this pinned count is stale and should be re-derived and updated intentionally"
                )
                continue
            }
            #expect(
                !text.contains(token),
                "\(path) took a wall clock again (\(token)) — MTP timing belongs to benchd, not this engine"
            )
        }
    }
}

@Test
func mtpRelevantHarnessFilesDeclareNoTimingField() throws {
    for path in mtpRelevantHarnessFiles {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        for token in forbiddenTimingFieldTokens {
            #expect(
                !text.contains(token),
                "\(path) declares a timing-named field (\(token)) — the engine's self-measured timing surface stays retired"
            )
        }
    }
}

// The wire-facing effective-spec echo carries only what a caller asked to
// have acknowledged (mode + depth) — never a timing value. Encoded and
// checked at the JSON level, not just by field-name grep, so a future field
// renamed around the token list above still fails this if it is a duration.
@Test
func mtpEffectiveSpecEchoCarriesNoTimingField() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = String(
        decoding: try encoder.encode(RuntimeWorkerEffectiveSpec.mtp(depth: 2)),
        as: UTF8.self
    )
    #expect(json == #"{"mode":"mtp","mtp":{"depth":2}}"#)
}
