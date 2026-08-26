import Foundation
import Testing

@testable import MLXFastRuntimeWorkerSupport

// DFlash became a first-class SCORED mode on this track (David ruling,
// 2026-08-26). Three properties of the scored path have to hold, and all three
// are WIRING properties — they are true today because of the ORDER and the
// SHARING of statements inside `Gemma4RuntimeWorker.handleWorkerRequest`, not
// because of any value a unit test can read back. A refactor that moved one
// line would break them silently and no existing test would notice.
//
// So they are pinned here as source-text tripwires, the style this repo already
// uses for cross-repo and cross-file invariants it cannot observe at runtime
// (`Gemma4AssistantHeadStagingTests.runtimeWorkerVerbIsWiredToTheSharedOptionSurface`,
// `MTPTimingRetirementTests`). CWD == package root under `swift test`, the same
// invariant `HarnessHashRootSetTests` relies on.
//
// WHAT THESE DO NOT COVER, and where that coverage lives: that the target bind
// actually refuses a mutated target is `TargetQuantizationBindTests`
// (`anInPlaceMutationAfterStartupIsCaughtOnlyByTheReCheck`,
// `aSubstitutedTargetInstanceIsRefusedByIdentity`); that a lawful head requant
// stays silent through both checks is the same suite's
// `aLawfullyRequantizedHeadIsSilentThroughBothChecks`; that a DFlash drafter
// requant is lawful and changes nothing on disk is
// `DFlashRequantOnLoadTests.theDrafterIsQuantizedOnLoadAndTheStagedBytesNeverChange`.
// These tests pin that the DFlash arm is wired INTO those guarantees.

@Suite("DFlash scored-window wiring")
struct DFlashScoredWindowWiringTests {

    private func workerSource() throws -> String {
        try String(
            contentsOfFile: "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift",
            encoding: .utf8)
    }

    /// The body of one `case "<verb>":` arm, up to the next `case "` at the same
    /// level. Crude but sufficient: the verb arms in `handleWorkerRequest` are
    /// written as a flat `switch` over string literals.
    private func caseBody(_ verb: String, in source: String) throws -> String {
        let opener = "case \"\(verb)\":"
        let start = try #require(
            source.range(of: opener), "the worker must still have a `\(opener)` arm")
        let rest = source[start.upperBound...]
        if let next = rest.range(of: "\n        case \"") {
            return String(rest[..<next.lowerBound])
        }
        return String(rest)
    }

    /// (1) TARGET BIND FIRST. `free_decode_begin` IS the measured prefill window
    /// for the scored regime — benchd brackets it — and it is the verb every
    /// DFlash scored run opens. The pre-measure re-check must therefore sit
    /// AHEAD of the route resolution and the session open, or a target mutated
    /// after startup would be inside the window before anything looked at it.
    ///
    /// Pinned as an ORDER, because that is what can regress: the call exists
    /// today, and moving it three lines down would leave every assertion about
    /// its existence green while defeating it entirely.
    @Test func theDFlashWindowRevalidatesTheTargetBeforeAnyRouting() throws {
        let body = try caseBody("free_decode_begin", in: try workerSource())

        let revalidate = try #require(
            body.range(of: "revalidateTargetForMeasuredWindow"),
            "the free_decode_begin arm must re-check the target bind")
        let routeResolution = try #require(
            body.range(of: "runtimeWorkerDecodeRoute"),
            "the free_decode_begin arm must resolve a decode route")
        let sessionOpen = try #require(
            body.range(of: "openSingleStreamFreeRunSession"),
            "the free_decode_begin arm must open the free-run session")

        #expect(
            revalidate.lowerBound < routeResolution.lowerBound,
            "the target re-check must precede route resolution")
        #expect(
            revalidate.lowerBound < sessionOpen.lowerBound,
            "the target re-check must precede the measured session open")
        // It must also name the phase it guards, so a refusal says which window
        // it fired in.
        #expect(body.contains("phase: \"free_decode_begin\""))
    }

    /// (2) NO PARALLEL PATH. serial, mtp and DFlash are measured through the
    /// SAME verbs on the SAME parent clock: benchd times `free_decode_begin`
    /// for prefill and `free_decode_run` for decode on every route, and the
    /// worker opens ONE width-1 session for all three. A DFlash-specific
    /// session opener or timing call would be a second measurement path whose
    /// numbers are not comparable to the arm it is scored against.
    ///
    /// The route is a SELECTION passed INTO the shared opener — `route:
    /// freeRoute` — not a branch to a different opener. That is the property.
    @Test func dflashSharesTheSingleStreamSessionAndParentClockVerbs() throws {
        let body = try caseBody("free_decode_begin", in: try workerSource())

        // Exactly ONE session opener in the arm, and the route is an argument
        // to it rather than a fork above it.
        let openerCount = body.components(separatedBy: "openSingleStreamFreeRunSession").count - 1
        #expect(
            openerCount == 1,
            "the free-run arm must open exactly one session for every route, got \(openerCount)")
        #expect(body.contains("route: freeRoute"))

        // No DFlash-specific opener or timing entry point anywhere in the
        // worker. If one is ever added, this goes red and the reviewer gets to
        // ask whether the two arms are still measured the same way.
        let source = try workerSource()
        for forbidden in [
            "openDFlashSession",
            "openSingleStreamDFlashSession",
            "measureDFlashDecode",
            "dflashDecodeWindow",
        ] {
            #expect(
                !source.contains(forbidden),
                "\(forbidden) would be a second measurement path for the DFlash arm")
        }
    }

    /// (3) THE BIND IS HEAD-AGNOSTIC BY CONSTRUCTION. A participant may
    /// re-quantize the DFlash drafter (participant-contract 4.4); the target may
    /// never be re-quantized. Those two coexist because the bind check looks at
    /// the TARGET ONLY — its parameter list has no drafter in it, so no head
    /// requant can reach it, and no future one can either without changing this
    /// signature.
    ///
    /// The behavioural halves are elsewhere and are cited at the top of this
    /// file; what is pinned here is the structural reason they compose.
    @Test func theTargetBindTakesNoDrafterAndSoCannotSeeAHeadRequant() throws {
        let bind = try String(
            contentsOfFile: "Sources/MLXFastHarness/Gemma4TargetQuantizationBind.swift",
            encoding: .utf8)
        let signature = try #require(
            bind.range(of: "func revalidateTargetForMeasuredWindow("),
            "the pre-measure re-check must still exist")
        let close = try #require(
            bind[signature.upperBound...].range(of: ") throws {"))
        let params = String(bind[signature.upperBound..<close.lowerBound])

        #expect(params.contains("verifiedTarget"))
        #expect(params.contains("currentTarget"))
        for headWord in ["drafter", "dflash", "DFlash", "head", "assistant"] {
            #expect(
                !params.contains(headWord),
                "the target bind must take no head input; found \(headWord) in: \(params)")
        }
    }
}
