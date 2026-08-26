// Copyright © 2026 Eigen Labs.
//
// Pure, deterministic unit tests for the monotonic deadline-lease state
// machine and the request-derived absolute safety ceiling. No engine, no MLX,
// no real sleeps — every transition takes an explicit `ContinuousClock.Instant`
// so lease expiry is exercised exactly. Because time is only ever the injected
// monotonic instant (never `Date`), these tests are also wall-clock-jump immune
// by construction.

import XCTest

@testable import MLXLMCommon

final class CBv2DeadlineLeaseTests: XCTestCase {
    private let t0 = ContinuousClock.now

    private func at(_ seconds: Double) -> ContinuousClock.Instant {
        t0.advanced(by: .seconds(seconds))
    }

    private func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    private func makeLease(
        admission: Double = 120, prefill: Double = 120, decode: Double = 120,
        backpressure: Double = 120, safety: Double = 100_000
    ) -> CBv2RequestLeaseState {
        CBv2RequestLeaseState(
            now: t0, admissionLease: admission, prefillLease: prefill,
            decodeLease: decode, backpressureLease: backpressure,
            safety: .seconds(safety), computedTokens: 0, generatedTokens: 0)
    }

    // MARK: Safety ceiling formula

    func testSafetyCeilingComposition() {
        // queue(120) + prefill(0) + decode(0) + reprefill(0) + slack(60)
        XCTAssertEqual(
            seconds(
                CBv2SafetyCeiling.duration(
                    promptTokens: 0, maxTokens: 0, admissionLease: 120, decodeFloorTPS: 5)),
            180, accuracy: 1e-6)
        // decode bound = 16000 / 5 = 3200 dominates.
        XCTAssertEqual(
            seconds(
                CBv2SafetyCeiling.duration(
                    promptTokens: 0, maxTokens: 16000, admissionLease: 120, decodeFloorTPS: 5)),
            3380, accuracy: 1e-6)
        // prefill bound = 2 * (2000 / 200) = 20 (one re-prefill included).
        XCTAssertEqual(
            seconds(
                CBv2SafetyCeiling.duration(
                    promptTokens: 2000, maxTokens: 0, admissionLease: 120, decodeFloorTPS: 5)),
            200, accuracy: 1e-6)
    }

    func testSafetyCeilingSaturatesInsteadOfTrapping() {
        // An extreme accepted maxTokens over the minimum decode floor would
        // otherwise compute seconds past Duration's representable range and
        // TRAP in .seconds(_:) — a remotely reachable process crash. The
        // ceiling must saturate at the clamp and remain a real bound.
        let ceiling = CBv2SafetyCeiling.duration(
            promptTokens: Int.max, maxTokens: Int.max,
            admissionLease: 120, decodeFloorTPS: 0.000001)
        XCTAssertEqual(
            seconds(ceiling), CBv2SafetyCeiling.maxCeilingSeconds, accuracy: 1)
        // Ordinary requests are far below the clamp (formula unchanged).
        let ordinary = CBv2SafetyCeiling.duration(
            promptTokens: 2000, maxTokens: 16000, admissionLease: 120, decodeFloorTPS: 5)
        XCTAssertLessThan(seconds(ordinary), CBv2SafetyCeiling.maxCeilingSeconds / 1000)
    }

    func testSafetyCeilingIsGenerousVersusRealThroughput() {
        // A 16k-token generation at a REAL 55 tok/s finishes in ~291s; the
        // ceiling for the same request must be far larger (pathology only).
        let ceiling = seconds(
            CBv2SafetyCeiling.duration(
                promptTokens: 2000, maxTokens: 16000, admissionLease: 120, decodeFloorTPS: 5))
        XCTAssertGreaterThan(ceiling, 3000)
        XCTAssertGreaterThan(ceiling, 291 * 10)
    }

    // MARK: Admission lease

    func testNeverAdmittedExpiresWithAdmissionTimeout() {
        let lease = makeLease(admission: 120)
        XCTAssertNil(lease.expiredCause(now: at(119), isRunning: false, isPaused: false))
        XCTAssertEqual(
            lease.expiredCause(now: at(121), isRunning: false, isPaused: false),
            .admissionTimeout)
    }

    func testAdmissionEndsAtFirstAdmission() {
        var lease = makeLease(admission: 120, prefill: 120)
        lease.markAdmitted(now: at(10))
        XCTAssertTrue(lease.isAdmitted)
        // Past the old admission window it is NO LONGER an admission timeout;
        // the prefill progress lease (armed at admission) governs instead.
        XCTAssertEqual(
            lease.expiredCause(now: at(200), isRunning: true, isPaused: false),
            .prefillStall)
    }

    func testAdmissionDoesNotReArmAfterPreemption() {
        var lease = makeLease(admission: 120, safety: 100_000)
        lease.markAdmitted(now: at(5))
        // Admitted then preempted (back in waiting → isRunning:false). Long
        // after the original admission window, it must NOT admission-timeout
        // and must NOT be stall-faulted while awaiting re-admission — only the
        // absolute safety ceiling bounds it.
        XCTAssertNil(lease.expiredCause(now: at(500), isRunning: false, isPaused: false))
    }

    // MARK: Prefill progress lease

    func testPrefillProgressRefreshesThenStalls() {
        var lease = makeLease(prefill: 120)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(50), computedTokens: 512, generatedTokens: 0)
        XCTAssertNil(lease.expiredCause(now: at(160), isRunning: true, isPaused: false))
        lease.recordProgress(now: at(160), computedTokens: 1024, generatedTokens: 0)
        XCTAssertNil(lease.expiredCause(now: at(279), isRunning: true, isPaused: false))
        // No further prefill progress → prefill stall.
        XCTAssertEqual(
            lease.expiredCause(now: at(281), isRunning: true, isPaused: false),
            .prefillStall)
    }

    // MARK: Decode progress lease

    func testDecodeStalledExpiresWithDecodeStall() {
        var lease = makeLease(decode: 120)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 5, generatedTokens: 1)
        XCTAssertNil(lease.expiredCause(now: at(129), isRunning: true, isPaused: false))
        XCTAssertEqual(
            lease.expiredCause(now: at(131), isRunning: true, isPaused: false),
            .decodeStall)
    }

    func testSlowContinuousDecodeSurvivesFarPastLegacyWall() {
        // 100 tokens, one every 100s (< the 120s decode lease). Total 10,000s
        // — ~83× the legacy 120s wall — yet the request NEVER expires while it
        // keeps producing tokens. This is the exact case the old single wall
        // killed.
        var lease = makeLease(decode: 120, safety: 100_000)
        lease.markAdmitted(now: at(0))
        for i in 1...100 {
            let now = at(Double(i) * 100)
            lease.recordProgress(now: now, computedTokens: i + 4, generatedTokens: i)
            XCTAssertNil(
                lease.expiredCause(now: now, isRunning: true, isPaused: false),
                "token \(i) at \(Double(i) * 100)s must not expire while progressing")
        }
        // Right after the last token it is still alive.
        XCTAssertNil(lease.expiredCause(now: at(10_050), isRunning: true, isPaused: false))
        // Only once it stops producing tokens for a full lease does it stall.
        XCTAssertEqual(
            lease.expiredCause(now: at(10_121), isRunning: true, isPaused: false),
            .decodeStall)
    }

    // MARK: Absolute safety ceiling

    func testSafetyCeilingFiresDespiteHealthyProgressLease() {
        // Progress leases keep getting refreshed (never trip), but the absolute
        // ceiling still catches a pathological request that dribbles forever.
        var lease = makeLease(decode: 120, safety: 500)
        lease.markAdmitted(now: at(0))
        for i in 1...5 {
            lease.recordProgress(
                now: at(Double(i) * 100), computedTokens: i + 4, generatedTokens: i)
        }
        // At 400s the decode lease (last refresh 500 + 120) is healthy…
        XCTAssertNil(lease.expiredCause(now: at(400), isRunning: true, isPaused: false))
        // …but past the 500s absolute ceiling, safety fires even though the
        // progress lease has NOT tripped.
        XCTAssertEqual(
            lease.expiredCause(now: at(501), isRunning: true, isPaused: false),
            .safetyDeadline)
    }

    // MARK: Backpressure lease

    func testBackpressureLeaseIsSeparateFromProgress() {
        var lease = makeLease(decode: 120, backpressure: 120)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 5, generatedTokens: 1)
        lease.markPaused(now: at(20))
        // While paused the progress lease is NOT consulted — a slow consumer is
        // not an engine stall. Only the backpressure lease applies.
        XCTAssertNil(lease.expiredCause(now: at(139), isRunning: true, isPaused: true))
        XCTAssertEqual(
            lease.expiredCause(now: at(141), isRunning: true, isPaused: true),
            .backpressureTimeout)
    }

    func testResumeGrantsFreshProgressWindow() {
        var lease = makeLease(decode: 120, backpressure: 120)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 5, generatedTokens: 1)
        lease.markPaused(now: at(20))
        lease.markResumed(now: at(50))
        // Backpressure cleared; a fresh 120s decode window from resume.
        XCTAssertNil(lease.expiredCause(now: at(169), isRunning: true, isPaused: false))
        XCTAssertEqual(
            lease.expiredCause(now: at(171), isRunning: true, isPaused: false),
            .decodeStall)
    }

    // MARK: Preemption rewind (PR#82 review P1)

    func testPreemptionRewindGrantsFreshPrefillWindowAndResetsWatermark() {
        // A decode-phase request is preempted: the scheduler rewinds
        // numComputedTokens to 0 and the request must re-prefill from scratch
        // (generated tokens kept). Without markPreempted, the stale computed
        // watermark (1000) makes every re-prefill chunk at or below 1000 fail
        // the progress test while the old decode-phase deadline keeps running
        // — a healthy re-prefill would be killed as .decodeStall.
        var lease = makeLease(prefill: 120, decode: 120, safety: 100_000)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 1000, generatedTokens: 8)
        lease.markPreempted(now: at(50))
        // Fresh PREFILL window from the preemption instant, not the stale
        // decode deadline armed at t=10.
        XCTAssertNil(lease.expiredCause(now: at(169), isRunning: true, isPaused: false))
        // Re-prefill chunks BELOW the old watermark refresh the lease again.
        lease.recordProgress(now: at(169), computedTokens: 256, generatedTokens: 8)
        XCTAssertNil(lease.expiredCause(now: at(288), isRunning: true, isPaused: false))
        lease.recordProgress(now: at(288), computedTokens: 700, generatedTokens: 8)
        XCTAssertNil(lease.expiredCause(now: at(407), isRunning: true, isPaused: false))
        // A genuinely stalled re-prefill still dies, as .prefillStall.
        XCTAssertEqual(
            lease.expiredCause(now: at(409), isRunning: true, isPaused: false),
            .prefillStall)
        // Admission was NOT re-armed by the preemption.
        XCTAssertTrue(lease.isAdmitted)
    }

    func testPreemptionRewindThenNewTokenReturnsToDecodePhase() {
        var lease = makeLease(prefill: 120, decode: 120, safety: 100_000)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 1000, generatedTokens: 8)
        lease.markPreempted(now: at(50))
        // Re-prefill completes and the FIRST NEW generated token flips the
        // lease back to the decode phase with a fresh decode window.
        lease.recordProgress(now: at(100), computedTokens: 1009, generatedTokens: 9)
        XCTAssertNil(lease.expiredCause(now: at(219), isRunning: true, isPaused: false))
        XCTAssertEqual(
            lease.expiredCause(now: at(221), isRunning: true, isPaused: false),
            .decodeStall)
    }

    func testReadmissionAfterLongQueueWaitGrantsFreshWindow() {
        // Preempted at t=50 (fresh prefill window 50+120=170), then the row
        // waits behind higher-priority work for LONGER than the prefill lease.
        // Waiting rows are correctly stall-exempt (isRunning:false → only the
        // ceiling applies), but the window itself kept ticking — so without a
        // re-admission refresh the row returns to RUNNING pre-expired and is
        // killed as .prefillStall before its first re-prefill chunk.
        var lease = makeLease(prefill: 120, decode: 120, safety: 100_000)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 1000, generatedTokens: 8)
        lease.markPreempted(now: at(50))
        // Long stall-exempt wait: alive at t=400 only because isRunning:false.
        XCTAssertNil(lease.expiredCause(now: at(400), isRunning: false, isPaused: false))
        // WITHOUT the refresh it would die instantly on re-admission:
        XCTAssertEqual(
            lease.expiredCause(now: at(400), isRunning: true, isPaused: false),
            .prefillStall)
        // Re-admission grants a fresh window without re-arming admission.
        lease.markReadmitted(now: at(400))
        XCTAssertTrue(lease.isAdmitted)
        XCTAssertNil(lease.expiredCause(now: at(519), isRunning: true, isPaused: false))
        // A genuinely stalled re-prefill still dies at the fresh window's end.
        XCTAssertEqual(
            lease.expiredCause(now: at(521), isRunning: true, isPaused: false),
            .prefillStall)
        // No-op on a never-admitted row: first admission owns that transition.
        var fresh = makeLease(admission: 120)
        fresh.markReadmitted(now: at(60))
        XCTAssertFalse(fresh.isAdmitted)
        XCTAssertEqual(
            fresh.expiredCause(now: at(121), isRunning: false, isPaused: false),
            .admissionTimeout)
    }

    // MARK: Backpressure precedence over the ceiling (PR#82 review P2)

    func testPausedRequestPastCeilingIsBackpressureNotSafety() {
        // The ceiling's formula has no backpressure allowance, so a request
        // paused by a slow consumer late in its budget must get the FULL
        // backpressure lease and terminate as health-neutral
        // .backpressureTimeout — never be misclassified as .safetyDeadline.
        var lease = makeLease(decode: 120, backpressure: 120, safety: 100)
        lease.markAdmitted(now: at(0))
        lease.recordProgress(now: at(10), computedTokens: 5, generatedTokens: 1)
        lease.markPaused(now: at(90))
        // Past the 100s ceiling but inside the pause's backpressure lease:
        // still alive, and NOT a safety death.
        XCTAssertNil(lease.expiredCause(now: at(150), isRunning: true, isPaused: true))
        // The backpressure lease (90+120) governs the paused terminal.
        XCTAssertEqual(
            lease.expiredCause(now: at(211), isRunning: true, isPaused: true),
            .backpressureTimeout)
        // The ceiling still applies at the first RUNNING check after resume —
        // pauses cannot unbound a request's lifetime.
        var resumed = lease
        resumed.markResumed(now: at(150))
        XCTAssertEqual(
            resumed.expiredCause(now: at(151), isRunning: true, isPaused: false),
            .safetyDeadline)
        // Defensive: paused with NO armed backpressure lease (bookkeeping
        // drift) still falls back to the ceiling backstop.
        var drifted = makeLease(safety: 100)
        drifted.markAdmitted(now: at(0))
        XCTAssertEqual(
            drifted.expiredCause(now: at(101), isRunning: true, isPaused: true),
            .safetyDeadline)
    }

    // MARK: Legacy kill-switch wall

    func testLegacyWallIsSingleLifetimeIgnoringProgress() {
        var lease = CBv2RequestLeaseState.legacy(now: t0, wall: 100)
        // Progress does not extend the legacy wall.
        lease.recordProgress(now: at(50), computedTokens: 5, generatedTokens: 5)
        XCTAssertNil(lease.expiredCause(now: at(99), isRunning: true, isPaused: false))
        XCTAssertEqual(
            lease.expiredCause(now: at(101), isRunning: true, isPaused: false),
            .legacyRequestTimeout)
        // Applies to a never-admitted (waiting) legacy request too.
        XCTAssertEqual(
            lease.expiredCause(now: at(101), isRunning: false, isPaused: false),
            .legacyRequestTimeout)
    }
}
