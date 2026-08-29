// Copyright © 2026 Eigen Labs.
//
// SCHED-001 — cold-cohort admission coalescing (host-side scheduling only).
//
// THE BUG. `EngineLoopV2.enqueue` hands a submitted request to the engine
// queue with `engineQueue.async` and does NOT kick a step; an idle engine is
// driven by `scheduleIdleRecheck()`, a `engineQueue.asyncAfter` poll armed
// `config.idleRecheckInterval` (1 ms) ahead. A closed cohort is submitted by
// one caller in a tight loop (`Gemma4RuntimeCohortDriver`) or by a task group
// (`Gemma4A4BRuntimeWeights.warmCohortShapes`). Whichever `enqueue` blocks
// land on the serial queue BEFORE that pending poll block are planned
// together; the rest wait for the next step. The cohort therefore prefills as
// k + (B - k) instead of one B-wide plan whenever the submit burst straddles
// the poll deadline.
//
// WHY IT COSTS. `EngineLoopV2.executeMixed` coalesces equal-length prompt
// chunks into ONE rectangular `[B, chunk]` trunk traversal, and skips the
// packed path entirely for a group of one (`group.rows.count > 1`). A 1 + 7
// split is therefore two full 30-layer traversals over the same 8192 tokens
// instead of one, and the solo row travels the unpacked per-row path.
//
// THE FIX. When the engine is COLD (nothing running, nothing in flight) and
// fewer rows are waiting than the configured cohort width, defer the step by
// a bounded, small amount and re-check, so the still-arriving cohort lands in
// one plan. Bounded three ways, so it can never wedge:
//
//   1. HARD CEILING — at most `window` from the first deferral, then the step
//      proceeds with whatever is queued.
//   2. QUIET RELEASE — if the waiting set has not GROWN for `quiet` AND no
//      submit is mid-flight (`CBv2EngineGauges.pendingSubmitCount`), the
//      burst is over; proceed. A lone request therefore costs at most
//      `quiet`, once, on a cold engine.
//   3. TARGET REACHED — `maxConcurrentRequests` rows are waiting; proceed
//      immediately (this is the cohort case: the wait ends within one poll
//      of the last submit).
//
// It never arms when `maxConcurrentRequests <= 1`, so every single-stream
// configuration is byte-identical to stock and pays nothing. It never arms
// with a running row, so decode — including cohort refill — is untouched. It
// reorders NOTHING: the same rows are planned in the same FIFO order by the
// same `SchedulerV2.plan()`, so tokens and per-stream order are unchanged.

import Foundation

/// SCHED-001 policy knobs. Read once, at first use.
public enum CBv2AdmissionCoalescing {
    private static func env(_ name: String) -> String? {
        ProcessInfo.processInfo.environment[name]
    }

    private static func flag(_ name: String, default defaultValue: Bool) -> Bool {
        guard let raw = env(name)?.trimmingCharacters(in: .whitespaces).lowercased(),
            !raw.isEmpty
        else { return defaultValue }
        return !["0", "off", "false", "no"].contains(raw)
    }

    private static func micros(_ name: String, default defaultValue: Int) -> Duration {
        let value = env(name).flatMap(Int.init).map { max(0, $0) } ?? defaultValue
        return .microseconds(value)
    }

    /// Kill switch. Default ON; `DARKBLOOM_CBV2_ADMISSION_COALESCE=0` reverts
    /// to stock admission exactly.
    public static let enabled = flag("DARKBLOOM_CBV2_ADMISSION_COALESCE", default: true)

    /// Hard ceiling on the total deferral for one cold cohort.
    static let window = micros("DARKBLOOM_CBV2_ADMISSION_COALESCE_US", default: 6000)

    /// Release early once the waiting set has been quiet (no new arrival) for
    /// this long — the "burst is over" heuristic that keeps a lone request
    /// from ever paying the full window.
    static let quiet = micros("DARKBLOOM_CBV2_ADMISSION_COALESCE_QUIET_US", default: 1500)

    /// Re-check cadence while deferring.
    static let poll = micros("DARKBLOOM_CBV2_ADMISSION_COALESCE_POLL_US", default: 200)

    /// `DARKBLOOM_CBV2_ADMISSION_TRACE=1` prints one stderr line per prefill
    /// forward with its real batch width — the diagnostic that proves (or
    /// disproves) the split. Never on in a measured run.
    static let trace = flag("DARKBLOOM_CBV2_ADMISSION_TRACE", default: false)

    static let pollSeconds: Double = {
        let c = poll.components
        return Double(c.seconds) + Double(c.attoseconds) * 1e-18
    }()

    private static let traceEpoch = DispatchTime.now().uptimeNanoseconds

    static func note(_ message: @autoclosure () -> String) {
        guard trace else { return }
        let epoch = traceEpoch
        let ms = Double(DispatchTime.now().uptimeNanoseconds &- epoch) / 1e6
        FileHandle.standardError.write(
            Data(String(format: "[adm] %10.3f %@\n", ms, message()).utf8))
    }

    // MARK: Observation seam (diagnostics + regression test)

    /// OFF in every measured run: one `Bool` load on the prefill-planning
    /// path, which already costs a 30-layer trunk traversal. Set by
    /// `Tests/MLXFastTests/AdmissionCoalescingTests.swift` so the admitted
    /// cohort width is observable without parsing stderr.
    nonisolated(unsafe) public static var captureWidths = false

    /// True only under a diagnostic run. Two plain `Bool` loads; the step
    /// path skips the (cheap, but not free) plan tally when this is false.
    @inline(__always)
    static var observing: Bool { trace || captureWidths }
    private static let captureLock = NSLock()
    nonisolated(unsafe) private static var capturedPlanRows: [Int] = []
    nonisolated(unsafe) private static var capturedForwardWidths: [Int] = []

    /// Prefill rows in one planned step (0 ⇒ not recorded).
    static func recordPlanPrefillRows(_ rows: Int, decodeRows: Int = 0) {
        if rows == 0 {
            if decodeRows > 0 { note("plan decode_rows=\(decodeRows)") }
            return
        }
        note("plan prefill_rows=\(rows) decode_rows=\(decodeRows)")
        guard captureWidths else { return }
        captureLock.lock()
        if capturedPlanRows.count < 4096 { capturedPlanRows.append(rows) }
        captureLock.unlock()
    }

    /// Batch width of ONE prompt forward, recorded where it actually runs.
    static func recordPrefillForwardWidth(_ width: Int, chunk: Int) {
        note("prefill forward batch=\(width) chunk=\(chunk)")
        guard captureWidths else { return }
        captureLock.lock()
        if capturedForwardWidths.count < 4096 { capturedForwardWidths.append(width) }
        captureLock.unlock()
    }

    public static func resetCapture() {
        captureLock.lock()
        capturedPlanRows = []
        capturedForwardWidths = []
        captureLock.unlock()
    }

    /// (prefill rows per planned step, batch width per prompt forward).
    public static func capturedWidths() -> (planRows: [Int], forwardWidths: [Int]) {
        captureLock.lock()
        defer { captureLock.unlock() }
        return (capturedPlanRows, capturedForwardWidths)
    }
}
