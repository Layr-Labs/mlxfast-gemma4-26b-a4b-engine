// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — monotonic per-request deadline leases.
//
// The pre-CBv2 scheduler applied its 120s timeout ONLY to requests that had
// not yet been admitted to engine work, using a monotonic clock. The CBv2
// conversion regressed that into a single wall-clock (`Date`) deadline over
// the WHOLE engine lifetime (waiting + prefill + decode + preemption +
// backpressure), so a healthy request still producing tokens was killed at
// 120s (a 16k-token generation needs ~290s). See
// docs/reports/2026-07-20-generation-deadline-incident-and-redesign.md.
//
// This file replaces that single flat wall with independent, MONOTONIC
// (`ContinuousClock`, never `Date`) leases, each with a typed terminal cause:
//
//   * admission lease   — bounds time before the request begins engine work;
//                         ends permanently at first admission; never re-arms
//                         after preemption. Cause: `.admissionTimeout`.
//   * prefill lease     — refreshed on confirmed finalized prefill progress.
//                         Cause: `.prefillStall`.
//   * decode lease      — refreshed on confirmed sampled/finalized token
//                         progress; a long generation that keeps producing
//                         tokens NEVER expires. Cause: `.decodeStall`.
//   * backpressure lease— bounds time a request may sit paused on downstream
//                         buffer pressure (health-neutral). Cause:
//                         `.backpressureTimeout`.
//   * safety ceiling    — a generous, request-derived ABSOLUTE bound that only
//                         catches pathology (indefinite dribble / logic
//                         errors), never the normal cutoff. Cause:
//                         `.safetyDeadline`.
//
// A kill-switch (`CBv2EngineLoopConfig.useLegacyRequestTimeout`) restores the
// legacy single total-lifetime wall for rollback; its terminal is the original
// `.error("request exceeded Ns deadline")` string (cause `.legacyRequestTimeout`
// when surfaced typed).
//
// This module is PURE, SYNCHRONOUS bookkeeping (no MLX, no I/O): the lease
// state machine and the safety-ceiling formula are fully unit-testable without
// model weights or a running engine.

import Foundation

// MARK: - Injectable monotonic clock

/// A monotonic clock source. Production uses `ContinuousClock`; tests inject a
/// fake so lease expiry is exercised deterministically with no real sleeps.
///
/// `ContinuousClock` is deliberate: it is immune to wall-clock (`Date`)
/// adjustments — an NTP step or manual date change can never make a lease
/// expire early or late.
public struct CBv2Clock: Sendable {
    public let now: @Sendable () -> ContinuousClock.Instant
    public init(now: @escaping @Sendable () -> ContinuousClock.Instant) {
        self.now = now
    }
    /// The live monotonic clock.
    public static let continuous = CBv2Clock { ContinuousClock.now }
}

// MARK: - Typed terminal cause

/// Machine-readable cause for a platform/engine-initiated terminal that is NOT
/// a natural `stop`/`length` completion or an explicit cancellation.
///
/// Carried on `CBv2FinishReason.terminal(cause:message:)` so the provider
/// bridge can classify health, retry, and billing WITHOUT parsing an error
/// string (the previous behavior, which flattened every deadline into a
/// generic string that the coordinator could not distinguish from an engine
/// fault). The reconciled usage rides the SAME `CBv2Event.finished(reason:
/// usage:)` envelope, so a typed terminal always carries the request's
/// prompt/completion token counts.
public enum CBv2TerminalCause: Sendable, Equatable {
    /// The request never began engine work before the admission lease expired
    /// (queue/admission timeout). Retryable capacity outcome; health-neutral.
    case admissionTimeout
    /// Prompt prefill stopped making confirmed finalized progress.
    case prefillStall
    /// Decode stopped making confirmed finalized token progress.
    case decodeStall
    /// The request-derived absolute safety ceiling fired — a pathology guard,
    /// never the normal completion cutoff.
    case safetyDeadline
    /// The request sat paused on output-stream/WebSocket backpressure longer
    /// than the backpressure lease. Health-neutral (downstream pressure, not
    /// an engine fault).
    case backpressureTimeout
    /// The single-step engine-health watchdog fired (a step wedged inside a
    /// blocking eval). Provider-health fault.
    case watchdog
    /// The legacy single total-lifetime wall fired (kill-switch behavior).
    case legacyRequestTimeout

    /// Human-readable diagnostic for logs/telemetry (never a wire contract —
    /// consumers switch on the typed case, not this text).
    public var diagnostic: String {
        switch self {
        case .admissionTimeout: return "admission lease expired before engine admission"
        case .prefillStall: return "prefill made no confirmed progress within the prefill lease"
        case .decodeStall: return "decode made no confirmed token progress within the decode lease"
        case .safetyDeadline: return "request exceeded its absolute safety ceiling"
        case .backpressureTimeout: return "request paused on backpressure past the backpressure lease"
        case .watchdog: return "engine step exceeded the watchdog timeout"
        case .legacyRequestTimeout: return "request exceeded the legacy total-lifetime deadline"
        }
    }
}

// MARK: - Request-derived absolute safety ceiling

/// The generous absolute bound of last resort. It is NOT the ordinary
/// generation cutoff — the progress leases are. It only catches pathology:
/// indefinite token dribble, missing progress events, or logic errors that
/// keep a request alive without real progress.
///
/// Measured from engine enqueue:
///
///   queue allowance                       (= admission lease)
/// + conservative prefill bound(promptTokens)
/// + conservative decode bound(maxOutputTokens, floor TPS)
/// + bounded preemption slack
///
/// The throughput floors are DELIBERATELY far below any real model so the
/// ceiling never fires for a healthy request. Real Apple-silicon decode runs
/// ~30–120 tok/s and prefill runs thousands of tok/s; the floors here are
/// 5 tok/s decode and 200 tok/s prefill (a 6–24× and 10×+ margin). A 32k-token
/// generation at a real 55 tok/s finishes in ~600s, while this ceiling for the
/// same request is well over an hour — pathology only.
public enum CBv2SafetyCeiling {
    /// Conservative prefill throughput floor (tokens/second). Real prefill is
    /// orders of magnitude faster; this only bounds pathology.
    static let prefillFloorTPS: Double = 200
    /// Bounded slack for one full post-preemption re-prefill plus margin.
    static let preemptionSlackSeconds: Double = 60
    /// Saturation bound for the computed ceiling (~31.7 years). `Duration`
    /// construction traps on values beyond its representable range, and an
    /// extreme accepted `maxTokens` (fixed-window backends cap retained KV by
    /// the window, not the request allowance) over a tiny decode floor can
    /// otherwise produce seconds past Int64 range — a remotely reachable
    /// process crash. Any request hitting this clamp is already bounded far
    /// beyond every real workload; the ceiling stays an absolute bound.
    static let maxCeilingSeconds: Double = 1e9

    /// Absolute ceiling duration, measured from enqueue.
    ///
    /// - Parameters:
    ///   - promptTokens: prompt length (drives the prefill bound).
    ///   - maxTokens: requested output allowance (drives the decode bound).
    ///   - admissionLease: the queue/admission allowance, reused as the queue
    ///     portion of the ceiling.
    ///   - decodeFloorTPS: conservative decode throughput floor (tokens/s).
    public static func duration(
        promptTokens: Int, maxTokens: Int,
        admissionLease: TimeInterval, decodeFloorTPS: Double
    ) -> Duration {
        let prompt = Double(max(0, promptTokens))
        let output = Double(max(0, maxTokens))
        let floorDecode = max(0.01, decodeFloorTPS)
        let prefillBound = prompt / prefillFloorTPS
        let decodeBound = output / floorDecode
        let seconds =
            max(0, admissionLease)
            + prefillBound
            + decodeBound
            + prefillBound  // one conservative re-prefill after a late preemption
            + preemptionSlackSeconds
        // Saturate before constructing: .seconds(_:) traps past Duration's
        // range, and `seconds` is attacker-influenceable through an accepted
        // extreme maxTokens (see maxCeilingSeconds).
        return .seconds(min(seconds, maxCeilingSeconds))
    }
}

// MARK: - Per-request lease state machine

/// The monotonic lease state for one in-flight request. Engine-thread-confined
/// (mutated only from the engine step thread), but pure and independently
/// unit-testable: every transition takes an explicit `now` instant.
struct CBv2RequestLeaseState {
    enum Phase: Equatable { case prefill, decode }

    /// When non-nil the request is under the legacy single-wall kill-switch:
    /// only this wall applies and it maps to `.legacyRequestTimeout`.
    private(set) var legacyWall: ContinuousClock.Instant?

    /// Admission lease deadline; nil once the request has been admitted. Never
    /// re-armed after preemption (admission is a one-time transition).
    private(set) var admissionDeadline: ContinuousClock.Instant?

    /// Current progress-lease deadline and the phase it represents.
    private(set) var progressDeadline: ContinuousClock.Instant
    private(set) var phase: Phase

    /// Backpressure lease deadline; active only while the request is paused.
    private(set) var backpressureDeadline: ContinuousClock.Instant?

    /// Absolute safety ceiling (set once at enqueue).
    private(set) var safetyDeadline: ContinuousClock.Instant

    private let prefillLease: Duration
    private let decodeLease: Duration
    private let backpressureLease: Duration

    /// Confirmed progress watermarks — a lease refreshes only when one of these
    /// actually advances (never from optimistic scheduler planning).
    private var lastComputedTokens: Int
    private var lastGeneratedTokens: Int

    /// Whether the request has begun engine work.
    var isAdmitted: Bool { legacyWall == nil && admissionDeadline == nil }
    var isLegacy: Bool { legacyWall != nil }

    // MARK: Construction

    /// New-behavior lease set armed at enqueue.
    init(
        now: ContinuousClock.Instant,
        admissionLease: TimeInterval,
        prefillLease: TimeInterval,
        decodeLease: TimeInterval,
        backpressureLease: TimeInterval,
        safety: Duration,
        computedTokens: Int,
        generatedTokens: Int
    ) {
        self.legacyWall = nil
        self.admissionDeadline = now.advanced(by: .seconds(admissionLease))
        self.prefillLease = .seconds(prefillLease)
        self.decodeLease = .seconds(decodeLease)
        self.backpressureLease = .seconds(backpressureLease)
        self.phase = .prefill
        // Placeholder until first admission arms the real prefill lease.
        self.progressDeadline = now.advanced(by: .seconds(prefillLease))
        self.backpressureDeadline = nil
        self.safetyDeadline = now.advanced(by: safety)
        self.lastComputedTokens = computedTokens
        self.lastGeneratedTokens = generatedTokens
    }

    /// Legacy kill-switch lease: a single total-lifetime wall.
    static func legacy(
        now: ContinuousClock.Instant, wall: TimeInterval
    ) -> CBv2RequestLeaseState {
        var s = CBv2RequestLeaseState(
            now: now, admissionLease: wall, prefillLease: wall, decodeLease: wall,
            backpressureLease: wall, safety: .seconds(wall),
            computedTokens: 0, generatedTokens: 0)
        s.legacyWall = now.advanced(by: .seconds(wall))
        s.admissionDeadline = nil
        return s
    }

    // MARK: Transitions

    /// First admission: permanently ends the admission lease and arms the
    /// prefill progress lease. Idempotent and never re-arms after preemption.
    mutating func markAdmitted(now: ContinuousClock.Instant) {
        guard legacyWall == nil, admissionDeadline != nil else { return }
        admissionDeadline = nil
        phase = .prefill
        progressDeadline = now.advanced(by: prefillLease)
    }

    /// Record CONFIRMED finalized progress (called from `finalize`, never from
    /// optimistic planning). A confirmed sampled/generated token refreshes the
    /// decode lease; a confirmed prefill-chunk advance refreshes the prefill
    /// lease. Progress also implies admission.
    mutating func recordProgress(
        now: ContinuousClock.Instant, computedTokens: Int, generatedTokens: Int
    ) {
        guard legacyWall == nil else { return }
        if admissionDeadline != nil { markAdmitted(now: now) }
        if generatedTokens > lastGeneratedTokens {
            lastGeneratedTokens = generatedTokens
            lastComputedTokens = computedTokens
            phase = .decode
            progressDeadline = now.advanced(by: decodeLease)
        } else if computedTokens > lastComputedTokens {
            lastComputedTokens = computedTokens
            phase = .prefill
            progressDeadline = now.advanced(by: prefillLease)
        }
    }

    /// The request paused on backpressure: arm the backpressure lease. The
    /// progress lease is not consulted while paused (a slow consumer is not an
    /// engine stall).
    mutating func markPaused(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        backpressureDeadline = now.advanced(by: backpressureLease)
    }

    /// The request resumed from backpressure: clear the backpressure lease and
    /// grant a fresh progress window (the paused interval was not the engine's
    /// fault, so it must not count as a progress stall).
    mutating func markResumed(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        backpressureDeadline = nil
        progressDeadline = now.advanced(by: phase == .prefill ? prefillLease : decodeLease)
    }

    /// Preemption rewound computation: the scheduler reset the record's
    /// computed-token count to zero and the request must re-prefill from
    /// scratch (generated tokens are kept). Reset the computed watermark and
    /// grant a fresh PREFILL progress window so the re-prefill's confirmed
    /// chunks refresh the lease — without this, every re-prefill chunk at or
    /// below the old watermark fails the `computedTokens >` test while the
    /// stale (possibly decode-phase) progress deadline keeps running, and a
    /// sufficiently long re-prefill is falsely killed as a stall. Never
    /// re-arms admission (a preempted row stays admitted); backpressure state
    /// is untouched (output-stream pause bookkeeping is orthogonal and its
    /// pause/resume events keep driving markPaused/markResumed).
    mutating func markPreempted(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        lastComputedTokens = 0
        phase = .prefill
        progressDeadline = now.advanced(by: prefillLease)
    }

    /// Re-admission from waiting (a preempted or capacity-requeued row
    /// entering engine work again): the fresh window granted at demotion has
    /// been ticking through the stall-exempt queue wait — progress checks are
    /// correctly suspended for waiting rows, but the deadline itself is not —
    /// so a wait longer than the lease leaves it pre-expired and the row
    /// would be killed as a stall before its first re-prefill chunk
    /// finalizes. Grant a fresh progress window for the current phase WITHOUT
    /// touching the (permanently ended) admission lease or backpressure
    /// state. No-op for a never-admitted row (first admission owns that
    /// transition).
    mutating func markReadmitted(now: ContinuousClock.Instant) {
        guard legacyWall == nil, admissionDeadline == nil else { return }
        progressDeadline = now.advanced(by: phase == .prefill ? prefillLease : decodeLease)
    }

    // MARK: Expiry

    /// The typed cause if any lease has expired, else nil.
    ///
    /// - Parameters:
    ///   - isRunning: the scheduler has this request in the RUNNING set
    ///     (progress-stall applies only to actively-running rows; a
    ///     preempted/requeued row awaiting re-admission is bounded only by the
    ///     safety ceiling, never faulted as a stall).
    ///   - isPaused: the request is paused on backpressure.
    func expiredCause(
        now: ContinuousClock.Instant, isRunning: Bool, isPaused: Bool
    ) -> CBv2TerminalCause? {
        if let wall = legacyWall {
            return now > wall ? .legacyRequestTimeout : nil
        }
        if let admission = admissionDeadline {
            // Never admitted: admission lease governs (it is always earlier
            // than the safety ceiling, which includes the admission allowance).
            if now > admission { return .admissionTimeout }
            if now > safetyDeadline { return .safetyDeadline }
            return nil
        }
        // Admitted. While genuinely paused on backpressure, the backpressure
        // lease governs and takes precedence over the safety ceiling: the
        // ceiling's formula includes no backpressure allowance, so a request
        // paused by a slow consumer late in its budget must be terminated as
        // health-neutral `.backpressureTimeout` (at the full lease), never
        // misclassified as engine pathology. This cannot unbound a request's
        // lifetime: each pause is bounded by one backpressure lease, and the
        // ceiling applies at the first running check between pauses. A paused
        // row with no armed lease (defensive: bookkeeping drift) still falls
        // back to the ceiling.
        if isPaused, backpressureDeadline != nil {
            if let bp = backpressureDeadline, now > bp { return .backpressureTimeout }
            return nil
        }
        // The absolute ceiling is the backstop for everything else.
        if now > safetyDeadline { return .safetyDeadline }
        if isPaused { return nil }
        if isRunning, now > progressDeadline {
            return phase == .prefill ? .prefillStall : .decodeStall
        }
        return nil
    }
}
