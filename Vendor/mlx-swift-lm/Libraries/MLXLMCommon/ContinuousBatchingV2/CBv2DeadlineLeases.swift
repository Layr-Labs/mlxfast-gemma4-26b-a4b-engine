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

public struct CBv2Clock: Sendable {
    public let now: @Sendable () -> ContinuousClock.Instant
    public init(now: @escaping @Sendable () -> ContinuousClock.Instant) {
        self.now = now
    }
    public static let continuous = CBv2Clock { ContinuousClock.now }
}

// MARK: - Typed terminal cause

public enum CBv2TerminalCause: Sendable, Equatable {
    case admissionTimeout
    case prefillStall
    case decodeStall
    case safetyDeadline
    case backpressureTimeout
    case watchdog
    case legacyRequestTimeout

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

public enum CBv2SafetyCeiling {
    static let prefillFloorTPS: Double = 200
    static let preemptionSlackSeconds: Double = 60
    static let maxCeilingSeconds: Double = 1e9

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
        return .seconds(min(seconds, maxCeilingSeconds))
    }
}

// MARK: - Per-request lease state machine

struct CBv2RequestLeaseState {
    enum Phase: Equatable { case prefill, decode }

    private(set) var legacyWall: ContinuousClock.Instant?

    private(set) var admissionDeadline: ContinuousClock.Instant?

    private(set) var progressDeadline: ContinuousClock.Instant
    private(set) var phase: Phase

    private(set) var backpressureDeadline: ContinuousClock.Instant?

    private(set) var safetyDeadline: ContinuousClock.Instant

    private let prefillLease: Duration
    private let decodeLease: Duration
    private let backpressureLease: Duration

    private var lastComputedTokens: Int
    private var lastGeneratedTokens: Int

    var isAdmitted: Bool { legacyWall == nil && admissionDeadline == nil }
    var isLegacy: Bool { legacyWall != nil }

    // MARK: Construction

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
        self.progressDeadline = now.advanced(by: .seconds(prefillLease))
        self.backpressureDeadline = nil
        self.safetyDeadline = now.advanced(by: safety)
        self.lastComputedTokens = computedTokens
        self.lastGeneratedTokens = generatedTokens
    }

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

    mutating func markAdmitted(now: ContinuousClock.Instant) {
        guard legacyWall == nil, admissionDeadline != nil else { return }
        admissionDeadline = nil
        phase = .prefill
        progressDeadline = now.advanced(by: prefillLease)
    }

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

    mutating func markPaused(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        backpressureDeadline = now.advanced(by: backpressureLease)
    }

    mutating func markResumed(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        backpressureDeadline = nil
        progressDeadline = now.advanced(by: phase == .prefill ? prefillLease : decodeLease)
    }

    mutating func markPreempted(now: ContinuousClock.Instant) {
        guard legacyWall == nil else { return }
        lastComputedTokens = 0
        phase = .prefill
        progressDeadline = now.advanced(by: prefillLease)
    }

    mutating func markReadmitted(now: ContinuousClock.Instant) {
        guard legacyWall == nil, admissionDeadline == nil else { return }
        progressDeadline = now.advanced(by: phase == .prefill ? prefillLease : decodeLease)
    }

    // MARK: Expiry

    func expiredCause(
        now: ContinuousClock.Instant, isRunning: Bool, isPaused: Bool
    ) -> CBv2TerminalCause? {
        if let wall = legacyWall {
            return now > wall ? .legacyRequestTimeout : nil
        }
        if let admission = admissionDeadline {
            if now > admission { return .admissionTimeout }
            if now > safetyDeadline { return .safetyDeadline }
            return nil
        }
        if isPaused, backpressureDeadline != nil {
            if let bp = backpressureDeadline, now > bp { return .backpressureTimeout }
            return nil
        }
        if now > safetyDeadline { return .safetyDeadline }
        if isPaused { return nil }
        if isRunning, now > progressDeadline {
            return phase == .prefill ? .prefillStall : .decodeStall
        }
        return nil
    }
}
