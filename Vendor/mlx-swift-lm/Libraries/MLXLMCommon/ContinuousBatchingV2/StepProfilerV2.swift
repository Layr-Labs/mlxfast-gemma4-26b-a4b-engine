// StepProfilerV2.swift
//
// Opt-in phase timers and independently armed event counters for the decode
// hot loops (v2 EngineLoopV2 and the legacy GenerationBatch/Scheduler path).
// BenchCBv2's profile mode uses phase timing to decompose per-step wall time;
// performance cells arm only graph-submission events for provenance.
//
// DISABLED by default: each instrumentation point reads one plain static Bool,
// so the production step path pays one predictable branch and nothing else.
// Timing can be enabled programmatically or via CBV2_STEP_PROFILE=1. Event
// counters are armed explicitly at an idle benchmark boundary.

import Foundation

public enum CBv2StepProfiler {

    nonisolated(unsafe) public static var enabled: Bool = {
        let environment = ProcessInfo.processInfo.environment
        let raw = environment["CBV2_STEP_PROFILE"] ?? environment["DARKBLOOM_CBV2_STEP_PROFILE"]
        return raw.map { ["1", "true", "yes", "on"].contains($0.lowercased()) } ?? false
    }()

    nonisolated(unsafe) public private(set) static var eventsEnabled = false

    private static let lock = NSLock()
    nonisolated(unsafe) private static var samples: [String: [Double]] = [:]
    nonisolated(unsafe) private static var eventCounts: [String: Int] = [:]

    @inline(__always)
    public static func record(_ phase: StaticString, seconds: Double) {
        guard enabled else { return }
        let key = "\(phase)"
        lock.lock()
        samples[key, default: []].append(seconds)
        lock.unlock()
    }

    @inline(__always)
    public static func time<T>(_ phase: StaticString, _ body: () -> T) -> T {
        guard enabled else { return body() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = body()
        record(phase, seconds: CFAbsoluteTimeGetCurrent() - start)
        return result
    }
    @inline(__always)
    public static func recordEvent(_ event: StaticString) {
        guard eventsEnabled else { return }
        let key = "\(event)"
        lock.lock()
        defer { lock.unlock() }
        guard eventsEnabled else { return }
        eventCounts[key, default: 0] += 1
    }

    public static func armEvents() {
        lock.lock()
        eventCounts.removeAll(keepingCapacity: true)
        eventsEnabled = true
        lock.unlock()
    }

    public static func snapshotAndDisarmEvents() -> [String: Int] {
        lock.lock()
        eventsEnabled = false
        defer { lock.unlock() }
        return eventCounts
    }

    public static func reset() {
        lock.lock()
        eventsEnabled = false
        samples.removeAll()
        eventCounts.removeAll()
        lock.unlock()
    }

    public static func snapshot() -> [String: [Double]] {
        lock.lock()
        defer { lock.unlock() }
        return samples.mapValues { $0.sorted() }
    }
    public static func eventCountsSnapshot() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return eventCounts
    }

    public static func summaryTable() -> String {
        let snap = snapshot()
        func pct(_ sorted: [Double], _ q: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let rank = q * Double(sorted.count - 1)
            let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
            if lo == hi { return sorted[lo] }
            let w = rank - Double(lo)
            return sorted[lo] * (1 - w) + sorted[hi] * w
        }
        var out = "| phase | n | total ms | mean ms | p50 ms | p95 ms | max ms |\n"
        out += "|---|---|---|---|---|---|---|\n"
        let rows = snap.map { (name: $0.key, sorted: $0.value) }
            .sorted { $0.sorted.reduce(0, +) > $1.sorted.reduce(0, +) }
        for row in rows {
            let s = row.sorted
            let total = s.reduce(0, +)
            out += String(
                format: "| %@ | %d | %.1f | %.3f | %.3f | %.3f | %.3f |\n",
                row.name, s.count, total * 1e3, total / Double(s.count) * 1e3,
                pct(s, 0.5) * 1e3, pct(s, 0.95) * 1e3, (s.last ?? 0) * 1e3)
        }
        return out
    }
}
