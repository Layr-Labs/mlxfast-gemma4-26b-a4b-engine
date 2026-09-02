// StopHoldback.swift
//
// Streaming stop-string matcher for ContinuousBatchingV2 (workstream E).
// One instance per in-flight request, fed the incremental text produced by
// `DetokenizerV2`.
//
// Guarantee: `ingest` only ever returns text that provably cannot be part
// of a stop string — the longest suffix of the pending text that is a
// prefix of ANY stop string is held back until it is either completed
// (stop: truncate at the match start, signal stop, emit nothing further)
// or disambiguated (released with the next chunk). This makes workstream
// B's one-step-late stop detection invisible: a user can never observe
// text at or past a stop match, even when the stop string spans several
// tokens or overlaps another candidate ("</s>" vs "</section>").
//
// `flush()` releases the held tail at natural end of generation (EOS /
// max-tokens), where no stop can complete anymore.
//
// Matching is over unicode scalars (exact substring semantics, no
// normalization), mirroring how OpenAI-compatible servers compare stop
// strings against the decoded stream.

import Foundation

public struct CBv2StopScan: Sendable, Equatable {
    public var text: String
    public var stopped: Bool

    public init(text: String, stopped: Bool) {
        self.text = text
        self.stopped = stopped
    }
}

public final class StopHoldback {

    private let stops: [[Unicode.Scalar]]
    private let maxHold: Int
    private var pending: [Unicode.Scalar] = []
    private var stopped = false

    public init(stopStrings: [String]) {
        self.stops = stopStrings.filter { !$0.isEmpty }.map { Array($0.unicodeScalars) }
        self.maxHold = stops.map(\.count).max() ?? 0
    }

    public var isPassthrough: Bool { stops.isEmpty }

    public func ingest(_ text: String) -> CBv2StopScan {
        if stopped {
            return CBv2StopScan(text: "", stopped: true)
        }
        if stops.isEmpty {
            return CBv2StopScan(text: text, stopped: false)
        }

        pending.append(contentsOf: text.unicodeScalars)

        if let matchStart = earliestMatchStart() {
            stopped = true
            let emit = String(String.UnicodeScalarView(pending[..<matchStart]))
            pending.removeAll()
            return CBv2StopScan(text: emit, stopped: true)
        }

        let hold = longestAmbiguousSuffix()
        let emitCount = pending.count - hold
        guard emitCount > 0 else {
            return CBv2StopScan(text: "", stopped: false)
        }
        let emit = String(String.UnicodeScalarView(pending[..<emitCount]))
        pending.removeFirst(emitCount)
        return CBv2StopScan(text: emit, stopped: false)
    }

    public func flush() -> String {
        guard !stopped, !pending.isEmpty else { return "" }
        let emit = String(String.UnicodeScalarView(pending))
        pending.removeAll()
        return emit
    }

    // MARK: - Internals

    private func earliestMatchStart() -> Int? {
        var earliest: Int? = nil
        for stop in stops {
            guard stop.count <= pending.count else { continue }
            let lastStart = pending.count - stop.count
            var start = 0
            while start <= lastStart {
                if earliest != nil, start >= earliest! { break }
                var i = 0
                while i < stop.count, pending[start + i] == stop[i] { i += 1 }
                if i == stop.count {
                    if earliest == nil || start < earliest! { earliest = start }
                    break
                }
                start += 1
            }
        }
        return earliest
    }

    private func longestAmbiguousSuffix() -> Int {
        var longest = 0
        for stop in stops {
            let maxLen = min(stop.count - 1, pending.count)
            guard maxLen > longest else { continue }
            var k = maxLen
            while k > longest {
                var matches = true
                let offset = pending.count - k
                for i in 0 ..< k where pending[offset + i] != stop[i] {
                    matches = false
                    break
                }
                if matches {
                    longest = k
                    break
                }
                k -= 1
            }
        }
        return longest
    }
}
