// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: per-request event stream with bounded
// buffering and scheduling backpressure.
//
// The engine thread NEVER blocks on a slow consumer: when a request's buffer
// reaches capacity the stream fires the backpressure callback, the engine
// pauses that request's scheduling (slot retained, decode skipped), and the
// buffer stops growing except for the ≤2 steps already in flight. Draining
// below the low watermark resumes scheduling. Events are never dropped.

import Foundation

public final class CBv2OutputStream: @unchecked Sendable {
    public let id: CBv2RequestID

    private let lock = NSLock()
    private var buffer: [CBv2Event] = []
    private var waiter: CheckedContinuation<CBv2Event?, Never>?
    private var finished = false
    private var terminalDelivered = false
    private var consumerCancelled = false
    private var pausedForBackpressure = false
    private var pendingEmissions = 0

    private let capacity: Int
    private var lowWatermark: Int { max(0, capacity / 2) }
    private let onBackpressure: (@Sendable (CBv2RequestID, Bool) -> Void)?
    private let onAbandoned: (@Sendable (CBv2RequestID) -> Void)?

    public init(
        id: CBv2RequestID,
        capacity: Int = 256,
        onBackpressure: (@Sendable (CBv2RequestID, Bool) -> Void)? = nil,
        onAbandoned: (@Sendable (CBv2RequestID) -> Void)? = nil
    ) {
        precondition(capacity > 0, "CBv2OutputStream capacity must be positive")
        self.id = id
        self.capacity = capacity
        self.onBackpressure = onBackpressure
        self.onAbandoned = onAbandoned
    }

    // MARK: Producer (engine thread / watchdog)

    public func reserveEmission() {
        var firePause = false
        lock.lock()
        pendingEmissions += 1
        if !pausedForBackpressure, !finished, buffer.count + pendingEmissions >= capacity {
            pausedForBackpressure = true
            firePause = true
        }
        lock.unlock()
        if firePause { onBackpressure?(id, true) }
    }

    public func emit(_ event: CBv2Event, consumingReservation: Bool = false) {
        var firePause = false
        var fireResume = false
        lock.lock()
        if consumingReservation { pendingEmissions = max(0, pendingEmissions - 1) }
        if finished {
            lock.unlock()
            return
        }
        if case .finished = event { finished = true }
        if let waiter {
            self.waiter = nil
            if case .finished = event { terminalDelivered = true }
            if pausedForBackpressure, buffer.count + pendingEmissions <= lowWatermark {
                pausedForBackpressure = false
                fireResume = true
            }
            lock.unlock()
            if fireResume { onBackpressure?(id, false) }
            waiter.resume(returning: event)
            return
        }
        buffer.append(event)
        if !pausedForBackpressure, !finished, buffer.count + pendingEmissions >= capacity {
            pausedForBackpressure = true
            firePause = true
        }
        lock.unlock()
        if firePause { onBackpressure?(id, true) }
    }

    public func finish(reason: CBv2FinishReason, usage: CBv2Usage) {
        emit(.finished(reason: reason, usage: usage))
    }

    public var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return finished
    }

    // MARK: Consumer

    public func makeStream() -> AsyncStream<CBv2Event> {
        AsyncStream(
            unfolding: { await self.next() },
            onCancel: {
                guard !self.isFinished else { return }
                self.onAbandoned?(self.id)
            })
    }

    func next() async -> CBv2Event? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<CBv2Event?, Never>) in
                var fireResume = false
                var result: CBv2Event??  // .some(event), .some(nil)=done, nil=parked
                lock.lock()
                if terminalDelivered || consumerCancelled {
                    result = .some(nil)
                } else if !buffer.isEmpty {
                    let event = buffer.removeFirst()
                    if case .finished = event { terminalDelivered = true }
                    if pausedForBackpressure, buffer.count + pendingEmissions <= lowWatermark {
                        pausedForBackpressure = false
                        fireResume = true
                    }
                    result = .some(event)
                } else {
                    waiter = c
                }
                lock.unlock()
                if fireResume { onBackpressure?(id, false) }
                if let result { c.resume(returning: result) }
            }
        } onCancel: {
            lock.lock()
            consumerCancelled = true
            let parked = waiter
            waiter = nil
            lock.unlock()
            parked?.resume(returning: nil)
        }
    }

    public var bufferedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return buffer.count
    }

    var pendingEmissionCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pendingEmissions
    }
}
