// ContiguousKVBackend.swift
//
// The v1 `CBv2KVBackend`: per-sequence contiguous MLX buffers
// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV`).
// The paged backend (workstream C) implements the same protocol behind a
// Metal kernel; the scheduler and models never see the difference.

import Foundation
import MLX

public struct CBv2ContiguousBackendConfig: Sendable {
    public var bytesCapacity: Int
    public var kvDType: DType

    public init(
        bytesCapacity: Int,
        kvDType: DType = .float16
    ) {
        self.bytesCapacity = bytesCapacity
        self.kvDType = kvDType
    }
}

public final class CBv2ContiguousKVBackend: CBv2KVBackend {

    public let config: CBv2ContiguousBackendConfig
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .contiguousUnquantized }

    private let lock = NSLock()
    private var live: [ObjectIdentifier: CBv2SequenceKV] = [:]
    private var reservations: [ObjectIdentifier: Int] = [:]
    private var liveBytesCapacity: Int

    public init(config: CBv2ContiguousBackendConfig) {
        self.config = config
        self.liveBytesCapacity = config.bytesCapacity
    }

    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return liveBytesCapacity
    }

    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        liveBytesCapacity = max(0, bytes)
        lock.unlock()
    }

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return live.values.reduce(0) { $0 + $1.byteCount }
    }

    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return accountedBytesLocked()
    }

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard promptLength <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "promptLength \(promptLength) exceeds maxLength \(maxLength)")
        }

        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            makeRow(kind: kind, promptLength: promptLength, maxLength: maxLength)
        }
        try registerReserving(
            state,
            estimates: rowEstimates(
                layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength))
        return state
    }

    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }

        guard plan.backend == prefixReuseBackend else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "prefix plan backend \(plan.backend.rawValue) != \(prefixReuseBackend.rawValue)")
        }
        guard plan.matchedBoundary <= maxLength,
            plan.replayStart >= 0,
            plan.replayStart <= plan.matchedBoundary,
            plan.replayTokens == plan.matchedBoundary - plan.replayStart
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix replay plan")
        }

        let expectedSnapshotOffset = plan.restoredFullTokens
        var sawOwningFull = false
        for (index, entry) in prefix.enumerated() {
            let kind = layerKinds[index]
            if kind.sharesKVWithLayer != nil {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is KV-shared but received a prefix snapshot")
                }
                continue
            }
            if case .slidingWindow = kind.attention {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index) is windowed but received a prefix snapshot")
                }
                continue
            }
            sawOwningFull = true
            guard let entry else {
                throw CBv2KVError.backendIneligible(
                    reason: "owning full layer \(index) is missing its prefix snapshot")
            }
            guard kind.sharesKVWithLayer == nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index) is KV-shared but received a prefix snapshot")
            }
            guard case .full = kind.attention else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "layer \(index) is windowed but received a prefix snapshot (windowed layers are recomputed)"
                )
            }
            guard entry.offset == expectedSnapshotOffset else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "prefix offset \(entry.offset) != planned \(expectedSnapshotOffset) at layer \(index)"
                )
            }
            guard entry.keys.dim(2) == entry.offset, entry.values.dim(2) == entry.offset else {
                throw CBv2KVError.backendIneligible(
                    reason: "full prefix snapshot at layer \(index) does not exactly cover its offset")
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix replay requires at least one storage-owning full layer")
        }

        let state = layerKinds.enumerated().map { index, kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    initialOffset: plan.replayStart)
            case .full:
                let entry = prefix[index]!
                if plan.strategy == .frozenFullReplay {
                    return CBv2FrozenReplayFullSequenceKV(
                        snapshot: entry,
                        replayStart: plan.replayStart,
                        maxLength: maxLength,
                        kvHeads: kind.kvHeads,
                        headDim: kind.headDim)
                }
                let row = makeRow(
                    kind: kind,
                    promptLength: expectedSnapshotOffset,
                    maxLength: maxLength)!
                _ = row.update(keys: entry.keys, values: entry.values)
                return row
            }
        }
        try registerReserving(
            state,
            estimates: adoptionRowEstimates(
                prefix: prefix,
                layerKinds: layerKinds,
                maxLength: maxLength))
        return state
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live.removeValue(forKey: key)
            reservations.removeValue(forKey: key)
        }
    }

    // MARK: - Private

    private func accountedBytesLocked() -> Int {
        live.reduce(0) { total, entry in
            total + max(entry.value.byteCount, reservations[entry.key] ?? 0)
        }
    }

    private func registerReserving(_ state: [CBv2SequenceKV?], estimates: [Int?]) throws {
        precondition(state.count == estimates.count, "estimate/state count mismatch")
        lock.lock()
        defer { lock.unlock() }
        let needed = zip(state, estimates).reduce(0) { total, pair in
            guard let row = pair.0 else { return total }
            return total + max(row.byteCount, pair.1 ?? 0)
        }
        let available = liveBytesCapacity - accountedBytesLocked()
        guard needed <= available else {
            throw CBv2KVError.capacityExhausted(needed: needed, available: max(0, available))
        }
        for (row, estimate) in zip(state, estimates) {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live[key] = row
            reservations[key] = estimate ?? 0
        }
    }

    private func makeRow(kind: CBv2LayerKind, promptLength: Int, maxLength: Int)
        -> CBv2SequenceKV?
    {
        guard kind.sharesKVWithLayer == nil else { return nil }
        switch kind.attention {
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        case .full:
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim)
        }
    }

    private func validate(layerKinds: [CBv2LayerKind]) throws {
        for (index, kind) in layerKinds.enumerated() {
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): non-positive window \(window)")
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count, source != index else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): invalid KV-share source \(source)")
                }
                guard layerKinds[source].sharesKVWithLayer == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): KV-share source \(source) is itself a shared layer")
                }
            }
        }
    }

    private func rowEstimates(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) -> [Int?] {
        let itemSize = config.kvDType.size
        return layerKinds.map { kind -> Int? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return window * kind.kvHeads * kind.headDim * itemSize * 2
            case .full:
                let slots = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
                return slots * kind.kvHeads * kind.headDim * itemSize * 2
            }
        }
    }

    private func adoptionRowEstimates(
        prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind],
        maxLength: Int
    ) -> [Int?] {
        layerKinds.enumerated().map { index, kind in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return window * kind.kvHeads * kind.headDim * config.kvDType.size * 2
            case .full:
                guard let entry = prefix[index] else { return 0 }
                return maxLength * kind.kvHeads * kind.headDim
                    * (entry.keys.dtype.size + entry.values.dtype.size)
            }
        }
    }
}
