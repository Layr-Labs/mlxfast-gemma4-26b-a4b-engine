// FullSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for FULL attention layers.
//
// One instance owns the K/V for ONE sequence at ONE layer. There is no shared
// batch frontier, no left padding, and no batch-wide trim: joining or leaving
// a batch never touches this object (batch membership is just list membership
// in `CBv2LayerCache.rows`).

import Foundation
import MLX

public enum CBv2CoreInstrumentation {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _hostSyncs = 0
    nonisolated(unsafe) private static var _positionOffsetsHostRebuilds = 0

    public static var hostSyncs: Int {
        lock.lock()
        defer { lock.unlock() }
        return _hostSyncs
    }

    public static var positionOffsetsHostRebuilds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _positionOffsetsHostRebuilds
    }

    static func recordHostSync() {
        lock.lock()
        defer { lock.unlock() }
        _hostSyncs += 1
    }

    static func recordPositionOffsetsHostRebuild() {
        lock.lock()
        defer { lock.unlock() }
        _positionOffsetsHostRebuilds += 1
    }
}

protocol CBv2InnerStateProviding {
    func cbv2InnerState() -> [MLXArray]
}

final class CBv2FullDecodeCohortPool {
    let rowCount: Int
    let kvHeads: Int
    let headDim: Int

    private(set) var keys: MLXArray
    private(set) var values: MLXArray
    private(set) var capacity: Int

    private let capacityLimit: Int

    init(
        keys: MLXArray, values: MLXArray, capacity: Int,
        rowCount: Int, kvHeads: Int, headDim: Int, capacityLimit: Int
    ) {
        self.keys = keys
        self.values = values
        self.capacity = capacity
        self.rowCount = rowCount
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacityLimit = capacityLimit
    }

    var nbytes: Int { keys.nbytes + values.nbytes }

    func rowAppend(
        index: Int, keys newKeys: MLXArray, values newValues: MLXArray,
        at offset: Int, count n: Int
    ) {
        ensureCapacity(offset + n)
        keys[index ..< (index + 1), 0..., offset ..< (offset + n), 0...] = newKeys
        values[index ..< (index + 1), 0..., offset ..< (offset + n), 0...] = newValues
    }

    func batchAppend(keys newKeys: MLXArray, values newValues: MLXArray, at offset: Int) {
        ensureCapacity(offset + 1)
        keys[0..., 0..., offset ..< (offset + 1), 0...] = newKeys
        values[0..., 0..., offset ..< (offset + 1), 0...] = newValues
    }

    func rowViews(index: Int, upTo offset: Int) -> (MLXArray, MLXArray) {
        (
            keys[index ..< (index + 1), 0..., ..<offset, 0...],
            values[index ..< (index + 1), 0..., ..<offset, 0...]
        )
    }

    func batchViews(upTo offset: Int) -> (MLXArray, MLXArray) {
        (keys[0..., 0..., ..<offset, 0...], values[0..., 0..., ..<offset, 0...])
    }

    private func ensureCapacity(_ needed: Int) {
        guard needed > capacity else { return }
        precondition(
            needed <= capacityLimit,
            "CBv2FullDecodeCohortPool: append past capacity limit (\(needed) > \(capacityLimit)) — admission bug"
        )
        let newCapacity = min(capacityLimit, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys, MLXArray.zeros([rowCount, kvHeads, growth, headDim], dtype: keys.dtype)],
            axis: 2)
        values = concatenated(
            [values, MLXArray.zeros([rowCount, kvHeads, growth, headDim], dtype: values.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}

public final class CBv2FullSequenceKV: CBv2DecodeRootCompactionCapableSequenceKV,
    CBv2InnerStateProviding
{

    static let initialSlack = 256

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    private var capacity: Int

    private(set) var cohortPool: CBv2FullDecodeCohortPool?
    private(set) var cohortIndex: Int = -1

    public init(promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int) {
        precondition(maxLength > 0, "CBv2FullSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2FullSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = min(maxLength, max(1, promptLength + Self.initialSlack))
    }

    public var byteCount: Int {
        if let pool = cohortPool {
            return pool.nbytes / pool.rowCount
        }
        return (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
    }

    public func advanceAfterFusedAppend() {
        precondition(
            cohortPool == nil && keys != nil && values != nil,
            "CBv2FullSequenceKV: fused append advance requires private storage")
        precondition(
            absoluteOffset + 1 <= maxLength,
            "CBv2FullSequenceKV: fused append past maxLength — admission bug")
        absoluteOffset += 1
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2FullSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2FullSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2FullSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        if let pool = cohortPool {
            pool.rowAppend(
                index: cohortIndex, keys: newKeys, values: newValues,
                at: absoluteOffset, count: n)
            absoluteOffset += n
            return pool.rowViews(index: cohortIndex, upTo: absoluteOffset)
        }

        ensureCapacity(absoluteOffset + n, keyTemplate: newKeys, valueTemplate: newValues)

        keys![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newKeys
        values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = newValues
        absoluteOffset += n

        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    func confirmPooledBatchAppend(_ n: Int) {
        precondition(cohortPool != nil, "CBv2FullSequenceKV: batch append without a pool")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )
        absoluteOffset += n
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        if let pool = cohortPool {
            let (poolKeys, poolValues) = pool.rowViews(
                index: cohortIndex, upTo: absoluteOffset)
            return (poolKeys, poolValues, absoluteOffset)
        }
        guard let keys, let values else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...],
            absoluteOffset
        )
    }

    public var supportsSpeculativeWrites: Bool { true }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2FullSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2FullSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
    }

    func cbv2InnerState() -> [MLXArray] {
        if let pool = cohortPool {
            return [pool.keys, pool.values]
        }
        return [keys, values].compactMap { $0 }
    }

    // MARK: - ATT-008 cohort pooling

    static func cohortPool(binding rows: [CBv2FullSequenceKV])
        -> CBv2FullDecodeCohortPool?
    {
        guard !rows.isEmpty else { return nil }

        if let pool = rows[0].cohortPool {
            guard pool.rowCount == rows.count else { return nil }
            for (index, row) in rows.enumerated() {
                guard row.cohortPool === pool, row.cohortIndex == index else {
                    return nil
                }
            }
            return pool
        }

        let head = rows[0]
        guard let headKeys = head.keys, let headValues = head.values else { return nil }
        let expectedShape = [1, head.kvHeads, head.capacity, head.headDim]
        for row in rows {
            guard row.cohortPool == nil,
                row.kvHeads == head.kvHeads,
                row.headDim == head.headDim,
                row.capacity == head.capacity,
                let rowKeys = row.keys, let rowValues = row.values,
                rowKeys.dtype == headKeys.dtype,
                rowValues.dtype == headValues.dtype,
                rowKeys.shape == expectedShape,
                rowValues.shape == expectedShape
            else { return nil }
        }

        let pool = CBv2FullDecodeCohortPool(
            keys: concatenated(rows.map { $0.keys! }, axis: 0),
            values: concatenated(rows.map { $0.values! }, axis: 0),
            capacity: head.capacity,
            rowCount: rows.count,
            kvHeads: head.kvHeads,
            headDim: head.headDim,
            capacityLimit: rows.map(\.maxLength).max()!)
        for (index, row) in rows.enumerated() {
            row.cohortPool = pool
            row.cohortIndex = index
            row.keys = nil
            row.values = nil
        }
        return pool
    }

    // MARK: - Private

    private func ensureCapacity(_ needed: Int, keyTemplate: MLXArray, valueTemplate: MLXArray) {
        if keys == nil {
            capacity = min(maxLength, max(capacity, needed))
            keys = MLXArray.zeros(
                [1, kvHeads, capacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, capacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
            return
        }
        guard needed > capacity else { return }

        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys!, MLXArray.zeros([1, kvHeads, growth, keys!.dim(3)], dtype: keys!.dtype)],
            axis: 2)
        values = concatenated(
            [values!, MLXArray.zeros([1, kvHeads, growth, values!.dim(3)], dtype: values!.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}
