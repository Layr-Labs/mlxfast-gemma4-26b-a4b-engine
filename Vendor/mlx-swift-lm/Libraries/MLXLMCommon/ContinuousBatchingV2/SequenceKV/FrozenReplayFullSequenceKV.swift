// FrozenReplayFullSequenceKV.swift
//
// Exact full-attention storage for hybrid prefix reuse. Cached K/V remains
// immutable through the matched boundary M while the logical read/replay
// cursor advances from C. At M the row transitions once to ordinary append.

import Foundation
import MLX

/// Full-attention row with independent logical and physical frontiers.
///
/// During `[C, M)` updates, newly projected K/V is deliberately ignored:
/// attention reads the exact cached prefix through the current chunk end.
/// This prevents incomplete sliding-window replay activations from poisoning
/// persistent downstream full-attention state. Chunks must not cross M; the
/// scheduler enforces that boundary from the same `CBv2PrefixReusePlan`.
public final class CBv2FrozenReplayFullSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {
    public private(set) var absoluteOffset: Int
    public var retainedCount: Int { absoluteOffset }
    public let frozenHighWater: Int
    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray
    private var values: MLXArray
    private var capacity: Int
    private var appendMode = false
    private var transitionCount = 0

    public init(
        snapshot: (keys: MLXArray, values: MLXArray, offset: Int),
        replayStart: Int,
        maxLength: Int,
        kvHeads: Int,
        headDim: Int
    ) {
        precondition(replayStart >= 0, "frozen replay start must be non-negative")
        precondition(snapshot.offset > replayStart, "frozen replay requires C < M")
        precondition(snapshot.offset <= maxLength, "frozen high-water exceeds maxLength")
        precondition(snapshot.keys.ndim == 4 && snapshot.values.ndim == 4)
        precondition(snapshot.keys.dim(0) == 1 && snapshot.values.dim(0) == 1)
        precondition(snapshot.keys.dim(1) == kvHeads && snapshot.values.dim(1) == kvHeads)
        precondition(
            snapshot.keys.dim(2) == snapshot.offset
                && snapshot.values.dim(2) == snapshot.offset,
            "frozen snapshot must exactly cover [0, M)")
        precondition(
            snapshot.keys.dim(3) == headDim && snapshot.values.dim(3) == headDim,
            "frozen snapshot head dimension mismatch")
        self.absoluteOffset = replayStart
        self.frozenHighWater = snapshot.offset
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        // Ownership transfer: retain the staged arrays directly. No adoption
        // copy, append buffer, rotation, or reallocation occurs before M.
        self.keys = snapshot.keys
        self.values = snapshot.values
        self.capacity = snapshot.offset
    }

    public var byteCount: Int {
        keys.nbytes + values.nbytes
    }

    public var supportsSpeculativeWrites: Bool { true }

    public func update(
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        validateUpdate(keys: newKeys, values: newValues)
        let count = newKeys.dim(2)

        if absoluteOffset < frozenHighWater {
            precondition(
                !appendMode,
                "frozen replay row entered append mode before matched boundary")
            precondition(
                absoluteOffset + count <= frozenHighWater,
                "frozen replay chunk crosses M; scheduler boundary split missing")
            absoluteOffset += count
            return logicalViews()
        }

        enterAppendMode()
        precondition(
            absoluteOffset + count <= maxLength,
            "frozen replay append past maxLength")
        ensureCapacity(absoluteOffset + count, keyTemplate: newKeys, valueTemplate: newValues)
        keys[.ellipsis, absoluteOffset ..< (absoluteOffset + count), 0...] = newKeys
        values[.ellipsis, absoluteOffset ..< (absoluteOffset + count), 0...] = newValues
        absoluteOffset += count
        return logicalViews()
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let views = logicalViews()
        return (views.0, views.1, absoluteOffset)
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "frozen replay rollback count must be non-negative")
        precondition(appendMode, "frozen replay cannot roll back immutable replay")
        precondition(
            absoluteOffset - n >= frozenHighWater,
            "frozen replay rollback cannot cross immutable high-water")
        absoluteOffset -= n
    }

    public func beginSpeculativeWrite() {
        precondition(appendMode, "speculation cannot begin during frozen replay")
    }

    public func commitSpeculativeWrite() {
        precondition(appendMode, "speculation cannot commit during frozen replay")
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values]
    }

    var didTransitionToAppendForTesting: Bool { appendMode }
    var transitionCountForTesting: Int { transitionCount }

    private func enterAppendMode() {
        guard !appendMode else { return }
        precondition(
            absoluteOffset == frozenHighWater,
            "frozen replay may transition only at M")
        appendMode = true
        transitionCount += 1
    }

    private func logicalViews() -> (MLXArray, MLXArray) {
        (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    private func validateUpdate(keys newKeys: MLXArray, values newValues: MLXArray) {
        let count = newKeys.dim(2)
        precondition(count > 0, "frozen replay update must be non-empty")
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1)
        precondition(newKeys.dim(1) == kvHeads && newValues.dim(1) == kvHeads)
        precondition(newValues.dim(2) == count)
        precondition(newKeys.dim(3) == headDim && newValues.dim(3) == headDim)
        precondition(
            newKeys.dtype == keys.dtype && newValues.dtype == values.dtype,
            "frozen replay K/V dtype changed across the matched boundary")
    }

    private func ensureCapacity(
        _ needed: Int,
        keyTemplate: MLXArray,
        valueTemplate: MLXArray
    ) {
        guard needed > capacity else { return }
        let target: Int
        if capacity == frozenHighWater {
            // Adoption reserved M + initialSlack before publication. The first
            // append must not jump to 2*M and outrun that physical promise.
            target = max(needed, frozenHighWater + CBv2FullSequenceKV.initialSlack)
        } else {
            target = max(capacity * 2, needed)
        }
        let newCapacity = min(maxLength, target)
        let growth = newCapacity - capacity
        keys = concatenated(
            [
                keys,
                MLXArray.zeros([1, kvHeads, growth, headDim], dtype: keyTemplate.dtype),
            ],
            axis: 2)
        values = concatenated(
            [
                values,
                MLXArray.zeros([1, kvHeads, growth, headDim], dtype: valueTemplate.dtype),
            ],
            axis: 2)
        capacity = newCapacity
    }
}
