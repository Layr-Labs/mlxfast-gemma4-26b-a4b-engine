
import Foundation
import MLX

public final class CBv2WindowedSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    public let window: Int

    public private(set) var absoluteOffset: Int

    private var oldestValidPosition: Int

    public var retainedCount: Int { absoluteOffset - oldestValidPosition }

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    private var borrowableChunkViews: (keys: MLXArray, values: MLXArray)?

    private var speculativeWriteArmed = false

    private var staged: (keys: MLXArray, values: MLXArray, basePosition: Int)?

    public init(window: Int, kvHeads: Int, headDim: Int, initialOffset: Int = 0) {
        precondition(window > 0, "CBv2WindowedSequenceKV: window must be > 0")
        precondition(initialOffset >= 0, "CBv2WindowedSequenceKV: negative initialOffset")
        self.window = window
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.absoluteOffset = initialOffset
        self.oldestValidPosition = initialOffset
    }

    public var byteCount: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (staged.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2WindowedSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2WindowedSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2WindowedSequenceKV: keys/values token count mismatch")
        precondition(n > 0, "CBv2WindowedSequenceKV: empty update")

        if speculativeWriteArmed {
            return stageSpeculativeUpdate(newKeys: newKeys, newValues: newValues, count: n)
        }
        precondition(
            staged == nil,
            "CBv2WindowedSequenceKV: plain update with a staged speculative write pending — commit first"
        )

        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)

        if n == 1 {
            borrowableChunkViews = nil  // ring == window: snapshot is exact
            writeRing(keys!, tokens: newKeys, firstPosition: absoluteOffset)
            writeRing(values!, tokens: newValues, firstPosition: absoluteOffset)
            absoluteOffset += 1
            oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
            return (
                temporalOrder(keys!, from: oldestValidPosition, to: absoluteOffset),
                temporalOrder(values!, from: oldestValidPosition, to: absoluteOffset)
            )
        }

        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        var kParts = ringSlices(keys!, from: historyFrom, to: absoluteOffset)
        var vParts = ringSlices(values!, from: historyFrom, to: absoluteOffset)
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        let writeCount = min(n, window)
        let firstWritten = absoluteOffset + n - writeCount
        let kTail = writeCount == n ? newKeys : newKeys[.ellipsis, (n - writeCount)..., 0...]
        let vTail = writeCount == n ? newValues : newValues[.ellipsis, (n - writeCount)..., 0...]
        writeRing(keys!, tokens: kTail, firstPosition: firstWritten)
        writeRing(values!, tokens: vTail, firstPosition: firstWritten)

        absoluteOffset += n
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)

        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    var canUseFullDecodeRing: Bool {
        !speculativeWriteArmed && staged == nil && keys != nil && values != nil
            && retainedCount == window
    }

    func fullDecodeRingBeforeWrite() -> (keys: MLXArray, values: MLXArray, start: Int)? {
        guard canUseFullDecodeRing else { return nil }
        return (keys!, values!, (oldestValidPosition + 1) % window)
    }

    func advanceFullDecodeRingWithoutWrite() {
        precondition(canUseFullDecodeRing)
        borrowableChunkViews = nil
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    func writeFullDecodeRing(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (keys: MLXArray, values: MLXArray, start: Int) {
        precondition(canUseFullDecodeRing,
            "CBv2WindowedSequenceKV: ring decode requires a full unstaged ring")
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2WindowedSequenceKV holds one sequence")
        precondition(newKeys.dim(1) == kvHeads && newKeys.dim(2) == 1,
            "CBv2WindowedSequenceKV: ring decode key shape mismatch")
        precondition(newValues.dim(2) == 1,
            "CBv2WindowedSequenceKV: ring decode value shape mismatch")

        borrowableChunkViews = nil
        writeRing(keys!, tokens: newKeys, firstPosition: absoluteOffset)
        writeRing(values!, tokens: newValues, firstPosition: absoluteOffset)
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
        return (keys!, values!, oldestValidPosition % window)
    }


    public var supportsSpeculativeWrites: Bool { true }

    public func beginSpeculativeWrite() {
        precondition(
            !speculativeWriteArmed,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite while already armed")
        precondition(
            staged == nil,
            "CBv2WindowedSequenceKV: beginSpeculativeWrite with a staged update pending — commit first"
        )
        speculativeWriteArmed = true
    }

    private func stageSpeculativeUpdate(
        newKeys: MLXArray, newValues: MLXArray, count n: Int
    ) -> (MLXArray, MLXArray) {
        let logical = snapshot()
        let historyCount = min(logical.keys.dim(2), window - 1)
        let historyFrom = logical.keys.dim(2) - historyCount
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if historyCount > 0 {
            kParts.append(logical.keys[.ellipsis, historyFrom..., 0...])
            vParts.append(logical.values[.ellipsis, historyFrom..., 0...])
        }
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        if let existing = staged {
            let confirmed = absoluteOffset - existing.basePosition
            let existingKeys = existing.keys[.ellipsis, ..<confirmed, 0...]
            let existingValues = existing.values[.ellipsis, ..<confirmed, 0...]
            staged = (
                concatenated([existingKeys, newKeys], axis: 2),
                concatenated([existingValues, newValues], axis: 2),
                existing.basePosition)
        } else {
            staged = (newKeys, newValues, absoluteOffset)
        }
        absoluteOffset += n
        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    public func commitSpeculativeWrite() {
        speculativeWriteArmed = false
        guard let staged else { return }
        self.staged = nil
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            allocateIfNeeded(keyTemplate: staged.keys, valueTemplate: staged.values)
            let writeCount = min(confirmed, window)
            let skip = confirmed - writeCount
            writeRing(
                keys!, tokens: staged.keys[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount)
            writeRing(
                values!, tokens: staged.values[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount)
        }
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
        borrowableChunkViews = nil
    }

    public func borrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if let views = borrowableChunkViews { return views }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    func decodeBorrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if staged != nil {
            precondition(
                borrowableChunkViews != nil,
                "CBv2WindowedSequenceKV: staged decode borrow after rollback — commit first")
            let views = borrowableChunkViews!
            precondition(
                views.keys.dim(2) <= window && views.values.dim(2) <= window,
                "CBv2WindowedSequenceKV: staged decode borrow exceeds window \(window)")
            return views
        }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        if let staged {
            return stagedSnapshot(staged)
        }
        guard let keys, let values, retainedCount > 0 else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            temporalOrder(keys, from: oldestValidPosition, to: absoluteOffset),
            temporalOrder(values, from: oldestValidPosition, to: absoluteOffset),
            absoluteOffset
        )
    }

    private func stagedSnapshot(
        _ staged: (keys: MLXArray, values: MLXArray, basePosition: Int)
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if let keys, let values, staged.basePosition > oldestValidPosition {
            kParts = ringSlices(keys, from: oldestValidPosition, to: staged.basePosition)
            vParts = ringSlices(values, from: oldestValidPosition, to: staged.basePosition)
        }
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            kParts.append(staged.keys[.ellipsis, ..<confirmed, 0...])
            vParts.append(staged.values[.ellipsis, ..<confirmed, 0...])
        }
        guard !kParts.isEmpty else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2),
            vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2),
            absoluteOffset
        )
    }

    public func fastForward(to offset: Int) {
        precondition(
            keys == nil && absoluteOffset == oldestValidPosition,
            "CBv2WindowedSequenceKV.fastForward requires a fresh state")
        precondition(
            !speculativeWriteArmed && staged == nil,
            "CBv2WindowedSequenceKV.fastForward with a speculative write pending")
        precondition(offset >= absoluteOffset, "fastForward cannot move backwards")
        absoluteOffset = offset
        oldestValidPosition = offset
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2WindowedSequenceKV.rollback: negative n")
        if let staged {
            precondition(
                n <= absoluteOffset - staged.basePosition,
                "CBv2WindowedSequenceKV.rollback(\(n)) exceeds staged range "
                    + "\(absoluteOffset - staged.basePosition)")
            absoluteOffset -= n
            borrowableChunkViews = nil
            return
        }
        precondition(
            n <= retainedCount,
            "CBv2WindowedSequenceKV.rollback(\(n)) exceeds retained \(retainedCount)")
        absoluteOffset -= n
        borrowableChunkViews = nil
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }


    private func ringSlices(_ array: MLXArray, from: Int, to: Int) -> [MLXArray] {
        guard to > from else { return [] }
        let count = to - from
        precondition(count <= window, "ring range exceeds window")
        let start = from % window
        if start + count <= window {
            return [array[.ellipsis, start ..< (start + count), 0...]]
        }
        return [
            array[.ellipsis, start ..< window, 0...],
            array[.ellipsis, 0 ..< (start + count - window), 0...],
        ]
    }

    private func temporalOrder(_ array: MLXArray, from: Int, to: Int) -> MLXArray {
        let slices = ringSlices(array, from: from, to: to)
        switch slices.count {
        case 0:
            return array[.ellipsis, 0 ..< 0, 0...]
        case 1:
            return slices[0]
        default:
            return concatenated(slices, axis: 2)
        }
    }

    private func writeRing(_ buffer: MLXArray, tokens: MLXArray, firstPosition: Int) {
        let n = tokens.dim(2)
        precondition(n <= window, "writeRing: more tokens than slots")
        let start = firstPosition % window
        if start + n <= window {
            buffer[.ellipsis, start ..< (start + n), 0...] = tokens
        } else {
            let first = window - start
            buffer[.ellipsis, start ..< window, 0...] = tokens[.ellipsis, ..<first, 0...]
            buffer[.ellipsis, 0 ..< (n - first), 0...] = tokens[.ellipsis, first..., 0...]
        }
    }

    private func allocateIfNeeded(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        guard keys == nil else { return }
        keys = MLXArray.zeros(
            [1, kvHeads, window, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
        values = MLXArray.zeros(
            [1, kvHeads, window, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
    }
}
