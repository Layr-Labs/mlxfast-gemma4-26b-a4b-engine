// WindowedSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for SLIDING-WINDOW attention.
//
// The window is enforced by STORAGE EVICTION keyed to absolute positions —
// never by masks over shared buffers (report 10 §4 invariant 6). Storage is a
// temporal-order linear buffer with a small bounded tail. Decode appends into
// that tail and returns one contiguous slice; only when the tail fills is the
// live window compacted back to the front. The RECENT end is always kept and
// `absoluteOffset` keeps counting past the window.

import Foundation
import MLX

/// `CBv2SequenceKV` for sliding-window attention.
///
/// ## Temporal-order guarantee
/// Every array this class returns (from `update` and `snapshot`) is in
/// temporal order (oldest → newest). Every ordinary decode return is one
/// zero-copy slice. A bounded compaction copies the live window only after the
/// linear tail is exhausted; there is no per-token post-wrap concatenation.
///
/// ## Multi-token updates (prefill chunks)
/// For `n > 1` the returned views are `retainedHistory (≤ window-1 entries)
/// ++ the n new tokens` — i.e. up to `window - 1 + n` entries, which can
/// exceed `retainedCount`. This is required for correctness: the FIRST token
/// of the chunk must attend to the `window - 1` tokens before it, which the
/// storage evicts as the chunk is written. The attention layer applies a
/// causal∧window mask whenever the returned length exceeds `window`
/// (a pure function of lengths — see `CBv2AttentionV1.maskMode`).
///
/// ## Rollback after window eviction
/// `rollback(n)` moves `absoluteOffset` back by `n`. Before the window first
/// fills this recovers the previous state exactly. After eviction, speculative
/// tokens have already evicted the `n` OLDEST in-window entries, so the
/// retained count shrinks to `window - n` until fresh tokens refill the
/// window. This
/// is tracked by `oldestValidPosition`, which is monotonically non-decreasing
/// — destroyed history can never be re-exposed, and the un-confirmed tail is
/// structurally unreachable (all views are keyed to
/// `[oldestValidPosition, absoluteOffset)`).
///
/// ## Staged speculative writes (MTP verify rounds)
/// The post-wrap window shrink above is CORRECT but numerically divergent
/// from a non-speculative run, which breaks the MTP acceptance gate (MTP-on
/// must be token-exact vs MTP-off for greedy). `beginSpeculativeWrite()`
/// therefore opens a STAGED transaction: one multi-token update or several
/// serial one-token updates return exactly the views their plain equivalents
/// would return and advance `absoluteOffset`, while destructive storage writes
/// (and the `oldestValidPosition` advance) are deferred to
/// `commitSpeculativeWrite()`. A final `rollback(m)` is a pure counter move —
/// nothing was destroyed — so after commit the state is value-exactly what
/// plain updates of only the confirmed tokens would have produced.
public final class CBv2WindowedSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    /// Window size in tokens == maximum number of logically retained slots.
    public let window: Int

    /// Decode tail policy. Large windows receive at most 256 additional slots
    /// (and at most 12.5%). For small windows compaction is cheap, while even
    /// minimum slack would be a disproportionate footprint, so they stay exact.
    private static let maxLinearSlack = 256
    private static let minimumWindowForLinearSlack = 64

    static func storageSlotCount(for window: Int) -> Int {
        guard window >= minimumWindowForLinearSlack else { return window }
        return window + min(maxLinearSlack, window / 8)
    }

    /// Absolute position of the next token to be written.
    public private(set) var absoluteOffset: Int

    /// Absolute position of the oldest entry that is still logically valid.
    /// Monotonically non-decreasing (see rollback discussion above).
    private var oldestValidPosition: Int

    public var retainedCount: Int { absoluteOffset - oldestValidPosition }

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    /// Absolute position represented by physical slot zero. Logical retained
    /// positions are always one contiguous subrange of the linear buffers.
    private var bufferBasePosition: Int

    /// Physical slots per K/V buffer (`window + bounded slack`).
    private let storageCapacity: Int

    /// Step-scoped PRE-EVICTION views captured by the most recent MULTI-token
    /// `update()` (`retainedHistory ++ chunk`, up to `window - 1 + n` entries).
    /// KV-borrowing layers (Gemma-4 cross-layer sharing) attend these instead
    /// of the post-eviction storage, so a chunk's earliest queries still see
    /// their full window — the storage writes have already evicted those
    /// entries. nil after a decode
    /// update, rollback, or before any update. Retaining these views keeps
    /// the pre-write buffer alive only until the next `update()` replaces
    /// them (bounded: one extra window-sized buffer between a chunk update
    /// and the following update).
    private var borrowableChunkViews: (keys: MLXArray, values: MLXArray)?

    /// Transaction opened by `beginSpeculativeWrite()` and closed by commit.
    /// Every intervening update stages instead of writing storage.
    private var speculativeWriteArmed = false

    /// The accumulated speculative updates plus the absolute position the
    /// transaction started at. The storage writes
    /// for the still-confirmed range `[basePosition, absoluteOffset)` happen
    /// at `commitSpeculativeWrite()`. At most one transaction per row.
    private var staged: (keys: MLXArray, values: MLXArray, basePosition: Int)?

    /// - Parameters:
    ///   - window: sliding window in tokens (> 0).
    ///   - initialOffset: absolute position this sequence starts at. Non-zero
    ///     when a prefix-cache hit starts finite-window replay at C. The row
    ///     starts empty at C while owning full rows may retain immutable K/V
    ///     through M; absolute RoPE positions therefore remain aligned.
    public init(window: Int, kvHeads: Int, headDim: Int, initialOffset: Int = 0) {
        precondition(window > 0, "CBv2WindowedSequenceKV: window must be > 0")
        precondition(initialOffset >= 0, "CBv2WindowedSequenceKV: negative initialOffset")
        self.window = window
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.absoluteOffset = initialOffset
        self.oldestValidPosition = initialOffset
        self.bufferBasePosition = initialOffset
        self.storageCapacity = Self.storageSlotCount(for: window)
    }

    public var byteCount: Int {
        // Staged tensors are physically held until commit (bounded: one
        // 1+k-token chunk per in-flight MTP round).
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

        if n == 1 {
            // Decode fast path: one linear append and one temporal-order slice.
            // Once the bounded tail fills, `store` compacts the live history;
            // every intervening post-window token remains concat-free.
            borrowableChunkViews = nil  // storage == window: snapshot is exact
            store(keys: newKeys, values: newValues, firstPosition: absoluteOffset)
            absoluteOffset += 1
            oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
            return (
                linearView(keys!, from: oldestValidPosition, to: absoluteOffset),
                linearView(values!, from: oldestValidPosition, to: absoluteOffset)
            )
        }

        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)

        // Prefill chunk: capture the retained history VIEW before storage
        // writes evict it (the view references the pre-write buffer contents;
        // slice-update produces a new buffer, so they stay valid).
        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        let returnedKeys: MLXArray
        let returnedValues: MLXArray
        if historyCount == 0 {
            returnedKeys = newKeys
            returnedValues = newValues
        } else {
            returnedKeys = concatenated(
                [linearView(keys!, from: historyFrom, to: absoluteOffset), newKeys], axis: 2)
            returnedValues = concatenated(
                [linearView(values!, from: historyFrom, to: absoluteOffset), newValues], axis: 2)
        }

        // Only the last `window` tokens matter when the chunk itself exceeds
        // the window. `store` chooses append, compaction, or tail replacement.
        store(keys: newKeys, values: newValues, firstPosition: absoluteOffset)

        absoluteOffset += n
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)

        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    // MARK: - Speculative (MTP) staging

    /// Staging always supported: storage defers its destructive writes to
    /// `commitSpeculativeWrite()`, so speculative rollback is value-exact.
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

    /// One update in the active transaction: return EXACTLY the views the
    /// corresponding plain update would return and advance `absoluteOffset`,
    /// but touch NEITHER the linear buffers NOR `oldestValidPosition` — the
    /// destructive writes are
    /// deferred to `commitSpeculativeWrite()` so a `rollback` in between is
    /// a pure counter move.
    ///
    /// n == 1 equivalence with the plain decode return: plain writes the
    /// token then returns storage `[max(oldest, offset+1-window),
    /// offset+1)`. `history ++ token` with `historyCount = min(retained,
    /// window - 1)` yields the same entries — below-full storage keeps all
    /// history, and a full window drops exactly the one entry (position
    /// `offset - window`) the plain write evicts. Pinned by
    /// `CBv2MTPKVStagingTests`.
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
        // KV-borrowing layers attend the SAME views this step (storage does
        // not hold the staged tokens yet) — set for n == 1 too, unlike the
        // plain decode path where post-write storage is already exact.
        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    public func commitSpeculativeWrite() {
        speculativeWriteArmed = false
        guard let staged else { return }
        self.staged = nil
        // Confirmed range after finalize-time rollback: [basePosition,
        // absoluteOffset). Write it into storage exactly as the plain
        // multi-token path would (only the last `window` matter when the
        // confirmed span exceeds the window); a fully rolled-back
        // (cancelled) row writes nothing.
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            store(
                keys: staged.keys[.ellipsis, ..<confirmed, 0...],
                values: staged.values[.ellipsis, ..<confirmed, 0...],
                firstPosition: staged.basePosition)
        }
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
        // Step-scoped views die at finalize (the plain paths replace them on
        // the next update; nothing borrows between commit and that update).
        borrowableChunkViews = nil
    }

    /// Views a KV-BORROWING layer must attend against in the CURRENT step:
    /// exactly what the most recent `update()` returned. After a multi-token
    /// (prefill-chunk) update this is the PRE-eviction history + chunk — the
    /// borrowing layer's chunk queries need the same `window - 1 + n` entries
    /// the source layer attended, and storage has already evicted the old
    /// ones. After a decode update (or a rollback) it is retained storage,
    /// identical to `snapshot()`. Step-scoped: valid between the source
    /// layer's `update()` and the next mutation; never retain across steps.
    public func borrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if let views = borrowableChunkViews { return views }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    /// Decode normally borrows the retained storage snapshot. During a staged
    /// serial MTP transaction the buffers deliberately have not been written,
    /// so the source layer's current logical post-update view is authoritative.
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

    /// The post-update physical ring for the exact full-window decode path.
    /// Consumers must restore logical order while reading; ordinary callers
    /// continue to use `decodeBorrowableViews()` / `snapshot()`.
    func decodePhysicalRingViews() -> (keys: MLXArray, values: MLXArray)? {
        guard staged == nil, retainedCount == window, let keys, let values else {
            return nil
        }
        return (keys, values)
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
            linearView(keys, from: oldestValidPosition, to: absoluteOffset),
            linearView(values, from: oldestValidPosition, to: absoluteOffset),
            absoluteOffset
        )
    }

    /// Value-exact snapshot while a speculative update is staged: storage
    /// holds `[oldestValidPosition, basePosition)` and the staged tensors
    /// hold the confirmed tail `[basePosition, absoluteOffset)` (rollback
    /// may have shrunk it). Transiently up to `window + n` entries — same
    /// exposure as a plain chunk update's pre-eviction views. Never on an
    /// engine path (windowed rows are not donated), but keeps
    /// `borrowableViews()`'s fallback correct after a mid-staged rollback.
    private func stagedSnapshot(
        _ staged: (keys: MLXArray, values: MLXArray, basePosition: Int)
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        var kParts: [MLXArray] = []
        var vParts: [MLXArray] = []
        if let keys, let values, staged.basePosition > oldestValidPosition {
            kParts = [linearView(keys, from: oldestValidPosition, to: staged.basePosition)]
            vParts = [linearView(values, from: oldestValidPosition, to: staged.basePosition)]
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

    /// Advance the position counter WITHOUT writing storage (contract
    /// `CBv2SequenceKV.fastForward(to:)`). Only valid on a FRESH state, used
    /// during prefix-cache adoption so the engine's trailing-window replay
    /// lands at true absolute positions.
    public func fastForward(to offset: Int) {
        precondition(
            keys == nil && absoluteOffset == oldestValidPosition,
            "CBv2WindowedSequenceKV.fastForward requires a fresh state")
        // A fully-rolled-back staged row can look "fresh" (offset back at
        // oldestValidPosition, storage never allocated) while a commit is
        // still owed — exclude it explicitly.
        precondition(
            !speculativeWriteArmed && staged == nil,
            "CBv2WindowedSequenceKV.fastForward with a speculative write pending")
        precondition(offset >= absoluteOffset, "fastForward cannot move backwards")
        absoluteOffset = offset
        oldestValidPosition = offset
        bufferBasePosition = offset
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2WindowedSequenceKV.rollback: negative n")
        if let staged {
            // Pure counter move: the staged tokens were never written to
            // storage, so nothing was destroyed and the retained window
            // does not shrink. The engine only ever rolls back tokens from
            // the staged round (a cancelled row may roll back ALL of them).
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
        // oldestValidPosition deliberately does NOT decrease: once the window
        // advanced, the rolled-back tokens logically evicted the oldest `n`
        // in-window entries, so the window shrinks until refilled. Before the
        // first eviction, oldestValidPosition is
        // still the initial offset and the rollback is a full recovery.
        absoluteOffset -= n
        // Any captured pre-eviction chunk views now cover rolled-back
        // positions — invalidate so borrowing falls back to storage.
        borrowableChunkViews = nil
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    // MARK: - Linear storage geometry

    /// One zero-copy temporal-order view over absolute positions `[from, to)`.
    private func linearView(_ array: MLXArray, from: Int, to: Int) -> MLXArray {
        precondition(from >= bufferBasePosition, "linear range precedes buffer base")
        precondition(to >= from, "negative linear range")
        let start = from - bufferBasePosition
        let end = to - bufferBasePosition
        precondition(end <= storageCapacity, "linear range exceeds storage capacity")
        return array[.ellipsis, start ..< end, 0...]
    }

    /// Store one committed chunk beginning at `firstPosition`.
    ///
    /// Fast path: append into the unused tail. Once that tail is full, retain
    /// only the history that the post-update window can still expose, compact
    /// it to slot zero, and append the chunk. Thus a window-sized copy occurs
    /// once per `storageCapacity - window + 1` decode tokens instead of a
    /// two-slice concatenation on every token after the window first fills.
    private func store(keys newKeys: MLXArray, values newValues: MLXArray, firstPosition: Int) {
        let n = newKeys.dim(2)
        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)

        // A window-sized (or larger) chunk replaces all prior attendable
        // history. Keep exactly its trailing window at the front, leaving the
        // bounded tail available for subsequent decode appends.
        if n >= window {
            let skip = n - window
            keys![.ellipsis, ..<window, 0...] = newKeys[.ellipsis, skip..., 0...]
            values![.ellipsis, ..<window, 0...] = newValues[.ellipsis, skip..., 0...]
            bufferBasePosition = firstPosition + skip
            return
        }

        let writeStart = firstPosition - bufferBasePosition
        if writeStart >= 0 && writeStart + n <= storageCapacity {
            keys![.ellipsis, writeStart ..< (writeStart + n), 0...] = newKeys
            values![.ellipsis, writeStart ..< (writeStart + n), 0...] = newValues
            return
        }

        // The append ran off the bounded tail (or a future caller moved to a
        // new absolute base). Compact only history that survives this update.
        let preserveFrom = max(oldestValidPosition, firstPosition + n - window)
        let preserveCount = firstPosition - preserveFrom
        precondition(preserveCount >= 0, "linear compaction has negative history")
        precondition(
            preserveCount + n <= window,
            "linear compaction exceeds logical window")

        if preserveCount == 0 {
            keys = MLXArray.zeros(
                [1, kvHeads, storageCapacity, keys!.dim(3)], dtype: keys!.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, storageCapacity, values!.dim(3)], dtype: values!.dtype)
        } else {
            let retainedKeys = linearView(keys!, from: preserveFrom, to: firstPosition)
            let retainedValues = linearView(values!, from: preserveFrom, to: firstPosition)
            let tailCount = storageCapacity - preserveCount
            keys = concatenated(
                [
                    retainedKeys,
                    MLXArray.zeros(
                        [1, kvHeads, tailCount, retainedKeys.dim(3)], dtype: retainedKeys.dtype),
                ], axis: 2)
            values = concatenated(
                [
                    retainedValues,
                    MLXArray.zeros(
                        [1, kvHeads, tailCount, retainedValues.dim(3)], dtype: retainedValues.dtype),
                ], axis: 2)
        }
        bufferBasePosition = preserveFrom
        keys![.ellipsis, preserveCount ..< (preserveCount + n), 0...] = newKeys
        values![.ellipsis, preserveCount ..< (preserveCount + n), 0...] = newValues
    }

    private func allocateIfNeeded(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        guard keys == nil else { return }
        keys = MLXArray.zeros(
            [1, kvHeads, storageCapacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
        values = MLXArray.zeros(
            [1, kvHeads, storageCapacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
    }
}
