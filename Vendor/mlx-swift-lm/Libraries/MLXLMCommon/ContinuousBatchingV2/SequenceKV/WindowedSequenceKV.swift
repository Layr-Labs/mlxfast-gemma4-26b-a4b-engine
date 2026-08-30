// WindowedSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for SLIDING-WINDOW attention.
//
// The window is enforced by STORAGE EVICTION keyed to absolute positions —
// never by masks over shared buffers (report 10 §4 invariant 6). Storage is
// a ring of exactly `window` slots; the token at absolute position `p` lives
// in physical slot `p % window`, so eviction is simply "newer tokens
// overwrite slots window positions behind them". The RECENT end is always
// kept. `absoluteOffset` keeps counting past the window.

import Foundation
import MLX
import MLXFast

/// `CBv2SequenceKV` for sliding-window attention.
///
/// ## Temporal-order guarantee
/// Every array this class returns (from `update` and `snapshot`) is in
/// temporal order (oldest → newest). Unwinding the ring costs at most one
/// concat of two slices; in the common pre-wrap decode case it is a single
/// zero-copy slice.
///
/// ## Multi-token updates (prefill chunks)
/// For `n > 1` the returned views are `retainedHistory (≤ window-1 entries)
/// ++ the n new tokens` — i.e. up to `window - 1 + n` entries, which can
/// exceed `retainedCount`. This is required for correctness: the FIRST token
/// of the chunk must attend to the `window - 1` tokens before it, which the
/// ring evicts as the chunk is written. The attention layer applies a
/// causal∧window mask whenever the returned length exceeds `window`
/// (a pure function of lengths — see `CBv2AttentionV1.maskMode`).
///
/// ## Rollback and un-wrapping
/// `rollback(n)` moves `absoluteOffset` back by `n`. Before the ring wraps
/// this recovers the previous state exactly. After wrapping, the speculative
/// tokens' writes have already destroyed the `n` OLDEST in-window entries
/// (their slots alias positions exactly `window` behind), so the retained
/// count shrinks to `window - n` until fresh tokens refill the window. This
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
/// would return and advance `absoluteOffset`, while destructive ring writes
/// (and the `oldestValidPosition` advance) are deferred to
/// `commitSpeculativeWrite()`. A final `rollback(m)` is a pure counter move —
/// nothing was destroyed — so after commit the state is value-exactly what
/// plain updates of only the confirmed tokens would have produced.
public final class CBv2WindowedSequenceKV: CBv2DecodeRootCompactionCapableSequenceKV,
    CBv2InnerStateProviding
{

    /// Window size in tokens == number of physical ring slots.
    public let window: Int

    /// Absolute position of the next token to be written.
    public private(set) var absoluteOffset: Int

    /// Absolute position of the oldest entry that is still physically valid.
    /// Monotonically non-decreasing (see rollback discussion above).
    private var oldestValidPosition: Int

    public var retainedCount: Int { absoluteOffset - oldestValidPosition }

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    /// KVQ-001: 8-bit affine mirror of the ring, maintained ALONGSIDE the
    /// bf16 ring (which stays the source of truth for every logical view,
    /// borrow, staging and rollback path). Only the B=8 full-ring decode
    /// pass-A kernels read it; halving their K/V bytes is the entire point.
    ///
    /// Layout: `[2, kvHeads, window, headDim + 4]` uint8 — plane 0 keys,
    /// plane 1 values; per (plane, head, slot) the first `headDim` bytes are
    /// the affine-quantized values and the trailing 4 bytes are the fp16
    /// (scale, bias) pair for that slot, so one buffer per row carries
    /// everything a kernel needs (Metal's 31-buffer limit rules out separate
    /// scale arrays at batch 8).
    ///
    /// Consistency: the mirror mirrors the bf16 ring slot-for-slot and is
    /// written at exactly the writes that mutate the ring (`writeRing`,
    /// `writeDecodeToken`, and the fused in-kernel store). Rollback moves
    /// counters only — both buffers keep the same bytes — so validity is
    /// tracked by the same `oldestValidPosition`/`absoluteOffset` pair and
    /// the mirror needs no bookkeeping of its own.
    private var quantMirror: MLXArray?

    /// `MLX_KV_QUANT=0` disables the quantized-ring read path wholesale
    /// (mirror never allocated, kernels take the established bf16 road).
    /// Default ON. `MLX_` prefix: the worker env sanitizer only passes
    /// that namespace through.
    static let quantEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `MLX_KV_QUANT_SIM=1` (default OFF, local only): after every ring
    /// write, overwrite the bf16 slot with its quantize→dequantize round
    /// trip, so the SINGLE-STREAM fallback path — the one a local
    /// `--local-iterate` golden run exercises — sees exactly the values the
    /// quantized kernels would reconstruct at B = 8. That turns the local
    /// teacher-forced golden into a real end-to-end drift measurement for
    /// this mechanism. Never set on the ranked box.
    static let quantSimulate: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KV_QUANT_SIM"] ?? "")
                .lowercased())
    }()

    private var quantEligible: Bool {
        Self.quantEnabled && headDim == 256 && window > 0
            && (window & (window - 1)) == 0
    }

    /// Step-scoped PRE-EVICTION views captured by the most recent MULTI-token
    /// `update()` (`retainedHistory ++ chunk`, up to `window - 1 + n` entries).
    /// KV-borrowing layers (Gemma-4 cross-layer sharing) attend these instead
    /// of the post-eviction ring, so a chunk's earliest queries still see
    /// their full window — the ring writes have already destroyed those
    /// entries (slot aliasing at distance `window`). nil after a decode
    /// update, rollback, or before any update. Retaining these views keeps
    /// the pre-write buffer alive only until the next `update()` replaces
    /// them (bounded: one extra window-sized buffer between a chunk update
    /// and the following update).
    private var borrowableChunkViews: (keys: MLXArray, values: MLXArray)?

    /// Transaction opened by `beginSpeculativeWrite()` and closed by commit.
    /// Every intervening update stages instead of writing the ring.
    private var speculativeWriteArmed = false

    /// The accumulated speculative updates plus the absolute position the
    /// transaction started at. The ring writes
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
    }

    public var byteCount: Int {
        // Staged tensors are physically held until commit (bounded: one
        // 1+k-token chunk per in-flight MTP round).
        // KVQ-001: the packed 8-bit mirror is real resident memory and must
        // be visible to the engine's capacity accounting (the original
        // KVQ-001 revision omitted it, under-counting ~850 MB at B=8).
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
            + (staged.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
            + (quantMirror?.nbytes ?? 0)
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
            // Decode fast path: one modular slot write, then the retained
            // window in temporal order (1 slice pre-wrap, 2-slice concat after).
            writeDecodeToken(keys: newKeys, values: newValues)
            return (
                temporalOrder(keys!, from: oldestValidPosition, to: absoluteOffset),
                temporalOrder(values!, from: oldestValidPosition, to: absoluteOffset)
            )
        }

        // Prefill chunk: capture the retained history VIEWS before the ring
        // writes evict it (the views reference the pre-write buffer contents;
        // slice-update produces a new buffer, so they stay valid).
        let historyCount = min(retainedCount, window - 1)
        let historyFrom = absoluteOffset - historyCount
        var kParts = ringSlices(keys!, from: historyFrom, to: absoluteOffset)
        var vParts = ringSlices(values!, from: historyFrom, to: absoluteOffset)
        kParts.append(newKeys)
        vParts.append(newValues)
        let returnedKeys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let returnedValues = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)

        // Write the new tokens into the ring. Only the last `window` matter
        // when the chunk itself exceeds the window.
        let writeCount = min(n, window)
        let firstWritten = absoluteOffset + n - writeCount
        let kTail = writeCount == n ? newKeys : newKeys[.ellipsis, (n - writeCount)..., 0...]
        let vTail = writeCount == n ? newValues : newValues[.ellipsis, (n - writeCount)..., 0...]
        writeRing(keys!, tokens: kTail, firstPosition: firstWritten, mirrorPlane: 0)
        writeRing(values!, tokens: vTail, firstPosition: firstWritten, mirrorPlane: 1)

        absoluteOffset += n
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)

        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    func decodeRingWrite(keys newKeys: MLXArray, values newValues: MLXArray) {
        precondition(staged == nil && newKeys.dim(2) == 1 && newValues.dim(2) == 1)
        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)
        writeDecodeToken(keys: newKeys, values: newValues)
    }

    var decodeRingView: (keys: MLXArray, values: MLXArray, start: Int)? {
        guard staged == nil, let keys, let values, retainedCount == window else { return nil }
        return (keys, values, oldestValidPosition % window)
    }

    /// KVQ-001: the packed 8-bit mirror for the same full-ring decode step
    /// `decodeRingView` describes, or nil when the quantized road is off.
    /// The same allocation serves the before-write (fused) step: the fused
    /// quantized kernel stores the new token's mirror bytes itself, exactly
    /// as it stores the bf16 ones.
    var decodeRingQuantView: MLXArray? {
        guard staged == nil, quantMirror != nil, retainedCount == window
        else { return nil }
        return quantMirror
    }

    /// The ring view a fused decode step should attend: the SAME allocations
    /// and the SAME start `decodeRingView` would report AFTER this step's
    /// one-token `decodeRingWrite`, offered before that write happens.
    ///
    /// Only defined on an already-full ring — the identical predicate the
    /// separate-write path uses — where the append evicts exactly one entry,
    /// so `oldestValidPosition` advances by one and the post-write start is
    /// `(oldestValidPosition + 1) % window`. The physical slot that write
    /// lands in is `absoluteOffset % window`, i.e. `(start + window - 1) %
    /// window` — the slot the returned start has just stepped past.
    var decodeRingViewBeforeWrite: (keys: MLXArray, values: MLXArray, start: Int)? {
        guard staged == nil, let keys, let values, retainedCount == window else { return nil }
        return (keys, values, (oldestValidPosition + 1) % window)
    }

    /// Bookkeeping half of a fused decode step. The attention kernel already
    /// stored this step's token into the ring allocation in place, so advance
    /// exactly the counters `writeDecodeToken` would and construct no
    /// `SliceUpdate`. Precondition mirrors `decodeRingViewBeforeWrite`, which
    /// the caller must have consulted for the very same step.
    func advanceDecodeRingAfterFusedWrite() {
        precondition(
            staged == nil && keys != nil && retainedCount == window,
            "CBv2WindowedSequenceKV: fused ring advance outside a full-ring decode step")
        borrowableChunkViews = nil
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    // MARK: - Speculative (MTP) staging

    /// Staging always supported: the ring defers its destructive writes to
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
    /// but touch NEITHER the ring
    /// buffers NOR `oldestValidPosition` — the destructive writes are
    /// deferred to `commitSpeculativeWrite()` so a `rollback` in between is
    /// a pure counter move.
    ///
    /// n == 1 equivalence with the plain decode return: plain writes the
    /// token then returns the ring `[max(oldest, offset+1-window),
    /// offset+1)`. `history ++ token` with `historyCount = min(retained,
    /// window - 1)` yields the same entries — a below-full ring keeps all
    /// history, and a full ring drops exactly the one entry (position
    /// `offset - window`) the plain write would have destroyed. Pinned by
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
        // KV-borrowing layers attend the SAME views this step (the ring does
        // not hold the staged tokens yet) — set for n == 1 too, unlike the
        // plain decode path where the post-write ring is already exact.
        borrowableChunkViews = (returnedKeys, returnedValues)
        return (returnedKeys, returnedValues)
    }

    public func commitSpeculativeWrite() {
        speculativeWriteArmed = false
        guard let staged else { return }
        self.staged = nil
        // Confirmed range after finalize-time rollback: [basePosition,
        // absoluteOffset). Write it into the ring exactly as the plain
        // multi-token path would (only the last `window` matter when the
        // confirmed span exceeds the window); a fully rolled-back
        // (cancelled) row writes nothing.
        let confirmed = absoluteOffset - staged.basePosition
        if confirmed > 0 {
            allocateIfNeeded(keyTemplate: staged.keys, valueTemplate: staged.values)
            let writeCount = min(confirmed, window)
            let skip = confirmed - writeCount
            writeRing(
                keys!, tokens: staged.keys[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount, mirrorPlane: 0)
            writeRing(
                values!, tokens: staged.values[.ellipsis, skip ..< confirmed, 0...],
                firstPosition: absoluteOffset - writeCount, mirrorPlane: 1)
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
    /// the source layer attended, and the ring has already evicted the old
    /// ones. After a decode update (or a rollback) it is the retained ring,
    /// identical to `snapshot()`. Step-scoped: valid between the source
    /// layer's `update()` and the next mutation; never retain across steps.
    public func borrowableViews() -> (keys: MLXArray, values: MLXArray) {
        if let views = borrowableChunkViews { return views }
        let snap = snapshot()
        return (snap.keys, snap.values)
    }

    /// Decode normally borrows the retained ring snapshot. During a staged
    /// serial MTP transaction the ring deliberately has not been written, so
    /// the source layer's current logical post-update view is authoritative.
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

    /// Value-exact snapshot while a speculative update is staged: the ring
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

    /// Advance the position counter WITHOUT writing storage (contract
    /// `CBv2SequenceKV.fastForward(to:)`). Only valid on a FRESH state, used
    /// during prefix-cache adoption so the engine's trailing-window replay
    /// lands at true absolute positions.
    public func fastForward(to offset: Int) {
        precondition(
            keys == nil && absoluteOffset == oldestValidPosition,
            "CBv2WindowedSequenceKV.fastForward requires a fresh state")
        // A fully-rolled-back staged row can look "fresh" (offset back at
        // oldestValidPosition, ring never allocated) while a commit is
        // still owed — exclude it explicitly.
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
            // Pure counter move: the staged tokens were never written to
            // the ring, so nothing was destroyed and the retained window
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
        // oldestValidPosition deliberately does NOT decrease: if the ring had
        // wrapped, the rolled-back tokens' writes destroyed the oldest `n`
        // in-window entries (slot aliasing at distance `window`), so the
        // window shrinks until refilled. Pre-wrap, oldestValidPosition is
        // still the initial offset and the rollback is a full recovery.
        absoluteOffset -= n
        // Any captured pre-eviction chunk views now cover rolled-back
        // positions — invalidate so borrowing falls back to the ring.
        borrowableChunkViews = nil
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values, quantMirror].compactMap { $0 }
    }

    // MARK: - Ring geometry

    private func writeDecodeToken(keys newKeys: MLXArray, values newValues: MLXArray) {
        borrowableChunkViews = nil
        writeRing(keys!, tokens: newKeys, firstPosition: absoluteOffset, mirrorPlane: 0)
        writeRing(values!, tokens: newValues, firstPosition: absoluteOffset, mirrorPlane: 1)
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    /// Views covering absolute positions `[from, to)` in temporal order:
    /// one slice when the modular range does not cross the wrap point,
    /// two slices when it does. `to - from` must be ≤ `window`.
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

    /// Write `tokens` (≤ window of them) into their modular slots, splitting
    /// at the wrap point when needed (at most two slice assignments).
    ///
    /// `mirrorPlane` (0 = keys, 1 = values) additionally keeps the KVQ-001
    /// mirror in step when the quantized road is on, and — under
    /// `MLX_KV_QUANT_SIM` — replaces the bf16 payload with its
    /// quantize→dequantize round trip so the single-stream fallback sees the
    /// mirror's numerics.
    private func writeRing(
        _ buffer: MLXArray, tokens: MLXArray, firstPosition: Int, mirrorPlane: Int? = nil
    ) {
        var tokens = tokens
        let n = tokens.dim(2)
        precondition(n <= window, "writeRing: more tokens than slots")
        if let plane = mirrorPlane, quantMirror != nil {
            writeMirror(plane: plane, tokens: tokens, firstPosition: firstPosition)
            if Self.quantSimulate {
                tokens = Self.quantRoundTrip(tokens)
            }
        }
        let start = firstPosition % window
        if start + n <= window {
            buffer[.ellipsis, start ..< (start + n), 0...] = tokens
        } else {
            let first = window - start
            buffer[.ellipsis, start ..< window, 0...] = tokens[.ellipsis, ..<first, 0...]
            buffer[.ellipsis, 0 ..< (n - first), 0...] = tokens[.ellipsis, first..., 0...]
        }
    }

    // MARK: - KVQ-001 quantized mirror

    /// fp16-rounded per-(head, token) affine parameters for `x`
    /// (`[..., headDim]` over the last axis). The fp16 rounding is applied
    /// BEFORE quantization so the host packer, the fused kernel's writer and
    /// the sim round trip all reconstruct with the identical (scale, bias)
    /// the mirror actually stores.
    private static func quantParams(_ f: MLXArray) -> (scale: MLXArray, bias: MLXArray) {
        let mn = f.min(axis: -1, keepDims: true)
        let mx = f.max(axis: -1, keepDims: true)
        let scale = maximum((mx - mn) / 255, MLXArray(Float(1e-6)))
            .asType(.float16).asType(.float32)
        let bias = mn.asType(.float16).asType(.float32)
        return (scale, bias)
    }

    /// `[1, kvHeads, n, headDim]` bf16 → packed mirror rows
    /// `[kvHeads, n, headDim + 4]` uint8 (values ++ fp16 scale ++ fp16 bias).
    private static func quantPack(_ x: MLXArray) -> MLXArray {
        let f = x[0].asType(.float32)
        let (scale, bias) = quantParams(f)
        let q = clip(round((f - bias) / scale), min: 0, max: 255).asType(.uint8)
        let sBytes = scale.asType(.float16).view(dtype: .uint8)
        let bBytes = bias.asType(.float16).view(dtype: .uint8)
        return concatenated([q, sBytes, bBytes], axis: -1)
    }

    /// KVQ-GPUPACK: `MLX_KV_QUANT_GPUPACK=0` routes mirror packing back
    /// through the host expression above. Default ON: one kernel dispatch
    /// replaces the ~8-op MLX expression per (plane, chunk) write, which is
    /// the whole prefill cost of the mirror.
    static let gpuPackEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_GPUPACK"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `MLX_KV_QUANT_PACK_CHECK=1` (local only): compute BOTH packers on
    /// every write and fail loudly on any byte mismatch. The GPU packer is
    /// only legitimate while it reproduces the host packer bit for bit.
    static let gpuPackCheck: Bool = {
        ["1", "true", "yes", "on"].contains(
            (ProcessInfo.processInfo.environment["MLX_KV_QUANT_PACK_CHECK"] ?? "")
                .lowercased())
    }()

    /// One threadgroup (32 lanes) per (head, token) row: simd min/max over
    /// the 256 payload values, the SAME fp16-rounded (scale, bias) the host
    /// packer stores (`metal::rint` matches MLX `round`), quantized bytes
    /// plus the fp16 tail written at the identical offsets.
    private static let quantPackKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kvq8_pack_d256_v1",
        inputNames: ["x"],
        outputNames: ["packed"],
        source: """
            constexpr int D = 256;
            constexpr int simd_width = 32;
            constexpr int row_stride = D + 4;

            const int row = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* xr = x + row * D;

            float vmin = 3.402823466e+38F;
            float vmax = -3.402823466e+38F;
            for (int i = lane; i < D; i += simd_width) {
                const float v = float(xr[i]);
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
            vmin = simd_min(vmin);
            vmax = simd_max(vmax);

            const half hs = half(max((vmax - vmin) / 255.0f, 1e-6f));
            const half hb = half(vmin);
            const float s = float(hs);
            const float b = float(hb);

            device uint8_t* out = packed + row * row_stride;
            for (int i = lane; i < D; i += simd_width) {
                const float q = metal::rint((float(xr[i]) - b) / s);
                out[i] = uint8_t(clamp(q, 0.0f, 255.0f));
            }
            if (lane == 0) {
                device half* tail = (device half*)(out + D);
                tail[0] = hs;
                tail[1] = hb;
            }
        """
    )

    /// `[1, kvHeads, n, headDim]` -> packed `[kvHeads, n, headDim + 4]`
    /// uint8, byte-identical to `quantPack`.
    private static func quantPackGPU(_ x: MLXArray) -> MLXArray {
        let kvHeads = x.dim(1)
        let n = x.dim(2)
        let rows = kvHeads * n
        return quantPackKernel(
            [x],
            template: [("T", x.dtype)],
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[kvHeads, n, x.dim(3) + 4]],
            outputDTypes: [.uint8]
        )[0]
    }

    /// The bf16 values the mirror reconstructs for `x` — the sim harness's
    /// stand-in for the quantized kernels' dequantized reads.
    static func quantRoundTrip(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let (scale, bias) = quantParams(f)
        let q = clip(round((f - bias) / scale), min: 0, max: 255)
        return (q * scale + bias).asType(x.dtype)
    }

    private func writeMirror(plane: Int, tokens: MLXArray, firstPosition: Int) {
        guard let quantMirror else { return }
        let packedFlat: MLXArray
        if Self.gpuPackCheck {
            let gpu = Self.quantPackGPU(tokens)
            let host = Self.quantPack(tokens)
            let mismatches = (gpu .!= host).sum().item(Int.self)
            if mismatches != 0 {
                FileHandle.standardError.write(
                    "[kvq-gpupack] BYTE MISMATCH: \(mismatches) bytes differ\n"
                        .data(using: .utf8)!)
            }
            packedFlat = host
        } else if Self.gpuPackEnabled {
            packedFlat = Self.quantPackGPU(tokens)
        } else {
            packedFlat = Self.quantPack(tokens)
        }
        let packed = packedFlat.expandedDimensions(axis: 0)
        let n = tokens.dim(2)
        let start = firstPosition % window
        if start + n <= window {
            quantMirror[plane ..< (plane + 1), 0..., start ..< (start + n), 0...] = packed
        } else {
            let first = window - start
            quantMirror[plane ..< (plane + 1), 0..., start ..< window, 0...] =
                packed[.ellipsis, ..<first, 0...]
            quantMirror[plane ..< (plane + 1), 0..., 0 ..< (n - first), 0...] =
                packed[.ellipsis, first..., 0...]
        }
    }

    private func allocateIfNeeded(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        guard keys == nil else { return }
        keys = MLXArray.zeros(
            [1, kvHeads, window, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
        values = MLXArray.zeros(
            [1, kvHeads, window, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
        if quantEligible, keyTemplate.dtype == .bfloat16,
            keyTemplate.dim(3) == headDim, valueTemplate.dim(3) == headDim
        {
            quantMirror = MLXArray.zeros(
                [2, kvHeads, window, headDim + 4], dtype: .uint8)
        }
    }
}
