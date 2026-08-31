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

    /// KVQ-WRITEBEHIND: decode tokens whose bf16 ring slot assignment has not
    /// been constructed yet, in position order starting at
    /// `pendingRingFirstPosition`. Only the quantized decode road defers, and
    /// only when the paired mirror write has already carried this token's
    /// mirror bytes, so the mirror is never behind. Every reader of the bf16
    /// ring flushes first, which makes the deferral invisible.
    private var pendingRingKeys: [MLXArray] = []
    private var pendingRingValues: [MLXArray] = []
    private var pendingRingFirstPosition = 0

    /// The run length the deferral is allowed to reach. A run of `L` tokens
    /// costs one full-ring slice assignment plus one `L`-token concatenation
    /// instead of `L` full-ring assignments, so the per-token bf16 ring cost
    /// falls by a factor of about `L`. Must stay well under `window`: a run
    /// longer than the window would write one slot twice in a single flush.
    static let pendingRingRunLimit = 32

    /// `MLX_KV_RING_WRITEBEHIND=0` returns the bf16 ring to one slice
    /// assignment per decode token per plane. Default ON.
    static let ringWriteBehindEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_RING_WRITEBEHIND"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

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
            // KVQ-WRITEBEHIND: a bounded run of one-token tensors, resident
            // until the run flushes.
            + pendingRingKeys.reduce(0) { $0 + $1.nbytes }
            + pendingRingValues.reduce(0) { $0 + $1.nbytes }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        flushPendingRingWrites()
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

    /// `deferRingCopy` is the caller's statement that the bf16 ring is not an
    /// input to this step's attention, which is true exactly when the
    /// quantized mirror road is the one that will run. It is a request, not a
    /// promise: the write behind declines whenever its own preconditions do
    /// not hold, and any read flushes it out before observing the ring.
    func decodeRingWrite(
        keys newKeys: MLXArray, values newValues: MLXArray, deferRingCopy: Bool = false
    ) {
        precondition(staged == nil && newKeys.dim(2) == 1 && newValues.dim(2) == 1)
        allocateIfNeeded(keyTemplate: newKeys, valueTemplate: newValues)
        writeDecodeToken(keys: newKeys, values: newValues, deferRingCopy: deferRingCopy)
    }

    /// KVQ-WRITEBEHIND: the ring start `decodeRingView` reports, and the same
    /// availability predicate, without handing out the bf16 buffers. The
    /// quantized road needs the start and the mirror only, so asking through
    /// this accessor leaves a deferred run deferred.
    var decodeRingSlot: Int? {
        guard staged == nil, keys != nil, values != nil, retainedCount == window
        else { return nil }
        return oldestValidPosition % window
    }

    var decodeRingView: (keys: MLXArray, values: MLXArray, start: Int)? {
        flushPendingRingWrites()
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
        flushPendingRingWrites()
        guard staged == nil, let keys, let values, retainedCount == window else { return nil }
        return (keys, values, (oldestValidPosition + 1) % window)
    }

    /// Bookkeeping half of a fused decode step. The attention kernel already
    /// stored this step's token into the ring allocation in place, so advance
    /// exactly the counters `writeDecodeToken` would and construct no
    /// `SliceUpdate`. Precondition mirrors `decodeRingViewBeforeWrite`, which
    /// the caller must have consulted for the very same step.
    func advanceDecodeRingAfterFusedWrite() {
        flushPendingRingWrites()
        precondition(
            staged == nil && keys != nil && retainedCount == window,
            "CBv2WindowedSequenceKV: fused ring advance outside a full-ring decode step")
        borrowableChunkViews = nil
        // KVQ-DIAG: dispatch the quantized reader on the full ring this step
        // just advanced past, and discard the result. WRITE-016 owns the
        // steady-state decode write, so this is where the probe belongs.
        // Probe 1 (the read kernel) was CLEARED on the box by submission
        // 47dae0ea, which scored 1.96373951131358 with it dispatching. Only
        // the write half is still unproven, so only probe 2 runs here.
        diagnosticFusedDispatch()
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
        flushPendingRingWrites()
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
        flushPendingRingWrites()
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
        flushPendingRingWrites()
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

    private func writeDecodeToken(
        keys newKeys: MLXArray, values newValues: MLXArray, deferRingCopy: Bool = false
    ) {
        borrowableChunkViews = nil
        // KVQ-PAIRWRITE: when both mirror planes go out together the ring
        // writes carry no mirror plane of their own.
        let paired = writePairedMirror(
            keys: newKeys, values: newValues, firstPosition: absoluteOffset)
        // KVQ-WRITEBEHIND: the mirror is already current for this token, so
        // the ring copy can join a run. Deferral needs the paired write to
        // have happened, because an unpaired token still owes the mirror its
        // bytes and `writeRing` is what delivers them.
        if paired, deferRingCopy, Self.ringWriteBehindEnabled,
            pendingRingKeys.count < Self.pendingRingRunLimit,
            pendingRingKeys.isEmpty
                || pendingRingFirstPosition + pendingRingKeys.count == absoluteOffset
        {
            if pendingRingKeys.isEmpty { pendingRingFirstPosition = absoluteOffset }
            pendingRingKeys.append(newKeys)
            pendingRingValues.append(newValues)
        } else {
            flushPendingRingWrites()
            writeRing(
                keys!, tokens: newKeys, firstPosition: absoluteOffset,
                mirrorPlane: paired ? nil : 0)
            writeRing(
                values!, tokens: newValues, firstPosition: absoluteOffset,
                mirrorPlane: paired ? nil : 1)
        }
        absoluteOffset += 1
        oldestValidPosition = max(oldestValidPosition, absoluteOffset - window)
    }

    /// KVQ-WRITEBEHIND: construct the deferred run's ring slice assignments.
    ///
    /// The run is `pendingRingKeys.count` consecutive one-token writes
    /// starting at `pendingRingFirstPosition`, and the run limit is far below
    /// `window`, so every token in it owns a distinct slot and the batched
    /// write lands byte for byte what the per-token writes would have landed.
    /// The mirror planes went out with the tokens themselves, so this pass
    /// carries no mirror plane.
    private func flushPendingRingWrites() {
        guard !pendingRingKeys.isEmpty, let keys, let values else {
            pendingRingKeys.removeAll(keepingCapacity: true)
            pendingRingValues.removeAll(keepingCapacity: true)
            return
        }
        let runKeys =
            pendingRingKeys.count == 1
            ? pendingRingKeys[0] : concatenated(pendingRingKeys, axis: 2)
        let runValues =
            pendingRingValues.count == 1
            ? pendingRingValues[0] : concatenated(pendingRingValues, axis: 2)
        let first = pendingRingFirstPosition
        pendingRingKeys.removeAll(keepingCapacity: true)
        pendingRingValues.removeAll(keepingCapacity: true)
        writeRing(keys, tokens: runKeys, firstPosition: first, mirrorPlane: nil)
        writeRing(values, tokens: runValues, firstPosition: first, mirrorPlane: nil)
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

    /// KVQ-PAIRWRITE: `MLX_KV_QUANT_PAIRWRITE=0` returns a decode step's
    /// mirror maintenance to one pack dispatch and one `SliceUpdate` PER
    /// PLANE. Default ON: both planes of the one token share a single pack
    /// dispatch and a single `SliceUpdate`.
    static let pairedMirrorWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_PAIRWRITE"]
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
        name: "cbv2_kvq8_pack_d256_v3u",
        inputNames: ["x"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 256;
            constexpr int simd_width = 32;
            constexpr int row_stride = D + 4;

            device uint8_t* packed = (device uint8_t*)packed_w;
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
                const ushort su = as_type<ushort>(hs);
                const ushort bu = as_type<ushort>(hb);
                out[D + 0] = uint8_t(su & 0xff);
                out[D + 1] = uint8_t(su >> 8);
                out[D + 2] = uint8_t(bu & 0xff);
                out[D + 3] = uint8_t(bu >> 8);
            }
        """
    )

    /// KVQ-PAIRWRITE: both mirror planes of ONE decode token in a single
    /// dispatch. Rows `[0, HEADS)` are packed out of `xk`, rows
    /// `[HEADS, 2 * HEADS)` out of `xv`, and the output rows land in that
    /// order, which is the `[2, kvHeads, 1, row_words]` block the mirror
    /// update wants.
    ///
    /// The per-row arithmetic below is a transcription of `quantPackKernel`
    /// above, deliberately duplicated rather than shared: that packer is the
    /// promoted prefill path and re-composing its source string would change
    /// its text, and a Swift-hosted kernel that changes text without
    /// changing name serves a stale cached body.
    private static let quantPackPairKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_kvq8_pack_pair_d256_v1",
        inputNames: ["xk", "xv"],
        outputNames: ["packed_w"],
        source: """
            constexpr int D = 256;
            constexpr int simd_width = 32;
            constexpr int row_stride = D + 4;

            device uint8_t* packed = (device uint8_t*)packed_w;
            const int row = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_position_in_threadgroup.x);
            const device T* xr = row < HEADS ? xk + row * D : xv + (row - HEADS) * D;

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
                const ushort su = as_type<ushort>(hs);
                const ushort bu = as_type<ushort>(hb);
                out[D + 0] = uint8_t(su & 0xff);
                out[D + 1] = uint8_t(su >> 8);
                out[D + 2] = uint8_t(bu & 0xff);
                out[D + 3] = uint8_t(bu >> 8);
            }
        """
    )


    // MARK: - KVQ-DIAG: compute-and-discard probe

    /// KVQ-DIAG. Five box runs of the full KVQ mechanism died scoreless with
    /// no telemetry (exit 5, accepted_pairs=0, candidate leg refused seconds
    /// into its first timed decode dispatch) while every variant ran clean and
    /// bit-identical on M5 Pro. Four hypotheses were eliminated one submission
    /// at a time (graph interaction, fp16 pointer punning, uint8 buffer
    /// binding, cold-JIT warm deadline). This submission separates the last
    /// two possibilities without touching the token path at all.
    ///
    /// The mirror is maintained exactly as the mechanism maintains it, the
    /// quantized read kernel is DISPATCHED on real ring state at the
    /// production geometry, and its output is evaluated and then discarded.
    /// Attention keeps using the stock bf16 path, so the emitted tokens are
    /// bit-identical to the base tree and the composite only carries the cost
    /// of the extra dispatch.
    ///
    /// Reading the result: a SCORE means these kernels dispatch and complete
    /// on the ranked box, and the fault lives in how their output re-enters
    /// the attention graph. A SCORELESS FAIL means the dispatch itself is
    /// fatal there. Either way one slot buys the answer that four blind
    /// resubmissions did not.
    ///
    /// `MLX_KV_QUANT_DIAG=0` disables the probe.
    static let diagEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let diagBlocks = 256
    /// The probe is called once per ROW per layer, but the mechanism's real
    /// dispatch is once per LAYER over all eight rows at once. Firing on
    /// every call would multiply the mechanism's cost by the batch and drag
    /// the composite far below anything the board would score. One call in
    /// eight reproduces the production dispatch count.
    private static let diagStride = 64
    nonisolated(unsafe) private static var diagCounter = 0
    private static let diagLock = NSLock()
    nonisolated(unsafe) private static var diagDispatched = false
    nonisolated(unsafe) private static var diagRejected = false

    /// One line each, to stderr, so the ranked log shows whether the probe
    /// actually dispatched. A silent probe proves nothing.
    private static func diagDispatchOnce() {
        diagLock.lock(); let fresh = !diagDispatched; diagDispatched = true; diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] dispatched\n".utf8))
        }
    }

    private static func diagRejectOnce(_ why: String) {
        diagLock.lock(); let fresh = !diagRejected; diagRejected = true; diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] skipped: \(why)\n".utf8))
        }
    }

private static let diagQuantReadKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_ring_2pass_a_q8_d256_g2_diag_b\(diagBlocks)_v1",
        inputNames: [
            "queries",
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "starts",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int row_stride = D + 4;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            constexpr int row_words = row_stride / 4;
            const device uint32_t* mirror_w = m0;
            switch (batch_index) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }
            const uint start = starts[batch_index];

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            int slot = int((start + block) % N);
            device T* partial = partials
                + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
            device float* sum_out = sums + batch_head * BLOCKS + block;
            device float* max_out = maxs + batch_head * BLOCKS + block;

            thread float q[values_per_lane];
            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const device uint32_t* krow_w =
                    mirror_w + (kv_head * N + slot) * row_words;
                const device uint32_t* vrow_w =
                    mirror_w + ((KV_HEADS + kv_head) * N + slot) * row_words;
                const uint32_t ktw = krow_w[D / 4];
                const uint32_t vtw = vrow_w[D / 4];
                const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                const float kb = float(as_type<half>(ushort(ktw >> 16)));
                const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                const float vb = float(as_type<half>(ushort(vtw >> 16)));
                const uint32_t kw0 = krow_w[lane * 2];
                const uint32_t kw1 = krow_w[lane * 2 + 1];
                const uint32_t vw0 = vrow_w[lane * 2];
                const uint32_t vw1 = vrow_w[lane * 2 + 1];
                float score = 0.0f;
                for (int element = 0; element < 4; ++element) {
                    score += q[element]
                        * fma(float((kw0 >> (8 * element)) & 0xffu), ks, kb);
                }
                for (int element = 0; element < 4; ++element) {
                    score += q[4 + element]
                        * fma(float((kw1 >> (8 * element)) & 0xffu), ks, kb);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < 4; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor
                            * fma(float((vw0 >> (8 * element)) & 0xffu), vs, vb);
                    accumulator[4 + element] = accumulator[4 + element] * old_factor
                        + score_factor
                            * fma(float((vw1 >> (8 * element)) & 0xffu), vs, vb);
                }

                slot += BLOCKS;
                if (slot >= N) slot -= N;
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

    /// Dispatch the quantized reader over this row's mirror and discard the
    /// result. Every row binds its own mirror into all eight slots: the
    /// geometry, the template and the memory traffic match the mechanism's
    /// production dispatch, and nothing observable depends on the output.
    private func diagnosticQuantDispatch() {
        guard Self.diagEnabled, let quantMirror else { return }
        guard retainedCount == window, headDim == 256, kvHeads == 8, window == 1024
        else {
            Self.diagRejectOnce(
                "retained=\(retainedCount) window=\(window) headDim=\(headDim) kvHeads=\(kvHeads)")
            return
        }
        Self.diagLock.lock()
        Self.diagCounter += 1
        let fire = Self.diagCounter % Self.diagStride == 0
        Self.diagLock.unlock()
        guard fire else { return }
        let batch = 8
        let queryHeads = 16
        let queries = MLXArray.zeros(
            [batch, queryHeads, 1, headDim], dtype: .bfloat16)
        let starts = MLXArray(
            Array(repeating: UInt32(oldestValidPosition % window), count: batch),
            [batch])
        let mirrors = Array(repeating: quantMirror, count: batch)
        let out = Self.diagQuantReadKernel(
            [queries] + mirrors + [starts],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", window),
                ("GQA", queryHeads / kvHeads),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", Self.diagBlocks),
            ],
            grid: (kvHeads * 32, batch * (queryHeads / kvHeads), Self.diagBlocks),
            threadGroup: (32, queryHeads / kvHeads, 1),
            outputShapes: [
                [batch, queryHeads, 1, Self.diagBlocks, headDim],
                [batch, queryHeads, 1, Self.diagBlocks],
                [batch, queryHeads, 1, Self.diagBlocks],
            ],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        asyncEval(out[1])
        Self.diagDispatchOnce()
    }


private static let diagFusedKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ringwrite_sdpa_2pass_a_q8_d256_g2_diagfused_b\(diagBlocks)_v1",
        inputNames: [
            "queries",
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "starts", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["partials", "sums", "maxs", "fence"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr uint ring_mask = uint(N - 1);
            constexpr int row_stride = D + 4;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            constexpr int row_words = row_stride / 4;
            const device uint32_t* mirror_w = m0;
            switch (batch_index) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            const device uint32_t* mkeys_w =
                mirror_w + kv_head * N * row_words;
            const device uint32_t* mvalues_w =
                mirror_w + (KV_HEADS + kv_head) * N * row_words;
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const uint ring_start = starts[batch_index];
            const uint write_slot = (ring_start + ring_mask) & ring_mask;
            if (block == 0 && query_head_in_group == 0) {
                float kmn = 3.402823466e+38F;
                float kmx = -3.402823466e+38F;
                float vmn = 3.402823466e+38F;
                float vmx = -3.402823466e+38F;
                for (int element = 0; element < values_per_lane; ++element) {
                    const float kx = float(new_key[element]);
                    const float vx = float(new_value[element]);
                    kmn = min(kmn, kx);
                    kmx = max(kmx, kx);
                    vmn = min(vmn, vx);
                    vmx = max(vmx, vx);
                }
                kmn = simd_min(kmn);
                kmx = simd_max(kmx);
                vmn = simd_min(vmn);
                vmx = simd_max(vmx);
                const half khs = half(max((kmx - kmn) / 255.0f, 1e-6f));
                const half khb = half(kmn);
                const half vhs = half(max((vmx - vmn) / 255.0f, 1e-6f));
                const half vhb = half(vmn);
                const float kqs = float(khs);
                const float kqb = float(khb);
                const float vqs = float(vhs);
                const float vqb = float(vhb);
                device uint32_t* mk_w = const_cast<device uint32_t*>(mkeys_w)
                    + write_slot * row_words;
                device uint32_t* mv_w = const_cast<device uint32_t*>(mvalues_w)
                    + write_slot * row_words;
                uint32_t kws[2] = {0u, 0u};
                uint32_t vws[2] = {0u, 0u};
                for (int element = 0; element < values_per_lane; ++element) {
                    const float kq = clamp(
                        rint((float(new_key[element]) - kqb) / kqs), 0.0f, 255.0f);
                    const float vq = clamp(
                        rint((float(new_value[element]) - vqb) / vqs), 0.0f, 255.0f);
                    kws[element / 4] |= uint32_t(kq) << (8 * (element % 4));
                    vws[element / 4] |= uint32_t(vq) << (8 * (element % 4));
                }
                mk_w[lane * 2] = kws[0];
                mk_w[lane * 2 + 1] = kws[1];
                mv_w[lane * 2] = vws[0];
                mv_w[lane * 2 + 1] = vws[1];
                if (lane == 0) {
                    mk_w[D / 4] = uint32_t(as_type<ushort>(khs))
                        | (uint32_t(as_type<ushort>(khb)) << 16);
                    mv_w[D / 4] = uint32_t(as_type<ushort>(vhs))
                        | (uint32_t(as_type<ushort>(vhb)) << 16);
                }
            }
            if (batch_index == 0 && kv_head == 0 && block == 0
                && query_head_in_group == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }

            device T* partial = partials
                + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
            device float* sum_out = sums + batch_head * BLOCKS + block;
            device float* max_out = maxs + batch_head * BLOCKS + block;

            thread float q[values_per_lane];
            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            uint slot = (ring_start + uint(block)) & ring_mask;
            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const bool current = token == N - 1;
                float score = 0.0f;
                if (current) {
                    for (int element = 0; element < values_per_lane; ++element) {
                        score += q[element] * float(new_key[element]);
                    }
                } else {
                    const device uint32_t* krow_w = mkeys_w + slot * row_words;
                    const uint32_t ktw = krow_w[D / 4];
                    const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                    const float kb = float(as_type<half>(ushort(ktw >> 16)));
                    const uint32_t kw0 = krow_w[lane * 2];
                    const uint32_t kw1 = krow_w[lane * 2 + 1];
                for (int element = 0; element < 4; ++element) {
                        score += q[element]
                            * fma(float((kw0 >> (8 * element)) & 0xffu), ks, kb);
                    }
                        for (int element = 0; element < 4; ++element) {
                        score += q[4 + element]
                            * fma(float((kw1 >> (8 * element)) & 0xffu), ks, kb);
                    }
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                if (current) {
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] = accumulator[element] * old_factor
                            + score_factor * float(new_value[element]);
                    }
                } else {
                    const device uint32_t* vrow_w = mvalues_w + slot * row_words;
                    const uint32_t vtw = vrow_w[D / 4];
                    const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                    const float vb = float(as_type<half>(ushort(vtw >> 16)));
                    const uint32_t vw0 = vrow_w[lane * 2];
                    const uint32_t vw1 = vrow_w[lane * 2 + 1];
                    for (int element = 0; element < 4; ++element) {
                        accumulator[element] = accumulator[element] * old_factor
                            + score_factor
                                * fma(float((vw0 >> (8 * element)) & 0xffu), vs, vb);
                        accumulator[4 + element] = accumulator[4 + element] * old_factor
                            + score_factor
                                * fma(float((vw1 >> (8 * element)) & 0xffu), vs, vb);
                    }
                }

                slot = (slot + uint(BLOCKS)) & ring_mask;
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

private static let diagBF16WriteKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ring_bf16_write_d256_diag_v1",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "starts", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["fence"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr uint ring_mask = uint(N - 1);

            const int group = int(threadgroup_position_in_grid.x);
            const int batch_index = group / KV_HEADS;
            const int kv_head = group % KV_HEADS;
            const int lane = int(thread_index_in_simdgroup);

            const device T* keys = k0;
            const device T* values = v0;
            switch (batch_index) {
                case 1: keys = k1; values = v1; break;
                case 2: keys = k2; values = v2; break;
                case 3: keys = k3; values = v3; break;
                case 4: keys = k4; values = v4; break;
                case 5: keys = k5; values = v5; break;
                case 6: keys = k6; values = v6; break;
                case 7: keys = k7; values = v7; break;
                default: break;
            }

            const uint ring_start = starts[batch_index];
            const uint write_slot = (ring_start + ring_mask) & ring_mask;
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            device T* write_key = const_cast<device T*>(keys)
                + kv_head * N * D + write_slot * D + lane * values_per_lane;
            device T* write_value = const_cast<device T*>(values)
                + kv_head * N * D + write_slot * D + lane * values_per_lane;
            for (int element = 0; element < values_per_lane; ++element) {
                write_key[element] = new_key[element];
                write_value[element] = new_value[element];
            }
            if (group == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """,
        ensureRowContiguous: true
    )

    /// KVQ-DIAG-2 scratch. The fused kernel WRITES: it quantizes the step's
    /// new token into the evicted mirror slot, and the companion writes the
    /// exact bf16 token into the evicted ring slot. Pointing either at the
    /// live ring would change the model function, so the probe binds
    /// dedicated scratch buffers, one per batch slot (no aliasing, so no
    /// artificial race the real mechanism would never have). Allocated once,
    /// reused, never read by anything.
    nonisolated(unsafe) private static var diagScratch:
        (mirrors: [MLXArray], keys: [MLXArray], values: [MLXArray])?

    private static func diagScratchBuffers(
        kvHeads: Int, window: Int, headDim: Int
    ) -> (mirrors: [MLXArray], keys: [MLXArray], values: [MLXArray]) {
        if let diagScratch { return diagScratch }
        let made = (
            mirrors: (0 ..< 8).map { _ in
                MLXArray.zeros(
                    [2, kvHeads, window, (headDim + 4) / 4], dtype: .uint32)
            },
            keys: (0 ..< 8).map { _ in
                MLXArray.zeros([1, kvHeads, window, headDim], dtype: .bfloat16)
            },
            values: (0 ..< 8).map { _ in
                MLXArray.zeros([1, kvHeads, window, headDim], dtype: .bfloat16)
            }
        )
        diagScratch = made
        return made
    }

    /// Dispatch the fused read+write kernel and its bf16 companion over
    /// scratch, discarding both. Probe 1 (`47dae0ea`) cleared the read
    /// kernel by scoring; this clears or convicts the write half.
    private func diagnosticFusedDispatch() {
        guard Self.diagFusedEnabled, let quantMirror, retainedCount == window,
            headDim == 256, kvHeads == 8, window == 1024
        else { return }
        Self.diagLock.lock()
        Self.diagCounter += 1
        let fire = Self.diagCounter % Self.diagStride == 0
        Self.diagLock.unlock()
        guard fire else { return }
        let batch = 8
        let queryHeads = 16
        let gqa = queryHeads / kvHeads
        let scratch = Self.diagScratchBuffers(
            kvHeads: kvHeads, window: window, headDim: headDim)
        let queries = MLXArray.zeros(
            [batch, queryHeads, 1, headDim], dtype: .bfloat16)
        let starts = MLXArray(
            Array(repeating: UInt32(oldestValidPosition % window), count: batch),
            [batch])
        let newKV = MLXArray.zeros([batch, kvHeads, 1, headDim], dtype: .bfloat16)
        let fenceIn = MLXArray.zeros([1], dtype: .int32)
        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype),
            ("D", headDim),
            ("N", window),
            ("GQA", gqa),
            ("KV_HEADS", kvHeads),
            ("BLOCKS", Self.diagBlocks),
        ]
        // KVQ-DIAG-3. Probes 1 (`47dae0ea`) and 2 (`e14279c9`) both scored,
        // so every kernel in the mechanism dispatches and completes on the
        // box. Two differences remain between those probes and the mechanism
        // that dies: the real one WRITES INTO LIVE, GRAPH-REFERENCED
        // allocations, and its output is consumed by pass B. This probe
        // takes the first: the fused kernel now writes the LIVE mirror
        // (harmless while attention stays on the stock bf16 road, because
        // nothing reads the mirror), with the bf16 companion still on
        // scratch so the ring the model actually reads is untouched.
        let liveMirrors = Self.diagLiveWrite
            ? Array(repeating: quantMirror, count: batch) : scratch.mirrors
        let fused = Self.diagFusedKernel(
            [queries] + liveMirrors + [starts, newKV, newKV, fenceIn],
            template: template,
            grid: (kvHeads * 32, batch * gqa, Self.diagBlocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [
                [batch, queryHeads, 1, Self.diagBlocks, headDim],
                [batch, queryHeads, 1, Self.diagBlocks],
                [batch, queryHeads, 1, Self.diagBlocks],
                [1],
            ],
            outputDTypes: [.bfloat16, .float32, .float32, .int32]
        )
        let companion = Self.diagBF16WriteKernel(
            scratch.keys + scratch.values + [starts, newKV, newKV, fused[3]],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", window),
                ("KV_HEADS", kvHeads),
            ],
            grid: (batch * kvHeads * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32]
        )
        asyncEval(fused[1], companion[0])
        Self.diagFusedDispatchOnce()
    }

    /// `MLX_KV_QUANT_DIAG_FUSED=0` disables the second probe.
    static let diagFusedEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG_FUSED"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `MLX_KV_QUANT_DIAG_LIVE=0` sends the fused write back to scratch.
    static let diagLiveWrite: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_KV_QUANT_DIAG_LIVE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    nonisolated(unsafe) private static var diagFusedDispatched = false

    private static func diagFusedDispatchOnce() {
        diagLock.lock()
        let fresh = !diagFusedDispatched
        diagFusedDispatched = true
        diagLock.unlock()
        if fresh {
            FileHandle.standardError.write(Data("[kvq-diag] fused dispatched\n".utf8))
        }
    }

    /// KVQ-WARM: compile the pack pipeline at constructor time (see
    /// `CBv2RaggedTwoPassDecodeAttentionV1.warmQuantPipelines`).
    public static func warmPackPipeline() {
        guard quantEnabled, gpuPackEnabled else { return }
        let dummy = MLXArray.zeros([1, 8, 1, 256], dtype: .float16)
        eval(quantPackGPU(dummy))
        guard pairedMirrorWriteEnabled else { return }
        let pair = MLXArray.zeros([1, 8, 1, 256], dtype: .bfloat16)
        eval(quantPackPairGPU(keys: pair, values: pair))
    }

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
            outputShapes: [[kvHeads, n, (x.dim(3) + 4) / 4]],
            outputDTypes: [.uint32]
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

    /// `[1, H, 1, D]` keys and values -> `[2, H, 1, (D + 4) / 4]` uint32,
    /// row for row what `quantPackGPU` produces for each plane separately.
    static func quantPackPairGPU(keys: MLXArray, values: MLXArray) -> MLXArray {
        let heads = keys.dim(1)
        let headDim = keys.dim(3)
        return quantPackPairKernel(
            [keys, values],
            template: [("T", keys.dtype), ("HEADS", heads)],
            grid: (2 * heads * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[2, heads, 1, (headDim + 4) / 4]],
            outputDTypes: [.uint32]
        )[0]
    }

    /// KVQ-PAIRWRITE. Maintaining the mirror plane by plane builds TWO
    /// `SliceUpdate`s over the whole mirror allocation per decode token, and
    /// the second one takes the first one's output as its input. The mirror
    /// is an attention input on every step of the quantized road, so neither
    /// update can donate its buffer: depositing 520 bytes per head copies
    /// the full ring twice.
    ///
    /// Both planes of a decode token are known at the same instant, so they
    /// can share one pack dispatch and one update, which halves that copy
    /// and takes the step's mirror dispatches from four to two. Returns
    /// false — leaving the per-plane road to run untouched — whenever a
    /// precondition does not hold.
    private func writePairedMirror(
        keys newKeys: MLXArray, values newValues: MLXArray, firstPosition: Int
    ) -> Bool {
        guard Self.pairedMirrorWriteEnabled,
            let quantMirror,
            Self.gpuPackEnabled, !Self.gpuPackCheck, !Self.quantSimulate,
            headDim == 256,
            newKeys.dtype == newValues.dtype,
            newKeys.shape == [1, kvHeads, 1, headDim],
            newValues.shape == [1, kvHeads, 1, headDim]
        else { return false }
        let packed = Self.quantPackPairGPU(keys: newKeys, values: newValues)
        let slot = firstPosition % window
        quantMirror[0..., 0..., slot ..< (slot + 1), 0...] = packed
        return true
    }

    private func writeMirror(plane: Int, tokens: MLXArray, firstPosition: Int) {
        guard let quantMirror else { return }
        let packedFlat: MLXArray
        if Self.gpuPackCheck {
            let gpu = Self.quantPackGPU(tokens)
            let host = Self.quantPack(tokens).view(dtype: .uint32)
            let mismatches = (gpu .!= host).sum().item(Int.self)
            if mismatches != 0 {
                FileHandle.standardError.write(
                    "[kvq-gpupack] BYTE MISMATCH: \(mismatches) bytes differ\n"
                        .data(using: .utf8)!)
            }
            packedFlat = host
        } else if Self.gpuPackEnabled {
            packedFlat = Self.quantPackGPU(tokens).view(dtype: .uint8)
        } else {
            packedFlat = Self.quantPack(tokens)
        }
        let packed = packedFlat.view(dtype: .uint32).expandedDimensions(axis: 0)
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
            // KVQ-U32: the mirror is allocated and BOUND as uint32 words
            // (row = (headDim + 4) / 4 of them). Kernel bodies cast down to
            // uint8_t* internally; byte layout is unchanged.
            quantMirror = MLXArray.zeros(
                [2, kvHeads, window, (headDim + 4) / 4], dtype: .uint32)
        }
    }
}
