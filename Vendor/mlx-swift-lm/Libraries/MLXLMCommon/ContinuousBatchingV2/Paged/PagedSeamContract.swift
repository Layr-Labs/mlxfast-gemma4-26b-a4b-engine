// PagedSeamContract.swift
// Declarations and admission rules for the paged KV seam. Behaviour lives
// in PagedKVPool / PagedSequenceKV / PagedLayerCache.

import Foundation
import MLX

// MARK: - Speculative span

/// Max extra positions one speculative round may write past the frontier.
/// Ring sizing and `supportsSpeculativeWrites` must agree on this literal.
public enum CBv2PagedSpeculation {

    /// Frozen literal. Consumers read `maxSpeculativeSpan`.
    static let declaredSpan = 8

    /// Worst-case positions written beyond the confirmed frontier in one
    /// round. Validated on first read so a larger MTP bound cannot silently
    /// undersize the ring (`precondition` stays live except `-Ounchecked`).
    public static let maxSpeculativeSpan: Int = {
        assertSpanCoversMTPBound()
        return declaredSpan
    }()

    /// `true` while the reserved span still covers a whole MTP round:
    /// `maxDraftTokens` drafted columns plus one target column.
    ///
    /// Reads `declaredSpan`, not `maxSpeculativeSpan`, so the validation can
    /// call it without recursing through the property it validates.
    static var spanCoversMTPBound: Bool {
        declaredSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1
    }

    /// Traps if the MTP draft bound has outgrown the reserved span.
    /// `maxDraftTokens` drafted columns plus one target column.
    ///
    /// A `precondition`, not an `assert`: an under-sized ring is silent KV
    /// corruption, and the only way to reach this condition is a source
    /// change to one of two constants, which CI catches first.
    static func assertSpanCoversMTPBound() {
        precondition(spanCoversMTPBound, spanDriftMessage)
    }

    /// The diagnostic, shared by the runtime check and its test copy so the
    /// two cannot describe the same drift differently.
    static var spanDriftMessage: String {
        """
        CBv2PagedSpeculation.maxSpeculativeSpan (\(declaredSpan)) no longer \
        covers CBv2MTPConfig.testedMaxDraftTokens + 1 \
        (\(CBv2MTPConfig.testedMaxDraftTokens + 1)). Raise the span and re-check \
        PagedKVPool.ringPageCount before raising the MTP draft bound.
        """
    }
}

// MARK: - Windowed ring geometry

/// Windowed ring formula. Must match `PagedKVPool.ringPageCount`.
///
/// Gather-before-write plus the gather fence keep a 65-page / 1024-window
/// ring valid. Changing either without resizing the ring is a revert.
public enum CBv2PagedRingGeometry {

    /// Widest gather a windowed row can be asked for: `window`, not
    /// `window - 1 + maxPrefillChunk`.
    public static func attendableTokens(window: Int) -> Int {
        PagedSequenceKV.maxWindowExposure(window: window)
    }

    /// Tokens the ring MUST cover, as the larger of two INDEPENDENT bounds.
    /// `PagedKVPool.checkedRingPageCount` refuses to build a pool that does
    /// not reach either, and checks them separately.
    ///
    ///   CACHE  `attendableTokens + maxSpeculativeSpan` — the widest gather
    ///          plus one speculative round, because writing position `p`
    ///          destroys whatever held `p - ringTokens`.
    ///   ROW    `maxPrefillChunk` — one `PagedSequenceKV.write` scatters a
    ///          whole chunk in ONE dispatch, and a chunk longer than the ring
    ///          puts two of its own tokens in one physical slot with no
    ///          ordering between them.
    ///
    /// The ROW bound used to be implied by the chunk term the CACHE bound
    /// carried. It is not implied any more. Which one binds is pure geometry:
    /// window 1,024 / chunk 512 is cache-bound (1,032 vs 512), while window
    /// 128 / chunk 2,048 is row-bound (2,048 vs 136) and a ring derived from
    /// the window alone would give it 9 pages for a 128-page write.
    public static func requiredTokens(window: Int, maxPrefillChunk: Int) -> Int {
        max(
            attendableTokens(window: window) + CBv2PagedSpeculation.maxSpeculativeSpan,
            maxPrefillChunk)
    }

    /// Ring length in pages:
    ///
    ///     ceil(max(window + maxSpeculativeSpan, maxPrefillChunk) / pageSize)
    ///
    /// gemma-4 (window 1,024, chunk 512, pageSize 16, span 8):
    /// `ceil(max(1032, 512) / 16) == 65` pages == 1,040 tokens.
    public static func ringPageCount(window: Int, pageSize: Int, maxPrefillChunk: Int) -> Int {
        let required = requiredTokens(window: window, maxPrefillChunk: maxPrefillChunk)
        return (required + pageSize - 1) / pageSize
    }
}

// MARK: - Rectangular MTP verification

/// Opt-in marker for a layer cache that can have its attention serialised
/// per query during MTP rectangular verification.
///
/// **Why this protocol exists.** `EngineLoopV2+MTPTargetVerification` used to
/// reach the `mtpSerializesRectangularAttention` flag through
/// `as? CBv2LayerCache` behind a `preconditionFailure`. `CBv2LayerCache` is
/// `final` and `PagedLayerCache` is a *sibling* conformer of
/// `CBv2AttendingLayerCache`, not a subclass, so that cast could never
/// succeed for a paged bank — and `preconditionFailure` is a `fatalError`:
/// daemon death, every co-resident model's in-flight requests lost, no
/// telemetry. It was unreachable only because gemma-4's windowed paged rows
/// failed the storage-eligibility gate first and no round was ever built,
/// which made removing that gate a latent process abort.
///
/// Both halves have since landed: the verifier `compactMap`s this marker and
/// degrades to the serial oracle when any cache in the bank does not conform,
/// and `PagedLayerCache` conforms.
///
/// **Contract.** Setting the flag to `true` obliges the cache to attend one
/// query position at a time for the duration of the round, so that each
/// column is bit-identical to that column run as a standalone `L == 1` decode.
/// Rectangular verification does NOT require batched multi-query attention:
/// the contiguous implementation already serialises attention and batches only
/// the weight-bound model body across the `1 + k` columns. A paged conformer is
/// therefore a column loop over the existing decode dispatch, not a new kernel.
///
/// Callers MUST degrade to serial verification for a cache that does not
/// conform, and MUST NOT trap.
protocol CBv2MTPRectangularSerializing: AnyObject {
    /// While `true`, attention is computed one query position at a time.
    /// Set for the duration of a rectangular verification round and cleared
    /// in a `defer`.
    var mtpSerializesRectangularAttention: Bool { get set }
}

/// The contiguous cache already owns the stored flag (`LayerCacheV2.swift`),
/// so conformance is declaration-only.
extension CBv2LayerCache: CBv2MTPRectangularSerializing {}

// MARK: - Row-side speculative transaction

/// A sequence row that can stage and roll back speculative writes.
///
/// The contiguous windowed ring already implements this behaviour behind
/// `CBv2SequenceKV`'s `beginSpeculativeWrite()` / `commitSpeculativeWrite()`
/// (which are protocol requirements with default no-ops, so a paged row
/// silently inherits the no-ops today). This protocol adds only the part that
/// does not exist anywhere yet: a way to ASK a row how much speculative room
/// it has, rather than assuming.
///
/// **Paged rows need no data staging.** A windowed paged ring aliases at
/// `ringPages * pageSize`, not at `window`, so a rolled-back speculative write
/// destroys only entries already outside every attendable range — provided the
/// ring reserves `CBv2PagedSpeculation.maxSpeculativeSpan`. The transaction is
/// therefore pure bookkeeping: a speculative base, a tightened rollback
/// precondition, deferred page frees, and restoration of the retained-count
/// input.
protocol CBv2PagedSpeculativeRow: AnyObject {
    /// Positions this row can write past its confirmed frontier without
    /// destroying an entry that is still attendable.
    ///
    /// Windowed rows: `ringPages * pageSize - window`. Full rows: unbounded in
    /// practice, reported as `Int.max`. A row is speculation-eligible when this
    /// is `>= CBv2PagedSpeculation.maxSpeculativeSpan`.
    var speculativeHeadroom: Int { get }
}

// MARK: - Windowed prefix adoption (WS-4.1)

/// Why a donated sliding window was refused at a matched boundary. Every case
/// is a normal, expected outcome whose handling is "fall back to replay" —
/// never a trap, never a partial install.
public enum CBv2PagedWindowRestoreRefusal: Error, Equatable, CustomStringConvertible {

    /// The snapshot ends at a different absolute position than the boundary
    /// being adopted. This is the common case and the reason this type
    /// exists: `PrefixCacheV2` can return a boundary far behind the donation
    /// endpoint the window was taken at.
    case boundaryMismatch(snapshotEnd: Int, requested: Int)

    /// Right boundary, wrong extent. A window shorter than
    /// `min(matchedBoundary, window)` is missing its oldest entries, and
    /// those are invisible to attention rather than recoverable by a short
    /// replay; a longer one is not a window at all.
    case inexactWindow(tokens: Int, required: Int)

    /// A full-attention row has no window to restore. Full layers come back
    /// through `PrefixCacheV2`'s per-layer snapshots, not through this seam.
    case notWindowed(requested: Int)

    public var description: String {
        switch self {
        case .boundaryMismatch(let snapshotEnd, let requested):
            return
                "windowed prefix refused: snapshot ends at absolute \(snapshotEnd) but the "
                + "adoption boundary is \(requested); installing it would place the donor's "
                + "keys at the wrong absolute positions"
        case .inexactWindow(let tokens, let required):
            return
                "windowed prefix refused: snapshot carries \(tokens) positions, the boundary "
                + "needs exactly \(required); a partial window is not an exact restore"
        case .notWindowed(let requested):
            return "windowed prefix refused: row at boundary \(requested) has no sliding window"
        }
    }
}

/// A donated sliding window, together with the ONE absolute boundary at which
/// it may be installed.
///
/// **Why this is a type and not three loose arguments** (PR#86 review, `:173`).
/// `PrefixCacheV2` indexes EVERY whole-block boundary of a donation — "Every
/// whole-block boundary of this entry is indexed, so shorter prefixes of a
/// long donation still hit" (`PrefixCacheV2.swift:124-126`) — and its lookup
/// scans longest-to-shortest and returns `matched = k * blockSize` for any `k`
/// up to the entry's block count (`PrefixCacheV2.swift:216-231`). A finished
/// paged row, by contrast, retains only the last `retainedCount` positions
/// ending at its own `absoluteOffset`. So a donation that ended near token
/// 4,096 is ALSO indexed at 1,024, and the trailing window it carries cannot
/// serve that hit.
///
/// Under the previous frozen signature — `restoreWindow(keys:values:base:)`,
/// with `base` an argument taken on trust and no boundary to check it against
/// — an adopter had exactly two ways to proceed and both were wrong:
///
///  * install the payload anyway, writing the donor's positions
///    `[3072, 4096)` into the adopter's `[0, 1024)`. `PagedSequenceKV` stores
///    by absolute position (`gatherRange` maps `p` to ring slot
///    `(p / pageSize) % ringPages`), so nothing downstream can notice: silent
///    wrong answers, no trap, no telemetry; or
///  * replay, which contradicts the same seam entry's "replay bound 25,600
///    tokens -> 0" claim.
///
/// The resolution is the one WS-4.2 reached independently on the provider side
/// (`SSDWindowSidecar.swift`, provider PR#588): the persisted form is
/// PER-BLOCK and content-addressed off the same chain hash as the
/// full-attention block, the window at boundary `M` is assembled from the
/// `W / blockSize` sidecars tiling `[M - W, M)`, and a boundary whose tiling
/// is incomplete is REFUSED rather than partially restored — "a PARTIAL
/// window restore is NOT exact (the missing oldest entries are invisible to
/// attention and cannot be recovered by a short replay ...), so a boundary is
/// adoptable only when EVERY tiling block is present". That rule is
/// `SSDWindowSidecarGeometry.coveredBlocks`, enforced in
/// `SSDWindowSidecar.rebuildWindow`, with the authenticated `windowBase`
/// anti-splice check in `SSDWindowSidecar.isBound`. Cited BY SYMBOL: these
/// are line anchors into another repository, and they have rotted once.
///
/// This type is the engine-side statement of the same rule, and it makes the
/// wrong-absolute-position outcome unrepresentable rather than merely
/// discouraged:
///
///  * `tokens` is read off `keys`, never supplied, so a snapshot cannot claim
///    an extent it does not carry;
///  * `endBoundary` is derived from `base + tokens`, so the position a payload
///    belongs at is a property OF the payload, not of the call;
///  * every install goes through `requireAdmissible(at:window:)`, which
///    refuses any boundary but that one.
public struct CBv2PagedWindowSnapshot {

    /// `[1, kvHeads, tokens, headDim]`, oldest position first.
    public let keys: MLXArray
    /// `[1, kvHeads, tokens, headDim]`, oldest position first.
    public let values: MLXArray
    /// Absolute position of the FIRST retained token.
    public let base: Int
    /// Retained positions. Read off `keys.dim(2)`; never a caller's claim.
    public let tokens: Int

    /// Absolute position one past the last retained token — the only boundary
    /// this payload may be installed at.
    public var endBoundary: Int { base + tokens }

    /// `nil` for anything that is not a well-formed window.
    ///
    /// Deliberately failable rather than trapping: a mis-shaped or corrupt
    /// donation must degrade to replay, which is always safe, and a cache
    /// read is not a place to abort a multi-tenant daemon.
    ///
    /// The argument list is the `(keys:values:base:)` tuple the provider's
    /// donor seam produces — `CBv2WindowedSequenceKV.windowSnapshot()` in
    /// `SSDWindowSidecar.swift`, and `SSDWindowSidecar.Window` — so bridging
    /// a donor snapshot is
    /// `CBv2PagedWindowSnapshot(keys: w.keys, values: w.values, base: w.base)`.
    ///
    /// The tuple SHAPE is shared across the two repositories; the protocol
    /// that used to carry it is not. The provider had an
    /// `SSDWindowSnapshotting` protocol with one conformer and a runtime
    /// `as?` probe, written in anticipation of a `PagedSequenceKV`
    /// conformance that WS-4.1 never landed; it has been deleted and the
    /// probe is a concrete cast on the CONTIGUOUS row. Nothing on the paged
    /// side conforms to anything here — a donated window reaches this type
    /// as three loose arrays through `makeSequenceState(adopting:)`.
    public init?(keys: MLXArray, values: MLXArray, base: Int) {
        guard base >= 0,
            keys.ndim == 4, values.ndim == 4,
            keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(1) == values.dim(1),
            keys.dim(2) == values.dim(2),
            keys.dim(3) == values.dim(3),
            keys.dim(2) > 0
        else { return nil }
        self.keys = keys
        self.values = values
        self.base = base
        self.tokens = keys.dim(2)
    }

    /// Throws unless this payload is an EXACT window for `matchedBoundary` on
    /// a row whose sliding window is `window` (`nil` for full attention).
    ///
    /// Admissible means both of:
    ///
    ///  * `endBoundary == matchedBoundary` — the window was taken at exactly
    ///    the boundary being adopted;
    ///  * `tokens == min(matchedBoundary, window)` — it is the whole window,
    ///    or, for a boundary shorter than one window, the whole history. The
    ///    two together pin `base == matchedBoundary - min(matchedBoundary,
    ///    window)`, so an admissible base is page-aligned whenever
    ///    `matchedBoundary` is, which WS-0.6 invariant 1 guarantees for a
    ///    matched block boundary (checked in `PagedKVPool.init`).
    ///
    /// ## WHAT ACTUALLY ADOPTS A WINDOW
    ///
    /// This contract once froze three ROW signatures for adoption —
    /// `windowSnapshot()`, `restoreWindow(_:at:)` and
    /// `installShared(_:upTo:)`. None of them was ever implemented, and
    /// `PagedSequenceKV` has no member of any of those names. The route that
    /// shipped is on the BACKEND, not the row:
    ///
    ///   * `PagedKVBackend.makeFrozenFullState` builds the snapshot inline
    ///     from the donated `(keys, values, offset)` triple, and is where
    ///     `requireAdmissible` is called — refusals become
    ///     `CBv2KVError.backendIneligible` during the validation phase, so
    ///     the install phase below it cannot throw and leave half-built
    ///     state.
    ///   * `PagedKVBackend.installWindow` places it: `fastForward(to:base)`
    ///     followed by chunked `row.write`.
    ///
    /// **Adoption is a BYTE WRITE, not a pointer swap.** The frozen entry
    /// promised page adoption under refcount, and that is not what runs:
    /// `installWindow` copies the payload into the row's own freshly
    /// allocated pages, `maxPrefillChunk` tokens at a time, because
    /// `PagedSequenceKV.write` refuses a windowed run longer than a chunk.
    /// No page is shared and no refcount is adopted — which is why
    /// `PagedKVGroup` has no retain operation and every page refcount is 0
    /// or 1.
    ///
    /// `installWindow` is `private` AND re-asserts this check on the
    /// boundary it was handed, so the two callers a future edit might add —
    /// one inside `PagedKVBackend`, one that makes it non-private — both hit
    /// the invariant instead of silently writing keys at wrong absolute
    /// positions.
    public func requireAdmissible(at matchedBoundary: Int, window: Int?) throws {
        guard let window, window > 0 else {
            throw CBv2PagedWindowRestoreRefusal.notWindowed(requested: matchedBoundary)
        }
        guard endBoundary == matchedBoundary else {
            throw CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: endBoundary, requested: matchedBoundary)
        }
        let required = min(matchedBoundary, window)
        guard tokens == required else {
            throw CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: tokens, required: required)
        }
    }
}
