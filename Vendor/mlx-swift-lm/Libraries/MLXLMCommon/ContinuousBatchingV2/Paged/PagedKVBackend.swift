// PagedKVBackend.swift
//
// `CBv2KVBackend` factory over a `PagedKVPool` (WS-C). Swappable with the
// WS-A contiguous backend behind the same contract: the scheduler and
// models never see the difference.
//
// Eligibility is validated at construction (engine build time), per the
// contract: unsupported head dims, shapes over the paged kernel's
// threadgroup-memory budget (`PagedAttentionKernel.ineligibilityReason` —
// dispatching one is an uncatchable Metal fatal; the kernel's head split
// keeps every supported head dim within budget, incl. Gemma-4 global
// layers at headDim 512 / GQA 8), quant schemes, or malformed KV-sharing
// throw `CBv2KVError.backendIneligible` before any request is admitted.
// Attention sinks ARE supported (they are a kernel parameter here).
//
// Admission model: the worst-case page count for a request's `maxLength`
// is reserved UP FRONT — at admission via `reserve(layerKinds:maxLength:)`,
// or (when no admission-time reservation was taken) lazily by
// `makeSequenceState`, which reconciles against any prior reservation so the
// pages are charged exactly once. `reserve` throws `capacityExhausted` when
// the pool cannot honor the demand, so an admission controller can reject or
// queue a request that would otherwise fail at materialization (Codex P2:
// several same-step admissions must not over-commit the pool). Physical
// pages materialize lazily as tokens are written (`bytesInUse` stays
// truthful); `CBv2SequenceKV.update` therefore never fails mid-decode. See
// PagedKVPool.swift for the rationale.
//
// Prefix adoption (WS-4.1): paged serves `.direct`, `.tailReplay` and BOTH
// forms of `.frozenFullReplay` — the dual-cursor replay, and the zero-replay
// restore that arrives once a window payload exists. See
// `makeSequenceState(adopting:plan:layerKinds:maxLength:)` for the three
// shapes. Two properties of that path are load-bearing and easy to lose:
//
//   * a refusal is a THROW, never a trap. The engine's only recovery is a
//     cold prefill, and it can only take it if the adoption reports failure
//     instead of aborting the daemon;
//   * validation of EVERY layer completes before the first page is reserved.
//     A frozen-full hit whose sliding side cannot be completed must leave the
//     pool exactly as it found it, because a half-restored hybrid does not
//     fail loudly — the sliding rows simply attend fewer keys than they
//     should, and nothing downstream can see it.

import Foundation
import MLX

public final class PagedKVBackend: CBv2KVBackend {
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .pagedFP16 }
    public let pool: PagedKVPool
    /// The model's per-layer structure this backend was built for.
    public let layerKinds: [CBv2LayerKind]
    /// WHEN this backend's slabs become MLX-resident. See
    /// PagedKVSlabCommitment.swift — the default defers the commitment past
    /// engine construction so an idle pool does not pre-empt a co-resident
    /// model's post-load headroom measurement (D1).
    public let slabCommitment: PagedKVSlabCommitment
    /// True once the slabs have actually been evaluated. Flipped exactly
    /// once, on the engine loop thread, by `commitSlabs()`.
    private(set) var slabsAreWired = false
    /// The only writer of `slabsAreWired`; `private(set)` keeps the flag out
    /// of reach of everything except `commitSlabs()`, which lives in the
    /// other file.
    func markSlabsWired() { slabsAreWired = true }

    public init(
        layerKinds: [CBv2LayerKind],
        config: PagedKVPoolConfig,
        slabCommitment: PagedKVSlabCommitment = .atFirstAdmission
    ) throws {
        for (index, kind) in layerKinds.enumerated() {
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count,
                    layerKinds[source].sharesKVWithLayer == nil
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) shares KV with invalid layer \(source)")
                }
                let src = layerKinds[source]
                guard src.kvHeads == kind.kvHeads, src.headDim == kind.headDim,
                    src.attention == kind.attention
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) KV-shares with structurally different layer "
                            + "\(source)")
                }
            }
            guard kind.kvHeads > 0, kind.queryHeads % kind.kvHeads == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): queryHeads \(kind.queryHeads) not a multiple "
                        + "of kvHeads \(kind.kvHeads)")
            }
            // Kernel-level static eligibility (head dim support + the part
            // kernel's threadgroup-memory budget). Checked for EVERY layer
            // that will dispatch paged attention — including KV-shared
            // layers, which borrow storage but launch with their own GQA.
            // One over-budget layer makes the whole model ineligible.
            if let reason = PagedAttentionKernel.ineligibilityReason(
                headDim: kind.headDim, gqa: kind.queryHeads / kind.kvHeads)
            {
                throw CBv2KVError.backendIneligible(reason: "layer \(index): \(reason)")
            }
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid sliding window \(window)")
            }
        }
        self.layerKinds = layerKinds
        self.slabCommitment = slabCommitment
        self.pool = try PagedKVPool(layerKinds: layerKinds, config: config)
        if slabCommitment == .atConstruction {
            // An eager commit that cannot fit fails the BUILD (throwing
            // `capacityExhausted`), which is the honest posture for the
            // profiler/single-slot deployments that opt into it.
            try commitSlabs()
        }
    }

    // MARK: - Admission-time reservation (Codex P2)

    /// Reserve the worst-case page demand for a request of `maxLength` tokens
    /// BEFORE it is admitted, so several same-step admissions cannot
    /// over-commit the pool. Charges the pool up front (reflected in
    /// `bytesReserved`) and throws `capacityExhausted` when it cannot fit —
    /// the admission controller then rejects or queues the request instead of
    /// accepting one that would only fail at `makeSequenceState`.
    ///
    /// A subsequent `makeSequenceState` for the SAME request must be told the
    /// pages are already held (`reserved: true`) so it does not double-charge;
    /// `release`/`makeSequenceState(adopting:)`/finish balance the hold via
    /// the per-row `reservedPages` bookkeeping exactly once. Balance an
    /// admission that never materializes with `unreserve(layerKinds:maxLength:)`.
    public func reserve(layerKinds: [CBv2LayerKind], maxLength: Int) throws {
        precondition(maxLength > 0)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        try pool.reserve(needs)
        // The charge succeeded, so this pool is no longer idle. Wire the
        // slabs NOW — before any row can reach `ensurePage` — so a deferred
        // commitment never turns an accepted admission into an unbacked
        // page. Deferring the ALLOCATION is the D1 fix; deferring the
        // GUARANTEE would be a daemon abort under load.
        do {
            try commitSlabs()
        } catch {
            // A refused commit must leave the pool exactly as it found it:
            // unwind the page charge so the rejected admission leaves no
            // residue and the retry re-charges from a clean ledger.
            pool.unreserve(needs)
            throw error
        }
    }

    /// Release an admission-time `reserve` that never reached
    /// `makeSequenceState` (rejected, superseded, or shut down).
    public func unreserve(layerKinds: [CBv2LayerKind], maxLength: Int) {
        pool.unreserve(pageNeeds(layerKinds: layerKinds, maxLength: maxLength))
    }

    // MARK: - CBv2KVBackend

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try makeSequenceState(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength,
            reserved: false)
    }

    /// `reserved: true` skips the pool reservation because `reserve(...)`
    /// already charged the pages at admission — the per-row `reservedPages`
    /// still governs allocation and `release` still unreserves them, so the
    /// hold is charged and released exactly once.
    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int, reserved: Bool
    ) throws -> [CBv2SequenceKV?] {
        precondition(maxLength >= promptLength && maxLength > 0)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        if !reserved {
            try pool.reserve(needs)
        }
        // Same guarantee as `reserve`: every page this row may touch is
        // backed before the row exists. Idempotent and free after the first
        // admission. On a refused commit, unwind exactly the charge THIS
        // call took — a `reserved: true` caller still owns its own hold and
        // balances it with `unreserve` per the admission contract above.
        do {
            try commitSlabs()
        } catch {
            if !reserved { pool.unreserve(needs) }
            throw error
        }
        var states: [CBv2SequenceKV?] = []
        states.reserveCapacity(layerKinds.count)
        for kind in layerKinds {
            if kind.sharesKVWithLayer != nil {
                states.append(nil)
            } else {
                let reserved = PagedKVPool.pageDemand(
                    kind: kind, maxLength: maxLength, config: pool.config)
                states.append(
                    PagedSequenceKV(
                        pool: pool, kind: kind, maxLength: maxLength, reservedPages: reserved))
            }
        }
        return states
    }

    /// Adopt a donated prefix. Snapshots are written into fresh pages via the
    /// in-place bulk-write kernel.
    ///
    /// Three shapes, one entry point:
    ///
    ///  * `.direct` / `.tailReplay` — owning full rows restored to C, windowed
    ///    rows fast-forwarded to C and left empty, engine replays `[C, M)`.
    ///    Both cursors are C, which is why paged has always served these.
    ///  * `.frozenFullReplay` with R == 0 (`requiresExactWindowRestore`) —
    ///    owning full rows restored to M and EVERY owning windowed row
    ///    restored to M from an admissible `CBv2PagedWindowSnapshot`. Both
    ///    cursors are M and there is nothing to replay, which is the form
    ///    that makes the paged-hybrid dual-cursor problem evaporate rather
    ///    than solving it. (The capability refusal that named that problem
    ///    is gone: `derive` no longer refuses paged hybrids.)
    ///  * `.frozenFullReplay` with R > 0 — owning full rows adopted FROZEN
    ///    through M via `PagedSequenceKV.adoptFrozen`, so their storage is
    ///    exact and immutable while the logical cursor reports C; windowed
    ///    rows fast-forwarded to C and left empty; engine replays `[C, M)`.
    ///    This is the genuine dual cursor, and it is the form the engine
    ///    actually produces today — `PrefixCacheV2` nils every windowed layer,
    ///    so no window payload reaches the restore form yet.
    ///
    /// EVERY refusal is a thrown `backendIneligible`, never a trap and never a
    /// partial install. `EngineLoopV2.applyAdoption` catches it, unreserves the
    /// admission charge and the request cold-prefills — which is the only safe
    /// answer, because a frozen-full hit that restores the full layers but
    /// cannot complete the sliding side leaves windows that are SHORT rather
    /// than absent: attention silently ignores the missing oldest entries and
    /// no later replay can recover them. Validation therefore completes for
    /// every layer BEFORE a single page is reserved.
    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard plan.backend == .pagedFP16 else {
            throw CBv2KVError.backendIneligible(
                reason: "paged adoption received \(plan.backend.rawValue) prefix plan")
        }
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }
        guard plan.matchedBoundary > 0, plan.matchedBoundary <= maxLength,
            plan.replayStart >= 0, plan.replayStart <= plan.matchedBoundary,
            plan.replayTokens == plan.matchedBoundary - plan.replayStart
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix replay plan")
        }
        if plan.strategy == .frozenFullReplay {
            return try makeFrozenFullState(
                adopting: prefix, plan: plan, layerKinds: layerKinds, maxLength: maxLength)
        }
        var matched = 0
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            precondition(
                layerKinds[index].sharesKVWithLayer == nil,
                "prefix donated to a KV-shared layer \(index)")
            precondition(
                matched == 0 || matched == entry.offset,
                "non-uniform prefix offsets (\(entry.offset) vs \(matched))")
            matched = entry.offset
        }
        guard matched == plan.restoredFullTokens, matched == plan.replayStart else {
            throw CBv2KVError.backendIneligible(
                reason: "paged prefix offset does not match ordinary replay plan")
        }
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, state) in states.enumerated() {
            guard let state = state as? PagedSequenceKV else { continue }
            if let snapshot = prefix[index] {
                precondition(
                    state.windowSize == nil,
                    "prefix donated to a windowed layer \(index)")
                var keys = snapshot.keys
                var values = snapshot.values
                if keys.ndim == 4 {
                    keys = keys.squeezed(axis: 0)
                    values = values.squeezed(axis: 0)
                }
                precondition(
                    keys.dim(1) == snapshot.offset,
                    "full-attention prefix snapshot must cover [0, offset)")
                state.write(keys: keys, values: values)
            } else if state.windowSize != nil, matched > 0 {
                state.fastForward(to: matched)
            }
        }
        return states
    }

    // MARK: - Frozen-full hybrid adoption (WS-4.1)

    private func makeFrozenFullState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        let matched = plan.matchedBoundary
        guard plan.restoredFullTokens == matched else {
            throw CBv2KVError.backendIneligible(
                reason: "frozen-full plan restores \(plan.restoredFullTokens) tokens but "
                    + "matched \(matched) — full rows must be exact through M")
        }
        let restoringWindows = plan.requiresExactWindowRestore
        if !restoringWindows {
            // The dual-cursor form. Neither condition is visible to the
            // planner, and both are cheap, so they are checked here.
            guard plan.replayStart > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "frozen replay with C == 0 saves nothing")
            }
            // The SAME bound contiguous uses, from the SAME function — paged
            // briefly kept its own edition (`requiredFrozenReplayTokens`,
            // `cbv2RequiredRecompute` plus one prefill chunk) because a
            // frozen paged row attended its chunk's freshly projected keys.
            // `PagedLayerCache.prefillKVWritingChunk` reads the cached
            // diagonal out of the frozen pages now, so the extra term is gone
            // and two copies of one bound would only drift. Still CHECKED
            // rather than assumed: a plan reaches this function from outside
            // the type. If a frozen paged row is ever made to attend fresh
            // projections again, this bound is short by one chunk and it will
            // accept a replay that leaves the sliding rows silently inexact.
            let required = cbv2RequiredRecompute(
                layerKinds: layerKinds, matched: plan.matchedBoundary)
            guard plan.replayTokens >= required else {
                throw CBv2KVError.backendIneligible(
                    reason: "frozen replay of \(plan.replayTokens) tokens is shorter than the "
                        + "\(required) this layout needs — the sliding rows would come back "
                        + "inexact")
            }
        }

        // --- Validate every layer BEFORE reserving a single page. A frozen
        //     full restore that cannot complete its sliding side must leave
        //     the pool untouched so the cold prefill can have the capacity.
        var fullKV: [Int: (keys: MLXArray, values: MLXArray)] = [:]
        var windows: [Int: CBv2PagedWindowSnapshot] = [:]
        var sawOwningFull = false
        for (index, kind) in layerKinds.enumerated() {
            let entry = prefix[index]
            guard kind.sharesKVWithLayer == nil else {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is KV-shared but received a prefix snapshot")
                }
                continue
            }
            // An owning full layer always needs its snapshot. A windowed one
            // needs a window under the restore form and must NOT have one
            // under the replay form, which the switch below decides.
            let needsEntry: Bool
            if case .slidingWindow = kind.attention {
                needsEntry = restoringWindows
            } else {
                needsEntry = true
            }
            guard let entry else {
                guard !needsEntry else {
                    throw CBv2KVError.backendIneligible(
                        reason: "frozen-full adoption at boundary \(matched): owning layer "
                            + "\(index) has no donated KV")
                }
                continue
            }
            switch kind.attention {
            case .full:
                guard entry.offset == matched else {
                    throw CBv2KVError.backendIneligible(
                        reason: "full prefix offset \(entry.offset) != matched \(matched) at "
                            + "layer \(index)")
                }
                let keys = Self.rowShaped(entry.keys)
                let values = Self.rowShaped(entry.values)
                guard keys.ndim == 3, values.ndim == 3,
                    keys.dim(0) == kind.kvHeads, values.dim(0) == kind.kvHeads,
                    keys.dim(2) == kind.headDim, values.dim(2) == kind.headDim,
                    keys.dim(1) == matched, values.dim(1) == matched
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "full prefix snapshot at layer \(index) does not exactly cover "
                            + "[0, \(matched)) at this layer's geometry")
                }
                fullKV[index] = (keys, values)
                sawOwningFull = true
            case .slidingWindow(let window):
                guard restoringWindows else {
                    // The replay form recomputes windowed layers from C. A
                    // payload here means the donor and the plan disagree
                    // about what is being adopted; installing it would put
                    // the row at M while every other row sits at C.
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is windowed but received a prefix snapshot "
                            + "under a replay plan (windowed layers are recomputed)")
                }
                guard entry.keys.ndim == 4,
                    let snapshot = CBv2PagedWindowSnapshot(
                        keys: entry.keys, values: entry.values,
                        base: entry.offset - entry.keys.dim(2))
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "windowed prefix at layer \(index) is not a well-formed "
                            + "window snapshot")
                }
                // The seam's rule, not a local reimplementation of it: the
                // payload may be installed at exactly one absolute boundary,
                // and a short window is refused rather than partially placed.
                do {
                    try snapshot.requireAdmissible(at: matched, window: window)
                } catch let refusal as CBv2PagedWindowRestoreRefusal {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): \(refusal.description)")
                }
                guard snapshot.keys.dim(1) == kind.kvHeads,
                    snapshot.keys.dim(3) == kind.headDim,
                    snapshot.values.dim(1) == kind.kvHeads,
                    snapshot.values.dim(3) == kind.headDim
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "windowed prefix at layer \(index) has the wrong KV geometry")
                }
                windows[index] = snapshot
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "frozen-full adoption requires at least one storage-owning full layer")
        }

        // --- Install. Everything above validated, so nothing here throws and
        //     no half-built state can escape.
        //
        //     Both forms leave EVERY row on the same logical cursor, which is
        //     what keeps RoPE offsets, masks and `retainedCount` uniform
        //     across layers: M for the restore form, C for the replay form.
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, state) in states.enumerated() {
            guard let row = state as? PagedSequenceKV else { continue }
            if let full = fullKV[index] {
                if restoringWindows {
                    row.write(keys: full.keys, values: full.values)
                } else {
                    // Storage [0, M) is exact and immutable; the cursor
                    // reports C so the replay of [C, M) advances it without
                    // overwriting a byte.
                    row.adoptFrozen(
                        keys: full.keys, values: full.values, replayStart: plan.replayStart)
                }
            } else if let window = windows[index] {
                installWindow(window, into: row, at: matched)
            } else if plan.replayStart > 0 {
                // Windowed row on the replay form: empty, at C, and its
                // `baseOffset` set so no query reaches behind the replay.
                row.fastForward(to: plan.replayStart)
            }
        }
        return states
    }

    /// Place a validated window at its one admissible base. A BYTE WRITE:
    /// the payload is copied into the row's own pages. Nothing is shared and
    /// no page refcount is adopted.
    ///
    /// `fastForward` is what makes the absolute positions right: it sets both
    /// `absoluteOffset` and `baseOffset` to `snapshot.base`, so the tokens
    /// land in the ring slots `(p / pageSize) % ringPages` a cold row would
    /// have used for the same absolute positions, and every later gather and
    /// decode window resolves against the true positions rather than against
    /// a payload that merely starts somewhere.
    ///
    /// The write is SPLIT: `PagedSequenceKV.write` refuses a windowed run
    /// longer than `maxPrefillChunk` because the ring cannot hold one, and a
    /// full window is longer than a chunk by construction (gemma-4: 1,024
    /// against 512).
    ///
    /// Admissibility is re-asserted here rather than trusted from the
    /// caller. `CBv2PagedWindowSnapshot.requireAdmissible` is what stops a
    /// donation taken at position 4,096 being written into an adopter's
    /// `[0, 1024)` — silent wrong answers, no trap, no telemetry — and the
    /// validating call is in `makeFrozenFullState`, twenty lines away and
    /// separated by an install loop. A `precondition` rather than a throw:
    /// the install phase is deliberately non-throwing so no half-built state
    /// can escape, and by this point the refusal path has already run, so
    /// this can only fire on a programming error.
    private func installWindow(
        _ snapshot: CBv2PagedWindowSnapshot, into row: PagedSequenceKV,
        at matchedBoundary: Int
    ) {
        do {
            try snapshot.requireAdmissible(at: matchedBoundary, window: row.windowSize)
        } catch {
            preconditionFailure(
                "[PagedKVBackend] installWindow reached with an inadmissible snapshot "
                    + "at boundary \(matchedBoundary): \(error) — validation must run in "
                    + "makeFrozenFullState before the install phase")
        }
        let keys = snapshot.keys.squeezed(axis: 0)
        let values = snapshot.values.squeezed(axis: 0)
        row.fastForward(to: snapshot.base)
        var written = 0
        while written < snapshot.tokens {
            let count = min(pool.config.maxPrefillChunk, snapshot.tokens - written)
            let range = written ..< (written + count)
            row.write(
                keys: keys[.ellipsis, range, 0...],
                values: values[.ellipsis, range, 0...])
            written += count
        }
    }

    /// `[1, kvHeads, tokens, headDim]` (a donated snapshot) or
    /// `[kvHeads, tokens, headDim]` (already row-shaped) -> row shape.
    private static func rowShaped(_ array: MLXArray) -> MLXArray {
        array.ndim == 4 && array.dim(0) == 1 ? array.squeezed(axis: 0) : array
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        for entry in state {
            guard let entry else { continue }
            guard let paged = entry as? PagedSequenceKV else {
                fatalError("[PagedKVBackend] release of a foreign sequence state")
            }
            paged.releaseStorage()
        }
    }

    public var bytesInUse: Int { pool.bytesInUse }
    public var bytesCapacity: Int { pool.bytesCapacity }
    /// Admission-relevant bytes (worst-case reservations of live requests).
    public var bytesReserved: Int { pool.bytesReserved }
    /// The slabs' allocation CEILING — `pageCount * pageBytes` over every
    /// group, poison pages included. Derived from page bookkeeping fixed at
    /// `PagedKVPool.init`, NOT from whether the slabs have been evaluated,
    /// so it is time-INVARIANT and safe for sizing/limit consumers.
    /// For "how much is resident right now" use `bytesWired`.
    public var bytesPhysical: Int { pool.bytesPhysical }
    /// Paged snapshots are views over the SHARED slabs; the donor's pages
    /// are recycled once its state is released, so donated snapshots must
    /// be device-materialized before the prefix cache indexes them.
    public var requiresMaterializedSnapshots: Bool { true }

    // MARK: - Helpers

    func pageNeeds(layerKinds: [CBv2LayerKind], maxLength: Int) -> [PagedKVGroupKey: Int] {
        var needs: [PagedKVGroupKey: Int] = [:]
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            let pages = PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: pool.config)
            needs[PagedKVGroupKey(kind), default: 0] += pages
        }
        return needs
    }

    /// One layer cache per model layer (KV-shared layers get a borrowing
    /// cache with no rows). `attentionSoftcap` comes from model config —
    /// it is not part of the contract's per-call surface.
    public func makeLayerCaches(attentionSoftcap: Float? = nil) -> [PagedLayerCache] {
        layerKinds.enumerated().map { index, kind in
            PagedLayerCache(
                layerIndex: index, kind: kind, pool: pool,
                attentionSoftcap: attentionSoftcap)
        }
    }
}
