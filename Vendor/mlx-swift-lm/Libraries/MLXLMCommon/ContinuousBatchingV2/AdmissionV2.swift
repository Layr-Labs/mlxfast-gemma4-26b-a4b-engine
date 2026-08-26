// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: truthful admission with soft reservations.
//
// Submit-time check: could the request EVER fit (worst case promptLen +
// maxTokens, window-capped per layer) in total KV capacity? Step-time check:
// vLLM-style optimism — admit while `bytesReserved + nextStepNeed <
// capacity - watermark`; preemption is the backstop when optimism loses.

import Foundation
import MLX

// MARK: - Capacity oracle (scheduler ↔ admission)

/// Soft KV-capacity oracle consulted by `SchedulerV2.plan()` for every
/// assignment. Implemented by `AdmissionV2`; tests use scripted fakes.
/// (WS-B-internal — not part of the frozen contract; see
/// docs/engine-v2/CONTRACT-ISSUES-B-scheduler.md §6.)
public protocol CBv2StepCapacity: AnyObject {
    /// Reserve headroom for `additionalTokens` more KV entries of request
    /// `id`. Throws `CBv2KVError.capacityExhausted` when the soft ledger
    /// would cross the watermark — the scheduler preempts in response.
    func reserve(id: CBv2RequestID, additionalTokens: Int) throws
    /// Reserve token-derived bytes plus an exact native-dtype adjustment in
    /// one admission operation.
    func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws
    /// Partially undo a reservation (optimistic-advance rollback).
    func unreserve(id: CBv2RequestID, tokens: Int)
    func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int)
    /// Drop every reservation held by `id` (finish/cancel/preempt).
    func releaseAll(id: CBv2RequestID)
    /// Non-throwing conservative probe used by the chained-decode fast path.
    func hasHeadroom(additionalTokens: Int) -> Bool
    /// The ledger's current byte ceiling (runtime-resizable). Feeds the
    /// engine's capacity snapshot so re-slices read back consistently.
    /// Defaulted to 0 (= unknown) so simple test capacities need not
    /// implement it.
    var bytesCapacity: Int { get }
}

extension CBv2StepCapacity {
    public var bytesCapacity: Int { 0 }
    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalBytes == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "capacity oracle cannot reserve exact native KV bytes")
        }
        try reserve(id: id, additionalTokens: additionalTokens)
    }
    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(bytes == 0, "capacity oracle cannot unreserve exact native KV bytes")
        unreserve(id: id, tokens: tokens)
    }
}

// MARK: - Backend storage policy

/// How many token ROWS of ONE layer's KV a backend actually occupies once a
/// sequence is bounded by `tokens` tokens. `AdmissionV2` multiplies these
/// rows by that layer's per-token byte cost, so the ledger charges what the
/// backend will really allocate instead of a figure inferred from the layer
/// kind alone.
///
/// The shipped policies differ exactly where the fixed-ring charge is right
/// and where it is wrong:
///
/// * CONTIGUOUS — `CBv2WindowedSequenceKV.allocateIfNeeded` allocates
///   `MLXArray.zeros([1, kvHeads, window, headDim])` on the FIRST write, so
///   a windowed layer occupies its whole ring from token one. Charging only
///   `min(tokens, window)` hid ~0.9 GB of overshoot at B=8 on Gemma-style
///   hybrids (PR#87).
/// * PAGED — `PagedKVPool.pageDemand` reserves
///   `min(ceil(tokens / pageSize), ringPageCount(window))` pages, so a short
///   row holds a handful of pages and NEVER the whole ring. Charging the
///   whole ring there rejects requests the pool can serve: a one-token
///   request needs ONE page, not 1,024 window rows (PR#87 review).
///
/// The backend answers for itself — admission never switches on a backend
/// enum, because the knowledge belongs where the allocation happens.
public protocol CBv2KVResidencyPolicy: Sendable {
    /// nil when the figure cannot be represented in `Int`; admission then
    /// fails cold instead of admitting on wrapped arithmetic.
    func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int?
    /// Rows are handed out in multiples of this (1 for per-token contiguous
    /// arrays, `pageSize` for paged). The conservative headroom probe rounds
    /// its token count up to this so a decode step that crosses a page
    /// boundary is never counted as costing a single row.
    var rowGranularity: Int { get }
}

extension CBv2KVResidencyPolicy {
    public var rowGranularity: Int { 1 }
}

/// Per-sequence contiguous arrays (`CBv2ContiguousKVBackend`): full layers
/// grow with the sequence; windowed layers allocate their whole `window`-row
/// ring on the first write and never grow again.
///
/// This is also the DEFAULT for any backend that does not declare a policy,
/// deliberately: it is the CONSERVATIVE one. A backend that forgets to
/// implement `kvResidency` over-charges its windowed rows and under-admits,
/// which costs throughput — never the reverse, which costs the process.
public struct CBv2ContiguousKVResidency: CBv2KVResidencyPolicy {
    public init() {}

    public func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int? {
        guard tokens >= 0 else { return nil }
        switch kind.attention {
        case .full: return tokens
        case .slidingWindow(let window): return max(0, window)
        }
    }
}

// MARK: - AdmissionV2

/// Byte-ledger admission controller. Thread-safe: `canEverFit` runs on the
/// submitter's thread while reserve/release run on the engine thread.
///
/// Bytes are estimated from `CBv2LayerKind`: layers that share KV storage
/// (`sharesKVWithLayer != nil`) own no bytes; sliding-window layers plateau
/// at `window` tokens — so a long-context request on Gemma-style hybrids is
/// charged truthfully, not as if every layer were full-attention.
///
/// The LEDGER charges `allocatedBytes(forTokens:)`, NOT
/// `estimatedBytes(forTokens:)`: occupancy, not retained content. What a
/// row OCCUPIES is a property of the BACKEND, so the conversion runs
/// through the `CBv2KVResidencyPolicy` the backend hands `EngineV2` — the
/// ledger never guesses from the layer kind alone.
///
/// * Contiguous rows allocate a fixed `window`-row ring on the FIRST write
///   (`CBv2WindowedSequenceKV.allocateIfNeeded`) instead of growing with
///   the sequence, so a 500-token request against a 1024 window occupies
///   the whole ring the moment it is touched. Charging only the retained
///   tokens left the gate believing it had margin it did not have (~0.9 GB
///   of hidden overshoot at B=8 on Gemma-style hybrids), and on unified
///   memory that surfaces as swap pressure, not a clean rejection (PR#87).
/// * Paged rows do NOT commit a ring: `PagedKVPool.pageDemand` caps the
///   reservation at `min(ceil(tokens / pageSize), ringPageCount)` pages.
///   Charging them the whole ring made `canEverFit` and the step ledger
///   reject short requests the pool can serve (PR#87 review).
public final class AdmissionV2: CBv2StepCapacity, @unchecked Sendable {
    public struct Config: Sendable {
        /// Fraction of capacity kept free as the optimism watermark.
        public var watermarkFraction: Double
        /// Default bytes per KV element (2 = fp16/bf16) — the fallback for
        /// layers not listed in `layerElementBytes`.
        public var elementBytes: Int
        /// OPTIONAL per-layer bytes-per-element (aligned to `layerKinds`),
        /// for models whose layers cache K/V at different precisions —
        /// e.g. GPT-OSS full-attention layers cache fp32 K/V (YarnRoPE
        /// computes in fp32) while sliding layers cache bf16. Assuming a
        /// flat 2 bytes/element there under-charges the fp32 rows ~2x and
        /// over-admits. Derive it from the model's probed cache dtypes
        /// (the compiled decode path probes per-layer dtypes at warmup —
        /// see `CBv2CompiledDecode`) via `Config.elementBytes(forDTypes:)`.
        /// nil ⇒ uniform `elementBytes`. Entries for KV-shared layers are
        /// ignored (those layers own no storage).
        public var layerElementBytes: [Int]?
        public init(
            watermarkFraction: Double = 0.05, elementBytes: Int = 2,
            layerElementBytes: [Int]? = nil
        ) {
            self.watermarkFraction = watermarkFraction
            self.elementBytes = elementBytes
            self.layerElementBytes = layerElementBytes
        }

        /// Build a per-layer element-bytes table from probed cache dtypes
        /// (nil entries — KV-shared layers, unprobed — fall back to
        /// `defaultElementBytes`). Conservative: a wider dtype is charged
        /// its full element size.
        public static func elementBytes(
            forDTypes dtypes: [DType?], defaultElementBytes: Int = 2
        ) -> [Int] {
            dtypes.map { $0.map(\.size) ?? defaultElementBytes }
        }
    }

    private let lock = NSLock()
    private let layerKinds: [CBv2LayerKind]
    /// Resolved bytes-per-element per layer (aligned to `layerKinds`).
    private let perLayerElementBytes: [Int]
    /// Watermark fraction retained so `updateBytesCapacity` can recompute
    /// `watermark` against the new capacity.
    private let watermarkFraction: Double
    /// Lock-protected: recomputed whenever the capacity changes.
    private var watermark: Int
    /// Per-token bytes if every storage-owning layer retained the token
    /// (upper bound; used for the conservative headroom probe).
    private let maxPerTokenBytes: Int
    /// Nominal bytes per token for storage-owning full-attention rows under
    /// this ledger's actual per-layer dtype assumptions.
    public let fullKVBytesPerToken: Int

    /// Total KV byte budget. Runtime-resizable via `updateBytesCapacity`
    /// (multi-model co-residency re-slicing); reads take the ledger lock.
    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity
    }
    private var _bytesCapacity: Int

    /// Bytes carved out of the budget for an EXTERNAL worst-case obligation
    /// the ledger cannot see per-request — today the compiled decode path's
    /// padding reserve. REFUNDABLE: if the obligation disappears (compiled
    /// decode disables itself at warmup after a trace failure), the engine
    /// calls `refundExternalReserve()` so admission is not permanently
    /// tighter than the hardware truth (PR#62 review). Lock-protected.
    private var externalReserveBytes: Int

    /// Backend storage policy: how many rows of each layer a row of this
    /// engine's backend really occupies. Immutable — a live engine never
    /// swaps backends.
    private let residency: any CBv2KVResidencyPolicy

    /// Cumulative reserved tokens per request. Under the CONTIGUOUS policy
    /// the token→byte conversion charges every windowed layer its whole
    /// fixed ring once, so decode reservations past a layer's window add
    /// zero bytes for that layer — and so do reservations below it, the
    /// ring being already paid. Under the PAGED policy the same conversion
    /// charges page-capped rows, so short rows stay cheap and the charge
    /// plateaus at the ring instead of starting there.
    private var reservedTokens: [CBv2RequestID: Int] = [:]
    private var reservedExactBytes: [CBv2RequestID: Int] = [:]
    private var ledgerBytes = 0

    public init(
        layerKinds: [CBv2LayerKind], bytesCapacity: Int, config: Config = .init(),
        residency: any CBv2KVResidencyPolicy = CBv2ContiguousKVResidency(),
        externalReserveBytes: Int = 0
    ) {
        self.residency = residency
        self.layerKinds = layerKinds
        self._bytesCapacity = bytesCapacity
        self.externalReserveBytes = max(0, externalReserveBytes)
        if let table = config.layerElementBytes {
            precondition(
                table.count == layerKinds.count,
                "AdmissionV2: layerElementBytes count \(table.count) != layer count \(layerKinds.count)"
            )
            self.perLayerElementBytes = table
        } else {
            self.perLayerElementBytes = Array(
                repeating: config.elementBytes, count: layerKinds.count)
        }
        self.watermarkFraction = config.watermarkFraction
        self.watermark = Int(Double(bytesCapacity) * config.watermarkFraction)
        var perToken = 0
        var fullPerToken = 0
        var accountingOverflow = false
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            guard
                let bytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: self.perLayerElementBytes[index]),
                let newPerToken = Self.add(perToken, bytes)
            else {
                accountingOverflow = true
                break
            }
            perToken = newPerToken
            if case .full = kind.attention {
                guard let newFullPerToken = Self.add(fullPerToken, bytes) else {
                    accountingOverflow = true
                    break
                }
                fullPerToken = newFullPerToken
            }
        }
        self.maxPerTokenBytes = accountingOverflow ? Int.max : perToken
        self.fullKVBytesPerToken = accountingOverflow ? Int.max : fullPerToken
    }

    // MARK: Estimation

    /// KV bytes whose CONTENT is retained after processing `tokens` tokens
    /// of one sequence (windowed layers cap at their window). This is NOT
    /// what a sequence occupies — see `allocatedBytes(forTokens:)`, the
    /// figure the ledger charges.
    public func estimatedBytes(forTokens tokens: Int) -> Int {
        estimatedBytesChecked(forTokens: tokens) ?? Int.max
    }

    private func estimatedBytesChecked(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return 0 }
        var total = 0
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = tokens
            case .slidingWindow(let window): retained = min(tokens, window)
            }
            guard retained >= 0,
                let perTokenBytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: perLayerElementBytes[index]),
                let retainedBytes = Self.multiply(retained, perTokenBytes),
                let newTotal = Self.add(total, retainedBytes)
            else { return nil }
            total = newTotal
        }
        return total
    }

    /// Bytes the BACKEND occupies at `tokens` beyond the content
    /// `estimatedBytes(forTokens:)` retains. `allocatedBytesChecked` folds
    /// this into every ledger charge, which is why no caller adds it a
    /// second time. nil on accounting overflow.
    ///
    /// Under the contiguous policy this is exactly the still-unfilled
    /// remainder of every fixed sliding ring — the windowed rows allocate
    /// the whole ring on first write, so the gap is real occupancy from the
    /// first token onward, and the name is literal. Under the paged policy
    /// there is no fixed ring to fill: the gap is the page-granularity
    /// remainder of `PagedKVPool.pageDemand`, which for a short row is a
    /// page or two rather than a whole window.
    public func fixedWindowBytesShortfall(afterReservingTokens tokens: Int) -> Int? {
        var total = 0
        let held = max(0, tokens)
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = held
            case .slidingWindow(let window): retained = max(0, min(held, window))
            }
            guard let occupied = residency.residentRows(layer: kind, tokens: held),
                occupied >= 0
            else { return nil }
            let missing = max(0, occupied - retained)
            guard
                let perTokenBytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: perLayerElementBytes[index]),
                let missingBytes = Self.multiply(missing, perTokenBytes),
                let newTotal = Self.add(total, missingBytes)
            else { return nil }
            total = newTotal
        }
        return total
    }

    /// KV bytes a sequence OCCUPIES once it has processed `tokens` tokens:
    /// retained content plus whatever the backend holds on top of it. This
    /// is what the ledger charges and what the backend actually allocates —
    /// a contiguous windowed layer costs its whole `window`-row ring from
    /// the first token, while the same layer on the paged backend costs
    /// `min(ceil(tokens / pageSize), ringPageCount)` pages.
    public func allocatedBytes(forTokens tokens: Int) -> Int {
        allocatedBytesChecked(forTokens: tokens) ?? Int.max
    }

    private func allocatedBytesChecked(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return 0 }
        guard let retained = estimatedBytesChecked(forTokens: tokens),
            let ringShortfall = fixedWindowBytesShortfall(afterReservingTokens: tokens),
            let total = Self.add(retained, ringShortfall)
        else { return nil }
        return total
    }

    /// Bytes a single request may ever reserve: `reserve` enforces
    /// `capacity - externalReserve - watermark`, so feasibility must be
    /// judged against the same ceiling. (Judging against full capacity
    /// admitted requests in `(ceiling, capacity]` that could NEVER reserve
    /// their last tokens — they hit the wall, self-preempted, restarted,
    /// and livelocked until their deadline.)
    public var admissibleBytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity - externalReserveBytes - watermark
    }

    /// The ceiling every reservation is checked against. Callers hold `lock`.
    private var reserveCeiling: Int { _bytesCapacity - externalReserveBytes - watermark }

    /// Runtime capacity update (multi-model co-residency re-slicing: the
    /// provider shrinks resident engines to fair shares before granting a
    /// newcomer, and grows survivors back on unload). The watermark is
    /// recomputed from the configured fraction against the new capacity.
    ///
    /// Shrink semantics are safe by construction: reservations already in
    /// the ledger are untouched (nothing is evicted here — preemption
    /// remains the scheduler's job), while NEW `reserve` calls fail with
    /// `capacityExhausted` until the pool drains below the new ceiling.
    /// Grow takes effect immediately.
    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        _bytesCapacity = max(0, bytes)
        watermark = Int(Double(_bytesCapacity) * watermarkFraction)
        lock.unlock()
    }

    /// Release the external carve-out (idempotent). Called by the engine
    /// when the obligation it covered no longer exists — compiled decode
    /// disabled itself at warmup, so its padding can never materialize and
    /// the bytes belong to regular admission again (PR#62 review).
    /// Live external (compiled padding) reserve — the construction value
    /// until `refundExternalReserve()`, then 0. Read by the engine loop's
    /// gauge publish so the snapshot's `kvBytesReserved` carries the FULL
    /// not-available-for-new-admissions figure (backend promises + this
    /// carve), keeping "capacity − reserved" truthful for planners.
    public var bytesExternallyReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return externalReserveBytes
    }

    public func refundExternalReserve() {
        lock.lock()
        externalReserveBytes = 0
        lock.unlock()
    }

    /// Truthful submit-time check: worst case (promptLen + maxTokens) vs the
    /// watermark-adjusted capacity (`admissibleBytesCapacity` — the most
    /// `reserve` will ever grant). Requests that could never fit are
    /// rejected up front; requests that fit only sometimes are admitted
    /// optimistically and preempted if optimism loses. Judged on OCCUPANCY
    /// (fixed sliding rings included), so the gate agrees with both what
    /// `reserve` charges and what the backend allocates.
    public func canEverFit(promptTokens: Int, maxTokens: Int) -> Bool {
        let (tokens, overflow) = promptTokens.addingReportingOverflow(max(maxTokens, 0))
        guard !overflow, let bytes = allocatedBytesChecked(forTokens: tokens) else {
            return false
        }
        return bytes <= admissibleBytesCapacity
    }

    // MARK: CBv2StepCapacity

    public func reserve(id: CBv2RequestID, additionalTokens: Int) throws {
        try reserve(id: id, additionalTokens: additionalTokens, additionalBytes: 0)
    }

    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalTokens >= 0, additionalBytes >= 0 else {
            throw CBv2KVError.backendIneligible(reason: "negative KV reservation")
        }
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let (new, tokenCountOverflow) = old.addingReportingOverflow(additionalTokens)
        let oldExact = reservedExactBytes[id] ?? 0
        let (newExact, exactOverflow) = oldExact.addingReportingOverflow(additionalBytes)
        guard !tokenCountOverflow, !exactOverflow,
            let oldTokenBytes = allocatedBytesChecked(forTokens: old),
            let newTokenBytes = allocatedBytesChecked(forTokens: new)
        else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (tokenDelta, tokenOverflow) = newTokenBytes.subtractingReportingOverflow(oldTokenBytes)
        let (delta, deltaOverflow) = tokenDelta.addingReportingOverflow(additionalBytes)
        guard !tokenOverflow, !deltaOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (after, afterOverflow) = ledgerBytes.addingReportingOverflow(delta)
        guard !afterOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        guard after <= reserveCeiling else {
            throw CBv2KVError.capacityExhausted(
                needed: delta,
                available: max(0, reserveCeiling - ledgerBytes))
        }
        reservedTokens[id] = new
        reservedExactBytes[id] = newExact
        ledgerBytes = after
    }

    public func unreserve(id: CBv2RequestID, tokens: Int) {
        unreserve(id: id, tokens: tokens, bytes: 0)
    }

    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(tokens >= 0 && bytes >= 0)
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = max(0, old - tokens)
        let oldExact = reservedExactBytes[id] ?? 0
        let newExact = max(0, oldExact - bytes)
        ledgerBytes += allocatedBytes(forTokens: new) - allocatedBytes(forTokens: old)
            + newExact - oldExact
        if new == 0 {
            reservedTokens.removeValue(forKey: id)
        } else {
            reservedTokens[id] = new
        }
        if newExact == 0 {
            reservedExactBytes.removeValue(forKey: id)
        } else {
            reservedExactBytes[id] = newExact
        }
    }

    public func releaseAll(id: CBv2RequestID) {
        lock.lock()
        defer { lock.unlock() }
        guard let old = reservedTokens.removeValue(forKey: id) else { return }
        let exact = reservedExactBytes.removeValue(forKey: id) ?? 0
        ledgerBytes -= allocatedBytes(forTokens: old) + exact
    }

    /// Conservative (window caps ignored): may under-report headroom, which
    /// only breaks a decode chain — never over-admits. The token count is
    /// first rounded UP to the backend's row granularity, so on a paged
    /// backend a single decode token that crosses a page boundary is priced
    /// as the whole page it can cost, not as one row.
    public func hasHeadroom(additionalTokens: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard additionalTokens >= 0,
            let chargedTokens = Self.roundUp(additionalTokens, to: residency.rowGranularity),
            let additionalBytes = Self.multiply(chargedTokens, maxPerTokenBytes),
            let after = Self.add(ledgerBytes, additionalBytes)
        else { return false }
        return after <= reserveCeiling
    }

    /// Smallest multiple of `granularity` at or above `value`; nil on
    /// overflow. A non-positive granularity means "no rounding".
    private static func roundUp(_ value: Int, to granularity: Int) -> Int? {
        guard granularity > 1 else { return value }
        guard let bumped = add(value, granularity - 1) else { return nil }
        return (bumped / granularity) * granularity
    }

    private static func storageBytesPerToken(
        kind: CBv2LayerKind,
        elementBytes: Int
    ) -> Int? {
        guard kind.kvHeads >= 0, kind.headDim >= 0, elementBytes >= 0,
            let elements = multiply(kind.kvHeads, kind.headDim),
            let kvElements = multiply(elements, 2)
        else { return nil }
        return multiply(kvElements, elementBytes)
    }

    private static func multiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }

    private static func add(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }

    // MARK: Telemetry

    /// Ledger bytes currently reserved (soft truth; the backend's
    /// `bytesInUse` is the hard truth once arrays materialize).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return ledgerBytes
    }

    public func snapshot(
        activeRequests: Int, waitingRequests: Int, activeTokens: Int, backendBytesInUse: Int? = nil
    ) -> CBv2CapacitySnapshot {
        lock.lock()
        let ledger = ledgerBytes
        // Honor the field's contract (see `CBv2CapacitySnapshot
        // .kvBytesReserved`): reserved carries the live external (compiled
        // padding) carve too, so "capacity − reserved" matches what
        // `canEverFit`/`reserve` will actually admit — the same figure the
        // engine loop's gauge publish reports. The carve is NOT storage,
        // so the in-use fallback stays ledger-only.
        let reserved = ledger + externalReserveBytes
        let capacity = _bytesCapacity
        lock.unlock()
        return CBv2CapacitySnapshot(
            activeRequests: activeRequests,
            waitingRequests: waitingRequests,
            kvBytesInUse: backendBytesInUse ?? ledger,
            kvBytesCapacity: capacity,
            kvBytesReserved: reserved,
            activeTokens: activeTokens)
    }
}
