// Copyright © 2026 Eigen Labs.
//
// WS-B: AdmissionV2 tests — truthful worst-case estimates (window-capped,
// KV-sharing aware), soft reservations with a watermark, rollback symmetry.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerAdmissionTests: XCTestCase {

    private func id(_ raw: UInt64) -> CBv2RequestID { CBv2RequestID(raw) }

    /// full + sliding-window(128) + KV-shared layer, kvHeads 2, headDim 8.
    private var hybridKinds: [CBv2LayerKind] {
        [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 0, headDim: 8, kvHeads: 2, queryHeads: 4),
        ]
    }

    func testEstimateRespectsWindowsAndKVSharing() {
        let admission = AdmissionV2(
            layerKinds: hybridKinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        // Per token per storage-owning layer: 2(K+V) × 2 heads × 8 dim × 2 B = 64 B.
        // 1000 tokens: full layer = 64_000; windowed = min(1000,128)×64 = 8192;
        // KV-shared layer owns NOTHING.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 1000), 64_000 + 8192)
        // Below the window, both storage layers charge fully.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 100), 100 * 64 * 2)
        XCTAssertEqual(admission.estimatedBytes(forTokens: 0), 0)
    }

    /// Per-layer element bytes (PR#62 review): GPT-OSS caches fp32 K/V on
    /// full-attention layers but bf16 on sliding ones. A flat 2 B/element
    /// under-charges the fp32 rows ~2x and over-admits — the estimate must
    /// honor a per-layer dtype table.
    func testPerLayerElementBytesForMixedPrecision() {
        // layer 0 full (fp32, 4 B), layer 1 sliding-window(128) (bf16, 2 B).
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 8, kvHeads: 2, queryHeads: 4),
        ]
        let table = AdmissionV2.Config.elementBytes(forDTypes: [.float32, .bfloat16])
        XCTAssertEqual(table, [4, 2])
        let mixed = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, layerElementBytes: table))
        // 1000 tokens: full(fp32) = 1000 × 2 × 2 × 8 × 4 = 128_000;
        // windowed(bf16) = min(1000,128) × 2 × 2 × 8 × 2 = 8192.
        XCTAssertEqual(mixed.estimatedBytes(forTokens: 1000), 128_000 + 8192)

        // The flat-2-bytes assumption under-charges the fp32 layer by 2x.
        let flat = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        XCTAssertEqual(flat.estimatedBytes(forTokens: 1000), 64_000 + 8192)
        XCTAssertGreaterThan(
            mixed.estimatedBytes(forTokens: 1000), flat.estimatedBytes(forTokens: 1000),
            "fp32 full-attention rows must be charged ~2x the flat bf16 assumption")

        // Feasibility + reservation both honor the wider dtype.
        XCTAssertLessThan(
            mixed.admissibleBytesCapacity - mixed.estimatedBytes(forTokens: 1000),
            flat.admissibleBytesCapacity - flat.estimatedBytes(forTokens: 1000))
    }

    /// A per-layer table shorter/longer than the layer count is a construction
    /// bug — caught eagerly.
    func testMismatchedElementBytesTableTraps() {
        // Length mismatch would silently mis-charge; ensure the helper builds
        // a correctly-sized table from probed dtypes (nil ⇒ default).
        let table = AdmissionV2.Config.elementBytes(
            forDTypes: [.float32, nil, .bfloat16], defaultElementBytes: 2)
        XCTAssertEqual(table, [4, 2, 2], "nil (KV-shared / unprobed) falls back to default")
    }

    func testCanEverFitIsWorstCase() {
        // Capacity fits exactly 100 tokens of the single full layer
        // (32 B/token: 2 × 1 head × 8 dim × 2 B).
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0))
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
        XCTAssertFalse(admission.canEverFit(promptTokens: 50, maxTokens: 51))
        // The check ignores current usage — feasibility, not availability.
        try? admission.reserve(id: id(1), additionalTokens: 100)
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
    }

    /// Regression (admission/reserve livelock): `canEverFit` judged against
    /// FULL capacity while `reserve` enforces `capacity - watermark`. A
    /// request whose worst case landed in `(capacity - watermark, capacity]`
    /// was admitted, could never reserve its final tokens, self-preempted,
    /// restarted, and livelocked until its deadline. Feasibility must use
    /// the same watermark-adjusted ceiling `reserve` enforces.
    func testCanEverFitRespectsWatermark() throws {
        // 1 full layer, kv1 hd8 eb2 → 32 B/token. Capacity 3200, watermark
        // 5% (160 B) → 3040 usable → 95 tokens is the true ceiling.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0.05))
        XCTAssertEqual(admission.admissibleBytesCapacity, 3040)

        // 95 tokens: admissible AND reservable end to end.
        XCTAssertTrue(admission.canEverFit(promptTokens: 45, maxTokens: 50))
        try admission.reserve(id: id(1), additionalTokens: 95)
        admission.releaseAll(id: id(1))

        // 96..100 tokens: previously admitted (<= full capacity) but the
        // solo reservation can never complete — must be rejected up front.
        for total in 96 ... 100 {
            XCTAssertFalse(
                admission.canEverFit(promptTokens: total - 50, maxTokens: 50),
                "\(total)-token worst case exceeds capacity - watermark")
            XCTAssertThrowsError(
                try admission.reserve(id: id(2), additionalTokens: total),
                "reserve agrees: \(total) tokens can never be granted")
            admission.releaseAll(id: id(2))
        }
    }

    func testSoftReserveThrowsAtWatermark() throws {
        // 1 full layer, kv1 hd1 eb2 → 4 B/token. Capacity 1000, watermark
        // 10% → 900 usable → 225 tokens.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0.1))
        try admission.reserve(id: id(1), additionalTokens: 225)
        XCTAssertThrowsError(try admission.reserve(id: id(2), additionalTokens: 1)) { error in
            guard case CBv2KVError.capacityExhausted(let needed, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(needed, 4)
            XCTAssertEqual(available, 0)
        }
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 1))
    }

    func testOverflowingCapacityInputsFailWithoutMutatingLedger() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let admission = AdmissionV2(
            layerKinds: kinds,
            bytesCapacity: Int.max,
            config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 1)
        XCTAssertThrowsError(
            try admission.reserve(id: id(1), additionalTokens: Int.max))
        XCTAssertEqual(admission.bytesReserved, 4)
        XCTAssertFalse(admission.canEverFit(promptTokens: Int.max, maxTokens: 1))
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: Int.max))
        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 0)
    }

    func testUnreserveAndReleaseRestoreHeadroom() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 400, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 60)  // 240 B
        try admission.reserve(id: id(2), additionalTokens: 40)  // 400 B
        XCTAssertEqual(admission.bytesReserved, 400)
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 1))

        // Optimistic-advance rollback path.
        admission.unreserve(id: id(2), tokens: 10)
        XCTAssertEqual(admission.bytesReserved, 360)
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 10))

        // Finish/cancel/preempt path.
        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 120)
        try admission.reserve(id: id(3), additionalTokens: 70)
        XCTAssertEqual(admission.bytesReserved, 400)
    }

    func testWindowedReservationPlateaus() throws {
        // Sliding window 4 → per-request bytes stop growing past 4 tokens,
        // so long decodes on windowed layers do not exhaust the ledger.
        let kinds = [CBv2LayerKind(attention: .slidingWindow(4), headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 64, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 4)  // 16 B (at the plateau)
        XCTAssertEqual(admission.bytesReserved, 16)
        for _ in 0 ..< 1000 {
            try admission.reserve(id: id(1), additionalTokens: 1)  // +0 B each
        }
        XCTAssertEqual(admission.bytesReserved, 16)
        // A second sequence still pays its own window.
        try admission.reserve(id: id(2), additionalTokens: 100)
        XCTAssertEqual(admission.bytesReserved, 32)
    }

    // MARK: Fixed sliding rings (Bug A — paged-KV plan §7 item 0.1)

    /// Gemma-4-26b shape: 25 sliding-window(1024) layers (head_dim 256, 8 kv
    /// heads) + 5 full layers (head_dim 512, 2 kv heads).
    private var gemmaKinds: [CBv2LayerKind] {
        Array(
            repeating: CBv2LayerKind(
                attention: .slidingWindow(1024), headDim: 256, kvHeads: 8, queryHeads: 8),
            count: 25)
            + Array(
                repeating: CBv2LayerKind(
                    attention: .full, headDim: 512, kvHeads: 2, queryHeads: 8),
                count: 5)
    }

    /// Regression (Bug A): the windowed backends allocate the WHOLE
    /// `window`-row ring on the first write (`WindowedSequenceKV` —
    /// `guard keys == nil else { return }`, then `MLXArray.zeros([1, kvHeads,
    /// window, ...])`), so a request shorter than the window occupies the
    /// full ring. Admission used to charge `min(tokens, window)`, leaving the
    /// gate blind to the difference. The charge must now equal the
    /// allocation.
    func testWindowedLayerIsChargedItsWholeRingBelowTheWindow() throws {
        let admission = AdmissionV2(
            layerKinds: gemmaKinds, bytesCapacity: 1 << 40,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        let tokens = 500  // well below the 1024 window

        // Per token per storage-owning layer:
        //   windowed 2(K+V) × 8 heads × 256 dim × 2 B = 8192 B
        //   full     2(K+V) × 2 heads × 512 dim × 2 B = 4096 B
        let windowedRingBytes = 25 * 1024 * 8192  // 209.7 MB — what is allocated
        let fullBytes = 5 * tokens * 4096

        // The ledger's charge now IS the allocation.
        XCTAssertEqual(admission.allocatedBytes(forTokens: tokens), windowedRingBytes + fullBytes)
        try admission.reserve(id: id(1), additionalTokens: tokens)
        XCTAssertEqual(admission.bytesReserved, windowedRingBytes + fullBytes)

        // Retention is unchanged and still window-capped — that estimate is
        // simply not what a sequence occupies. The gap it used to hide is
        // ~107 MB for ONE such request.
        XCTAssertEqual(admission.estimatedBytes(forTokens: tokens), 25 * tokens * 8192 + fullBytes)
        XCTAssertEqual(
            admission.allocatedBytes(forTokens: tokens) - admission.estimatedBytes(forTokens: tokens),
            try XCTUnwrap(admission.fixedWindowBytesShortfall(afterReservingTokens: tokens)))

        // Past the window the two agree, and the ring is never charged twice:
        // reservations below AND above the window add nothing for it.
        try admission.reserve(id: id(1), additionalTokens: 2048 - tokens)
        XCTAssertEqual(admission.bytesReserved, windowedRingBytes + 5 * 2048 * 4096)
        XCTAssertEqual(
            admission.allocatedBytes(forTokens: 2048), admission.estimatedBytes(forTokens: 2048))

        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 0, "rollback stays symmetric")
    }

    /// The overshoot this closes: eight 500-token requests looked like
    /// 0.9 GB to the old ledger while allocating 1.76 GB. The gate must now
    /// stop admitting before the hardware does.
    func testWindowedOvershootIsRejectedRatherThanHidden() {
        let capacity = 1_000_000_000
        let admission = AdmissionV2(
            layerKinds: gemmaKinds, bytesCapacity: capacity,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        let tokens = 500

        // What the old (retention-only) ledger believed eight rows cost —
        // comfortably inside the budget.
        XCTAssertLessThan(8 * admission.estimatedBytes(forTokens: tokens), capacity)
        // What they actually allocate.
        XCTAssertGreaterThan(8 * admission.allocatedBytes(forTokens: tokens), capacity)

        var admitted = 0
        for raw in 1 ... 8 {
            do {
                try admission.reserve(id: id(UInt64(raw)), additionalTokens: tokens)
                admitted += 1
            } catch {
                break
            }
        }
        XCTAssertEqual(admitted, 4, "the ledger stops at the real ceiling, not the imagined one")
        XCTAssertLessThanOrEqual(admission.bytesReserved, capacity)
    }

    /// The prefix-cache adoption path reserves `capacityReservationTokens`
    /// plus the plan's `initialAdditionalCapacityBytes`. Now that the token
    /// charge carries the fixed rings, `EngineV2.makePrefixLookup` no longer
    /// routes `fixedWindowBytesShortfall` through the plan — the ring must
    /// land in the ledger exactly ONCE.
    func testAdoptionReservationChargesFixedRingExactlyOnce() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(16), headDim: 8, kvHeads: 1, queryHeads: 2),
        ]
        // 2(K+V) × 1 head × 8 dim × 2 B = 32 B per token per layer.
        let bytesPerToken = 32
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 20, config: .init(watermarkFraction: 0))
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)
        let maximumSequenceLength = 40
        let plan = try XCTUnwrap(
            capability.plan(
                matchedBoundary: 24,
                exactStagedFullKVBytes: 24 * bytesPerToken,
                maximumSequenceLength: maximumSequenceLength,
                nominalFullKVBytesPerToken: admission.fullKVBytesPerToken))
        XCTAssertEqual(plan.strategy, .tailReplay)
        XCTAssertEqual(plan.capacityReservationTokens, 8)
        let ringBytes = 16 * bytesPerToken
        XCTAssertEqual(
            plan.initialAdditionalCapacityBytes,
            (maximumSequenceLength - plan.capacityReservationTokens) * bytesPerToken,
            "the plan carries the full-row span only — the ring is the token charge's job")

        try admission.reserve(
            id: id(777),
            additionalTokens: plan.capacityReservationTokens,
            additionalBytes: plan.initialAdditionalCapacityBytes)
        // Backend truth: the full row is sized to the whole sequence, the
        // sliding row to its ring. Charging the ring twice would land
        // `ringBytes` above this.
        let exactBackendBytes = maximumSequenceLength * bytesPerToken + ringBytes
        XCTAssertEqual(admission.bytesReserved, exactBackendBytes)

        admission.unreserve(
            id: id(777),
            tokens: plan.capacityReservationTokens,
            bytes: plan.initialAdditionalCapacityBytes)
        XCTAssertEqual(admission.bytesReserved, 0, "adoption rollback balances exactly")
    }

    func testSnapshotReportsLedgerOrBackendTruth() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 25)  // 100 B
        let soft = admission.snapshot(activeRequests: 1, waitingRequests: 2, activeTokens: 25)
        XCTAssertEqual(soft.kvBytesInUse, 100)
        XCTAssertEqual(soft.kvBytesCapacity, 1000)
        XCTAssertEqual(soft.activeRequests, 1)
        XCTAssertEqual(soft.waitingRequests, 2)
        XCTAssertEqual(soft.activeTokens, 25)

        let hard = admission.snapshot(
            activeRequests: 1, waitingRequests: 2, activeTokens: 25, backendBytesInUse: 88)
        XCTAssertEqual(hard.kvBytesInUse, 88, "backend truth wins when provided")
    }

    // MARK: External reserve (compiled padding carve-out, PR#62 review)

    /// The compiled decode path carves its worst-case padding reserve out of
    /// the ledger at engine build. If warmup tracing disables compiled
    /// decode, the engine refunds the carve-out — admission must then judge
    /// against the FULL budget again, not stay permanently tighter than the
    /// hardware truth.
    func testExternalReserveTightensAdmissionAndRefundRestores() throws {
        // Single full layer, kvHeads 1, headDim 4 ⇒ 16 B/token.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1600,
            config: .init(watermarkFraction: 0),
            externalReserveBytes: 800)

        // 60 tokens = 960 B: fits the hardware budget (1600) but not the
        // reserve-reduced ceiling (800).
        XCTAssertEqual(admission.admissibleBytesCapacity, 800)
        XCTAssertFalse(admission.canEverFit(promptTokens: 40, maxTokens: 20))
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 60))
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 60)) { error in
            guard case CBv2KVError.capacityExhausted(_, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(available, 800, "available must reflect the carve-out")
        }

        // Refund (compiled decode disabled at warmup): full budget again.
        admission.refundExternalReserve()
        XCTAssertEqual(admission.admissibleBytesCapacity, 1600)
        XCTAssertTrue(admission.canEverFit(promptTokens: 40, maxTokens: 20))
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 60))
        XCTAssertNoThrow(try admission.reserve(id: id(1), additionalTokens: 60))
        // 960 B reserved; 40 more tokens (640 B) tops out exactly at 1600.
        XCTAssertNoThrow(try admission.reserve(id: id(2), additionalTokens: 40))
        XCTAssertThrowsError(try admission.reserve(id: id(3), additionalTokens: 1))

        // Refund is idempotent.
        admission.refundExternalReserve()
        XCTAssertEqual(admission.admissibleBytesCapacity, 1600)
    }

    // MARK: Runtime capacity updates (multi-model co-residency re-slicing)

    /// Shrink below current usage: in-flight reservations are untouched
    /// (nothing evicted), NEW reserves fail until the pool drains below the
    /// new ceiling, then admission resumes under the NEW ceiling.
    func testUpdateBytesCapacityShrinkUnderLoad() throws {
        // 1 full layer, kv1 hd1 eb2 ⇒ 4 B/token. No watermark for exact math.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0))

        // Fill near the ceiling: 960 of 1000 bytes.
        try admission.reserve(id: id(1), additionalTokens: 140)  // 560 B
        try admission.reserve(id: id(2), additionalTokens: 100)  // 400 B
        XCTAssertEqual(admission.bytesReserved, 960)

        // Shrink BELOW current usage.
        admission.updateBytesCapacity(500)
        XCTAssertEqual(admission.bytesCapacity, 500)

        // Existing reservations are untouched — the ledger still carries
        // them and unreserve/releaseAll stay symmetric.
        XCTAssertEqual(admission.bytesReserved, 960)

        // New reserves fail while the pool sits above the new ceiling —
        // including zero-cost feasibility (canEverFit judges vs 500 now).
        XCTAssertFalse(admission.canEverFit(promptTokens: 100, maxTokens: 60))
        XCTAssertThrowsError(try admission.reserve(id: id(3), additionalTokens: 1)) { error in
            guard case CBv2KVError.capacityExhausted(_, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(available, 0, "over-ceiling pool must report zero availability")
        }
        // Even the in-flight requests cannot GROW their reservations — the
        // scheduler preempts on this signal; the ledger never over-admits.
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 1))
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 1))

        // Partial drain that STILL exceeds the ceiling: still exhausted
        // (560 B remain > 500).
        admission.releaseAll(id: id(2))
        XCTAssertEqual(admission.bytesReserved, 560)
        XCTAssertThrowsError(try admission.reserve(id: id(3), additionalTokens: 1))

        // Drain below the new ceiling: admission resumes, and the NEW
        // ceiling binds exactly (125 tokens == 500 B).
        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 0)
        XCTAssertTrue(admission.canEverFit(promptTokens: 100, maxTokens: 25))
        try admission.reserve(id: id(5), additionalTokens: 125)
        XCTAssertEqual(admission.bytesReserved, 500)
        XCTAssertThrowsError(try admission.reserve(id: id(6), additionalTokens: 1))
    }

    /// Grow takes effect immediately: a ledger at the old ceiling admits
    /// more the moment the capacity rises.
    func testUpdateBytesCapacityGrowAdmitsImmediately() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 400, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 100)  // 400 B — at the ceiling
        XCTAssertThrowsError(try admission.reserve(id: id(2), additionalTokens: 1))
        XCTAssertFalse(admission.canEverFit(promptTokens: 100, maxTokens: 1))

        admission.updateBytesCapacity(800)
        XCTAssertEqual(admission.bytesCapacity, 800)
        XCTAssertTrue(admission.canEverFit(promptTokens: 100, maxTokens: 100))
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 100))
        try admission.reserve(id: id(2), additionalTokens: 100)  // 800 B total
        XCTAssertEqual(admission.bytesReserved, 800)
        XCTAssertThrowsError(try admission.reserve(id: id(3), additionalTokens: 1))
    }

    /// The watermark is recomputed from the CONFIGURED fraction against the
    /// new capacity — observable through `admissibleBytesCapacity`
    /// (capacity − externalReserve − watermark) and the reserve boundary.
    func testUpdateBytesCapacityRecomputesWatermark() throws {
        // 4 B/token; 10% watermark. 1000 ⇒ admissible 900.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0.1))
        XCTAssertEqual(admission.admissibleBytesCapacity, 900)

        // Grow: watermark scales up with the new capacity (2000 ⇒ 200).
        admission.updateBytesCapacity(2000)
        XCTAssertEqual(admission.admissibleBytesCapacity, 1800)
        try admission.reserve(id: id(1), additionalTokens: 450)  // exactly 1800 B
        XCTAssertThrowsError(try admission.reserve(id: id(2), additionalTokens: 1))
        admission.releaseAll(id: id(1))

        // Shrink: watermark scales down (500 ⇒ 50).
        admission.updateBytesCapacity(500)
        XCTAssertEqual(admission.admissibleBytesCapacity, 450)
        // Snapshot reports the live capacity.
        XCTAssertEqual(
            admission.snapshot(activeRequests: 0, waitingRequests: 0, activeTokens: 0)
                .kvBytesCapacity,
            500)

        // Degenerate input clamps to zero — nothing is ever admissible.
        admission.updateBytesCapacity(-5)
        XCTAssertEqual(admission.bytesCapacity, 0)
        XCTAssertFalse(admission.canEverFit(promptTokens: 1, maxTokens: 0))
    }

    /// `snapshot().kvBytesReserved` honors the field's contract: it carries
    /// the live external (compiled padding) carve on top of the per-request
    /// ledger — so "capacity − reserved" matches what `reserve` actually
    /// admits, the same figure the engine loop's gauge publish reports —
    /// and drops it after the warmup refund. The carve is a reservation,
    /// not storage: the in-use fallback stays ledger-only.
    /// (Fixture shape = 4 bytes/token: one full layer, kvHeads 1, headDim 1,
    /// elementBytes 2, K+V.)
    func testSnapshotReservedIncludesLiveExternalReserve() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000,
            config: .init(watermarkFraction: 0),
            externalReserveBytes: 300)

        // Idle: reserved = the standing carve alone; in-use stays 0.
        let idle = admission.snapshot(activeRequests: 0, waitingRequests: 0, activeTokens: 0)
        XCTAssertEqual(idle.kvBytesReserved, 300)
        XCTAssertEqual(idle.kvBytesInUse, 0)
        XCTAssertEqual(admission.bytesExternallyReserved, 300)

        // With a live reservation (100 tokens = 400 B): reserved = ledger +
        // carve, and "capacity − reserved" IS the remaining admit ceiling —
        // exactly 75 more tokens fit, one more byte-worth throws.
        try admission.reserve(id: id(9), additionalTokens: 100)
        let busy = admission.snapshot(activeRequests: 1, waitingRequests: 0, activeTokens: 100)
        XCTAssertEqual(busy.kvBytesReserved, 700)
        XCTAssertEqual(busy.kvBytesInUse, 400, "the carve is not storage")
        XCTAssertEqual(busy.kvBytesCapacity - busy.kvBytesReserved, 300)
        try admission.reserve(id: id(9), additionalTokens: 75)
        XCTAssertThrowsError(try admission.reserve(id: id(9), additionalTokens: 1))

        // Refund (compiled decode disabled at warmup): the carve leaves
        // reserved; the ledger remains.
        admission.refundExternalReserve()
        XCTAssertEqual(admission.bytesExternallyReserved, 0)
        let refunded = admission.snapshot(activeRequests: 1, waitingRequests: 0, activeTokens: 175)
        XCTAssertEqual(refunded.kvBytesReserved, 700)
    }

    // MARK: Backend-aware occupancy (PR#87 review — paged over-charge)

    /// gemma-4 windowed shape, 8,192 B per token per layer.
    private var gemmaWindowedKind: CBv2LayerKind {
        CBv2LayerKind(attention: .slidingWindow(1024), headDim: 256, kvHeads: 8, queryHeads: 8)
    }

    /// The gemma-4 paged sizing: pageSize 16, chunk 512. The ring it implies
    /// is deliberately NOT restated here — it is `ringPageCount`'s to define
    /// and it has already moved once (WS-1.2 took gemma-4's from 97 pages to
    /// 65 once `PagedLayerCache` gathered BEFORE writing). Every expectation
    /// below that depends on it reads `PagedKVPool.ringPageCount` instead of
    /// a literal, which is what carried this suite across that change.
    private var gemmaPagedConfig: PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 1 << 30, maxPrefillChunk: 512, nominalMaxSequenceLength: 8192)
    }

    /// The whole-ring charge PR#87 introduced is right for CONTIGUOUS rows
    /// and wrong for PAGED ones, so the two must diverge. On one gemma-4
    /// windowed layer (8,192 B/token, window 1,024):
    ///
    /// | tokens | contiguous rows | paged rows  | contiguous B | paged B     |
    /// |--------|-----------------|-------------|--------------|-------------|
    /// |      1 |           1,024 |          16 |    8,388,608 |     131,072 |
    /// |    500 |           1,024 |         512 |    8,388,608 |   4,194,304 |
    /// |  2,000 |           1,024 | whole ring  |    8,388,608 | ring × 8,192 |
    ///
    /// The short-row figures are LITERAL — one page and 32 pages are the
    /// whole point of the fix and must not drift. The 2,000-token row is
    /// derived from `ringPageCount` on purpose: the ring's width is under
    /// active revision, and pinning a literal there would turn a correct
    /// engine change into a red test (it already did once, at 1,552).
    /// What must hold regardless of the ring's width is the SHAPE: past the
    /// ring the paged charge stops tracking tokens and plateaus at the ring,
    /// and that plateau is at least the window — charging less is a
    /// free-list underflow, not a rejected request. Paged is therefore not
    /// uniformly cheaper; this is a change of policy, not a relaxation.
    func testWindowedChargeDivergesByBackendStoragePolicy() {
        let kinds = [gemmaWindowedKind]
        let bytesPerToken = 2 * 8 * 256 * 2  // 8,192
        let contiguous = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 40,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        let paged = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 40,
            config: .init(watermarkFraction: 0, elementBytes: 2),
            residency: CBv2PagedKVResidency(config: gemmaPagedConfig))

        for tokens in [1, 500, 2000] {
            XCTAssertEqual(
                contiguous.allocatedBytes(forTokens: tokens), 1024 * bytesPerToken,
                "contiguous windowed rows own the whole ring from token one (PR#87)")
        }
        XCTAssertEqual(paged.allocatedBytes(forTokens: 1), 16 * bytesPerToken)
        XCTAssertEqual(paged.allocatedBytes(forTokens: 500), 512 * bytesPerToken)
        let ringRows = PagedKVPool.ringPageCount(
            window: 1024, config: gemmaPagedConfig) * gemmaPagedConfig.pageSize
        XCTAssertEqual(paged.allocatedBytes(forTokens: 2000), ringRows * bytesPerToken)
        XCTAssertGreaterThanOrEqual(
            ringRows, 1024,
            "the ring must cover the window or the paged charge under-states the pool")
        XCTAssertEqual(
            paged.allocatedBytes(forTokens: 100_000), ringRows * bytesPerToken,
            "past the ring the charge plateaus instead of tracking tokens")

        // Retention is a property of the MODEL, not the backend — unchanged.
        for admission in [contiguous, paged] {
            XCTAssertEqual(admission.estimatedBytes(forTokens: 1), bytesPerToken)
            XCTAssertEqual(admission.estimatedBytes(forTokens: 500), 500 * bytesPerToken)
            XCTAssertEqual(admission.estimatedBytes(forTokens: 2000), 1024 * bytesPerToken)
        }
    }

    /// The P1 itself: a short request the paged pool CAN serve was being
    /// rejected because admission charged it a whole 1,024-row window.
    ///
    /// "The pool can serve it" is demonstrated, not asserted — the real
    /// `PagedKVBackend.reserve` takes the pages first. The pool is
    /// deliberately sized BELOW one contiguous ring, which is the regime
    /// where the two policies disagree.
    func testPagedShortRequestIsAdmittedWhereWholeRingChargingRejectsIt() throws {
        // One windowed layer, 512 B/token, page = 16 tokens = 8,192 B.
        // capacity 40 pages ⇒ 20,480 B less than the 524,288 B ring.
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(1024), headDim: 64, kvHeads: 2, queryHeads: 4)
        ]
        let pageBytes = 2 * 2 * 16 * 64 * 2  // 8,192
        let backend: any CBv2KVBackend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 40 * pageBytes, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 1024))
        let capacity = backend.bytesCapacity
        XCTAssertLessThan(
            capacity, 1024 * 512,
            "fixture must be smaller than one contiguous ring or the bug cannot show")

        // The whole-ring ledger PR#87 shipped, and the backend's own.
        let wholeRing = AdmissionV2(
            layerKinds: kinds, bytesCapacity: capacity, config: .init(watermarkFraction: 0))
        let backendAware = AdmissionV2(
            layerKinds: kinds, bytesCapacity: capacity, config: .init(watermarkFraction: 0),
            residency: backend.kvResidency)

        for tokens in [1, 64, 500] {
            // The pool really can serve it: it hands over the pages.
            let paged = try XCTUnwrap(backend as? PagedKVBackend)
            XCTAssertNoThrow(
                try paged.reserve(layerKinds: kinds, maxLength: tokens),
                "\(tokens)-token row is within the pool's page budget")
            paged.unreserve(layerKinds: kinds, maxLength: tokens)

            XCTAssertFalse(
                wholeRing.canEverFit(promptTokens: tokens, maxTokens: 0),
                "the bug: \(tokens)-token request charged all 1,024 window rows")
            XCTAssertTrue(
                backendAware.canEverFit(promptTokens: tokens, maxTokens: 0),
                "paged charges pages, so the gate agrees with the pool")
            XCTAssertNoThrow(try backendAware.reserve(id: id(1), additionalTokens: tokens))
            backendAware.releaseAll(id: id(1))
        }

        // One page per short row, so the ledger admits as many as the pool
        // has usable pages — every one of which the old charge refused.
        XCTAssertEqual(backendAware.allocatedBytes(forTokens: 1), pageBytes)
        var admitted = 0
        while (try? backendAware.reserve(id: id(UInt64(admitted + 1)), additionalTokens: 1)) != nil {
            admitted += 1
            if admitted > capacity / pageBytes { break }
        }
        XCTAssertEqual(admitted, capacity / pageBytes)

        // Not a relaxation: what the pool cannot serve is still refused. A
        // 2,000-token row saturates the whole ring, and the ring dwarfs this
        // fixture's 40 pages under either sizing it has had (65 now, 97
        // before WS-1.2), so both the ledger and the pool say no.
        XCTAssertFalse(backendAware.canEverFit(promptTokens: 2000, maxTokens: 0))
        XCTAssertThrowsError(try (backend as! PagedKVBackend).reserve(
            layerKinds: kinds, maxLength: 2000))
    }

    /// The ledger's paged charge is DERIVED from `PagedKVPool.pageDemand`,
    /// never restated, so admission cannot drift from the reservation the
    /// pool actually takes. Under-charging is a free-list underflow
    /// (`PagedKVGroup.allocatePage` traps — a daemon abort, not a rejection).
    func testPagedResidencyTracksPoolPageDemandExactly() {
        let config = gemmaPagedConfig
        let policy = CBv2PagedKVResidency(config: config)
        let kinds = [
            gemmaWindowedKind,
            CBv2LayerKind(attention: .slidingWindow(97), headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .full, headDim: 512, kvHeads: 2, queryHeads: 8),
        ]
        for kind in kinds {
            for tokens in [0, 1, 15, 16, 17, 500, 1023, 1024, 1552, 1553, 2000, 8192] {
                let expected =
                    PagedKVPool.pageDemand(kind: kind, maxLength: tokens, config: config)
                    * config.pageSize
                XCTAssertEqual(
                    policy.residentRows(layer: kind, tokens: tokens), expected,
                    "\(kind.attention) at \(tokens) tokens")
            }
        }
        XCTAssertEqual(policy.rowGranularity, config.pageSize)
    }

    /// A backend that does not declare `kvResidency` inherits the
    /// CONSERVATIVE policy: whole-ring charging, the status quo. A future
    /// backend that forgets must over-charge and under-admit, never
    /// over-commit the device.
    func testUndeclaredBackendResidencyDefaultsToWholeRingCharging() {
        let backend: any CBv2KVBackend = HarnessKVBackend()
        XCTAssertTrue(backend.kvResidency is CBv2ContiguousKVResidency)
        let kind = gemmaWindowedKind
        XCTAssertEqual(backend.kvResidency.residentRows(layer: kind, tokens: 1), 1024)
        XCTAssertEqual(backend.kvResidency.rowGranularity, 1)

        // And the paged backend answers for itself through the same seam.
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(1024), headDim: 64, kvHeads: 2, queryHeads: 4)
        ]
        guard
            let paged = try? PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 * 8192, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: 1024))
        else { return XCTFail("paged fixture must build") }
        let asProtocol: any CBv2KVBackend = paged
        XCTAssertTrue(
            asProtocol.kvResidency is CBv2PagedKVResidency,
            "the override must be a witness, not a static-dispatch shadow")
        XCTAssertEqual(asProtocol.kvResidency.residentRows(layer: kinds[0], tokens: 1), 16)
    }
}
