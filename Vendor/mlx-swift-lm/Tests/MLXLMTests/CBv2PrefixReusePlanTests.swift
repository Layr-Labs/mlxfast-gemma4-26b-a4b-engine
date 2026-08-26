import XCTest

@testable import MLXLMCommon

final class CBv2FrozenReplayPlanTests: XCTestCase {
    func testFullyBidirectionalLayersDisablePrefixReuse() {
        let causal = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        var bidirectional = causal
        bidirectional.isBidirectional = true

        XCTAssertTrue(cbv2LayerKindsAllowPrefixReuse([causal]))
        XCTAssertFalse(cbv2LayerKindsAllowPrefixReuse([causal, bidirectional]))
    }

    private func full(sharedWith source: Int? = nil) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            sharesKVWithLayer: source,
            headDim: 8,
            kvHeads: 2,
            queryHeads: 4)
    }

    private func sliding(_ window: Int) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window),
            headDim: 8,
            kvHeads: 2,
            queryHeads: 4)
    }

    func testInterleavedHybridProducesExplicitFrozenPlan() throws {
        let kinds = [full(), sliding(16), sliding(16), full()]
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .contiguousUnquantized)
        XCTAssertTrue(capability.isSupported)
        XCTAssertEqual(capability.strategy, .frozenFullReplay)
        XCTAssertEqual(capability.conservativeReplayBoundTokens, 32)

        let plan = try XCTUnwrap(capability.plan(matchedBoundary: 72))
        XCTAssertEqual(plan.matchedBoundary, 72)
        XCTAssertEqual(plan.replayStart, 40)
        XCTAssertEqual(plan.replayTokens, 32)
        XCTAssertEqual(plan.prefillTokensSaved, 40)
        XCTAssertEqual(plan.restoredFullTokens, 72)
        XCTAssertEqual(plan.capacityReservationTokens, 72)
        XCTAssertEqual(plan.fullKVBytesPerToken, 128)
        XCTAssertEqual(plan.stagedFullKVBytes, 72 * 128)
        XCTAssertEqual(plan.residentFullKVBytes, 72 * 128)
        XCTAssertEqual(plan.clampedChunk(start: 32, proposed: 16), 8)
        XCTAssertEqual(plan.clampedChunk(start: 40, proposed: 40), 32)
        XCTAssertEqual(plan.clampedChunk(start: 72, proposed: 16), 16)

        let fp32Admission = AdmissionV2(
            layerKinds: kinds,
            bytesCapacity: 1 << 28,
            config: .init(
                watermarkFraction: 0,
                layerElementBytes: [4, 2, 2, 4]))
        XCTAssertEqual(fp32Admission.fullKVBytesPerToken, 256)
        let fp32WindowShortfall = try XCTUnwrap(
            fp32Admission.fixedWindowBytesShortfall(afterReservingTokens: 72))
        let nativeFP32 = try XCTUnwrap(
            capability.plan(
                matchedBoundary: 72,
                exactStagedFullKVBytes: 72 * 256,
                maximumSequenceLength: 200,
                nominalFullKVBytesPerToken: fp32Admission.fullKVBytesPerToken,
                fixedWindowCapacityBytes: fp32WindowShortfall))
        XCTAssertEqual(nativeFP32.fullKVBytesPerToken, 256)
        XCTAssertEqual(nativeFP32.nominalFullKVBytesPerToken, 256)
        XCTAssertEqual(nativeFP32.additionalFullKVBytesPerToken, 0)
        XCTAssertEqual(nativeFP32.fullCapacityTokensReserved, 200)
        XCTAssertEqual(
            nativeFP32.initialAdditionalCapacityBytes,
            (200 - 72) * 256)
        XCTAssertEqual(nativeFP32.stagedFullKVBytes, 72 * 256)
        XCTAssertEqual(nativeFP32.residentFullKVBytes, 72 * 256)
    }

    func testBackendSupportFailsColdForUnknownHybrid() {
        let kinds = [full(), sliding(16), full()]
        // WS-4.1: paged no longer fails cold on an interleaved hybrid. It
        // derives `.frozenFullReplay` like contiguous, and pays the SAME
        // conservative replay. It briefly paid one extra window, because
        // `PagedLayerCache.prefillKV` attended the chunk's freshly projected
        // keys where the contiguous frozen row hands back the cached ones, so
        // the first exact position moved by up to one prefill chunk.
        // `prefillKVWritingChunk` reads the cached diagonal out of the frozen
        // pages now, so the two bounds are one number — and this assertion is
        // what keeps them that way.
        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .pagedFP16)
        XCTAssertTrue(paged.isSupported)
        XCTAssertNil(paged.unsupportedReason)
        XCTAssertEqual(paged.strategy, .frozenFullReplay)
        XCTAssertEqual(paged.conservativeReplayBoundTokens, 16)
        XCTAssertEqual(
            CBv2PrefixReuseCapability.derive(
                layerKinds: kinds, backend: .contiguousUnquantized
            ).conservativeReplayBoundTokens,
            paged.conservativeReplayBoundTokens)

        let unknown = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .unknown)
        XCTAssertFalse(unknown.isSupported)
        XCTAssertEqual(unknown.unsupportedReason, .unknownBackend)

        let forwardShared = [
            full(sharedWith: 1),
            full(),
        ]
        XCTAssertEqual(
            CBv2PrefixReuseCapability.derive(
                layerKinds: forwardShared,
                backend: .contiguousUnquantized
            ).unsupportedReason,
            .invalidLayout)
        let mismatchedShared = [
            sliding(16),
            full(sharedWith: 0),
        ]
        XCTAssertEqual(
            CBv2PrefixReuseCapability.derive(
                layerKinds: mismatchedShared,
                backend: .contiguousUnquantized
            ).unsupportedReason,
            .invalidLayout)
    }

    func testMatchesBelowAtAndAboveReplayBoundFailOrAdoptExactly() {
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(16), sliding(16), full()],
            backend: .contiguousUnquantized)
        XCTAssertEqual(capability.conservativeReplayBoundTokens, 32)
        XCTAssertNil(capability.plan(matchedBoundary: 31))
        XCTAssertNil(capability.plan(matchedBoundary: 32))
        let above = capability.plan(matchedBoundary: 33)
        XCTAssertEqual(above?.replayStart, 1)
        XCTAssertEqual(above?.replayTokens, 32)
        XCTAssertEqual(above?.clampedChunk(start: 0, proposed: 2), 1)
        XCTAssertEqual(above?.clampedChunk(start: 1, proposed: 31), 31)
        XCTAssertEqual(above?.clampedChunk(start: 1, proposed: 33), 32)
        XCTAssertEqual(above?.clampedChunk(start: 33, proposed: 1), 1)
    }

    /// The two shapes Gate G2 measures, and where frozen-full reuse first
    /// fires on each.
    ///
    /// Paged used to grant `windowCount * maxWindow + maxWindow`, and that
    /// extra window cost twice: every adopted request replayed a window it
    /// did not need, AND the first boundary that could adopt at all moved out
    /// by the same amount. `plan` refuses at or below its own bound —
    /// `replayStart` would be 0 and the hit would save nothing — so the bound
    /// IS the floor, and narrowing it lowers the floor.
    ///
    /// `firstBoundary` is the first boundary `PrefixCacheV2` can actually
    /// return that clears the bound: lookups match whole hash blocks, so M is
    /// always a multiple of `CBv2BlockHasher.defaultBlockSize`.
    func testRealModelShapesShareTheContiguousReplayBoundAndItsFloor() throws {
        struct Shape {
            let name: String
            let kinds: [CBv2LayerKind]
            /// `windowCount * maxWindow` — what both backends grant now.
            let bound: Int
            /// `bound + maxWindow` — what paged granted before the narrowing.
            let previousBound: Int
            let firstBoundary: Int
            let savedNow: Int
            /// What that same boundary saved under `previousBound`; 0 means
            /// the adoption was REFUSED outright and the request cold-filled.
            let savedBefore: Int
        }
        let shapes = [
            // gemma-4-26B: 25 sliding layers of window 1,024 plus 5 full.
            Shape(
                name: "gemma-4-26B",
                kinds: (0 ..< 30).map { $0 < 25 ? sliding(1024) : full() },
                bound: 25600, previousBound: 26624,
                firstBoundary: 25856, savedNow: 256, savedBefore: 0),
            // gpt-oss-20b: 24 layers alternating sliding(128) / full.
            Shape(
                name: "gpt-oss-20b",
                kinds: (0 ..< 24).map { $0.isMultiple(of: 2) ? sliding(128) : full() },
                bound: 1536, previousBound: 1664,
                firstBoundary: 1792, savedNow: 256, savedBefore: 128),
        ]

        let block = CBv2BlockHasher.defaultBlockSize
        for shape in shapes {
            let paged = CBv2PrefixReuseCapability.derive(
                layerKinds: shape.kinds, backend: .pagedFP16)
            let contiguous = CBv2PrefixReuseCapability.derive(
                layerKinds: shape.kinds, backend: .contiguousUnquantized)
            XCTAssertEqual(paged.strategy, .frozenFullReplay, shape.name)
            XCTAssertEqual(
                paged.conservativeReplayBoundTokens, shape.bound, shape.name)
            XCTAssertEqual(
                paged.conservativeReplayBoundTokens,
                contiguous.conservativeReplayBoundTokens,
                "\(shape.name): both backends must pay ONE bound")

            // The floor, exactly: nothing at the bound, a plan one past it.
            XCTAssertNil(paged.plan(matchedBoundary: shape.bound), shape.name)
            let atFloor = try XCTUnwrap(
                paged.plan(matchedBoundary: shape.bound + 1), shape.name)
            XCTAssertEqual(atFloor.replayTokens, shape.bound, shape.name)
            XCTAssertEqual(atFloor.replayStart, 1, shape.name)

            // `firstBoundary` is the least block-aligned boundary above it.
            XCTAssertEqual(
                shape.firstBoundary, (shape.bound / block + 1) * block, shape.name)
            let first = try XCTUnwrap(
                paged.plan(matchedBoundary: shape.firstBoundary), shape.name)
            XCTAssertEqual(first.replayTokens, shape.bound, shape.name)
            XCTAssertEqual(first.prefillTokensSaved, shape.savedNow, shape.name)

            // And what that same boundary was worth before the narrowing,
            // derived from the old grant rather than asserted as a constant.
            XCTAssertEqual(
                shape.savedBefore,
                max(0, shape.firstBoundary - shape.previousBound),
                "\(shape.name): the recorded before/after saving must follow from the "
                    + "two bounds, not from a number typed into a test")
            XCTAssertEqual(
                shape.previousBound - shape.bound,
                shape.kinds.compactMap {
                    if case .slidingWindow(let w) = $0.attention { return w }
                    return nil
                }.max(),
                "\(shape.name): the term removed was exactly one maxWindow")
        }
    }

    /// The relation `PagedKVBackend.makeFrozenFullState` checks on every
    /// frozen adoption: the planner's grant must cover `cbv2RequiredRecompute`.
    ///
    /// It used to be able to FAIL. The backend added the pool's prefill chunk
    /// to its requirement, and the planner — which cannot see pool config —
    /// guessed `maxWindow` as a proxy, so any layout with `window < chunk`
    /// demanded more than it was granted and refused every adoption
    /// (gpt-oss-20b: 2,048 demanded against 1,664 granted). The requirement
    /// now reads layer shapes only, so grant and requirement are the same
    /// expression and the relation is EQUALITY, not slack. Pinned as equality
    /// on purpose: `>=` would silently tolerate a bound that grew back.
    ///
    /// Pool-chunk independence is checked where a pool exists, in
    /// `CBv2PrefixReusePagedFrozenFullTests.theReplayBoundIsIndependentOf
    /// ThePoolChunk`; here the point is that no chunk can enter at all.
    func testGrantEqualsTheAdoptionRequirementForEveryLayout() throws {
        for windowCount in [1, 2, 5, 12, 25] {
            for window in [8, 16, 128, 512, 1024] {
                // One owning full layer downstream is what selects
                // `.frozenFullReplay` over `.tailReplay`.
                let kinds = (0 ..< windowCount).map { _ in sliding(window) } + [full()]
                let bound = windowCount * window
                for backend in [CBv2PrefixReuseBackend.pagedFP16, .contiguousUnquantized] {
                    let label = "\(backend.rawValue) \(windowCount)x\(window)"
                    let capability = CBv2PrefixReuseCapability.derive(
                        layerKinds: kinds, backend: backend)
                    XCTAssertEqual(capability.strategy, .frozenFullReplay, label)
                    XCTAssertEqual(capability.conservativeReplayBoundTokens, bound, label)
                    for matched in [bound + 1, 2 * bound, 4 * bound] {
                        let plan = try XCTUnwrap(
                            capability.plan(matchedBoundary: matched), "\(label) M=\(matched)")
                        XCTAssertEqual(
                            plan.replayTokens,
                            cbv2RequiredRecompute(layerKinds: kinds, matched: matched),
                            "\(label) M=\(matched): granted and required must be one number — "
                                + "a shortfall refuses the adoption, a surplus is replay the "
                                + "request pays for and does not need")
                    }
                }
            }
        }
    }

    func testSafeLayoutsPreserveDirectAndTailReplay() throws {
        let direct = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), full()],
            backend: .pagedFP16)
        XCTAssertEqual(direct.strategy, .direct)
        let directPlan = try XCTUnwrap(direct.plan(matchedBoundary: 512))
        XCTAssertEqual(directPlan.replayStart, 512)
        XCTAssertEqual(directPlan.restoredFullTokens, 512)

        let tail = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(64), sliding(128)],
            backend: .pagedFP16)
        XCTAssertEqual(tail.strategy, .tailReplay)
        let tailPlan = try XCTUnwrap(tail.plan(matchedBoundary: 512))
        XCTAssertEqual(tailPlan.replayTokens, 256)
        XCTAssertEqual(tailPlan.replayStart, 256)
        XCTAssertEqual(tailPlan.restoredFullTokens, 256)
        XCTAssertEqual(tailPlan.capacityReservationTokens, 256)

        let shared = CBv2PrefixReuseCapability.derive(
            layerKinds: [full(), sliding(64), full(sharedWith: 0)],
            backend: .contiguousUnquantized)
        XCTAssertEqual(shared.strategy, .tailReplay)
    }

    func testFrozenReplayReservesMOnceAndRollbackBalancesExactly() throws {
        let kinds = [full(), sliding(16), sliding(16), full()]
        let plan = try XCTUnwrap(
            CBv2PrefixReuseCapability.derive(
                layerKinds: kinds,
                backend: .contiguousUnquantized
            ).plan(
                matchedBoundary: 72,
                exactStagedFullKVBytes: 72 * 256,
                maximumSequenceLength: 74))
        let admission = AdmissionV2(
            layerKinds: kinds,
            bytesCapacity: 1 << 28,
            config: .init(watermarkFraction: 0))
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 64,
                prefillChunkSize: 32),
            capacity: admission)
        let request = CBv2Request(
            id: CBv2RequestID(991),
            promptTokens: makePromptTokens(length: 73, seed: 991),
            sampling: .init(temperature: 0),
            maxTokens: 1)
        let rec = try scheduler.enqueue(request)
        rec.prefixReusePlan = plan
        rec.numComputedTokens = plan.replayStart
        try admission.reserve(
            id: rec.id,
            additionalTokens: plan.capacityReservationTokens,
            additionalBytes: plan.initialAdditionalCapacityBytes)
        let reservedAtM =
            admission.estimatedBytes(forTokens: plan.matchedBoundary)
            + plan.initialAdditionalCapacityBytes
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        let replay = scheduler.plan()
        XCTAssertEqual(replay.assignments.map(\.numTokens), [32])
        XCTAssertEqual(
            admission.bytesReserved,
            reservedAtM,
            "replay below M must consume the atomic M reservation, not add R")
        scheduler.rollback(replay)
        XCTAssertEqual(rec.numComputedTokens, plan.replayStart)
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        rec.numComputedTokens = plan.matchedBoundary
        let tail = scheduler.plan()
        XCTAssertEqual(tail.assignments.map(\.numTokens), [1])
        XCTAssertEqual(
            admission.bytesReserved,
            reservedAtM,
            "full native span and fixed sliding rings were prepaid at adoption")
        scheduler.rollback(tail)
        XCTAssertEqual(admission.bytesReserved, reservedAtM)

        XCTAssertTrue(scheduler.requeueOnCapacity(rec.id))
        XCTAssertNil(rec.prefixReusePlan)
        XCTAssertEqual(rec.numComputedTokens, 0)
        XCTAssertEqual(admission.bytesReserved, 0)
    }

    func testTailReplayPrepaysFullSlidingRingAndNativeFullSpan() throws {
        let kinds = [full(), sliding(16)]
        let admission = AdmissionV2(
            layerKinds: kinds,
            bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0))
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .contiguousUnquantized)
        let matched = 24
        let replayStart = 8
        let exactFullBytesPerToken = 64
        let plan = try XCTUnwrap(capability.plan(
            matchedBoundary: matched,
            exactStagedFullKVBytes: matched * exactFullBytesPerToken,
            maximumSequenceLength: 40,
            nominalFullKVBytesPerToken: admission.fullKVBytesPerToken))
        XCTAssertEqual(plan.strategy, .tailReplay)
        XCTAssertEqual(plan.replayStart, replayStart)
        // The fixed sliding ring is charged ONCE, inside the token
        // reservation (`AdmissionV2.allocatedBytes(forTokens:)`) — do not
        // route `fixedWindowBytesShortfall` through the plan as well.
        XCTAssertEqual(
            plan.initialAdditionalCapacityBytes,
            (40 - replayStart) * exactFullBytesPerToken)

        try admission.reserve(
            id: CBv2RequestID(777),
            additionalTokens: plan.capacityReservationTokens,
            additionalBytes: plan.initialAdditionalCapacityBytes)
        let exactBackendBytes =
            40 * exactFullBytesPerToken
            + 16 * 64
        XCTAssertEqual(admission.bytesReserved, exactBackendBytes)
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 32,
                prefillChunkSize: 16),
            capacity: admission)
        let rec = try scheduler.enqueue(CBv2Request(
            id: CBv2RequestID(777),
            promptTokens: makePromptTokens(length: 25, seed: 777),
            sampling: .init(temperature: 0),
            maxTokens: 4))
        rec.prefixReusePlan = plan
        rec.numComputedTokens = replayStart
        let replay = scheduler.plan()
        XCTAssertEqual(replay.assignments.map(\.numTokens), [16])
        XCTAssertEqual(admission.bytesReserved, exactBackendBytes)
        let postBoundary = scheduler.plan()
        XCTAssertEqual(postBoundary.assignments.map(\.numTokens), [1])
        XCTAssertEqual(
            admission.bytesReserved,
            exactBackendBytes,
            "prepaid full span and fixed ring must not be charged again")
        scheduler.rollback(postBoundary)
        scheduler.rollback(replay)
        admission.releaseAll(id: CBv2RequestID(777))
        XCTAssertEqual(admission.bytesReserved, 0)
    }

    func testFixedWindowShortfallOverflowFailsCold() {
        let admission = AdmissionV2(
            layerKinds: [
                CBv2LayerKind(
                    attention: .slidingWindow(Int.max),
                    headDim: 1,
                    kvHeads: 1,
                    queryHeads: 1)
            ],
            bytesCapacity: Int.max,
            config: .init(watermarkFraction: 0))
        XCTAssertNil(admission.fixedWindowBytesShortfall(afterReservingTokens: 0))
    }
}
