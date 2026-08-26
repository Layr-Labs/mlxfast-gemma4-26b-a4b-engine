// CBv2PagedPoolGuardTests.swift
//
// Track P (WS-0.5 / 0.6 / 1.3 / 6.4): the pool's construction guards, the
// reserved poison page's ACCOUNTING, the admission-charge residency
// identity, and the adaptive partition sizer.
//
// Deliberately disjoint from the poison-page tests in
// CBv2PagedBackendTests, which cover allocation and slab contents. This
// file covers the parts those cannot see: reservation arithmetic, free-list
// churn, and the charge/residency identity that WS-1.2's ring shrink will
// land on top of.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged pool guards")
struct CBv2PagedPoolGuardTests {

    private func fullKind(
        headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func windowedKind(_ window: Int) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func config(
        capacityBytes: Int = 8 << 20, maxPrefillChunk: Int = 64,
        nominalMaxLen: Int = 1024, prefixSharingBlockSize: Int? = nil
    ) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: capacityBytes, maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: nominalMaxLen,
            prefixSharingBlockSize: prefixSharingBlockSize)
    }

    private let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)

    // MARK: - WS-0.5: poison-page accounting

    /// The poison page is excluded from the RESERVATION ceiling, not merely
    /// from the free list. Reserving every usable page must succeed and must
    /// leave the poison page unreserved and unreachable; reserving one more
    /// must throw rather than let a row allocate into it.
    ///
    /// This is the half `poisonPageIsReservedAndNeverAllocatable` cannot
    /// see: that test drains the free list directly, which bypasses
    /// admission entirely. A pool that reserved `pageCount` pages while only
    /// `pageCount - 1` are allocatable would pass it and still underflow the
    /// free list on the last row's last page — a process abort under load.
    @Test func poisonPageIsExcludedFromTheReservationCeiling() throws {
        let pool = try PagedKVPool(layerKinds: [fullKind()], config: config())
        let usable = pool.usablePageCount(group: key)
        let physical = pool.group(key).pageCount
        #expect(usable == physical - 1)

        try pool.reserve([key: usable])
        #expect(throws: CBv2KVError.self) { try pool.reserve([key: 1]) }

        // Every reserved page must be materializable — this is exactly the
        // promise `reserve` makes to `PagedSequenceKV.ensurePage`.
        var pages: [Int32] = []
        for _ in 0 ..< usable {
            let page = pool.allocatePage(group: key)
            #expect(pool.isAllocatablePage(page, group: key))
            pages.append(page)
        }
        #expect(!pages.contains(pool.poisonPage(group: key)))
        #expect(Set(pages).count == usable, "a page was handed out twice")

        pool.freePages(group: key, pages: pages)
        pool.unreserve([key: usable])
        #expect(pool.bytesReserved == 0)
        #expect(pool.bytesInUse == 0)
    }

    /// Repeated allocate/free churn must never work the poison page into the
    /// free list. The free list is a LIFO stack that `freePage` appends to,
    /// so a single mis-accounted release would surface as the poison page
    /// being handed to the very next allocation.
    @Test func poisonPageSurvivesFreeListChurn() throws {
        let pool = try PagedKVPool(layerKinds: [fullKind()], config: config())
        let poison = pool.poisonPage(group: key)
        for _ in 0 ..< 64 {
            var batch: [Int32] = []
            for _ in 0 ..< 8 {
                let page = pool.allocatePage(group: key)
                #expect(page != poison)
                batch.append(page)
            }
            pool.freePages(group: key, pages: batch.reversed())
        }
        #expect(!pool.group(key).freeList.contains(poison))
        #expect(pool.group(key).refCounts[Int(poison)] == 1, "poison refcount must stay pinned")
        #expect(pool.bytesInUse == 0)
    }

    /// `bytesCapacity` is what capacity planning binds to. It must report
    /// USABLE bytes: counting the poison page would promise a page that
    /// admission can never hand out.
    @Test func capacityReportsUsableBytesOnly() throws {
        let pool = try PagedKVPool(layerKinds: [fullKind()], config: config())
        let g = pool.group(key)
        #expect(pool.bytesCapacity == (g.pageCount - 1) * g.pageBytes)
        #expect(pool.bytesCapacity <= 8 << 20, "usable capacity must stay inside the budget")
    }

    // MARK: - WS-3.2c: deferred frees

    /// Deferred frees hold their pages OUT of the free list until the drain,
    /// which is the entire point: a rolled-back page must not be recycled to
    /// another row while a speculative round's captures still name it.
    @Test func deferredFreesAreHeldUntilDrained() throws {
        let pool = try PagedKVPool(layerKinds: [fullKind()], config: config())
        let pages = (0 ..< 4).map { _ in pool.allocatePage(group: key) }
        let inUse = pool.bytesInUse

        pool.deferFreePages(group: key, pages: pages)
        #expect(pool.bytesInUse == inUse, "a deferred page is still held")
        for page in pages {
            #expect(!pool.group(key).freeList.contains(page), "deferred page re-entered the pool")
        }

        pool.drainDeferredFrees()
        #expect(pool.bytesInUse == 0)
        pool.drainDeferredFrees()  // idempotent on an empty queue
        #expect(pool.bytesInUse == 0)
    }

    // MARK: - WS-0.6: construction guards

    /// The block/page divisibility invariant is a property of two constants
    /// declared in files with no cross-reference. It must be asserted at
    /// construction, and it must hold for the shipped pair.
    @Test func blockSizeIsAWholeNumberOfPages() throws {
        #expect(CBv2BlockHasher.defaultBlockSize % CBv2PagedDefaults.pageSize == 0)
        // A page size that does not divide the block must be refused, and the
        // refusal must name the block size so the operator can act on it.
        // (24 also fails the PTOK guard; the point is that it is CATCHABLE
        // and never reaches MLX allocation.)
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: PagedKVPoolConfig(
                    pageSize: 24, capacityBytes: 8 << 20, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 1024))
        }
    }

    /// A pool that DECLARES it will share prefix blocks must be able to
    /// complete one in a chunk plus the frontier's partial page; a pool that
    /// declares nothing keeps working with any chunk size, because
    /// `maxPrefillChunk` is an operator/ITL knob and must never be able to
    /// fail engine build for a pool that never shares.
    @Test func prefixSharingBlockSizeArmsTheChunkCoverageGuard() throws {
        // Unarmed: a tiny chunk is fine.
        _ = try PagedKVPool(layerKinds: [fullKind()], config: config(maxPrefillChunk: 16))

        // Armed and satisfied: 512 + 16 covers a 256-token block.
        _ = try PagedKVPool(
            layerKinds: [fullKind()],
            config: config(
                maxPrefillChunk: 512,
                prefixSharingBlockSize: CBv2BlockHasher.defaultBlockSize))

        // Armed and violated: 64 + 16 cannot complete a 256-token block.
        do {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: config(
                    maxPrefillChunk: 64,
                    prefixSharingBlockSize: CBv2BlockHasher.defaultBlockSize))
            Issue.record("expected the chunk-coverage guard to refuse the pool")
        } catch CBv2KVError.backendIneligible(let reason) {
            #expect(reason.contains("maxPrefillChunk 64"), "reason: \(reason)")
        }

        // Armed with a block that is not a whole number of pages.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: config(maxPrefillChunk: 512, prefixSharingBlockSize: 100))
        }
    }

    /// The guards must stay CATCHABLE under hostile arithmetic — the chunk
    /// span is `maxPrefillChunk + pageSize` and `maxPrefillChunk` is
    /// operator-influenced, so the sum has to be overflow-checked rather
    /// than trapping in the runtime.
    @Test func armedGuardSurvivesOverflowingChunk() {
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: PagedKVPoolConfig(
                    capacityBytes: 1 << 20,
                    maxPrefillChunk: Int.max,
                    nominalMaxSequenceLength: 1024,
                    maxBufferLength: Int.max,
                    prefixSharingBlockSize: CBv2BlockHasher.defaultBlockSize))
        }
    }

    // MARK: - WS-1.3: the admission charge is the residency bound

    /// `pageDemand` is not a bound, it is an IDENTITY: for a row written
    /// from position 0 it equals the peak `table.count` exactly.
    ///
    /// This is the gate WS-1.2 lands on. Shrinking `ringPageCount` shrinks
    /// the charge through the `min`, and if the shrunk ring is ever smaller
    /// than what `ensurePage` actually touches, every row aborts on a
    /// free-list underflow under load instead of failing a request. Any
    /// change to either side that breaks the identity fails HERE, at desk
    /// speed, instead of in production.
    @Test(arguments: [8, 16, 32])
    func chargeEqualsPeakResidency(pageSize: Int) {
        for window in [32, 128, 1024] {
            for chunk in [16, 64, 512] {
                let cfg = PagedKVPoolConfig(
                    pageSize: pageSize, capacityBytes: 8 << 20, maxPrefillChunk: chunk,
                    nominalMaxSequenceLength: 4096)
                let ring = PagedKVPool.ringPageCount(window: window, config: cfg)
                for maxLength in [1, 2, 15, 16, 17, 255, 256, 257, 1536, 1552, 2000, 4096] {
                    // Peak table size = 1 + the highest ring slot the row's
                    // own writes reach (PagedSequenceKV.ensurePage).
                    var peak = 0
                    for position in 0 ..< maxLength {
                        peak = max(peak, (position / pageSize) % ring + 1)
                    }
                    let charge = PagedKVPool.pageDemand(
                        kind: CBv2LayerKind(
                            attention: .slidingWindow(window), headDim: 64, kvHeads: 2,
                            queryHeads: 4),
                        maxLength: maxLength, config: cfg)
                    #expect(
                        charge == peak,
                        """
                        windowed charge \(charge) != peak residency \(peak) \
                        (pageSize \(pageSize), window \(window), chunk \(chunk), \
                        maxLength \(maxLength), ring \(ring))
                        """)
                }
            }
        }
    }

    /// Full-attention rows hold every page they write, so the charge is the
    /// page span of `maxLength` with no ring to cap it.
    @Test func fullAttentionChargeIsThePageSpan() {
        let cfg = PagedKVPoolConfig(capacityBytes: 8 << 20, nominalMaxSequenceLength: 1024)
        for maxLength in [1, 16, 17, 32, 100, 4096] {
            #expect(
                PagedKVPool.pageDemand(kind: fullKind(), maxLength: maxLength, config: cfg)
                    == (maxLength + 15) / 16)
        }
    }

    /// A row must never be able to write past its charge. This is the
    /// contract `reserve` depends on, exercised end to end rather than
    /// arithmetically.
    @Test func windowedRowNeverExceedsItsCharge() throws {
        let cfg = config(capacityBytes: 32 << 20, maxPrefillChunk: 16, nominalMaxLen: 4096)
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let maxLength = 512
        let charge = PagedKVPool.pageDemand(
            kind: kinds[0], maxLength: maxLength, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: maxLength)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)

        var peak = 0
        while row.absoluteOffset + 16 <= maxLength {
            row.write(
                keys: MLXArray.zeros([2, 16, 64], dtype: .float16),
                values: MLXArray.zeros([2, 16, 64], dtype: .float16))
            peak = max(peak, row.table.count)
        }
        #expect(peak == charge, "peak residency \(peak) != charge \(charge)")
    }

    // MARK: - Gate G1: per-sequence KV footprint vs the contiguous backend

    /// Gemma-4-26b shape, as `CBv2SchedulerAdmissionTests` declares it:
    /// 25 sliding-window(1024) layers (head_dim 256, 8 kv heads) + 5 full
    /// layers (head_dim 512, 2 kv heads).
    private var gemma4Kinds: [CBv2LayerKind] {
        Array(
            repeating: CBv2LayerKind(
                attention: .slidingWindow(1024), headDim: 256, kvHeads: 8, queryHeads: 8),
            count: 25)
            + Array(
                repeating: CBv2LayerKind(
                    attention: .full, headDim: 512, kvHeads: 2, queryHeads: 8),
                count: 5)
    }

    /// The shipped per-sequence KV charge, through the shipped ledger.
    private func perSequenceKVBytes(
        residency: any CBv2KVResidencyPolicy, tokens: Int
    ) -> Int {
        AdmissionV2(
            layerKinds: gemma4Kinds, bytesCapacity: 1 << 50,
            config: .init(watermarkFraction: 0, elementBytes: 2),
            residency: residency
        ).allocatedBytes(forTokens: tokens)
    }

    /// GATE G1. Paged per-sequence KV must not exceed contiguous.
    ///
    /// It did: the 97-page ring held 1,552 tokens for a 1,024-token window,
    /// so every sliding layer over-committed by `ring - window == 528` tokens
    /// once a request outgrew the window. That is 25 of gemma-4's 30 layers,
    /// and it is why the gate failed at 10k while passing at 1k (the ring is
    /// not yet reached) and reading much better at 100k (the five full layers
    /// dominate the total).
    ///
    /// The 65-page ring holds 1,040 tokens, so the over-commit is 16 tokens —
    /// one page — and that is a FLOOR, not slack worth chasing:
    /// `ringPageCount` cannot go below `ceil((window + maxSpeculativeSpan) /
    /// pageSize)`, and `chargeEqualsPeakResidency` forbids charging less than
    /// the ring a row actually touches.
    ///
    /// Asserted in exact bytes rather than as a float ratio so a regression
    /// names the layer arithmetic instead of a rounded percentage.
    @Test(arguments: [1024, 10240, 102_400])
    func pagedPerSequenceKVDoesNotExceedContiguous(tokens: Int) throws {
        let cfg = PagedKVPoolConfig(
            capacityBytes: 8 << 30, maxPrefillChunk: 512, nominalMaxSequenceLength: tokens)
        let paged = perSequenceKVBytes(
            residency: CBv2PagedKVResidency(config: cfg), tokens: tokens)
        let contiguous = perSequenceKVBytes(
            residency: CBv2ContiguousKVResidency(), tokens: tokens)

        // The sliding layers are the whole story; state their two figures so
        // a failure says WHICH side moved.
        let ringTokens = PagedKVPool.ringPageCount(window: 1024, config: cfg) * cfg.pageSize
        #expect(ringTokens == 1040, "gemma-4's windowed ring is 65 pages == 1,040 tokens")

        // Paged may be at most one page per sliding layer over contiguous,
        // and only once the request outgrows the window.
        let slidingLayers = 25
        let slidingBytesPerToken = 2 * 8 * 256 * 2
        let allowance = tokens > 1024 ? slidingLayers * cfg.pageSize * slidingBytesPerToken : 0
        #expect(
            paged <= contiguous + allowance,
            """
            \(tokens) tokens: paged \(paged) B vs contiguous \(contiguous) B, over by \
            \(paged - contiguous) B against an allowance of \(allowance) B (one \
            \(cfg.pageSize)-token page on each of \(slidingLayers) sliding layers). The old \
            97-page ring was over by \(slidingLayers * (1552 - 1024) * slidingBytesPerToken) B \
            at this context.
            """)
        // ...and never UNDER, which would mean the charge stopped covering
        // what a row can touch — a free-list underflow, not a saving.
        #expect(
            paged >= contiguous,
            "\(tokens) tokens: paged \(paged) B is BELOW contiguous \(contiguous) B")

        // The exact figures the gate publishes, so a regression names the
        // arithmetic rather than a rounded percentage. Sliding layers cost
        // 2 * 8 * 256 * 2 == 8,192 B/token, full layers 2 * 2 * 512 * 2 ==
        // 4,096 B/token.
        let expected: [Int: (paged: Int, contiguous: Int)] = [
            1024: (230_686_720, 230_686_720),
            10240: (422_707_200, 419_430_400),
            102_400: (2_310_144_000, 2_306_867_200),
        ]
        let want = try #require(expected[tokens])
        #expect(paged == want.paged, "\(tokens) tokens: paged \(paged) B != \(want.paged) B")
        #expect(
            contiguous == want.contiguous,
            "\(tokens) tokens: contiguous \(contiguous) B != \(want.contiguous) B")

        // The BEFORE figure, measured rather than remembered. A 1,552-token
        // ring is what `ringPageCount` returns when the ROW bound is forced
        // to 1,552 — the same 97 pages the old cache bound
        // (`window - 1 + maxPrefillChunk + span`) produced at chunk 512.
        var oldCfg = cfg
        oldCfg.maxPrefillChunk = 1552
        #expect(PagedKVPool.ringPageCount(window: 1024, config: oldCfg) == 97)
        let before = perSequenceKVBytes(
            residency: CBv2PagedKVResidency(config: oldCfg), tokens: tokens)
        let expectedBefore: [Int: Int] = [
            1024: 230_686_720, 10240: 527_564_800, 102_400: 2_415_001_600,
        ]
        #expect(
            before == (try #require(expectedBefore[tokens])),
            "\(tokens) tokens: pre-shrink paged \(before) B")
        #expect(before >= paged, "the shrink must not have made the footprint larger")
    }

    // MARK: - WS-6.4: adaptive partition sizing

    /// Every value the sizer can return must be a page multiple inside
    /// `[minPartitionTokens, partitionTokens]`. `PTOK % pageSize == 0` is a
    /// kernel-launch precondition — an uncatchable trap — so this is
    /// swept rather than spot-checked.
    @Test func partitionSizerAlwaysReturnsAPageMultiple() {
        for pageSize in [8, 16, 32, 64, 256] {
            for batch in [1, 2, 4, 8, 32] {
                for kvHeads in [1, 2, 8, 16] {
                    for splits in [1, 2, 4] {
                        for length in [1, 7, 128, 512, 1024, 5000, 131_072] {
                            let ptok = PagedAttentionKernel.partitionTokensForDispatch(
                                maxAttendLength: length, batch: batch, kvHeads: kvHeads,
                                headSplits: splits, pageSize: pageSize)
                            #expect(ptok % pageSize == 0, "PTOK \(ptok) % pageSize \(pageSize)")
                            #expect(ptok >= pageSize)
                            #expect(ptok <= PagedAttentionKernel.partitionTokens)
                            #expect(
                                PagedAttentionKernel.partitionTokenLadder.contains(ptok)
                                    || ptok == PagedAttentionKernel.partitionTokens,
                                """
                                PTOK \(ptok) is off the ladder — every distinct value is a \
                                separate JIT variant, so the set must stay bounded
                                """)
                        }
                    }
                }
            }
        }
    }

    /// The point of WS-6.4: a short B == 1 decode must fill the GPU.
    ///
    /// GPT-OSS decode shape — 8 kv heads, GQA 8 (so `headsPerThreadgroup`
    /// keeps one threadgroup per kv head), B == 1. At 512 tokens the fixed
    /// 256-token partition gives `8 * 2 == 16` threadgroups on a 40-core
    /// GPU, which is where the B == 1 deficit comes from.
    ///
    /// The contract is exact: the sizer either reaches
    /// `partitionTargetThreadgroups` or it is sitting on the smallest rung
    /// with nothing left to give. It may never launch FEWER threadgroups
    /// than the fixed partition would.
    @Test(arguments: [256, 512, 1024, 2048, 4096])
    func partitionSizerFillsTheGpuAtShortContextB1(length: Int) {
        let kvHeads = 8
        let splits = 1
        let pageSize = 16
        func threadgroups(_ ptok: Int) -> Int {
            kvHeads * splits * ((length + ptok - 1) / ptok)
        }
        let fixed = threadgroups(PagedAttentionKernel.partitionTokens)
        let ptok = PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: length, batch: 1, kvHeads: kvHeads, headSplits: splits,
            pageSize: pageSize)
        let adaptive = threadgroups(ptok)
        #expect(ptok >= PagedAttentionKernel.minPartitionTokens)
        #expect(adaptive >= fixed, "length \(length): \(adaptive) < fixed \(fixed)")
        #expect(
            adaptive >= PagedAttentionKernel.partitionTargetThreadgroups
                || ptok == PagedAttentionKernel.minPartitionTokens,
            """
            length \(length): \(adaptive) threadgroups (PTOK \(ptok)) misses the \
            \(PagedAttentionKernel.partitionTargetThreadgroups) target without being \
            pinned at the floor
            """)
    }

    /// Where the GPU is already saturated the sizer must NOT shrink the
    /// partition: smaller partitions cost partial-buffer bytes
    /// (`[B, queryHeads, maxParts, headDim]` fp32) and merge work for
    /// occupancy that is already there. B == 8 is the target operating
    /// point, so a regression here would trade the batching win away.
    ///
    /// "Already saturated" is relative to `partitionTargetThreadgroups`,
    /// which is operator-settable: raising the knob above what the fixed
    /// partition launches is a REQUEST for the shrink, so the maximum-rung
    /// expectation is conditioned on the fixed dispatch actually meeting
    /// the configured target. The unconditional half — never fewer
    /// threadgroups than the fixed partition would launch — holds at every
    /// setting, including the kill switch, and keeps this non-vacuous when
    /// the target is raised past saturation.
    @Test func partitionSizerKeepsTheMaximumWhenAlreadySaturated() {
        let kvHeads = 8
        let splits = 1
        let batch = 8
        func threadgroups(_ ptok: Int, _ length: Int) -> Int {
            kvHeads * splits * batch * ((length + ptok - 1) / ptok)
        }
        for length in [512, 1024, 8192] {
            let ptok = PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: length, batch: batch, kvHeads: kvHeads,
                headSplits: splits, pageSize: 16)
            let fixed = threadgroups(PagedAttentionKernel.partitionTokens, length)
            #expect(
                threadgroups(ptok, length) >= fixed,
                "B=8 length \(length): PTOK \(ptok) launches fewer threadgroups than the fixed partition")
            guard fixed >= PagedAttentionKernel.partitionTargetThreadgroups else { continue }
            #expect(
                ptok == PagedAttentionKernel.partitionTokens,
                "B=8 length \(length) shrank PTOK to \(ptok)")
        }
    }

    /// The sizer must be a pure function of the dispatch shape: the kernel
    /// cache is keyed by PTOK, so a sizer that drifted between identical
    /// steps would re-JIT the pipeline on every decode.
    @Test func partitionSizerIsStableForAShape() {
        let first = PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: 777, batch: 3, kvHeads: 8, headSplits: 1, pageSize: 16)
        for _ in 0 ..< 16 {
            #expect(
                PagedAttentionKernel.partitionTokensForDispatch(
                    maxAttendLength: 777, batch: 3, kvHeads: 8, headSplits: 1, pageSize: 16)
                    == first)
        }
    }

    /// Wall-clock sweep of the decode dispatch at GPT-OSS shapes, used to
    /// pick `partitionTargetThreadgroups`. Off by default (it is a
    /// benchmark, not an assertion).
    ///
    /// PTOK is chosen inside `decode`, so the sweep varies it the way
    /// production would — through the env knob, one value per PROCESS:
    ///
    ///   for t in 0 128 256 512; do
    ///     DARKBLOOM_CBV2_PAGED_PTOK_BENCH=1 DARKBLOOM_CBV2_PAGED_PTOK_TARGET=$t \
    ///       swift test --filter partitionSizingBenchmark 2>&1 | grep ptok;
    ///   done
    ///
    /// `TARGET=0` is the kill switch and reproduces the pre-WS-6.4 fixed
    /// 256-token partition, so it is the baseline column.
    @Test func partitionSizingBenchmark() throws {
        guard
            ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PAGED_PTOK_BENCH"] == "1"
        else { return }
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        let kvHeads = 8
        let queryHeads = 64
        let headDim = 64
        let pageSize = 16
        let dtype = DType.float16
        let target = PagedAttentionKernel.partitionTargetThreadgroups

        for (batch, context) in [
            (1, 256), (1, 512), (1, 1024), (1, 4096),
            (2, 512), (4, 1024), (8, 1024), (8, 4096),
        ] {
            let pagesPerRow = (context + pageSize - 1) / pageSize
            let slabPages = max(8, batch * pagesPerRow)
            let kSlab = MLXArray.zeros([slabPages, kvHeads, pageSize, headDim], dtype: dtype)
            let vSlab = MLXArray.zeros([slabPages, kvHeads, pageSize, headDim], dtype: dtype)
            let fence = MLXArray.zeros([1], dtype: .int32)
            let maxPages = max(8, pagesPerRow)
            var flat = [Int32](repeating: 0, count: batch * maxPages)
            for row in 0 ..< batch {
                for page in 0 ..< pagesPerRow {
                    flat[row * maxPages + page] = Int32(row * pagesPerRow + page)
                }
            }
            let tables = MLXArray(flat, [batch, maxPages])
            let (seqinfo, _) = PagedAttentionKernel.seqinfo(
                (0 ..< batch).map { _ in
                    PagedAttentionKernel.SeqInfoRow(
                        attendStart: 0, attendLength: context, tableLength: pagesPerRow)
                })
            let params = MLXArray([Float](repeating: 0, count: 8))
            let queries = MLXArray.zeros([batch, queryHeads, headDim], dtype: dtype)
            eval(kSlab, vSlab, tables, seqinfo, params, queries, fence)

            func dispatch() -> MLXArray {
                PagedAttentionKernel.decode(
                    queries: queries, kSlab: kSlab, vSlab: vSlab, tables: tables,
                    seqinfo: seqinfo, maxAttendLength: context, sinks: nil, params: params,
                    softcap: false, pageSize: pageSize, writeFence: fence,
                    kernelSource: source
                ).out
            }
            for _ in 0 ..< 8 { eval(dispatch()) }  // JIT + warm the pipeline

            let iterations = 200
            let started = Date()
            for _ in 0 ..< iterations { eval(dispatch()) }
            let micros = Date().timeIntervalSince(started) * 1e6 / Double(iterations)

            let hpt = PagedAttentionKernel.headsPerThreadgroup(
                headDim: headDim, gqa: queryHeads / kvHeads)
            let splits = (queryHeads / kvHeads) / hpt
            let ptok = PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: context, batch: batch, kvHeads: kvHeads,
                headSplits: splits, pageSize: pageSize)
            let groups = kvHeads * splits * batch * ((context + ptok - 1) / ptok)
            print(
                String(
                    format:
                        "ptok target=%4d  B=%d ctx=%5d  PTOK=%3d  threadgroups=%4d  %8.1f us",
                    target, batch, context, ptok, groups, micros))
        }
    }
}
