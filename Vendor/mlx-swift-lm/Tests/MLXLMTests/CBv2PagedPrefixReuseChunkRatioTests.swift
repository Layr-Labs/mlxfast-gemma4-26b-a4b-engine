// CBv2PagedPrefixReuseChunkRatioTests.swift
//
// Paged frozen-full replay when the pool's prefill chunk is LARGER than the
// model's sliding window.
//
// `CBv2PrefixReusePagedFrozenFullTests` covers the opposite ratio (window 16
// against chunk 8) and every other paged reuse suite inherits that fixture,
// so the chunk > window regime had no coverage at all. It is not a corner:
// gpt-oss-20b is 12 sliding layers of window 128 against the default
// 512-token chunk, and it is the model where prefix reuse matters most,
// because its replay bound is 1,536 tokens rather than gemma-4's 25,600 —
// reuse fires at a prompt length an operator actually sends.
//
// WHAT THE GAP COST, measured by the Gate G2 parity harness on real weights:
// paged returned `adoption_failed` with saved = 0 on a 28,672-token prompt
// where contiguous saved 26,880, after BOTH arms matched 28,416 tokens. The
// lookup hit and the plan was derived; only the adoption refused.
//
// The refusal was CORRECT — it was refusing an under-provisioned plan.
// `PagedLayerCache.prefillKV` used to assemble
// `gather([baseOffset, queryStart)) ++ chunk` with the chunk half being the
// freshly projected K/V, so a frozen paged row contributed exact keys before
// the current chunk and poisoned ones inside it. That cost paged one extra
// `maxPrefillChunk` of replay over contiguous, the planner could not see pool
// config so it paid `maxWindow` as a proxy, and the proxy is only sufficient
// when `maxPrefillChunk <= maxWindow`. gemma-4 satisfied that (512 against
// 1,024) and gpt-oss did not (512 against 128), so the backend was handed
// 1,664 tokens of replay where it demanded 2,048 and refused.
//
// The chunk term is now GONE rather than capped: the gather is hoisted, so
// paged's bound EQUALS contiguous's `windowCount * maxWindow` and
// `cbv2RequiredRecompute(layerKinds:matched:)` is the single shared
// definition, reading layer shapes and nothing else. gpt-oss went from
// saved = 0 to saved = 26,880 — full parity with contiguous, not a shortfall.
//
// This suite pins the two things that can silently come back:
//
//   1. the bound must stay INDEPENDENT of `maxPrefillChunk` and identical
//      across backends. Any reintroduced chunk term or paged-specific pad
//      re-creates the gpt-oss refusal, and it re-creates it as a
//      configuration-dependent bug that a gemma-4-shaped fixture cannot see;
//   2. adoption in the chunk > window ratio must be accepted AND token-exact.
//      Widening acceptance without exactness trades the frozen-full defect
//      back for a throughput number — the windows come back full-length but
//      built from too few keys, which nothing downstream can detect.
//
// (2) is the one with teeth. A test that only asserts acceptance passes for
// a "fix" that merely lowers the bound, which is precisely the unsafe change.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2PrefixReuse: paged frozen replay with chunk > window")
struct CBv2PagedPrefixReuseChunkRatioTests {

    // MARK: - Fixture

    /// Window 8 against a 32-token chunk: the same 1:4 ratio gpt-oss-20b runs
    /// (128 against 512), small enough to decode end to end.
    ///
    /// `TinyTestModel.make(stackedSlidingFull:)` yields
    /// `[full, sliding, sliding, full]` — an owning full layer downstream of
    /// the windowed ones, which is what selects `.frozenFullReplay`.
    private static let window = 8
    private static let chunkSize = 32
    private static let maxLength = 256
    private static let matched = 64
    private static let slidingLayers = 2

    /// `windowCount * maxWindow`, with no chunk term and no paged pad.
    private static let expectedReplay = slidingLayers * window  // 16

    private func makeModel() -> TinyTestModel {
        TinyTestModel.make(
            seed: 0x9F17_C0DE, headDim: 64, stackedSlidingFull: true,
            windowSize: Self.window)
    }

    private func makeBackend(_ kinds: [CBv2LayerKind]) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 64 << 20,
                maxPrefillChunk: Self.chunkSize,
                nominalMaxSequenceLength: Self.maxLength))
    }

    /// Prefill `prompt[from ..< upTo]`, honouring the plan's chunk clamp
    /// exactly as `SchedulerV2` does — it still splits at C and M, which a
    /// frozen row requires because storage below M is immutable and above it
    /// is appendable, and one write cannot do both.
    @discardableResult
    private func prefill(
        model: TinyTestModel, backend: PagedKVBackend, state: [CBv2SequenceKV?],
        prompt: [Int], from: Int, upTo: Int, plan: CBv2PrefixReusePlan? = nil
    ) -> [CBv2AttendingLayerCache] {
        let caches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
        for (i, kind) in model.layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            caches[i].setRows([state[i]!])
        }
        var index = from
        while index < upTo {
            var count = min(Self.chunkSize, upTo - index)
            if let plan { count = plan.clampedChunk(start: index, proposed: count) }
            let slice = Array(prompt[index ..< (index + count)])
            _ = model.forward(
                tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                caches: caches)
            index += slice.count
        }
        return caches
    }

    /// Prefill then greedy-decode. Greedy is the amplifier: a window short by
    /// one key perturbs the logits, one argmax flips, and the trajectories
    /// separate permanently.
    private func run(
        model: TinyTestModel, backend: PagedKVBackend, state: [CBv2SequenceKV?],
        prompt: [Int], from: Int, steps: Int, plan: CBv2PrefixReusePlan? = nil
    ) -> [Int] {
        let caches = prefill(
            model: model, backend: backend, state: state, prompt: prompt,
            from: from, upTo: prompt.count - 1, plan: plan)
        var current = prompt.last!
        var generated: [Int] = []
        for _ in 0 ..< steps {
            let logits = model.forward(
                tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
            current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
            generated.append(current)
        }
        return generated
    }

    /// Donor arm: a cold prefill of `prompt[0 ..< matched)` snapshotted per
    /// layer. The replay form recomputes windowed layers, so only the full
    /// rows are donated — exactly what `PrefixCacheV2` hands over.
    private func donateFullRows(
        model: TinyTestModel, backend: PagedKVBackend, prompt: [Int], matched: Int
    ) throws -> [(keys: MLXArray, values: MLXArray, offset: Int)?] {
        let kinds = model.layerKinds
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: matched, maxLength: Self.maxLength)
        prefill(
            model: model, backend: backend, state: state, prompt: prompt,
            from: 0, upTo: matched)
        let prefix = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard kind.sharesKVWithLayer == nil, let row = state[index],
                case .full = kind.attention
            else { return nil }
            let snapshot = row.snapshot()
            // Paged snapshots are lazy views over the SHARED slabs; the
            // donor's pages are recycled the instant its state is released.
            eval(snapshot.keys, snapshot.values)
            return snapshot
        }
        backend.release(state)
        return prefix
    }

    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    /// gpt-oss-20b as `config.json` declares it: 24 layers alternating
    /// `sliding_attention` (window 128) and `full_attention`, head_dim 64,
    /// 8 KV heads.
    private func gptOssLayerKinds() -> [CBv2LayerKind] {
        (0 ..< 24).map { index in
            CBv2LayerKind(
                attention: index.isMultiple(of: 2) ? .slidingWindow(128) : .full,
                headDim: 64,
                kvHeads: 8,
                queryHeads: 64)
        }
    }

    /// gemma-4-26B: 25 sliding layers of window 1,024, then 5 full.
    private func gemma4LayerKinds() -> [CBv2LayerKind] {
        (0 ..< 30).map { index in
            CBv2LayerKind(
                attention: index < 25 ? .slidingWindow(1024) : .full,
                headDim: index < 25 ? 256 : 512,
                kvHeads: 4,
                queryHeads: 8)
        }
    }

    // MARK: - 1. The bound, at real production geometry

    /// The arithmetic that produced `saved = 0`, pinned at the real geometry
    /// it produced it on — and now pinned at parity.
    ///
    /// 26,880 is the number the G2 harness measures on contiguous. Paged
    /// reporting anything less than that is the regression this test exists
    /// to catch; paged reporting 0 is the bug it was written for.
    @Test func gptOssGeometryReusesAsMuchAsContiguous() throws {
        let kinds = gptOssLayerKinds()
        #expect(kinds.filter { $0.attention == .slidingWindow(128) }.count == 12)

        let contiguous = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)
        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        #expect(contiguous.strategy == .frozenFullReplay)
        #expect(paged.strategy == .frozenFullReplay)
        #expect(contiguous.conservativeReplayBoundTokens == 1536, "12 sliding layers x 128")
        #expect(
            paged.conservativeReplayBoundTokens == 1536,
            "paged pays no pad over contiguous once the chunk term is gone")

        // The boundary the G2 parity harness actually matched on both arms.
        let plan = try #require(
            paged.plan(matchedBoundary: 28416, maximumSequenceLength: 28672))
        #expect(plan.replayTokens == 1536)
        #expect(plan.replayStart == 26880)
        #expect(
            plan.prefillTokensSaved == 26880,
            """
            the harness measures 26880 on contiguous; paged must match it, and it \
            returned 0 while the bound carried a maxPrefillChunk term
            """)

        // The backend's own demand, which is what refuses an adoption. It
        // reads layer shapes only — no pool config reaches it.
        #expect(
            cbv2RequiredRecompute(layerKinds: kinds, matched: plan.matchedBoundary)
                <= plan.replayTokens,
            "a plan the backend will refuse must never be emitted")
    }

    // MARK: - 2. The bound is independent of the pool chunk

    /// The invariant that now carries the gpt-oss defect.
    ///
    /// The refusal was configuration-dependent: identical model, identical
    /// match, and the adoption succeeded or failed purely on the pool's
    /// `maxPrefillChunk` relative to the window. So the property worth
    /// pinning is not a numeric bound but an INDEPENDENCE — the bound must
    /// not be a function of the chunk, and paged must not carry a pad that
    /// contiguous does not.
    ///
    /// `chunk` is swept as a deliberately inert dimension. The day it stops
    /// being inert, this fails for every shape whose window is under it,
    /// which is exactly the population that shipped broken last time.
    @Test func theReplayBoundIsIndependentOfThePoolChunk() throws {
        for windowCount in [1, 2, 5, 12, 25] {
            for window in [8, 16, 128, 512, 1024] {
                var kinds = (0 ..< windowCount).map { _ in
                    CBv2LayerKind(
                        attention: .slidingWindow(window),
                        headDim: 64, kvHeads: 2, queryHeads: 4)
                }
                // One owning full layer downstream: what selects
                // `.frozenFullReplay` over `.tailReplay`.
                kinds.append(
                    CBv2LayerKind(
                        attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4))

                let expected = windowCount * window
                let contiguous = CBv2PrefixReuseCapability.derive(
                    layerKinds: kinds, backend: .contiguousUnquantized)
                let paged = CBv2PrefixReuseCapability.derive(
                    layerKinds: kinds, backend: .pagedFP16)
                #expect(paged.strategy == .frozenFullReplay)
                #expect(
                    paged.conservativeReplayBoundTokens == expected,
                    """
                    \(windowCount) x window \(window): paged bound must be \
                    windowCount x maxWindow with no pad
                    """)
                #expect(
                    paged.conservativeReplayBoundTokens
                        == contiguous.conservativeReplayBoundTokens,
                    """
                    \(windowCount) x window \(window): paged must not cost more \
                    replay than contiguous
                    """)

                let plan = try #require(
                    paged.plan(
                        matchedBoundary: 2 * expected,
                        maximumSequenceLength: 4 * expected))
                #expect(plan.replayTokens == expected)

                // A pool of ANY chunk must be servable by this one plan.
                // `cbv2RequiredRecompute` takes no chunk, so the loop proves
                // the demand cannot move with one.
                let demanded = cbv2RequiredRecompute(
                    layerKinds: kinds, matched: plan.matchedBoundary)
                for chunk in [8, 16, 128, 512, 1024, 2048] {
                    #expect(
                        plan.replayTokens >= demanded,
                        """
                        \(windowCount) sliding layers x window \(window), pool chunk \
                        \(chunk): granted \(plan.replayTokens) but the backend demands \
                        \(demanded) — adoption refuses and the hit saves nothing
                        """)
                }
            }
        }
    }

    /// Both shipping models, at the boundary the harness matches, with the
    /// saved-token counts the gate reports.
    @Test func shippingModelsSaveWhatContiguousSaves() throws {
        for (name, kinds, bound, saved) in [
            ("gpt-oss-20b", gptOssLayerKinds(), 1536, 26880),
            ("gemma-4-26B", gemma4LayerKinds(), 25600, 2816),
        ] {
            let paged = CBv2PrefixReuseCapability.derive(
                layerKinds: kinds, backend: .pagedFP16)
            let contiguous = CBv2PrefixReuseCapability.derive(
                layerKinds: kinds, backend: .contiguousUnquantized)
            #expect(paged.conservativeReplayBoundTokens == bound, "\(name)")
            #expect(contiguous.conservativeReplayBoundTokens == bound, "\(name)")

            let plan = try #require(
                paged.plan(matchedBoundary: 28416, maximumSequenceLength: 28672))
            #expect(plan.prefillTokensSaved == saved, "\(name)")

            // `clampedChunk` no longer caps — it only splits at C and M, and
            // that split is still mandatory: a frozen row's storage is
            // immutable below M and appendable above it, so one write can
            // never straddle.
            #expect(plan.clampedChunk(start: plan.replayStart, proposed: 512) == 512)
            #expect(
                plan.clampedChunk(start: plan.matchedBoundary - 1, proposed: 512) == 1,
                "\(name): a chunk must not cross M")
            #expect(plan.clampedChunk(start: plan.matchedBoundary, proposed: 512) == 512)
        }
    }

    // MARK: - 3. Adoption is accepted in this ratio

    /// The regression. While the bound carried a chunk term, `derive` granted
    /// `2 * 8 + 8 == 24` replay tokens and the backend demanded
    /// `2 * 8 + 32 == 48`, so this threw `backendIneligible` and the engine
    /// cold-prefilled — the unit-scope shape of gpt-oss's `adoption_failed`.
    @Test func adoptionIsAcceptedWhenTheChunkExceedsTheWindow() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x1CE_B00D)

        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(
                matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        #expect(plan.replayTokens == Self.expectedReplay)
        #expect(plan.replayStart == Self.matched - Self.expectedReplay)

        let backend = try makeBackend(kinds)
        let prefix = try donateFullRows(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)

        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)

        // Every row on the same logical cursor C, full rows frozen through M.
        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            #expect(row.absoluteOffset == plan.replayStart, "layer \(index)")
            switch kind.attention {
            case .full:
                #expect(row.frozenHighWater == Self.matched)
            case .slidingWindow:
                #expect(row.frozenHighWater == 0)
                #expect(row.retainedCount == 0, "sliding rows rebuild from C")
            }
        }
        backend.release(state)
    }

    // MARK: - 4. And it is exact

    /// The safety half, and the reason the narrowed bound needs coverage in
    /// THIS ratio specifically. Accepting the adoption is only correct if the
    /// sliding rows come back bit-exact; a bound that is too short reproduces
    /// the frozen-full defect, where the windows are full-length but built
    /// from too few keys and nothing downstream can see it.
    ///
    /// Decoded over three full window turnovers so a single missing key
    /// cannot hide inside the prompt.
    @Test func frozenReplayIsTokenExactWhenTheChunkExceedsTheWindow() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x1CE_B00D)
        let steps = 3 * Self.window

        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let cold = run(
            model: model, backend: coldBackend, state: coldState, prompt: prompt,
            from: 0, steps: steps)
        coldBackend.release(coldState)

        let backend = try makeBackend(kinds)
        let prefix = try donateFullRows(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(
                matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)

        let adopted = run(
            model: model, backend: backend, state: state, prompt: prompt,
            from: plan.replayStart, steps: steps, plan: plan)

        // The frozen rows must still hold the donated bytes at [0, M) after
        // the replay wrote straight through them.
        for (index, kind) in kinds.enumerated() {
            guard case .full = kind.attention else { continue }
            let row = try #require(state[index] as? PagedSequenceKV)
            let original = try #require(prefix[index])
            let (keys, values) = row.gatherRange(start: 0, count: Self.matched)
            #expect(
                maxAbsDiff(keys, original.keys) == 0,
                "layer \(index) frozen keys were overwritten by the replay")
            #expect(maxAbsDiff(values, original.values) == 0)
        }
        backend.release(state)

        let agreed = zip(adopted, cold).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adopted == cold,
            """
            frozen replay with chunk \(Self.chunkSize) > window \(Self.window) diverged \
            from a cold twin (\(agreed) of \(steps) tokens matched before divergence) — \
            the replay bound is too short for this ratio and the sliding rows came back \
            inexact
            """)
    }
}
