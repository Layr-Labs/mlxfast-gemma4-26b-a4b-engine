// CBv2PagedPrefixReuseBoundNarrowingTests.swift
//
// The consequence of dropping paged's `+ maxWindow`: a frozen-full adoption
// FIRES at a matched boundary that used to be refused outright, and is exact
// when it does.
//
// `CBv2PrefixReuseCapability.derive` granted `.pagedFP16` one extra window of
// conservative replay on top of `windowCount * maxWindow`, because a frozen
// paged row attended its chunk's freshly projected keys — poisoned, during a
// replay — where `CBv2FrozenReplayFullSequenceKV` hands back the cached ones.
// `PagedLayerCache.prefillKVWritingChunk` now reads the cached diagonal out of
// the frozen pages (proven bit-identical to a cold twin across five chunk
// alignments and under query blocking in `CBv2PagedFrozenChunkGather`), so the
// term is gone and both backends pay one bound.
//
// The extra window was not merely wasted replay. `plan` refuses at or below
// its own bound — `replayStart` would be 0 and the hit would save nothing —
// so the bound is also the FLOOR on the matched boundary at which reuse can
// fire at all. Paged's floor sat one window above contiguous's, and every
// boundary in between was a confirmed cache hit that the paged backend threw
// away. This suite pins the boundary in that band.
//
// Two properties, and the second is the one that makes the first safe:
//
//   1. at a boundary the OLD bound refused, paged now adopts, and the adopted
//      row is bit-exact against a cold twin — the REBUILT SLIDING WINDOWS
//      compared with `arrayEqual`, not just the tokens. Token equality is the
//      user-visible property but a weak instrument for the bound on this
//      fixture (P4_FrozenChunkGather measured that halving R still yields
//      identical tokens here); the windows discriminate.
//   2. the narrowing is CONFINED to the frozen-full branch. Non-frozen
//      layouts are untouched, and the clamp a frozen plan imposes is inert
//      above M, so the ordinary prefill of `[M, promptLength)` — where the
//      tokens actually are — is computed exactly as an unadopted one.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedPrefixReuse: the narrowed frozen-replay bound")
struct CBv2PagedPrefixReuseBoundNarrowingTests {

    // MARK: - Fixture

    /// `[full, sliding(16), sliding(16), full]` — an owning full layer after
    /// windowed ones, which is what selects `.frozenFullReplay`.
    private static let window = 16
    private static let slidingLayers = 2
    private static let chunkSize = 8
    private static let maxLength = 256

    /// `windowCount * maxWindow`. Both backends grant this.
    private static var bound: Int { slidingLayers * window }
    /// `bound + maxWindow`. What `.pagedFP16` granted before the narrowing.
    private static var previousBound: Int { bound + window }

    /// A boundary INSIDE the band the extra window used to sterilise:
    /// above the shared bound, at or below the old paged grant. Reuse fires
    /// here now and refused here before.
    private static let matched = 48

    private func makeModel() -> TinyTestModel {
        TinyTestModel.make(
            seed: 0x8B0D_1CE5, headDim: 64, stackedSlidingFull: true,
            windowSize: Self.window)
    }

    private func makeBackend(_ kinds: [CBv2LayerKind], chunk: Int = Self.chunkSize) throws
        -> PagedKVBackend
    {
        try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 64 << 20,
                maxPrefillChunk: chunk,
                nominalMaxSequenceLength: Self.maxLength))
    }

    private func sliding(_ window: Int) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func full() -> CBv2LayerKind {
        CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    /// Prefill `prompt[from ..< upTo]` through `caches`, honouring the plan's
    /// boundary split exactly as `SchedulerV2` does — a frozen chunk must
    /// never straddle M, or `PagedSequenceKV.write` traps.
    private func prefill(
        model: TinyTestModel, caches: [CBv2AttendingLayerCache], prompt: [Int],
        from: Int, upTo: Int, chunk: Int = Self.chunkSize, plan: CBv2PrefixReusePlan? = nil
    ) {
        var index = from
        while index < upTo {
            var count = min(chunk, upTo - index)
            if let plan { count = plan.clampedChunk(start: index, proposed: count) }
            let slice = Array(prompt[index ..< (index + count)])
            _ = model.forward(
                tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                caches: caches)
            index += slice.count
        }
    }

    private func caches(
        _ backend: PagedKVBackend, _ kinds: [CBv2LayerKind], _ state: [CBv2SequenceKV?]
    ) -> [CBv2AttendingLayerCache] {
        let caches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
        for (i, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
            caches[i].setRows([state[i]!])
        }
        return caches
    }

    /// Greedy decode. A window short by one key perturbs the logits, one
    /// argmax flips, and the trajectories separate permanently.
    private func decode(
        model: TinyTestModel, caches: [CBv2AttendingLayerCache], from token: Int, steps: Int
    ) -> [Int] {
        var current = token
        var generated: [Int] = []
        for _ in 0 ..< steps {
            let logits = model.forward(
                tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
            current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
            generated.append(current)
        }
        return generated
    }

    /// A cold prefill of `[0, matched)` with the owning FULL rows snapshotted,
    /// which is what `PrefixCacheV2` donates: the replay form recomputes every
    /// windowed layer, so a window payload here would be a contradiction.
    private func donateFullRows(
        model: TinyTestModel, backend: PagedKVBackend, prompt: [Int], matched: Int
    ) throws -> [(keys: MLXArray, values: MLXArray, offset: Int)?] {
        let kinds = model.layerKinds
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: matched, maxLength: Self.maxLength)
        prefill(
            model: model, caches: caches(backend, kinds, state), prompt: prompt,
            from: 0, upTo: matched)
        let donated = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard kind.sharesKVWithLayer == nil, case .full = kind.attention,
                let row = state[index]
            else { return nil }
            let snapshot = row.snapshot()
            // Paged snapshots are lazy views over the SHARED slabs; the
            // donor's pages are recycled the instant its state is released.
            eval(snapshot.keys, snapshot.values)
            return snapshot
        }
        backend.release(state)
        return donated
    }

    private func assertIdentical(
        _ got: MLXArray, _ want: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, "\(what): shape", sourceLocation: sourceLocation)
        guard got.shape == want.shape else { return }
        let delta = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        #expect(
            arrayEqual(got, want).item(Bool.self),
            "\(what) are not bit-identical: max |delta| = \(delta)",
            sourceLocation: sourceLocation)
    }

    // MARK: - 1. The boundary the old bound refused

    /// The headline, and the whole point of the narrowing.
    ///
    /// M = 48 sits above the shared bound (32) and at the old paged grant
    /// (48). Paged produced NO PLAN here — `replayStart` was 0 — so a
    /// confirmed cache hit was discarded and the request cold-prefilled,
    /// while contiguous served the same match. It now adopts, replays R = 32
    /// from C = 16, and saves 16 tokens.
    ///
    /// "Previously refused" is asserted with live code rather than as a
    /// historical claim: a layout whose CURRENT bound is the old paged grant
    /// still returns nil at this very boundary, which is the planner rule
    /// that produced the refusal.
    @Test func adoptionFiresAtABoundaryTheExtraWindowUsedToRefuse() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        #expect(kinds.count == 4, "shape must be [full, sliding, sliding, full]")
        let prompt = makePromptTokens(length: 96, seed: 0x2B0D_4C11)
        let steps = 3 * Self.window

        let paged = CBv2PrefixReuseCapability.derive(layerKinds: kinds, backend: .pagedFP16)
        let contiguous = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)
        #expect(paged.conservativeReplayBoundTokens == Self.bound)
        #expect(
            paged.conservativeReplayBoundTokens
                == contiguous.conservativeReplayBoundTokens,
            "the narrowing's premise: one bound for both backends")

        // The band. M is above the shared bound and at or below the old one.
        #expect(Self.matched > Self.bound)
        #expect(Self.matched <= Self.previousBound)

        // The refusal, demonstrated rather than remembered: three sliding
        // layers make the CURRENT bound 48, which is the old paged grant, and
        // a bound of 48 still refuses M = 48 today.
        let oldBoundLayout = [full()] + Array(repeating: sliding(Self.window), count: 3) + [full()]
        let oldBoundCapability = CBv2PrefixReuseCapability.derive(
            layerKinds: oldBoundLayout, backend: .pagedFP16)
        #expect(oldBoundCapability.conservativeReplayBoundTokens == Self.previousBound)
        #expect(
            oldBoundCapability.plan(
                matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength) == nil,
            """
            a bound of \(Self.previousBound) saves nothing at M = \(Self.matched) — this \
            is exactly what paged did at this boundary before the narrowing
            """)

        // And with the narrowed bound it is a real plan.
        let plan = try #require(
            paged.plan(matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        #expect(!plan.requiresExactWindowRestore)
        #expect(plan.replayTokens == Self.bound)
        #expect(plan.replayStart == Self.matched - Self.bound)
        #expect(plan.prefillTokensSaved == Self.matched - Self.bound)
        #expect(
            plan.replayTokens
                >= cbv2RequiredRecompute(layerKinds: kinds, matched: Self.matched),
            "the grant must clear the shared cone bound the backend guard checks")

        // --- Cold arm: no adoption at all, straight through.
        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let coldCaches = caches(coldBackend, kinds, coldState)
        prefill(
            model: model, caches: coldCaches, prompt: prompt, from: 0, upTo: prompt.count - 1)
        let coldTokens = decode(
            model: model, caches: coldCaches, from: prompt.last!, steps: steps)
        coldBackend.release(coldState)

        // --- Reference windows: a cold row stopped exactly at M, which is
        //     what the replay must reproduce.
        let referenceBackend = try makeBackend(kinds)
        let referenceState = try referenceBackend.makeSequenceState(
            layerKinds: kinds, promptLength: Self.matched, maxLength: Self.maxLength)
        prefill(
            model: model, caches: caches(referenceBackend, kinds, referenceState),
            prompt: prompt, from: 0, upTo: Self.matched)

        // --- Adopted arm.
        let backend = try makeBackend(kinds)
        let prefix = try donateFullRows(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)
        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)
        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            #expect(row.absoluteOffset == plan.replayStart, "layer \(index) starts at C")
            switch kind.attention {
            case .full:
                #expect(row.frozenHighWater == Self.matched)
            case .slidingWindow:
                #expect(row.frozenHighWater == 0)
                #expect(row.retainedCount == 0, "sliding rows rebuild from C")
            }
        }

        // Replay [C, M) and compare the REBUILT WINDOWS. This is the
        // assertion that measures the replay; the tokens below do not.
        let adoptedCaches = caches(backend, kinds, state)
        prefill(
            model: model, caches: adoptedCaches, prompt: prompt,
            from: plan.replayStart, upTo: Self.matched, plan: plan)
        var comparedWindows = 0
        for (index, kind) in kinds.enumerated() {
            guard case .slidingWindow = kind.attention else { continue }
            let rebuilt = try #require(state[index] as? PagedSequenceKV)
            let reference = try #require(referenceState[index] as? PagedSequenceKV)
            #expect(rebuilt.absoluteOffset == Self.matched)
            #expect(reference.absoluteOffset == Self.matched)
            let span = min(Self.matched, Self.window)
            let got = rebuilt.gatherRange(start: Self.matched - span, count: span)
            let want = reference.gatherRange(start: Self.matched - span, count: span)
            assertIdentical(
                got.keys, want.keys,
                "layer \(index) window keys rebuilt by a replay of R = \(plan.replayTokens)")
            assertIdentical(got.values, want.values, "layer \(index) window values")
            comparedWindows += 1
        }
        #expect(comparedWindows == Self.slidingLayers, "premise: there are windows to rebuild")
        referenceBackend.release(referenceState)

        // The frozen rows must still hold the donated bytes after the replay
        // wrote straight through them.
        for (index, kind) in kinds.enumerated() {
            guard case .full = kind.attention else { continue }
            let row = try #require(state[index] as? PagedSequenceKV)
            let original = try #require(prefix[index])
            let (keys, values) = row.gatherRange(start: 0, count: Self.matched)
            assertIdentical(keys, original.keys, "layer \(index) frozen keys survived the replay")
            assertIdentical(values, original.values, "layer \(index) frozen values")
        }

        // Finish the prompt above M and decode: the end-to-end property.
        prefill(
            model: model, caches: adoptedCaches, prompt: prompt,
            from: Self.matched, upTo: prompt.count - 1, plan: plan)
        let adoptedTokens = decode(
            model: model, caches: adoptedCaches, from: prompt.last!, steps: steps)
        backend.release(state)

        let agreed = zip(adoptedTokens, coldTokens).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adoptedTokens == coldTokens,
            """
            an adoption at M = \(Self.matched) (R = \(plan.replayTokens), \
            C = \(plan.replayStart)) diverged from a cold twin after \(agreed) of \
            \(steps) tokens
            """)
    }

    // MARK: - 2. Nothing else moved

    /// The narrowing is confined to the `.frozenFullReplay` branch, and the
    /// clamp a frozen plan imposes is inert outside `[0, M)`.
    ///
    /// Three ways this change could have leaked into requests that never
    /// adopt, each checked here:
    ///
    ///  1. a non-frozen layout's bound could have moved. `.direct` and
    ///     `.tailReplay` never carried the extra window, so both backends
    ///     must still agree, and the frozen layout must now agree with the
    ///     tail-replay layout that has the SAME sliding shape — that equality
    ///     IS the delta this ticket landed, in one line.
    ///  2. the clamp could bind above M and chunk the ordinary prefill of
    ///     `[M, promptLength)` — 26k+ tokens on the real harness prompt —
    ///     into window-sized pieces.
    ///  3. the ordinary prefill could compute something different under a
    ///     plan than without one. Compared bit for bit, per layer.
    @Test func theOrdinaryPathIsUntouchedByTheNarrowing() throws {
        let slidingOnly = Array(repeating: sliding(Self.window), count: Self.slidingLayers)
        let frozenShape = [full()] + slidingOnly + [full()]
        let allFull = [full(), full()]

        for backend in [CBv2PrefixReuseBackend.pagedFP16, .contiguousUnquantized] {
            let tail = CBv2PrefixReuseCapability.derive(layerKinds: slidingOnly, backend: backend)
            let frozen = CBv2PrefixReuseCapability.derive(
                layerKinds: frozenShape, backend: backend)
            let direct = CBv2PrefixReuseCapability.derive(layerKinds: allFull, backend: backend)
            #expect(tail.strategy == .tailReplay, "\(backend.rawValue)")
            #expect(frozen.strategy == .frozenFullReplay, "\(backend.rawValue)")
            #expect(direct.strategy == .direct, "\(backend.rawValue)")
            #expect(direct.conservativeReplayBoundTokens == 0, "\(backend.rawValue)")
            #expect(tail.conservativeReplayBoundTokens == Self.bound, "\(backend.rawValue)")
            #expect(
                frozen.conservativeReplayBoundTokens == tail.conservativeReplayBoundTokens,
                """
                \(backend.rawValue): adding an owning full layer downstream must not \
                change the replay bound. For .pagedFP16 it used to add one window, and \
                that difference is precisely what this ticket removed
                """)
        }

        // (2) The clamp splits at C and M and is inert everywhere else.
        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: frozenShape, backend: .pagedFP16)
        let plan = try #require(
            paged.plan(matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        let wideChunk = 4 * Self.window
        for offset in [0, 1, Self.window, 4 * Self.window] {
            #expect(
                plan.clampedChunk(start: Self.matched + offset, proposed: wideChunk)
                    == wideChunk,
                "the prefill above M keeps the pool's full chunk (offset \(offset))")
        }
        #expect(plan.clampedChunk(start: 0, proposed: wideChunk) == plan.replayStart)
        #expect(
            plan.clampedChunk(start: plan.replayStart, proposed: wideChunk)
                == Self.matched - plan.replayStart)
        #expect(
            plan.clampedChunk(start: plan.replayStart, proposed: Self.chunkSize)
                == Self.chunkSize,
            "below M the replay leg gets the pool's chunk — the one-window cap is gone")

        // (3) End to end: the ordinary segment [M, promptLength) computes the
        //     same bytes with and without a plan driving the chunking.
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x2B0D_4C11)
        let upTo = prompt.count - 1

        var arms: [[CBv2SequenceKV?]] = []
        var backends: [PagedKVBackend] = []
        var tokens: [[Int]] = []
        for usePlan in [false, true] {
            let backend = try makeBackend(kinds, chunk: wideChunk)
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
            let armCaches = caches(backend, kinds, state)
            // Identical below M, so any difference above it is the clamp's.
            prefill(
                model: model, caches: armCaches, prompt: prompt, from: 0,
                upTo: Self.matched, chunk: Self.chunkSize)
            prefill(
                model: model, caches: armCaches, prompt: prompt, from: Self.matched,
                upTo: upTo, chunk: wideChunk, plan: usePlan ? plan : nil)
            tokens.append(
                decode(model: model, caches: armCaches, from: prompt.last!, steps: Self.window))
            arms.append(state)
            backends.append(backend)
        }
        defer { for (index, backend) in backends.enumerated() { backend.release(arms[index]) } }

        for (index, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
            let plain = try #require(arms[0][index] as? PagedSequenceKV)
            let planned = try #require(arms[1][index] as? PagedSequenceKV)
            #expect(plain.absoluteOffset == planned.absoluteOffset, "layer \(index) cursor")
            // Only what each row still HOLDS: a sliding row's ring has
            // recycled everything older than `retainedCount`, and asking for
            // an evicted position is a trap, not a mismatch.
            let count = min(plain.retainedCount, planned.retainedCount)
            let start = plain.absoluteOffset - count
            guard count > 0 else { continue }
            let unplanned = plain.gatherRange(start: start, count: count)
            let underPlan = planned.gatherRange(start: start, count: count)
            assertIdentical(
                underPlan.keys, unplanned.keys,
                "layer \(index) keys: the clamp changed an ordinary prefill above M")
            assertIdentical(underPlan.values, unplanned.values, "layer \(index) values")
        }
        #expect(tokens[0] == tokens[1], "an ordinary prefill must not notice the plan")
    }
}
