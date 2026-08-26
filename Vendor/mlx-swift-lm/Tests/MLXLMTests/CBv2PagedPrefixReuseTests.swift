// CBv2PagedPrefixReuseTests.swift
//
// WS-4.1. Paged prefix reuse for an INTERLEAVED HYBRID layout — sliding
// layers with a storage-owning full layer downstream of them, which is
// gemma-4's shape (25 sliding + 5 full, `num_kv_shared_layers: 0`).
//
// The capability used to refuse this outright:
// `CBv2PrefixReuseCapability.derive` answered a "paged hybrid requires a dual
// cursor" refusal for `.pagedFP16`, so a paged gemma-4 never got a plan at
// all and `PagedKVBackend`'s own guard was dead code behind it. (That refusal
// case has since been deleted — `derive` cannot produce it.) That is a straight capability regression against the
// contiguous backend, which serves the same layout.
//
// What actually blocks paged is NOT the sliding rows. `fastForward(to:)`
// already gives a windowed row an initial absolute offset AND sets its
// `baseOffset`, and `CBv2PagedBackendTests.adoptedRowGreedyDecodeMatchesCold
// Row` already proves an offset row token-exact. It is the FULL rows:
// `.frozenFullReplay` needs storage that stays immutable through M while the
// logical cursor advances from C, and `PagedSequenceKV.write` appends at the
// frontier unconditionally.
//
// This suite covers the form that needs no second cursor at all — the one
// `PagedSeamContract` describes at `:509-515`. Every sliding row is restored
// EXACTLY at M from a `CBv2PagedWindowSnapshot`, so both cursors sit at M and
// R collapses to zero.
//
// Three properties, in the order they can fail:
//
//   1. an accepted restore is TOKEN-EXACT against a cold twin, greedy-decoded
//      over three full window turnovers;
//   2. a restore that cannot be completed EXACTLY refuses — it never installs
//      a partial window, and it leaves the pool byte-for-byte as it found it
//      so the cold prefill it degrades to can have the capacity;
//   3. the restored payload lands at the right ABSOLUTE positions, and a
//      payload for any other boundary is refused rather than placed.
//
// (3) is the one with teeth. `PagedSequenceKV` stores by absolute position,
// so a window installed at the wrong base is invisible: no trap, no
// telemetry, just wrong answers.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2PrefixReuse: paged frozen-full hybrid")
struct CBv2PrefixReusePagedFrozenFullTests {

    // MARK: - Fixture

    /// `[full, sliding(16), sliding(16), full]` — an owning full layer AFTER
    /// windowed ones, which is what makes `hasOwningFullAfterWindow` hold and
    /// selects `.frozenFullReplay`. headDim 64: the paged kernel supports
    /// {64, 128, 256, 512}.
    private static let window = 16
    private static let chunkSize = 8
    private static let maxLength = 256
    /// Matched boundary. Two pages at the default page size, and > `window`
    /// so the restored payload is a true window rather than a whole history.
    private static let matched = 32

    private func makeModel() -> TinyTestModel {
        TinyTestModel.make(
            seed: 0x4A11_B0AD, headDim: 64, stackedSlidingFull: true,
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

    /// Prefill `prompt[from ..< upTo]` in `chunkSize` chunks. Returns the
    /// caches so a caller can keep decoding through them.
    ///
    /// `plan` reproduces the scheduler's boundary split: a frozen replay
    /// chunk must never straddle M, because a row whose storage is frozen
    /// below M and appendable above it cannot serve one call that does both.
    /// `PagedSequenceKV.write` traps rather than guessing, so this is the
    /// same discipline the engine applies through `clampedChunk`.
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

    /// Prefill `prompt[from ..< count - 1]`, then greedy-decode `steps`
    /// tokens. Greedy decoding is the amplifier: a window short by one key
    /// perturbs the logits, one argmax flips, and the trajectories separate
    /// permanently.
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

    /// The donation a per-block window sidecar would reconstruct at `M`: the
    /// LAST `min(M, window)` positions of a row that ended at `M`.
    ///
    /// A live paged row retains more than a window mid-prefill
    /// (`retainedCount` is `min(written, window - 1 + lastUpdateTokens)`), and
    /// `CBv2PagedWindowSnapshot.requireAdmissible` refuses anything that is
    /// not exactly the window — a longer payload is not a window at all. The
    /// slice is the sidecar's job; here it is the fixture's.
    private func donatedWindow(
        _ row: CBv2SequenceKV, window: Int, boundary: Int
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let snapshot = row.snapshot()
        eval(snapshot.keys, snapshot.values)
        let tokens = min(boundary, window)
        let total = snapshot.keys.dim(2)
        return (
            keys: snapshot.keys[.ellipsis, (total - tokens) ..< total, 0...],
            values: snapshot.values[.ellipsis, (total - tokens) ..< total, 0...],
            offset: snapshot.offset
        )
    }

    private func donatedFull(
        _ row: CBv2SequenceKV
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let snapshot = row.snapshot()
        // Paged snapshots are lazy views over the SHARED slabs; the donor's
        // pages are recycled the instant its state is released.
        eval(snapshot.keys, snapshot.values)
        return snapshot
    }

    /// Donor arm: a cold prefill of `prompt[0 ..< matched)`, snapshotted per
    /// layer exactly as a prefix cache plus a window sidecar would hold it,
    /// then released.
    private func donate(
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
            guard kind.sharesKVWithLayer == nil, let row = state[index] else { return nil }
            switch kind.attention {
            case .full:
                return donatedFull(row)
            case .slidingWindow(let window):
                return donatedWindow(row, window: window, boundary: matched)
            }
        }
        backend.release(state)
        return prefix
    }

    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    private func restorePlan(
        _ capability: CBv2PrefixReuseCapability, matched: Int
    ) throws -> CBv2PrefixReusePlan {
        try #require(
            capability.plan(
                matchedBoundary: matched,
                maximumSequenceLength: Self.maxLength,
                restoringWindowsAtBoundary: true))
    }

    // MARK: - 1. Capability and plan shape

    /// The refusal that actually blocked paged was in `derive`, not in the
    /// backend: with `strategy == nil` no plan was ever produced, so
    /// `PagedKVBackend`'s `.frozenFullReplay` guard could not run.
    @Test func pagedHybridDerivesFrozenFullInsteadOfRefusing() throws {
        let kinds = makeModel().layerKinds
        #expect(kinds.count == 4)
        #expect(kinds[1].attention == .slidingWindow(Self.window))
        #expect(kinds[3].attention == .full, "an owning full layer AFTER a windowed one")

        let paged = CBv2PrefixReuseCapability.derive(layerKinds: kinds, backend: .pagedFP16)
        #expect(paged.isSupported)
        #expect(paged.unsupportedReason == nil)
        #expect(paged.strategy == .frozenFullReplay)

        let contiguous = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)
        #expect(contiguous.strategy == .frozenFullReplay)
        // Two windowed layers x window 16. Paged pays the SAME bound as
        // contiguous. It briefly paid one window more, because a frozen paged
        // row attended its chunk's freshly projected keys where the
        // contiguous frozen row hands back the cached ones;
        // `PagedLayerCache.prefillKVWritingChunk` reads the cached diagonal
        // out of the frozen pages now, so the two bounds are equal and this
        // assertion is what keeps them that way.
        #expect(contiguous.conservativeReplayBoundTokens == 2 * Self.window)
        #expect(paged.conservativeReplayBoundTokens == 2 * Self.window)
        #expect(
            paged.conservativeReplayBoundTokens
                == contiguous.conservativeReplayBoundTokens)
    }

    /// The two frozen-full forms, and the reach difference between them.
    ///
    /// The replay form cannot produce a plan at or below its own replay
    /// bound: `replayStart` would be 0, which saves nothing, and `plan`
    /// returns nil. That is a real limit on the CONTIGUOUS backend too — for
    /// gemma-4 (25 sliding layers, window 1,024) it means frozen-full reuse
    /// never fires below a matched boundary of 25,600 tokens. The restore
    /// form has no such floor: it fires at every boundary the prefix cache
    /// can return.
    @Test func windowRestoreFiresAtBoundariesTheReplayFormCannotReach() throws {
        let kinds = makeModel().layerKinds
        let paged = CBv2PrefixReuseCapability.derive(layerKinds: kinds, backend: .pagedFP16)
        let contiguous = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)

        // At or below the replay bound the replay form has nothing to save.
        for boundary in [8, 16, 32] {
            #expect(
                paged.plan(matchedBoundary: boundary, maximumSequenceLength: Self.maxLength)
                    == nil,
                "replay form must not claim a saving at boundary \(boundary)")
            let restored = try restorePlan(paged, matched: boundary)
            #expect(restored.requiresExactWindowRestore)
            #expect(restored.replayTokens == 0)
            #expect(restored.replayStart == boundary)
            #expect(restored.prefillTokensSaved == boundary)
        }
        #expect(contiguous.plan(matchedBoundary: 32, maximumSequenceLength: Self.maxLength) == nil)

        // Above the bound both forms produce a plan; only the shapes differ.
        let replay = try #require(
            paged.plan(matchedBoundary: 64, maximumSequenceLength: Self.maxLength))
        #expect(replay.strategy == .frozenFullReplay)
        #expect(replay.replayTokens == 2 * Self.window)
        #expect(replay.replayStart == 64 - 2 * Self.window)
        #expect(replay.restoredFullTokens == 64, "full rows are exact through M in both forms")
        #expect(!replay.requiresExactWindowRestore)

        let restored = try restorePlan(paged, matched: 64)
        #expect(restored.restoredFullTokens == 64)
        #expect(restored.capacityReservationTokens == 64)
        #expect(restored.requiresExactWindowRestore)

        // The opt-in is frozen-full only: a tail-replay layout restores its
        // full rows to C, so its windowed rows must march from C as well.
        let tail = CBv2PrefixReuseCapability.derive(
            layerKinds: [kinds[0], kinds[1]], backend: .pagedFP16)
        #expect(tail.strategy == .tailReplay)
        let tailPlan = try #require(
            tail.plan(
                matchedBoundary: 64, maximumSequenceLength: Self.maxLength,
                restoringWindowsAtBoundary: true))
        #expect(tailPlan.replayTokens == Self.window)
        #expect(!tailPlan.requiresExactWindowRestore)
    }

    // MARK: - 2. Token exactness

    /// The acceptance property: a paged gemma-4-shaped request ACCEPTS a
    /// `.frozenFullReplay` plan and produces output identical to a cold twin,
    /// token for token.
    @Test func acceptedFrozenFullRestoreIsTokenExactAgainstAColdTwin() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 49, seed: 0xBEE_F00D)
        let steps = 3 * Self.window

        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let cold = run(
            model: model, backend: coldBackend, state: coldState, prompt: prompt,
            from: 0, steps: steps)
        coldBackend.release(coldState)

        let backend = try makeBackend(kinds)
        let prefix = try donate(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try restorePlan(capability, matched: Self.matched)

        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)
        let adopted = run(
            model: model, backend: backend, state: state, prompt: prompt,
            from: Self.matched, steps: steps)
        backend.release(state)

        let agreed = zip(adopted, cold).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adopted == cold,
            """
            adopted hybrid diverged from a cold twin decoded from the same prompt \
            (\(agreed) of \(steps) tokens matched before divergence)
            """)
    }

    // MARK: - 3. Absolute positions

    /// The restored window must sit at its TRUE absolute positions, not
    /// merely somewhere. `PagedSequenceKV` maps position `p` to ring slot
    /// `(p / pageSize) % ringPages`, so an install at the wrong base is
    /// undetectable downstream — this is the assertion that would catch it.
    @Test func restoredWindowOccupiesItsTrueAbsolutePositions() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 49, seed: 0xA11_60_0D)
        let matched = Self.matched
        let base = matched - Self.window

        let backend = try makeBackend(kinds)
        let prefix = try donate(
            model: model, backend: backend, prompt: prompt, matched: matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let state = try backend.makeSequenceState(
            adopting: prefix, plan: try restorePlan(capability, matched: matched),
            layerKinds: kinds, maxLength: Self.maxLength)
        defer { backend.release(state) }

        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            switch kind.attention {
            case .full:
                #expect(row.absoluteOffset == matched)
                #expect(row.baseOffset == 0, "a restored full row owns [0, M)")
            case .slidingWindow:
                #expect(row.absoluteOffset == matched)
                #expect(
                    row.baseOffset == base,
                    "layer \(index) window must begin at absolute \(base), not at 0")
                #expect(row.retainedCount == Self.window)

                // The bytes at those absolute positions are the donated ones.
                // A one-page shift would still gather cleanly here and would
                // still be wrong; only comparing CONTENT at ABSOLUTE indices
                // catches it.
                let donated = try #require(prefix[index])
                let (keys, values) = row.gatherRange(start: base, count: Self.window)
                #expect(maxAbsDiff(keys, donated.keys) == 0)
                #expect(maxAbsDiff(values, donated.values) == 0)

                // And the range the decode kernel will be handed is the
                // window ending at M, expressed in the same absolute frame:
                // the query sits at the last restored position, M - 1, and
                // its window is exactly [M - W, M). Installed one page low,
                // `baseOffset` would clamp this to 0 and the row would
                // attend keys it never wrote.
                let attend = row.decodeAttendRange
                #expect(attend.start == base)
                #expect(attend.length == Self.window)
                #expect(attend.start + attend.length == matched)
            }
        }
    }

    /// A payload is admissible at exactly ONE boundary. `PrefixCacheV2`
    /// indexes every whole-block boundary of a donation, so a window taken at
    /// 32 will routinely be offered against a match at 16 — and installing it
    /// there would write the donor's positions `[16, 32)` into the adopter's
    /// `[0, 16)`.
    @Test func aWindowFromAnotherBoundaryIsRefusedNotRelocated() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 49, seed: 0xD1F_F007)

        let backend = try makeBackend(kinds)
        let prefix = try donate(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)

        // Same payload, offered at a shorter boundary the same donation is
        // also indexed at. The full snapshots are sliced to match, so the
        // ONLY thing wrong is the window's absolute extent.
        let shorter = 16
        let relocated = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard let entry = prefix[index] else { return nil }
            guard case .full = kind.attention else { return entry }
            return (
                keys: entry.keys[.ellipsis, 0 ..< shorter, 0...],
                values: entry.values[.ellipsis, 0 ..< shorter, 0...],
                offset: shorter
            )
        }
        try expectRefusal(
            backend: backend, prefix: relocated,
            plan: try restorePlan(capability, matched: shorter),
            kinds: kinds)

        // And the mirror image: a payload whose base is shifted one page
        // forward of where it belongs, offered at its own boundary.
        let shifted = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard let entry = prefix[index] else { return nil }
            guard case .slidingWindow = kind.attention else { return entry }
            return (keys: entry.keys, values: entry.values, offset: entry.offset + 16)
        }
        try expectRefusal(
            backend: backend, prefix: shifted,
            plan: try restorePlan(capability, matched: Self.matched),
            kinds: kinds)
    }

    // MARK: - 4. Refusal, never a partial restore

    /// A frozen-full hit restores the full layers to M. If the sliding side
    /// cannot then be completed EXACTLY, the request must cold-prefill — a
    /// short window is not a smaller window, it is a window missing its
    /// oldest entries, which attention silently ignores and no replay can
    /// recover. Every refusal must also leave the pool untouched, or the
    /// cold prefill it degrades to inherits a leak.
    @Test func anIncompleteRestoreRefusesAndLeavesThePoolUntouched() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 49, seed: 0xDEAD_10CC)
        let matched = Self.matched

        let backend = try makeBackend(kinds)
        let prefix = try donate(
            model: model, backend: backend, prompt: prompt, matched: matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try restorePlan(capability, matched: matched)

        func mutating(
            layer: Int,
            _ transform: ((keys: MLXArray, values: MLXArray, offset: Int)?) -> (
                keys: MLXArray, values: MLXArray, offset: Int
            )?
        ) -> [(keys: MLXArray, values: MLXArray, offset: Int)?] {
            var copy = prefix
            copy[layer] = transform(prefix[layer])
            return copy
        }

        // (a) A sliding layer with no donated window at all. This is the
        //     production case today: `PrefixCacheV2.isCacheable` drops every
        //     windowed layer, so nothing in the engine donates one.
        try expectRefusal(
            backend: backend, prefix: mutating(layer: 1) { _ in nil }, plan: plan, kinds: kinds)

        // (b) A window short by one position. The missing entry is the OLDEST
        //     one, which is exactly the entry a short replay cannot rebuild.
        try expectRefusal(
            backend: backend,
            prefix: mutating(layer: 2) { entry in
                guard let entry else { return nil }
                return (
                    keys: entry.keys[.ellipsis, 1 ..< Self.window, 0...],
                    values: entry.values[.ellipsis, 1 ..< Self.window, 0...],
                    offset: entry.offset
                )
            },
            plan: plan, kinds: kinds)

        // (c) An owning full layer with no snapshot.
        try expectRefusal(
            backend: backend, prefix: mutating(layer: 3) { _ in nil }, plan: plan, kinds: kinds)

        // (d) A full snapshot that does not reach M.
        try expectRefusal(
            backend: backend,
            prefix: mutating(layer: 0) { entry in
                guard let entry else { return nil }
                return (
                    keys: entry.keys[.ellipsis, 0 ..< (matched - 8), 0...],
                    values: entry.values[.ellipsis, 0 ..< (matched - 8), 0...],
                    offset: matched - 8
                )
            },
            plan: plan, kinds: kinds)

        // The refusals degrade to a cold prefill on the SAME pool, and that
        // prefill must be the ordinary one — this is the property that makes
        // refusing safe rather than merely loud.
        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let expected = run(
            model: model, backend: coldBackend, state: coldState, prompt: prompt,
            from: 0, steps: 2 * Self.window)
        coldBackend.release(coldState)

        let fallbackState = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let fallback = run(
            model: model, backend: backend, state: fallbackState, prompt: prompt,
            from: 0, steps: 2 * Self.window)
        backend.release(fallbackState)
        #expect(fallback == expected)
    }

    // MARK: - 5. The dual-cursor replay form

    /// The form the engine actually produces today: full rows exact and
    /// IMMUTABLE through M with the logical cursor at C, sliding rows empty at
    /// C, `[C, M)` replayed.
    ///
    /// Two properties, and the SECOND is the one with teeth.
    ///
    ///  1. The frozen rows stay frozen. If `PagedSequenceKV.write` appended
    ///     during the replay, the exact cached full K/V for `[C, M)` would be
    ///     overwritten with projections taken from sliding rows that do not
    ///     have their windows yet, and the damage is PERMANENT — full
    ///     attention keeps every key forever.
    ///  2. The REBUILT SLIDING WINDOWS are bit-equal to a cold twin's at the
    ///     same absolute positions. Token equality alone does NOT discriminate
    ///     on this fixture: P4_FrozenChunkGather measured that halving R to 16
    ///     still yields identical tokens here, because layer 0 reads the exact
    ///     embedding (so its fresh projections already equal the cached ones)
    ///     and layer 3 feeds only discarded pre-M logits. A test that asserted
    ///     tokens alone would pass on a replay that is half as long as the
    ///     cone arithmetic requires. The window comparison is what fails.
    @Test func frozenReplayFormIsTokenExactAgainstAColdTwin() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x5EE_D5)
        let matched = 64
        let steps = 3 * Self.window

        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let cold = run(
            model: model, backend: coldBackend, state: coldState, prompt: prompt,
            from: 0, steps: steps)
        coldBackend.release(coldState)

        let backend = try makeBackend(kinds)
        let donated = try donate(
            model: model, backend: backend, prompt: prompt, matched: matched)
        // The replay form recomputes windowed layers, so it must be offered
        // full snapshots only — exactly what `PrefixCacheV2` hands over.
        let prefix = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            if case .slidingWindow = kind.attention { return nil }
            return donated[index]
        }

        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(matchedBoundary: matched, maximumSequenceLength: Self.maxLength))
        #expect(!plan.requiresExactWindowRestore)
        #expect(plan.replayTokens == 2 * Self.window)
        #expect(plan.replayStart == matched - 2 * Self.window)
        #expect(
            plan.replayTokens
                >= cbv2RequiredRecompute(layerKinds: kinds, matched: matched),
            "paged's grant must clear the shared bound, same as contiguous")

        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)

        // Both cursors are C; only the full rows carry storage past it.
        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            #expect(row.absoluteOffset == plan.replayStart, "layer \(index)")
            switch kind.attention {
            case .full:
                #expect(row.frozenHighWater == matched)
                #expect(row.baseOffset == 0)
                #expect(row.retainedCount == plan.replayStart)
            case .slidingWindow:
                #expect(row.frozenHighWater == 0)
                #expect(row.baseOffset == plan.replayStart)
                #expect(row.retainedCount == 0, "sliding rows rebuild from C")
            }
        }

        // (2) Replay [C, M) ONLY, and compare the rebuilt sliding windows
        //     against a cold row prefilled straight through to M. This is the
        //     assertion that measures the replay; the tokens below do not.
        prefill(
            model: model, backend: backend, state: state, prompt: prompt,
            from: plan.replayStart, upTo: matched, plan: plan)

        let windowBackend = try makeBackend(kinds)
        let windowState = try windowBackend.makeSequenceState(
            layerKinds: kinds, promptLength: matched, maxLength: Self.maxLength)
        prefill(
            model: model, backend: windowBackend, state: windowState, prompt: prompt,
            from: 0, upTo: matched)
        for (index, kind) in kinds.enumerated() {
            guard case .slidingWindow = kind.attention else { continue }
            let rebuilt = try #require(state[index] as? PagedSequenceKV)
            let reference = try #require(windowState[index] as? PagedSequenceKV)
            #expect(rebuilt.absoluteOffset == matched && reference.absoluteOffset == matched)
            let span = min(matched, Self.window)
            let got = rebuilt.gatherRange(start: matched - span, count: span)
            let want = reference.gatherRange(start: matched - span, count: span)
            #expect(
                maxAbsDiff(got.keys, want.keys) == 0,
                "layer \(index) rebuilt its window inexactly — the replay is too short")
            #expect(maxAbsDiff(got.values, want.values) == 0, "layer \(index) values")
        }
        windowBackend.release(windowState)

        // (3) Finish the prompt and decode; tokens are the end-to-end check.
        let adopted = run(
            model: model, backend: backend, state: state, prompt: prompt,
            from: matched, steps: steps, plan: plan)

        // The frozen rows must still hold the donated bytes at [0, M) after
        // the replay wrote straight through them.
        for (index, kind) in kinds.enumerated() {
            guard case .full = kind.attention else { continue }
            let row = try #require(state[index] as? PagedSequenceKV)
            let original = try #require(prefix[index])
            let (keys, values) = row.gatherRange(start: 0, count: matched)
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
            frozen replay diverged from a cold twin decoded from the same prompt \
            (\(agreed) of \(steps) tokens matched before divergence)
            """)
    }

    /// A replay shorter than the layout's cone bound is refused, not
    /// attempted.
    ///
    /// A plan reaches the backend from outside the type, so the relation is
    /// checked rather than assumed even though `derive` now grants exactly
    /// this. A too-short replay leaves the sliding rows inexact, which is
    /// invisible: the rows are full-length, just wrong.
    @Test func aReplayShorterThanTheConeBoundIsRefused() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0xC0FF_EE11)
        let matched = 64

        // 2 sliding layers x window 16 — the SHARED bound, no paged extra.
        #expect(cbv2RequiredRecompute(layerKinds: kinds, matched: matched) == 2 * Self.window)
        #expect(
            cbv2RequiredRecompute(layerKinds: [kinds[0]], matched: matched) == 0,
            "no windowed layer, no replay")

        let backend = try makeBackend(kinds)
        let donated = try donate(
            model: model, backend: backend, prompt: prompt, matched: matched)
        let prefix = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            if case .slidingWindow = kind.attention { return nil }
            return donated[index]
        }
        let short = CBv2PrefixReusePlan(
            backend: .pagedFP16,
            strategy: .frozenFullReplay,
            matchedBoundary: matched,
            replayStart: matched - 24,
            replayTokens: 24,
            prefillTokensSaved: matched - 24,
            restoredFullTokens: matched,
            capacityReservationTokens: matched,
            nominalFullKVBytesPerToken: 0,
            fullKVBytesPerToken: 0,
            additionalFullKVBytesPerToken: 0,
            initialAdditionalCapacityBytes: 0,
            fullCapacityTokensReserved: matched,
            stagedFullKVBytes: 0,
            residentFullKVBytes: 0)
        try expectRefusal(backend: backend, prefix: prefix, plan: short, kinds: kinds)

        // And a replay plan that ALSO carries windows is a contradiction: one
        // row would sit at M while every other row sits at C.
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let replay = try #require(
            capability.plan(matchedBoundary: matched, maximumSequenceLength: Self.maxLength))
        try expectRefusal(backend: backend, prefix: donated, plan: replay, kinds: kinds)
    }

    /// REGRESSION, gpt-oss-20b (G2 parity gate, 2026-07-25). A layout whose
    /// window is SMALLER than the pool's prefill chunk lost prefix reuse
    /// entirely: the planner granted `windowCount*maxWindow + maxWindow` while
    /// the backend demanded `windowCount*maxWindow + maxPrefillChunk`, so
    /// adoption refused after a perfectly good match. Measured on the real
    /// model: 12 sliding layers of window 128 against a 512-token chunk,
    /// granted 1,664 against a demanded 2,048, a match of 28,416 tokens, and
    /// 26,752 tokens of saving thrown away — while contiguous served the same
    /// match. The falsified comment read "every shipping config".
    ///
    /// The chunk term is now gone entirely (`PagedLayerCache
    /// .prefillKVWritingChunk` reads the frozen diagonal out of the pages),
    /// so the invariant this test defends is stronger than the original fix:
    /// THE BOUND DOES NOT DEPEND ON `maxPrefillChunk` AT ALL. Any future
    /// change that reintroduces a chunk term brings the gpt-oss refusal back
    /// with it, and fails here.
    @Test func theReplayBoundIsIndependentOfThePoolChunk() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let matched = 64
        // window 16 against a 32-token pool chunk — the gpt-oss relation.
        let chunk = 2 * Self.window
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 64 << 20, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: Self.maxLength))
        #expect(backend.pool.config.maxPrefillChunk > Self.window, "the defect's precondition")

        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(matchedBoundary: matched, maximumSequenceLength: Self.maxLength))
        // Granted, demanded, and the shared bound are ONE number, whatever the
        // pool's chunk is.
        #expect(plan.replayTokens == 2 * Self.window)
        #expect(cbv2RequiredRecompute(layerKinds: kinds, matched: matched) == plan.replayTokens)

        // The replay leg is no longer capped: the chunk that reaches a frozen
        // row below M is the pool's, exactly as above M. If a cap ever comes
        // back it must come back WITH a bound that pays for it.
        #expect(plan.clampedChunk(start: plan.replayStart, proposed: chunk) == chunk)
        #expect(plan.clampedChunk(start: matched, proposed: chunk) == chunk)
        #expect(
            plan.clampedChunk(start: matched - Self.window, proposed: chunk) == Self.window,
            "still split at M")

        // And it adopts rather than refusing.
        let prompt = makePromptTokens(length: 96, seed: 0x5EE_D5)
        let donated = try donate(
            model: model, backend: backend, prompt: prompt, matched: matched)
        let prefix = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            if case .slidingWindow = kind.attention { return nil }
            return donated[index]
        }
        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)
        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            #expect(row.absoluteOffset == plan.replayStart, "layer \(index)")
            if case .full = kind.attention { #expect(row.frozenHighWater == matched) }
        }
        backend.release(state)
    }

    // MARK: - Refusal helper

    /// Adoption must THROW (so `EngineLoopV2.applyAdoption` can fall back),
    /// must not trap, and must not have charged the pool.
    private func expectRefusal(
        backend: PagedKVBackend,
        prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        kinds: [CBv2LayerKind],
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let before = (backend.bytesInUse, backend.bytesReserved)
        #expect(throws: CBv2KVError.self, sourceLocation: sourceLocation) {
            _ = try backend.makeSequenceState(
                adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)
        }
        #expect(
            backend.bytesInUse == before.0,
            "a refused adoption must not materialize pages", sourceLocation: sourceLocation)
        #expect(
            backend.bytesReserved == before.1,
            "a refused adoption must not hold a reservation", sourceLocation: sourceLocation)
    }
}
