// CBv2PackedPrefillTests.swift — rectangular [B, L] packed prompt prefill.
//
// `EngineLoopV2.executeMixed` groups the step's PROMPT rows by
// (chunk length, samples flag) and, when BOTH base capability gates agree,
// runs each equal-length group of >1 rows as ONE rectangular `[B, L]`
// forward through `prefillOutput` instead of B separate `[1, L]` forwards.
// The base gates are:
//
//   * `CBv2LayerCacheProvider.supportsPackedPrefill` — the caches keep rows
//     independent (`CBv2LayerCacheBank`: true only when every cache is the
//     contiguous `CBv2LayerCache`), and
//   * `CBv2PackedPrefillSteppableModel.supportsPackedPrefill` — the model's
//     prompt forward is shape-generic over the batch axis
//     (`Gemma4TextModel.cbv2SupportsPackedPrefill == true`).
//
// Span-bearing rows additionally require the model's packed-multimodal claim
// and the provider's row-aligned span-binding claim. The fixture below omits
// that stronger model claim so its span row pins the singleton fallback.
//
// This suite pins the whole seam:
//
//  1. `testFourEqualPromptRowsRunAsOneRectangularForward` — packing actually
//     happens: four equal chunks become ONE `[4, chunk]` prefill, not four
//     `[1, chunk]` prefills (both the intermediate and the frontier chunk).
//  2. `testPackedRowsAreParityWithSingletonExecutionB2` / `...B4` — the
//     load-bearing test: every row's sampled first token matches the
//     singleton-path run, and EVERY layer's per-row KV (offset, retained
//     length, retained contents) is bit-identical.
//  3. `testPackedRowsWithDifferentPromptsDoNotContaminateEachOther` and
//     `testPackedRowsWithDifferentCacheOffsetsStayIndependent` — no
//     cross-row contamination, including rows at different pre-existing
//     absolute offsets in the same packed group.
//  4. `testDifferentChunkLengthsAreNotPackedTogether`,
//     `testDifferentSamplesFlagsAreNotPackedTogether`,
//     `testSingleRowGroupTakesTheSingletonPath` — grouping rules.
//  5. `testCacheProviderVetoPreventsPacking`,
//     `testModelVetoPreventsPacking`,
//     `testSpanBearingChunkFallsBackWithoutMultimodalClaim`,
//     `testLayerCacheBankVouchesOnlyForContiguousCaches` — fail-closed gates.
//  6. `testDecodeRowsKeepTheirOwnRectangularBatchAlongsidePacking` — decode
//     is untouched: it stays a `[Bdecode, 1]` `forward`, and its tokens are
//     unchanged by packing.
//  7. `testContinuationAfterPackedPrefillMatchesSingletonReference` — the
//     decode tokens FOLLOWING a packed prefill match the singleton
//     reference for several steps.
//  8. `testGemma4AdapterAdvertisesAndHonorsPackedPrefill` — the real
//     shipping conformer (`Gemma4TextModel` behind
//     `CBv2SteppableLanguageModelAdapter`) advertises the capability and is
//     per-row parity under packing, on a TINY random-weight config.
//
// Everything runs on tiny seeded-random fixtures (`TinyTestModel`, a 4-layer
// Gemma 4 config with vocab 64). No checkpoints are downloaded.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Bit-identity helper

/// Exact equality (no tolerance). Packed and singleton execution must be
/// byte-neutral in the KV they leave behind, not merely close.
private func cbv2PackedBitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape else { return false }
    if a.size == 0 { return true }
    return (a .== b).all().item(Bool.self)
}

// MARK: - Recording model wrapper

/// One model-facing call, as the engine issued it.
enum CBv2PackedCall: Equatable, CustomStringConvertible {
    /// A prompt chunk through the `prefillOutput` seam.
    case prefill(shape: [Int], requirement: CBv2PrefillRequirement, spliced: Bool)
    /// A `CBv2SteppableModel.forward` — decode, in this suite.
    case forward(shape: [Int])

    var description: String {
        switch self {
        case .prefill(let shape, let requirement, let spliced):
            return "prefill\(shape) \(requirement)\(spliced ? " spliced" : "")"
        case .forward(let shape):
            return "forward\(shape)"
        }
    }
}

/// Wraps ANY `CBv2SteppableModel` and records the SHAPE of every call the
/// engine makes, while letting the test dictate the model-side packing
/// claim. Prefill forwards to the inner model's own narrowing seam when it
/// has one (the Gemma 4 adapter), else reproduces `prefillOutput`'s
/// fallback (full logits + slice) so the numbers are unchanged either way.
final class CBv2PackedRecordingModel: CBv2PackedPrefillSteppableModel,
    CBv2MultimodalSteppableModel, @unchecked Sendable
{
    let inner: CBv2SteppableModel
    private let packedClaim: Bool
    private let lock = NSLock()
    private var _calls: [CBv2PackedCall] = []

    init(_ inner: CBv2SteppableModel, packed: Bool) {
        self.inner = inner
        self.packedClaim = packed
    }

    // MARK: recording

    var calls: [CBv2PackedCall] {
        lock.lock()
        defer { lock.unlock() }
        return _calls
    }

    /// Shapes of the prompt-chunk calls, in order.
    var prefillShapes: [[Int]] {
        calls.compactMap {
            if case .prefill(let shape, _, _) = $0 { return shape }
            return nil
        }
    }

    /// Shapes of the plain `forward` calls (decode), in order.
    var forwardShapes: [[Int]] {
        calls.compactMap {
            if case .forward(let shape) = $0 { return shape }
            return nil
        }
    }

    /// Prompt-chunk calls that carried more than one row.
    var packedPrefillShapes: [[Int]] { prefillShapes.filter { $0[0] > 1 } }

    func reset() {
        lock.lock()
        _calls = []
        lock.unlock()
    }

    private func record(_ call: CBv2PackedCall) {
        lock.lock()
        _calls.append(call)
        lock.unlock()
    }

    // MARK: CBv2PackedPrefillSteppableModel

    var supportsPackedPrefill: Bool { packedClaim }

    func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        record(
            .prefill(
                shape: tokens.shape, requirement: requirement,
                spliced: inputEmbeddings != nil))
        if let narrowing = inner as? CBv2PrefillSteppableModel {
            return narrowing.prefill(
                tokens: tokens, inputEmbeddings: inputEmbeddings,
                caches: caches, requirement: requirement)
        }
        // Same fallback `EngineLoopV2.prefillOutput` applies to a
        // non-narrowing model: full logits, sliced here.
        let logits: MLXArray
        if let inputEmbeddings {
            guard let multimodal = inner as? CBv2MultimodalSteppableModel else {
                preconditionFailure("spliced prefill against a non-multimodal fixture")
            }
            logits = multimodal.forward(
                tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        } else {
            logits = inner.forward(tokens: tokens, caches: caches)
        }
        switch requirement {
        case .evaluationOnly: return logits[0..., -1, 0 ..< 1]
        case .lastPositionLogits: return logits[0..., -1, 0...]
        }
    }

    // MARK: CBv2SteppableModel

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        record(.forward(shape: tokens.shape))
        return inner.forward(tokens: tokens, caches: caches)
    }

    // MARK: CBv2MultimodalSteppableModel

    var supportsMultimodalPrefill: Bool {
        (inner as? CBv2MultimodalSteppableModel)?.supportsMultimodalPrefill ?? false
    }

    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        guard let multimodal = inner as? CBv2MultimodalSteppableModel else {
            preconditionFailure("embedPromptTokens against a non-multimodal fixture")
        }
        return multimodal.embedPromptTokens(tokens)
    }

    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        record(.forward(shape: tokens.shape))
        guard let multimodal = inner as? CBv2MultimodalSteppableModel else {
            preconditionFailure("embedding forward against a non-multimodal fixture")
        }
        return multimodal.forward(
            tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
    }
}

// MARK: - Cache provider with a controllable packing claim

/// A real `CBv2LayerCacheBank` (production contiguous caches) whose
/// `supportsPackedPrefill` answer the test dictates, so the engine's
/// PROVIDER gate can be exercised independently of the caches' true
/// capability.
final class CBv2PackedTestCacheProvider: CBv2LayerCacheProvider, CBv2CompositionInvalidating {
    let bank: CBv2LayerCacheBank
    private let claim: Bool

    init(layerKinds: [CBv2LayerKind], claimsPackedPrefill: Bool) {
        self.bank = CBv2LayerCacheBank(layerKinds: layerKinds)
        self.claim = claimsPackedPrefill
    }

    var supportsMultimodalSpans: Bool { bank.supportsMultimodalSpans }
    var supportsPackedPrefill: Bool { claim }

    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        bank.layerCaches(rowStates: rowStates)
    }

    func invalidateBoundComposition() { bank.invalidateBoundComposition() }
    func releaseBoundRows() { bank.releaseBoundRows() }
}

// MARK: - Synchronous mixed-step driver

/// Drives `EngineLoopV2.executeMixed` step by step on the calling thread —
/// the loop is never started, so scheduling is fully deterministic and both
/// the model call log and the per-request KV state are directly inspectable.
///
/// Mirrors exactly what `finalize` does for the parts this suite needs:
/// materialize the sampled tokens and confirm them back into the scheduler.
final class CBv2PackedHarness {
    let loop: EngineLoopV2
    let scheduler: SchedulerV2
    let model: CBv2PackedRecordingModel
    let provider: CBv2PackedTestCacheProvider
    let backend: CBv2ContiguousKVBackend
    let layerKinds: [CBv2LayerKind]

    private(set) var order: [CBv2RequestID] = []
    private(set) var generated: [CBv2RequestID: [Int]] = [:]
    private var nextRaw: UInt64 = 1

    init(
        inner: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        modelClaimsPacking: Bool = true,
        providerClaimsPacking: Bool = true,
        prefillChunkSize: Int,
        maxBatchedTokensPerStep: Int,
        maxConcurrentRequests: Int = 8
    ) {
        self.layerKinds = layerKinds
        self.model = CBv2PackedRecordingModel(inner, packed: modelClaimsPacking)
        self.provider = CBv2PackedTestCacheProvider(
            layerKinds: layerKinds, claimsPackedPrefill: providerClaimsPacking)
        self.backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27))
        self.scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrentRequests,
                maxBatchedTokensPerStep: maxBatchedTokensPerStep,
                prefillChunkSize: prefillChunkSize,
                maxWaiting: 64))
        self.loop = EngineLoopV2(
            model: model,
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: provider,
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2NullDetokenizerFactory(),
            scheduler: scheduler,
            capacity: nil,
            config: CBv2EngineLoopConfig(),
            gauges: CBv2EngineGauges(kvBytesCapacity: 1 << 27))
    }

    // MARK: submission

    @discardableResult
    func enqueue(
        prompt: [Int], maxTokens: Int = 64,
        multimodal: CBv2ResolvedMultimodal? = nil
    ) throws -> CBv2RequestID {
        let id = CBv2RequestID(nextRaw)
        nextRaw += 1
        var request = CBv2Request(
            id: id, promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens)
        if let multimodal {
            // The scheduler needs the SPANS (block snapping); the resolved
            // embeddings live on the loop, exactly as `enqueue` places them.
            request.multimodal = CBv2MultimodalInput(spans: multimodal.spans) { [] }
        }
        _ = try scheduler.enqueue(request)
        if let multimodal { loop.multimodalByID[id] = multimodal }
        order.append(id)
        return id
    }

    // MARK: stepping

    /// One planned mixed step. Returns the (id, token) pairs sampled.
    @discardableResult
    func step() -> [(id: CBv2RequestID, token: Int)] {
        let plan = scheduler.plan()
        guard let inFlight = loop.executeMixed(plan) else { return [] }
        var sampled: [(id: CBv2RequestID, token: Int)] = []
        if let tokens = inFlight.sampledTokens {
            let host = tokens.asArray(Int32.self)
            for (index, id) in inFlight.sampledRows.enumerated() {
                let token = Int(host[index])
                scheduler.recordSampled(id: id, token: token)
                generated[id, default: []].append(token)
                sampled.append((id, token))
            }
        }
        if !inFlight.evalTargets.isEmpty { eval(inFlight.evalTargets) }
        return sampled
    }

    func run(steps: Int) {
        for _ in 0 ..< steps { step() }
    }

    // MARK: inspection

    func kv(_ id: CBv2RequestID) -> [CBv2SequenceKV?] {
        loop.kvStates[id] ?? []
    }

    func absoluteOffsets(_ id: CBv2RequestID) -> [Int] {
        kv(id).map { $0?.absoluteOffset ?? -1 }
    }
}

// MARK: - Suite

final class CBv2PackedPrefillTests: XCTestCase {

    private let weightSeed: UInt64 = 0xC0FFEE

    // MARK: helpers

    /// Two harnesses over the SAME weights: one with packing enabled, one
    /// with the model-side gate off (so every row takes the singleton
    /// `[1, chunk]` path). Everything else — scheduler config, backend,
    /// caches, sampler — is identical, so the ONLY difference is packing.
    private func makePair(
        base: TinyTestModel, prefillChunkSize: Int, maxBatchedTokensPerStep: Int,
        maxConcurrentRequests: Int = 8
    ) -> (packed: CBv2PackedHarness, singleton: CBv2PackedHarness) {
        (
            CBv2PackedHarness(
                inner: base, layerKinds: base.layerKinds, modelClaimsPacking: true,
                prefillChunkSize: prefillChunkSize,
                maxBatchedTokensPerStep: maxBatchedTokensPerStep,
                maxConcurrentRequests: maxConcurrentRequests),
            CBv2PackedHarness(
                inner: base, layerKinds: base.layerKinds, modelClaimsPacking: false,
                prefillChunkSize: prefillChunkSize,
                maxBatchedTokensPerStep: maxBatchedTokensPerStep,
                maxConcurrentRequests: maxConcurrentRequests)
        )
    }

    /// Every layer's per-row KV must agree exactly: absolute offset,
    /// retained length, and the retained K/V contents.
    private func assertKVBitIdentical(
        _ packed: [CBv2SequenceKV?], _ singleton: [CBv2SequenceKV?],
        row: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertFalse(packed.isEmpty, "\(row): test premise — the model has layers",
            file: file, line: line)
        XCTAssertEqual(
            packed.count, singleton.count, "\(row): layer count", file: file, line: line)
        for (layer, pair) in zip(packed, singleton).enumerated() {
            let (packedKV, singletonKV) = pair
            guard let packedKV, let singletonKV else {
                XCTAssertNil(
                    packedKV, "\(row) layer \(layer): KV-shared layers must match",
                    file: file, line: line)
                XCTAssertNil(
                    singletonKV, "\(row) layer \(layer): KV-shared layers must match",
                    file: file, line: line)
                continue
            }
            XCTAssertEqual(
                packedKV.absoluteOffset, singletonKV.absoluteOffset,
                "\(row) layer \(layer): absolute offset must be identical after a packed prefill",
                file: file, line: line)
            XCTAssertEqual(
                packedKV.retainedCount, singletonKV.retainedCount,
                "\(row) layer \(layer): retained KV length must be identical",
                file: file, line: line)
            let packedSnapshot = packedKV.snapshot()
            let singletonSnapshot = singletonKV.snapshot()
            eval(
                packedSnapshot.keys, packedSnapshot.values,
                singletonSnapshot.keys, singletonSnapshot.values)
            XCTAssertTrue(
                cbv2PackedBitIdentical(packedSnapshot.keys, singletonSnapshot.keys),
                "\(row) layer \(layer): retained KEYS must be bit-identical between packed "
                    + "and singleton execution",
                file: file, line: line)
            XCTAssertTrue(
                cbv2PackedBitIdentical(packedSnapshot.values, singletonSnapshot.values),
                "\(row) layer \(layer): retained VALUES must be bit-identical between packed "
                    + "and singleton execution",
                file: file, line: line)
        }
    }

    /// Shared body of the parity tests: run the same prompts through both
    /// harnesses and assert per-row token + KV agreement.
    @discardableResult
    private func assertParity(
        prompts: [[Int]], chunkSize: Int, budget: Int, steps: Int,
        expectPackedShape: [Int], file: StaticString = #filePath, line: UInt = #line
    ) -> (packed: CBv2PackedHarness, singleton: CBv2PackedHarness) {
        let base = TinyTestModel.make(seed: weightSeed)
        let (packed, singleton) = makePair(
            base: base, prefillChunkSize: chunkSize, maxBatchedTokensPerStep: budget)

        for prompt in prompts {
            XCTAssertNoThrow(try packed.enqueue(prompt: prompt), file: file, line: line)
            XCTAssertNoThrow(try singleton.enqueue(prompt: prompt), file: file, line: line)
        }
        packed.run(steps: steps)
        singleton.run(steps: steps)

        // Test premise: packing really happened on one side and never on
        // the other.
        XCTAssertTrue(
            packed.model.prefillShapes.contains(expectPackedShape),
            "test premise: expected a packed \(expectPackedShape) prompt forward, saw "
                + "\(packed.model.prefillShapes)",
            file: file, line: line)
        XCTAssertTrue(
            singleton.model.packedPrefillShapes.isEmpty,
            "test premise: the reference run must stay on the [1, chunk] path, saw "
                + "\(singleton.model.packedPrefillShapes)",
            file: file, line: line)

        for (index, id) in packed.order.enumerated() {
            let referenceID = singleton.order[index]
            let packedTokens = packed.generated[id] ?? []
            let singletonTokens = singleton.generated[referenceID] ?? []
            XCTAssertFalse(
                packedTokens.isEmpty,
                "row \(index): test premise — the row must have sampled something",
                file: file, line: line)
            XCTAssertEqual(
                packedTokens, singletonTokens,
                "row \(index): packed execution must sample exactly the singleton tokens",
                file: file, line: line)
            // ABSOLUTE (not just relative) check that a packed chunk still
            // performs every K/V write and advances every offset: the row
            // consumed its whole prompt plus one position per decode step.
            let expectedOffset = prompts[index].count + max(0, packedTokens.count - 1)
            for (layer, state) in packed.kv(id).enumerated() {
                guard let state else { continue }
                XCTAssertEqual(
                    state.absoluteOffset, expectedOffset,
                    "row \(index) layer \(layer): a packed prefill must advance the offset "
                        + "through every prompt token",
                    file: file, line: line)
            }
            assertKVBitIdentical(
                packed.kv(id), singleton.kv(referenceID), row: "row \(index)",
                file: file, line: line)
        }
        return (packed, singleton)
    }

    // MARK: 1. Packing actually happens

    /// Four requests with EQUAL prompt lengths are admitted in one step and
    /// must reach the model as ONE `[4, chunk]` forward — for the
    /// intermediate chunk (`.evaluationOnly`) AND for the frontier chunk
    /// (`.lastPositionLogits`) — never as four `[1, chunk]` forwards.
    func testFourEqualPromptRowsRunAsOneRectangularForward() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let harness = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        for seed in 0 ..< 4 {
            try harness.enqueue(prompt: makePromptTokens(length: 16, seed: UInt64(2000 + seed)))
        }

        // Step 1 — all four admitted, chunk [0, 8): intermediate.
        harness.model.reset()
        harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [.prefill(shape: [4, 8], requirement: .evaluationOnly, spliced: false)],
            "four equal intermediate chunks must be ONE rectangular [4, 8] forward")

        // Step 2 — chunk [8, 16): the frontier, still four equal rows.
        harness.model.reset()
        let sampled = harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [.prefill(shape: [4, 8], requirement: .lastPositionLogits, spliced: false)],
            "four equal frontier chunks must be ONE rectangular [4, 8] forward")
        XCTAssertEqual(sampled.count, 4, "every packed frontier row samples exactly one token")
        XCTAssertEqual(
            Set(sampled.map(\.id)), Set(harness.order),
            "every request must receive its own sampled token")
    }

    // MARK: 2. Per-row parity — the load-bearing test

    func testPackedRowsAreParityWithSingletonExecutionB2() throws {
        // Prompt 24 / chunk 8 ⇒ 3 prompt chunks, the last sampling.
        let prompts = (0 ..< 2).map { makePromptTokens(length: 24, seed: UInt64(3100 + $0)) }
        assertParity(
            prompts: prompts, chunkSize: 8, budget: 64, steps: 3,
            expectPackedShape: [2, 8])
    }

    func testPackedRowsAreParityWithSingletonExecutionB4() throws {
        let prompts = (0 ..< 4).map { makePromptTokens(length: 24, seed: UInt64(3200 + $0)) }
        assertParity(
            prompts: prompts, chunkSize: 8, budget: 64, steps: 3,
            expectPackedShape: [4, 8])
    }

    // MARK: 3. No cross-row contamination

    /// Rows with DIFFERENT prompt contents in one packed group must each get
    /// their own result. `TinyTestModel` is a real transformer, so a row's
    /// first token is a function of its own tokens — if a packed row picked
    /// up a neighbour's hidden state or KV, this diverges from the
    /// row-at-a-time reference.
    func testPackedRowsWithDifferentPromptsDoNotContaminateEachOther() throws {
        let prompts = (0 ..< 4).map { makePromptTokens(length: 24, seed: UInt64(4100 + $0)) }
        XCTAssertEqual(
            Set(prompts.map { $0.description }).count, prompts.count,
            "test premise: the four prompts must differ")
        let (packed, _) = assertParity(
            prompts: prompts, chunkSize: 8, budget: 64, steps: 3,
            expectPackedShape: [4, 8])

        // …and the outcome really is prompt-dependent, so parity is not
        // vacuous (a bug that copied row 0 everywhere would collapse these).
        let firstTokens = packed.order.map { packed.generated[$0]?.first ?? -1 }
        XCTAssertTrue(
            Set(firstTokens).count > 1,
            "test premise: the fixture's first token must depend on the row's own prompt "
                + "(got \(firstTokens)) — otherwise a cross-row leak would be invisible")
    }

    /// Rows packed together at DIFFERENT pre-existing absolute offsets: a
    /// stale/shared position offset or a KV view that spans the cohort
    /// shows up here as a token or KV mismatch against the singleton run.
    func testPackedRowsWithDifferentCacheOffsetsStayIndependent() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let (packed, singleton) = makePair(
            base: base, prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        let promptA = makePromptTokens(length: 32, seed: 4200)
        let promptB = makePromptTokens(length: 24, seed: 4201)

        for harness in [packed, singleton] {
            // A alone first: it reaches offset 8 before B ever runs.
            try harness.enqueue(prompt: promptA)
            harness.step()
            try harness.enqueue(prompt: promptB)
        }

        let aID = packed.order[0]
        let bID = packed.order[1]
        XCTAssertEqual(
            packed.absoluteOffsets(aID).first, 8,
            "test premise: row A must already hold 8 tokens of KV")
        XCTAssertTrue(
            packed.kv(bID).isEmpty,
            "test premise: row B has no KV state allocated before its first step")

        // Step 2 packs A's [8, 16) chunk with B's [0, 8) chunk: same length,
        // same samples flag, DIFFERENT absolute offsets.
        packed.model.reset()
        packed.step()
        singleton.step()
        XCTAssertEqual(
            packed.model.prefillShapes, [[2, 8]],
            "equal-length chunks at different offsets must still pack")

        packed.run(steps: 4)
        singleton.run(steps: 4)

        for (index, id) in packed.order.enumerated() {
            XCTAssertEqual(
                packed.generated[id] ?? [], singleton.generated[singleton.order[index]] ?? [],
                "row \(index): a packed row at its own offset must sample the singleton tokens")
            assertKVBitIdentical(
                packed.kv(id), singleton.kv(singleton.order[index]),
                row: "offset row \(index)")
        }
    }

    // MARK: 4. Grouping rules

    /// Rows with DIFFERENT chunk lengths are never coalesced — and the
    /// leftover single-row group falls through to the singleton path.
    func testDifferentChunkLengthsAreNotPackedTogether() throws {
        // Budget 20 with chunk 8: rows 1 and 2 take 8 tokens each, row 3
        // takes the remaining 4. All three are intermediate chunks
        // (prompt 24), so ONLY the length differs.
        let base = TinyTestModel.make(seed: weightSeed)
        let harness = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 20)
        for seed in 0 ..< 3 {
            try harness.enqueue(prompt: makePromptTokens(length: 24, seed: UInt64(5100 + seed)))
        }

        harness.model.reset()
        harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [
                .prefill(shape: [2, 8], requirement: .evaluationOnly, spliced: false),
                .prefill(shape: [1, 4], requirement: .evaluationOnly, spliced: false),
            ],
            "only the two 8-token chunks may pack; the 4-token chunk must stay on the "
                + "singleton path")
    }

    /// Rows with the same chunk length but different `samples` flags
    /// (frontier vs intermediate) must NOT be packed: they need different
    /// `CBv2PrefillRequirement`s.
    func testDifferentSamplesFlagsAreNotPackedTogether() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let harness = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        // 16 tokens ⇒ chunk [0, 8) is intermediate; 8 tokens ⇒ chunk [0, 8)
        // IS the frontier. Same length, different samples flag.
        try harness.enqueue(prompt: makePromptTokens(length: 16, seed: 5200))
        try harness.enqueue(prompt: makePromptTokens(length: 8, seed: 5201))

        harness.model.reset()
        let sampled = harness.step()
        XCTAssertEqual(
            harness.model.prefillShapes, [[1, 8], [1, 8]],
            "an intermediate chunk and a frontier chunk of equal length must not pack")
        XCTAssertEqual(
            harness.model.calls,
            [
                .prefill(shape: [1, 8], requirement: .evaluationOnly, spliced: false),
                .prefill(shape: [1, 8], requirement: .lastPositionLogits, spliced: false),
            ],
            "each row must ask for its OWN requirement")
        XCTAssertEqual(sampled.count, 1, "only the frontier row samples this step")
        XCTAssertEqual(sampled.first?.id, harness.order[1])
    }

    /// A group of exactly one row is left to the per-request loop.
    func testSingleRowGroupTakesTheSingletonPath() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let harness = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        try harness.enqueue(prompt: makePromptTokens(length: 24, seed: 5300))

        harness.model.reset()
        harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [.prefill(shape: [1, 8], requirement: .evaluationOnly, spliced: false)],
            "a lone prompt row must run as [1, chunk], never as a degenerate [1, chunk] pack "
                + "through a different code path")
    }

    // MARK: 5. Fail-closed capability gates

    /// (a) The cache provider vetoes: no packing, even though the model
    /// advertises it and the caches are genuinely contiguous.
    func testCacheProviderVetoPreventsPacking() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let vetoed = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            modelClaimsPacking: true, providerClaimsPacking: false,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        XCTAssertTrue(
            vetoed.provider.bank.supportsPackedPrefill,
            "test premise: the underlying bank IS capable — only the claim is withheld")

        let prompts = (0 ..< 4).map { makePromptTokens(length: 24, seed: UInt64(6100 + $0)) }
        for prompt in prompts { try vetoed.enqueue(prompt: prompt) }
        vetoed.run(steps: 3)

        XCTAssertEqual(
            vetoed.model.prefillShapes, Array(repeating: [1, 8], count: 12),
            "a provider that does not vouch for row independence must never be packed into")
        XCTAssertTrue(vetoed.model.packedPrefillShapes.isEmpty)

        // …and it still produces the reference tokens.
        let reference = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds, modelClaimsPacking: false,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        for prompt in prompts { try reference.enqueue(prompt: prompt) }
        reference.run(steps: 3)
        for (index, id) in vetoed.order.enumerated() {
            XCTAssertEqual(
                vetoed.generated[id] ?? [],
                reference.generated[reference.order[index]] ?? [],
                "row \(index): vetoing packing must not change the result")
        }
    }

    /// (b) The model vetoes: no packing, even though the provider vouches.
    func testModelVetoPreventsPacking() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let vetoed = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            modelClaimsPacking: false, providerClaimsPacking: true,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        for seed in 0 ..< 4 {
            try vetoed.enqueue(prompt: makePromptTokens(length: 24, seed: UInt64(6200 + seed)))
        }
        vetoed.run(steps: 3)
        XCTAssertEqual(
            vetoed.model.prefillShapes, Array(repeating: [1, 8], count: 12),
            "a model that does not claim packed prefill must only ever see [1, chunk]")
    }

    /// A model that conforms only to `CBv2PrefillSteppableModel` (no packed
    /// refinement at all) is likewise never packed — the default is closed.
    func testNonConformingModelIsNeverPacked() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let narrowing = CBv2PrefillNarrowingModel(base)
        XCTAssertFalse(
            narrowing is CBv2PackedPrefillSteppableModel,
            "test premise: the fixture must not conform to the packed refinement")

        let scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 8, maxBatchedTokensPerStep: 64, prefillChunkSize: 8))
        let loop = EngineLoopV2(
            model: narrowing,
            layerKinds: base.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: base.layerKinds),
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2NullDetokenizerFactory(),
            scheduler: scheduler,
            capacity: nil,
            config: CBv2EngineLoopConfig(),
            gauges: CBv2EngineGauges(kvBytesCapacity: 1 << 27))
        for raw in 1 ... 4 {
            _ = try scheduler.enqueue(
                CBv2Request(
                    id: CBv2RequestID(UInt64(raw)),
                    promptTokens: makePromptTokens(length: 24, seed: UInt64(6300 + raw)),
                    sampling: .init(temperature: 0), maxTokens: 8))
        }
        _ = loop.executeMixed(scheduler.plan())

        let shapes = narrowing.log.compactMap { entry -> [Int]? in
            if case .prefill(let call) = entry { return call.tokenShape }
            return nil
        }
        XCTAssertEqual(
            shapes, Array(repeating: [1, 8], count: 4),
            "the default `supportsPackedPrefill` is false — a merely prefill-capable model "
                + "must keep the per-request path")
    }

    /// (c) Without the stronger packed-multimodal model claim, a span chunk
    /// stays singleton even beside same-shape text rows.
    func testSpanBearingChunkFallsBackWithoutMultimodalClaim() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let harness = CBv2PackedHarness(
            inner: base, layerKinds: base.layerKinds,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)

        // Span [2, 6) lies wholly inside the first 8-token chunk, so the
        // vision row's chunk is EXACTLY 8 tokens — the same length, and the
        // same (intermediate) requirement, as its two text neighbours.
        let spans = [CBv2ImageSpan(tokenOffset: 2, length: 4)]
        let (visionPrompt, images) = CBv2VisionFixtures.make(
            model: base, length: 16, spans: spans, seed: 6400)
        let resolved = try CBv2MultimodalPlan.resolve(
            CBv2VisionFixtures.input(images),
            promptTokenCount: visionPrompt.count,
            model: harness.model,
            cacheProvider: harness.provider,
            maxBatchedTokensPerStep: 64)

        try harness.enqueue(prompt: makePromptTokens(length: 16, seed: 6401))
        try harness.enqueue(prompt: visionPrompt, multimodal: resolved)
        try harness.enqueue(prompt: makePromptTokens(length: 16, seed: 6402))

        harness.model.reset()
        harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [
                .prefill(shape: [2, 8], requirement: .evaluationOnly, spliced: false),
                .prefill(shape: [1, 8], requirement: .evaluationOnly, spliced: true),
            ],
            "the two text rows pack; the span-bearing chunk must stay per-request and arrive "
                + "with spliced embeddings")

        // The vision row's SECOND chunk carries no span, so it becomes
        // packable again alongside its neighbours — the exclusion is a pure
        // function of has-spans, not of the request.
        harness.model.reset()
        harness.step()
        XCTAssertEqual(
            harness.model.calls,
            [.prefill(shape: [3, 8], requirement: .lastPositionLogits, spliced: false)],
            "a span-free chunk of a multimodal request is packable like any text chunk")
    }

    /// `CBv2LayerCacheBank` vouches ONLY when every cache is the contiguous
    /// `CBv2LayerCache` — one foreign cache closes the gate.
    func testLayerCacheBankVouchesOnlyForContiguousCaches() {
        let kinds = CBv2SchedFixtures.tinyLayerKinds(layers: 2)
        let contiguous = CBv2LayerCacheBank(layerKinds: kinds)
        XCTAssertTrue(
            contiguous.supportsPackedPrefill,
            "a bank of contiguous CBv2LayerCaches keeps rows independent")
        XCTAssertTrue(
            contiguous.supportsPackedMultimodalSpans,
            "contiguous caches bind one optional span context per row")

        let mixed = CBv2LayerCacheBank(caches: [
            CBv2LayerCache(layerIndex: 0, kind: kinds[0]),
            CBv2SchedMockLayerCache(layerIndex: 1, kind: kinds[1], rows: []),
        ])
        XCTAssertFalse(
            mixed.supportsPackedPrefill,
            "one cache that makes no per-row claim must veto packing for the whole bank")
        XCTAssertFalse(
            mixed.supportsPackedMultimodalSpans,
            "one cache without row-aligned span binding must veto packed vision")
    }

    // MARK: 6. Decode is untouched

    /// A mixed step of decode rows + packable prompt rows must keep decode
    /// on its own rectangular `[Bdecode, 1]` forward, and packing must not
    /// change the decode tokens.
    func testDecodeRowsKeepTheirOwnRectangularBatchAlongsidePacking() throws {
        let base = TinyTestModel.make(seed: weightSeed)
        let (packed, singleton) = makePair(
            base: base, prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        let decodePrompts = (0 ..< 2).map { makePromptTokens(length: 4, seed: UInt64(7100 + $0)) }
        let promptRows = (0 ..< 2).map { makePromptTokens(length: 24, seed: UInt64(7200 + $0)) }

        for harness in [packed, singleton] {
            // Two short requests prefill (and sample) in step 1, so they are
            // decode-ready in step 2.
            for prompt in decodePrompts { try harness.enqueue(prompt: prompt) }
            harness.step()
            for prompt in promptRows { try harness.enqueue(prompt: prompt) }
        }

        packed.model.reset()
        let packedSampled = packed.step()
        singleton.model.reset()
        let singletonSampled = singleton.step()

        XCTAssertEqual(
            packed.model.forwardShapes, [[2, 1]],
            "the two decode rows must run as ONE rectangular [2, 1] forward")
        XCTAssertEqual(
            packed.model.prefillShapes, [[2, 8]],
            "the two equal prompt chunks must pack into one [2, 8] forward")
        XCTAssertEqual(
            singleton.model.forwardShapes, [[2, 1]],
            "the reference run decodes identically")
        XCTAssertEqual(
            singleton.model.prefillShapes, [[1, 8], [1, 8]],
            "test premise: the reference run does not pack")

        let packedDecode = packedSampled.map(\.token)
        let singletonDecode = singletonSampled.map(\.token)
        XCTAssertEqual(packedSampled.count, 2, "only the two decode rows sample this step")
        XCTAssertEqual(
            packedDecode, singletonDecode,
            "packing the prompt rows must not change the decode rows' tokens")

        // Continue the mixed run: every row must still agree.
        packed.run(steps: 5)
        singleton.run(steps: 5)
        for (index, id) in packed.order.enumerated() {
            XCTAssertEqual(
                packed.generated[id] ?? [], singleton.generated[singleton.order[index]] ?? [],
                "row \(index): mixed decode+packed-prefill stream must match the reference")
            assertKVBitIdentical(
                packed.kv(id), singleton.kv(singleton.order[index]), row: "mixed row \(index)")
        }
    }

    // MARK: 7. Continuation after a packed prefill

    func testContinuationAfterPackedPrefillMatchesSingletonReference() throws {
        // 3 prefill steps (prompt 24 / chunk 8) then 8 chained decode steps.
        let prompts = (0 ..< 4).map { makePromptTokens(length: 24, seed: UInt64(8100 + $0)) }
        let (packed, _) = assertParity(
            prompts: prompts, chunkSize: 8, budget: 64, steps: 11,
            expectPackedShape: [4, 8])
        for id in packed.order {
            XCTAssertEqual(
                (packed.generated[id] ?? []).count, 9,
                "one frontier token + 8 decode tokens per row")
        }
        // Decode after the packed prefill is rectangular over all four rows.
        XCTAssertEqual(
            packed.model.forwardShapes, Array(repeating: [4, 1], count: 8),
            "post-prefill decode must be a plain [4, 1] batch")
    }

    // MARK: 8. The real conformer: Gemma 4 behind the steppable adapter

    /// Tiny 4-layer Gemma 4 (random weights, vocab 64, two KV-shared
    /// layers): the adapter must advertise the model's claim, and packed
    /// execution must be per-row parity with the singleton path.
    func testGemma4AdapterAdvertisesAndHonorsPackedPrefill() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 16,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6
            }
            """
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(0xB0A7)
        let gemma = Gemma4TextModel(config)
        eval(gemma)

        XCTAssertTrue(
            gemma.cbv2SupportsPackedPrefill,
            "Gemma4TextModel claims rectangular-prompt safety")
        XCTAssertTrue(
            gemma.cbv2SupportsPackedMultimodalPrefill,
            "Gemma4TextModel explicitly claims row-local embedding/span safety")
        let adapter = CBv2SteppableLanguageModelAdapter(gemma)
        XCTAssertTrue(
            adapter.supportsPackedPrefill,
            "the steppable adapter must propagate the model's claim")
        XCTAssertTrue(
            adapter.supportsPackedMultimodalPrefill,
            "the steppable adapter must propagate the stronger multimodal claim")

        let kinds = config.cbv2LayerKinds
        let prompts = (0 ..< 2).map {
            makePromptTokens(length: 24, seed: UInt64(9100 + $0), vocabSize: 64)
        }

        let packed = CBv2PackedHarness(
            inner: adapter, layerKinds: kinds, modelClaimsPacking: true,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        let singleton = CBv2PackedHarness(
            inner: adapter, layerKinds: kinds, modelClaimsPacking: false,
            prefillChunkSize: 8, maxBatchedTokensPerStep: 64)
        for prompt in prompts {
            try packed.enqueue(prompt: prompt)
            try singleton.enqueue(prompt: prompt)
        }
        packed.run(steps: 6)
        singleton.run(steps: 6)

        XCTAssertTrue(
            packed.model.prefillShapes.contains([2, 8]),
            "test premise: Gemma 4's equal prompt chunks must pack, saw "
                + "\(packed.model.prefillShapes)")
        XCTAssertTrue(singleton.model.packedPrefillShapes.isEmpty)

        for (index, id) in packed.order.enumerated() {
            XCTAssertEqual(
                packed.generated[id] ?? [], singleton.generated[singleton.order[index]] ?? [],
                "Gemma 4 row \(index): packed tokens must match the singleton run")
            assertKVBitIdentical(
                packed.kv(id), singleton.kv(singleton.order[index]),
                row: "gemma4 row \(index)")
        }
    }
}
