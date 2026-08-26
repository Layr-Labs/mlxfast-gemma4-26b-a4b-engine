// CBv2PrefillOutputTests.swift — the prompt-output seam (PrefillOutputV2.swift).
//
// `EngineLoopV2.prefillOutput` is an OPT-IN narrowing of the prompt forward:
// models conforming to `CBv2PrefillSteppableModel` return only what the
// engine consumes (a [B, 1] evaluation handle for intermediate chunks,
// [B, vocab] for the frontier chunk); everything else keeps the established
// full-logits `forward` and is sliced by the engine.
//
// This suite pins the seam from both sides:
//
//  1. `testFallbackSlicingMatchesForwardExactly` — a model that conforms
//     ONLY to `CBv2SteppableModel` gets byte-identical output to the old
//     slice-after-forward (`forward(...)[0..., -1, 0...]` /
//     `[0..., -1, 0 ..< 1]`).
//  2. `testEngineRequestsEvaluationOnlyForIntermediateChunksAndLogitsAtFrontier`
//     — a conforming model records exactly one `prefill` per prompt chunk,
//     `.evaluationOnly` for every intermediate chunk and
//     `.lastPositionLogits` for the frontier chunk.
//  3. `testNarrowedPathIsTokenIdenticalToFallback` — greedy token parity
//     between a genuinely narrowing model (frontier logits projected from
//     the LAST hidden row only; no vocabulary projection at all for
//     intermediate chunks) and the full-logits fallback.
//  4. `testNarrowedPathLeavesIdenticalKVState` — every layer cache's
//     offset/retained count/K/V contents match after prefill, and the next
//     decode step samples the same token.
//  5. `testDecodePathNeverCallsPrefill` (+ the compiled-decode twin) —
//     decode is out of scope for the seam: zero `prefill` calls once prefill
//     is done, decode still routes through `forward`.
//  6. `testMultimodalChunkReachesPrefillOutputWithSplicedEmbeddings` —
//     a span-bearing chunk arrives at `prefillOutput` with non-nil spliced
//     embeddings and the right requirement (both an intermediate span chunk
//     and a frontier span chunk).
//
// Fixtures reuse `TinyTestModel` / `HarnessLayerCache` / `HarnessKVBackend`
// (CBv2Fixtures.swift), `CBv2SchedMockBackend` / `CBv2SchedMockCacheProvider`
// (CBv2SchedulerTestSupport.swift) and the vision fixtures + collectors from
// CBv2MultimodalTests.swift / CBv2SchedulerTestSupport.swift. No model
// downloads: weights are seeded random.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

// MARK: - Bit-identity helper

/// Exact equality (no tolerance): these paths must be byte-neutral, not
/// merely close.
private func cbv2BitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape else { return false }
    return (a .== b).all().item(Bool.self)
}

// MARK: - Fixture models over TinyTestModel's weights

/// Recorded `prefill(...)` invocation.
struct CBv2PrefillCall {
    let requirement: CBv2PrefillRequirement
    let tokenShape: [Int]
    let inputEmbeddings: MLXArray?
    let tokens: MLXArray
}

/// One entry of the model-call log, so ORDER (prefill during prompt, forward
/// during decode) is assertable.
enum CBv2PrefillLogEntry {
    case forward(shape: [Int])
    case prefill(CBv2PrefillCall)
}

/// A steppable model that is deliberately NOT `CBv2PrefillSteppableModel`:
/// the engine must slice its full logits itself. Multimodal-capable so the
/// same fallback covers the embedding path.
class CBv2PrefillFallbackModel: CBv2SteppableModel, CBv2MultimodalSteppableModel,
    @unchecked Sendable
{
    let base: TinyTestModel
    private let lock = NSLock()
    private var _log: [CBv2PrefillLogEntry] = []

    init(_ base: TinyTestModel) { self.base = base }

    var log: [CBv2PrefillLogEntry] {
        lock.lock()
        defer { lock.unlock() }
        return _log
    }

    func record(_ entry: CBv2PrefillLogEntry) {
        lock.lock()
        _log.append(entry)
        lock.unlock()
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        record(.forward(shape: tokens.shape))
        return base.forward(tokens: tokens, caches: caches)
    }

    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        base.embedPromptTokens(tokens)
    }

    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        record(.forward(shape: tokens.shape))
        return base.forward(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
    }
}

/// The opt-in conformer. GENUINELY narrows:
///  - `.evaluationOnly` runs the whole trunk (every K/V write, every offset
///    advance) and returns `finalNorm(h)[0..., -1, 0 ..< 1]` — the vocabulary
///    projection is never built;
///  - `.lastPositionLogits` projects ONLY the last hidden row through the
///    LM head, returning [B, vocab].
///
/// The trunk is a faithful replica of `TinyTestModel`'s private
/// `forwardV2(hidden:caches:)` (validated bit-exactly against
/// `TinyTestModel.forward` by `testNarrowingFixtureTrunkMatchesFullForward`),
/// so any KV/numeric divergence is the seam's, not the fixture's.
class CBv2PrefillNarrowingModel: CBv2PrefillFallbackModel, CBv2PrefillSteppableModel {

    func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        record(
            .prefill(
                CBv2PrefillCall(
                    requirement: requirement, tokenShape: tokens.shape,
                    inputEmbeddings: inputEmbeddings, tokens: tokens)))
        let hidden = inputEmbeddings ?? base.embed(tokens)
        let normed = trunk(hidden: hidden, caches: caches)
        switch requirement {
        case .evaluationOnly:
            // No vocabulary projection at all — a [B, 1] handle whose graph
            // still transitively covers the whole chunk (and its KV writes).
            return normed[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            // LM head on the LAST ROW ONLY ⇒ [B, vocab].
            return base.lmHead(normed[0..., -1, 0...])
        }
    }

    /// `TinyTestModel.forwardV2(hidden:caches:)` minus the LM head.
    func trunk(hidden: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let kinds = base.layerKinds
        precondition(caches.count == base.blocks.count)
        var h = hidden
        // KV-shared layers reuse the SOURCE layer's PRE-update offsets.
        let sharedSourceOffsets: [Int: MLXArray] = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().compactMap { index, kind in
                kind.sharesKVWithLayer.map { source in (index, caches[source].positionOffsets) }
            })
        for (i, block) in base.blocks.enumerated() {
            if let source = kinds[i].sharesKVWithLayer {
                h = block.forwardV2Borrowing(
                    h, cache: caches[i], source: caches[source],
                    sourceOffsets: sharedSourceOffsets[i]!)
            } else {
                h = block.forwardV2(h, cache: caches[i])
            }
        }
        return base.finalNorm(h)
    }
}

// MARK: - Synchronous prefill/decode driver (harness caches, greedy)

/// Drives `EngineLoopV2.prefillOutput` chunk-by-chunk over the harness layer
/// caches, then greedy-decodes through `CBv2SteppableModel.forward` — the
/// same discipline `executeMixed` uses, minus the async engine, so KV state
/// is directly inspectable.
private final class CBv2PrefillOutputDriver {
    let loop: EngineLoopV2
    let model: CBv2SteppableModel
    let layerKinds: [CBv2LayerKind]
    let backend: HarnessKVBackend
    private(set) var kv: [CBv2SequenceKV?] = []
    private(set) var caches: [HarnessLayerCache] = []

    init(model: CBv2SteppableModel, layerKinds: [CBv2LayerKind]) {
        self.model = model
        self.layerKinds = layerKinds
        self.backend = HarnessKVBackend()
        self.loop = CBv2PrefillOutputDriver.makeLoop(model: model, layerKinds: layerKinds)
    }

    /// A loop instance built purely for its `prefillOutput` seam: never
    /// started, so nothing else in it runs.
    static func makeLoop(
        model: CBv2SteppableModel, layerKinds: [CBv2LayerKind]
    ) -> EngineLoopV2 {
        EngineLoopV2(
            model: model,
            layerKinds: layerKinds,
            backend: CBv2SchedMockBackend(),
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: layerKinds),
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2NullDetokenizerFactory(),
            scheduler: SchedulerV2(config: CBv2SchedulerConfig()),
            capacity: nil,
            config: CBv2EngineLoopConfig(),
            gauges: CBv2EngineGauges(kvBytesCapacity: 1 << 30))
    }

    /// Chunked prefill of the WHOLE prompt through `prefillOutput`.
    /// Returns the frontier chunk's [1, vocab] logits.
    @discardableResult
    func prefill(prompt: [Int], chunkSize: Int) throws -> MLXArray {
        kv = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: prompt.count,
            maxLength: prompt.count + 64)
        caches = layerKinds.enumerated().map { index, kind in
            HarnessLayerCache(
                layerIndex: index, kind: kind,
                rows: kind.sharesKVWithLayer == nil ? [kv[index]!] : [])
        }
        var frontier: MLXArray?
        var start = 0
        while start < prompt.count {
            let count = min(chunkSize, prompt.count - start)
            let isFrontier = start + count == prompt.count
            let tokens = MLXArray(prompt[start ..< start + count].map(Int32.init))
                .reshaped(1, count)
            let output = loop.prefillOutput(
                tokens: tokens, inputEmbeddings: nil,
                caches: caches.map { $0 as CBv2AttendingLayerCache },
                requirement: isFrontier ? .lastPositionLogits : .evaluationOnly)
            eval(output)
            if isFrontier { frontier = output }
            start += count
        }
        return try XCTUnwrap(frontier)
    }

    /// Greedy continuation: the frontier logits give the first token, then
    /// `steps - 1` rectangular [1, 1] decode forwards.
    func greedy(frontier: MLXArray, steps: Int) -> [Int] {
        var generated: [Int] = []
        var next = Int(argMax(frontier, axis: -1).asArray(Int32.self)[0])
        generated.append(next)
        for _ in 1 ..< steps {
            let tokens = MLXArray([Int32(next)]).reshaped(1, 1)
            let logits = model.forward(
                tokens: tokens, caches: caches.map { $0 as CBv2AttendingLayerCache })
            next = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
            generated.append(next)
        }
        return generated
    }

    /// One [1, 1] decode step without recording it in `generated` — used to
    /// compare the first post-prefill token across paths.
    func nextDecodeToken(after token: Int) -> Int {
        let tokens = MLXArray([Int32(token)]).reshaped(1, 1)
        let logits = model.forward(
            tokens: tokens, caches: caches.map { $0 as CBv2AttendingLayerCache })
        return Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
    }
}

// MARK: - Suite

final class CBv2PrefillOutputTests: XCTestCase {

    private let hiddenSeed: UInt64 = 0xC0FFEE

    /// Fresh harness caches over a fresh KV state (so two runs of the same
    /// forward are directly comparable).
    private func freshCaches(
        _ layerKinds: [CBv2LayerKind], backend: HarnessKVBackend, promptLength: Int
    ) throws -> [CBv2AttendingLayerCache] {
        let kv = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: promptLength + 8)
        return layerKinds.enumerated().map { index, kind in
            HarnessLayerCache(
                layerIndex: index, kind: kind,
                rows: kind.sharesKVWithLayer == nil ? [kv[index]!] : [])
        }
    }

    // MARK: 0. Fixture self-check

    /// The narrowing fixture's trunk replica must be bit-exact vs
    /// `TinyTestModel.forward`, otherwise tests 3/4 would measure the
    /// fixture, not the seam.
    func testNarrowingFixtureTrunkMatchesFullForward() throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let narrowing = CBv2PrefillNarrowingModel(base)
        let backend = HarnessKVBackend()
        let prompt = makePromptTokens(length: 12, seed: 1)
        let tokens = MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count)

        let cachesA = try freshCaches(base.layerKinds, backend: backend, promptLength: 12)
        let reference = base.forward(tokens: tokens, caches: cachesA)

        let cachesB = try freshCaches(base.layerKinds, backend: backend, promptLength: 12)
        let replica = base.lmHead(narrowing.trunk(hidden: base.embed(tokens), caches: cachesB))

        eval(reference, replica)
        XCTAssertTrue(
            cbv2BitIdentical(reference, replica),
            "the fixture's trunk replica must reproduce TinyTestModel.forward exactly")
    }

    // MARK: 1. Fallback equivalence — the engine slices, byte-for-byte

    func testFallbackSlicingMatchesForwardExactly() throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let fallback = CBv2PrefillFallbackModel(base)
        XCTAssertFalse(
            fallback is CBv2PrefillSteppableModel,
            "test premise: the fallback fixture must NOT conform to the prefill protocol")

        let loop = CBv2PrefillOutputDriver.makeLoop(
            model: fallback, layerKinds: base.layerKinds)
        let backend = HarnessKVBackend()

        for chunkLength in [1, 5, 13] {
            let prompt = makePromptTokens(length: chunkLength, seed: UInt64(100 + chunkLength))
            let tokens = MLXArray(prompt.map(Int32.init)).reshaped(1, chunkLength)

            // Reference: the pre-seam behavior — full logits, then slice.
            let refCaches = try freshCaches(
                base.layerKinds, backend: backend, promptLength: chunkLength)
            let full = fallback.forward(tokens: tokens, caches: refCaches)
            let refLast = full[0..., -1, 0...]
            let refEval = full[0..., -1, 0 ..< 1]

            let logitsCaches = try freshCaches(
                base.layerKinds, backend: backend, promptLength: chunkLength)
            let seamLast = loop.prefillOutput(
                tokens: tokens, inputEmbeddings: nil, caches: logitsCaches,
                requirement: .lastPositionLogits)

            let evalCaches = try freshCaches(
                base.layerKinds, backend: backend, promptLength: chunkLength)
            let seamEval = loop.prefillOutput(
                tokens: tokens, inputEmbeddings: nil, caches: evalCaches,
                requirement: .evaluationOnly)

            eval(refLast, refEval, seamLast, seamEval)

            XCTAssertEqual(
                seamLast.shape, [1, base.config.vocabSize],
                ".lastPositionLogits must be [B, vocab] (L=\(chunkLength))")
            XCTAssertEqual(
                seamEval.shape, [1, 1],
                ".evaluationOnly must be [B, 1] (L=\(chunkLength))")
            XCTAssertTrue(
                cbv2BitIdentical(seamLast, refLast),
                ".lastPositionLogits must equal forward(...)[0..., -1, 0...] byte-for-byte "
                    + "(L=\(chunkLength))")
            XCTAssertTrue(
                cbv2BitIdentical(seamEval, refEval),
                ".evaluationOnly must equal forward(...)[0..., -1, 0 ..< 1] byte-for-byte "
                    + "(L=\(chunkLength))")
        }
    }

    /// The multimodal (spliced-embedding) fallback slices the SAME way.
    func testFallbackSlicingMatchesEmbeddingForwardExactly() throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let fallback = CBv2PrefillFallbackModel(base)
        let loop = CBv2PrefillOutputDriver.makeLoop(
            model: fallback, layerKinds: base.layerKinds)
        let backend = HarnessKVBackend()

        let length = 9
        let prompt = makePromptTokens(length: length, seed: 321)
        let tokens = MLXArray(prompt.map(Int32.init)).reshaped(1, length)
        MLXRandom.seed(0xBEEF)
        let embeddings = MLXRandom.normal([1, length, base.config.hiddenSize]) * 0.6
        eval(embeddings)

        let refCaches = try freshCaches(base.layerKinds, backend: backend, promptLength: length)
        let full = fallback.forward(
            tokens: tokens, inputEmbeddings: embeddings, caches: refCaches)
        let refLast = full[0..., -1, 0...]
        let refEval = full[0..., -1, 0 ..< 1]

        let lastCaches = try freshCaches(base.layerKinds, backend: backend, promptLength: length)
        let seamLast = loop.prefillOutput(
            tokens: tokens, inputEmbeddings: embeddings, caches: lastCaches,
            requirement: .lastPositionLogits)
        let evalCaches = try freshCaches(base.layerKinds, backend: backend, promptLength: length)
        let seamEval = loop.prefillOutput(
            tokens: tokens, inputEmbeddings: embeddings, caches: evalCaches,
            requirement: .evaluationOnly)

        eval(refLast, refEval, seamLast, seamEval)
        XCTAssertTrue(
            cbv2BitIdentical(seamLast, refLast),
            "embedding-path .lastPositionLogits must equal the sliced embedding forward")
        XCTAssertTrue(
            cbv2BitIdentical(seamEval, refEval),
            "embedding-path .evaluationOnly must equal the sliced embedding forward")
    }

    // MARK: 2. The opt-in path is actually taken, once per chunk

    func testEngineRequestsEvaluationOnlyForIntermediateChunksAndLogitsAtFrontier()
        async throws
    {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let model = CBv2PrefillNarrowingModel(base)
        // Prompt 24 / chunk 8 ⇒ exactly three prompt chunks, the last one
        // reaching the frontier (24 % 8 == 0 keeps a trailing single token
        // from being planned as a decode row instead).
        let prompt = makePromptTokens(length: 24, seed: 700)
        let engine = makeEngine(model: model, base: base, prefillChunkSize: 8)

        let collected = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: 4)))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)

        let prefills = model.log.compactMap { entry -> CBv2PrefillCall? in
            if case .prefill(let call) = entry { return call }
            return nil
        }
        XCTAssertEqual(
            prefills.count, 3,
            "one prefill(...) per prompt chunk — no chunk may bypass the seam, none may "
                + "be visited twice")
        XCTAssertEqual(
            prefills.map(\.requirement),
            [.evaluationOnly, .evaluationOnly, .lastPositionLogits],
            "intermediate chunks must ask for .evaluationOnly; the frontier chunk for "
                + ".lastPositionLogits")
        XCTAssertEqual(
            prefills.map(\.tokenShape), [[1, 8], [1, 8], [1, 8]],
            "prompt chunks are [1, chunk]")
        XCTAssertTrue(
            prefills.allSatisfy { $0.inputEmbeddings == nil },
            "a text-only request must never carry spliced embeddings")
        XCTAssertEqual(
            prefills.filter { $0.requirement == .lastPositionLogits }.count, 1,
            "exactly one frontier chunk")
    }

    // MARK: 3. Token parity: narrowed vs full-logits fallback

    func testNarrowedPathIsTokenIdenticalToFallback() throws {
        for chunkSize in [4, 7, 32] {
            let base = TinyTestModel.make(seed: hiddenSeed)
            let prompt = makePromptTokens(length: 26, seed: 800)
            let steps = 12

            let fallbackDriver = CBv2PrefillOutputDriver(
                model: CBv2PrefillFallbackModel(base), layerKinds: base.layerKinds)
            let fallbackFrontier = try fallbackDriver.prefill(
                prompt: prompt, chunkSize: chunkSize)
            let fallbackTokens = fallbackDriver.greedy(
                frontier: fallbackFrontier, steps: steps)

            let narrowDriver = CBv2PrefillOutputDriver(
                model: CBv2PrefillNarrowingModel(base), layerKinds: base.layerKinds)
            let narrowFrontier = try narrowDriver.prefill(prompt: prompt, chunkSize: chunkSize)
            let narrowTokens = narrowDriver.greedy(frontier: narrowFrontier, steps: steps)

            eval(fallbackFrontier, narrowFrontier)
            // NOT bit-identical, and that is inherent to the optimization:
            // projecting one hidden row ([1, hidden] @ [hidden, vocab]) uses a
            // different GEMM shape than projecting the whole chunk and
            // slicing, so the last mantissa bits can differ. What must hold
            // is that the difference is float noise and the ARGMAX (hence
            // every sampled token) is unchanged. `Gemma4TextModel.cbv2Prefill`
            // narrows exactly this way (`applyLMHead(hidden[0..., -1, 0...])`),
            // so this bound is the real contract, not a fixture artifact.
            // Measured here: max |Δ| ≈ 3.6e-7 on fp32 logits of magnitude ~4
            // (about one ULP); 1e-5 leaves ~25× headroom while still failing
            // on any real numeric divergence.
            let maxDiff = abs(narrowFrontier - fallbackFrontier).max().item(Float.self)
            XCTAssertLessThan(
                maxDiff, 1e-5,
                "frontier logits must match the sliced full logits to float noise "
                    + "(chunkSize \(chunkSize))")
            XCTAssertEqual(
                Int(argMax(narrowFrontier, axis: -1).asArray(Int32.self)[0]),
                Int(argMax(fallbackFrontier, axis: -1).asArray(Int32.self)[0]),
                "the frontier argmax must be unchanged (chunkSize \(chunkSize))")
            XCTAssertEqual(
                narrowTokens, fallbackTokens,
                "narrowed prefill must produce the identical greedy token sequence "
                    + "(chunkSize \(chunkSize))")
            XCTAssertEqual(narrowTokens.count, steps)
        }
    }

    /// Same parity through the REAL engine (async loop, production sampler),
    /// not just the synchronous driver.
    func testNarrowedPathIsTokenIdenticalToFallbackThroughEngine() async throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let prompt = makePromptTokens(length: 24, seed: 801)
        let budget = 10

        let fallbackEngine = makeEngine(
            model: CBv2PrefillFallbackModel(base), base: base, prefillChunkSize: 8)
        let fallback = await cbv2SchedCollect(
            try fallbackEngine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: budget)))
        await fallbackEngine.shutdown()

        let narrowEngine = makeEngine(
            model: CBv2PrefillNarrowingModel(base), base: base, prefillChunkSize: 8)
        let narrowed = await cbv2SchedCollect(
            try narrowEngine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: budget)))
        await narrowEngine.shutdown()

        XCTAssertEqual(fallback.finishReason, .length)
        XCTAssertEqual(narrowed.finishReason, .length)
        XCTAssertEqual(fallback.tokens.count, budget)
        XCTAssertEqual(
            narrowed.tokens, fallback.tokens,
            "engine-level greedy output must be unchanged by the narrowing opt-in")
    }

    // MARK: 4. KV parity after prefill

    func testNarrowedPathLeavesIdenticalKVState() throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let prompt = makePromptTokens(length: 26, seed: 900)
        let chunkSize = 7

        let fallbackDriver = CBv2PrefillOutputDriver(
            model: CBv2PrefillFallbackModel(base), layerKinds: base.layerKinds)
        let fallbackFrontier = try fallbackDriver.prefill(prompt: prompt, chunkSize: chunkSize)

        let narrowDriver = CBv2PrefillOutputDriver(
            model: CBv2PrefillNarrowingModel(base), layerKinds: base.layerKinds)
        let narrowFrontier = try narrowDriver.prefill(prompt: prompt, chunkSize: chunkSize)

        XCTAssertEqual(fallbackDriver.kv.count, narrowDriver.kv.count)
        XCTAssertFalse(fallbackDriver.kv.isEmpty, "test premise: the model has layers")

        for (layer, pair) in zip(fallbackDriver.kv, narrowDriver.kv).enumerated() {
            let (fallbackKV, narrowKV) = pair
            guard let fallbackKV, let narrowKV else {
                XCTAssertNil(fallbackKV, "layer \(layer): KV-shared layers must match")
                XCTAssertNil(narrowKV, "layer \(layer): KV-shared layers must match")
                continue
            }
            XCTAssertEqual(
                narrowKV.absoluteOffset, fallbackKV.absoluteOffset,
                "layer \(layer): absolute offset must be identical after prefill")
            XCTAssertEqual(
                narrowKV.absoluteOffset, prompt.count,
                "layer \(layer): prefill must advance every layer by the whole prompt")
            XCTAssertEqual(
                narrowKV.retainedCount, fallbackKV.retainedCount,
                "layer \(layer): retained KV length must be identical after prefill")

            let fallbackViews = try XCTUnwrap(
                (fallbackKV as? HarnessReadableKV)?.currentViews(),
                "layer \(layer): fallback KV must retain content")
            let narrowViews = try XCTUnwrap(
                (narrowKV as? HarnessReadableKV)?.currentViews(),
                "layer \(layer): narrowed KV must retain content")
            eval(fallbackViews.0, fallbackViews.1, narrowViews.0, narrowViews.1)
            XCTAssertTrue(
                cbv2BitIdentical(narrowViews.0, fallbackViews.0),
                "layer \(layer): retained KEYS must be bit-identical")
            XCTAssertTrue(
                cbv2BitIdentical(narrowViews.1, fallbackViews.1),
                "layer \(layer): retained VALUES must be bit-identical")
        }

        // …and the very next decode step agrees.
        eval(fallbackFrontier, narrowFrontier)
        let fallbackFirst = Int(argMax(fallbackFrontier, axis: -1).asArray(Int32.self)[0])
        let narrowFirst = Int(argMax(narrowFrontier, axis: -1).asArray(Int32.self)[0])
        XCTAssertEqual(narrowFirst, fallbackFirst, "first sampled token must match")
        XCTAssertEqual(
            narrowDriver.nextDecodeToken(after: narrowFirst),
            fallbackDriver.nextDecodeToken(after: fallbackFirst),
            "the decode step immediately after prefill must sample the same token")
    }

    // MARK: 5. Decode never touches the seam

    func testDecodePathNeverCallsPrefill() async throws {
        let base = TinyTestModel.make(seed: hiddenSeed)
        let model = CBv2PrefillNarrowingModel(base)
        let prompt = makePromptTokens(length: 24, seed: 1000)
        let budget = 6
        let engine = makeEngine(model: model, base: base, prefillChunkSize: 8)

        let collected = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: budget)))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.tokens.count, budget)

        let entries = model.log
        // Prefill first (3 chunks), then decode — and NOTHING after the last
        // prefill may be a prefill call.
        let prefillIndices = entries.indices.filter {
            if case .prefill = entries[$0] { return true }
            return false
        }
        XCTAssertEqual(prefillIndices, [0, 1, 2], "the seam is used only for prompt chunks")

        let decodeEntries = entries.dropFirst(3)
        XCTAssertFalse(decodeEntries.isEmpty, "the decode path must have run")
        for entry in decodeEntries {
            switch entry {
            case .prefill(let call):
                XCTFail(
                    "decode must never call prefill(...) (requirement \(call.requirement), "
                        + "shape \(call.tokenShape))")
            case .forward(let shape):
                XCTAssertEqual(
                    shape, [1, 1],
                    "decode must go through forward(...) with a rectangular [B, 1] batch")
            }
        }
        // The first token comes from the frontier prefill chunk; the rest
        // come from [B, 1] decode forwards. Chained decode launches step
        // N+1 before finalizing N, so the last request may burn one extra
        // (discarded) decode forward — never an extra prefill.
        XCTAssertTrue(
            (budget - 1 ... budget).contains(decodeEntries.count),
            "expected \(budget - 1) or \(budget) decode forwards (chained-decode overshoot), "
                + "got \(decodeEntries.count)")
    }

    // MARK: 6. Multimodal chunks reach the seam with spliced embeddings

    func testMultimodalChunkReachesPrefillOutputWithSplicedEmbeddings() async throws {
        // prompt 16, span [4, 10), chunk 8 ⇒ snapped chunks [0,4), [4,12),
        // [12,16). Chunk 0's naive boundary (8) lands inside the span so it
        // shrinks to the span start; chunk 1 starts at the span and its full
        // 8 tokens already cover it, so it carries the 6 image rows plus 2
        // trailing text rows. The SPAN chunk is intermediate
        // (.evaluationOnly); the frontier chunk is plain text
        // (.lastPositionLogits).
        let base = TinyTestModel.make(seed: hiddenSeed)
        let model = CBv2PrefillNarrowingModel(base)
        let spans = [CBv2ImageSpan(tokenOffset: 4, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: base, length: 16, spans: spans, seed: 1100)
        let engine = makeEngine(model: model, base: base, prefillChunkSize: 8)

        let collected = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: 4,
                    multimodal: CBv2VisionFixtures.input(images))))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)

        let prefills = model.log.compactMap { entry -> CBv2PrefillCall? in
            if case .prefill(let call) = entry { return call }
            return nil
        }
        XCTAssertEqual(
            prefills.map(\.tokenShape), [[1, 4], [1, 8], [1, 4]],
            "span snapping must produce chunks [0,4), [4,12), [12,16)")
        XCTAssertEqual(
            prefills.map(\.requirement),
            [.evaluationOnly, .evaluationOnly, .lastPositionLogits])

        let spanCall = prefills[1]
        let spliced = try XCTUnwrap(
            spanCall.inputEmbeddings,
            "the span-bearing chunk must reach prefillOutput with non-nil spliced embeddings")
        XCTAssertEqual(
            spliced.shape, [1, 8, base.config.hiddenSize],
            "spliced embeddings are [1, chunk, hidden]")
        XCTAssertEqual(
            spanCall.requirement, .evaluationOnly,
            "an intermediate span chunk still asks only for an evaluation handle")

        // The splice really happened: the span's rows carry the image
        // embedding, the rest keep their token embeddings.
        let plain = base.embedPromptTokens(spanCall.tokens)
        eval(spliced, plain, images[0].embedding)
        XCTAssertTrue(
            cbv2BitIdentical(
                spliced[0..., 0 ..< 6, 0...], images[0].embedding.asType(spliced.dtype)),
            "chunk rows 0..<6 are the image span — they must be the image embedding")
        XCTAssertTrue(
            cbv2BitIdentical(spliced[0..., 6 ..< 8, 0...], plain[0..., 6 ..< 8, 0...]),
            "chunk rows 6..<8 are trailing TEXT — they must keep their token embeddings")
        XCTAssertFalse(
            cbv2BitIdentical(spliced, plain),
            "spliced embeddings must differ from the raw placeholder-token embeddings")

        // Text-only chunks stay on the token path.
        XCTAssertNil(prefills[0].inputEmbeddings, "leading text chunk carries no embeddings")
        XCTAssertNil(prefills[2].inputEmbeddings, "trailing text chunk carries no embeddings")
    }

    /// A span chunk that IS the frontier must reach the seam with non-nil
    /// embeddings AND `.lastPositionLogits`.
    func testFrontierMultimodalChunkRequestsLastPositionLogits() async throws {
        // prompt 10, span [4, 10) ⇒ chunks [0,4) and [4,10); the span chunk
        // computes through the last prompt token, so it samples.
        let base = TinyTestModel.make(seed: hiddenSeed)
        let model = CBv2PrefillNarrowingModel(base)
        let spans = [CBv2ImageSpan(tokenOffset: 4, length: 6)]
        let (prompt, images) = CBv2VisionFixtures.make(
            model: base, length: 10, spans: spans, seed: 1101)
        let engine = makeEngine(model: model, base: base, prefillChunkSize: 8)

        let collected = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: 4,
                    multimodal: CBv2VisionFixtures.input(images))))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)

        let prefills = model.log.compactMap { entry -> CBv2PrefillCall? in
            if case .prefill(let call) = entry { return call }
            return nil
        }
        XCTAssertEqual(prefills.map(\.tokenShape), [[1, 4], [1, 6]])
        XCTAssertEqual(prefills.map(\.requirement), [.evaluationOnly, .lastPositionLogits])
        let frontier = prefills[1]
        XCTAssertNotNil(
            frontier.inputEmbeddings,
            "a frontier span chunk must still arrive with spliced embeddings")
        XCTAssertEqual(frontier.inputEmbeddings?.shape, [1, 6, base.config.hiddenSize])
    }

    // MARK: - Engine construction (production caches + contiguous backend)

    private func makeEngine(
        model: CBv2SteppableModel, base: TinyTestModel, prefillChunkSize: Int
    ) -> EngineV2 {
        EngineV2(
            model: model,
            layerKinds: base.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: base.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 5),
            schedulerConfig: CBv2SchedulerConfig(
                maxBatchedTokensPerStep: 256, prefillChunkSize: prefillChunkSize,
                maxWaiting: 4))
    }
}
