// CBv2InvarianceTests.swift — WS-G: THE core batch-composition invariance
// suite (report 12 item 5, plan §6 "Correctness harness").
//
// Contract under test: a request's greedy output is TOKEN-EXACT identical no
// matter what batchmates it runs with — alone, alongside 1/2/3 neighbors,
// joining mid-stream, or through a churn storm of joins/leaves. This is the
// property the v1 left-padded engine could not guarantee (scalar RoPE
// offsets, shared `_idx`, batch-wide trims) and the property that makes the
// v2 per-request-KV design correct by construction.
//
// Every arm but the last runs on `TinyTestModel` (seeded random weights, no
// downloads) through the v2 path (per-request KV + per-row attention). At
// integration the same suites re-point at the real WS-A/WS-B implementations.
//
// The LAST arm is the PAGED one. `TinyTestModel`'s head dim (16) is not one
// the paged kernel accepts, so it cannot reach `PagedAttentionKernel.decode`
// at all; that arm therefore carries its own minimal fixture and states the
// same contract one layer down, as bit-identity of a row's attention output.
// It is a regression gate for the WS-6.4 partition sizer — see its doc
// comment for the environment that must break it.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2InvarianceTests: XCTestCase {

    /// ≥64 decode steps per the spec.
    private static let soloSteps = 64

    // Shared fixture: one model per class (weights deterministic via seed).
    // XCTest runs the class's tests serially; `nonisolated(unsafe)` matches
    // that execution model.
    private nonisolated(unsafe) static var model: TinyTestModel!
    private nonisolated(unsafe) static var subjectPrompt: [Int] = []
    private nonisolated(unsafe) static var soloOutput: [Int] = []

    override class func setUp() {
        super.setUp()
        model = TinyTestModel.make(seed: 0xC0FFEE)
        // Subject prompt is longer than the sliding window (16) so windowed
        // layers are exercised from the first decode step, and 64 decode
        // steps cross the window repeatedly.
        subjectPrompt = makePromptTokens(length: 24, seed: 1)
        soloOutput = try! CBv2HarnessEngine.runSolo(
            model: model, prompt: subjectPrompt, maxTokens: soloSteps)
    }

    override class func tearDown() {
        model = nil
        super.tearDown()
    }

    private var model: TinyTestModel { Self.model }

    // MARK: - Solo == batched with N neighbors

    /// Decode the subject alongside `neighborCount` different requests; the
    /// subject's output must equal its solo run token-for-token.
    private func assertSubjectUnaffected(byNeighbors neighborCount: Int) throws {
        let engine = CBv2HarnessEngine(model: model)
        let subject = try engine.join(prompt: Self.subjectPrompt, maxTokens: Self.soloSteps)

        for n in 0 ..< neighborCount {
            // Different lengths (some > window) and different content.
            let length = 5 + n * 11
            try engine.join(
                prompt: makePromptTokens(length: length, seed: 100 + UInt64(n)),
                maxTokens: Self.soloSteps)
        }

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), Self.soloOutput,
            "subject output changed with \(neighborCount) neighbor(s) in the batch")
    }

    func testSoloEqualsBatchWithOneNeighbor() throws {
        try assertSubjectUnaffected(byNeighbors: 1)
    }

    func testSoloEqualsBatchWithTwoNeighbors() throws {
        try assertSubjectUnaffected(byNeighbors: 2)
    }

    func testSoloEqualsBatchWithThreeNeighbors() throws {
        try assertSubjectUnaffected(byNeighbors: 3)
    }

    // MARK: - Mid-stream join, neighbors finishing around the subject

    /// The subject joins while two neighbors are mid-decode; one neighbor
    /// finishes early (leaves), another joins later, and the first-joined
    /// neighbors all finish before the subject. Subject must be unaffected.
    func testJoinMidStreamWithNeighborsFinishingAround() throws {
        let engine = CBv2HarnessEngine(model: model)

        let n0 = try engine.join(
            prompt: makePromptTokens(length: 7, seed: 200), maxTokens: 12)
        let n1 = try engine.join(
            prompt: makePromptTokens(length: 30, seed: 201), maxTokens: 40)

        // Neighbors decode for a while before the subject arrives.
        for _ in 0 ..< 8 { engine.decodeStep() }

        let subject = try engine.join(prompt: Self.subjectPrompt, maxTokens: Self.soloSteps)

        // n0 finishes after 4 more steps (12 total) and leaves; n2 joins.
        for _ in 0 ..< 4 { engine.decodeStep() }
        XCTAssertTrue(engine.finishedIDs.contains(n0), "n0 should have hit maxTokens")
        engine.remove(id: n0)

        let n2 = try engine.join(
            prompt: makePromptTokens(length: 18, seed: 202), maxTokens: 20)

        // Run until every neighbor has finished and left, subject still going.
        while !(engine.active.first(where: { $0.id == n1 })?.finished ?? true) {
            engine.decodeStep()
        }
        engine.remove(id: n1)
        while !(engine.active.first(where: { $0.id == n2 })?.finished ?? true) {
            engine.decodeStep()
        }
        engine.remove(id: n2)

        // Subject finishes (possibly alone again).
        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), Self.soloOutput,
            "mid-stream join with churn around the subject changed its output")
    }

    // MARK: - Sliding-window crossing during the run

    /// Same invariance with prompts and runs specifically sized so windowed
    /// layers cross their eviction boundary mid-decode for every request.
    func testWindowCrossingUnderBatching() throws {
        // Short prompt (< window): the window boundary is crossed DURING
        // decode, not during prefill.
        let prompt = makePromptTokens(length: 6, seed: 300)
        let steps = 3 * TinyTestModelConfig().windowSize  // cross 3× over

        let solo = try CBv2HarnessEngine.runSolo(model: model, prompt: prompt, maxTokens: steps)

        let engine = CBv2HarnessEngine(model: model)
        let subject = try engine.join(prompt: prompt, maxTokens: steps)
        // Neighbors on both sides of their own window boundaries.
        try engine.join(prompt: makePromptTokens(length: 40, seed: 301), maxTokens: steps)
        try engine.join(prompt: makePromptTokens(length: 15, seed: 302), maxTokens: steps)

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), solo,
            "window crossing under batching diverged from the solo run")
    }

    // MARK: - Sinks (GPT-OSS shape) under batching

    /// Same core invariance on the sink-enabled fixture: per-head learned
    /// sinks are folded into every row's softmax identically regardless of
    /// batch composition.
    func testInvarianceWithAttentionSinks() throws {
        let sinkModel = TinyTestModel.make(seed: 0xC0FFEE, withSinks: true)
        let prompt = makePromptTokens(length: 24, seed: 400)

        let solo = try CBv2HarnessEngine.runSolo(model: sinkModel, prompt: prompt, maxTokens: 48)

        let engine = CBv2HarnessEngine(model: sinkModel)
        let subject = try engine.join(prompt: prompt, maxTokens: 48)
        try engine.join(prompt: makePromptTokens(length: 9, seed: 401), maxTokens: 48)
        try engine.join(prompt: makePromptTokens(length: 33, seed: 402), maxTokens: 48)

        while !(engine.active.first(where: { $0.id == subject })?.finished ?? true) {
            engine.decodeStep()
        }

        XCTAssertEqual(
            engine.generated(for: subject), solo,
            "sink model output changed under batching")
    }

    // MARK: - Paged backend: a row's partitioning must not see its batchmates

    /// `pagedattention.metal` (:12-16) states design goal 1 as "a row's
    /// partition count depends only on ITS OWN attended length (fixed PTOK),
    /// so its arithmetic — including summation order — is bit-identical
    /// regardless of batchmates". WS-6.4's adaptive sizer voided the
    /// parenthetical: `PagedAttentionKernel.partitionTokensForDispatch`
    /// (PagedAttentionKernel.swift:313) derives PTOK from the DISPATCH's
    /// batch size and the batch-wide `maxAttendLength`, so a 1024-token row
    /// is split 16 ways alone and 4 ways beside a 2048-token batchmate.
    /// Different partition counts mean a different online-softmax merge
    /// order — batch-composition dependence, the one thing this suite
    /// exists to forbid. `partitionTargetDefault` (:256) is 0 as of v0.8.0
    /// precisely to shut it off; this arm is what holds that shut.
    ///
    /// REGRESSION GATE. Restoring the pre-v0.8.0 adaptation must break it:
    ///
    ///     DARKBLOOM_CBV2_PAGED_PTOK_TARGET=128 swift test --filter CBv2Invariance
    ///
    /// FIXTURE. `TinyTestModel` cannot carry this arm at all: its head dim is
    /// 16 (CBv2Fixtures.swift:472) and the paged kernel accepts only
    /// 64/128/256/512 (`PagedAttentionKernel.supportedHeadDims`, :159), so
    /// `CBv2HarnessEngine` can never reach `PagedAttentionKernel.decode`.
    /// This is therefore the smallest fixture that DOES reach it with a
    /// genuine multi-row batch: one paged attention layer on the GPT-OSS
    /// decode shape (8 kv heads, GQA 8, head dim 64 — the shape the sizer's
    /// own tuning note measures) driven as a recurrent greedy decoder,
    /// attention out -> mix -> logits -> argmax -> next query. The recurrence
    /// is what a real stack supplies and what turns a one-ULP attention
    /// difference into a flipped token; the KV is fixed prefilled context, so
    /// `maxAttendLength` — and hence the sizer's input — is constant per run.
    func testPagedRowInvariantToBatchmateAttendedLength() throws {
        let source: String
        do {
            source = try PagedAttentionResources.loadSourceForCurrentProcess()
        } catch {
            throw XCTSkip("paged attention runtime resource unavailable: \(error)")
        }

        let kvHeads = 8
        let queryHeads = 64
        let headDim = 64
        let pageSize = 16
        let dtype = DType.float16
        let features = queryHeads * headDim
        let vocab = 1024
        let subjectContext = 1024
        let batchmateContext = 2048
        let steps = Self.soloSteps
        // Post-norm query magnitude. A unit-RMS query against unit-variance
        // keys puts the pre-softmax scores at std 1 over a 1024-token
        // context — a near-uniform softmax, i.e. an AVERAGING operator, and
        // averaging is a contraction that damps exactly the divergence under
        // test (measured: unit gain holds the same tokens for 1024 steps
        // while the attention output is already wrong). Real decode
        // attention is peaked, not uniform; a gain of 8 puts the scores at
        // std 8, which is an ordinary attention-logit range and makes the
        // recurrence expansive the way a real stack is.
        let queryGain: Float = 8

        let hpt = PagedAttentionKernel.headsPerThreadgroup(
            headDim: headDim, gqa: queryHeads / kvHeads)
        let splits = (queryHeads / kvHeads) / hpt
        func partitionTokens(batch: Int, maxAttendLength: Int, target: Int) -> Int {
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxAttendLength, batch: batch, kvHeads: kvHeads,
                headSplits: splits, pageSize: pageSize, target: target)
        }

        // Non-vacuity guard. This fixture is only a regression test while the
        // ADAPTIVE sizer still hands this row two different partition lengths
        // depending on its batchmates. If a ladder or shape change ever makes
        // the two agree, the arm has stopped testing anything and must be
        // re-sized — it must not go quiet.
        let legacyTarget = 128
        XCTAssertNotEqual(
            partitionTokens(batch: 1, maxAttendLength: subjectContext, target: legacyTarget),
            partitionTokens(batch: 2, maxAttendLength: batchmateContext, target: legacyTarget),
            "fixture no longer exercises batch-composition-dependent partitioning")

        // Shared, deterministic readout weights: identical in both runs, so
        // the ONLY difference between them is the decode dispatch's shape.
        let mix =
            (MLXRandom.normal([features, features], key: MLXRandom.key(0x5EED_0001))
            * (1.0 / Float(features).squareRoot())).asType(dtype)
        let unembed = MLXRandom.normal([features, vocab], key: MLXRandom.key(0x5EED_0002))
        let embed = MLXRandom.normal([vocab, features], key: MLXRandom.key(0x5EED_0003))
            .asType(dtype)
        var paramValues = [Float](repeating: 0, count: 8)
        paramValues[1] = 1.0 / Float(headDim).squareRoot()  // scale; params[0] = softcap
        let params = MLXArray(paramValues)
        eval(mix, unembed, embed, params)

        /// Greedy-decode `contexts.count` rows for `steps` steps, one paged
        /// dispatch per step. Returns row 0's tokens plus its raw per-step
        /// attention output — the bitwise witness.
        func run(contexts: [Int]) -> (tokens: [Int], attention: [[Float]]) {
            let batch = contexts.count
            let pages = contexts.map { ($0 + pageSize - 1) / pageSize }

            // Row r's KV is keyed on r alone, so row 0's pages hold identical
            // bytes at identical physical indices whether or not row 1 exists.
            var keyBlocks: [MLXArray] = []
            var valueBlocks: [MLXArray] = []
            for row in contexts.indices {
                let shape = [pages[row], kvHeads, pageSize, headDim]
                keyBlocks.append(
                    MLXRandom.normal(shape, key: MLXRandom.key(0x51AB_0000 + UInt64(row)))
                        .asType(dtype))
                valueBlocks.append(
                    MLXRandom.normal(shape, key: MLXRandom.key(0x51AB_1000 + UInt64(row)))
                        .asType(dtype))
            }
            let kSlab = concatenated(keyBlocks, axis: 0)
            let vSlab = concatenated(valueBlocks, axis: 0)

            let maxPages = max(8, pages.max() ?? 8)
            var table = [Int32](repeating: 0, count: batch * maxPages)
            var physical = 0
            for row in contexts.indices {
                for page in 0 ..< pages[row] {
                    table[row * maxPages + page] = Int32(physical)
                    physical += 1
                }
            }
            let tables = MLXArray(table, [batch, maxPages])
            let (seqinfo, maxAttendLength) = PagedAttentionKernel.seqinfo(
                contexts.indices.map { row in
                    PagedAttentionKernel.SeqInfoRow(
                        attendStart: 0, attendLength: contexts[row], tableLength: pages[row])
                })
            let fence = MLXArray.zeros([1], dtype: .int32)

            var queries = concatenated(
                contexts.indices.map { row in
                    MLXRandom.normal(
                        [1, queryHeads, headDim],
                        key: MLXRandom.key(0x0B1E_0000 + UInt64(row))
                    ).asType(dtype)
                }, axis: 0)
            eval(kSlab, vSlab, tables, seqinfo, fence, queries)

            var tokens: [Int] = []
            var attention: [[Float]] = []
            for _ in 0 ..< steps {
                let out = PagedAttentionKernel.decode(
                    queries: queries, kSlab: kSlab, vSlab: vSlab, tables: tables,
                    seqinfo: seqinfo, maxAttendLength: maxAttendLength, sinks: nil,
                    params: params, softcap: false, pageSize: pageSize,
                    writeFence: fence, kernelSource: source
                ).out
                eval(out)
                attention.append(out[0].asArray(Float.self))

                // Readout runs per row on a [1, features] slice, so the
                // subject's downstream arithmetic is shape-identical in both
                // runs: a [2, F] matmul is under no obligation to reduce row 0
                // the same way a [1, F] one does, and that would masquerade as
                // the very divergence under test.
                var nextQueries: [MLXArray] = []
                for row in 0 ..< batch {
                    let hidden = out[row].reshaped([1, features])
                    let token = argMax(
                        matmul(hidden.asType(.float32), unembed), axis: -1
                    ).item(Int.self)
                    if row == 0 { tokens.append(token) }
                    let mixed = (matmul(hidden, mix) + embed[token].reshaped([1, features]))
                        .asType(.float32)
                    let normed =
                        mixed
                        * (rsqrt(mean(mixed * mixed, axis: -1, keepDims: true) + 1e-6)
                            * queryGain)
                    nextQueries.append(
                        normed.asType(dtype).reshaped([1, queryHeads, headDim]))
                }
                queries = batch == 1 ? nextQueries[0] : concatenated(nextQueries, axis: 0)
                eval(queries)
            }
            return (tokens, attention)
        }

        let solo = run(contexts: [subjectContext])
        let batched = run(contexts: [subjectContext, batchmateContext])

        let target = PagedAttentionKernel.partitionTargetThreadgroups
        let dispatched =
            "PTOK solo="
            + "\(partitionTokens(batch: 1, maxAttendLength: subjectContext, target: target))"
            + " batched="
            + "\(partitionTokens(batch: 2, maxAttendLength: batchmateContext, target: target))"
            + " (\(PagedAttentionKernel.partitionTargetEnvironmentKey)=\(target))"

        // The property, in the form the user sees it.
        let tokenDivergence =
            zip(solo.tokens, batched.tokens)
            .enumerated().first { $0.element.0 != $0.element.1 }?.offset
        XCTAssertNil(
            tokenDivergence,
            "subject row's decoded token \(tokenDivergence ?? -1) changed because a "
                + "\(batchmateContext)-token batchmate joined its dispatch — \(dispatched)")

        // The same property one layer down, where it is deterministic instead
        // of contingent on an argmax landing near a tie: the row's attention
        // output must be bit-identical, which is what design goal 1 promises
        // and what every downstream readout inherits.
        var firstDivergentStep: Int?
        var maxDifferingElements = 0
        for step in 0 ..< steps {
            let differing = zip(solo.attention[step], batched.attention[step])
                .reduce(into: 0) { $0 += $1.0 == $1.1 ? 0 : 1 }
            guard differing > 0 else { continue }
            if firstDivergentStep == nil { firstDivergentStep = step }
            maxDifferingElements = max(maxDifferingElements, differing)
        }
        XCTAssertNil(
            firstDivergentStep,
            "subject row's attention output is not bit-identical across batch "
                + "composition: first diverges at step \(firstDivergentStep ?? -1), up to "
                + "\(maxDifferingElements)/\(features) elements per step — \(dispatched)")
    }

    // MARK: - Stochastic sampling with per-request seeds (WS-E)

    /// The contract keys the RNG on (seed, requestID, per-request step), so
    /// a SEEDED stochastic request must reproduce token-exactly under
    /// batching. Runs the REAL EngineV2 with the production
    /// `CBv2DefaultSampler` (LogitsPipelineV2 + SamplerV2): once solo, once
    /// with two stochastic neighbors — the subject keeps the same request
    /// id and seed, so its keyed noise stream is identical in both runs.
    func testPerRequestSeedInvarianceUnderBatching() async throws {
        let prompt = makePromptTokens(length: 12, seed: 500)
        let sampling = CBv2SamplingParams(temperature: 0.8, topP: 0.9, seed: 42)
        let maxTokens = 24

        func run(withNeighbors: Bool) async throws -> [Int] {
            let engine = EngineV2(
                model: model,
                layerKinds: model.layerKinds,
                backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 7),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 4, prefillChunkSize: 8))
            let subject = CBv2Request(
                id: CBv2RequestID(1), promptTokens: prompt, sampling: sampling,
                maxTokens: maxTokens)
            var neighborStreams: [AsyncStream<CBv2Event>] = []
            let subjectStream = try engine.submit(subject)
            if withNeighbors {
                for n: UInt64 in 2 ... 3 {
                    let neighbor = CBv2Request(
                        id: CBv2RequestID(n),
                        promptTokens: makePromptTokens(length: 6 + Int(n) * 9, seed: 500 + n),
                        sampling: CBv2SamplingParams(
                            temperature: 0.7, topK: 20, seed: 1000 + n),
                        maxTokens: maxTokens)
                    neighborStreams.append(try engine.submit(neighbor))
                }
            }
            let collected = await cbv2SchedCollect(subjectStream)
            for stream in neighborStreams {
                _ = await cbv2SchedCollect(stream)
            }
            await engine.shutdown()
            XCTAssertEqual(collected.finishReason, .length)
            XCTAssertEqual(collected.tokens.count, maxTokens)
            return collected.tokens
        }

        let solo = try await run(withNeighbors: false)
        let batched = try await run(withNeighbors: true)
        XCTAssertEqual(
            solo, batched,
            "seeded stochastic request diverged under batching — keyed RNG must be a pure function of (seed, requestID, per-request step)"
        )
    }

    // MARK: - Churn storm

    /// 50 seeded random join/leave events; every request that ever ran must
    /// produce output token-exact vs its own solo run (compared over however
    /// many tokens it actually generated before finish/cancel).
    func testChurnStormEveryRequestMatchesSolo() throws {
        var rng = CBv2HarnessRNG(seed: 0xD00D)
        let engine = CBv2HarnessEngine(model: model)
        let maxConcurrent = 4

        struct Record {
            let prompt: [Int]
            let maxTokens: Int
            var output: [Int] = []
        }
        var records: [UInt64: Record] = [:]
        var liveIDs: [CBv2RequestID] = []

        func joinRandom() throws {
            let prompt = makePromptTokens(
                length: Int.random(in: 3 ... 28, using: &rng),
                seed: rng.next())
            let maxTokens = Int.random(in: 4 ... 24, using: &rng)
            let id = try engine.join(prompt: prompt, maxTokens: maxTokens)
            records[id.raw] = Record(prompt: prompt, maxTokens: maxTokens)
            liveIDs.append(id)
        }

        func leave(_ id: CBv2RequestID) {
            // Capture what the request generated (finish OR early cancel).
            records[id.raw]?.output = engine.generated(for: id) ?? []
            engine.remove(id: id)
            liveIDs.removeAll { $0 == id }
        }

        var events = 0
        while events < 50 {
            let wantJoin =
                liveIDs.isEmpty
                || (liveIDs.count < maxConcurrent && Bool.random(using: &rng))
            if wantJoin {
                try joinRandom()
            } else {
                // Random leave: prefer finished rows, otherwise cancel one.
                let id = engine.finishedIDs.first ?? liveIDs.randomElement(using: &rng)!
                leave(id)
            }
            events += 1

            // A few decode steps between events; harvest natural finishes so
            // finished rows don't keep decoding forever.
            if !liveIDs.isEmpty {
                for _ in 0 ..< Int.random(in: 1 ... 4, using: &rng) {
                    guard !engine.active.isEmpty else { break }
                    engine.decodeStep()
                    for id in engine.finishedIDs { leave(id) }
                }
            }
        }

        // Drain the stragglers.
        while !engine.active.isEmpty {
            engine.decodeStep()
            for id in engine.finishedIDs { leave(id) }
        }

        XCTAssertGreaterThanOrEqual(records.count, 20, "storm should admit many requests")

        // Every request must match the prefix of its solo run.
        for (raw, record) in records.sorted(by: { $0.key < $1.key }) {
            guard !record.output.isEmpty else { continue }
            let solo = try CBv2HarnessEngine.runSolo(
                model: model, prompt: record.prompt, maxTokens: record.maxTokens)
            XCTAssertEqual(
                record.output, Array(solo.prefix(record.output.count)),
                "request cbv2-\(raw) diverged from its solo run during the churn storm")
        }
    }
}
