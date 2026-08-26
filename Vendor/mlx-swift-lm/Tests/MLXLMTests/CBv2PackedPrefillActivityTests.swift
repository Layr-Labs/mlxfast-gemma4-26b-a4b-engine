// CBv2PackedPrefillActivityTests.swift — packed prefill as EVIDENCE.
//
// `CBv2Engine.packedPrefillActivity()` is the out-of-module answer to a
// question a capability flag cannot settle: did the rectangular `[B, chunk]`
// packed-prefill path actually RUN?
//
// `isSupported` is configuration — the caches vouch for per-row independence
// and the model's prompt forward is batch-generic. `rowsExecuted` /
// `groupsExecuted` are measurements: they move only inside
// `EngineLoopV2.executeMixed`, at the rectangular forward itself. A model
// that claims packing and is never given two equal-length prompt chunks in
// one step must read zero.
//
// This suite drives REAL model shapes through a REAL `EngineV2` (submit ->
// stream -> shutdown) and reads the counter through the `any CBv2Engine`
// existential, exactly as a benchmark harness outside MLXLMCommon would:
//
//  1. testGPTOSSShapeNeverPacksAndGemma4ShapeDoes — the contrast. GPT-OSS
//     (no `CBv2LanguageModelPrefillForwardable` conformance, so the adapter
//     answers false) stays at zero under the SAME concurrent traffic that
//     drives Gemma 4's counter positive. Both arms cross-check the counter
//     against the batch dimensions the MODEL actually saw, so the counter
//     has to count the right thing, not merely be nonzero.
//  2. testClaimedPackedPrefillCapabilityReadsZeroUntilExercised — the same
//     Gemma 4 engine, one request at a time: capability true, evidence zero.
//     Configuration alone cannot satisfy the counter.
//  3. testEngineWithoutAPackedPrefillPathReportsNoActivity — the protocol
//     default is fail-closed.
//
// Tiny random-weight configs (vocab 64, 4 layers). No checkpoints.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Passes every call straight through to the wrapped model and INHERITS its
/// real packed-prefill claim (unlike `CBv2PackedRecordingModel`, whose claim
/// the test dictates) while recording the batch dimension of each prompt
/// forward. Nothing numeric changes: the engine reaches the same
/// `prefill` implementation it would without the wrapper.
final class CBv2ActivityRecordingModel: CBv2PackedPrefillSteppableModel, @unchecked Sendable {
    private let inner: any CBv2PackedPrefillSteppableModel
    private let lock = NSLock()
    private var _prefillShapes: [[Int]] = []

    init(_ inner: any CBv2PackedPrefillSteppableModel) {
        self.inner = inner
    }

    /// Shapes of every prompt-chunk forward, in order.
    var prefillShapes: [[Int]] {
        lock.lock()
        defer { lock.unlock() }
        return _prefillShapes
    }

    /// Prompt-chunk forwards that carried more than one row — the packed
    /// ones, as seen from the model side.
    var packedShapes: [[Int]] { prefillShapes.filter { $0[0] > 1 } }

    var supportsPackedPrefill: Bool { inner.supportsPackedPrefill }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        inner.forward(tokens: tokens, caches: caches)
    }

    func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        lock.lock()
        _prefillShapes.append(tokens.shape)
        lock.unlock()
        return inner.prefill(
            tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
            requirement: requirement)
    }
}

final class CBv2PackedPrefillActivityTests: XCTestCase {

    // MARK: - Fixtures

    /// Deterministic prompt tokens (no MLX involvement — a plain LCG).
    private func promptTokens(length: Int, seed: UInt64, vocabSize: Int = 64) -> [Int] {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1
        return (0 ..< length).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabSize - 1)) + 1
        }
    }

    /// 4-layer Gemma 4 text model: the shipping conformer that claims
    /// rectangular-prompt safety (`cbv2SupportsPackedPrefill == true`).
    ///
    /// `headDim` is a parameter because the paged kernel only accepts
    /// 64/128/256/512 — the paged arm needs a head dim it can serve, the
    /// contiguous arms stay on the cheap tiny one.
    private func makeGemma4(headDim: Int = 8, globalHeadDim: Int = 16) throws -> (
        Gemma4TextModel, [CBv2LayerKind]
    ) {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": \(headDim),
                "global_head_dim": \(globalHeadDim),
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
        let model = Gemma4TextModel(config)
        eval(model)
        return (model, config.cbv2LayerKinds)
    }

    /// 4-layer GPT-OSS: alternating sliding/full with sinks, and NO prompt
    /// narrowing conformance — the control shape that must never pack.
    private func makeGPTOSS() throws -> (GPTOSSModel, [CBv2LayerKind]) {
        let json = """
            {
                "model_type": "gpt_oss",
                "num_hidden_layers": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": 64,
                "rms_norm_eps": 1e-5,
                "hidden_size": 32,
                "intermediate_size": 32,
                "head_dim": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "sliding_window": 16
            }
            """
        let config = try JSONDecoder.json5().decode(
            GPTOSSConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(0x60557)
        let model = GPTOSSModel(config)
        eval(model)
        return (model, config.cbv2LayerKinds)
    }

    /// Production stack: the given KV backend + cache provider (defaulting
    /// to contiguous, which vouches for packing), the model behind the
    /// shipping steppable adapter. The engine is typed as the PUBLIC
    /// protocol so every assertion below goes through the surface an
    /// out-of-module harness has.
    private func makeEngine(
        model: any LanguageModel, layerKinds: [CBv2LayerKind], prefillChunkSize: Int,
        backend: CBv2KVBackend? = nil, cacheProvider: CBv2LayerCacheProvider? = nil
    ) -> (engine: any CBv2Engine, recorder: CBv2ActivityRecordingModel) {
        let recorder = CBv2ActivityRecordingModel(
            CBv2SteppableLanguageModelAdapter(model))
        let engine = EngineV2(
            model: recorder,
            layerKinds: layerKinds,
            backend: backend
                ?? CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27)),
            cacheProvider: cacheProvider ?? CBv2LayerCacheBank(layerKinds: layerKinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 8, maxBatchedTokensPerStep: 256,
                prefillChunkSize: prefillChunkSize, maxWaiting: 16))
        return (engine, recorder)
    }

    private func request(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    /// Submit every prompt back-to-back (so the rows share steps), then drain.
    private func runConcurrently(
        _ engine: any CBv2Engine, prompts: [[Int]], maxTokens: Int
    ) async throws -> [CBv2SchedCollected] {
        var streams: [AsyncStream<CBv2Event>] = []
        for (index, prompt) in prompts.enumerated() {
            streams.append(
                try engine.submit(
                    request(id: UInt64(index + 1), prompt: prompt, maxTokens: maxTokens)))
        }
        var collected: [CBv2SchedCollected] = []
        for stream in streams { collected.append(await cbv2SchedCollect(stream)) }
        return collected
    }

    // MARK: - 1. Executed vs never executed

    /// Identical traffic, two model shapes. The counter is not a restatement
    /// of the capability: it is zero for the shape whose prompt forward the
    /// engine refuses to batch, and positive for the one it does batch — and
    /// it gets there by running prompts through `submit`, never by the test
    /// touching the counter's storage. Both arms are reconciled against the
    /// batch dimensions the model itself observed.
    func testGPTOSSShapeNeverPacksAndGemma4ShapeDoes() async throws {
        // 8 prefill chunks per row: even if a row is admitted a step or two
        // late, the cohort still shares many equal-length chunk steps.
        let prompts = (0 ..< 4).map { promptTokens(length: 128, seed: UInt64(4100 + $0)) }

        // -- control: GPT-OSS shape ------------------------------------
        let (gptoss, gptossKinds) = try makeGPTOSS()
        XCTAssertFalse(
            CBv2SteppableLanguageModelAdapter(gptoss).supportsPackedPrefill,
            "test premise: GPT-OSS makes no rectangular-prompt claim")
        let control = makeEngine(
            model: gptoss, layerKinds: gptossKinds, prefillChunkSize: 16)
        XCTAssertEqual(
            control.engine.packedPrefillActivity(), .none,
            "a fresh engine has neither capability nor evidence for a non-packing model")

        let controlRuns = try await runConcurrently(
            control.engine, prompts: prompts, maxTokens: 4)
        await control.engine.shutdown()
        for (index, run) in controlRuns.enumerated() {
            XCTAssertEqual(run.finishReason, .length, "control row \(index) must complete")
            XCTAssertEqual(run.tokens.count, 4, "control row \(index)")
        }
        let controlAfter = control.engine.packedPrefillActivity()
        XCTAssertFalse(
            controlAfter.isSupported,
            "the model's prompt forward is not batch-generic, so packing is not available")
        XCTAssertEqual(
            control.recorder.packedShapes, [],
            "test premise: the model must have seen only [1, chunk] prompt forwards")
        XCTAssertGreaterThan(
            control.recorder.prefillShapes.count, 0,
            "test premise: prompt chunks really were executed")
        XCTAssertEqual(
            controlAfter.rowsExecuted, 0,
            "no row may be reported as packed when the packed path is never entered")
        XCTAssertEqual(controlAfter.groupsExecuted, 0)
        XCTAssertFalse(controlAfter.didExecute)

        // -- subject: Gemma 4 shape ------------------------------------
        let (gemma, gemmaKinds) = try makeGemma4()
        XCTAssertTrue(
            CBv2SteppableLanguageModelAdapter(gemma).supportsPackedPrefill,
            "test premise: Gemma 4 claims rectangular-prompt safety")
        let subject = makeEngine(
            model: gemma, layerKinds: gemmaKinds, prefillChunkSize: 16)
        let subjectBefore = subject.engine.packedPrefillActivity()
        XCTAssertTrue(
            subjectBefore.isSupported, "both gates agree before a single step has run")
        XCTAssertFalse(
            subjectBefore.didExecute,
            "capability is not evidence: nothing has been packed yet")

        let subjectRuns = try await runConcurrently(
            subject.engine, prompts: prompts, maxTokens: 4)
        await subject.engine.shutdown()
        for (index, run) in subjectRuns.enumerated() {
            XCTAssertEqual(run.finishReason, .length, "subject row \(index) must complete")
            XCTAssertEqual(run.tokens.count, 4, "subject row \(index)")
        }
        let subjectAfter = subject.engine.packedPrefillActivity()
        XCTAssertTrue(subjectAfter.isSupported)
        XCTAssertTrue(
            subjectAfter.didExecute,
            "four equal-length concurrent prompts must have taken the packed path, saw "
                + "\(subject.recorder.prefillShapes)")
        // The counter is reconciled against the model's own view: one group
        // per rectangular forward, and its rows are that forward's batch dim.
        XCTAssertEqual(
            subjectAfter.groupsExecuted, subject.recorder.packedShapes.count,
            "one counted group per rectangular prompt forward the model saw")
        XCTAssertEqual(
            subjectAfter.rowsExecuted,
            subject.recorder.packedShapes.reduce(0) { $0 + $1[0] },
            "counted rows must be the batch dimensions of those forwards, saw "
                + "\(subject.recorder.packedShapes)")
    }

    // MARK: - 1b. Backend independence

    /// The counter is incremented in the backend-agnostic step path, so a
    /// PAGED gemma-4 stack must report packing exactly like the contiguous
    /// one (`PagedLayerCache.keepsRowsIndependentWhenPackedByConstruction`
    /// is true, so the bank vouches). A harness differencing the counters
    /// per backend arm would otherwise read a false "claimed but never
    /// packed" on paged.
    func testPagedBackendCountsPackedPrefillLikeContiguous() async throws {
        // The paged kernel serves head dims 64/128/256/512 only (production
        // gemma-4 is 256 sliding / 512 full).
        let (gemma, kinds) = try makeGemma4(headDim: 64, globalHeadDim: 64)
        let paged: PagedKVBackend
        do {
            paged = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 512))
        } catch let error as CBv2KVError {
            throw XCTSkip("paged backend unavailable on this hardware: \(error)")
        }
        let bank = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
        XCTAssertTrue(
            bank.supportsPackedPrefill,
            "test premise: the paged caches vouch for per-row independence")

        let (engine, recorder) = makeEngine(
            model: gemma, layerKinds: kinds, prefillChunkSize: 16,
            backend: paged, cacheProvider: bank)
        XCTAssertTrue(
            engine.packedPrefillActivity().isSupported,
            "both gates agree on the paged stack too")

        let prompts = (0 ..< 4).map { promptTokens(length: 128, seed: UInt64(4300 + $0)) }
        let runs = try await runConcurrently(engine, prompts: prompts, maxTokens: 4)
        await engine.shutdown()
        for (index, run) in runs.enumerated() {
            XCTAssertEqual(run.finishReason, .length, "paged row \(index) must complete")
        }

        let activity = engine.packedPrefillActivity()
        XCTAssertTrue(
            activity.didExecute,
            "the paged arm must count packed groups, saw \(recorder.prefillShapes)")
        XCTAssertEqual(
            activity.groupsExecuted, recorder.packedShapes.count,
            "one counted group per rectangular prompt forward, on paged as on contiguous")
        XCTAssertEqual(
            activity.rowsExecuted, recorder.packedShapes.reduce(0) { $0 + $1[0] })
    }

    // MARK: - 2. Claimed but unexercised

    /// The same Gemma 4 engine that packs above, fed one request at a time.
    /// A single row can never form a group, so the capability stays true and
    /// the evidence stays zero — the counter cannot be satisfied by
    /// configuration.
    func testClaimedPackedPrefillCapabilityReadsZeroUntilExercised() async throws {
        let (gemma, kinds) = try makeGemma4()
        let (engine, recorder) = makeEngine(
            model: gemma, layerKinds: kinds, prefillChunkSize: 16)

        for index in 0 ..< 4 {
            let collected = await cbv2SchedCollect(
                try engine.submit(
                    request(
                        id: UInt64(index + 1),
                        prompt: promptTokens(length: 64, seed: UInt64(4200 + index)),
                        maxTokens: 3)))
            XCTAssertEqual(collected.finishReason, .length, "row \(index) must complete")
            let activity = engine.packedPrefillActivity()
            XCTAssertTrue(
                activity.isSupported, "row \(index): the capability is claimed throughout")
            XCTAssertEqual(
                activity.rowsExecuted, 0,
                "row \(index): a solo row cannot form a packed group, so a claimed-but-"
                    + "unexercised capability must read zero")
            XCTAssertEqual(activity.groupsExecuted, 0, "row \(index)")
            XCTAssertFalse(activity.didExecute, "row \(index)")
        }
        await engine.shutdown()

        XCTAssertEqual(
            recorder.packedShapes, [],
            "test premise: sequential requests never gave the engine a cohort to pack")
        XCTAssertEqual(
            recorder.prefillShapes.count, 16,
            "test premise: 4 requests x 4 chunks of 16 really were prefilled")
        XCTAssertEqual(
            engine.packedPrefillActivity(),
            CBv2PackedPrefillActivity(isSupported: true, rowsExecuted: 0, groupsExecuted: 0),
            "sixteen prefill chunks through a packing-capable engine, none of them packed")
    }

    // MARK: - 3. Fail-closed default

    /// An engine with no packed-prefill path at all reports neither
    /// capability nor execution, rather than inheriting a truthy default.
    func testEngineWithoutAPackedPrefillPathReportsNoActivity() {
        let engine: any CBv2Engine = CBv2NoPackedPrefillEngine()
        let activity = engine.packedPrefillActivity()
        XCTAssertFalse(activity.isSupported)
        XCTAssertEqual(activity.rowsExecuted, 0)
        XCTAssertEqual(activity.groupsExecuted, 0)
        XCTAssertFalse(activity.didExecute)
        XCTAssertEqual(activity, .none)
    }
}

/// Minimal conformer used only to pin the protocol's fail-closed default.
private final class CBv2NoPackedPrefillEngine: CBv2Engine, @unchecked Sendable {
    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        AsyncStream { $0.finish() }
    }
    func cancel(_ id: CBv2RequestID) {}
    func capacity() -> CBv2CapacitySnapshot { CBv2EngineGauges(kvBytesCapacity: 0).read() }
    func shutdown() async {}
}
