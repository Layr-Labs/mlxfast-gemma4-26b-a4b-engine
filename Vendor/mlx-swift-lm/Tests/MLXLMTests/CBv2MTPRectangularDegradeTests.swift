// CBv2MTPRectangularDegradeTests.swift
//
// WS-3.0: a layer-cache bank that cannot serialise its attention per query
// column MUST degrade MTP target verification to the serial oracle. It must
// never trap.
//
// `EngineLoopV2+MTPTargetVerification` used to reach the serialisation flag
// through `as? CBv2LayerCache` behind a `preconditionFailure`. `CBv2LayerCache`
// is `final`, and every other attending cache — `PagedLayerCache`, and the
// decorator below — is a SIBLING conformer of `CBv2AttendingLayerCache`, never
// a subclass, so that cast could not succeed for a non-contiguous bank. The
// trap was a `fatalError`: daemon death, every co-resident model's in-flight
// requests lost, and no telemetry. It stayed unreachable only because paged
// windowed rows failed the speculative-storage gate first (WS-3.3 opens it).
//
// The gate is now conformance to `CBv2MTPRectangularSerializing`
// (`Paged/PagedSeamContract.swift`), matching the affirmative-capability
// convention already used by `CBv2PackedPrefillCapableCache` and
// `CBv2MultimodalSpanCapableCache`.
//
// Every engine test here runs the REAL engine over the tiny random-init
// Gemma-4 target + assistant drafter used by `CBv2MTPRoundSmokeTests`, on the
// contiguous backend, in forced `.rectangular` mode. The two arms differ in
// exactly ONE axis: whether the bank's caches conform to
// `CBv2MTPRectangularSerializing`. Serial is the correctness oracle, so the
// degraded arm must stay token-exact with both the rectangular arm and the
// MTP-off baseline.

import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - A cache that is exactly a paged cache's shape to the engine

/// Forwards the whole `CBv2AttendingLayerCache` surface to a real
/// `CBv2LayerCache` while deliberately NOT conforming to
/// `CBv2MTPRectangularSerializing`.
///
/// This models a non-contiguous bank structurally: same attention numerics,
/// same last-query and packed-prefill claims, same legacy `KVCache` inner
/// state — so a parity comparison against the plain bank isolates the single
/// axis under test. Conformances are deliberately enumerated rather than
/// inherited; `CBv2LayerCache` is `final`, which is the whole reason the
/// deleted downcast was unsatisfiable.
private final class CBv2NonSerializingLayerCache {
    let inner: CBv2LayerCache

    init(layerIndex: Int, kind: CBv2LayerKind) {
        self.inner = CBv2LayerCache(layerIndex: layerIndex, kind: kind)
    }
}

extension CBv2NonSerializingLayerCache: CBv2AttendingLayerCache {
    var layerIndex: Int { inner.layerIndex }
    var kind: CBv2LayerKind { inner.kind }
    var rows: [CBv2SequenceKV] { inner.rows }
    var positionOffsets: MLXArray { inner.positionOffsets }

    func setRows(_ rows: [CBv2SequenceKV]) { inner.setRows(rows) }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        inner.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        inner.attendBorrowing(source: source, queries: queries, scale: scale, sinks: sinks)
    }
}

extension CBv2NonSerializingLayerCache: CBv2LastQueryPrefillLayerCache {
    func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        inner.updateAndAttendLastQuery(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
    }
}

extension CBv2NonSerializingLayerCache: CBv2SpanMaskBinding, CBv2MultimodalSpanCapableCache {
    func bindSpanContext(_ context: CBv2SpanChunkContext?) { inner.bindSpanContext(context) }
    var honorsSpanMaskContexts: Bool { inner.honorsSpanMaskContexts }
}

extension CBv2NonSerializingLayerCache: CBv2PackedPrefillCapableCache {
    var keepsRowsIndependentWhenPacked: Bool { inner.keepsRowsIndependentWhenPacked }
}

extension CBv2NonSerializingLayerCache: KVCache {
    var offset: Int { inner.offset }
    var maxSize: Int? { inner.maxSize }
    func innerState() -> [MLXArray] { inner.innerState() }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        inner.update(keys: keys, values: values)
    }
    var state: [MLXArray] {
        get { inner.state }
        set { inner.state = newValue }
    }
    var metaState: [String] {
        get { inner.metaState }
        set { inner.metaState = newValue }
    }
    var isTrimmable: Bool { inner.isTrimmable }
    @discardableResult func trim(_ n: Int) -> Int { inner.trim(n) }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        inner.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func copy() -> any KVCache { inner.copy() }
}

// MARK: - Suite

@Suite("CBv2MTPRectangularDegrade", .serialized)
struct CBv2MTPRectangularDegradeTests {

    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16

    /// Telemetry key `mtpBuildTargetVerification` records when it refuses to
    /// run rectangular over a bank it cannot serialise.
    private let degradeReason = "rectangular_cache_unsupported"

    // MARK: Fixtures (same shapes as CBv2MTPRoundSmokeTests)

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func drafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": \(hiddenSize),
                "use_ordered_embeddings": false,
                "num_centroids": 16,
                "centroid_intermediate_top_k": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 32,
                    "global_head_dim": 32,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": \(slidingWindow),
                    "final_logit_softcapping": null,
                    "tie_word_embeddings": true,
                    "vocab_size": \(vocabSize),
                    "vocab_size_per_layer_input": \(vocabSize),
                    "rms_norm_eps": 1e-6,
                    "hidden_size_per_layer_input": 0,
                    "use_double_wide_mlp": false
                }
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    private struct Fixture {
        let target: Gemma4TextModel
        let drafter: Gemma4AssistantDraftModel
    }

    private func makeFixture(seed: UInt64 = 0x5EED) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    /// `serializing: false` swaps every cache for the non-conforming
    /// decorator; nothing else about the engine changes.
    private func makeEngine(
        _ fixture: Fixture, mtp: Bool, serializing: Bool = true
    ) throws -> EngineV2 {
        let kinds = fixture.target.cbv2LayerKinds
        let caches: [any CBv2AttendingLayerCache] =
            serializing
            ? kinds.enumerated().map { CBv2LayerCache(layerIndex: $0.offset, kind: $0.element) }
            : kinds.enumerated().map {
                CBv2NonSerializingLayerCache(layerIndex: $0.offset, kind: $0.element)
            }
        let mtpDrafter: Gemma4CBv2MTPDrafter? =
            mtp
            ? try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
            : nil
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: mtpDrafter,
            mtpConfig: CBv2MTPConfig(
                enabled: mtp, maxDraftTokens: 2, maxSpeculativeBatch: 2,
                fixedDraftTokens: 2,
                // Forced rectangular: the strategy switch must be overruled
                // by the conformance check, not by the automatic work cap.
                verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 8))
    }

    private func run(_ engine: EngineV2, _ request: CBv2Request) async throws
        -> CBv2SchedCollected
    {
        await cbv2SchedCollect(try engine.submit(request))
    }

    private func greedyRequest(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0),
            maxTokens: maxTokens)
    }

    // MARK: - (1) Non-conforming bank degrades instead of aborting

    /// The acceptance test for WS-3.0. Against the deleted `as? CBv2LayerCache`
    /// guard this did not fail — it killed the test process with
    /// "MTP rectangular verification requires CBv2 layer caches".
    @Test func nonSerializingBankDegradesToSerialInsteadOfTrapping() async throws {
        let fixture = try makeFixture()
        // Prompt 24 > window 16 with 32 generated tokens, so verify rounds
        // really run and the sliding ring wraps through them.
        let prompt = makePromptTokens(length: 24, seed: 11, vocabSize: vocabSize)

        // Same non-conforming bank on BOTH legs, so the only variable is MTP.
        let off = try makeEngine(fixture, mtp: false, serializing: false)
        let baseline = try await run(off, greedyRequest(id: 1, prompt: prompt, maxTokens: 32))
        await off.shutdown()
        #expect(baseline.tokens.count == 32)

        let on = try makeEngine(fixture, mtp: true, serializing: false)
        let degraded = try await run(on, greedyRequest(id: 1, prompt: prompt, maxTokens: 32))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()

        // Liveness: the daemon survived a bank it cannot serialise.
        #expect(degraded.finishReason == .length)
        #expect(metrics.rounds >= 1, "no verify round ran — the degrade path was never exercised")

        // Strategy: every round took the serial oracle, and said so.
        #expect(metrics.rectangularVerificationRounds == 0)
        #expect(metrics.serialVerificationRounds >= 1)
        #expect(
            metrics.controllerFallbacks[degradeReason, default: 0]
                == metrics.serialVerificationRounds,
            "degrade must be reported once per round, not silently")

        // Correctness: serial is the oracle, so degrading is lossless.
        #expect(
            degraded.tokens == baseline.tokens,
            "degraded MTP diverged: on=\(degraded.tokens) off=\(baseline.tokens)")
    }

    // MARK: - (2) Control arm: a conforming bank still runs rectangular

    /// Proves the degrade in (1) is caused by non-conformance and nothing
    /// else: identical fixture, identical config, conforming caches.
    @Test func serializingBankStillRunsRectangular() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 24, seed: 11, vocabSize: vocabSize)

        let off = try makeEngine(fixture, mtp: false)
        let baseline = try await run(off, greedyRequest(id: 2, prompt: prompt, maxTokens: 32))
        await off.shutdown()

        let on = try makeEngine(fixture, mtp: true, serializing: true)
        let rectangular = try await run(on, greedyRequest(id: 2, prompt: prompt, maxTokens: 32))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()

        #expect(metrics.rounds >= 1)
        #expect(metrics.rectangularVerificationRounds >= 1)
        #expect(metrics.controllerFallbacks[degradeReason, default: 0] == 0)
        #expect(rectangular.tokens == baseline.tokens)
    }

    // MARK: - (3) Why the deleted cast was a daemon abort, not a defensive assert

    /// `CBv2LayerCache` is `final` and `PagedLayerCache` is a sibling
    /// conformer, so `as? CBv2LayerCache` can NEVER succeed for a paged bank
    /// — whatever else that bank conforms to. The old guard was therefore an
    /// unconditional `fatalError` for paged, armed the moment WS-3.3 let a
    /// paged row into a round.
    @Test func pagedBankCanNeverSatisfyTheDeletedContiguousCast() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .slidingWindow(32), headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 8 << 20, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 1024))
        let caches = backend.makeLayerCaches()
        #expect(caches.count == kinds.count)
        for cache in caches {
            #expect(
                !(cache is CBv2LayerCache),
                "a paged cache can never be the final contiguous class")
        }
    }

    // MARK: - (4) WS-3.5: the round's captures are fenced ahead of its writes

    private func pagedFixture(
        heads: Int, dim: Int
    ) throws -> (backend: PagedKVBackend, row: PagedSequenceKV, state: [CBv2SequenceKV?]) {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: dim, kvHeads: heads, queryHeads: heads * 2)
        ]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 8 << 20, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 1024))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let row = try #require(state[0] as? PagedSequenceKV)
        return (backend, row, state)
    }

    private func tile(_ value: Float, heads: Int, tokens: Int, dim: Int) -> MLXArray {
        MLXArray(
            [Float](repeating: value, count: heads * tokens * dim), [heads, tokens, dim]
        ).asType(.float16)
    }

    /// The back-edge exists and is value-neutral. `MLXArray` is a class, so
    /// `!==` is an exact statement that the group's fence is now a NEW graph
    /// node — which is the whole mechanism: every subsequent write consumes
    /// `group.writeFence`, so it now transitively depends on the captures.
    @Test func publishAddsAValueNeutralBackEdgeToTheGroupFence() throws {
        let (heads, dim) = (2, 64)
        let fixture = try pagedFixture(heads: heads, dim: dim)
        defer { fixture.backend.release(fixture.state) }
        let row = fixture.row
        row.write(
            keys: tile(1, heads: heads, tokens: 8, dim: dim),
            values: tile(2, heads: heads, tokens: 8, dim: dim))
        let snapshot = row.snapshot()

        let group = fixture.backend.pool.group(row.groupKey)
        let fenceBefore = group.writeFence
        eval(fenceBefore)
        let valueBefore = fenceBefore.item(Int32.self)

        let unfenceable = CBv2MTPCaptureFence.publish(
            [(row: row as CBv2SequenceKV, keys: snapshot.keys, values: snapshot.values)])

        #expect(unfenceable.isEmpty, "a paged row must be fenceable by graph edge, not by eval")
        #expect(group.writeFence !== fenceBefore, "no back-edge was published")
        eval(group.writeFence)
        #expect(
            group.writeFence.item(Int32.self) == valueBefore,
            "the back-edge must not change the fence's value")
    }

    /// A capture with no tokens is a no-op, not a trap and not an eval.
    ///
    /// `PagedKVPool.gather` returns `[1, H, 0, D]` for `count == 0` and never
    /// builds a page index, so an empty capture read no slab bytes and has
    /// nothing for a later write to clobber. It also has no element for the
    /// fence probe to index — the reason the probe is guarded rather than
    /// unconditional. The row must NOT be reported unfenceable: that would
    /// send the caller off to `eval` an empty array.
    @Test func publishSkipsAZeroTokenCaptureWithoutTrappingOrEvaluating() throws {
        let (heads, dim) = (2, 64)
        let fixture = try pagedFixture(heads: heads, dim: dim)
        defer { fixture.backend.release(fixture.state) }
        let row = fixture.row
        // Never written: `retainedCount` is 0, so the snapshot is empty.
        let snapshot = row.snapshot()
        #expect(snapshot.keys.dim(2) == 0, "an unwritten row must snapshot to zero tokens")

        let group = fixture.backend.pool.group(row.groupKey)
        let fenceBefore = group.writeFence
        eval(fenceBefore)
        let valueBefore = fenceBefore.item(Int32.self)

        let unfenceable = CBv2MTPCaptureFence.publish(
            [(row: row as CBv2SequenceKV, keys: snapshot.keys, values: snapshot.values)])

        #expect(unfenceable.isEmpty, "an empty capture needs no eval fallback")
        #expect(
            group.writeFence === fenceBefore,
            "an empty capture read nothing, so it must not publish an edge")
        eval(group.writeFence)
        #expect(group.writeFence.item(Int32.self) == valueBefore)
    }

    /// The invariant the edge buys: a write that lands on the captured slots
    /// after `publish` cannot be observed by the capture, even when both are
    /// forced in one `eval`. Rollback-then-rewrite is the strongest possible
    /// aliasing — the round's new tokens reuse the exact physical slots the
    /// capture named.
    @Test func capturedBytesSurviveAWriteOverTheSameSlots() throws {
        let (heads, dim) = (2, 64)
        let fixture = try pagedFixture(heads: heads, dim: dim)
        defer { fixture.backend.release(fixture.state) }
        let row = fixture.row
        row.write(
            keys: tile(1, heads: heads, tokens: 8, dim: dim),
            values: tile(2, heads: heads, tokens: 8, dim: dim))
        let snapshot = row.snapshot()
        let group = fixture.backend.pool.group(row.groupKey)

        CBv2MTPCaptureFence.publish(
            [(row: row as CBv2SequenceKV, keys: snapshot.keys, values: snapshot.values)])

        // The "round": roll back to position 4 and write different bytes
        // into positions 4..<8 — the same pages, the same slots.
        row.rollback(4)
        row.write(
            keys: tile(7, heads: heads, tokens: 4, dim: dim),
            values: tile(9, heads: heads, tokens: 4, dim: dim))

        // One eval, captures and write together: exactly the shape of the
        // round's single `asyncEval`.
        eval([snapshot.keys, snapshot.values, group.writeFence])

        let keyTail = snapshot.keys[0..., 0..., 4 ..< 8, 0...].asArray(Float.self)
        let valueTail = snapshot.values[0..., 0..., 4 ..< 8, 0...].asArray(Float.self)
        #expect(
            keyTail.allSatisfy { $0 == 1 },
            "capture observed the round's own keys — the fence back-edge is missing")
        #expect(
            valueTail.allSatisfy { $0 == 2 },
            "capture observed the round's own values — the fence back-edge is missing")
    }

    /// A row with no reachable write fence must be REPORTED, not silently
    /// skipped: the caller turns the residue into a real `eval`. This is the
    /// only protection a future recyclable-but-unfenced backend would get.
    @Test func rowsWithNoReachableFenceAreReportedForEval() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 22))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 64)
        defer { backend.release(state) }
        let row = try #require(state[0])
        let keys = MLXArray([Float](repeating: 0, count: 4), [4])
        let values = MLXArray([Float](repeating: 0, count: 4), [4])

        let unfenceable = CBv2MTPCaptureFence.publish(
            [(row: row, keys: keys, values: values)])

        #expect(unfenceable.count == 2)
        #expect(unfenceable[0] === keys)
        #expect(unfenceable[1] === values)
    }
}
