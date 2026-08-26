// CBv2LastQueryPrefillTests.swift
//
// Final-layer prompt specialization for ContinuousBatchingV2:
//  - `CBv2AttentionV1.updateAndAttendLastQuery` / `CBv2LayerCache`
//    (Libraries/MLXLMCommon/ContinuousBatchingV2/{AttentionV1,LayerCacheV2,
//    LastQueryPrefillV2}.swift), and
//  - the Gemma-4 policy seam + trunk wiring
//    (Libraries/MLXLLM/Models/Gemma4Text.swift).
//
// What each suite pins:
//
//  1. `CBv2LastQueryPrefillAttentionTests` — the primitive is EXACTLY the
//     last row of ordinary chunk attention: B == 1, B > 1, and a chunk that
//     lands on pre-existing cached history (offset > 0).
//  2. `CBv2LastQueryPrefillCacheTests` — offsets advance by the K/V length
//     (L), NOT by the query length (1), and the committed K/V bytes are
//     identical to what `updateAndAttend` would have stored.
//  3. `CBv2LastQueryPrefillGatingTests` — every geometry the primitive's
//     preconditions reject (sliding window, KV-shared, qL != 1, kvL == 1)
//     is already excluded by the POSITIVE gate `gemma4UseLastQueryPrefill`,
//     so the preconditions are unreachable. No test here intentionally
//     crashes the process.
//  4. `CBv2LastQueryPrefillPolicyTests` — the full truth table of
//     `gemma4UseLastQueryPrefill`, one false branch per case.
//  5. `CBv2LastQueryPrefillLayerParityTests` — a real `Gemma4DecoderLayer`
//     final layer driven three ways on identical fresh CBv2 caches:
//     unnarrowed, tail-narrowed, tail-narrowed + last-query. Deterministic
//     (the toggles are parameters, not env).
//  6. `CBv2LastQueryPrefillModelParityTests` — a TINY synthetic Gemma text
//     model, multi-chunk prompt through the CBv2 prompt trunk
//     (`cbv2Prefill`) vs the ordinary full forward: frontier logits within
//     tolerance, identical greedy token, bit-identical KV in EVERY layer.
//
// Tiny synthetic configs with seeded random weights only — no checkpoint,
// no download.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Shared helpers

/// Largest elementwise |a - b| in float32. Both operands are evaluated.
private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
    let diff = MLX.abs(a.asType(.float32) - b.asType(.float32))
    let m = diff.size == 0 ? MLXArray(Float(0)) : diff.max()
    eval(m)
    return m.item(Float.self)
}

/// Bit-for-bit equality (no tolerance) — used for KV state, which must be
/// produced by the identical projection graph on every path.
private func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape else { return false }
    if a.size == 0 { return true }
    let same = MLX.all(a .== b)
    eval(same)
    return same.item(Bool.self)
}

/// Full-attention layer kind with no sinks and no KV sharing.
private func fullKind(
    headDim: Int = 8, kvHeads: Int = 2, queryHeads: Int = 4
) -> CBv2LayerKind {
    CBv2LayerKind(
        attention: .full, sharesKVWithLayer: nil, hasSinks: false,
        headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
}

private func makeFullRows(
    count: Int, kvHeads: Int, headDim: Int, maxLength: Int
) -> [CBv2FullSequenceKV] {
    (0 ..< count).map { _ in
        CBv2FullSequenceKV(
            promptLength: maxLength, maxLength: maxLength,
            kvHeads: kvHeads, headDim: headDim)
    }
}

// MARK: - 1. Attention primitive equivalence

@Suite("CBv2LastQueryPrefill attention primitive")
struct CBv2LastQueryPrefillAttentionTests {

    private struct Fixture {
        var kind: CBv2LayerKind
        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        var history: (keys: MLXArray, values: MLXArray)?
        var scale: Float
    }

    private func fixture(
        batch: Int, chunk: Int, historyLength: Int, seed: UInt64
    ) -> Fixture {
        MLXRandom.seed(seed)
        let kind = fullKind()
        let q = MLXRandom.normal([batch, kind.queryHeads, chunk, kind.headDim])
        let k = MLXRandom.normal([batch, kind.kvHeads, chunk, kind.headDim])
        let v = MLXRandom.normal([batch, kind.kvHeads, chunk, kind.headDim])
        var history: (keys: MLXArray, values: MLXArray)?
        if historyLength > 0 {
            history = (
                MLXRandom.normal([batch, kind.kvHeads, historyLength, kind.headDim]),
                MLXRandom.normal([batch, kind.kvHeads, historyLength, kind.headDim])
            )
        }
        eval(q, k, v)
        if let history { eval(history.keys, history.values) }
        return Fixture(
            kind: kind, queries: q, keys: k, values: v, history: history,
            scale: 1.0 / Float(kind.headDim).squareRoot())
    }

    /// Seed `count` fresh rows with the fixture's history (written through
    /// the SAME `update` path both sides use, so the two runs start from
    /// byte-identical cache state).
    private func seededRows(_ f: Fixture, count: Int, maxLength: Int) -> [CBv2FullSequenceKV] {
        let rows = makeFullRows(
            count: count, kvHeads: f.kind.kvHeads, headDim: f.kind.headDim,
            maxLength: maxLength)
        guard let history = f.history else { return rows }
        for (index, row) in rows.enumerated() {
            _ = row.update(
                keys: history.keys[index ..< index + 1],
                values: history.values[index ..< index + 1])
        }
        return rows
    }

    private func assertLastQueryMatchesFinalRow(
        batch: Int, chunk: Int, historyLength: Int, seed: UInt64,
        _ label: Comment
    ) {
        let f = fixture(batch: batch, chunk: chunk, historyLength: historyLength, seed: seed)
        let maxLength = chunk + historyLength + 8

        // Reference: ordinary chunk attention over the whole query rectangle.
        let referenceRows = seededRows(f, count: batch, maxLength: maxLength)
        let reference = CBv2AttentionV1.updateAndAttend(
            rows: referenceRows, kind: f.kind,
            queries: f.queries, keys: f.keys, values: f.values,
            scale: f.scale, sinks: nil)
        let referenceLast = reference[0..., 0..., (chunk - 1) ..< chunk, 0...]

        // Specialization: the frontier query row only, same K/V rectangle,
        // on a fresh identical row.
        let lastQueryRows = seededRows(f, count: batch, maxLength: maxLength)
        let qLast = f.queries[0..., 0..., (chunk - 1) ..< chunk, 0...]
        let specialized = CBv2AttentionV1.updateAndAttendLastQuery(
            rows: lastQueryRows, kind: f.kind,
            queries: qLast, keys: f.keys, values: f.values,
            scale: f.scale, sinks: nil)

        eval(referenceLast, specialized)
        #expect(
            specialized.shape == [batch, f.kind.queryHeads, 1, f.kind.headDim],
            "last-query output must be [B, queryHeads, 1, headDim] — \(label)")
        #expect(
            maxAbsDiff(specialized, referenceLast) < 2e-6,
            "last-query attention must equal the final row of chunk attention — \(label)")

        // Both sides must also have committed the SAME cache state.
        for index in 0 ..< batch {
            #expect(
                referenceRows[index].absoluteOffset == lastQueryRows[index].absoluteOffset,
                "row \(index) offset — \(label)")
            let refSnap = referenceRows[index].snapshot()
            let lqSnap = lastQueryRows[index].snapshot()
            #expect(bitIdentical(refSnap.keys, lqSnap.keys), "row \(index) keys — \(label)")
            #expect(bitIdentical(refSnap.values, lqSnap.values), "row \(index) values — \(label)")
        }
    }

    @Test func singleRowChunkMatchesFinalQueryRow() {
        assertLastQueryMatchesFinalRow(
            batch: 1, chunk: 12, historyLength: 0, seed: 0xB1_0C,
            "B=1, fresh row")
    }

    @Test func batchedRowsMatchFinalQueryRow() {
        assertLastQueryMatchesFinalRow(
            batch: 3, chunk: 9, historyLength: 0, seed: 0xB2_0C,
            "B=3, fresh rows")
    }

    @Test func chunkOnExistingHistoryMatchesFinalQueryRow() {
        // offset > 0: the chunk's newest query must see history + chunk.
        assertLastQueryMatchesFinalRow(
            batch: 1, chunk: 7, historyLength: 11, seed: 0xB3_0C,
            "B=1, offset 11")
    }

    @Test func batchedChunkOnExistingHistoryMatchesFinalQueryRow() {
        assertLastQueryMatchesFinalRow(
            batch: 2, chunk: 6, historyLength: 5, seed: 0xB4_0C,
            "B=2, offset 5")
    }

    /// The specialization is mask-free by construction: its result must be
    /// invariant to how much history precedes the chunk only through the
    /// keys it actually attends — i.e. it is exactly `attend(qLast, all KV)`.
    /// A query row that ignored the chunk's own keys would not match.
    @Test func specializedRowDependsOnTheWholeChunk() {
        let f = fixture(batch: 1, chunk: 8, historyLength: 0, seed: 0xB5_0C)
        let rowsA = seededRows(f, count: 1, maxLength: 32)
        let qLast = f.queries[0..., 0..., 7 ..< 8, 0...]
        let outA = CBv2AttentionV1.updateAndAttendLastQuery(
            rows: rowsA, kind: f.kind, queries: qLast,
            keys: f.keys, values: f.values, scale: f.scale, sinks: nil)

        // Perturb an EARLY key of the chunk: the frontier query attends it,
        // so the output must move.
        var perturbedKeys = f.keys
        perturbedKeys[0..., 0..., 0 ..< 1, 0...] = f.keys[0..., 0..., 0 ..< 1, 0...] + 3.0
        let rowsB = seededRows(f, count: 1, maxLength: 32)
        let outB = CBv2AttentionV1.updateAndAttendLastQuery(
            rows: rowsB, kind: f.kind, queries: qLast,
            keys: perturbedKeys, values: f.values, scale: f.scale, sinks: nil)

        eval(outA, outB)
        #expect(
            maxAbsDiff(outA, outB) > 1e-4,
            "the frontier query must attend the chunk's own earliest key")
    }
}

// MARK: - 2. Cache offset + retained-content semantics

@Suite("CBv2LastQueryPrefill cache semantics")
struct CBv2LastQueryPrefillCacheTests {

    @Test func offsetsAdvanceByKeyLengthNotQueryLength() {
        MLXRandom.seed(0xC0_FF)
        let kind = fullKind()
        let batch = 2
        let chunk = 10
        let scale = 1.0 / Float(kind.headDim).squareRoot()

        let queries = MLXRandom.normal([batch, kind.queryHeads, chunk, kind.headDim])
        let keys = MLXRandom.normal([batch, kind.kvHeads, chunk, kind.headDim])
        let values = MLXRandom.normal([batch, kind.kvHeads, chunk, kind.headDim])
        eval(queries, keys, values)

        let referenceRows = makeFullRows(
            count: batch, kvHeads: kind.kvHeads, headDim: kind.headDim, maxLength: 64)
        let lastQueryRows = makeFullRows(
            count: batch, kvHeads: kind.kvHeads, headDim: kind.headDim, maxLength: 64)

        let referenceCache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: referenceRows)
        let lastQueryCache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: lastQueryRows)

        let referenceOffsetsBefore = referenceCache.positionOffsets + 0
        let lastQueryOffsetsBefore = lastQueryCache.positionOffsets + 0
        eval(referenceOffsetsBefore, lastQueryOffsetsBefore)

        _ = referenceCache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        _ = lastQueryCache.updateAndAttendLastQuery(
            queries: queries[0..., 0..., (chunk - 1) ..< chunk, 0...],
            keys: keys, values: values, scale: scale, sinks: nil)

        // Host offsets: advanced by L, not by 1.
        for index in 0 ..< batch {
            #expect(
                lastQueryRows[index].absoluteOffset == chunk,
                "row \(index) must consume all \(chunk) positions, not 1")
            #expect(
                lastQueryRows[index].absoluteOffset == referenceRows[index].absoluteOffset,
                "row \(index) offset must match ordinary chunk attention")
            #expect(lastQueryRows[index].retainedCount == chunk, "row \(index) retained")
        }
        // Device offsets follow the same rule (the model reads these for RoPE).
        #expect(
            bitIdentical(lastQueryCache.positionOffsets, referenceCache.positionOffsets),
            "positionOffsets must advance by keys.dim(2)")
        #expect(
            bitIdentical(
                lastQueryCache.positionOffsets, lastQueryOffsetsBefore + Int32(chunk)),
            "positionOffsets must advance by exactly \(chunk)")
        // Legacy scalar view agrees.
        #expect(lastQueryCache.offset == chunk)
        // No host rebuild happened inside the step.
        #expect(lastQueryCache.positionOffsetsHostRebuilds == 0)

        // Retained content is byte-identical to the ordinary path's.
        for index in 0 ..< batch {
            let refSnap = referenceRows[index].snapshot()
            let lqSnap = lastQueryRows[index].snapshot()
            #expect(refSnap.offset == lqSnap.offset, "row \(index) snapshot offset")
            #expect(bitIdentical(refSnap.keys, lqSnap.keys), "row \(index) retained keys")
            #expect(bitIdentical(refSnap.values, lqSnap.values), "row \(index) retained values")
            // ...and equals the source rectangle written verbatim.
            #expect(
                bitIdentical(lqSnap.keys, keys[index ..< index + 1]),
                "row \(index) keys must be the chunk verbatim")
            #expect(
                bitIdentical(lqSnap.values, values[index ..< index + 1]),
                "row \(index) values must be the chunk verbatim")
        }
    }

    @Test func offsetsAdvanceByKeyLengthAcrossSuccessiveChunks() {
        MLXRandom.seed(0xC1_FF)
        let kind = fullKind(headDim: 8, kvHeads: 1, queryHeads: 2)
        let scale = 1.0 / Float(kind.headDim).squareRoot()
        let rows = makeFullRows(
            count: 1, kvHeads: kind.kvHeads, headDim: kind.headDim, maxLength: 64)
        let cache = CBv2LayerCache(layerIndex: 3, kind: kind, rows: rows)

        var expectedOffset = 0
        for chunk in [5, 9, 4] {
            let queries = MLXRandom.normal([1, kind.queryHeads, 1, kind.headDim])
            let keys = MLXRandom.normal([1, kind.kvHeads, chunk, kind.headDim])
            let values = MLXRandom.normal([1, kind.kvHeads, chunk, kind.headDim])
            eval(queries, keys, values)
            _ = cache.updateAndAttendLastQuery(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
            expectedOffset += chunk
            #expect(rows[0].absoluteOffset == expectedOffset)
        }
        eval(cache.positionOffsets)
        #expect(bitIdentical(cache.positionOffsets, MLXArray([Int32(expectedOffset)])))
        #expect(cache.positionOffsetsHostRebuilds == 0)
    }

    /// The contiguous cache is the capability claim the model policy reads.
    @Test func contiguousLayerCacheIsCapable() {
        // Erased to the protocol the model actually probes.
        let cache: any CBv2AttendingLayerCache = CBv2LayerCache(layerIndex: 0, kind: fullKind())
        #expect(cache is any CBv2LastQueryPrefillLayerCache)
    }
}

// MARK: - Config fixtures (tiny synthetic Gemma text configs)

private enum TinyGemma {

    /// `layerTypes` verbatim, tiny dims, PLE + MoE off, dense MLP.
    static func config(
        layerTypes: [String],
        numKvSharedLayers: Int = 0,
        hiddenSize: Int = 32,
        headDim: Int = 8,
        globalHeadDim: Int = 16,
        slidingWindow: Int = 16,
        vocabSize: Int = 64
    ) throws -> Gemma4TextConfiguration {
        let types = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": \(layerTypes.count),
                "intermediate_size": \(hiddenSize * 2),
                "num_attention_heads": 2,
                "head_dim": \(headDim),
                "global_head_dim": \(globalHeadDim),
                "num_key_value_heads": 1,
                "num_kv_shared_layers": \(numKvSharedLayers),
                "layer_types": [\(types)],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// Final layer is full attention and owns its K/V — the shape the
    /// specialization is defined for.
    static func lastQueryCapableConfig() throws -> Gemma4TextConfiguration {
        try config(layerTypes: [
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
        ])
    }

    /// Final layer is sliding — tail narrowing still applies, last-query
    /// prefill must not.
    static func slidingFinalConfig() throws -> Gemma4TextConfiguration {
        try config(layerTypes: [
            "full_attention", "sliding_attention", "full_attention", "sliding_attention",
        ])
    }

    /// Final layer borrows K/V from an earlier layer (writes nothing).
    static func sharedFinalConfig() throws -> Gemma4TextConfiguration {
        try config(
            layerTypes: [
                "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            ],
            numKvSharedLayers: 2)
    }

    /// Per-layer contiguous CBv2 caches with one bound row each (storage
    /// owners only), exactly as the engine binds them.
    static func caches(
        for config: Gemma4TextConfiguration, promptLength: Int, maxLength: Int
    ) throws -> (backend: CBv2ContiguousKVBackend, caches: [CBv2LayerCache]) {
        let kinds = config.cbv2LayerKinds
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        let caches = kinds.enumerated().map {
            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
        }
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: promptLength, maxLength: maxLength)
        for (index, kind) in kinds.enumerated() where kind.sharesKVWithLayer == nil {
            guard let row = state[index] else {
                Issue.record("storage-owning layer \(index) has no row")
                continue
            }
            caches[index].setRows([row])
        }
        return (backend, caches)
    }
}

// MARK: - Env knobs, mirrored so the fixtures adapt instead of going vacuous

/// The trunk's prompt-path knobs are read into file-scope `let`s in
/// Gemma4Text.swift, so a test process cannot flip them at runtime. Mirror
/// the SAME parsing here so the fixtures pick chunk sizes that genuinely
/// straddle the threshold and the expectations track any override.
private enum PrefillKnobs {
    static let tailRows: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_TAIL_ROWS"], let value = Int(raw)
        else { return 1 }
        return max(0, value)
    }()

    static let minChunk: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_TAIL_MIN_CHUNK"], let value = Int(raw)
        else { return 128 }
        return max(2, value)
    }()

    static let lastQueryEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_LAST_QUERY"]
        else { return true }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// A chunk long enough for the trunk's final-layer specializations.
    static var narrowingChunk: Int { max(minChunk, 8) + 32 }
    /// A chunk too short for them.
    static var unnarrowedChunk: Int { max(2, minChunk - 1) }
}

/// Wraps a real `CBv2LayerCache` and records which attention entry point the
/// trunk dispatched to. Conforms to `CBv2LastQueryPrefillLayerCache` so the
/// model's capability probe sees a capable cache.
private final class SpyLayerCache: CBv2LastQueryPrefillLayerCache, KVCache {
    let inner: CBv2LayerCache
    private(set) var updateAndAttendCalls = 0
    private(set) var lastQueryCalls = 0
    private(set) var lastQueryShapes: [(q: [Int], k: [Int])] = []

    init(_ inner: CBv2LayerCache) { self.inner = inner }

    var layerIndex: Int { inner.layerIndex }
    var kind: CBv2LayerKind { inner.kind }
    var rows: [CBv2SequenceKV] { inner.rows }
    func setRows(_ rows: [CBv2SequenceKV]) { inner.setRows(rows) }
    var positionOffsets: MLXArray { inner.positionOffsets }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        updateAndAttendCalls += 1
        return inner.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
    }

    func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        lastQueryCalls += 1
        lastQueryShapes.append((queries.shape, keys.shape))
        return inner.updateAndAttendLastQuery(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        inner.attendBorrowing(source: source, queries: queries, scale: scale, sinks: sinks)
    }

    // Legacy KVCache passthrough (the trunk carries caches as [KVCache]).
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

/// A CBv2 attending cache that does NOT claim the last-query capability —
/// stands in for the paged / custom backends the model must fall back for.
private final class IncapableLayerCache: CBv2AttendingLayerCache {
    let layerIndex: Int
    let kind: CBv2LayerKind
    private(set) var storedRows: [CBv2SequenceKV] = []

    init(layerIndex: Int, kind: CBv2LayerKind) {
        self.layerIndex = layerIndex
        self.kind = kind
    }

    var rows: [CBv2SequenceKV] { storedRows }
    func setRows(_ rows: [CBv2SequenceKV]) { storedRows = rows }
    var positionOffsets: MLXArray { MLXArray(storedRows.map { Int32($0.absoluteOffset) }) }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        CBv2AttentionV1.updateAndAttend(
            rows: storedRows, kind: kind, queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        CBv2AttentionV1.attendBorrowing(
            sourceRows: source.rows, sourceKind: source.kind, kind: kind,
            queries: queries, scale: scale, sinks: sinks)
    }
}

// MARK: - 3. Preconditions are unreachable (positive gating)

@Suite("CBv2LastQueryPrefill precondition gating")
struct CBv2LastQueryPrefillGatingTests {

    /// The primitive's preconditions demand: full attention, storage-owning,
    /// qL == 1, kvL > 1. Each is guaranteed BEFORE the call by the policy
    /// function, so none of them is reachable from the model. These tests
    /// assert the positive gate instead of tripping the trap.

    @Test func slidingFinalLayerIsGatedOff() throws {
        let config = try TinyGemma.slidingFinalConfig()
        let last = config.numHiddenLayers - 1
        #expect(config.layerTypes[last] == "sliding_attention")
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        #expect(
            gemma4UseLastQueryPrefill(
                config, layerIdx: last, batchSize: 1, sequenceLength: 256,
                outputTailRows: 1, hasCapableCache: true, enabled: true) == false,
            "a sliding-window final layer must never reach the full-attention precondition")
    }

    @Test func kvSharedFinalLayerIsGatedOff() throws {
        let config = try TinyGemma.sharedFinalConfig()
        let last = config.numHiddenLayers - 1
        #expect(config.layerTypes[last] == "full_attention")
        #expect(config.layerUsesSharedKV(layerIdx: last))
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        #expect(
            gemma4UseLastQueryPrefill(
                config, layerIdx: last, batchSize: 1, sequenceLength: 256,
                outputTailRows: 1, hasCapableCache: true, enabled: true) == false,
            "a KV-shared final layer owns no storage to commit")
    }

    /// `qL != 1` can only arise from a tail wider than one row; the policy
    /// requires `outputTailRows == 1`, and `forwardV2` derives
    /// `outputStart == L - 1` from it, so the query rectangle is always
    /// exactly one row when the specialization runs.
    @Test func multiRowTailIsGatedOff() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let last = config.numHiddenLayers - 1
        for tail in [nil, 0, 2, 8] as [Int?] {
            #expect(
                gemma4UseLastQueryPrefill(
                    config, layerIdx: last, batchSize: 1, sequenceLength: 256,
                    outputTailRows: tail, hasCapableCache: true, enabled: true) == false,
                "outputTailRows \(String(describing: tail)) must not produce qL != 1")
        }
    }

    /// `kvL == 1` is the decode geometry; the policy requires `L > 1`.
    @Test func singleTokenChunkIsGatedOff() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let last = config.numHiddenLayers - 1
        for length in [0, 1] {
            #expect(
                gemma4UseLastQueryPrefill(
                    config, layerIdx: last, batchSize: 1, sequenceLength: length,
                    outputTailRows: 1, hasCapableCache: true, enabled: true) == false,
                "sequenceLength \(length) must not reach the kvL > 1 precondition")
        }
    }

    /// Capability is claimed by the CACHE CLASS, not by the layer kind: a
    /// `CBv2LayerCache` conforms even for a sliding layer, which is exactly
    /// why the config predicate — not the conformance — is the attention
    /// gate. Non-contiguous backends drop the capability entirely.
    @Test func capabilityIsAClassClaimNotAKindClaim() {
        let slidingKind = CBv2LayerKind(
            attention: .slidingWindow(16), headDim: 8, kvHeads: 1, queryHeads: 2)
        let sliding: any CBv2AttendingLayerCache = CBv2LayerCache(
            layerIndex: 0, kind: slidingKind)
        let incapable: any CBv2AttendingLayerCache = IncapableLayerCache(
            layerIndex: 0, kind: fullKind())
        #expect(sliding is any CBv2LastQueryPrefillLayerCache)
        #expect((incapable is any CBv2LastQueryPrefillLayerCache) == false)
    }

    @Test func incapableCacheIsGatedOff() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let last = config.numHiddenLayers - 1
        let cache: any CBv2AttendingLayerCache = IncapableLayerCache(
            layerIndex: last, kind: config.cbv2LayerKinds[last])
        #expect(
            gemma4UseLastQueryPrefill(
                config, layerIdx: last, batchSize: 1, sequenceLength: 256,
                outputTailRows: 1,
                hasCapableCache: cache is any CBv2LastQueryPrefillLayerCache,
                enabled: true) == false)
    }
}

// MARK: - 4. Policy truth table

@Suite("CBv2LastQueryPrefill policy truth table")
struct CBv2LastQueryPrefillPolicyTests {

    private func decide(
        _ config: Gemma4TextConfiguration,
        layerIdx: Int? = nil,
        batchSize: Int = 1,
        sequenceLength: Int = 256,
        outputTailRows: Int? = 1,
        hasCapableCache: Bool = true,
        enabled: Bool = true
    ) -> Bool {
        gemma4UseLastQueryPrefill(
            config,
            layerIdx: layerIdx ?? (config.numHiddenLayers - 1),
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            outputTailRows: outputTailRows,
            hasCapableCache: hasCapableCache,
            enabled: enabled)
    }

    @Test func allConditionsTrue() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        #expect(gemma4SupportsLastQueryPrefill(config))
        #expect(decide(config), "the fully-satisfied case must select last-query prefill")
    }

    @Test func disabledByKillSwitch() throws {
        #expect(decide(try TinyGemma.lastQueryCapableConfig(), enabled: false) == false)
    }

    @Test func incapableCache() throws {
        #expect(decide(try TinyGemma.lastQueryCapableConfig(), hasCapableCache: false) == false)
    }

    @Test func tailRowsMustBeExactlyOne() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        #expect(decide(config, outputTailRows: nil) == false)
        #expect(decide(config, outputTailRows: 0) == false)
        #expect(decide(config, outputTailRows: 2) == false)
        #expect(decide(config, outputTailRows: 1))
    }

    @Test func onlyTheFinalLayer() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        for idx in 0 ..< (config.numHiddenLayers - 1) {
            #expect(decide(config, layerIdx: idx) == false, "layer \(idx) is not final")
        }
        #expect(decide(config, layerIdx: config.numHiddenLayers - 1))
    }

    @Test func requiresMultiTokenChunk() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        #expect(decide(config, sequenceLength: 1) == false)
        #expect(decide(config, sequenceLength: 2))
    }

    @Test func requiresNonEmptyBatch() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        #expect(decide(config, batchSize: 0) == false)
        #expect(decide(config, batchSize: 1))
        #expect(decide(config, batchSize: 4))
    }

    @Test func slidingFinalLayerConfigIsRejected() throws {
        let config = try TinyGemma.slidingFinalConfig()
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        #expect(decide(config) == false)
    }

    @Test func sharedFinalLayerConfigIsRejected() throws {
        let config = try TinyGemma.sharedFinalConfig()
        #expect(config.numKvSharedLayers > 0)
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        #expect(decide(config) == false)
    }

    /// The geometry predicate also rejects a config whose declared
    /// `layer_types` disagrees with `num_hidden_layers`.
    @Test func layerTypeCountMustMatchLayerCount() throws {
        var config = try TinyGemma.lastQueryCapableConfig()
        #expect(gemma4SupportsLastQueryPrefill(config))
        config.layerTypes = Array(config.layerTypes.dropLast())
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        #expect(decide(config) == false)
    }
}

// MARK: - 4b. The SHIPPING model selects this path (anti-dead-code pin)

/// Last-query prefill was proposed for deletion as "dead on gemma-4-26B"
/// on the theory that `num_kv_shared_layers` defaults to 20, so the final
/// layer is always KV-shared and `gemma4SupportsLastQueryPrefill` is always
/// false. That default only applies when the key is ABSENT. Every shipping
/// gemma-4-26B-A4B checkpoint states `"num_kv_shared_layers": 0`
/// explicitly, so the final layer owns its K/V and the specialization is
/// SELECTED on every production prompt chunk. A Swift struct default is
/// evidence about absent keys only, never about a shipping checkpoint.
///
/// This suite exists because the rest of the file could not refute that
/// claim. `sharedFinalLayerConfigIsRejected` above looks like it pins the
/// production model and does not: `TinyGemma.sharedFinalConfig()`
/// DELIBERATELY passes `numKvSharedLayers: 2` to construct the rejected
/// geometry, so it pins the NEGATIVE case only. Nothing here asserted
/// anything about the shape actually shipped — which is exactly what made
/// the wrong conclusion look tested. Do not delete this suite to make a
/// removal easier; it is the removal's counter-evidence.
///
/// The fixture below is the verbatim `text_config` shape of
/// `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (30 layers, 5-periodic
/// sliding/full pattern ending in `full_attention`). It is a literal, not
/// a checkpoint read, so it needs no download and cannot go vacuous.
@Suite("CBv2LastQueryPrefill shipping-model shape")
struct CBv2LastQueryPrefillProductionShapeTests {

    /// gemma-4-26B-A4B-it-qat-4bit `text_config`, scalars verbatim.
    static func shippingConfig() throws -> Gemma4TextConfiguration {
        let pattern = (0 ..< 5).flatMap { _ in
            Array(repeating: "sliding_attention", count: 5) + ["full_attention"]
        }
        let types = pattern.map { "\"\($0)\"" }.joined(separator: ", ")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "intermediate_size": 2112,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "layer_types": [\(types)],
                "sliding_window": 1024,
                "attention_k_eq_v": true,
                "enable_moe_block": true,
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "use_bidirectional_attention": "vision",
                "tie_word_embeddings": true,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "rms_norm_eps": 1e-6
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// The premise of the "dead code" claim, refuted at its root: the
    /// shipping checkpoint states 0 shared layers, so the decoder default
    /// of 20 never applies and the FINAL layer owns its K/V.
    @Test func shippingConfigHasNoSharedFinalLayer() throws {
        let config = try Self.shippingConfig()
        #expect(config.numHiddenLayers == 30)
        #expect(config.layerTypes.count == 30)
        #expect(config.layerTypes.last == "full_attention")
        #expect(config.numKvSharedLayers == 0)
        #expect(config.layerUsesSharedKV(layerIdx: config.numHiddenLayers - 1) == false)
    }

    @Test func shippingConfigSupportsLastQueryPrefill() throws {
        #expect(gemma4SupportsLastQueryPrefill(try Self.shippingConfig()))
    }

    /// The real `CBv2LayerCache` is what the engine binds, and it conforms
    /// — so the trunk's `hasCapableCache` probe (Gemma4Text.swift, the
    /// `fullCache[idx] is any CBv2LastQueryPrefillLayerCache` argument) is
    /// true for every layer in production.
    @Test func engineLayerCacheIsCapable() throws {
        let config = try Self.shippingConfig()
        let kinds = config.cbv2LayerKinds
        let final = CBv2LayerCache(layerIndex: kinds.count - 1, kind: kinds[kinds.count - 1])
        #expect(final is any CBv2LastQueryPrefillLayerCache)
    }

    /// End of the chain: with the shipping config, a capable cache, and the
    /// trunk's own knob values, the policy seam SELECTS last-query prefill
    /// for the final layer of any prompt chunk the trunk narrows. Deleting
    /// the feature would change production behaviour.
    @Test func shippingConfigSelectsLastQueryPrefillOnFinalLayer() throws {
        let config = try Self.shippingConfig()
        let chunk = PrefillKnobs.narrowingChunk
        let selected = gemma4UseLastQueryPrefill(
            config,
            layerIdx: config.numHiddenLayers - 1,
            batchSize: 1,
            sequenceLength: chunk,
            outputTailRows: PrefillKnobs.tailRows > 0 ? min(PrefillKnobs.tailRows, chunk) : nil,
            hasCapableCache: true,
            enabled: PrefillKnobs.lastQueryEnabled)
        let expected = PrefillKnobs.lastQueryEnabled && PrefillKnobs.tailRows == 1
        #expect(
            selected == expected,
            "shipping gemma-4-26B selects last-query prefill under default knobs")
    }
}

// MARK: - 5/6. Decoder-layer parity (deterministic, env-independent)

@Suite("CBv2LastQueryPrefill decoder-layer parity")
struct CBv2LastQueryPrefillLayerParityTests {

    private struct Run {
        var output: MLXArray
        var keys: MLXArray
        var values: MLXArray
        var offset: Int
    }

    /// Run the FINAL decoder layer of a tiny config over one `[1, L, H]`
    /// chunk on a fresh contiguous CBv2 cache.
    private func runFinalLayer(
        config: Gemma4TextConfiguration,
        layer: Gemma4DecoderLayer,
        hidden: MLXArray,
        outputTailRows: Int?,
        useLastQueryPrefill: Bool
    ) throws -> Run {
        let layerIdx = config.numHiddenLayers - 1
        let kind = config.cbv2LayerKinds[layerIdx]
        let rows = makeFullRows(
            count: 1, kvHeads: kind.kvHeads, headDim: kind.headDim,
            maxLength: hidden.dim(1) + 16)
        let cache = CBv2LayerCache(layerIndex: layerIdx, kind: kind, rows: rows)
        let (out, _, _) = layer(
            hidden,
            mask: nil,
            cache: cache,
            perLayerInput: nil,
            sharedKV: nil,
            positionOffset: nil,
            v2SharedSource: nil,
            outputTailRows: outputTailRows,
            useLastQueryPrefill: useLastQueryPrefill)
        eval(out)
        eval(cache.innerState())
        let snapshot = rows[0].snapshot()
        eval(snapshot.keys, snapshot.values)
        return Run(
            output: out, keys: snapshot.keys, values: snapshot.values,
            offset: snapshot.offset)
    }

    private func fixtureLayer(_ config: Gemma4TextConfiguration, seed: UInt64) -> (
        Gemma4DecoderLayer, MLXArray
    ) {
        MLXRandom.seed(seed)
        let layer = Gemma4DecoderLayer(config, layerIdx: config.numHiddenLayers - 1)
        eval(layer)
        // Small magnitudes keep the tiny random layer numerically tame.
        let hidden = MLXRandom.normal([1, 40, config.hiddenSize]) * 0.5
        eval(hidden)
        return (layer, hidden)
    }

    /// Test 5 (layer scope): tail narrowing + last-query ENABLED vs the
    /// unnarrowed reference. The toggle is a parameter, so this holds
    /// regardless of the process's env knobs.
    @Test func lastQueryPrefillMatchesUnnarrowedFinalRow() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let (layer, hidden) = fixtureLayer(config, seed: 0xD1_0A)
        let L = hidden.dim(1)

        let reference = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: nil, useLastQueryPrefill: false)
        let specialized = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: 1, useLastQueryPrefill: true)

        #expect(reference.output.shape == [1, L, config.hiddenSize])
        #expect(specialized.output.shape == [1, 1, config.hiddenSize])
        #expect(
            maxAbsDiff(
                specialized.output, reference.output[0..., (L - 1) ..< L, 0...]) < 1e-5,
            "last-query prefill must reproduce the unnarrowed frontier row")

        #expect(specialized.offset == L, "the chunk consumed every position")
        #expect(specialized.offset == reference.offset)
        #expect(bitIdentical(specialized.keys, reference.keys), "final-layer K must be identical")
        #expect(
            bitIdentical(specialized.values, reference.values),
            "final-layer V must be identical")
    }

    /// Test 6 (layer scope): tail narrowing ALONE (last-query off).
    @Test func tailNarrowingAloneMatchesUnnarrowedFinalRow() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let (layer, hidden) = fixtureLayer(config, seed: 0xD2_0A)
        let L = hidden.dim(1)

        let reference = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: nil, useLastQueryPrefill: false)
        let narrowed = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: 1, useLastQueryPrefill: false)

        #expect(narrowed.output.shape == [1, 1, config.hiddenSize])
        #expect(
            maxAbsDiff(narrowed.output, reference.output[0..., (L - 1) ..< L, 0...]) < 1e-5,
            "tail narrowing must not change the retained row")
        #expect(narrowed.offset == reference.offset)
        #expect(bitIdentical(narrowed.keys, reference.keys))
        #expect(bitIdentical(narrowed.values, reference.values))
    }

    /// A wider tail narrows without changing any retained row either.
    @Test func multiRowTailNarrowingMatchesUnnarrowedRows() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let (layer, hidden) = fixtureLayer(config, seed: 0xD3_0A)
        let L = hidden.dim(1)
        let tail = 4

        let reference = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: nil, useLastQueryPrefill: false)
        let narrowed = try runFinalLayer(
            config: config, layer: layer, hidden: hidden,
            outputTailRows: tail, useLastQueryPrefill: false)

        #expect(narrowed.output.shape == [1, tail, config.hiddenSize])
        #expect(
            maxAbsDiff(narrowed.output, reference.output[0..., (L - tail) ..< L, 0...]) < 1e-5)
        #expect(bitIdentical(narrowed.keys, reference.keys))
        #expect(bitIdentical(narrowed.values, reference.values))
    }

    /// Tail narrowing on a SLIDING final layer (last-query is not legal
    /// there) still preserves the retained row and the windowed ring.
    @Test func tailNarrowingOnSlidingFinalLayer() throws {
        let config = try TinyGemma.slidingFinalConfig()
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        MLXRandom.seed(0xD4_0A)
        let layerIdx = config.numHiddenLayers - 1
        let kind = config.cbv2LayerKinds[layerIdx]
        guard case .slidingWindow(let window) = kind.attention else {
            Issue.record("expected a sliding final layer")
            return
        }
        let layer = Gemma4DecoderLayer(config, layerIdx: layerIdx)
        eval(layer)
        let hidden = MLXRandom.normal([1, 40, config.hiddenSize]) * 0.5
        eval(hidden)
        let L = hidden.dim(1)

        func run(_ tail: Int?) -> (MLXArray, MLXArray, MLXArray, Int) {
            let row = CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
            let cache = CBv2LayerCache(layerIndex: layerIdx, kind: kind, rows: [row])
            let (out, _, _) = layer(
                hidden, mask: nil, cache: cache, perLayerInput: nil, sharedKV: nil,
                positionOffset: nil, v2SharedSource: nil,
                outputTailRows: tail, useLastQueryPrefill: false)
            eval(out)
            eval(cache.innerState())
            let snap = row.snapshot()
            eval(snap.keys, snap.values)
            return (out, snap.keys, snap.values, snap.offset)
        }

        let (fullOut, fullK, fullV, fullOffset) = run(nil)
        let (narrowOut, narrowK, narrowV, narrowOffset) = run(1)

        #expect(narrowOut.shape == [1, 1, config.hiddenSize])
        #expect(maxAbsDiff(narrowOut, fullOut[0..., (L - 1) ..< L, 0...]) < 1e-5)
        #expect(narrowOffset == fullOffset)
        #expect(narrowOffset == L)
        #expect(bitIdentical(narrowK, fullK))
        #expect(bitIdentical(narrowV, fullV))
    }
}

// MARK: - 5/6. Model-level parity through the CBv2 prompt trunk

@Suite("CBv2LastQueryPrefill model parity")
struct CBv2LastQueryPrefillModelParityTests {

    private struct PrefillResult {
        /// `[1, vocab]` logits for the prompt's frontier position.
        var frontierLogits: MLXArray
        /// Per-layer (keys, values, absoluteOffset) for storage owners.
        var kv: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    }

    /// Run `chunks` through the CBv2 PROMPT path (`cbv2Prefill`, which is
    /// where the final-layer narrowing + last-query specialization live).
    private func runCBv2Prefill(
        model: Gemma4TextModel, config: Gemma4TextConfiguration, chunks: [MLXArray],
        totalTokens: Int
    ) throws -> PrefillResult {
        let (_, caches) = try TinyGemma.caches(
            for: config, promptLength: totalTokens, maxLength: totalTokens + 16)
        let kvCaches: [KVCache] = caches
        var frontier: MLXArray?
        for (index, chunk) in chunks.enumerated() {
            let isLast = index == chunks.count - 1
            let out = model.cbv2Prefill(
                chunk, inputEmbedding: nil, cache: kvCaches,
                requirement: isLast ? .lastPositionLogits : .evaluationOnly)
            eval(out)
            eval(caches.flatMap { $0.innerState() })
            if isLast { frontier = out }
        }
        return PrefillResult(
            frontierLogits: frontier!, kv: snapshotKV(caches))
    }

    /// Reference: the ordinary (non-scheduled) forward — no narrowing, no
    /// last-query specialization — over the same chunks and cache shape.
    private func runReferenceForward(
        model: Gemma4TextModel, config: Gemma4TextConfiguration, chunks: [MLXArray],
        totalTokens: Int
    ) throws -> PrefillResult {
        let (_, caches) = try TinyGemma.caches(
            for: config, promptLength: totalTokens, maxLength: totalTokens + 16)
        let kvCaches: [KVCache] = caches
        var frontier: MLXArray?
        for (index, chunk) in chunks.enumerated() {
            let logits = model(chunk, cache: kvCaches)
            eval(logits)
            eval(caches.flatMap { $0.innerState() })
            if index == chunks.count - 1 {
                frontier = logits[0..., (chunk.dim(1) - 1) ..< chunk.dim(1), 0...]
                    .reshaped(1, -1)
                eval(frontier!)
            }
        }
        return PrefillResult(
            frontierLogits: frontier!, kv: snapshotKV(caches))
    }

    private func snapshotKV(_ caches: [CBv2LayerCache])
        -> [(keys: MLXArray, values: MLXArray, offset: Int)?]
    {
        caches.map { cache in
            guard let row = cache.rows.first else { return nil }
            let snap = row.snapshot()
            eval(snap.keys, snap.values)
            return (snap.keys, snap.values, snap.offset)
        }
    }

    private func greedyToken(_ logits: MLXArray) -> Int {
        let arg = argMax(logits.reshaped(-1), axis: 0)
        eval(arg)
        return arg.item(Int.self)
    }

    private func assertParity(
        config: Gemma4TextConfiguration, seed: UInt64, chunkSizes: [Int],
        _ label: Comment
    ) throws {
        MLXRandom.seed(seed)
        let model = Gemma4TextModel(config)
        eval(model)

        let total = chunkSizes.reduce(0, +)
        var tokens: [Int32] = []
        tokens.reserveCapacity(total)
        for i in 0 ..< total {
            tokens.append(Int32((i * 7 + 3) % config.vocabSize))
        }
        var chunks: [MLXArray] = []
        var cursor = 0
        for size in chunkSizes {
            chunks.append(
                MLXArray(Array(tokens[cursor ..< cursor + size]))[.newAxis, .ellipsis])
            cursor += size
        }

        let specialized = try runCBv2Prefill(
            model: model, config: config, chunks: chunks, totalTokens: total)
        let reference = try runReferenceForward(
            model: model, config: config, chunks: chunks, totalTokens: total)

        #expect(
            specialized.frontierLogits.shape == reference.frontierLogits.shape,
            "frontier logits shape — \(label)")
        let logitDiff = maxAbsDiff(specialized.frontierLogits, reference.frontierLogits)
        #expect(logitDiff < 1e-5, "frontier logits diff \(logitDiff) — \(label)")
        #expect(
            greedyToken(specialized.frontierLogits) == greedyToken(reference.frontierLogits),
            "greedy first token — \(label)")

        #expect(specialized.kv.count == reference.kv.count)
        for index in 0 ..< specialized.kv.count {
            guard let a = specialized.kv[index], let b = reference.kv[index] else {
                #expect(specialized.kv[index] == nil && reference.kv[index] == nil)
                continue
            }
            #expect(a.offset == b.offset, "layer \(index) offset — \(label)")
            #expect(a.offset == total, "layer \(index) must hold every prompt token — \(label)")
            #expect(bitIdentical(a.keys, b.keys), "layer \(index) K — \(label)")
            #expect(bitIdentical(a.values, b.values), "layer \(index) V — \(label)")
        }
    }

    /// Test 5: a config whose final layer IS eligible, so the CBv2 prompt
    /// trunk takes tail narrowing + last-query prefill.
    @Test func multiChunkPromptMatchesFullForwardWithLastQueryEligibleConfig() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let chunk = PrefillKnobs.narrowingChunk
        #expect(gemma4SupportsLastQueryPrefill(config))
        #expect(
            gemma4UseLastQueryPrefill(
                config, layerIdx: config.numHiddenLayers - 1, batchSize: 1,
                sequenceLength: chunk, outputTailRows: 1, hasCapableCache: true,
                enabled: true),
            "the fixture must actually select the specialization")
        try assertParity(
            config: config, seed: 0xE1_0F, chunkSizes: [chunk, chunk],
            "last-query-eligible final layer")
    }

    /// Test 6: a config whose final layer is SLIDING — last-query prefill is
    /// off by construction, so the trunk exercises tail narrowing alone.
    @Test func multiChunkPromptMatchesFullForwardWithTailNarrowingOnly() throws {
        let config = try TinyGemma.slidingFinalConfig()
        let chunk = PrefillKnobs.narrowingChunk
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        try assertParity(
            config: config, seed: 0xE2_0F, chunkSizes: [chunk, chunk],
            "tail narrowing only (sliding final layer)")
    }

    /// A KV-shared trailing block: the final layer borrows, so neither the
    /// specialization nor a storage commit may happen there — parity must
    /// still hold end to end.
    @Test func multiChunkPromptMatchesFullForwardWithSharedFinalLayer() throws {
        let config = try TinyGemma.sharedFinalConfig()
        let chunk = PrefillKnobs.narrowingChunk
        #expect(gemma4SupportsLastQueryPrefill(config) == false)
        try assertParity(
            config: config, seed: 0xE3_0F, chunkSizes: [chunk, chunk],
            "KV-shared final layer")
    }
}

// MARK: - Trunk dispatch (proves the parity suites are not vacuous)

@Suite("CBv2LastQueryPrefill trunk dispatch")
struct CBv2LastQueryPrefillDispatchTests {

    /// Build the model's caches with the FINAL layer wrapped in a spy.
    private func spiedCaches(
        for config: Gemma4TextConfiguration, promptLength: Int
    ) throws -> (all: [KVCache], spy: SpyLayerCache) {
        let (_, caches) = try TinyGemma.caches(
            for: config, promptLength: promptLength, maxLength: promptLength + 16)
        let last = caches.count - 1
        let spy = SpyLayerCache(caches[last])
        var all: [KVCache] = caches
        all[last] = spy
        return (all, spy)
    }

    private func prefill(
        _ config: Gemma4TextConfiguration, tokens: Int, seed: UInt64
    ) throws -> SpyLayerCache {
        MLXRandom.seed(seed)
        let model = Gemma4TextModel(config)
        eval(model)
        let (caches, spy) = try spiedCaches(for: config, promptLength: tokens)
        let ids = (0 ..< tokens).map { Int32(($0 * 5 + 1) % config.vocabSize) }
        let out = model.cbv2Prefill(
            MLXArray(ids)[.newAxis, .ellipsis], inputEmbedding: nil,
            cache: caches, requirement: .lastPositionLogits)
        eval(out)
        return spy
    }

    /// A long enough chunk on an eligible config MUST reach
    /// `updateAndAttendLastQuery` with a one-row Q and the full K rectangle.
    @Test func longPromptChunkDispatchesLastQueryPrefill() throws {
        guard PrefillKnobs.tailRows == 1, PrefillKnobs.lastQueryEnabled else {
            // The env kill switches are off in this process; the parity
            // suites still hold, but this dispatch claim does not apply.
            return
        }
        let config = try TinyGemma.lastQueryCapableConfig()
        let tokens = PrefillKnobs.narrowingChunk
        let spy = try prefill(config, tokens: tokens, seed: 0xF1_0A)
        #expect(spy.lastQueryCalls == 1, "the final layer must take the specialization")
        #expect(spy.updateAndAttendCalls == 0, "and must not also run chunk attention")
        #expect(spy.lastQueryShapes.first?.q[2] == 1, "Q must be exactly one row")
        #expect(spy.lastQueryShapes.first?.k[2] == tokens, "K must be the whole chunk")
        #expect(spy.rows.first?.absoluteOffset == tokens, "offsets advance by the K/V length")
    }

    /// A chunk below the tail threshold keeps the unnarrowed final layer.
    @Test func shortPromptChunkKeepsOrdinaryChunkAttention() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let tokens = PrefillKnobs.unnarrowedChunk
        let spy = try prefill(config, tokens: tokens, seed: 0xF2_0A)
        #expect(spy.lastQueryCalls == 0, "short chunks must not specialize")
        #expect(spy.updateAndAttendCalls == 1)
        #expect(spy.rows.first?.absoluteOffset == tokens)
    }

    /// Ordinary (non-prompt) forwards — decode, MTP verify, legacy calls —
    /// never reach the specialization even at prompt length.
    @Test func nonPromptForwardNeverDispatchesLastQueryPrefill() throws {
        let config = try TinyGemma.lastQueryCapableConfig()
        let tokens = PrefillKnobs.narrowingChunk
        MLXRandom.seed(0xF3_0A)
        let model = Gemma4TextModel(config)
        eval(model)
        let (caches, spy) = try spiedCaches(for: config, promptLength: tokens)
        let ids = (0 ..< tokens).map { Int32(($0 * 5 + 1) % config.vocabSize) }
        let logits = model(MLXArray(ids)[.newAxis, .ellipsis], cache: caches)
        eval(logits)
        #expect(logits.shape == [1, tokens, config.vocabSize])
        #expect(spy.lastQueryCalls == 0, "only the CBv2 prompt trunk may specialize")
        #expect(spy.updateAndAttendCalls == 1)
    }

    /// A sliding final layer never specializes, however long the chunk.
    @Test func slidingFinalLayerNeverDispatchesLastQueryPrefill() throws {
        let config = try TinyGemma.slidingFinalConfig()
        let tokens = PrefillKnobs.narrowingChunk
        let spy = try prefill(config, tokens: tokens, seed: 0xF4_0A)
        #expect(spy.lastQueryCalls == 0)
        #expect(spy.updateAndAttendCalls == 1)
    }
}
