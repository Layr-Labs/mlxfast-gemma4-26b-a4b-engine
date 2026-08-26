// CBv2QueryBlockAttentionTests.swift
//
// Query-block sub-blocking of multi-token prompt attention
// (`CBv2AttentionV1.attendQueryBlocks`, AttentionV1.swift).
//
// The change splits a `[1, L]` prefill chunk into `queryBlockSize`-wide query
// blocks and slices K/V to each block's own visible span, instead of computing
// the whole `[L, kL]` score rectangle and masking the excess away. Every
// failure mode of that transformation is SILENT:
//
//   * `visibleStart` too late  -> tokens lose part of their sliding window
//     (quality loss, no crash, no shape error);
//   * `visibleEnd` too late    -> FUTURE LEAKAGE on full-attention layers
//     (catastrophic, and invisible to any test that only compares against a
//     reference that shares the same off-by-one);
//   * wrong `historyCount` on the borrow path -> KV-shared layers silently
//     desync from their source layer.
//
// So this suite deliberately does NOT rest on a single oracle:
//
//  1. `CBv2QueryBlockAttentionParityTests` — the production entry point
//     (`updateAndAttend`, which blocks whenever `shouldBlockQueries(L)`)
//     against TWO independent references: (a) the unblocked single-call path
//     rebuilt from the production primitives it actually calls
//     (`MLXFast.scaledDotProductAttention` + `CBv2AttentionV1.maskMode` over
//     the FULL rectangle), and (b) a from-scratch fp32 attention driven by
//     ABSOLUTE token positions, which shares no code with the implementation.
//     Output TENSORS are compared, never sampled tokens. Max abs diffs are
//     printed for every case.
//  2. `CBv2QueryBlockAttentionSerialTests` — `blockSize == 1` is BIT-equal to
//     the pre-change serial-query path (reconstructed verbatim here from the
//     code it replaced), pinning the "thin wrapper" refactor claim.
//  3. `CBv2QueryBlockAttentionCausalityTests` — strict causality by
//     PERTURBATION: change K/V at position p+1, and every query <= p must be
//     bit-identical. An off-by-one on `visibleEnd` cannot survive this, and it
//     needs no reference implementation at all.
//  4. `CBv2QueryBlockAttentionWindowTests` — the sliding-window boundary
//     pinned from BOTH sides: perturbing `p - window` must not move query p,
//     perturbing `p - window + 1` must.
//  5. `CBv2QueryBlockAttentionKVStateTests` — blocking changes only the
//     attention computation: committed offset, retained length and retained
//     K/V bytes are identical to a row that only ran `update()`.
//  6. `CBv2QueryBlockAttentionBorrowTests` — the same parity + causality for
//     `attendBorrowing` (Gemma-4 cross-layer KV sharing), including
//     pre-existing history, where `historyCount` is derived from the SOURCE
//     row's pre-eviction borrow views.
//  7. `CBv2QueryBlockAttentionGatingTests` — `shouldBlockQueries` truth table
//     against the live `queryBlockSize` (including the `0` kill switch), plus
//     whole-mask parity for q-blocked vision spans and mixed packed
//     vision/text rows with independent per-row contexts.
//
// Tiny synthetic tensors, seeded RNG, no model download, no checkpoint.
//
// NOTE on reachable block sizes: `attendQueryBlocks` is `private` to
// `CBv2AttentionV1`, so a test module cannot call it with an arbitrary
// `blockSize`. The two block widths reachable through the real entry points
// are exercised here: `1` (via `serializeQueries: true`) and the live
// `CBv2AttentionV1.queryBlockSize` (128 by default, or whatever
// `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK` selects — run the suite twice to cover
// another width, e.g. `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK=32 swift test`). Every
// test below adapts to the live value rather than assuming 128.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

// MARK: - Geometry

private let kvHeads = 2
private let queryHeads = 4
private let headDim = 64
private let attnScale: Float = 1.0 / Float(headDim).squareRoot()

/// Live block width. Read once so every expectation below is stated against
/// the value the implementation will actually use in this process.
private let liveBlock = CBv2AttentionV1.queryBlockSize

private func layerKind(window: Int?, sharesKVWithLayer: Int? = nil) -> CBv2LayerKind {
    let attention: CBv2LayerKind.Attention
    if let window { attention = .slidingWindow(window) } else { attention = .full }
    return CBv2LayerKind(
        attention: attention, sharesKVWithLayer: sharesKVWithLayer, hasSinks: false,
        headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
}

private func makeRow(window: Int?, maxLength: Int) -> CBv2SequenceKV {
    if let window {
        return CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
    }
    return CBv2FullSequenceKV(
        promptLength: maxLength, maxLength: maxLength, kvHeads: kvHeads, headDim: headDim)
}

/// The views a chunk BORROW must attend — the source row's step-scoped
/// pre-eviction views for a windowed source, its snapshot otherwise. Mirrors
/// the selection rule in `CBv2AttentionV1.chunkBorrowViews` (which is private)
/// using only public row API.
private func borrowViews(of row: CBv2SequenceKV) -> (MLXArray, MLXArray) {
    if let windowed = row as? CBv2WindowedSequenceKV { return windowed.borrowableViews() }
    let snapshot = row.snapshot()
    return (snapshot.keys, snapshot.values)
}

// MARK: - Comparison helpers

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
    if a.size == 0 { return 0 }
    let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
    eval(diff)
    return diff.item(Float.self)
}

private func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape else { return false }
    if a.size == 0 { return true }
    let same = MLX.all(a .== b)
    eval(same)
    return same.item(Bool.self)
}

/// Rows `[..., lo ..< hi, ...]` of a `[B, heads, L, headDim]` attention output.
private func queryRows(_ out: MLXArray, _ lo: Int, _ hi: Int) -> MLXArray {
    out[0..., 0..., lo ..< hi, 0...]
}

// MARK: - References

/// The UNBLOCKED single-call path, rebuilt from the two production primitives
/// `CBv2AttentionV1.attend` calls when `softcap == nil`: one SDPA over the
/// whole `[L, kL]` rectangle with `maskMode(L:kL:window:)`. This is exactly
/// what the code did before query blocking existed.
private func singleCallReference(
    queries: MLXArray, keys: MLXArray, values: MLXArray, window: Int?
) -> MLXArray {
    let L = queries.dim(2)
    let kL = keys.dim(2)
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: attnScale,
        mask: CBv2AttentionV1.maskMode(L: L, kL: kL, window: window), sinks: nil)
}

/// From-scratch fp32 attention over ABSOLUTE token positions. Shares no code
/// with `AttentionV1` — the mask is stated directly from the definition
/// ("query at absolute position q attends key at absolute position k iff
/// k <= q and, when windowed, k > q - window"), so a relative-coordinate
/// off-by-one in the implementation cannot hide inside it.
private func absolutePositionReference(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    queryStart: Int, keyStart: Int, window: Int?
) -> MLXArray {
    let L = queries.dim(2)
    let kL = keys.dim(2)
    let groups = queries.dim(1) / keys.dim(1)

    let qPos = MLXArray(Int32(queryStart) ..< Int32(queryStart + L)).reshaped([L, 1])
    let kPos = MLXArray(Int32(keyStart) ..< Int32(keyStart + kL)).reshaped([1, kL])
    var allow = kPos .<= qPos
    if let window { allow = allow .&& (kPos .> (qPos - Int32(window))) }

    let q = queries.asType(.float32) * attnScale
    var k = keys.asType(.float32)
    var v = values.asType(.float32)
    if groups > 1 {
        k = repeated(k, count: groups, axis: 1)
        v = repeated(v, count: groups, axis: 1)
    }
    var scores = matmul(q, k.transposed(0, 1, 3, 2))
    scores = which(allow.reshaped([1, 1, L, kL]), scores, MLXArray(Float(-1e30)))
    return matmul(softmax(scores, axis: -1, precise: true), v).asType(queries.dtype)
}

/// The PRE-CHANGE serial-query path, transcribed verbatim from the code
/// `attendSerialQueries` used before it was refactored into
/// `attendQueryBlocks(blockSize: 1)`:
///
/// ```
/// for column in 0 ..< newTokenCount {
///     let visibleEnd = historyCount + column + 1
///     let visibleStart = window.map { max(0, visibleEnd - $0) } ?? 0
///     attend(q[column..<column+1], k[visibleStart..<visibleEnd], ...,
///            L: 1, kL: visibleEnd - visibleStart, window: nil)
/// }
/// concatenated(outputs, axis: 2)
/// ```
private func legacySerialReference(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    newTokenCount: Int, window: Int?
) -> MLXArray {
    let historyCount = keys.dim(2) - newTokenCount
    precondition(historyCount >= 0)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(newTokenCount)
    for column in 0 ..< newTokenCount {
        let visibleEnd = historyCount + column + 1
        let visibleStart = window.map { max(0, visibleEnd - $0) } ?? 0
        outputs.append(
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., column ..< (column + 1), 0...],
                keys: keys[0..., 0..., visibleStart ..< visibleEnd, 0...],
                values: values[0..., 0..., visibleStart ..< visibleEnd, 0...],
                scale: attnScale,
                mask: CBv2AttentionV1.maskMode(
                    L: 1, kL: visibleEnd - visibleStart, window: nil),
                sinks: nil))
    }
    return concatenated(outputs, axis: 2)
}

// MARK: - Fixture

/// One chunk of `n` new tokens landing on `history` pre-existing tokens.
private struct ChunkFixture {
    let n: Int
    let history: Int
    let window: Int?
    let queries: MLXArray
    let keys: MLXArray
    let values: MLXArray
    let historyKeys: MLXArray?
    let historyValues: MLXArray?

    init(n: Int, history: Int, window: Int?, seed: UInt64) {
        MLXRandom.seed(seed)
        self.n = n
        self.history = history
        self.window = window
        self.queries = MLXRandom.normal([1, queryHeads, n, headDim])
        self.keys = MLXRandom.normal([1, kvHeads, n, headDim])
        self.values = MLXRandom.normal([1, kvHeads, n, headDim])
        if history > 0 {
            self.historyKeys = MLXRandom.normal([1, kvHeads, history, headDim])
            self.historyValues = MLXRandom.normal([1, kvHeads, history, headDim])
        } else {
            self.historyKeys = nil
            self.historyValues = nil
        }
        eval(queries, keys, values)
        if let historyKeys, let historyValues { eval(historyKeys, historyValues) }
    }

    var maxLength: Int { history + n + 8 }

    /// A fresh row pre-loaded with `history` tokens through the SAME `update`
    /// path every side uses, so all runs start from byte-identical state.
    func freshRow() -> CBv2SequenceKV {
        let row = makeRow(window: window, maxLength: maxLength)
        if let historyKeys, let historyValues {
            _ = row.update(keys: historyKeys, values: historyValues)
        }
        return row
    }

    /// A fresh row with the chunk already committed, plus the views that
    /// commit returned (what the attention path attends).
    func committedRow() -> (row: CBv2SequenceKV, keys: MLXArray, values: MLXArray, offsetBefore: Int)
    {
        let row = freshRow()
        let offsetBefore = row.absoluteOffset
        let (k, v) = row.update(keys: keys, values: values)
        return (row, k, v, offsetBefore)
    }

    /// Copy of the chunk K/V with position `index` shifted by `delta`.
    func perturbed(keysAt index: Int, delta: Float) -> MLXArray {
        var k = keys
        k[0..., 0..., index ..< (index + 1), 0...] =
            keys[0..., 0..., index ..< (index + 1), 0...] + delta
        eval(k)
        return k
    }

    func perturbed(valuesAt index: Int, delta: Float) -> MLXArray {
        var v = values
        v[0..., 0..., index ..< (index + 1), 0...] =
            values[0..., 0..., index ..< (index + 1), 0...] + delta
        eval(v)
        return v
    }
}

private func windowLabel(_ window: Int?) -> String {
    window.map(String.init) ?? "nil"
}

private func diffLabel(_ value: Float?) -> String {
    guard let value else { return "skipped" }
    return String(value)
}

/// `#expect` takes a `Comment`, which is only expressible from a string
/// LITERAL — computed messages have to go through `rawValue`.
private func note(_ message: String) -> Comment {
    Comment(rawValue: message)
}

// MARK: - 1. Numeric parity with the unblocked single-call path

@Suite("CBv2QueryBlockAttention parity")
struct CBv2QueryBlockAttentionParityTests {

    /// `L * kL` above which the from-scratch fp32 oracle (which materializes
    /// the whole score rectangle) is skipped; the single-call oracle still
    /// runs for those cases.
    private static let naiveBudget = 1_500_000

    private struct Result {
        var vsSingleCall: Float
        var vsAbsolute: Float?
        var blocked: Bool
        var kL: Int
    }

    private func runCase(
        n: Int, history: Int, window: Int?, serialize: Bool, seed: UInt64
    ) -> Result {
        let fixture = ChunkFixture(n: n, history: history, window: window, seed: seed)
        let kind = layerKind(window: window)

        // Production path: blocks whenever shouldBlockQueries(L) (or always,
        // at blockSize 1, when serializeQueries is set).
        let productionRow = fixture.freshRow()
        let produced = CBv2AttentionV1.updateAndAttend(
            rows: [productionRow], kind: kind,
            queries: fixture.queries, keys: fixture.keys, values: fixture.values,
            scale: attnScale, sinks: nil, softcap: nil, spanContexts: nil,
            serializeQueries: serialize)

        // Oracle A: one SDPA over the full rectangle on an identical row.
        let reference = fixture.committedRow()
        let single = singleCallReference(
            queries: fixture.queries, keys: reference.keys, values: reference.values,
            window: window)

        eval(produced, single)
        #expect(
            produced.shape == [1, queryHeads, n, headDim],
            "blocked attention must preserve the output shape")

        let kL = reference.keys.dim(2)
        var result = Result(
            vsSingleCall: maxAbsDiff(produced, single),
            vsAbsolute: nil,
            blocked: serialize || CBv2AttentionV1.shouldBlockQueries(n),
            kL: kL)

        // Oracle B: absolute-position fp32 attention, no shared code.
        if n * kL <= Self.naiveBudget {
            let absolute = absolutePositionReference(
                queries: fixture.queries, keys: reference.keys, values: reference.values,
                queryStart: reference.offsetBefore,
                keyStart: reference.offsetBefore - (kL - n),
                window: window)
            eval(absolute)
            result.vsAbsolute = maxAbsDiff(produced, absolute)
        }
        return result
    }

    /// The load-bearing parity sweep. Note `n = 17` and `n = 129` produce a
    /// ragged final block at the default width, `n = 129` is one token past
    /// the gate, and `history` covers below-window, exactly-window-minus-one,
    /// exactly-window and far-past-window.
    @Test func blockedChunkAttentionMatchesTheUnblockedPath() {
        let windows: [Int?] = [1024, nil]
        let lengths = [17, 64, 100, 129, 512, 2048]
        let histories = [0, 500, 1023, 1024, 5000]

        var worstSingle: Float = 0
        var worstAbsolute: Float = 0
        var worstSingleLabel = "-"
        var worstAbsoluteLabel = "-"
        var seed: UInt64 = 0xB10C_0001

        print("[cbv2-query-block] queryBlockSize = \(liveBlock)")
        for window in windows {
            for n in lengths {
                for history in histories {
                    seed &+= 1
                    let r = runCase(
                        n: n, history: history, window: window, serialize: false, seed: seed)
                    let label =
                        "window=\(windowLabel(window)) n=\(n) history=\(history) kL=\(r.kL)"
                    let dAbs = diffLabel(r.vsAbsolute)
                    print(
                        "[cbv2-query-block] \(label) blocked=\(r.blocked) dVsSingleCall=\(r.vsSingleCall) dVsAbsolute=\(dAbs)"
                    )

                    if r.vsSingleCall > worstSingle {
                        worstSingle = r.vsSingleCall
                        worstSingleLabel = label
                    }
                    if let d = r.vsAbsolute, d > worstAbsolute {
                        worstAbsolute = d
                        worstAbsoluteLabel = label
                    }

                    if r.blocked {
                        // Last-ulp tolerance: the same non-zero terms are
                        // summed, but a different kL retiles the reduction.
                        #expect(
                            r.vsSingleCall < 2e-5,
                            "blocked output must match the unblocked path — \(label)")
                    } else {
                        // Not blocked: the gate says this chunk takes the
                        // pre-change path, so it must be BIT-identical.
                        #expect(
                            r.vsSingleCall == 0,
                            "unblocked chunks must be bit-identical to the pre-change path — \(label)"
                        )
                    }
                    if let d = r.vsAbsolute {
                        #expect(
                            d < 5e-5,
                            "blocked output must match absolute-position attention — \(label)")
                    }
                }
            }
        }
        print("[cbv2-query-block] WORST vs single-call = \(worstSingle) (\(worstSingleLabel))")
        print(
            "[cbv2-query-block] WORST vs absolute-position fp32 = \(worstAbsolute) (\(worstAbsoluteLabel))"
        )
    }

    /// Same sweep at block width 1 (`serializeQueries`), which is the extreme
    /// end of the blocking transformation: every query gets its own K/V slice.
    @Test func blockWidthOneMatchesTheUnblockedPath() {
        let windows: [Int?] = [1024, 64, nil]
        let lengths = [17, 129, 300]
        let histories = [0, 1024]

        var worstSingle: Float = 0
        var worstAbsolute: Float = 0
        var seed: UInt64 = 0xB10C_1001

        for window in windows {
            for n in lengths {
                for history in histories {
                    seed &+= 1
                    let r = runCase(
                        n: n, history: history, window: window, serialize: true, seed: seed)
                    let label = "window=\(windowLabel(window)) n=\(n) history=\(history)"
                    let dAbs = diffLabel(r.vsAbsolute)
                    print(
                        "[cbv2-query-block b=1] \(label) kL=\(r.kL) dVsSingleCall=\(r.vsSingleCall) dVsAbsolute=\(dAbs)"
                    )
                    worstSingle = max(worstSingle, r.vsSingleCall)
                    if let d = r.vsAbsolute { worstAbsolute = max(worstAbsolute, d) }
                    #expect(
                        r.vsSingleCall < 2e-5,
                        "blockSize 1 must match the unblocked path — \(label)")
                    if let d = r.vsAbsolute {
                        #expect(
                            d < 5e-5,
                            "blockSize 1 must match absolute-position attention — \(label)")
                    }
                }
            }
        }
        print("[cbv2-query-block b=1] WORST vs single-call = \(worstSingle)")
        print("[cbv2-query-block b=1] WORST vs absolute-position fp32 = \(worstAbsolute)")
    }

    /// Packed prefill (`EngineLoopV2.executeMixed`) sends a RECTANGULAR
    /// `[B > 1, L > 1]` chunk through the same `updateAndAttendRow` loop, so
    /// those rows are blocked too. Blocking is per row and must stay so: each
    /// row's output has to be BIT-identical to running that row alone, even
    /// though the rows sit at different absolute offsets (which is exactly
    /// what `historyCount` is derived from, per row).
    @Test func packedRectangularRowsAreIndependentUnderBlocking() {
        let n = max(300, liveBlock * 2 + 40)
        let histories = [0, 500, 1024]
        let batch = histories.count

        for window in [nil, 64, 1024] as [Int?] {
            MLXRandom.seed(0xB10C_1501)
            let kind = layerKind(window: window)
            let queries = MLXRandom.normal([batch, queryHeads, n, headDim])
            let keys = MLXRandom.normal([batch, kvHeads, n, headDim])
            let values = MLXRandom.normal([batch, kvHeads, n, headDim])
            let history = histories.map { count -> (MLXArray, MLXArray)? in
                guard count > 0 else { return nil }
                return (
                    MLXRandom.normal([1, kvHeads, count, headDim]),
                    MLXRandom.normal([1, kvHeads, count, headDim])
                )
            }
            eval(queries, keys, values)
            for entry in history { if let entry { eval(entry.0, entry.1) } }

            func freshRow(_ index: Int) -> CBv2SequenceKV {
                let row = makeRow(
                    window: window, maxLength: histories[index] + n + 8)
                if let entry = history[index] { _ = row.update(keys: entry.0, values: entry.1) }
                return row
            }

            let packedRows = (0 ..< batch).map(freshRow)
            let packed = CBv2AttentionV1.updateAndAttend(
                rows: packedRows, kind: kind,
                queries: queries, keys: keys, values: values, scale: attnScale, sinks: nil)
            eval(packed)
            #expect(packed.shape == [batch, queryHeads, n, headDim], "packed output shape")

            for index in 0 ..< batch {
                let soloRow = freshRow(index)
                let solo = CBv2AttentionV1.updateAndAttend(
                    rows: [soloRow], kind: kind,
                    queries: queries[index ..< (index + 1)],
                    keys: keys[index ..< (index + 1)],
                    values: values[index ..< (index + 1)],
                    scale: attnScale, sinks: nil)
                eval(solo)
                let slice = packed[index ..< (index + 1)]
                let label =
                    "window=\(windowLabel(window)) row=\(index) history=\(histories[index])"
                let delta = maxAbsDiff(slice, solo)
                #expect(
                    bitIdentical(slice, solo),
                    note("packed row must equal the singleton run — \(label) (d=\(delta))"))
                #expect(
                    packedRows[index].absoluteOffset == soloRow.absoluteOffset,
                    note("packed row offset must match the singleton run — \(label)"))
                #expect(
                    bitIdentical(
                        packedRows[index].snapshot().keys, soloRow.snapshot().keys),
                    note("packed row keys must match the singleton run — \(label)"))
            }
        }
    }
}

// MARK: - 2. blockSize == 1 is EXACTLY the pre-change serial path

@Suite("CBv2QueryBlockAttention serial-path equivalence")
struct CBv2QueryBlockAttentionSerialTests {

    /// `attendSerialQueries` now delegates to `attendQueryBlocks(blockSize: 1)`.
    /// That is only a safe refactor if the composed call sequence is
    /// unchanged, so this asserts BIT equality (no tolerance) against the
    /// transcribed pre-change loop.
    @Test func serializedQueriesAreBitIdenticalToTheLegacyLoop() {
        let windows: [Int?] = [1024, 64, nil]
        let lengths = [2, 17, 129, 300]
        let histories = [0, 500, 1024]
        var seed: UInt64 = 0xB10C_2001

        for window in windows {
            for n in lengths {
                for history in histories {
                    seed &+= 1
                    let fixture = ChunkFixture(
                        n: n, history: history, window: window, seed: seed)
                    let kind = layerKind(window: window)

                    let productionRow = fixture.freshRow()
                    let produced = CBv2AttentionV1.updateAndAttend(
                        rows: [productionRow], kind: kind,
                        queries: fixture.queries, keys: fixture.keys, values: fixture.values,
                        scale: attnScale, sinks: nil, softcap: nil, spanContexts: nil,
                        serializeQueries: true)

                    let reference = fixture.committedRow()
                    let legacy = legacySerialReference(
                        queries: fixture.queries, keys: reference.keys,
                        values: reference.values, newTokenCount: n, window: window)

                    eval(produced, legacy)
                    let label = "window=\(windowLabel(window)) n=\(n) history=\(history)"
                    let delta = maxAbsDiff(produced, legacy)
                    #expect(
                        bitIdentical(produced, legacy),
                        note(
                            "blockSize 1 must be BIT-identical to the pre-change serial path — \(label) (maxAbsDiff=\(delta))"
                        ))
                }
            }
        }
    }
}

// MARK: - 3. Strict causality — no future leakage

@Suite("CBv2QueryBlockAttention causality")
struct CBv2QueryBlockAttentionCausalityTests {

    /// Perturb the K (and separately the V) of chunk position `p + 1` and
    /// assert every query up to and including `p` is BIT-unchanged, while
    /// query `p + 1` does move (proving the perturbation was observable at
    /// all). This needs no reference implementation, so an off-by-one shared
    /// with a reference cannot hide from it.
    private func assertNoLeakage(
        n: Int, history: Int, window: Int?, boundary p: Int, seed: UInt64
    ) {
        let fixture = ChunkFixture(n: n, history: history, window: window, seed: seed)
        let kind = layerKind(window: window)
        precondition(p + 1 < n)

        func run(keys: MLXArray, values: MLXArray) -> MLXArray {
            let row = fixture.freshRow()
            let out = CBv2AttentionV1.updateAndAttend(
                rows: [row], kind: kind,
                queries: fixture.queries, keys: keys, values: values,
                scale: attnScale, sinks: nil)
            eval(out)
            return out
        }

        let baseline = run(keys: fixture.keys, values: fixture.values)
        let blocked = CBv2AttentionV1.shouldBlockQueries(n)
        let label =
            "window=\(windowLabel(window)) n=\(n) history=\(history) p=\(p) blocked=\(blocked)"

        // Future VALUE vector.
        let valuePerturbed = run(
            keys: fixture.keys, values: fixture.perturbed(valuesAt: p + 1, delta: 7.0))
        let vPast = maxAbsDiff(
            queryRows(baseline, 0, p + 1), queryRows(valuePerturbed, 0, p + 1))
        let vSelf = maxAbsDiff(
            queryRows(baseline, p + 1, p + 2), queryRows(valuePerturbed, p + 1, p + 2))
        #expect(
            bitIdentical(queryRows(baseline, 0, p + 1), queryRows(valuePerturbed, 0, p + 1)),
            note("queries <= p must not see V at p+1 — \(label) (maxAbsDiff=\(vPast))"))
        #expect(
            vSelf > 1e-3,
            note("query p+1 MUST see its own V — perturbation unobservable — \(label)"))

        // Future KEY vector.
        let keyPerturbed = run(
            keys: fixture.perturbed(keysAt: p + 1, delta: 7.0), values: fixture.values)
        let kPast = maxAbsDiff(queryRows(baseline, 0, p + 1), queryRows(keyPerturbed, 0, p + 1))
        let kSelf = maxAbsDiff(
            queryRows(baseline, p + 1, p + 2), queryRows(keyPerturbed, p + 1, p + 2))
        #expect(
            bitIdentical(queryRows(baseline, 0, p + 1), queryRows(keyPerturbed, 0, p + 1)),
            note("queries <= p must not see K at p+1 — \(label) (maxAbsDiff=\(kPast))"))
        #expect(
            kSelf > 1e-3,
            note("query p+1 MUST see its own K — perturbation unobservable — \(label)"))
    }

    /// Boundaries chosen around the block edges of the live width: the last
    /// query of a block, the first query of the next block, and interior
    /// positions. `visibleEnd` is per-BLOCK, so a `+1` there leaks the first
    /// token of the following block into the whole current block.
    private func boundaries(n: Int) -> [Int] {
        let b = max(1, liveBlock)
        var candidates = [0, 1, b - 2, b - 1, b, b + 1, 2 * b - 1, 2 * b, n - 2]
        candidates.append(contentsOf: [n / 2, n / 2 + 1])
        return Array(Set(candidates.filter { $0 >= 0 && $0 + 1 < n })).sorted()
    }

    @Test func fullAttentionNeverLeaksFutureTokens() {
        let n = max(300, liveBlock * 2 + 40)
        var seed: UInt64 = 0xB10C_3001
        for history in [0, 512] {
            for p in boundaries(n: n) {
                seed &+= 1
                assertNoLeakage(n: n, history: history, window: nil, boundary: p, seed: seed)
            }
        }
    }

    @Test func slidingWindowAttentionNeverLeaksFutureTokens() {
        let n = max(300, liveBlock * 2 + 40)
        var seed: UInt64 = 0xB10C_3501
        for window in [64, 1024] {
            for history in [0, 512] {
                for p in boundaries(n: n) {
                    seed &+= 1
                    assertNoLeakage(
                        n: n, history: history, window: window, boundary: p, seed: seed)
                }
            }
        }
    }
}

// MARK: - 4. Sliding-window boundary, pinned from both sides

@Suite("CBv2QueryBlockAttention window boundary")
struct CBv2QueryBlockAttentionWindowTests {

    private func runWithChunkKV(
        _ fixture: ChunkFixture, keys: MLXArray, values: MLXArray
    ) -> MLXArray {
        let row = fixture.freshRow()
        let out = CBv2AttentionV1.updateAndAttend(
            rows: [row], kind: layerKind(window: fixture.window),
            queries: fixture.queries, keys: keys, values: values,
            scale: attnScale, sinks: nil)
        eval(out)
        return out
    }

    /// Query at chunk index `p` (absolute `history + p`) must attend exactly
    /// `[p - window + 1, p]`. Perturbing `p - window` (one token too old)
    /// must be invisible; perturbing `p - window + 1` (the oldest visible
    /// token) must move the output. Together these pin `visibleStart` in both
    /// directions — a floor that is one too late silently truncates the
    /// window, a floor one too early silently widens it.
    private func assertWindowFloor(
        window: Int, n: Int, history: Int, probe p: Int, seed: UInt64
    ) {
        let fixture = ChunkFixture(n: n, history: history, window: window, seed: seed)
        precondition(p - window >= 0 && p < n)
        let baseline = runWithChunkKV(fixture, keys: fixture.keys, values: fixture.values)
        let label = "window=\(window) n=\(n) history=\(history) p=\(p)"

        for (name, index, mustMove) in [
            ("outside (p-window)", p - window, false),
            ("inside (p-window+1)", p - window + 1, true),
        ] {
            let keyRun = runWithChunkKV(
                fixture, keys: fixture.perturbed(keysAt: index, delta: 7.0),
                values: fixture.values)
            let valueRun = runWithChunkKV(
                fixture, keys: fixture.keys,
                values: fixture.perturbed(valuesAt: index, delta: 7.0))
            let keyDelta = maxAbsDiff(queryRows(baseline, p, p + 1), queryRows(keyRun, p, p + 1))
            let valueDelta = maxAbsDiff(
                queryRows(baseline, p, p + 1), queryRows(valueRun, p, p + 1))
            print(
                "[cbv2-query-block window] \(label) \(name) index=\(index) dK=\(keyDelta) dV=\(valueDelta)"
            )
            if mustMove {
                #expect(
                    keyDelta > 1e-3,
                    note("query p must attend the key at p-window+1 — \(label) \(name)"))
                #expect(
                    valueDelta > 1e-3,
                    note("query p must attend the value at p-window+1 — \(label) \(name)"))
            } else {
                #expect(
                    bitIdentical(queryRows(baseline, p, p + 1), queryRows(keyRun, p, p + 1)),
                    note("query p must NOT attend the key at p-window — \(label) (d=\(keyDelta))"))
                #expect(
                    bitIdentical(queryRows(baseline, p, p + 1), queryRows(valueRun, p, p + 1)),
                    note(
                        "query p must NOT attend the value at p-window — \(label) (d=\(valueDelta))"
                    ))
            }
        }
    }

    @Test func slidingWindowFloorIsExact() {
        let n = max(300, liveBlock * 2 + 40)
        var seed: UInt64 = 0xB10C_4001
        let window = 64
        // Probes inside the first block, straddling a block edge, and deep in
        // a later block: `visibleStart` is recomputed per block, so a wrong
        // floor can appear in one block and not another.
        let probes = Array(Set([
            window, window + 1, liveBlock - 1, liveBlock, liveBlock + 1,
            2 * liveBlock, n - 1,
        ].filter { $0 - window >= 0 && $0 < n })).sorted()
        for history in [0, 512] {
            for p in probes {
                seed &+= 1
                assertWindowFloor(window: window, n: n, history: history, probe: p, seed: seed)
            }
        }
    }

    /// Control: with FULL attention the same "outside the window" position is
    /// visible, so the sliding-window test above is actually detecting the
    /// window and not an artifact of the perturbation.
    @Test func fullAttentionHasNoWindowFloor() {
        let n = max(300, liveBlock * 2 + 40)
        let p = min(n - 1, 2 * liveBlock)
        let fixture = ChunkFixture(n: n, history: 0, window: nil, seed: 0xB10C_4501)
        let baseline = runWithChunkKV(fixture, keys: fixture.keys, values: fixture.values)
        let far = runWithChunkKV(
            fixture, keys: fixture.keys, values: fixture.perturbed(valuesAt: 0, delta: 7.0))
        #expect(
            maxAbsDiff(queryRows(baseline, p, p + 1), queryRows(far, p, p + 1)) > 1e-4,
            "a full-attention query must see token 0 regardless of distance")
    }
}

// MARK: - 5. Blocking must not touch committed KV state

@Suite("CBv2QueryBlockAttention KV state")
struct CBv2QueryBlockAttentionKVStateTests {

    private func assertCommitUnchanged(n: Int, history: Int, window: Int?, seed: UInt64) {
        let fixture = ChunkFixture(n: n, history: history, window: window, seed: seed)
        let kind = layerKind(window: window)
        let label = "window=\(windowLabel(window)) n=\(n) history=\(history)"

        // (a) blocking on (n > queryBlockSize), (b) blockSize 1, (c) no
        // attention at all — only `update()`. All three must commit the same
        // bytes: `update` runs before any attention decision.
        let blockedRow = fixture.freshRow()
        _ = CBv2AttentionV1.updateAndAttend(
            rows: [blockedRow], kind: kind,
            queries: fixture.queries, keys: fixture.keys, values: fixture.values,
            scale: attnScale, sinks: nil)

        let serialRow = fixture.freshRow()
        _ = CBv2AttentionV1.updateAndAttend(
            rows: [serialRow], kind: kind,
            queries: fixture.queries, keys: fixture.keys, values: fixture.values,
            scale: attnScale, sinks: nil, softcap: nil, spanContexts: nil,
            serializeQueries: true)

        let commitOnlyRow = fixture.freshRow()
        _ = commitOnlyRow.update(keys: fixture.keys, values: fixture.values)

        let expected = commitOnlyRow.snapshot()
        for (name, row) in [("blocked", blockedRow), ("serial", serialRow)] {
            #expect(
                row.absoluteOffset == commitOnlyRow.absoluteOffset,
                "\(name) absoluteOffset — \(label)")
            #expect(
                row.retainedCount == commitOnlyRow.retainedCount,
                "\(name) retainedCount — \(label)")
            let actual = row.snapshot()
            #expect(actual.offset == expected.offset, "\(name) snapshot offset — \(label)")
            #expect(
                bitIdentical(actual.keys, expected.keys),
                "\(name) retained keys must be bit-identical — \(label)")
            #expect(
                bitIdentical(actual.values, expected.values),
                "\(name) retained values must be bit-identical — \(label)")
        }
    }

    @Test func committedKVIsIndependentOfBlocking() {
        var seed: UInt64 = 0xB10C_5001
        let n = max(300, liveBlock * 2 + 40)
        for window in [nil, 64, 1024] as [Int?] {
            for history in [0, 500, 1024, 5000] {
                seed &+= 1
                assertCommitUnchanged(n: n, history: history, window: window, seed: seed)
            }
        }
    }
}

// MARK: - 6. Borrow path (Gemma-4 cross-layer KV sharing)

@Suite("CBv2QueryBlockAttention borrow path")
struct CBv2QueryBlockAttentionBorrowTests {

    /// The borrowing layer's `historyCount` comes from the SOURCE row's
    /// pre-eviction chunk views, not from its own state. A wrong
    /// `historyCount` shifts every block's visible span and silently desyncs
    /// the shared layer from its source.
    private func assertBorrowParity(
        n: Int, history: Int, window: Int?, serialize: Bool, seed: UInt64
    ) {
        let fixture = ChunkFixture(n: n, history: history, window: window, seed: seed)
        let sourceKind = layerKind(window: window)
        let sharedKind = layerKind(window: window, sharesKVWithLayer: 0)

        // The source layer already appended this step's tokens earlier in the
        // forward pass — that is the precondition of `attendBorrowing`.
        let sourceRow = fixture.freshRow()
        let offsetBefore = sourceRow.absoluteOffset
        _ = sourceRow.update(keys: fixture.keys, values: fixture.values)
        let produced = CBv2AttentionV1.attendBorrowing(
            sourceRows: [sourceRow], sourceKind: sourceKind, kind: sharedKind,
            queries: fixture.queries, scale: attnScale, sinks: nil, softcap: nil,
            spanContexts: nil, serializeQueries: serialize)

        // Identical twin, same commit, references taken from the same views.
        let twin = fixture.freshRow()
        _ = twin.update(keys: fixture.keys, values: fixture.values)
        let (bk, bv) = borrowViews(of: twin)
        let kL = bk.dim(2)
        let single = singleCallReference(
            queries: fixture.queries, keys: bk, values: bv, window: window)
        let absolute = absolutePositionReference(
            queries: fixture.queries, keys: bk, values: bv,
            queryStart: offsetBefore, keyStart: offsetBefore - (kL - n), window: window)

        eval(produced, single, absolute)
        let label =
            "window=\(windowLabel(window)) n=\(n) history=\(history) kL=\(kL) b=\(serialize ? 1 : liveBlock)"
        let dSingle = maxAbsDiff(produced, single)
        let dAbsolute = maxAbsDiff(produced, absolute)
        print("[cbv2-query-block borrow] \(label) dVsSingleCall=\(dSingle) dVsAbsolute=\(dAbsolute)")
        #expect(
            produced.shape == [1, queryHeads, n, headDim], "borrow output shape — \(label)")
        #expect(dSingle < 2e-5, "borrowed blocked attention must match single-call — \(label)")
        #expect(dAbsolute < 5e-5, "borrowed blocked attention must match absolute — \(label)")

        // Borrowing must not mutate the source row.
        #expect(
            sourceRow.absoluteOffset == twin.absoluteOffset,
            "attendBorrowing must not advance the source row — \(label)")
        #expect(
            bitIdentical(sourceRow.snapshot().keys, twin.snapshot().keys),
            "attendBorrowing must not mutate source keys — \(label)")
        #expect(
            bitIdentical(sourceRow.snapshot().values, twin.snapshot().values),
            "attendBorrowing must not mutate source values — \(label)")
    }

    @Test func borrowedChunkAttentionMatchesTheUnblockedPath() {
        var seed: UInt64 = 0xB10C_6001
        for window in [nil, 64, 1024] as [Int?] {
            for n in [17, 129, 512] {
                for history in [0, 500, 1024, 5000] {
                    seed &+= 1
                    assertBorrowParity(
                        n: n, history: history, window: window, serialize: false, seed: seed)
                }
            }
        }
    }

    @Test func borrowedChunkAtBlockWidthOneMatchesTheUnblockedPath() {
        var seed: UInt64 = 0xB10C_6501
        for window in [nil, 64, 1024] as [Int?] {
            for history in [0, 1024] {
                seed &+= 1
                assertBorrowParity(
                    n: 129, history: history, window: window, serialize: true, seed: seed)
            }
        }
    }

    /// Causality on the borrow path: the shared layer must not see the future
    /// either, even though its K/V comes from someone else's storage.
    @Test func borrowedChunkAttentionNeverLeaksFutureTokens() {
        let n = max(300, liveBlock * 2 + 40)
        var seed: UInt64 = 0xB10C_6901
        for window in [nil, 64] as [Int?] {
            for p in [liveBlock - 1, liveBlock, 2 * liveBlock].filter({ $0 + 1 < n }) {
                seed &+= 1
                let fixture = ChunkFixture(n: n, history: 512, window: window, seed: seed)
                let sourceKind = layerKind(window: window)
                let sharedKind = layerKind(window: window, sharesKVWithLayer: 0)

                func run(keys: MLXArray, values: MLXArray) -> MLXArray {
                    let row = fixture.freshRow()
                    _ = row.update(keys: keys, values: values)
                    let out = CBv2AttentionV1.attendBorrowing(
                        sourceRows: [row], sourceKind: sourceKind, kind: sharedKind,
                        queries: fixture.queries, scale: attnScale, sinks: nil)
                    eval(out)
                    return out
                }

                let baseline = run(keys: fixture.keys, values: fixture.values)
                let perturbed = run(
                    keys: fixture.keys, values: fixture.perturbed(valuesAt: p + 1, delta: 7.0))
                let label = "window=\(windowLabel(window)) n=\(n) p=\(p)"
                #expect(
                    bitIdentical(
                        queryRows(baseline, 0, p + 1), queryRows(perturbed, 0, p + 1)),
                    "borrowed queries <= p must not see V at p+1 — \(label)")
                #expect(
                    maxAbsDiff(
                        queryRows(baseline, p + 1, p + 2), queryRows(perturbed, p + 1, p + 2))
                        > 1e-3,
                    "borrowed query p+1 MUST see its own V — \(label)")
            }
        }
    }
}

// MARK: - 7. Gating

@Suite("CBv2QueryBlockAttention gating")
struct CBv2QueryBlockAttentionGatingTests {

    /// `queryBlockSize` is a lazily-initialized `static let`, so its value is
    /// fixed for the process. Pin the parse contract against the live env.
    @Test func blockWidthFollowsTheEnvironmentKnob() {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_ATTN_QUERY_BLOCK"]
        if let raw, let value = Int(raw), value >= 0 {
            #expect(
                liveBlock == value,
                "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK=\(raw) must select block width \(value)")
        } else {
            #expect(liveBlock == 128, "default query block width is 128 (raw=\(raw ?? "unset"))")
        }
    }

    @Test func decodeIsNeverBlocked() {
        #expect(
            !CBv2AttentionV1.shouldBlockQueries(1),
            "L == 1 (decode) must always take the single-call path")
        #expect(
            !CBv2AttentionV1.shouldBlockQueries(0),
            "L == 0 must never be blocked")
    }

    @Test func gateTracksTheLiveBlockWidth() {
        guard liveBlock > 0 else {
            // Kill switch engaged: nothing is ever blocked.
            for L in [1, 2, 127, 128, 129, 512, 100_000] {
                #expect(
                    !CBv2AttentionV1.shouldBlockQueries(L),
                    "queryBlockSize == 0 must disable blocking for every L (L=\(L))")
            }
            return
        }
        #expect(
            !CBv2AttentionV1.shouldBlockQueries(liveBlock),
            "L == queryBlockSize must not be blocked (a single block is the single call)")
        if liveBlock > 1 {
            #expect(
                !CBv2AttentionV1.shouldBlockQueries(liveBlock - 1),
                "L < queryBlockSize must not be blocked")
        }
        #expect(
            CBv2AttentionV1.shouldBlockQueries(liveBlock + 1),
            "L == queryBlockSize + 1 must be blocked")
        #expect(
            CBv2AttentionV1.shouldBlockQueries(liveBlock * 4),
            "long chunks must be blocked")
    }

    /// Vision chunks retain q-blocking without losing the bidirectional
    /// overlay. A span that crosses a q-block boundary must make each touched
    /// block retain the complete span's K/V, then match the whole-rectangle
    /// reference within the ordinary blocked-attention reduction tolerance.
    @Test func spanBearingChunksRetainOverlayUnderQueryBlocking() {
        let n = max(300, liveBlock * 2 + 40)
        for window in [nil, 1024] as [Int?] {
            for history in [0, 512] {
                let fixture = ChunkFixture(
                    n: n, history: history, window: window, seed: 0xB10C_7001)
                let kind = layerKind(window: window)
                let chunkEnd = history + n
                // Ordered, non-overlapping, fully inside the chunk (the
                // scheduler's invariant). One block deliberately STRADDLES the
                // first block edge when the live width leaves room: blocking a
                // span chunk would cut that bidirectional block in half.
                var blocks = [CBv2ImageSpan(tokenOffset: history + 20, length: 16)]
                if liveBlock >= 48, liveBlock + 16 < n {
                    blocks.append(
                        CBv2ImageSpan(tokenOffset: history + liveBlock - 8, length: 24))
                } else {
                    blocks.append(CBv2ImageSpan(tokenOffset: history + n / 2, length: 24))
                }
                let context = CBv2SpanChunkContext(chunkEnd: chunkEnd, blocks: blocks)

                let row = fixture.freshRow()
                let produced = CBv2AttentionV1.updateAndAttend(
                    rows: [row], kind: kind,
                    queries: fixture.queries, keys: fixture.keys, values: fixture.values,
                    scale: attnScale, sinks: nil, softcap: nil, spanContexts: [context])

                let reference = fixture.committedRow()
                let mask = CBv2AttentionV1.spanChunkMask(
                    L: n, kL: reference.keys.dim(2), window: window, context: context)
                let single = MLXFast.scaledDotProductAttention(
                    queries: fixture.queries, keys: reference.keys, values: reference.values,
                    scale: attnScale, mask: .array(mask), sinks: nil)

                eval(produced, single)
                let gate = CBv2AttentionV1.shouldBlockQueries(n)
                let label =
                    "window=\(windowLabel(window)) n=\(n) history=\(history) shouldBlockQueries=\(gate)"
                let delta = maxAbsDiff(produced, single)
                #expect(
                    delta < 2e-5,
                    note(
                        "q-blocked span chunks must match the whole-span mask — \(label) (maxAbsDiff=\(delta))"
                    ))

                // And the span overlay must actually be doing something —
                // otherwise the assertion above would pass trivially against a
                // plain causal path too.
                let plain = singleCallReference(
                    queries: fixture.queries, keys: reference.keys,
                    values: reference.values, window: window)
                eval(plain)
                #expect(
                    maxAbsDiff(produced, plain) > 1e-4,
                    "the bidirectional span overlay must change the result — \(label)")
            }
        }
    }

    @Test func packedVisionAndTextRowsKeepIndependentQueryBlockMasks() {
        let n = max(300, liveBlock * 2 + 40)
        let history = 96
        for window in [nil, 64] as [Int?] {
            let vision = ChunkFixture(
                n: n, history: history, window: window, seed: 0xB10C_7101)
            let text = ChunkFixture(
                n: n, history: history, window: window, seed: 0xB10C_7102)
            let kind = layerKind(window: window)
            let boundary = max(32, liveBlock)
            let context = CBv2SpanChunkContext(
                chunkEnd: history + n,
                blocks: [
                    CBv2ImageSpan(
                        tokenOffset: history + boundary - 8, length: 24)
                ])

            let rows = [vision.freshRow(), text.freshRow()]
            let produced = CBv2AttentionV1.updateAndAttend(
                rows: rows, kind: kind,
                queries: concatenated([vision.queries, text.queries], axis: 0),
                keys: concatenated([vision.keys, text.keys], axis: 0),
                values: concatenated([vision.values, text.values], axis: 0),
                scale: attnScale, sinks: nil, softcap: nil,
                spanContexts: [context, nil])

            let visionReference = vision.committedRow()
            let visionMask = CBv2AttentionV1.spanChunkMask(
                L: n, kL: visionReference.keys.dim(2),
                window: window, context: context)
            let expectedVision = MLXFast.scaledDotProductAttention(
                queries: vision.queries, keys: visionReference.keys,
                values: visionReference.values, scale: attnScale,
                mask: .array(visionMask), sinks: nil)
            let textReference = text.committedRow()
            let expectedText = singleCallReference(
                queries: text.queries, keys: textReference.keys,
                values: textReference.values, window: window)
            eval(produced, expectedVision, expectedText)

            let label = "window=\(windowLabel(window)) n=\(n)"
            #expect(
                maxAbsDiff(produced[0 ..< 1], expectedVision) < 2e-5,
                "packed vision row must retain its bidirectional overlay — \(label)")
            #expect(
                maxAbsDiff(produced[1 ..< 2], expectedText) < 2e-5,
                "neighboring text row must retain q-block causal/window attention — \(label)")
        }
    }

    @Test func separatedSpansEmitOnlyIntersectingMaskTermsAndKeepParity() {
        let n = max(400, liveBlock * 3)
        let spans = [
            CBv2ImageSpan(tokenOffset: 20, length: 16),
            CBv2ImageSpan(tokenOffset: 140, length: 16),
            CBv2ImageSpan(tokenOffset: 280, length: 16),
        ]
        let context = CBv2SpanChunkContext(chunkEnd: n, blocks: spans)

        #expect(
            Array(
                CBv2AttentionV1.spanBlocksIntersectingQueryRange(
                    queryAbsoluteStart: 0, queryCount: 64, context: context))
                == [spans[0]])
        #expect(
            CBv2AttentionV1.spanBlocksIntersectingQueryRange(
                queryAbsoluteStart: 64, queryCount: 32, context: context
            ).isEmpty)
        #expect(
            Array(
                CBv2AttentionV1.spanBlocksIntersectingQueryRange(
                    queryAbsoluteStart: 96, queryCount: 96, context: context))
                == [spans[1]])
        #expect(
            Array(
                CBv2AttentionV1.spanBlocksIntersectingQueryRange(
                    queryAbsoluteStart: 224, queryCount: 96, context: context))
                == [spans[2]])

        let fixture = ChunkFixture(
            n: n, history: 0, window: 64, seed: 0xB10C_7201)
        let row = fixture.freshRow()
        let produced = CBv2AttentionV1.updateAndAttend(
            rows: [row], kind: layerKind(window: 64),
            queries: fixture.queries, keys: fixture.keys, values: fixture.values,
            scale: attnScale, sinks: nil, spanContexts: [context])
        let reference = fixture.committedRow()
        let wholeMask = CBv2AttentionV1.spanChunkMask(
            L: n, kL: reference.keys.dim(2), window: 64, context: context)
        let expected = MLXFast.scaledDotProductAttention(
            queries: fixture.queries, keys: reference.keys, values: reference.values,
            scale: attnScale, mask: .array(wholeMask), sinks: nil)
        eval(produced, expected)
        #expect(
            maxAbsDiff(produced, expected) < 2e-5,
            "intersecting-only span terms must preserve whole-mask attention parity")
    }
}
