import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// TARGET-FORWARD IDENTITY invariants (exactness round three, 2026-08-25).
//
// The round-three box arbiter run (both legs on the shared width-1 CBv2
// executor, d69f11c3) settled the mechanism class: the SERIAL leg matches
// the legacy-recorded tapes on every prompt and every retry, while the
// MTP-BOUND leg alone diverges at deterministic prompt-specific steps —
// so binding the assistant-head drafter to the engine alters the TARGET's
// computed stream on the production tuple (the F3 flag from #29).
//
// The invariant this suite pins, at the strongest level a laptop can
// certify (BIT identity, not argmax identity): with a head involved,
// drafting or not, the target's logits for the committed path must be
// bit-identical to the plain forward's. Both tests are GREEN on this
// machine — the with/without-head entry points and the staged-vs-ring
// attention geometries are bit-identical at fixture scale here — which is
// the plainly-stated verdict of the round-three diagnosis: the flip is
// BOX-ONLY near-tie kernel physics (M5 + production dims + the QAT
// checkpoint; trap 3.1's class), not laptop-reproducible. These tests are
// regression tripwires for the identity, and the per-round sidecar
// diagnostics (RuntimeWorkerFreeRunDiagnostics.swift) are what turn the
// next box run into the localizing experiment: the divergence step aligns
// against the verify-round audit spans in the sidecar, separating
// verify-column flips from seed-step flips from chain-cadence plain-round
// flips.
//
// Entry-point identity, verified by reading at 77144a32 (cited in the PR):
// `cbv2ForwardWithHidden` (Gemma4Text.swift:2257) and the plain forward
// (:1554) share `forwardTrunk`; `capturePreNorm` only gates whether the
// already-computed pre-norm hidden is RETURNED (:1789-1790) — the logits
// graph is op-identical. The residual with-head differences are graph
// CONTEXT: staged-KV view geometry in verify columns, the extra retained
// hidden output, and chain-cadence differences in when plain rounds run
// non-chained. All value-identical at op level; only kernel dispatch can
// tell them apart, and only on silicon/shapes where a near-tie exists.

@Suite("MTPTargetForwardIdentity", .serialized)
struct MTPTargetForwardIdentityTests {

    private let vocabSize = 64
    private let hiddenSize = 32
    private let slidingWindow = 12

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
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

    private func promptTokens(length: Int, seed: Int) -> [Int] {
        var value = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
        return (0 ..< length).map { _ in
            value = value &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(value % UInt64(vocabSize))
        }
    }

    /// P1: forwardWithHidden vs plain forward — bit-identity of logits over
    /// a full greedy decode, same model, two independently-built but
    /// identically-advanced legacy cache stacks (legacy stack suffices: the
    /// two entry points share forwardTrunk; the probe is about the ENTRY
    /// POINT difference itself).
    @Test
    func p1ForwardWithHiddenBitEqualsPlainForward() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        MLXRandom.seed(0xB17_0001)
        let model = Gemma4TextModel(try targetConfig())
        eval(model)
        let prompt = promptTokens(length: 14, seed: 11)
        let n = 24

        let cacheA = model.newCache(parameters: nil)
        let cacheB = model.newCache(parameters: nil)
        let seedLogitsA = model(
            MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count]), cache: cacheA)
        let (seedLogitsB, _) = model.cbv2ForwardWithHidden(
            MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count]), caches: cacheB)
        eval(cacheA)
        eval(cacheB)
        var mismatches = 0
        var firstMismatch = -1
        func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let last = a[0..., -1, 0...]
            let lastB = b[0..., -1, 0...]
            return allClose(last, lastB, rtol: 0, atol: 0).item(Bool.self)
        }
        if !bitEqual(seedLogitsA, seedLogitsB) {
            mismatches += 1
            firstMismatch = 0
        }
        var last = seedLogitsA[0..., -1, 0...].argMax(axis: -1).item(Int.self)
        for step in 0 ..< n {
            let tok = MLXArray([Int32(last)]).reshaped([1, 1])
            let logitsA = model(tok, cache: cacheA)
            let (logitsB, _) = model.cbv2ForwardWithHidden(tok, caches: cacheB)
            if !bitEqual(logitsA, logitsB) {
                mismatches += 1
                if firstMismatch < 0 { firstMismatch = step + 1 }
            }
            last = logitsA[0..., -1, 0...].argMax(axis: -1).item(Int.self)
        }
        print("P1 forwardWithHidden-vs-forward: mismatches=\(mismatches) first=\(firstMismatch)")
        #expect(mismatches == 0)
    }

    /// P2: staged-transaction attention views vs plain ring views — the
    /// verify-column geometry. Two identically-advanced CBv2 width-1 engine
    /// states are impractical to construct outside the engine, so drive the
    /// REAL classes at the KV level: identical windowed rings, one plain
    /// decode step vs the same step under an armed staged write.
    @Test
    func p2StagedViewAttentionBitEqualsRingViewAttention() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        MLXRandom.seed(0xB17_0002)
        let kvHeads = 1
        let headDim = 16
        let window = 12
        // Advance two rings identically well past wrap.
        let ringA = CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        let ringB = CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        var lastKV: (MLXArray, MLXArray)? = nil
        for position in 0 ..< 40 {
            MLXRandom.seed(UInt64(1000 + position))
            let k = MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16)
            let v = MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16)
            _ = ringA.update(keys: k, values: v)
            _ = ringB.update(keys: k, values: v)
            lastKV = (k, v)
        }
        _ = lastKV
        // The probe step: same new K/V, plain vs staged.
        MLXRandom.seed(0xB17_0003)
        let nk = MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16)
        let nv = MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16)
        let q = MLXRandom.normal([1, 2, 1, headDim]).asType(.float16)

        let (plainK, plainV) = ringA.update(keys: nk, values: nv)
        ringB.beginSpeculativeWrite()
        let (stagedK, stagedV) = ringB.update(keys: nk, values: nv)

        // Same VALUES?
        let valuesEqual =
            allClose(plainK, stagedK, rtol: 0, atol: 0).item(Bool.self)
            && allClose(plainV, stagedV, rtol: 0, atol: 0).item(Bool.self)

        // Same attention BITS? (the fused SDPA the layer dispatches)
        let scale = Float(1.0 / Double(headDim).squareRoot())
        func attend(_ keys: MLXArray, _ values: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: q,
                keys: keys.asType(.float16),
                values: values.asType(.float16),
                scale: scale,
                mask: .none)
        }
        let outPlain = attend(plainK, plainV)
        let outStaged = attend(stagedK, stagedV)
        let attentionBitsEqual = allClose(outPlain, outStaged, rtol: 0, atol: 0).item(Bool.self)
        ringB.rollback(0)
        ringB.commitSpeculativeWrite()
        print(
            "P2 staged-vs-ring: valuesEqual=\(valuesEqual) "
            + "attentionBitsEqual=\(attentionBitsEqual) "
            + "plainShape=\(plainK.shape) stagedShape=\(stagedK.shape)")
        #expect(valuesEqual)
        #expect(attentionBitsEqual)
    }
}
