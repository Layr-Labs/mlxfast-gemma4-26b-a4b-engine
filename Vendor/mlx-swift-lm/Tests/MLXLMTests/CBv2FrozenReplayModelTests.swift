import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CBv2FrozenReplayGemmaClassTests: XCTestCase {
    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        eval(lhs, rhs)
        return zip(
            lhs.reshaped([-1]).asArray(Float.self),
            rhs.reshaped([-1]).asArray(Float.self)
        ).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    private func process(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        state: [CBv2SequenceKV?],
        tokens: ArraySlice<Int>
    ) -> MLXArray {
        let bank = CBv2LayerCacheBank(layerKinds: layerKinds)
        var logits = MLXArray.zeros([1, 1, 64])
        for token in tokens {
            logits = model.forward(
                tokens: MLXArray([Int32(token)]).reshaped(1, 1),
                caches: bank.layerCaches(rowStates: [state]))
            eval(logits)
        }
        return logits
    }

    func testActualGemma4TextModelKEqualsVHybridIsFrozenExact() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "attention_k_eq_v": true,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 4,
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6
            }
            """
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self,
            from: Data(json.utf8))
        MLXRandom.seed(0x47454D4D41)
        let gemma = Gemma4TextModel(config)
        eval(gemma)
        let model = CBv2SteppableLanguageModelAdapter(gemma)
        let kinds = gemma.cbv2LayerKinds
        let prompt = makePromptTokens(length: 41, seed: 0x474, vocabSize: 64)
        let matched = 40

        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds,
            backend: .contiguousUnquantized)
        let plan = try XCTUnwrap(capability.plan(matchedBoundary: matched))
        XCTAssertEqual(plan.strategy, .frozenFullReplay)
        XCTAssertEqual(plan.replayTokens, 20)
        XCTAssertEqual(plan.replayStart, 20)

        let coldBackend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 28))
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds,
            promptLength: prompt.count,
            maxLength: prompt.count + 64)
        let coldLogits = process(
            model: model,
            layerKinds: kinds,
            state: coldState,
            tokens: prompt[...])

        // Recreate the old mutating C-bound replay to pin the counterexample.
        let oldState: [CBv2SequenceKV?] = try zip(kinds, coldState).map {
            kind, donor in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return CBv2WindowedSequenceKV(
                    window: window,
                    kvHeads: kind.kvHeads,
                    headDim: kind.headDim,
                    initialOffset: plan.replayStart)
            case .full:
                let snapshot = try XCTUnwrap(donor?.snapshot())
                let row = CBv2FullSequenceKV(
                    promptLength: plan.replayStart,
                    maxLength: prompt.count + 64,
                    kvHeads: kind.kvHeads,
                    headDim: kind.headDim)
                _ = row.update(
                    keys: snapshot.keys[.ellipsis, 0 ..< plan.replayStart, 0...],
                    values: snapshot.values[.ellipsis, 0 ..< plan.replayStart, 0...])
                return row
            }
        }
        let oldLogits = process(
            model: model,
            layerKinds: kinds,
            state: oldState,
            tokens: prompt[plan.replayStart...])

        let prefix = zip(kinds, coldState).map {
            kind, donor -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard kind.sharesKVWithLayer == nil,
                case .full = kind.attention,
                let snapshot = donor?.snapshot()
            else { return nil }
            return (
                snapshot.keys[.ellipsis, 0 ..< matched, 0...],
                snapshot.values[.ellipsis, 0 ..< matched, 0...],
                matched)
        }
        let frozenBackend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 28))
        let frozenState = try frozenBackend.makeSequenceState(
            adopting: prefix,
            plan: plan,
            layerKinds: kinds,
            maxLength: prompt.count + 64)
        let frozenLogits = process(
            model: model,
            layerKinds: kinds,
            state: frozenState,
            tokens: prompt[plan.replayStart...])

        XCTAssertGreaterThan(maxAbsDiff(coldLogits, oldLogits), 0)
        XCTAssertEqual(maxAbsDiff(coldLogits, frozenLogits), 0, accuracy: 0)

        let fullLayer = try XCTUnwrap(kinds.firstIndex { kind in
            guard kind.sharesKVWithLayer == nil else { return false }
            if case .full = kind.attention { return true }
            return false
        })
        let coldFull = try XCTUnwrap(coldState[fullLayer]?.snapshot())
        let oldFull = try XCTUnwrap(oldState[fullLayer]?.snapshot())
        let frozenFull = try XCTUnwrap(frozenState[fullLayer]?.snapshot())
        XCTAssertGreaterThan(maxAbsDiff(coldFull.keys, oldFull.keys), 0)
        XCTAssertGreaterThan(maxAbsDiff(coldFull.values, oldFull.values), 0)
        XCTAssertEqual(maxAbsDiff(coldFull.keys, frozenFull.keys), 0, accuracy: 0)
        XCTAssertEqual(maxAbsDiff(coldFull.values, frozenFull.values), 0, accuracy: 0)
    }
}
