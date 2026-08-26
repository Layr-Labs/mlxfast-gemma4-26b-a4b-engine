import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Gemma 4 DFlash TARGET CONFORMANCE + drafter BIND — ported 2026-08-25 from
// Layr-Labs/mlx-swift-lm `origin/dflash-framework-updates`
// (Tests/MLXLMTests/Gemma4DFlashForwardTests.swift @ d41c3003), adapted to
// this engine's vendored `Gemma4Text.swift` (which descends from
// mlx-swift-lm main @ ed55bee, not that branch).
//
// Dropped from the port: `autoVectorVerifySuffixUsesConservativeEnvelope`
// and the sequential/mixed/auto verify coverage. Those exercise the fork's
// verify-path FUSIONS, which this graft deliberately does not carry (see
// Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift).
//
// Added here, not in the fork: `logitsForDFlashHidden` must NOT softcap
// (the drafter applies its own), and the bind/draftBlock pair that is the
// whole point of the real-loader lane.
//
// GPU-free at fixture scale, forced onto `.cpu`.
@Suite("Gemma4 DFlash target conformance")
struct Gemma4DFlashForwardTests {

    private func tinyGemma4Config(
        hiddenLayers: Int = 4,
        sharedLayers: Int = 0,
        slidingWindow: Int = 16
    ) throws -> Gemma4TextConfiguration {
        let layerTypes = (0 ..< hiddenLayers)
            .map { $0 % 2 == 0 ? #""sliding_attention""# : #""full_attention""# }
            .joined(separator: ", ")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": \(hiddenLayers),
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 8,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": \(sharedLayers),
                "layer_types": [\(layerTypes)],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// A drafter whose geometry matches `tinyGemma4Config`'s target: same
    /// hidden size, same vocab, `num_target_layers == num_hidden_layers`.
    private func tinyDFlashConfig(
        numTargetLayers: Int = 4,
        targetLayerIds: String = "0, 3",
        blockSize: Int = 4
    ) throws -> DFlashConfiguration {
        let json = """
            {
                "architectures": ["DFlashDraftModel"],
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "max_position_embeddings": 128,
                "block_size": \(blockSize),
                "num_target_layers": \(numTargetLayers),
                "layer_types": ["full_attention", "full_attention"],
                "tie_word_embeddings": true,
                "dflash_config": {
                    "target_layer_ids": [\(targetLayerIds)],
                    "mask_token_id": 4
                }
            }
            """
        return try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
    }

    // MARK: - Target conformance

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3])
            let reference = model(tokens, cache: model.newCache(parameters: nil))

            #expect(allClose(forward.logits, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    /// Tap ORDER is the config's order, not sorted order — the drafter's `fc`
    /// projection was trained against that concatenation.
    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let ascending = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3])
            let descending = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [3, 1])

            #expect(ascending.hiddenStates.count == 2)
            #expect(ascending.hiddenStates[0].shape == [1, 3, 16])
            #expect(ascending.targetHidden.shape == [1, 3, 32])
            // Same two layers, opposite request order: slot 0 of one is slot 1
            // of the other.
            #expect(
                allClose(
                    ascending.hiddenStates[0], descending.hiddenStates[1],
                    rtol: 1e-6, atol: 1e-6
                ).item(Bool.self))
            #expect(
                allClose(
                    ascending.hiddenStates[1], descending.hiddenStates[0],
                    rtol: 1e-6, atol: 1e-6
                ).item(Bool.self))
        }
    }

    @Test func greedyTokensMatchSoftcappedLogitsArgmax() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config())
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3])
            let greedy = try model.forwardGreedyTokensForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3])

            #expect((greedy.tokens .== forward.logits.argMax(axis: -1)).all().item(Bool.self))
        }
    }

    /// `logitsForDFlashHidden` is the RAW head. The drafter applies its own
    /// `final_logit_softcapping`; handing it the target's already-capped
    /// logits would cap twice, with the wrong constant. The target's own
    /// logits keep the cap — hence the two must NOT be equal on a config
    /// whose cap actually binds.
    @Test func drafterBorrowedLMHeadIsNotSoftcapped() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try tinyGemma4Config()
            #expect(config.finalLogitSoftcapping > 0)
            let model = Gemma4TextModel(config)
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            // The target's own logits are capped: |logit| < cap everywhere,
            // by construction of tanh(x/c)*c.
            let capped = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [0, 3]
            ).logits
            #expect(
                abs(capped).max().item(Float.self) < config.finalLogitSoftcapping)

            // The drafter's borrowed head is not. Drive it with a hidden whose
            // magnitude puts the raw projection well outside the cap; a capped
            // head could not produce it.
            let hidden =
                MLXArray.ones([1, 1, config.hiddenSize]) * Float(200)
            let raw = model.logitsForDFlashHidden(hidden)
            eval(raw)
            // logitsForDFlashHidden must be the RAW head: the drafter applies
            // its own final_logit_softcapping, so a capped value here would
            // be capped twice with the wrong constant.
            #expect(abs(raw).max().item(Float.self) > config.finalLogitSoftcapping)
        }
    }

    @Test func capturesSharedKVLayerHiddenStates() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(
                try tinyGemma4Config(hiddenLayers: 4, sharedLayers: 2))
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]

            let forward = try model.forwardForDFlash(
                tokens,
                cache: model.newCache(parameters: nil),
                targetLayerIds: [1, 3])

            #expect(forward.logits.shape == [1, 3, 32])
            #expect(forward.targetHidden.shape == [1, 3, 32])
        }
    }

    @Test func rejectsOutOfRangeTargetLayerIdsAtForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config(hiddenLayers: 4))
            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    MLXArray([Int32(1)])[.newAxis, .ellipsis],
                    cache: model.newCache(parameters: nil),
                    targetLayerIds: [0, 4])
            }
        }
    }

    // MARK: - Drafter bind (the real-loader lane's whole point)

    @Test func bindsToMatchingGemma4Target() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config(hiddenLayers: 4))
            eval(model)
            let drafter = DFlashDraftModel(config: try tinyDFlashConfig(numTargetLayers: 4))
            eval(drafter)
            try drafter.bind(target: model)

            let forward = try model.forwardForDFlash(
                MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis],
                cache: model.newCache(parameters: nil),
                targetLayerIds: drafter.config.targetLayerIds)
            // The drafter conditions on the LAST position's taps.
            let lastHidden = forward.targetHidden[0..., (-1)..., 0...]
            let drafted = try drafter.draftBlock(
                bonus: 1,
                targetHidden: lastHidden,
                cache: try drafter.makeCache(),
                blockSize: 4)
            eval(drafted)
            // blockSize - 1 proposals: the bonus column is not a proposal.
            #expect(drafted.shape == [1, 3])
        }
    }

    /// `num_target_layers` must equal the TARGET's layer count. This is the
    /// check that makes the real A4B pairing (drafter 30 / target 30) a fact
    /// rather than a hope.
    @Test func refusesDrafterWithWrongTargetLayerCount() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try tinyGemma4Config(hiddenLayers: 4))
            let drafter = DFlashDraftModel(
                config: try tinyDFlashConfig(numTargetLayers: 6, targetLayerIds: "0, 5"))
            #expect(
                throws: DFlashError.incompatibleDrafter(
                    field: "numTargetLayers", drafter: "6", target: "4")
            ) {
                try drafter.bind(target: model)
            }
        }
    }

    @Test func refusesForwardBeforeBind() throws {
        try Device.withDefaultDevice(.cpu) {
            let drafter = DFlashDraftModel(config: try tinyDFlashConfig())
            eval(drafter)
            #expect(throws: DFlashError.drafterNotBound) {
                _ = try drafter.draftBlock(
                    bonus: 1,
                    targetHidden: MLXArray.zeros([1, 1, 32]),
                    cache: try drafter.makeCache(),
                    blockSize: 4)
            }
        }
    }

    // MARK: - Depth ceiling (the echo == what runs)

    /// The dflash `effective_spec` echo and the round loop read the SAME
    /// resolver, so an unclamped default cannot be echoed. A drafter with a
    /// trained block of 4 caps depth at 3 regardless of what a caller asks.
    @Test func depthCeilingComesFromTheBoundDrafterBlock() throws {
        let drafter = DFlashDraftModel(config: try tinyDFlashConfig(blockSize: 4))
        let ceiling = gemma4DFlashMaxDepth(for: drafter)
        #expect(ceiling == 3)
        #expect(
            RuntimeWorkerSpecRegistry.resolveDFlashDepth(nil, maxDepth: ceiling) == 3)
        #expect(
            RuntimeWorkerSpecRegistry.resolveDFlashDepth(16, maxDepth: ceiling) == 3)
        #expect(
            RuntimeWorkerSpecRegistry.resolveDFlashDepth(2, maxDepth: ceiling) == 2)
        #expect(
            RuntimeWorkerSpecRegistry.resolveDFlashDepth(0, maxDepth: ceiling) == 1)
    }

    /// The engine's own block ceiling still applies on top of the drafter's:
    /// the real z-lab head's trained block is 16, and
    /// `experimentalDFlashMaxBlockSize` is 16, so the ceiling is 15.
    @Test func depthCeilingIsAlsoBoundedByTheEngineBlockCeiling() throws {
        let drafter = DFlashDraftModel(config: try tinyDFlashConfig(blockSize: 64))
        #expect(
            gemma4DFlashMaxDepth(for: drafter)
                == MLXFastConstants.experimentalDFlashMaxBlockSize - 1)
    }
}
