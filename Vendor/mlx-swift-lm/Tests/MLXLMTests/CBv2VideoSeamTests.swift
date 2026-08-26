// CBv2VideoSeamTests.swift — video-through-CBv2 (Gemma4 seam + engine
// exactness for the per-frame video span shape).
//
// v0.7.5 design: one image-span per sampled frame. The Gemma4 processor
// emits each sampled frame as its own `timestamp + boi + <video>×count + eoi`
// block; the ordinary text tokens between blocks mean the engine never
// coalesces across frames, so video rides the existing CBv2 vision span
// machinery unchanged. The new engine seam is
// `Gemma4.perVideoFrameVisionFeatures(pixels:frames:)` (the video-budget
// mirror of `perImageVisionFeatures`) plus the `videoPlaceholderTokenId`
// accessor the provider carves spans with.
//
// This suite proves, with no model downloads:
//  (a) PROCESSOR STRUCTURE: a synthetic 2-frame video through the REAL
//      `Gemma4Processor` expands into exactly two per-frame placeholder
//      runs, delimited by boi/eoi and separated by ordinary text tokens
//      (the mm:ss timestamp), sized to the derived per-frame video budget;
//  (b) NO COALESCING: span carving at the video placeholder id yields two
//      spans that `CBv2MultimodalPlan.coalescedBlocks` keeps separate, and
//      the reference mask discriminates the two-block shape from a
//      wrongly-merged single block;
//  (c) TOKEN-EXACT: CBv2 span prefill of the processor-derived two-frame
//      prompt through the REAL EngineV2 (chunked, span-snapped) matches the
//      wrapper-semantics full-sequence reference token-for-token for the
//      first N decode steps — the video analog of the PR#63 image-span
//      exactness suite (CBv2MultimodalTests), whose reference mimics the
//      MLXVLM wrapper's `prepare` + generate loop;
//  (d) SEAM BUDGET: on a tiny random-weight MLXVLM `Gemma4`,
//      `perVideoFrameVisionFeatures` runs the real tower at the VIDEO patch
//      budget — per-frame soft-token counts match the processor's
//      placeholder runs (regressing `isVideo: false` here inflates them to
//      the image budget and fails), the wrapper's own `prepare` accepts the
//      congruent counts (maskedScatter throws on any mismatch), and
//      `videoPlaceholderTokenId` mirrors `video_token_id` including its
//      258_884 default.
//
// Direct VLM and CBv2 forwards now share one `Gemma4TextModel` object. The
// engine contract below therefore exercises the same language tower as the
// wrapper-semantics reference rather than a separately reconstructed module.

import AVFoundation
import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXRandom
@testable import MLXVLM
import XCTest

@testable import MLXLMCommon

// MARK: - Stub tokenizer (drives the REAL Gemma4Processor deterministically)

/// Deterministic tokenizer for the Gemma4 processor: `<|video|>`/`<|image|>`
/// map to small fixed ids, ordinary text maps per-UTF8-byte into a text band
/// well away from the placeholder/delimiter ids, and the chat template
/// linearizes the Gemma4MessageGenerator content parts (one marker token per
/// media part). All emitted ids stay below TinyTestModel's vocab (128).
private struct Gemma4VideoStubTokenizer: Tokenizer {
    static let imageMarkerId = 6
    static let videoMarkerId = 7

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // Text band [20, 80): never collides with markers (6/7) or the
        // processor config's boi/eoi (10/11).
        Array(text.utf8).map { 20 + Int($0 % 60) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
    }

    func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: false) }
    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|image|>": return Self.imageMarkerId
        case "<|video|>": return Self.videoMarkerId
        default: return nil
        }
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var out = [1]  // bos-ish leading token
        for message in messages {
            if let parts = message["content"] as? [[String: String]] {
                for part in parts {
                    switch part["type"] {
                    case "image": out.append(Self.imageMarkerId)
                    case "video": out.append(Self.videoMarkerId)
                    default:
                        out.append(
                            contentsOf: encode(
                                text: part["text"] ?? "", addSpecialTokens: false))
                    }
                }
            } else if let text = message["content"] as? String {
                out.append(contentsOf: encode(text: text, addSpecialTokens: false))
            }
        }
        return out
    }
}

// MARK: - Suite

final class CBv2VideoSeamTests: XCTestCase {

    /// Delimiters the tiny processor config stamps around every frame block.
    private static let boiTokenId = 10
    private static let eoiTokenId = 11
    /// Geometry: patch 8, pooling 2, video budget 4 ⇒ a 64×64 frame resizes
    /// to 32×32 ⇒ (32/8)² = 16 patches ⇒ 16/2² = 4 soft tokens per frame.
    /// The IMAGE budget stays 16, so a video/image budget mix-up quadruples
    /// the per-frame count and fails the structure assertions.
    private static let expectedPerFrameSoftTokens = 4
    private static let imageSoftTokens = 16

    /// Tiny processor config congruent with `tinyVLMConfigJSON`'s vision
    /// geometry. `video_processor.max_soft_tokens` is the VIDEO budget.
    private static let processorConfigJSON = """
        {
            "processor_class": "Gemma4Processor",
            "patch_size": 8,
            "max_soft_tokens": \(imageSoftTokens),
            "pooling_kernel_size": 2,
            "image_seq_length": \(imageSoftTokens),
            "boi_token_id": \(boiTokenId),
            "eoi_token_id": \(eoiTokenId),
            "video_processor": { "max_soft_tokens": \(expectedPerFrameSoftTokens) }
        }
        """

    /// Tiny random-weight Gemma4 VLM: 2 text layers [sliding, full], PLE and
    /// MoE off, `use_bidirectional_attention: "vision"` (the production
    /// video mask path), 1-layer vision tower sharing the processor's patch
    /// geometry. Placeholder ids match the stub tokenizer's markers.
    private static let tinyVLMConfigJSON = """
        {
            "model_type": "gemma4",
            "image_token_id": \(Gemma4VideoStubTokenizer.imageMarkerId),
            "video_token_id": \(Gemma4VideoStubTokenizer.videoMarkerId),
            "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "intermediate_size": 64,
                "vocab_size": 512,
                "rms_norm_eps": 1e-6,
                "sliding_window": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false,
                "enable_moe_block": false,
                "use_bidirectional_attention": "vision"
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "patch_size": 8,
                "position_embedding_size": 64,
                "default_output_length": \(imageSoftTokens),
                "pooling_kernel_size": 2
            }
        }
        """

    private func makeProcessor() throws -> Gemma4Processor {
        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: Data(Self.processorConfigJSON.utf8))
        return Gemma4Processor(config, tokenizer: Gemma4VideoStubTokenizer())
    }

    private func makeTinyVLM(configJSON: String = CBv2VideoSeamTests.tinyVLMConfigJSON) throws
        -> MLXVLM.Gemma4
    {
        let config = try JSONDecoder().decode(
            MLXVLM.Gemma4Configuration.self, from: Data(configJSON.utf8))
        return MLXVLM.Gemma4(config)
    }

    /// Two solid-color 64×64 frames at t = 0s and t = 2s. The `.frames` video
    /// path samples min(estimated, provided) frames, so exactly these two
    /// survive sampling (and stamp "00:00" / "00:02" timestamps).
    private func twoFrameVideo() -> UserInput.Video {
        func frame(_ seconds: Double, _ red: CGFloat) -> UserInput.VideoFrame {
            let image = CIImage(color: CIColor(red: red, green: 0.4, blue: 0.6))
                .cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
            return UserInput.VideoFrame(
                frame: image,
                timeStamp: CMTime(seconds: seconds, preferredTimescale: 600))
        }
        return .frames([frame(0, 0.2), frame(2, 0.8)])
    }

    /// Run the REAL processor over the 2-frame fixture.
    private func prepareTwoFrameInput() async throws -> LMInput {
        let processor = try makeProcessor()
        let input = UserInput(
            chat: [.user("describe the clip", videos: [twoFrameVideo()])])
        return try await processor.prepare(input: input)
    }

    /// Maximal runs of `id` in `tokens`, in prompt order — the carving rule
    /// the provider's span builder applies to placeholder ids.
    private func placeholderRuns(in tokens: [Int], id: Int) -> [CBv2ImageSpan] {
        var spans: [CBv2ImageSpan] = []
        var index = 0
        while index < tokens.count {
            guard tokens[index] == id else {
                index += 1
                continue
            }
            let start = index
            while index < tokens.count, tokens[index] == id { index += 1 }
            spans.append(CBv2ImageSpan(tokenOffset: start, length: index - start))
        }
        return spans
    }

    private func promptTokens(of input: LMInput) -> [Int] {
        input.text.tokens.asArray(Int32.self).map(Int.init)
    }

    // MARK: (a)+(b) Processor structure — per-frame blocks, text between

    func testProcessorEmitsPerFrameBlocksSeparatedByText() async throws {
        let lmInput = try await prepareTwoFrameInput()
        let tokens = promptTokens(of: lmInput)

        // Two sampled frames survive as pixels (`.frames` caps sampling at
        // the provided count), resized to the VIDEO budget grid (32×32).
        let video = try XCTUnwrap(lmInput.video, "processor must emit video pixels")
        XCTAssertEqual(video.pixels.dim(0), 2, "exactly the two provided frames")
        XCTAssertEqual(video.frames?.count, 2)
        for frame in video.frames ?? [] {
            XCTAssertEqual(frame.h, 32, "video-budget resize target")
            XCTAssertEqual(frame.w, 32)
        }

        // Exactly two placeholder runs, each the derived per-frame count.
        let spans = placeholderRuns(
            in: tokens, id: Gemma4VideoStubTokenizer.videoMarkerId)
        XCTAssertEqual(spans.count, 2, "one span per sampled frame")
        for span in spans {
            XCTAssertEqual(
                span.length, Self.expectedPerFrameSoftTokens,
                "per-frame soft-token count must be the VIDEO budget, not the image budget")
        }

        // Delimiters: boi immediately before each run, eoi immediately after.
        for span in spans {
            XCTAssertEqual(tokens[span.tokenOffset - 1], Self.boiTokenId, "boi before the block")
            XCTAssertEqual(tokens[span.end], Self.eoiTokenId, "eoi after the block")
        }

        // The frame blocks are separated by ORDINARY text tokens: eoi, the
        // " 00:02 " timestamp (text band), then boi — at least 3 tokens, at
        // least one of them plain text.
        let gap = Array(tokens[spans[0].end ..< spans[1].tokenOffset])
        XCTAssertGreaterThanOrEqual(gap.count, 3, "frames must not be adjacent")
        XCTAssertTrue(
            gap.contains { $0 >= 20 && $0 < 80 },
            "the mm:ss timestamp must sit between the frame blocks as ordinary text")

        // Span carving is what the engine consumes: the plan keeps the two
        // frame blocks SEPARATE (adjacency is the only merge condition).
        XCTAssertEqual(
            CBv2MultimodalPlan.coalescedBlocks(spans: spans), spans,
            "text-separated frame spans must never coalesce")

        // Total placeholder count == what the vision features must fill.
        XCTAssertEqual(
            tokens.filter { $0 == Gemma4VideoStubTokenizer.videoMarkerId }.count,
            2 * Self.expectedPerFrameSoftTokens)
    }

    /// The two-block mask is a DIFFERENT attention shape than one merged
    /// block: under per-frame blocks, frame-1 queries cannot see frame-2
    /// keys and the timestamp text between them stays causal. This is the
    /// discrimination that makes the exactness test below able to detect
    /// wrongful coalescing.
    func testFrameSpansSeparatedByTextDoNotCoalesce() async throws {
        let lmInput = try await prepareTwoFrameInput()
        let tokens = promptTokens(of: lmInput)
        let spans = placeholderRuns(
            in: tokens, id: Gemma4VideoStubTokenizer.videoMarkerId)
        XCTAssertEqual(spans.count, 2)

        let T = tokens.count
        let correct = CBv2VisionReference.mask(length: T, window: nil, blocks: spans)
        let merged = CBv2VisionReference.mask(
            length: T, window: nil,
            blocks: [
                CBv2ImageSpan(
                    tokenOffset: spans[0].tokenOffset,
                    length: spans[1].end - spans[0].tokenOffset)
            ])
        eval(correct, merged)

        // Frame-1 query → frame-2 key: forbidden per-frame, allowed merged.
        let q = spans[0].tokenOffset
        let k = spans[1].tokenOffset
        XCTAssertFalse(
            correct[0, 0, q, k].item(Bool.self),
            "per-frame blocks must not attend forward across the timestamp text")
        XCTAssertTrue(
            merged[0, 0, q, k].item(Bool.self),
            "a merged block WOULD attend forward — the shapes are distinguishable")
    }

    // MARK: (c) Token-exact: CBv2 span prefill vs wrapper-semantics reference

    /// The video analog of PR#63's image-span exactness: the REAL EngineV2
    /// (chunked prefill, span snapping, bidirectional span masks) must match
    /// the wrapper-semantics full-sequence reference token-for-token on the
    /// processor-derived two-frame prompt. TinyTestModel weights; synthetic
    /// per-frame embeddings sized by the REAL processor structure.
    func testTwoFrameVideoEngineExactness() async throws {
        let lmInput = try await prepareTwoFrameInput()
        let prompt = promptTokens(of: lmInput)
        let spans = placeholderRuns(
            in: prompt, id: Gemma4VideoStubTokenizer.videoMarkerId)
        XCTAssertEqual(spans.count, 2)
        XCTAssertLessThan(
            prompt.max() ?? 0, 128, "processor ids must fit TinyTestModel's vocab")

        let model = TinyTestModel.make(seed: 0xC0FFEE)
        MLXRandom.seed(0xF1DE0)
        let images = spans.map { span -> CBv2VisionReference.Image in
            let embedding = MLXRandom.normal([1, span.length, model.config.hiddenSize]) * 0.7
            eval(embedding)
            return .init(span: span, embedding: embedding)
        }
        let decodeSteps = 10

        // Wrapper-semantics reference: full-sequence forward, causal(∧window)
        // ∨ bidirectional per (non-coalesced) block, greedy re-decode per
        // step — mimicking the MLXVLM `prepare` + generate loop.
        let reference = CBv2VisionReference.greedy(
            model: model, prompt: prompt, images: images, decodeSteps: decodeSteps)

        // CBv2 span prefill through the REAL engine. prefillChunkSize 8 puts
        // naive chunk boundaries inside both frame blocks, exercising the
        // span-snapping path the video shape rides in production.
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 5),
            schedulerConfig: CBv2SchedulerConfig(
                maxBatchedTokensPerStep: 256, prefillChunkSize: 8, maxWaiting: 4))
        let request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: decodeSteps,
            multimodal: CBv2VisionFixtures.input(images))
        let collected = await cbv2SchedCollect(try engine.submit(request))
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(
            collected.tokens, reference,
            "CBv2 video span prefill must match the wrapper-semantics reference token-for-token"
        )
        await engine.shutdown()
    }

    // MARK: (d) Gemma4 seam — video budget, count congruence, accessor

    func testPerVideoFrameFeaturesUseVideoBudgetAndMatchProcessorCounts() async throws {
        let lmInput = try await prepareTwoFrameInput()
        let tokens = promptTokens(of: lmInput)
        let video = try XCTUnwrap(lmInput.video)
        let vlm = try makeTinyVLM()

        // The NEW seam: per-frame features at the VIDEO budget.
        let perFrame = vlm.perVideoFrameVisionFeatures(
            pixels: video.pixels, frames: video.frames)
        XCTAssertEqual(perFrame.count, 2, "one feature array per sampled frame")
        for features in perFrame {
            XCTAssertEqual(
                features.shape, [1, Self.expectedPerFrameSoftTokens, 32],
                "per-frame features must use the VIDEO patch budget in the text hidden size")
        }

        // Regression tripwire for hardcoding `isVideo: false`: the image
        // seam on the SAME pixels yields the (larger) image budget.
        let asImages = vlm.perImageVisionFeatures(
            pixels: video.pixels, frames: video.frames)
        XCTAssertEqual(asImages.count, 2)
        for features in asImages {
            XCTAssertEqual(features.shape, [1, Self.imageSoftTokens, 32])
        }
        XCTAssertNotEqual(
            perFrame[0].dim(1), asImages[0].dim(1),
            "video and image budgets must be distinguishable on this fixture")
        XCTAssertEqual(
            perFrame[0].dtype, asImages[0].dtype,
            "both seams cast to the language model's embedding dtype")

        // Count congruence — the invariant CBv2 splicing relies on: the
        // seam's total soft tokens equal the processor's placeholder count.
        let placeholderCount = tokens.filter {
            $0 == Gemma4VideoStubTokenizer.videoMarkerId
        }.count
        XCTAssertEqual(
            perFrame.reduce(0) { $0 + $1.dim(1) }, placeholderCount,
            "seam features must fill exactly the processor's video placeholders")

        // The wrapper itself agrees end to end: `prepare` scatters the same
        // per-frame counts (maskedScatter THROWS on any mismatch) and runs
        // the bidirectional-vision text path over the tiny trunk.
        let result = try vlm.prepare(
            lmInput, cache: vlm.newCache(parameters: nil), windowSize: nil)
        guard case .logits(let output) = result else {
            return XCTFail("Gemma4.prepare must return logits for a video prompt")
        }
        eval(output.logits)
        XCTAssertEqual(output.logits.shape, [1, tokens.count, 512])

        // Accessors the provider carves spans with.
        XCTAssertEqual(
            vlm.videoPlaceholderTokenId, Gemma4VideoStubTokenizer.videoMarkerId)
        XCTAssertEqual(
            vlm.imagePlaceholderTokenId, Gemma4VideoStubTokenizer.imageMarkerId)
    }

    /// The wrapper owns one shared Gemma4TextModel: repeated access, direct
    /// VLM, CBv2, strict loading, and parameter accounting all meet at that
    /// same module boundary. Vision tensors remain outside text sanitization.
    func testSharedTextTowerOwnershipStrictLoadAndSanitizerBoundary() throws {
        let untiedJSON = Self.tinyVLMConfigJSON.replacingOccurrences(
            of: "\"tie_word_embeddings\": true",
            with: "\"tie_word_embeddings\": false")
        let vlm = try makeTinyVLM(configJSON: untiedJSON)
        let ownedTextModel = vlm.textModel
        XCTAssertTrue(ownedTextModel === vlm.textModel)

        let adapter = CBv2SteppableLanguageModelAdapter(ownedTextModel)
        XCTAssertTrue(
            adapter.supportsMultimodalPrefill,
            "the VLM-owned text tower must expose Gemma4 vision-span prefill to CBv2")

        let vlmParameters = vlm.parameters().flattened()
        let textParameters = ownedTextModel.parameters().flattened()
        let vlmKeys = Set(vlmParameters.map(\.0))
        XCTAssertTrue(vlmKeys.contains("language_model.model.embed_tokens.weight"))
        XCTAssertTrue(vlmKeys.contains("language_model.lm_head.weight"))
        for (key, _) in textParameters {
            XCTAssertTrue(
                vlmKeys.contains("language_model.\(key)"),
                "VLM parameter tree must own the shared text parameter \(key)")
        }
        XCTAssertEqual(
            vlmParameters.filter { $0.0.hasPrefix("language_model.") }
                .reduce(0) { $0 + $1.1.nbytes },
            textParameters.reduce(0) { $0 + $1.1.nbytes },
            "the VLM must account for exactly one text tower")

        let exactTree = Dictionary(uniqueKeysWithValues: vlmParameters)
        try vlm.update(
            parameters: ModuleParameters.unflattened(exactTree), verify: [.all])

        let sanitized = vlm.sanitize(weights: [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([512, 32]),
            "model.language_model.lm_head.weight": MLXArray.zeros([512, 32]),
            "model.language_model.layers.0.experts.gate_up_proj":
                MLXArray.zeros([4, 2]),
            "language_model.model.layers.0.experts.switch_glu.gate_proj.scales":
                MLXArray.zeros([2, 1]),
            "model.vision_tower.transformer.layers.0.self_attn.k_proj.linear.weight":
                MLXArray.zeros([2, 2]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNotNil(
            sanitized[
                "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"])
        XCTAssertNotNil(
            sanitized[
                "language_model.model.layers.0.experts.switch_glu.up_proj.weight"])
        XCTAssertNotNil(
            sanitized[
                "language_model.model.layers.0.experts.switch_glu.gate_proj.scales"])
        XCTAssertNotNil(
            sanitized[
                "vision_tower.transformer.layers.0.self_attn.k_proj.weight"])
        XCTAssertNil(
            sanitized["language_model.model.layers.0.experts.gate_up_proj"])
    }

    /// Gemma4 VLM adapters have always interpreted explicit keys relative to
    /// decoder layers. The shared-tower cutover must retain that root contract,
    /// while the nil-key default still reaches every Linear descendant.
    func testSharedTowerPreservesExplicitAndDefaultLoRARoots() throws {
        func module(named name: String, in layer: Module) -> Module? {
            layer.namedModules().first { $0.0 == name }?.1
        }

        let explicitVLM = try makeTinyVLM()
        XCTAssertEqual(explicitVLM.loraLayers.count, 2)
        for (root, decoder) in zip(
            explicitVLM.loraLayers, explicitVLM.textModel.decoderLayers)
        {
            XCTAssertTrue(
                root === decoder,
                "LoRA roots must be the decoder objects owned by the shared tower")
        }
        XCTAssertTrue(
            explicitVLM.loraLayers.allSatisfy {
                module(named: "self_attn.q_proj", in: $0) is Linear
                    && module(named: "mlp.gate_proj", in: $0) is Linear
            },
            "VLM LoRA roots must be canonical decoder layers")

        let explicitAdapter = try LoRAContainer.from(
            model: explicitVLM,
            configuration: LoRAConfiguration(
                numLayers: 1,
                loraParameters: .init(
                    rank: 2, scale: 1, keys: ["self_attn.q_proj"])))
        XCTAssertFalse(
            module(named: "self_attn.q_proj", in: explicitVLM.loraLayers[0])
                is LoRALinear)
        XCTAssertTrue(
            module(named: "self_attn.q_proj", in: explicitVLM.loraLayers[1])
                is LoRALinear)
        XCTAssertFalse(
            module(named: "mlp.gate_proj", in: explicitVLM.loraLayers[1])
                is LoRALinear)
        XCTAssertTrue(
            explicitAdapter.parameters.flattened().allSatisfy {
                $0.0.hasPrefix("language_model.model.layers.1.self_attn.q_proj.")
            },
            "explicit decoder-relative keys must not cross into the vision tower")

        let defaultVLM = try makeTinyVLM()
        let defaultKeys = Set(defaultVLM.loraDefaultKeys)
        XCTAssertTrue(defaultKeys.contains("self_attn.q_proj"))
        XCTAssertTrue(defaultKeys.contains("mlp.gate_proj"))
        XCTAssertFalse(defaultKeys.contains("q_proj"))

        let defaultAdapter = try LoRAContainer.from(
            model: defaultVLM,
            configuration: LoRAConfiguration(
                numLayers: 1,
                loraParameters: .init(rank: 2, scale: 1)))
        XCTAssertTrue(
            module(named: "self_attn.q_proj", in: defaultVLM.loraLayers[1])
                is LoRALinear)
        XCTAssertTrue(
            module(named: "mlp.gate_proj", in: defaultVLM.loraLayers[1])
                is LoRALinear)
        XCTAssertTrue(
            defaultAdapter.parameters.flattened().allSatisfy {
                $0.0.hasPrefix("language_model.model.layers.1.")
            },
            "default LoRA must remain inside the owned decoder-layer boundary")
    }

    /// Production VLM configs keep quantization beside `text_config`. Both
    /// accepted root spellings must overlay the canonical configuration that
    /// the shared target and automatic MTP policy consume.
    func testRootOnlyQuantizationDrivesCanonicalTextConfigurationAndMTPPolicy() throws {
        for rootKey in ["quantization", "quantization_config"] {
            var json = Self.tinyVLMConfigJSON.replacingOccurrences(
                of: "\"model_type\": \"gemma4\",",
                with:
                    "\"model_type\": \"gemma4\", \"\(rootKey)\": {\"bits\": 4, \"group_size\": 64},")
            json = json.replacingOccurrences(
                of: "\"hidden_size\": 32", with: "\"hidden_size\": 1536")
            json = json.replacingOccurrences(
                of: "\"num_hidden_layers\": 2", with: "\"num_hidden_layers\": 35")

            let config = try JSONDecoder().decode(
                MLXVLM.Gemma4Configuration.self, from: Data(json.utf8))
            XCTAssertEqual(config.quantization?.bits, 4)
            XCTAssertEqual(config.quantization?.groupSize, 64)
            XCTAssertEqual(config.textConfig.quantizationBits, 4)
            XCTAssertEqual(config.textConfig.quantizationGroupSize, 64)

            let policy = Gemma4MTPAutomaticPolicy.automatic(for: config.textConfig)
            XCTAssertEqual(policy.family, .e2b)
            XCTAssertFalse(policy.supportsBatchedMTP)
            XCTAssertEqual(
                policy.strategy(forBatchSize: 4), .singleStream(blockSize: 3))
        }
    }


    /// Nested Gemma4 VLM configs predate the canonical text-only defaults.
    /// Omitting fields must retain the former VLM topology rather than silently
    /// selecting the 4B text defaults.
    func testNestedVLMOmissionsUseFormerDefaults() throws {
        let json = """
            {
                "model_type": "gemma4",
                "text_config": {},
                "vision_config": {}
            }
            """
        let config = try JSONDecoder().decode(
            MLXVLM.Gemma4Configuration.self, from: Data(json.utf8))
        let text = config.textConfig

        XCTAssertEqual(text.hiddenSize, 2816)
        XCTAssertEqual(text.numHiddenLayers, 30)
        XCTAssertEqual(text.intermediateSize, 2112)
        XCTAssertEqual(text.numAttentionHeads, 16)
        XCTAssertEqual(text.numKeyValueHeads, 8)
        XCTAssertEqual(text.slidingWindow, 1024)
        XCTAssertEqual(text.layerTypes, Array(repeating: "sliding_attention", count: 30))
        XCTAssertEqual(text.finalLogitSoftcapping, 0)
        XCTAssertEqual(text.hiddenSizePerLayerInput, 0)
        XCTAssertEqual(text.vocabSizePerLayerInput, 0)
        XCTAssertEqual(text.numKvSharedLayers, 0)
        XCTAssertFalse(text.useDoubleWideMlp)
        XCTAssertEqual(text.fullPartialRotaryFactor, 0.25)
    }

    /// The deleted VLM DTO supported these switches, while the canonical
    /// tower intentionally does not. True values must fail at configuration
    /// decode instead of silently changing attention parameters or RoPE math.
    func testNestedVLMRejectsUnsupportedAttentionTopologyFlags() throws {
        for field in ["attention_bias", "rope_traditional"] {
            let json = """
                {
                    "model_type": "gemma4",
                    "text_config": { "\(field)": true },
                    "vision_config": {}
                }
                """
            XCTAssertThrowsError(
                try JSONDecoder().decode(
                    MLXVLM.Gemma4Configuration.self, from: Data(json.utf8)),
                "\(field)=true must fail explicitly")
        }
    }


    /// Alias-only quantization must reach BaseConfiguration and the real
    /// loadWeights quantization pass before strict parameter verification.
    func testQuantizationConfigAliasStrictLoadsQuantizedVLMWeights() throws {
        let json = Self.tinyVLMConfigJSON.replacingOccurrences(
            of: "\"model_type\": \"gemma4\",",
            with:
                "\"model_type\": \"gemma4\", \"quantization_config\": {\"bits\": 4, \"group_size\": 32},")
        let data = Data(json.utf8)
        let config = try JSONDecoder().decode(
            MLXVLM.Gemma4Configuration.self, from: data)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let quantization = try XCTUnwrap(baseConfig.perLayerQuantization)

        let source = MLXVLM.Gemma4(config)
        quantize(model: source, groupSize: 32, bits: 4) { path, module in
            guard path.hasPrefix("language_model"),
                let linear = module as? Linear
            else { return false }
            return linear.weight.dim(-1).isMultiple(of: 32)
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))

        let loaded = MLXVLM.Gemma4(config)
        try loadWeights(
            modelDirectory: directory, model: loaded,
            perLayerQuantization: quantization)

        let qProjection = loaded.textModel.decoderLayers[0].namedModules()
            .first { $0.0 == "self_attn.q_proj" }?.1
        XCTAssertTrue(qProjection is QuantizedLinear)
        XCTAssertTrue(
            loaded.parameters().flattened().contains {
                $0.0 == "language_model.model.layers.0.self_attn.q_proj.scales"
            })
    }

    /// Direct VLM generation must honor the requested full-attention bound;
    /// the no-bound path remains the canonical standard cache.
    func testDirectVLMFullAttentionCacheHonorsMaxKVSize() throws {
        let vlm = try makeTinyVLM()
        let bounded = vlm.newCache(
            parameters: GenerateParameters(maxKVSize: 7))
        XCTAssertEqual(bounded.count, 2)
        XCTAssertTrue(bounded[0] is RotatingKVCache)
        XCTAssertEqual(bounded[0].maxSize, 16)
        XCTAssertTrue(bounded[1] is RotatingKVCache)
        XCTAssertEqual(bounded[1].maxSize, 7)

        let unbounded = vlm.newCache(parameters: nil)
        XCTAssertTrue(unbounded[1] is StandardKVCache)
        XCTAssertNil(unbounded[1].maxSize)
    }
    /// `video_token_id` absent from config.json ⇒ the accessor reports the
    /// Gemma4 default 258_884 — the same fallback the processor uses when
    /// the tokenizer cannot resolve `<|video|>`, keeping model and processor
    /// in sync on real checkpoints.
    func testVideoPlaceholderTokenIdDefaultsTo258884() throws {
        var json = Self.tinyVLMConfigJSON
        json = json.replacingOccurrences(
            of: "\"video_token_id\": \(Gemma4VideoStubTokenizer.videoMarkerId),",
            with: "")
        XCTAssertFalse(json.contains("video_token_id"), "fixture must omit the key")
        let vlm = try makeTinyVLM(configJSON: json)
        XCTAssertEqual(vlm.videoPlaceholderTokenId, 258_884)
    }
}
