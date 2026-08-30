// Copyright © 2026 Apple Inc.

// ContinuousBatchingV2 MTP model seam tests (weight-free, tiny random-init
// fixtures):
//  - capture-layer indices (`Gemma4TextModel.cbv2MTPCaptureLayers`),
//  - `cbv2ForwardWithHidden` logits parity vs the plain CBv2 forward
//    ([1, chunk] prefill AND rectangular [B, 1] decode) + adapter plumbing,
//  - drafter chain parity: `Gemma4CBv2MTPDrafter` over CBv2 snapshot
//    captures produces the exact draft ids of the v1 greedy chain
//    (`runGemma4MTPGreedyRound` semantics over legacy caches),
//  - batch invariance: each row of a padded/masked B=2 round drafts the
//    same ids as its own solo B=1 round.

import Foundation
import MLX
@testable import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

@Suite("CBv2MTPModelSeam", .serialized)
struct CBv2MTPModelSeamTests {

    // MARK: - Config fixtures

    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 64

    @Test func slidingMaskUsesStrictAbsoluteWindowBoundary() {
        func row(start: Int, anchor: Int, count: Int) -> CBv2MTPRowCapture {
            let kv = MLXArray.zeros([1, 1, count, 4], dtype: .float32)
            return CBv2MTPRowCapture(
                fullKeys: kv, fullValues: kv,
                slidingKeys: kv, slidingValues: kv,
                slidingStart: start, anchor: anchor)
        }

        // At anchor == window, absolute key 0 lies exactly on the excluded
        // lower boundary while key 1 is visible.
        let boundary = Gemma4CBv2MTPDrafter.slidingMask(
            rows: [row(start: 0, anchor: slidingWindow, count: slidingWindow)],
            tMax: slidingWindow, window: slidingWindow, dtype: .float32)
        let boundaryMask = boundary?.reshaped([-1]).asArray(Float.self)
        #expect(boundaryMask?.first == -Float.infinity)
        #expect(boundaryMask?[1] == 0)

        // The same rule holds after the rotating window advances: the oldest
        // retained absolute key remains exactly `anchor-window`.
        let advanced = Gemma4CBv2MTPDrafter.slidingMask(
            rows: [row(start: 1, anchor: slidingWindow + 1, count: slidingWindow)],
            tMax: slidingWindow, window: slidingWindow, dtype: .float32)
        let advancedMask = advanced?.reshaped([-1]).asArray(Float.self)
        #expect(advancedMask?.first == -Float.infinity)
        #expect(advancedMask?[1] == 0)

        // Before the first wrap every retained key is strictly in range, so
        // the allocation-free no-mask path remains available.
        #expect(
            Gemma4CBv2MTPDrafter.slidingMask(
                rows: [row(start: 0, anchor: slidingWindow - 1, count: slidingWindow - 1)],
                tMax: slidingWindow - 1, window: slidingWindow, dtype: .float32) == nil)
    }

    @Test func cachedDrafterRoPEPreservesSingletonQueryAxis() {
        let hidden = MLXArray((0 ..< 128).map(Float.init)).reshaped([2, 1, 64])
        let table = DrafterRoPETable(
            cos: MLXArray.ones([2, 16], dtype: .float32),
            sin: MLXArray.zeros([2, 16], dtype: .float32),
            dims: 32,
            startPosition: 11,
            windowAhead: 2,
            base: 10_000)

        let rotated = Gemma4CBv2MTPDrafter.applyCachedDrafterRoPE(
            hidden: hidden,
            table: table,
            positionOffset: .batch(MLXArray([Int32(11), Int32(12)])))
        eval(rotated)

        #expect(rotated.shape == [2, 1, 64])
        #expect(allClose(rotated, hidden, rtol: 0, atol: 0).item(Bool.self))
    }

    /// 6-layer target, last 2 KV-shared. Non-shared prefix layer types
    /// [sliding, full, full, sliding] ⇒ capture layers full=2, sliding=3.
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

    /// Full-attention-only target: no non-shared sliding layer, so the MTP
    /// capture geometry is unavailable.
    private func fullOnlyConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": ["full_attention", "full_attention"],
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

    /// 2-layer fully-KV-shared drafter matching the target's hidden/vocab.
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

    // MARK: - CBv2 cache helpers

    private struct CBv2Rig {
        let kinds: [CBv2LayerKind]
        let backend: CBv2ContiguousKVBackend
        let bank: CBv2LayerCacheBank

        init(config: Gemma4TextConfiguration) {
            self.kinds = config.cbv2LayerKinds
            self.backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
            self.bank = CBv2LayerCacheBank(layerKinds: kinds)
        }

        func newRow(promptLength: Int) throws -> [CBv2SequenceKV?] {
            try backend.makeSequenceState(
                layerKinds: kinds, promptLength: promptLength,
                maxLength: promptLength + 32)
        }

        func attendingCaches(_ rows: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
            bank.layerCaches(rowStates: rows)
        }

        func kvCaches(_ rows: [[CBv2SequenceKV?]]) -> [KVCache] {
            attendingCaches(rows).map { cache in
                guard let kv = cache as? KVCache else {
                    fatalError("CBv2 layer cache \(type(of: cache)) must conform to KVCache")
                }
                return kv
            }
        }
    }

    private func tokens2d(_ prompt: [Int]) -> MLXArray {
        MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
    }

    private func greedyToken(_ logits: MLXArray) -> Int {
        Int(logits[0..., -1, 0...].asType(.float32).argMax(axis: -1).item(Int32.self))
    }

    private func lastHiddenSlice(_ lastHidden: MLXArray) -> MLXArray {
        lastHidden[0..., -1 ..< lastHidden.dim(1), 0...]
    }

    /// Build one row's `CBv2MTPRowCapture` from its CBv2 sequence state
    /// (snapshot views at the model's two capture layers).
    private func rowCapture(
        rowState: [CBv2SequenceKV?], captureLayers: CBv2MTPCaptureLayers
    ) throws -> CBv2MTPRowCapture {
        let fullRow = try #require(rowState[captureLayers.full])
        let slidingRow = try #require(rowState[captureLayers.sliding])
        let fullSnap = fullRow.snapshot()
        let slidingSnap = slidingRow.snapshot()
        return CBv2MTPRowCapture(
            fullKeys: fullSnap.keys, fullValues: fullSnap.values,
            slidingKeys: slidingSnap.keys, slidingValues: slidingSnap.values,
            slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
            anchor: fullRow.absoluteOffset)
    }

    /// Chain `steps` draft steps through the CBv2 drafter adapter, returning
    /// per-row draft ids. Seeds are each row's bonus token; `hidden` is the
    /// stacked [B, 1, H] carry.
    private func chainDrafts(
        _ cbDrafter: Gemma4CBv2MTPDrafter, prepared: CBv2MTPPreparedCapture,
        seeds: [Int], hidden: MLXArray, steps: Int
    ) -> [[Int]] {
        let batch = seeds.count
        var tok = MLXArray(seeds.map { Int32($0) }).reshaped([batch, 1])
        var carry = hidden
        var drafts = [[Int]](repeating: [], count: batch)
        for _ in 0 ..< steps {
            let (next, newHidden) = cbDrafter.draftStep(
                tokens: tok, hidden: carry, prepared: prepared)
            eval(next, newHidden)
            for (row, id) in next.asArray(Int32.self).enumerated() {
                drafts[row].append(Int(id))
            }
            tok = next.reshaped([batch, 1])
            carry = newHidden
        }
        return drafts
    }

    // MARK: - (a) Capture indices

    @Test func captureLayerIndices() throws {
        MLXRandom.seed(0x517)
        let model = Gemma4TextModel(try targetConfig())
        let layers = try #require(model.cbv2MTPCaptureLayers)
        #expect(layers.full == 2)
        #expect(layers.sliding == 3)
    }

    @Test func captureLayersNilWithoutSlidingLayer() throws {
        MLXRandom.seed(0x517)
        let model = Gemma4TextModel(try fullOnlyConfig())
        #expect(model.cbv2MTPCaptureLayers == nil)
    }

    @Test func adapterAnswersCaptureLayersAtRuntime() throws {
        MLXRandom.seed(0x517)
        let gemma = Gemma4TextModel(try targetConfig())
        let gemmaAdapter = CBv2SteppableLanguageModelAdapter(gemma)
        #expect(gemmaAdapter.mtpCaptureLayers == gemma.cbv2MTPCaptureLayers)
        #expect(gemmaAdapter.mtpCaptureLayers != nil)
        #expect(gemmaAdapter.mtpTargetIdentity == ObjectIdentifier(gemma))

        // Non-CBv2MTPForwardable models answer nil (no trap).
        let tinyAdapter = CBv2SteppableLanguageModelAdapter(TinyTestModel.make())
        #expect(tinyAdapter.mtpCaptureLayers == nil)
        #expect(tinyAdapter.mtpTargetIdentity == nil)
    }

    @Test func driverRejectsDrafterBoundToDifferentTargetInstance() throws {
        MLXRandom.seed(0x518)
        let boundTarget = Gemma4TextModel(try targetConfig())
        let engineTarget = Gemma4TextModel(try targetConfig())
        let assistant = try Gemma4AssistantDraftModel(config: drafterConfig())
        let drafter = try Gemma4CBv2MTPDrafter(
            drafter: assistant, target: boundTarget)

        let driver = CBv2MTPRoundDriver.build(
            model: CBv2SteppableLanguageModelAdapter(engineTarget),
            drafter: drafter,
            config: CBv2MTPConfig(enabled: true, fixedDraftTokens: 2))

        #expect(driver == nil)
    }

    // MARK: - (b) Logits parity: cbv2ForwardWithHidden vs plain forward

    @Test func forwardWithHiddenLogitsParity() throws {
        MLXRandom.seed(0xBEEF)
        let config = try targetConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let rig = CBv2Rig(config: config)
        let adapter = CBv2SteppableLanguageModelAdapter(model)

        let prompts = [
            makePromptTokens(length: 12, seed: 11, vocabSize: vocabSize),
            makePromptTokens(length: 20, seed: 12, vocabSize: vocabSize),
        ]

        // [1, chunk] prefill: identical prompts through twin rows, one via
        // the plain forward, one via cbv2ForwardWithHidden.
        var plainRows: [[CBv2SequenceKV?]] = []
        var hiddenRows: [[CBv2SequenceKV?]] = []
        for (i, prompt) in prompts.enumerated() {
            let tokens = tokens2d(prompt)

            let plainRow = try rig.newRow(promptLength: prompt.count)
            let plainLogits = model(tokens, cache: rig.kvCaches([plainRow]))

            let hiddenRow = try rig.newRow(promptLength: prompt.count)
            let (logits, lastHidden) = model.cbv2ForwardWithHidden(
                tokens, caches: rig.kvCaches([hiddenRow]))

            eval(plainLogits, logits, lastHidden)
            #expect(logits.shape == [1, prompt.count, vocabSize], "prefill row \(i)")
            #expect(lastHidden.shape == [1, prompt.count, hiddenSize], "prefill row \(i)")
            #expect(
                allClose(plainLogits, logits).item(Bool.self),
                "prefill logits diverged for row \(i)")

            plainRows.append(plainRow)
            hiddenRows.append(hiddenRow)
        }

        // Rectangular [B, 1] decode over both rows, hidden leg through the
        // adapter's forwardWithHidden (exercises the cache conversion).
        let step = MLXArray([Int32(3), Int32(7)]).reshaped([2, 1])
        let plainLogits = model(step, cache: rig.kvCaches(plainRows))
        let (logits, lastHidden) = adapter.forwardWithHidden(
            tokens: step, caches: rig.attendingCaches(hiddenRows))
        eval(plainLogits, logits, lastHidden)
        #expect(logits.shape == [2, 1, vocabSize])
        #expect(lastHidden.shape == [2, 1, hiddenSize])
        #expect(allClose(plainLogits, logits).item(Bool.self), "decode logits diverged")
    }

    // MARK: - (c) Drafter chain parity vs the v1 greedy chain

    @Test func drafterChainParityVsV1() throws {
        MLXRandom.seed(0xD0D0)
        let config = try targetConfig()
        let target = Gemma4TextModel(config)
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)

        // Prompt < both windows so legacy RotatingKVCache and CBv2 windowed
        // eviction retain identical content AND the v1 sliding mask is .none.
        let prompt = makePromptTokens(length: 24, seed: 21, vocabSize: vocabSize)
        let k = 3

        // --- v1 leg: legacy caches + the runGemma4MTPGreedyRound chain. ---
        let legacyCache = target.newCache(parameters: nil)
        let prefill = target.forwardForMTP(tokens2d(prompt), cache: legacyCache)
        let v1Bonus = greedyToken(prefill.logits)
        let v1SharedKV = prefill.capturedSharedKV

        // Mirror runGemma4MTPGreedyRound's draft chain exactly (that round
        // loop does not expose the raw draft ids).
        let driveOffset = legacyCache[0].offset
        let positionOffset = Gemma4.PositionOffset.scalar(driveOffset)
        var v1Hidden = lastHiddenSlice(prefill.lastHidden)
        let masks = drafter.makeMasks(
            queryLen: 1, sharedKV: v1SharedKV,
            positionOffset: positionOffset, dtype: v1Hidden.dtype)
        var v1Tok = MLXArray([Int32(v1Bonus)])[.newAxis, .ellipsis]
        var v1Drafts: [Int] = []
        for _ in 0 ..< k {
            let inputsEmbeds = concatenated(
                [target.embedTokensForDrafter(v1Tok), v1Hidden], axis: -1)
            let (newHidden, logits) = drafter(
                inputsEmbeds: inputsEmbeds, sharedKV: v1SharedKV,
                positionOffset: positionOffset, masks: masks)
            let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
            eval(sampled)
            v1Drafts.append(Int(sampled.item(Int32.self)))
            v1Tok = sampled[.newAxis, .ellipsis]
            v1Hidden = newHidden
        }

        // --- CBv2 leg: same prompt through CBv2 caches + snapshot capture. ---
        let rig = CBv2Rig(config: config)
        let rowState = try rig.newRow(promptLength: prompt.count)
        let (logits, lastHidden) = target.cbv2ForwardWithHidden(
            tokens2d(prompt), caches: rig.kvCaches([rowState]))
        let cbBonus = greedyToken(logits)
        #expect(cbBonus == v1Bonus, "prefill bonus diverged before any drafting")

        let captureLayers = try #require(target.cbv2MTPCaptureLayers)
        let capture = try rowCapture(rowState: rowState, captureLayers: captureLayers)
        #expect(capture.anchor == driveOffset)
        #expect(capture.slidingStart == 0)

        let cbDrafter = try Gemma4CBv2MTPDrafter(drafter: drafter, target: target)
        let prepared = cbDrafter.prepare(rows: [capture])
        let cbDrafts = chainDrafts(
            cbDrafter, prepared: prepared, seeds: [cbBonus],
            hidden: lastHiddenSlice(lastHidden), steps: k)

        #expect(
            cbDrafts[0] == v1Drafts,
            "CBv2 draft chain diverged from v1: cbv2=\(cbDrafts[0]) v1=\(v1Drafts)")
    }

    // MARK: - (d) Batch invariance (padding + masks)

    @Test func batchedDraftsMatchSoloDrafts() throws {
        MLXRandom.seed(0xFACE)
        let config = try targetConfig()
        let target = Gemma4TextModel(config)
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)

        let rig = CBv2Rig(config: config)
        let captureLayers = try #require(target.cbv2MTPCaptureLayers)
        let cbDrafter = try Gemma4CBv2MTPDrafter(drafter: drafter, target: target)
        let k = 3

        // Different prompt lengths force real padding + per-row masks.
        let prompts = [
            makePromptTokens(length: 9, seed: 31, vocabSize: vocabSize),
            makePromptTokens(length: 17, seed: 32, vocabSize: vocabSize),
        ]

        var captures: [CBv2MTPRowCapture] = []
        var bonuses: [Int] = []
        var hiddens: [MLXArray] = []
        var soloDrafts: [[Int]] = []
        for prompt in prompts {
            let rowState = try rig.newRow(promptLength: prompt.count)
            let (logits, lastHidden) = target.cbv2ForwardWithHidden(
                tokens2d(prompt), caches: rig.kvCaches([rowState]))
            let bonus = greedyToken(logits)
            let hidden = lastHiddenSlice(lastHidden)
            let capture = try rowCapture(rowState: rowState, captureLayers: captureLayers)

            let prepared = cbDrafter.prepare(rows: [capture])
            soloDrafts.append(
                chainDrafts(
                    cbDrafter, prepared: prepared, seeds: [bonus],
                    hidden: hidden, steps: k)[0])

            captures.append(capture)
            bonuses.append(bonus)
            hiddens.append(hidden)
        }
        #expect(captures[0].anchor != captures[1].anchor)

        let prepared = cbDrafter.prepare(rows: captures)
        let batchDrafts = chainDrafts(
            cbDrafter, prepared: prepared, seeds: bonuses,
            hidden: concatenated(hiddens, axis: 0), steps: k)

        for row in 0 ..< prompts.count {
            #expect(
                batchDrafts[row] == soloDrafts[row],
                "row \(row) batch drafts \(batchDrafts[row]) != solo \(soloDrafts[row])")
        }
    }

    // MARK: - (e) Official target-hidden and accepted-index semantics

    @Test func targetConditioningHiddenIsPreFinalNorm() throws {
        MLXRandom.seed(0xA55157)
        let config = try targetConfig()
        let target = Gemma4TextModel(config)
        eval(target)
        let rig = CBv2Rig(config: config)
        let prompt = makePromptTokens(length: 11, seed: 77, vocabSize: vocabSize)
        let tokens = tokens2d(prompt)

        let referenceRow = try rig.newRow(promptLength: prompt.count)
        let (postNorm, preNorm) = target.model.callCapturingPreNorm(
            tokens, cache: rig.kvCaches([referenceRow]))
        let seamRow = try rig.newRow(promptLength: prompt.count)
        let (_, conditioningHidden) = target.cbv2ForwardWithHidden(
            tokens, caches: rig.kvCaches([seamRow]))
        eval(postNorm, preNorm, conditioningHidden)

        #expect(
            allClose(conditioningHidden, preNorm, rtol: 0, atol: 0).item(Bool.self),
            "the assistant conditioning slot must be the decoder output before model.norm")
        #expect(
            !allClose(conditioningHidden, postNorm, rtol: 0, atol: 0).item(Bool.self),
            "a post-final-norm hidden would silently change official assistant inputs")
    }

    @Test func lastAcceptedTokenUsesAcceptedDraftHiddenColumn() {
        let depth = 3
        let hidden = MLXArray((0 ..< 8).map(Int32.init)).reshaped([2, 4, 1])
        for accepted in 0 ... depth {
            let column = CBv2MTPHiddenIndex.carryColumn(
                targetOutputIndex: accepted, draftDepth: depth)
            #expect(column == accepted)
            let selected = hidden[0, column, 0].item(Int32.self)
            #expect(selected == Int32(accepted))
        }
        // The second row proves the same column is selected per row rather
        // than flattening across the rectangular batch.
        let column = CBv2MTPHiddenIndex.carryColumn(
            targetOutputIndex: 2, draftDepth: depth)
        #expect(hidden[1, column, 0].item(Int32.self) == 6)
    }
}
