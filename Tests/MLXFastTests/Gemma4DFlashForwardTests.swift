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
// Forced onto `.cpu` and fixture-scale, but constructing the models still
// needs the built MLX runtime, which hosted CI does not have: box-only, every
// test gated behind MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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

    @Test func preprojectedDraftContextMatchesRawTargetHidden() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
            let raw = forward.targetHidden
            let projected = try drafter.projectTargetHidden(raw)
            let rawTokens = try drafter.draftBlock(
                bonus: 1,
                targetHidden: raw,
                cache: try drafter.makeCache(),
                blockSize: 4)
            let projectedTokens = try drafter.draftBlock(
                bonus: 1,
                projectedContext: projected,
                cache: try drafter.makeCache(),
                blockSize: 4)
            eval(rawTokens, projectedTokens)

            #expect(projected.shape == [1, 3, 16])
            #expect((rawTokens .== projectedTokens).all().item(Bool.self))
        }
    }

    /// `num_target_layers` must equal the TARGET's layer count. This is the
    /// check that makes the real A4B pairing (drafter 30 / target 30) a fact
    /// rather than a hope.
    @Test func refusesDrafterWithWrongTargetLayerCount() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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

    @Test func d7WidthPolicyPromotesAfterTwoFullC4Rounds() {
        var policy = Gemma4DFlashWidthPolicy(requestedDepth: 7)
        #expect(policy.currentDepth == 3)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
        #expect(policy.currentDepth == 3)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
        #expect(policy.currentDepth == 7)
    }

    @Test func d7WidthPolicyResetsAndDemotesWithoutTreatingTailAsFailure() {
        var policy = Gemma4DFlashWidthPolicy(requestedDepth: 7)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 2)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
        #expect(policy.currentDepth == 3)
        policy.record(roundBlockSize: 4, maxEmitCount: 4, committed: 4)
        #expect(policy.currentDepth == 7)
        policy.record(roundBlockSize: 7, maxEmitCount: 6, committed: 6)
        #expect(policy.currentDepth == 7)
        policy.record(roundBlockSize: 8, maxEmitCount: 7, committed: 7)
        #expect(policy.currentDepth == 7)
        policy.record(roundBlockSize: 8, maxEmitCount: 8, committed: 3)
        #expect(policy.currentDepth == 3)
    }

    @Test func nonD7WidthPoliciesRemainFixed() {
        for depth in [1, 2, 3, 4, 5, 6, 8, 11] {
            var policy = Gemma4DFlashWidthPolicy(requestedDepth: depth)
            policy.record(
                roundBlockSize: depth + 1,
                maxEmitCount: depth + 1,
                committed: depth + 1)
            policy.record(
                roundBlockSize: depth + 1,
                maxEmitCount: depth + 1,
                committed: 1)
            #expect(policy.currentDepth == depth)
        }
    }

    @Test func recurrenceRequiresThreeConfirmationsAndChoosesShortestPeriod() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: [1, 2, 3, 1, 2], accepted: 0, proposed: 3)
        #expect(policy.phase == .exactDFlashC4)
        policy.record(committedTokens: [3], accepted: 0, proposed: 3)
        #expect(policy.phase == .periodicExactWide(cycle: [1, 2, 3]))
        #expect(policy.makeDraftTokens(count: 8) == [1, 2, 3, 1, 2, 3, 1, 2])
    }

    @Test func exactAndWideVerifierWidthsRemainSeparateConstructionInputs() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        #expect(policy.verifierBlockSize(remaining: 127) == 4)
        #expect(policy.verifierBlockSize(remaining: 1) == 4)
        #expect(policy.verifierBlockSize(remaining: 2) == 4)

        policy.record(
            committedTokens: [1, 2, 3, 1, 2, 3],
            accepted: 0,
            proposed: 3)
        #expect(policy.verifierBlockSize(remaining: 1) == 16)
    }

    @Test func d15RecurrencePublishesOnlyItsConstructionCertifiedWidths() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth:
                RuntimeWorkerDFlashFreeRunSession.d15ExactC4VerifierWidth,
            verifierWidth:
                RuntimeWorkerDFlashFreeRunSession.d15FidelityVerifierWidth)

        for remaining in 1...512 {
            #expect(policy.verifierBlockSize(remaining: remaining) == 4)
        }

        policy.record(
            committedTokens: [1, 2, 3, 1, 2, 3],
            accepted: 0,
            proposed: 3)
        for remaining in 1...512 {
            #expect(
                policy.verifierBlockSize(remaining: remaining)
                    == RuntimeWorkerDFlashFreeRunSession.d15FidelityVerifierWidth)
        }
    }

    @Test func replayPlanPreservesFramesContextsAndCommittedFrontiers() {
        let frames = [
            Gemma4DFlashReplayFrame(
                bonus: 11,
                context: "first-wide",
                blockSize: 32,
                generatedTokenCountBeforeRound: 1),
            Gemma4DFlashReplayFrame(
                bonus: 22,
                context: "second-wide",
                blockSize: 32,
                generatedTokenCountBeforeRound: 33),
            Gemma4DFlashReplayFrame(
                bonus: 33,
                context: "rejected-current",
                blockSize: 32,
                generatedTokenCountBeforeRound: 65),
        ]

        let plan = gemma4DFlashReplayPlan(
            frames: frames,
            promptTokenCount: 1_024,
            action: .replayWideDrafterAndDemote)

        #expect(plan.map(\.bonus) == [11, 22, 33])
        #expect(
            plan.map(\.context)
                == ["first-wide", "second-wide", "rejected-current"])
        #expect(plan.map(\.blockSize) == [32, 32, 32])
        #expect(plan.map(\.committedDraftOffset) == [1_024, 1_056, 1_088])

        let terminalPlan = gemma4DFlashReplayPlan(
            frames: frames,
            promptTokenCount: 1_024,
            action: .none)
        #expect(terminalPlan.isEmpty)
    }

    @Test func recurrencePromotesPeriod18AfterTwentyThreeTokens() {
        let cycle = [
            100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 101, 102,
        ]
        let leadIn = [900, 901]
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 32)

        policy.record(
            committedTokens: leadIn + cycle + Array(cycle.prefix(2)),
            accepted: 0,
            proposed: 3)
        #expect(policy.phase == .exactDFlashC4)

        policy.record(
            committedTokens: [cycle[2]],
            accepted: 0,
            proposed: 3)

        let nextAlignedCycle = Array(cycle[3...] + cycle[..<3])
        #expect(policy.phase == .periodicExactWide(cycle: nextAlignedCycle))
        #expect(
            policy.makeDraftTokens(count: 5)
                == Array((cycle + cycle)[3 ..< 8]))
    }

    @Test func recurrenceRejectsThePeriod3TwoTokenFalsePositive() {
        let cycle = [
            100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 101, 102,
        ]
        let observed = [900, 901] + cycle + Array(cycle.prefix(3))
        #expect(Array(observed.suffix(2)) == Array(observed.dropLast(3).suffix(2)))
        #expect(Array(observed.suffix(3)) != Array(observed.dropLast(3).suffix(3)))

        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 32)
        policy.record(committedTokens: observed, accepted: 0, proposed: 3)

        let nextAlignedPeriod18 = Array(cycle[3...] + cycle[..<3])
        #expect(policy.phase == .periodicExactWide(cycle: nextAlignedPeriod18))
        #expect(policy.phase != .periodicExactWide(cycle: Array(observed.suffix(3))))
    }

    @Test func recurrenceRejectsPeriodOneAndPeriodsAboveThirtyTwo() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: Array(repeating: 7, count: 64),
            accepted: 0,
            proposed: 3)
        #expect(policy.phase == .exactDFlashC4)
        let long = Array(0 ..< 33)
        policy.record(committedTokens: long + long, accepted: 0, proposed: 3)
        #expect(policy.phase == .exactDFlashC4)
    }

    @Test func recurrenceDoesNotMistakeADuplicateCycleTailForPeriodOne() {
        let cycle = [1, 2, 3, 3]
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: cycle + cycle,
            accepted: 0,
            proposed: 3)

        #expect(policy.phase == .periodicExactWide(cycle: cycle))
    }

    @Test func wideRejectionDemotesNextRoundAndClearsTheCycle() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: [4, 5, 6, 4, 5, 6],
            accepted: 0,
            proposed: 3)
        #expect(policy.phase == .periodicExactWide(cycle: [4, 5, 6]))
        let action = policy.record(
            committedTokens: [4, 99],
            accepted: 1,
            proposed: 15,
            isTerminalRound: false)
        #expect(policy.phase == .exactDFlashC4)
        #expect(policy.makeDraftTokens(count: 3) == nil)
        #expect(action == .replayWideDrafterAndDemote)
    }

    @Test func terminalWideTailSkipsReplayAndDemotion() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: [4, 5, 6, 4, 5, 6],
            accepted: 0,
            proposed: 3)

        let action = policy.record(
            committedTokens: [4],
            accepted: 0,
            proposed: 15,
            isTerminalRound: true)

        #expect(action == .none)
        #expect(policy.phase == .periodicExactWide(cycle: [4, 5, 6]))
    }

    @Test func wideTailKeepsItsConstructionBoundPhysicalWidth() {
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 16)
        policy.record(
            committedTokens: [4, 5, 6, 4, 5, 6],
            accepted: 0,
            proposed: 3)
        #expect(policy.verifierBlockSize(remaining: 1) == 16)
        #expect(policy.verifierBlockSize(remaining: 127) == 16)
    }

    @Test func fullyAcceptedWideRoundAdvancesThePeriod18CycleOffset() {
        let cycle = Array(100 ..< 118)
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 32)
        policy.record(
            committedTokens: cycle + cycle,
            accepted: 0,
            proposed: 3)
        #expect(policy.makeDraftTokens(count: 4) == [100, 101, 102, 103])

        let committed = Array((cycle + cycle).prefix(32))
        policy.record(committedTokens: committed, accepted: 31, proposed: 31)

        #expect(policy.makeDraftTokens(count: 4) == [114, 115, 116, 117])
    }

    @Test func consecutiveWideRoundsContinueFromTheUpdatedCycleOffset() {
        let cycle = Array(100 ..< 118)
        var policy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: 4, verifierWidth: 32)
        policy.record(
            committedTokens: cycle + cycle,
            accepted: 0,
            proposed: 3)
        policy.record(
            committedTokens: Array((cycle + cycle).prefix(32)),
            accepted: 31,
            proposed: 31)
        let fromFourteen = Array(cycle[14...] + cycle + cycle)
        policy.record(
            committedTokens: Array(fromFourteen.prefix(32)),
            accepted: 31,
            proposed: 31)

        #expect(policy.makeDraftTokens(count: 4) == [110, 111, 112, 113])
    }

    @Test func d15SessionRoutesOnceIntoDedicatedRecurrenceLoop() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)

        let runStart = try #require(source.range(of: "func run(targetN: Int)"))
        let standardStart = try #require(
            source.range(
                of: "func runStandardRounds(",
                range: runStart.upperBound ..< source.endIndex))
        let run = source[runStart.lowerBound ..< standardStart.lowerBound]
        #expect(run.contains("if depth == 15"))
        #expect(run.contains("return try runD15ProposalRounds("))

        #expect(
            source.contains(
                "let maximumPhysicalVerifierWidth =\n"
                    + "            depth == 15 ? Self.d15FidelityVerifierWidth : depth + 1"))
        #expect(
            source.contains(
                "self.maximumPhysicalVerifierWidth = maximumPhysicalVerifierWidth"))

        let recurrenceStart = try #require(
            source.range(of: "func runD15ProposalRounds("))
        let recurrenceLoop = source[recurrenceStart.lowerBound ..< source.endIndex]
        #expect(recurrenceLoop.contains("let phase = proposalPhasePolicy.phase"))
        #expect(recurrenceLoop.contains("switch phase"))
        #expect(recurrenceLoop.contains("runDFlashGreedyRound("))
        #expect(recurrenceLoop.contains("runDFlashGreedyProposalRound("))
        #expect(
            recurrenceLoop.contains(
                "let isTerminalRound = round.tokens.count == remaining"))
        #expect(recurrenceLoop.contains("wideReplayFrames.append(wideReplayFrame)"))
        #expect(recurrenceLoop.contains("case .replayWideDrafterAndDemote:"))
        #expect(
            recurrenceLoop.contains(
                "try replaySkippedWideDrafterBlocks(replayPlan)"))
        #expect(recurrenceLoop.contains("let replayPlan = gemma4DFlashReplayPlan("))
        #expect(recurrenceLoop.contains("wideReplayFrames.removeAll("))
        #expect(
            recurrenceLoop.contains(
                "private func replaySkippedWideDrafterBlocks("))
        #expect(recurrenceLoop.contains("try drafter.draftBlock("))
        #expect(recurrenceLoop.contains("eval(replayedDraftTokens)"))
        #expect(recurrenceLoop.contains("trimPromptCache("))
        let stopBoundary = try #require(
            recurrenceLoop.range(
                of: "if let stopIndex = round.tokens.firstIndex"))
        let bonusMutation = try #require(
            recurrenceLoop.range(of: "bonus = round.bonus"))
        let frameMutation = try #require(
            recurrenceLoop.range(
                of: "wideReplayFrames.append(wideReplayFrame)"))
        let policyMutation = try #require(
            recurrenceLoop.range(of: "proposalPhasePolicy.record("))
        #expect(stopBoundary.lowerBound < bonusMutation.lowerBound)
        #expect(stopBoundary.lowerBound < frameMutation.lowerBound)
        #expect(stopBoundary.lowerBound < policyMutation.lowerBound)
        let stopBlock = recurrenceLoop[
            stopBoundary.lowerBound ..< bonusMutation.lowerBound]
        #expect(stopBlock.contains("let upToStop = Array(round.tokens[...stopIndex])"))
        #expect(stopBlock.contains("committedTokens: upToStop"))
        #expect(stopBlock.contains("generatedTokenCount += upToStop.count"))
        let replayStart = try #require(
            recurrenceLoop.range(
                of: "private func replaySkippedWideDrafterBlocks("))
        let replayHelper = recurrenceLoop[replayStart.lowerBound...]
        let replayBeforeTrim = try #require(
            replayHelper.range(of: "try drafter.draftBlock("))
        let trimAfterReplay = try #require(
            replayHelper.range(of: "trimPromptCache("))
        #expect(replayBeforeTrim.lowerBound < trimAfterReplay.lowerBound)
        #expect(
            recurrenceLoop.contains(
                "proposalPhasePolicy.record(\n"
                    + "                committedTokens: round.tokens,\n"
                    + "                accepted: round.accepted,\n"
                    + "                proposed: proposed,\n"
                    + "                isTerminalRound: isTerminalRound)"))
        for forbidden in [
            "structuralC4",
            "period2Wide",
            "try?",
            "fallback",
        ] {
            #expect(!recurrenceLoop.contains(forbidden))
        }
    }

    /// The dflash `effective_spec` echo and the round loop read the SAME
    /// resolver, so an unclamped default cannot be echoed. A drafter with a
    /// trained block of 4 caps depth at 3 regardless of what a caller asks.
    @Test func depthCeilingComesFromTheBoundDrafterBlock() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
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

    /// The selected submission ceiling applies on top of the drafter and
    /// engine ceilings. The target-verified D15 period-2 lane uses the trained
    /// block's full width after its runtime proposal route is installed.
    @Test func depthCeilingUsesTheSelectedFidelityPassingDepth() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let drafter = DFlashDraftModel(config: try tinyDFlashConfig(blockSize: 64))
        #expect(gemma4DFlashMaxDepth(for: drafter) == 15)
    }
}
