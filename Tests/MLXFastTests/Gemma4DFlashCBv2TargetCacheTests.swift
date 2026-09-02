import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

@Suite("Gemma 4 DFlash CBv2 target cache", .serialized)
struct Gemma4DFlashCBv2TargetCacheTests {
    private enum ConstructionFailure: Error {
        case expected
    }

    private final class CompiledBankIdentityToken {}

    private var runtimeTestsEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 6,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 8,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 4,
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

    @Test func rowStateOwnershipReleasesExactlyOnceOnlyOnConstructionFailure() throws {
        var failureReleaseCount = 0
        do {
            _ = try withDFlashCBv2RowStateOwnership(
                17,
                release: { state in
                    #expect(state == 17)
                    failureReleaseCount += 1
                },
                prepare: { _ in
                    throw ConstructionFailure.expected
                }) as Int
            Issue.record("expected construction failure")
        } catch ConstructionFailure.expected {
            // Expected: the helper released the not-yet-published state.
        }
        #expect(failureReleaseCount == 1)

        var successReleaseCount = 0
        let prepared = withDFlashCBv2RowStateOwnership(
            23,
            release: { _ in successReleaseCount += 1 },
            prepare: { $0 + 1 })
        #expect(prepared == 24)
        #expect(successReleaseCount == 0)
    }

    @Test func d15CapacityIncludesTheFullUnalignedFinalFidelityRectangle() {
        let promptTokenCount = 7
        let maximumTokens =
            RuntimeWorkerDFlashFreeRunSession.targetCacheMaximumTokens(
                promptTokenCount: promptTokenCount,
                depth: 15)
        let expected = promptTokenCount
            + MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
            + RuntimeWorkerDFlashFreeRunSession.d15FidelityVerifierWidth - 1
        #expect(maximumTokens == expected)

        // With one requested token left, the periodic route retains its
        // construction width. Its exclusive end lands on the allocation.
        let finalVerifierStart = promptTokenCount
            + MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens - 1
        #expect(
            finalVerifierStart
                + RuntimeWorkerDFlashFreeRunSession.d15FidelityVerifierWidth
                == maximumTokens)

        let standardMaximumTokens =
            RuntimeWorkerDFlashFreeRunSession.targetCacheMaximumTokens(
                promptTokenCount: promptTokenCount,
                depth: 14)
        #expect(
            standardMaximumTokens == promptTokenCount
                + MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens + 1)
    }

    @Test func c16IsInstalledAcrossTheImmutableVerifierSurface() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        func source(_ path: String) throws -> String {
            try String(
                contentsOf: repositoryRoot.appendingPathComponent(path),
                encoding: .utf8)
        }

        let model = try source(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        #expect(model.contains(
            "private static let certifiedColumns = [2, 3, 4, 8, 16]"))
        #expect(model.contains("let shapes = [2, 3, 4, 8, 16].map"))

        let route = try source(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "MTP/Gemma4MTPVerifierRoute.swift")
        #expect(route.contains(
            "public static let certifiedColumns: Set<Int> = [2, 3, 4, 8, 16]"))

        let head = try source(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "Gemma4MMAQuantizedGEMV.swift")
        #expect(head.contains("[2, 3, 4, 8, 16].contains(columns)"))

        let glue = try source(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                + "Gemma4PrefillGlueV1.swift")
        #expect(glue.contains(
            "private static let verifierColumns: Set<Int> = [2, 3, 4, 8, 16]"))
    }

    @Test func inheritedCompiledVerifierBankPreservesObjectIdentity() {
        let inherited = CompiledBankIdentityToken()
        let preserved: CompiledBankIdentityToken? =
            RuntimeWorkerDFlashFreeRunSession.inheritedCompiledVerifierBank(
                inherited)
        #expect(preserved === inherited)
    }

    @Test func d15ConstructsCertifiedCBv2AndEnablesItAfterPrefill() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)

        #expect(
            source.contains(
                "certifiedRectangularVerification: depth == 15"))
        #expect(!source.contains("target.newCache(parameters: nil)"))
        #expect(
            source.contains(
                "private let certifiedRectangularVerification: Bool"))
        #expect(
            source.contains(
                "certifiedRectangularVerification: Bool = false"))

        let cacheStart = try #require(
            source.range(of: "final class Gemma4DFlashCBv2TargetCache"))
        let compiledStart = try #require(
            source.range(
                of: "final class Gemma4DFlashCompiledVerifierBank",
                range: cacheStart.upperBound ..< source.endIndex))
        let cache = source[cacheStart.lowerBound ..< compiledStart.lowerBound]
        let ownershipStart = try #require(
            cache.range(of: "let prepared = try withDFlashCBv2RowStateOwnership("))
        let supportCheck = try #require(
            cache.range(of: "bank.supportsCertifiedMTPRectangularVerification"))
        let layerCacheBinding = try #require(
            cache.range(
                of: "let layerCaches = bank.layerCaches",
                range: supportCheck.upperBound ..< cache.endIndex))
        let unsupportedCertification =
            cache[supportCheck.lowerBound ..< layerCacheBinding.lowerBound]
        let propertyPublish = try #require(
            cache.range(of: "self.certifiedRectangularVerification ="))
        let objectPublication = try #require(
            cache.range(of: "self.backend = backend"))
        let ownedPreparation =
            cache[ownershipStart.lowerBound ..< objectPublication.lowerBound]
        #expect(ownershipStart.lowerBound < supportCheck.lowerBound)
        #expect(supportCheck.lowerBound < propertyPublish.lowerBound)
        #expect(ownedPreparation.contains("try target.newCacheV2"))
        #expect(ownedPreparation.contains("allSatisfy(\\.supportsSpeculativeWrites)"))
        #expect(ownedPreparation.contains("guard let cache = layer as? KVCache"))
        #expect(!unsupportedCertification.contains("backend.release"))
        #expect(unsupportedCertification.contains("throw MLXFastError.invalidInput("))
        #expect(
            unsupportedCertification.contains(
                "requires a construction-certified CBv2 attention bank"))
        #expect(
            cache.components(separatedBy: "backend.release(").count == 3)

        let finishStart = try #require(
            cache.range(of: "func finishPrefill(evaluating outputs: [MLXArray]) throws"))
        let rootsStart = try #require(
            cache.range(
                of: "var evaluationRoots:",
                range: finishStart.upperBound ..< cache.endIndex))
        let finish = cache[finishStart.lowerBound ..< rootsStart.lowerBound]
        let evalBarrier = try #require(
            finish.range(of: "eval(outputs + evaluationRoots)"))
        let install = try #require(
            finish.range(of: "let installed = try pendingCertifiedInstaller()"))
        let publish = try #require(
            finish.range(of: "certifiedForward = installed"))
        let enable = try #require(
            finish.range(of: "bank.setCertifiedMTPRectangularVerification(true)"))
        #expect(evalBarrier.lowerBound < install.lowerBound)
        #expect(install.lowerBound < publish.lowerBound)
        #expect(publish.lowerBound < enable.lowerBound)
        #expect(
            source.components(separatedBy:
                "bank.setCertifiedMTPRectangularVerification(true)").count == 2)
    }

    @Test func certifiedCBv2ForwardDirectlyDispatchesBetweenConstructionBoundC4AndC16() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)
        let forwardStart = try #require(
            source.range(
                of: "func forwardGreedy(",
                range: source.startIndex ..< source.endIndex))
        let rootsStart = try #require(
            source.range(
                of: "func evaluationRoots(cache:",
                range: forwardStart.upperBound ..< source.endIndex))
        let forward = source[forwardStart.lowerBound ..< rootsStart.lowerBound]

        #expect(forward.contains("return certifiedForward(verifyInput)"))
        #expect(!forward.contains("if verifyInput"))
        #expect(!forward.contains("certifiedForwards["))
        #expect(!forward.contains("guard"))
        #expect(!forward.contains("precondition"))
        #expect(!forward.contains("target.forwardGreedyTokensForDFlash("))
        #expect(!forward.contains("supportsCertifiedMTPRectangularVerification"))
        #expect(!forward.contains("try?"))
        #expect(!forward.contains("fallback"))
    }

    @Test func d15InstallsExactlyC4AndC16AfterPrefillBeforeEnablingCBv2() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)

        #expect(source.contains("static let d15FidelityVerifierWidth = 16"))
        #expect(source.contains("static let d15ExactC4VerifierWidth = 4"))
        #expect(source.contains("static let d15RequiredVerifierColumns: Set<Int> = [4, 16]"))

        let cacheStart = try #require(
            source.range(of: "final class Gemma4DFlashCBv2TargetCache"))
        let compiledStart = try #require(
            source.range(
                of: "final class Gemma4DFlashCompiledVerifierBank",
                range: cacheStart.upperBound ..< source.endIndex))
        let cache = source[cacheStart.lowerBound ..< compiledStart.lowerBound]
        #expect(cache.contains("targetLayerIds: [Int]"))
        #expect(cache.contains("requiredVerifierColumns: Set<Int>"))
        #expect(cache.contains(
            "target.bindCertifiedDFlashGreedyForward("))
        #expect(cache.contains(
            "private let pendingCertifiedInstaller: () throws -> ((MLXArray) ->"))
        #expect(cache.contains(
            "private var certifiedForward: (MLXArray) -> DFlashGreedyTargetForward"))
        #expect(cache.contains("pendingCertifiedInstaller = {"))
        #expect(cache.contains("let c4 = try target.bindCertifiedDFlashGreedyForward("))
        #expect(cache.contains("let c16 = try target.bindCertifiedDFlashGreedyForward("))
        #expect(cache.contains("if input.dim(1) == 4"))
        #expect(cache.contains("return c4(input)"))
        #expect(cache.contains("return c16(input)"))

        let finishStart = try #require(cache.range(
            of: "func finishPrefill(evaluating outputs: [MLXArray]) throws"))
        let initAndStoredState = cache[..<finishStart.lowerBound]
        #expect(!initAndStoredState.contains("try pendingCertifiedInstaller()"))
        let finishEnd = try #require(cache.range(
            of: "var evaluationRoots:",
            range: finishStart.upperBound ..< cache.endIndex))
        let finish = cache[finishStart.lowerBound ..< finishEnd.lowerBound]
        let evalBarrier = try #require(finish.range(
            of: "eval(outputs + evaluationRoots)"))
        let bind = try #require(finish.range(
            of: "let installed = try pendingCertifiedInstaller()"))
        let publish = try #require(finish.range(
            of: "certifiedForward = installed"))
        let enable = try #require(finish.range(
            of: "bank.setCertifiedMTPRectangularVerification(true)"))
        #expect(evalBarrier.lowerBound < bind.lowerBound)
        #expect(bind.lowerBound < publish.lowerBound)
        #expect(publish.lowerBound < enable.lowerBound)

        let construction = try #require(source.range(
            of: "let prefillCacheTransaction = try Gemma4DFlashCBv2TargetCache("))
        let draftCache = try #require(source.range(
            of: "let draftCache = try drafter.makeCache()",
            range: construction.upperBound ..< source.endIndex))
        let d15Construction = source[construction.lowerBound ..< draftCache.lowerBound]
        #expect(d15Construction.contains(
            "targetLayerIds: drafter.config.targetLayerIds"))
        #expect(d15Construction.contains(
            "requiredVerifierColumns: depth == 15\n"
                + "                ? Self.d15RequiredVerifierColumns : []"))

        let modelSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift"),
            encoding: .utf8)
        let binderStart = try #require(modelSource.range(
            of: "public func bindCertifiedDFlashGreedyForward("))
        let binderEnd = try #require(modelSource.range(
            of: "public func bindDirectDFlashGreedyForward(",
            range: binderStart.upperBound ..< modelSource.endIndex))
        let binder = modelSource[binderStart.lowerBound ..< binderEnd.lowerBound]
        #expect(binder.contains("cbv2MTPVerifierContext("))
        #expect(binder.contains("certifiedDFlashGreedyVerifier("))
        #expect(binder.contains(
            "CBv2CertifiedContiguousLayerCacheStack(cache)"))
        #expect(binder.contains("cacheStack: cacheStack"))
        #expect(!binder.contains("callCapturingValidatedDFlashHiddenStates("))
        #expect(binder.contains("verifier: verifier"))
        let bodyStart = try #require(binder.range(of: "body: { [self] input in"))
        let hotBody = binder[bodyStart.lowerBound ..< binder.endIndex]
        #expect(!hotBody.contains("DFlashTargetValidation.validateTargetLayerIds"))
        #expect(!hotBody.contains("ProcessInfo.processInfo.environment"))
        #expect(!hotBody.contains("try?"))
        #expect(!hotBody.contains("fallback"))
    }

    @Test func certifiedDFlashBinderUsesDedicatedNonoptionalVerifierModelPath() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let modelSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)
        let dflashSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift"),
            encoding: .utf8)

        let binderStart = try #require(dflashSource.range(
            of: "public func bindCertifiedDFlashGreedyForward("))
        let binderEnd = try #require(dflashSource.range(
            of: "public func bindDirectDFlashGreedyForward(",
            range: binderStart.upperBound ..< dflashSource.endIndex))
        let binder = dflashSource[binderStart.lowerBound ..< binderEnd.lowerBound]
        #expect(binder.contains("certifiedDFlashGreedyVerifier("))
        #expect(!binder.contains("model.callCapturingCertifiedDFlashVerifierHiddenStates("))
        #expect(!binder.contains("model.callCapturingValidatedDFlashHiddenStates("))

        let wrapperStart = try #require(modelSource.range(
            of: "func certifiedDFlashGreedyVerifier("))
        let wrapperEnd = try #require(modelSource.range(
            of: "func applyRawLMHead(",
            range: wrapperStart.upperBound ..< modelSource.endIndex))
        let wrapper = modelSource[wrapperStart.lowerBound ..< wrapperEnd.lowerBound]
        #expect(wrapper.contains(
            "model.callCapturingCertifiedDFlashVerifierHiddenStates("))
        #expect(wrapper.contains("applyCertifiedMTPVerifierHead("))
        #expect(wrapper.contains("verifier: Gemma4MTPVerifierContext"))

        let dedicatedStart = try #require(modelSource.range(
            of: "func callCapturingCertifiedDFlashVerifierHiddenStates("))
        let nextMethod = try #require(modelSource.range(
            of: "fileprivate func callCapturingMTPVerifierDiagnostic(",
            range: dedicatedStart.upperBound ..< modelSource.endIndex))
        let dedicated = modelSource[
            dedicatedStart.lowerBound ..< nextMethod.lowerBound]
        #expect(dedicated.contains(
            "bindings: Gemma4MTPVerifierModelBindings"))
        #expect(dedicated.contains(
            "cacheStack: CBv2CertifiedContiguousLayerCacheStack"))
        #expect(dedicated.contains("cacheStack.positionOffsets + 0"))
        #expect(dedicated.contains("cacheStack.layers[index]"))
        #expect(dedicated.contains(".callCertifiedMTPVerifier("))
        for forbidden in [
            "Gemma4MTPVerifierContext?",
            "verifier?",
            "if let verifier",
            "verifier: nil",
            "fallback",
            "stock",
            "try?",
            "as!",
            "compactMap",
            "unifiedPositionOffset",
            "??",
        ] {
            #expect(!wrapper.contains(forbidden))
            #expect(!dedicated.contains(forbidden))
        }

        let attentionStart = try #require(modelSource.range(
            of: "fileprivate func callCertifiedMTPVerifier("))
        let attentionEnd = try #require(modelSource.range(
            of: "// MARK: - MoE",
            range: attentionStart.upperBound ..< modelSource.endIndex))
        let attention = modelSource[
            attentionStart.lowerBound ..< attentionEnd.lowerBound]
        #expect(attention.contains(
            "let valueRaw = bindings.value(x, keyRaw)"))
        #expect(attention.contains("bindings.normalizeV(valueRaw)"))

        let cacheSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/"
                    + "ContinuousBatchingV2/LayerCacheV2.swift"),
            encoding: .utf8)
        #expect(cacheSource.contains(
            "public struct CBv2CertifiedContiguousLayerCacheStack"))
        #expect(cacheSource.contains(
            "unifiedPositionStateIdentity == stateIdentity"))
    }

    @Test func sessionInstallsPersistentCompiledVerifierBankAfterCBv2Prefill() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let session = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)
        let worker = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)
        #expect(session.contains("final class Gemma4DFlashCompiledVerifierBank"))
        #expect(session.contains("installPrefillState("))
        #expect(session.contains("bindCompiledDFlashGreedyForward"))
        #expect(worker.contains("var dflashCompiledVerifierBank:"))
        #expect(worker.contains("compiledVerifierBank: dflashCompiledVerifierBank"))
    }

    @Test func d15UsesPrefilledConcreteCBv2BeforeCompiledBankConstruction() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let session = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)
        let round = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift"),
            encoding: .utf8)

        #expect(session.contains("static let d15FidelityVerifierWidth = 16"))
        #expect(session.contains("requiredVerifierColumns:"))
        #expect(session.contains("func runD15ProposalRounds("))
        #expect(session.contains("runDFlashGreedyRound("))
        #expect(session.contains("runDFlashGreedyProposalRound("))
        #expect(round.contains("public func runDFlashGreedyRound"))
        #expect(round.contains("public func runDFlashGreedyProposalRound"))

        let runStart = try #require(session.range(of: "func run(targetN: Int)"))
        let standardStart = try #require(
            session.range(
                of: "func runStandardRounds(",
                range: runStart.upperBound ..< session.endIndex))
        let run = session[runStart.lowerBound ..< standardStart.lowerBound]
        let d15Branch = try #require(run.range(of: "if depth == 15"))
        let d15Return = try #require(
            run.range(
                of: "return try runD15ProposalRounds(",
                range: d15Branch.upperBound ..< run.endIndex))
        let compiledConstruction = try #require(
            run.range(of: "Gemma4DFlashCompiledVerifierBank("))
        #expect(d15Branch.lowerBound < d15Return.lowerBound)
        #expect(d15Return.lowerBound < compiledConstruction.lowerBound)
        let d15Setup = run[d15Branch.lowerBound ..< d15Return.lowerBound]
        #expect(
            d15Setup.contains(
                "self.targetCache = prefillTargetCacheTransaction.cache"))
        #expect(d15Setup.contains("self.prefillTargetCacheTransaction = nil"))
        #expect(!d15Setup.contains("compiledVerifierBank = nil"))
        #expect(
            session.contains(
                "self.compiledVerifierBank = Self.inheritedCompiledVerifierBank("))
        #expect(
            run.contains(
                "targetCacheTransaction: prefillTargetCacheTransaction"))
        #expect(
            run.contains(
                "let requiredVerifierColumns = Set(\n"
                    + "            2 ... MLXFastConstants.experimentalDFlashMaxBlockSize)"))
        #expect(!run.contains("requiredVerifierColumns.insert"))

        let d15Start = try #require(
            session.range(
                of: "func runD15ProposalRounds(",
                range: standardStart.upperBound ..< session.endIndex))
        let d15 = session[d15Start.lowerBound ..< session.endIndex]
        #expect(
            d15.contains(
                "targetCacheTransaction: Gemma4DFlashCBv2TargetCache"))
        #expect(!d15.contains("Gemma4DFlashCompiledVerifierBank"))
        #expect(d15.contains("let phase = proposalPhasePolicy.phase"))
        #expect(d15.contains("case .exactDFlashC4:"))
        #expect(d15.contains("case .periodicExactWide:"))
        #expect(d15.contains("runDFlashGreedyRound("))
        #expect(d15.contains("runDFlashGreedyProposalRound("))
        #expect(
            d15.contains(
                "let isTerminalRound = round.tokens.count == remaining"))
        #expect(d15.contains("wideReplayFrames.append(wideReplayFrame)"))
        #expect(d15.contains("case .replayWideDrafterAndDemote:"))
        #expect(
            d15.contains(
                "try replaySkippedWideDrafterBlocks(replayPlan)"))
        #expect(d15.contains("let replayPlan = gemma4DFlashReplayPlan("))
        #expect(d15.contains("eval(replayedDraftTokens)"))
        #expect(d15.contains("trimPromptCache("))
        #expect(
            d15.contains(
                "targetCacheTransaction: targetCacheTransaction"))
        #expect(!d15.contains("supports("))
        #expect(!d15.contains("ProcessInfo.processInfo.environment"))
        #expect(!d15.contains("try?"))
        #expect(!d15.contains("fallback"))
    }

    @Test func standardLoopDoesNotInspectD15RecurrencePath() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)
        let standardStart = try #require(source.range(of: "func runStandardRounds("))
        let d15Start = try #require(
            source.range(
                of: "func runD15ProposalRounds(",
                range: standardStart.upperBound ..< source.endIndex))
        let standard = source[standardStart.lowerBound ..< d15Start.lowerBound]

        #expect(
            standard.contains(
                "targetCacheTransaction: Gemma4DFlashCompiledVerifierBank"))
        #expect(!standard.contains("proposalPhasePolicy"))
        #expect(!standard.contains("Gemma4DFlashProposalPhase"))
        #expect(!standard.contains("runDFlashGreedyProposalRound"))
    }

    @Test func dflashRunResponseEchoesConstructionOwnedPhysicalWidth() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let session = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)
        let worker = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)
        let trustedWorker = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastTrustedHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)

        #expect(worker.contains("let physicalVerifierWidth: Int?"))
        #expect(
            worker.contains(
                "state.dflashFreeRunSession?.maximumPhysicalVerifierWidth"))
        #expect(
            worker.contains(
                "physicalVerifierWidth: dflashPhysicalVerifierWidth"))
        #expect(
            worker.contains(
                "case physicalVerifierWidth = \"physical_verifier_width\""))
        #expect(trustedWorker.contains("let physicalVerifierWidth: Int?"))
        #expect(
            trustedWorker.contains(
                "case physicalVerifierWidth = \"physical_verifier_width\""))
        #expect(
            session.contains(
                "depth == 15 ? Self.d15FidelityVerifierWidth : depth + 1"))
    }

    @Test func rejectedCertifiedRectangleMatchesCommittedSerialNextStep() throws {
        guard runtimeTestsEnabled else { return }

        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4TextModel(try targetConfig())
            eval(model)
            let promptTokens = [1, 2, 3, 4, 5, 6, 7, 8]
            let prompt = MLXArray(promptTokens.map(Int32.init))[.newAxis, .ellipsis]
            let legacyCache = model.newCache(parameters: nil)
            let contiguous = try Gemma4DFlashCBv2TargetCache(
                target: model,
                promptTokenCount: promptTokens.count,
                maxLength: promptTokens.count + 16,
                certifiedRectangularVerification: true)

            let legacyPrefill = try model.forwardGreedyTokensForDFlash(
                prompt, cache: legacyCache, targetLayerIds: [1, 5])
            let contiguousPrefill = try model.forwardGreedyTokensForDFlash(
                prompt, cache: contiguous.cache, targetLayerIds: [1, 5])
            try contiguous.finishPrefill(
                evaluating: [contiguousPrefill.tokens, contiguousPrefill.targetHidden])
            eval(legacyPrefill.tokens, legacyPrefill.targetHidden)

            #expect(
                legacyPrefill.tokens.asArray(Int32.self)
                    == contiguousPrefill.tokens.asArray(Int32.self))

            let verifyTokens = MLXArray([Int32(9), 10, 11])[.newAxis, .ellipsis]
            contiguous.beginRound()
            let contiguousVerify = try model.forwardGreedyTokensForDFlash(
                verifyTokens, cache: contiguous.cache, targetLayerIds: [1, 5])
            eval(
                [contiguousVerify.tokens, contiguousVerify.targetHidden]
                    + contiguous.evaluationRoots)
            var transactionCache = contiguous.cache
            let keptHidden = try contiguous.finishRound(
                target: model,
                cache: &transactionCache,
                verifyInput: verifyTokens,
                acceptedTokenCount: 0,
                rejectedTokenCount: 2,
                targetLayerIds: [1, 5],
                verifiedTargetHidden: contiguousVerify.targetHidden)
            eval(keptHidden)

            // Serial commits only the rectangle's first column. The CBv2
            // transaction evaluated all three columns, then must become
            // value-equivalent to this one-column history after rollback.
            let legacyCommitted = try model.forwardGreedyTokensForDFlash(
                verifyTokens[0..., 0 ..< 1],
                cache: legacyCache,
                targetLayerIds: [1, 5])
            eval(legacyCommitted.tokens, legacyCommitted.targetHidden)

            #expect(contiguous.storageOffsets.allSatisfy { $0 == promptTokens.count + 1 })
            #expect(keptHidden.dim(1) == 1)

            let next = MLXArray([Int32(12)])[.newAxis, .ellipsis]
            let legacyNext = try model.forwardGreedyTokensForDFlash(
                next, cache: legacyCache, targetLayerIds: [1, 5])
            let contiguousNext = try model.forwardGreedyTokensForDFlash(
                next, cache: contiguous.cache, targetLayerIds: [1, 5])
            eval(
                [legacyNext.tokens, legacyNext.targetHidden,
                 contiguousNext.tokens, contiguousNext.targetHidden]
                    + contiguous.evaluationRoots)

            #expect(
                legacyNext.tokens.asArray(Int32.self)
                    == contiguousNext.tokens.asArray(Int32.self))
            #expect(
                allClose(
                    legacyNext.targetHidden, contiguousNext.targetHidden,
                    rtol: 1e-4, atol: 1e-4
                ).item(Bool.self))
        }
    }

    @Test func fixedVerifierCacheCanResetAndRewindAcrossCompiledCalls() throws {
        guard runtimeTestsEnabled else { return }

        Device.withDefaultDevice(.cpu) {
            let cache = CompilableKVCache(maxLength: 8)
            let prefix = MLXArray([Float(1), 2]).reshaped(1, 1, 2, 1)
            cache.installCommittedPrefix(
                keys: prefix, values: prefix + 100, offset: 2)
            eval(cache)

            let stateArrays = cache.innerState()
            let step = compile { arrays in
                let savedState = stateArrays.map { $0._copyContextInternal() }
                for (state, tracer) in zip(stateArrays, arrays.dropFirst()) {
                    state._updateInternal(tracer)
                }
                _ = cache.update(keys: arrays[0], values: arrays[0] + 100)
                let updatedState = cache.innerState().map {
                    $0._copyContextInternal()
                }
                for (state, saved) in zip(stateArrays, savedState) {
                    state._updateInternal(saved)
                }
                return [arrays[0].sum()] + updatedState
            }

            func runStep(_ input: MLXArray) -> [MLXArray] {
                let outputs = step([input] + stateArrays)
                for (state, updated) in zip(stateArrays, outputs.dropFirst()) {
                    state._updateInternal(updated)
                }
                return [outputs[0]]
            }

            let first = runStep(MLXArray([Float(3)]).reshaped(1, 1, 1, 1))
            eval(first + cache.innerState())
            #expect(cache.offset == 3)
            #expect(
                cache.keys![.ellipsis, ..<3, 0...].asArray(Float.self)
                    == [1, 2, 3])

            cache.rewindCompiledOffset(by: 1)
            eval(cache)
            #expect(cache.offset == 2)

            let replacement = MLXArray([Float(7), 8]).reshaped(1, 1, 2, 1)
            cache.installCommittedPrefix(
                keys: replacement, values: replacement + 100, offset: 2)
            eval(cache)
            let second = runStep(MLXArray([Float(9)]).reshaped(1, 1, 1, 1))
            eval(second + cache.innerState())
            #expect(cache.offset == 3)
            #expect(
                cache.keys![.ellipsis, ..<3, 0...].asArray(Float.self)
                    == [7, 8, 9])
        }
    }
}
