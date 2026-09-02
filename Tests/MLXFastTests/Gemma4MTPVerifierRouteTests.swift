@testable import MLXLMCommon
@testable import MLXLLM
import Foundation
import Testing

@Suite("Gemma 4 MTP verifier route")
struct Gemma4MTPVerifierRouteTests {
    @Test
    func fixedC16RouteIsSerialReductionOnlyAndDoesNotAdmitOtherWidths() throws {
        let route = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: false, tiedHeadVersion: 26)

        for columns in [2, 3, 4, 8, 16] {
            #expect(route.supports(.init(batch: 1, columns: columns)))
            #expect(route.attentionStrategy(columns: columns) == .serializedDecode)
            #expect(
                Gemma4B1MTPFullAttentionGeometry.visibleKeyLengths(
                    historyLength: 64, columns: columns)
                    == (0..<columns).map { 65 + $0 })
            #expect(CBv2AttentionQKVMMA8V1.supportsVerifierColumns(columns))
            #expect(CBv2DenseMLPQMVV1.supportsVerifierColumns(columns))
            #expect(Gemma4MMAQuantizedGEMV.supportsVerifierColumns(columns))
            #expect(CBv2Gemma4MTPRouterProjection.supportsVerifierColumns(columns))
        }
        for columns in [1, 5, 6, 7, 9, 15, 17, 32] {
            #expect(!route.supports(.init(batch: 1, columns: columns)))
            #expect(route.strategy(for: .qkv, columns: columns) == nil)
            #expect(route.attentionStrategy(columns: columns) == nil)
        }
        #expect(!route.supports(.init(batch: 8, columns: 8)))

        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/"
                    + "MTP/Gemma4MTPVerifierRoute.swift"),
            encoding: .utf8)
        #expect(source.contains("case sharedSerialReduction"))
        #expect(source.contains("public static let certifiedColumns: Set<Int> = [2, 3, 4, 8, 16]"))
        #expect(!source.contains("case stockModule"))

        let c8Projections: [CBv2Gemma4MTPVerifierProjection] = [
            .qkv, .attentionOutput, .denseGateUp, .denseDown,
            .expert, .router, .tiedHead,
        ]
        for projection in c8Projections {
            #expect(route.strategy(for: projection, columns: 8)
                == .sharedSerialReduction)
            #expect(route.strategy(for: projection, columns: 16)
                == .sharedSerialReduction)
        }
    }

    @Test
    func sharedKVAttentionCandidateIsFullOnlyAndProductionStaysSerial() {
        let route = CBv2Gemma4MTPVerifierRoute.production
        for columns in 2...4 {
            #expect(
                route.candidateAttentionStrategy(
                    kind: .full, columns: columns) == .sharedKVExact)
            #expect(
                route.candidateAttentionStrategy(
                    kind: .slidingWindow(1_024), columns: columns)
                    == .serializedDecode)
            #expect(route.attentionStrategy(columns: columns) == .serializedDecode)
        }
        #expect(route.candidateAttentionStrategy(kind: .full, columns: 1) == nil)
        #expect(route.candidateAttentionStrategy(kind: .full, columns: 5) == nil)
    }

    @Test
    func sharedKVAttentionUsesAnIndependentVisibleEndPerColumn() {
        for columns in 2...4 {
            for history in [0, 1_024, 4_095, 4_096, 8_191, 8_192, 16_384, 65_536,
                131_072]
            {
                #expect(
                    Gemma4B1MTPFullAttentionGeometry.visibleKeyLengths(
                        historyLength: history, columns: columns)
                        == (0..<columns).map { history + $0 + 1 })
            }
        }
    }

    @Test
    func singlePromptShapesAreExplicitAndCannotAliasB8() throws {
        for columns in 2...4 {
            let b1 = CBv2Gemma4MTPVerifierShape(batch: 1, columns: columns)
            let b8 = CBv2Gemma4MTPVerifierShape(batch: 8, columns: columns)
            #expect(CBv2Gemma4MTPVerifierRoute.production.supports(b1))
            #expect(b1 != b8)
        }
        #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
            .init(batch: 1, columns: 1)))
        #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
            .init(batch: 1, columns: 5)))
        #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
            .init(batch: 2, columns: 2)))
    }

    @Test
    func installedTrunkUsesPreboundScaledEmbedding() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let trunkStart = try #require(source.range(of: "    private func forwardTrunk("))
        let pleStart = try #require(source.range(
            of: "        // Compute per-layer inputs (PLE)",
            range: trunkStart.upperBound..<source.endIndex))
        let trunkEntry = String(source[trunkStart.lowerBound..<pleStart.lowerBound])
        let verifierEmbedding = try #require(trunkEntry.range(
            of: "h = verifier.bindings.scaledEmbedding(inputs)"))
        let ordinaryFusedEmbedding = try #require(trunkEntry.range(
            of: "Gemma4FusedScaledEmbedding.apply"))
        #expect(verifierEmbedding.upperBound < ordinaryFusedEmbedding.lowerBound)

        let bindingsStart = try #require(source.range(
            of: "private struct Gemma4MTPVerifierModelBindings"))
        let contextStart = try #require(source.range(
            of: "public struct Gemma4MTPVerifierContext",
            range: bindingsStart.upperBound..<source.endIndex))
        let bindings = String(source[bindingsStart.lowerBound..<contextStart.lowerBound])
        #expect(bindings.contains("let scaledEmbedding: (MLXArray) -> MLXArray"))
        #expect(source.contains(
            "let scaledEmbedding: (MLXArray) -> MLXArray = { tokens in"))
        #expect(source.contains("embedding(tokens) * verifierEmbedScale"))
        #expect(source.contains("scaledEmbedding: scaledEmbedding"))
    }

    @Test
    func installedAttentionPathBypassesFailableQKVHelpers() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let forwardStart = try #require(source.range(of: "    private func forwardV2("))
        let forwardEnd = try #require(source.range(
            of: "// MARK: - MoE (26B-A4B)",
            range: forwardStart.upperBound..<source.endIndex))
        let forward = String(source[forwardStart.lowerBound..<forwardEnd.lowerBound])
        let normalizationStart = try #require(
            forward.range(of: "        var appliedRope = false"))
        let firstFailableHelper = try #require(forward.range(
            of: "gemma4FusedQKVNorm(",
            range: normalizationStart.upperBound..<forward.endIndex))
        let directRoute = try #require(forward.range(
            of: "        if verifier != nil {",
            range: normalizationStart.upperBound..<firstFailableHelper.lowerBound))
        let direct = String(
            forward[directRoute.lowerBound..<firstFailableHelper.lowerBound])

        #expect(direct.contains("queries = qNorm(queryRaw).transposed(0, 2, 1, 3)"))
        #expect(direct.contains("k = kNorm(kRaw).transposed(0, 2, 1, 3)"))
        #expect(direct.contains("v = vNorm(vRaw).transposed(0, 2, 1, 3)"))
        #expect(direct.contains(
            "queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: queryPositionOffset)"))
        #expect(direct.contains(
            "k = gemma4ApplyRotaryPosition(rope, to: k, offset: captured)"))
        #expect(!direct.contains("gemma4FusedQKVNorm"))
    }

    @Test
    func installedAttentionSeparatesC2ToC4AndC8SerialReductionBinders() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let bindStart = try #require(source.range(
            of: "    fileprivate func bindMTPVerifier(\n"
                + "        columns: Int,\n"
                + "        route: CBv2Gemma4MTPVerifierRoute\n"
                + "    ) throws -> Gemma4MTPVerifierAttentionBindings"))
        let bindEnd = try #require(source.range(
            of: "    /// Exact B8/L1 Q/K/V projection",
            range: bindStart.upperBound..<source.endIndex))
        let binder = String(source[bindStart.lowerBound..<bindEnd.lowerBound])

        #expect(binder.contains(
            "let outputValue = try quantized(\n"
                + "            oProj, component: \"\\(prefix) attention output projection\")"))
        #expect(binder.contains("switch outputStrategy"))
        #expect(binder.contains("case .combined:"))
        #expect(binder.contains(
            "output = CBv2AttentionOQMVV1.bindB1Verifier("))
        #expect(binder.contains("case .sharedSerialReduction:"))
        #expect(binder.contains(
            "output = CBv2AttentionOQMVV1.bindB1SharedSerialReductionVerifier("))
        #expect(binder.contains(
            "inDim: outputValue.scales.dim(1) * outputValue.groupSize,"))
        #expect(binder.contains("weight: outputValue.weight, scales: outputValue.scales,"))
        #expect(binder.contains("biases: outputValue.biases, groupSize: outputValue.groupSize,"))
        #expect(binder.contains("bits: outputValue.bits, mode: outputValue.mode"))
        #expect(!binder.contains("outputValue(input)"))
    }

    @Test
    func operatorDiagnosticCapturesTheSameCBv2TrunkWithAndWithoutInstalledBindings() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let diagnosticStart = try #require(source.range(
            of: "    fileprivate func callCapturingMTPVerifierDiagnostic("))
        let trunkStart = try #require(source.range(
            of: "    private func forwardTrunk(",
            range: diagnosticStart.upperBound..<source.endIndex))
        let diagnostic = String(
            source[diagnosticStart.lowerBound..<trunkStart.lowerBound])
        #expect(diagnostic.contains(
            "Gemma4DFlashHiddenCapture(\n"
                + "            layerIds: Array(layers.indices), layerCount: layers.count)"))
        #expect(diagnostic.contains("dFlashHiddenCapture: hiddenCapture"))
        #expect(diagnostic.contains("verifier: verifier"))
        #expect(diagnostic.contains("hiddenCapture.orderedHiddenStates()"))

        let publicStart = try #require(source.range(
            of: "    public func cbv2MTPVerifierDiagnosticForward("))
        let publicDiagnostic = String(source[publicStart.lowerBound..<source.endIndex])
        #expect(publicDiagnostic.contains("installedVerifier: Bool"))
        #expect(publicDiagnostic.contains(
            "let verifier: Gemma4MTPVerifierContext? = installedVerifier ? context : nil"))
        #expect(publicDiagnostic.contains(
            "model.callCapturingMTPVerifierDiagnostic("))
        #expect(publicDiagnostic.contains("layerHiddenStates:"))
        #expect(publicDiagnostic.contains("inputHidden:"))
    }

    @Test
    func installedLayerPathUsesConstructionBoundRectangularGlue() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let routerStart = try #require(source.range(
            of: "fileprivate func bindMTPVerifier(\n"
                + "        columns: Int,\n"
                + "        route: CBv2Gemma4MTPVerifierRoute\n"
                + "    ) throws -> Gemma4MTPVerifierRouterBindings"))
        let routerEnd = try #require(source.range(
            of: "    // MARK: ZIP-ROUTER-001 stages",
            range: routerStart.upperBound..<source.endIndex))
        let routerPath = String(source[routerStart.lowerBound..<routerEnd.lowerBound])
        #expect(!routerPath.contains("Gemma4FusedRouterTop8.bindVerifier("))
        #expect(routerPath.contains("projection: bound"))
        let verifierSelection = try #require(
            routerPath.range(of: "return selectTopK(verifier(normed))"))
        let decodeOnlyFusedSelection = try #require(
            routerPath.range(of: "Gemma4FusedRouterTop8.apply"))
        #expect(verifierSelection.upperBound < decodeOnlyFusedSelection.lowerBound)

        let layerStart = try #require(source.range(
            of: "    public func callAsFunction(\n"
                + "        _ x: MLXArray,\n"
                + "        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,"))
        let ordinaryRoute = try #require(source.range(
            of: "        // PREFIX-001: only build the joined producer",
            range: layerStart.upperBound..<source.endIndex))
        let verifierLayerPath = String(
            source[layerStart.lowerBound..<ordinaryRoute.lowerBound])
        #expect(verifierLayerPath.contains("if let verifier {"))
        #expect(verifierLayerPath.contains(
            "var out = verifier.glue.normResidual(attnOut, residual)"))
        #expect(verifierLayerPath.contains(
            "let (n1, n2) = verifier.glue.dualPreNorm(out)"))
        #expect(verifierLayerPath.contains(
            "out = verifier.glue.branchTail(h1Raw, h2Raw, residual2)"))
        #expect(verifierLayerPath.contains("return (out, kvPair, attnPositionOffset)"))
        #expect(!verifierLayerPath.contains("postAttentionLayernorm(attnOut)"))
        #expect(!verifierLayerPath.contains("preFeedforwardLayernorm(out)"))
        #expect(!verifierLayerPath.contains("preFeedforwardLayernorm2(out)"))
        #expect(!verifierLayerPath.contains("Gemma4FusedLayerGlue"))
        #expect(!verifierLayerPath.contains("Gemma4PrefillGlueV1"))

        let bindingsStart = try #require(source.range(
            of: "public struct Gemma4MTPVerifierLayerBindings"))
        let bindingsEnd = try #require(source.range(
            of: "private struct Gemma4MTPVerifierModelBindings",
            range: bindingsStart.upperBound..<source.endIndex))
        let bindings = String(source[bindingsStart.lowerBound..<bindingsEnd.lowerBound])
        #expect(bindings.contains(
            "let glue: Gemma4PrefillGlueV1.VerifierBindings"))
        #expect(source.contains("Gemma4PrefillGlueV1.bindVerifier("))
    }

    @Test
    func exactProductionConstructionFixtureInstallsEveryFixedWidthEntrypoint() throws {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
        let installed = try fixture.install()
        let shapes = [2, 3, 4, 8, 16].map {
            CBv2Gemma4MTPVerifierShape(batch: 1, columns: $0)
        }

        #expect(fixture.topology.vocabularySize == 262_144)
        #expect(fixture.topology.requiredProjectionEntrypoints == 266)
        #expect(fixture.topology.requiredGlueEntrypoints == 90)
        #expect(fixture.topology.requiredEntrypoints == 356)
        #expect(installed.shapes == shapes)
        #expect(fixture.cbv2MTPVerifierInstalled)
        for shape in shapes {
            #expect(
                installed.entrypointCount(shape: shape)
                    == fixture.topology.requiredEntrypoints)
            #expect(installed.context(
                batch: shape.batch, columns: shape.columns) != nil)
        }
        #expect(installed.context(batch: 8, columns: 2) == nil)
    }

    @Test
    func oneMissingC8BindingLeavesNoPublishedContexts() {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
            .removingBinding(
                component: "layer 0 q projection",
                shape: .init(batch: 1, columns: 8))

        do {
            _ = try fixture.install()
            Issue.record("construction accepted a missing C8 projection binding")
        } catch let error as Gemma4MTPVerifierInstallationError {
            #expect(error.description.contains("layer 0 q projection"))
            #expect(!fixture.cbv2MTPVerifierInstalled)
            #expect(fixture.installedShapeCount == 0)
        } catch {
            Issue.record("unexpected installation error: \(error)")
        }
    }

    @Test
    func everyMissingC8EntrypointLeavesNoPublishedContexts() {
        let components = [
            "layer 0 q projection",
            "layer 0 k projection",
            "layer 0 v projection",
            "layer 0 attention output projection",
            "layer 0 dense gate projection",
            "layer 0 dense up projection",
            "layer 0 dense down projection",
            "layer 0 router projection",
            "layer 0 expert projection",
            "layer 0 post-attention glue",
            "layer 0 dual-prenorm glue",
            "layer 0 branch-tail glue",
            "tied language-model head",
        ]

        for component in components {
            let fixture = Gemma4MTPVerifierConstructionFixture.production
                .removingBinding(
                    component: component,
                    shape: .init(batch: 1, columns: 8))
            do {
                _ = try fixture.install()
                Issue.record("construction accepted missing C8 \(component)")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
                #expect(!fixture.cbv2MTPVerifierInstalled)
                #expect(fixture.installedShapeCount == 0)
            } catch {
                Issue.record("unexpected installation error for \(component): \(error)")
            }
        }
    }

    @Test
    func everyMissingC16EntrypointLeavesNoPublishedContexts() {
        let components = [
            "layer 0 q projection",
            "layer 0 k projection",
            "layer 0 v projection",
            "layer 0 attention output projection",
            "layer 0 dense gate projection",
            "layer 0 dense up projection",
            "layer 0 dense down projection",
            "layer 0 router projection",
            "layer 0 expert projection",
            "layer 0 post-attention glue",
            "layer 0 dual-prenorm glue",
            "layer 0 branch-tail glue",
            "tied language-model head",
        ]

        for component in components {
            let fixture = Gemma4MTPVerifierConstructionFixture.production
                .removingBinding(
                    component: component,
                    shape: .init(batch: 1, columns: 16))
            do {
                _ = try fixture.install()
                Issue.record("construction accepted missing C16 \(component)")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
                #expect(!fixture.cbv2MTPVerifierInstalled)
                #expect(fixture.installedShapeCount == 0)
                #expect(fixture.installedContexts == nil)
            } catch {
                Issue.record("unexpected error for missing C16 \(component): \(error)")
            }
        }
    }

    @Test
    func oneMissingRectangularGlueBindingLeavesNoPublishedContexts() {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
            .removingBinding(
                component: "layer 0 branch-tail glue",
                shape: .init(batch: 1, columns: 4))

        do {
            _ = try fixture.install()
            Issue.record("construction accepted a missing rectangular glue binding")
        } catch let error as Gemma4MTPVerifierInstallationError {
            #expect(error.description.contains("layer 0 branch-tail glue"))
            #expect(!fixture.cbv2MTPVerifierInstalled)
            #expect(fixture.installedShapeCount == 0)
        } catch {
            Issue.record("unexpected installation error: \(error)")
        }
    }

    @Test
    func oneMissingB1ProjectionLeavesNoPartiallyPublishedContexts() {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
            .removingBinding(
                component: "layer 0 q projection",
                shape: .init(batch: 1, columns: 3))

        do {
            _ = try fixture.install()
            Issue.record("construction accepted a missing B1 projection")
        } catch let error as Gemma4MTPVerifierInstallationError {
            #expect(error.description.contains("layer 0 q projection"))
            #expect(!fixture.cbv2MTPVerifierInstalled)
            #expect(fixture.installedShapeCount == 0)
        } catch {
            Issue.record("unexpected installation error: \(error)")
        }
    }

    @Test
    func ordinaryForwardIsTableIndependentAndExplicitVerificationOwnsShapeLookup() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        #expect(source.contains(
            "private var installedMTPVerifierContexts: "
                + "[CBv2Gemma4MTPVerifierShape: Gemma4MTPVerifierContext]?"))
        #expect(source.contains(
            "public func cbv2MTPVerifierContext(\n"
                + "        batch: Int, columns: Int"))
        #expect(source.contains("let shapes = [2, 3, 4, 8, 16].map"))
        #expect(source.contains("batch: 1, columns: $0"))
        #expect(source.contains("CBv2AttentionQKVMMA8V1.bindB1Verifier("))
        #expect(source.contains("CBv2DenseMLPQMVV1.bindB1Verifier("))
        #expect(source.contains("CBv2Gemma4MTPRouterProjection.bindB1Verifier("))
        #expect(source.contains("CBv2Gemma4MTPExpertProjection.bindB1Verifier("))
        #expect(source.contains("Gemma4MMAQuantizedGEMV.bindB1Verifier("))

        let ordinaryStart = try #require(source.range(
            of: "    public func cbv2ForwardWithHidden("))
        let bindStart = try #require(source.range(
            of: "    public func cbv2BindRectangularVerificationForward()",
            range: ordinaryStart.upperBound..<source.endIndex))
        let ordinary = String(
            source[ordinaryStart.lowerBound..<bindStart.lowerBound])
        let binding = String(source[bindStart.lowerBound..<source.endIndex])

        // Prompt/prefill, decode seeds, and serial-oracle columns all use the
        // ordinary seam. It must be identical whether the private table is
        // absent or already installed because it never reads that table.
        #expect(ordinary.contains("verifier: nil"))
        #expect(!ordinary.contains("installedMTPVerifierContexts"))
        #expect(!ordinary.contains("CBv2Gemma4MTPVerifierShape"))
        #expect(!ordinary.contains("preconditionFailure"))

        // The adapter calls this binder once at construction. An uninstalled
        // generic model receives the ordinary rectangular implementation;
        // an installed production model receives a closure over the immutable
        // exact table. The measured closure never performs exact-or-stock
        // eligibility or fallback.
        #expect(binding.contains(
            "guard let contexts = installedMTPVerifierContexts else"))
        #expect(binding.contains(
            "cbv2ForwardWithHidden(tokens, caches: caches)"))
        #expect(binding.contains(
            "let shape = CBv2Gemma4MTPVerifierShape("))
        #expect(binding.contains(
            "batch: tokens.dim(0), columns: tokens.dim(1)"))
        #expect(binding.contains("contexts[shape]"))
        #expect(binding.contains(
            #"no installed B\(shape.batch)/C\(shape.columns) route"#))
        #expect(!binding.contains("tokens.dim(0) == 8"))
        #expect(!binding.contains("tokens.ndim"))
    }

    @Test
    func dflashSmallRowsUseTheInstalledFixedWidthVerifierWithoutFallback() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let dflash = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift"),
            encoding: .utf8)
        let model = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)

        #expect(dflash.contains(
            "let verifier = dFlashInstalledVerifier(for: inputs)"))
        #expect(dflash.contains(
            "verifier: verifier"))
        #expect(dflash.contains(
            "applyLMHead(forward.postNorm, verifier: verifier).argMax(axis: -1)"))

        let captureStart = try #require(model.range(
            of: "    func callCapturingDFlashHiddenStates("))
        let trunkStart = try #require(model.range(
            of: "    private func forwardTrunk(",
            range: captureStart.upperBound..<model.endIndex))
        let capture = String(model[captureStart.lowerBound..<trunkStart.lowerBound])
        #expect(capture.contains("verifier: Gemma4MTPVerifierContext? = nil"))
        #expect(capture.contains("verifier: verifier"))

        // Width is the only runtime-varying route key. The immutable table was
        // validated and installed while loading the exact target, so DFlash
        // must not add an eligible-or-stock branch over model metadata here.
        #expect(dflash.contains("inputs.dim(0), columns: inputs.dim(1)"))
        #expect(!dflash.contains("try-custom"))
        #expect(!dflash.contains("eligible"))
    }

    @Test
    func dflashSlicesAcceptedTapRowsBeforeConcatenatingDraftContext() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift"),
            encoding: .utf8)

        #expect(source.contains("let acceptedHiddenStates = hiddenStates.map"))
        #expect(source.contains("0 ..< accepted + 1"))
        #expect(source.contains("concatenated(acceptedHiddenStates, axis: -1)"))
    }

    @Test
    func rectangularTargetVerificationUsesExplicitPhaseSeam() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let contracts = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/MTPContractsV2.swift"),
            encoding: .utf8)
        let adapter = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/SteppableAdapterV2.swift"),
            encoding: .utf8)
        let verification = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/EngineLoopV2+MTPTargetVerification.swift"),
            encoding: .utf8)

        #expect(contracts.contains(
            "func cbv2BindRectangularVerificationForward()"))
        #expect(contracts.contains(
            "func forwardRectangularVerificationWithHidden("))
        #expect(adapter.contains(
            "forwardable.cbv2BindRectangularVerificationForward()"))
        #expect(adapter.contains(
            "mtpRectangularVerificationForward(tokens, asKVCaches(caches))"))

        let serialStart = try #require(verification.range(
            of: "    func mtpBuildSerialTargetVerification("))
        let rectangularStart = try #require(verification.range(
            of: "    func mtpBuildCertifiedRectangularVerification(",
            range: serialStart.upperBound..<verification.endIndex))
        let automaticStart = try #require(verification.range(
            of: "    func mtpBuildGenericAutomaticVerification(",
            range: rectangularStart.upperBound..<verification.endIndex))
        let serial = String(
            verification[serialStart.lowerBound..<rectangularStart.lowerBound])
        let rectangular = String(
            verification[rectangularStart.lowerBound..<automaticStart.lowerBound])
        #expect(serial.contains("mtp.model.forwardWithHidden("))
        #expect(!serial.contains("forwardRectangularVerificationWithHidden"))
        #expect(rectangular.contains(
            "mtp.model.forwardRectangularVerificationWithHidden("))
    }

    @Test
    func secondInstallationIsRejectedWithoutReplacingPublishedContexts() throws {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
        let first = try fixture.install()
        let firstShapes = first.shapes
        let firstCounts = firstShapes.map(first.entrypointCount(shape:))

        do {
            _ = try fixture.install()
            Issue.record("construction replaced an already published verifier table")
        } catch let error as Gemma4MTPVerifierInstallationError {
            #expect(error.description.contains("already installed"))
        } catch {
            Issue.record("unexpected second-install error: \(error)")
        }

        #expect(fixture.cbv2MTPVerifierInstalled)
        #expect(fixture.installedShapeCount == 5)
        let retained = try #require(fixture.installedContexts)
        #expect(retained.shapes == firstShapes)
        #expect(retained.shapes.map(retained.entrypointCount(shape:)) == firstCounts)

        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift"),
            encoding: .utf8)
        let installStart = try #require(source.range(
            of: "    public func installCBv2MTPVerifier() throws {"))
        let oneShot = try #require(source.range(
            of: "guard installedMTPVerifierContexts == nil else",
            range: installStart.upperBound..<source.endIndex))
        let topology = try #require(source.range(
            of: "let expectedLayerTypes", range: installStart.upperBound..<source.endIndex))
        #expect(oneShot.lowerBound < topology.lowerBound)
    }

    @Test
    func constructionRefusesEveryPinnedTopologyMismatchAndUntiedHead() {
        let incompatible: [(String, Gemma4MTPVerifierConstructionFixture)] = [
            ("hidden size", .production.replacingTopology(hiddenSize: 2_815)),
            ("layer count", .production.replacingTopology(layerCount: 29)),
            ("expert count", .production.replacingTopology(expertCount: 127)),
            ("top-k", .production.replacingTopology(topK: 7)),
            ("untied head", .production.replacingTiedHead(false)),
        ]

        for (component, fixture) in incompatible {
            do {
                _ = try fixture.install()
                Issue.record("construction accepted incompatible \(component)")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
            } catch {
                Issue.record("unexpected error for \(component): \(error)")
            }
        }
    }

    @Test
    func constructionRefusesWrongQuantizationStorageAndMissingAffineBiases() {
        let component = "layer 0 q projection"
        let incompatible: [(String, Gemma4MTPVerifierConstructionFixture)] = [
            ("group size", .production.replacingQuantization(
                component: component, groupSize: 32)),
            ("bits", .production.replacingQuantization(
                component: component, bits: 8)),
            ("mode", .production.replacingQuantization(
                component: component, mode: .mxfp4)),
            ("weight dtype", .production.replacingQuantization(
                component: component, weightDType: .float16)),
            ("scale dtype", .production.replacingQuantization(
                component: component, scaleDType: .float32)),
            ("bias dtype", .production.replacingQuantization(
                component: component, biasDType: .float32)),
            ("affine biases", .production.replacingQuantization(
                component: component, hasAffineBiases: false)),
            ("layout", .production.replacingQuantization(
                component: component, layout: "transposed")),
        ]

        for (reason, fixture) in incompatible {
            do {
                _ = try fixture.install()
                Issue.record("construction accepted wrong \(reason)")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
            } catch {
                Issue.record("unexpected error for \(reason): \(error)")
            }
        }
    }

    @Test
    func constructionRefusesEveryMissingProjectionBinding() {
        let components = [
            "layer 0 q projection",
            "layer 0 k projection",
            "layer 0 v projection",
            "layer 0 attention output projection",
            "layer 0 dense gate projection",
            "layer 0 dense up projection",
            "layer 0 dense down projection",
            "layer 0 router projection",
            "layer 0 expert projection",
            "tied language-model head",
        ]

        for component in components {
            let fixture = Gemma4MTPVerifierConstructionFixture.production
                .removingBinding(component: component, columns: 3)
            do {
                _ = try fixture.install()
                Issue.record("construction accepted missing \(component) binding")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
            } catch {
                Issue.record("unexpected error for \(component): \(error)")
            }
        }
    }

    @Test
    func constructionRefusesUnsupportedRouteAndHeadVersions() {
        let incompatible: [(String, Gemma4MTPVerifierConstructionFixture)] = [
            ("route version", .production.replacingRouteVersion(2)),
            ("head version", .production.replacingHeadVersion(13)),
        ]

        for (component, fixture) in incompatible {
            do {
                _ = try fixture.install()
                Issue.record("construction accepted unsupported \(component)")
            } catch let error as Gemma4MTPVerifierInstallationError {
                #expect(error.description.contains(component))
            } catch {
                Issue.record("unexpected error for \(component): \(error)")
            }
        }
    }

    @Test
    func projectionStrategiesContainNoStockOrGenericFallback() {
        let route = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: false, tiedHeadVersion: 26)

        for columns in 2...4 {
            #expect(route.strategy(for: .qkv, columns: columns) == .combined)
            #expect(route.strategy(for: .attentionOutput, columns: columns) == .combined)
            #expect(route.strategy(for: .denseDown, columns: columns) == .combined)
            #expect(route.strategy(for: .expert, columns: columns) == .combined)
            #expect(route.strategy(for: .router, columns: columns) == .independentB8)
        }
        #expect(route.strategy(for: .denseGateUp, columns: 2) == .combined)
        #expect(route.strategy(for: .denseGateUp, columns: 3) == .combined)
        #expect(route.strategy(for: .denseGateUp, columns: 4) == .independentB8)
        #expect(route.strategy(for: .tiedHead, columns: 2) == .independentB8)
        #expect(route.strategy(for: .tiedHead, columns: 3) == .combined)
        #expect(route.strategy(for: .tiedHead, columns: 4) == .combined)
        #expect(route.strategy(for: .qkv, columns: 1) == nil)
        #expect(route.strategy(for: .qkv, columns: 5) == nil)
    }

    @Test
    func exactKernelGeometriesAreLimitedToProductionWidths() {
        let route = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: false, tiedHeadVersion: 26)

        for columns in 2...4 {
            #expect(route.gateUpRows(columns: columns) == .four)
            for outputWidth in [1024, 2048, 4096, 8192] {
                #expect(
                    route.qkvGeometry(outputWidth: outputWidth, columns: columns)
                        == .ks2Tile2)
            }
        }
        #expect(route.gateUpRows(columns: 1) == nil)
        #expect(route.gateUpRows(columns: 5) == nil)
        #expect(route.qkvGeometry(outputWidth: 3072, columns: 4) == nil)
        #expect(route.qkvGeometry(outputWidth: 4096, columns: 1) == nil)
    }

    @Test
    func everyProductionWidthUsesDecodeSerializedAttention() {
        let route = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: false, tiedHeadVersion: 26)

        for columns in 2...4 {
            #expect(route.attentionStrategy(columns: columns) == .serializedDecode)
        }
        #expect(route.attentionStrategy(columns: 1) == nil)
        #expect(route.attentionStrategy(columns: 5) == nil)
    }

    @Test
    func bindersAndPolicyAreConstructionBound() {
        for columns in 2...4 {
            #expect(CBv2AttentionQKVMMA8V1.supportsVerifierColumns(columns))
            #expect(CBv2AttentionOQMVV1.supportsVerifierColumns(columns))
            #expect(CBv2DenseMLPQMVV1.supportsVerifierColumns(columns))
            #expect(Gemma4MMAQuantizedGEMV.supportsVerifierColumns(columns))
            #expect(CBv2Gemma4MTPExpertProjection.supportsVerifierColumns(columns))
            #expect(CBv2Gemma4MTPRouterProjection.supportsVerifierColumns(columns))
        }
        for columns in [1, 5] {
            #expect(!CBv2AttentionQKVMMA8V1.supportsVerifierColumns(columns))
            #expect(!CBv2AttentionOQMVV1.supportsVerifierColumns(columns))
            #expect(!CBv2DenseMLPQMVV1.supportsVerifierColumns(columns))
            #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierColumns(columns))
            #expect(!CBv2Gemma4MTPExpertProjection.supportsVerifierColumns(columns))
            #expect(!CBv2Gemma4MTPRouterProjection.supportsVerifierColumns(columns))
        }

        let mma8GateUp = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: true, tiedHeadVersion: 26)
        for columns in 2...4 {
            #expect(mma8GateUp.strategy(for: .denseGateUp, columns: columns) == .independentB8)
        }

        for version in [14, 15, 16, 26] {
            #expect(Gemma4MMAQuantizedGEMV.isVerifierCompatible(version: version))
            let route = CBv2Gemma4MTPVerifierRoute.testing(
                gateUpUsesMMA8: false, tiedHeadVersion: version)
            #expect(route.strategy(for: .tiedHead, columns: 2) == .independentB8)
            #expect(route.strategy(for: .tiedHead, columns: 3) == .combined)
            #expect(route.strategy(for: .tiedHead, columns: 4) == .combined)
        }
        for version in [1, 13, 17, 25] {
            #expect(!Gemma4MMAQuantizedGEMV.isVerifierCompatible(version: version))
        }
        let incompatibleHead = CBv2Gemma4MTPVerifierRoute.testing(
            gateUpUsesMMA8: false, tiedHeadVersion: 1)
        for columns in 2...4 {
            #expect(incompatibleHead.strategy(for: .tiedHead, columns: columns) == .independentB8)
        }
    }

    @Test
    func independentHeadAssemblyPreservesThePositionAxis() throws {
        let assembly = try #require(
            Gemma4MMAQuantizedGEMV.independentB8Assembly(
                columns: 2, outDim: 236_800))
        #expect(assembly.columnShape == [8, 1, 236_800])
        #expect(assembly.concatenationAxis == 1)
        #expect(assembly.outputShape == [8, 2, 236_800])
        #expect(Gemma4MMAQuantizedGEMV.independentB8Assembly(
            columns: 1, outDim: 236_800) == nil)
        #expect(Gemma4MMAQuantizedGEMV.independentB8Assembly(
            columns: 5, outDim: 236_800) == nil)
    }

    @Test
    func checkedProjectionProbesRejectWeightsBeforeDimensionAccess() {
        #expect(CBv2DenseMLPQMVV1.supportsVerifierWeightRank(2))
        #expect(!CBv2DenseMLPQMVV1.supportsVerifierWeightRank(0))
        #expect(!CBv2DenseMLPQMVV1.supportsVerifierWeightRank(1))
        #expect(Gemma4MMAQuantizedGEMV.supportsVerifierWeightRank(2))
        #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierWeightRank(0))
        #expect(!Gemma4MMAQuantizedGEMV.supportsVerifierWeightRank(1))
    }
}
