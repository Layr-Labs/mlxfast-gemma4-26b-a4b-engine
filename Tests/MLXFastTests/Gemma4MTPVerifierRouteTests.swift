@testable import MLXLMCommon
@testable import MLXLLM
import Foundation
import Testing

@Suite("Gemma 4 MTP verifier route")
struct Gemma4MTPVerifierRouteTests {
    @Test
    func installedLayerPathHasNoFailableDecodeHelperFallbacks() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let routerStart = try #require(source.range(
            of: "fileprivate func bindMTPVerifier(\n"
                + "        columns: Int\n"
                + "    ) throws -> (MLXArray) -> MLXArray"))
        let routerEnd = try #require(source.range(
            of: "    // MARK: ZIP-ROUTER-001 stages",
            range: routerStart.upperBound..<source.endIndex))
        let routerPath = String(source[routerStart.lowerBound..<routerEnd.lowerBound])
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
        #expect(verifierLayerPath.contains("if verifier != nil"))
        #expect(verifierLayerPath.contains("if let verifier {"))
        #expect(verifierLayerPath.contains("let postAttn = postAttentionLayernorm(attnOut)"))
        #expect(verifierLayerPath.contains(
            "let n1 = preFeedforwardLayernorm(out)"))
        #expect(verifierLayerPath.contains(
            "let n2 = preFeedforwardLayernorm2(out)"))
        #expect(verifierLayerPath.contains("return (out, kvPair, attnPositionOffset)"))
        #expect(!verifierLayerPath.contains("Gemma4FusedLayerGlue"))
        #expect(!verifierLayerPath.contains("Gemma4PrefillGlueV1"))
    }

    @Test
    func exactProductionConstructionFixtureInstallsEveryFixedWidthEntrypoint() throws {
        let fixture = Gemma4MTPVerifierConstructionFixture.production
        let installed = try fixture.install()

        #expect(fixture.topology.vocabularySize == 262_144)
        #expect(installed.columns == [2, 3, 4])
        for columns in 2...4 {
            #expect(
                installed.entrypointCount(columns: columns)
                    == Gemma4MTPVerifierConstructionFixture.production
                        .requiredEntrypointCount(columns: columns))
        }
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
