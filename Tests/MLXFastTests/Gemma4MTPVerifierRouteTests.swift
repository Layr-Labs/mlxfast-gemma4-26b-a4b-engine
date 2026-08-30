@testable import MLXLMCommon
import Testing

@Suite("Gemma 4 MTP verifier route")
struct Gemma4MTPVerifierRouteTests {
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
