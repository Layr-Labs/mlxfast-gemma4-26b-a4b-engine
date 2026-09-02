import Foundation
import MLX
@testable import MLXLMCommon
import MLXRandom
import Testing

@Suite("Gemma 4 MTP verifier expert kernels", .serialized)
struct Gemma4MTPVerifierExpertKernelTests {
    private static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    private func expertPlane(
        experts: Int, inDim: Int, outDim: Int, seed: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let packed = inDim / 8
        let groups = inDim / 64
        let expertCodes = MLXArray((0..<experts).map { index in
            UInt32(truncatingIfNeeded: index &* (1_000_003 + seed) &+ seed)
        }).reshaped([experts, 1, 1])
        let outputCodes = MLXArray((0..<outDim).map { index in
            UInt32(truncatingIfNeeded: index &* (2_654_435_761 + seed) &+ seed * 17)
        }).reshaped([1, outDim, 1])
        let packedCodes = MLXArray((0..<packed).map { index in
            UInt32(truncatingIfNeeded: index &* (2_246_822_519 + seed) &+ seed * 31)
        }).reshaped([1, 1, packed])
        let weight = broadcast(expertCodes, to: [experts, outDim, packed])
            + broadcast(outputCodes, to: [experts, outDim, packed])
            + broadcast(packedCodes, to: [experts, outDim, packed])

        let expertMetadata = MLXArray((0..<experts).map { index in
            Float(96 + (index * 7 + seed) % 43) / 128
        }).reshaped([experts, 1, 1])
        let outputMetadata = MLXArray((0..<outDim).map { index in
            Float((index * 11 + seed) % 31) / 256
        }).reshaped([1, outDim, 1])
        let groupMetadata = MLXArray((0..<groups).map { index in
            Float((index * 13 + seed) % 29) / 256
        }).reshaped([1, 1, groups])
        let scales = (
            broadcast(expertMetadata, to: [experts, outDim, groups])
                + broadcast(outputMetadata, to: [experts, outDim, groups])
                + broadcast(groupMetadata, to: [experts, outDim, groups])
        ).asType(.bfloat16)
        let biases = (
            broadcast(expertMetadata - 1, to: [experts, outDim, groups])
                + broadcast(outputMetadata, to: [experts, outDim, groups])
                - broadcast(groupMetadata, to: [experts, outDim, groups])
        ).asType(.bfloat16)
        return (weight, scales, biases)
    }

    private func ordinaryB8ExpertBlock(
        x: MLXArray, indices: MLXArray, routeWeights: MLXArray,
        gate: (MLXArray, MLXArray, MLXArray),
        up: (MLXArray, MLXArray, MLXArray),
        down: (MLXArray, MLXArray, MLXArray)
    ) -> MLXArray {
        let (lhsIndices, sortedIndices, inverseOrder) = gatherSortIndices(
            indices: indices, numExperts: 128, expertPrefixBounds: true)
        let sortedX = MLX.expandedDimensions(x, axes: [-2, -3])
            .flattened(start: 0, end: -3)
        func project(
            _ input: MLXArray,
            _ plane: (MLXArray, MLXArray, MLXArray),
            lhs: MLXArray?
        ) -> MLXArray {
            gatherQuantizedMM(
                input, plane.0, scales: plane.1, biases: plane.2,
                lhsIndices: lhs, rhsIndices: sortedIndices,
                transpose: true, groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: true)
        }
        let gateOutput = project(sortedX, gate, lhs: lhsIndices)
        let upOutput = project(sortedX, up, lhs: lhsIndices)
        let activated = CBv2Gemma4MTPExpertProjection.productionGeGLU(
            gate: gateOutput, up: upOutput)
        let downOutput = project(activated, down, lhs: nil)
        let unsorted = unflatten(
            downOutput[inverseOrder], axis: 0, shape: indices.shape)
        return weightedExpertSum(
            MLX.squeezed(unsorted, axis: -2), routeWeights)
    }

    /// The canonical physical-B1/L1 SwitchGLU route. Eight assignments do
    /// not cross SwitchGLU's `indices.size >= 64` sorting threshold, so the
    /// verifier identity oracle must keep the established unsorted gathers.
    private func ordinaryB1ExpertBlock(
        x: MLXArray, indices: MLXArray, routeWeights: MLXArray,
        gate: (MLXArray, MLXArray, MLXArray),
        up: (MLXArray, MLXArray, MLXArray),
        down: (MLXArray, MLXArray, MLXArray)
    ) -> MLXArray {
        let expandedX = MLX.expandedDimensions(x, axes: [-2, -3])
        func project(
            _ input: MLXArray,
            _ plane: (MLXArray, MLXArray, MLXArray)
        ) -> MLXArray {
            gatherQuantizedMM(
                input, plane.0, scales: plane.1, biases: plane.2,
                lhsIndices: nil, rhsIndices: indices,
                transpose: true, groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: false)
        }
        let gateOutput = project(expandedX, gate)
        let upOutput = project(expandedX, up)
        let activated = CBv2Gemma4MTPExpertProjection.productionGeGLU(
            gate: gateOutput, up: upOutput)
        let downOutput = project(activated, down)
        return weightedExpertSum(
            MLX.squeezed(downOutput, axis: -2), routeWeights)
    }
    @Test
    func routerBinderAcceptsOnlyProductionAffineQ8Group64() {
        #expect(CBv2Gemma4MTPRouterProjection.supportsProductionQuantization(
            groupSize: 64, bits: 8, mode: .affine))
        #expect(!CBv2Gemma4MTPRouterProjection.supportsProductionQuantization(
            groupSize: 64, bits: 4, mode: .affine))
        #expect(!CBv2Gemma4MTPRouterProjection.supportsProductionQuantization(
            groupSize: 32, bits: 8, mode: .affine))
    }

    @Test
    func verifierExpertGeometrySupportsOnlyC2ThroughC4AssignmentBlocks() {
        #expect(CBv2Gemma4ExpertVerifierGeometry.supports(assignments: 128))
        #expect(CBv2Gemma4ExpertVerifierGeometry.supports(assignments: 192))
        #expect(CBv2Gemma4ExpertVerifierGeometry.supports(assignments: 256))
        #expect(!CBv2Gemma4ExpertVerifierGeometry.supports(assignments: 64))
        #expect(!CBv2Gemma4ExpertVerifierGeometry.supports(assignments: 320))
    }

    @Test
    func physicalB1ExpertRoutePreservesC2ToC8AndUsesStableRank128ForC16() throws {
        for columns in [2, 3, 4, 8, 16] {
            #expect(CBv2Gemma4MTPExpertProjection.supportsVerifierColumns(columns))
        }
        for columns in [1, 5, 7, 9, 15, 17] {
            #expect(!CBv2Gemma4MTPExpertProjection.supportsVerifierColumns(columns))
        }

        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift"),
            encoding: .utf8)
        let expertStart = try #require(source.range(
            of: "public enum CBv2Gemma4MTPExpertProjection"))
        let binderStart = try #require(source.range(
            of: "    public static func bindB1Verifier(",
            range: expertStart.upperBound..<source.endIndex))
        let independentStart = try #require(source.range(
            of: "    public static func bindIndependentB8(",
            range: binderStart.upperBound..<source.endIndex))
        let binder = String(source[binderStart.lowerBound..<independentStart.lowerBound])
        let c8Start = try #require(binder.range(of: "        if columns == 8 {"))
        let c16Start = try #require(binder.range(
            of: "        if columns == 16 {",
            range: c8Start.upperBound..<binder.endIndex))
        let c2ToC4Start = try #require(binder.range(
            of: "\n        return { x, indices, routeWeights in",
            range: c16Start.upperBound..<binder.endIndex))
        let c8 = String(binder[c8Start.lowerBound..<c16Start.lowerBound])
        let c16 = String(binder[c16Start.lowerBound..<c2ToC4Start.lowerBound])
        let c2ToC4 = String(binder[c2ToC4Start.lowerBound..<binder.endIndex])
        #expect(c8.contains("runCombinedC8("))
        #expect(!c8.contains("runCombinedC16("))
        #expect(c16.contains("runCombinedC16("))
        #expect(!c16.contains("runCombinedC8("))
        #expect(!c8.contains("?"))
        #expect(!c16.contains("?"))
        #expect(c2ToC4.contains("return runCombined("))
        #expect(!c2ToC4.contains("runCombinedC8("))
        #expect(!c2ToC4.contains("runCombinedC16("))

        let c8HelperStart = try #require(source.range(
            of: "    private static func runCombinedC8("))
        let c16HelperStart = try #require(source.range(
            of: "    private static func runCombinedC16(",
            range: c8HelperStart.upperBound..<source.endIndex))
        let c8Helper = String(
            source[c8HelperStart.lowerBound..<c16HelperStart.lowerBound])
        #expect(c8Helper.contains("routeSimdRank64Kernel("))
        #expect(c8Helper.contains("grid: (64, 1, 1)"))
        #expect(c8Helper.contains("threadGroup: (64, 1, 1)"))
        let independentB1Start = try #require(source.range(
            of: "    private static func runIndependentB1(",
            range: c16HelperStart.upperBound..<source.endIndex))
        let c16Helper = String(
            source[c16HelperStart.lowerBound..<independentB1Start.lowerBound])
        #expect(c16Helper.contains("routeStableRank128Kernel("))
        #expect(c16Helper.contains("grid: (128, 1, 1)"))
        #expect(c16Helper.contains("threadGroup: (128, 1, 1)"))
        #expect(c16Helper.contains("outputShapes: [[128], [128], [128]]"))
        #expect(source.contains("threadgroup uint keys[128]"))
        #expect(source.contains("other < key || (other == key && source < assignment)"))
        #expect(source.contains("row_order[rank] = assignment / 8"))
        #expect(source.contains("sorted_keys[rank] = key"))
        #expect(source.contains("inverse_order[assignment] = rank"))
        #expect(!c16.contains("ProcessInfo"))
        #expect(!c16.contains("fallback"))
        #expect(!c16.contains("catch"))
        #expect(!c16Helper.contains("ProcessInfo"))
        #expect(!c16Helper.contains("fallback"))
        #expect(!c16Helper.contains("catch"))
        #expect(!source.contains("routeSimdRank32Kernel"))
        #expect(!source.contains("b1RouteBinding(columns:"))
    }

    @Test(.enabled(if: runtimeEnabled))
    func independentB8RouterPreservesQ8ScoresAndExactTop8Ordering() throws {
        let inDim = 2816
        let outDim = 128
        let bits = 8
        let weightValues: [UInt32] = (0..<(outDim * inDim / 4)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 71)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 29) % 47) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 17) % 37 - 18) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 4])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)
        for columns in 2...4 {
            let router = try #require(CBv2Gemma4MTPRouterProjection.bindIndependentB8(
                columns: columns, weight: weight, scales: scales, biases: biases,
                groupSize: 64, bits: bits, mode: .affine))
            let xValues: [Float] = (0..<(8 * columns * inDim)).map { index in
                Float((index * 53 + columns * 23) % 283 - 141) / 128.0
            }
            let x = MLXArray(xValues).reshaped([8, columns, inDim])
                .asType(.bfloat16)
            let candidateScores = router(x)
            let referenceScores = concatenated(
                (0..<columns).map { column in
                    quantizedMM(
                        x[0..., column..<(column + 1), 0...], weight,
                        scales: scales, biases: biases, transpose: true,
                        groupSize: 64, bits: bits, mode: .affine)
                },
                axis: 1)
            let kth = outDim - 8
            let candidateTop8 = argPartition(
                candidateScores, kth: kth, axis: -1)[.ellipsis, kth...]
            let referenceTop8 = argPartition(
                referenceScores, kth: kth, axis: -1)[.ellipsis, kth...]
            eval(candidateScores, referenceScores, candidateTop8, referenceTop8)

            #expect(candidateScores.shape == [8, columns, outDim])
            #expect(
                allClose(candidateScores, referenceScores, rtol: 0, atol: 0)
                    .item(Bool.self))
            #expect(all(candidateTop8 .== referenceTop8).item(Bool.self))
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func physicalB1RouterPreservesIndependentQ8ScoresForEveryInstalledWidth() throws {
        let inDim = 2816
        let outDim = 128
        let bits = 8
        let weightValues: [UInt32] = (0..<(outDim * inDim / 4)).map { index in
            UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 103)
        }
        let scaleValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float(128 + (index * 31) % 43) / 128.0
        }
        let biasValues: [Float] = (0..<(outDim * inDim / 64)).map { index in
            Float((index * 19) % 41 - 20) / 128.0
        }
        let weight = MLXArray(weightValues).reshaped([outDim, inDim / 4])
        let scales = MLXArray(scaleValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)
        let biases = MLXArray(biasValues).reshaped([outDim, inDim / 64])
            .asType(.bfloat16)

        for columns in [2, 3, 4, 8, 16] {
            let router = try #require(CBv2Gemma4MTPRouterProjection.bindB1Verifier(
                columns: columns, weight: weight, scales: scales, biases: biases,
                groupSize: 64, bits: bits, mode: .affine))
            let xValues: [Float] = (0..<(columns * inDim)).map { index in
                Float((index * 59 + columns * 29) % 293 - 146) / 128.0
            }
            let x = MLXArray(xValues).reshaped([1, columns, inDim])
                .asType(.bfloat16)
            let candidate = router(x)
            let reference = concatenated(
                (0..<columns).map { column in
                    quantizedMM(
                        x[0..., column..<(column + 1), 0...], weight,
                        scales: scales, biases: biases, transpose: true,
                        groupSize: 64, bits: bits, mode: .affine)
                },
                axis: 1)
            eval(candidate, reference)

            #expect(candidate.shape == [1, columns, outDim])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    @Test(.enabled(if: runtimeEnabled))
    func combinedProductionGeGLUBlockIsExactToIndependentB8Positions() throws {
        let experts = 128
        let hidden = 2816
        let intermediate = 704
        let gate = expertPlane(experts: experts, inDim: hidden, outDim: intermediate, seed: 3)
        let up = expertPlane(experts: experts, inDim: hidden, outDim: intermediate, seed: 17)
        let down = expertPlane(experts: experts, inDim: intermediate, outDim: hidden, seed: 29)

        MLXRandom.seed(4_704_2816)
        for columns in 2...4 {
            let combined = try #require(CBv2Gemma4MTPExpertProjection.bindVerifier(
                columns: columns,
                gateWeight: gate.0, gateScales: gate.1, gateBiases: gate.2,
                upWeight: up.0, upScales: up.1, upBiases: up.2,
                downWeight: down.0, downScales: down.1, downBiases: down.2,
                groupSize: 64, bits: 4, mode: .affine))
            let x = MLXRandom.normal([8, columns, hidden], dtype: .bfloat16)
            let indices = MLXArray(
                (0..<(8 * columns * 8)).map { assignment in
                    let batch = assignment / (columns * 8)
                    let column = (assignment / 8) % columns
                    let slot = assignment % 8
                    return UInt32((batch * 11 + column * 17 + slot * 23) % experts)
                }
            ).reshaped([8, columns, 8])
            let routeWeights = MLXRandom.uniform(
                low: 0, high: 1, [8, columns, 8]).asType(.bfloat16)

            let candidate = combined(x, indices, routeWeights)
            let reference = concatenated(
                (0..<columns).map { column in
                    ordinaryB8ExpertBlock(
                        x: x[0..., column, 0...],
                        indices: indices[0..., column, 0...],
                        routeWeights: routeWeights[0..., column, 0...],
                        gate: gate, up: up, down: down
                    ).reshaped([8, 1, hidden])
                },
                axis: 1)
            eval(candidate, reference)

            #expect(candidate.shape == [8, columns, hidden])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    @Test(.enabled(if: runtimeEnabled), arguments: [false, true])
    func physicalB1ExpertsPreserveIndependentPositionsForRepeatedAndDisjointAssignments(
        disjoint: Bool
    ) throws {
        let experts = 128
        let hidden = 2816
        let intermediate = 704
        let gate = expertPlane(experts: experts, inDim: hidden, outDim: intermediate, seed: 37)
        let up = expertPlane(experts: experts, inDim: hidden, outDim: intermediate, seed: 53)
        let down = expertPlane(experts: experts, inDim: intermediate, outDim: hidden, seed: 71)

        MLXRandom.seed(disjoint ? 4_704_2817 : 4_704_2818)
        for columns in [2, 3, 4, 8, 16] {
            let combined = try #require(CBv2Gemma4MTPExpertProjection.bindB1Verifier(
                columns: columns,
                gateWeight: gate.0, gateScales: gate.1, gateBiases: gate.2,
                upWeight: up.0, upScales: up.1, upBiases: up.2,
                downWeight: down.0, downScales: down.1, downBiases: down.2,
                groupSize: 64, bits: 4, mode: .affine))
            let independent = try #require(
                CBv2Gemma4MTPExpertProjection.bindIndependentB1Verifier(
                    columns: columns,
                    gateWeight: gate.0, gateScales: gate.1, gateBiases: gate.2,
                    upWeight: up.0, upScales: up.1, upBiases: up.2,
                    downWeight: down.0, downScales: down.1, downBiases: down.2,
                    groupSize: 64, bits: 4, mode: .affine))
            let x = MLXRandom.normal([1, columns, hidden], dtype: .bfloat16)
            let indices = MLXArray(
                (0..<(columns * 8)).map { assignment in
                    let column = assignment / 8
                    let slot = assignment % 8
                    // Repeated assignments deliberately share each slot's plane
                    // across columns. Disjoint assignments give every position a
                    // separate expert plane, exposing any cross-expert aliasing.
                    return UInt32(disjoint ? column * 8 + slot : slot * 7)
                }
            ).reshaped([1, columns, 8])
            let routeWeights = MLXRandom.uniform(
                low: 0, high: 1, [1, columns, 8]).asType(.bfloat16)

            let candidate = combined(x, indices, routeWeights)
            let independentCandidate = independent(x, indices, routeWeights)
            let reference = concatenated(
                (0..<columns).map { column in
                    ordinaryB1ExpertBlock(
                        x: x[0..., column, 0...].reshaped([1, hidden]),
                        indices: indices[0..., column, 0...].reshaped([1, 8]),
                        routeWeights: routeWeights[0..., column, 0...].reshaped([1, 8]),
                        gate: gate, up: up, down: down
                    ).reshaped([1, 1, hidden])
                },
                axis: 1)
            eval(candidate, independentCandidate, reference)

            #expect(candidate.shape == [1, columns, hidden])
            #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
            #expect(independentCandidate.shape == [1, columns, hidden])
            #expect(
                allClose(independentCandidate, reference, rtol: 0, atol: 0)
                    .item(Bool.self))
        }
    }
}
