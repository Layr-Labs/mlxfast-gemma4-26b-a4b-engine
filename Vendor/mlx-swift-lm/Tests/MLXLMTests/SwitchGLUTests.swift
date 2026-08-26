import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// Serialized: SwitchGLU's init probes custom activations through compiled
// functions (`silu(probe)`), and its forward runs compiled GLU kernels.
// Running these tests in parallel can deadlock MLX-Swift's per-
// CompiledFunction locks (one test tracing inside a compiled call while
// another blocks on the same function's lock from init).
@Suite(.serialized)
struct SwitchGLUTests {
    @Test func fusedGateUpInitializerShapeRemainsPublicAndLoadable() {
        let defaultGLU = SwitchGLU(
            inputDims: 8, hiddenDims: 4, numExperts: 2,
            fuseGateUp: true)
        let customGLU = SwitchGLU(
            inputDims: 8, hiddenDims: 4, numExperts: 2,
            activation: { $0 + 0.5 }, fuseGateUp: true)

        for glu in [defaultGLU, customGLU] {
            let keys = Set(glu.parameters().flattened().map(\.0))
            #expect(keys.contains("gate_up_proj.weight"))
            #expect(!keys.contains("gate_proj.weight"))
            #expect(!keys.contains("up_proj.weight"))

            let x = MLXArray.ones([1, 8])
            let indices = MLXArray([Int32(0), 1]).reshaped(1, 2)
            let output = glu(x, indices)
            eval(output)
            #expect(output.shape == [1, 2, 8])
        }
    }

    @Test func weightedExpertUnsortMatchesLegacyForB1B2B4PrefillReorderings() {
        let topK = 8
        let hidden = 2816
        let tolerance: Float = 0.01

        resetWeightedExpertUnsortStats()
        for batch in [1, 2, 4] {
            // Eight prompt rows per batch keeps all three cases on the sorted
            // dispatch (64/128/256 assignments) while changing token order.
            let tokens = batch * 8
            let assignments = tokens * topK
            MLXRandom.seed(UInt64(80 + batch))
            let original = MLXRandom.normal([tokens, topK, hidden]).asType(.bfloat16)
            let weights = softmax(
                MLXRandom.normal([tokens, topK]).asType(.bfloat16),
                axis: -1,
                precise: true)
            let expertIndices = MLXArray(
                (0 ..< assignments).map {
                    UInt32(($0 * 37 + ($0 / topK) * 11 + batch) % 128)
                }
            )
            let order = argSort(expertIndices)
            let inverseOrder = argSort(order)
            let sorted = original.reshaped(assignments, hidden)[order]

            let expected = weightedExpertSum(original, weights)
            let actual = weightedExpertUnsort(
                sortedOutputs: sorted,
                inverseOrder: inverseOrder,
                weights: weights)
            eval(expected, actual)

            let maxDiff = max(abs(expected - actual)).item(Float.self)
            #expect(actual.shape == [tokens, hidden])
            #expect(
                maxDiff <= tolerance,
                Comment(rawValue: "B\(batch) reordered prefill max diff \(maxDiff)"))
        }

        let provenance = weightedExpertUnsortProvenance(requested: true)
        #expect(provenance.effectiveCalls == 3)
        #expect(provenance.engaged)
        #expect(!provenance.missingExpectedEngagement)

        // Snapshot disarms the boundary-scoped probe. A later production call
        // remains uncounted until the next explicit reset/arm.
        let unarmedOutputs = MLXArray.zeros([64, hidden], dtype: .bfloat16)
        let unarmedOrder = MLXArray((0 ..< 64).map { UInt32($0) })
        let unarmedWeights = MLXArray.zeros([8, topK], dtype: .bfloat16)
        _ = weightedExpertUnsort(
            sortedOutputs: unarmedOutputs,
            inverseOrder: unarmedOrder,
            weights: unarmedWeights)
        #expect(weightedExpertUnsortStats().effectiveCalls == 3)
    }

    @Test func combinedWeightedReductionPreservesDisabledDecodeAndGenericFallbacks() {
        let inputDims = 64
        let hiddenDims = 32
        let numExperts = 8
        let topK = 8
        let tolerance: Float = 0.01
        let glu = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            activation: { $0 + 0.25 },
            bias: false)

        glu.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": values(
                numExperts * hiddenDims * inputDims, offset: 1000
            ).reshaped(numExperts, hiddenDims, inputDims).asType(.bfloat16),
            "up_proj.weight": values(
                numExperts * hiddenDims * inputDims, offset: 5000
            ).reshaped(numExperts, hiddenDims, inputDims).asType(.bfloat16),
            "down_proj.weight": values(
                numExperts * inputDims * hiddenDims, offset: 9000
            ).reshaped(numExperts, inputDims, hiddenDims).asType(.bfloat16),
        ]))

        resetWeightedExpertUnsortStats()
        let cases: [
            (name: String, rows: Int, enabled: Bool, productionPrefill: Bool, dtype: DType)
        ] = [
            ("disabled-sorted", 8, false, true, .bfloat16),
            ("decode-b1", 1, true, false, .bfloat16),
            ("decode-b2", 2, true, false, .bfloat16),
            ("decode-b4", 4, true, false, .bfloat16),
            ("decode-b8-sorted-size", 8, true, false, .bfloat16),
            ("custom-near-geometry", 8, true, true, .bfloat16),
            ("unsupported-dtype", 8, true, true, .float32),
        ]
        for testCase in cases {
            let x = values(testCase.rows * inputDims, offset: 13000)
                .reshaped(testCase.rows, inputDims).asType(testCase.dtype)
            let indices = MLXArray(
                (0 ..< testCase.rows * topK).map {
                    UInt32(($0 * 5 + $0 / topK) % numExperts)
                }
            ).reshaped(testCase.rows, topK)
            let weights = values(testCase.rows * topK, offset: 17)
                .reshaped(testCase.rows, topK).asType(testCase.dtype)

            let expected = weightedExpertSum(glu(x, indices), weights)
            let actual = glu.callAndWeightedReduce(
                x,
                indices,
                weights: weights,
                fuseSortedReduction: testCase.enabled,
                isProductionPrefill: testCase.productionPrefill)
            eval(expected, actual)

            let maxDiff = max(abs(expected - actual)).item(Float.self)
            #expect(
                maxDiff <= tolerance,
                Comment(rawValue: "\(testCase.name) fallback max diff \(maxDiff)"))
        }

        let provenance = weightedExpertUnsortProvenance(requested: true)
        #expect(provenance.effectiveCalls == 0)
        #expect(!provenance.engaged)
        #expect(provenance.missingExpectedEngagement)
    }


    /// Regression for the removed runtime fused gate+up cache: SwitchGLU must
    /// retain NO weight copies beyond its parameters. The removed cache
    /// concatenated a full second copy of the quantized gate+up expert weights
    /// onto the module (stored OUTSIDE `parameters()`, so quantize/update never
    /// saw it) on the first forward — ~8 GiB (qat-4bit) / ~15 GiB (8bit)
    /// model-wide on Gemma 4 26B, and the root cause of the v0.7.3 production
    /// incident. This pins both views:
    ///
    ///   * parameter identity — every parameter array is the SAME instance
    ///     after N decode-shaped forwards (nothing re-pointed, nothing added);
    ///   * module-retained bytes — a reflection walk over every MLXArray
    ///     reachable from the module (deduped by instance) measures EXACTLY
    ///     what the module holds. The baseline is taken BEFORE any forward —
    ///     the old cache built in `callAsFunction` ahead of its size gate, so
    ///     any warm-up forward (whatever its shape) would hide a reintroduced
    ///     build; and unlike the process-global `GPU.activeMemory` counter,
    ///     the walk is immune to allocations from concurrently-running suites,
    ///     so the assertion can be exact instead of noise-margined.
    @Test func decodeShapedForwardsRetainNoExtraWeightCopies() {
        let inputDims = 2048
        let hiddenDims = 2048
        let numExperts = 8

        let glu = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: glu, groupSize: 64, bits: 4)
        MLX.eval(glu.parameters().flattened().map(\.1))

        let paramsBefore = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        let retainedBefore = Self.moduleRetainedBytes(glu)

        // Decode shape: B=1, top_k=8 → indices.size == 8, the exact per-call
        // shape the removed fused dispatch keyed on.
        let x = values(inputDims, offset: 100).reshaped(1, inputDims).asType(.float32)
        let indices = MLXArray((0 ..< 8).map(Int32.init)).reshaped(1, 8)
        for _ in 0 ..< 8 {
            eval(glu(x, indices))
        }

        let retainedAfter = Self.moduleRetainedBytes(glu)
        #expect(
            retainedAfter == retainedBefore,
            Comment(
                rawValue: "8 decode-shaped forwards grew module-retained arrays from "
                    + "\(retainedBefore) to \(retainedAfter) bytes — SwitchGLU is "
                    + "caching weights again"))

        let paramsAfter = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        #expect(paramsAfter.count == paramsBefore.count)
        for (key, before) in paramsBefore {
            #expect(
                paramsAfter[key] === before,
                Comment(rawValue: "parameter \(key) was replaced during forwards"))
        }
    }

    /// Total bytes of every MLXArray reachable from `root` via reflection,
    /// deduped by array instance. Sees arrays stored in plain (non-parameter)
    /// properties — exactly where the removed cache lived — and only this
    /// module's arrays, so concurrent suites cannot perturb the measurement.
    private static func moduleRetainedBytes(_ root: Any) -> Int {
        var seenArrays = Set<ObjectIdentifier>()
        var seenObjects = Set<ObjectIdentifier>()
        var total = 0
        func walk(_ value: Any) {
            if let array = value as? MLXArray {
                if seenArrays.insert(ObjectIdentifier(array)).inserted {
                    total += array.nbytes
                }
                return
            }
            let mirror = Mirror(reflecting: value)
            if mirror.displayStyle == .class {
                let id = ObjectIdentifier(value as AnyObject)
                if !seenObjects.insert(id).inserted { return }
            }
            for child in mirror.children {
                walk(child.value)
            }
        }
        walk(root)
        return total
    }

    private func values(_ count: Int, offset: Int = 0) -> MLXArray {
        MLXArray((offset ..< offset + count).map { Float($0) / 1000 })
    }
}
