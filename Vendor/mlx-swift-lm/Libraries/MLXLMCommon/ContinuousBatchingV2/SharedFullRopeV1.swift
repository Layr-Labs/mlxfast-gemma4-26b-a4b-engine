// A forward-local proportional-RoPE table shared by the full D512 layers.
import Foundation
import MLX
import MLXFast

public enum CBv2SharedFullRopeV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SHARED_FULL_ROPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// The factory constructs the original proportional layer itself. The
    /// configuration token cannot label an independently supplied array.
    public struct Parameters {
        public let rope: ProportionalRoPE
        public let frequencies: MLXArray
        fileprivate let dimensions: Int
        fileprivate let traditional: Bool
        fileprivate let baseBits: UInt32
        fileprivate let partialBits: UInt32
        fileprivate let factorBits: UInt32

        fileprivate init(
            rope: ProportionalRoPE, frequencies: MLXArray,
            dimensions: Int, base: Float, partialRotaryFactor: Float
        ) {
            self.rope = rope
            self.frequencies = frequencies
            self.dimensions = dimensions
            self.traditional = false
            self.baseBits = base.bitPattern
            self.partialBits = partialRotaryFactor.bitPattern
            self.factorBits = Float(1).bitPattern
        }

        fileprivate func matchesConfiguration(_ other: Parameters) -> Bool {
            dimensions == other.dimensions && traditional == other.traditional
                && baseBits == other.baseBits && partialBits == other.partialBits
                && factorBits == other.factorBits
        }
    }

    /// Same constructor/configuration as Gemma's original full-attention path;
    /// the existing factor * MLX.pow expression and +inf padding remain intact.
    public static func makeParameters(
        dimensions: Int, base: Float, partialRotaryFactor: Float
    ) -> Parameters? {
        guard dimensions == 512, base.isFinite, base > 0,
            partialRotaryFactor.isFinite, partialRotaryFactor > 0,
            partialRotaryFactor <= 1,
            Int(partialRotaryFactor * Float(dimensions) / 2.0) > 0
        else { return nil }
        let rope = ProportionalRoPE(
            dims: dimensions, traditional: false, base: base,
            scalingConfig: [
                "type": .string("proportional"),
                "partial_rotary_factor": .float(partialRotaryFactor),
            ])
        guard let frequencies = rope.frequencyTable,
            frequencies.dtype == .float32, frequencies.shape == [256]
        else { return nil }
        return Parameters(
            rope: rope, frequencies: frequencies, dimensions: dimensions,
            base: base, partialRotaryFactor: partialRotaryFactor)
    }

    public struct Table {
        fileprivate let values: MLXArray
        fileprivate let positionOffsets: MLXArray
        fileprivate let parameters: Parameters

        fileprivate init(
            values: MLXArray, positionOffsets: MLXArray, parameters: Parameters
        ) {
            self.values = values
            self.positionOffsets = positionOffsets
            self.parameters = parameters
        }

        func matchingValues(
            positionOffsets: MLXArray, parameters: Parameters?,
            ropeFrequencies: MLXArray, dimensions: Int
        ) -> MLXArray? {
            guard let parameters, dimensions == 512,
                positionOffsets.dtype == .int32, positionOffsets.shape == [8],
                positionOffsets === self.positionOffsets,
                self.parameters.matchesConfiguration(parameters),
                parameters.dimensions == dimensions,
                ropeFrequencies === parameters.frequencies,
                parameters.rope.frequencyTable === parameters.frequencies,
                ropeFrequencies.dtype == .float32, ropeFrequencies.shape == [256]
            else { return nil }
            return values
        }
    }

    static func canMake(positionOffsets: MLXArray, parameters: Parameters) -> Bool {
        enabled && CBv2RaggedComposedD512DecodeAttentionV1.normRopeKernelAvailable
            && parameters.dimensions == 512
            && parameters.rope.frequencyTable === parameters.frequencies
            && parameters.frequencies.dtype == .float32
            && parameters.frequencies.shape == [256]
            && positionOffsets.dtype == .int32 && positionOffsets.shape == [8]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_shared_full_rope_trig_b8_d512_v1",
        inputNames: ["position_offsets", "rope_freqs"], outputNames: ["rope_trig"],
        source: """
            const int batch_index = int(thread_position_in_grid.x) / 256;
            const int pair = int(thread_position_in_grid.x) % 256;
            const float L = static_cast<float>(position_offsets[batch_index]);
            const float inv_freq = 1.0f / rope_freqs[pair];
            const float theta = L * inv_freq;
            const float costheta = metal::fast::cos(theta);
            const float sintheta = metal::fast::sin(theta);
            const int trig_index = (batch_index * 256 + pair) * 2;
            rope_trig[trig_index] = costheta;
            rope_trig[trig_index + 1] = sintheta;
            """,
        ensureRowContiguous: true)

    public static func make(
        positionOffsets: MLXArray, parameters: Parameters
    ) -> Table? {
        guard canMake(positionOffsets: positionOffsets, parameters: parameters)
        else { return nil }
        let values = kernel(
            [positionOffsets, parameters.frequencies],
            grid: (8 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[8, 256, 2]], outputDTypes: [.float32])[0]
        return Table(values: values, positionOffsets: positionOffsets, parameters: parameters)
    }

    public struct EmbeddingProducts {
        public let hidden: MLXArray
        public let normed: MLXArray
        public let qkvRunsum: MLXArray
        public let slidingTable: CBv2SharedSlidingRopeV1.Table
        public let fullTable: Table
    }

    /// Only the fixed dual producer may supply the sibling table's values.
    /// There is no entry point adopting arbitrary precomputed trig arrays.
    public static func makeWithInputEmbedding(
        tokens: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        embedScale: Float, inputNormWeight: MLXArray, positionOffsets: MLXArray,
        slidingParameters: CBv2SharedSlidingRopeV1.Parameters,
        fullParameters: Parameters
    ) -> EmbeddingProducts? {
        guard canMake(positionOffsets: positionOffsets, parameters: fullParameters),
            let produced = CBv2SharedSlidingRopeV1.makeWithFullInputEmbedding(
                tokens: tokens, weight: weight, scales: scales, biases: biases,
                embedScale: embedScale, inputNormWeight: inputNormWeight,
                positionOffsets: positionOffsets, parameters: slidingParameters,
                fullParameters: fullParameters)
        else { return nil }
        return EmbeddingProducts(
            hidden: produced.hidden, normed: produced.normed,
            qkvRunsum: produced.qkvRunsum, slidingTable: produced.table,
            fullTable: Table(
                values: produced.fullValues, positionOffsets: positionOffsets,
                parameters: fullParameters))
    }
}
