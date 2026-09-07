// SharedSlidingRopeV1.swift
//
// One offset-derived float32 sine/cosine table per batch-8 decode forward.
// Layers may share it only when their sealed base-RoPE parameters match.

import Foundation
import MLX
import MLXFast

public enum CBv2SharedSlidingRopeV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_SHARED_SLIDING_ROPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// A sealed association between host configuration and its actual device
    /// arrays. Consumers cannot associate arbitrary frequency values with a
    /// matching host base. Each layer keeps its own established parameter arrays.
    public struct Parameters {
        public let log2Base: MLXArray
        public let inverseFrequencies: MLXArray?
        fileprivate let log2BaseBits: UInt32
        fileprivate let dimensions: Int

        fileprivate init(log2BaseValue: Float, dimensions: Int) {
            let log2Base = MLXArray([log2BaseValue])
            self.log2Base = log2Base
            self.inverseFrequencies =
                CBv2RaggedTwoPassDecodeAttentionV1.makeBaseRopeInverseFrequencies(
                    log2Base: log2Base, dimensions: dimensions)
            self.log2BaseBits = log2BaseValue.bitPattern
            self.dimensions = dimensions
        }

        fileprivate func matchesConfiguration(_ other: Parameters) -> Bool {
            dimensions == other.dimensions
                && log2BaseBits == other.log2BaseBits
                && (inverseFrequencies != nil) == (other.inverseFrequencies != nil)
        }

        fileprivate func owns(log2Base: MLXArray, inverseFrequencies: MLXArray?) -> Bool {
            guard ObjectIdentifier(log2Base) == ObjectIdentifier(self.log2Base)
            else { return false }
            switch (self.inverseFrequencies, inverseFrequencies) {
            case (nil, nil): return true
            case (let expected?, let actual?):
                return ObjectIdentifier(expected) == ObjectIdentifier(actual)
            default: return false
            }
        }
    }

    /// The scalar and optional inverse table have the same construction and
    /// arithmetic as the resident sliding kernel's existing per-layer inputs.
    public static func makeParameters(
        log2BaseValue: Float, dimensions: Int
    ) -> Parameters {
        Parameters(log2BaseValue: log2BaseValue, dimensions: dimensions)
    }

    public struct Table {
        fileprivate let values: MLXArray
        // Retain the exact immutable forward snapshot to prevent identity reuse.
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
            positionOffsets: MLXArray,
            parameters: Parameters?,
            ropeLog2Base: MLXArray,
            inverseFrequencies: MLXArray?,
            dimensions: Int
        ) -> MLXArray? {
            guard let parameters,
                dimensions == 256,
                parameters.dimensions == dimensions,
                ObjectIdentifier(positionOffsets) == ObjectIdentifier(self.positionOffsets),
                self.parameters.matchesConfiguration(parameters),
                parameters.owns(
                    log2Base: ropeLog2Base, inverseFrequencies: inverseFrequencies)
            else { return nil }
            return values
        }
    }

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_shared_sliding_rope_trig_b8_d256_v1",
        inputNames: ["position_offsets", "rope_parameters"],
        outputNames: ["rope_trig"],
        source: """
            const uint pair = thread_position_in_grid.x;
            const uint batch_index = thread_position_in_grid.y;
            const float L = static_cast<float>(position_offsets[batch_index]);
            const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
            const float inv_freq = ROPE_INV_FREQS
                ? rope_parameters[pair]
                : metal::exp2(-d * rope_parameters[0]);
            const float theta = L * inv_freq;
            const float costheta = metal::fast::cos(theta);
            const float sintheta = metal::fast::sin(theta);
            const size_t trig_base =
                (size_t(batch_index) * (D / 2) + size_t(pair)) * 2;
            rope_trig[trig_base] = costheta;
            rope_trig[trig_base + 1] = sintheta;
            """,
        ensureRowContiguous: true)

    /// Use the same immutable `.batch(offsets)` snapshot passed to every layer
    /// in this forward. A live mutable cache offset array is not a snapshot.
    /// This table is forward-local and contains only shape/offset-derived data.
    public static func make(
        positionOffsets: MLXArray, parameters: Parameters
    ) -> Table? {
        guard enabled,
            parameters.dimensions == 256,
            Float(bitPattern: parameters.log2BaseBits).isFinite,
            positionOffsets.dtype == .int32,
            positionOffsets.shape == [8]
        else { return nil }
        let values = kernel(
            [positionOffsets, parameters.inverseFrequencies ?? parameters.log2Base],
            template: [
                ("D", parameters.dimensions),
                ("ROPE_INV_FREQS", parameters.inverseFrequencies != nil),
            ],
            grid: (parameters.dimensions / 2, 8, 1),
            threadGroup: (parameters.dimensions / 2, 1, 1),
            outputShapes: [[8, parameters.dimensions / 2, 2]],
            outputDTypes: [.float32])[0]
        return Table(values: values, positionOffsets: positionOffsets, parameters: parameters)
    }

    /// A fixed embedding producer may create a table; arbitrary arrays cannot
    /// be adopted as a table. Its original three outputs retain their types.
    public struct EmbeddingProducts {
        public let hidden: MLXArray
        public let normed: MLXArray
        public let qkvRunsum: MLXArray
        public let table: Table
    }

    private static let embeddingFusionEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_EMBED_SHARED_ROPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let embeddingSource = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_sums[32];

            // Match rms_single_row<T, 4>: one logical thread owns four
            // adjacent hidden values. Two adjacent logical threads share one
            // affine-4 packed word but consume disjoint nibble halves.
            const uint base = row * 2816u + lid * 4u;
            const uint word_col = lid >> 1;
            const uint code_base = (lid & 1u) << 2;
            const int raw_token = tokens[row];
            const int vocab = w_shape[0];
            const size_t token = size_t(
                raw_token < 0 ? raw_token + vocab : raw_token);
            const uint packed = w[token * 352u + size_t(word_col)];
            const size_t gindex = token * 44u + size_t(lid >> 4);
            const T scale = scales[gindex];
            const T bias = biases[gindex];
            const T es = embed_scale;

            T hiddenv[4];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                const uint8_t d =
                    (packed >> (4u * (code_base + uint(i)))) & 0x0f;
                // Preserve both stock BF16 boundaries in their original order.
                const T dequantized = scale * d + bias;
                hiddenv[i] = dequantized * es;
                hidden[base + uint(i)] = hiddenv[i];
            }

            // Exact active inputNormWithQKVRunsum tree: four ordered squares,
            // one SIMD reduction, the same 22 populated cross-SIMD lanes, and
            // precise rsqrt. Every SIMD group repeats the final deterministic
            // combine, which is the existing one-barrier broadcast form.
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                const float xi = float(hiddenv[i]);
                acc += xi * xi;
            }
            acc = simd_sum(acc);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = simd_sum(
                simd_lane_id < 22 ? local_sums[simd_lane_id] : 0.0f);
            const float inv = metal::precise::rsqrt(
                acc / 2816.0f + 1e-06f);

            T normedv[4];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                normedv[i] = norm_w[lid * 4u + uint(i)]
                    * static_cast<T>(float(hiddenv[i]) * inv);
                normed[base + uint(i)] = normedv[i];
            }

            // Byte-for-byte arithmetic/order of qkvRunsumEpilogue(normedv).
            float qkv_sum = 0.0f;
            qkv_sum += normedv[0] + normedv[1] + normedv[2] + normedv[3];
            qkv_sum += simd_shuffle_xor(qkv_sum, 1u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 2u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 4u);
            qkv_sum += simd_shuffle_xor(qkv_sum, 8u);
            if ((lid & 15u) == 0u) {
                qkv_rs[row * 44u + lid / 16u] = qkv_sum;
            }

            // Forward-local table: one pair per lane in the first four SIMD groups.
            // No thread in this kernel consumes these independent output values.
            if (lid < 128u) {
                const uint pair = lid;
                const uint batch_index = row;
                const float L = static_cast<float>(position_offsets[batch_index]);
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = ROPE_INV_FREQS
                    ? rope_parameters[pair]
                    : metal::exp2(-d * rope_parameters[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const size_t trig_base =
                    (size_t(batch_index) * (D / 2) + size_t(pair)) * 2;
                rope_trig[trig_base] = costheta;
                rope_trig[trig_base + 1] = sintheta;
            }
            """

    private static let embeddingKernel = MLXFast.metalKernel(
        name: "gemma4_scaled_embedding_input_norm_qkv_rs_shared_rope_b8_2816_bf16_v1",
        inputNames: [
            "tokens", "w", "scales", "biases", "embed_scale", "norm_w",
            "position_offsets", "rope_parameters",
        ],
        outputNames: ["hidden", "normed", "qkv_rs", "rope_trig"],
        source: embeddingSource,
        ensureRowContiguous: true)


    private static let fullEmbeddingSource = embeddingSource + "\n\n" + """
            // Independent full-attention pairs use eight further SIMD groups.
            if (lid >= 128u && lid < 384u) {
                const int pair = int(lid) - 128;
                const int batch_index = int(row);
                const float L = static_cast<float>(position_offsets[batch_index]);
                const float inv_freq = 1.0f / full_rope_freqs[pair];
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const int trig_index = (batch_index * 256 + pair) * 2;
                full_rope_trig[trig_index] = costheta;
                full_rope_trig[trig_index + 1] = sintheta;
            }
            """

    private static let fullEmbeddingKernel = MLXFast.metalKernel(
        name: "gemma4_scaled_embedding_input_norm_qkv_rs_shared_rope_b8_2816_bf16_v1_full_rope_v1",
        inputNames: [
            "tokens", "w", "scales", "biases", "embed_scale", "norm_w",
            "position_offsets", "rope_parameters", "full_rope_freqs",
        ],
        outputNames: ["hidden", "normed", "qkv_rs", "rope_trig", "full_rope_trig"],
        source: fullEmbeddingSource,
        ensureRowContiguous: true)

    // Only the fixed dual producer returns these values to the sealed full
    // factory. Neither table exposes an arbitrary-array adoption initializer.
    struct FullEmbeddingProducts {
        let hidden: MLXArray
        let normed: MLXArray
        let qkvRunsum: MLXArray
        let table: Table
        let fullValues: MLXArray
    }

    /// Produce the same three embedding products and the canonical trig table
    /// with one fixed B8 kernel. The caller retains its quantization and feature
    /// gates; this public array entry also checks every accessed shape and type.
    public static func makeWithInputEmbedding(
        tokens: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        embedScale: Float,
        inputNormWeight: MLXArray,
        positionOffsets: MLXArray,
        parameters: Parameters
    ) -> EmbeddingProducts? {
        guard enabled, embeddingFusionEnabled,
            CBv2RaggedTwoPassDecodeAttentionV1.residentNormRopeKernelAvailable,
            parameters.dimensions == 256,
            Float(bitPattern: parameters.log2BaseBits).isFinite,
            positionOffsets.dtype == .int32, positionOffsets.shape == [8],
            tokens.dtype == .int32, tokens.shape == [8, 1],
            weight.dtype == .uint32, weight.ndim == 2,
            weight.dim(0) > 0, weight.dim(1) == 352,
            scales.dtype == .bfloat16,
            scales.shape == [weight.dim(0), 44],
            biases.dtype == .bfloat16, biases.shape == scales.shape,
            inputNormWeight.dtype == .bfloat16, inputNormWeight.shape == [2816],
            embedScale == Float(2816).squareRoot()
        else { return nil }
        let outputs = embeddingKernel(
            [
                tokens, weight, scales, biases,
                embedScale.asMLXArray(dtype: .bfloat16), inputNormWeight,
                positionOffsets, parameters.inverseFrequencies ?? parameters.log2Base,
            ],
            template: [
                ("T", DType.bfloat16), ("D", 256),
                ("ROPE_INV_FREQS", parameters.inverseFrequencies != nil),
            ],
            grid: (8 * 704, 1, 1), threadGroup: (704, 1, 1),
            outputShapes: [[8, 1, 2816], [8, 1, 2816], [8, 44], [8, 128, 2]],
            outputDTypes: [.bfloat16, .bfloat16, .float32, .float32])
        let table = Table(
            values: outputs[3], positionOffsets: positionOffsets, parameters: parameters)
        return EmbeddingProducts(
            hidden: outputs[0], normed: outputs[1], qkvRunsum: outputs[2], table: table)
    }

    static func makeWithFullInputEmbedding(
        tokens: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        embedScale: Float,
        inputNormWeight: MLXArray,
        positionOffsets: MLXArray,
        parameters: Parameters,
        fullParameters: CBv2SharedFullRopeV1.Parameters
    ) -> FullEmbeddingProducts? {
        guard enabled, embeddingFusionEnabled,
            CBv2SharedFullRopeV1.canMake(
                positionOffsets: positionOffsets, parameters: fullParameters),
            CBv2RaggedTwoPassDecodeAttentionV1.residentNormRopeKernelAvailable,
            parameters.dimensions == 256,
            Float(bitPattern: parameters.log2BaseBits).isFinite,
            positionOffsets.dtype == .int32, positionOffsets.shape == [8],
            tokens.dtype == .int32, tokens.shape == [8, 1],
            weight.dtype == .uint32, weight.ndim == 2,
            weight.dim(0) > 0, weight.dim(1) == 352,
            scales.dtype == .bfloat16,
            scales.shape == [weight.dim(0), 44],
            biases.dtype == .bfloat16, biases.shape == scales.shape,
            inputNormWeight.dtype == .bfloat16, inputNormWeight.shape == [2816],
            embedScale == Float(2816).squareRoot()
        else { return nil }
        let outputs = fullEmbeddingKernel(
            [
                tokens, weight, scales, biases,
                embedScale.asMLXArray(dtype: .bfloat16), inputNormWeight,
                positionOffsets, parameters.inverseFrequencies ?? parameters.log2Base,
                fullParameters.frequencies,
            ],
            template: [
                ("T", DType.bfloat16), ("D", 256),
                ("ROPE_INV_FREQS", parameters.inverseFrequencies != nil),
            ],
            grid: (8 * 704, 1, 1), threadGroup: (704, 1, 1),
            outputShapes: [[8, 1, 2816], [8, 1, 2816], [8, 44], [8, 128, 2], [8, 256, 2]],
            outputDTypes: [.bfloat16, .bfloat16, .float32, .float32, .float32])
        let table = Table(
            values: outputs[3], positionOffsets: positionOffsets, parameters: parameters)
        return FullEmbeddingProducts(
            hidden: outputs[0], normed: outputs[1], qkvRunsum: outputs[2], table: table,
            fullValues: outputs[4])
    }
}
