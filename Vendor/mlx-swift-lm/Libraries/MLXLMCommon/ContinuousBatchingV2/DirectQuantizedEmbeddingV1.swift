// DirectQuantizedEmbeddingV1.swift
//
// Exact B=8 decode lookup for the pinned affine-4 Gemma embedding. Stock
// QuantizedEmbedding gathers packed weight, scale, and bias rows into three
// temporary arrays before affine dequantization. This subclass reads those
// source rows directly and transcribes the stock dequantization expression.

import Foundation
import MLX
import MLXFast
import MLXNN

public final class CBv2DirectQuantizedEmbeddingV1: QuantizedEmbedding {
    private static let batch = 8
    private static let vocabulary = 262_144
    private static let hidden = 2_816
    private static let packedPerRow = 352
    private static let groupsPerRow = 44
    private static let packsPerGroup = 8

    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DIRECT_QUANTIZED_EMBEDDING"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_direct_affine4_embedding_v1",
        inputNames: ["tokens", "weight", "scales", "biases"],
        outputNames: ["out"],
        source: """
            const uint packed_index = uint(thread_position_in_grid.x);
            const uint row = packed_index / PACKED_PER_ROW;
            const uint packed_column = packed_index - row * PACKED_PER_ROW;
            const uint token = uint(tokens[row]);

            const uint value = weight[token * PACKED_PER_ROW + packed_column];
            const uint group = token * GROUPS_PER_ROW
                + packed_column / PACKS_PER_GROUP;
            const T scale = scales[group];
            const T bias = biases[group];
            const uint output_base = row * HIDDEN + packed_column * 8;

            // Verbatim affine-4 dequantization order from quantized.h:
            //     out[i] = scale * d + bias
            // Assignment retains the stock BF16 rounding boundary.
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                const uint8_t d = (value >> (4 * i)) & 0x0f;
                out[output_base + i] = scale * d + bias;
            }
        """,
        ensureRowContiguous: false
    )

    public init(
        _ embedding: Embedding, groupSize: Int, bits: Int,
        mode: QuantizationMode
    ) {
        super.init(
            weight: embedding.weight, groupSize: groupSize, bits: bits,
            mode: mode)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard Self.enabled,
            groupSize == 64,
            bits == 4,
            mode == .affine,
            x.dtype == .int32,
            x.shape == [Self.batch, 1],
            x.strides == [1, 1],
            weight.dtype == .uint32,
            weight.shape == [Self.vocabulary, Self.packedPerRow],
            weight.strides == [Self.packedPerRow, 1],
            scales.dtype == .bfloat16,
            scales.shape == [Self.vocabulary, Self.groupsPerRow],
            scales.strides == [Self.groupsPerRow, 1],
            let biases,
            biases.dtype == .bfloat16,
            biases.shape == scales.shape,
            biases.strides == scales.strides
        else { return super.callAsFunction(x) }

        CBv2EngageMark.once("direct-embedding")
        return Self.kernel(
            [x, weight, scales, biases],
            template: [
                ("T", scales.dtype),
                ("PACKED_PER_ROW", Self.packedPerRow),
                ("GROUPS_PER_ROW", Self.groupsPerRow),
                ("PACKS_PER_GROUP", Self.packsPerGroup),
                ("HIDDEN", Self.hidden),
            ],
            grid: (Self.batch * Self.packedPerRow, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[Self.batch, 1, Self.hidden]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
