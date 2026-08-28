// FullAttentionDecodeOutputPack.swift
//
// One-dispatch raw payload packing for the fixed Gemma 4 full-attention
// B=8 decode output shape. All unsupported inputs fail closed to the caller's
// established concatenate path.

import MLX
import MLXFast

enum CBv2FullAttentionDecodeOutputPack {
    private static let batch = 8
    private static let queryHeads = 16
    private static let queryLength = 1
    private static let headDim = 512
    private static let elementsPerRow = queryHeads * queryLength * headDim
    private static let rowShape = [1, queryHeads, queryLength, headDim]
    private static let outputShape = [batch, queryHeads, queryLength, headDim]

    private static let packKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_full_attention_decode_pack_b8_h16_d512_bf16_v1",
        inputNames: ["row0", "row1", "row2", "row3", "row4", "row5", "row6", "row7"],
        outputNames: ["out"],
        source: """
            const uint element = uint(thread_position_in_grid.x);
            const uint row = uint(thread_position_in_grid.y);

            const device ushort* source =
                reinterpret_cast<const device ushort*>(row0);
            switch (row) {
                case 1:
                    source = reinterpret_cast<const device ushort*>(row1);
                    break;
                case 2:
                    source = reinterpret_cast<const device ushort*>(row2);
                    break;
                case 3:
                    source = reinterpret_cast<const device ushort*>(row3);
                    break;
                case 4:
                    source = reinterpret_cast<const device ushort*>(row4);
                    break;
                case 5:
                    source = reinterpret_cast<const device ushort*>(row5);
                    break;
                case 6:
                    source = reinterpret_cast<const device ushort*>(row6);
                    break;
                case 7:
                    source = reinterpret_cast<const device ushort*>(row7);
                    break;
                default:
                    break;
            }

            device ushort* destination = reinterpret_cast<device ushort*>(out);
            destination[row * ELEMENTS_PER_ROW + element] = source[element];
        """,
        ensureRowContiguous: true
    )

    /// Packs exactly eight `[1, 16, 1, 512]` BF16 row outputs along axis 0.
    /// The kernel treats both source and destination as `ushort`, so every
    /// payload bit and the concatenate row order are preserved without any
    /// floating-point conversion or arithmetic.
    static func pack(_ rows: [MLXArray]) -> MLXArray? {
        guard rows.count == batch else { return nil }
        for row in rows {
            guard row.dtype == .bfloat16, row.shape == rowShape else { return nil }
        }

        let packed = packKernel(
            rows,
            template: [("ELEMENTS_PER_ROW", elementsPerRow)],
            grid: (elementsPerRow, batch, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )
        guard packed.count == 1,
            packed[0].dtype == .bfloat16,
            packed[0].shape == outputShape
        else { return nil }
        return packed[0]
    }
}
