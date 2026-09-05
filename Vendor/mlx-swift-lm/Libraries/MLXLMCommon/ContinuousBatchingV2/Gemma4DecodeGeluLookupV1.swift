// Exact BF16 GeGLU on the dense and routed-expert decode rectangles.
// A BF16 gate has only 65,536 bit patterns. Evaluate the incumbent compiled
// expression once for that complete domain, with an up operand of one, then
// index its rounded activation and multiply by the current up operand.
// The table contains mathematical function values, independent of weights,
// prompts, token positions, and previous outputs. Its 128 KiB stays immutable.

import Foundation
import MLX
import MLXFast

public enum Gemma4DecodeGeluLookupV1 {
    private static let enabled: Bool = {
        let environment = ProcessInfo.processInfo.environment
        guard MLXHardwareInfo.isCompiledDecodeSupported,
            environment["MLX_DISABLE_COMPILE"] == nil
        else { return false }
        guard let value = environment["DARKBLOOM_GEMMA4_DECODE_GELU_LOOKUP"] else {
            return true
        }
        return !["0", "false", "no", "off"].contains(value.lowercased())
    }()

    // Swift initializes this holder once. Its only array is evaluated before
    // publication and is never mutated; concurrent readers share the values.
    private final class Table: @unchecked Sendable {
        let values: MLXArray

        init() {
            let gate = MLXArray((0..<65536).map { UInt16($0) }).view(dtype: .bfloat16)
            let product = MLX.compile(shapeless: false) {
                (gate: MLXArray, up: MLXArray) -> MLXArray in
                (0.5 * gate
                    * (1 + tanh(sqrt(2 / Float.pi)
                        * (gate + 0.044715 * gate * gate * gate)))) * up
            }
            values = product(gate, MLXArray.ones([65536], dtype: .bfloat16))
            eval(values)
        }
    }

    private static let table = Table()
    private static let kernel = MLXFast.metalKernel(
        name: "gemma4_decode_gelu_bf16_lookup_v1",
        inputNames: ["gate", "up", "table"], outputNames: ["out"],
        source: """
            const uint first = thread_position_in_grid.x * 4;
            const uint row = first / N;
            const uint column = first % N;
            const long gb = long(row) * gate_strides[0];
            const long ub = long(row) * up_strides[0];
            if (gate_strides[NDIM - 1] == 1 && up_strides[NDIM - 1] == 1) {
                const packed_ushort4 g = *reinterpret_cast<const device packed_ushort4*>(gate + gb + column);
                const packed_ushort4 u = *reinterpret_cast<const device packed_ushort4*>(up + ub + column);
                ushort4 result;
                #pragma unroll
                for (uint j = 0; j < 4; ++j) result[j] = bfloat16_to_uint16(table[uint(g[j])] * uint16_to_bfloat16(u[j]));
                *reinterpret_cast<device ushort4*>(out + first) = result;
            } else {
                #pragma unroll
                for (uint j = 0; j < 4; ++j) {
                    const long gi = gb + long(column + j) * gate_strides[NDIM - 1];
                    const long ui = ub + long(column + j) * up_strides[NDIM - 1];
                    out[first + j] = table[uint(bfloat16_to_uint16(gate[gi]))] * up[ui];
                }
            }
        """,
        ensureRowContiguous: false)

    // Shared only inside MLXLMCommon with the dense projection epilogue.
    // Return the already evaluated immutable mathematical table, never a
    // value derived from a model input or prior invocation.
    static func denseFusionTable() -> MLXArray? {
        guard enabled, StreamOrDevice.default == .gpu else { return nil }
        return table.values
    }

    /// The three incumbent decode signatures; other shapes retain their
    /// current implementation. Strides are consumed by Metal after evaluation,
    /// avoiding both speculative host stride inspection and copies of the
    /// dense gate/up slices. Packed input vectors require only BF16 alignment.
    public static func apply(_ gate: MLXArray, _ up: MLXArray) -> MLXArray? {
        guard enabled, StreamOrDevice.default == .gpu,
            gate.dtype == .bfloat16, up.dtype == .bfloat16,
            gate.shape == up.shape
        else { return nil }
        let shape = gate.shape
        let dense = shape == [8, 1, 2112]
        let expert = shape == [64, 1, 704] || shape == [64, 704]
        guard dense || expert else { return nil }
        CBv2EngageMark.once("decode-gelu-lookup")
        return kernel(
            [gate, up, table.values],
            template: [("N", gate.dim(-1)), ("NDIM", gate.ndim)],
            grid: (gate.size / 4, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [shape], outputDTypes: [.bfloat16], stream: .gpu)[0]
    }
}
