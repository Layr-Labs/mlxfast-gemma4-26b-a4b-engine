// Split-weight expert QMV and exact typed GeGLU in one explicit model primitive.
// Original affine4/g64 weights remain untouched; no joined storage or views.
import Foundation
import MLX
import MLXFast

internal enum Gemma4ExpertSplitGeGLUV1 {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_EXPERT_SPLIT_GEGLU"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "gemma4_expert_split_geglu_v1",
        inputNames: ["gateW", "gateS", "gateB", "upW", "upS", "upB",
                     "x", "lhs", "rhs"],
        outputNames: ["out"],
        source: """
            const uint assignment = threadgroup_position_in_grid.z;
            const uint column = threadgroup_position_in_grid.y * 4u;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint expert = rhs[assignment];
            uint runOffset = 0;
            for (uint prior = assignment; prior > 0; --prior) {
                if (rhs[prior - 1] != expert) break;
                ++runOffset;
            }
            if ((runOffset & 3u) != 0u) return;
            uint runLength = 1;
            while (runLength < 4 && assignment + runLength < 64
                   && rhs[assignment + runLength] == expert) ++runLength;
            const device uint* w = (sg == 0 ? gateW : upW) + expert * 704u * 352u;
            const device T* scales = (sg == 0 ? gateS : upS) + expert * 704u * 44u;
            const device T* biases = (sg == 0 ? gateB : upB) + expert * 704u * 44u;
            threadgroup T rounded[32];
            threadgroup T* plane = rounded + sg * runLength * 4u;
            if (runLength == 1)
                gemma4_expert_split_geglu_v1::project<T,1>(
                    w,scales,biases,x,lhs,assignment,column,lane,plane);
            else if (runLength == 2)
                gemma4_expert_split_geglu_v1::project<T,2>(
                    w,scales,biases,x,lhs,assignment,column,lane,plane);
            else if (runLength == 3)
                gemma4_expert_split_geglu_v1::project<T,3>(
                    w,scales,biases,x,lhs,assignment,column,lane,plane);
            else
                gemma4_expert_split_geglu_v1::project<T,4>(
                    w,scales,biases,x,lhs,assignment,column,lane,plane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && lane < runLength * 4u) {
                const T g = rounded[lane];
                const T u = rounded[runLength * 4u + lane];
                out[(assignment + lane / 4u) * 704u + column + lane % 4u] =
                    gemma4_expert_split_geglu_v1::product(g,u);
            }
            """,
        header: """
            // Same affine4/g64 tape as the incumbent vector helpers.
            namespace gemma4_expert_split_geglu_v1 {
            template <typename T>
            inline float load8(const device T* x, thread float* v) {
                float sum = 0;
                for (int i = 0; i < 8; i += 4) {
                    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                    v[i] = x[i];
                    v[i + 1] = x[i + 1] / 16.0f;
                    v[i + 2] = x[i + 2] / 256.0f;
                    v[i + 3] = x[i + 3] / 4096.0f;
                }
                return sum;
            }
            inline float dot8(uint word, const thread float* v,
                              float scale, float bias, float sum) {
                const uint lo = word & 0xffffu;
                const uint hi = word >> 16;
                float accum = (v[0] * (lo & 0x000fu) + v[1] * (lo & 0x00f0u)
                             + v[2] * (lo & 0x0f00u) + v[3] * (lo & 0xf000u));
                accum += (v[4] * (hi & 0x000fu) + v[5] * (hi & 0x00f0u)
                        + v[6] * (hi & 0x0f00u) + v[7] * (hi & 0xf000u));
                return scale * accum + sum * bias;
            }
            template <typename T>
            inline T product(T gate, T up) {
                const T cubic0 = static_cast<T>(static_cast<T>(0.044715f) * gate);
                const T cubic1 = static_cast<T>(cubic0 * gate);
                const T cubic2 = static_cast<T>(cubic1 * gate);
                const T inner = static_cast<T>(gate + cubic2);
                const T scaled = static_cast<T>(static_cast<T>(0.7978846f) * inner);
                const T curved = metal::precise::tanh(scaled);
                const T shifted = static_cast<T>(static_cast<T>(1.0f) + curved);
                const T halfGate = static_cast<T>(static_cast<T>(0.5f) * gate);
                const T gelu = static_cast<T>(halfGate * shifted);
                return static_cast<T>(gelu * up);
            }
            template <typename T, int R>
            inline void project(const device uint* w, const device T* scales,
                                const device T* biases, const device T* x,
                                const device uint* lhs, uint assignment,
                                uint column, uint lane, threadgroup T* rounded) {
                thread float result[R][4] = {{0}};
                thread uint packed[4];
                thread float ss[4], bb[4], v[8];
                thread uint inputRows[R];
                for (int r = 0; r < R; ++r) inputRows[r] = lhs[assignment + r];
                w += column * 352u + lane;
                scales += column * 44u + lane / 8u;
                biases += column * 44u + lane / 8u;
                for (int block = 0; block < 11; ++block) {
                    for (int c = 0; c < 4; ++c) {
                        packed[c] = w[c * 352u];
                        ss[c] = scales[c * 44u];
                        bb[c] = biases[c * 44u];
                    }
                    // One live eight-value input buffer, as in the incumbent quad stream.
                    for (int r = 0; r < R; ++r) {
                        const float sum = load8<T>(
                            x + inputRows[r] * 2816u + uint(block) * 256u + lane * 8u, v);
                        for (int c = 0; c < 4; ++c)
                            result[r][c] += dot8(packed[c], v, ss[c], bb[c], sum);
                    }
                    w += 32;
                    scales += 4;
                    biases += 4;
                }
                for (int r = 0; r < R; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        const float value = simd_sum(result[r][c]);
                        if (lane == 0) rounded[r * 4 + c] = static_cast<T>(value);
                    }
                }
            }
            } // namespace gemma4_expert_split_geglu_v1
            """ + "\n",
        ensureRowContiguous: true
    )

    /// Called only with the explicit production GeGLU profile, validated
    /// affine4/g64 split storage, and the ordinary raw-key sorted route table.
    /// A pair of SIMD groups computes four gate/up columns. Same-expert
    /// leaders retain the incumbent four-assignment reuse. Each projection
    /// rounds to BF16 before a 64-byte threadgroup exchange and typed GeGLU.
    static func apply(
        x: MLXArray, lhs: MLXArray, rhs: MLXArray,
        storage: SwitchGateUpFusedStorage
    ) -> MLXArray? {
        guard enabled, x.dtype == .bfloat16, x.shape == [8, 1, 2816],
            lhs.dtype == .uint32, lhs.shape == [64],
            rhs.dtype == .uint32, rhs.shape == [64]
        else { return nil }
        let result = kernel(
            [storage.gateWeight, storage.gateScales, storage.gateBiases,
             storage.upWeight, storage.upScales, storage.upBiases, x, lhs, rhs],
            template: [("T", DType.bfloat16)],
            grid: (64, 176, 64),
            threadGroup: (64, 1, 1),
            outputShapes: [[64, 1, 704]],
            outputDTypes: [.bfloat16]
        )[0]
        CBv2EngageMark.once("expert-split-geglu")
        return result
    }
}
