import MLX
import MLXFast

/// Exact physical-B1, C2-C4 full-attention candidate for Gemma 4 verification.
///
/// QK and AV preserve the frozen ordinary B1/L1 per-output accumulation and
/// reduction order while sharing immutable K/V loads across the independent
/// verification columns. Softmax remains one independent precise dispatch per
/// column because every column has a different causal visible end.
///
/// This type is deliberately not installed by the production route yet. Its
/// callable is exposed to the exact operator and isolated performance gates;
/// promotion is a later construction-policy change after those gates pass.
public enum Gemma4B1MTPFullAttentionV1 {
    public typealias Attention = (
        _ queries: MLXArray,
        _ keys: MLXArray,
        _ values: MLXArray,
        _ historyLength: Int
    ) -> MLXArray

    private static let queryHeads = 16
    private static let kvHeads = 2
    private static let gqa = 8
    private static let headDim = 512

    public static func bind(columns: Int) -> Attention? {
        switch columns {
        case 2: return bindFixedColumns(c2)
        case 3: return bindFixedColumns(c3)
        case 4: return bindFixedColumns(c4)
        default: return nil
        }
    }

    private struct Kernels {
        let columns: Int
        let qk: MLXFast.MLXFastKernel
        let blockSoftmax: MLXFast.MLXFastKernel
        let loopedSoftmax: MLXFast.MLXFastKernel
        let av: MLXFast.MLXFastKernel

        init(columns: Int) {
            self.columns = columns
            qk = makeQK(columns: columns)
            blockSoftmax = makeBlockSoftmax(columns: columns)
            loopedSoftmax = makeLoopedSoftmax(columns: columns)
            av = makeSharedAV(columns: columns)
        }
    }

    private static let c2 = Kernels(columns: 2)
    private static let c3 = Kernels(columns: 3)
    private static let c4 = Kernels(columns: 4)

    private static func bindFixedColumns(_ kernels: Kernels) -> Attention {
        let columns = kernels.columns
        return { queries, keys, values, historyLength in
            let visibleLengths = Gemma4B1MTPFullAttentionGeometry.visibleKeyLengths(
                historyLength: historyLength, columns: columns)
            let keyLength = historyLength + columns
            let params = MLXArray([UInt32(historyLength)])
            let template: [(String, any KernelTemplateArg)] = [
                ("T", DType.bfloat16)
            ]

            let chunks = (keyLength + 63) / 64
            let scoreShapes = visibleLengths.map { [1, queryHeads, $0] }
            let scores = kernels.qk(
                [queries, keys, params],
                template: template,
                grid: (32, 4, kvHeads * chunks),
                threadGroup: (32, 4, 1),
                outputShapes: scoreShapes,
                outputDTypes: Array(repeating: .bfloat16, count: columns))

            var probabilities: [MLXArray] = []
            probabilities.reserveCapacity(columns)
            for column in 0..<columns {
                let visible = visibleLengths[column]
                let softmaxParams = MLXArray([UInt32(visible)])
                let softmax: MLXFast.MLXFastKernel
                let threads: Int
                if visible <= 4_096 {
                    softmax = kernels.blockSoftmax
                    threads = ((visible + 3) / 4 + 31) / 32 * 32
                } else {
                    softmax = kernels.loopedSoftmax
                    threads = 1_024
                }
                probabilities.append(
                    softmax(
                        [scores[column], softmaxParams],
                        template: template,
                        grid: (threads * queryHeads, 1, 1),
                        threadGroup: (threads, 1, 1),
                        outputShapes: [[1, queryHeads, visible]],
                        outputDTypes: [.bfloat16])[0])
            }

            return kernels.av(
                probabilities + [values, params],
                template: template,
                grid: (32, 4, kvHeads * 8),
                threadGroup: (32, 4, 1),
                outputShapes: [[1, queryHeads, columns, headDim]],
                outputDTypes: [.bfloat16])[0]
        }
    }

    private static func makeQK(columns: Int) -> MLXFast.MLXFastKernel {
        let outputNames = (0..<columns).map { "scores\($0)" }
        let writes = (0..<columns).map { column in
            """
                        const int visible_\(column) = history_length + \(column + 1);
                        if (out_row + tm < visible_\(column)) {
                            scores\(column)[
                                size_t(kv_head * GQA + h) * visible_\(column)
                                + out_row + tm] =
                                static_cast<T>(result[\(column)][h][tm]);
                        }
            """
        }.joined(separator: "\n")

        return MLXFast.metalKernel(
            name: "gemma4_b1_mtp_full_qk_c\(columns)_bf16_g8_d512_v1",
            inputNames: ["queries", "keys", "params"],
            outputNames: outputNames,
            source: """
                constexpr int COLUMNS = \(columns);
                constexpr int D = 512;
                constexpr int GQA = 8;

                const int history_length = int(params[0]);
                const int key_length = history_length + COLUMNS;
                const int n_chunks = (key_length + 63) / 64;
                const int z = int(threadgroup_position_in_grid.z);
                const int chunk = z % n_chunks;
                const int kv_head = z / n_chunks;
                const int sg = int(simdgroup_index_in_threadgroup);
                const int lane = int(thread_index_in_simdgroup);

                const device T* key_plane =
                    keys + size_t(kv_head) * key_length * D;
                const int virtual_groups = (key_length + 15) / 16;
                const int vtg_lo = chunk * 4;
                const int vtg_hi = min(vtg_lo + 4, virtual_groups);

                for (int vtg = vtg_lo; vtg < vtg_hi; ++vtg) {
                    int out_row = vtg * 16 + sg * 4;
                    if (out_row >= key_length) continue;
                    if (key_length >= 4 && out_row + 4 > key_length) {
                        out_row = key_length - 4;
                    }

                    float result[COLUMNS][GQA][4] = {{{0.0f}}};
                    T inter[4];
                    int bn = lane * 4;
                    for (int i = 0; i < 4; ++i) {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            const bool valid_key = out_row + tm < key_length;
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                inter[tn] = valid_key
                                    ? key_plane[
                                          size_t(out_row + tm) * D + bn + tn]
                                    : static_cast<T>(0);
                            }
                            #pragma clang loop unroll(full)
                            for (int c = 0; c < COLUMNS; ++c) {
                                #pragma clang loop unroll(full)
                                for (int h = 0; h < GQA; ++h) {
                                    const device T* query = queries
                                        + size_t(kv_head * GQA + h)
                                            * COLUMNS * D
                                        + size_t(c) * D;
                                    #pragma clang loop unroll(full)
                                    for (int tn = 0; tn < 4; ++tn) {
                                        result[c][h][tm] += inter[tn]
                                            * static_cast<float>(query[bn + tn]);
                                    }
                                }
                            }
                        }
                        bn += 128;
                    }
                    #pragma clang loop unroll(full)
                    for (int c = 0; c < COLUMNS; ++c) {
                        #pragma clang loop unroll(full)
                        for (int h = 0; h < GQA; ++h) {
                            #pragma clang loop unroll(full)
                            for (int tm = 0; tm < 4; ++tm) {
                                #pragma clang loop unroll(full)
                                for (ushort delta = 16; delta >= 1; delta >>= 1) {
                                    result[c][h][tm] +=
                                        simd_shuffle_down(result[c][h][tm], delta);
                                }
                            }
                        }
                    }
                    if (lane == 0) {
                        #pragma clang loop unroll(full)
                        for (int h = 0; h < GQA; ++h) {
                            #pragma clang loop unroll(full)
                            for (int tm = 0; tm < 4; ++tm) {
                \(writes)
                            }
                        }
                    }
                }
            """,
            ensureRowContiguous: true)
    }

    private static func makeBlockSoftmax(columns: Int) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "gemma4_b1_mtp_full_softmax_block_c\(columns)_bf16_v1",
            inputNames: ["scores", "params"],
            outputNames: ["probs"],
            source: """
                const int axis_size = int(params[0]);
                const int gid = int(threadgroup_position_in_grid.x);
                const int lid = int(thread_position_in_threadgroup.x);
                const int simd_lane_id = int(thread_index_in_simdgroup);
                const int simd_group_id = int(simdgroup_index_in_threadgroup);
                threadgroup float local_max[32];
                threadgroup float local_normalizer[32];
                float ld[4];
                const device T* in = scores + size_t(gid) * axis_size + lid * 4;
                if (lid * 4 + 4 <= axis_size) {
                    for (int i = 0; i < 4; ++i) ld[i] = static_cast<float>(in[i]);
                } else {
                    for (int i = 0; i < 4; ++i) {
                        ld[i] = lid * 4 + i < axis_size
                            ? static_cast<float>(in[i]) : -INFINITY;
                    }
                }
                if (simd_group_id == 0) {
                    local_max[simd_lane_id] = -INFINITY;
                    local_normalizer[simd_lane_id] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float maxval = -3.402823466e+38F;
                for (int i = 0; i < 4; ++i) {
                    maxval = maxval < ld[i] ? ld[i] : maxval;
                }
                maxval = simd_max(maxval);
                if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    maxval = simd_max(local_max[simd_lane_id]);
                    if (simd_lane_id == 0) local_max[0] = maxval;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maxval = local_max[0];
                float normalizer = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const float exp_x = fast::exp(ld[i] - maxval);
                    ld[i] = exp_x;
                    normalizer += exp_x;
                }
                normalizer = simd_sum(normalizer);
                if (simd_lane_id == 0) {
                    local_normalizer[simd_group_id] = normalizer;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    normalizer = simd_sum(local_normalizer[simd_lane_id]);
                    if (simd_lane_id == 0) local_normalizer[0] = normalizer;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                normalizer = 1 / local_normalizer[0];
                device T* out = probs + size_t(gid) * axis_size + lid * 4;
                for (int i = 0; i < 4 && lid * 4 + i < axis_size; ++i) {
                    out[i] = static_cast<T>(ld[i] * normalizer);
                }
            """,
            ensureRowContiguous: true)
    }

    private static func makeLoopedSoftmax(columns: Int) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "gemma4_b1_mtp_full_softmax_looped_c\(columns)_bf16_v1",
            inputNames: ["scores", "params"],
            outputNames: ["probs"],
            source: """
                const int axis_size = int(params[0]);
                const int gid = int(threadgroup_position_in_grid.x);
                const int lid = int(thread_position_in_threadgroup.x);
                const int lsize = int(threads_per_threadgroup.x);
                const int simd_lane_id = int(thread_index_in_simdgroup);
                const int simd_group_id = int(simdgroup_index_in_threadgroup);
                const device T* in = scores + size_t(gid) * axis_size;
                threadgroup float local_max[32];
                threadgroup float local_normalizer[32];
                float prevmax;
                float maxval = -3.402823466e+38F;
                float normalizer = 0.0f;
                const int rounds = (axis_size + 4 * lsize - 1) / (4 * lsize);
                for (int r = 0; r < rounds; ++r) {
                    const int offset = r * lsize * 4 + lid * 4;
                    float vals[4];
                    for (int i = 0; i < 4; ++i) {
                        vals[i] = offset + i < axis_size
                            ? static_cast<float>(in[offset + i]) : -INFINITY;
                    }
                    prevmax = maxval;
                    for (int i = 0; i < 4; ++i) {
                        maxval = maxval < vals[i] ? vals[i] : maxval;
                    }
                    normalizer *= fast::exp(prevmax - maxval);
                    for (int i = 0; i < 4; ++i) {
                        normalizer += fast::exp(vals[i] - maxval);
                    }
                }
                prevmax = maxval;
                maxval = simd_max(maxval);
                normalizer *= fast::exp(prevmax - maxval);
                normalizer = simd_sum(normalizer);
                prevmax = maxval;
                if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maxval = simd_max(local_max[simd_lane_id]);
                normalizer *= fast::exp(prevmax - maxval);
                if (simd_lane_id == 0) {
                    local_normalizer[simd_group_id] = normalizer;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                normalizer = simd_sum(local_normalizer[simd_lane_id]);
                normalizer = 1 / normalizer;
                device T* out = probs + size_t(gid) * axis_size;
                for (int r = 0; r < rounds; ++r) {
                    const int offset = r * lsize * 4 + lid * 4;
                    for (int i = 0; i < 4 && offset + i < axis_size; ++i) {
                        out[offset + i] = static_cast<T>(
                            fast::exp(static_cast<float>(in[offset + i]) - maxval)
                                * normalizer);
                    }
                }
            """,
            ensureRowContiguous: true)
    }

    private static func makeSharedAV(columns: Int) -> MLXFast.MLXFastKernel {
        let probabilityNames = (0..<columns).map { "probs\($0)" }
        let probabilityPointers = (0..<columns).map { column in
            """
                const device T* prob_\(column) = probs\(column)
                    + size_t(kv_head * GQA) * (history_length + \(column + 1));
            """
        }.joined(separator: "\n")
        let updates = (0..<columns).map { column in
            """
                            if (row_index < history_length + \(column + 1)) {
                                #pragma clang loop unroll(full)
                                for (int h = 0; h < GQA; ++h) {
                                    const float coefficient = static_cast<float>(
                                        prob_\(column)[
                                            size_t(h) * (history_length + \(column + 1))
                                            + row_index]);
                                    #pragma clang loop unroll(full)
                                    for (int tn = 0; tn < 4; ++tn) {
                                        result[\(column)][h][tn] +=
                                            coefficient * inter[tn];
                                    }
                                }
                            }
            """
        }.joined(separator: "\n")
        let writes = (0..<columns).map { column in
            """
                        device T* output_\(column) = out
                            + size_t(kv_head * GQA + h) * COLUMNS * D
                            + size_t(\(column)) * D + out_col;
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            output_\(column)[tn] =
                                static_cast<T>(result[\(column)][h][tn]);
                        }
            """
        }.joined(separator: "\n")
        return MLXFast.metalKernel(
            name: "gemma4_b1_mtp_full_av_c\(columns)_bf16_g8_d512_v1",
            inputNames: probabilityNames + ["values", "params"],
            outputNames: ["out"],
            source: """
                constexpr int COLUMNS = \(columns);
                constexpr int D = 512;
                constexpr int GQA = 8;
                constexpr int SM = 8;
                constexpr int SN = 4;
                constexpr int TILES = 8;

                const int history_length = int(params[0]);
                const int key_length = history_length + COLUMNS;
                const int z = int(threadgroup_position_in_grid.z);
                const int tile = z % TILES;
                const int kv_head = z / TILES;
                const int sg = int(simdgroup_index_in_threadgroup);
                const int lane = int(thread_index_in_simdgroup);
                const int thrM = lane / SN;
                const int thrN = lane % SN;
                int row_index = thrM * 4;
                const int out_col = tile * (SN * 4 * 4)
                    + (SN * sg + thrN) * 4;
                const device T* value_plane =
                    values + size_t(kv_head) * key_length * D;
                \(probabilityPointers)
                float result[COLUMNS][GQA][4] = {{{0.0f}}};
                T inter[4];
                const int stride = SM * 4;
                while (row_index < key_length) {
                    threadgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        if (row_index + tm >= key_length) break;
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            inter[tn] = value_plane[
                                size_t(row_index + tm) * D + out_col + tn];
                        }
                        const int row_index_saved = row_index;
                        row_index += tm;
                \(updates)
                        row_index = row_index_saved;
                    }
                    row_index += stride;
                }
                #pragma clang loop unroll(full)
                for (int c = 0; c < COLUMNS; ++c) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            #pragma clang loop unroll(full)
                            for (ushort reduction = SM / 2;
                                reduction >= 1; reduction >>= 1) {
                                result[c][h][tn] += simd_shuffle_down(
                                    result[c][h][tn], SN * reduction);
                            }
                        }
                    }
                }
                if (thrM == 0) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                \(writes)
                    }
                }
            """,
            ensureRowContiguous: true)
    }

}
