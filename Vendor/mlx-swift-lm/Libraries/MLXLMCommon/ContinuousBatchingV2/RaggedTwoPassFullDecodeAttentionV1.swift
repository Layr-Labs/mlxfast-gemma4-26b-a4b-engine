// RaggedTwoPassFullDecodeAttentionV1.swift
//
// Batch-wide two-pass vector attention for the exact Gemma 4 FULL-attention
// decode shape: 2 KV heads, GQA 8, D=512, B=8, qL=1. Sliding ATT-006 is
// 8 KV heads / D=256 / N=1024 fixed; this helper is the global-layer sibling.
// N is a runtime input so decode steps 1024...1152 share one compiled kernel
// per block-count, matching MLX's sdpa_vector_2pass partition.

import Foundation
import MLX
import MLXFast

enum CBv2RaggedTwoPassFullDecodeAttentionV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RAGGED_TWO_PASS_FULL_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 2
    private static let gqa = 8
    private static let headDim = 512
    private static let minSequence = 1024
    private static let maxSequence = 1152

    /// MLX `sdpa_vector_2pass` block count at qL=1, GQA=8, this N.
    static func mlxBlocks(sequenceLength n: Int) -> Int {
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"],
            let value = Int(raw), value > 0
        {
            return value
        }
        let nSimds = gqa * 1
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s":
            if n > 1024 && nSimds > 4 {
                if n <= 8192 { return 128 }
                if n <= 32768 { return 256 }
                if n <= 65536 { return 512 }
                return 1024
            }
            return 64
        case "d":
            var blocks = 128
            if nSimds <= 2 && n > 8192 {
                blocks = 256
            } else if nSimds >= 6 {
                if n >= 16384 && n < 65536 {
                    blocks = 512
                } else if n >= 65536 {
                    blocks = 1024
                }
            }
            return blocks
        default:
            return nSimds >= 4 ? 64 : 32
        }
    }

    private static let passASource = """
        constexpr int simd_width = 32;
        constexpr int values_per_lane = D / simd_width;

        const int kv_head = int(threadgroup_position_in_grid.x);
        const int batch_index = int(threadgroup_position_in_grid.y);
        const int block = int(threadgroup_position_in_grid.z);
        const int query_head_in_group = int(thread_position_in_threadgroup.y);
        const int query_head = GQA * kv_head + query_head_in_group;
        const int batch_head = batch_index * 16 + query_head;
        const int lane = int(thread_index_in_simdgroup);
        const int N = int(seq_len[0]);

        const device T* keys = k0;
        const device T* values = v0;
        switch (batch_index) {
            case 1: keys = k1; values = v1; break;
            case 2: keys = k2; values = v2; break;
            case 3: keys = k3; values = v3; break;
            case 4: keys = k4; values = v4; break;
            case 5: keys = k5; values = v5; break;
            case 6: keys = k6; values = v6; break;
            case 7: keys = k7; values = v7; break;
            default: break;
        }

        const device T* query =
            queries + batch_head * D + lane * values_per_lane;
        keys += kv_head * N * D + block * D + lane * values_per_lane;
        values += kv_head * N * D + block * D + lane * values_per_lane;
        device T* partial = partials
            + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
        device float* sum_out = sums + batch_head * BLOCKS + block;
        device float* max_out = maxs + batch_head * BLOCKS + block;

        thread float q[values_per_lane];
        thread float accumulator[values_per_lane];
        for (int element = 0; element < values_per_lane; ++element) {
            q[element] = 1.0f * float(query[element]);
            accumulator[element] = 0.0f;
        }

        float max_score = -3.402823466e+38F;
        float sum_exp_score = 0.0f;
        for (int token = block; token < N; token += BLOCKS) {
            float score = 0.0f;
            for (int element = 0; element < values_per_lane; ++element) {
                score += q[element] * float(keys[element]);
            }
            score = simd_sum(score);

            const float new_max = max(max_score, score);
            const float old_factor = fast::exp(max_score - new_max);
            const float score_factor = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * old_factor + score_factor;
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = accumulator[element] * old_factor
                    + score_factor * float(values[element]);
            }

            keys += BLOCKS * D;
            values += BLOCKS * D;
        }

        if (lane == 0) {
            sum_out[0] = sum_exp_score;
            max_out[0] = max_score;
        }
        for (int element = 0; element < values_per_lane; ++element) {
            partial[element] = T(accumulator[element]);
        }
        """

    private static let passBSource = """
        constexpr int simd_width = 32;
        constexpr int values_per_lane = D / simd_width;

        const int batch_head = int(threadgroup_position_in_grid.x);
        const int simdgroup = int(simdgroup_index_in_threadgroup);
        const int lane = int(thread_index_in_simdgroup);

        partials += batch_head * BLOCKS * D
            + simdgroup * D + lane * values_per_lane;
        sums += batch_head * BLOCKS;
        maxs += batch_head * BLOCKS;
        out += batch_head * D + simdgroup * values_per_lane;

        thread float accumulator[values_per_lane];
        for (int element = 0; element < values_per_lane; ++element) {
            accumulator[element] = 0.0f;
        }
        threadgroup float partial_outputs[simd_width * simd_width];

        float sum_exp_score = 0.0f;
        float max_score = -3.402823466e+38F;
        for (int block = 0; block < BLOCKS / simd_width; ++block) {
            max_score = max(max_score, maxs[lane + simd_width * block]);
        }
        max_score = simd_max(max_score);

        for (int block = 0; block < BLOCKS / simd_width; ++block) {
            const float factor = fast::exp(
                maxs[lane + simd_width * block] - max_score);
            sum_exp_score += factor * sums[lane + simd_width * block];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int block = 0; block < BLOCKS / simd_width; ++block) {
            const float factor = fast::exp(maxs[simdgroup] - max_score);
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] += factor * float(partials[element]);
            }
            maxs += simd_width;
            sums += simd_width;
            partials += simd_width * D;
        }

        for (int element = 0; element < values_per_lane; ++element) {
            partial_outputs[lane * simd_width + simdgroup] = accumulator[element];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accumulator[element] = simd_sum(
                partial_outputs[simdgroup * simd_width + lane]);
            accumulator[element] = sum_exp_score == 0.0f
                ? accumulator[element]
                : accumulator[element] / sum_exp_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            for (int element = 0; element < values_per_lane; ++element) {
                out[element] = T(accumulator[element]);
            }
        }
        """

    private static let kernelsByBlocks: [Int: (MLXFast.MLXFastKernel, MLXFast.MLXFastKernel)] = {
        var table: [Int: (MLXFast.MLXFastKernel, MLXFast.MLXFastKernel)] = [:]
        for blockCount in [32, 64, 128] {
            let passA = MLXFast.metalKernel(
                name: "cbv2_ragged8_sdpa_2pass_a_bf16_d512_g8_b\(blockCount)_v1",
                inputNames: [
                    "queries",
                    "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
                    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "seq_len",
                ],
                outputNames: ["partials", "sums", "maxs"],
                source: passASource,
                ensureRowContiguous: true)
            let passB = MLXFast.metalKernel(
                name: "cbv2_ragged8_sdpa_2pass_b_bf16_d512_b\(blockCount)_v1",
                inputNames: ["partials", "sums", "maxs"],
                outputNames: ["out"],
                source: passBSource,
                ensureRowContiguous: true)
            table[blockCount] = (passA, passB)
        }
        return table
    }()

    static func attend(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        scale: Float
    ) -> MLXArray? {
        guard enabled,
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.count == batch,
            values.count == batch
        else { return nil }

        let sequenceLength = keys[0].dim(2)
        guard sequenceLength >= minSequence, sequenceLength <= maxSequence else {
            return nil
        }
        let blocks = mlxBlocks(sequenceLength: sequenceLength)
        guard blocks > 0, blocks.isMultiple(of: 32), let pair = kernelsByBlocks[blocks] else {
            return nil
        }

        for index in 0 ..< batch {
            let key = keys[index]
            let value = values[index]
            guard key.dtype == .bfloat16,
                value.dtype == .bfloat16,
                key.shape == [1, kvHeads, sequenceLength, headDim],
                value.shape == key.shape
            else { return nil }
        }

        let seq = MLXArray([Int32(sequenceLength)])
        let inputs: [MLXArray] = [queries] + keys + values + [seq]
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = pair.0(
            inputs,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("GQA", gqa),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch * gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )

        return pair.1(
            passA,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
            ],
            grid: (batch * queryHeads * 1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
