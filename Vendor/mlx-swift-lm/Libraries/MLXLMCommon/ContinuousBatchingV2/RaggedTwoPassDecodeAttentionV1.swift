// RaggedTwoPassDecodeAttentionV1.swift
//
// Batch-wide dispatch of MLX's established two-pass vector attention for the
// exact Gemma 4 sliding decode shape. Every row keeps its own contiguous K/V
// array and every query head keeps the same block-strided pass-A traversal,
// BF16 partial store, and pass-B merge order. Only eight identical row-local
// dispatch pairs become one batch-wide pair.

import Foundation
import MLX
import MLXFast

enum CBv2RaggedTwoPassDecodeAttentionV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RAGGED_TWO_PASS_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let ringEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RING_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 8
    private static let gqa = 2
    private static let headDim = 256
    private static let sequenceLength = 1024
    private static let fullKVHeads = 2
    private static let fullGQA = 8
    private static let fullHeadDim = 512

    /// Mirrors `sdpa_vector_2pass` at N=1024 and qL=1/GQA=2. Honor the same
    /// diagnostic override so a process can never run mismatched partitions.
    private static let blocks: Int = {
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"],
            let value = Int(raw), value > 0
        {
            return value
        }
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s": return 64
        case "d": return 128
        default: return 32
        }
    }()

    private static let passAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_linear_sdpa_2pass_a_bf16_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "lens",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

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
            keys += kv_head * C * D + lane * values_per_lane;
            values += kv_head * C * D + lane * values_per_lane;
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
            for (int token = block; token < int(lens[batch_index]); token += BLOCKS) {
                const device T* key = keys + token * D;
                const device T* value = values + token * D;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += q[element] * float(key[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor * float(value[element]);
                }
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

    private static let ringPassAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ring_sdpa_2pass_a_bf16_d256_g2_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "starts",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

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
            keys += kv_head * N * D + lane * values_per_lane;
            values += kv_head * N * D + lane * values_per_lane;
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

            uint slot = (starts[batch_index] + uint(block)) % uint(N);
            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const device T* key = keys + slot * D;
                const device T* value = values + slot * D;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += q[element] * float(key[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor * float(value[element]);
                }

                slot += BLOCKS;
                if (slot >= uint(N)) {
                    slot -= uint(N);
                }
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

    private static let passBKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_2pass_b_bf16_d256_b\(blocks)_v1",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
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
        """,
        ensureRowContiguous: true
    )

    static func attendRing(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        starts: [Int],
        scale: Float
    ) -> MLXArray? {
        guard enabled,
            ringEnabled,
            blocks > 0,
            blocks.isMultiple(of: 32),
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.count == batch,
            values.count == batch,
            starts.count == batch,
            starts.allSatisfy({ (0 ..< sequenceLength).contains($0) })
        else { return nil }

        for index in 0 ..< batch {
            let key = keys[index]
            let value = values[index]
            guard key.dtype == .bfloat16,
                value.dtype == .bfloat16,
                key.shape == [1, kvHeads, sequenceLength, headDim],
                value.shape == key.shape
            else { return nil }
        }

        let startArray = MLXArray(starts.map(UInt32.init))
        let inputs = [queries] + keys + values + [startArray]
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = ringPassAKernel(
            inputs,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch * gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )

        return passBKernel(
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

    static func attendFullBatch(
        queries: MLXArray, keys: [MLXArray], values: [MLXArray],
        lens: [Int], capacity: Int, scale: Float
    ) -> MLXArray? {
        guard fullBatchEnabled,
            blocks > 0, blocks.isMultiple(of: 32), scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, fullHeadDim],
            keys.count == batch, values.count == batch, lens.count == batch,
            lens.allSatisfy({ (sequenceLength ... capacity).contains($0) })
        else { return nil }

        for index in 0 ..< batch {
            let key = keys[index]
            let value = values[index]
            guard key.dtype == .bfloat16, value.dtype == .bfloat16,
                key.shape == [1, fullKVHeads, capacity, fullHeadDim],
                value.shape == key.shape
            else { return nil }
        }

        let lengths = MLXArray(lens.map(UInt32.init))
        let partialShape = [batch, queryHeads, 1, blocks, fullHeadDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        // The partition is pinned here so pass A and pass B always share it.
        let passA = passAKernel(
            [queries] + keys + values + [lengths],
            template: [
                ("T", queries.dtype), ("D", fullHeadDim), ("C", capacity),
                ("GQA", fullGQA), ("BLOCKS", blocks),
            ],
            grid: (fullKVHeads * 32, batch * fullGQA, blocks),
            threadGroup: (32, fullGQA, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        return passBKernel(
            passA,
            template: [("T", queries.dtype), ("D", fullHeadDim), ("BLOCKS", blocks)],
            grid: (batch * queryHeads * 1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[batch, queryHeads, 1, fullHeadDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    private static var fullBatchEnabled: Bool {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_FULL_BATCH_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }

    static func attend(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        scale: Float
    ) -> MLXArray? {
        guard enabled,
            blocks > 0,
            blocks.isMultiple(of: 32),
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.count == batch,
            values.count == batch
        else { return nil }

        for index in 0 ..< batch {
            let key = keys[index]
            let value = values[index]
            guard key.dtype == .bfloat16,
                value.dtype == .bfloat16,
                key.shape == [1, kvHeads, sequenceLength, headDim],
                value.shape == key.shape
            else { return nil }
        }

        let inputs = [queries] + keys + values
            + [MLXArray(Array(repeating: UInt32(sequenceLength), count: batch))]
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = passAKernel(
            inputs,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("C", sequenceLength),
                ("GQA", gqa),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch * gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )

        return passBKernel(
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
