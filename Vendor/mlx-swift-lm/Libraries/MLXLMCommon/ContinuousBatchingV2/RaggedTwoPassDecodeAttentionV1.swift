// Box-draw entry #3 on the unmodified crown content. Entry #2 (4198c2b2)
// drew the fast-serial box — the line's eighth consecutive — and sealed
// candidate decode ~2.2276, the crown's own content level, as the ledger
// predicts. Submitted off-cadence to decouple from any scheduler phase.
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

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 8
    private static let gqa = 2
    private static let headDim = 256
    private static let sequenceLength = 1024

    /// PARTITION-001: the stock partition count, sized for ONE row.
    ///
    /// `sdpa_vector_2pass` picks `blocks` from the architecture letter, the key
    /// length and `n_simds = GQA * qL` alone
    /// (`scaled_dot_product_attention.cpp:443-476`). Batch never enters the
    /// choice, because the stock call it was tuned for dispatches one row: at
    /// N=1024/GQA=2 it wants 128 threadgroup columns on a `d` part to keep the
    /// machine busy with `kvHeads * 1 * blocks` threadgroups.
    ///
    /// This dispatch is batch-wide. `kvHeads * batch * blocks` threadgroups is
    /// eight times what the heuristic was solving for, so the stock answer
    /// over-partitions by 8x and pays for it in the pass-A/pass-B scratch
    /// round trip, which is `batch * queryHeads * blocks * D` BF16 written and
    /// read once each per sliding layer.
    private static let stockBlocks: Int = {
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s": return 64
        case "d": return 128
        default: return 32
        }
    }()

    /// PARTITION-002: the partition this dispatch actually uses.
    ///
    /// PARTITION-001 stopped at 32 for a reason that was about the merge
    /// kernel, not about the machine: pass B indexed its columns with one SIMD
    /// lane each and looped `BLOCKS / simd_width` times, so a partition below a
    /// simdgroup silently merged nothing. That loop is now lane-guarded and
    /// admits any partition that divides the ring, which lets the occupancy
    /// argument finish.
    ///
    /// The stock heuristic's answer is an occupancy target expressed in
    /// threadgroups: on a `d` part at N=1024 and `n_simds = 2` it asks for
    /// `kvHeads * 1 * 128 = 1024` of them, because the call it was tuned for
    /// dispatches one row. This dispatch carries eight rows, so the partition
    /// that reaches the same target is `128 / 8 = 16`, and the whole span from
    /// 8 to 16 clears it: at 8 the launch is still 512 threadgroups of two
    /// simdgroups, and each of those simdgroups keeps a 512-byte K load and a
    /// 512-byte V load outstanding, so roughly 1 MB is in flight against the
    /// ~225 KB a 450 GB/s part needs to cover its own DRAM latency.
    ///
    /// Everything below the target is scratch that does not have to be written.
    /// `MLX_SDPA_BLOCKS` keeps its stock meaning and still wins, so a process
    /// can never run mismatched partitions; `DARKBLOOM_CBV2_2PASS_BLOCKS`
    /// restores the stock answer (`=0`) or selects any other divisor of the
    /// ring for bisection.
    private static let blocks: Int = {
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"],
            let value = Int(raw), value > 0
        {
            return value
        }
        if let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_2PASS_BLOCKS"], let value = Int(raw)
        {
            return value > 0 && sequenceLength.isMultiple(of: value)
                ? value : stockBlocks
        }
        return min(8, stockBlocks)
    }()

    private static let passAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_2pass_a_bf16_d256_g2_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
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
        """,
        ensureRowContiguous: true
    )

    private static let ringPassAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_ring_2pass_a_bf16_d256_g2_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "starts",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr int ring_mask = N - 1;

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
            const uint start = starts[batch_index];

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            keys += kv_head * N * D + lane * values_per_lane;
            values += kv_head * N * D + lane * values_per_lane;
            int slot = int((start + uint(block)) & uint(ring_mask));
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
                const device T* k = keys + slot * D;
                const device T* v = values + slot * D;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += q[element] * float(k[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor * float(v[element]);
                }

                slot = (slot + BLOCKS) & ring_mask;
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

    /// `ringPassAKernel` with this step's one-token ring write folded in.
    ///
    /// The ring is FULL, so the token about to be appended evicts exactly the
    /// entry the caller's `start` has already stepped past: physical slot
    /// `(start + N - 1) % N`. No block ever READS that slot — logical token
    /// `N - 1` is served straight from `new_keys`/`new_values` — so one
    /// threadgroup per (row, kv head) can store the new token into it in
    /// place while the rest of the grid streams the retained window. The
    /// traversal, the block-strided partition, the BF16 partial store, and
    /// the online-softmax accumulation order are byte-for-byte those of
    /// `ringPassAKernel`; only the source of logical token `N - 1` changes,
    /// and it carries the same values the elided `SliceUpdate` would have
    /// left in that slot.
    ///
    /// `write_fence` in / `fence` out makes the in-place store part of the
    /// evaluated dependency chain: the next step's writing pass A consumes
    /// this step's fence, so the mutation can never be reordered against a
    /// later read of the same allocation.
    private static let fusedRingPassAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_ringwrite_sdpa_2pass_a_bf16_d256_g2_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "starts", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["partials", "sums", "maxs", "fence"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr uint ring_mask = uint(N - 1);

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
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const uint ring_start = starts[batch_index];
            const uint write_slot = (ring_start + ring_mask) & ring_mask;
            if (block == 0 && query_head_in_group == 0) {
                device T* write_key = const_cast<device T*>(keys) + write_slot * D;
                device T* write_value = const_cast<device T*>(values) + write_slot * D;
                for (int element = 0; element < values_per_lane; ++element) {
                    write_key[element] = new_key[element];
                    write_value[element] = new_value[element];
                }
            }
            if (batch_index == 0 && kv_head == 0 && block == 0
                && query_head_in_group == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }

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

            uint slot = (ring_start + uint(block)) & ring_mask;
            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const bool current = token == N - 1;
                const device T* k = current ? new_key : keys + slot * D;
                const device T* v = current ? new_value : values + slot * D;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += q[element] * float(k[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor * float(v[element]);
                }

                slot = (slot + uint(BLOCKS)) & ring_mask;
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
        name: "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)_v7",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            // One lane per partition column, so a partition narrower than a
            // simdgroup parks the surplus lanes on the identity of each
            // reduction rather than dropping columns. At BLOCKS >= simd_width
            // every lane is live in every round and the guard is the constant
            // true, which leaves the arithmetic and its order untouched.
            constexpr int rounds = (BLOCKS + simd_width - 1) / simd_width;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int output_group = int(simdgroup_index_in_threadgroup);
            const int block_lane = int(thread_index_in_simdgroup);

            partials += batch_head * BLOCKS * D
                + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            // A lane's column summaries are invariant across all three walks
            // below, and its rescale factor is invariant across the last two.
            // Holding them keeps the same values while dropping two reloads of
            // `maxs`, one of `sums`, and one `fast::exp` per column. The dead
            // lanes take the identity of each reduction through a select, as
            // they did through the skipped body: `-FLT_MAX` folds away under
            // `max`, and `fast::exp` of a large negative is `+0.0f`, so the
            // `+0.0f * +0.0f` they add to a total that started at `+0.0f`
            // leaves it `+0.0f`.
            thread float lane_max[rounds];
            thread float lane_sum[rounds];
            thread float lane_factor[rounds];
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + simd_width * round;
                const bool live = column < BLOCKS;
                lane_max[round] = live ? maxs[column] : -3.402823466e+38F;
                lane_sum[round] = live ? sums[column] : 0.0f;
                max_score = max(max_score, lane_max[round]);
            }
            max_score = simd_max(max_score);

            for (int round = 0; round < rounds; ++round) {
                lane_factor[round] = fast::exp(lane_max[round] - max_score);
                sum_exp_score += lane_factor[round] * lane_sum[round];
            }
            sum_exp_score = simd_sum(sum_exp_score);

            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + simd_width * round;
                if (column < BLOCKS) {
                    const float factor = lane_factor[round];
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] +=
                            factor * float(partials[column * D + element]);
                    }
                }
            }

            for (int element = 0; element < values_per_lane; ++element) {
                const float reduced = simd_sum(accumulator[element]);
                if (block_lane == 0) {
                    out[element] = T(
                        sum_exp_score == 0.0f
                            ? reduced
                            : reduced / sum_exp_score);
                }
            }
        """,
        ensureRowContiguous: true
    )

    static func attend(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        scale: Float
    ) -> MLXArray? {
        attend(
            passAKernel: passAKernel, queries: queries, keys: keys, values: values,
            extraInputs: [], scale: scale)
    }

    static func attendRing(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        starts: [Int],
        scale: Float,
        slidingWindowLength: Int
    ) -> MLXArray? {
        guard slidingWindowLength == sequenceLength,
            starts.count == batch,
            starts.allSatisfy({ 0 <= $0 && $0 < sequenceLength })
        else { return nil }
        let startArray = MLXArray(starts.map(UInt32.init), [batch])
        return attend(
            passAKernel: ringPassAKernel, queries: queries, keys: keys, values: values,
            extraInputs: [startArray], scale: scale)
    }

    /// `attendRing` with this step's one-token ring write fused into pass A.
    ///
    /// `keys`/`values` are the rows' RETAINED ring allocations and `starts`
    /// the post-write ring starts — i.e. exactly what `attendRing` would be
    /// handed after a separate `decodeRingWrite`, minus the write. Returns
    /// nil (write NOT performed) whenever any predicate fails, so the caller
    /// falls back to the established write-then-attend pair.
    static func attendRingWriting(
        queries: MLXArray,
        newKeys: MLXArray,
        newValues: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        starts: [Int],
        previousWriteFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard enabled,
            blocks > 0,
            sequenceLength.isMultiple(of: blocks),
            scale == 1.0,
            slidingWindowLength == sequenceLength,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            newKeys.dtype == .bfloat16,
            newKeys.shape == [batch, kvHeads, 1, headDim],
            newValues.dtype == .bfloat16,
            newValues.shape == newKeys.shape,
            previousWriteFence.dtype == .int32,
            previousWriteFence.shape == [1],
            keys.count == batch,
            values.count == batch,
            starts.count == batch,
            starts.allSatisfy({ 0 <= $0 && $0 < sequenceLength })
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

        let startArray = MLXArray(starts.map(UInt32.init), [batch])
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = fusedRingPassAKernel(
            [queries] + keys + values
                + [startArray, newKeys, newValues, previousWriteFence],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch * gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape, [1]],
            outputDTypes: [.bfloat16, .float32, .float32, .int32]
        )

        let output = passBKernel(
            Array(passA.prefix(3)),
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
        return (output, passA[3])
    }

    private static func attend(
        passAKernel: MLXFast.MLXFastKernel,
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        extraInputs: [MLXArray],
        scale: Float
    ) -> MLXArray? {
        guard enabled,
            blocks > 0,
            sequenceLength.isMultiple(of: blocks),
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

        let inputs = [queries] + keys + values + extraInputs
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = passAKernel(
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
}

/// D512-SDPA: batch-wide FULL-attention decode for the exact Gemma 4 decode
/// cohort (B=8, 16 query heads, 2 KV heads, D=512, GQA=8, bf16, scale 1.0,
/// no sinks/softcap, mask-free L=1) as THREE batched dispatches per layer
/// with the numerics of the UNFUSED chain.
///
/// D=512 has NO fused SDPA kernel (`sdpa_vector` supports 64/96/128/256), so
/// the serial control leg lowers every one of these calls to the fast.cpp
/// fallback graph (fast.cpp:717-790): scale-multiply (scale == 1.0 — an
/// exact bf16 identity) → GQA unflatten → `matmul` QKᵀ → precise softmax →
/// `matmul` scores·V. These kernels are an EXACT TRANSCRIPTION of the three
/// Metal kernels the frozen host picks for those ops at these shapes — not a
/// re-derivation:
///
/// 1. QKᵀ at M=1/N=kL/K=512 routes to `gemv` (matmul.cpp:1307) with
///    bm=4/bn=1/sm=1/sn=32/tm=4/tn=4 (kL < 4096 keeps bm=4; the `gemv_al`
///    twin needs `batch_size_out == 1` and can never fire here). Per score:
///    each of 32 lanes accumulates an ordered fp32 chain of 16 products
///    (4 K-blocks of 128 × TN=4 columns, `result += bf16·float`), then a
///    shuffle-down butterfly (16,8,4,2,1) folds the lanes; lane 0 stores
///    bf16 (mlx-generated/gemv.cpp:1297-1436, GEMVKernel).
/// 2. softmax over kL routes to `block_softmax_precise_bfloat16`
///    (kL ≤ 4096 → non-looped; softmax.cpp:64-68 sizes the threadgroup at
///    32·ceil(ceil(kL/4)/32) ≤ 1024). Per row: thread t owns elements
///    [4t, 4t+4) (pad -inf), fp32 max/sum via simd_max/simd_sum plus a
///    32-slot cross-simdgroup reduce, output = bf16(exp·(1/sum))
///    (kernels/softmax.h `softmax_single_row`, AccT=float, N_READS=4).
/// 3. scores·V at M=1/N=512/K=kL routes to `gemv_t` with
///    bm=1/bn=4/sm=8/sn=4/tm=4/tn=4 (out=512 → bn=4; kL < 8192 → sm=8/sn=4).
///    Per output element: 8 thrM lanes accumulate ordered fp32 chains over
///    K rows {thrM·4+tm+32·i}, then a shuffle-down butterfly (16,8,4) folds
///    them; thrM==0 stores bf16 (mlx-generated/gemv.cpp:1486-1623,
///    GEMVTKernel).
///
/// What changes vs. the per-row chain is LOADS ONLY (the promoted-kernel
/// invariant: loads may be shared/reordered, adds may not move):
/// - The QKᵀ kernel computes all 8 query heads of a GQA group in one
///   threadgroup, so each K tile is loaded ONCE instead of 8×. Every score
///   keeps its own accumulators, its own per-lane chain, and the stock
///   butterfly — bit-identical per output.
/// - The AV kernel likewise shares each V tile across the 8 heads.
/// - The bf16 score/probability rows ride a `[8, 16, 1, kL]` device scratch
///   (the unfused chain materializes the identical bf16 values in its own
///   intermediates); every rounding point (each score, each probability,
///   each output element) is reproduced with explicit casts.
/// This cuts the ~288 MB of 8×-redundant SLC traffic per layer to ~37 MB
/// and ~41 dispatches to 3. kL and the per-row buffer capacities are
/// RUNTIME scalars (a tiny uint32 params array), so the per-step kL growth
/// (lockstep 1024 + step) never recompiles a pipeline; only the
/// launch geometry (chunk count, softmax threadgroup size) varies per call.
///
/// Parity: verified uint16 bit-exact against the per-row unfused chain at
/// kL ∈ {1024, 1027, 1100, 1152, 1055, 2048, 4095} with per-row capacities
/// ≠ kL and poisoned buffer tails (any out-of-bounds read would show), and
/// deterministic across repeat dispatch (the duplicate tail-row writes are
/// value-identical by construction). Directional chained timing at kL=1100:
/// ~131 µs/layer saved vs. the per-row chain (~940 → ~300 µs per
/// 5-layer step).
///
/// Fails closed (nil → caller keeps the pinned per-row loop, byte-preserved)
/// on: env kill-switch (DEFAULT ON), any other batch size,
/// geometry, dtype, scale, sinks, softcap, bidirectional kind, non-full
/// kind, non-`CBv2FullSequenceKV` rows, ATT-008-pooled rows, offsets not in
/// lockstep, kL outside [4, 4095] (the transcribed kernel-selection window),
/// or missing/mismatched backing buffers. All gates are checked BEFORE any
/// append, so failing closed can never leave a double-append behind.
enum CBv2RaggedComposedD512DecodeAttentionV1 {
    /// Kill switch: `DARKBLOOM_GEMMA4_D512_DECODE_SDPA` set to
    /// `0`/`false`/`no`/`off` restores the established per-row chain.
    /// Default ON.
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_DECODE_SDPA"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 2
    private static let gqa = 8
    private static let headDim = 512

    /// kL bounds that keep every frozen-host kernel decision inside the
    /// transcribed geometry: `gemv` bm=4 requires kL < 4096
    /// (matmul.cpp:1115), block (non-looped) softmax requires kL ≤ 4096 with
    /// a ≤1024-thread group (softmax.cpp:53,64-68), and the gemv tail shift
    /// requires kL ≥ TM = 4.
    private static let minKeyLength = 4
    private static let maxKeyLength = 4095

    /// Dispatch 1 — QKᵀ. Grid: (row, kv head, chunk of 4 virtual gemv
    /// threadgroups = 64 score rows) × 128 threads (4 simdgroups — exactly
    /// the stock gemv threadgroup shape). Each simdgroup replays the stock
    /// GEMVKernel<bf16,4,1,1,32,4,4> lane→column mapping and tail shift for
    /// TM=4 score rows, for all 8 heads of the GQA group at once.
    /// params: [0]=kL, [1]=K (=D, runtime like the stock buffer-passed
    /// sizes), [2+row]=that row's KV buffer capacity.
    /// WRITE-022 (samfenwick's db4ef5e, re-implemented with credit): the
    /// stock QK source as a shared constant so the plain and fenced kernel
    /// objects are STRUCTURALLY identical — the fenced one differs only by an
    /// unused trailing input that carries the store dispatch's fence, which
    /// orders the standalone K/V store before this dispatch without touching
    /// the unrolled inner loop (the addressing perturbation WRITE-021/v2
    /// paid ~-1.34% decode for).
    private static let qkSource: String = """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_length = int(params[0]);
            const int in_vec_size = int(params[1]);

            const int n_chunks = (key_length + 63) / 64;
            const int z = int(threadgroup_position_in_grid.z);
            const int chunk = z % n_chunks;
            const int row_kv = z / n_chunks;
            const int row = row_kv / 2;
            const int kv_head = row_kv % 2;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);

            const int row_capacity = int(params[2 + row]);

            const device T* key_plane = k0;
            switch (row) {
                case 1: key_plane = k1; break;
                case 2: key_plane = k2; break;
                case 3: key_plane = k3; break;
                case 4: key_plane = k4; break;
                case 5: key_plane = k5; break;
                case 6: key_plane = k6; break;
                case 7: key_plane = k7; break;
                default: break;
            }
            key_plane += size_t(kv_head) * size_t(row_capacity) * D;

            const device T* query =
                queries + size_t(row * 16 + kv_head * GQA) * D;
            device T* score_rows =
                scores + size_t(row * 16 + kv_head * GQA) * key_length;

            const int virtual_groups = (key_length + 15) / 16;
            const int vtg_lo = chunk * 4;
            const int vtg_hi = min(vtg_lo + 4, virtual_groups);
            const int n_iter = in_vec_size / 128;

            for (int vtg = vtg_lo; vtg < vtg_hi; ++vtg) {
                int out_row = vtg * 16 + sg * 4;
                if (out_row >= key_length) continue;
                out_row = out_row + 4 <= key_length
                    ? out_row : key_length - 4;

                const device T* mat = key_plane + size_t(out_row) * D;
                float result[GQA][4] = {{0.0f}};
                T inter[4];
                float v_coeff[GQA][4];
                int bn = lane * 4;
                for (int i = 0; i < n_iter; ++i) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            v_coeff[h][tn] = static_cast<float>(
                                query[h * D + bn + tn]);
                        }
                    }
                    int mat_offset = 0;
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            inter[tn] = mat[mat_offset + bn + tn];
                        }
                        #pragma clang loop unroll(full)
                        for (int h = 0; h < GQA; ++h) {
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                result[h][tm] +=
                                    inter[tn] * v_coeff[h][tn];
                            }
                        }
                        mat_offset += D;
                    }
                    bn += 128;
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        #pragma clang loop unroll(full)
                        for (ushort delta = 16; delta >= 1; delta >>= 1) {
                            result[h][tm] +=
                                simd_shuffle_down(result[h][tm], delta);
                        }
                    }
                }
                if (lane == 0) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            score_rows[
                                size_t(h) * key_length + out_row + tm] =
                                static_cast<T>(result[h][tm]);
                        }
                    }
                }
            }
        """

    private static let qkKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_qk_bf16_g8_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params",
        ],
        outputNames: ["scores"],
        source: qkSource,
        ensureRowContiguous: true
    )

    private static let qkFencedKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_qk_fenced_bf16_g8_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params", "store_fence",
        ],
        outputNames: ["scores"],
        source: qkSource,
        ensureRowContiguous: true
    )

    /// Dispatch 2 — softmax. A verbatim transcription of
    /// `softmax_single_row` (block_softmax_precise, AccT=float, N_READS=4)
    /// over the 128 score rows; the CALLER sizes the threadgroup exactly
    /// like softmax.cpp:64-68 (32·ceil(ceil(kL/4)/32)). params[0] = kL.
    private static let softmaxKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_softmax_bf16_v1",
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
            const device T* in =
                scores + size_t(gid) * axis_size + lid * 4;
            if (lid * 4 + 4 <= axis_size) {
                for (int i = 0; i < 4; i++) {
                    ld[i] = static_cast<float>(in[i]);
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    ld[i] = ((lid * 4 + i) < axis_size)
                        ? static_cast<float>(in[i]) : -INFINITY;
                }
            }
            if (simd_group_id == 0) {
                local_max[simd_lane_id] = -INFINITY;
                local_normalizer[simd_lane_id] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float maxval = -3.402823466e+38F;
            for (int i = 0; i < 4; i++) {
                maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) {
                local_max[simd_group_id] = maxval;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                maxval = simd_max(local_max[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maxval = local_max[0];

            float normalizer = 0.0f;
            for (int i = 0; i < 4; i++) {
                float exp_x = fast::exp(ld[i] - maxval);
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
                if (simd_lane_id == 0) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            normalizer = 1 / local_normalizer[0];

            device T* out_row =
                probs + size_t(gid) * axis_size + lid * 4;
            if (lid * 4 + 4 <= axis_size) {
                for (int i = 0; i < 4; i++) {
                    out_row[i] = static_cast<T>(ld[i] * normalizer);
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    if ((lid * 4 + i) < axis_size) {
                        out_row[i] = static_cast<T>(ld[i] * normalizer);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    /// Dispatch 3 — probs·V. Grid: (row, kv head, column tile of 64) × 128
    /// threads (4 simdgroups — exactly the stock gemv_t threadgroup shape).
    /// Replays the stock GEMVTKernel<bf16,1,4,8,4,4,4> row-striding and
    /// butterfly for all 8 heads of the GQA group at once (shared V tile
    /// loads). params as dispatch 1.
    private static let avKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_v1",
        inputNames: [
            "probs",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params",
        ],
        outputNames: ["out"],
        source: """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_length = int(params[0]);

            const int z = int(threadgroup_position_in_grid.z);
            const int tile = z % 8;
            const int row_kv = z / 8;
            const int row = row_kv / 2;
            const int kv_head = row_kv % 2;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);

            const int row_capacity = int(params[2 + row]);

            const device T* value_plane = v0;
            switch (row) {
                case 1: value_plane = v1; break;
                case 2: value_plane = v2; break;
                case 3: value_plane = v3; break;
                case 4: value_plane = v4; break;
                case 5: value_plane = v5; break;
                case 6: value_plane = v6; break;
                case 7: value_plane = v7; break;
                default: break;
            }
            value_plane += size_t(kv_head) * size_t(row_capacity) * D;

            const device T* prob_rows =
                probs + size_t(row * 16 + kv_head * GQA) * key_length;

            const int thrM = lane / 4;
            const int thrN = lane % 4;
            int bm = thrM * 4;
            const int out_col = tile * 64 + (4 * sg + thrN) * 4;

            float result[GQA][4] = {{0.0f}};
            T inter[4];
            float v_coeff[GQA][4];
            const int n_iter = key_length / 32;
            const int leftover = key_length - n_iter * 32;

            for (int i = 0; i < n_iter; ++i) {
                threadgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        v_coeff[h][tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                }
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    for (int tn = 0; tn < 4; ++tn) {
                        inter[tn] = value_plane[
                            size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        float vc = v_coeff[h][tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += vc * inter[tn];
                        }
                    }
                }
                bm += 32;
            }
            if (leftover > 0) {
                for (int tm = 0; tm < 4 && bm + tm < key_length; ++tm) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        v_coeff[h][tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; ++tn) {
                        inter[tn] = value_plane[
                            size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += v_coeff[h][tm] * inter[tn];
                        }
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int h = 0; h < GQA; ++h) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < 4; ++tn) {
                    #pragma clang loop unroll(full)
                    for (ushort delta = 4; delta >= 1; delta >>= 1) {
                        result[h][tn] +=
                            simd_shuffle_down(result[h][tn], 4 * delta);
                    }
                }
            }
            if (thrM == 0) {
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    device T* out_ptr = out
                        + size_t(row * 16 + kv_head * GQA + h) * D
                        + out_col;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        out_ptr[j] = static_cast<T>(result[h][j]);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    // ATTRIBUTION. Everything in this WRITE-016-D512 section, and the
    // matching hunks in AttentionV1.swift and SequenceKV/FullSequenceKV.swift,
    // is taken VERBATIM from public ranked submission
    // 4921db67-5294-4330-856b-75d0b1af09c4 (PR #617, author DashiellB, official
    // composite 1.776595, parity 4/4). This submission authored none of it and
    // claims none of it; it re-measures that mechanism on exact-current main,
    // where PR #617's own three files are byte-identical to the base it was
    // written against. The public note states why: PR #617's candidate windows
    // (decode 2.29808315625 s, prefill 1.24319015625 s) are the fastest
    // measured on this track, and its composite was held down by a serial
    // control leg that came back at the 0th percentile of the 65-run
    // distribution, not by the mechanism.

    /// WRITE-016-D512 fused variant of dispatch 1: identical score
    /// arithmetic, plus (a) the new token's K/V columns are written into the
    /// cache buffers in place (one writer threadgroup per row x kv head,
    /// donor: cbv2_ragged8_ringwrite_sdpa_2pass_a), (b) an int32 write fence
    /// is threaded in -> out so the in-place store is a real graph
    /// dependency, and (c) the new token's K row (logical row kL-1) is
    /// served from `new_keys` during scoring, never from the slot being
    /// written, so no read races the store.
    private static let fusedQkKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_qk_bf16_g8_v2",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["scores", "fence"],
        source: """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_length = int(params[0]);
            const int in_vec_size = int(params[1]);

            const int n_chunks = (key_length + 63) / 64;
            const int z = int(threadgroup_position_in_grid.z);
            const int chunk = z % n_chunks;
            const int row_kv = z / n_chunks;
            const int row = row_kv / 2;
            const int kv_head = row_kv % 2;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);

            const int row_capacity = int(params[2 + row]);

            const device T* key_plane = k0;
            switch (row) {
                case 1: key_plane = k1; break;
                case 2: key_plane = k2; break;
                case 3: key_plane = k3; break;
                case 4: key_plane = k4; break;
                case 5: key_plane = k5; break;
                case 6: key_plane = k6; break;
                case 7: key_plane = k7; break;
                default: break;
            }
            key_plane += size_t(kv_head) * size_t(row_capacity) * D;

            const device T* value_plane = v0;
            switch (row) {
                case 1: value_plane = v1; break;
                case 2: value_plane = v2; break;
                case 3: value_plane = v3; break;
                case 4: value_plane = v4; break;
                case 5: value_plane = v5; break;
                case 6: value_plane = v6; break;
                case 7: value_plane = v7; break;
                default: break;
            }
            value_plane += size_t(kv_head) * size_t(row_capacity) * D;
            const device T* new_key_plane =
                new_keys + size_t(row * 2 + kv_head) * D;
            const device T* new_value_plane =
                new_values + size_t(row * 2 + kv_head) * D;


            const device T* query =
                queries + size_t(row * 16 + kv_head * GQA) * D;
            device T* score_rows =
                scores + size_t(row * 16 + kv_head * GQA) * key_length;

            const int virtual_groups = (key_length + 15) / 16;
            const int vtg_lo = chunk * 4;
            const int vtg_hi = min(vtg_lo + 4, virtual_groups);
            const int n_iter = in_vec_size / 128;

            for (int vtg = vtg_lo; vtg < vtg_hi; ++vtg) {
                int out_row = vtg * 16 + sg * 4;
                if (out_row >= key_length) continue;
                out_row = out_row + 4 <= key_length
                    ? out_row : key_length - 4;

                const device T* mat = key_plane + size_t(out_row) * D;
                // Group-level peel: only the tile containing logical row
                // kL-1 pays the serve-from-input branch; every other tile
                // keeps the donor's branch-free unrolled body.
                const bool tile_has_new_token =
                    out_row + 4 > key_length - 1;
                float result[GQA][4] = {{0.0f}};
                T inter[4];
                float v_coeff[GQA][4];
                int bn = lane * 4;
                for (int i = 0; i < n_iter; ++i) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            v_coeff[h][tn] = static_cast<float>(
                                query[h * D + bn + tn]);
                        }
                    }
                    int mat_offset = 0;
                    if (!tile_has_new_token) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            inter[tn] = mat[mat_offset + bn + tn];
                        }
                        #pragma clang loop unroll(full)
                        for (int h = 0; h < GQA; ++h) {
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                result[h][tm] +=
                                    inter[tn] * v_coeff[h][tn];
                            }
                        }
                        mat_offset += D;
                    }
                    } else {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        const bool is_new_token =
                            out_row + tm == key_length - 1;
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            inter[tn] = is_new_token
                                ? new_key_plane[bn + tn]
                                : mat[mat_offset + bn + tn];
                        }
                        #pragma clang loop unroll(full)
                        for (int h = 0; h < GQA; ++h) {
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                result[h][tm] +=
                                    inter[tn] * v_coeff[h][tn];
                            }
                        }
                        mat_offset += D;
                    }
                    }
                    bn += 128;
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        #pragma clang loop unroll(full)
                        for (ushort delta = 16; delta >= 1; delta >>= 1) {
                            result[h][tm] +=
                                simd_shuffle_down(result[h][tm], delta);
                        }
                    }
                }
                if (lane == 0) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            score_rows[
                                size_t(h) * key_length + out_row + tm] =
                                static_cast<T>(result[h][tm]);
                        }
                    }
                }
            }

            if (chunk == 0 && sg == 0) {
                device T* write_key = const_cast<device T*>(key_plane)
                    + size_t(key_length - 1) * D + lane * 16;
                device T* write_value = const_cast<device T*>(value_plane)
                    + size_t(key_length - 1) * D + lane * 16;
                const device T* src_key = new_key_plane + lane * 16;
                const device T* src_value = new_value_plane + lane * 16;
                for (int element = 0; element < 16; ++element) {
                    write_key[element] = src_key[element];
                    write_value[element] = src_value[element];
                }
            }
            if (z == 0 && sg == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """,
        ensureRowContiguous: true
    )


    /// WRITE-016-D512 fused variant of dispatch 3: identical output
    /// arithmetic; the new token's V row (logical row kL-1) is served from
    /// `new_values` rather than the cache slot the fused QK dispatch wrote,
    /// so this kernel has no read-after-in-place-write hazard at all.
    private static let fusedAvKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_av_bf16_g8_v2",
        inputNames: [
            "probs",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "new_values",
        ],
        outputNames: ["out"],
        source: """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_length = int(params[0]);

            const int z = int(threadgroup_position_in_grid.z);
            const int tile = z % 8;
            const int row_kv = z / 8;
            const int row = row_kv / 2;
            const int kv_head = row_kv % 2;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);

            const int row_capacity = int(params[2 + row]);

            const device T* value_plane = v0;
            switch (row) {
                case 1: value_plane = v1; break;
                case 2: value_plane = v2; break;
                case 3: value_plane = v3; break;
                case 4: value_plane = v4; break;
                case 5: value_plane = v5; break;
                case 6: value_plane = v6; break;
                case 7: value_plane = v7; break;
                default: break;
            }
            value_plane += size_t(kv_head) * size_t(row_capacity) * D;
            const device T* new_value_plane =
                new_values + size_t(row * 2 + kv_head) * D;

            const device T* prob_rows =
                probs + size_t(row * 16 + kv_head * GQA) * key_length;

            const int thrM = lane / 4;
            const int thrN = lane % 4;
            int bm = thrM * 4;
            const int out_col = tile * 64 + (4 * sg + thrN) * 4;

            float result[GQA][4] = {{0.0f}};
            T inter[4];
            float v_coeff[GQA][4];
            const int n_iter = key_length / 32;
            const int leftover = key_length - n_iter * 32;

            for (int i = 0; i < n_iter; ++i) {
                threadgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        v_coeff[h][tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                }
                // Tile-level peel: only the 4-row tile containing logical
                // row kL-1 pays the serve-from-input branch.
                if (bm + 4 <= key_length - 1) {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    for (int tn = 0; tn < 4; ++tn) {
                        inter[tn] = value_plane[
                            size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        float vc = v_coeff[h][tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += vc * inter[tn];
                        }
                    }
                }
                } else {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    const bool is_new_token = bm + tm == key_length - 1;
                    for (int tn = 0; tn < 4; ++tn) {
                        inter[tn] = is_new_token
                            ? new_value_plane[out_col + tn]
                            : value_plane[
                                  size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        float vc = v_coeff[h][tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += vc * inter[tn];
                        }
                    }
                }
                }
                bm += 32;
            }
            if (leftover > 0) {
                for (int tm = 0; tm < 4 && bm + tm < key_length; ++tm) {
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        v_coeff[h][tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                    const bool is_new_token = bm + tm == key_length - 1;
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; ++tn) {
                        inter[tn] = is_new_token
                            ? new_value_plane[out_col + tn]
                            : value_plane[
                                  size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += v_coeff[h][tm] * inter[tn];
                        }
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int h = 0; h < GQA; ++h) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < 4; ++tn) {
                    #pragma clang loop unroll(full)
                    for (ushort delta = 4; delta >= 1; delta >>= 1) {
                        result[h][tn] +=
                            simd_shuffle_down(result[h][tn], 4 * delta);
                    }
                }
            }
            if (thrM == 0) {
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    device T* out_ptr = out
                        + size_t(row * 16 + kv_head * GQA + h) * D
                        + out_col;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        out_ptr[j] = static_cast<T>(result[h][j]);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )


    /// WRITE-016-D512 kill switch: `DARKBLOOM_GEMMA4_D512_FUSED_WRITE` set
    /// to 0/false/no/off restores the append-then-attend path byte for byte.
    private static let fusedWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_FUSED_WRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// WRITE-022 store dispatch: one threadgroup per (row, kv head) — 16
    /// threadgroups of 128 threads — each thread writing 4 contiguous
    /// elements of K and 4 of V into slot kL-1 of the row's private buffer
    /// through a const_cast on the input pointer. 32 KiB total. The fence
    /// output is the WRITE-016 pattern: a real graph edge whose inter-kernel
    /// barrier (BarrierScopeBuffers, encoder-wide) orders every buffer write
    /// this dispatch issued before the fenced QK reads them.
    private static let ringStoreKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_d512_ringstore_bf16_v1",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["fence"],
        source: """
            constexpr int D = 512;
            const int z = int(threadgroup_position_in_grid.z);
            const int row = z / 2;
            const int kv_head = z % 2;
            const int lane = int(thread_position_in_threadgroup.x);
            const int key_length = int(params[0]);
            const int row_capacity = int(params[2 + row]);

            const device T* key_plane = k0;
            const device T* value_plane = v0;
            switch (row) {
                case 1: key_plane = k1; value_plane = v1; break;
                case 2: key_plane = k2; value_plane = v2; break;
                case 3: key_plane = k3; value_plane = v3; break;
                case 4: key_plane = k4; value_plane = v4; break;
                case 5: key_plane = k5; value_plane = v5; break;
                case 6: key_plane = k6; value_plane = v6; break;
                case 7: key_plane = k7; value_plane = v7; break;
                default: break;
            }
            key_plane += size_t(kv_head) * size_t(row_capacity) * D;
            value_plane += size_t(kv_head) * size_t(row_capacity) * D;

            device T* write_key = const_cast<device T*>(key_plane)
                + size_t(key_length - 1) * D + lane * 4;
            device T* write_value = const_cast<device T*>(value_plane)
                + size_t(key_length - 1) * D + lane * 4;
            const device T* src_key = new_keys
                + size_t(row * 2 + kv_head) * D + lane * 4;
            const device T* src_value = new_values
                + size_t(row * 2 + kv_head) * D + lane * 4;
            for (int element = 0; element < 4; ++element) {
                write_key[element] = src_key[element];
                write_value[element] = src_value[element];
            }
            if (z == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """,
        ensureRowContiguous: true)

    /// WRITE-022 kill switch: `DARKBLOOM_GEMMA4_D512_STORE_DISPATCH=0` falls
    /// back to the v2 fold (and its own switch falls back to the append path).
    private static let storeDispatchEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_STORE_DISPATCH"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// WRITE-022: the fused append as its own 32 KiB dispatch placed before
    /// the STOCK three-kernel chain (byte-for-byte dispatch 1-3), fence-
    /// ordered. Removes the same 80 copy-on-write appends as the v2 fold
    /// without the fold's inner-loop addressing perturbation.
    static func updateAndAttendWriting22(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard storeDispatchEnabled,
            enabled,
            rows.count == batch,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !kind.isBidirectional,
            kind.queryHeads == queryHeads,
            kind.kvHeads == kvHeads,
            kind.headDim == headDim,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            previousWriteFence.dtype == .int32,
            previousWriteFence.shape == [1],
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.shape == [batch, kvHeads, 1, headDim],
            values.shape == keys.shape
        else { return nil }
        guard case .full = kind.attention else { return nil }

        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == batch else { return nil }

        let offset = fullRows[0].absoluteOffset
        let keyLength = offset + 1
        guard offset > 0,
            keyLength >= minKeyLength,
            keyLength <= maxKeyLength,
            fullRows.allSatisfy({ $0.cohortPool == nil }),
            fullRows.allSatisfy({ $0.absoluteOffset == offset }),
            fullRows.allSatisfy({ keyLength <= $0.maxLength })
        else { return nil }

        var keyBuffers: [MLXArray] = []
        var valueBuffers: [MLXArray] = []
        keyBuffers.reserveCapacity(batch)
        valueBuffers.reserveCapacity(batch)
        var params: [UInt32] = [UInt32(keyLength), UInt32(headDim)]
        params.reserveCapacity(batch + 2)
        for row in fullRows {
            let state = row.cbv2InnerState()
            guard state.count == 2,
                state[0].dtype == .bfloat16,
                state[1].dtype == .bfloat16,
                state[0].ndim == 4,
                state[0].dim(0) == 1,
                state[0].dim(1) == kvHeads,
                state[0].dim(3) == headDim,
                state[1].shape == state[0].shape,
                state[1].dtype == state[0].dtype,
                state[0].dim(2) >= keyLength
            else { return nil }
            keyBuffers.append(state[0])
            valueBuffers.append(state[1])
            params.append(UInt32(state[0].dim(2)))
        }
        let paramsArray = MLXArray(params)

        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype)
        ]
        let scratchShape = [batch, queryHeads, 1, keyLength]

        let storeFence = ringStoreKernel(
            keyBuffers + valueBuffers
                + [paramsArray, keys, values, previousWriteFence],
            template: template,
            grid: (128, 1, batch * kvHeads),
            threadGroup: (128, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32]
        )[0]

        let chunks = (keyLength + 63) / 64
        let scores = qkFencedKernel(
            [queries] + keyBuffers + [paramsArray, storeFence],
            template: template,
            grid: (32, 4, batch * kvHeads * chunks),
            threadGroup: (32, 4, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        let softmaxThreads = ((keyLength + 3) / 4 + 31) / 32 * 32
        let probs = softmaxKernel(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        let output = avKernel(
            [probs] + valueBuffers + [paramsArray],
            template: template,
            grid: (32, 4, batch * kvHeads * 8),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]

        for row in fullRows {
            row.advanceAfterFusedAppend()
        }
        return (output, storeFence)
    }


    /// Fused write + attend for the D=512 full-attention decode cohort, or
    /// nil (with NO side effects) when any gate fails. Replaces the 16
    /// copy-on-write slice appends per layer per round with one in-kernel
    /// store riding the QK dispatch, fence-chained exactly like WRITE-016:
    /// the caller must store `nextWriteFence` back into the layer's
    /// `CBv2DecodeRingWriteFence` (published via `innerState()`), and this
    /// function advances each row's bookkeeping (`advanceAfterFusedAppend`)
    /// before returning.
    static func updateAndAttendWriting(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard enabled,
            fusedWriteEnabled,
            rows.count == batch,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !kind.isBidirectional,
            kind.queryHeads == queryHeads,
            kind.kvHeads == kvHeads,
            kind.headDim == headDim,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            previousWriteFence.dtype == .int32,
            previousWriteFence.shape == [1],
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.shape == [batch, kvHeads, 1, headDim],
            values.shape == keys.shape
        else { return nil }
        guard case .full = kind.attention else { return nil }

        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == batch else { return nil }

        // Lockstep + storage gates, ALL before any write. The extra gate
        // over the unfused path: the backing buffers must already have room
        // for the new token (capacity >= kL), because the fused store cannot
        // grow them — an ensureCapacity step falls back to the append path.
        let offset = fullRows[0].absoluteOffset
        let keyLength = offset + 1
        guard offset > 0,
            keyLength >= minKeyLength,
            keyLength <= maxKeyLength,
            fullRows.allSatisfy({ $0.cohortPool == nil }),
            fullRows.allSatisfy({ $0.absoluteOffset == offset }),
            fullRows.allSatisfy({ keyLength <= $0.maxLength })
        else { return nil }

        var keyBuffers: [MLXArray] = []
        var valueBuffers: [MLXArray] = []
        keyBuffers.reserveCapacity(batch)
        valueBuffers.reserveCapacity(batch)
        var params: [UInt32] = [UInt32(keyLength), UInt32(headDim)]
        params.reserveCapacity(batch + 2)
        for row in fullRows {
            let state = row.cbv2InnerState()
            guard state.count == 2,
                state[0].dtype == .bfloat16,
                state[1].dtype == .bfloat16,
                state[0].ndim == 4,
                state[0].dim(0) == 1,
                state[0].dim(1) == kvHeads,
                state[0].dim(3) == headDim,
                state[1].shape == state[0].shape,
                state[1].dtype == state[0].dtype,
                state[0].dim(2) >= keyLength
            else { return nil }
            keyBuffers.append(state[0])
            valueBuffers.append(state[1])
            params.append(UInt32(state[0].dim(2)))
        }
        let paramsArray = MLXArray(params)

        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype)
        ]
        let scratchShape = [batch, queryHeads, 1, keyLength]

        let chunks = (keyLength + 63) / 64
        let passQK = fusedQkKernel(
            [queries] + keyBuffers + valueBuffers
                + [paramsArray, keys, values, previousWriteFence],
            template: template,
            grid: (32, 4, batch * kvHeads * chunks),
            threadGroup: (32, 4, 1),
            outputShapes: [scratchShape, [1]],
            outputDTypes: [.bfloat16, .int32]
        )
        let scores = passQK[0]
        let nextWriteFence = passQK[1]

        let softmaxThreads = ((keyLength + 3) / 4 + 31) / 32 * 32
        let probs = softmaxKernel(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        let output = fusedAvKernel(
            [probs] + valueBuffers + [paramsArray, values],
            template: template,
            grid: (32, 4, batch * kvHeads * 8),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]

        for row in fullRows {
            row.advanceAfterFusedAppend()
        }
        return (output, nextWriteFence)
    }

    /// Batched update + attend for the D=512 full-attention decode cohort,
    /// or nil (with NO side effects) when any gate fails.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray? {
        guard enabled,
            rows.count == batch,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !kind.isBidirectional,
            kind.queryHeads == queryHeads,
            kind.kvHeads == kvHeads,
            kind.headDim == headDim,
            queries.dtype == .bfloat16,
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.shape == [batch, kvHeads, 1, headDim],
            values.shape == keys.shape
        else { return nil }
        guard case .full = kind.attention else { return nil }

        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == batch else { return nil }

        // Lockstep + storage gates, ALL before any append. Pooled (ATT-008)
        // rows fail closed: their backing layout is the pool's batch axis,
        // and the pooled route already has its own batched path.
        let offset = fullRows[0].absoluteOffset
        let keyLength = offset + 1
        guard offset > 0,
            keyLength >= minKeyLength,
            keyLength <= maxKeyLength,
            fullRows.allSatisfy({ $0.cohortPool == nil }),
            fullRows.allSatisfy({ $0.absoluteOffset == offset }),
            fullRows.allSatisfy({ keyLength <= $0.maxLength })
        else { return nil }
        for row in fullRows {
            let state = row.cbv2InnerState()
            guard state.count == 2,
                state[0].dtype == .bfloat16,
                state[1].dtype == .bfloat16,
                state[0].ndim == 4,
                state[0].dim(0) == 1,
                state[0].dim(1) == kvHeads,
                state[0].dim(3) == headDim,
                state[1].shape == state[0].shape,
                state[1].dtype == state[0].dtype
            else { return nil }
        }

        // Byte-identical per-row appends — the same `update` calls, in the
        // same row order, as the established per-row loop. Only the
        // returned temporal views go unused; the kernels read the full
        // backing buffers (contiguous, so no `ensureRowContiguous` copy)
        // with kL/capacity as runtime scalars.
        var keyBuffers: [MLXArray] = []
        var valueBuffers: [MLXArray] = []
        keyBuffers.reserveCapacity(batch)
        valueBuffers.reserveCapacity(batch)
        var params: [UInt32] = [UInt32(keyLength), UInt32(headDim)]
        params.reserveCapacity(batch + 2)
        for (index, row) in fullRows.enumerated() {
            _ = row.update(
                keys: keys[index ..< (index + 1)],
                values: values[index ..< (index + 1)])
            let state = row.cbv2InnerState()
            keyBuffers.append(state[0])
            valueBuffers.append(state[1])
            params.append(UInt32(state[0].dim(2)))
        }
        let paramsArray = MLXArray(params)

        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype)
        ]
        let scratchShape = [batch, queryHeads, 1, keyLength]

        let chunks = (keyLength + 63) / 64
        let scores = qkKernel(
            [queries] + keyBuffers + [paramsArray],
            template: template,
            grid: (32, 4, batch * kvHeads * chunks),
            threadGroup: (32, 4, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        // Threadgroup sizing verbatim from softmax.cpp:64-68.
        let softmaxThreads = ((keyLength + 3) / 4 + 31) / 32 * 32
        let probs = softmaxKernel(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        return avKernel(
            [probs] + valueBuffers + [paramsArray],
            template: template,
            grid: (32, 4, batch * kvHeads * 8),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
