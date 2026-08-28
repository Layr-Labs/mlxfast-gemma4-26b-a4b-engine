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
            int slot = int((start + block) % N);
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

                slot += BLOCKS;
                if (slot >= N) slot -= N;
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
        name: "cbv2_ragged8_ringwrite_sdpa_2pass_a_bf16_d256_g2_b\(blocks)_offsets_v2",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "position_offsets", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["partials", "sums", "maxs", "fence"],
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
            const device T* new_key = new_keys
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            const device T* new_value = new_values
                + (batch_index * KV_HEADS + kv_head) * D + lane * values_per_lane;
            // The cache bank already owns one shared PRE-step absolute
            // position tensor. On a full ring, the post-write physical start
            // is `(absolute_offset + 1) % N`; derive it in-register instead
            // of uploading the same [B] uint32 start tensor in every layer.
            const uint ring_start =
                (uint(position_offsets[batch_index]) + 1u) % uint(N);
            const uint write_slot = (ring_start + uint(N - 1)) % uint(N);
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

            uint slot = (ring_start + uint(block)) % uint(N);
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
        geometries: [CBv2RingGeometry?],
        preStepPositionOffsets: MLXArray?,
        previousWriteFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard enabled,
            let preStepPositionOffsets,
            preStepPositionOffsets.dtype == .int32,
            preStepPositionOffsets.shape == [batch],
            blocks > 0,
            blocks.isMultiple(of: 32),
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

        // HOST-SHAPE-001. The rows' K/V here are their PERSISTENT ring
        // allocations, whose dtype and shape were settled once at allocation
        // and cannot have moved since (see `CBv2RingGeometry`). Re-deriving
        // them per step meant, for each of the eight rows, two `.shape` calls
        // plus the two `[Int]` literals they were compared against -- and this
        // predicate runs on every sliding layer of every decode step, so the
        // ruled cohort paid tens of thousands of throwaway array allocations
        // per generation to reconfirm a constant. When every row carries the
        // attestation and it agrees with this kernel's compiled geometry, the
        // integer comparisons below settle it. Any unattested row (non-bf16
        // storage, an unallocated ring, a count mismatch) falls through to the
        // original per-row revalidation, and a failure there still returns nil
        // with the write NOT performed.
        var attested = geometries.count == batch
        if attested {
            for geometry in geometries {
                guard let geometry,
                    geometry.kvHeads == kvHeads,
                    geometry.length == sequenceLength,
                    geometry.headDim == headDim
                else {
                    attested = false
                    break
                }
            }
        }
        if !attested {
            for index in 0 ..< batch {
                let key = keys[index]
                let value = values[index]
                guard key.dtype == .bfloat16,
                    value.dtype == .bfloat16,
                    key.shape == [1, kvHeads, sequenceLength, headDim],
                    value.shape == key.shape
                else { return nil }
            }
        }

        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = fusedRingPassAKernel(
            [queries] + keys + values
                + [preStepPositionOffsets, newKeys, newValues, previousWriteFence],
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
