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
        name: "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)_v2",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int output_group = int(simdgroup_index_in_threadgroup);
            const int block_lane = int(thread_index_in_simdgroup);

            partials += batch_head * BLOCKS * D
                + block_lane * D + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                max_score = max(
                    max_score, maxs[block_lane + simd_width * block]);
            }
            max_score = simd_max(max_score);

            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                const float factor = fast::exp(
                    maxs[block_lane + simd_width * block] - max_score);
                sum_exp_score +=
                    factor * sums[block_lane + simd_width * block];
            }
            sum_exp_score = simd_sum(sum_exp_score);

            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                const float factor = fast::exp(maxs[block_lane] - max_score);
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] +=
                        factor * float(partials[element]);
                }
                maxs += simd_width;
                sums += simd_width;
                partials += simd_width * D;
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

/// D512-SDPA: batch-wide FULL-attention decode for the exact Gemma 4 decode
/// cohort (B=8, 16 query heads, 2 KV heads, D=512, GQA=8, bf16, scale 1.0,
/// no sinks/softcap, mask-free L=1).
///
/// The AOT metallib now exports `sdpa_vector_*_512_512` and
/// `sdpa_vector_2pass_{1,2}_*_512`, but the frozen host gate
/// (`sdpa_vector_supported_head_dim`) still only lists 64/96/128/256, so
/// `MLXFast.scaledDotProductAttention` will not pick D=512. This helper
/// launches the same kernels the host would: `sdpa_vector_2pass_1` then
/// `sdpa_vector_2pass_2` when `kL >= 1024` on M-series (else single-pass
/// `sdpa_vector`), transcribed at BD=32 / BN=32 / `qk_per_thread = D/BD = 16`
/// — the template already legal at D=512, never instantiated until now.
/// Decode-time split, not a compile-time macro. kL and per-row capacities
/// stay RUNTIME scalars so lockstep growth never recompiles a pipeline.
///
/// Fails closed (nil → caller keeps the pinned per-row loop, byte-preserved)
/// on: env kill-switch (DEFAULT ON), any other batch size, geometry, dtype,
/// scale, sinks, softcap, bidirectional kind, non-full kind,
/// non-`CBv2FullSequenceKV` rows, ATT-008-pooled rows, offsets not in
/// lockstep, or missing/mismatched backing buffers. All gates are checked
/// BEFORE any append, so failing closed can never leave a double-append
/// behind. Sliding D=256 `fusedRingWrite` / `attendRingWriting` is untouched.
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

    /// True when the vector launch can fire (kill-switch, block count).
    /// Callers that append first must check this before mutating KV.
    static var isPrepared: Bool {
        enabled && blocks > 0 && blocks.isMultiple(of: 32)
    }

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 2
    private static let gqa = 8
    private static let headDim = 512

    /// Mirrors `sdpa_vector_2pass` block count. Honor `MLX_SDPA_BLOCKS`.
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

    /// Host `eval_gpu` 2-pass split: M-series (`s`/`d`) uses 2-pass at
    /// `kL >= 1024`; GQA also 2-passes at `kL >= 4096` on other Apple GPUs.
    private static let isMSeries: Bool = {
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s", "d": return true
        default: return false
        }
    }()

    /// `sdpa_vector_2pass_1` at D=512 / GQA=8. Grid is total threads
    /// `(kvHeads*32, B*GQA, blocks)` so threadgroups are `(kvHeads, B, blocks)`
    /// with group `(32, GQA, 1)` — same encoding as the sliding D=256 2-pass.
    /// `params[0] = kL`, `params[1+row] = that row's KV capacity`.
    private static let passAKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_2pass_a_bf16_d512_g8_b\(blocks)_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int GQA = 8;
            constexpr int simd_width = 32;
            constexpr int qk_per_thread = D / simd_width;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            const int key_length = int(params[0]);
            const int row_capacity = int(params[1 + batch_index]);

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
                queries + size_t(batch_head) * D + lane * qk_per_thread;
            keys += size_t(kv_head) * size_t(row_capacity) * D
                + size_t(block) * D + lane * qk_per_thread;
            values += size_t(kv_head) * size_t(row_capacity) * D
                + size_t(block) * D + lane * qk_per_thread;
            device T* partial = partials
                + size_t(batch_head) * BLOCKS * D
                + size_t(block) * D + lane * qk_per_thread;
            device float* sum_out = sums + size_t(batch_head) * BLOCKS + block;
            device float* max_out = maxs + size_t(batch_head) * BLOCKS + block;

            thread float q[qk_per_thread];
            thread float accumulator[qk_per_thread];
            for (int element = 0; element < qk_per_thread; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < key_length; token += BLOCKS) {
                float score = 0.0f;
                for (int element = 0; element < qk_per_thread; ++element) {
                    score += q[element] * float(keys[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < qk_per_thread; ++element) {
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
            for (int element = 0; element < qk_per_thread; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )

    /// `sdpa_vector_2pass_2` at D=512. Same 1024-thread merge as sliding D=256;
    /// `values_per_lane = D/32 = 16` fills the 512-wide output.
    private static let passBKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d512_b\(blocks)_v1",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int output_group = int(simdgroup_index_in_threadgroup);
            const int block_lane = int(thread_index_in_simdgroup);

            partials += batch_head * BLOCKS * D
                + block_lane * D + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                max_score = max(
                    max_score, maxs[block_lane + simd_width * block]);
            }
            max_score = simd_max(max_score);

            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                const float factor = fast::exp(
                    maxs[block_lane + simd_width * block] - max_score);
                sum_exp_score +=
                    factor * sums[block_lane + simd_width * block];
            }
            sum_exp_score = simd_sum(sum_exp_score);

            for (int block = 0; block < BLOCKS / simd_width; ++block) {
                const float factor = fast::exp(maxs[block_lane] - max_score);
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] +=
                        factor * float(partials[element]);
                }
                maxs += simd_width;
                sums += simd_width;
                partials += simd_width * D;
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

    /// `sdpa_vector` single-pass at D=512. One 1024-thread group per
    /// (row, query head); 32 simdgroups walk kL with BN=32.
    private static let singlePassKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_vector_bf16_d512_g8_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params",
        ],
        outputNames: ["out"],
        source: """
            constexpr int D = 512;
            constexpr int GQA = 8;
            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int qk_per_thread = D / BD;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int batch_index = batch_head / 16;
            const int query_head = batch_head % 16;
            const int kv_head = query_head / GQA;
            const int simd_gid = int(simdgroup_index_in_threadgroup);
            const int simd_lid = int(thread_index_in_simdgroup);

            const int key_length = int(params[0]);
            const int row_capacity = int(params[1 + batch_index]);

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
                queries + size_t(batch_head) * D + simd_lid * qk_per_thread;
            keys += size_t(kv_head) * size_t(row_capacity) * D
                + size_t(simd_gid) * D + simd_lid * qk_per_thread;
            values += size_t(kv_head) * size_t(row_capacity) * D
                + size_t(simd_gid) * D + simd_lid * qk_per_thread;

            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            thread float q[qk_per_thread];
            thread float o[qk_per_thread];
            for (int element = 0; element < qk_per_thread; ++element) {
                q[element] = 1.0f * float(query[element]);
                o[element] = 0.0f;
            }

            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = simd_gid; token < key_length; token += BN) {
                float score = 0.0f;
                for (int element = 0; element < qk_per_thread; ++element) {
                    score += q[element] * float(keys[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float factor = fast::exp(max_score - new_max);
                const float exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int element = 0; element < qk_per_thread; ++element) {
                    o[element] = o[element] * factor
                        + exp_score * float(values[element]);
                }

                keys += BN * D;
                values += BN * D;
            }

            if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = max_scores[simd_lid];
            const float new_max = simd_max(max_score);
            const float factor = fast::exp(max_score - new_max);
            sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

            for (int element = 0; element < qk_per_thread; ++element) {
                outputs[simd_lid * BD + simd_gid] = o[element];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[element] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                o[element] = sum_exp_score == 0.0f
                    ? o[element]
                    : (o[element] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            device T* out_ptr =
                out + size_t(batch_head) * D + simd_gid * qk_per_thread;
            if (simd_lid == 0) {
                for (int element = 0; element < qk_per_thread; ++element) {
                    out_ptr[element] = T(o[element]);
                }
            }
        """,
        ensureRowContiguous: true
    )

    /// Launch vector SDPA over already-appended ragged K/V, or nil when any
    /// launch gate fails. Callers must not append before this returns nil.
    static func attendVector(
        queries: MLXArray,
        keys: [MLXArray],
        values: [MLXArray],
        keyLength: Int,
        scale: Float
    ) -> MLXArray? {
        guard enabled,
            blocks > 0,
            blocks.isMultiple(of: 32),
            scale == 1.0,
            keyLength >= 1,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            keys.count == batch,
            values.count == batch
        else { return nil }

        var params: [UInt32] = [UInt32(keyLength)]
        params.reserveCapacity(batch + 1)
        for index in 0 ..< batch {
            let key = keys[index]
            let value = values[index]
            guard key.dtype == .bfloat16,
                value.dtype == .bfloat16,
                key.ndim == 4,
                key.dim(0) == 1,
                key.dim(1) == kvHeads,
                key.dim(2) >= keyLength,
                key.dim(3) == headDim,
                value.shape == key.shape
            else { return nil }
            params.append(UInt32(key.dim(2)))
        }

        let paramsArray = MLXArray(params)
        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype)
        ]
        let useTwoPass =
            (isMSeries && keyLength >= 1024)
            || keyLength >= 4096

        if useTwoPass {
            let partialShape = [batch, queryHeads, 1, blocks, headDim]
            let summaryShape = [batch, queryHeads, 1, blocks]
            let passA = passAKernel(
                [queries] + keys + values + [paramsArray],
                template: template + [
                    ("D", headDim),
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

        return singlePassKernel(
            [queries] + keys + values + [paramsArray],
            template: template,
            grid: (batch * queryHeads * 1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    /// Batched update + attend for the D=512 full-attention decode cohort,
    /// or nil (with NO side effects) when any gate fails.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray? {
        guard enabled,
            blocks > 0,
            blocks.isMultiple(of: 32),
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
            keyLength >= 1,
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
        for (index, row) in fullRows.enumerated() {
            _ = row.update(
                keys: keys[index ..< (index + 1)],
                values: values[index ..< (index + 1)])
            let state = row.cbv2InnerState()
            keyBuffers.append(state[0])
            valueBuffers.append(state[1])
        }
        guard let output = attendVector(
            queries: queries, keys: keyBuffers, values: valueBuffers,
            keyLength: keyLength, scale: scale)
        else {
            preconditionFailure(
                "CBv2RaggedComposedD512DecodeAttentionV1: vector SDPA refused after append")
        }
        return output
    }
}
