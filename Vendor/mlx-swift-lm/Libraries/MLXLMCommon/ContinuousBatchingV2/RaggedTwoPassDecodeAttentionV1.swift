// Ranked replication by delordemm1 of the parity-clean resident q4 merge
// published in submission bc839700. This marker is non-executable; the kernel
// and host implementation below remain byte-identical to that measured tree.
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

    private static let q4ResidentMergeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_Q4_RESIDENT_MERGE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 8
    private static let gqa = 2
    private static let headDim = 256
    private static let sequenceLength = 1024

    private static let stockBlocks: Int = {
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s": return 64
        case "d": return 128
        default: return 32
        }
    }()

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

    private static let combineColumns: Int = {
        let capped = min(blocks, 32)
        return capped > 0 && (capped & (capped - 1)) == 0 ? capped : 32
    }()

    private static let combineSets = 32 / combineColumns
    private static let combineThreads = (32 / combineSets) * 32

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

    static let gqaPairedPassAEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DECODE_GQA_PAIRED_PASSA"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let fusedRingPassAPairedKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name:
                "cbv2_ragged8_ringwrite_sdpa_2pass_a_gqapair_bf16_d256_g2"
                + "_b\(blocks)_vr_qreg_v3",
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
                static_assert(GQA == 2, "this kernel pairs exactly two query heads");
                constexpr uint ring_mask = uint(N - 1);

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int block = int(threadgroup_position_in_grid.z);
                const int query_head = GQA * kv_head;
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
                if (block == 0) {
                    device T* write_key = const_cast<device T*>(keys) + write_slot * D;
                    device T* write_value = const_cast<device T*>(values) + write_slot * D;
                    for (int element = 0; element < values_per_lane; ++element) {
                        write_key[element] = new_key[element];
                        write_value[element] = new_value[element];
                    }
                }
                if (batch_index == 0 && kv_head == 0 && block == 0 && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                device T* partial = partials
                    + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
                device float* sum_out = sums + batch_head * BLOCKS + block;
                device float* max_out = maxs + batch_head * BLOCKS + block;

                thread float q_lo[values_per_lane];
                thread float q_hi[values_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                for (int element = 0; element < values_per_lane; ++element) {
                    q_lo[element] = 1.0f * float(query[element]);
                    q_hi[element] = 1.0f * float(query[D + element]);
                    acc_lo[element] = 0.0f;
                    acc_hi[element] = 0.0f;
                }

                uint slot = (ring_start + uint(block)) & ring_mask;
                float max_lo = -3.402823466e+38F;
                float max_hi = -3.402823466e+38F;
                float sum_lo = 0.0f;
                float sum_hi = 0.0f;
                for (int token = block; token < N; token += BLOCKS) {
                    const bool current = token == N - 1;
                    const device T* k = current ? new_key : keys + slot * D;
                    const device T* v = current ? new_value : values + slot * D;
                    float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float key_element = float(k[element]);
                        score_lo += q_lo[element] * key_element;
                        score_hi += q_hi[element] * key_element;
                    }
                    score_lo = simd_sum(score_lo);
                    score_hi = simd_sum(score_hi);

                    const float new_max_lo = max(max_lo, score_lo);
                    const float new_max_hi = max(max_hi, score_hi);
                    const float old_factor_lo = fast::exp(max_lo - new_max_lo);
                    const float old_factor_hi = fast::exp(max_hi - new_max_hi);
                    const float score_factor_lo = fast::exp(score_lo - new_max_lo);
                    const float score_factor_hi = fast::exp(score_hi - new_max_hi);
                    max_lo = new_max_lo;
                    max_hi = new_max_hi;
                    sum_lo = sum_lo * old_factor_lo + score_factor_lo;
                    sum_hi = sum_hi * old_factor_hi + score_factor_hi;
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float value_element = float(v[element]);
                        acc_lo[element] = acc_lo[element] * old_factor_lo
                            + score_factor_lo * value_element;
                        acc_hi[element] = acc_hi[element] * old_factor_hi
                            + score_factor_hi * value_element;
                    }

                    slot = (slot + uint(BLOCKS)) & ring_mask;
                }

                if (lane == 0) {
                    sum_out[0] = sum_lo;
                    max_out[0] = max_lo;
                    sum_out[BLOCKS] = sum_hi;
                    max_out[BLOCKS] = max_hi;
                }
                for (int element = 0; element < values_per_lane; ++element) {
                    partial[element] = T(acc_lo[element]);
                    partial[BLOCKS * D + element] = T(acc_hi[element]);
                }
            """,
            ensureRowContiguous: true
        )

    static let gqaPairedPassAVec4Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_DECODE_PAIRED_PASSA_VEC4"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let fusedRingPassAPairedVec4Kernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name:
                "cbv2_ragged8_ringwrite_sdpa_2pass_a_gqapair_vec4_bf16_d256_g2"
                + "_b\(blocks)_v1",
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
                constexpr int vectors_per_lane = values_per_lane / 4;
                static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
                static_assert(GQA == 2, "this kernel pairs exactly two query heads");
                static_assert(values_per_lane % 4 == 0, "lane run must be a multiple of four");
                constexpr uint ring_mask = uint(N - 1);
                typedef vec<T, 4> T4;

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int block = int(threadgroup_position_in_grid.z);
                const int query_head = GQA * kv_head;
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
                if (block == 0) {
                    device T4* write_key = reinterpret_cast<device T4*>(
                        const_cast<device T*>(keys) + write_slot * D);
                    device T4* write_value = reinterpret_cast<device T4*>(
                        const_cast<device T*>(values) + write_slot * D);
                    const device T4* source_key =
                        reinterpret_cast<const device T4*>(new_key);
                    const device T4* source_value =
                        reinterpret_cast<const device T4*>(new_value);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        write_key[chunk] = source_key[chunk];
                        write_value[chunk] = source_value[chunk];
                    }
                }
                if (batch_index == 0 && kv_head == 0 && block == 0 && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                device T* partial = partials
                    + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
                device float* sum_out = sums + batch_head * BLOCKS + block;
                device float* max_out = maxs + batch_head * BLOCKS + block;

                thread T4 q_lo_vectors[vectors_per_lane];
                thread T4 q_hi_vectors[vectors_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                {
                    const device T4* query_lo =
                        reinterpret_cast<const device T4*>(query);
                    const device T4* query_hi =
                        reinterpret_cast<const device T4*>(query + D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        q_lo_vectors[chunk] = query_lo[chunk];
                        q_hi_vectors[chunk] = query_hi[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            acc_lo[chunk * 4 + j] = 0.0f;
                            acc_hi[chunk * 4 + j] = 0.0f;
                        }
                    }
                }

                uint slot = (ring_start + uint(block)) & ring_mask;
                float max_lo = -3.402823466e+38F;
                float max_hi = -3.402823466e+38F;
                float sum_lo = 0.0f;
                float sum_hi = 0.0f;
                for (int token = block; token < N; token += BLOCKS) {
                    const bool current = token == N - 1;
                    const device T4* k = reinterpret_cast<const device T4*>(
                        current ? new_key : keys + slot * D);
                    const device T4* v = reinterpret_cast<const device T4*>(
                        current ? new_value : values + slot * D);
                    float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    thread T4 value_vectors[vectors_per_lane];
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 key_vector = k[chunk];
                        const T4 q_lo_vector = q_lo_vectors[chunk];
                        const T4 q_hi_vector = q_hi_vectors[chunk];
                        value_vectors[chunk] = v[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            const float key_element = float(key_vector[j]);
                            score_lo += float(q_lo_vector[j]) * key_element;
                            score_hi += float(q_hi_vector[j]) * key_element;
                        }
                    }
                    score_lo = simd_sum(score_lo);
                    score_hi = simd_sum(score_hi);

                    const float new_max_lo = max(max_lo, score_lo);
                    const float new_max_hi = max(max_hi, score_hi);
                    const float old_factor_lo = fast::exp(max_lo - new_max_lo);
                    const float old_factor_hi = fast::exp(max_hi - new_max_hi);
                    const float score_factor_lo = fast::exp(score_lo - new_max_lo);
                    const float score_factor_hi = fast::exp(score_hi - new_max_hi);
                    max_lo = new_max_lo;
                    max_hi = new_max_hi;
                    sum_lo = sum_lo * old_factor_lo + score_factor_lo;
                    sum_hi = sum_hi * old_factor_hi + score_factor_hi;
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 value_vector = value_vectors[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            const int element = chunk * 4 + j;
                            const float value_element = float(value_vector[j]);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }

                    slot = (slot + uint(BLOCKS)) & ring_mask;
                }

                if (lane == 0) {
                    sum_out[0] = sum_lo;
                    max_out[0] = max_lo;
                    sum_out[BLOCKS] = sum_hi;
                    max_out[BLOCKS] = max_hi;
                }
                {
                    device T4* partial_lo = reinterpret_cast<device T4*>(partial);
                    device T4* partial_hi =
                        reinterpret_cast<device T4*>(partial + BLOCKS * D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        T4 lo_vector;
                        T4 hi_vector;
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            lo_vector[j] = T(acc_lo[chunk * 4 + j]);
                            hi_vector[j] = T(acc_hi[chunk * 4 + j]);
                        }
                        partial_lo[chunk] = lo_vector;
                        partial_hi[chunk] = hi_vector;
                    }
                }
            """,
            ensureRowContiguous: true
        )

    private static let passBKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name:
            "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)"
            + "_c\(combineColumns)_vec4_v6",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            typedef vec<T, 4> T4;
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int vectors_per_lane = values_per_lane / 4;
            static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
            // COMBINE-PACK-001: a lane owns one partition column of one output
            // group, and a simdgroup carries `sets` output groups side by
            // side. COLS is min(BLOCKS, simd_width) rounded to a power of two,
            // so every lane holds a live column and the surplus-lane guard
            // below is the constant true whenever the partition fits a
            // simdgroup. At COLS == simd_width this is the incumbent one
            // output group per simdgroup, one column per lane.
            constexpr int sets = simd_width / COLS;
            constexpr int rounds = (BLOCKS + COLS - 1) / COLS;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            const int block_lane = lane % COLS;
            const int output_group =
                int(simdgroup_index_in_threadgroup) * sets + lane / COLS;

            partials += batch_head * BLOCKS * D
                + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            // COMBINE-HOIST-001: a lane's column summaries are invariant
            // across the three passes below, and its rescale factor is
            // invariant across the last two. The incumbent re-read `maxs`
            // three times and `sums` once, and evaluated the same
            // `fast::exp` twice per column. Each is kept in a register
            // instead. `rounds` is a compile-time constant, so these are
            // named registers rather than an indexed stack array.
            thread float lane_max[rounds];
            thread float lane_sum[rounds];
            thread float lane_factor[rounds];
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                const bool live = column < BLOCKS;
                lane_max[round] = live ? maxs[column] : -3.402823466e+38F;
                lane_sum[round] = live ? sums[column] : 0.0f;
                max_score = max(max_score, lane_max[round]);
            }
            // The columns of one output group sit on the contiguous lane run
            // [set * COLS, set * COLS + COLS), so an ascending xor butterfly
            // bounded at COLS never leaves the set. It is the same tree the
            // full-width reduction ran over the live columns, with the rounds
            // that only folded in the identity dropped.
            for (int stride = 1; stride < COLS; stride <<= 1) {
                max_score =
                    max(max_score, simd_shuffle_xor(max_score, ushort(stride)));
            }

            for (int round = 0; round < rounds; ++round) {
                lane_factor[round] = fast::exp(lane_max[round] - max_score);
                sum_exp_score += lane_factor[round] * lane_sum[round];
            }
            for (int stride = 1; stride < COLS; stride <<= 1) {
                sum_exp_score += simd_shuffle_xor(sum_exp_score, ushort(stride));
            }

            // A lane's run of the column is contiguous, so it is read as
            // four-wide vectors of the same element type. Each component is
            // widened where it is multiplied, so every product and every
            // accumulator update is the one the element walk performed.
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                if (column < BLOCKS) {
                    const float factor = lane_factor[round];
                    const device T4* partial_vectors =
                        reinterpret_cast<const device T4*>(
                            partials + column * D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 partial_vector = partial_vectors[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            accumulator[chunk * 4 + j] +=
                                factor * float(partial_vector[j]);
                        }
                    }
                }
            }

            for (int element = 0; element < values_per_lane; ++element) {
                float reduced = accumulator[element];
                for (int stride = 1; stride < COLS; stride <<= 1) {
                    reduced += simd_shuffle_xor(reduced, ushort(stride));
                }
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

    private static let passBFoldKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name:
            "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)"
            + "_c\(combineColumns)_vec4_xfold_v1",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            typedef vec<T, 4> T4;
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int vectors_per_lane = values_per_lane / 4;
            static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
            // COMBINE-PACK-001: a lane owns one partition column of one output
            // group, and a simdgroup carries `sets` output groups side by
            // side. COLS is min(BLOCKS, simd_width) rounded to a power of two,
            // so every lane holds a live column and the surplus-lane guard
            // below is the constant true whenever the partition fits a
            // simdgroup. At COLS == simd_width this is the incumbent one
            // output group per simdgroup, one column per lane.
            constexpr int sets = simd_width / COLS;
            constexpr int rounds = (BLOCKS + COLS - 1) / COLS;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            const int block_lane = lane % COLS;
            const int output_group =
                int(simdgroup_index_in_threadgroup) * sets + lane / COLS;

            partials += batch_head * BLOCKS * D
                + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            // COMBINE-HOIST-001: a lane's column summaries are invariant
            // across the three passes below, and its rescale factor is
            // invariant across the last two. The incumbent re-read `maxs`
            // three times and `sums` once, and evaluated the same
            // `fast::exp` twice per column. Each is kept in a register
            // instead. `rounds` is a compile-time constant, so these are
            // named registers rather than an indexed stack array.
            thread float lane_max[rounds];
            thread float lane_sum[rounds];
            thread float lane_factor[rounds];
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                const bool live = column < BLOCKS;
                lane_max[round] = live ? maxs[column] : -3.402823466e+38F;
                lane_sum[round] = live ? sums[column] : 0.0f;
                max_score = max(max_score, lane_max[round]);
            }
            // The columns of one output group sit on the contiguous lane run
            // [set * COLS, set * COLS + COLS), so an ascending xor butterfly
            // bounded at COLS never leaves the set. It is the same tree the
            // full-width reduction ran over the live columns, with the rounds
            // that only folded in the identity dropped.
            for (int stride = 1; stride < COLS; stride <<= 1) {
                max_score =
                    max(max_score, simd_shuffle_xor(max_score, ushort(stride)));
            }

            for (int round = 0; round < rounds; ++round) {
                lane_factor[round] = fast::exp(lane_max[round] - max_score);
                sum_exp_score += lane_factor[round] * lane_sum[round];
            }
            for (int stride = 1; stride < COLS; stride <<= 1) {
                sum_exp_score += simd_shuffle_xor(sum_exp_score, ushort(stride));
            }

            // A lane's run of the column is contiguous, so it is read as
            // four-wide vectors of the same element type. Each component is
            // widened where it is multiplied, so every product and every
            // accumulator update is the one the element walk performed.
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                if (column < BLOCKS) {
                    const float factor = lane_factor[round];
                    const device T4* partial_vectors =
                        reinterpret_cast<const device T4*>(
                            partials + column * D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 partial_vector = partial_vectors[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            accumulator[chunk * 4 + j] +=
                                factor * float(partial_vector[j]);
                        }
                    }
                }
            }

            // COMBINE-XFOLD-001: fold the lane run with ONE cross-lane
            // butterfly instead of `values_per_lane` independent chains, and
            // let every lane store its own element.
            //
            // The incumbent ran one log2(COLS) shuffle chain per element, so
            // every lane carried every element to the last step, and then the
            // single lane with `block_lane == 0` issued the whole run of
            // stores while the other COLS-1 lanes sat in the branch shadow.
            //
            // Step k pairs the lanes that differ in bit k of `block_lane`,
            // exactly as chain step k did. Each lane keeps the half of the
            // element set whose low index bit equals its own bit k and sends
            // the half its partner keeps, so `simd_shuffle_xor(upper ? a : b,
            // stride)` returns the partner's evaluation of that select and
            // the partner has the opposite `upper`. After the last step a
            // lane holds, in `fold[j]`, the complete sum for element
            // `j * COLS + block_lane`.
            //
            // Every element's additions therefore still fold lane bit 0
            // first, then bit 1, and so on in the same ascending order, over
            // the same values, so each stored element is bit for bit the one
            // the chains produced. `sum_exp_score` is bitwise equal on every
            // lane of the set already: its own butterfly gives each lane the
            // same operands in commuted pairs, and IEEE addition is
            // commutative.
            //
            // Each step is a literal-bound block rather than a loop over a
            // runtime delta, so the array is addressed with compile-time
            // indices and cannot be spilled.
            constexpr bool lane_fold = values_per_lane >= COLS;
            if (lane_fold) {
                float fold[values_per_lane];
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    fold[element] = accumulator[element];
                }
                if (COLS > 1) {
                    const bool upper = (block_lane & 1) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 2; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(1));
                    }
                }
                if (COLS > 2) {
                    const bool upper = (block_lane & 2) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 4; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(2));
                    }
                }
                if (COLS > 4) {
                    const bool upper = (block_lane & 4) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 8; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(4));
                    }
                }
                if (COLS > 8) {
                    const bool upper = (block_lane & 8) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 16; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(8));
                    }
                }
                if (COLS > 16) {
                    const bool upper = (block_lane & 16) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 32; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(16));
                    }
                }
                // The run the lane kept is contiguous across the set, so the
                // COLS stores of one output group are one coalesced run
                // instead of a serial run issued by one lane.
                #pragma clang loop unroll(full)
                for (int j = 0; j < values_per_lane / COLS; ++j) {
                    out[j * COLS + block_lane] = T(
                        sum_exp_score == 0.0f
                            ? fold[j]
                            : fold[j] / sum_exp_score);
                }
            } else {
                // A partition wider than the lane run cannot be folded into
                // the lane index; that shape keeps the incumbent chains.
                for (int element = 0; element < values_per_lane; ++element) {
                    float reduced = accumulator[element];
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        reduced += simd_shuffle_xor(reduced, ushort(stride));
                    }
                    if (block_lane == 0) {
                        out[element] = T(
                            sum_exp_score == 0.0f
                                ? reduced
                                : reduced / sum_exp_score);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let combineFold: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PASSA_COMBFOLD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static var passBActive: MLXFast.MLXFastKernel {
        if combineFold {
            CBv2EngageMark.once("passb-xfold")
            return passBFoldKernel
        }
        return passBKernel
    }

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

    private static let portQuantReadKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_ring_2pass_a_q4g64_d256_g2_port_b\(blocks)_v1",
        inputNames: [
            "queries",
            "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "starts",
        ],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int row_stride = D + 4;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int batch_index = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = batch_index * 16 + query_head;
            const int lane = int(thread_index_in_simdgroup);

            // KVQ4: 4-bit payload in 32 words (8 nibbles each) plus one tail word
            // per 64-element group holding that group's fp16 (scale, bias).
            constexpr int row_words = D / 8 + D / 64;
            const device uint32_t* mirror_w = m0;
            switch (batch_index) {
                case 1: mirror_w = m1; break;
                case 2: mirror_w = m2; break;
                case 3: mirror_w = m3; break;
                case 4: mirror_w = m4; break;
                case 5: mirror_w = m5; break;
                case 6: mirror_w = m6; break;
                case 7: mirror_w = m7; break;
                default: break;
            }
            const uint start = starts[batch_index];

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
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
                const device uint32_t* krow_w =
                    mirror_w + (kv_head * N + slot) * row_words;
                const device uint32_t* vrow_w =
                    mirror_w + ((KV_HEADS + kv_head) * N + slot) * row_words;
                // One lane owns 8 consecutive elements, so it sits wholly
                // inside one 64-element group: group = (lane * 8) / 64.
                const int group = lane / 8;
                const uint32_t ktw = krow_w[D / 8 + group];
                const uint32_t vtw = vrow_w[D / 8 + group];
                const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                const float kb = float(as_type<half>(ushort(ktw >> 16)));
                const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                const float vb = float(as_type<half>(ushort(vtw >> 16)));
                const uint32_t kw = krow_w[lane];
                const uint32_t vw = vrow_w[lane];
                float score = 0.0f;
                for (int element = 0; element < 8; ++element) {
                    score += q[element]
                        * fma(float((kw >> (4 * element)) & 0xfu), ks, kb);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < 8; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor
                            * fma(float((vw >> (4 * element)) & 0xfu), vs, vb);
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

    private static let portQuantFusedWriteKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_ragged8_sdpa_ringwrite_2pass_a_q4g64_d256_g2_regpack_vec4_carry_pair_b\(blocks)_v5",
            inputNames: [
                "queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "new_keys", "new_values", "write_fence",
            ],
            outputNames: ["partials", "sums", "maxs", "fence"],
            source: """
                constexpr int simd_width = 32;
                constexpr int values_per_lane = D / simd_width;
                constexpr int payload_words = D / 8;
                constexpr int row_words = payload_words + D / 64;
                constexpr int current_block = (N - 1) % BLOCKS;

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int block = int(threadgroup_position_in_grid.z);
                // GQA-PAIR: one simdgroup serves BOTH query heads of its
                // group. The packed words, the metadata and the nibble
                // dequant are read and computed once and feed two independent
                // online-softmax chains that never mix.
                const int query_head = GQA * kv_head;
                const int batch_head = batch_index * 16 + query_head;
                const int lane = int(thread_index_in_simdgroup);

                const device uint32_t* mirror_w = m0;
                switch (batch_index) {
                    case 1: mirror_w = m1; break;
                    case 2: mirror_w = m2; break;
                    case 3: mirror_w = m3; break;
                    case 4: mirror_w = m4; break;
                    case 5: mirror_w = m5; break;
                    case 6: mirror_w = m6; break;
                    case 7: mirror_w = m7; break;
                    default: break;
                }
                const device uint32_t* mkeys_w =
                    mirror_w + kv_head * N * row_words;
                const device uint32_t* mvalues_w =
                    mirror_w + (KV_HEADS + kv_head) * N * row_words;
                const device T* new_key = new_keys
                    + (batch_index * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const device T* new_value = new_values
                    + (batch_index * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const uint start = starts[batch_index];
                const uint write_slot = (start + uint(N - 1)) % uint(N);

                half khs = half(0.0f);
                half khb = half(0.0f);
                half vhs = half(0.0f);
                half vhb = half(0.0f);
                uint32_t kword = 0u;
                uint32_t vword = 0u;
                if (block == current_block) {
                    float kmin = 3.402823466e+38F;
                    float kmax = -3.402823466e+38F;
                    float vmin = 3.402823466e+38F;
                    float vmax = -3.402823466e+38F;
                    // The lane's own elements, loaded once and held in
                    // registers: the extrema pass and the quantization pass
                    // below read the same values, so the second pass reads
                    // `kv`/`vv` instead of issuing the loads again.
                    float kv[values_per_lane];
                    float vv[values_per_lane];
                    // The lane owns `values_per_lane` contiguous elements
                    // starting at a multiple of that count, so the span is
                    // vector-aligned and the eight scalar loads become two
                    // four-wide ones per plane.
                    using T4 = vec<T, 4>;
                    const device T4* kvec =
                        reinterpret_cast<const device T4*>(new_key);
                    const device T4* vvec =
                        reinterpret_cast<const device T4*>(new_value);
                    #pragma unroll
                    for (int q = 0; q < values_per_lane / 4; ++q) {
                        const T4 kq4 = kvec[q];
                        const T4 vq4 = vvec[q];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            kv[q * 4 + j] = float(kq4[j]);
                            vv[q * 4 + j] = float(vq4[j]);
                            kmin = min(kmin, kv[q * 4 + j]);
                            kmax = max(kmax, kv[q * 4 + j]);
                            vmin = min(vmin, vv[q * 4 + j]);
                            vmax = max(vmax, vv[q * 4 + j]);
                        }
                    }
                    for (uint mask = 1; mask < 8; mask <<= 1) {
                        kmin = min(kmin, simd_shuffle_xor(kmin, mask));
                        kmax = max(kmax, simd_shuffle_xor(kmax, mask));
                        vmin = min(vmin, simd_shuffle_xor(vmin, mask));
                        vmax = max(vmax, simd_shuffle_xor(vmax, mask));
                    }
                    khs = half(max((kmax - kmin) / 15.0f, 1e-6f));
                    khb = half(kmin);
                    vhs = half(max((vmax - vmin) / 15.0f, 1e-6f));
                    vhb = half(vmin);
                    const float ks = float(khs);
                    const float kb = float(khb);
                    const float vs = float(vhs);
                    const float vb = float(vhb);
                    #pragma unroll
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float kq = metal::rint((kv[element] - kb) / ks);
                        const float vq = metal::rint((vv[element] - vb) / vs);
                        kword |= uint32_t(clamp(kq, 0.0f, 15.0f)) << (4 * element);
                        vword |= uint32_t(clamp(vq, 0.0f, 15.0f)) << (4 * element);
                    }

                    {
                        device uint32_t* write_key =
                            const_cast<device uint32_t*>(mkeys_w)
                            + write_slot * row_words;
                        device uint32_t* write_value =
                            const_cast<device uint32_t*>(mvalues_w)
                            + write_slot * row_words;
                        write_key[lane] = kword;
                        write_value[lane] = vword;
                        if (lane % 8 == 0) {
                            write_key[payload_words + lane / 8] =
                                uint32_t(as_type<ushort>(khs))
                                | (uint32_t(as_type<ushort>(khb)) << 16);
                            write_value[payload_words + lane / 8] =
                                uint32_t(as_type<ushort>(vhs))
                                | (uint32_t(as_type<ushort>(vhb)) << 16);
                        }
                    }
                }
                if (batch_index == 0 && kv_head == 0
                    && block == current_block && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                const device T* query =
                    queries + batch_head * D + lane * values_per_lane;
                device T* partial = partials
                    + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
                device float* sum_out = sums + batch_head * BLOCKS + block;
                device float* max_out = maxs + batch_head * BLOCKS + block;

                thread float q_lo[values_per_lane];
                thread float q_hi[values_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                for (int element = 0; element < values_per_lane; ++element) {
                    q_lo[element] = float(query[element]);
                    q_hi[element] = float(query[D + element]);
                    acc_lo[element] = 0.0f;
                    acc_hi[element] = 0.0f;
                }

                float max_lo = -3.402823466e+38F;
                float max_hi = -3.402823466e+38F;
                float sum_lo = 0.0f;
                float sum_hi = 0.0f;
                uint slot = (start + uint(block)) % uint(N);
                // The walk holds the next position's packed words while it
                // works on the current one, so each load is issued a whole
                // iteration before its value is needed. The prefetch is
                // suppressed when the next position is the current token,
                // which is served from `kword`/`vword` and whose slot this
                // kernel is storing into; every address formed is therefore a
                // slot the one-stage walk also reads.
                const bool prefetch_first = block < N - 1;
                uint next_slot = slot + uint(BLOCKS);
                if (next_slot >= uint(N)) next_slot -= uint(N);
                uint32_t kw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + lane] : 0u;
                uint32_t vw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + lane] : 0u;
                uint32_t ktw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + payload_words + lane / 8] : 0u;
                uint32_t vtw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + payload_words + lane / 8] : 0u;
                for (int token = block; token < N; token += BLOCKS) {
                    const bool current = token == N - 1;
                    const uint32_t kw = current ? kword : kw_pre;
                    const uint32_t vw = current ? vword : vw_pre;
                    const uint32_t ktw = current
                        ? (uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16))
                        : ktw_pre;
                    const uint32_t vtw = current
                        ? (uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16))
                        : vtw_pre;
                    if (token + BLOCKS < N - 1) {
                        kw_pre = mkeys_w[next_slot * row_words + lane];
                        vw_pre = mvalues_w[next_slot * row_words + lane];
                        ktw_pre =
                            mkeys_w[next_slot * row_words + payload_words + lane / 8];
                        vtw_pre =
                            mvalues_w[next_slot * row_words + payload_words + lane / 8];
                        next_slot += uint(BLOCKS);
                        if (next_slot >= uint(N)) next_slot -= uint(N);
                    }
                    const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                    const float kb = float(as_type<half>(ushort(ktw >> 16)));
                    const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                    const float vb = float(as_type<half>(ushort(vtw >> 16)));
                    float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float key_element =
                            fma(float((kw >> (4 * element)) & 0xfu), ks, kb);
                        score_lo += q_lo[element] * key_element;
                        score_hi += q_hi[element] * key_element;
                    }
                    score_lo = simd_sum(score_lo);
                    score_hi = simd_sum(score_hi);

                    const float new_max_lo = max(max_lo, score_lo);
                    const float new_max_hi = max(max_hi, score_hi);
                    const float old_factor_lo = fast::exp(max_lo - new_max_lo);
                    const float old_factor_hi = fast::exp(max_hi - new_max_hi);
                    const float score_factor_lo = fast::exp(score_lo - new_max_lo);
                    const float score_factor_hi = fast::exp(score_hi - new_max_hi);
                    max_lo = new_max_lo;
                    max_hi = new_max_hi;
                    sum_lo = sum_lo * old_factor_lo + score_factor_lo;
                    sum_hi = sum_hi * old_factor_hi + score_factor_hi;
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float value_element =
                            fma(float((vw >> (4 * element)) & 0xfu), vs, vb);
                        acc_lo[element] = acc_lo[element] * old_factor_lo
                            + score_factor_lo * value_element;
                        acc_hi[element] = acc_hi[element] * old_factor_hi
                            + score_factor_hi * value_element;
                    }

                }

                if (lane == 0) {
                    sum_out[0] = sum_lo;
                    max_out[0] = max_lo;
                    sum_out[BLOCKS] = sum_hi;
                    max_out[BLOCKS] = max_hi;
                }
                for (int element = 0; element < values_per_lane; ++element) {
                    partial[element] = T(acc_lo[element]);
                    partial[BLOCKS * D + element] = T(acc_hi[element]);
                }
            """,
            ensureRowContiguous: true
        )

    private static let portQuantFusedWriteResidentKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_ragged8_sdpa_ringwrite_q4g64_d256_g2_regpack_vec4_carry_pair_b8_resident_colred_vload_c3",
            inputNames: [
                "queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "new_keys", "new_values", "write_fence",
            ],
            outputNames: ["out", "fence"],
            source: """
                typedef vec<T, 4> T4;
                constexpr int simd_width = 32;
                constexpr int values_per_lane = D / simd_width;
                constexpr int vectors_per_lane = values_per_lane / 4;
                constexpr int payload_words = D / 8;
                constexpr int row_words = payload_words + D / 64;
                constexpr int current_block = (N - 1) % BLOCKS;
                constexpr int COLS = BLOCKS;
                constexpr int sets = simd_width / COLS;
                constexpr int rounds = (BLOCKS + COLS - 1) / COLS;
                static_assert(BLOCKS == 8, "resident kernel requires eight blocks");
                static_assert(GQA == 2, "resident kernel requires GQA two");
                static_assert(values_per_lane % 4 == 0, "lane run is four-wide");

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int block = int(simdgroup_index_in_threadgroup);
                const int query_head = GQA * kv_head;
                const int batch_head = batch_index * 16 + query_head;
                const int lane = int(thread_index_in_simdgroup);

                threadgroup T local_partials[GQA * BLOCKS * D];
                threadgroup float local_sums[GQA * BLOCKS];
                threadgroup float local_maxs[GQA * BLOCKS];

                const device uint32_t* mirror_w = m0;
                switch (batch_index) {
                    case 1: mirror_w = m1; break;
                    case 2: mirror_w = m2; break;
                    case 3: mirror_w = m3; break;
                    case 4: mirror_w = m4; break;
                    case 5: mirror_w = m5; break;
                    case 6: mirror_w = m6; break;
                    case 7: mirror_w = m7; break;
                    default: break;
                }
                const device uint32_t* mkeys_w =
                    mirror_w + kv_head * N * row_words;
                const device uint32_t* mvalues_w =
                    mirror_w + (KV_HEADS + kv_head) * N * row_words;
                const device T* new_key = new_keys
                    + (batch_index * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const device T* new_value = new_values
                    + (batch_index * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const uint start = starts[batch_index];
                const uint write_slot = (start + uint(N - 1)) % uint(N);

                half khs = half(0.0f);
                half khb = half(0.0f);
                half vhs = half(0.0f);
                half vhb = half(0.0f);
                uint32_t kword = 0u;
                uint32_t vword = 0u;
                if (block == current_block) {
                    float kmin = 3.402823466e+38F;
                    float kmax = -3.402823466e+38F;
                    float vmin = 3.402823466e+38F;
                    float vmax = -3.402823466e+38F;
                    float kv[values_per_lane];
                    float vv[values_per_lane];
                    const device T4* kvec =
                        reinterpret_cast<const device T4*>(new_key);
                    const device T4* vvec =
                        reinterpret_cast<const device T4*>(new_value);
                    #pragma unroll
                    for (int q = 0; q < values_per_lane / 4; ++q) {
                        const T4 kq4 = kvec[q];
                        const T4 vq4 = vvec[q];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            kv[q * 4 + j] = float(kq4[j]);
                            vv[q * 4 + j] = float(vq4[j]);
                            kmin = min(kmin, kv[q * 4 + j]);
                            kmax = max(kmax, kv[q * 4 + j]);
                            vmin = min(vmin, vv[q * 4 + j]);
                            vmax = max(vmax, vv[q * 4 + j]);
                        }
                    }
                    for (uint mask = 1; mask < 8; mask <<= 1) {
                        kmin = min(kmin, simd_shuffle_xor(kmin, mask));
                        kmax = max(kmax, simd_shuffle_xor(kmax, mask));
                        vmin = min(vmin, simd_shuffle_xor(vmin, mask));
                        vmax = max(vmax, simd_shuffle_xor(vmax, mask));
                    }
                    khs = half(max((kmax - kmin) / 15.0f, 1e-6f));
                    khb = half(kmin);
                    vhs = half(max((vmax - vmin) / 15.0f, 1e-6f));
                    vhb = half(vmin);
                    const float ks = float(khs);
                    const float kb = float(khb);
                    const float vs = float(vhs);
                    const float vb = float(vhb);
                    #pragma unroll
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float kq = metal::rint((kv[element] - kb) / ks);
                        const float vq = metal::rint((vv[element] - vb) / vs);
                        kword |= uint32_t(clamp(kq, 0.0f, 15.0f)) << (4 * element);
                        vword |= uint32_t(clamp(vq, 0.0f, 15.0f)) << (4 * element);
                    }

                    device uint32_t* write_key =
                        const_cast<device uint32_t*>(mkeys_w)
                        + write_slot * row_words;
                    device uint32_t* write_value =
                        const_cast<device uint32_t*>(mvalues_w)
                        + write_slot * row_words;
                    write_key[lane] = kword;
                    write_value[lane] = vword;
                    if (lane % 8 == 0) {
                        write_key[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16);
                        write_value[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16);
                    }
                }
                if (batch_index == 0 && kv_head == 0
                    && block == current_block && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                const device T* query =
                    queries + batch_head * D + lane * values_per_lane;
                threadgroup T* partial = local_partials
                    + block * D + lane * values_per_lane;
                threadgroup float* sum_out = local_sums + block;
                threadgroup float* max_out = local_maxs + block;

                thread float q_lo[values_per_lane];
                thread float q_hi[values_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                const device T4* qvec = reinterpret_cast<const device T4*>(query);
                const device T4* qvec_hi = reinterpret_cast<const device T4*>(query + D);
                #pragma clang loop unroll(full)
                for (int q = 0; q < values_per_lane / 4; ++q) {
                    const T4 q4_lo = qvec[q];
                    const T4 q4_hi = qvec_hi[q];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        q_lo[q * 4 + j] = float(q4_lo[j]);
                        q_hi[q * 4 + j] = float(q4_hi[j]);
                    }
                }
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    acc_lo[element] = 0.0f;
                    acc_hi[element] = 0.0f;
                }

                float max_lo = -3.402823466e+38F;
                float max_hi = -3.402823466e+38F;
                float sum_lo = 0.0f;
                float sum_hi = 0.0f;
                uint slot = (start + uint(block)) % uint(N);
                const bool prefetch_first = block < N - 1;
                uint next_slot = slot + uint(BLOCKS);
                if (next_slot >= uint(N)) next_slot -= uint(N);
                uint32_t kw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + lane] : 0u;
                uint32_t vw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + lane] : 0u;
                uint32_t ktw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + payload_words + lane / 8] : 0u;
                uint32_t vtw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + payload_words + lane / 8] : 0u;
                for (int token = block; token < N; token += BLOCKS) {
                    const bool current = token == N - 1;
                    const uint32_t kw = current ? kword : kw_pre;
                    const uint32_t vw = current ? vword : vw_pre;
                    const uint32_t ktw = current
                        ? (uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16))
                        : ktw_pre;
                    const uint32_t vtw = current
                        ? (uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16))
                        : vtw_pre;
                    if (token + BLOCKS < N - 1) {
                        kw_pre = mkeys_w[next_slot * row_words + lane];
                        vw_pre = mvalues_w[next_slot * row_words + lane];
                        ktw_pre =
                            mkeys_w[next_slot * row_words + payload_words + lane / 8];
                        vtw_pre =
                            mvalues_w[next_slot * row_words + payload_words + lane / 8];
                        next_slot += uint(BLOCKS);
                        if (next_slot >= uint(N)) next_slot -= uint(N);
                    }
                    const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                    const float kb = float(as_type<half>(ushort(ktw >> 16)));
                    const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                    const float vb = float(as_type<half>(ushort(vtw >> 16)));
                    float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float key_element =
                            fma(float((kw >> (4 * element)) & 0xfu), ks, kb);
                        score_lo += q_lo[element] * key_element;
                        score_hi += q_hi[element] * key_element;
                    }
                    score_lo = simd_sum(score_lo);
                    score_hi = simd_sum(score_hi);

                    const float new_max_lo = max(max_lo, score_lo);
                    const float new_max_hi = max(max_hi, score_hi);
                    const float old_factor_lo = fast::exp(max_lo - new_max_lo);
                    const float old_factor_hi = fast::exp(max_hi - new_max_hi);
                    const float score_factor_lo = fast::exp(score_lo - new_max_lo);
                    const float score_factor_hi = fast::exp(score_hi - new_max_hi);
                    max_lo = new_max_lo;
                    max_hi = new_max_hi;
                    sum_lo = sum_lo * old_factor_lo + score_factor_lo;
                    sum_hi = sum_hi * old_factor_hi + score_factor_hi;
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float value_element =
                            fma(float((vw >> (4 * element)) & 0xfu), vs, vb);
                        acc_lo[element] = acc_lo[element] * old_factor_lo
                            + score_factor_lo * value_element;
                        acc_hi[element] = acc_hi[element] * old_factor_hi
                            + score_factor_hi * value_element;
                    }
                }

                if (lane == 0) {
                    sum_out[0] = sum_lo;
                    max_out[0] = max_lo;
                    sum_out[BLOCKS] = sum_hi;
                    max_out[BLOCKS] = max_hi;
                }
                threadgroup T4* partial_vec_lo =
                    reinterpret_cast<threadgroup T4*>(partial);
                threadgroup T4* partial_vec_hi =
                    reinterpret_cast<threadgroup T4*>(partial + BLOCKS * D);
                #pragma clang loop unroll(full)
                for (int q = 0; q < values_per_lane / 4; ++q) {
                    T4 p4_lo, p4_hi;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        p4_lo[j] = T(acc_lo[q * 4 + j]);
                        p4_hi[j] = T(acc_hi[q * 4 + j]);
                    }
                    partial_vec_lo[q] = p4_lo;
                    partial_vec_hi[q] = p4_hi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const int block_lane = lane % COLS;
                const int output_group = block * sets + lane / COLS;
                #pragma clang loop unroll(full)
                for (int head = 0; head < GQA; ++head) {
                    const threadgroup T* head_partials = local_partials
                        + head * BLOCKS * D + output_group * values_per_lane;
                    const threadgroup float* head_sums =
                        local_sums + head * BLOCKS;
                    const threadgroup float* head_maxs =
                        local_maxs + head * BLOCKS;
                    device T* head_out = out
                        + (batch_head + head) * D
                        + output_group * values_per_lane;

                    thread float accumulator[values_per_lane];
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] = 0.0f;
                    }
                    thread float lane_max[rounds];
                    thread float lane_sum[rounds];
                    thread float lane_factor[rounds];
                    float sum_exp_score = 0.0f;
                    float max_score = -3.402823466e+38F;
                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        const bool live = column < BLOCKS;
                        lane_max[round] =
                            live ? head_maxs[column] : -3.402823466e+38F;
                        lane_sum[round] = live ? head_sums[column] : 0.0f;
                        max_score = max(max_score, lane_max[round]);
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        max_score = max(
                            max_score,
                            simd_shuffle_xor(max_score, ushort(stride)));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        lane_factor[round] =
                            fast::exp(lane_max[round] - max_score);
                        sum_exp_score += lane_factor[round] * lane_sum[round];
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        sum_exp_score +=
                            simd_shuffle_xor(sum_exp_score, ushort(stride));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        if (column < BLOCKS) {
                            const float factor = lane_factor[round];
                            const threadgroup T4* partial_vectors =
                                reinterpret_cast<const threadgroup T4*>(
                                    head_partials + column * D);
                            #pragma clang loop unroll(full)
                            for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                                const T4 partial_vector = partial_vectors[chunk];
                                #pragma clang loop unroll(full)
                                for (int j = 0; j < 4; ++j) {
                                    accumulator[chunk * 4 + j] +=
                                        factor * float(partial_vector[j]);
                                }
                            }
                        }
                    }

                    // Each step trades the half of the live slots whose
                    // element bit matches the partner's column bit and keeps
                    // the other half, so the live count runs 8, 4, 2, 1 and
                    // the lane that survives for an element is the one whose
                    // column index equals it. Every node of the addition tree
                    // pairs the same two column subtrees the per-element
                    // butterfly paired.
                    #pragma clang loop unroll(full)
                    for (int step = 0; (1 << step) < COLS; ++step) {
                        const ushort stride = ushort(1 << step);
                        const bool upper = (block_lane & int(stride)) != 0;
                        const int live = values_per_lane >> step;
                        #pragma clang loop unroll(full)
                        for (int slot = 0; slot < live; slot += 2) {
                            const float keep = upper
                                ? accumulator[slot + 1]
                                : accumulator[slot];
                            const float trade = upper
                                ? accumulator[slot]
                                : accumulator[slot + 1];
                            accumulator[slot >> 1] =
                                keep + simd_shuffle_xor(trade, stride);
                        }
                    }
                    head_out[block_lane] = T(
                        sum_exp_score == 0.0f
                            ? accumulator[0]
                            : accumulator[0] / sum_exp_score);
                }
            """,
            ensureRowContiguous: true
        )


    /// Multi-column twin of the resident kernel for the wide MTP verify: one
    /// threadgroup per (row, kv head) walks the rectangle's C columns in
    /// order, each column running the single-column body with its own ring
    /// start, query, new K/V and output rows (row-major `[B * C, ...]`, no
    /// fold), separated by a device-memory barrier so column c+1 attends the
    /// slot column c packed. Per column the arithmetic is the single-column
    /// kernel's.
    private static let portQuantFusedWriteResidentColumnsKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_wide_sdpa_ringwrite_q4g64_d256_g2_resident_columns_v1",
            inputNames: [
                "queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "new_keys", "new_values", "write_fence",
            ],
            outputNames: ["out", "fence"],
            source: """
                typedef vec<T, 4> T4;
                constexpr int simd_width = 32;
                constexpr int values_per_lane = D / simd_width;
                constexpr int vectors_per_lane = values_per_lane / 4;
                constexpr int payload_words = D / 8;
                constexpr int row_words = payload_words + D / 64;
                constexpr int current_block = (N - 1) % BLOCKS;
                constexpr int COLS = BLOCKS;
                constexpr int sets = simd_width / COLS;
                constexpr int rounds = (BLOCKS + COLS - 1) / COLS;
                static_assert(BLOCKS == 8, "resident kernel requires eight blocks");
                static_assert(GQA == 2, "resident kernel requires GQA two");
                static_assert(values_per_lane % 4 == 0, "lane run is four-wide");

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int block = int(simdgroup_index_in_threadgroup);
                const int query_head = GQA * kv_head;
                const int lane = int(thread_index_in_simdgroup);

                threadgroup T local_partials[GQA * BLOCKS * D];
                threadgroup float local_sums[GQA * BLOCKS];
                threadgroup float local_maxs[GQA * BLOCKS];

                const device uint32_t* mirror_w = m0;
                switch (batch_index) {
                    case 1: mirror_w = m1; break;
                    case 2: mirror_w = m2; break;
                    case 3: mirror_w = m3; break;
                    case 4: mirror_w = m4; break;
                    case 5: mirror_w = m5; break;
                    case 6: mirror_w = m6; break;
                    case 7: mirror_w = m7; break;
                    default: break;
                }
                const device uint32_t* mkeys_w =
                    mirror_w + kv_head * N * row_words;
                const device uint32_t* mvalues_w =
                    mirror_w + (KV_HEADS + kv_head) * N * row_words;
                const uint start0 = starts[batch_index];
                for (int c = 0; c < C; ++c) {
                if (c > 0) {
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                }
                const int batch_head = (batch_index * C + c) * 16 + query_head;
                const device T* new_key = new_keys
                    + ((batch_index * C + c) * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const device T* new_value = new_values
                    + ((batch_index * C + c) * KV_HEADS + kv_head) * D
                    + lane * values_per_lane;
                const uint start = start0 + uint(c);
                const uint write_slot = (start + uint(N - 1)) % uint(N);

                half khs = half(0.0f);
                half khb = half(0.0f);
                half vhs = half(0.0f);
                half vhb = half(0.0f);
                uint32_t kword = 0u;
                uint32_t vword = 0u;
                if (block == current_block) {
                    float kmin = 3.402823466e+38F;
                    float kmax = -3.402823466e+38F;
                    float vmin = 3.402823466e+38F;
                    float vmax = -3.402823466e+38F;
                    float kv[values_per_lane];
                    float vv[values_per_lane];
                    const device T4* kvec =
                        reinterpret_cast<const device T4*>(new_key);
                    const device T4* vvec =
                        reinterpret_cast<const device T4*>(new_value);
                    #pragma unroll
                    for (int q = 0; q < values_per_lane / 4; ++q) {
                        const T4 kq4 = kvec[q];
                        const T4 vq4 = vvec[q];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            kv[q * 4 + j] = float(kq4[j]);
                            vv[q * 4 + j] = float(vq4[j]);
                            kmin = min(kmin, kv[q * 4 + j]);
                            kmax = max(kmax, kv[q * 4 + j]);
                            vmin = min(vmin, vv[q * 4 + j]);
                            vmax = max(vmax, vv[q * 4 + j]);
                        }
                    }
                    for (uint mask = 1; mask < 8; mask <<= 1) {
                        kmin = min(kmin, simd_shuffle_xor(kmin, mask));
                        kmax = max(kmax, simd_shuffle_xor(kmax, mask));
                        vmin = min(vmin, simd_shuffle_xor(vmin, mask));
                        vmax = max(vmax, simd_shuffle_xor(vmax, mask));
                    }
                    khs = half(max((kmax - kmin) / 15.0f, 1e-6f));
                    khb = half(kmin);
                    vhs = half(max((vmax - vmin) / 15.0f, 1e-6f));
                    vhb = half(vmin);
                    const float ks = float(khs);
                    const float kb = float(khb);
                    const float vs = float(vhs);
                    const float vb = float(vhb);
                    #pragma unroll
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float kq = metal::rint((kv[element] - kb) / ks);
                        const float vq = metal::rint((vv[element] - vb) / vs);
                        kword |= uint32_t(clamp(kq, 0.0f, 15.0f)) << (4 * element);
                        vword |= uint32_t(clamp(vq, 0.0f, 15.0f)) << (4 * element);
                    }

                    device uint32_t* write_key =
                        const_cast<device uint32_t*>(mkeys_w)
                        + write_slot * row_words;
                    device uint32_t* write_value =
                        const_cast<device uint32_t*>(mvalues_w)
                        + write_slot * row_words;
                    write_key[lane] = kword;
                    write_value[lane] = vword;
                    if (lane % 8 == 0) {
                        write_key[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16);
                        write_value[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16);
                    }
                }
                if (c == 0 && batch_index == 0 && kv_head == 0
                    && block == current_block && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                const device T* query =
                    queries + batch_head * D + lane * values_per_lane;
                threadgroup T* partial = local_partials
                    + block * D + lane * values_per_lane;
                threadgroup float* sum_out = local_sums + block;
                threadgroup float* max_out = local_maxs + block;

                thread float q_lo[values_per_lane];
                thread float q_hi[values_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                const device T4* qvec = reinterpret_cast<const device T4*>(query);
                const device T4* qvec_hi = reinterpret_cast<const device T4*>(query + D);
                #pragma clang loop unroll(full)
                for (int q = 0; q < values_per_lane / 4; ++q) {
                    const T4 q4_lo = qvec[q];
                    const T4 q4_hi = qvec_hi[q];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        q_lo[q * 4 + j] = float(q4_lo[j]);
                        q_hi[q * 4 + j] = float(q4_hi[j]);
                    }
                }
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    acc_lo[element] = 0.0f;
                    acc_hi[element] = 0.0f;
                }

                float max_lo = -3.402823466e+38F;
                float max_hi = -3.402823466e+38F;
                float sum_lo = 0.0f;
                float sum_hi = 0.0f;
                uint slot = (start + uint(block)) % uint(N);
                const bool prefetch_first = block < N - 1;
                uint next_slot = slot + uint(BLOCKS);
                if (next_slot >= uint(N)) next_slot -= uint(N);
                uint32_t kw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + lane] : 0u;
                uint32_t vw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + lane] : 0u;
                uint32_t ktw_pre = prefetch_first
                    ? mkeys_w[slot * row_words + payload_words + lane / 8] : 0u;
                uint32_t vtw_pre = prefetch_first
                    ? mvalues_w[slot * row_words + payload_words + lane / 8] : 0u;
                for (int token = block; token < N; token += BLOCKS) {
                    const bool current = token == N - 1;
                    const uint32_t kw = current ? kword : kw_pre;
                    const uint32_t vw = current ? vword : vw_pre;
                    const uint32_t ktw = current
                        ? (uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16))
                        : ktw_pre;
                    const uint32_t vtw = current
                        ? (uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16))
                        : vtw_pre;
                    if (token + BLOCKS < N - 1) {
                        kw_pre = mkeys_w[next_slot * row_words + lane];
                        vw_pre = mvalues_w[next_slot * row_words + lane];
                        ktw_pre =
                            mkeys_w[next_slot * row_words + payload_words + lane / 8];
                        vtw_pre =
                            mvalues_w[next_slot * row_words + payload_words + lane / 8];
                        next_slot += uint(BLOCKS);
                        if (next_slot >= uint(N)) next_slot -= uint(N);
                    }
                    const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                    const float kb = float(as_type<half>(ushort(ktw >> 16)));
                    const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                    const float vb = float(as_type<half>(ushort(vtw >> 16)));
                    float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float key_element =
                            fma(float((kw >> (4 * element)) & 0xfu), ks, kb);
                        score_lo += q_lo[element] * key_element;
                        score_hi += q_hi[element] * key_element;
                    }
                    score_lo = simd_sum(score_lo);
                    score_hi = simd_sum(score_hi);

                    const float new_max_lo = max(max_lo, score_lo);
                    const float new_max_hi = max(max_hi, score_hi);
                    const float old_factor_lo = fast::exp(max_lo - new_max_lo);
                    const float old_factor_hi = fast::exp(max_hi - new_max_hi);
                    const float score_factor_lo = fast::exp(score_lo - new_max_lo);
                    const float score_factor_hi = fast::exp(score_hi - new_max_hi);
                    max_lo = new_max_lo;
                    max_hi = new_max_hi;
                    sum_lo = sum_lo * old_factor_lo + score_factor_lo;
                    sum_hi = sum_hi * old_factor_hi + score_factor_hi;
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float value_element =
                            fma(float((vw >> (4 * element)) & 0xfu), vs, vb);
                        acc_lo[element] = acc_lo[element] * old_factor_lo
                            + score_factor_lo * value_element;
                        acc_hi[element] = acc_hi[element] * old_factor_hi
                            + score_factor_hi * value_element;
                    }
                }

                if (lane == 0) {
                    sum_out[0] = sum_lo;
                    max_out[0] = max_lo;
                    sum_out[BLOCKS] = sum_hi;
                    max_out[BLOCKS] = max_hi;
                }
                threadgroup T4* partial_vec_lo =
                    reinterpret_cast<threadgroup T4*>(partial);
                threadgroup T4* partial_vec_hi =
                    reinterpret_cast<threadgroup T4*>(partial + BLOCKS * D);
                #pragma clang loop unroll(full)
                for (int q = 0; q < values_per_lane / 4; ++q) {
                    T4 p4_lo, p4_hi;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        p4_lo[j] = T(acc_lo[q * 4 + j]);
                        p4_hi[j] = T(acc_hi[q * 4 + j]);
                    }
                    partial_vec_lo[q] = p4_lo;
                    partial_vec_hi[q] = p4_hi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const int block_lane = lane % COLS;
                const int output_group = block * sets + lane / COLS;
                #pragma clang loop unroll(full)
                for (int head = 0; head < GQA; ++head) {
                    const threadgroup T* head_partials = local_partials
                        + head * BLOCKS * D + output_group * values_per_lane;
                    const threadgroup float* head_sums =
                        local_sums + head * BLOCKS;
                    const threadgroup float* head_maxs =
                        local_maxs + head * BLOCKS;
                    device T* head_out = out
                        + (batch_head + head) * D
                        + output_group * values_per_lane;

                    thread float accumulator[values_per_lane];
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] = 0.0f;
                    }
                    thread float lane_max[rounds];
                    thread float lane_sum[rounds];
                    thread float lane_factor[rounds];
                    float sum_exp_score = 0.0f;
                    float max_score = -3.402823466e+38F;
                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        const bool live = column < BLOCKS;
                        lane_max[round] =
                            live ? head_maxs[column] : -3.402823466e+38F;
                        lane_sum[round] = live ? head_sums[column] : 0.0f;
                        max_score = max(max_score, lane_max[round]);
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        max_score = max(
                            max_score,
                            simd_shuffle_xor(max_score, ushort(stride)));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        lane_factor[round] =
                            fast::exp(lane_max[round] - max_score);
                        sum_exp_score += lane_factor[round] * lane_sum[round];
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        sum_exp_score +=
                            simd_shuffle_xor(sum_exp_score, ushort(stride));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        if (column < BLOCKS) {
                            const float factor = lane_factor[round];
                            const threadgroup T4* partial_vectors =
                                reinterpret_cast<const threadgroup T4*>(
                                    head_partials + column * D);
                            #pragma clang loop unroll(full)
                            for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                                const T4 partial_vector = partial_vectors[chunk];
                                #pragma clang loop unroll(full)
                                for (int j = 0; j < 4; ++j) {
                                    accumulator[chunk * 4 + j] +=
                                        factor * float(partial_vector[j]);
                                }
                            }
                        }
                    }

                    // Each step trades the half of the live slots whose
                    // element bit matches the partner's column bit and keeps
                    // the other half, so the live count runs 8, 4, 2, 1 and
                    // the lane that survives for an element is the one whose
                    // column index equals it. Every node of the addition tree
                    // pairs the same two column subtrees the per-element
                    // butterfly paired.
                    #pragma clang loop unroll(full)
                    for (int step = 0; (1 << step) < COLS; ++step) {
                        const ushort stride = ushort(1 << step);
                        const bool upper = (block_lane & int(stride)) != 0;
                        const int live = values_per_lane >> step;
                        #pragma clang loop unroll(full)
                        for (int slot = 0; slot < live; slot += 2) {
                            const float keep = upper
                                ? accumulator[slot + 1]
                                : accumulator[slot];
                            const float trade = upper
                                ? accumulator[slot]
                                : accumulator[slot + 1];
                            accumulator[slot >> 1] =
                                keep + simd_shuffle_xor(trade, stride);
                        }
                    }
                    head_out[block_lane] = T(
                        sum_exp_score == 0.0f
                            ? accumulator[0]
                            : accumulator[0] / sum_exp_score);
                }
                }
            """,
            ensureRowContiguous: true
        )

    /// Single-pass twin of the columns kernel: simdgroup g owns the ring
    /// positions congruent to start + g (mod 8) and scores each one once
    /// against all C column queries, so the ring streams once per round
    /// instead of once per column. Column c's state in simdgroup g is block
    /// (g - c) mod 8 of the per-column kernel (same slots, same order; column
    /// c takes steps 0..127 when g >= c and 1..128 otherwise) and is stored
    /// at that block index for the unchanged combine, so every (row, column)
    /// output is bit-identical. Column c's new K/V sits at position
    /// start + N - 1 + c: packed by simdgroup 7 for c = 0 and by simdgroup
    /// c - 1 otherwise, consumed from registers, and written to the ring after
    /// the pass so the evicted slot is read before it is overwritten.
    private static let wideColumnsKernelHeader = """
// One ring token against the C column queries of one query head of a
// wide-verify row: the per-column kernel's inner-loop arithmetic, applied to
// the columns in [first_column, last_column] in column order. Constant
// bounds fold the column test away, which keeps the hot loop branch-free.
// R rounds the dequantized key and value elements (float keeps them as the
// target attends them; the kernel's T reproduces a bf16 dequantized copy).
template <int C, int VPL, typename R = float>
inline void cbv2_wide_columns_token(
    const uint32_t kw, const uint32_t vw, const uint32_t ktw, const uint32_t vtw,
    const int first_column, const int last_column,
    thread float (&q)[C][VPL], thread float (&acc)[C][VPL],
    thread float (&maxs)[C], thread float (&sums)[C]) {
  const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
  const float kb = float(as_type<half>(ushort(ktw >> 16)));
  const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
  const float vb = float(as_type<half>(ushort(vtw >> 16)));
  float key_element[VPL];
  float value_element[VPL];
  #pragma clang loop unroll(full)
  for (int element = 0; element < VPL; ++element) {
    key_element[element] = float(R(fma(float((kw >> (4 * element)) & 0xfu), ks, kb)));
    value_element[element] = float(R(fma(float((vw >> (4 * element)) & 0xfu), vs, vb)));
  }
  #pragma clang loop unroll(full)
  for (int c = 0; c < C; ++c) {
    if (c < first_column || c > last_column) continue;
    float score = 0.0f;
    #pragma clang loop unroll(full)
    for (int element = 0; element < VPL; ++element) {
      score += q[c][element] * key_element[element];
    }
    score = simd_sum(score);
    const float new_max = max(maxs[c], score);
    const float old_factor = fast::exp(maxs[c] - new_max);
    const float score_factor = fast::exp(score - new_max);
    maxs[c] = new_max;
    sums[c] = sums[c] * old_factor + score_factor;
    #pragma clang loop unroll(full)
    for (int element = 0; element < VPL; ++element) {
      acc[c][element] = acc[c][element] * old_factor
          + score_factor * value_element[element];
    }
  }
}
"""

    /// Single-pass twin of the columns kernel. Simdgroup (g, h) owns the ring
    /// positions congruent to start + g (mod 8) for query head h and scores
    /// each one once against all C column queries, so the ring streams once
    /// per round instead of once per column and twice as many simdgroups are
    /// in flight. Column c's state in simdgroup g is block (g - c) mod 8 of
    /// the per-column kernel (same slots, same order; column c takes steps
    /// 0..127 when g >= c and 1..128 otherwise) and is stored at that block
    /// index for the unchanged combine, so every (row, column) output is
    /// bit-identical. Column c's new K/V sits at position start + N - 1 + c:
    /// packed by simdgroup 7 for c = 0 and by simdgroup c - 1 otherwise,
    /// consumed from registers, and written to the ring after the pass so the
    /// evicted slot is read before it is overwritten.
    private static let portQuantFusedWriteResidentColumnsSinglePassKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_wide_sdpa_ringwrite_q4g64_d256_g2_resident_columns_v4",
            inputNames: [
                "queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "new_keys", "new_values", "write_fence",
            ],
            outputNames: ["out", "fence"],
            source: """
                typedef vec<T, 4> T4;
                constexpr int simd_width = 32;
                constexpr int values_per_lane = D / simd_width;
                constexpr int vectors_per_lane = values_per_lane / 4;
                constexpr int payload_words = D / 8;
                constexpr int row_words = payload_words + D / 64;
                constexpr int COLS = BLOCKS;
                constexpr int sets = simd_width / COLS;
                constexpr int rounds = (BLOCKS + COLS - 1) / COLS;
                constexpr int steps = N / BLOCKS;
                static_assert(BLOCKS == 8, "resident kernel requires eight blocks");
                static_assert(GQA == 2, "resident kernel requires GQA two");
                static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
                static_assert(N % BLOCKS == 0, "window is a multiple of the block count");
                static_assert(C >= 2 && C < BLOCKS, "one packed column per simdgroup");

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int simd = int(simdgroup_index_in_threadgroup);
                const int group = simd % BLOCKS;
                const int head = simd / BLOCKS;
                const int query_head = GQA * kv_head + head;
                const int lane = int(thread_index_in_simdgroup);

                threadgroup T local_partials[GQA * BLOCKS * D];
                threadgroup float local_sums[GQA * BLOCKS];
                threadgroup float local_maxs[GQA * BLOCKS];

                const device uint32_t* mirror_w = m0;
                switch (batch_index) {
                    case 1: mirror_w = m1; break;
                    case 2: mirror_w = m2; break;
                    case 3: mirror_w = m3; break;
                    case 4: mirror_w = m4; break;
                    case 5: mirror_w = m5; break;
                    case 6: mirror_w = m6; break;
                    case 7: mirror_w = m7; break;
                    default: break;
                }
                const device uint32_t* mkeys_w =
                    mirror_w + kv_head * N * row_words;
                const device uint32_t* mvalues_w =
                    mirror_w + (KV_HEADS + kv_head) * N * row_words;
                const uint start0 = starts[batch_index];
                const int packs_column = group == BLOCKS - 1
                    ? 0 : (group + 1 < C ? group + 1 : -1);
                const uint first_slot = (start0 + uint(group)) % uint(N);
                const uint write_slot = packs_column == 0
                    ? (start0 + uint(N - 1)) % uint(N) : first_slot;

                half khs = half(0.0f);
                half khb = half(0.0f);
                half vhs = half(0.0f);
                half vhb = half(0.0f);
                uint32_t kword = 0u;
                uint32_t vword = 0u;
                if (packs_column >= 0) {
                    const device T* new_key = new_keys
                        + ((batch_index * C + packs_column) * KV_HEADS + kv_head) * D
                        + lane * values_per_lane;
                    const device T* new_value = new_values
                        + ((batch_index * C + packs_column) * KV_HEADS + kv_head) * D
                        + lane * values_per_lane;
                    float kmin = 3.402823466e+38F;
                    float kmax = -3.402823466e+38F;
                    float vmin = 3.402823466e+38F;
                    float vmax = -3.402823466e+38F;
                    float kv[values_per_lane];
                    float vv[values_per_lane];
                    const device T4* kvec =
                        reinterpret_cast<const device T4*>(new_key);
                    const device T4* vvec =
                        reinterpret_cast<const device T4*>(new_value);
                    #pragma unroll
                    for (int q = 0; q < values_per_lane / 4; ++q) {
                        const T4 kq4 = kvec[q];
                        const T4 vq4 = vvec[q];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            kv[q * 4 + j] = float(kq4[j]);
                            vv[q * 4 + j] = float(vq4[j]);
                            kmin = min(kmin, kv[q * 4 + j]);
                            kmax = max(kmax, kv[q * 4 + j]);
                            vmin = min(vmin, vv[q * 4 + j]);
                            vmax = max(vmax, vv[q * 4 + j]);
                        }
                    }
                    for (uint mask = 1; mask < 8; mask <<= 1) {
                        kmin = min(kmin, simd_shuffle_xor(kmin, mask));
                        kmax = max(kmax, simd_shuffle_xor(kmax, mask));
                        vmin = min(vmin, simd_shuffle_xor(vmin, mask));
                        vmax = max(vmax, simd_shuffle_xor(vmax, mask));
                    }
                    khs = half(max((kmax - kmin) / 15.0f, 1e-6f));
                    khb = half(kmin);
                    vhs = half(max((vmax - vmin) / 15.0f, 1e-6f));
                    vhb = half(vmin);
                    const float ks = float(khs);
                    const float kb = float(khb);
                    const float vs = float(vhs);
                    const float vb = float(vhb);
                    #pragma unroll
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float kq = metal::rint((kv[element] - kb) / ks);
                        const float vq = metal::rint((vv[element] - vb) / vs);
                        kword |= uint32_t(clamp(kq, 0.0f, 15.0f)) << (4 * element);
                        vword |= uint32_t(clamp(vq, 0.0f, 15.0f)) << (4 * element);
                    }
                }
                if (head == 0 && packs_column == 0 && batch_index == 0 && kv_head == 0
                    && lane == 0) {
                    fence[0] = write_fence[0] + 1;
                }

                thread float q[C][values_per_lane];
                thread float acc[C][values_per_lane];
                thread float maxs[C];
                thread float sums[C];
                #pragma clang loop unroll(full)
                for (int c = 0; c < C; ++c) {
                    const device T4* qvec = reinterpret_cast<const device T4*>(
                        queries + ((batch_index * C + c) * 16 + query_head) * D
                        + lane * values_per_lane);
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < values_per_lane / 4; ++v) {
                        const T4 q4 = qvec[v];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            q[c][v * 4 + j] = float(q4[j]);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        acc[c][element] = 0.0f;
                    }
                    maxs[c] = -3.402823466e+38F;
                    sums[c] = 0.0f;
                }

                // Simdgroup 7's packed token is its step 127 (column 0's new
                // key); a packing simdgroup g < 7 appends its column g + 1 key
                // as step 128, after 128 device steps.
                const int last_device_step = packs_column == 0 ? steps - 2 : steps - 1;
                uint next_slot = first_slot + uint(BLOCKS);
                if (next_slot >= uint(N)) next_slot -= uint(N);
                uint32_t kw = mkeys_w[first_slot * row_words + lane];
                uint32_t vw = mvalues_w[first_slot * row_words + lane];
                uint32_t ktw = mkeys_w[first_slot * row_words + payload_words + lane / 8];
                uint32_t vtw = mvalues_w[first_slot * row_words + payload_words + lane / 8];
                uint32_t kw_pre = mkeys_w[next_slot * row_words + lane];
                uint32_t vw_pre = mvalues_w[next_slot * row_words + lane];
                uint32_t ktw_pre = mkeys_w[next_slot * row_words + payload_words + lane / 8];
                uint32_t vtw_pre = mvalues_w[next_slot * row_words + payload_words + lane / 8];
                cbv2_wide_columns_token<C, values_per_lane>(
                    kw, vw, ktw, vtw, 0, group, q, acc, maxs, sums);
                for (int step = 1; step <= last_device_step; ++step) {
                    kw = kw_pre;
                    vw = vw_pre;
                    ktw = ktw_pre;
                    vtw = vtw_pre;
                    if (step < last_device_step) {
                        next_slot += uint(BLOCKS);
                        if (next_slot >= uint(N)) next_slot -= uint(N);
                        kw_pre = mkeys_w[next_slot * row_words + lane];
                        vw_pre = mvalues_w[next_slot * row_words + lane];
                        ktw_pre =
                            mkeys_w[next_slot * row_words + payload_words + lane / 8];
                        vtw_pre =
                            mvalues_w[next_slot * row_words + payload_words + lane / 8];
                    }
                    cbv2_wide_columns_token<C, values_per_lane>(
                        kw, vw, ktw, vtw, 0, C - 1, q, acc, maxs, sums);
                }
                if (packs_column >= 0) {
                    cbv2_wide_columns_token<C, values_per_lane>(
                        kword, vword,
                        uint32_t(as_type<ushort>(khs)) | (uint32_t(as_type<ushort>(khb)) << 16),
                        uint32_t(as_type<ushort>(vhs)) | (uint32_t(as_type<ushort>(vhb)) << 16),
                        packs_column == 0 ? 0 : group + 1, C - 1, q, acc, maxs, sums);
                }

                if (head == 0 && packs_column >= 0) {
                    device uint32_t* write_key =
                        const_cast<device uint32_t*>(mkeys_w)
                        + write_slot * row_words;
                    device uint32_t* write_value =
                        const_cast<device uint32_t*>(mvalues_w)
                        + write_slot * row_words;
                    write_key[lane] = kword;
                    write_value[lane] = vword;
                    if (lane % 8 == 0) {
                        write_key[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(khs))
                            | (uint32_t(as_type<ushort>(khb)) << 16);
                        write_value[payload_words + lane / 8] =
                            uint32_t(as_type<ushort>(vhs))
                            | (uint32_t(as_type<ushort>(vhb)) << 16);
                    }
                }

                const int block_lane = lane % COLS;
                const int output_group = group * sets + lane / COLS;
                const threadgroup T* head_partials = local_partials
                    + head * BLOCKS * D + output_group * values_per_lane;
                const threadgroup float* head_sums = local_sums + head * BLOCKS;
                const threadgroup float* head_maxs = local_maxs + head * BLOCKS;
                #pragma clang loop unroll(full)
                for (int c = 0; c < C; ++c) {
                    const int block = (group - c + BLOCKS) % BLOCKS;
                    if (c > 0) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (lane == 0) {
                        local_sums[head * BLOCKS + block] = sums[c];
                        local_maxs[head * BLOCKS + block] = maxs[c];
                    }
                    threadgroup T4* partial_vec = reinterpret_cast<threadgroup T4*>(
                        local_partials + head * BLOCKS * D + block * D
                        + lane * values_per_lane);
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < values_per_lane / 4; ++v) {
                        T4 p4;
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            p4[j] = T(acc[c][v * 4 + j]);
                        }
                        partial_vec[v] = p4;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    device T* head_out = out
                        + ((batch_index * C + c) * 16 + query_head) * D
                        + output_group * values_per_lane;
                    thread float accumulator[values_per_lane];
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        accumulator[element] = 0.0f;
                    }
                    thread float lane_max[rounds];
                    thread float lane_sum[rounds];
                    thread float lane_factor[rounds];
                    float sum_exp_score = 0.0f;
                    float max_score = -3.402823466e+38F;
                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        const bool live = column < BLOCKS;
                        lane_max[round] =
                            live ? head_maxs[column] : -3.402823466e+38F;
                        lane_sum[round] = live ? head_sums[column] : 0.0f;
                        max_score = max(max_score, lane_max[round]);
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        max_score = max(
                            max_score,
                            simd_shuffle_xor(max_score, ushort(stride)));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        lane_factor[round] =
                            fast::exp(lane_max[round] - max_score);
                        sum_exp_score += lane_factor[round] * lane_sum[round];
                    }
                    #pragma clang loop unroll(full)
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        sum_exp_score +=
                            simd_shuffle_xor(sum_exp_score, ushort(stride));
                    }

                    #pragma clang loop unroll(full)
                    for (int round = 0; round < rounds; ++round) {
                        const int column = block_lane + COLS * round;
                        if (column < BLOCKS) {
                            const float factor = lane_factor[round];
                            const threadgroup T4* partial_vectors =
                                reinterpret_cast<const threadgroup T4*>(
                                    head_partials + column * D);
                            #pragma clang loop unroll(full)
                            for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                                const T4 partial_vector = partial_vectors[chunk];
                                #pragma clang loop unroll(full)
                                for (int j = 0; j < 4; ++j) {
                                    accumulator[chunk * 4 + j] +=
                                        factor * float(partial_vector[j]);
                                }
                            }
                        }
                    }

                    #pragma clang loop unroll(full)
                    for (int step = 0; (1 << step) < COLS; ++step) {
                        const ushort stride = ushort(1 << step);
                        const bool upper = (block_lane & int(stride)) != 0;
                        const int live = values_per_lane >> step;
                        #pragma clang loop unroll(full)
                        for (int slot = 0; slot < live; slot += 2) {
                            const float keep = upper
                                ? accumulator[slot + 1]
                                : accumulator[slot];
                            const float trade = upper
                                ? accumulator[slot]
                                : accumulator[slot + 1];
                            accumulator[slot >> 1] =
                                keep + simd_shuffle_xor(trade, stride);
                        }
                    }
                    head_out[block_lane] = T(
                        sum_exp_score == 0.0f
                            ? accumulator[0]
                            : accumulator[0] / sum_exp_score);
                }
            """,
            header: wideColumnsKernelHeader,
            ensureRowContiguous: true
        )

    /// Test hook: `false` routes the wide verify through the per-column `_v1`
    /// columns kernel the single-pass `_v4` is proven bit-identical against.
    nonisolated(unsafe) static var wideColumnsSinglePass = true

    /// `queries` `[8*C, 16, 1, D]`, `newKeys`/`newValues` `[8*C, kvHeads, 1, D]`
    /// row-major (row = cohort row * C + column); `startArray` is the `[8]`
    /// uint32 first-retained slot for column 0 per row. Output `[8*C, 16, 1, D]`.
    static func attendRingQuantWritingColumns(
        queries: MLXArray,
        mirrors: [MLXArray],
        startArray: MLXArray,
        newKeys: MLXArray,
        newValues: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int,
        columns: Int
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
            CBv2WindowedSequenceKV.quantEnabled,
            !CBv2WindowedSequenceKV.quantSimulate,
            !CBv2WindowedSequenceKV.gpuPackCheck,
            q4ResidentMergeEnabled, blocks == 8, combineColumns == 8, combineThreads == 256,
            slidingWindowLength == sequenceLength,
            columns >= 2, columns <= 4,
            startArray.shape == [batch], startArray.dtype == .uint32,
            enabled,
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch * columns, queryHeads, 1, headDim],
            newKeys.dtype == .bfloat16,
            newKeys.shape == [batch * columns, kvHeads, 1, headDim],
            newValues.dtype == .bfloat16,
            newValues.shape == newKeys.shape,
            previousWriteFence.dtype == .int32,
            previousWriteFence.shape == [1],
            mirrors.count == batch,
            mirrors.allSatisfy({
                $0.dtype == .uint32
                    && $0.shape == [2, kvHeads, sequenceLength, headDim / 8 + headDim / 64]
            })
        else { return nil }
        let inputs = [queries] + mirrors + [startArray, newKeys, newValues, previousWriteFence]
        let singlePass = wideColumnsSinglePass
        let kernel = singlePass
            ? portQuantFusedWriteResidentColumnsSinglePassKernel
            : portQuantFusedWriteResidentColumnsKernel
        let threads = blocks * 32 * (singlePass ? gqa : 1)
        let outputs = kernel(
            inputs,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", blocks),
                ("C", columns),
            ],
            grid: (kvHeads * threads, batch, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch * columns, queryHeads, 1, headDim], [1]],
            outputDTypes: [.bfloat16, .int32]
        )
        CBv2EngageMark.once(
            singlePass ? "kvq4-resident-columns-v4" : "kvq4-resident-columns")
        return (outputs[0], outputs[1])
    }

    /// Read-only twin of the single-pass kernel for the MTP drafter: one
    /// query column per row over the 1023 retained keys after the boundary
    /// slot `starts[row]` (the slot the drafter's sliding rule masks out),
    /// with the per-column kernel's token arithmetic over K/V rounded to T
    /// (the bytes of the dequantized copies it replaces) and float partials
    /// through the combine, like the stock SDPA it replaces. Nothing is
    /// written; `read_fence` is consumed for ordering only.
    private static let portQuantReadResidentKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_mtp_sdpa_q4g64_d256_g2_resident_read_v3",
            inputNames: [
                "queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "read_fence",
            ],
            outputNames: ["out"],
            source: """
                typedef vec<T, 4> T4;
                constexpr int simd_width = 32;
                constexpr int values_per_lane = D / simd_width;
                constexpr int vectors_per_lane = values_per_lane / 4;
                constexpr int payload_words = D / 8;
                constexpr int row_words = payload_words + D / 64;
                constexpr int COLS = BLOCKS;
                constexpr int sets = simd_width / COLS;
                constexpr int rounds = (BLOCKS + COLS - 1) / COLS;
                constexpr int steps = N / BLOCKS;
                static_assert(BLOCKS == 8, "resident kernel requires eight blocks");
                static_assert(GQA == 2, "resident kernel requires GQA two");
                static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
                static_assert(N % BLOCKS == 0, "window is a multiple of the block count");

                const int kv_head = int(threadgroup_position_in_grid.x);
                const int batch_index = int(threadgroup_position_in_grid.y);
                const int simd = int(simdgroup_index_in_threadgroup);
                const int group = simd % BLOCKS;
                const int head = simd / BLOCKS;
                const int query_head = GQA * kv_head + head;
                const int lane = int(thread_index_in_simdgroup);
                (void)read_fence[0];

                threadgroup float local_partials[GQA * BLOCKS * D];
                threadgroup float local_sums[GQA * BLOCKS];
                threadgroup float local_maxs[GQA * BLOCKS];

                const device uint32_t* mirror_w = m0;
                switch (batch_index) {
                    case 1: mirror_w = m1; break;
                    case 2: mirror_w = m2; break;
                    case 3: mirror_w = m3; break;
                    case 4: mirror_w = m4; break;
                    case 5: mirror_w = m5; break;
                    case 6: mirror_w = m6; break;
                    case 7: mirror_w = m7; break;
                    default: break;
                }
                const device uint32_t* mkeys_w =
                    mirror_w + kv_head * N * row_words;
                const device uint32_t* mvalues_w =
                    mirror_w + (KV_HEADS + kv_head) * N * row_words;
                const uint start = starts[batch_index];

                thread float q[1][values_per_lane];
                thread float acc[1][values_per_lane];
                thread float maxs[1];
                thread float sums[1];
                const device T4* qvec = reinterpret_cast<const device T4*>(
                    queries + (batch_index * 16 + query_head) * D + lane * values_per_lane);
                #pragma clang loop unroll(full)
                for (int v = 0; v < values_per_lane / 4; ++v) {
                    const T4 q4 = qvec[v];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        q[0][v * 4 + j] = float(q4[j]);
                    }
                }
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    acc[0][element] = 0.0f;
                }
                maxs[0] = -3.402823466e+38F;
                sums[0] = 0.0f;

                // Block 0's first token is the boundary slot itself: skip it.
                const int first_step = group == 0 ? 1 : 0;
                uint slot = (start + uint(group + first_step * BLOCKS)) % uint(N);
                uint32_t kw = mkeys_w[slot * row_words + lane];
                uint32_t vw = mvalues_w[slot * row_words + lane];
                uint32_t ktw = mkeys_w[slot * row_words + payload_words + lane / 8];
                uint32_t vtw = mvalues_w[slot * row_words + payload_words + lane / 8];
                slot += uint(BLOCKS);
                if (slot >= uint(N)) slot -= uint(N);
                uint32_t kw_pre = mkeys_w[slot * row_words + lane];
                uint32_t vw_pre = mvalues_w[slot * row_words + lane];
                uint32_t ktw_pre = mkeys_w[slot * row_words + payload_words + lane / 8];
                uint32_t vtw_pre = mvalues_w[slot * row_words + payload_words + lane / 8];
                cbv2_wide_columns_token<1, values_per_lane, T>(
                    kw, vw, ktw, vtw, 0, 0, q, acc, maxs, sums);
                for (int step = first_step + 1; step < steps; ++step) {
                    kw = kw_pre;
                    vw = vw_pre;
                    ktw = ktw_pre;
                    vtw = vtw_pre;
                    if (step + 1 < steps) {
                        slot += uint(BLOCKS);
                        if (slot >= uint(N)) slot -= uint(N);
                        kw_pre = mkeys_w[slot * row_words + lane];
                        vw_pre = mvalues_w[slot * row_words + lane];
                        ktw_pre = mkeys_w[slot * row_words + payload_words + lane / 8];
                        vtw_pre = mvalues_w[slot * row_words + payload_words + lane / 8];
                    }
                    cbv2_wide_columns_token<1, values_per_lane, T>(
                        kw, vw, ktw, vtw, 0, 0, q, acc, maxs, sums);
                }

                if (lane == 0) {
                    local_sums[head * BLOCKS + group] = sums[0];
                    local_maxs[head * BLOCKS + group] = maxs[0];
                }
                threadgroup float4* partial_vec = reinterpret_cast<threadgroup float4*>(
                    local_partials + head * BLOCKS * D + group * D + lane * values_per_lane);
                #pragma clang loop unroll(full)
                for (int v = 0; v < values_per_lane / 4; ++v) {
                    float4 p4;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        p4[j] = acc[0][v * 4 + j];
                    }
                    partial_vec[v] = p4;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const int block_lane = lane % COLS;
                const int output_group = group * sets + lane / COLS;
                const threadgroup float* head_partials = local_partials
                    + head * BLOCKS * D + output_group * values_per_lane;
                const threadgroup float* head_sums = local_sums + head * BLOCKS;
                const threadgroup float* head_maxs = local_maxs + head * BLOCKS;
                device T* head_out = out
                    + (batch_index * 16 + query_head) * D
                    + output_group * values_per_lane;
                thread float accumulator[values_per_lane];
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = 0.0f;
                }
                thread float lane_max[rounds];
                thread float lane_sum[rounds];
                thread float lane_factor[rounds];
                float sum_exp_score = 0.0f;
                float max_score = -3.402823466e+38F;
                #pragma clang loop unroll(full)
                for (int round = 0; round < rounds; ++round) {
                    const int column = block_lane + COLS * round;
                    const bool live = column < BLOCKS;
                    lane_max[round] =
                        live ? head_maxs[column] : -3.402823466e+38F;
                    lane_sum[round] = live ? head_sums[column] : 0.0f;
                    max_score = max(max_score, lane_max[round]);
                }
                #pragma clang loop unroll(full)
                for (int stride = 1; stride < COLS; stride <<= 1) {
                    max_score = max(
                        max_score,
                        simd_shuffle_xor(max_score, ushort(stride)));
                }

                #pragma clang loop unroll(full)
                for (int round = 0; round < rounds; ++round) {
                    lane_factor[round] =
                        fast::exp(lane_max[round] - max_score);
                    sum_exp_score += lane_factor[round] * lane_sum[round];
                }
                #pragma clang loop unroll(full)
                for (int stride = 1; stride < COLS; stride <<= 1) {
                    sum_exp_score +=
                        simd_shuffle_xor(sum_exp_score, ushort(stride));
                }

                #pragma clang loop unroll(full)
                for (int round = 0; round < rounds; ++round) {
                    const int column = block_lane + COLS * round;
                    if (column < BLOCKS) {
                        const float factor = lane_factor[round];
                        const threadgroup float4* partial_vectors =
                            reinterpret_cast<const threadgroup float4*>(
                                head_partials + column * D);
                        #pragma clang loop unroll(full)
                        for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                            const float4 partial_vector = partial_vectors[chunk];
                            #pragma clang loop unroll(full)
                            for (int j = 0; j < 4; ++j) {
                                accumulator[chunk * 4 + j] +=
                                    factor * partial_vector[j];
                            }
                        }
                    }
                }

                #pragma clang loop unroll(full)
                for (int step = 0; (1 << step) < COLS; ++step) {
                    const ushort stride = ushort(1 << step);
                    const bool upper = (block_lane & int(stride)) != 0;
                    const int live = values_per_lane >> step;
                    #pragma clang loop unroll(full)
                    for (int slot = 0; slot < live; slot += 2) {
                        const float keep = upper
                            ? accumulator[slot + 1]
                            : accumulator[slot];
                        const float trade = upper
                            ? accumulator[slot]
                            : accumulator[slot + 1];
                        accumulator[slot >> 1] =
                            keep + simd_shuffle_xor(trade, stride);
                    }
                }
                head_out[block_lane] = T(
                    sum_exp_score == 0.0f
                        ? accumulator[0]
                        : accumulator[0] / sum_exp_score);
            """,
            header: wideColumnsKernelHeader,
            ensureRowContiguous: true
        )

    /// Drafter attention over the target's q4 sliding mirror: `queries`
    /// `[8, 16, 1, D]`, `startArray` the `[8]` uint32 boundary slot per row
    /// (the row's anchor mod window; that slot is skipped), `readFence`
    /// consumed for ordering. Output `[8, 16, 1, D]`; nil off the eight-row
    /// D256 ring geometry.
    static func attendRingQuantReading(
        queries: MLXArray,
        mirrors: [MLXArray],
        startArray: MLXArray,
        readFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int
    ) -> MLXArray? {
        guard CBv2WindowedSequenceKV.quantEnabled,
            blocks == 8, combineColumns == 8, combineThreads == 256,
            slidingWindowLength == sequenceLength,
            startArray.shape == [batch], startArray.dtype == .uint32,
            enabled,
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            readFence.dtype == .int32,
            readFence.shape == [1],
            mirrors.count == batch,
            mirrors.allSatisfy({
                $0.dtype == .uint32
                    && $0.shape == [2, kvHeads, sequenceLength, headDim / 8 + headDim / 64]
            })
        else { return nil }
        let threads = blocks * 32 * gqa
        let outputs = portQuantReadResidentKernel(
            [queries] + mirrors + [startArray, readFence],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * threads, batch, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )
        CBv2EngageMark.once("mtp-drafter-q4-mirror-attention")
        return outputs[0]
    }

    public static func selfTestReadKernel(mirror: MLXArray, queries: MLXArray) {
        let startArray = MLXArray(Array(repeating: UInt32(0), count: batch), [batch])
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let out = portQuantReadKernel(
            [queries] + Array(repeating: mirror, count: batch) + [startArray],
            template: [
                ("T", queries.dtype), ("D", headDim), ("N", sequenceLength),
                ("GQA", gqa), ("KV_HEADS", kvHeads), ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch * gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        let maxs = out[2].reshaped([-1]).asArray(Float.self)
        let partial = out[0].reshaped([-1]).asArray(Float.self)
        FileHandle.standardError.write(Data(
            "[kvq4-kernel] blocks=\(blocks) maxs[0..3]=\(maxs[0..<4]) partial[0..5]=\(partial[0..<6])\n".utf8))
    }

    static func attendRingQuant(
        queries: MLXArray,
        mirrors: [MLXArray],
        starts: [Int],
        scale: Float,
        slidingWindowLength: Int
    ) -> MLXArray? {
        guard CBv2WindowedSequenceKV.quantEnabled,
            slidingWindowLength == sequenceLength,
            starts.count == batch,
            starts.allSatisfy({ 0 <= $0 && $0 < sequenceLength }),
            enabled, blocks > 0, sequenceLength.isMultiple(of: blocks),
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            headDim == 256, kvHeads == 8,
            mirrors.count == batch,
            mirrors.allSatisfy({
                $0.dtype == .uint32
                    && $0.shape == [2, kvHeads, sequenceLength, headDim / 8 + headDim / 64]
            })
        else { return nil }

        let startArray = MLXArray(starts.map(UInt32.init), [batch])
        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = portQuantReadKernel(
            [queries] + mirrors + [startArray],
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
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        CBv2EngageMark.once("kvq8port")
        return passBActive(
            passA,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
                ("COLS", combineColumns),
            ],
            grid: (batch * queryHeads * combineThreads, 1, 1),
            threadGroup: (combineThreads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    static func attendRingQuantWriting(
        queries: MLXArray,
        mirrors: [MLXArray],
        starts: [Int],
        newKeys: MLXArray,
        newValues: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard starts.count == batch,
            starts.allSatisfy({ 0 <= $0 && $0 < sequenceLength })
        else { return nil }
        return attendRingQuantWriting(
            queries: queries, mirrors: mirrors,
            startArray: MLXArray(starts.map(UInt32.init), [batch]),
            newKeys: newKeys, newValues: newValues,
            previousWriteFence: previousWriteFence, scale: scale,
            slidingWindowLength: slidingWindowLength)
    }

    /// `startArray` is the `[8]` uint32 first-retained slot per row, host or
    /// device built (chained MTP rounds derive it from the device position
    /// chain).
    static func attendRingQuantWriting(
        queries: MLXArray,
        mirrors: [MLXArray],
        startArray: MLXArray,
        newKeys: MLXArray,
        newValues: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float,
        slidingWindowLength: Int
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        guard CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
            CBv2WindowedSequenceKV.quantEnabled,
            !CBv2WindowedSequenceKV.quantSimulate,
            !CBv2WindowedSequenceKV.gpuPackCheck,
            slidingWindowLength == sequenceLength,
            startArray.shape == [batch], startArray.dtype == .uint32,
            enabled, blocks > 0, sequenceLength.isMultiple(of: blocks),
            scale == 1.0,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim],
            newKeys.dtype == .bfloat16,
            newKeys.shape == [batch, kvHeads, 1, headDim],
            newValues.dtype == .bfloat16,
            newValues.shape == newKeys.shape,
            previousWriteFence.dtype == .int32,
            previousWriteFence.shape == [1],
            mirrors.count == batch,
            mirrors.allSatisfy({
                $0.dtype == .uint32
                    && $0.shape == [2, kvHeads, sequenceLength, headDim / 8 + headDim / 64]
            })
        else { return nil }

        let inputs = [queries] + mirrors
            + [startArray, newKeys, newValues, previousWriteFence]
        if q4ResidentMergeEnabled,
            blocks == 8,
            combineColumns == 8,
            combineThreads == 256
        {
            let resident = portQuantFusedWriteResidentKernel(
                inputs,
                template: [
                    ("T", queries.dtype),
                    ("D", headDim),
                    ("N", sequenceLength),
                    ("GQA", gqa),
                    ("KV_HEADS", kvHeads),
                    ("BLOCKS", blocks),
                ],
                grid: (kvHeads * blocks * 32, batch, 1),
                threadGroup: (blocks * 32, 1, 1),
                outputShapes: [[batch, queryHeads, 1, headDim], [1]],
                outputDTypes: [.bfloat16, .int32]
            )
            CBv2EngageMark.once("kvq4-fused-live-write")
            CBv2EngageMark.once("kvq4-resident-merge")
            return (resident[0], resident[1])
        }

        let partialShape = [batch, queryHeads, 1, blocks, headDim]
        let summaryShape = [batch, queryHeads, 1, blocks]
        let passA = portQuantFusedWriteKernel(
            inputs,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("KV_HEADS", kvHeads),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, batch, blocks),
            threadGroup: (32, 1, 1),
            outputShapes: [partialShape, summaryShape, summaryShape, [1]],
            outputDTypes: [.bfloat16, .float32, .float32, .int32]
        )
        let output = passBActive(
            Array(passA.prefix(3)),
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
                ("COLS", combineColumns),
            ],
            grid: (batch * queryHeads * combineThreads, 1, 1),
            threadGroup: (combineThreads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
        CBv2EngageMark.once("kvq4-fused-live-write")
        return (output, passA[3])
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
        let paired = gqaPairedPassAEnabled && gqa == 2
        let vectorized = paired && gqaPairedPassAVec4Enabled && headDim % 4 == 0
        if paired {
            CBv2EngageMark.once("decode-gqa-paired-passa")
        }
        if vectorized {
            CBv2EngageMark.once("decode-gqa-paired-passa-vec4")
        }
        let pairedKernel =
            vectorized ? fusedRingPassAPairedVec4Kernel : fusedRingPassAPairedKernel
        let passA = (paired ? pairedKernel : fusedRingPassAKernel)(
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
            grid: paired
                ? (kvHeads * 32, batch, blocks)
                : (kvHeads * 32, batch * gqa, blocks),
            threadGroup: paired ? (32, 1, 1) : (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape, [1]],
            outputDTypes: [.bfloat16, .float32, .float32, .int32]
        )

        let output = passBActive(
            Array(passA.prefix(3)),
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
                ("COLS", combineColumns),
            ],
            grid: (batch * queryHeads * combineThreads, 1, 1),
            threadGroup: (combineThreads, 1, 1),
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

        return passBActive(
            passA,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
                ("COLS", combineColumns),
            ],
            grid: (batch * queryHeads * combineThreads, 1, 1),
            threadGroup: (combineThreads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

enum CBv2RaggedComposedD512DecodeAttentionV1 {
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

    private static let minKeyLength = 4
    private static let maxKeyLength = 4095

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
            constexpr int n_iter = D / 128;

            for (int vtg = vtg_lo; vtg < vtg_hi; ++vtg) {
                int out_row = vtg * 16 + sg * 4;
                if (out_row >= key_length) continue;
                out_row = out_row + 4 <= key_length
                    ? out_row : key_length - 4;

                const device T* mat = key_plane + size_t(out_row) * D;
                // XFOLD: one flat accumulator over the same 32 partial sums,
                // so the cross-lane fold below can address the whole set with
                // compile-time indices.
                float result[GQA * 4] = {0.0f};
                // KTILE: the 4x4 key tile is shared by all GQA heads, the
                // query block is not. Staging the tile costs 16 halves and
                // frees the 32-float per-head staging array.
                typedef vec<T, 4> T4;
                T4 k_tile[4];
                float q_coeff[4];
                int bn = lane * 4;
                for (int i = 0; i < n_iter; ++i) {
                    int mat_offset = 0;
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        k_tile[tm] = *reinterpret_cast<const device T4*>(
                            mat + mat_offset + bn);
                        mat_offset += D;
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        const T4 q_raw = *reinterpret_cast<const device T4*>(
                            query + h * D + bn);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            q_coeff[tn] = static_cast<float>(q_raw[tn]);
                        }
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                result[h * 4 + tm] +=
                                    k_tile[tm][tn] * q_coeff[tn];
                            }
                        }
                    }
                    bn += 128;
                }
                // XFOLD: the 32 sums fold across the simdgroup as ONE
                // butterfly over the whole set rather than 32 independent
                // shuffle-down chains. Step K halves the set every lane still
                // carries, so the traffic is 16 + 8 + 4 + 2 + 1 = 31 shuffles
                // instead of 32 * 5 = 160, and the live accumulator collapses
                // 32 -> 16 -> 8 -> 4 -> 2 -> 1 instead of staying 32 wide for
                // the whole fold.
                //
                // Exactness: step K merges the group holding lane l with the
                // group holding lane l ^ K, so after k steps every lane's
                // group is the coset of the same subgroup <16, 8, ...>. The
                // merge hierarchy is therefore the SAME for every lane and the
                // same as the shuffle-down form's: 16 pairs {l, l+16}, 8 quads
                // {l, l+8, l+16, l+24}, and so on. Only the left/right order
                // at each node varies with the lane, and float addition is
                // commutative, so every sum is bit-identical. Measured: 0 of
                // 402,944 fp32 words and 0 of 2,687,232 bf16 words differ,
                // against a deliberately reassociated control that moves 66%
                // of the fp32 words.
                //
                // Landing: after the five steps bit i of a lane's surviving
                // value index equals bit i of the lane, so lane l holds the sum
                // for (h, tm) = (l >> 2, l & 3). Lane 0's run of 32 serialised
                // stores becomes one store per lane, to 32 distinct addresses.
                {
                    const bool hi = (lane & 16) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 16; ++j) {
                        const float a = result[j];
                        const float b = result[16 + j];
                        result[j] = (hi ? b : a)
                            + simd_shuffle_xor(hi ? a : b, ushort(16));
                    }
                }
                {
                    const bool hi = (lane & 8) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 8; ++j) {
                        const float a = result[j];
                        const float b = result[8 + j];
                        result[j] = (hi ? b : a)
                            + simd_shuffle_xor(hi ? a : b, ushort(8));
                    }
                }
                {
                    const bool hi = (lane & 4) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        const float a = result[j];
                        const float b = result[4 + j];
                        result[j] = (hi ? b : a)
                            + simd_shuffle_xor(hi ? a : b, ushort(4));
                    }
                }
                {
                    const bool hi = (lane & 2) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 2; ++j) {
                        const float a = result[j];
                        const float b = result[2 + j];
                        result[j] = (hi ? b : a)
                            + simd_shuffle_xor(hi ? a : b, ushort(2));
                    }
                }
                {
                    const bool hi = (lane & 1) != 0;
                    const float a = result[0];
                    const float b = result[1];
                    result[0] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(1));
                }
                score_rows[size_t(lane >> 2) * key_length + out_row
                    + (lane & 3)] = static_cast<T>(result[0]);
            }
        """

    private static let qkKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_qk_bf16_g8_xfold_v3_vec1",
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
        name: "cbv2_ragged8_sdpa_d512_qk_fenced_bf16_g8_xfold_v3_vec1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params", "store_fence",
        ],
        outputNames: ["scores"],
        source: qkSource,
        ensureRowContiguous: true
    )

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

    private static let avColumnTiles: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_AV_TILES"], let value = Int(raw)
        else { return 16 }
        return value == 8 || value == 16 ? value : 16
    }()

    private static let avTileColumns = headDim / avColumnTiles
    private static let avSimdgroups = avTileColumns / 16

    private static let avKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_xfold_v3_t\(avColumnTiles)_vec1",
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
            const int tile = z % \(avColumnTiles);
            const int row_kv = z / \(avColumnTiles);
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
            const int out_col = tile * \(avTileColumns) + (4 * sg + thrN) * 4;

            // XFOLD: one flat accumulator over the same 32 partial sums, so
            // the cross-lane fold below can address the whole set with
            // compile-time indices.
            float result[GQA * 4] = {0.0f};
            // VTILE: the 4x4 value tile is shared by all GQA heads, the
            // probability block is not. Staging the tile costs 16 halves and
            // frees the 32-float per-head staging array.
            typedef vec<T, 4> T4;
            T4 v_tile[4];
            float p_coeff[4];
            const int n_iter = key_length / 32;
            const int leftover = key_length - n_iter * 32;

            for (int i = 0; i < n_iter; ++i) {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    v_tile[tm] = *reinterpret_cast<const device T4*>(
                        value_plane + size_t(bm + tm) * D + out_col);
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        p_coeff[tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        float vc = p_coeff[tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h * 4 + tn] += vc * v_tile[tm][tn];
                        }
                    }
                }
                bm += 32;
            }
            if (leftover > 0) {
                for (int tm = 0; tm < 4 && bm + tm < key_length; ++tm) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; ++tn) {
                        v_tile[0][tn] = value_plane[
                            size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        const float pc = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h * 4 + tn] += pc * v_tile[0][tn];
                        }
                    }
                }
            }
            // XFOLD: the 32 sums fold across the eight lanes that share this
            // thrN as ONE butterfly over the whole set rather than 32
            // independent shuffle-down chains. The traffic is 16 + 8 + 4 = 28
            // shuffles instead of 32 * 3 = 96, and the live accumulator
            // collapses 32 -> 16 -> 8 -> 4 instead of staying 32 wide.
            //
            // Exactness: as in the QK fold, step K merges the group holding
            // lane l with the group holding lane l ^ K, so the merge hierarchy
            // over the eight thrM lanes is the same coset chain for every lane
            // and the same one the shuffle-down form built. Only the left and
            // right order at each node varies, and float addition is
            // commutative. Measured bit-identical alongside the QK fold.
            //
            // Landing: bit i of a lane's surviving head index equals bit i + 2
            // of the lane, so the lane finishes holding head thrM's four
            // columns. The eight thrM == 0 lanes' run of 32 stores becomes
            // four stores on every lane, to the same 32 addresses per thrN.
            {
                const bool hi = (lane & 16) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 16; ++j) {
                    const float a = result[j];
                    const float b = result[16 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(16));
                }
            }
            {
                const bool hi = (lane & 8) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; ++j) {
                    const float a = result[j];
                    const float b = result[8 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(8));
                }
            }
            {
                const bool hi = (lane & 4) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    const float a = result[j];
                    const float b = result[4 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(4));
                }
            }
            {
                device T* out_ptr = out
                    + size_t(row * 16 + kv_head * GQA + thrM) * D
                    + out_col;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    out_ptr[j] = static_cast<T>(result[j]);
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let fusedQkKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_qk_bf16_g8_ktile_v3_vec1",
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
            constexpr int n_iter = D / 128;

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
                // KTILE: the 4x4 key tile is shared by all GQA heads, the
                // query block is not. Staging the tile costs 16 halves and
                // frees the 32-float per-head staging array. The peel keeps
                // both arms, it now only decides how the tile is filled.
                typedef vec<T, 4> T4;
                T4 k_tile[4];
                float q_coeff[4];
                int bn = lane * 4;
                for (int i = 0; i < n_iter; ++i) {
                    int mat_offset = 0;
                    if (!tile_has_new_token) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        k_tile[tm] = *reinterpret_cast<const device T4*>(
                            mat + mat_offset + bn);
                        mat_offset += D;
                    }
                    } else {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        const bool is_new_token =
                            out_row + tm == key_length - 1;
                        k_tile[tm] = is_new_token
                            ? *reinterpret_cast<const device T4*>(
                                  new_key_plane + bn)
                            : *reinterpret_cast<const device T4*>(
                                  mat + mat_offset + bn);
                        mat_offset += D;
                    }
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        const T4 q_raw = *reinterpret_cast<const device T4*>(
                            query + h * D + bn);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            q_coeff[tn] = static_cast<float>(q_raw[tn]);
                        }
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            #pragma clang loop unroll(full)
                            for (int tn = 0; tn < 4; ++tn) {
                                result[h][tm] +=
                                    k_tile[tm][tn] * q_coeff[tn];
                            }
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
                typedef vec<T, 4> T4;
                device T4* write_key = reinterpret_cast<device T4*>(
                    const_cast<device T*>(key_plane)
                        + size_t(key_length - 1) * D + lane * 16);
                device T4* write_value = reinterpret_cast<device T4*>(
                    const_cast<device T*>(value_plane)
                        + size_t(key_length - 1) * D + lane * 16);
                const device T4* src_key = reinterpret_cast<const device T4*>(
                    new_key_plane + lane * 16);
                const device T4* src_value =
                    reinterpret_cast<const device T4*>(
                        new_value_plane + lane * 16);
                #pragma clang loop unroll(full)
                for (int element = 0; element < 4; ++element) {
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

    private static let fusedAvKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_av_bf16_g8_vtile_v3_vec1",
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
            // VTILE: the 4x4 value tile is shared by all GQA heads, the
            // probability block is not. Staging the tile costs 16 halves and
            // frees the 32-float per-head staging array. The peel keeps both
            // arms, it now only decides how the tile is filled.
            typedef vec<T, 4> T4;
            T4 v_tile[4];
            float p_coeff[4];
            const int n_iter = key_length / 32;
            const int leftover = key_length - n_iter * 32;

            for (int i = 0; i < n_iter; ++i) {
                // Tile-level peel: only the 4-row tile containing logical
                // row kL-1 pays the serve-from-input branch.
                if (bm + 4 <= key_length - 1) {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    v_tile[tm] = *reinterpret_cast<const device T4*>(
                        value_plane + size_t(bm + tm) * D + out_col);
                }
                } else {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    const bool is_new_token = bm + tm == key_length - 1;
                    v_tile[tm] = is_new_token
                        ? *reinterpret_cast<const device T4*>(
                              new_value_plane + out_col)
                        : *reinterpret_cast<const device T4*>(
                              value_plane + size_t(bm + tm) * D + out_col);
                }
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        p_coeff[tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                    }
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        float vc = p_coeff[tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += vc * v_tile[tm][tn];
                        }
                    }
                }
                bm += 32;
            }
            if (leftover > 0) {
                for (int tm = 0; tm < 4 && bm + tm < key_length; ++tm) {
                    const bool is_new_token = bm + tm == key_length - 1;
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; ++tn) {
                        v_tile[0][tn] = is_new_token
                            ? new_value_plane[out_col + tn]
                            : value_plane[
                                  size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        const float pc = static_cast<float>(
                            prob_rows[size_t(h) * key_length + bm + tm]);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h][tn] += pc * v_tile[0][tn];
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

    private static let fusedWriteEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_FUSED_WRITE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let ringStoreKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_d512_ringstore_bf16_v1_vec1",
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

            typedef vec<T, 4> T4;
            device T4* write_key = reinterpret_cast<device T4*>(
                const_cast<device T*>(key_plane)
                    + size_t(key_length - 1) * D + lane * 4);
            device T4* write_value = reinterpret_cast<device T4*>(
                const_cast<device T*>(value_plane)
                    + size_t(key_length - 1) * D + lane * 4);
            const device T4* src_key = reinterpret_cast<const device T4*>(
                new_keys + size_t(row * 2 + kv_head) * D + lane * 4);
            const device T4* src_value = reinterpret_cast<const device T4*>(
                new_values + size_t(row * 2 + kv_head) * D + lane * 4);
            write_key[0] = src_key[0];
            write_value[0] = src_value[0];
            if (z == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """,
        ensureRowContiguous: true)

    private static let storeDispatchEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_STORE_DISPATCH"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()


    /// Per-row key lengths for the ragged verify road: `params` is
    /// `[stride, D, capacity x8, key_length x8]`; the scratch keeps a
    /// max-length stride and every row runs the uniform body's statements over
    /// its own length, so a row's scores are the uniform kernel's.
    private static func transformed(
        _ source: String, _ replacements: [(String, String)]
    ) -> String {
        var text = source
        for (old, new) in replacements {
            precondition(
                text.components(separatedBy: old).count == 2,
                "kernel source marker missing: \(old)")
            text = text.replacingOccurrences(of: old, with: new)
        }
        return text
    }

    private static let qkRowsSource: String = transformed(
        qkSource,
        [
            (
                "const int key_length = int(params[0]);",
                "const int key_stride = int(params[0]);"
            ),
            (
                "const int n_chunks = (key_length + 63) / 64;",
                "const int n_chunks = (key_stride + 63) / 64;"
            ),
            (
                "const int kv_head = row_kv % 2;",
                "const int kv_head = row_kv % 2;\n            const int key_length = int(params[10 + row]);"
            ),
            (
                "scores + size_t(row * 16 + kv_head * GQA) * key_length;",
                "scores + size_t(row * 16 + kv_head * GQA) * key_stride;"
            ),
            (
                "score_rows[size_t(lane >> 2) * key_length + out_row",
                "score_rows[size_t(lane >> 2) * key_stride + out_row"
            ),
        ])

    private static let qkFencedRowsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_qk_fenced_bf16_g8_xfold_rows_v1",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params", "store_fence",
        ],
        outputNames: ["scores"],
        source: qkRowsSource,
        ensureRowContiguous: true
    )

    private static let softmaxRowsSource: String = """
            const int axis_stride = int(params[0]);
            const int gid = int(threadgroup_position_in_grid.x);
            const int axis_size = int(params[10 + gid / 16]);
            const int lid = int(thread_position_in_threadgroup.x);
            const int simd_lane_id = int(thread_index_in_simdgroup);
            const int simd_group_id = int(simdgroup_index_in_threadgroup);

            threadgroup float local_max[32];
            threadgroup float local_normalizer[32];

            float ld[4];
            const device T* in =
                scores + size_t(gid) * axis_stride + lid * 4;
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
                probs + size_t(gid) * axis_stride + lid * 4;
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
        """

    private static let softmaxRowsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_softmax_bf16_rows_v1",
        inputNames: ["scores", "params"],
        outputNames: ["probs"],
        source: softmaxRowsSource,
        ensureRowContiguous: true
    )

    private static let avRowsSource: String = """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_stride = int(params[0]);

            const int z = int(threadgroup_position_in_grid.z);
            const int tile = z % \(avColumnTiles);
            const int row_kv = z / \(avColumnTiles);
            const int row = row_kv / 2;
            const int kv_head = row_kv % 2;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);

            const int row_capacity = int(params[2 + row]);
            const int key_length = int(params[10 + row]);

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
                probs + size_t(row * 16 + kv_head * GQA) * key_stride;

            const int thrM = lane / 4;
            const int thrN = lane % 4;
            int bm = thrM * 4;
            const int out_col = tile * \(avTileColumns) + (4 * sg + thrN) * 4;

            // XFOLD: one flat accumulator over the same 32 partial sums, so
            // the cross-lane fold below can address the whole set with
            // compile-time indices.
            float result[GQA * 4] = {0.0f};
            // VTILE: the 4x4 value tile is shared by all GQA heads, the
            // probability block is not. Staging the tile costs 16 halves and
            // frees the 32-float per-head staging array.
            typedef vec<T, 4> T4;
            T4 v_tile[4];
            float p_coeff[4];
            const int n_iter = key_length / 32;
            const int leftover = key_length - n_iter * 32;

            for (int i = 0; i < n_iter; ++i) {
                #pragma clang loop unroll(full)
                for (int tm = 0; tm < 4; ++tm) {
                    v_tile[tm] = *reinterpret_cast<const device T4*>(
                        value_plane + size_t(bm + tm) * D + out_col);
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < GQA; ++h) {
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        p_coeff[tm] = static_cast<float>(
                            prob_rows[size_t(h) * key_stride + bm + tm]);
                    }
                    #pragma clang loop unroll(full)
                    for (int tm = 0; tm < 4; ++tm) {
                        float vc = p_coeff[tm];
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h * 4 + tn] += vc * v_tile[tm][tn];
                        }
                    }
                }
                bm += 32;
            }
            if (leftover > 0) {
                for (int tm = 0; tm < 4 && bm + tm < key_length; ++tm) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; ++tn) {
                        v_tile[0][tn] = value_plane[
                            size_t(bm + tm) * D + out_col + tn];
                    }
                    #pragma clang loop unroll(full)
                    for (int h = 0; h < GQA; ++h) {
                        const float pc = static_cast<float>(
                            prob_rows[size_t(h) * key_stride + bm + tm]);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; ++tn) {
                            result[h * 4 + tn] += pc * v_tile[0][tn];
                        }
                    }
                }
            }
            // XFOLD: the 32 sums fold across the eight lanes that share this
            // thrN as ONE butterfly over the whole set rather than 32
            // independent shuffle-down chains. The traffic is 16 + 8 + 4 = 28
            // shuffles instead of 32 * 3 = 96, and the live accumulator
            // collapses 32 -> 16 -> 8 -> 4 instead of staying 32 wide.
            //
            // Exactness: as in the QK fold, step K merges the group holding
            // lane l with the group holding lane l ^ K, so the merge hierarchy
            // over the eight thrM lanes is the same coset chain for every lane
            // and the same one the shuffle-down form built. Only the left and
            // right order at each node varies, and float addition is
            // commutative. Measured bit-identical alongside the QK fold.
            //
            // Landing: bit i of a lane's surviving head index equals bit i + 2
            // of the lane, so the lane finishes holding head thrM's four
            // columns. The eight thrM == 0 lanes' run of 32 stores becomes
            // four stores on every lane, to the same 32 addresses per thrN.
            {
                const bool hi = (lane & 16) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 16; ++j) {
                    const float a = result[j];
                    const float b = result[16 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(16));
                }
            }
            {
                const bool hi = (lane & 8) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 8; ++j) {
                    const float a = result[j];
                    const float b = result[8 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(8));
                }
            }
            {
                const bool hi = (lane & 4) != 0;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    const float a = result[j];
                    const float b = result[4 + j];
                    result[j] = (hi ? b : a)
                        + simd_shuffle_xor(hi ? a : b, ushort(4));
                }
            }
            {
                device T* out_ptr = out
                    + size_t(row * 16 + kv_head * GQA + thrM) * D
                    + out_col;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    out_ptr[j] = static_cast<T>(result[j]);
                }
            }
        """

    private static let avRowsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_xfold_rows_t\(avColumnTiles)_v1",
        inputNames: [
            "probs",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params",
        ],
        outputNames: ["out"],
        source: avRowsSource,
        ensureRowContiguous: true
    )

    private static let ringStoreRowsSource: String = """
            constexpr int D = 512;
            const int z = int(threadgroup_position_in_grid.z);
            const int row = z / 2;
            const int kv_head = z % 2;
            const int lane = int(thread_position_in_threadgroup.x);
            const int key_length = int(params[10 + row]);
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

            typedef vec<T, 4> T4;
            device T4* write_key = reinterpret_cast<device T4*>(
                const_cast<device T*>(key_plane)
                    + size_t(key_length - 1) * D + lane * 4);
            device T4* write_value = reinterpret_cast<device T4*>(
                const_cast<device T*>(value_plane)
                    + size_t(key_length - 1) * D + lane * 4);
            const device T4* src_key = reinterpret_cast<const device T4*>(
                new_keys + size_t(row * 2 + kv_head) * D + lane * 4);
            const device T4* src_value = reinterpret_cast<const device T4*>(
                new_values + size_t(row * 2 + kv_head) * D + lane * 4);
            write_key[0] = src_key[0];
            write_value[0] = src_value[0];
            if (z == 0 && lane == 0) {
                fence[0] = write_fence[0] + 1;
            }
        """

    private static let ringStoreRowsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_d512_ringstore_bf16_rows_v1",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["fence"],
        source: ringStoreRowsSource,
        ensureRowContiguous: true)

    /// Wide MTP verify twins of the `_rows_v1` kernels. The rectangle is
    /// row-major (`row * C + column`); column `c` of a row stores at and
    /// attends `params[10 + row] + c` keys, and every statement a key reaches
    /// is the rows kernel's, so each (row, column) result is the per-column
    /// pass's bit for bit. `params[18 + row]` is the row's element offset
    /// inside its K/V input, so one cohort pool `[8, 2, capacity, D]` passed
    /// as every plane addresses like eight private buffers.
    private static let qkColumnsSource: String = transformed(
        qkRowsSource,
        [
            (
                "const int row_kv = z / n_chunks;",
                "const int rect = z / n_chunks;\n            const int column = rect % C;\n            const int row_kv = rect / C;"
            ),
            (
                "const int key_length = int(params[10 + row]);",
                "const int key_length = int(params[10 + row]) + column;"
            ),
            (
                "queries + size_t(row * 16 + kv_head * GQA) * D;",
                "queries + size_t((row * C + column) * 16 + kv_head * GQA) * D;"
            ),
            (
                "scores + size_t(row * 16 + kv_head * GQA) * key_stride;",
                "scores + size_t((row * C + column) * 16 + kv_head * GQA) * key_stride;"
            ),
            (
                "key_plane += size_t(kv_head) * size_t(row_capacity) * D;",
                "key_plane += size_t(params[18 + row]) + size_t(kv_head) * size_t(row_capacity) * D;"
            ),
        ])

    private static let softmaxColumnsSource: String = transformed(
        softmaxRowsSource,
        [
            (
                "const int axis_size = int(params[10 + gid / 16]);",
                "const int axis_size = int(params[10 + gid / 16 / C]) + (gid / 16) % C;"
            )
        ])

    private static let avColumnsSource: String = transformed(
        avRowsSource,
        [
            (
                "const int row_kv = z / \(avColumnTiles);",
                "const int rect = z / \(avColumnTiles);\n            const int column = rect % C;\n            const int row_kv = rect / C;"
            ),
            (
                "const int key_length = int(params[10 + row]);",
                "const int key_length = int(params[10 + row]) + column;"
            ),
            (
                "probs + size_t(row * 16 + kv_head * GQA) * key_stride;",
                "probs + size_t((row * C + column) * 16 + kv_head * GQA) * key_stride;"
            ),
            (
                "+ size_t(row * 16 + kv_head * GQA + thrM) * D",
                "+ size_t((row * C + column) * 16 + kv_head * GQA + thrM) * D"
            ),
            (
                "value_plane += size_t(kv_head) * size_t(row_capacity) * D;",
                "value_plane += size_t(params[18 + row]) + size_t(kv_head) * size_t(row_capacity) * D;"
            ),
        ])

    private static let ringStoreColumnsSource: String = transformed(
        ringStoreRowsSource,
        [
            (
                "const int row = z / 2;",
                "const int row_kv = z / C;\n            const int column = z % C;\n            const int row = row_kv / 2;"
            ),
            (
                "const int kv_head = z % 2;",
                "const int kv_head = row_kv % 2;"
            ),
            (
                "const int key_length = int(params[10 + row]);",
                "const int key_length = int(params[10 + row]) + column;"
            ),
            (
                "new_keys + size_t(row * 2 + kv_head) * D + lane * 4);",
                "new_keys + size_t((row * C + column) * 2 + kv_head) * D + lane * 4);"
            ),
            (
                "new_values + size_t(row * 2 + kv_head) * D + lane * 4);",
                "new_values + size_t((row * C + column) * 2 + kv_head) * D + lane * 4);"
            ),
            (
                "key_plane += size_t(kv_head) * size_t(row_capacity) * D;",
                "key_plane += size_t(params[18 + row]) + size_t(kv_head) * size_t(row_capacity) * D;"
            ),
            (
                "value_plane += size_t(kv_head) * size_t(row_capacity) * D;",
                "value_plane += size_t(params[18 + row]) + size_t(kv_head) * size_t(row_capacity) * D;"
            ),
        ])

    private static let ringStoreColumnsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_d512_ringstore_bf16_columns_v2",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "new_keys", "new_values", "write_fence",
        ],
        outputNames: ["fence"],
        source: ringStoreColumnsSource,
        ensureRowContiguous: true)

    private static let qkFencedColumnsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_qk_fenced_bf16_g8_xfold_columns_v2",
        inputNames: [
            "queries",
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "params", "store_fence",
        ],
        outputNames: ["scores"],
        source: qkColumnsSource,
        ensureRowContiguous: true
    )

    private static let softmaxColumnsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_softmax_bf16_columns_v1",
        inputNames: ["scores", "params"],
        outputNames: ["probs"],
        source: softmaxColumnsSource,
        ensureRowContiguous: true
    )

    private static let avColumnsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_xfold_columns_t\(avColumnTiles)_v2",
        inputNames: [
            "probs",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params",
        ],
        outputNames: ["out"],
        source: avColumnsSource,
        ensureRowContiguous: true
    )

    /// Ragged twin of `updateAndAttendWriting22`: each row stores its new K/V
    /// at its own length and attends its own prefix. `lengths` is the int32
    /// `[8]` per-row key length INCLUDING the new token (device or host
    /// built); `maxKeyLength` is a host upper bound used for the grid and the
    /// scratch stride only. Takes over only when the fused uniform road is
    /// refused (rows at different offsets), so decode keeps its kernels.
    /// `columns` > 1 is the wide MTP verify rectangle `[8*C, ...]`, row-major
    /// (`row * C + column`): `lengths` is then column 0's length, column `c`
    /// attends `lengths + c` keys, and `maxKeyLength` bounds the LAST column.
    static func updateAndAttendWritingRagged(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        previousWriteFence: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?,
        lengths: MLXArray, maxKeyLength: Int, columns: Int = 1
    ) -> (output: MLXArray, nextWriteFence: MLXArray)? {
        let rectangle = batch * columns
        guard enabled, storeDispatchEnabled,
            rows.count == batch, columns >= 1, columns <= 4,
            scale == 1.0, sinks == nil, softcap == nil,
            !kind.isBidirectional,
            kind.kvHeads == kvHeads, kind.headDim == headDim, kind.queryHeads == queryHeads,
            queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16,
            queries.shape == [rectangle, queryHeads, 1, headDim],
            keys.shape == [rectangle, kvHeads, 1, headDim],
            values.shape == keys.shape,
            previousWriteFence.dtype == .int32, previousWriteFence.shape == [1],
            lengths.dtype == .int32, lengths.shape == [batch],
            maxKeyLength >= minKeyLength, maxKeyLength <= Self.maxKeyLength
        else { return nil }
        guard case .full = kind.attention else { return nil }
        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        let wide = columns > 1
        guard fullRows.count == batch,
            fullRows.allSatisfy({ maxKeyLength <= $0.maxLength }),
            let planes = raggedPlanes(fullRows, columns: columns, allowPool: wide)
        else { return nil }

        let paramsArray = concatenated(
            [
                MLXArray([UInt32(maxKeyLength), UInt32(headDim)] + planes.capacities),
                lengths.asType(.uint32),
                MLXArray(planes.offsets),
            ], axis: 0)
        var template: [(String, any KernelTemplateArg)] = [("T", queries.dtype)]
        if wide { template.append(("C", columns)) }
        let scratchShape = [rectangle, queryHeads, 1, maxKeyLength]
        let keyBuffers = planes.keys
        let valueBuffers = planes.values
        let kernels = wide
            ? (store: ringStoreColumnsKernel, qk: qkFencedColumnsKernel,
               softmax: softmaxColumnsKernel, av: avColumnsKernel)
            : (store: ringStoreRowsKernel, qk: qkFencedRowsKernel,
               softmax: softmaxRowsKernel, av: avRowsKernel)

        let storeFence = kernels.store(
            keyBuffers + valueBuffers + [paramsArray, keys, values, previousWriteFence],
            template: template,
            grid: (128, 1, rectangle * kvHeads),
            threadGroup: (128, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32]
        )[0]
        let chunks = (maxKeyLength + 63) / 64
        let scores = kernels.qk(
            [queries] + keyBuffers + [paramsArray, storeFence],
            template: template,
            grid: (32, 4, rectangle * kvHeads * chunks),
            threadGroup: (32, 4, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]
        let softmaxThreads = ((maxKeyLength + 3) / 4 + 31) / 32 * 32
        let probs = kernels.softmax(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * rectangle * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]
        let output = kernels.av(
            [probs] + valueBuffers + [paramsArray],
            template: template,
            grid: (32, avSimdgroups, rectangle * kvHeads * avColumnTiles),
            threadGroup: (32, avSimdgroups, 1),
            outputShapes: [[rectangle, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
        for row in fullRows {
            if planes.pooled {
                row.confirmPooledBatchAppend(columns)
            } else {
                for _ in 0 ..< columns { row.advanceAfterFusedAppend() }
            }
        }
        CBv2EngageMark.once(
            !wide ? "d512-ragged-rows"
                : planes.pooled ? "d512-ragged-columns-pooled" : "d512-ragged-columns")
        return (output, storeFence)
    }

    private struct RaggedPlanes {
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        var capacities: [UInt32] = []
        var offsets: [UInt32] = []
        var pooled = false
    }

    /// Per-row K/V planes for the ragged roads: eight private buffers, or
    /// (wide only) the rows' ONE existing cohort pool addressed through
    /// per-row element offsets. Never forms a pool. A row needs capacity for
    /// its OWN new slots only: the kernels touch slots below each row's key
    /// length, and host counters never lag below the device truth.
    private static func raggedPlanes(
        _ fullRows: [CBv2FullSequenceKV], columns: Int, allowPool: Bool
    ) -> RaggedPlanes? {
        var planes = RaggedPlanes()
        if let pool = fullRows[0].cohortPool {
            guard allowPool, pool.rowCount == batch else { return nil }
            for (index, row) in fullRows.enumerated() {
                guard row.cohortPool === pool, row.cohortIndex == index else { return nil }
            }
            planes.pooled = true
        }
        for row in fullRows {
            let state = row.cbv2InnerState()
            guard state.count == 2,
                state[0].dtype == .bfloat16, state[1].dtype == .bfloat16,
                state[0].ndim == 4, state[0].dim(0) == (planes.pooled ? batch : 1),
                state[0].dim(1) == kvHeads, state[0].dim(3) == headDim,
                state[1].shape == state[0].shape,
                state[0].dim(2) >= row.absoluteOffset + columns
            else { return nil }
            let capacity = state[0].dim(2)
            planes.keys.append(state[0])
            planes.values.append(state[1])
            planes.capacities.append(UInt32(capacity))
            planes.offsets.append(
                UInt32((planes.pooled ? row.cohortIndex : 0) * kvHeads * capacity * headDim))
        }
        return planes
    }

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
            grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
            threadGroup: (32, avSimdgroups, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]

        for row in fullRows {
            row.advanceAfterFusedAppend()
        }
        return (output, storeFence)
    }

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
            grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
            threadGroup: (32, avSimdgroups, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
