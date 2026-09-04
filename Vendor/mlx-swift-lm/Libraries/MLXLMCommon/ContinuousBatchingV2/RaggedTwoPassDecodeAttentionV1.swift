// Ranked replication by delordemm1 of the parity-clean resident q4 merge
// published in submission bc839700. This marker is non-executable; the kernel
// and host implementation below remain byte-identical to that measured tree.
// Independent current-crown replication by delordemm1; the executable PNIB
// kernel source below is preserved exactly from the public PR #2603 artifact.
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

public enum CBv2RaggedTwoPassDecodeAttentionV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RAGGED_TWO_PASS_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Keep all eight q4 pass-A partitions resident in one threadgroup and
    /// transcribe pass B from threadgroup memory. The established two-dispatch
    /// path remains available as a same-binary control.
    private static let q4ResidentMergeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_Q4_RESIDENT_MERGE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// F4: consume raw decode Q/K/V in the resident q4 kernel, applying the
    /// exact Gemma sliding-layer RMSNorm and base RoPE prologue there. Off
    /// leaves the established `gemma4_b8_qkv_rms_norm_rope_v2_vec1` graph as
    /// the attention input and selects the untouched crown resident kernel.
    static let q4ResidentNormRopeEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_Q4_RESIDENT_NORM_ROPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private struct ResidentNormRopeInputs {
        let normalizedKeys: ObjectIdentifier
        let normalizedValues: ObjectIdentifier
        let rawQueries: MLXArray
        let rawKeys: MLXArray
        let rawValues: MLXArray
        let qWeight: MLXArray
        let kWeight: MLXArray
        let positionOffsets: MLXArray
        let ropeLog2Base: MLXArray
    }

    private static let residentNormRopeLock = NSLock()
    nonisolated(unsafe) private static var residentNormRopeInputs:
        [ObjectIdentifier: ResidentNormRopeInputs] = [:]

    /// Register the raw producer inputs behind an already-built exact
    /// norm+RoPE fallback. The fallback arrays remain what the generic cache
    /// sees; only the exact resident path consumes this carrier, so any miss
    /// evaluates the established graph unchanged.
    @discardableResult
    public static func registerResidentNormRope(
        normalizedQueries: MLXArray,
        normalizedKeys: MLXArray,
        normalizedValues: MLXArray,
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray,
        qWeight: MLXArray,
        kWeight: MLXArray,
        positionOffsets: MLXArray,
        ropeLog2Base: MLXArray,
        eps: Float,
        appliedRope: Bool
    ) -> Bool {
        guard q4ResidentNormRopeEnabled,
            appliedRope,
            eps == 1.0e-6,
            enabled,
            q4ResidentMergeEnabled,
            blocks == 8,
            combineColumns == 8,
            combineThreads == 256,
            CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
            CBv2WindowedSequenceKV.q4BF16RingElideEnabled,
            CBv2WindowedSequenceKV.quantEnabled,
            !CBv2WindowedSequenceKV.quantSimulate,
            !CBv2WindowedSequenceKV.gpuPackCheck,
            normalizedQueries.dtype == .bfloat16,
            normalizedQueries.shape == [batch, queryHeads, 1, headDim],
            normalizedKeys.dtype == .bfloat16,
            normalizedKeys.shape == [batch, kvHeads, 1, headDim],
            normalizedValues.dtype == .bfloat16,
            normalizedValues.shape == normalizedKeys.shape,
            rawQueries.dtype == .bfloat16,
            rawQueries.shape == [batch, 1, queryHeads, headDim],
            rawKeys.dtype == .bfloat16,
            rawKeys.shape == [batch, 1, kvHeads, headDim],
            rawValues.dtype == .bfloat16,
            rawValues.shape == rawKeys.shape,
            qWeight.dtype == .bfloat16,
            qWeight.shape == [headDim],
            kWeight.dtype == .bfloat16,
            kWeight.shape == [headDim],
            positionOffsets.dtype == .int32,
            positionOffsets.shape == [batch],
            ropeLog2Base.dtype == .float32,
            ropeLog2Base.shape == [1]
        else { return false }

        residentNormRopeLock.lock()
        if residentNormRopeInputs.count >= 64 {
            residentNormRopeInputs.removeAll(keepingCapacity: true)
        }
        residentNormRopeInputs[ObjectIdentifier(normalizedQueries)] =
            ResidentNormRopeInputs(
                normalizedKeys: ObjectIdentifier(normalizedKeys),
                normalizedValues: ObjectIdentifier(normalizedValues),
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                qWeight: qWeight,
                kWeight: kWeight,
                positionOffsets: positionOffsets,
                ropeLog2Base: ropeLog2Base)
        residentNormRopeLock.unlock()
        return true
    }

    @inline(__always)
    private static func takeResidentNormRope(
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) -> ResidentNormRopeInputs? {
        residentNormRopeLock.lock()
        let inputs = residentNormRopeInputs.removeValue(
            forKey: ObjectIdentifier(queries))
        residentNormRopeLock.unlock()
        guard let inputs,
            inputs.normalizedKeys == ObjectIdentifier(keys),
            inputs.normalizedValues == ObjectIdentifier(values)
        else { return nil }
        return inputs
    }

    /// Extra products of the F4 resident attention dispatch. The attention API
    /// intentionally remains an `MLXArray`; Gemma takes this host-side carrier
    /// immediately after `updateAndAttend` returns, before reshaping `output`.
    /// F2 is removed: `runsumTable` remains API-compatible and is always nil so
    /// Gemma must use `CBv2AttentionOQMVV1.runsumTable(for:)` unchanged.
    public struct ResidentProducts {
        public let runsumTable: MLXArray?
        public let normalizedKeys: MLXArray?
        public let normalizedValues: MLXArray?
    }

    private static let residentProductsLock = NSLock()
    nonisolated(unsafe) private static var residentProducts:
        [ObjectIdentifier: ResidentProducts] = [:]

    /// Shared with the D=512 chain (`CBv2RaggedComposedD512DecodeAttentionV1`),
    /// whose NORMROPE-D512 / ORS-D512 folds publish through the same carrier.
    @inline(__always)
    fileprivate static func publishResidentProducts(
        _ products: ResidentProducts, for output: MLXArray
    ) {
        residentProductsLock.lock()
        // The intended consumer takes the entry synchronously. Keep misuse or
        // old call sites bounded without retaining an unbounded lazy graph.
        if residentProducts.count >= 64 {
            residentProducts.removeAll(keepingCapacity: true)
        }
        residentProducts[ObjectIdentifier(output)] = products
        residentProductsLock.unlock()
    }

    /// Consume the extra outputs associated with this exact attention array.
    /// An empty carrier is the F4-off signal: use the established standalone
    /// producer and the already-normalized K/V values. This deliberately takes
    /// the same consumer branch as the parity-proven F4-on/F2-off state.
    @inline(__always)
    public static func takeResidentProducts(for output: MLXArray) -> ResidentProducts? {
        residentProductsLock.lock()
        let products = residentProducts.removeValue(forKey: ObjectIdentifier(output))
        residentProductsLock.unlock()
        // The F4 kill switch is also a hard consumer-side barrier. Even if a
        // future call-site ordering bug leaves a carrier behind, OFF cannot
        // substitute resident K/V or suppress any standalone crown producer.
        guard q4ResidentNormRopeEnabled else {
            return ResidentProducts(
                runsumTable: nil, normalizedKeys: nil, normalizedValues: nil)
        }
        return products
    }

    private static let batch = 8
    private static let queryHeads = 16
    private static let kvHeads = 8
    private static let gqa = 2
    private static let headDim = 256
    private static let sequenceLength = 1024

    /// HOST-ALLOC-TABLES kill switch: `DARKBLOOM_GEMMA4_HOST_ALLOC_TABLES` set
    /// to `0`/`false`/`no`/`off` builds the ring-start vector and the D512
    /// params vector fresh on every call (the shipped constructors below are
    /// the fallback branch either way). Default ON.
    static let hostAllocTablesEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_HOST_ALLOC_TABLES"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    // Precomputed uniform start vectors for the ranked cohort geometry.
    // All batch-8 streams advance in lockstep, so every decode step passes
    // starts == [s]*8 with 0 <= s < sequenceLength. One evaluated [8]
    // UInt32 array per s: immutable after init, lock-free, zero-alloc on
    // the hot path. Anything else falls back to a fresh allocation.
    // Keyed on ring offsets only, never on tokens.
    nonisolated(unsafe) private static let uniformStarts8: [MLXArray?] = {
        var table = [MLXArray?](repeating: nil, count: sequenceLength)
        var toEval: [MLXArray] = []
        toEval.reserveCapacity(sequenceLength)
        for s in 0..<sequenceLength {
            let arr = MLXArray(
                Array(repeating: UInt32(s), count: batch), [batch])
            table[s] = arr
            toEval.append(arr)
        }
        eval(toEval)
        return table
    }()

    private static func getStartArray(starts: [Int], batch batchParam: Int) -> MLXArray {
        if hostAllocTablesEnabled, batchParam == batch, starts.count == batch,
            let s0 = starts.first, starts.allSatisfy({ $0 == s0 }),
            s0 >= 0, s0 < uniformStarts8.count,
            let hit = uniformStarts8[s0]
        {
            return hit
        }
        return MLXArray(starts.map(UInt32.init), [batchParam])
    }

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

    /// Attribution: COMBINE-PACK-001 and COMBINE-HOIST-001 below are adapted
    /// from samfenwick's public Yukon submission
    /// `0ca873cb-d1b3-43e0-b75d-88d49c206812` (`3dcce32`). That sealed run
    /// passed parity and measured the fastest absolute decode window among the
    /// recent public candidates (2.085229 seconds).
    /// Off-cadence retest: the first stacked run also passed parity and kept
    /// 13.871 ms of absolute decode gain, but missed promotion after its paired
    /// serial prefill control shifted by 25.274 ms versus the record run.
    ///
    /// COMBINE-PACK-001: how many partition columns one simdgroup of the merge
    /// dispatch carries.
    ///
    /// The merge indexes a partition column with a lane, so with the partition
    /// PARTITION-002 settled on it runs eight live lanes and twenty-four dead
    /// ones in every simdgroup of every threadgroup. Packing `32 / COLS`
    /// output groups into the one simdgroup fills those lanes instead. It is
    /// the same reduction over the same columns in the same order, so the
    /// merge is unchanged arithmetically; only the lane a column lands on and
    /// the number of threads launched move.
    ///
    /// At a partition of a simdgroup or wider this is `32`, `sets` is one and
    /// the packing is the incumbent one thread per column, so the stock
    /// partitions and the `DARKBLOOM_CBV2_2PASS_BLOCKS` bisection route keep
    /// the shape they had.
    private static let combineColumns: Int = {
        let capped = min(blocks, 32)
        return capped > 0 && (capped & (capped - 1)) == 0 ? capped : 32
    }()

    /// Output groups per simdgroup. `D / values_per_lane` is always 32, so the
    /// merge needs `32 / combineSets` simdgroups to cover the head.
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

    /// GQA-PAIR-001: one simdgroup serves BOTH query heads of its GQA group.
    ///
    /// The incumbent lays a GQA group across two simdgroups of one threadgroup,
    /// one query head each. Both simdgroups walk the same block-strided slot
    /// sequence of the same `(row, kv head)` ring, so every K row and every V
    /// row of the retained window is issued TWICE, once per simdgroup, for the
    /// same 512 bytes. The second issue is a cache hit rather than a second
    /// DRAM read, but it is still a full set of load instructions and a second
    /// pass through the L1 return path, and the sliding layers stream 67 MB of
    /// ring per decode step across 25 of the 30 layers.
    ///
    /// Here the lane holds both heads' queries and both heads' accumulators,
    /// loads each K element and each V element ONCE, and feeds the two
    /// independent online-softmax chains from that one register. The two chains
    /// never mix: each keeps its own `max`, its own `sum`, its own accumulator
    /// bank, and its own `simd_sum`, and each expression is character for
    /// character the incumbent's with the shared load hoisted out. So every
    /// output element's add sequence is unchanged and the partition, traversal
    /// order and BF16 partial store are unchanged.
    ///
    /// Occupancy: the launch keeps its 512 threadgroups and drops from two
    /// simdgroups to one, so half as many K/V loads are outstanding. The
    /// incumbent's own sizing note puts roughly 1 MB in flight against the
    /// ~225 KB needed to cover DRAM latency on a 450 GB/s part, so halving it
    /// still clears that bar with margin.
    ///
    /// Kill switch: `DARKBLOOM_CBV2_DECODE_GQA_PAIRED_PASSA=0`.
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

    /// VEC4-PASSA-001: the paired pass A reads K and V four elements at a
    /// time instead of one.
    ///
    /// A lane owns `D / 32 == 8` contiguous elements of the head, so each
    /// retained slot costs it eight scalar 2-byte loads for K and eight more
    /// for V. Every one of those addresses is `lane * 8 + slot * D` plus a
    /// constant, so the whole run of eight is 16 contiguous bytes at a
    /// 16-byte-aligned address. Issuing it as two `vec<T, 4>` loads asks the
    /// same bytes of the same cache line in one quarter of the instructions.
    ///
    /// The pairing rider already showed that the ranked part is sensitive to
    /// load-issue count on this kernel even where the box this was written on
    /// is not: `ee77bc4` measured a flat local null and returned decode gain
    /// `2.23188325717` against the `2.20170392058` of the tree it replaced,
    /// +1.37% of decode. This is the same lever one step further down, and it
    /// touches nothing but addressing width.
    ///
    /// Kill switch: `DARKBLOOM_CBV2_DECODE_PAIRED_PASSA_VEC4=0` falls back to
    /// the scalar paired kernel, which stays in the file untouched.
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

    /// COMBINE-XFOLD-001 selector. ON by default;
    /// `DARKBLOOM_GEMMA4_PASSA_COMBFOLD=0` restores the incumbent merge
    /// kernel byte for byte, including its cached pipeline name.
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

    /// Shipped q4g64 pass-A with one live mirror-slot write. The logical new
    /// token is always consumed from the just-computed packed word, so no
    /// threadgroup races the in-place store. The returned fence orders the
    /// next decode dispatch after this write completes.
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

    /// B8-Q4-RESIDENT-001: the eight block SIMD groups that previously ran as
    /// separate pass-A threadgroups now share one 256-thread threadgroup. Each
    /// group executes the incumbent block-strided walk and casts both head
    /// accumulators through BF16 threadgroup storage. After a barrier, the same
    /// threads execute the incumbent COLS=8 pass-B mapping and xor trees. This
    /// removes only the global partial write/read and the second dispatch.
    private static let portQuantFusedWriteResidentKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_ragged8_sdpa_ringwrite_q4g64_d256_g2_regpack_vec4_carry_pair_b8_resident_colred_vload_c3_ey29_ey32_yp3_ey51_yrp1_ey82",
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

    /// ORSFOLD-001. Default ON. `DARKBLOOM_CBV2_O_RS_FOLD=0` selects the
    /// four-output resident kernel and leaves the standalone
    /// `cbv2_b8_rs_table_dyn_v1` prepass to build the o_proj table.
    static let oRunsumFoldEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_O_RS_FOLD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// ORSFOLD-001 epilogue. Each combine lane already holds, in `ov`, the one
    /// BF16 element it is about to store into `out`, and the o_proj activation
    /// is that store read back head-major. Lane `l` owns element
    /// `(block * 4 + l / 8) * 8 + (l % 8)`, so the eight lanes `l & ~7 ..
    /// (l & ~7) + 7` are exactly one aligned run of eight consecutive elements,
    /// which is the run one lane of the prepass reads as a `uint4`. Shuffling
    /// that octet into `xt` and writing the two four-element partials as the
    /// same two statements the prepass uses reproduces its value exactly,
    /// including the BF16 rounding of each partial. A 64-wide group is eight
    /// such octets, held by the four octets of block `2k` and the four of block
    /// `2k+1`; xor 8 and xor 16 walk the octet index inside a block and give
    /// `((v0+v1)+(v2+v3))`, the partner block gives `((v4+v5)+(v6+v7))`, and
    /// the final add is the prepass's xor-4 stage. Same tree, same order.
    private static let residentORunsumFold = """
                    const ushort ow = as_type<ushort>(ov);
                    const ushort obase = ushort(lane & ~7);
                    thread T xt[8];
                    #pragma clang loop unroll(full)
                    for (int p = 0; p < 8; ++p) {
                        xt[p] = as_type<T>(
                            simd_shuffle(ow, ushort(obase + ushort(p))));
                    }
                    float rsv = 0;
                    rsv += xt[0] + xt[1] + xt[2] + xt[3];
                    rsv += xt[4] + xt[5] + xt[6] + xt[7];
                    rsv += simd_shuffle_xor(rsv, 8u);
                    rsv += simd_shuffle_xor(rsv, 16u);
                    if (lane == 0) {
                        local_o_rs[head * BLOCKS + block] = rsv;
                    }
        """

    private static let residentORunsumStore = """
                threadgroup_barrier(mem_flags::mem_threadgroup);
                constexpr int rs_groups = D / 64;
                const int rs_flat = block * simd_width + lane;
                if (rs_flat < GQA * rs_groups) {
                    const int rs_head = rs_flat / rs_groups;
                    const int rs_group = rs_flat % rs_groups;
                    o_rs[batch_index * (KV_HEADS * GQA * rs_groups)
                            + (query_head + rs_head) * rs_groups + rs_group] =
                        local_o_rs[rs_head * BLOCKS + 2 * rs_group]
                        + local_o_rs[rs_head * BLOCKS + 2 * rs_group + 1];
                }
        """

    /// SLIDING-PREFETCH-DEPTH (SPD2). Default ON.
    ///
    /// The resident sliding walk keeps ONE iteration's packed K/V words
    /// outstanding: four `uint32` per lane, 288 distinct bytes per simdgroup
    /// (a 144-byte K row and a 144-byte V row). The dispatch below is
    /// `kvHeads * batch = 64` threadgroups of eight simdgroups, so 147,456
    /// bytes are in flight device-wide. This plane moves ~471.9 MB per decode
    /// step, the largest decode term after the MoE gather, and unlike every
    /// other plane this project has put a carry or a prefetch on it is not
    /// three to four times OVER-supplied.
    ///
    /// But read 147,456 B against the walk's OWN achieved bandwidth, not
    /// against the part's peak. The body issues roughly 200 machine
    /// instructions per 288 bytes, so at full issue it asks for well under
    /// half of 614.4 GB/s; comparing against the ~246-307 KB that SATURATING
    /// the part would need is the wrong denominator. Per core: one holding two
    /// of the 64 threadgroups runs sixteen simdgroups, covers ~570-910 ns per
    /// token step, and is already covered at depth one; one holding a single
    /// threadgroup runs eight, covers ~290-460 ns, and is short by about 1.6x.
    /// With 64 threadgroups on a 40-core part the split is 24 cores at two and
    /// 16 at one, so depth two certainly removes a stall on the sixteen light
    /// cores -- which finish early and are not the critical path -- and helps
    /// the heavy ones only if the loaded DRAM latency exceeds ~570 ns.
    ///
    /// This is therefore a bounded bet. What is measured offline: issued work
    /// per token step falls 5.5% (227 -> 214.5 optimized AIR instructions, the
    /// loop overhead now amortising over two steps), latency cover doubles,
    /// register use rises 11 per thread (68 -> 79 on applegpu_g18p, 68 -> 76
    /// on g17p) without crossing an occupancy tier, and kernel text grows 13%.
    ///
    /// Depth two is NOT a deeper copy chain. A rotating `kw_pre = kw_pre2`
    /// pair would read the second stage one iteration after its load issued
    /// and stall there, so the cover would stay at one iteration however many
    /// stages were added. The walk is instead expanded modulo `PF`: the token
    /// loop steps `PF * BLOCKS` and a fully unrolled inner loop gives each
    /// phase its own prefetch register, so a loaded word stays in the register
    /// it was loaded into until it is consumed `PF` token steps later.
    ///
    /// Bit-exact: the same slots are read, in the same order, exactly once
    /// each, and each is consumed by the same token it was consumed by before.
    /// Only the issue point moves earlier.
    ///
    /// `DARKBLOOM_CBV2_SLIDING_PREFETCH_DEPTH=1` restores the shipped
    /// depth-one walk and the shipped kernel names, byte for byte.
    static let slidingPrefetchDepth2: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_SLIDING_PREFETCH_DEPTH"], let value = Int(raw)
        else { return true }
        return value >= 2
    }()

    /// PF2-TAIL-PEEL. Default on only inside the promoted depth-two arm.
    /// Set `DARKBLOOM_CBV2_SLIDING_PREFETCH_PEEL=0` to restore the promoted
    /// modulo-two walk and its `_spd2` registration byte for byte.
    static let slidingPrefetchPeelEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_SLIDING_PREFETCH_PEEL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// RING-OFF-044. Carry the resident sliding ring cursor as a row-word
    /// offset instead of a slot index, so the peeled walk's inner phase drops
    /// its `slot * row_words` multiply and addresses the mirror with a pure
    /// add chain. Address-identical, so it changes no float operation.
    /// Set `DARKBLOOM_CBV2_SLIDING_RING_OFFSET=0` to restore the promoted
    /// peeled body and its `_spd2_lp1` registration byte for byte.
    static let ringOffsetEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_SLIDING_RING_OFFSET"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()
    /// SLIDE-PNIB (port of PR 2393 onto the RING-OFF tip). Paired-nibble
    /// unpack in the resident sliding walks. Set
    /// `DARKBLOOM_CBV2_SLIDING_PNIB=0` to restore the shipped scalar
    /// `float((w >> 4e) & 0xf)` chain and the shipped registration byte for
    /// byte on every walk arm.
    static let slidingNibblePairUnpack: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_SLIDING_PNIB"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Unpacks the eight 4-bit lanes of one packed word into `dest` with four
    /// `half2` extractions instead of eight scalar integer-to-float converts.
    ///
    /// `0x6400` is `1024.0h`, whose ULP is exactly one, so `0x6400 | n` is the
    /// half `1024 + n` for every `n` in `0...15` and the subtraction returns
    /// `n` with no rounding in either step. `float()` of that half is the same
    /// float the shipped `float((w >> 4e) & 0xfu)` produced, so the fma below
    /// consumes identical operands in the identical order.
    ///
    /// The pair is `(p, p + values_per_lane / 2)`, which is where the two
    /// nibbles that share a 32-bit half2 register already live in the packed
    /// word. No address is formed, no load moves, no branch changes and no
    /// accumulation is reassociated.
    private static func nibblePairUnpack(
        _ word: String, into dest: String, indent: String
    ) -> String {
        guard slidingNibblePairUnpack else { return "" }
        let pad = indent
        return """
            float \(dest)[values_per_lane];
            \(pad)#pragma clang loop unroll(full)
            \(pad)for (int p = 0; p < values_per_lane / 2; ++p) {
            \(pad)    const half2 nib = as_type<half2>(
            \(pad)        ((((\(word)) >> (4 * p)) & 0xfu)
            \(pad)            | ((((\(word))
            \(pad)                >> (4 * (p + values_per_lane / 2))) & 0xfu) << 16))
            \(pad)        | 0x64006400u) - half2(1024.0h, 1024.0h);
            \(pad)    \(dest)[p] = float(nib[0]);
            \(pad)    \(dest)[p + values_per_lane / 2] = float(nib[1]);
            \(pad)}
            \(pad)
            """
    }

    /// The dequantization operand for element `element` of `word`.
    private static func nibbleOperand(_ word: String, _ dest: String) -> String {
        slidingNibblePairUnpack
            ? "\(dest)[element]"
            : "float((\(word) >> (4 * element)) & 0xfu)"
    }

    /// MLX keys its custom-kernel library cache by kernel NAME and re-JITs a
    /// name whose generated source changed (`backend/metal/custom_kernel.cpp`
    /// `:56-70`, `device.cpp:770-796`), so a changed body takes a changed
    /// name. Empty on the depth-one arm, while peel-off keeps `_spd2`.
    private static let slidingPrefetchKey =
        (slidingPrefetchDepth2
            ? (slidingPrefetchPeelEnabled
                ? (ringOffsetEnabled ? "_spd2_lp1_ro1" : "_spd2_lp1")
                : "_spd2")
            : "")
        + (slidingNibblePairUnpack ? "_pn1" : "")

    /// The shipped depth-one ring walk, verbatim.
    private static let residentSlidingWalkDepth1 = """
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
                    \(nibblePairUnpack("kw", into: "key_el", indent: "                    "))float score_lo = 0.0f;
                    float score_hi = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float key_element =
                            fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                    \(nibblePairUnpack("vw", into: "value_el", indent: "                    "))#pragma clang loop unroll(full)
                    for (int element = 0; element < values_per_lane; ++element) {
                        const float value_element =
                            fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                        acc_lo[element] = acc_lo[element] * old_factor_lo
                            + score_factor_lo * value_element;
                        acc_hi[element] = acc_hi[element] * old_factor_hi
                            + score_factor_hi * value_element;
                    }
                }
            """

    /// SPD2: the same walk, expanded modulo two.
    private static let residentSlidingWalkDepth2 = """
            uint slot = (start + uint(block)) % uint(N);
                // SLIDING-PREFETCH-DEPTH. `PF` iterations of the walk are outstanding
                // instead of one. `kw_pre[u]` is a PHASE register, not a stage of a copy
                // chain: the inner loop is fully unrolled at compile time, so the word
                // loaded into `kw_pre[u]` on one outer step is consumed by that same
                // `kw_pre[u]` PF token steps later with no register move in between. A
                // rotating `A = B` pair would resolve B's load one step early and buy no
                // latency cover at all.
                //
                // `block` is `simdgroup_index_in_threadgroup` over a `BLOCKS * 32` thread
                // group, so `0 <= block < BLOCKS`; with `N % (PF * BLOCKS) == 0` the outer
                // loop runs `N / (PF * BLOCKS)` times and the inner phase never steps past
                // the ring, exactly as the one-stage walk did.
                //
                // A position `t` is walked, and is not the current token, precisely when
                // `t < N - 1`. That one predicate is the seed guard, the re-issue guard,
                // and the shipped `token + BLOCKS < N - 1` alike. The current token is
                // served from `kword`/`vword` and its slot is the one this kernel stores
                // into, so no address formed here is a slot the one-stage walk did not
                // also read, in the same order, exactly once.
                constexpr int PF = 2;
                static_assert(N % (PF * BLOCKS) == 0,
                    "depth-two walk needs an even number of token steps");
                uint pf_slot = slot;
                thread uint32_t kw_pre[PF];
                thread uint32_t vw_pre[PF];
                thread uint32_t ktw_pre[PF];
                thread uint32_t vtw_pre[PF];
                #pragma clang loop unroll(full)
                for (int u = 0; u < PF; ++u) {
                    const bool prefetch_first = block + u * BLOCKS < N - 1;
                    kw_pre[u] = prefetch_first
                        ? mkeys_w[pf_slot * row_words + lane] : 0u;
                    vw_pre[u] = prefetch_first
                        ? mvalues_w[pf_slot * row_words + lane] : 0u;
                    ktw_pre[u] = prefetch_first
                        ? mkeys_w[pf_slot * row_words + payload_words + lane / 8] : 0u;
                    vtw_pre[u] = prefetch_first
                        ? mvalues_w[pf_slot * row_words + payload_words + lane / 8] : 0u;
                    pf_slot += uint(BLOCKS);
                    if (pf_slot >= uint(N)) pf_slot -= uint(N);
                }
                uint next_slot = pf_slot;
                for (int token = block; token < N; token += PF * BLOCKS) {
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < PF; ++u) {
                        const int tok = token + u * BLOCKS;
                        const bool current = tok == N - 1;
                        const uint32_t kw = current ? kword : kw_pre[u];
                        const uint32_t vw = current ? vword : vw_pre[u];
                        const uint32_t ktw = current
                            ? (uint32_t(as_type<ushort>(khs))
                                | (uint32_t(as_type<ushort>(khb)) << 16))
                            : ktw_pre[u];
                        const uint32_t vtw = current
                            ? (uint32_t(as_type<ushort>(vhs))
                                | (uint32_t(as_type<ushort>(vhb)) << 16))
                            : vtw_pre[u];
                        if (tok + PF * BLOCKS < N - 1) {
                            kw_pre[u] = mkeys_w[next_slot * row_words + lane];
                            vw_pre[u] = mvalues_w[next_slot * row_words + lane];
                            ktw_pre[u] =
                                mkeys_w[next_slot * row_words + payload_words + lane / 8];
                            vtw_pre[u] =
                                mvalues_w[next_slot * row_words + payload_words + lane / 8];
                            next_slot += uint(BLOCKS);
                            if (next_slot >= uint(N)) next_slot -= uint(N);
                        }
                        const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                        const float kb = float(as_type<half>(ushort(ktw >> 16)));
                        const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                        const float vb = float(as_type<half>(ushort(vtw >> 16)));
                        \(nibblePairUnpack("kw", into: "key_el", indent: "                        "))float score_lo = 0.0f;
                        float score_hi = 0.0f;
                        #pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float key_element =
                                fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                        \(nibblePairUnpack("vw", into: "value_el", indent: "                        "))#pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float value_element =
                                fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }
                }
            """

    /// PF2-TAIL-PEEL: preserve the promoted two-phase latency distance while
    /// removing the current-token selects and guarded reissue from 62 of 64
    /// outer trips. The final two trips retain the promoted boundary logic.
    private static let residentSlidingWalkDepth2Peeled = """
            uint slot = (start + uint(block)) % uint(N);
                constexpr int PF = 2;
                constexpr int OUTER = N / (PF * BLOCKS);
                static_assert(N % (PF * BLOCKS) == 0,
                    "peeled depth-two walk needs integral phase groups");
                static_assert(OUTER >= 2,
                    "peeled depth-two walk needs two boundary trips");
                static_assert(PF * BLOCKS <= N - 1,
                    "both depth-two seed positions must be historical");

                uint pf_slot = slot;
                thread uint32_t kw_pre[PF];
                thread uint32_t vw_pre[PF];
                thread uint32_t ktw_pre[PF];
                thread uint32_t vtw_pre[PF];
                #pragma clang loop unroll(full)
                for (int u = 0; u < PF; ++u) {
                    kw_pre[u] = mkeys_w[pf_slot * row_words + lane];
                    vw_pre[u] = mvalues_w[pf_slot * row_words + lane];
                    ktw_pre[u] =
                        mkeys_w[pf_slot * row_words + payload_words + lane / 8];
                    vtw_pre[u] =
                        mvalues_w[pf_slot * row_words + payload_words + lane / 8];
                    pf_slot += uint(BLOCKS);
                    if (pf_slot >= uint(N)) pf_slot -= uint(N);
                }
                uint next_slot = pf_slot;
                int token = block;

                // The first OUTER-2 trips consume only historical rows and
                // every same-phase successor is also historical. Loads and
                // next_slot updates therefore follow the promoted order with
                // no predicate in the hot loop.
                for (; token < N - 2 * PF * BLOCKS; token += PF * BLOCKS) {
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < PF; ++u) {
                        const uint32_t kw = kw_pre[u];
                        const uint32_t vw = vw_pre[u];
                        const uint32_t ktw = ktw_pre[u];
                        const uint32_t vtw = vtw_pre[u];
                        kw_pre[u] = mkeys_w[next_slot * row_words + lane];
                        vw_pre[u] = mvalues_w[next_slot * row_words + lane];
                        ktw_pre[u] =
                            mkeys_w[next_slot * row_words + payload_words + lane / 8];
                        vtw_pre[u] =
                            mvalues_w[next_slot * row_words + payload_words + lane / 8];
                        next_slot += uint(BLOCKS);
                        if (next_slot >= uint(N)) next_slot -= uint(N);
                        const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                        const float kb = float(as_type<half>(ushort(ktw >> 16)));
                        const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                        const float vb = float(as_type<half>(ushort(vtw >> 16)));
                        \(nibblePairUnpack("kw", into: "key_el", indent: "                        "))float score_lo = 0.0f;
                        float score_hi = 0.0f;
                        #pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float key_element =
                                fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                        \(nibblePairUnpack("vw", into: "value_el", indent: "                        "))#pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float value_element =
                                fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }
                }

                // Penultimate phase 1 for block 7 must not load the live write
                // slot, and final phase 1 for block 7 consumes the new token.
                // Retain the promoted predicates for exactly these two trips.
                #pragma clang loop unroll(disable)
                for (; token < N; token += PF * BLOCKS) {
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < PF; ++u) {
                        const int tok = token + u * BLOCKS;
                        const bool current = tok == N - 1;
                        const uint32_t kw = current ? kword : kw_pre[u];
                        const uint32_t vw = current ? vword : vw_pre[u];
                        const uint32_t ktw = current
                            ? (uint32_t(as_type<ushort>(khs))
                                | (uint32_t(as_type<ushort>(khb)) << 16))
                            : ktw_pre[u];
                        const uint32_t vtw = current
                            ? (uint32_t(as_type<ushort>(vhs))
                                | (uint32_t(as_type<ushort>(vhb)) << 16))
                            : vtw_pre[u];
                        if (tok + PF * BLOCKS < N - 1) {
                            kw_pre[u] = mkeys_w[next_slot * row_words + lane];
                            vw_pre[u] = mvalues_w[next_slot * row_words + lane];
                            ktw_pre[u] =
                                mkeys_w[next_slot * row_words + payload_words + lane / 8];
                            vtw_pre[u] =
                                mvalues_w[next_slot * row_words + payload_words + lane / 8];
                            next_slot += uint(BLOCKS);
                            if (next_slot >= uint(N)) next_slot -= uint(N);
                        }
                        const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                        const float kb = float(as_type<half>(ushort(ktw >> 16)));
                        const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                        const float vb = float(as_type<half>(ushort(vtw >> 16)));
                        \(nibblePairUnpack("kw", into: "key_el", indent: "                        "))float score_lo = 0.0f;
                        float score_hi = 0.0f;
                        #pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float key_element =
                                fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                        \(nibblePairUnpack("vw", into: "value_el", indent: "                        "))#pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float value_element =
                                fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }
                }
            """


    /// RING-OFF-044. The peeled walk with its ring cursor folded into a
    /// row-word offset. Byte-for-byte the promoted body apart from the
    /// cursor: the same slots are read, in the same order, at the same
    /// addresses, and every float operation is untouched.
    private static let residentSlidingWalkDepth2PeeledOffset = """
            uint slot = (start + uint(block)) % uint(N);
                constexpr int PF = 2;
                constexpr int OUTER = N / (PF * BLOCKS);
                static_assert(N % (PF * BLOCKS) == 0,
                    "peeled depth-two walk needs integral phase groups");
                static_assert(OUTER >= 2,
                    "peeled depth-two walk needs two boundary trips");
                static_assert(PF * BLOCKS <= N - 1,
                    "both depth-two seed positions must be historical");

                // RING-OFF: carry the ring cursor as a row-word offset
                // rather than a slot index. `slot * row_words` is monotone
                // on [0, N) and every advance is the same `BLOCKS` step, so
                // the wrap compare is exact at `N * row_words` and every
                // address below is the address the promoted walk computed.
                constexpr uint ROW_W = uint(row_words);
                constexpr uint RING_W = uint(N) * ROW_W;
                constexpr uint RING_STEP = uint(BLOCKS) * ROW_W;
                static_assert(RING_W / ROW_W == uint(N),
                    "ring word span must not overflow a 32-bit cursor");
                const uint pay_lane = uint(lane);
                const uint meta_lane = uint(payload_words) + uint(lane) / 8u;
                uint pf_off = slot * ROW_W;
                thread uint32_t kw_pre[PF];
                thread uint32_t vw_pre[PF];
                thread uint32_t ktw_pre[PF];
                thread uint32_t vtw_pre[PF];
                #pragma clang loop unroll(full)
                for (int u = 0; u < PF; ++u) {
                    kw_pre[u] = mkeys_w[pf_off + pay_lane];
                    vw_pre[u] = mvalues_w[pf_off + pay_lane];
                    ktw_pre[u] =
                        mkeys_w[pf_off + meta_lane];
                    vtw_pre[u] =
                        mvalues_w[pf_off + meta_lane];
                    pf_off += RING_STEP;
                    if (pf_off >= RING_W) pf_off -= RING_W;
                }
                uint next_off = pf_off;
                int token = block;

                // The first OUTER-2 trips consume only historical rows and
                // every same-phase successor is also historical. Loads and
                // next_off updates therefore follow the promoted order with
                // no predicate in the hot loop.
                for (; token < N - 2 * PF * BLOCKS; token += PF * BLOCKS) {
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < PF; ++u) {
                        const uint32_t kw = kw_pre[u];
                        const uint32_t vw = vw_pre[u];
                        const uint32_t ktw = ktw_pre[u];
                        const uint32_t vtw = vtw_pre[u];
                        kw_pre[u] = mkeys_w[next_off + pay_lane];
                        vw_pre[u] = mvalues_w[next_off + pay_lane];
                        ktw_pre[u] =
                            mkeys_w[next_off + meta_lane];
                        vtw_pre[u] =
                            mvalues_w[next_off + meta_lane];
                        next_off += RING_STEP;
                        if (next_off >= RING_W) next_off -= RING_W;
                        const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                        const float kb = float(as_type<half>(ushort(ktw >> 16)));
                        const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                        const float vb = float(as_type<half>(ushort(vtw >> 16)));
                        \(nibblePairUnpack("kw", into: "key_el", indent: "                        "))float score_lo = 0.0f;
                        float score_hi = 0.0f;
                        #pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float key_element =
                                fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                        \(nibblePairUnpack("vw", into: "value_el", indent: "                        "))#pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float value_element =
                                fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }
                }

                // Penultimate phase 1 for block 7 must not load the live write
                // slot, and final phase 1 for block 7 consumes the new token.
                // Retain the promoted predicates for exactly these two trips.
                #pragma clang loop unroll(disable)
                for (; token < N; token += PF * BLOCKS) {
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < PF; ++u) {
                        const int tok = token + u * BLOCKS;
                        const bool current = tok == N - 1;
                        const uint32_t kw = current ? kword : kw_pre[u];
                        const uint32_t vw = current ? vword : vw_pre[u];
                        const uint32_t ktw = current
                            ? (uint32_t(as_type<ushort>(khs))
                                | (uint32_t(as_type<ushort>(khb)) << 16))
                            : ktw_pre[u];
                        const uint32_t vtw = current
                            ? (uint32_t(as_type<ushort>(vhs))
                                | (uint32_t(as_type<ushort>(vhb)) << 16))
                            : vtw_pre[u];
                        if (tok + PF * BLOCKS < N - 1) {
                            kw_pre[u] = mkeys_w[next_off + pay_lane];
                            vw_pre[u] = mvalues_w[next_off + pay_lane];
                            ktw_pre[u] =
                                mkeys_w[next_off + meta_lane];
                            vtw_pre[u] =
                                mvalues_w[next_off + meta_lane];
                            next_off += RING_STEP;
                            if (next_off >= RING_W) next_off -= RING_W;
                        }
                        const float ks = float(as_type<half>(ushort(ktw & 0xffffu)));
                        const float kb = float(as_type<half>(ushort(ktw >> 16)));
                        const float vs = float(as_type<half>(ushort(vtw & 0xffffu)));
                        const float vb = float(as_type<half>(ushort(vtw >> 16)));
                        \(nibblePairUnpack("kw", into: "key_el", indent: "                        "))float score_lo = 0.0f;
                        float score_hi = 0.0f;
                        #pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float key_element =
                                fma(\(nibbleOperand("kw", "key_el")), ks, kb);
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
                        \(nibblePairUnpack("vw", into: "value_el", indent: "                        "))#pragma clang loop unroll(full)
                        for (int element = 0; element < values_per_lane; ++element) {
                            const float value_element =
                                fma(\(nibbleOperand("vw", "value_el")), vs, vb);
                            acc_lo[element] = acc_lo[element] * old_factor_lo
                                + score_factor_lo * value_element;
                            acc_hi[element] = acc_hi[element] * old_factor_hi
                                + score_factor_hi * value_element;
                        }
                    }
                }
            """

    private static var residentSlidingWalk: String {
        if !slidingPrefetchDepth2 { return residentSlidingWalkDepth1 }
        guard slidingPrefetchPeelEnabled else { return residentSlidingWalkDepth2 }
        return ringOffsetEnabled
            ? residentSlidingWalkDepth2PeeledOffset : residentSlidingWalkDepth2Peeled
    }

    /// F4: exact decode Q/K/V normalization and sliding RoPE in the resident
    /// attention prologue. SIMD groups 0...3 own Q0, Q1, K, V respectively, so
    /// every raw element is normalized once; all eight resident attention groups
    /// reuse the BF16-rounded threadgroup rows.
    private static func residentNormRopeSource(withORunsum: Bool) -> String {
        """
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
                threadgroup T local_queries[GQA * D];
                threadgroup T local_key[D];
                threadgroup T local_value[D];
                \(withORunsum ? "threadgroup float local_o_rs[GQA * BLOCKS];" : "")

                // Transcribe gemma4_b8_qkv_rms_norm_rope_v2_vec1 one row per
                // designated SIMD group. Each lane owns the same two four-value
                // runs as the standalone kernel's two 32-lane SIMD groups.
                if (block < GQA + 2) {
                    const device T* raw_row = raw_queries;
                    const device T* weight_row = q_weight;
                    threadgroup T* normalized_row = local_queries;
                    bool weighted = true;
                    if (block < GQA) {
                        raw_row += (batch_head + block) * D;
                        normalized_row += block * D;
                    } else if (block == GQA) {
                        raw_row = raw_keys
                            + (batch_index * KV_HEADS + kv_head) * D;
                        weight_row = k_weight;
                        normalized_row = local_key;
                    } else {
                        raw_row = raw_values
                            + (batch_index * KV_HEADS + kv_head) * D;
                        normalized_row = local_value;
                        weighted = false;
                    }

                    const device T4* raw_vectors =
                        reinterpret_cast<const device T4*>(raw_row);
                    const device T4* weight_vectors =
                        reinterpret_cast<const device T4*>(weight_row);
                    const T4 vin_first = raw_vectors[lane];
                    const T4 vin_second = raw_vectors[lane + simd_width];
                    float sum_first = 0.0f;
                    float sum_second = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < 4; ++i) {
                        const float first = float(vin_first[i]);
                        const float second = float(vin_second[i]);
                        sum_first += first * first;
                        sum_second += second * second;
                    }
                    sum_first = simd_sum(sum_first);
                    sum_second = simd_sum(sum_second);

                    // Same second 32-lane tree as partials[0], partials[1],
                    // partials[2...31] = 0 in the standalone kernel.
                    float sum = lane == 0 ? sum_first
                        : (lane == 1 ? sum_second : 0.0f);
                    sum = simd_sum(sum);
                    float inverse_rms = 0.0f;
                    if (lane == 0) {
                        inverse_rms = metal::precise::rsqrt(
                            sum / float(D) + 1.0e-6f);
                    }
                    inverse_rms = simd_shuffle(inverse_rms, ushort(0));

                    const T4 weight_first = weight_vectors[lane];
                    const T4 weight_second = weight_vectors[lane + simd_width];
                    threadgroup T4* normalized_vectors =
                        reinterpret_cast<threadgroup T4*>(normalized_row);
                    T4 rounded_first;
                    T4 rounded_second;
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < 4; ++i) {
                        const T normalized_first =
                            T(float(vin_first[i]) * inverse_rms);
                        const T normalized_second =
                            T(float(vin_second[i]) * inverse_rms);
                        rounded_first[i] = weighted
                            ? T(weight_first[i] * normalized_first)
                            : T(1) * normalized_first;
                        rounded_second[i] = weighted
                            ? T(weight_second[i] * normalized_second)
                            : T(1) * normalized_second;
                    }
                    normalized_vectors[lane] = rounded_first;
                    normalized_vectors[lane + simd_width] = rounded_second;

                    if (block == GQA + 1) {
                        device T4* value_vectors =
                            reinterpret_cast<device T4*>(
                                v_out + (batch_index * KV_HEADS + kv_head) * D);
                        value_vectors[lane] = rounded_first;
                        value_vectors[lane + simd_width] = rounded_second;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Q and K preserve the standalone norm kernel's BF16 store
                // boundary before applying the base-route sliding RoPE.
                if (block < GQA + 1) {
                    threadgroup T* normalized_row = block < GQA
                        ? local_queries + block * D : local_key;
                    device T* key_output =
                        k_out + (batch_index * KV_HEADS + kv_head) * D;
                    const float L = static_cast<float>(
                        position_offsets[batch_index]);
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < 4; ++i) {
                        const int pair = lane * 4 + i;
                        const float d = static_cast<float>(pair)
                            / static_cast<float>(D / 2);
                        const float inv_freq =
                            metal::exp2(-d * rope_log2_base[0]);
                        const float theta = L * inv_freq;
                        const float costheta = metal::fast::cos(theta);
                        const float sintheta = metal::fast::sin(theta);
                        const float x1 =
                            static_cast<float>(normalized_row[pair]);
                        const float x2 =
                            static_cast<float>(normalized_row[pair + D / 2]);
                        const T rx1 = static_cast<T>(
                            x1 * costheta - x2 * sintheta);
                        const T rx2 = static_cast<T>(
                            x1 * sintheta + x2 * costheta);
                        normalized_row[pair] = rx1;
                        normalized_row[pair + D / 2] = rx2;
                        if (block == GQA) {
                            key_output[pair] = rx1;
                            key_output[pair + D / 2] = rx2;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

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
                const threadgroup T* new_key =
                    local_key + lane * values_per_lane;
                const threadgroup T* new_value =
                    local_value + lane * values_per_lane;
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
                    const threadgroup T4* kvec =
                        reinterpret_cast<const threadgroup T4*>(new_key);
                    const threadgroup T4* vvec =
                        reinterpret_cast<const threadgroup T4*>(new_value);
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

                const threadgroup T* query =
                    local_queries + lane * values_per_lane;
                threadgroup T* partial = local_partials
                    + block * D + lane * values_per_lane;
                threadgroup float* sum_out = local_sums + block;
                threadgroup float* max_out = local_maxs + block;

                thread float q_lo[values_per_lane];
                thread float q_hi[values_per_lane];
                thread float acc_lo[values_per_lane];
                thread float acc_hi[values_per_lane];
                const threadgroup T4* qvec =
                    reinterpret_cast<const threadgroup T4*>(query);
                const threadgroup T4* qvec_hi =
                    reinterpret_cast<const threadgroup T4*>(query + D);
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
                \(residentSlidingWalk)

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
                    const T ov = T(
                        sum_exp_score == 0.0f
                            ? accumulator[0]
                            : accumulator[0] / sum_exp_score);
                    head_out[block_lane] = ov;
                \(withORunsum ? residentORunsumFold : "")
                }
                \(withORunsum ? residentORunsumStore : "")
            """
    }

    private static let portQuantFusedWriteResidentNormRopeKernel: MLXFast.MLXFastKernel =
        MLXFast.metalKernel(
            name: "cbv2_ragged8_sdpa_ringwrite_q4g64_d256_g2_regpack_vec4_carry_pair_b8_resident_colred_vload_c3_f4_normrope_v1_ey29_ey32_yp3_ey51_yrp1\(slidingPrefetchKey)",
            inputNames: [
                "raw_queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "raw_keys", "raw_values", "q_weight", "k_weight",
                "position_offsets", "rope_log2_base", "write_fence",
            ],
            outputNames: ["out", "fence", "k_out", "v_out"],
            source: residentNormRopeSource(withORunsum: false),
            ensureRowContiguous: true
        )

    /// ORSFOLD-001 arm: the same kernel with the o_proj run-sum table as a
    /// fifth output, so the standalone prepass never runs on a sliding layer.
    private static let portQuantFusedWriteResidentNormRopeORunsumKernel:
        MLXFast.MLXFastKernel = MLXFast.metalKernel(
            name: "cbv2_ragged8_sdpa_ringwrite_q4g64_d256_g2_regpack_vec4_carry_pair_b8_resident_colred_vload_c3_f4_normrope_ors_v1_ey29_ey32_yp3_ey51_yrp1\(slidingPrefetchKey)",
            inputNames: [
                "raw_queries",
                "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
                "starts", "raw_keys", "raw_values", "q_weight", "k_weight",
                "position_offsets", "rope_log2_base", "write_fence",
            ],
            outputNames: ["out", "fence", "k_out", "v_out", "o_rs"],
            source: residentNormRopeSource(withORunsum: true),
            ensureRowContiguous: true
        )

    /// KVQ-PORT: `attendRing` reading the packed 8-bit mirror instead of the
    /// bf16 ring, with the result CONSUMED by pass B exactly as the stock
    /// road consumes it. This is the separate-write road only: the promoted
    /// stock mechanism still owns the ring write, so no fused kernel, no
    /// bf16 companion and no participant-authored fence are involved.
    ///
    /// Three diagnostic submissions established what this is allowed to
    /// assume. `47dae0ea` (1.9637) proved the read kernel dispatches and
    /// completes on the ranked box; `e14279c9` (2.0147) proved the fused
    /// writer and its companion do too; `3f7e5b55` (2.0122) proved writing
    /// in place into a live, graph-referenced mirror is fine there. All
    /// three discarded their output. What none of them exercised is pass B
    /// consuming a quantized pass A, which is what this does.
    /// KVQ4 kernel self-test: dispatch the real read kernel over a mirror
    /// whose every slot holds a known ramp, with a one-hot query. Pass A's
    /// `sums` then equals exp(0)=1 per block only if the dequantized dot
    /// products are what the host model predicts, so a mismatch localizes
    /// the fault to the kernel rather than the layout.
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

        let startArray = getStartArray(starts: starts, batch: batch)
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

    /// Exact B8/D256 q4g64 ring attention which packs this step's new K/V
    /// into the live mirror from pass A. Pass B consumes the first three
    /// outputs, while the fourth output is the next step's write fence.
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
        guard CBv2WindowedSequenceKV.q4FusedMirrorWriteEnabled,
            CBv2WindowedSequenceKV.quantEnabled,
            !CBv2WindowedSequenceKV.quantSimulate,
            !CBv2WindowedSequenceKV.gpuPackCheck,
            slidingWindowLength == sequenceLength,
            starts.count == batch,
            starts.allSatisfy({ 0 <= $0 && $0 < sequenceLength }),
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

        let startArray = getStartArray(starts: starts, batch: batch)
        let inputs = [queries] + mirrors
            + [startArray, newKeys, newValues, previousWriteFence]
        if q4ResidentMergeEnabled,
            blocks == 8,
            combineColumns == 8,
            combineThreads == 256
        {
            if let normRope = takeResidentNormRope(
                queries: queries, keys: newKeys, values: newValues)
            {
                let residentInputs =
                    [normRope.rawQueries] + mirrors + [
                        startArray,
                        normRope.rawKeys,
                        normRope.rawValues,
                        normRope.qWeight,
                        normRope.kWeight,
                        normRope.positionOffsets,
                        normRope.ropeLog2Base,
                        previousWriteFence,
                    ]
                let residentTemplate: [(String, any KernelTemplateArg)] = [
                    ("T", normRope.rawQueries.dtype),
                    ("D", headDim),
                    ("N", sequenceLength),
                    ("GQA", gqa),
                    ("KV_HEADS", kvHeads),
                    ("BLOCKS", blocks),
                ]
                let residentShapes = [
                    [batch, queryHeads, 1, headDim], [1],
                    [batch, kvHeads, 1, headDim],
                    [batch, kvHeads, 1, headDim],
                ]
                let residentDTypes: [DType] = [
                    .bfloat16, .int32, .bfloat16, .bfloat16,
                ]
                let resident: [MLXArray]
                let oRunsum: MLXArray?
                if oRunsumFoldEnabled {
                    resident = portQuantFusedWriteResidentNormRopeORunsumKernel(
                        residentInputs,
                        template: residentTemplate,
                        grid: (kvHeads * blocks * 32, batch, 1),
                        threadGroup: (blocks * 32, 1, 1),
                        outputShapes: residentShapes
                            + [[batch, queryHeads * headDim / 64]],
                        outputDTypes: residentDTypes + [.float32]
                    )
                    oRunsum = resident[4]
                    CBv2EngageMark.once("o-runsum-resident-fold")
                } else {
                    resident = portQuantFusedWriteResidentNormRopeKernel(
                        residentInputs,
                        template: residentTemplate,
                        grid: (kvHeads * blocks * 32, batch, 1),
                        threadGroup: (blocks * 32, 1, 1),
                        outputShapes: residentShapes,
                        outputDTypes: residentDTypes
                    )
                    oRunsum = nil
                }
                publishResidentProducts(
                    ResidentProducts(
                        runsumTable: oRunsum,
                        normalizedKeys: resident[2],
                        normalizedValues: resident[3]),
                    for: resident[0])
                CBv2EngageMark.once("kvq4-fused-live-write")
                CBv2EngageMark.once("kvq4-resident-merge")
                CBv2EngageMark.once("kvq4-resident-norm-rope")
                if slidingPrefetchDepth2 && slidingPrefetchPeelEnabled {
                    CBv2EngageMark.once("sliding-prefetch-pf2-tail-peel")
                }
                return (resident[0], resident[1])
            }
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
        let startArray = getStartArray(starts: starts, batch: batch)
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

        let startArray = getStartArray(starts: starts, batch: batch)
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
public enum CBv2RaggedComposedD512DecodeAttentionV1 {
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

    // Single-entry params cache for the shared [kL, D, per-row caps...]
    // layout. All full-attention rows share one offset per step, so the
    // layers of a step hit one entry; guarded by a lock, hit logic
    // otherwise identical to a fresh allocation.
    private struct CachedD512Params {
        var params: [UInt32] = []
        var array: MLXArray?
    }
    nonisolated(unsafe) private static var cachedD512Params = CachedD512Params()
    private static let d512ParamsLock = NSLock()

    private static func getD512ParamsArray(params: [UInt32]) -> MLXArray {
        guard CBv2RaggedTwoPassDecodeAttentionV1.hostAllocTablesEnabled else {
            return MLXArray(params)
        }
        d512ParamsLock.lock()
        if cachedD512Params.params == params, let arr = cachedD512Params.array {
            d512ParamsLock.unlock()
            return arr
        }
        d512ParamsLock.unlock()
        let arr = MLXArray(params)
        d512ParamsLock.lock()
        cachedD512Params = CachedD512Params(params: params, array: arr)
        d512ParamsLock.unlock()
        return arr
    }

    // MARK: NORMROPE-D512 / ORS-D512

    /// NORMROPE-D512 kill switch: `DARKBLOOM_GEMMA4_D512_NORMROPE` set to
    /// `0`/`false`/`no`/`off` leaves the standalone
    /// `gemma4_b8_qkv_rms_norm_rope_v2_vec1` dispatch as the attention input
    /// and selects the untouched WRITE-022 store kernel. Default ON.
    static let normRopeFoldEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_NORMROPE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// ORS-D512 kill switch: `DARKBLOOM_GEMMA4_D512_ORS` set to
    /// `0`/`false`/`no`/`off` selects the untouched dispatch-3 kernel and
    /// leaves the standalone `cbv2_b8_rs_table_dyn_v1` prepass to build the
    /// o_proj run-sum table. Default ON. The fold rides the CUT-13 `_sv1`
    /// dispatch-3 twin only, so `DARKBLOOM_GEMMA4_D512_SOFTMAX_VEC=0` also
    /// restores the prepass.
    static let oRunsumFoldEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_ORS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// NORMROPE-D512 carrier. The three normalized arrays are held weakly:
    /// the entry is only honoured while the exact objects the producer built
    /// are still alive, so an address reused by a later array can never match
    /// a stale record.
    private struct FullNormRopeInputs {
        weak var normalizedQueries: MLXArray?
        weak var normalizedKeys: MLXArray?
        weak var normalizedValues: MLXArray?
        let rawQueries: MLXArray
        let rawKeys: MLXArray
        let qWeight: MLXArray
        let kWeight: MLXArray
        let positionOffsets: MLXArray
        let ropeFrequencies: MLXArray
    }

    private static let fullNormRopeLock = NSLock()
    nonisolated(unsafe) private static var fullNormRopeInputs:
        [ObjectIdentifier: FullNormRopeInputs] = [:]

    /// NORMROPE-D512: register the raw full-attention decode projections
    /// behind the already-built exact norm+RoPE arrays. The fallback arrays
    /// remain what the generic cache sees; only the fused store dispatch
    /// consumes this carrier, so any miss evaluates the established
    /// `gemma4_b8_qkv_rms_norm_rope_v2_vec1` graph unchanged. `rawValues`
    /// must be the raw key projection itself (Gemma's k-eq-v full layers), so
    /// V is `RMSNormNoScale` of the same row the K RMSNorm reduces.
    @discardableResult
    public static func registerFullNormRope(
        normalizedQueries: MLXArray,
        normalizedKeys: MLXArray,
        normalizedValues: MLXArray,
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray,
        qWeight: MLXArray,
        kWeight: MLXArray,
        positionOffsets: MLXArray,
        ropeFrequencies: MLXArray,
        eps: Float,
        appliedRope: Bool
    ) -> Bool {
        guard normRopeFoldEnabled,
            enabled,
            storeDispatchEnabled,
            appliedRope,
            eps == 1.0e-6,
            rawValues === rawKeys,
            normalizedQueries.dtype == .bfloat16,
            normalizedQueries.shape == [batch, queryHeads, 1, headDim],
            normalizedKeys.dtype == .bfloat16,
            normalizedKeys.shape == [batch, kvHeads, 1, headDim],
            normalizedValues.dtype == .bfloat16,
            normalizedValues.shape == normalizedKeys.shape,
            rawQueries.dtype == .bfloat16,
            rawQueries.shape == [batch, 1, queryHeads, headDim],
            rawKeys.dtype == .bfloat16,
            rawKeys.shape == [batch, 1, kvHeads, headDim],
            qWeight.dtype == .bfloat16,
            qWeight.shape == [headDim],
            kWeight.dtype == .bfloat16,
            kWeight.shape == [headDim],
            positionOffsets.dtype == .int32,
            positionOffsets.shape == [batch],
            ropeFrequencies.dtype == .float32,
            ropeFrequencies.shape == [headDim / 2]
        else { return false }

        fullNormRopeLock.lock()
        if fullNormRopeInputs.count >= 64 {
            fullNormRopeInputs.removeAll(keepingCapacity: true)
        }
        fullNormRopeInputs[ObjectIdentifier(normalizedQueries)] =
            FullNormRopeInputs(
                normalizedQueries: normalizedQueries,
                normalizedKeys: normalizedKeys,
                normalizedValues: normalizedValues,
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                qWeight: qWeight,
                kWeight: kWeight,
                positionOffsets: positionOffsets,
                ropeFrequencies: ropeFrequencies)
        fullNormRopeLock.unlock()
        return true
    }

    @inline(__always)
    private static func takeFullNormRope(
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) -> FullNormRopeInputs? {
        fullNormRopeLock.lock()
        let inputs = fullNormRopeInputs.removeValue(
            forKey: ObjectIdentifier(queries))
        fullNormRopeLock.unlock()
        guard let inputs,
            inputs.normalizedQueries === queries,
            inputs.normalizedKeys === keys,
            inputs.normalizedValues === values
        else { return nil }
        return inputs
    }

    /// LASTQ-D512 kill switch: `DARKBLOOM_GEMMA4_LASTQ_D512` set to
    /// `0`/`false`/`no`/`off` restores the per-row last-query attend loop in
    /// `CBv2AttentionV1.updateAndAttendLastQuery`. Default ON.
    private static let lastQueryPrefillEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_LASTQ_D512"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

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

    /// CUT-13: alignment-gated vec4 twins of dispatch 2 and dispatch 3.
    ///
    /// The probability-row stride is `key_length`, which is not always a
    /// multiple of four, so the softmax load/store loops and the AV
    /// `p_coeff` loops above issue four scalar transactions per row chunk.
    /// Each `_sv1` twin computes one threadgroup-uniform branch
    /// `(axis_size & 3) == 0` (softmax) / `(key_length & 3) == 0` (AV) at
    /// the top of the kernel. When the row is 4-aligned:
    ///
    /// - softmax: `gid * axis_size` and `lid * 4` are multiples of four
    ///   elements, so each guarded chunk (`lid * 4 + 4 <= axis_size`) starts
    ///   on an 8-byte boundary for bf16 and covers exactly the four elements
    ///   the scalar run touches. Threads outside that guard — the whole
    ///   masked tail threadgroup included — keep the scalar path, so no
    ///   transaction is issued past the row end.
    /// - AV: `prob_rows` is offset by `(row * 16 + kv_head * GQA) *
    ///   key_length`, each head row by `h * key_length`, and `bm = 32 * i +
    ///   4 * thrM`; all three terms are multiples of four elements when
    ///   `key_length % 4 == 0`, and `bm + 4 <= 32 * n_iter <= key_length`
    ///   keeps every vec4 inside the row. The leftover tail stays scalar.
    ///
    /// Exactness: identical per-element `static_cast<float>` conversions and
    /// `static_cast<T>(... * normalizer)` store expressions, identical
    /// simd_max/simd_sum reduction order, identical addresses; only the
    /// memory transaction width changes. The scalar arm of every branch is
    /// the promoted source verbatim. The ranked decode key length runs
    /// 1024..1152, so about half the steps take the vector arm.
    ///
    /// Kill switch: `DARKBLOOM_GEMMA4_D512_SOFTMAX_VEC` set to
    /// `0`/`false`/`no`/`off` restores all three promoted kernels byte for
    /// byte, including their cached pipeline names. Default ON.
    private static let softmaxVecEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_SOFTMAX_VEC"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let softmaxVecKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_softmax_bf16_v1_sv1",
        inputNames: ["scores", "params"],
        outputNames: ["probs"],
        source: """
            const int axis_size = int(params[0]);
            const int gid = int(threadgroup_position_in_grid.x);
            const int lid = int(thread_position_in_threadgroup.x);
            const int simd_lane_id = int(thread_index_in_simdgroup);
            const int simd_group_id = int(simdgroup_index_in_threadgroup);
            const int num_simdgroups = (axis_size + 127) / 128;

            typedef vec<T, 4> T4;
            const bool row_vec4 = (axis_size & 3) == 0;

            threadgroup float local_max[32];
            threadgroup float local_normalizer[32];

            float ld[4];
            const device T* in =
                scores + size_t(gid) * axis_size + lid * 4;
            if (lid * 4 + 4 <= axis_size) {
                if (row_vec4) {
                    const T4 raw = *reinterpret_cast<const device T4*>(in);
                    for (int i = 0; i < 4; i++) {
                        ld[i] = static_cast<float>(raw[i]);
                    }
                } else {
                    for (int i = 0; i < 4; i++) {
                        ld[i] = static_cast<float>(in[i]);
                    }
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    ld[i] = ((lid * 4 + i) < axis_size)
                        ? static_cast<float>(in[i]) : -INFINITY;
                }
            }

            float maxval = -3.402823466e+38F;
            for (int i = 0; i < 4; i++) {
                maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) {
                local_max[simd_group_id] = maxval;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            maxval = -3.402823466e+38F;
            for (int s = 0; s < num_simdgroups; ++s) {
                const float sm = local_max[s];
                maxval = (maxval < sm) ? sm : maxval;
            }

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

            normalizer = 0.0f;
            for (int s = 0; s < num_simdgroups; ++s) {
                normalizer += local_normalizer[s];
            }
            const float inv_normalizer = 1.0f / normalizer;

            device T* out_row =
                probs + size_t(gid) * axis_size + lid * 4;
            if (lid * 4 + 4 <= axis_size) {
                if (row_vec4) {
                    T4 out_vec;
                    for (int i = 0; i < 4; i++) {
                        out_vec[i] = static_cast<T>(ld[i] * inv_normalizer);
                    }
                    *reinterpret_cast<device T4*>(out_row) = out_vec;
                } else {
                    for (int i = 0; i < 4; i++) {
                        out_row[i] = static_cast<T>(ld[i] * inv_normalizer);
                    }
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    if ((lid * 4 + i) < axis_size) {
                        out_row[i] = static_cast<T>(ld[i] * inv_normalizer);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static var softmaxActive: MLXFast.MLXFastKernel {
        softmaxVecEnabled ? softmaxVecKernel : softmaxKernel
    }

    /// AV-TILES-001: the column tiling of dispatch 3.
    ///
    /// probs·V is the heaviest byte stream of the D=512 decode chain — it
    /// reads every row's whole value plane once per full-attention layer, and
    /// nothing else in the chain touches that much memory. It shipped as 8
    /// column tiles of 64 per row and kv head, so the launch is
    /// `batch * kvHeads * 8` = 128 threadgroups of four simdgroups. Every
    /// threadgroup walks the entire key length, so they all cost the same,
    /// and 128 divides no shipped core count evenly: the cores that draw one
    /// threadgroup more than their neighbours hold the whole dispatch open
    /// for a full extra tile while the rest stand idle.
    ///
    /// Halving the tile to 32 columns doubles the launch to 256 threadgroups
    /// of two simdgroups, which cuts that residue term roughly in half at the
    /// same total thread count and the same per-lane register footprint.
    ///
    /// Exactness: a simdgroup keeps its lane→column stride, its 4×4 value
    /// tile, its key-major walk over the same 32-key blocks and the same
    /// cross-lane butterfly, because none of those read `sg`. Only the base
    /// column a simdgroup starts from moves, so every output element is
    /// accumulated from the same terms in the same order. The value plane
    /// stays partitioned into disjoint column runs and a simdgroup still
    /// issues one 32-byte contiguous run per key row, so the coalescing is
    /// unchanged too; only the probability rows, which are small next to the
    /// value plane and stay resident, are re-read by twice as many tiles.
    ///
    /// `DARKBLOOM_GEMMA4_D512_AV_TILES=8` restores the incumbent geometry.
    private static let avColumnTiles: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_D512_AV_TILES"], let value = Int(raw)
        else { return 16 }
        return value == 8 || value == 16 ? value : 16
    }()

    /// Columns one threadgroup of dispatch 3 owns, and the simdgroups it
    /// needs to cover them at the frozen 16 columns per simdgroup.
    private static let avTileColumns = headDim / avColumnTiles
    private static let avSimdgroups = avTileColumns / 16

    /// Dispatch 3 — probs·V. Grid: (row, kv head, column tile) × 32·SG
    /// threads. Replays the stock GEMVTKernel<bf16,1,4,8,4,4,4> row-striding
    /// and butterfly for all 8 heads of the GQA group at once (shared V tile
    /// loads). params as dispatch 1.
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

    /// CUT-13: alignment-gated vec4 twin of dispatch 3 above. Identical to
    /// the promoted kernel except for the threadgroup-uniform
    /// `(key_length & 3) == 0` gate and the `p_coeff` loop: the 4-aligned
    /// arm loads each head's probability chunk as one `vec<T, 4>` from
    /// `prob_rows + h * key_length + bm`, an address that is a multiple of
    /// four elements (row base `(row * 16 + kv_head * GQA) * key_length`,
    /// head stride `h * key_length`, `bm = 32 * i + 4 * thrM` all are, and
    /// `bm + 4 <= 32 * n_iter <= key_length` stays in row). Per-element
    /// `static_cast<float>`, the XFOLD butterfly and every store are
    /// unchanged; the scalar arm and the leftover tail are the promoted
    /// source verbatim.
    private static let avVecKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_xfold_v3_t\(avColumnTiles)_vec1_sv1",
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
            const bool row_vec4 = (key_length & 3) == 0;

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
                    if (row_vec4) {
                        const T4 p_raw = *reinterpret_cast<const device T4*>(
                            prob_rows + size_t(h) * key_length + bm);
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(p_raw[tm]);
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(
                                prob_rows[size_t(h) * key_length + bm + tm]);
                        }
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

    private static var avActive: MLXFast.MLXFastKernel {
        softmaxVecEnabled ? avVecKernel : avKernel
    }

    /// ORS-D512: how many per-threadgroup partials one o_proj 64-group of the
    /// activation is split into. At the AV-TILES-001 default of 32-column
    /// tiles a 64-group spans two threadgroups, so dispatch 3 emits a
    /// `[8, 256]` pair table and the o_proj kernel adds each pair (the
    /// prepass's own final xor-4 stage); at `DARKBLOOM_GEMMA4_D512_AV_TILES=8`
    /// the group sits inside one threadgroup and the `[8, 128]` table is
    /// emitted whole.
    private static let avORunsumPartials = avColumnTiles / 8

    /// ORS-D512 twin of the CUT-13 dispatch-3 kernel: the same body verbatim,
    /// plus an epilogue that emits the o_proj run-sum table (or its pair
    /// partials) for the activation this dispatch just stored, so the
    /// standalone `cbv2_b8_rs_table_dyn_v1` prepass never runs on a full
    /// layer.
    ///
    /// Exactness. The prepass lane `fm` reads the eight consecutive BF16
    /// values `64g + 8fm ... + 7` of the o_proj input, forms the two quads
    /// `((x0+x1)+x2)+x3` and `((x4+x5)+x6)+x7` in T, adds them into a float
    /// as `(0 + qA) + qB`, then butterflies xor 1, 2, 4 over `fm`. Here the
    /// o_proj input is this kernel's own `out` plane read back head-major
    /// (column `head * 512 + d`), and after the XFOLD every lane holds the
    /// four consecutive stored columns `out_col ... + 3` of head `thrM`, so
    /// the octet `fm` is the lane pair `(thrN, thrN ^ 1)` of one simdgroup:
    /// the even lane holds `x0..x3`, the odd lane `x4..x7`. Each lane
    /// gathers the partner's four stored bit patterns, writes the two quads
    /// and the float sum as the prepass's own statements in the prepass's
    /// order, and both lanes of the pair hold the identical octet value.
    /// `fm` bit 0 is lane bit 1 (`simd_shuffle_xor 2`), bit 1 is the
    /// simdgroup pair (threadgroup memory), bit 2 is the second simdgroup
    /// pair at 64-column tiles or the neighbouring threadgroup at 32-column
    /// tiles (the pair table). Every node adds the same two subtrees the
    /// prepass's butterfly adds; float addition is commutative, so the
    /// left/right order at a node is immaterial. The `out` store, the
    /// accumulation and the fold above it are the promoted text.
    private static let avVecORunsumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_sdpa_d512_av_bf16_g8_xfold_v3_t\(avColumnTiles)_vec1_sv1_ors1",
        inputNames: [
            "probs",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params",
        ],
        outputNames: ["out", "rs"],
        source: """
            constexpr int D = 512;
            constexpr int GQA = 8;

            const int key_length = int(params[0]);
            const bool row_vec4 = (key_length & 3) == 0;

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
                    if (row_vec4) {
                        const T4 p_raw = *reinterpret_cast<const device T4*>(
                            prob_rows + size_t(h) * key_length + bm);
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(p_raw[tm]);
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(
                                prob_rows[size_t(h) * key_length + bm + tm]);
                        }
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
            // ORS-D512 epilogue: the o_proj run-sum octet tree of
            // cbv2_b8_rs_table_dyn_v1 over the values just stored.
            threadgroup float rs_partial[\(avSimdgroups)][32];
            {
                thread ushort own[4];
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    own[j] = as_type<ushort>(static_cast<T>(result[j]));
                }
                const ushort partner = ushort(lane ^ 1);
                const bool upper = (lane & 1) != 0;
                thread T xt[8];
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    const ushort other = simd_shuffle(own[j], partner);
                    xt[j] = as_type<T>(upper ? other : own[j]);
                    xt[4 + j] = as_type<T>(upper ? own[j] : other);
                }
                float rsv = 0;
                rsv += xt[0] + xt[1] + xt[2] + xt[3];
                rsv += xt[4] + xt[5] + xt[6] + xt[7];
                rsv += simd_shuffle_xor(rsv, 2u);
                rs_partial[sg][lane] = rsv;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rsv += rs_partial[sg ^ 1][lane];
                \(avSimdgroups == 4
                    ? "rsv += rs_partial[sg ^ 2][lane] + rs_partial[sg ^ 3][lane];"
                    : "")
                if (sg == 0 && thrN == 0) {
                    rs[row * (16 * \(avColumnTiles))
                        + (kv_head * GQA + thrM) * \(avColumnTiles) + tile] = rsv;
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


    /// WRITE-016-D512 fused variant of dispatch 3: identical output
    /// arithmetic; the new token's V row (logical row kL-1) is served from
    /// `new_values` rather than the cache slot the fused QK dispatch wrote,
    /// so this kernel has no read-after-in-place-write hazard at all.
    private static let fusedAvKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_av_bf16_g8_xfold_v3_t\(avColumnTiles)_vec1",
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
            const device T* new_value_plane =
                new_values + size_t(row * 2 + kv_head) * D;

            const device T* prob_rows =
                probs + size_t(row * 16 + kv_head * GQA) * key_length;

            const int thrM = lane / 4;
            const int thrN = lane % 4;
            int bm = thrM * 4;
            const int out_col = tile * \(avTileColumns) + (4 * sg + thrN) * 4;

            float result[GQA * 4] = {0.0f};
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
                            result[h * 4 + tn] += vc * v_tile[tm][tn];
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
                            result[h * 4 + tn] += pc * v_tile[0][tn];
                        }
                    }
                }
            }
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

    /// CUT-13: alignment-gated vec4 twin of the fused dispatch 3 above.
    /// Same gate, same `p_coeff` vec4 arm and same alignment proof as
    /// `avVecKernel` (`probs` here is the fresh scratch the dispatch-2
    /// output allocation hands this kernel); the value-tile peel, the
    /// shuffle-down fold and every store are the promoted source verbatim.
    private static let fusedAvVecKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_writesdpa_d512_av_bf16_g8_xfold_v3_t\(avColumnTiles)_vec1_sv1",
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
            const bool row_vec4 = (key_length & 3) == 0;

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
            const device T* new_value_plane =
                new_values + size_t(row * 2 + kv_head) * D;

            const device T* prob_rows =
                probs + size_t(row * 16 + kv_head * GQA) * key_length;

            const int thrM = lane / 4;
            const int thrN = lane % 4;
            int bm = thrM * 4;
            const int out_col = tile * \(avTileColumns) + (4 * sg + thrN) * 4;

            float result[GQA * 4] = {0.0f};
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
                    if (row_vec4) {
                        const T4 p_raw = *reinterpret_cast<const device T4*>(
                            prob_rows + size_t(h) * key_length + bm);
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(p_raw[tm]);
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (int tm = 0; tm < 4; ++tm) {
                            p_coeff[tm] = static_cast<float>(
                                prob_rows[size_t(h) * key_length + bm + tm]);
                        }
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
                            result[h * 4 + tn] += pc * v_tile[0][tn];
                        }
                    }
                }
            }
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

    private static var fusedAvActive: MLXFast.MLXFastKernel {
        softmaxVecEnabled ? fusedAvVecKernel : fusedAvKernel
    }


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

    /// NORMROPE-D512: the WRITE-022 store dispatch with the full layers' Q/K
    /// RMSNorm + RoPE folded in, so the standalone
    /// `gemma4_b8_qkv_rms_norm_rope_v2_vec1` dispatch leaves the chain.
    ///
    /// Geometry: 144 threadgroups of 128 threads — exactly the standalone
    /// kernel's one-row-per-threadgroup, four-values-per-thread launch over
    /// its 16 K rows and 128 Q rows (`threads = D / 4`). Threadgroups 0...15
    /// are the incumbent store's (row, kv head) pairs: they reduce the raw key
    /// row, write V = `RMSNormNoScale(raw key)` and K = `RoPE(k_weight ·
    /// RMSNorm(raw key))` into slot kL-1 of the row's private buffers (and
    /// mirror both rows into `k_out`/`v_out` for the layer's KV-pair return).
    /// Threadgroups 16...143 normalize and rotate one query row each into
    /// `q_out`, the `[8, 16, 1, 512]` plane dispatch 1 reads.
    ///
    /// Exactness: every statement of the standalone kernel's K-row and Q-row
    /// arithmetic is transcribed with the same thread geometry — per-lane
    /// four-value square sum, one `simd_sum` per simdgroup, the four partials
    /// in lanes 0...3 (lanes 4...31 zero) of a second `simd_sum`,
    /// `precise::rsqrt(sum / D + 1e-6)`, the BF16 `rounded[]` store boundary
    /// before RoPE, `inv_freq = 1 / rope_freqs[pair]` from the 256-entry
    /// proportional table (+inf pass-through pairs included), `fast::cos/sin`
    /// of `L * inv_freq`, and the same two rotation expressions. The ring
    /// slot receives the K row the standalone kernel would have handed the
    /// incumbent store, so dispatches 1...3 read identical bytes.
    private static let ringStoreNormRopeKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ragged8_d512_ringstore_normrope_freqs_bf16_v1_vec1",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "params", "raw_queries", "raw_keys", "q_weight", "k_weight",
            "position_offsets", "rope_freqs", "write_fence",
        ],
        outputNames: ["fence", "q_out", "k_out", "v_out"],
        source: """
            constexpr int D = 512;
            constexpr int KV_ROWS = 16;
            constexpr int Q_HEADS = 16;
            constexpr int K_HEADS = 2;
            constexpr int reads = 4;
            typedef vec<T, 4> T4;

            const int z = int(threadgroup_position_in_grid.z);
            const int lid = int(thread_position_in_threadgroup.x);
            const int lane = int(thread_index_in_simdgroup);
            const int simd_group = int(simdgroup_index_in_threadgroup);
            const int key_length = int(params[0]);

            // Threadgroups 0...15 own one (row, kv head) K/V pair each, in the
            // incumbent store's order; 16...143 own one query row each, in the
            // standalone kernel's `local_row = batch * 16 + head` order.
            const bool is_key = z < KV_ROWS;
            const int local_row = is_key ? z : z - KV_ROWS;
            const int batch_index = is_key ? local_row / K_HEADS : local_row / Q_HEADS;
            const int kv_head = local_row % K_HEADS;

            const device T* input = is_key ? raw_keys : raw_queries;
            const device T* weight = is_key ? k_weight : q_weight;
            input += local_row * D + lid * reads;
            weight += lid * reads;

            device T* key_slot = k_out;
            device T* value_slot = v_out;
            if (is_key) {
                const int row_capacity = int(params[2 + batch_index]);
                const device T* key_plane = k0;
                const device T* value_plane = v0;
                switch (batch_index) {
                    case 1: key_plane = k1; value_plane = v1; break;
                    case 2: key_plane = k2; value_plane = v2; break;
                    case 3: key_plane = k3; value_plane = v3; break;
                    case 4: key_plane = k4; value_plane = v4; break;
                    case 5: key_plane = k5; value_plane = v5; break;
                    case 6: key_plane = k6; value_plane = v6; break;
                    case 7: key_plane = k7; value_plane = v7; break;
                    default: break;
                }
                key_slot = const_cast<device T*>(key_plane)
                    + size_t(kv_head) * size_t(row_capacity) * D
                    + size_t(key_length - 1) * D;
                value_slot = const_cast<device T*>(value_plane)
                    + size_t(kv_head) * size_t(row_capacity) * D
                    + size_t(key_length - 1) * D;
            }
            device T* output_row = is_key ? key_slot : (q_out + local_row * D);
            // Query rows never dereference the mirrors; keep their pointers
            // inside the K/V allocations anyway.
            device T* key_mirror = k_out + (is_key ? local_row : 0) * D;
            device T* value_mirror = v_out + (is_key ? local_row : 0) * D;

            const T4 vin = *reinterpret_cast<const device T4*>(input);
            float sum = 0.0f;
            for (int i = 0; i < reads; ++i) {
                const float value = float(vin[i]);
                sum += value * value;
            }
            sum = simd_sum(sum);

            threadgroup float partials[32];
            threadgroup float inverse_rms;
            threadgroup T rounded[D];
            if (simd_group == 0) partials[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) partials[simd_group] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum = simd_sum(partials[lane]);
                if (lane == 0) {
                    inverse_rms = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const T4 wv = *reinterpret_cast<const device T4*>(weight);
            for (int i = 0; i < reads; ++i) {
                const int element = lid * reads + i;
                const T normalized = T(float(vin[i]) * inverse_rms);
                // Reproduce the separate norm kernel's BF16 output-store
                // boundary before any RoPE arithmetic reads the value.
                rounded[element] = T(wv[i] * normalized);
            }
            // K rows also carry V: the same raw row and the same normalizer,
            // written with RMSNormNoScale's own final expression, straight
            // into slot kL-1 of the value ring (and the layer's V return).
            if (is_key) {
                T4 sharedv;
                for (int i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    sharedv[i] = T(1) * normalized;
                }
                *reinterpret_cast<device T4*>(value_slot + lid * reads) = sharedv;
                *reinterpret_cast<device T4*>(value_mirror + lid * reads) = sharedv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lid * reads < D / 2) {
                const float L = static_cast<float>(position_offsets[batch_index]);
                for (int i = 0; i < reads; ++i) {
                    const int pair = lid * reads + i;
                    const float inv_freq = 1.0f / rope_freqs[pair];
                    const float theta = L * inv_freq;
                    const float costheta = metal::fast::cos(theta);
                    const float sintheta = metal::fast::sin(theta);
                    const float x1 = static_cast<float>(rounded[pair]);
                    const float x2 = static_cast<float>(rounded[pair + D / 2]);
                    const float rx1 = x1 * costheta - x2 * sintheta;
                    const float rx2 = x1 * sintheta + x2 * costheta;
                    output_row[pair] = static_cast<T>(rx1);
                    output_row[pair + D / 2] = static_cast<T>(rx2);
                    if (is_key) {
                        key_mirror[pair] = static_cast<T>(rx1);
                        key_mirror[pair + D / 2] = static_cast<T>(rx2);
                    }
                }
            }
            if (z == 0 && lid == 0) {
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
        let paramsArray = getD512ParamsArray(params: params)

        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype)
        ]
        let scratchShape = [batch, queryHeads, 1, keyLength]

        // NORMROPE-D512: with the raw projections registered for this exact
        // (queries, keys, values) triple, the store dispatch normalizes and
        // rotates them itself and hands dispatch 1 its own `q_out`; the
        // registered fallback arrays are then never evaluated. A miss keeps
        // the WRITE-022 store byte for byte.
        let storeFence: MLXArray
        let liveQueries: MLXArray
        var normalizedKeys: MLXArray? = nil
        var normalizedValues: MLXArray? = nil
        if normRopeFoldEnabled,
            let normRope = takeFullNormRope(
                queries: queries, keys: keys, values: values)
        {
            let stored = ringStoreNormRopeKernel(
                keyBuffers + valueBuffers + [
                    paramsArray,
                    normRope.rawQueries,
                    normRope.rawKeys,
                    normRope.qWeight,
                    normRope.kWeight,
                    normRope.positionOffsets,
                    normRope.ropeFrequencies,
                    previousWriteFence,
                ],
                template: template,
                grid: (128, 1, batch * kvHeads + batch * queryHeads),
                threadGroup: (128, 1, 1),
                outputShapes: [
                    [1],
                    [batch, queryHeads, 1, headDim],
                    [batch, kvHeads, 1, headDim],
                    [batch, kvHeads, 1, headDim],
                ],
                outputDTypes: [.int32, .bfloat16, .bfloat16, .bfloat16]
            )
            storeFence = stored[0]
            liveQueries = stored[1]
            normalizedKeys = stored[2]
            normalizedValues = stored[3]
            CBv2EngageMark.once("d512-normrope-store")
        } else {
            storeFence = ringStoreKernel(
                keyBuffers + valueBuffers
                    + [paramsArray, keys, values, previousWriteFence],
                template: template,
                grid: (128, 1, batch * kvHeads),
                threadGroup: (128, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.int32]
            )[0]
            liveQueries = queries
        }

        let chunks = (keyLength + 63) / 64
        let scores = qkFencedKernel(
            [liveQueries] + keyBuffers + [paramsArray, storeFence],
            template: template,
            grid: (32, 4, batch * kvHeads * chunks),
            threadGroup: (32, 4, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        let softmaxThreads = ((keyLength + 3) / 4 + 31) / 32 * 32
        let probs = softmaxActive(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        // ORS-D512: dispatch 3 also emits the o_proj run-sum table (or its
        // per-threadgroup pair partials) for the activation it stores.
        let output: MLXArray
        let oRunsum: MLXArray?
        if oRunsumFoldEnabled, softmaxVecEnabled {
            let attended = avVecORunsumKernel(
                [probs] + valueBuffers + [paramsArray],
                template: template,
                grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
                threadGroup: (32, avSimdgroups, 1),
                outputShapes: [
                    [batch, queryHeads, 1, headDim],
                    [batch, queryHeads * headDim / 64 * avORunsumPartials],
                ],
                outputDTypes: [.bfloat16, .float32]
            )
            output = attended[0]
            oRunsum = attended[1]
            CBv2EngageMark.once(
                avORunsumPartials == 1 ? "d512-ors-av-table" : "d512-ors-av-pairs")
        } else {
            output = avActive(
                [probs] + valueBuffers + [paramsArray],
                template: template,
                grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
                threadGroup: (32, avSimdgroups, 1),
                outputShapes: [[batch, queryHeads, 1, headDim]],
                outputDTypes: [.bfloat16]
            )[0]
            oRunsum = nil
        }

        if oRunsum != nil || normalizedKeys != nil {
            CBv2RaggedTwoPassDecodeAttentionV1.publishResidentProducts(
                CBv2RaggedTwoPassDecodeAttentionV1.ResidentProducts(
                    runsumTable: oRunsum,
                    normalizedKeys: normalizedKeys,
                    normalizedValues: normalizedValues),
                for: output)
        }

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
        let paramsArray = getD512ParamsArray(params: params)

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
        let probs = softmaxActive(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        let output = fusedAvActive(
            [probs] + valueBuffers + [paramsArray, values],
            template: template,
            grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
            threadGroup: (32, avSimdgroups, 1),
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
        return dispatchChain(
            queries: queries, keyBuffers: keyBuffers, valueBuffers: valueBuffers,
            params: params, keyLength: keyLength)
    }

    /// LASTQ-D512: the attend-only twin of `updateAndAttend` for the final
    /// layer's last-query prefill (see LastQueryPrefillV2.swift). The caller
    /// (`CBv2AttentionV1.updateAndAttendLastQuery`) has ALREADY committed
    /// every row's full K/V chunk — the byte-identical per-row `update`
    /// calls in row order — so this runs the stock three-dispatch chain over
    /// the committed buffers with kL = the rows' common committed length and
    /// appends NOTHING of its own (a chunk can never be double-committed;
    /// the WRITE variants are not applicable for exactly that reason).
    ///
    /// Admission mirrors `updateAndAttend` with the decode-only gates
    /// translated: kL is the committed length (not `offset + 1` — nothing is
    /// appended), there is no new-token keys/values shape to check, and
    /// `absoluteOffset <= maxLength` is a `CBv2FullSequenceKV` invariant
    /// rather than an append-fit gate. Fails closed (nil, NO side effects)
    /// on any miss, exactly like `updateAndAttend`.
    ///
    /// Exactness: same claim and same scope as `updateAndAttend`. The
    /// per-row graph this replaces is the SAME `attend(..., L: 1, ...)`
    /// call — maskMode `.none` at L == 1, scale 1.0, no sinks/softcap,
    /// same committed `[1, kvHeads, kL, headDim]` views — the decode
    /// per-row loop makes at the same kL, and the chain is parity-verified
    /// bit-exact against that graph at kL ∈ {1024, 1027, 1100, 1152, 1055,
    /// 2048, 4095} (see the enum header). How the committed bytes got there
    /// (one-token decode append vs one whole-chunk prefill append) is not
    /// an input the kernels can observe.
    static func attendCommitted(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray? {
        guard enabled,
            lastQueryPrefillEnabled,
            rows.count == batch,
            scale == 1.0,
            sinks == nil,
            softcap == nil,
            !kind.isBidirectional,
            kind.queryHeads == queryHeads,
            kind.kvHeads == kvHeads,
            kind.headDim == headDim,
            queries.dtype == .bfloat16,
            queries.shape == [batch, queryHeads, 1, headDim]
        else { return nil }
        guard case .full = kind.attention else { return nil }

        let fullRows = rows.compactMap { $0 as? CBv2FullSequenceKV }
        guard fullRows.count == batch else { return nil }

        // Lockstep + storage gates, ALL before any dispatch. Pooled
        // (ATT-008) rows fail closed exactly like the decode path: their
        // backing layout is the pool's batch axis.
        let keyLength = fullRows[0].absoluteOffset
        guard keyLength >= minKeyLength,
            keyLength <= maxKeyLength,
            fullRows.allSatisfy({ $0.cohortPool == nil }),
            fullRows.allSatisfy({ $0.absoluteOffset == keyLength })
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
        return dispatchChain(
            queries: queries, keyBuffers: keyBuffers, valueBuffers: valueBuffers,
            params: params, keyLength: keyLength)
    }

    /// The stock three-dispatch chain (QKᵀ → precise softmax → probs·V) over
    /// collected committed row buffers, byte-for-byte as `updateAndAttend`
    /// always dispatched it. `params` = [kL, D, per-row KV buffer
    /// capacities...]. Shared by the decode append path and the last-query
    /// attend-only path so the two cannot drift.
    private static func dispatchChain(
        queries: MLXArray, keyBuffers: [MLXArray], valueBuffers: [MLXArray],
        params: [UInt32], keyLength: Int
    ) -> MLXArray {
        let paramsArray = getD512ParamsArray(params: params)

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
        let probs = softmaxActive(
            [scores, paramsArray],
            template: template,
            grid: (softmaxThreads * batch * queryHeads, 1, 1),
            threadGroup: (softmaxThreads, 1, 1),
            outputShapes: [scratchShape],
            outputDTypes: [.bfloat16]
        )[0]

        return avActive(
            [probs] + valueBuffers + [paramsArray],
            template: template,
            grid: (32, avSimdgroups, batch * kvHeads * avColumnTiles),
            threadGroup: (32, avSimdgroups, 1),
            outputShapes: [[batch, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
