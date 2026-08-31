// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import MLX
import MLXFast

/// Exact two-stage argmax for a small batch with a large vocabulary.
///
/// The stock Metal ArgReduce launches one threadgroup per row. A [8, 262144]
/// head therefore scans all logits with just eight groups. Here each group
/// scans 4096 values, then one SIMD group per row reduces the tile winners.
/// Only the decomposition changes: no values are added, scaled or rounded.
///
/// This is called on the sampler's already-transformed logits, after grammar,
/// penalties and probability filtering. It neither replaces those transforms
/// nor changes raw-logprob capture or sampler state.
enum CBv2ParallelArgMaxV1 {
    private static let tileSize = 4096
    private static let tileThreads = 256
    private static let finalThreads = 32

    /// Independent fallback for attribution and emergency bisection.
    /// An absent setting enables the implementation; an explicit off value
    /// restores the stock argMax + int32 conversion.
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_PARALLEL_ARGMAX"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }()

    private static let metalAvailable: Bool = {
        var available = false
        return mlx_metal_is_available(&available) == 0 && available
    }()

    /// Stock ArgMax starts from {index: 0, value: -infinity}, updates each
    /// lane only for a strictly greater value, and breaks inter-lane ties by
    /// the smaller original index. Thus NaNs never win and an all-NaN/-inf
    /// row returns zero. Each tile retains GLOBAL indices, including the
    /// stock zero sentinel when it contains no value greater than -inf.
    /// BF16 -> float is exact, and float32 inputs need no conversion.
    private static let header = """
        struct CBv2ArgMaxPairV1 {
            uint index;
            float value;
        };

        inline CBv2ArgMaxPairV1 cbv2_argmax_choose_v1(
            CBv2ArgMaxPairV1 best, CBv2ArgMaxPairV1 current) {
            if (best.value < current.value ||
                (best.value == current.value && best.index > current.index)) {
                return current;
            }
            return best;
        }

        inline CBv2ArgMaxPairV1 cbv2_argmax_simd_v1(CBv2ArgMaxPairV1 best) {
            #pragma unroll
            for (uint offset = 16; offset > 0; offset >>= 1) {
                CBv2ArgMaxPairV1 neighbor = {
                    simd_shuffle_down(best.index, offset),
                    simd_shuffle_down(best.value, offset),
                };
                best = cbv2_argmax_choose_v1(best, neighbor);
            }
            return best;
        }
        """

    private static let tileKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_parallel_argmax_tiles_v2",
        inputNames: ["logits"],
        outputNames: ["tile_values", "tile_indices"],
        source: """
            const uint tile = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const size_t row_base = size_t(row) * VOCAB;
            const uint tile_base = tile * TILE_SIZE;

            CBv2ArgMaxPairV1 best = {0u, Limits<float>::min};
            for (uint offset = lid * 4; offset < TILE_SIZE; offset += 256 * 4) {
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const uint index = tile_base + offset + i;
                    const float value = float(logits[row_base + index]);
                    // Classify NaNs by bits so fast-math cannot discard the
                    // stock rule that an unordered comparison never wins.
                    const bool is_nan =
                        (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
                    if (!is_nan && value > best.value) {
                        best = {index, value};
                    }
                }
            }

            best = cbv2_argmax_simd_v1(best);
            threadgroup float group_values[8];
            threadgroup uint group_indices[8];
            if (lane == 0) {
                group_values[sg] = best.value;
                group_indices[sg] = best.index;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg != 0) return;

            best = {0u, Limits<float>::min};
            if (lane < 8) {
                best = {group_indices[lane], group_values[lane]};
            }
            best = cbv2_argmax_simd_v1(best);
            if (lane == 0) {
                const uint output_index = row * TILES + tile;
                tile_values[output_index] = best.value;
                tile_indices[output_index] = best.index;
            }
            """,
        header: header,
        ensureRowContiguous: true
    )

    private static let finalKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_parallel_argmax_finalize_v2",
        inputNames: ["tile_values", "tile_indices"],
        outputNames: ["tokens"],
        source: """
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            CBv2ArgMaxPairV1 best = {0u, Limits<float>::min};
            #pragma unroll
            for (uint tile = lane; tile < TILES; tile += 32) {
                const uint index = row * TILES + tile;
                CBv2ArgMaxPairV1 current = {
                    tile_indices[index], tile_values[index],
                };
                best = cbv2_argmax_choose_v1(best, current);
            }
            best = cbv2_argmax_simd_v1(best);
            if (lane == 0) {
                // VOCAB <= 2^20, so this is exactly stock uint32 -> int32.
                tokens[row] = int(best.index);
            }
            """,
        header: header,
        ensureRowContiguous: true
    )

    /// Returns nil outside the narrow performance envelope. All shape and
    /// dtype checks use lazy metadata only. Never read Swift array strides:
    /// they are unstable before evaluation. Instead both kernels REQUIRE
    /// row-contiguous storage via MLX's eval-time ensureRowContiguous check;
    /// a strided input is bit-copied before the kernel, never misindexed.
    static func apply(_ logits: MLXArray) -> MLXArray? {
        guard enabled, logits.ndim == 2,
            logits.dtype == .bfloat16 || logits.dtype == .float32
        else { return nil }
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        guard rows > 0, rows <= 8,
            vocab >= 65536, vocab <= 1_048_576,
            vocab % tileSize == 0
        else { return nil }

        // Preserve CPU execution, including a task-scoped CPU stream on a
        // Metal-capable host. These checks inspect host metadata only.
        guard metalAvailable else { return nil }
        let stream = StreamOrDevice.default
        var device = mlx_device_new()
        defer { _ = mlx_device_free(device) }
        var deviceType = MLX_CPU
        guard mlx_stream_get_device(&device, stream.ctx) == 0,
            mlx_device_get_type(&deviceType, device) == 0,
            deviceType == MLX_GPU
        else { return nil }

        let tiles = vocab / tileSize
        let partial = tileKernel(
            [logits],
            template: [("VOCAB", vocab), ("TILE_SIZE", tileSize), ("TILES", tiles)],
            grid: (tiles * tileThreads, rows, 1),
            threadGroup: (tileThreads, 1, 1),
            outputShapes: [[rows, tiles], [rows, tiles]],
            outputDTypes: [.float32, .uint32],
            stream: stream
        )
        let tokens = finalKernel(
            partial,
            template: [("TILES", tiles)],
            grid: (finalThreads, rows, 1),
            threadGroup: (finalThreads, 1, 1),
            outputShapes: [[rows]],
            outputDTypes: [.int32],
            stream: stream
        )[0]
        CBv2EngageMark.once("parallel-greedy-argmax")
        return tokens
    }
}
