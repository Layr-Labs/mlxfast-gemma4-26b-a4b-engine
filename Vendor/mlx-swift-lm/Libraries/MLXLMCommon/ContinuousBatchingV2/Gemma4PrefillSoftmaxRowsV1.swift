// PREFILL-SOFTMAX-ROWPACK: pack R score rows into one threadgroup on the
// prefill plane, leaving every arithmetic step of MLX's own kernel in place.
//
// The composed prefill attention normalises a score rectangle per q-block:
// `[B, kvHeads, repeats, L, kL]` with `L = 128` and `kL` walking the eight
// causal q-blocks of a 1024-token chunk (128, 256, ... 1024). MLX dispatches
// `block_softmax_precise` for that, i.e. `softmax_single_row<T, float, 4>`
// (`backend/metal/kernels/softmax.h`), whose host (`backend/metal/softmax.cpp`)
// gives EACH ROW its own threadgroup of `ceil(kL / 4 / 32) * 32` threads.
//
// At these widths that threadgroup is tiny -- 32 threads at kL 128, 256 at
// kL 1024 -- and the kernel's runtime turns out to be quantised by the number
// of simdgroups per row rather than by the bytes it moves. Measured solo on
// this machine, one layer's eight widths:
//
//     kL      128  256  384  512  640  768  896 1024
//     us     39.0 68.0  134  131  260  259  250  250     (161-269 GB/s)
//
// kL 640 runs at 161 GB/s against a 350-365 GB/s streaming roof: the rectangle
// is not DRAM-bound, the launch shape is. Filling the threadgroup with R
// independent rows fixes exactly that, and nothing else:
//
//     best   34.4 38.9 62.2 87.0  109  132  158  181     (370-385 GB/s)
//     R         8    8    8    4    4    4    4    2
//
// 1354 us -> 803 us per layer, 29 layers = -16 ms per 8x1024 prefill step.
//
// EXACTNESS. This is a threadgroup-to-row REMAPPING, not a new reduction. Row
// `r` of the pack owns simdgroups `[r*spr, (r+1)*spr)` and threads
// `[r*tpr, (r+1)*tpr)` of the threadgroup, where `tpr = kL / 4` and
// `spr = tpr / 32`. Because `tpr` is a whole multiple of 32 (the `kL % 128 == 0`
// guard), the row-local thread index `rlid = lid - r*tpr` has the SAME residue
// mod 32 as `lid`, so `thread_index_in_simdgroup` is the row-local lane of the
// stock kernel and the row-local simdgroup index is `rsg = simd_group_id - r*spr`.
// Every step downstream then reads the same operands in the same order as
// `softmax_single_row`:
//
//   * the same four elements per thread at the same offsets (`ld[i]`),
//   * `Limits<float>::finite_min` seed and the same `maxval` walk,
//   * `simd_max` over the same 32 lanes holding the same values,
//   * a 32-slot cross-simd vector per row whose unused slots are seeded to
//     `Limits<float>::min` / `0` by the same `rsg == 0` lanes the stock kernel
//     uses, then reduced by the same `simd_max` / `simd_sum`,
//   * `softmax_exp` == `fast::exp` on `ld[i] - maxval`, fp32 accumulation in
//     thread-read order, `1 / normalizer`, one `T(ld[i] * normalizer)` store.
//
// The only thing that changes is WHICH threadgroup a row lands in, which no
// arithmetic depends on. Proven, not argued: `ScratchPrefillSoftmaxRowsTests`
// compares this kernel against `MLX.softmax(x, axis: -1, precise: true)` on the
// uint16 bit view over the production geometry at all eight widths, with random
// scores and with rows that carry the composed path's `0xFF7F` causal fill.
//
// SCOPE. bfloat16 only, `kL` a multiple of 128 in `128...1024`, a row count
// divisible by R, and a threadgroup that fits 1024 threads. Everything else --
// decode's `[B*heads, 1]` softmax rows, the router's `[8192, 128]` plane
// (`rows % R` and dtype pass, so it is excluded by the CALL SITE: only the
// composed prefill SDPA calls this), any non-contiguous or non-bf16 input --
// returns nil and takes `MLX.softmax` unchanged.
//
// Kill switch: `DARKBLOOM_GEMMA4_PREFILL_SOFTMAX_ROWS=0`.

import Foundation
import MLX
import MLXFast

public enum Gemma4PrefillSoftmaxRowsV1 {

    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_SOFTMAX_ROWS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// `SOFTMAX_N_READS` (`backend/metal/kernels/defines.h`), the elements each
    /// thread of `softmax_single_row` owns.
    private static let nReads = 4

    /// Rows packed into one threadgroup, per key width. The stock kernel's cost
    /// is `2^ceil(log2(kL / 128))` units of latency regardless of R, so R is
    /// chosen to fill the threadgroup without crossing 1024 threads; these are
    /// the measured optima (`$T/pprof/roofline/ub2`).
    public static func rowsPerGroup(kL: Int) -> Int {
        if kL <= 384 { return 8 }
        if kL <= 896 { return 4 }
        return 2
    }

    /// `axis_size` and `R` ride as one-element `int32` inputs rather than
    /// template arguments so that ALL eight widths share ONE jit-compiled
    /// library. The ranked leg has no warm-up: eight specialisations would pay
    /// eight first-use compiles inside the measured window.
    ///
    /// `MLXArray(Int32)` is a host construction (`mlx_array_new_int`), not an
    /// `astype` dispatch, but the two values are memoized anyway -- they are
    /// read-only inputs, never written, so sharing them across graphs is safe.
    private static let scalarLock = NSLock()
    nonisolated(unsafe) private static var scalarCache: [Int32: MLXArray] = [:]

    private static func scalar(_ value: Int) -> MLXArray {
        let key = Int32(value)
        scalarLock.lock()
        defer { scalarLock.unlock() }
        if let hit = scalarCache[key] { return hit }
        let made = MLXArray(key)
        if scalarCache.count < 64 { scalarCache[key] = made }
        return made
    }

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prefill_softmax_rows_v1",
        inputNames: ["scores", "axis", "rgroup"],
        outputNames: ["out"],
        source: """
            constexpr int SMR_N_READS = 4;

            // One 32-slot cross-simd vector per packed row, exactly the
            // `local_max` / `local_normalizer` of `softmax_single_row`.
            threadgroup float local_max[SMR_MAX_ROWS][32];
            threadgroup float local_normalizer[SMR_MAX_ROWS][32];

            const int axis_size = axis;
            const int rows_per_group = rgroup;
            const int tpr = axis_size / SMR_N_READS;   // threads per row
            const int spr = tpr / 32;                  // simdgroups per row

            const int lid_all = int(thread_position_in_threadgroup.x);
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const int r = int(simd_group_id) / spr;
            const int rsg = int(simd_group_id) - r * spr;
            const int lid = lid_all - r * tpr;
            const size_t row =
                size_t(threadgroup_position_in_grid.y) * rows_per_group + r;

            // `kL % 128 == 0` and `tpr * SMR_N_READS == axis_size`, so every
            // thread's four elements are in range: the stock kernel's aligned
            // branch, with its ragged twin unreachable.
            float ld[SMR_N_READS];
            const device T* in = scores + row * size_t(axis_size) + lid * SMR_N_READS;
            for (int i = 0; i < SMR_N_READS; i++) {
                ld[i] = static_cast<float>(in[i]);
            }

            if (rsg == 0) {
                local_max[r][simd_lane_id] = Limits<float>::min;
                local_normalizer[r][simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Get the max
            float maxval = Limits<float>::finite_min;
            for (int i = 0; i < SMR_N_READS; i++) {
                maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) {
                local_max[r][rsg] = maxval;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rsg == 0) {
                maxval = simd_max(local_max[r][simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_max[r][0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maxval = local_max[r][0];

            // Compute exp(x_i - maxval) and the partial sums
            float normalizer = 0;
            for (int i = 0; i < SMR_N_READS; i++) {
                float exp_x = smr_softmax_exp(ld[i] - maxval);
                ld[i] = exp_x;
                normalizer += exp_x;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane_id == 0) {
                local_normalizer[r][rsg] = normalizer;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rsg == 0) {
                normalizer = simd_sum(local_normalizer[r][simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_normalizer[r][0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            normalizer = 1 / local_normalizer[r][0];

            // Normalize and write to the output
            device T* op = out + row * size_t(axis_size) + lid * SMR_N_READS;
            for (int i = 0; i < SMR_N_READS; i++) {
                op[i] = static_cast<T>(ld[i] * normalizer);
            }
            """,
        header: """
            // Largest R the packed threadgroup vectors are sized for.
            constant constexpr const int SMR_MAX_ROWS = 8;

            // `softmax_exp` (`backend/metal/kernels/softmax.h`), verbatim.
            inline float smr_softmax_exp(float x) {
              return fast::exp(x);
            }
            """,
        ensureRowContiguous: true
    )

    /// `MLX.softmax(x, axis: -1, precise: true)` with R rows per threadgroup.
    /// Returns nil off the prefill score-rectangle plane.
    public static func softmax(_ x: MLXArray) -> MLXArray? {
        guard enabled, x.dtype == .bfloat16, x.ndim >= 2 else { return nil }
        let axisSize = x.dim(x.ndim - 1)
        guard axisSize >= 128, axisSize <= 1024, axisSize % 128 == 0 else { return nil }
        let total = x.size
        guard total > 0, total % axisSize == 0 else { return nil }
        let rows = total / axisSize
        let r = rowsPerGroup(kL: axisSize)
        guard rows % r == 0 else { return nil }
        let threadsPerRow = axisSize / nReads
        let threadgroup = r * threadsPerRow
        guard threadgroup > 0, threadgroup <= 1024 else { return nil }

        CBv2EngageMark.once("prefill-softmax-rowpack")
        return kernel(
            [x, scalar(axisSize), scalar(r)],
            template: [("T", x.dtype)],
            grid: (threadgroup, rows / r, 1),
            threadGroup: (threadgroup, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
    }
}
