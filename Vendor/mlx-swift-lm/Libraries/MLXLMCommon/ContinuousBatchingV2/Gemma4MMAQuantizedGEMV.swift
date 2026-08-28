// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFast

/// MMA-001 --- simdgroup-matrix affine-4/8 quantized GEMV for the tied LM head.
///
/// WHY. Gemma 4 26B A4B ties its embedding to the LM head, so `applyLMHead` is
/// one `quantizedMM` against a `[262144, 2816]` affine group-64 plane. At the
/// ranked cohort geometry the activation is `[8, 1, 2816]`, i.e. M = 8. Stock
/// `quantizedMM` routes that to ordinary `affine_qmv` (K = 2816 is not a
/// multiple of 512, so `qmv_fast` never launches), and the promoted tree
/// further routes N >= 8192 to `qmv_affine4_g64_quad_stream_impl`. That quad
/// serves FOUR activation rows per weight fetch, so an M = 8 head streams the
/// whole 369 MB vocab plane TWICE. The plane is the single largest contiguous
/// read in a decode step; halving its traffic is the entire mechanism here.
///
/// WHAT. One `MLXFast.metalKernel` that serves all EIGHT rows from a single
/// weight fetch by handing the unpacked codes to the simdgroup matrix units:
///
///   * a threadgroup is 4 simdgroups x 32 lanes = 128 threads and owns 32
///     output columns; each simdgroup owns 8 columns, i.e. one 8x8 `float`
///     accumulator tile whose rows are the eight activation rows;
///   * per affine group of 64 the simdgroup unpacks its 8 weight rows into
///     threadgroup memory ALREADY MULTIPLIED BY THE GROUP SCALE, and the
///     8-row activation tile is staged alongside it;
///   * nine `simdgroup_multiply_accumulate` calls consume the group: eight for
///     the 64 weight columns, and a ninth that carries the affine bias term as
///     a single extra K slot (see the algebra below), so the bias needs no
///     second accumulator and no separate reduction.
///
/// THE ALGEBRA. Affine quantization dequantizes as `w_k = s_g * q_k + b_g`, so
///
///     sum_k x_k * w_k == s_g * (sum_k x_k * q_k) + b_g * (sum_k x_k)
///
/// which is exactly how stock `qdot` closes (`scale * accum + sum * bias`).
/// This kernel folds `s_g` into the staged code (`s_g * q_k`, see EXACTNESS)
/// and appends `(sum_k x_k, b_g)` as K slot 64 of the same tile, so one MMA
/// chain produces both terms.
///
/// EXACTNESS. This is NOT bit-identical to stock and is not claimed to be.
/// Two properties are nevertheless held deliberately:
///
///   1. `s_g * q_k` is computed in `float` and is EXACT. `s_g` is bf16 (8-bit
///      significand) and `q_k` is an integer in 0...15 (4 bits), so the product
///      needs at most 12 significand bits and is exactly representable in
///      `float`. Folding the scale into the staged weight therefore introduces
///      no rounding that stock does not also have.
///   2. The `sum_k x_k` term reproduces stock `load_vector`'s BF16 QUAD-SUM
///      QUIRK verbatim. Stock sums each aligned group of four activations at
///      the ACTIVATION dtype before widening to `float`
///      (`sum += x[i] + x[i+1] + x[i+2] + x[i+3]`, `sum` a `float`), which is
///      measurably off exact math. The bias term is a first-class part of the
///      logit, so matching stock's answer -- not exact math -- is what keeps
///      greedy argmax aligned with the golden tape. The staging code below
///      writes that expression in stock's exact form and types.
///
/// What genuinely differs is FLOAT ACCUMULATION ORDER: the matrix units reduce
/// a tile as a tree, stock accumulates sequentially per lane and closes with a
/// `simd_sum` over 64 lanes. Both are float sums of the same terms.
///
/// FAIL-CLOSED. `apply` returns `nil` -- and every caller falls back to stock
/// `asLinear`/`quantizedMM` -- unless every one of these holds:
///
///   * `DARKBLOOM_GEMMA4_MMA_HEAD` is not one of `0`/`false`/`no`/`off`;
///   * the activation is bf16 and shaped `[8, K]` or `[8, 1, K]` -- eight
///     BATCH rows at one position, never an eight-token prefill chunk;
///   * affine group size 64, bits 4 (the tied head's quantization);
///   * K % 64 == 0 (whole affine groups) and K % 8 == 0 (whole uint32 words);
///   * N % 32 == 0 (whole threadgroups) and N >= 8192.
///
/// The N >= 8192 floor is load-bearing and must NOT be widened. A threadgroup
/// here claims only 32 output columns, so narrow planes (q/k/v/o, the dense
/// MLP) cannot fill the machine and measured SLOWER than ordinary pair/quad
/// QMV in isolation. Only the tied vocab plane is meant to enter.
public enum Gemma4MMAQuantizedGEMV {

    /// Rows the accumulator tile carries --- the ranked cohort's batch.
    private static let mRows = 8
    /// Output columns one simdgroup owns (one 8x8 tile).
    private static let colsPerSimdgroup = 8
    /// Simdgroups per threadgroup.
    private static let simdgroupsPerThreadgroup = 4
    /// Output columns one threadgroup owns.
    private static let colsPerThreadgroup = colsPerSimdgroup * simdgroupsPerThreadgroup
    /// Threads per threadgroup (one Apple simdgroup is 32 lanes).
    private static let threadsPerThreadgroup = simdgroupsPerThreadgroup * 32
    /// Narrowest plane allowed to enter. See FAIL-CLOSED above.
    private static let minOutputWidth = 8192

    /// `false` only when `DARKBLOOM_GEMMA4_MMA_HEAD` is an explicit off value.
    /// Resolved once; the kill switch is a process-level decision.
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_MMA_HEAD"]
        else { return true }
        switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
        case "0", "false", "no", "off": return false
        default: return true
        }
    }()

    /// Kernel source. `T` is the activation/scale dtype, `K` the contraction
    /// length, `N` the output width; all three arrive as template constants so
    /// the loop trip counts and the threadgroup allocation are compile-time.
    ///
    /// Layout notes for the two staging buffers:
    ///
    ///   * `Xs[m * A_STRIDE + j]` --- activation tile. `j` in 0..<64 is the
    ///     group's activations, `j == 64` is the bf16-quad `sum_k x_k` that
    ///     multiplies the affine bias, `j` in 65..<72 is zero padding so the
    ///     ninth 8-wide MMA slice contributes only the bias term.
    ///   * `Ws[sg][j * 8 + n]` --- staged weight tile for simdgroup `sg`, same
    ///     `j` convention: scaled codes, then the bias, then zeros.
    ///
    /// The pad slots are written once before the group loop and never touched
    /// again, so the loop stages exactly 64 + 1 slots per group.
    private static let source = """
        constexpr uint M_ROWS = 8;
        constexpr uint GROUP = 64;
        constexpr uint N_SG = 4;
        constexpr uint N_PSG = 8;
        constexpr uint A_STRIDE = 72;
        // W_STRIDE is 73 -- ODD ON PURPOSE. Threadgroup memory banks are the
        // word index mod 32, and a lane stages eight CONSECUTIVE j for one
        // output column n. With lane -> (n = lane/4, u = lane%4) the store
        // address is 73n + 8u + b, whose bank is (9n + 8u + b) mod 32: 9n is
        // distinct mod 8 across n in 0..7 and 8u picks one of four offsets, so
        // the 32 lanes hit 32 DISTINCT banks. A natural even stride (72, or a
        // j-major [j][n] layout) puts eight lanes on one bank and costs an
        // 8-way conflict on the kernel's dominant store.
        constexpr uint W_STRIDE = 73;
        constexpr uint N_GROUPS = K / GROUP;
        constexpr uint W_ROW_U32 = K / 8;
        constexpr uint G_ROW = K / GROUP;

        const uint lid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tg = threadgroup_position_in_grid.x;

        threadgroup float Xs[M_ROWS * A_STRIDE];
        threadgroup float Ws[N_SG][N_PSG * W_STRIDE];
        threadgroup float Os[N_SG][M_ROWS * N_PSG];
        threadgroup float XSum[M_ROWS * N_GROUPS];

        // Zero the seven pad slots that follow the bias slot. Written once:
        // the group loop only ever rewrites j in 0...64.
        if (lid < M_ROWS * 7) {
            Xs[(lid / 7) * A_STRIDE + 65 + (lid % 7)] = 0.0f;
        }
        for (uint t = lane; t < 7 * N_PSG; t += 32) {
            Ws[sg][(t / 7) * W_STRIDE + 65 + (t % 7)] = 0.0f;
        }

        // --- sum_k x_k for every (row, group), ONCE ---------------------
        // Hoisted out of the group loop: it depends only on the activation, so
        // recomputing it per group merely serialised 64 loads onto 8 threads
        // while the other 120 waited at the barrier.
        //
        // The addend is written in stock `load_vector<T, float, 8, 4>`'s exact
        // form -- a float accumulator, and the four-term sum evaluated at the
        // ACTIVATION dtype (`sum += x[i] + x[i+1] + x[i+2] + x[i+3];`),
        // including its two-addends-per-eight-values grouping. That bf16
        // quad-sum is what stock multiplies by the affine bias, so matching it
        // (not exact math) is what keeps greedy argmax on stock's answer.
        for (uint cell = lid; cell < M_ROWS * N_GROUPS; cell += N_SG * 32) {
            const device T* xp = x + (cell / N_GROUPS) * K + (cell % N_GROUPS) * GROUP;
            float s = 0.0f;
            for (uint c = 0; c < GROUP / 8; ++c) {
                const uint i = c * 8;
                s += xp[i + 0] + xp[i + 1] + xp[i + 2] + xp[i + 3];
                s += xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
            }
            XSum[cell] = s;
        }

        const uint n0 = tg * (N_SG * N_PSG);
        const uint sgN0 = n0 + sg * N_PSG;

        // Weight staging assignment. See W_STRIDE: n = lane/4 keeps the eight
        // output columns one per bank class, u = lane%4 (and u+4) covers the
        // group's eight uint32 words.
        const uint wn = lane / 4;
        const uint wu = lane % 4;
        const uint wRow = sgN0 + wn;
        const device uint32_t* wRowPtr = w + wRow * W_ROW_U32;
        threadgroup float* wDst = Ws[sg] + wn * W_STRIDE;

        // A zero-valued simdgroup_matrix is all-zero under either reading of
        // the scalar constructor (diagonal fill or broadcast fill).
        simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);

        threadgroup float* d0 = wDst + wu * 8;
        threadgroup float* d1 = wDst + (wu + 4) * 8;

        // SOFTWARE PIPELINE. Each lane consumes only two uint32 of weight per
        // group, so a naive loop issues 8 bytes of DRAM demand and then blocks
        // on a barrier -- far too few loads in flight to cover DRAM latency,
        // and the kernel measured ~66 GB/s against a ~273 GB/s part. The next
        // group's words and scale are therefore requested BEFORE the staging
        // barrier and consumed only after this group's MMAs, so the fetch has
        // the whole MMA phase to land.
        uint32_t p0 = wRowPtr[wu];
        uint32_t p1 = wRowPtr[wu + 4];
        float sc = float(scales[wRow * G_ROW]);
        float bs = lane < N_PSG ? float(biases[(sgN0 + lane) * G_ROW]) : 0.0f;

        for (uint g = 0; g < N_GROUPS; ++g) {
            // Protect the previous iteration's MMA reads before restaging.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- activations: 512 values, 4 per thread ------------------
            // Staged rather than read straight from device as an f32 view: an
            // 8x8 activation tile taken from device memory is eight rows at a
            // K-float stride, i.e. eight cache lines per tile, and measured
            // SLOWER (5.95 ms vs 5.83 ms) than staging the group once into
            // threadgroup memory where the tile rows are adjacent.
            {
                const uint m = lid / 16;
                const uint j0 = (lid % 16) * 4;
                const device T* xp = x + m * K + g * GROUP + j0;
                threadgroup float* dst = Xs + m * A_STRIDE + j0;
                dst[0] = float(xp[0]);
                dst[1] = float(xp[1]);
                dst[2] = float(xp[2]);
                dst[3] = float(xp[3]);
            }
            if (lid < M_ROWS) {
                Xs[lid * A_STRIDE + 64] = XSum[lid * N_GROUPS + g];
            }

            // --- weights: 8 rows x 8 uint32 words, 2 words per lane -----
            // `sc * float(code)` is EXACT: a bf16 significand (8 bits) times a
            // 4-bit integer needs at most 12 significand bits, so folding the
            // group scale into the staged code adds no rounding stock lacks.
            for (uint b = 0; b < 8; ++b) {
                d0[b] = sc * float((p0 >> (4 * b)) & 0xF);
                d1[b] = sc * float((p1 >> (4 * b)) & 0xF);
            }
            // One lane per output column -- `wn` covers only two columns per
            // quad of lanes, so the bias slot is assigned separately.
            if (lane < N_PSG) {
                Ws[sg][lane * W_STRIDE + 64] = bs;
            }

            // Request group g+1 now; it is consumed after the MMAs below.
            uint32_t n0w = 0;
            uint32_t n1w = 0;
            float nsc = 0.0f;
            float nbs = 0.0f;
            if (g + 1 < N_GROUPS) {
                const uint wBase = ((g + 1) * GROUP) / 8;
                n0w = wRowPtr[wBase + wu];
                n1w = wRowPtr[wBase + wu + 4];
                nsc = float(scales[wRow * G_ROW + g + 1]);
                if (lane < N_PSG) {
                    nbs = float(biases[(sgN0 + lane) * G_ROW + g + 1]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- eight code slices plus the bias slice ------------------
            // Ws is [n][j], so B is read transposed to present [j][n].
            for (uint t = 0; t < 9; ++t) {
                simdgroup_matrix<float, 8, 8> A;
                simdgroup_matrix<float, 8, 8> B;
                simdgroup_load(A, Xs + t * 8, A_STRIDE);
                simdgroup_load(B, Ws[sg] + t * 8, W_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, A, B, acc);
            }

            p0 = n0w;
            p1 = n1w;
            sc = nsc;
            bs = nbs;
        }

        simdgroup_store(acc, Os[sg], N_PSG);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = lid; idx < M_ROWS * N_SG * N_PSG; idx += N_SG * 32) {
            const uint m = idx / (N_SG * N_PSG);
            const uint c = idx % (N_SG * N_PSG);
            out[m * N + n0 + c] = T(Os[c / N_PSG][m * N_PSG + (c % N_PSG)]);
        }
        """

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: source,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    /// Tied-head GEMV, or `nil` when any gate above fails.
    ///
    /// - Parameters:
    ///   - x: activation, `[8, K]` or anything that flattens to it, bf16.
    ///   - w: packed affine codes, `[N, K * bits / 32]` uint32.
    ///   - scales: `[N, K / groupSize]`, same dtype as `x`.
    ///   - biases: `[N, K / groupSize]`, same dtype as `x`.
    /// - Returns: `[8, N]` in `x`'s dtype, or `nil`.
    public static func apply(
        x: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        guard enabled else { return nil }
        guard let biases else { return nil }
        guard groupSize == 64, bits == 4 else { return nil }
        guard x.dtype == .bfloat16, scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        guard w.dtype == .uint32 else { return nil }
        guard w.ndim == 2, scales.ndim == 2, biases.ndim == 2 else { return nil }

        // [8, K] or [8, 1, K] ONLY -- the decode cohort's own shape, eight
        // BATCH rows at one position. A [1, 8, K] eight-token prefill chunk
        // has the same element count and must NOT enter: the gate is meant to
        // claim the cohort's decode head, and prefill stays on stock.
        guard x.ndim == 2 || (x.ndim == 3 && x.dim(1) == 1) else { return nil }
        guard x.dim(0) == mRows else { return nil }
        let k = x.dim(-1)
        guard k > 0, x.size == mRows * k else { return nil }

        let n = w.dim(0)
        guard n >= minOutputWidth, n % colsPerThreadgroup == 0 else { return nil }
        guard k % groupSize == 0, k % 8 == 0 else { return nil }
        guard w.dim(1) == k * bits / 32 else { return nil }
        guard scales.dim(0) == n, biases.dim(0) == n else { return nil }
        guard scales.dim(1) == k / groupSize, biases.dim(1) == k / groupSize else { return nil }

        let threadgroups = n / colsPerThreadgroup
        let outputs = kernel(
            [x.reshaped([mRows, k]), w, scales, biases],
            template: [("T", x.dtype), ("K", k), ("N", n)],
            grid: (threadgroups * threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[mRows, n]],
            outputDTypes: [x.dtype]
        )
        return outputs[0]
    }
}
