// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFast

public enum Gemma4MMAQuantizedGEMV {

    public struct ActivationSums {
        fileprivate let values: MLXArray
    }

    private static let mRows = 8
    private static let colsPerSimdgroup = 8
    private static let simdgroupsPerThreadgroup = 4
    private static let colsPerThreadgroup = colsPerSimdgroup * simdgroupsPerThreadgroup
    private static let threadsPerThreadgroup = simdgroupsPerThreadgroup * 32
    private static let minOutputWidth = 8192

    private static func environmentFlag(_ name: String) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name] else { return true }
        switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
        case "0", "false", "no", "off": return false
        default: return true
        }
    }

    private static let enabled = environmentFlag("DARKBLOOM_GEMMA4_MMA_HEAD")

    private static let version: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MMA_HEAD_VERSION"]
        else { return defaultVersion }
        switch raw.trimmingCharacters(in: .whitespaces) {
        case "1": return 1
        case "2": return 2
        case "3": return 3
        case "4": return 4
        case "5": return 5
        case "6": return 6
        case "7": return 7
        case "8": return 8
        case "9": return 9
        case "10": return 10
        case "11": return 11
        case "12": return 12
        case "13": return 13
        case "14": return 14
        case "15": return 15
        case "16": return 16
        case "26": return 26
        case "27": return 27
        default: return defaultVersion
        }
    }()

    private static let defaultVersion = 27

    public static var consumesActivationSums: Bool {
        guard enabled else { return false }
        switch version {
        case 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 27:
            return true
        default:
            return false
        }
    }

    public static func activationSums(
        produced values: MLXArray, for x: MLXArray
    ) -> ActivationSums? {
        guard consumesActivationSums,
            x.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) == mRows,
            x.dim(1) == 1,
            x.dim(2) == 2816,
            x.size == mRows * 2816,
            values.dtype == .float32,
            values.ndim == 1,
            values.size == mRows * (2816 / 64)
        else { return nil }
        return ActivationSums(values: values)
    }

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

    // MARK: - Version 2 --- bf16 operands on the matrix units

    private static let sourceV2 = """
        constexpr uint M_ROWS = 8;
        constexpr uint GROUP = 64;
        constexpr uint N_SG = 4;
        constexpr uint N_PSG = 8;
        constexpr uint A_STRIDE = 72;
        constexpr uint W_STRIDE = 74;
        constexpr uint N_GROUPS = K / GROUP;
        constexpr uint W_ROW_U32 = K / 8;
        constexpr uint G_ROW = K / GROUP;

        const uint lid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tg = threadgroup_position_in_grid.x;

        threadgroup T Xs[M_ROWS * A_STRIDE];
        threadgroup T Ws[N_SG][N_PSG * W_STRIDE];
        threadgroup float Sd[N_SG][64];
        threadgroup float Bb[N_SG][64];
        threadgroup float Xb[64];
        threadgroup float Os[N_SG][M_ROWS * N_PSG];
        threadgroup float XSum[M_ROWS * N_GROUPS];

        // The three rescale tiles are mostly structural zeros that never
        // change: Sd is diagonal, Bb carries the bias in row 0 only, Xb the
        // activation sum in column 0 only. Written once here; the group loop
        // rewrites exactly 8 live slots in each.
        for (uint i = lid; i < 64; i += N_SG * 32) {
            Xb[i] = 0.0f;
        }
        for (uint i = lane; i < 64; i += 32) {
            Sd[sg][i] = 0.0f;
            Bb[sg][i] = 0.0f;
        }

        // --- sum_k x_k for every (row, group), ONCE ---------------------
        // Identical to version 1, and for the same reason: it depends only on
        // the activation, and it is written in stock
        // `load_vector<T, float, 8, 4>`'s exact form -- a float accumulator
        // with the four-term addend evaluated at the ACTIVATION dtype -- so
        // the affine bias is multiplied by stock's bf16 quad-sum, not by exact
        // math. That is what keeps greedy argmax on stock's answer.
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

        const uint wn = lane / 4;
        const uint wu = lane % 4;
        const uint wRow = sgN0 + wn;
        const device uint32_t* wRowPtr = w + wRow * W_ROW_U32;
        threadgroup T* d0 = Ws[sg] + wn * W_STRIDE + wu * 8;
        threadgroup T* d1 = Ws[sg] + wn * W_STRIDE + (wu + 4) * 8;

        // Software pipeline, as in version 1: each lane demands only two
        // uint32 of weight per group, far too few in flight to cover DRAM
        // latency, so group g+1 is requested before the staging barrier and
        // consumed after this group's MMAs.
        // Version 1 needed the scale of the weight row THIS LANE stages, to
        // fold into the code. Version 2 stages raw codes, so the only scale it
        // wants is the one that rescales output COLUMN `lane` -- read by the
        // same eight lanes that fill the diagonal and the bias row.
        uint32_t p0 = wRowPtr[wu];
        uint32_t p1 = wRowPtr[wu + 4];
        float sd = lane < N_PSG ? float(scales[(sgN0 + lane) * G_ROW]) : 0.0f;
        float bs = lane < N_PSG ? float(biases[(sgN0 + lane) * G_ROW]) : 0.0f;

        simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint g = 0; g < N_GROUPS; ++g) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- activations: staged AT THEIR OWN DTYPE ------------------
            {
                const uint m = lid / 16;
                const uint j0 = (lid % 16) * 4;
                const device T* xp = x + m * K + g * GROUP + j0;
                threadgroup T* dst = Xs + m * A_STRIDE + j0;
                dst[0] = xp[0];
                dst[1] = xp[1];
                dst[2] = xp[2];
                dst[3] = xp[3];
            }

            // --- weights: RAW codes, exact in bf16 -----------------------
            for (uint b = 0; b < 8; ++b) {
                d0[b] = T(float((p0 >> (4 * b)) & 0xF));
                d1[b] = T(float((p1 >> (4 * b)) & 0xF));
            }

            // --- the three rescale tiles' live slots ---------------------
            if (lane < N_PSG) {
                Sd[sg][lane * 8 + lane] = sd;
                Bb[sg][lane] = bs;
            }
            if (lid < M_ROWS) {
                Xb[lid * 8] = XSum[lid * N_GROUPS + g];
            }

            uint32_t n0w = 0;
            uint32_t n1w = 0;
            float nsd = 0.0f;
            float nbs = 0.0f;
            if (g + 1 < N_GROUPS) {
                const uint wBase = ((g + 1) * GROUP) / 8;
                n0w = wRowPtr[wBase + wu];
                n1w = wRowPtr[wBase + wu + 4];
                if (lane < N_PSG) {
                    nsd = float(scales[(sgN0 + lane) * G_ROW + g + 1]);
                    nbs = float(biases[(sgN0 + lane) * G_ROW + g + 1]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- the group's exact dot product, on the matrix units ------
            simdgroup_matrix<float, 8, 8> accg = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint t = 0; t < 8; ++t) {
                simdgroup_matrix<T, 8, 8> A;
                simdgroup_matrix<T, 8, 8> B;
                simdgroup_load(A, Xs + t * 8, A_STRIDE);
                simdgroup_load(B, Ws[sg] + t * 8, W_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(accg, A, B, accg);
            }

            // --- reapply the group scale, then the affine bias -----------
            // acc += accg * diag(s_g)   -- scales column n by s_g[n]
            // acc += xsum_col * bias_row -- the rank-1 affine bias term
            {
                simdgroup_matrix<float, 8, 8> S;
                simdgroup_matrix<float, 8, 8> XBm;
                simdgroup_matrix<float, 8, 8> BBm;
                simdgroup_load(S, Sd[sg], 8);
                simdgroup_multiply_accumulate(acc, accg, S, acc);
                simdgroup_load(XBm, Xb, 8);
                simdgroup_load(BBm, Bb[sg], 8);
                simdgroup_multiply_accumulate(acc, XBm, BBm, acc);
            }

            p0 = n0w;
            p1 = n1w;
            sd = nsd;
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

    private static let kernelV2: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v2",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: sourceV2,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 3 --- deeper weight pipeline, batched affine bias

    private static let sourceV3 = """
        constexpr uint M_ROWS = 8;
        constexpr uint GROUP = 64;
        constexpr uint N_SG = 4;
        constexpr uint N_PSG = 8;
        // Activation staged TRANSPOSED as [j][m], so X_STRIDE spans the eight
        // batch rows (+1 to break the power-of-two bank stride).
        constexpr uint X_STRIDE = 9;
        constexpr uint W_STRIDE = 74;
        constexpr uint N_GROUPS = K / GROUP;
        constexpr uint W_ROW_U32 = K / 8;
        constexpr uint G_ROW = K / GROUP;

        const uint lid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tg = threadgroup_position_in_grid.x;

        threadgroup T Xs[GROUP * X_STRIDE];
        threadgroup T Ws[N_SG][N_PSG * W_STRIDE];
        // Sd is the per-group diagonal during the loop and the result tile
        // afterwards; the two lifetimes do not overlap.
        threadgroup float Sd[N_SG][64];
        threadgroup float XbB[64];
        threadgroup float BbB[N_SG][64];
        threadgroup float XSum[M_ROWS * N_GROUPS];

        for (uint i = lane; i < 64; i += 32) {
            Sd[sg][i] = 0.0f;
        }

        // --- sum_k x_k for every (row, group), ONCE ---------------------
        // Unchanged from versions 1 and 2, and for the same reason: written in
        // stock `load_vector<T, float, 8, 4>`'s exact form -- a float
        // accumulator with the four-term addend evaluated at the ACTIVATION
        // dtype -- so the affine bias is multiplied by stock's bf16 quad-sum
        // rather than by exact math.
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

        const uint wn = lane / 4;
        const uint wu = lane % 4;
        const uint wRow = sgN0 + wn;
        const device uint32_t* wRowPtr = w + wRow * W_ROW_U32;
        threadgroup T* d0 = Ws[sg] + wn * W_STRIDE + wu * 8;
        threadgroup T* d1 = Ws[sg] + wn * W_STRIDE + (wu + 4) * 8;
        const device T* sRow = scales + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
        const device T* bRow = biases + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;

        // Two groups of weight in flight at all times. Stage A is consumed by
        // this iteration, stage B by the next, and the loop issues the loads
        // for the one after that.
        uint32_t p0a = wRowPtr[wu];
        uint32_t p1a = wRowPtr[wu + 4];
        float sda = lane < N_PSG ? float(sRow[0]) : 0.0f;
        float bsa = lane < N_PSG ? float(bRow[0]) : 0.0f;
        uint32_t p0b = 0;
        uint32_t p1b = 0;
        float sdb = 0.0f;
        float bsb = 0.0f;
        if (N_GROUPS > 1) {
            p0b = wRowPtr[GROUP / 8 + wu];
            p1b = wRowPtr[GROUP / 8 + wu + 4];
            sdb = lane < N_PSG ? float(sRow[1]) : 0.0f;
            bsb = lane < N_PSG ? float(bRow[1]) : 0.0f;
        }

        simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint g = 0; g < N_GROUPS; ++g) {
            const uint gg = g % 8;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Activation staged TRANSPOSED: Xs[j][m]. A thread takes ONE k and
            // four ROWS, so its four threadgroup writes are CONSECUTIVE in the
            // staged layout; the alternative (four consecutive k of one row)
            // makes every write a stride-X_STRIDE scatter. The device side
            // gives up coalescing to pay for that, but the whole activation is
            // 45 KB and every threadgroup reads the same bytes, so those loads
            // are cache hits rather than DRAM traffic.
            {
                const uint j = lid / 2;
                const uint m0 = (lid % 2) * 4;
                const device T* xp = x + m0 * K + g * GROUP + j;
                threadgroup T* dst = Xs + j * X_STRIDE + m0;
                dst[0] = xp[0 * K];
                dst[1] = xp[1 * K];
                dst[2] = xp[2 * K];
                dst[3] = xp[3 * K];
            }

            for (uint b = 0; b < 8; ++b) {
                d0[b] = T(float((p0a >> (4 * b)) & 0xF));
                d1[b] = T(float((p1a >> (4 * b)) & 0xF));
            }

            const bool lastGroup = (g + 1 == N_GROUPS);
            if (lane < N_PSG) {
                Sd[sg][lane * 8 + lane] = sda;
                // Transposed bias pair: BbT[n][gg].
                BbB[sg][lane * 8 + gg] = bsa;
                // Partial final block: clear only the K slots ABOVE this
                // group's, with the same lane that wrote the live one, so no
                // two threads ever touch the same address in this phase.
                if (lastGroup) {
                    for (uint j = gg + 1; j < 8; ++j) {
                        BbB[sg][lane * 8 + j] = 0.0f;
                    }
                }
            }
            if (lid < M_ROWS) {
                // Transposed bias pair: XbT[gg][m].
                XbB[gg * 8 + lid] = XSum[lid * N_GROUPS + g];
                if (lastGroup) {
                    for (uint j = gg + 1; j < 8; ++j) {
                        XbB[j * 8 + lid] = 0.0f;
                    }
                }
            }

            // Request group g+2 while g+1 is still outstanding.
            uint32_t p0c = 0;
            uint32_t p1c = 0;
            float sdc = 0.0f;
            float bsc = 0.0f;
            if (g + 2 < N_GROUPS) {
                const uint wBase = ((g + 2) * GROUP) / 8;
                p0c = wRowPtr[wBase + wu];
                p1c = wRowPtr[wBase + wu + 4];
                if (lane < N_PSG) {
                    sdc = float(sRow[g + 2]);
                    bsc = float(bRow[g + 2]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // accgT[n][m] = sum_j W[n][j] * X^T[j][m]. BOTH tiles now load
            // untransposed, in the layout each was stored in.
            simdgroup_matrix<float, 8, 8> accg = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint t = 0; t < 8; ++t) {
                simdgroup_matrix<T, 8, 8> A;
                simdgroup_matrix<T, 8, 8> B;
                simdgroup_load(A, Ws[sg] + t * 8, W_STRIDE);
                simdgroup_load(B, Xs + t * 8 * X_STRIDE, X_STRIDE);
                simdgroup_multiply_accumulate(accg, A, B, accg);
            }

            // Per-column on `out` is per-ROW on `outT`, so the diagonal is the
            // LEFT operand here.
            {
                simdgroup_matrix<float, 8, 8> S;
                simdgroup_load(S, Sd[sg], 8);
                simdgroup_multiply_accumulate(acc, S, accg, acc);
            }

            // One rank-8 bias MMA per eight groups instead of one rank-1 MMA
            // per group.
            // Transposed too: outT[n][m] += sum_gg BbT[n][gg] * XbT[gg][m].
            if (gg == 7 || lastGroup) {
                simdgroup_matrix<float, 8, 8> XBm;
                simdgroup_matrix<float, 8, 8> BBm;
                simdgroup_load(BBm, BbB[sg], 8);
                simdgroup_load(XBm, XbB, 8);
                simdgroup_multiply_accumulate(acc, BBm, XBm, acc);
            }

            p0a = p0b;
            p1a = p1b;
            sda = sdb;
            bsa = bsb;
            p0b = p0c;
            p1b = p1c;
            sdb = sdc;
            bsb = bsc;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc, Sd[sg], N_PSG);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The staged tile is outT[n][m], so the indices exchange here.
        for (uint idx = lid; idx < M_ROWS * N_SG * N_PSG; idx += N_SG * 32) {
            const uint m = idx / (N_SG * N_PSG);
            const uint c = idx % (N_SG * N_PSG);
            out[m * N + n0 + c] = T(Sd[c / N_PSG][(c % N_PSG) * N_PSG + m]);
        }
        """

    private static let kernelV3: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v3",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: sourceV3,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 4 --- direct weight-fragment construction

    private static let sourceV4: String = {
        var result = sourceV3

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce("constexpr uint W_STRIDE = 74;\n", with: "")
        replaceOnce("threadgroup T Ws[N_SG][N_PSG * W_STRIDE];\n", with: "")
        replaceOnce(
            """
            threadgroup T* d0 = Ws[sg] + wn * W_STRIDE + wu * 8;
            threadgroup T* d1 = Ws[sg] + wn * W_STRIDE + (wu + 4) * 8;

            """,
            with: ""
        )
        replaceOnce(
            """
            const uint wn = lane / 4;
            const uint wu = lane % 4;
            const uint wRow = sgN0 + wn;
            const device uint32_t* wRowPtr = w + wRow * W_ROW_U32;
            const device T* sRow = scales + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            const device T* bRow = biases + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            """,
            with: """
            const device T* sRow = scales + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            const device T* bRow = biases + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            """
        )
        replaceOnce(
            """
            // Two groups of weight in flight at all times. Stage A is consumed by
            // this iteration, stage B by the next, and the loop issues the loads
            // for the one after that.
            uint32_t p0a = wRowPtr[wu];
            uint32_t p1a = wRowPtr[wu + 4];
            float sda = lane < N_PSG ? float(sRow[0]) : 0.0f;
            float bsa = lane < N_PSG ? float(bRow[0]) : 0.0f;
            uint32_t p0b = 0;
            uint32_t p1b = 0;
            float sdb = 0.0f;
            float bsb = 0.0f;
            if (N_GROUPS > 1) {
                p0b = wRowPtr[GROUP / 8 + wu];
                p1b = wRowPtr[GROUP / 8 + wu + 4];
                sdb = lane < N_PSG ? float(sRow[1]) : 0.0f;
                bsb = lane < N_PSG ? float(bRow[1]) : 0.0f;
            }
            """,
            with: """
            // Scale and bias metadata stay two groups ahead of use.
            float sda = lane < N_PSG ? float(sRow[0]) : 0.0f;
            float bsa = lane < N_PSG ? float(bRow[0]) : 0.0f;
            float sdb = 0.0f;
            float bsb = 0.0f;
            if (N_GROUPS > 1) {
                sdb = lane < N_PSG ? float(sRow[1]) : 0.0f;
                bsb = lane < N_PSG ? float(bRow[1]) : 0.0f;
            }
            """
        )
        replaceOnce(
            """
                for (uint b = 0; b < 8; ++b) {
                    d0[b] = T(float((p0a >> (4 * b)) & 0xF));
                    d1[b] = T(float((p1a >> (4 * b)) & 0xF));
                }

            """,
            with: ""
        )
        replaceOnce(
            """
                // Request group g+2 while g+1 is still outstanding.
                uint32_t p0c = 0;
                uint32_t p1c = 0;
                float sdc = 0.0f;
                float bsc = 0.0f;
                if (g + 2 < N_GROUPS) {
                    const uint wBase = ((g + 2) * GROUP) / 8;
                    p0c = wRowPtr[wBase + wu];
                    p1c = wRowPtr[wBase + wu + 4];
                    if (lane < N_PSG) {
                        sdc = float(sRow[g + 2]);
                        bsc = float(bRow[g + 2]);
                    }
                }
            """,
            with: """
                // Request group g+2 metadata while g+1 is outstanding.
                float sdc = 0.0f;
                float bsc = 0.0f;
                if (g + 2 < N_GROUPS && lane < N_PSG) {
                    sdc = float(sRow[g + 2]);
                    bsc = float(bRow[g + 2]);
                }
            """
        )
        replaceOnce(
            """
                p0a = p0b;
                p1a = p1b;
                sda = sdb;
                bsa = bsb;
                p0b = p0c;
                p1b = p1c;
                sdb = sdc;
                bsb = bsc;
            """,
            with: """
                sda = sdb;
                bsa = bsb;
                sdb = sdc;
                bsb = bsc;
            """
        )
        replaceOnce(
            """
                // accgT[n][m] = sum_j W[n][j] * X^T[j][m]. BOTH tiles now load
                // untransposed, in the layout each was stored in.
                simdgroup_matrix<float, 8, 8> accg = simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint t = 0; t < 8; ++t) {
                    simdgroup_matrix<T, 8, 8> A;
                    simdgroup_matrix<T, 8, 8> B;
                    simdgroup_load(A, Ws[sg] + t * 8, W_STRIDE);
                    simdgroup_load(B, Xs + t * 8 * X_STRIDE, X_STRIDE);
                    simdgroup_multiply_accumulate(accg, A, B, accg);
                }
            """,
            with: """
                // accgT[n][m] = sum_j W[n][j] * X^T[j][m]. Build the
                // weight fragment directly: an 8x8 matrix distributes two
                // adjacent row-major elements to each lane. Four lanes own
                // one output row and therefore issue the same packed-word
                // address; the device coalescer merges those reads while each
                // lane extracts its own nibble pair.
                const uint fragmentRow =
                    ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
                const uint fragmentCol =
                    ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
                const device uint32_t* fragmentWRow =
                    w + (sgN0 + fragmentRow) * W_ROW_U32;
                simdgroup_matrix<float, 8, 8> accg =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint t = 0; t < 8; ++t) {
                    simdgroup_matrix<T, 8, 8> A;
                    simdgroup_matrix<T, 8, 8> B;
                    const uint packed = fragmentWRow[g * (GROUP / 8) + t];
                    A.thread_elements()[0] =
                        T(float((packed >> (4 * fragmentCol)) & 0xFu));
                    A.thread_elements()[1] =
                        T(float((packed >> (4 * (fragmentCol + 1))) & 0xFu));
                    simdgroup_load(B, Xs + t * 8 * X_STRIDE, X_STRIDE);
                    simdgroup_multiply_accumulate(accg, A, B, accg);
                }
            """
        )

        return result
    }()

    private static let kernelV4: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v4",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: sourceV4,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 5 --- one activation-sum prepass

    private static let xSumSource = """
        constexpr uint M_ROWS = 8;
        constexpr uint GROUP = 64;
        constexpr uint N_GROUPS = K / GROUP;
        const uint cell = thread_position_in_grid.x;
        if (cell >= M_ROWS * N_GROUPS) return;

        const device T* xp =
            x + (cell / N_GROUPS) * K + (cell % N_GROUPS) * GROUP;
        float s = 0.0f;
        #pragma unroll
        for (uint c = 0; c < GROUP / 8; ++c) {
            const uint i = c * 8;
            s += xp[i + 0] + xp[i + 1] + xp[i + 2] + xp[i + 3];
            s += xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
        }
        xSums[cell] = s;
        """

    private static let xSumKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_xsum_m8_v27_unroll",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: xSumSource,
        ensureRowContiguous: true
    )

    private static let xSumRowsSource: String = {
        let marker = "constexpr uint M_ROWS = 8;"
        let count = xSumSource.components(separatedBy: marker).count
        precondition(count == 2, "xSumRowsSource replacement count \(count): \(marker)")
        return xSumSource.replacingOccurrences(
            of: marker, with: "constexpr uint M_ROWS = 8 * RT;")
    }()

    private static let xSumRowsKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_xsum_m8_rows_v1",
        inputNames: ["x"],
        outputNames: ["xSums"],
        source: xSumRowsSource,
        ensureRowContiguous: true
    )

    private static let sourceV5: String = {
        var result = sourceV4

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce("threadgroup float XSum[M_ROWS * N_GROUPS];\n", with: "")
        replaceOnce(
            """
            // --- sum_k x_k for every (row, group), ONCE ---------------------
            // Unchanged from versions 1 and 2, and for the same reason: written in
            // stock `load_vector<T, float, 8, 4>`'s exact form -- a float
            // accumulator with the four-term addend evaluated at the ACTIVATION
            // dtype -- so the affine bias is multiplied by stock's bf16 quad-sum
            // rather than by exact math.
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

            """,
            with: ""
        )
        replaceOnce(
            "XbB[gg * 8 + lid] = XSum[lid * N_GROUPS + g];",
            with: "XbB[gg * 8 + lid] = xSums[lid * N_GROUPS + g];"
        )
        replaceOnce(
            """
                simdgroup_matrix<float, 8, 8> accg =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint t = 0; t < 8; ++t) {
            """,
            with: """
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow + g * (GROUP / 8));
                const uint4 packedLo = packedGroup[0];
                const uint4 packedHi = packedGroup[1];
                simdgroup_matrix<float, 8, 8> accg =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint t = 0; t < 8; ++t) {
            """
        )
        replaceOnce(
            "const uint packed = fragmentWRow[g * (GROUP / 8) + t];",
            with: "const uint packed = t < 4 ? packedLo[t] : packedHi[t - 4];"
        )

        return result
    }()

    private static let kernelV5: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v5",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV5,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 6 --- direct result-fragment rescale

    private static let sourceV6: String = {
        var result = sourceV5

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            for (uint i = lane; i < 64; i += 32) {
                Sd[sg][i] = 0.0f;
            }

            """,
            with: ""
        )
        replaceOnce(
            "        Sd[sg][lane * 8 + lane] = sda;\n",
            with: ""
        )
        replaceOnce(
            """
                // Per-column on `out` is per-ROW on `outT`, so the diagonal is the
                // LEFT operand here.
                {
                    simdgroup_matrix<float, 8, 8> S;
                    simdgroup_load(S, Sd[sg], 8);
                    simdgroup_multiply_accumulate(acc, S, accg, acc);
                }
            """,
            with: """
            // Per-column on `out` is per-ROW on `outT`. Each lane's two
            // fragment elements share `fragmentRow`; lanes 0...7 already hold
            // the corresponding scales in `sda`.
            const float rowScale = simd_shuffle(sda, ushort(fragmentRow));
            acc.thread_elements()[0] = metal::fma(
                rowScale, accg.thread_elements()[0], acc.thread_elements()[0]);
            acc.thread_elements()[1] = metal::fma(
                rowScale, accg.thread_elements()[1], acc.thread_elements()[1]);
            """
        )

        return result
    }()

    private static let kernelV6: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v6",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV6,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 7 --- direct result-fragment output

    private static let sourceV7: String = {
        var result = sourceV6

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce("threadgroup float Sd[N_SG][64];\n", with: "")
        replaceOnce(
            """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc, Sd[sg], N_PSG);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The staged tile is outT[n][m], so the indices exchange here.
            for (uint idx = lid; idx < M_ROWS * N_SG * N_PSG; idx += N_SG * 32) {
                const uint m = idx / (N_SG * N_PSG);
                const uint c = idx % (N_SG * N_PSG);
                out[m * N + n0 + c] = T(Sd[c / N_PSG][(c % N_PSG) * N_PSG + m]);
            }
            """,
            with: """
            const uint outputRow =
                ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
            const uint outputPair =
                ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
            const uint outputN = sgN0 + outputRow;
            out[outputPair * N + outputN] = T(acc.thread_elements()[0]);
            out[(outputPair + 1) * N + outputN] = T(acc.thread_elements()[1]);
            """
        )

        return result
    }()

    private static let kernelV7: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v7",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV7,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 8 --- early aligned weight loads

    private static let sourceV8: String = {
        var result = sourceV7

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
                const uint fragmentRow =
                    ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
                const uint fragmentCol =
                    ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
                const device uint32_t* fragmentWRow =
                    w + (sgN0 + fragmentRow) * W_ROW_U32;
            """,
            with: ""
        )
        replaceOnce(
            """
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint g = 0; g < N_GROUPS; ++g) {
                const uint gg = g % 8;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            """,
            with: """
            const uint fragmentRow =
                ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
            const uint fragmentCol =
                ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
            const device uint32_t* fragmentWRow =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint g = 0; g < N_GROUPS; ++g) {
                const uint gg = g % 8;
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow + g * (GROUP / 8));
                const uint4 packedLo = packedGroup[0];
                const uint4 packedHi = packedGroup[1];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        )
        replaceOnce(
            """
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow + g * (GROUP / 8));
                const uint4 packedLo = packedGroup[0];
                const uint4 packedHi = packedGroup[1];
                simdgroup_matrix<float, 8, 8> accg =
            """,
            with: """
                simdgroup_matrix<float, 8, 8> accg =
            """
        )
        replaceOnce(
            """
            const uint outputRow =
                ((lane & 6u) >> 1u) + ((lane & 16u) >> 2u);
            const uint outputPair =
                ((lane & 1u) << 1u) + ((lane & 8u) >> 1u);
            const uint outputN = sgN0 + outputRow;
            out[outputPair * N + outputN] = T(acc.thread_elements()[0]);
            out[(outputPair + 1) * N + outputN] = T(acc.thread_elements()[1]);
            """,
            with: """
            const uint outputN = sgN0 + fragmentRow;
            out[fragmentCol * N + outputN] = T(acc.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN] = T(acc.thread_elements()[1]);
            """
        )

        return result
    }()

    private static let kernelV8: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v8",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV8,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 9 --- early activation cache loads

    private static let sourceV9: String = {
        var result = sourceV8

        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
                const uint4 packedHi = packedGroup[1];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            """,
            with: """
                const uint4 packedHi = packedGroup[1];
                const uint activationJ = lid / 2;
                const uint activationM0 = (lid % 2) * 4;
                const device T* activationPtr =
                    x + activationM0 * K + g * GROUP + activationJ;
                const T activation0 = activationPtr[0 * K];
                const T activation1 = activationPtr[1 * K];
                const T activation2 = activationPtr[2 * K];
                const T activation3 = activationPtr[3 * K];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        )
        replaceOnce(
            """
                {
                    const uint j = lid / 2;
                    const uint m0 = (lid % 2) * 4;
                    const device T* xp = x + m0 * K + g * GROUP + j;
                    threadgroup T* dst = Xs + j * X_STRIDE + m0;
                    dst[0] = xp[0 * K];
                    dst[1] = xp[1 * K];
                    dst[2] = xp[2 * K];
                    dst[3] = xp[3 * K];
                }
            """,
            with: """
                {
                    threadgroup T* dst =
                        Xs + activationJ * X_STRIDE + activationM0;
                    dst[0] = activation0;
                    dst[1] = activation1;
                    dst[2] = activation2;
                    dst[3] = activation3;
                }
            """
        )

        return result
    }()

    private static let kernelV9: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v9",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV9,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 10 --- direct activation and affine fragments

    private static let sourceV10: String = {
        var result = sourceV9

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV10 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce("constexpr uint X_STRIDE = 9;\n", with: "")
        replaceOnce("threadgroup T Xs[GROUP * X_STRIDE];\n", with: "")
        replaceOnce("threadgroup float XbB[64];\n", with: "")
        replaceOnce("threadgroup float BbB[N_SG][64];\n", with: "")
        replaceOnce(
            """
                const uint activationJ = lid / 2;
                const uint activationM0 = (lid % 2) * 4;
                const device T* activationPtr =
                    x + activationM0 * K + g * GROUP + activationJ;
                const T activation0 = activationPtr[0 * K];
                const T activation1 = activationPtr[1 * K];
                const T activation2 = activationPtr[2 * K];
                const T activation3 = activationPtr[3 * K];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            """,
            with: ""
        )
        replaceOnce(
            """
                {
                    threadgroup T* dst =
                        Xs + activationJ * X_STRIDE + activationM0;
                    dst[0] = activation0;
                    dst[1] = activation1;
                    dst[2] = activation2;
                    dst[3] = activation3;
                }
            """,
            with: ""
        )
        replaceOnce(
            """
                if (lane < N_PSG) {
                    // Transposed bias pair: BbT[n][gg].
                    BbB[sg][lane * 8 + gg] = bsa;
                    // Partial final block: clear only the K slots ABOVE this
                    // group's, with the same lane that wrote the live one, so no
                    // two threads ever touch the same address in this phase.
                    if (lastGroup) {
                        for (uint j = gg + 1; j < 8; ++j) {
                            BbB[sg][lane * 8 + j] = 0.0f;
                        }
                    }
                }
                if (lid < M_ROWS) {
                    // Transposed bias pair: XbT[gg][m].
                    XbB[gg * 8 + lid] = xSums[lid * N_GROUPS + g];
                    if (lastGroup) {
                        for (uint j = gg + 1; j < 8; ++j) {
                            XbB[j * 8 + lid] = 0.0f;
                        }
                    }
                }

            """,
            with: ""
        )
        replaceOnce("    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n", with: "")
        replaceOnce(
            """
                    simdgroup_load(B, Xs + t * 8 * X_STRIDE, X_STRIDE);
                    simdgroup_multiply_accumulate(accg, A, B, accg);
            """,
            with: """
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] = x[fragmentCol * K + activationK];
                    B.thread_elements()[1] =
                        x[(fragmentCol + 1) * K + activationK];
                    simdgroup_multiply_accumulate(accg, A, B, accg);
            """
        )
        replaceOnce(
            """
                if (gg == 7 || lastGroup) {
                    simdgroup_matrix<float, 8, 8> XBm;
                    simdgroup_matrix<float, 8, 8> BBm;
                    simdgroup_load(BBm, BbB[sg], 8);
                    simdgroup_load(XBm, XbB, 8);
                    simdgroup_multiply_accumulate(acc, BBm, XBm, acc);
                }
            """,
            with: """
                if (gg == 7 || lastGroup) {
                    const uint biasBlock = g - gg;
                    const uint biasCol0 = biasBlock + fragmentCol;
                    const uint biasCol1 = biasCol0 + 1;
                    const uint biasRow = biasBlock + fragmentRow;
                    const device T* fragmentBRow =
                        biases + (sgN0 + fragmentRow) * G_ROW;
                    simdgroup_matrix<float, 8, 8> BBm;
                    BBm.thread_elements()[0] = biasCol0 < N_GROUPS
                        ? float(fragmentBRow[biasCol0]) : 0.0f;
                    BBm.thread_elements()[1] = biasCol1 < N_GROUPS
                        ? float(fragmentBRow[biasCol1]) : 0.0f;
                    simdgroup_matrix<float, 8, 8> XBm;
                    XBm.thread_elements()[0] = biasRow < N_GROUPS
                        ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                    XBm.thread_elements()[1] = biasRow < N_GROUPS
                        ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
                    simdgroup_multiply_accumulate(acc, BBm, XBm, acc);
                }
            """
        )

        return result
    }()

    private static let kernelV10: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v10",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV10,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 11 --- just-in-time row-scale loads

    private static let sourceV11: String = {
        var result = sourceV10

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV11 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            const device T* bRow = biases + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            """,
            with: ""
        )
        replaceOnce(
            """
            // Scale and bias metadata stay two groups ahead of use.
            float sda = lane < N_PSG ? float(sRow[0]) : 0.0f;
            float bsa = lane < N_PSG ? float(bRow[0]) : 0.0f;
            float sdb = 0.0f;
            float bsb = 0.0f;
            if (N_GROUPS > 1) {
                sdb = lane < N_PSG ? float(sRow[1]) : 0.0f;
                bsb = lane < N_PSG ? float(bRow[1]) : 0.0f;
            }
            """,
            with: ""
        )
        replaceOnce(
            """
                const uint4 packedHi = packedGroup[1];
            """,
            with: """
                const uint4 packedHi = packedGroup[1];
                const float sda =
                    lane < N_PSG ? float(sRow[g]) : 0.0f;
            """
        )
        replaceOnce(
            """
                // Request group g+2 metadata while g+1 is outstanding.
                float sdc = 0.0f;
                float bsc = 0.0f;
                if (g + 2 < N_GROUPS && lane < N_PSG) {
                    sdc = float(sRow[g + 2]);
                    bsc = float(bRow[g + 2]);
                }

            """,
            with: ""
        )
        replaceOnce(
            """
                sda = sdb;
                bsa = bsb;
                sdb = sdc;
                bsb = bsc;
            """,
            with: ""
        )

        return result
    }()

    private static let kernelV11: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v11",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV11,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 12 --- direct fragment-row scales

    private static let sourceV12: String = {
        var result = sourceV11

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV12 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            const device T* sRow = scales + (sgN0 + min(lane, N_PSG - 1)) * G_ROW;
            """,
            with: ""
        )
        replaceOnce(
            """
            const device uint32_t* fragmentWRow =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
            const device uint32_t* fragmentWRow =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device T* fragmentSRow =
                scales + (sgN0 + fragmentRow) * G_ROW;
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);
            """
        )
        replaceOnce(
            """
                const float sda =
                    lane < N_PSG ? float(sRow[g]) : 0.0f;
            """,
            with: """
                const float rowScale = float(fragmentSRow[g]);
            """
        )
        replaceOnce(
            """
            const float rowScale = simd_shuffle(sda, ushort(fragmentRow));
            """,
            with: ""
        )

        return result
    }()

    private static let kernelV12: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v12",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV12,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 13 --- exact eight-group block loop

    private static let sourceV13: String = {
        var result = sourceV12

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV13 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            for (uint g = 0; g < N_GROUPS; ++g) {
                const uint gg = g % 8;
            """,
            with: """
            for (uint biasBlock = 0; biasBlock < N_GROUPS; biasBlock += 8) {
                const uint blockGroups = min(8u, N_GROUPS - biasBlock);
                for (uint gg = 0; gg < blockGroups; ++gg) {
                    const uint g = biasBlock + gg;
            """
        )
        replaceOnce("    const bool lastGroup = (g + 1 == N_GROUPS);\n", with: "")
        replaceOnce("        const uint biasBlock = g - gg;\n", with: "")
        replaceOnce(
            """
                if (gg == 7 || lastGroup) {
            """,
            with: """
                }
                {
            """
        )

        return result
    }()

    private static let kernelV13: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v13",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV13,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 14 --- full affine-block fast path

    private static let sourceV14: String = {
        var result = sourceV13

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV14 replacement count \(count)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
                    simdgroup_matrix<float, 8, 8> BBm;
                    BBm.thread_elements()[0] = biasCol0 < N_GROUPS
                        ? float(fragmentBRow[biasCol0]) : 0.0f;
                    BBm.thread_elements()[1] = biasCol1 < N_GROUPS
                        ? float(fragmentBRow[biasCol1]) : 0.0f;
                    simdgroup_matrix<float, 8, 8> XBm;
                    XBm.thread_elements()[0] = biasRow < N_GROUPS
                        ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                    XBm.thread_elements()[1] = biasRow < N_GROUPS
                        ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
                    simdgroup_multiply_accumulate(acc, BBm, XBm, acc);
            """,
            with: """
                    simdgroup_matrix<float, 8, 8> BBm;
                    simdgroup_matrix<float, 8, 8> XBm;
                    if (biasBlock + 8 <= N_GROUPS) {
                        BBm.thread_elements()[0] = float(fragmentBRow[biasCol0]);
                        BBm.thread_elements()[1] = float(fragmentBRow[biasCol1]);
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
                    } else {
                        BBm.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow[biasCol0]) : 0.0f;
                        BBm.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow[biasCol1]) : 0.0f;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
                    }
                    simdgroup_multiply_accumulate(acc, BBm, XBm, acc);
            """
        )
        return result
    }()

    private static let kernelV14: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v14",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV14,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 15 --- two output tiles per SIMD group

    private static let sourceV15: String = {
        var result = sourceV14

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV15 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            const uint n0 = tg * (N_SG * N_PSG);
            const uint sgN0 = n0 + sg * N_PSG;
            """,
            with: """
            const uint n0 = tg * (N_SG * N_PSG * 2);
            const uint sgN0 = n0 + sg * N_PSG * 2;
            """
        )
        replaceOnce(
            """
            const device uint32_t* fragmentWRow =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device T* fragmentSRow =
                scales + (sgN0 + fragmentRow) * G_ROW;
            simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
            const device uint32_t* fragmentWRow0 =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow1 =
                w + (sgN0 + N_PSG + fragmentRow) * W_ROW_U32;
            const device T* fragmentSRow0 =
                scales + (sgN0 + fragmentRow) * G_ROW;
            const device T* fragmentSRow1 =
                scales + (sgN0 + N_PSG + fragmentRow) * G_ROW;
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            """
        )
        replaceOnce(
            """
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow + g * (GROUP / 8));
                const uint4 packedLo = packedGroup[0];
                const uint4 packedHi = packedGroup[1];
            """,
            with: """
                const device uint4* packedGroup0 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow0 + g * (GROUP / 8));
                const device uint4* packedGroup1 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow1 + g * (GROUP / 8));
                const uint4 packedLo0 = packedGroup0[0];
                const uint4 packedHi0 = packedGroup0[1];
                const uint4 packedLo1 = packedGroup1[0];
                const uint4 packedHi1 = packedGroup1[1];
            """
        )
        replaceOnce(
            "const float rowScale = float(fragmentSRow[g]);",
            with: """
                const float rowScale0 = float(fragmentSRow0[g]);
                const float rowScale1 = float(fragmentSRow1[g]);
            """
        )
        replaceOnce(
            """
                simdgroup_matrix<float, 8, 8> accg =
                    simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
                simdgroup_matrix<float, 8, 8> accg0 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg1 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
            """
        )
        replaceOnce(
            """
                    simdgroup_matrix<T, 8, 8> A;
                    simdgroup_matrix<T, 8, 8> B;
                    const uint packed = t < 4 ? packedLo[t] : packedHi[t - 4];
                    A.thread_elements()[0] =
                        T(float((packed >> (4 * fragmentCol)) & 0xFu));
                    A.thread_elements()[1] =
                        T(float((packed >> (4 * (fragmentCol + 1))) & 0xFu));
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] = x[fragmentCol * K + activationK];
                    B.thread_elements()[1] =
                        x[(fragmentCol + 1) * K + activationK];
                    simdgroup_multiply_accumulate(accg, A, B, accg);
            """,
            with: """
                    simdgroup_matrix<T, 8, 8> A0;
                    simdgroup_matrix<T, 8, 8> A1;
                    simdgroup_matrix<T, 8, 8> B;
                    const uint packed0 = t < 4 ? packedLo0[t] : packedHi0[t - 4];
                    const uint packed1 = t < 4 ? packedLo1[t] : packedHi1[t - 4];
                    A0.thread_elements()[0] =
                        T(float((packed0 >> (4 * fragmentCol)) & 0xFu));
                    A0.thread_elements()[1] =
                        T(float((packed0 >> (4 * (fragmentCol + 1))) & 0xFu));
                    A1.thread_elements()[0] =
                        T(float((packed1 >> (4 * fragmentCol)) & 0xFu));
                    A1.thread_elements()[1] =
                        T(float((packed1 >> (4 * (fragmentCol + 1))) & 0xFu));
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] = x[fragmentCol * K + activationK];
                    B.thread_elements()[1] =
                        x[(fragmentCol + 1) * K + activationK];
                    simdgroup_multiply_accumulate(accg0, A0, B, accg0);
                    simdgroup_multiply_accumulate(accg1, A1, B, accg1);
            """
        )
        replaceOnce(
            """
            acc.thread_elements()[0] = metal::fma(
                rowScale, accg.thread_elements()[0], acc.thread_elements()[0]);
            acc.thread_elements()[1] = metal::fma(
                rowScale, accg.thread_elements()[1], acc.thread_elements()[1]);
            """,
            with: """
            acc0.thread_elements()[0] = metal::fma(
                rowScale0, accg0.thread_elements()[0], acc0.thread_elements()[0]);
            acc0.thread_elements()[1] = metal::fma(
                rowScale0, accg0.thread_elements()[1], acc0.thread_elements()[1]);
            acc1.thread_elements()[0] = metal::fma(
                rowScale1, accg1.thread_elements()[0], acc1.thread_elements()[0]);
            acc1.thread_elements()[1] = metal::fma(
                rowScale1, accg1.thread_elements()[1], acc1.thread_elements()[1]);
            """
        )
        replaceOnce(
            """
                    const device T* fragmentBRow =
                        biases + (sgN0 + fragmentRow) * G_ROW;
                    simdgroup_matrix<float, 8, 8> BBm;
                    simdgroup_matrix<float, 8, 8> XBm;
            """,
            with: """
                    const device T* fragmentBRow0 =
                        biases + (sgN0 + fragmentRow) * G_ROW;
                    const device T* fragmentBRow1 =
                        biases + (sgN0 + N_PSG + fragmentRow) * G_ROW;
                    simdgroup_matrix<float, 8, 8> BBm0;
                    simdgroup_matrix<float, 8, 8> BBm1;
                    simdgroup_matrix<float, 8, 8> XBm;
            """
        )
        replaceOnce(
            """
                        BBm.thread_elements()[0] = float(fragmentBRow[biasCol0]);
                        BBm.thread_elements()[1] = float(fragmentBRow[biasCol1]);
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
            """,
            with: """
                        BBm0.thread_elements()[0] = float(fragmentBRow0[biasCol0]);
                        BBm0.thread_elements()[1] = float(fragmentBRow0[biasCol1]);
                        BBm1.thread_elements()[0] = float(fragmentBRow1[biasCol0]);
                        BBm1.thread_elements()[1] = float(fragmentBRow1[biasCol1]);
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
            """
        )
        replaceOnce(
            """
                        BBm.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow[biasCol0]) : 0.0f;
                        BBm.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow[biasCol1]) : 0.0f;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
            """,
            with: """
                        BBm0.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow0[biasCol0]) : 0.0f;
                        BBm0.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow0[biasCol1]) : 0.0f;
                        BBm1.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow1[biasCol0]) : 0.0f;
                        BBm1.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow1[biasCol1]) : 0.0f;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
            """
        )
        replaceOnce(
            "simdgroup_multiply_accumulate(acc, BBm, XBm, acc);",
            with: """
                    simdgroup_multiply_accumulate(acc0, BBm0, XBm, acc0);
                    simdgroup_multiply_accumulate(acc1, BBm1, XBm, acc1);
            """
        )
        replaceOnce(
            """
            const uint outputN = sgN0 + fragmentRow;
            out[fragmentCol * N + outputN] = T(acc.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN] = T(acc.thread_elements()[1]);
            """,
            with: """
            const uint outputN0 = sgN0 + fragmentRow;
            const uint outputN1 = outputN0 + N_PSG;
            out[fragmentCol * N + outputN0] = T(acc0.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN0] = T(acc0.thread_elements()[1]);
            out[fragmentCol * N + outputN1] = T(acc1.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN1] = T(acc1.thread_elements()[1]);
            """
        )
        return result
    }()

    private static let kernelV15: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v15_dualtile",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV15,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 16 --- four output tiles per SIMD group

    private static let sourceV16: String = {
        var result = sourceV15

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV16 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            const uint n0 = tg * (N_SG * N_PSG * 2);
            const uint sgN0 = n0 + sg * N_PSG * 2;
            """,
            with: """
            const uint n0 = tg * (N_SG * N_PSG * 4);
            const uint sgN0 = n0 + sg * N_PSG * 4;
            """
        )
        replaceOnce(
            """
            const device uint32_t* fragmentWRow0 =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow1 =
                w + (sgN0 + N_PSG + fragmentRow) * W_ROW_U32;
            const device T* fragmentSRow0 =
                scales + (sgN0 + fragmentRow) * G_ROW;
            const device T* fragmentSRow1 =
                scales + (sgN0 + N_PSG + fragmentRow) * G_ROW;
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
            const device uint32_t* fragmentWRow0 =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow1 =
                w + (sgN0 + N_PSG + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow2 =
                w + (sgN0 + N_PSG * 2 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow3 =
                w + (sgN0 + N_PSG * 3 + fragmentRow) * W_ROW_U32;
            const device T* fragmentSRow0 =
                scales + (sgN0 + fragmentRow) * G_ROW;
            const device T* fragmentSRow1 =
                scales + (sgN0 + N_PSG + fragmentRow) * G_ROW;
            const device T* fragmentSRow2 =
                scales + (sgN0 + N_PSG * 2 + fragmentRow) * G_ROW;
            const device T* fragmentSRow3 =
                scales + (sgN0 + N_PSG * 3 + fragmentRow) * G_ROW;
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc2 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc3 = simdgroup_matrix<float, 8, 8>(0.0f);
            """
        )
        replaceOnce(
            """
                const device uint4* packedGroup0 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow0 + g * (GROUP / 8));
                const device uint4* packedGroup1 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow1 + g * (GROUP / 8));
                const uint4 packedLo0 = packedGroup0[0];
                const uint4 packedHi0 = packedGroup0[1];
                const uint4 packedLo1 = packedGroup1[0];
                const uint4 packedHi1 = packedGroup1[1];
            """,
            with: """
                const device uint4* packedGroup0 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow0 + g * (GROUP / 8));
                const device uint4* packedGroup1 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow1 + g * (GROUP / 8));
                const device uint4* packedGroup2 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow2 + g * (GROUP / 8));
                const device uint4* packedGroup3 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow3 + g * (GROUP / 8));
                const uint4 packedLo0 = packedGroup0[0];
                const uint4 packedHi0 = packedGroup0[1];
                const uint4 packedLo1 = packedGroup1[0];
                const uint4 packedHi1 = packedGroup1[1];
                const uint4 packedLo2 = packedGroup2[0];
                const uint4 packedHi2 = packedGroup2[1];
                const uint4 packedLo3 = packedGroup3[0];
                const uint4 packedHi3 = packedGroup3[1];
            """
        )
        replaceOnce(
            """
                const float rowScale0 = float(fragmentSRow0[g]);
                const float rowScale1 = float(fragmentSRow1[g]);
            """,
            with: """
                const float rowScale0 = float(fragmentSRow0[g]);
                const float rowScale1 = float(fragmentSRow1[g]);
                const float rowScale2 = float(fragmentSRow2[g]);
                const float rowScale3 = float(fragmentSRow3[g]);
            """
        )
        replaceOnce(
            """
                simdgroup_matrix<float, 8, 8> accg0 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg1 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
                simdgroup_matrix<float, 8, 8> accg0 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg1 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg2 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg3 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
            """
        )
        replaceOnce(
            """
                    simdgroup_matrix<T, 8, 8> A0;
                    simdgroup_matrix<T, 8, 8> A1;
                    simdgroup_matrix<T, 8, 8> B;
                    const uint packed0 = t < 4 ? packedLo0[t] : packedHi0[t - 4];
                    const uint packed1 = t < 4 ? packedLo1[t] : packedHi1[t - 4];
                    A0.thread_elements()[0] =
                        T(float((packed0 >> (4 * fragmentCol)) & 0xFu));
                    A0.thread_elements()[1] =
                        T(float((packed0 >> (4 * (fragmentCol + 1))) & 0xFu));
                    A1.thread_elements()[0] =
                        T(float((packed1 >> (4 * fragmentCol)) & 0xFu));
                    A1.thread_elements()[1] =
                        T(float((packed1 >> (4 * (fragmentCol + 1))) & 0xFu));
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] = x[fragmentCol * K + activationK];
                    B.thread_elements()[1] =
                        x[(fragmentCol + 1) * K + activationK];
                    simdgroup_multiply_accumulate(accg0, A0, B, accg0);
                    simdgroup_multiply_accumulate(accg1, A1, B, accg1);
            """,
            with: """
                    simdgroup_matrix<float, 8, 8> A0;
                    simdgroup_matrix<float, 8, 8> A1;
                    simdgroup_matrix<float, 8, 8> A2;
                    simdgroup_matrix<float, 8, 8> A3;
                    simdgroup_matrix<float, 8, 8> B;
                    const uint packed0 = t < 4 ? packedLo0[t] : packedHi0[t - 4];
                    const uint packed1 = t < 4 ? packedLo1[t] : packedHi1[t - 4];
                    const uint packed2 = t < 4 ? packedLo2[t] : packedHi2[t - 4];
                    const uint packed3 = t < 4 ? packedLo3[t] : packedHi3[t - 4];
                    A0.thread_elements()[0] =
                        float((packed0 >> (4 * fragmentCol)) & 0xFu);
                    A0.thread_elements()[1] =
                        float((packed0 >> (4 * (fragmentCol + 1))) & 0xFu);
                    A1.thread_elements()[0] =
                        float((packed1 >> (4 * fragmentCol)) & 0xFu);
                    A1.thread_elements()[1] =
                        float((packed1 >> (4 * (fragmentCol + 1))) & 0xFu);
                    A2.thread_elements()[0] =
                        float((packed2 >> (4 * fragmentCol)) & 0xFu);
                    A2.thread_elements()[1] =
                        float((packed2 >> (4 * (fragmentCol + 1))) & 0xFu);
                    A3.thread_elements()[0] =
                        float((packed3 >> (4 * fragmentCol)) & 0xFu);
                    A3.thread_elements()[1] =
                        float((packed3 >> (4 * (fragmentCol + 1))) & 0xFu);
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] =
                        float(x[fragmentCol * K + activationK]);
                    B.thread_elements()[1] =
                        float(x[(fragmentCol + 1) * K + activationK]);
                    simdgroup_multiply_accumulate(accg0, A0, B, accg0);
                    simdgroup_multiply_accumulate(accg1, A1, B, accg1);
                    simdgroup_multiply_accumulate(accg2, A2, B, accg2);
                    simdgroup_multiply_accumulate(accg3, A3, B, accg3);
            """
        )
        replaceOnce(
            """
            acc0.thread_elements()[0] = metal::fma(
                rowScale0, accg0.thread_elements()[0], acc0.thread_elements()[0]);
            acc0.thread_elements()[1] = metal::fma(
                rowScale0, accg0.thread_elements()[1], acc0.thread_elements()[1]);
            acc1.thread_elements()[0] = metal::fma(
                rowScale1, accg1.thread_elements()[0], acc1.thread_elements()[0]);
            acc1.thread_elements()[1] = metal::fma(
                rowScale1, accg1.thread_elements()[1], acc1.thread_elements()[1]);
            """,
            with: """
            acc0.thread_elements()[0] = metal::fma(
                rowScale0, accg0.thread_elements()[0], acc0.thread_elements()[0]);
            acc0.thread_elements()[1] = metal::fma(
                rowScale0, accg0.thread_elements()[1], acc0.thread_elements()[1]);
            acc1.thread_elements()[0] = metal::fma(
                rowScale1, accg1.thread_elements()[0], acc1.thread_elements()[0]);
            acc1.thread_elements()[1] = metal::fma(
                rowScale1, accg1.thread_elements()[1], acc1.thread_elements()[1]);
            acc2.thread_elements()[0] = metal::fma(
                rowScale2, accg2.thread_elements()[0], acc2.thread_elements()[0]);
            acc2.thread_elements()[1] = metal::fma(
                rowScale2, accg2.thread_elements()[1], acc2.thread_elements()[1]);
            acc3.thread_elements()[0] = metal::fma(
                rowScale3, accg3.thread_elements()[0], acc3.thread_elements()[0]);
            acc3.thread_elements()[1] = metal::fma(
                rowScale3, accg3.thread_elements()[1], acc3.thread_elements()[1]);
            """
        )
        replaceOnce(
            """
                    const device T* fragmentBRow0 =
                        biases + (sgN0 + fragmentRow) * G_ROW;
                    const device T* fragmentBRow1 =
                        biases + (sgN0 + N_PSG + fragmentRow) * G_ROW;
                    simdgroup_matrix<float, 8, 8> BBm0;
                    simdgroup_matrix<float, 8, 8> BBm1;
                    simdgroup_matrix<float, 8, 8> XBm;
            """,
            with: """
                    const device T* fragmentBRow0 =
                        biases + (sgN0 + fragmentRow) * G_ROW;
                    const device T* fragmentBRow1 =
                        biases + (sgN0 + N_PSG + fragmentRow) * G_ROW;
                    const device T* fragmentBRow2 =
                        biases + (sgN0 + N_PSG * 2 + fragmentRow) * G_ROW;
                    const device T* fragmentBRow3 =
                        biases + (sgN0 + N_PSG * 3 + fragmentRow) * G_ROW;
                    simdgroup_matrix<float, 8, 8> BBm0;
                    simdgroup_matrix<float, 8, 8> BBm1;
                    simdgroup_matrix<float, 8, 8> BBm2;
                    simdgroup_matrix<float, 8, 8> BBm3;
                    simdgroup_matrix<float, 8, 8> XBm;
            """
        )
        replaceOnce(
            """
                        BBm0.thread_elements()[0] = float(fragmentBRow0[biasCol0]);
                        BBm0.thread_elements()[1] = float(fragmentBRow0[biasCol1]);
                        BBm1.thread_elements()[0] = float(fragmentBRow1[biasCol0]);
                        BBm1.thread_elements()[1] = float(fragmentBRow1[biasCol1]);
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
            """,
            with: """
                        BBm0.thread_elements()[0] = float(fragmentBRow0[biasCol0]);
                        BBm0.thread_elements()[1] = float(fragmentBRow0[biasCol1]);
                        BBm1.thread_elements()[0] = float(fragmentBRow1[biasCol0]);
                        BBm1.thread_elements()[1] = float(fragmentBRow1[biasCol1]);
                        BBm2.thread_elements()[0] = float(fragmentBRow2[biasCol0]);
                        BBm2.thread_elements()[1] = float(fragmentBRow2[biasCol1]);
                        BBm3.thread_elements()[0] = float(fragmentBRow3[biasCol0]);
                        BBm3.thread_elements()[1] = float(fragmentBRow3[biasCol1]);
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
            """
        )
        replaceOnce(
            """
                        BBm0.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow0[biasCol0]) : 0.0f;
                        BBm0.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow0[biasCol1]) : 0.0f;
                        BBm1.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow1[biasCol0]) : 0.0f;
                        BBm1.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow1[biasCol1]) : 0.0f;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
            """,
            with: """
                        BBm0.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow0[biasCol0]) : 0.0f;
                        BBm0.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow0[biasCol1]) : 0.0f;
                        BBm1.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow1[biasCol0]) : 0.0f;
                        BBm1.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow1[biasCol1]) : 0.0f;
                        BBm2.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow2[biasCol0]) : 0.0f;
                        BBm2.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow2[biasCol1]) : 0.0f;
                        BBm3.thread_elements()[0] = biasCol0 < N_GROUPS
                            ? float(fragmentBRow3[biasCol0]) : 0.0f;
                        BBm3.thread_elements()[1] = biasCol1 < N_GROUPS
                            ? float(fragmentBRow3[biasCol1]) : 0.0f;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
            """
        )
        replaceOnce(
            """
                    simdgroup_multiply_accumulate(acc0, BBm0, XBm, acc0);
                    simdgroup_multiply_accumulate(acc1, BBm1, XBm, acc1);
            """,
            with: """
                    simdgroup_multiply_accumulate(acc0, BBm0, XBm, acc0);
                    simdgroup_multiply_accumulate(acc1, BBm1, XBm, acc1);
                    simdgroup_multiply_accumulate(acc2, BBm2, XBm, acc2);
                    simdgroup_multiply_accumulate(acc3, BBm3, XBm, acc3);
            """
        )
        replaceOnce(
            """
            const uint outputN0 = sgN0 + fragmentRow;
            const uint outputN1 = outputN0 + N_PSG;
            out[fragmentCol * N + outputN0] = T(acc0.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN0] = T(acc0.thread_elements()[1]);
            out[fragmentCol * N + outputN1] = T(acc1.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN1] = T(acc1.thread_elements()[1]);
            """,
            with: """
            const uint outputN0 = sgN0 + fragmentRow;
            const uint outputN1 = outputN0 + N_PSG;
            const uint outputN2 = outputN0 + N_PSG * 2;
            const uint outputN3 = outputN0 + N_PSG * 3;
            out[fragmentCol * N + outputN0] = T(acc0.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN0] = T(acc0.thread_elements()[1]);
            out[fragmentCol * N + outputN1] = T(acc1.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN1] = T(acc1.thread_elements()[1]);
            out[fragmentCol * N + outputN2] = T(acc2.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN2] = T(acc2.thread_elements()[1]);
            out[fragmentCol * N + outputN3] = T(acc3.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN3] = T(acc3.thread_elements()[1]);
            """
        )
        return result
    }()

    private static let kernelV16: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v16_quadtile",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV16,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 26 --- reused packed-weight cursor

    private static let sourceV26: String = {
        var result = sourceV16

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV26 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            const device uint32_t* fragmentWRow0 =
                w + (sgN0 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow1 =
                w + (sgN0 + N_PSG + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow2 =
                w + (sgN0 + N_PSG * 2 + fragmentRow) * W_ROW_U32;
            const device uint32_t* fragmentWRow3 =
                w + (sgN0 + N_PSG * 3 + fragmentRow) * W_ROW_U32;
            """,
            with: """
            """
        )
        replaceOnce(
            """
                const device uint4* packedGroup0 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow0 + g * (GROUP / 8));
                const device uint4* packedGroup1 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow1 + g * (GROUP / 8));
                const device uint4* packedGroup2 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow2 + g * (GROUP / 8));
                const device uint4* packedGroup3 =
                    reinterpret_cast<const device uint4*>(
                        fragmentWRow3 + g * (GROUP / 8));
                const uint4 packedLo0 = packedGroup0[0];
                const uint4 packedHi0 = packedGroup0[1];
                const uint4 packedLo1 = packedGroup1[0];
                const uint4 packedHi1 = packedGroup1[1];
                const uint4 packedLo2 = packedGroup2[0];
                const uint4 packedHi2 = packedGroup2[1];
                const uint4 packedLo3 = packedGroup3[0];
                const uint4 packedHi3 = packedGroup3[1];
            """,
            with: """
                const uint packedWord0 =
                    (sgN0 + fragmentRow) * W_ROW_U32 + g * (GROUP / 8);
                const uint packedTileStride = N_PSG * W_ROW_U32;
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(w + packedWord0);
                const uint4 packedLo0 = packedGroup[0];
                const uint4 packedHi0 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride);
                const uint4 packedLo1 = packedGroup[0];
                const uint4 packedHi1 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride * 2);
                const uint4 packedLo2 = packedGroup[0];
                const uint4 packedHi2 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride * 3);
                const uint4 packedLo3 = packedGroup[0];
                const uint4 packedHi3 = packedGroup[1];
            """
        )
        return result
    }()

    private static let kernelV26: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v26_weight_cursor",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV26,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 27 --- compile-time affine-block inner trip

    private static let sourceV27: String = {
        var result = sourceV26

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV27 replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            for (uint biasBlock = 0; biasBlock < N_GROUPS; biasBlock += 8) {
                const uint blockGroups = min(8u, N_GROUPS - biasBlock);
                for (uint gg = 0; gg < blockGroups; ++gg) {
                    const uint g = biasBlock + gg;
            """,
            with: """
            #pragma unroll
            for (uint biasBlock = 0; biasBlock < N_GROUPS; biasBlock += 8) {
                #pragma unroll
                for (uint gg = 0; gg < 8; ++gg) {
                    const uint g = biasBlock + gg;
                    if (g >= N_GROUPS) continue;
            """
        )

        return result
    }()

    private static let kernelV27: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v27_unroll_blocks_fpmma_v1",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV27,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 27 carry --- weight-operand register carry

    private static let carryEnabled = environmentFlag("DARKBLOOM_GEMMA4_MMA_HEAD_CARRY")

    private static let sourceV27Carry: String = {
        var result = sourceV27

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(
                count == 2, "sourceV27Carry replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc2 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc3 = simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc2 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc3 = simdgroup_matrix<float, 8, 8>(0.0f);

            // Weight operands carried one group ahead. Both row bases are
            // functions of the fragment row alone, so the value each trip
            // consumes is the value the in-place read produced.
            const uint carryWordBase = (sgN0 + fragmentRow) * W_ROW_U32;
            const uint carryTileStride = N_PSG * W_ROW_U32;
            uint4 carryLo0;
            uint4 carryHi0;
            uint4 carryLo1;
            uint4 carryHi1;
            uint4 carryLo2;
            uint4 carryHi2;
            uint4 carryLo3;
            uint4 carryHi3;
            T carryS0;
            T carryS1;
            T carryS2;
            T carryS3;
            {
                const device uint4* carryGroup =
                    reinterpret_cast<const device uint4*>(w + carryWordBase);
                carryLo0 = carryGroup[0];
                carryHi0 = carryGroup[1];
                carryGroup = reinterpret_cast<const device uint4*>(
                    w + carryWordBase + carryTileStride);
                carryLo1 = carryGroup[0];
                carryHi1 = carryGroup[1];
                carryGroup = reinterpret_cast<const device uint4*>(
                    w + carryWordBase + carryTileStride * 2);
                carryLo2 = carryGroup[0];
                carryHi2 = carryGroup[1];
                carryGroup = reinterpret_cast<const device uint4*>(
                    w + carryWordBase + carryTileStride * 3);
                carryLo3 = carryGroup[0];
                carryHi3 = carryGroup[1];
                carryS0 = fragmentSRow0[0];
                carryS1 = fragmentSRow1[0];
                carryS2 = fragmentSRow2[0];
                carryS3 = fragmentSRow3[0];
            }
            """
        )

        replaceOnce(
            """
                const uint packedWord0 =
                    (sgN0 + fragmentRow) * W_ROW_U32 + g * (GROUP / 8);
                const uint packedTileStride = N_PSG * W_ROW_U32;
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(w + packedWord0);
                const uint4 packedLo0 = packedGroup[0];
                const uint4 packedHi0 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride);
                const uint4 packedLo1 = packedGroup[0];
                const uint4 packedHi1 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride * 2);
                const uint4 packedLo2 = packedGroup[0];
                const uint4 packedHi2 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWord0 + packedTileStride * 3);
                const uint4 packedLo3 = packedGroup[0];
                const uint4 packedHi3 = packedGroup[1];
                    const float rowScale0 = float(fragmentSRow0[g]);
                const float rowScale1 = float(fragmentSRow1[g]);
                const float rowScale2 = float(fragmentSRow2[g]);
                const float rowScale3 = float(fragmentSRow3[g]);
            """,
            with: """
                const uint4 packedLo0 = carryLo0;
                const uint4 packedHi0 = carryHi0;
                const uint4 packedLo1 = carryLo1;
                const uint4 packedHi1 = carryHi1;
                const uint4 packedLo2 = carryLo2;
                const uint4 packedHi2 = carryHi2;
                const uint4 packedLo3 = carryLo3;
                const uint4 packedHi3 = carryHi3;
                const float rowScale0 = float(carryS0);
                const float rowScale1 = float(carryS1);
                const float rowScale2 = float(carryS2);
                const float rowScale3 = float(carryS3);
                const uint gNext = min(g + 1u, N_GROUPS - 1u);
                const uint packedWordNext = carryWordBase + gNext * (GROUP / 8);
                const device uint4* packedGroup =
                    reinterpret_cast<const device uint4*>(w + packedWordNext);
                carryLo0 = packedGroup[0];
                carryHi0 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWordNext + carryTileStride);
                carryLo1 = packedGroup[0];
                carryHi1 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWordNext + carryTileStride * 2);
                carryLo2 = packedGroup[0];
                carryHi2 = packedGroup[1];
                packedGroup = reinterpret_cast<const device uint4*>(
                    w + packedWordNext + carryTileStride * 3);
                carryLo3 = packedGroup[0];
                carryHi3 = packedGroup[1];
                carryS0 = fragmentSRow0[gNext];
                carryS1 = fragmentSRow1[gNext];
                carryS2 = fragmentSRow2[gNext];
                carryS3 = fragmentSRow3[gNext];
            """
        )

        return result
    }()

    private static let kernelV27Carry: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v27_unroll_blocks_carry_fpmma_v2",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV27Carry,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    // MARK: - Version 27 rows --- RT eight-row tiles share one weight fetch

    private static let rowsEnabled = environmentFlag("DARKBLOOM_GEMMA4_MMA_HEAD_ROWS")

    // Every row tile runs the eight-row body's statements in the eight-row
    // order on the same dequantized operands; only the B fragment, the sums
    // and the stores are indexed by the tile. The activation arrives as
    // [K][pair][RT][2] so a lane's RT bf16 pairs are one contiguous run and
    // a simdgroup's B fragment loads coalesce; the unpack is the exact
    // bf16 -> float widening the eight-row body performs.
    private static let sourceV27Rows: String = {
        var result = sourceV27Carry

        func replaceOnce(_ old: String, with new: String) {
            let count = result.components(separatedBy: old).count
            precondition(count == 2, "sourceV27Rows replacement count \(count): \(old)")
            result = result.replacingOccurrences(of: old, with: new)
        }

        replaceOnce(
            """
            simdgroup_matrix<float, 8, 8> acc0 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc1 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc2 = simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> acc3 = simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
            const device uint* xPairs = reinterpret_cast<const device uint*>(x);
            simdgroup_matrix<float, 8, 8> acc0[RT];
            simdgroup_matrix<float, 8, 8> acc1[RT];
            simdgroup_matrix<float, 8, 8> acc2[RT];
            simdgroup_matrix<float, 8, 8> acc3[RT];
            #pragma clang loop unroll(full)
            for (int r = 0; r < RT; ++r) {
                acc0[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                acc1[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                acc2[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                acc3[r] = simdgroup_matrix<float, 8, 8>(0.0f);
            }
            """
        )

        replaceOnce(
            """
                simdgroup_matrix<float, 8, 8> accg0 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg1 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg2 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<float, 8, 8> accg3 =
                    simdgroup_matrix<float, 8, 8>(0.0f);
            """,
            with: """
                simdgroup_matrix<float, 8, 8> accg0[RT];
                simdgroup_matrix<float, 8, 8> accg1[RT];
                simdgroup_matrix<float, 8, 8> accg2[RT];
                simdgroup_matrix<float, 8, 8> accg3[RT];
                #pragma clang loop unroll(full)
                for (int r = 0; r < RT; ++r) {
                    accg0[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                    accg1[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                    accg2[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                    accg3[r] = simdgroup_matrix<float, 8, 8>(0.0f);
                }
            """
        )

        replaceOnce("        simdgroup_matrix<float, 8, 8> B;\n", with: "")

        replaceOnce(
            """
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    B.thread_elements()[0] =
                        float(x[fragmentCol * K + activationK]);
                    B.thread_elements()[1] =
                        float(x[(fragmentCol + 1) * K + activationK]);
                    simdgroup_multiply_accumulate(accg0, A0, B, accg0);
                    simdgroup_multiply_accumulate(accg1, A1, B, accg1);
                    simdgroup_multiply_accumulate(accg2, A2, B, accg2);
                    simdgroup_multiply_accumulate(accg3, A3, B, accg3);
            """,
            with: """
                    const uint activationK = g * GROUP + t * 8 + fragmentRow;
                    const device uint* activationPairs =
                        xPairs + (activationK * 4 + (fragmentCol >> 1)) * RT;
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < RT; ++r) {
                        const uint pair = activationPairs[r];
                        simdgroup_matrix<float, 8, 8> B;
                        B.thread_elements()[0] = as_type<float>(pair << 16);
                        B.thread_elements()[1] = as_type<float>(pair & 0xFFFF0000u);
                        simdgroup_multiply_accumulate(accg0[r], A0, B, accg0[r]);
                        simdgroup_multiply_accumulate(accg1[r], A1, B, accg1[r]);
                        simdgroup_multiply_accumulate(accg2[r], A2, B, accg2[r]);
                        simdgroup_multiply_accumulate(accg3[r], A3, B, accg3[r]);
                    }
            """
        )

        replaceOnce(
            """
            acc0.thread_elements()[0] = metal::fma(
                rowScale0, accg0.thread_elements()[0], acc0.thread_elements()[0]);
            acc0.thread_elements()[1] = metal::fma(
                rowScale0, accg0.thread_elements()[1], acc0.thread_elements()[1]);
            acc1.thread_elements()[0] = metal::fma(
                rowScale1, accg1.thread_elements()[0], acc1.thread_elements()[0]);
            acc1.thread_elements()[1] = metal::fma(
                rowScale1, accg1.thread_elements()[1], acc1.thread_elements()[1]);
            acc2.thread_elements()[0] = metal::fma(
                rowScale2, accg2.thread_elements()[0], acc2.thread_elements()[0]);
            acc2.thread_elements()[1] = metal::fma(
                rowScale2, accg2.thread_elements()[1], acc2.thread_elements()[1]);
            acc3.thread_elements()[0] = metal::fma(
                rowScale3, accg3.thread_elements()[0], acc3.thread_elements()[0]);
            acc3.thread_elements()[1] = metal::fma(
                rowScale3, accg3.thread_elements()[1], acc3.thread_elements()[1]);
            """,
            with: """
            #pragma clang loop unroll(full)
            for (int r = 0; r < RT; ++r) {
                acc0[r].thread_elements()[0] = metal::fma(
                    rowScale0, accg0[r].thread_elements()[0], acc0[r].thread_elements()[0]);
                acc0[r].thread_elements()[1] = metal::fma(
                    rowScale0, accg0[r].thread_elements()[1], acc0[r].thread_elements()[1]);
                acc1[r].thread_elements()[0] = metal::fma(
                    rowScale1, accg1[r].thread_elements()[0], acc1[r].thread_elements()[0]);
                acc1[r].thread_elements()[1] = metal::fma(
                    rowScale1, accg1[r].thread_elements()[1], acc1[r].thread_elements()[1]);
                acc2[r].thread_elements()[0] = metal::fma(
                    rowScale2, accg2[r].thread_elements()[0], acc2[r].thread_elements()[0]);
                acc2[r].thread_elements()[1] = metal::fma(
                    rowScale2, accg2[r].thread_elements()[1], acc2[r].thread_elements()[1]);
                acc3[r].thread_elements()[0] = metal::fma(
                    rowScale3, accg3[r].thread_elements()[0], acc3[r].thread_elements()[0]);
                acc3[r].thread_elements()[1] = metal::fma(
                    rowScale3, accg3[r].thread_elements()[1], acc3[r].thread_elements()[1]);
            }
            """
        )

        replaceOnce(
            """
                        XBm.thread_elements()[0] =
                            xSums[fragmentCol * N_GROUPS + biasRow];
                        XBm.thread_elements()[1] =
                            xSums[(fragmentCol + 1) * N_GROUPS + biasRow];
                    } else {
            """,
            with: """
                    } else {
            """
        )

        replaceOnce(
            """
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[fragmentCol * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(fragmentCol + 1) * N_GROUPS + biasRow] : 0.0f;
                    }
                            simdgroup_multiply_accumulate(acc0, BBm0, XBm, acc0);
                    simdgroup_multiply_accumulate(acc1, BBm1, XBm, acc1);
                    simdgroup_multiply_accumulate(acc2, BBm2, XBm, acc2);
                    simdgroup_multiply_accumulate(acc3, BBm3, XBm, acc3);
            """,
            with: """
                    }
                    #pragma clang loop unroll(full)
                    for (int r = 0; r < RT; ++r) {
                        const uint sumM = uint(r) * 8u + fragmentCol;
                        XBm.thread_elements()[0] = biasRow < N_GROUPS
                            ? xSums[sumM * N_GROUPS + biasRow] : 0.0f;
                        XBm.thread_elements()[1] = biasRow < N_GROUPS
                            ? xSums[(sumM + 1) * N_GROUPS + biasRow] : 0.0f;
                        simdgroup_multiply_accumulate(acc0[r], BBm0, XBm, acc0[r]);
                        simdgroup_multiply_accumulate(acc1[r], BBm1, XBm, acc1[r]);
                        simdgroup_multiply_accumulate(acc2[r], BBm2, XBm, acc2[r]);
                        simdgroup_multiply_accumulate(acc3[r], BBm3, XBm, acc3[r]);
                    }
            """
        )

        replaceOnce(
            """
            out[fragmentCol * N + outputN0] = T(acc0.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN0] = T(acc0.thread_elements()[1]);
            out[fragmentCol * N + outputN1] = T(acc1.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN1] = T(acc1.thread_elements()[1]);
            out[fragmentCol * N + outputN2] = T(acc2.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN2] = T(acc2.thread_elements()[1]);
            out[fragmentCol * N + outputN3] = T(acc3.thread_elements()[0]);
            out[(fragmentCol + 1) * N + outputN3] = T(acc3.thread_elements()[1]);
            """,
            with: """
            #pragma clang loop unroll(full)
            for (int r = 0; r < RT; ++r) {
                const uint outputM = uint(r) * 8u + fragmentCol;
                out[outputM * N + outputN0] = T(acc0[r].thread_elements()[0]);
                out[(outputM + 1) * N + outputN0] = T(acc0[r].thread_elements()[1]);
                out[outputM * N + outputN1] = T(acc1[r].thread_elements()[0]);
                out[(outputM + 1) * N + outputN1] = T(acc1[r].thread_elements()[1]);
                out[outputM * N + outputN2] = T(acc2[r].thread_elements()[0]);
                out[(outputM + 1) * N + outputN2] = T(acc2[r].thread_elements()[1]);
                out[outputM * N + outputN3] = T(acc3[r].thread_elements()[0]);
                out[(outputM + 1) * N + outputN3] = T(acc3[r].thread_elements()[1]);
            }
            """
        )

        return result
    }()

    private static let kernelV27Rows: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_mma_affine4_qmv_m8_v27_unroll_blocks_carry_fpmma_rows_v2",
        inputNames: ["x", "w", "scales", "biases", "xSums"],
        outputNames: ["out"],
        source: sourceV27Rows,
        header: "#include <metal_simdgroup_matrix>\n",
        ensureRowContiguous: true
    )

    private static func admittedShape(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        rows: Int, colsPerThreadgroup selectedColsPerThreadgroup: Int,
        groupSize: Int, bits: Int
    ) -> (k: Int, n: Int)? {
        guard groupSize == 64, bits == 4 else { return nil }
        guard x.dtype == .bfloat16, scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        guard w.dtype == .uint32 else { return nil }
        guard w.ndim == 2, scales.ndim == 2, biases.ndim == 2 else { return nil }

        guard x.ndim == 2 || (x.ndim == 3 && x.dim(1) == 1) else { return nil }
        guard x.dim(0) == rows else { return nil }
        let k = x.dim(-1)
        guard k > 0, x.size == rows * k else { return nil }

        let n = w.dim(0)
        guard n >= minOutputWidth, n % selectedColsPerThreadgroup == 0 else { return nil }
        guard k % groupSize == 0, k % 8 == 0 else { return nil }
        guard w.dim(1) == k * bits / 32 else { return nil }
        guard scales.dim(0) == n, biases.dim(0) == n else { return nil }
        guard scales.dim(1) == k / groupSize, biases.dim(1) == k / groupSize else { return nil }
        return (k, n)
    }

    private static func activationSumCells(
        _ kernel: MLXFast.MLXFastKernel, flatX: MLXArray, cells: Int,
        template: [(String, any KernelTemplateArg)]
    ) -> MLXArray {
        let sumThreads = 128
        let sumThreadgroups = (cells + sumThreads - 1) / sumThreads
        return kernel(
            [flatX],
            template: template,
            grid: (sumThreadgroups * sumThreads, 1, 1),
            threadGroup: (sumThreads, 1, 1),
            outputShapes: [[cells]],
            outputDTypes: [.float32]
        )[0]
    }

    /// One dispatch for a wide MTP verify rectangle of 16/24/32 rows; nil
    /// whenever the eight-row kernel would not be the selected path.
    public static func applyRows(
        x: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        guard enabled, rowsEnabled, version == 27, let biases, x.ndim >= 2 else { return nil }
        let rows = x.dim(0)
        let rowTiles = CBv2MTPWideVerifyContext.rowTileCount(rows: rows, tile: mRows)
        let selectedColsPerThreadgroup = colsPerThreadgroup * 4
        guard rowTiles > 1,
            case let (k, n)? = admittedShape(
                x: x, w: w, scales: scales, biases: biases, rows: rows,
                colsPerThreadgroup: selectedColsPerThreadgroup,
                groupSize: groupSize, bits: bits)
        else { return nil }

        let flatX = x.reshaped([rows, k])
        let xSums = activationSumCells(
            xSumRowsKernel, flatX: flatX, cells: rows * (k / groupSize),
            template: [("T", x.dtype), ("K", k), ("RT", rowTiles)])
        let pairedX = flatX.reshaped([rowTiles, 4, 2, k]).transposed(3, 1, 0, 2)
        CBv2EngageMark.once("mma-head-rows")
        let threadgroups = n / selectedColsPerThreadgroup
        return kernelV27Rows(
            [pairedX, w, scales, biases, xSums],
            template: [("T", x.dtype), ("K", k), ("N", n), ("RT", rowTiles)],
            grid: (threadgroups * threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [x.dtype]
        )[0]
    }

    public static func apply(
        x: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        activationSums: ActivationSums? = nil
    ) -> MLXArray? {
        guard enabled else { return nil }
        if let wide = applyRows(
            x: x, w: w, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
        {
            return wide
        }
        if (x.ndim == 2 || (x.ndim == 3 && x.dim(1) == 1)), x.dim(0) > mRows,
            let tiled = CBv2MTPWideVerifyContext.rowTiles(x, tile: mRows, {
                apply(
                    x: $0, w: w, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, activationSums: nil)
            })
        {
            return tiled
        }
        let selectedColsPerThreadgroup = version == 16 || version == 26 || version == 27
            ? colsPerThreadgroup * 4
            : (version == 15 ? colsPerThreadgroup * 2 : colsPerThreadgroup)
        guard let biases,
            case let (k, n)? = admittedShape(
                x: x, w: w, scales: scales, biases: biases, rows: mRows,
                colsPerThreadgroup: selectedColsPerThreadgroup,
                groupSize: groupSize, bits: bits)
        else { return nil }

        let flatX = x.reshaped([mRows, k])
        let threadgroups = n / selectedColsPerThreadgroup
        let selected: MLXFast.MLXFastKernel
        let inputs: [MLXArray]
        switch version {
        case 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 27:
            let sumCells = mRows * (k / groupSize)
            let xSums: MLXArray
            if let activationSums,
                activationSums.values.dtype == .float32,
                activationSums.values.ndim == 1,
                activationSums.values.size == sumCells
            {
                xSums = activationSums.values
            } else {
                xSums = activationSumCells(
                    xSumKernel, flatX: flatX, cells: sumCells,
                    template: [("T", x.dtype), ("K", k)])
            }
            switch version {
            case 27:
                if carryEnabled {
                    CBv2EngageMark.once("mma-head-carry")
                    selected = kernelV27Carry
                } else {
                    selected = kernelV27
                }
            case 26: selected = kernelV26
            case 16: selected = kernelV16
            case 15: selected = kernelV15
            case 14: selected = kernelV14
            case 13: selected = kernelV13
            case 12: selected = kernelV12
            case 11: selected = kernelV11
            case 10: selected = kernelV10
            case 9: selected = kernelV9
            case 8: selected = kernelV8
            case 7: selected = kernelV7
            case 6: selected = kernelV6
            default: selected = kernelV5
            }
            inputs = [flatX, w, scales, biases, xSums]
        case 4:
            selected = kernelV4
            inputs = [flatX, w, scales, biases]
        case 3:
            selected = kernelV3
            inputs = [flatX, w, scales, biases]
        case 2:
            selected = kernelV2
            inputs = [flatX, w, scales, biases]
        default:
            selected = kernel
            inputs = [flatX, w, scales, biases]
        }
        let outputs = selected(
            inputs,
            template: [("T", x.dtype), ("K", k), ("N", n)],
            grid: (threadgroups * threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[mRows, n]],
            outputDTypes: [x.dtype]
        )
        return outputs[0]
    }
}
