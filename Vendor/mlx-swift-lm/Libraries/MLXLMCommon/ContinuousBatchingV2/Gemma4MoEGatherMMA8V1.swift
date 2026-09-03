// MOE-MMA8-001: matrix-unit tier for the routed-expert gather projections at
// the batch-8, decode-length-one cohort geometry.
//
// WHAT RUNS TODAY. All three routed-expert projections of a decode round
// (gate, up: K = 2816, N = 704; down: K = 704, N = 2816; 4-bit affine, g64)
// go through the frozen host's `affine_gather_qmv_bfloat16_gs_64_b_4`
// (`GatherQMM::eval_gpu`, `quantized.cpp`: M = 1, B = 64 assignments, so
// neither the `gather_qmm_rhs` block-GEMM nor any `_nax` family is reachable).
// That kernel is the scalar QMV road: each lane streams 4-byte packed words,
// unpacks nibbles with masks and multiplies them one by one on the ALUs, and
// the only cross-row weight reuse is the run-length election (pair / triple /
// quad on gate/up, pair only on down). The expert planes are ~80 % of the
// weight bytes a decode round moves (about 50 unique experts x 30 layers x
// ~3.3 MiB), and they are the LAST decode plane still on that road: Q/K/V
// (`AttentionQKVMMA8V1`), o_proj (`AttentionOQMVV1` MMA-O8), the dense MLP
// (`DenseMLPQMVV1` MMA-MLP-001) and the tied head all moved to the
// `simdgroup_float8x8` tier, and each move was promoted on the ranked box.
//
// WHAT THIS DOES. The promoted Q/K/V body (`qkv_mma8_affine4_g64_impl`) is
// carried here with the operand addressing generalized to a gathered
// expert plane: one threadgroup per (assignment, output-column tile group)
// elects the run LEADER of its sorted same-expert run (the same backward
// `run_offset` scan the incumbent gather performs), and the leader serves up
// to eight assignment rows of that run from ONE pass over the expert's weight
// tile. Rows past the run length are clamped to the leader's row and their
// outputs are never stored, so every one of the 64 assignment rows is
// written exactly once by exactly one leader. Per 64-value K group each lane
// reads one aligned 8-byte word per weight stream, one scale and one bias,
// builds eight `simdgroup_float8x8` activation operands once, and runs one
// eight-step `simdgroup_multiply_accumulate` chain per stream; the affine
// close `acc += s * C + rs * b` is the incumbent's own exact decomposition
// (integer nibbles times bf16 activations are exact in fp32; only the
// summation ORDER inside the matrix unit differs from the lane-serial QMV).
// The gate|up pair shares the activation operands and the run sums across the
// two planes (one launch writes both outputs), and TILES adjacent column
// tiles per threadgroup share them again.
//
// NUMERICS CLASS. This is a one-ulp-class reassociation of the fp32 sums,
// the same class as the promoted MMA tiers above; it is NOT bit-exact against
// the incumbent gather and is priced by the ranked per-stream 10 % token
// budget, not by a zero-flip rule. `Gemma4MoEGatherMMA8ParityTests` pins the
// candidate's error against an fp32 dequantized reference to the incumbent's
// own error band and checks every output row is written for every run
// pattern.
//
// Kill switches (same executable, byte-for-byte incumbent path restored):
//   DARKBLOOM_GEMMA4_MOE_MMA8=0        both planes back to the gather QMV
//   DARKBLOOM_GEMMA4_MOE_MMA8_DOWN=0   down plane only
//   DARKBLOOM_GEMMA4_MOE_MMA8_TILES=1|2|4  column tiles per threadgroup
// Engage marks: `moe-gateup-mma8`, `moe-down-mma8`.

import Foundation
import MLX
import MLXFast

public enum Gemma4MoEGatherMMA8V1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MOE_MMA8"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    public static let downEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MOE_MMA8_DOWN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Column tiles (of eight) per threadgroup; each tile is one more weight
    /// stream in flight per lane. 2 is the shipped default; 1 and 4 exist for
    /// local sweeps only.
    public static let tiles: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MOE_MMA8_TILES"],
            let v = Int(raw.trimmingCharacters(in: .whitespaces)),
            [1, 2, 4].contains(v)
        else { return 2 }
        return v
    }()

    static let assignments = 64
    static let cohortRows = 8
    static let experts = 128
    static let modelDim = 2816
    static let expertDim = 704
    static let groupSize = 64
    static let bits = 4
    static let simdWidth = 32
    static let simdGroups = 2

    /// Identity source-row table for the down plane (assignment `a` reads
    /// activation row `a`). Built once; the same table the host hands the
    /// incumbent down gather (`switchDownIdentity64`).
    nonisolated(unsafe) static let identity64: MLXArray = {
        let a = MLXArray((0..<assignments).map { UInt32($0) })
        eval(a)
        return a
    }()

    // MARK: - Metal

    static let kernelHeader = """
        #include <metal_simdgroup_matrix>

        #ifndef METAL_FUNC
        #define METAL_FUNC inline
        #endif

        // ---- Verbatim operand helpers from the promoted Q/K/V MMA8 tier ----
        struct mma8_coord {
          short fm;
          short fn;
        };

        inline mma8_coord mma8_lane(uint lane) {
          const short qid = short(lane / 4);
          return {
              short((qid & 4) + short((lane / 2) % 4)),
              short((qid & 2) * 2 + short(lane % 2) * 2)};
        }

        template <typename T, bool TWO_BYTE = (sizeof(T) == 2)>
        struct mma8_u16 {
          static inline T cast(ushort u) {
            return T(0);
          }
        };

        template <typename T>
        struct mma8_u16<T, true> {
          static inline T cast(ushort u) {
            return as_type<T>(u);
          }
        };

        template <typename T>
        inline float mma8_lo(uint u) {
          return float(mma8_u16<T>::cast(ushort(u & 0xFFFFu)));
        }

        template <typename T>
        inline float mma8_hi(uint u) {
          return float(mma8_u16<T>::cast(ushort(u >> 16)));
        }

        template <typename T>
        inline float mma8_runsum4(uint4 r) {
          thread T xt[8];
          xt[0] = mma8_u16<T>::cast(ushort(r.x & 0xFFFFu));
          xt[1] = mma8_u16<T>::cast(ushort(r.x >> 16));
          xt[2] = mma8_u16<T>::cast(ushort(r.y & 0xFFFFu));
          xt[3] = mma8_u16<T>::cast(ushort(r.y >> 16));
          xt[4] = mma8_u16<T>::cast(ushort(r.z & 0xFFFFu));
          xt[5] = mma8_u16<T>::cast(ushort(r.z >> 16));
          xt[6] = mma8_u16<T>::cast(ushort(r.w & 0xFFFFu));
          xt[7] = mma8_u16<T>::cast(ushort(r.w >> 16));
          float sum = 0;
          sum += xt[0] + xt[1] + xt[2] + xt[3];
          sum += xt[4] + xt[5] + xt[6] + xt[7];
          return sum;
        }

        #define MMA8_SETB(BB, W, HI) BB.thread_elements()[0] = mma8_##HI<T>(r0.W); BB.thread_elements()[1] = mma8_##HI<T>(r1.W);

        #define MMA8_STEP(BB, J) A.thread_elements()[0] = float(extract_bits(wv.x, 4 * (J), 4)); A.thread_elements()[1] = float(extract_bits(wv.y, 4 * (J), 4)); simdgroup_multiply_accumulate(C, A, BB, C);

        // ---- MOE-MMA8-001 gathered-expert body ----
        //
        // STREAMS = PLANES * TILES weight streams per lane: plane p (gate = 0,
        // up = 1; the down plane has one) times column tile t. Every stream
        // owns a private `C`, scale, bias and accumulator pair; the
        // activation operands B0..B7 and the run sums `rs` are built once per
        // K group and shared by all streams, exactly as the promoted
        // multi-tile Q/K/V body shares them across its tiles.
        //
        // K split: simdgroup 0 walks groups [0, gh), simdgroup 1 walks
        // [gh, G) (G = K / 64, gh = ceil(G / 2)); simdgroup 0 adds simdgroup
        // 1's partials through `red` and stores. For K = 704 (G = 11)
        // simdgroup 1's sixth trip is dead: its loads clamp to G - 1 and its
        // accumulate is skipped.
        // Weight storage: ONE row-contiguous plane of NROWS rows per expert
        // (NROWS = PLANES * NFIX); plane p's output column n lives at storage
        // row p * NFIX + n. For gate|up this is the loader's fused gate|up
        // storage ([128, 1408, K/8]), of which the split gate/up parameters
        // are zero-copy row slices; for down it is the down plane itself.
        template <typename T, int KFIX, int NFIX, int PLANES, int TILES>
        METAL_FUNC void moe_gather_mma8_affine4_g64_impl(
            const device uint32_t* w0,
            const device T* s0,
            const device T* b0,
            const device T* x,
            const device uint32_t* lhs,
            device T* y0,
            device T* y1,
            const uint assignment,
            const uint run_len,
            const uint expert,
            const int n0,
            threadgroup float2* red,
            uint simd_gid,
            uint simd_lid) {
          constexpr int K = KFIX;
          constexpr int N = NFIX;
          constexpr int NROWS = PLANES * NFIX;
          constexpr int G = K / 64;
          constexpr int gh = (G + 1) / 2;
          constexpr int STREAMS = PLANES * TILES;
          const int g0 = (simd_gid == 1) ? gh : 0;
          const mma8_coord c = mma8_lane(simd_lid);

          // Activation rows for this lane's two fragment columns. Rows past the
          // run are clamped onto the leader's row (valid memory, output never
          // stored).
          const uint rowA = uint(c.fn);
          const uint rowB = uint(c.fn) + 1u;
          const uint srcA = lhs[assignment + min(rowA, run_len - 1u)];
          const uint srcB = lhs[assignment + min(rowB, run_len - 1u)];
          const device T* x0 = x + ulong(srcA) * ulong(K) + 8 * c.fm;
          const device T* x1 = x + ulong(srcB) * ulong(K) + 8 * c.fm;

          const ulong wplane = ulong(expert) * ulong(NROWS) * ulong(K / 2);
          const ulong splane = ulong(expert) * ulong(NROWS) * ulong(G);

          const device uint8_t* wrow[STREAMS];
          const device T* srow[STREAMS];
          const device T* brow[STREAMS];
          thread float acc0[STREAMS];
          thread float acc1[STREAMS];
        #pragma clang loop unroll(full)
          for (int st = 0; st < STREAMS; ++st) {
            const int p = st / TILES;
            const int t = st - p * TILES;
            const int nt = p * NFIX + n0 + t * 8 + c.fm;
            wrow[st] = (const device uint8_t*)w0 + wplane + ulong(nt) * ulong(K / 2) + 4 * c.fn;
            srow[st] = s0 + splane + ulong(nt) * ulong(G);
            brow[st] = b0 + splane + ulong(nt) * ulong(G);
            acc0[st] = 0.0f;
            acc1[st] = 0.0f;
          }

          simdgroup_float8x8 A;
          simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

          // Two-deep register carry of the packed words, scale and bias:
          // group g0 (and g0 + 1) are read before the walk; every trip reads
          // group gi + 2 while gi is consumed. Indices clamp inside this
          // simdgroup's range, so the value a trip consumes is the value the
          // clamped address holds; the surplus reads are discarded.
          const int gmax = min(G, g0 + gh) - 1;
          uint2 wv_next[STREAMS];
          uint2 wv_next2[STREAMS];
          T s_next[STREAMS];
          T b_next[STREAMS];
        #pragma clang loop unroll(full)
          for (int st = 0; st < STREAMS; ++st) {
            wv_next[st] = *((const device uint2*)(wrow[st] + 32 * g0));
            wv_next2[st] = *((const device uint2*)(wrow[st] + 32 * min(g0 + 1, gmax)));
            s_next[st] = srow[st][g0];
            b_next[st] = brow[st][g0];
          }

        #pragma unroll
          for (int gi = 0; gi < gh; ++gi) {
            const int g = g0 + gi;
            const bool live = g < G;
            const int gc = min(g, gmax);

            uint2 wv_cur[STREAMS];
            float s_cur[STREAMS];
            float b_cur[STREAMS];
        #pragma clang loop unroll(full)
            for (int st = 0; st < STREAMS; ++st) {
              wv_cur[st] = wv_next[st];
              s_cur[st] = float(s_next[st]);
              b_cur[st] = float(b_next[st]);
            }
            const int g_next = min(g + 1, gmax);
            const int g_next2 = min(g + 2, gmax);
        #pragma clang loop unroll(full)
            for (int st = 0; st < STREAMS; ++st) {
              wv_next[st] = wv_next2[st];
              wv_next2[st] = *((const device uint2*)(wrow[st] + 32 * g_next2));
              s_next[st] = srow[st][g_next];
              b_next[st] = brow[st][g_next];
            }

            const uint4 r0 = *((const device uint4*)(x0 + 64 * gc));
            const uint4 r1 = *((const device uint4*)(x1 + 64 * gc));

            float2 rs = float2(mma8_runsum4<T>(r0), mma8_runsum4<T>(r1));
            rs += simd_shuffle_xor(rs, 2u);
            rs += simd_shuffle_xor(rs, 4u);
            rs += simd_shuffle_xor(rs, 16u);

            MMA8_SETB(B0, x, lo)
            MMA8_SETB(B1, x, hi)
            MMA8_SETB(B2, y, lo)
            MMA8_SETB(B3, y, hi)
            MMA8_SETB(B4, z, lo)
            MMA8_SETB(B5, z, hi)
            MMA8_SETB(B6, w, lo)
            MMA8_SETB(B7, w, hi)

            if (live) {
        #pragma clang loop unroll(full)
              for (int st = 0; st < STREAMS; ++st) {
                const uint2 wv = wv_cur[st];
                simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
                MMA8_STEP(B0, 0)
                MMA8_STEP(B1, 1)
                MMA8_STEP(B2, 2)
                MMA8_STEP(B3, 3)
                MMA8_STEP(B4, 4)
                MMA8_STEP(B5, 5)
                MMA8_STEP(B6, 6)
                MMA8_STEP(B7, 7)
                acc0[st] += s_cur[st] * C.thread_elements()[0] + rs.x * b_cur[st];
                acc1[st] += s_cur[st] * C.thread_elements()[1] + rs.y * b_cur[st];
              }
            }
          }

          // Cross-simdgroup close: simdgroup 0 adds simdgroup 1's partials in
          // stream order, then stores.
          if (simd_gid == 1) {
        #pragma clang loop unroll(full)
            for (int st = 0; st < STREAMS; ++st) {
              red[st * 32 + simd_lid] = float2(acc0[st], acc1[st]);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_gid == 1) {
            return;
          }
          const bool storeA = rowA < run_len;
          const bool storeB = rowB < run_len;
          const ulong yA = ulong(assignment + rowA) * ulong(N);
          const ulong yB = ulong(assignment + rowB) * ulong(N);
        #pragma clang loop unroll(full)
          for (int st = 0; st < STREAMS; ++st) {
            const float2 other = red[st * 32 + simd_lid];
            const float vA = acc0[st] + other.x;
            const float vB = acc1[st] + other.y;
            const int p = st / TILES;
            const int t = st - p * TILES;
            device T* yp = (p == 0) ? y0 : y1;
            const int col = n0 + t * 8 + c.fm;
            if (storeA) {
              yp[yA + ulong(col)] = static_cast<T>(vA);
            }
            if (storeB) {
              yp[yB + ulong(col)] = static_cast<T>(vB);
            }
          }
        }
        """

    /// Leader election + dispatch. Identical run semantics to the incumbent
    /// `affine_gather_qmv` election (`quantized.h`): sorted raw expert keys,
    /// backward scan for the run offset, leader = offset 0, run length capped
    /// at eight (top-8 over eight rows cannot repeat an expert inside a row).
    static func kernelSource(planes: Int, tiles: Int, k: Int, n: Int) -> String {
        let secondPlane = planes == 2
        return """
            const uint3 tid = threadgroup_position_in_grid;
            const uint assignment = tid.z;
            const uint expert = keys[assignment];
            uint run_offset = 0u;
            for (uint prior = assignment; prior > 0u; --prior) {
              if (keys[prior - 1u] != expert) {
                break;
              }
              run_offset += 1u;
            }
            if (run_offset != 0u) {
              return;
            }
            uint run_len = 1u;
            while (run_len < 8u && assignment + run_len < 64u &&
                   keys[assignment + run_len] == expert) {
              run_len += 1u;
            }
            threadgroup float2 red[\(planes * tiles * 32)];
            moe_gather_mma8_affine4_g64_impl<T, \(k), \(n), \(planes), \(tiles)>(
                w0, s0, b0,
                x, lhs,
                y0, \(secondPlane ? "y1" : "y0"),
                assignment, run_len, expert, int(tid.y) * \(8 * tiles), red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """
    }

    private static let gateUpKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_moe_gateup_gather_mma8_affine4_g64_k2816_n704_t\(tiles)_v1",
        inputNames: ["x", "lhs", "keys", "w0", "s0", "b0"],
        outputNames: ["y0", "y1"],
        source: kernelSource(planes: 2, tiles: tiles, k: modelDim, n: expertDim),
        header: kernelHeader,
        ensureRowContiguous: true)

    private static let downKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b8_moe_down_gather_mma8_affine4_g64_k704_n2816_t\(tiles)_v1",
        inputNames: ["x", "lhs", "keys", "w0", "s0", "b0"],
        outputNames: ["y0"],
        source: kernelSource(planes: 1, tiles: tiles, k: expertDim, n: modelDim),
        header: kernelHeader,
        ensureRowContiguous: true)

    // MARK: - Host

    struct Plane {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray

        /// Refuses anything but the pinned Gemma 4 expert geometry. `rows`
        /// is the storage row count per expert (704 for a split plane, 1408
        /// for the fused gate|up storage, 2816 for down).
        init?(
            weight: MLXArray, scales: MLXArray, biases: MLXArray?,
            groupSize: Int, bits: Int, mode: QuantizationMode,
            inDim: Int, rows: Int
        ) {
            guard groupSize == Gemma4MoEGatherMMA8V1.groupSize,
                bits == Gemma4MoEGatherMMA8V1.bits,
                mode == .affine,
                let biases,
                weight.dtype == .uint32,
                weight.shape == [Gemma4MoEGatherMMA8V1.experts, rows, inDim / 8],
                scales.dtype == .bfloat16,
                scales.shape == [Gemma4MoEGatherMMA8V1.experts, rows, inDim / 64],
                biases.dtype == .bfloat16,
                biases.shape == scales.shape
            else { return nil }
            self.weight = weight
            self.scales = scales
            self.biases = biases
        }
    }

    static func routeTablesAdmissible(rowOrder: MLXArray, sortedKeys: MLXArray) -> Bool {
        rowOrder.dtype == .uint32 && rowOrder.ndim == 1 && rowOrder.size == assignments
            && sortedKeys.dtype == .uint32 && sortedKeys.ndim == 1
            && sortedKeys.size == assignments
    }

    /// Gate and up projections in one launch over the loader's FUSED gate|up
    /// storage (`[128, 1408, 352]` uint32 with `[128, 1408, 44]` bf16 scales
    /// and biases; rows 0..<704 are gate, 704..<1408 are up). The split
    /// `gate_proj` / `up_proj` parameters are zero-copy row slices of that
    /// storage and therefore NOT row-contiguous, which is why the tier reads
    /// the storage itself: a custom kernel input that is not row-contiguous
    /// is copied on every call. `x` is the pre-norm cohort plane
    /// (`[8, 2816]` or `[8, 1, 2816]`), `rowOrder[a]` the cohort row of
    /// sorted assignment `a`, `sortedKeys[a]` its RAW expert id (a tagged
    /// prefix-bounds carrier is refused by the caller). Returns
    /// `[64, 1, 704]` gate and up planes, or nil when any pin fails.
    public static func gateUp(
        x: MLXArray, rowOrder: MLXArray, sortedKeys: MLXArray,
        fusedWeight: MLXArray, fusedScales: MLXArray, fusedBiases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        initNaN: Bool = false
    ) -> (gate: MLXArray, up: MLXArray)? {
        guard enabled,
            x.dtype == .bfloat16,
            x.size == cohortRows * modelDim,
            x.ndim == 2 || (x.ndim == 3 && x.dim(1) == 1),
            x.dim(0) == cohortRows, x.dim(x.ndim - 1) == modelDim,
            routeTablesAdmissible(rowOrder: rowOrder, sortedKeys: sortedKeys),
            let g = Plane(
                weight: fusedWeight, scales: fusedScales, biases: fusedBiases,
                groupSize: groupSize, bits: bits, mode: mode,
                inDim: modelDim, rows: 2 * expertDim)
        else { return nil }
        CBv2EngageMark.once("moe-gateup-mma8")
        let x2 = x.ndim == 2 ? x : x.reshaped([cohortRows, modelDim])
        let tileGroups = expertDim / (8 * tiles)
        let outs = gateUpKernel(
            [x2, rowOrder, sortedKeys, g.weight, g.scales, g.biases],
            template: [("T", DType.bfloat16)],
            grid: (simdWidth, tileGroups * simdGroups, assignments),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[assignments, expertDim], [assignments, expertDim]],
            outputDTypes: [.bfloat16, .bfloat16],
            initValue: initNaN ? Float.nan : nil)
        return (
            outs[0].reshaped([assignments, 1, expertDim]),
            outs[1].reshaped([assignments, 1, expertDim])
        )
    }

    /// Down projection of the activated plane (`[64, 1, 704]` or
    /// `[64, 704]`, assignment order). Returns `[64, 1, 2816]`, or nil.
    public static func down(
        activated x: MLXArray, sortedKeys: MLXArray, proj: QuantizedSwitchLinear,
        initNaN: Bool = false
    ) -> MLXArray? {
        guard enabled, downEnabled,
            x.dtype == .bfloat16,
            x.size == assignments * expertDim,
            x.ndim == 2 || (x.ndim == 3 && x.dim(1) == 1),
            x.dim(0) == assignments, x.dim(x.ndim - 1) == expertDim,
            sortedKeys.dtype == .uint32, sortedKeys.ndim == 1,
            sortedKeys.size == assignments,
            proj.bias == nil,
            let d = Plane(
                weight: proj.weight, scales: proj.scales, biases: proj.biases,
                groupSize: proj.groupSize, bits: proj.bits, mode: proj.mode,
                inDim: expertDim, rows: modelDim)
        else { return nil }
        CBv2EngageMark.once("moe-down-mma8")
        let x2 = x.ndim == 2 ? x : x.reshaped([assignments, expertDim])
        let tileGroups = modelDim / (8 * tiles)
        let out = downKernel(
            [x2, identity64, sortedKeys, d.weight, d.scales, d.biases],
            template: [("T", DType.bfloat16)],
            grid: (simdWidth, tileGroups * simdGroups, assignments),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[assignments, modelDim]],
            outputDTypes: [.bfloat16],
            initValue: initNaN ? Float.nan : nil)[0]
        return out.reshaped([assignments, 1, modelDim])
    }
}
