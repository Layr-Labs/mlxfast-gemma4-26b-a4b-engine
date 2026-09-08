import Foundation
import MLX
import MLXFast

/// Prompt-plane QKᵀ stage: the eight query-block score rectangles issued as
/// one launch. Sibling of `CBv2GroupedPrefillPVV1` (the eight value products).
///
/// The ranked prompt walk still produces eight score rectangles whose visible
/// key extent grows linearly with the block index (128, 256, …, 1024). Each
/// rectangle is the composed fallback's QKᵀ plus the causal-mask addMM
/// epilogue. This kernel keeps that arithmetic — the same NAX `gemm_loop`
/// body the steel twins use, `transpose_b` on the contiguous key plane, then
/// `acc + (−0.0)` on an admitted column and `acc + bfloat16.min` on a masked
/// column, stored as T — and only changes the launch: the block index rides
/// the grid's z axis so eight steel GEMMs become one encoder.
///
/// Decode (L = 1) never admits. Kill switch:
/// `DARKBLOOM_GEMMA4_GROUPED_PREFILL_QK=0`. Engage mark: `prefill-grouped-qk`.
enum CBv2GroupedPrefillQKV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_GROUPED_PREFILL_QK"] else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private struct Geometry {
        let bm: Int
        let bk: Int
        let wm: Int
        let swizzle: Int
    }

    private static let geometry: Geometry? = {
        #if canImport(Metal)
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return nil
        }
        let override = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        let architecture = override.isEmpty ? GPU.deviceInfo().architecture : override
        let suffix = architecture.last
        guard architecture.count >= 3,
            let generation = Int(architecture.dropLast().suffix(2)),
            generation >= (suffix == "p" ? 18 : 17)
        else { return nil }
        if suffix == "s" || suffix == "c" || suffix == "d" {
            return Geometry(bm: 64, bk: 256, wm: 2, swizzle: 2)
        }
        return Geometry(bm: 128, bk: 512, wm: 4, swizzle: 0)
        #else
        return nil
        #endif
    }()

    /// The grouped-PV NAX helpers plus the plain `gemm_loop` those helpers
    /// omit (they only host the softmax-loader twin) and the causal-add
    /// epilogue. Reopens `mlx::steel` after the PV header closes it.
    private static let header = CBv2GroupedPrefillPVNAXSourceV1.header + #"""

namespace mlx {
namespace steel {

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    typename AccumType = float>
auto gemm_loop(
    const device T* A,
    const device T* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const bool has_output = sgp_sm > 0 && sgp_sn > 0;
  (void)has_output;

  int gemm_k_iterations_ = gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    if constexpr (
        (DARKBLOOM_GEMMA4_NAX_SKIP_EMPTY != 0) &&
        (!kAlignedM || !kAlignedN)) {
      if (!has_output)
        continue;
    }

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;
      const int k = kk1;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else if constexpr (!transpose_a) {
        Atile.load_rows(A + A_offset, lda, sgp_sm);
      } else {
        Atile.load_safe(A + A_offset, lda, short2(sgp_sm, SK));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else if constexpr (transpose_b) {
        Btile.load_rows(B + B_offset, ldb, sgp_sn);
      } else {
        Btile.load_safe(B + B_offset, ldb, short2(sgp_sn, SK));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});
    }

    A += transpose_a ? (BK * lda) : BK;
    B += transpose_b ? BK : (BK * ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);
    if constexpr (
        (DARKBLOOM_GEMMA4_NAX_SKIP_EMPTY != 0) &&
        (!kAlignedM || !kAlignedN)) {
      if (!has_output)
        return Dtile;
    }

    const short rem_bk = K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;

      const int k = kk1;
      const short psk = max(0, rem_bk - k);

      const short2 Aklims =
          transpose_a ? short2(sgp_sm, psk) : short2(psk, sgp_sm);
      const short2 Bklims =
          transpose_b ? short2(psk, sgp_sn) : short2(sgp_sn, psk);

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      Atile.load_safe(A + A_offset, lda, Aklims);
      Btile.load_safe(B + B_offset, ldb, Bklims);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});
    }
  }

  return Dtile;
}

template <short TM, short TN>
METAL_FUNC void apply_causal_bias(
    thread NAXTile<float, TM, TN>& Dtile,
    const int qpos0,
    const int col0,
    const short sm,
    const short sn) {
  using Frag = typename NAXTile<float, TM, TN>::NAXFrag_t;
  const short2 sc = BaseNAXFrag::get_coord();
  const float open = -0.0f;
  const float shut = as_type<float>(0xFF7F0000u);
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      thread auto& frag = Dtile.frag_at(mm, nn);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Frag::kElemRows; i++) {
        const short row =
            mm * Frag::kFragRows + sc.y + i * Frag::kElemRowsJump;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < Frag::kElemCols; j++) {
          const short col = nn * Frag::kFragCols + sc.x + j;
          if (row < sm && col < sn) {
            const int qpos = qpos0 + int(row);
            const int key = col0 + int(col);
            frag[i * Frag::kElemCols + j] =
                frag[i * Frag::kElemCols + j]
                + (qpos >= key ? open : shut);
          }
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

"""#

    private static let source = #"""
        using namespace mlx::steel;
        const uint3 tid = threadgroup_position_in_grid;
        const int block = int(tid.z) / (BATCH * 16);
        const int batchHead = int(tid.z) % (BATCH * 16);
        const int b = batchHead / 16;
        const int h = batchHead % 16;
        const int tid_y = (int(tid.y) << SWIZZLE) +
            (int(tid.x) & ((1 << SWIZZLE) - 1));
        const int tid_x = int(tid.x) >> SWIZZLE;
        const int nTiles = block + 1;
        if (tid_y >= 128 / BM || tid_x >= nTiles) return;
        const int kL = nTiles * 128;
        constexpr short SM = BM / WM;
        constexpr short SN = 128 / 4;
        const short tm = SM * (simdgroup_index_in_threadgroup / 4);
        const short tn = SN * (simdgroup_index_in_threadgroup % 4);
        const int row = tid_y * BM + tm;
        const int col = tid_x * 128 + tn;
        const int kv = h / (16 / KVHEADS);
        const device T* Q = queries +
            ((size_t(b) * 16 + h) * 1024 + block * 128) * D;
        const device T* K = keys +
            (size_t(b) * KVHEADS + kv) * 1024 * D;
        device T* S = nullptr;
        switch (block) {
            case 0: S = s0; break;
            case 1: S = s1; break;
            case 2: S = s2; break;
            case 3: S = s3; break;
            case 4: S = s4; break;
            case 5: S = s5; break;
            case 6: S = s6; break;
            default: S = s7; break;
        }
        threadgroup_barrier(mem_flags::mem_none);
        S += (size_t(batchHead) * 128 + row) * kL + col;
        dispatch_bool(D % BK == 0, [&](auto alignedK) {
            auto tile = gemm_loop<
                T, SM, SN, 32, BK, false, true, true, true, alignedK.value, float>(
                Q + row * D, K + col * D, D, D, D, D / BK, SM, SN);
            apply_causal_bias(
                tile, block * 128 + row, col, SM, SN);
            tile.store(S, kL);
        });
        """#

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_grouped_prefill_qk_nax_v1",
        inputNames: ["queries", "keys"],
        outputNames: (0..<8).map { "s\($0)" },
        source: source,
        header: header,
        ensureRowContiguous: true)

    /// Eight score rectangles `[B, 16, 128, (block+1)*128]`, or nil.
    static func project(
        queries: MLXArray, keys: MLXArray, queryPlane: MLXArray
    ) -> [MLXArray]? {
        guard enabled, let geometry,
            queries.ndim == 4, keys.ndim == 4, queryPlane.ndim == 5,
            queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            queryPlane.dtype == .bfloat16,
            queries.dim(0) >= 1, queries.dim(0) <= 8,
            queries.dim(1) == 16, queries.dim(2) == 1024,
            keys.dim(0) == queries.dim(0),
            keys.dim(2) == 1024, keys.dim(3) == queries.dim(3),
            (keys.dim(1) == 8 && queries.dim(3) == 256)
                || (keys.dim(1) == 2 && queries.dim(3) == 512),
            queryPlane.dim(0) == queries.dim(0),
            queryPlane.dim(1) * queryPlane.dim(2) == 16,
            queryPlane.dim(3) == 1024,
            queryPlane.dim(4) == queries.dim(3)
        else { return nil }

        let batch = queries.dim(0)
        let dim = queries.dim(3)
        let kvHeads = keys.dim(1)
        let qFlat = queryPlane.reshaped([batch, 16, 1024, dim])
        let swizzleTile = 1 << geometry.swizzle
        let tilesN = 1024 / 128 * swizzleTile
        let tilesM = (128 / geometry.bm + swizzleTile - 1) / swizzleTile
        var outputShapes: [[Int]] = []
        outputShapes.reserveCapacity(8)
        for block in 0..<8 {
            outputShapes.append([batch, 16, 128, (block + 1) * 128])
        }
        CBv2EngageMark.once("prefill-grouped-qk")
        return kernel(
            [qFlat, keys],
            template: [("T", queries.dtype), ("BATCH", batch), ("D", dim),
                ("KVHEADS", kvHeads), ("BM", geometry.bm), ("BK", geometry.bk),
                ("WM", geometry.wm), ("SWIZZLE", geometry.swizzle)],
            grid: (tilesN * 32, tilesM * 4, batch * 16 * 8 * geometry.wm),
            threadGroup: (32, 4, geometry.wm),
            outputShapes: outputShapes,
            outputDTypes: Array(repeating: .bfloat16, count: 8))
    }
}
