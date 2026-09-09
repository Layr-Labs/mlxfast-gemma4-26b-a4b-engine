// Copyright © 2023-2024 Apple Inc. Vendored quantized NAX arithmetic retained.
// The compact fragment loader changes addresses only; the GEMM body is copied
// from quantized_nax.h with fixed aligned production geometry.
enum Gemma4CompactExpertInputNAXSourceV1 {
    static let header = CBv2GroupedPrefillPVNAXSourceV1.header + #"""
using namespace metal;
using namespace mlx::steel;
// Match the Compiled primitive's typed tape exactly. Swift converts every
// scalar literal to the array dtype, and each primitive writes a bfloat16
// temporary before the next primitive reads it.
template <typename T>
inline T gemma4_geglu_compiled_tape(T gate, T up) {
  const T cubic_0 = static_cast<T>(static_cast<T>(0.044715f) * gate);
  const T cubic_1 = static_cast<T>(cubic_0 * gate);
  const T cubic_2 = static_cast<T>(cubic_1 * gate);
  const T inner = static_cast<T>(gate + cubic_2);
  const T scaled =
      static_cast<T>(static_cast<T>(0.7978845608028654f) * inner);
  const T curved = metal::precise::tanh(scaled);
  const T shifted = static_cast<T>(static_cast<T>(1.0f) + curved);
  const T half_gate = static_cast<T>(static_cast<T>(0.5f) * gate);
  const T gelu = static_cast<T>(half_gate * shifted);
  return static_cast<T>(gelu * up);
}

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename U, int N, int bits>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

// Expert-segment elision for affine_gather_qmm_rhs_nax: the per-tile
// segment loop re-runs the full K-loop once per distinct expert in the
// row tile and discards out-of-segment rows at store_slice. The helpers
// below let a simdgroup skip A loads and MMA for 16-row NAX fragment
// rows that fall wholly outside the current segment's stored row band.
// Fragment rows are independent accumulators, so eliding rows that are
// never stored cannot change any stored element's accumulation sequence.
// Compile-time source constant by design: an enable must never ride a
// function constant magnitude (pipeline-key law).
MLX_MTL_CONST bool kGatherRhsSegmentElide = true;
MLX_MTL_CONST bool kGatherRhsSortedEndpointElide = true;
MLX_MTL_CONST bool kGatherRhsSegmentFenceElide = true;

// Loads one 16-row fragment row of an A tile from device memory. The
// address arithmetic matches NAXTile::load exactly for that fragment row
// (row offset mm * kFragRows), so the loaded values are identical to the
// full-tile load for the surviving rows.
template <typename U, typename ATile>
METAL_FUNC void gather_rhs_load_frag_row(
    const short mm,
    thread ATile& Atile,
    const device U* src,
    const int ld) {
  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < ATile::kTileCols; ++kk) {
    ATile::NAXFrag_t::load(
        Atile.frag_at(mm, kk),
        src,
        ld,
        Int<1>{},
        short(mm * ATile::kFragRows),
        short(kk * ATile::kFragCols));
  }
}

// Issues the mm-th fragment row's MMA op sequence of tile_matmad_nax's
// TN-even branch, unchanged: same operands, same per-fragment
// accumulation chain, only the dead fragment rows' ops are absent.
template <typename CTile, typename ATile, typename BTile, bool transpose_b>
METAL_FUNC void gather_rhs_mma_frag_row(
    const short mm,
    thread CTile& C,
    thread ATile& A,
    thread BTile& B,
    metal::bool_constant<transpose_b> tb) {
  constexpr short TN = CTile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  constexpr auto ta = metal::bool_constant<false>{};
  static_assert(TN % 2 == 0, "Segment elision expects even TN");
  STEEL_PRAGMA_UNROLL
  for (short nn = 0; nn < TN; nn += 2) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; ++kk) {
      CTile::NAXFrag_t::mma(
          C.frag_at(mm, nn),
          C.frag_at(mm, nn + 1),
          A.frag_at(mm, kk, ta),
          ta,
          B.frag_at(kk, nn, tb),
          B.frag_at(kk, nn + 1, tb),
          tb);
    }
  }
}

// DARKBLOOM GEMMA4 NAX GATHER-RHS ROW-STRIP TILING.
// affine_gather_qmm_rhs_nax covers a BM x BN output tile with WM x WN
// simdgroups. The launch shape is fixed by the host (32, WN, WM) and the host
// is not editable, so the threadgroup is always 4 simdgroups over a 64 x 64
// tile. Stock splits that tile 2 x 2, so each simdgroup owns 32 rows x 32
// cols and the two simdgroups that share a row band each fetch the SAME 32
// rows of the activation operand from device memory: A is read twice per
// threadgroup per K step. This constant instead lays the same 4 simdgroups
// out as 4 row strips of 16 rows x 64 cols. The strips are disjoint in M, so
// every A fragment is fetched exactly once, and the B operand -- which
// already lives in threadgroup memory as Ws -- is read wider instead.
//
// Nothing about the K loop moves. BK, SK and TK are untouched, the k, kk1 and
// k_remain loops keep their bounds and their order, and every output element
// still accumulates over exactly the same k values in exactly the same
// sequence. Only which simdgroup owns an element, and how the owner's
// fragments are shaped, change.
//
// COMPOSITION WITH THE SEGMENT ELISION ON THIS KERNEL. The elision is
// expressed at Dtile.kFragRows (16 row) granularity and stays at exactly that
// granularity here: stock gives a simdgroup TM = 2 fragment rows of a 32 row
// band, the strip layout gives TM = 1 fragment row of a 16 row band, and the
// union over the 4 simdgroups is the same 64 rows either way. The live-band
// guard fr < seg_hi && fr + kFragRows > seg_lo tests fr and seg_lo/seg_hi in
// the same tm-relative frame in both layouts, so it decides the same
// intersection of absolute rows against the same segment. offset and
// offset_next stay threadgroup uniform, seg_lo/seg_hi stay simdgroup uniform,
// and gather_rhs_mma_frag_row keeps issuing exactly the TN-even op sequence
// of the shared helper, so the partial-band path and the full path still
// agree op for op. Narrowing the band from 32 rows to 16 can only move a band
// from partial to whole or to empty; it can never make a whole band partial,
// so the elision's own correctness argument is unweakened.
//
// The kernel's template parameters, and therefore every kernel-name string
// the host builds, are untouched: BM, BN, BK, WM and WN all keep their values
// and only the interior mapping is re-derived from them.
//
// Kill switch: build with -DDARKBLOOM_GEMMA4_NAX_GATHER_TILING=0 and SGM/SGN
// fold back to WM/WN, reproducing the shipped expressions byte for byte.
// Independent of the qmm-t family's switch.
#ifndef DARKBLOOM_GEMMA4_NAX_GATHER_TILING
#define DARKBLOOM_GEMMA4_NAX_GATHER_TILING 1
#endif

// DARKBLOOM GEMMA4 NAX VOLATILE-FENCE ELIDE.
// Every K-step loop body in the accelerated GEMM family declares an
// uninitialised volatile int that is never written and is read once through
// a discarded-value cast. With the elide on, neither the declaration nor the
// read is emitted; no value in the kernel is derived from it.
// Kill switch: build with -DDARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE=0 to restore
// the incumbent declaration and read at every site.
#ifndef DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
#define DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE 1
#endif


template <typename U, typename ATile>
METAL_FUNC void compact_load_frag_row(
    short mm, thread ATile& tile, const device U* src, int ld,
    const device uint32_t* rows, short valid_rows) {
  using Frag = typename ATile::NAXFrag_t;
  const short2 sc = Frag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < ATile::kTileCols; ++kk) {
    thread auto& dst = tile.frag_at(mm, kk);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < Frag::kElemRows; ++i) {
      const short row = sc.y + mm * ATile::kFragRows + i * Frag::kElemRowsJump;
      // Match NAXFrag::load's lane/element assignment, changing only row addressing.
      const size_t src_row = row < valid_rows ? size_t(rows[row]) : 0;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < Frag::kElemCols; ++j) {
        const short col = sc.x + kk * ATile::kFragCols + j;
        dst[i * Frag::kElemCols + j] =
            row < valid_rows ? src[src_row * size_t(ld) + col] : U(0);
      }
    }
  }
}
template <typename U, typename ATile>
METAL_FUNC void compact_load_tile(
    thread ATile& tile, const device U* src, int ld,
    const device uint32_t* rows, short valid_rows) {
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < ATile::kTileRows; ++mm) {
    compact_load_frag_row(mm, tile, src, ld, rows, valid_rows);
  }
}

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
METAL_FUNC void compact_gather_qmm_rhs_nax(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device uint32_t* indices,
    const device uint32_t* row_order,
    device T* y,
    const int M,
    const int N,
    const int K,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id,
    threadgroup T* Ws) {
  constexpr bool align_M = true, align_N = true, align_K = true;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Shared weight tile is allocated by the enclosing kernel.

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  // The host dispatch surface is trusted and not editable, so admission is a
  // uniform runtime predicate inside the kernel. All template terms fold at
  // compile time; only the exact target geometry reaches the compact close.
  const bool gemma4_gather_rhs_geglu =
      transpose && metal::is_same_v<T, bfloat> && group_size == 64 &&
      bits == 4 && M >= 512 && N == 1408 && K == 2816;
  const bool segment_fence_elide = kGatherRhsSegmentFenceElide &&
      transpose && metal::is_same_v<T, bfloat> && group_size == 64 &&
      bits == 4 && M >= 512 && K_it > 0 &&
      ((N == 1408 && K == 2816) || (N == 2816 && K == 704));
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  // x stays compact; row_order supplies each fragment's source row.
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  // Simdgroup grid over the BM x BN tile. Stock is WM x WN; the row-strip
  // layout stacks the same WM*WN simdgroups in the row direction only, so no
  // two of them share a row band. See the note on the enable above, including
  // why this leaves the segment elision's granularity and guard unchanged.
  constexpr int SGM =
      (DARKBLOOM_GEMMA4_NAX_GATHER_TILING != 0) ? (WM * WN) : WM;
  constexpr int SGN = (DARKBLOOM_GEMMA4_NAX_GATHER_TILING != 0) ? 1 : WN;
  static_assert(SGM * SGN == WM * WN, "simdgroup count must be preserved");
  static_assert(BM % (SGM * 16) == 0, "row strip must be a fragment multiple");
  static_assert(BN % (SGN * 16) == 0, "col strip must be a fragment multiple");

  constexpr short SM = BM / SGM;
  constexpr short SN = BN / SGN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  // gather_rhs_mma_frag_row issues the shared helper's TN-even op sequence and
  // has no branch for an odd TN; an odd TN would silently emit no arithmetic.
  static_assert(TN % 2 == 0, "gather segment elision requires an even TN");

  const short tm = SM * (simd_group_id / SGN);
  const short tn = SN * (simd_group_id % SGN);

  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    // gather_qmm_rhs is dispatched only for right-sorted indices. If this
    // segment's expert matches the tile endpoint, sortedness proves that the
    // remaining suffix is one segment and the per-row probe can stop here.
    if (kGatherRhsSortedEndpointElide &&
        indices[y_row + tgp_bm - 1] == index) {
      n = tgp_bm;
    } else {
      for (; n < tgp_bm; n++) {
        if (indices[y_row + n] != index) {
          offset_next = n;
          index_next = indices[y_row + n];
          break;
        }
      }
    }
    // The first retained K barrier already rendezvous all loaders.
    if (!segment_fence_elide) {
      threadgroup_barrier(mem_flags::mem_none);
    }

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x;
    const device uint32_t* source_rows = row_order + y_row + tm;

    // This simdgroup's stored row band for the current expert segment,
    // hoisted ahead of the K-loop (it depends only on offset, offset_next,
    // tm and sgp_sm, all known here). The stock path computes the full
    // tile and discards rows outside [seg_lo, seg_hi) at store_slice; with
    // the elision enabled those rows' A loads and MMA ops are skipped
    // instead. Cooperative weight loads and every threadgroup_barrier stay
    // unconditional, so barrier convergence is preserved, and seg_* are
    // uniform within a simdgroup (offset/offset_next are threadgroup
    // uniform). With the enable off both flags fold to false and only the
    // stock path below runs.
    const short seg_lo = min(int(sgp_sm), max(0, offset - tm));
    const short seg_hi = min(int(sgp_sm), max(0, offset_next - tm));
    const bool seg_empty = kGatherRhsSegmentElide && (seg_hi <= seg_lo);
    const bool seg_partial = kGatherRhsSegmentElide && !seg_empty &&
        !(seg_lo == 0 && seg_hi == sgp_sm);

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (seg_partial && kAlignedM.value) {
            // 16-row fragment-row granularity: only fragment rows that
            // intersect [seg_lo, seg_hi) load A and issue MMA. Each live
            // fragment row runs the exact op sequence of the stock path.
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              volatile int compiler_barrier;
#endif

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              STEEL_PRAGMA_UNROLL
              for (short mm = 0; mm < TM; mm++) {
                const short fr = short(mm * Dtile.kFragRows);
                if (fr < seg_hi && short(fr + Dtile.kFragRows) > seg_lo) {
                  compact_load_frag_row(mm, Atile, xn + kk1, K, source_rows, sgp_sm);
                  gather_rhs_mma_frag_row(
                      mm,
                      Dtile,
                      Atile,
                      Btile,
                      metal::bool_constant<transpose>{});
                }
              }

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              (void)compiler_barrier;
#endif
            }
          } else if (!seg_empty) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              volatile int compiler_barrier;
#endif

              if constexpr (kAlignedM.value) {
                compact_load_tile(Atile, xn + kk1, K, source_rows, sgp_sm);
              } else {
                compact_load_tile(Atile, xn + kk1, K, source_rows, sgp_sm);
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              (void)compiler_barrier;
#endif
            }
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Elision here is band-granular only (seg_empty): a partial band
          // runs the stock tail, whose extra MMA lands in fragment rows
          // that are never stored.
          if (!seg_empty) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              volatile int compiler_barrier;
#endif

              const short psk = min(int(SK), max(0, (BK - kk1)));
              compact_load_tile(Atile, xn + kk1, K, source_rows, sgp_sm);

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
              (void)compiler_barrier;
#endif
            }
          }
        }

        // Stores read private accumulators only. The next K-entry barrier
        // protects Ws before the next segment can overwrite shared weights.
        if (!segment_fence_elide) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // The exact production arm lays this 64-column tile out as adjacent
        // 16-column gate/up pairs. Round both GEMM closes to T, then reproduce
        // every bfloat16 temporary of the compiled GeGLU tape. Compact rows
        // occupy the physical prefix of the ordinary 1408-wide allocation.
        if (gemma4_gather_rhs_geglu) {
          static_assert(TN % 2 == 0, "GeGLU epilogue requires paired fragments");
          NAXTile<AccumType, TM, TN / 2> Otile;
          const_for_loop<0, TM, 1>([&](auto mm) {
            const_for_loop<0, TN / 2, 1>([&](auto nn) {
              thread auto& gate =
                  Dtile.frag_at(short(mm), short(nn) * 2);
              thread auto& up =
                  Dtile.frag_at(short(mm), short(nn) * 2 + 1);
              thread auto& out = Otile.frag_at(short(mm), short(nn));
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < Dtile.kElemsPerFrag; ++i) {
                const T g = static_cast<T>(gate[i]);
                const T u = static_cast<T>(up[i]);
                out[i] = float(gemma4_geglu_compiled_tape(g, u));
              }
            });
          });
          if (!seg_empty) {
            device T* compact_y =
                y - y_row_long * N - y_col_long +
                y_row_long * (N / 2) + size_t(tid.x) * (BN / 2) +
                tm * (N / 2) + tn / 2;
            if (seg_lo == 0 && seg_hi == SM) {
              Otile.store(compact_y, N / 2);
            } else {
              Otile.store_slice(
                  compact_y,
                  N / 2,
                  short2(0, seg_lo),
                  short2(SN / 2, seg_hi));
            }
          }
        } else {
          // Store results to device memory. seg_lo/seg_hi are the stock
          // m_lo_lim/m_hi_lim, hoisted ahead of the K-loop.
          if (!seg_empty) {
            if constexpr (kAlignedN.value) {
              if (seg_lo == 0 && seg_hi == SM) {
                Dtile.store(y + tm * N + tn, N);
              } else {
                Dtile.store_slice(
                    y + tm * N + tn, N, short2(0, seg_lo), short2(SN, seg_hi));
              }
            } else {
              Dtile.store_slice(
                  y + tm * N + tn,
                  N,
                  short2(0, seg_lo),
                  short2(sgp_sn, seg_hi));
            }
          }
        }
      });
    });
  }
}

"""#
}
