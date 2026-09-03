// Copyright © 2025 Apple Inc.

using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// DARKBLOOM GEMMA4 NAX SKIP-EMPTY.
// Restores NAX-SKIP-EMPTY-001 (upstream ml-explore/mlx 66a0407) behind a kill
// switch. A steel NAX GEMM simdgroup whose output extent is empty in either
// dimension (sgp_sm <= 0 or sgp_sn <= 0) stores nothing: the tail store is
// store_safe with a zero or negative extent, which writes no element. Such a
// simdgroup skips its A and B loads and its MMA instead of computing an
// accumulator that is then discarded.
//
// BARRIER SAFETY, the hazard this mechanism has to clear. The skip in the main
// K loop is placed strictly AFTER threadgroup_barrier(mem_flags::mem_none), so
// every simdgroup -- skipping or not -- still executes that barrier on every
// iteration, and the loop trip count gemm_k_iterations_ is threadgroup uniform.
// No threadgroup barrier is ever enclosed by the skip. In the unaligned-K tail
// the only synchronisation is a simdgroup_barrier, the skip is placed after it
// as well, and nothing after the early return contains any barrier at all.
// has_output is derived from sgp_sm and sgp_sn, which are simdgroup uniform, so
// all lanes of a simdgroup take the same branch and no intra-simdgroup
// divergence is introduced either.
//
// THREADGROUP MEMORY. gemm_loop reads A and B straight from device memory into
// per-simdgroup register tiles; it declares no threadgroup array, runs no
// cooperative loader and writes no threadgroup memory. A skipped simdgroup
// therefore produces nothing any other simdgroup reads.
//
// Kill switch: build with -DDARKBLOOM_GEMMA4_NAX_SKIP_EMPTY=0 and both guards
// fold to the incumbent unconditional form.
#ifndef DARKBLOOM_GEMMA4_NAX_SKIP_EMPTY
#define DARKBLOOM_GEMMA4_NAX_SKIP_EMPTY 1
#endif

// clang-format off
template <
    bool kAlignedM,
    bool kAlignedN,
    class NAXTile_t,
    typename T>
void gemm_epilogue(
    thread NAXTile_t& Dtile,
    const device T* C,
    const constant GEMMParams* params,
    const constant GEMMAddMMParams* addmm_params,
    const short sgp_sm, 
    const short sgp_sn) { // clang-format on

  (void)params;

  using V = typename NAXTile_t::elem_type;

  constexpr short TM = NAXTile_t::kTileRows;
  constexpr short TN = NAXTile_t::kTileCols;
  constexpr short kElemsPerFrag = NAXTile_t::kElemsPerFrag;

  using CFrag = typename NAXTile_t::NAXFrag_t;
  using cfrag_t = typename CFrag::template dtype_frag_t<T>;

  const_for_loop<0, TM, 1>([&](auto mm) {
    const_for_loop<0, TN, 1>([&](auto nn) {
      auto m = mm * Int<CFrag::kFragRows>{};
      auto n = nn * Int<CFrag::kFragCols>{};

      cfrag_t celems;

      if constexpr (kAlignedM && kAlignedN) {
        CFrag::load(celems, C, addmm_params->ldc, addmm_params->fdc, m, n);
      } else {
        CFrag::load_safe(
            celems,
            C,
            addmm_params->ldc,
            addmm_params->fdc,
            sgp_sm,
            sgp_sn,
            m,
            n);
      }

      thread auto& delems = Dtile.template frag_at<mm, nn>();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        if (do_axpby) {
          delems[i] = addmm_params->alpha * delems[i] +
              addmm_params->beta * static_cast<V>(celems[i]);
        } else {
          delems[i] += static_cast<V>(celems[i]);
        }
      }
    });
  });
}

// CAUSAL-CLOAD synthesized epilogue: adds the composed-prefill causal-bias
// constants the loaded operand would have supplied, per accumulator element,
// from the output coordinates the tile already knows. Element order, insert
// point, and widening match gemm_epilogue's non-axpby path exactly.
// clang-format off
template <class NAXTile_t>
void gemm_epilogue_causal_synth(
    thread NAXTile_t& Dtile,
    const int row0,
    const int col0,
    const int diag) { // clang-format on
  using V = typename NAXTile_t::elem_type;

  constexpr short TM = NAXTile_t::kTileRows;
  constexpr short TN = NAXTile_t::kTileCols;

  using CFrag = typename NAXTile_t::NAXFrag_t;

  const short2 sc = CFrag::get_coord();
  const V mask_add = static_cast<V>(as_type<float>(0xFF7F0000u));
  const V pass_add = static_cast<V>(-0.0f);

  const_for_loop<0, TM, 1>([&](auto mm) {
    const_for_loop<0, TN, 1>([&](auto nn) {
      thread auto& delems = Dtile.template frag_at<mm, nn>();

      const int mbase = row0 + sc.y + int(mm) * CFrag::kFragRows;
      const int nbase = col0 + sc.x + int(nn) * CFrag::kFragCols;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < CFrag::kElemRows; i++) {
        const int row = mbase + i * CFrag::kElemRowsJump;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < CFrag::kElemCols; j++) {
          const int col = nbase + j;
          delems[i * CFrag::kElemCols + j] +=
              (col - row <= diag) ? pass_add : mask_add;
        }
      }
    });
  });
}

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // CAUSAL-CLOAD eligibility (see steel_gemm_fused.h).
  constexpr bool kCausalBiasSynthEligible =
      !transpose_a && transpose_b && metal::is_same_v<T, bfloat16_t>;
  bool c_bstride_zero = true;

  // PREFILL-ATTN-TRAFFIC (at1) eligibility (signature and exactness argument
  // in steel_gemm_fused.h; the loader-side transform lives in
  // gemm_loop_softmax, gemm_nax.h).
  constexpr bool kSoftmaxLoaderEligible =
      !transpose_a && !transpose_b && metal::is_same_v<T, bfloat16_t>;
  bool softmax_loader = false;

  // Adjust for batch
  if (has_batch) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

    if (use_out_source) {
      const constant auto* C_bstrides = B_bstrides + params->batch_ndim;
      C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
      for (int d = 0; d < params->batch_ndim; d++) {
        c_bstride_zero = c_bstride_zero && (C_bstrides[d] == 0);
      }
    }
  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;

    if (use_out_source) {
      C += addmm_params->batch_stride_c * tid.z;
      c_bstride_zero = addmm_params->batch_stride_c == 0;
    }
  }

  D += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_M ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_N ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  D += tm * params->ldd + tn;

  if (use_out_source) {
    C += tm * addmm_params->ldc + tn * addmm_params->fdc;
  }

  const device T* sm_stats = nullptr;
  if constexpr (kSoftmaxLoaderEligible) {
    if (use_out_source) {
      if (do_axpby && addmm_params->fdc == 0 && addmm_params->ldc == 4 &&
          addmm_params->alpha == 1.0f &&
          as_type<uint>(addmm_params->beta) == 0x80000000u) {
        softmax_loader = true;
        sm_stats = C;
      }
    }
  }

  NAXTile<AccumType, TM, TN> Dtile;

  dispatch_bool(align_K, [&](auto kAlignedK) {
    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
        bool loop_done = false;
        if constexpr (kSoftmaxLoaderEligible) {
          if (softmax_loader) {
            // PREFILL-ATTN-TRAFFIC (at1): the softmax-in-loader twin.
            Dtile = gemm_loop_softmax<
                T,
                SM,
                SN,
                SK,
                BK,
                transpose_a,
                transpose_b,
                kAlignedM.value,
                kAlignedN.value,
                kAlignedK.value,
                AccumType>(
                A,
                B,
                params->lda,
                params->ldb,
                params->K,
                params->gemm_k_iterations_aligned,
                sgp_sm,
                sgp_sn,
                sm_stats);
            loop_done = true;
          }
        }
        if (!loop_done) {
          Dtile = gemm_loop<
              T,
              SM,
              SN,
              SK,
              BK,
              transpose_a,
              transpose_b,
              kAlignedM.value,
              kAlignedN.value,
              kAlignedK.value,
              AccumType>(
              A,
              B,
              params->lda,
              params->ldb,
              params->K,
              params->gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);
        }
        if ((DARKBLOOM_GEMMA4_NAX_SKIP_EMPTY == 0) ||
            ((kAlignedM.value || sgp_sm > 0) &&
             (kAlignedN.value || sgp_sn > 0))) {
          // PREFILL-ATTN-TRAFFIC (at1): a consumed statistics operand adds
          // nothing; the tile is stored as the plain matmul stores it.
          if (use_out_source && !softmax_loader) {
            bool synthesized = false;
            if constexpr (kCausalBiasSynthEligible) {
              if (!do_axpby && kAlignedM.value && kAlignedN.value &&
                  addmm_params->fdc == 1 &&
                  addmm_params->ldc == params->N + 1 &&
                  params->M <= params->N && c_bstride_zero) {
                // CAUSAL-CLOAD: signature and exactness argument in
                // steel_gemm_fused.h; the synthesized addend is bit-identical
                // to the loaded one on every element.
                gemm_epilogue_causal_synth(
                    Dtile, c_row + tm, c_col + tn, params->N - params->M);
                synthesized = true;
              }
            }
            if (!synthesized) {
              gemm_epilogue<kAlignedM.value, kAlignedN.value>(
                  Dtile, C, params, addmm_params, sgp_sm, sgp_sn);
            }
          }
          if constexpr (kAlignedM && kAlignedN) {
            Dtile.store(D, int(params->ldd));
          } else {
            Dtile.store_safe(D, int(params->ldd), short2(sgp_sn, sgp_sm));
          }
        }
      });
    });
  });
}
