// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;

namespace mlx::steel {

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

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
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

} // namespace mlx::steel
