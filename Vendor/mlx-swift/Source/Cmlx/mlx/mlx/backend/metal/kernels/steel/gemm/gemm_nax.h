// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;

// Match the Compiled primitive's typed tape exactly. Swift converts every
// scalar literal to the array dtype, and each primitive writes a bfloat16
// temporary before the next primitive reads it.
template <typename T>
inline T gemma4_dense_geglu_compiled_tape(T gate, T up) {
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

// PREFILL-ATTN-TRAFFIC (at1): apply the prompt softmax's per-element
// expression T(fast::exp(float(s) - maxval) * normalizer) to the A
// fragments this lane holds (rows and columns inside (row_limit, col_limit)
// only -- elements outside were zero-filled by load_safe and stay zero, as in
// the plain load). rmax / rinv hold the lane's rows in fragment order:
// index mm * kElemRows + i for fragment row mm and element row i.
template <typename T, short RA, short CA>
METAL_FUNC void softmax_transform_atile(
    thread NAXTile<T, RA, CA>& Atile,
    const thread float* rmax,
    const thread float* rinv,
    const short2 sc,
    const short row_limit,
    const short col_limit) {
  using Frag = typename NAXTile<T, RA, CA>::NAXFrag_t;
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < RA; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < CA; kk++) {
      thread auto& frag = Atile.frag_at(mm, kk);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Frag::kElemRows; i++) {
        const short row =
            mm * Frag::kFragRows + sc.y + i * Frag::kElemRowsJump;
        const float m = rmax[mm * Frag::kElemRows + i];
        const float inv = rinv[mm * Frag::kElemRows + i];
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < Frag::kElemCols; j++) {
          const short col = kk * Frag::kFragCols + sc.x + j;
          if (row < row_limit && col < col_limit) {
            const float s = static_cast<float>(frag[i * Frag::kElemCols + j]);
            frag[i * Frag::kElemCols + j] =
                static_cast<T>(metal::fast::exp(s - m) * inv);
          }
        }
      }
    }
  }
}

// PREFILL-ATTN-TRAFFIC (at1): gemm_loop's twin for the composed prompt
// attention's P.V product (signature and exactness argument in
// steel_gemm_fused.h). Same loads, same tensor ops, same K order and
// accumulator; the one addition is that every A element this lane loads is
// replaced by T(fast::exp(float(s) - maxval) * normalizer) for its row
// before the tensor op consumes it. The row statistics sit at
// sm_stats + row * 4 (bf16 words carrying the fp32 bit patterns) relative to
// this simdgroup's first row; this lane's rows are mm * 16 + sc.y + i * 8.
// Non-transposed A only. gemm_loop itself is untouched.
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
auto gemm_loop_softmax(
    const device T* A,
    const device T* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn,
    const device T* sm_stats) {
  static_assert(!transpose_a, "at1: non-transposed A operand only");
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

  constexpr short kSmRows = TM * BaseNAXFrag::kElemRows;
  float sm_rmax[kSmRows];
  float sm_rinv[kSmRows];
  const short2 sm_sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
      const short row = mm * BaseNAXFrag::kFragRows + sm_sc.y +
          i * BaseNAXFrag::kElemRowsJump;
      const short r =
          metal::max(short(0), metal::min(row, short(sgp_sm - 1)));
      const uint2 w =
          *reinterpret_cast<const device uint2*>(sm_stats + r * 4);
      sm_rmax[mm * BaseNAXFrag::kElemRows + i] = as_type<float>(w.x);
      sm_rinv[mm * BaseNAXFrag::kElemRows + i] = as_type<float>(w.y);
    }
  }

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
      softmax_transform_atile(
          Atile,
          sm_rmax,
          sm_rinv,
          sm_sc,
          kAlignedM ? short(SM) : sgp_sm,
          short(SK));

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
      softmax_transform_atile(Atile, sm_rmax, sm_rinv, sm_sc, sgp_sm, psk);
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
