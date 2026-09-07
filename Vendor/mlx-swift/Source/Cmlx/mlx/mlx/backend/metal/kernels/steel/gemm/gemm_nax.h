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

// DARKBLOOM GEMMA4 NAX TGP-BARRIER ELIDE.
// Both K loops of the accelerated GEMM family open every BK step with
// threadgroup_barrier(mem_flags::mem_none). That barrier is a vestige of the
// non-NAX steel GEMM, whose cooperative loaders stage A and B through the
// threadgroup arrays As/Bs and must fence readers against writers once per
// step. The NAX bodies stage nothing: gemm_loop and gemm_loop_softmax load A
// and B straight from device memory into per-simdgroup NAXTile registers,
// declare no threadgroup array, run no cooperative loader and write no
// threadgroup memory, and neither does any of the four NAX kernels that call
// them (steel_gemm_fused_nax.h, steel_gemm_gather_nax.h,
// steel_gemm_segmented_nax.h, steel_gemm_splitk_nax.h -- none declares a
// threadgroup array at all). mem_none requests no memory fence, so the call
// is a pure execution rendezvous ordering accesses to storage that does not
// exist, and it costs the threadgroup one full lockstep per BK step.
//
// EXACTNESS. Removing an execution barrier cannot change a value. Dtile is a
// register accumulator private to its simdgroup; A and B are read-only device
// operands; D is written after the loop into disjoint per-simdgroup extents.
// No simdgroup reads anything another simdgroup produced, so no output element
// depends on the order in which the simdgroups advance. Every element still
// accumulates over exactly the same k values in exactly the same sequence.
// The kernel's template parameters and every kernel-name string the host
// builds are untouched.
//
// COMPOSITION WITH NAX-SKIP-EMPTY. That mechanism's hazard argument places its
// `continue` strictly after this barrier so that a skipping simdgroup still
// reaches the barrier every iteration and threadgroup-uniform barrier arrival
// is preserved. With the elide on there is no barrier in the loop to arrive
// at, so the hazard the argument answers cannot arise: Metal constrains
// uniform arrival only at barriers that exist. The elide therefore weakens
// nothing in SKIP-EMPTY, and with the elide off the incumbent text and the
// incumbent argument both stand byte for byte.
//
// Kill switch: build with -DDARKBLOOM_GEMMA4_NAX_TGP_BARRIER_ELIDE=0 to
// restore the barrier at every site.
#ifndef DARKBLOOM_GEMMA4_NAX_TGP_BARRIER_ELIDE
#define DARKBLOOM_GEMMA4_NAX_TGP_BARRIER_ELIDE 1
#endif

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
#if !DARKBLOOM_GEMMA4_NAX_TGP_BARRIER_ELIDE
    threadgroup_barrier(mem_flags::mem_none);
#endif
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

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
      volatile int compiler_barrier;
#endif

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

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
      (void)compiler_barrier;
#endif
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
#if !DARKBLOOM_GEMMA4_NAX_TGP_BARRIER_ELIDE
    threadgroup_barrier(mem_flags::mem_none);
#endif
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

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
      volatile int compiler_barrier;
#endif

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

#if !DARKBLOOM_GEMMA4_NAX_VOLATILE_ELIDE
      (void)compiler_barrier;
#endif
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
