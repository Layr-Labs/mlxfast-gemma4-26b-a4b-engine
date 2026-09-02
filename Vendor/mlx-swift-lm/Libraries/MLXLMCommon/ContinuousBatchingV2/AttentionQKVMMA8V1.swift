// Tight-grid host for the promoted Q/K/V matrix-unit tier
// (`gemma4_qmv_mma8_affine4_g64_impl`, mlx-generated/quantized.cpp). The
// frozen MLX host launches `grid.x = M = 8` x-groups for these shapes and the
// tier keeps only `tid.x == 0`, so 7 of 8 threadgroups exist to hit an early
// return: q_proj full (N=8192) launches 8192 threadgroups where 1024 work.
// This host dispatches the SAME kernel text with grid.x = 1 — zero dead
// launches, identical arithmetic, identical weight traffic (already one pass).
// The pattern is the established tight-grid family (TiedLMHeadQMVV1,
// DenseMLPQMVV1, AttentionOQMVV1).
//
// MMA-MT-001 adds a second body under the same host and the same guards: two
// eight-column output tiles per simdgroup sharing one build of the x-side
// simdgroup operands and one run-sum reduction per K group. Same arithmetic,
// same accumulation order, bit-identical outputs;
// `DARKBLOOM_GEMMA4_QKV_MMA8_MULTITILE=0` restores the single-tile dispatch.
//
// MMA-RS-001 adds a run-sum PREPASS. The affine bias term `rs` depends only
// on x and the group index, so its value is identical for every output tile
// of every projection that consumes the same x; the bodies above recompute it
// per K group per tile (or tile pair). The prepass computes each (row, group)
// run sum once, with the identical expression tree and butterfly stage order,
// into a `[8, K/64]` fp32 table the main bodies read instead. Same values in
// the same accumulator steps, bit-identical outputs;
// `DARKBLOOM_GEMMA4_MMA8_RS_PREPASS=0` restores the in-kernel reductions.
// On QKFUSE-001 the fused Q|K dispatch reads the same table: its entries are
// per activation row and per 64-group of K, independent of N, so the
// concatenated-N dispatch consumes the identical floats.

import Foundation
import MLX
import MLXFast

public enum CBv2AttentionQKVMMA8V1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_QKV_MMA8_TIGHT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// MMA-RS-001 arm. Default ON. Applies to the Q/K/V host and the o_proj
    /// host below; both read the same env so a submission runs one policy.
    /// `DARKBLOOM_GEMMA4_MMA8_RS_PREPASS=0` restores the in-kernel run-sum
    /// reductions byte for byte in the same executables.
    public static let rsPrepassEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MMA8_RS_PREPASS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let batch = 8
    private static let sequence = 1
    private static let inputWidth = 2816
    private static let groupSize = 64
    private static let bits = 4
    private static let simdWidth = 32
    private static let simdGroups = 2
    private static let outputsPerGroup = 8

    private static let mma8KernelHeader = """
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

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

template <typename T, int KS, int KFIX>
METAL_FUNC void qkv_mma8_affine4_g64_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow =
      (const device uint8_t*)w + (n0 + c.fm) * (K / 2) + 4 * c.fn;
  const device T* srow = scales + (n0 + c.fm) * G;
  const device T* brow = biases + (n0 + c.fm) * G;
  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

  uint2 wv_next = *((const device uint2*)(wrow + 32 * g0));
  uint2 wv_next2 =
      *((const device uint2*)(wrow + 32 * (g0 + min(1, nGroups - 1))));
  T s_next = srow[g0];
  T b_next = brow[g0];

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;
    const uint2 wv = wv_next;
    const float s = float(s_next);
    const float b = float(b_next);
    const int g_next = g0 + min(gi + 1, nGroups - 1);
    const int g_next2 = g0 + min(gi + 2, nGroups - 1);
    wv_next = wv_next2;
    wv_next2 = *((const device uint2*)(wrow + 32 * g_next2));
    s_next = srow[g_next];
    b_next = brow[g_next];

    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

    float2 rs = float2(mma8_runsum4<T>(r0), mma8_runsum4<T>(r1));
    rs += simd_shuffle_xor(rs, 2u);
    rs += simd_shuffle_xor(rs, 4u);
    rs += simd_shuffle_xor(rs, 16u);

    // Each operand is filled immediately before the step that reads it, so
    // one is live at a time instead of eight. `r0`/`r1` are not written
    // across the run, so every fill yields the value it yielded before and
    // the eight steps keep their order into `C`. The multitile body keeps
    // its hoisted fills: there one fill set serves both tiles.
    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    MMA8_SETB(B0, x, lo)
    MMA8_STEP(B0, 0)
    MMA8_SETB(B1, x, hi)
    MMA8_STEP(B1, 1)
    MMA8_SETB(B2, y, lo)
    MMA8_STEP(B2, 2)
    MMA8_SETB(B3, y, hi)
    MMA8_STEP(B3, 3)
    MMA8_SETB(B4, z, lo)
    MMA8_STEP(B4, 4)
    MMA8_SETB(B5, z, hi)
    MMA8_STEP(B5, 5)
    MMA8_SETB(B6, w, lo)
    MMA8_STEP(B6, 6)
    MMA8_SETB(B7, w, hi)
    MMA8_STEP(B7, 7)

    acc0 += s * C.thread_elements()[0] + rs.x * b;
    acc1 += s * C.thread_elements()[1] + rs.y * b;
  }

  if (KS == 2) {
    if (simd_gid == 1) {
      red[simd_lid] = float2(acc0, acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
    const float2 other = red[simd_lid];
    acc0 = acc0 + other.x;
    acc1 = acc1 + other.y;
  }

  y[c.fn * N + n0 + c.fm] = static_cast<T>(acc0);
  y[(c.fn + 1) * N + n0 + c.fm] = static_cast<T>(acc1);
}

// MMA-MT-001: the promoted Q/K/V matrix-unit body with TWO eight-column output
// tiles per simdgroup instead of one.
//
// The incumbent rebuilds the whole x side of the product for every output
// tile: two `uint4` activation loads, two `mma8_runsum4` reductions (sixteen
// bf16 -> fp32 widenings and fourteen adds), three `simd_shuffle_xor`
// butterflies and eight `simdgroup_float8x8` operand fills, per K group, per
// tile. Only the weight side differs between tiles. At N = 8192 that x-side
// work runs 1024 times per projection over the same eight activation rows.
//
// This body hoists it out of the tile dimension: one fragment build per K
// group feeds both tiles, and the eight `simdgroup_multiply_accumulate` chains
// per tile are the only thing that repeats.
//
// Bit-exactness. `rs` and B0..B7 are values, not addresses, produced by the
// same expressions in the same order as the incumbent, so both tiles see
// exactly the operands the incumbent's own two threadgroups saw. Each tile
// keeps a private `C`, a private `s`/`b` pair and a private accumulator, and
// accumulates over the same ascending `g` range with the identical
// `acc += s * C.thread_elements()[i] + rs[i] * b` step. KS = 2 keeps the
// identical [0, gh) / [gh, G) split across the threadgroup's two simdgroups
// and the identical simdgroup-0-adds-simdgroup-1 close, per tile. Every output
// word is therefore the same float sum accumulated in the same order.
template <typename T, int KS, int TILES, int KFIX, int SPLIT = 0>
METAL_FUNC void qkv_mma8_affine4_g64_mt(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid,
    device T* y2 = nullptr) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow[TILES];
  const device T* srow[TILES];
  const device T* brow[TILES];
  thread float acc0[TILES];
  thread float acc1[TILES];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    wrow[t] = (const device uint8_t*)w + (nt + c.fm) * (K / 2) + 4 * c.fn;
    srow[t] = scales + (nt + c.fm) * G;
    brow[t] = biases + (nt + c.fm) * G;
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

  // The per-group weight operands are carried in registers: group g0's
  // packed word, scale and bias are read before the walk, and each trip
  // reads the next group's while the current group's stay resident. The
  // addresses are functions of the group index alone, so the value each
  // trip consumes is the value the in-place read produced. The clamp on
  // `g_next` keeps the last trip inside the simdgroup's group range; the
  // value it re-reads is discarded at loop exit.
  uint2 wv_next[TILES];
  uint2 wv_next2[TILES];
  T s_next[TILES];
  T b_next[TILES];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    wv_next[t] = *((const device uint2*)(wrow[t] + 32 * g0));
    wv_next2[t] =
        *((const device uint2*)(wrow[t] + 32 * (g0 + min(1, nGroups - 1))));
    s_next[t] = srow[t][g0];
    b_next[t] = brow[t][g0];
  }

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;

    uint2 wv_cur[TILES];
    float s_cur[TILES];
    float b_cur[TILES];
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      wv_cur[t] = wv_next[t];
      s_cur[t] = float(s_next[t]);
      b_cur[t] = float(b_next[t]);
    }
    const int g_next = g0 + min(gi + 1, nGroups - 1);
    const int g_next2 = g0 + min(gi + 2, nGroups - 1);
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      wv_next[t] = wv_next2[t];
      wv_next2[t] = *((const device uint2*)(wrow[t] + 32 * g_next2));
      s_next[t] = srow[t][g_next];
      b_next[t] = brow[t][g_next];
    }

    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

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

#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      const uint2 wv = wv_cur[t];
      const float s = s_cur[t];
      const float b = b_cur[t];

      simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
      MMA8_STEP(B0, 0)
      MMA8_STEP(B1, 1)
      MMA8_STEP(B2, 2)
      MMA8_STEP(B3, 3)
      MMA8_STEP(B4, 4)
      MMA8_STEP(B5, 5)
      MMA8_STEP(B6, 6)
      MMA8_STEP(B7, 7)

      acc0[t] += s * C.thread_elements()[0] + rs.x * b;
      acc1[t] += s * C.thread_elements()[1] + rs.y * b;
    }
  }

  if (KS == 2) {
    if (simd_gid == 1) {
#pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        red[t * 32 + simd_lid] = float2(acc0[t], acc1[t]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      const float2 other = red[t * 32 + simd_lid];
      acc0[t] = acc0[t] + other.x;
      acc1[t] = acc1[t] + other.y;
    }
  }

#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    if (SPLIT == 0) {
      y[c.fn * N + nt + c.fm] = static_cast<T>(acc0[t]);
      y[(c.fn + 1) * N + nt + c.fm] = static_cast<T>(acc1[t]);
    } else {
      // Fused Q||K plane: columns below SPLIT belong to Q, the rest to K.
      // Both rows of a store pair share one column, so the branch is uniform.
      const int col = nt + c.fm;
      if (col < SPLIT) {
        y[c.fn * SPLIT + col] = static_cast<T>(acc0[t]);
        y[(c.fn + 1) * SPLIT + col] = static_cast<T>(acc1[t]);
      } else {
        const int n2 = N - SPLIT;
        const int c2 = col - SPLIT;
        y2[c.fn * n2 + c2] = static_cast<T>(acc0[t]);
        y2[(c.fn + 1) * n2 + c2] = static_cast<T>(acc1[t]);
      }
    }
  }
}

// MMA-RS-001: the single-tile body with the affine bias run sums read from a
// precomputed `[8, G]` fp32 table instead of rebuilt per K group. The table's
// (row, g) entry is the value the incumbent's own runsum4 pair and 2/4/16
// butterfly produce for that row and group, so the `acc += s * C + rs[i] * b`
// step consumes the identical float. Everything else is the incumbent text.
template <typename T, int KS, int KFIX>
METAL_FUNC void qkv_mma8_affine4_g64_rsp(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device float* rs_table,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow =
      (const device uint8_t*)w + (n0 + c.fm) * (K / 2) + 4 * c.fn;
  const device T* srow = scales + (n0 + c.fm) * G;
  const device T* brow = biases + (n0 + c.fm) * G;
  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;
    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

    const float2 rs = float2(
        rs_table[c.fn * G + g], rs_table[(c.fn + 1) * G + g]);

    MMA8_SETB(B0, x, lo)
    MMA8_SETB(B1, x, hi)
    MMA8_SETB(B2, y, lo)
    MMA8_SETB(B3, y, hi)
    MMA8_SETB(B4, z, lo)
    MMA8_SETB(B5, z, hi)
    MMA8_SETB(B6, w, lo)
    MMA8_SETB(B7, w, hi)

    const uint2 wv = *((const device uint2*)(wrow + 32 * g));
    const float s = float(srow[g]);
    const float b = float(brow[g]);

    simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
    MMA8_STEP(B0, 0)
    MMA8_STEP(B1, 1)
    MMA8_STEP(B2, 2)
    MMA8_STEP(B3, 3)
    MMA8_STEP(B4, 4)
    MMA8_STEP(B5, 5)
    MMA8_STEP(B6, 6)
    MMA8_STEP(B7, 7)

    acc0 += s * C.thread_elements()[0] + rs.x * b;
    acc1 += s * C.thread_elements()[1] + rs.y * b;
  }

  if (KS == 2) {
    if (simd_gid == 1) {
      red[simd_lid] = float2(acc0, acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
    const float2 other = red[simd_lid];
    acc0 = acc0 + other.x;
    acc1 = acc1 + other.y;
  }

  y[c.fn * N + n0 + c.fm] = static_cast<T>(acc0);
  y[(c.fn + 1) * N + n0 + c.fm] = static_cast<T>(acc1);
}

// MMA-RS-001: the two-tile MMA-MT-001 body with the run sums read from the
// precomputed table. One table entry pair per (K group, row pair) replaces the
// per-group runsum4 pair and butterfly; the fragment build, the weight side,
// the accumulators and the KS = 2 close keep the MT body's text, so every
// output word is the same float sum in the same order.
// QKFUSE-001 merge: the SPLIT/y2 store split mirrors the MT body's, so the
// fused concatenated-N dispatch can consume this rsp body unchanged.
template <typename T, int KS, int TILES, int KFIX, int SPLIT = 0>
METAL_FUNC void qkv_mma8_affine4_g64_mt_rsp(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device float* rs_table,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid,
    device T* y2 = nullptr) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow[TILES];
  const device T* srow[TILES];
  const device T* brow[TILES];
  thread float acc0[TILES];
  thread float acc1[TILES];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    wrow[t] = (const device uint8_t*)w + (nt + c.fm) * (K / 2) + 4 * c.fn;
    srow[t] = scales + (nt + c.fm) * G;
    brow[t] = biases + (nt + c.fm) * G;
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  const device T* x0 = x + c.fn * K + 8 * c.fm;
  const device T* x1 = x0 + K;

  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;
    const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
    const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

    const float2 rs = float2(
        rs_table[c.fn * G + g], rs_table[(c.fn + 1) * G + g]);

    MMA8_SETB(B0, x, lo)
    MMA8_SETB(B1, x, hi)
    MMA8_SETB(B2, y, lo)
    MMA8_SETB(B3, y, hi)
    MMA8_SETB(B4, z, lo)
    MMA8_SETB(B5, z, hi)
    MMA8_SETB(B6, w, lo)
    MMA8_SETB(B7, w, hi)

#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      const uint2 wv = *((const device uint2*)(wrow[t] + 32 * g));
      const float s = float(srow[t][g]);
      const float b = float(brow[t][g]);

      simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
      MMA8_STEP(B0, 0)
      MMA8_STEP(B1, 1)
      MMA8_STEP(B2, 2)
      MMA8_STEP(B3, 3)
      MMA8_STEP(B4, 4)
      MMA8_STEP(B5, 5)
      MMA8_STEP(B6, 6)
      MMA8_STEP(B7, 7)

      acc0[t] += s * C.thread_elements()[0] + rs.x * b;
      acc1[t] += s * C.thread_elements()[1] + rs.y * b;
    }
  }

  if (KS == 2) {
    if (simd_gid == 1) {
#pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        red[t * 32 + simd_lid] = float2(acc0[t], acc1[t]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      const float2 other = red[t * 32 + simd_lid];
      acc0[t] = acc0[t] + other.x;
      acc1[t] = acc1[t] + other.y;
    }
  }

#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    if (SPLIT == 0) {
      y[c.fn * N + nt + c.fm] = static_cast<T>(acc0[t]);
      y[(c.fn + 1) * N + nt + c.fm] = static_cast<T>(acc1[t]);
    } else {
      // Fused Q||K plane: columns below SPLIT belong to Q, the rest to K.
      // Both rows of a store pair share one column, so the branch is uniform.
      const int col = nt + c.fm;
      if (col < SPLIT) {
        y[c.fn * SPLIT + col] = static_cast<T>(acc0[t]);
        y[(c.fn + 1) * SPLIT + col] = static_cast<T>(acc1[t]);
      } else {
        const int n2 = N - SPLIT;
        const int c2 = col - SPLIT;
        y2[c.fn * n2 + c2] = static_cast<T>(acc0[t]);
        y2[(c.fn + 1) * n2 + c2] = static_cast<T>(acc1[t]);
      }
    }
  }
}
"""


    /// C2X-RSP: the rsp bodies with the packed codes carried two groups
    /// deep and the activation run, the run-sum pair, the scale and the bias
    /// carried one group ahead. Derived from the rsp text above at load time;
    /// every address is a function of the group index alone and every
    /// arithmetic statement is the rsp body's own.
    private static let mma8C2XKernelHeader: String = {
        func slice(_ from: String, to: String?) -> String {
            let start = mma8KernelHeader.range(of: from)!
            let end = to.flatMap { mma8KernelHeader.range(of: $0, range: start.upperBound ..< mma8KernelHeader.endIndex) }
            return String(mma8KernelHeader[start.lowerBound ..< (end?.lowerBound ?? mma8KernelHeader.endIndex)])
        }
        func replaceOnce(_ text: inout String, _ old: String, with new: String) {
            precondition(text.components(separatedBy: old).count == 2)
            text = text.replacingOccurrences(of: old, with: new)
        }
        let stOldLoop = """
              simdgroup_float8x8 A;
              simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

            #pragma unroll
              for (int gi = 0; gi < nGroups; ++gi) {
                const int g = g0 + gi;
                const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
                const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

                const float2 rs = float2(
                    rs_table[c.fn * G + g], rs_table[(c.fn + 1) * G + g]);

            """
        let stNewLoop = """
              simdgroup_float8x8 A;
              simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

              uint2 wv_next = *((const device uint2*)(wrow + 32 * g0));
              uint2 wv_next2 =
                  *((const device uint2*)(wrow + 32 * (g0 + min(1, nGroups - 1))));
              T s_next = srow[g0];
              T b_next = brow[g0];
              uint4 r0_next = *((const device uint4*)(x0 + 64 * g0));
              uint4 r1_next = *((const device uint4*)(x1 + 64 * g0));
              float2 rs_next = float2(
                  rs_table[c.fn * G + g0], rs_table[(c.fn + 1) * G + g0]);

            #pragma unroll
              for (int gi = 0; gi < nGroups; ++gi) {
                const uint4 r0 = r0_next;
                const uint4 r1 = r1_next;
                const float2 rs = rs_next;
                const uint2 wv = wv_next;
                const float s = float(s_next);
                const float b = float(b_next);
                const int g_next = g0 + min(gi + 1, nGroups - 1);
                const int g_next2 = g0 + min(gi + 2, nGroups - 1);
                wv_next = wv_next2;
                wv_next2 = *((const device uint2*)(wrow + 32 * g_next2));
                s_next = srow[g_next];
                b_next = brow[g_next];
                r0_next = *((const device uint4*)(x0 + 64 * g_next));
                r1_next = *((const device uint4*)(x1 + 64 * g_next));
                rs_next = float2(
                    rs_table[c.fn * G + g_next], rs_table[(c.fn + 1) * G + g_next]);

            """
        let stOldWeights = """
                const uint2 wv = *((const device uint2*)(wrow + 32 * g));
                const float s = float(srow[g]);
                const float b = float(brow[g]);


            """
        let mtNewLoop = """
              simdgroup_float8x8 A;
              simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

              uint2 wv_next[TILES];
              uint2 wv_next2[TILES];
              T s_next[TILES];
              T b_next[TILES];
            #pragma clang loop unroll(full)
              for (int t = 0; t < TILES; ++t) {
                wv_next[t] = *((const device uint2*)(wrow[t] + 32 * g0));
                wv_next2[t] =
                    *((const device uint2*)(wrow[t] + 32 * (g0 + min(1, nGroups - 1))));
                s_next[t] = srow[t][g0];
                b_next[t] = brow[t][g0];
              }
              uint4 r0_next = *((const device uint4*)(x0 + 64 * g0));
              uint4 r1_next = *((const device uint4*)(x1 + 64 * g0));
              float2 rs_next = float2(
                  rs_table[c.fn * G + g0], rs_table[(c.fn + 1) * G + g0]);

            #pragma unroll
              for (int gi = 0; gi < nGroups; ++gi) {
                const uint4 r0 = r0_next;
                const uint4 r1 = r1_next;
                const float2 rs = rs_next;
                uint2 wv_cur[TILES];
                float s_cur[TILES];
                float b_cur[TILES];
            #pragma clang loop unroll(full)
                for (int t = 0; t < TILES; ++t) {
                  wv_cur[t] = wv_next[t];
                  s_cur[t] = float(s_next[t]);
                  b_cur[t] = float(b_next[t]);
                }
                const int g_next = g0 + min(gi + 1, nGroups - 1);
                const int g_next2 = g0 + min(gi + 2, nGroups - 1);
            #pragma clang loop unroll(full)
                for (int t = 0; t < TILES; ++t) {
                  wv_next[t] = wv_next2[t];
                  wv_next2[t] = *((const device uint2*)(wrow[t] + 32 * g_next2));
                  s_next[t] = srow[t][g_next];
                  b_next[t] = brow[t][g_next];
                }
                r0_next = *((const device uint4*)(x0 + 64 * g_next));
                r1_next = *((const device uint4*)(x1 + 64 * g_next));
                rs_next = float2(
                    rs_table[c.fn * G + g_next], rs_table[(c.fn + 1) * G + g_next]);

            """
        let mtOldWeights = """
                for (int t = 0; t < TILES; ++t) {
                  const uint2 wv = *((const device uint2*)(wrow[t] + 32 * g));
                  const float s = float(srow[t][g]);
                  const float b = float(brow[t][g]);

            """
        let mtNewWeights = """
                for (int t = 0; t < TILES; ++t) {
                  const uint2 wv = wv_cur[t];
                  const float s = s_cur[t];
                  const float b = b_cur[t];

            """
        var st = slice(
            "template <typename T, int KS, int KFIX>\nMETAL_FUNC void qkv_mma8_affine4_g64_rsp(",
            to: "\n// MMA-RS-001: the two-tile MMA-MT-001 body")
        replaceOnce(&st, "qkv_mma8_affine4_g64_rsp(", with: "qkv_mma8_affine4_g64_rsp_c2x(")
        replaceOnce(&st, stOldLoop, with: stNewLoop)
        replaceOnce(&st, stOldWeights, with: "")
        var mt = slice(
            "template <typename T, int KS, int TILES, int KFIX, int SPLIT = 0>\nMETAL_FUNC void qkv_mma8_affine4_g64_mt_rsp(",
            to: nil)
        replaceOnce(&mt, "qkv_mma8_affine4_g64_mt_rsp(", with: "qkv_mma8_affine4_g64_mt_rsp_c2x(")
        replaceOnce(&mt, stOldLoop, with: mtNewLoop)
        replaceOnce(&mt, mtOldWeights, with: mtNewWeights)
        func trimmed(_ t: String) -> String {
            var v = t
            while v.hasSuffix("\n") { v.removeLast() }
            return v
        }
        return mma8KernelHeader + "\n" + trimmed(st) + "\n\n" + trimmed(mt) + "\n"
    }()


    private static let mma8RowsKernelHeader = mma8KernelHeader + """

// Row-tiled twin of `qkv_mma8_affine4_g64_mt` for the wide MTP verify
// rectangle: RT eight-row tiles of activations share every weight fetch.
// Each row tile runs the multitile body's own statements in its own order,
// so a row's output word is the eight-row kernel's for that row.
template <typename T, int KS, int TILES, int RT, int KFIX>
METAL_FUNC void qkv_mma8_affine4_g64_mt_rows(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  constexpr int K = KFIX;
  constexpr int G = K / 64;
  constexpr int gh = (G + 1) / 2;
  constexpr int nGroups = (KS == 2) ? gh : G;
  const int g0 = (KS == 2 && simd_gid == 1) ? gh : 0;
  const mma8_coord c = mma8_lane(simd_lid);

  const device uint8_t* wrow[TILES];
  const device T* srow[TILES];
  const device T* brow[TILES];
  thread float acc0[TILES][RT];
  thread float acc1[TILES][RT];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
    wrow[t] = (const device uint8_t*)w + (nt + c.fm) * (K / 2) + 4 * c.fn;
    srow[t] = scales + (nt + c.fm) * G;
    brow[t] = biases + (nt + c.fm) * G;
#pragma clang loop unroll(full)
    for (int r = 0; r < RT; ++r) {
      acc0[t][r] = 0.0f;
      acc1[t][r] = 0.0f;
    }
  }

  const device T* xbase = x + c.fn * K + 8 * c.fm;

  simdgroup_float8x8 A;
  simdgroup_float8x8 B0, B1, B2, B3, B4, B5, B6, B7;

  uint2 wv_next[TILES];
  uint2 wv_next2[TILES];
  T s_next[TILES];
  T b_next[TILES];
#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    wv_next[t] = *((const device uint2*)(wrow[t] + 32 * g0));
    wv_next2[t] =
        *((const device uint2*)(wrow[t] + 32 * (g0 + min(1, nGroups - 1))));
    s_next[t] = srow[t][g0];
    b_next[t] = brow[t][g0];
  }

#pragma unroll
  for (int gi = 0; gi < nGroups; ++gi) {
    const int g = g0 + gi;

    uint2 wv_cur[TILES];
    float s_cur[TILES];
    float b_cur[TILES];
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      wv_cur[t] = wv_next[t];
      s_cur[t] = float(s_next[t]);
      b_cur[t] = float(b_next[t]);
    }
    const int g_next = g0 + min(gi + 1, nGroups - 1);
    const int g_next2 = g0 + min(gi + 2, nGroups - 1);
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
      wv_next[t] = wv_next2[t];
      wv_next2[t] = *((const device uint2*)(wrow[t] + 32 * g_next2));
      s_next[t] = srow[t][g_next];
      b_next[t] = brow[t][g_next];
    }

#pragma clang loop unroll(full)
    for (int r = 0; r < RT; ++r) {
      const device T* x0 = xbase + (8 * r) * K;
      const device T* x1 = x0 + K;
      const uint4 r0 = *((const device uint4*)(x0 + 64 * g));
      const uint4 r1 = *((const device uint4*)(x1 + 64 * g));

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

#pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
        const uint2 wv = wv_cur[t];
        const float s = s_cur[t];
        const float b = b_cur[t];

        simdgroup_float8x8 C = simdgroup_float8x8(0.0f);
        MMA8_STEP(B0, 0)
        MMA8_STEP(B1, 1)
        MMA8_STEP(B2, 2)
        MMA8_STEP(B3, 3)
        MMA8_STEP(B4, 4)
        MMA8_STEP(B5, 5)
        MMA8_STEP(B6, 6)
        MMA8_STEP(B7, 7)

        acc0[t][r] += s * C.thread_elements()[0] + rs.x * b;
        acc1[t][r] += s * C.thread_elements()[1] + rs.y * b;
      }
    }
  }

  if (KS == 2) {
    if (simd_gid == 1) {
#pragma clang loop unroll(full)
      for (int t = 0; t < TILES; ++t) {
#pragma clang loop unroll(full)
        for (int r = 0; r < RT; ++r) {
          red[(t * RT + r) * 32 + simd_lid] = float2(acc0[t][r], acc1[t][r]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 1) {
      return;
    }
#pragma clang loop unroll(full)
    for (int t = 0; t < TILES; ++t) {
#pragma clang loop unroll(full)
      for (int r = 0; r < RT; ++r) {
        const float2 other = red[(t * RT + r) * 32 + simd_lid];
        acc0[t][r] = acc0[t][r] + other.x;
        acc1[t][r] = acc1[t][r] + other.y;
      }
    }
  }

#pragma clang loop unroll(full)
  for (int t = 0; t < TILES; ++t) {
    const int nt = n0 + t * 8;
#pragma clang loop unroll(full)
    for (int r = 0; r < RT; ++r) {
      y[(8 * r + c.fn) * N + nt + c.fm] = static_cast<T>(acc0[t][r]);
      y[(8 * r + c.fn + 1) * N + nt + c.fm] = static_cast<T>(acc1[t][r]);
    }
  }
}
"""

    /// Wide MTP verify: one dispatch for RT eight-row tiles (rows = 8 * RT).
    private static let multiTileRowsKernel = MLXFast.metalKernel(
        name: "cbv2_wide_qkv_mma8_affine4_g64_mt2_k2816_rows_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64 * RT];
            qkv_mma8_affine4_g64_mt_rows<T, 2, 2, RT, 2816>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8RowsKernelHeader,
        ensureRowContiguous: true)

    /// MMA-MT-001 arm. Default ON.
    /// `DARKBLOOM_GEMMA4_QKV_MMA8_MULTITILE=0` restores the promoted
    /// single-tile dispatch byte for byte in the same executable.
    public static let multiTileEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_QKV_MMA8_MULTITILE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Output tiles per simdgroup. Two is the shipped width: it halves the
    /// x-side work per output column while keeping the incumbent's
    /// threadgroup count within a factor of two, which on the local box is
    /// where the amortisation stops paying for the extra live registers.
    private static let tilesPerGroup = 2

    private static let multiTileKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_carry2_v4",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt<T, 2, 2, 2816>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    /// QKFUSE-001 arm. Default ON.
    /// `DARKBLOOM_GEMMA4_QKV_FUSE_QK=0` restores the two separate dispatches.
    public static let fuseQKEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_QKV_FUSE_QK"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// QKVFUSE-001 arm. Default ON. The exact B8 sliding shape joins Q, K,
    /// and V into one output plane; zero restores the three separate roads.
    public static let fuseQKVEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_QKV_FUSE_QKV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let fusedSlidingKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_carry2_qk6144_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y", "y2"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt<T, 2, 2, 2816, 4096>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup, y2);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let fusedFullKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_carry2_qk9216_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y", "y2"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt<T, 2, 2, 2816, 8192>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup, y2);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    // MMA-RS-001 on QKFUSE-001: the fused Q||K dispatch consumes the shared
    // run-sum table. The table is per activation row and per 64-group of K,
    // independent of N, so the concatenated-N rsp body reads the identical
    // entries the separate Q and K rsp dispatches would; the SPLIT store
    // keeps QKFUSE-001's two-buffer layout.
    private static let fusedSlidingRspKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_carry2_qk6144_rsp_v1",
        inputNames: ["x", "w", "scales", "biases", "rs_table"],
        outputNames: ["y", "y2"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt_rsp<T, 2, 2, 2816, 4096>(
                w, scales, biases, x, rs_table, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup, y2);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let fusedFullRspKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_carry2_qk9216_rsp_v1",
        inputNames: ["x", "w", "scales", "biases", "rs_table"],
        outputNames: ["y", "y2"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt_rsp<T, 2, 2, 2816, 8192>(
                w, scales, biases, x, rs_table, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup, y2);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    /// Register carry on the production (rsp) bodies: `none` is the tip's
    /// text, `c2x` the two-deep weight / one-ahead activation carry.
    public enum Carry: String, Sendable { case none, c2x }

    private static func rspC2XKernel(_ name: String, tiles: Int, split: Int) -> MLXFast.MLXFastKernel {
        let outputs = split > 0 ? ["y", "y2"] : ["y"]
        let call = tiles == 2
            ? "threadgroup float2 red[64];\n    qkv_mma8_affine4_g64_mt_rsp_c2x<T, 2, 2, 2816\(split > 0 ? ", \(split)" : "")>("
            : "threadgroup float2 red[32];\n    qkv_mma8_affine4_g64_rsp_c2x<T, 2, 2816>("
        let n0 = tiles == 2 ? "int(tid.y) * 16" : "int(tid.y) * 8"
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["x", "w", "scales", "biases", "rs_table"],
            outputNames: outputs,
            source: """
                const uint3 tid = threadgroup_position_in_grid;
                \(call)
                    w, scales, biases, x, rs_table, y,
                    w_shape[0], \(n0), red,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup\(split > 0 ? ", y2" : ""));
                return;
                """,
            header: mma8C2XKernelHeader,
            ensureRowContiguous: true)
    }

    private static let multiTileRspC2XKernel = rspC2XKernel(
        "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_rsp_c2x_v1", tiles: 2, split: 0)
    private static let mma8RspC2XKernel = rspC2XKernel(
        "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_k2816_rsp_c2x_v1", tiles: 1, split: 0)
    private static let fusedSlidingRspC2XKernel = rspC2XKernel(
        "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_qk6144_rsp_c2x_v1", tiles: 2, split: 4096)
    private static let fusedFullRspC2XKernel = rspC2XKernel(
        "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_qk9216_rsp_c2x_v1", tiles: 2, split: 8192)
    private static let fusedQKVRspC2XKernel = rspC2XKernel(
        "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_qkv8192_rsp_c2x_v1", tiles: 2, split: 0)

    /// Eight-row body: two eight-column tiles per simdgroup (`mt2`) or one
    /// (`st`). Both run the same statements per tile.
    public enum Body: String, Sendable { case mt2, st }
    /// Wide verify mapping: `rows` folds RT row tiles into one simdgroup
    /// pair; `ztile` gives each row tile its own threadgroup (grid.z) running
    /// the eight-row body unchanged.
    public enum WideRoad: String, Sendable { case rows, ztile }

    private static func envChoice<E: RawRepresentable>(_ name: String, _ fallback: E) -> E
    where E.RawValue == String {
        guard let raw = ProcessInfo.processInfo.environment[name],
            let choice = E(rawValue: raw.trimmingCharacters(in: .whitespaces).lowercased())
        else { return fallback }
        return choice
    }

    public static let defaultBody: Body = envChoice(
        "DARKBLOOM_GEMMA4_QKV_MMA8_BODY", multiTileEnabled ? .mt2 : .st)
    public static let defaultWideRoad: WideRoad = envChoice("DARKBLOOM_GEMMA4_QKV_MMA8_WIDE", .ztile)
    public static let defaultCarry: Carry = envChoice("DARKBLOOM_GEMMA4_QKV_MMA8_CARRY", .c2x)

    private static func zTileKernel(_ name: String, body: Body) -> MLXFast.MLXFastKernel {
        let call = body == .mt2
            ? "threadgroup float2 red[64];\n    qkv_mma8_affine4_g64_mt<T, 2, 2, 2816>("
            : "threadgroup float2 red[32];\n    qkv_mma8_affine4_g64_impl<T, 2, 2816>("
        let n0 = body == .mt2 ? "int(tid.y) * 16" : "int(tid.y) * 8"
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["x", "w", "scales", "biases"],
            outputNames: ["y"],
            source: """
                const uint3 tid = threadgroup_position_in_grid;
                \(call)
                    w, scales, biases, x + tid.z * (8 * 2816), y + tid.z * (8 * w_shape[0]),
                    w_shape[0], \(n0), red,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
                return;
                """,
            header: mma8KernelHeader,
            ensureRowContiguous: true)
    }

    private static let zTileMT2Kernel = zTileKernel(
        "cbv2_wide_qkv_mma8_affine4_g64_mt2_k2816_ztile_v1", body: .mt2)
    private static let zTileSTKernel = zTileKernel(
        "cbv2_wide_qkv_mma8_affine4_g64_st_k2816_ztile_v1", body: .st)

    private static let mma8Kernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_k2816_carry2_bfill_v4",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            qkv_mma8_affine4_g64_impl<T, 2, 2816>(
                w, scales, biases, x, y,
                w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    // MMA-RS-001: the run-sum prepass. One threadgroup per (row quad, K
    // group); lane r*8+fm computes `mma8_runsum4` over the same aligned
    // 16-byte activation run the main body's lane (fm, row) reads, and the
    // three xor butterflies (masks 1, 2, 4 over this layout's lane bits are
    // the incumbent's masks 2, 4, 16 over ITS lane bits: both walk fm bit 0,
    // then fm bit 1, then fm bit 2) reduce the eight fm partials in the same
    // balanced fp32 tree. Lane fm == 0 stores the row's group total.
    //
    // Bit-exactness. Every lane of the incumbent ends its butterfly holding
    // ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7)) for its row; float
    // addition is commutative, so that value is lane-independent bit for bit,
    // and this kernel stores exactly it. The table entry therefore equals the
    // `rs` component the incumbent computes, and the rsp bodies consume the
    // identical float in the identical accumulator step.
    private static let runsumTableKernel = MLXFast.metalKernel(
        name: "cbv2_b8_rs_table_k2816_v1",
        inputNames: ["x"],
        outputNames: ["rs"],
        source: """
            constexpr int K = 2816;
            constexpr int G = K / 64;
            const int rowQuad = int(threadgroup_position_in_grid.y) / G;
            const int g = int(threadgroup_position_in_grid.y) % G;
            const int lane = int(thread_position_in_threadgroup.x);
            const int r = lane >> 3;
            const int fm = lane & 7;
            const uint4 run =
                *((const device uint4*)(x + (rowQuad * 4 + r) * K + 64 * g + 8 * fm));
            float v = mma8_runsum4<T>(run);
            v += simd_shuffle_xor(v, 1u);
            v += simd_shuffle_xor(v, 2u);
            v += simd_shuffle_xor(v, 4u);
            if (fm == 0) {
                rs[(rowQuad * 4 + r) * G + g] = v;
            }
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let multiTileRspKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_k2816_rsp_v1",
        inputNames: ["x", "w", "scales", "biases", "rs_table"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt_rsp<T, 2, 2, 2816>(
                w, scales, biases, x, rs_table, y,
                w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let mma8RspKernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_k2816_rsp_v1",
        inputNames: ["x", "w", "scales", "biases", "rs_table"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            qkv_mma8_affine4_g64_rsp<T, 2, 2816>(
                w, scales, biases, x, rs_table, y,
                w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    /// MMA-RS-001 table for one activation tensor. Returns nil unless the
    /// tensor matches the exact decode shape the rsp bodies admit, so a nil
    /// table always means "use the incumbent dispatch".
    @inline(__always)
    public static func runsumTable(for x: MLXArray) -> MLXArray? {
        guard rsPrepassEnabled,
            x.dtype == .bfloat16,
            x.ndim == 3,
            x.dim(0) == batch,
            x.dim(1) == sequence,
            x.dim(2) == inputWidth
        else { return nil }
        return runsumTableKernel(
            [x],
            template: [("T", x.dtype)],
            grid: (simdWidth, 2 * (inputWidth / groupSize), 1),
            threadGroup: (simdWidth, 1, 1),
            outputShapes: [[batch, inputWidth / groupSize]],
            outputDTypes: [.float32]
        )[0]
    }

    @inline(__always)
    private static let fusedLock = NSLock()

    private struct FusedQKPlane {
        let qWeight: MLXArray
        let qScales: MLXArray
        let qBiases: MLXArray
        let kWeight: MLXArray
        let kScales: MLXArray
        let kBiases: MLXArray
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray

        @inline(__always)
        func matches(
            qWeight candidateQWeight: MLXArray,
            qScales candidateQScales: MLXArray,
            qBiases candidateQBiases: MLXArray,
            kWeight candidateKWeight: MLXArray,
            kScales candidateKScales: MLXArray,
            kBiases candidateKBiases: MLXArray
        ) -> Bool {
            ObjectIdentifier(qWeight) == ObjectIdentifier(candidateQWeight)
                && ObjectIdentifier(qScales) == ObjectIdentifier(candidateQScales)
                && ObjectIdentifier(qBiases) == ObjectIdentifier(candidateQBiases)
                && ObjectIdentifier(kWeight) == ObjectIdentifier(candidateKWeight)
                && ObjectIdentifier(kScales) == ObjectIdentifier(candidateKScales)
                && ObjectIdentifier(kBiases) == ObjectIdentifier(candidateKBiases)
                && qWeight.shape == candidateQWeight.shape
                && qWeight.dtype == candidateQWeight.dtype
                && qScales.shape == candidateQScales.shape
                && qScales.dtype == candidateQScales.dtype
                && qBiases.shape == candidateQBiases.shape
                && qBiases.dtype == candidateQBiases.dtype
                && kWeight.shape == candidateKWeight.shape
                && kWeight.dtype == candidateKWeight.dtype
                && kScales.shape == candidateKScales.shape
                && kScales.dtype == candidateKScales.dtype
                && kBiases.shape == candidateKBiases.shape
                && kBiases.dtype == candidateKBiases.dtype
                && weight.shape
                    == [candidateQWeight.dim(0) + candidateKWeight.dim(0), candidateQWeight.dim(1)]
                && weight.dtype == candidateQWeight.dtype
                && scales.shape
                    == [candidateQScales.dim(0) + candidateKScales.dim(0), candidateQScales.dim(1)]
                && scales.dtype == candidateQScales.dtype
                && biases.shape == scales.shape
                && biases.dtype == candidateQBiases.dtype
        }
    }

    nonisolated(unsafe) private static var fusedPlanes:
        [ObjectIdentifier: FusedQKPlane] = [:]
    nonisolated(unsafe) private static var fusedQKVPlanes:
        [ObjectIdentifier: (MLXArray, MLXArray, MLXArray, MLXArray)] = [:]

    /// QKFUSE-001. One dispatch for the layer's Q and K projections.
    ///
    /// Q and K read the SAME activation at decode (`queryInput === x` whenever
    /// last-query prefill is off) and share K=2816, group size, bit width and
    /// quantization mode, so their weight planes concatenate along the output
    /// axis into one plane. Row r of the fused output is row r of whichever
    /// source plane it came from: the tier accumulates each output column
    /// independently, so this is bit-exact, not a reassociation. Threadgroup
    /// count is unchanged (768 for 4096+2048, the same 512+256 the two
    /// dispatches launch), so the whole saving is one encoder per layer.
    ///
    /// The concatenation is a pure re-layout of the shipped quantized words.
    /// No value is re-quantized and no numerical format changes.
    ///
    /// MMA-RS-001 merge: `rsTable` is the activation's precomputed run-sum
    /// table. The table is indexed by activation row and 64-group of K only —
    /// never by N — so the concatenated-N dispatch reads the identical
    /// entries the separate Q and K rsp dispatches would, and the rsp body's
    /// `acc[t] += s * C + rs * b` step consumes the identical float in the
    /// identical order per output column. Nil keeps the incumbent fused
    /// dispatch (in-kernel run sums).
    public static func fusedQKMatmul(
        x: MLXArray,
        qWeight: MLXArray, qScales: MLXArray, qBiases: MLXArray?,
        kWeight: MLXArray, kScales: MLXArray, kBiases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        cacheKey: ObjectIdentifier,
        rsTable: MLXArray? = nil,
        carry: Carry? = nil
    ) -> (MLXArray, MLXArray)? {
        guard enabled, fuseQKEnabled, multiTileEnabled,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let qBiases, let kBiases,
            x.dtype == .bfloat16,
            qScales.dtype == x.dtype, qBiases.dtype == x.dtype,
            kScales.dtype == x.dtype, kBiases.dtype == x.dtype,
            qWeight.dtype == .uint32, kWeight.dtype == .uint32,
            x.ndim == 3,
            x.dim(0) == batch, x.dim(1) == sequence, x.dim(2) == inputWidth,
            x.size == batch * sequence * inputWidth,
            qWeight.ndim == 2, kWeight.ndim == 2,
            qWeight.dim(1) == inputWidth * Self.bits / 32,
            kWeight.dim(1) == inputWidth * Self.bits / 32
        else { return nil }

        let qWidth = qWeight.dim(0)
        let kWidth = kWeight.dim(0)
        guard liveFusedSplit(qWidth), liveOutputWidth(kWidth),
            qScales.shape == [qWidth, inputWidth / Self.groupSize],
            qBiases.shape == qScales.shape,
            kScales.shape == [kWidth, inputWidth / Self.groupSize],
            kBiases.shape == kScales.shape
        else { return nil }

        let tableReady =
            rsTable != nil
            && rsTable!.dtype == .float32
            && rsTable!.shape == [batch, inputWidth / Self.groupSize]

        let c2x = (carry ?? defaultCarry) == .c2x
        let kernel =
            qWidth == 4096
            ? (tableReady ? (c2x ? fusedSlidingRspC2XKernel : fusedSlidingRspKernel) : fusedSlidingKernel)
            : (tableReady ? (c2x ? fusedFullRspC2XKernel : fusedFullRspKernel) : fusedFullKernel)
        let total = qWidth + kWidth
        let yTiles = total / outputsPerGroup
        guard yTiles % tilesPerGroup == 0 else { return nil }

        fusedLock.lock()
        var plane = fusedPlanes[cacheKey]
        if plane?.matches(
            qWeight: qWeight, qScales: qScales, qBiases: qBiases,
            kWeight: kWeight, kScales: kScales, kBiases: kBiases) != true
        {
            let w = concatenated([qWeight, kWeight], axis: 0)
            let s = concatenated([qScales, kScales], axis: 0)
            let b = concatenated([qBiases, kBiases], axis: 0)
            eval(w, s, b)
            plane = FusedQKPlane(
                qWeight: qWeight, qScales: qScales, qBiases: qBiases,
                kWeight: kWeight, kScales: kScales, kBiases: kBiases,
                weight: w, scales: s, biases: b)
            fusedPlanes[cacheKey] = plane
        }
        fusedLock.unlock()
        guard let plane else { return nil }

        let outputs = kernel(
            tableReady
                ? [x, plane.weight, plane.scales, plane.biases, rsTable!]
                : [x, plane.weight, plane.scales, plane.biases],
            template: [("T", x.dtype)],
            grid: (simdWidth, (yTiles / tilesPerGroup) * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, qWidth], [batch, sequence, kWidth]],
            outputDTypes: [x.dtype, x.dtype])
        return (outputs[0], outputs[1])
    }

    /// QKVFUSE-001. One c2x MMA8 dispatch for the sliding layer's Q, K and V
    /// projections. The three quantized planes are concatenated only along N;
    /// every output column executes the same rsp-c2x statements and accumulator
    /// sequence it executes in the separate dispatch. The returned tensors are
    /// views of the joined output, so splitting introduces no arithmetic.
    public static func fusedQKVMatmul(
        x: MLXArray,
        qWeight: MLXArray, qScales: MLXArray, qBiases: MLXArray?,
        kWeight: MLXArray, kScales: MLXArray, kBiases: MLXArray?,
        vWeight: MLXArray, vScales: MLXArray, vBiases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        cacheKey: ObjectIdentifier,
        rsTable: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard enabled, fuseQKVEnabled, multiTileEnabled,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let qBiases, let kBiases, let vBiases,
            x.dtype == .bfloat16,
            qScales.dtype == x.dtype, qBiases.dtype == x.dtype,
            kScales.dtype == x.dtype, kBiases.dtype == x.dtype,
            vScales.dtype == x.dtype, vBiases.dtype == x.dtype,
            qWeight.dtype == .uint32, kWeight.dtype == .uint32,
            vWeight.dtype == .uint32,
            x.shape == [batch, sequence, inputWidth],
            qWeight.shape == [4096, inputWidth * Self.bits / 32],
            kWeight.shape == [2048, inputWidth * Self.bits / 32],
            vWeight.shape == [2048, inputWidth * Self.bits / 32],
            qScales.shape == [4096, inputWidth / Self.groupSize],
            qBiases.shape == qScales.shape,
            kScales.shape == [2048, inputWidth / Self.groupSize],
            kBiases.shape == kScales.shape,
            vScales.shape == [2048, inputWidth / Self.groupSize],
            vBiases.shape == vScales.shape,
            let rsTable,
            rsTable.dtype == .float32,
            rsTable.shape == [batch, inputWidth / Self.groupSize]
        else { return nil }

        fusedLock.lock()
        var plane = fusedQKVPlanes[cacheKey]
        if plane == nil {
            let w = concatenated([qWeight, kWeight, vWeight], axis: 0)
            let s = concatenated([qScales, kScales, vScales], axis: 0)
            let b = concatenated([qBiases, kBiases, vBiases], axis: 0)
            eval(w, s, b)
            // Retain the identity owner with the plane. Test-created arrays
            // are short-lived; without this reference their ObjectIdentifier
            // can be reused for a later, unrelated plane.
            plane = (qWeight, w, s, b)
            fusedQKVPlanes[cacheKey] = plane
        }
        fusedLock.unlock()
        guard let (_, fw, fs, fb) = plane else { return nil }

        let joined = fusedQKVRspC2XKernel(
            [x, fw, fs, fb, rsTable],
            template: [("T", x.dtype)],
            grid: (simdWidth, (8192 / outputsPerGroup / tilesPerGroup) * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, 8192]],
            outputDTypes: [x.dtype]
        )[0]
        return (
            joined[.ellipsis, 0 ..< 4096],
            joined[.ellipsis, 4096 ..< 6144],
            joined[.ellipsis, 6144 ..< 8192])
    }

    /// Q widths the fused kernels bake as a compile-time split point.
    private static func liveFusedSplit(_ width: Int) -> Bool {
        width == 4096 || width == 8192
    }

    private static func liveOutputWidth(_ width: Int) -> Bool {
        width == 1024 || width == 2048 || width == 4096 || width == 8192
    }

    public static func matmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        rsTable: MLXArray? = nil,
        body: Body? = nil,
        wideRoad: WideRoad? = nil,
        carry: Carry? = nil
    ) -> MLXArray? {
        let rows = x.ndim == 3 ? x.dim(0) : 0
        let rowTiles = CBv2MTPWideVerifyContext.rowTileCount(rows: rows, tile: batch)
        guard enabled,
            groupSize == Self.groupSize,
            bits == Self.bits,
            mode == .affine,
            let biases,
            x.dtype == .bfloat16,
            scales.dtype == x.dtype,
            biases.dtype == x.dtype,
            weight.dtype == .uint32,
            x.ndim == 3,
            rowTiles >= 1,
            x.dim(1) == sequence,
            x.dim(2) == inputWidth,
            weight.ndim == 2,
            weight.dim(1) == inputWidth * Self.bits / 32
        else { return nil }
        let body = body ?? defaultBody
        let c2x = (carry ?? defaultCarry) == .c2x

        let outputWidth = weight.dim(0)
        guard liveOutputWidth(outputWidth),
            x.size == rows * sequence * inputWidth,
            scales.shape == [outputWidth, inputWidth / Self.groupSize],
            biases.shape == scales.shape
        else { return nil }

        let tableReady =
            rsTable != nil
            && rsTable!.dtype == .float32
            && rsTable!.shape == [batch, inputWidth / Self.groupSize]

        let yTiles = outputWidth / outputsPerGroup
        if rowTiles > 1, (wideRoad ?? defaultWideRoad) == .ztile {
            let multiTile = body == .mt2 && yTiles % tilesPerGroup == 0
            return (multiTile ? zTileMT2Kernel : zTileSTKernel)(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, (multiTile ? yTiles / tilesPerGroup : yTiles) * simdGroups, rowTiles),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[rows, sequence, outputWidth]],
                outputDTypes: [x.dtype]
            )[0]
        }
        if rowTiles > 1 {
            guard multiTileEnabled, yTiles % tilesPerGroup == 0 else { return nil }
            return multiTileRowsKernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype), ("RT", rowTiles)],
                grid: (simdWidth, (yTiles / tilesPerGroup) * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[rows, sequence, outputWidth]],
                outputDTypes: [x.dtype]
            )[0]
        }
        if body == .mt2, yTiles % tilesPerGroup == 0 {
            if tableReady {
                return (c2x ? multiTileRspC2XKernel : multiTileRspKernel)(
                    [x, weight, scales, biases, rsTable!],
                    template: [("T", x.dtype)],
                    grid: (simdWidth, (yTiles / tilesPerGroup) * simdGroups, 1),
                    threadGroup: (simdWidth, simdGroups, 1),
                    outputShapes: [[batch, sequence, outputWidth]],
                    outputDTypes: [x.dtype]
                )[0]
            }
            return multiTileKernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, (yTiles / tilesPerGroup) * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, sequence, outputWidth]],
                outputDTypes: [x.dtype]
            )[0]
        }
        if tableReady {
            return (c2x ? mma8RspC2XKernel : mma8RspKernel)(
                [x, weight, scales, biases, rsTable!],
                template: [("T", x.dtype)],
                grid: (simdWidth, yTiles * simdGroups, 1),
                threadGroup: (simdWidth, simdGroups, 1),
                outputShapes: [[batch, sequence, outputWidth]],
                outputDTypes: [x.dtype]
            )[0]
        }
        return mma8Kernel(
            [x, weight, scales, biases],
            template: [("T", x.dtype)],
            grid: (simdWidth, yTiles * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[batch, sequence, outputWidth]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
