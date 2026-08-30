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

template <typename T, int KS>
METAL_FUNC void qkv_mma8_affine4_g64_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  const int G = K / 64;
  const int gh = (G + 1) / 2;
  const int g_begin = (KS == 2 && simd_gid == 1) ? gh : 0;
  const int g_end = (KS == 2 && simd_gid == 0) ? gh : G;
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

  for (int g = g_begin; g < g_end; ++g) {
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
template <typename T, int KS, int TILES>
METAL_FUNC void qkv_mma8_affine4_g64_mt(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int n0,
    threadgroup float2* red,
    uint simd_gid,
    uint simd_lid) {
  const int G = K / 64;
  const int gh = (G + 1) / 2;
  const int g_begin = (KS == 2 && simd_gid == 1) ? gh : 0;
  const int g_end = (KS == 2 && simd_gid == 0) ? gh : G;
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

  for (int g = g_begin; g < g_end; ++g) {
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
    y[c.fn * N + nt + c.fm] = static_cast<T>(acc0[t]);
    y[(c.fn + 1) * N + nt + c.fm] = static_cast<T>(acc1[t]);
  }
}
"""

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
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_mt2_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[64];
            qkv_mma8_affine4_g64_mt<T, 2, 2>(
                w, scales, biases, x, y,
                x_shape[x_ndim - 1], w_shape[0], int(tid.y) * 16, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    private static let mma8Kernel = MLXFast.metalKernel(
        name: "cbv2_b8_l1_qkv_mma8_affine4_g64_tight_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            threadgroup float2 red[32];
            qkv_mma8_affine4_g64_impl<T, 2>(
                w, scales, biases, x, y,
                x_shape[x_ndim - 1], w_shape[0], int(tid.y) * 8, red,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            return;
            """,
        header: mma8KernelHeader,
        ensureRowContiguous: true)

    @inline(__always)
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
        mode: QuantizationMode
    ) -> MLXArray? {
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
            x.dim(0) == batch,
            x.dim(1) == sequence,
            x.dim(2) == inputWidth,
            weight.ndim == 2,
            weight.dim(1) == inputWidth * Self.bits / 32
        else { return nil }

        let outputWidth = weight.dim(0)
        guard liveOutputWidth(outputWidth),
            x.size == batch * sequence * inputWidth,
            scales.shape == [outputWidth, inputWidth / Self.groupSize],
            biases.shape == scales.shape
        else { return nil }

        let yTiles = outputWidth / outputsPerGroup
        if multiTileEnabled, yTiles % tilesPerGroup == 0 {
            return multiTileKernel(
                [x, weight, scales, biases],
                template: [("T", x.dtype)],
                grid: (simdWidth, (yTiles / tilesPerGroup) * simdGroups, 1),
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
