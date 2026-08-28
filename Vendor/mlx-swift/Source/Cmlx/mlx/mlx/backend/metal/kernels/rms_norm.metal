// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant bool has_w [[function_constant(20)]];

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_looped(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_single_row(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the computation and accumulators
  float thread_x[N_READS];
  float thread_w[N_READS];
  float thread_g[N_READS];
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_w[i] = w[w_stride * i];
      thread_g[i] = g[i];

      sumx2 += thread_x[i] * thread_x[i];
      sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        thread_w[i] = w[w_stride * i];
        thread_g[i] = g[i];

        sumx2 += thread_x[i] * thread_x[i];
        sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      gx[i] = static_cast<T>(
          thread_g[i] * thread_w[i] * normalizer -
          thread_x[i] * meangwx * normalizer3);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        gx[i] = static_cast<T>(
            thread_g[i] * thread_w[i] * normalizer -
            thread_x[i] * meangwx * normalizer3);
        if (has_w) {
          gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_looped(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the accumulators
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        sumx2 += xi * xi;
        sumgwx += xi * wi * gi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          sumx2 += xi * xi;
          sumgwx += xi * wi * gi;
        }
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        gx[i + r] =
            static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
        if (has_w) {
          gw[i + r] = static_cast<T>(gi * xi * normalizer);
        }
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          gx[i + r] =
              static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
          if (has_w) {
            gw[i + r] = static_cast<T>(gi * xi * normalizer);
          }
        }
      }
    }
  }
}

template <typename T>
[[kernel]] void fused_qk_rms_rope_inplace(
    device T* qk [[buffer(0)]],
    const device T* w_q [[buffer(1)]],
    const device T* w_k [[buffer(2)]],
    const device int* offset [[buffer(3)]],
    constant const float& scale [[buffer(4)]],
    constant const int64_t& q_stride0 [[buffer(5)]],
    constant const int64_t& q_stride1 [[buffer(6)]],
    constant const int64_t& q_stride2 [[buffer(7)]],
    constant const int64_t& k_stride0 [[buffer(8)]],
    constant const int64_t& k_stride1 [[buffer(9)]],
    constant const int64_t& k_stride2 [[buffer(10)]],
    constant const int64_t& offset_stride [[buffer(11)]],
    constant const int& n_q_heads [[buffer(12)]],
    constant const int& n_kv_heads [[buffer(13)]],
    constant const float& base [[buffer(14)]],
    constant const float& eps [[buffer(15)]],
    constant const uint& head_dim [[buffer(16)]],
    constant const int& q_offset [[buffer(17)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  constexpr int SIMD_SIZE = 32;
  constexpr int N_READS = 4;
  const int D = int(head_dim);
  const int half = D / 2;
  const bool is_q = int(pos.z) < n_q_heads;
  const int head = is_q ? int(pos.z) : int(pos.z) - n_q_heads;
  const int b = int(pos.y / 1024);
  const int t = int(pos.y % 1024);
  const int d_half = int(pos.x);
  if (d_half >= half) return;
  const device T* w = is_q ? w_q : w_k;
  const int64_t s0 = is_q ? q_stride0 : k_stride0;
  const int64_t s1 = is_q ? q_stride1 : k_stride1;
  const int64_t s2 = is_q ? q_stride2 : k_stride2;
  const int qk_batch_stride = 1024 * (is_q ? n_q_heads : n_kv_heads) * D;
  device T* base_ptr = qk + (is_q ? 0 : q_offset) + int64_t(b) * int64_t(qk_batch_stride);
  const int64_t row_base = int64_t(t) * s1 + int64_t(head) * s0;
  threadgroup float local_inv[1];
  threadgroup float partials[SIMD_SIZE];
  float acc = 0;
  for (int d = int(pos.x); d < D; d += int(grid.x)) {
    float v = float(base_ptr[row_base + int64_t(d) * s2]);
    acc += v * v;
  }
  acc = simd_sum(acc);
  if (simdgroup_index_in_threadgroup == 0) partials[thread_index_in_simdgroup] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index_in_simdgroup == 0) partials[simdgroup_index_in_threadgroup] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index_in_threadgroup == 0) {
    float s = simd_sum(partials[thread_index_in_simdgroup]);
    if (thread_index_in_simdgroup == 0) local_inv[0] = metal::precise::rsqrt(s / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = local_inv[0];
  float d_norm = float(d_half) / float(half);
  float inv_freq = metal::exp2(-d_norm * base);
  int off = offset[b * int(offset_stride)];
  float L = scale * float(t + off);
  float th = L * inv_freq;
  float c = metal::fast::cos(th);
  float s_ = metal::fast::sin(th);
  int64_t i1 = row_base + int64_t(d_half) * s2;
  int64_t i2 = row_base + int64_t(d_half + half) * s2;
  float x1 = float(base_ptr[i1]) * inv * float(w[d_half]);
  float x2 = float(base_ptr[i2]) * inv * float(w[d_half + half]);
  float rx1 = x1 * c - x2 * s_;
  float rx2 = x1 * s_ + x2 * c;
  base_ptr[i1] = T(rx1);
  base_ptr[i2] = T(rx2);
}

// clang-format off
#define instantiate_rms(name, itype)                                \
  instantiate_kernel("rms" #name, rms_single_row, itype)            \
  instantiate_kernel("vjp_rms" #name, vjp_rms_single_row, itype)    \
  instantiate_kernel("rms_looped" #name, rms_looped, itype)         \
  instantiate_kernel("vjp_rms_looped" #name, vjp_rms_looped, itype)

instantiate_rms(float32, float)
instantiate_rms(float16, half)
instantiate_rms(bfloat16, bfloat16_t)
instantiate_kernel("fused_qk_rms_rope_inplace_bfloat16", fused_qk_rms_rope_inplace, bfloat16_t)
instantiate_kernel("fused_qk_rms_rope_inplace_float16", fused_qk_rms_rope_inplace, half)
instantiate_kernel("fused_qk_rms_rope_inplace_float32", fused_qk_rms_rope_inplace, float) // clang-format on
