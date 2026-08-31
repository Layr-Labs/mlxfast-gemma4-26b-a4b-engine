// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

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

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

// One affine-4 dot product against a packed weight vector ALREADY held in
// registers. Byte-for-byte the bits == 4 arm of qdot: same nibble masks, same
// four-term expression, same accumulation order over i, same
// `scale * accum + sum * bias` close. Only the residence of `w` differs
// (thread instead of device), which is what lets one weight fetch serve four
// cohort input rows without holding four rows of x live at once.
template <typename U, int values_per_thread>
inline U qdot_affine4_registered(
    const thread uint16_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (w[i] & 0x000f) +
         x_thread[4 * i + 1] * (w[i] & 0x00f0) +
         x_thread[4 * i + 2] * (w[i] & 0x0f00) +
         x_thread[4 * i + 3] * (w[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// Consume the same two adjacent packed uint16 values through one aligned
// 32-bit device load. The low and high halves retain the original arithmetic
// order while halving the explicit weight-load instructions.
template <typename U, int values_per_thread>
inline U qdot_affine4_registered_word(
    uint packed_word,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(values_per_thread == 8, "Word load expects eight 4-bit values");
  const uint packed0 = packed_word & 0xffffu;
  const uint packed1 = packed_word >> 16;
  U accum =
      (x_thread[0] * (packed0 & 0x000f) +
       x_thread[1] * (packed0 & 0x00f0) +
       x_thread[2] * (packed0 & 0x0f00) +
       x_thread[3] * (packed0 & 0xf000));
  accum +=
      (x_thread[4] * (packed1 & 0x000f) +
       x_thread[5] * (packed1 & 0x00f0) +
       x_thread[6] * (packed1 & 0x0f00) +
       x_thread[7] * (packed1 & 0xf000));
  return scale * accum + sum * bias;
}

// Two independent affine-4 dot products over one packed weight vector. Each
// accumulator retains the scalar qdot operation order; only the packed weight
// load is shared between adjacent assignments routed to the same expert.
template <typename U, int values_per_thread>
inline void qdot_affine4_pair(
    const device uint8_t* w,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  static_assert(values_per_thread == 8, "Word load expects eight 4-bit values");
  const uint packedWord = *((const device uint*)w);
  const uint packed0 = packedWord & 0xffffu;
  const uint packed1 = packedWord >> 16;
  U accum0 =
      (x0[0] * (packed0 & 0x000f) +
       x0[1] * (packed0 & 0x00f0) +
       x0[2] * (packed0 & 0x0f00) +
       x0[3] * (packed0 & 0xf000));
  U accum1 =
      (x1[0] * (packed0 & 0x000f) +
       x1[1] * (packed0 & 0x00f0) +
       x1[2] * (packed0 & 0x0f00) +
       x1[3] * (packed0 & 0xf000));
  accum0 +=
      (x0[4] * (packed1 & 0x000f) +
       x0[5] * (packed1 & 0x00f0) +
       x0[6] * (packed1 & 0x0f00) +
       x0[7] * (packed1 & 0xf000));
  accum1 +=
      (x1[4] * (packed1 & 0x000f) +
       x1[5] * (packed1 & 0x00f0) +
       x1[6] * (packed1 & 0x0f00) +
       x1[7] * (packed1 & 0xf000));
  out0 = scale * accum0 + sum0 * bias;
  out1 = scale * accum1 + sum1 * bias;
}

// One affine-8 dot product against a byte weight vector ALREADY held in
// registers. Byte-for-byte the bits == 8 arm of qdot: same per-element
// multiply, same accumulation order over i, same `scale * accum + sum * bias`
// close. Only the residence of `w` differs (thread instead of device).
template <typename U, int values_per_thread>
inline U qdot_affine8_registered(
    const thread uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  for (int i = 0; i < values_per_thread; i++) {
    accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

// Two independent affine-8 dot products over one byte weight vector. Keep the
// per-row scalar accumulation order of qdot while sharing each weight load.
template <typename U, int values_per_thread>
inline void qdot_affine8_pair(
    const device uint8_t* w,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  U accum0 = 0;
  U accum1 = 0;
  for (int i = 0; i < values_per_thread; i++) {
    const uint8_t packed = w[i];
    accum0 += x0[i] * packed;
    accum1 += x1[i] * packed;
  }
  out0 = scale * accum0 + sum0 * bias;
  out1 = scale * accum1 + sum1 * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
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

template <typename T, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + quad_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(w + row * in_vec_size_w * quads_per_simd);
    const device T* sl = scales + row * in_vec_size_g * quads_per_simd;
    const device T* bl = biases + row * in_vec_size_g * quads_per_simd;

    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

// Exact-order affine4/g64 multi-row QMV. The frozen host launches M x-groups
// for each 8-output tile. Pair adjacent input rows in one group while keeping
// the stock two-simdgroup by four-output-row layout. Each active group caches a
// weight tile once and applies the stock arithmetic independently to one or two
// inputs; unused host groups return without reading weights. load_vector, the
// qdot expression, K accumulation order, and simd_sum remain identical to
// qmv_fast_impl for every output element.
// Word form: the caller now hands the same eight packed bytes as one aligned
// ulong, the load shape the promoted affine2 readout kernel already uses for
// this address pattern. Each 16-bit slice is exactly the uint16 the loop form
// read (little endian), the four nibble masks and the accumulation order over
// i are unchanged, and the slices are written explicitly so no variable
// 16 * i shift is left to the compiler.
template <typename U>
inline U qdot_affine4_loaded(
    ulong w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  const uint p0 = uint(w & 0xfffful);
  const uint p1 = uint((w >> 16) & 0xfffful);
  const uint p2 = uint((w >> 32) & 0xfffful);
  const uint p3 = uint(w >> 48);
  U accum =
      (x_thread[0] * (p0 & 0x000f) +
       x_thread[1] * (p0 & 0x00f0) +
       x_thread[2] * (p0 & 0x0f00) +
       x_thread[3] * (p0 & 0xf000));
  accum +=
      (x_thread[4] * (p1 & 0x000f) +
       x_thread[5] * (p1 & 0x00f0) +
       x_thread[6] * (p1 & 0x0f00) +
       x_thread[7] * (p1 & 0xf000));
  accum +=
      (x_thread[8] * (p2 & 0x000f) +
       x_thread[9] * (p2 & 0x00f0) +
       x_thread[10] * (p2 & 0x0f00) +
       x_thread[11] * (p2 & 0xf000));
  accum +=
      (x_thread[12] * (p3 & 0x000f) +
       x_thread[13] * (p3 & 0x00f0) +
       x_thread[14] * (p3 & 0x0f00) +
       x_thread[15] * (p3 & 0xf000));
  return scale * accum + sum * bias;
}

inline float2 qdot_affine4_loaded_pair(
    ulong w,
    const thread float* x0,
    const thread float* x1,
    float scale,
    float bias,
    float2 sum) {
  const uint p0 = uint(w & 0xfffful);
  const uint p1 = uint((w >> 16) & 0xfffful);
  const uint p2 = uint((w >> 32) & 0xfffful);
  const uint p3 = uint(w >> 48);
  float2 accum =
      (float2(x0[0], x1[0]) * (p0 & 0x000f) +
       float2(x0[1], x1[1]) * (p0 & 0x00f0) +
       float2(x0[2], x1[2]) * (p0 & 0x0f00) +
       float2(x0[3], x1[3]) * (p0 & 0xf000));
  accum +=
      (float2(x0[4], x1[4]) * (p1 & 0x000f) +
       float2(x0[5], x1[5]) * (p1 & 0x00f0) +
       float2(x0[6], x1[6]) * (p1 & 0x0f00) +
       float2(x0[7], x1[7]) * (p1 & 0xf000));
  accum +=
      (float2(x0[8], x1[8]) * (p2 & 0x000f) +
       float2(x0[9], x1[9]) * (p2 & 0x00f0) +
       float2(x0[10], x1[10]) * (p2 & 0x0f00) +
       float2(x0[11], x1[11]) * (p2 & 0xf000));
  accum +=
      (float2(x0[12], x1[12]) * (p3 & 0x000f) +
       float2(x0[13], x1[13]) * (p3 & 0x00f0) +
       float2(x0[14], x1[14]) * (p3 & 0x0f00) +
       float2(x0[15], x1[15]) * (p3 & 0xf000));
  return scale * accum + sum * bias;
}

template <typename T, int M>
METAL_FUNC void qmv_fast_crossrow_affine4_g64(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(M >= 2 && M <= 9, "multi-row QMV supports M in [2, 9]");
  constexpr int inputs_per_group = 2;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int in_vec_bytes_per_row_divisor = 2;
  constexpr int bytes_per_lane = 8;

  const int first_m = int(tid.x) * inputs_per_group;
  if (first_m >= M) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;
  const int in_vec_size_w = in_vec_size / in_vec_bytes_per_row_divisor;
  const int in_vec_size_g = in_vec_size / 64;

  const bool has_pair = first_m + 1 < M;
  thread float2 pair_result[rows_per_simd];
  thread float single_result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    pair_result[r] = 0.0f;
    single_result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    // One aligned 8-byte load per row replaces four 2-byte loads -- the load
    // shape qmv_fast_singlerow_affine2_g64 already uses for this address
    // pattern. This fast path only dispatches with in_vec_size a multiple of
    // block_size, so the row stride in_vec_size_w, the block offset k / 2 and
    // the lane offset simd_lid * bytes_per_lane are all multiples of 8 over
    // the weight base.
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];

    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint8_t* wb =
          reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 2 + simd_lid * bytes_per_lane;
      packed[r] = *reinterpret_cast<const device ulong*>(wb);
      const int group_index =
          row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm0 =
        x + first_m * in_vec_size + k + simd_lid * values_per_thread;
    const float sum0 =
        load_vector<T, float, values_per_thread, 4>(xm0, x0);
    if (has_pair) {
      thread float x1[values_per_thread];
      const device T* xm1 = xm0 + in_vec_size;
      const float sum1 =
          load_vector<T, float, values_per_thread, 4>(xm1, x1);
      for (int r = 0; r < rows_per_simd; r++) {
        pair_result[r] += qdot_affine4_loaded_pair(
            packed[r], x0, x1, scale_local[r], bias_local[r],
            float2(sum0, sum1));
      }
    } else {
      for (int r = 0; r < rows_per_simd; r++) {
        single_result[r] += qdot_affine4_loaded<float>(
            packed[r], x0, scale_local[r], bias_local[r], sum0);
      }
    }
  }

  if (has_pair) {
    for (int r = 0; r < rows_per_simd; r++) {
      const float reduced0 = simd_sum(pair_result[r].x);
      const float reduced1 = simd_sum(pair_result[r].y);
      if (simd_lid == 0) {
        y[first_m * out_vec_size + out_row + r] = static_cast<T>(reduced0);
        y[(first_m + 1) * out_vec_size + out_row + r] =
            static_cast<T>(reduced1);
      }
    }
  } else {
    for (int r = 0; r < rows_per_simd; r++) {
      const float reduced = simd_sum(single_result[r]);
      if (simd_lid == 0) {
        y[first_m * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// Wider row sharing for the affine4/g64 multi-row QMV. Same contract as
// qmv_fast_crossrow_affine4_g64: the frozen host launches M x-groups for each
// 8-output tile, so a group that claims NA adjacent input rows lets the
// remaining host groups return without reading weights. NA up to 4 shares one
// nibble mask and one integer-to-float conversion across NA inputs while
// holding only four x values per input live at a time, so the register
// footprint stays near the two-input kernel's. load_vector, the qdot
// expression, the K accumulation order and simd_sum are unchanged for every
// output element.
template <typename T, int NA, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid) {
  static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint16_t packed[rows_per_simd][4];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      // Same one-aligned-ulong load shape as the pair kernel above; the
      // register layout and every consumer stay uint16, so the unpack is a
      // byte-identical relabeling of the same loaded bytes.
      const ulong w01 = *reinterpret_cast<const device ulong*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane);
      packed[r][0] = uint16_t(w01 & 0xfffful);
      packed[r][1] = uint16_t((w01 >> 16) & 0xfffful);
      packed[r][2] = uint16_t((w01 >> 32) & 0xfffful);
      packed[r][3] = uint16_t(w01 >> 48);
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * values_per_thread + 4 * i;
        thread float xc[4];
        if (DIRECT_NIBBLES) {
          xc[0] = static_cast<float>(xm[0]);
          xc[1] = static_cast<float>(xm[1]);
          xc[2] = static_cast<float>(xm[2]);
          xc[3] = static_cast<float>(xm[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        } else {
          sums[m] += load_vector<T, float, 4, 4>(xm, xc);
        }
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < rows_per_simd; r++) {
        if (DIRECT_NIBBLES) {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * ((packed[r][i] >> 4) & 0x000f) +
                         a2 * ((packed[r][i] >> 8) & 0x000f) +
                         a3 * ((packed[r][i] >> 12) & 0x000f));
        } else {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * (packed[r][i] & 0x00f0) +
                         a2 * (packed[r][i] & 0x0f00) +
                         a3 * (packed[r][i] & 0xf000));
        }
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] =
            static_cast<T>(reduced);
      }
    }
  }
}

// Single-row (M == 1) affine2/g64 fast QMV for the coarse compact draft
// readout (out_vec_size == 98_336, bits == 2) of the promoted draft-rerank
// scheme, at 32 values per lane: each lane loads ONE uint64 (32 packed
// 2-bit values) per row per k-block, halving load count and k-blocks versus
// the generic 16-value form. Duo values are extracted by shift and
// multiplied by the UNSCALED activation: (x / 4^k) * (w & (3 << 2k)) and
// x * ((w >> 2k) & 3) are the same real product (power-of-two scaling is
// exact in FP32), so every elementary product equals the generic
// qmv_fast_impl<T, 64, 2> value; the wider lane coverage reassociates the
// FP32 partial sums, which is safe for this stage because the coarse
// shortlist is approximate by design and the exact affine-4 rerank plus
// target verification decide every emitted token. The serial leg runs no
// 2-bit matmul (all its projections are affine-4), and out_vec_size ==
// 98_336 exists only in the compact draft readout, so the dispatch gate
// below cannot touch the serial numerator or the denominator band.
template <typename T>
METAL_FUNC void qmv_fast_singlerow_affine2_g64(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;  // 32 values x 2 bits = 8 bytes
  const int in_vec_size_w = in_vec_size / 4;   // weight bytes per output row
  const int in_vec_size_g = in_vec_size / 64;  // scale groups per output row

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint8_t* ws = reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 4 + simd_lid * bytes_per_lane;
      packed[r] = *reinterpret_cast<const device ulong*>(ws);
      // 32 values per lane = half of one 64-value group.
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 32; j++) {
        accum += x0[j] * float((packed[r] >> (2 * j)) & 0x03ul);
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// IPG = ceil(M / ceil(M / 4)): the fewest weight streams reachable at NA <= 4,
// with the remainder spread evenly so no group runs a one-row tail.
template <typename T, int M, int IPG, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_m(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(M >= 3 && M <= 9, "wide multi-row QMV dispatch covers M in [3, 9]");
  static_assert(M % IPG != 1, "a one-input tail group is not instantiated");
  constexpr int TAIL = M % IPG;
  const int first_m = int(tid.x) * IPG;
  if (first_m >= M) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  if (TAIL == 0 || M - first_m >= IPG) {
    qmv_fast_crossrow_affine4_g64_wide<T, IPG, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  } else {
    qmv_fast_crossrow_affine4_g64_wide<
        T, (TAIL >= 2 ? TAIL : 2), DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k <= in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k <= in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int tail_values = static_cast<int>(in_vec_size - k);
    if (tail_values > 0) {
      // Affine callers keep K a whole number of quantization groups and k
      // advances by whole blocks, so the tail is a whole number of
      // values_per_thread lane packets: routed-expert down_proj K=704 leaves
      // 192 values = 24 complete packets, dense down_proj K=2112 (8-bit)
      // leaves 64 = 16.  Active lanes run the fixed unrolled loader and qdot;
      // the dynamic safe-tail remains only for a genuinely partial packet,
      // which no affine caller presents.
      if (tail_values % values_per_thread == 0) {
        const uint active_tail_lanes = uint(tail_values / values_per_thread);
        if (simd_lid < active_tail_lanes) {
          U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device T* sl = scales + row * in_vec_size_g;
            const device T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] +=
                qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
          }
        }
      } else {
        const int remaining = clamp(
            static_cast<int>(tail_values - simd_lid * values_per_thread),
            0,
            values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe<T, U, values_per_thread, bits>(
              x, x_thread, remaining);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const device T* sl = scales + row * in_vec_size_g;
            const device T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] += qdot_safe<U, values_per_thread, bits>(
                wl, x_thread, s, b, sum, remaining);
          }
        }
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_pair_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    device T* y0,
    device T* y1,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread float x1_thread[values_per_thread];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    float sum0 = load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
    float sum1 = load_vector<T, float, values_per_thread, 4>(x1, x1_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine4_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
  }

  // Every Gemma 4 caller entering this specialized g64 path has K aligned to
  // 64.  The final block therefore contains an integral number of complete
  // eight-value lane packets (32 lanes for K=2816, 24 for expert down_proj
  // K=704); no active lane needs the generic dynamic safe-tail loops.
  const uint active_tail_lanes =
      uint((in_vec_size - k) / values_per_thread);
  if (simd_lid < active_tail_lanes) {
    float sum0 =
        load_vector<T, float, values_per_thread, 4>(x0, x0_thread);
    float sum1 =
        load_vector<T, float, values_per_thread, 4>(x1, x1_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine4_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
    }
  }
}

// Four-row weight-stream sharing for the ordinary (plain-order) affine4/g64
// QMV, held to a PAIR-SIZED register footprint. Same geometry as
// qmv_affine4_g64_pair_impl: two simdgroups by four output rows per
// 64-thread group, values_per_thread = 8, block_size = 256,
// scale_step_per_thread = 8, same load_vector / load_vector_safe tail.
//
// RESIDENCY is the whole point. The retired four-row quad this replaces held
// all four cohort rows of x live across a block (4 x 8 = 32 floats) on top of
// its 16 accumulators, and measured SLOWER than the two-row pair it was meant
// to beat -- 226.3 us against 208.7 us at N = 8192 on the ranked box, with the
// halved weight stream never converting. This kernel fetches the block's
// packed weights and scale/bias into registers ONCE and then walks the four
// input rows in sequence through a SINGLE eight-value x buffer, so only one
// row of x is live at a time -- the discipline of
// qmv_fast_crossrow_affine4_g64_wide, which keeps four x values per input row
// live and states the same reason. With it the collapse converts: 197.5 us at
// N = 8192, under BOTH the pair kernel and the retired quad.
//
// Exactness is unchanged and argued the same way: each (output row, input row)
// pair keeps its own accumulator, its own K-loop order, and its own simd_sum;
// `qdot_affine4_registered` is the bits == 4 arm of `qdot` verbatim. Only the
// LOADS are shared, so every output element's add sequence is identical to
// stock qmv_impl.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_quad_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    device T* y0,
    device T* y1,
    device T* y2,
    device T* y3,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread uint packed[results_per_simdgroup];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};
  thread float result2[results_per_simdgroup] = {0};
  thread float result3[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  x2 += simd_lid * values_per_thread;
  x3 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;
  y3 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] =
          *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 4>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x3, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
    x3 += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] =
          *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum =
        load_vector_safe<T, float, values_per_thread, 4>(x0, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x1, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x2, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x3, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    result3[row] = simd_sum(result3[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
      y3[row] = static_cast<T>(result3[row]);
    }
  }
}

// Three-row weight-stream sharing: qmv_affine4_g64_quad_stream_impl with the
// fourth input row deleted, for same-expert gather runs of exactly three. The
// register discipline (one live x buffer), K-loop order, per-(output, input)
// accumulators, and qdot_affine4_registered arithmetic are the quad's own, so
// every output element's add sequence remains identical to stock qmv_impl.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine4_g64_triple_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    device T* y0,
    device T* y1,
    device T* y2,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread uint packed[results_per_simdgroup];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};
  thread float result2[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  x2 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] =
          *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 4>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 4>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] =
          *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum =
        load_vector_safe<T, float, values_per_thread, 4>(x0, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x1, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum =
        load_vector_safe<T, float, values_per_thread, 4>(x2, x_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
    }
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_pair_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    device T* y0,
    device T* y1,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 4;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread float x1_thread[values_per_thread];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    float sum0 = load_vector<T, float, values_per_thread, 8>(x0, x0_thread);
    float sum1 = load_vector<T, float, values_per_thread, 8>(x1, x1_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine8_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
  }

  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    float sum0 = load_vector_safe<T, float, values_per_thread, 8>(
        x0, x0_thread, remaining);
    float sum1 = load_vector_safe<T, float, values_per_thread, 8>(
        x1, x1_thread, remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float dot0;
      float dot1;
      qdot_affine8_pair<float, values_per_thread>(
          wl, x0_thread, x1_thread, sl[0], bl[0], sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
    }
  }
}

// Four-row byte-weight-stream sharing for the affine8/g64 QMV, held to a
// PAIR-SIZED register footprint, exactly as
// qmv_affine4_g64_quad_stream_impl does for the nibble path. Same geometry as
// qmv_affine8_g64_pair_impl: two simdgroups by four output rows per 64-thread
// group, values_per_thread = 4, block_size = 128, scale_step_per_thread = 16,
// same load_vector / load_vector_safe tail.
//
// The block's byte weights and scale/bias are fetched into registers ONCE and
// the four cohort input rows then walk through a SINGLE four-value x buffer,
// so only one row of x is live at a time. This is the reason an earlier
// affine-8 quad that held all four rows of x live measured as a regression:
// the residency, not the arithmetic. Each (output row, input row) pair keeps
// its own accumulator, its own K-loop order and its own simd_sum, and
// qdot_affine8_registered is the bits == 8 arm of qdot verbatim, so every
// output element's add sequence is identical to stock qmv_impl -- only the
// LOADS are shared.
template <typename T, const int group_size, const int bits>
METAL_FUNC void qmv_affine8_g64_quad_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    device T* y0,
    device T* y1,
    device T* y2,
    device T* y3,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 4;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread uint8_t packed[results_per_simdgroup][bytes_per_thread];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result0[results_per_simdgroup] = {0};
  thread float result1[results_per_simdgroup] = {0};
  thread float result2[results_per_simdgroup] = {0};
  thread float result3[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  x1 += simd_lid * values_per_thread;
  x2 += simd_lid * values_per_thread;
  x3 += simd_lid * values_per_thread;
  y0 += out_row;
  y1 += out_row;
  y2 += out_row;
  y3 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }

    ws += block_size;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
    x1 += block_size;
    x2 += block_size;
    x3 += block_size;
  }

  // Dense Gemma 4 K is g64-aligned, so the tail is always a whole number of
  // four-value lane packets.  In particular down_proj K=2112 leaves exactly
  // 16 active lanes; use the fixed unrolled load instead of four dynamic
  // safe-tail loops while preserving each lane's qdot and simd_sum order.
  const uint active_tail_lanes =
      uint((in_vec_size - k) / values_per_thread);
  if (simd_lid < active_tail_lanes) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      for (int i = 0; i < bytes_per_thread; i++) {
        packed[row][i] = wl[i];
      }
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum = load_vector<T, float, values_per_thread, 8>(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = load_vector<T, float, values_per_thread, 8>(x3, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result3[row] += qdot_affine8_registered<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    result1[row] = simd_sum(result1[row]);
    result2[row] = simd_sum(result2[row]);
    result3[row] = simd_sum(result3[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
      y1[row] = static_cast<T>(result1[row]);
      y2[row] = static_cast<T>(result2[row]);
      y3[row] = static_cast<T>(result3[row]);
    }
  }
}

// GROUP-EXACT-MMA -- the fp32 `simdgroup_float8x8` body for the M = 8 decode
// cohort on 4-bit affine g64 weights. `A` holds the raw weight codes
// (8 output rows x 8 k-slots), `B` holds X^T (8 k-slots x 8 cohort rows) and
// `C` is zeroed per g64 group, so each group's 64 products are formed exactly
// (a bf16 x times a 4-bit code needs at most 12 significant bits) and summed
// by the matrix unit before the single combined `acc += s * C + rs * b` close
// in ascending k. Inside group g, fragment j and slot s name
// k(j, s) = 64 g + 8 s + j; a dot product is order free, so A and B may share
// any bijection, and this one makes every lane's fragment one contiguous
// load: 16 nibbles (`uint2`) of one weight row for A, two 8-value runs
// (`uint4`) of two cohort rows for B.
struct mma8_coord {
  short fm;
  short fn;
};

// steel/gemm/mma.h's `get_coord` arithmetic, reproduced locally so the same
// text compiles wherever this body is pasted: lane (fm, fn) owns elements
// (fm, fn) and (fm, fn + 1) of every 8x8 operand.
inline mma8_coord mma8_lane(uint lane) {
  const short qid = short(lane / 4);
  return {
      short((qid & 4) + short((lane / 2) % 4)),
      short((qid & 2) * 2 + short(lane % 2) * 2)};
}

// The x-side loads pull sixteen bytes at a time and split them into eight
// 16-bit lanes, which only makes sense for a 2-byte T. `affine_qmv` is also
// instantiated for `float`; the tier gate carries `sizeof(T) == 2` so the
// float instantiation never runs this body, and this primary template is what
// lets it still compile.
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

// Widening a 16-bit float to fp32 is exact, so these two reproduce the
// reference's own operand values bit for bit.
template <typename T>
inline float mma8_lo(uint u) {
  return float(mma8_u16<T>::cast(ushort(u & 0xFFFFu)));
}

template <typename T>
inline float mma8_hi(uint u) {
  return float(mma8_u16<T>::cast(ushort(u >> 16)));
}

// Textual twin of `load_vector<T, float, 8, 4>`'s `sum` on the same aligned
// 8-run that the reference lane owns: the parenthesised 4-tuple is evaluated
// on T exactly as in the reference, then the two trees are added in fp32. The
// bias term of the affine form therefore reuses the reference's own
// elementary values, not a re-derived sum.
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

#define MMA8_SETB(BB, W, HI)                       \
  BB.thread_elements()[0] = mma8_##HI<T>(r0.W);    \
  BB.thread_elements()[1] = mma8_##HI<T>(r1.W);

#define MMA8_STEP(BB, J)                                          \
  A.thread_elements()[0] = float(extract_bits(wv.x, 4 * (J), 4)); \
  A.thread_elements()[1] = float(extract_bits(wv.y, 4 * (J), 4)); \
  simdgroup_multiply_accumulate(C, A, BB, C);

// x is [8, K] with K % 64 == 0, w is packed [N, K / 8] uint32, scales and
// biases are [N, K / 64], y is [8, N]. `n0` is the first of the eight output
// rows this threadgroup owns. KS = 2 splits the K / 64 groups between the two
// simdgroups of the host's (32, 2, 1) threadgroup; an odd group count gives
// the extra group to simdgroup 0, which is deterministic and independent of
// scheduling. `red` is 32 float2 of threadgroup memory for the KS = 2 close.
template <typename T, int KS>
METAL_FUNC void gemma4_qmv_mma8_affine4_g64_impl(
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

    // Each B lane owns the two 8-runs whose run sums the C lane (fm, fn)
    // needs; three xor-butterfly steps over the fm lane bits broadcast
    // RS[g][fn] and RS[g][fn + 1] to all eight lanes of the fn column group.
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

template <typename T, const int group_size, const int bits>
METAL_FUNC void qvm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    const int in_vec_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales += out_col / group_size + simd_lid * out_vec_size_g;
  biases += out_col / group_size + simd_lid * out_vec_size_g;
  x += tid.x * in_vec_stride + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);

      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
      scale = 0;
      bias = 0;
    }
    qouter<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local, x_local, scale, bias, result);
  }

// Accumulate in the simdgroup
#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_n_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  auto wl = (const device uint8_t*)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T, int group_size, int bits, int D, bool batched>
[[kernel]] void affine_qmv_quad(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_quad_impl<T, group_size, bits, D>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      quad_gid,
      quad_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 ntg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  if (!batched && group_size == 64 && bits == 2 && out_vec_size == 98336 &&
      ntg.x == 1) {
    // M == 1 coarse draft readout (draft-rerank scheme): the ONE 2-bit shape
    // in the scored path; proposal-only by construction (see kernel header).
    qmv_fast_singlerow_affine2_g64<T>(
        w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid,
        simd_lid);
    return;
  }
  if (!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024) {
    if (out_vec_size >= 4096) {
      // Wide row sharing needs enough output tiles to keep the machine fed;
      // below 4096 outputs the reduced x-group count thins the grid, so the
      // promoted pair kernel is kept there byte-for-byte.
      switch (ntg.x) {
        case 2:
          qmv_fast_crossrow_affine4_g64<T, 2>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 3:
          qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 4:
          qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 5:
          qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 6:
          qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 7:
          qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 8:
          // 3+3+2, not 4+4. M = 8 is the only hot width whose EVEN split needs
          // two simultaneous vec<float,4> accumulators in every active worker;
          // M = 9 uses three-lane vectors and profiles CHEAPER despite more work
          // (319 / 437 / 216 us for M = 7 / 8 / 9 in the public cross-row study)
          // — a register cliff, not work scaling.
          // Exact: these lanes carry INDEPENDENT input rows and are never reduced
          // across (simd_sum reduces along K WITHIN a row), so moving a row from
          // lane 3 of a four-wide vector to lane 0 of a two-wide one cannot
          // reorder its scalar chain. Template admits it: M in [3,9], 8 % 3 == 2
          // (no one-row tail), IPG 3 inside the wide helper's [2,4].
          // Receipts: 85d5bca3 2.91143, yzxoi 2.92675.
          // SYNERGY with the streak gate above, which is why they ship together:
          // gate 2 reaches the width-8 verify SOONER, so this kernel fires MORE.
          qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:
          break;
      }
    } else {
      switch (ntg.x) {
        case 2:
          qmv_fast_crossrow_affine4_g64<T, 2>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 3:
          qmv_fast_crossrow_affine4_g64<T, 3>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 4:
          qmv_fast_crossrow_affine4_g64<T, 4>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 5:
          qmv_fast_crossrow_affine4_g64<T, 5>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 6:
          qmv_fast_crossrow_affine4_g64<T, 6>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 7:
          qmv_fast_crossrow_affine4_g64<T, 7>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 8:
          qmv_fast_crossrow_affine4_g64<T, 8>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 9:
          qmv_fast_crossrow_affine4_g64<T, 9>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:
          break;
      }
    }
  }
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 ntg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  if (!batched && group_size == 64 && bits == 4 && ntg.x == 8 &&
      ntg.z == 1 && in_vec_size % 64 == 0 && out_vec_size >= 8 &&
      out_vec_size % 8 == 0) {
    // The ruled decode cohort presents eight input rows to ordinary QMV.
    // MMA-QKV S1 -- GROUP-EXACT-MMA tier. It replaces, for the 4-bit affine
    // g64 dense decode projections wide enough to fill the machine (q/k/v:
    // N = 1024 / 2048 / 4096 / 8192 over K = 2816; the tied head only if its
    // Swift MMA kernel is bypassed, since that road is tried first), the
    // scalar quad_stream tier below: instead of eight per-lane 8-term chains
    // that are each scaled and then reduced by `simd_sum`, one fp32
    // `simdgroup_float8x8` multiply-accumulate chain forms all 64 products of
    // a g64 group and sums them before the single `s * C + rs * b` close. Every
    // elementary term is the reference's own -- the products x * q are exact in
    // fp32 (a bf16 x carries 8 significant bits, a code 4), scales and biases
    // widen exactly, `mma8_runsum4` reproduces `load_vector`'s bf16 4-tuple sum
    // order on the same aligned 8-run, and the group closes are chained in
    // ascending k -- so the ONLY numeric deviation is fp32 reassociation inside
    // the 64-wide group dot (plus the two-halves add of the KS = 2 split). This
    // is the first non-bit-exact QMV tier here; measured against the stock M = 1
    // road over 50 random cohorts per plane at K = 2816, the deviation is at
    // most 1 bf16 ulp for every output above the 2^-10 * row-max magnitude gate
    // (non-zero fraction ~1.4e-4, 0 argmax flips over 400 rows per plane), it
    // is run-to-run bitwise deterministic, and the body measured 0.41-0.51x the
    // quad_stream body net of the dispatch floor on an M4 Max. Outputs cancelled
    // below ~2^-8 of their term mass can show a second relative ulp; they are
    // numerically negligible and never argmax candidates. KILL SWITCH: set
    // `kGemma4QmvMma8Affine4` to false and this branch vanishes at compile time,
    // restoring the quad_stream and pair tiers below byte for byte -- nothing
    // beneath this block was edited. Raising `kGemma4QmvMma8Affine4FloorN`
    // returns individual planes the same way. (MSL forbids a program-scope
    // `constexpr`, so the two switches live at the top of the tier they guard.)
    constexpr bool kGemma4QmvMma8Affine4 = true;
    constexpr int kGemma4QmvMma8Affine4FloorN = 1024;
    if (kGemma4QmvMma8Affine4 && sizeof(T) == 2 && ntg.z == 1 &&
        in_vec_size % 64 == 0 &&
        out_vec_size >= kGemma4QmvMma8Affine4FloorN && out_vec_size % 8 == 0) {
      // Seven of the eight host x-groups retire before any load; the eighth
      // produces all eight cohort columns of its eight output rows.
      if (tid.x != 0) {
        return;
      }
      threadgroup float2 red[32];
      gemma4_qmv_mma8_affine4_g64_impl<T, 2>(
          w,
          scales,
          biases,
          x,
          y,
          in_vec_size,
          out_vec_size,
          8 * int(tid.y),
          red,
          simd_gid,
          simd_lid);
      return;
    }
    if (out_vec_size >= 1024) {
      // WIDE-N tier -- every non-`fast` 4-bit decode plane on this model:
      // full-attention k_proj N = 1024 (k_eq_v), k/v_proj N = 2048, sliding
      // q_proj N = 4096, full q_proj N = 8192, tied lm_head N = 262144. K is
      // 2816, not a multiple of 512, so none of these reach affine_qmv_fast
      // and its cross-row family; before this they ran two-row pair (and, at
      // N >= 8192, a four-row quad that held 32 floats of x live and lost to
      // the pair it replaced).
      //
      // One packed-weight stream feeds FOUR cohort rows in two active x-groups
      // (4+4); the remaining host groups return. Per-row qdot, K-loop and
      // simd_sum keep the stock qmv_impl sequence for every output element --
      // only loads are shared.
      //
      // Measured on the ranked box (M4 Pro, B = 8, streamed weight pool so
      // every dispatch pulls from DRAM), us/dispatch, incumbent -> this:
      //   N = 1024   27.3 -> 26.1     N = 2048   54.1 -> 51.3
      //   N = 4096  106.7 -> 101.6    N = 8192  233.6 -> 201.5
      // The floor sits at 1024 because that is the smallest plane measured to
      // convert; below it (router.proj N = 128) two active x-groups leave only
      // (N / 8) * 2 threadgroups and the promoted pair kernel is kept
      // byte-for-byte.
      const int first_m = int(tid.x) * 4;
      if (first_m >= 8) {
        return;
      }
      qmv_affine4_g64_quad_stream_impl<T, 64, 4>(
          w,
          scales,
          biases,
          x + first_m * in_vec_size,
          x + (first_m + 1) * in_vec_size,
          x + (first_m + 2) * in_vec_size,
          x + (first_m + 3) * in_vec_size,
          y + first_m * out_vec_size,
          y + (first_m + 1) * out_vec_size,
          y + (first_m + 2) * out_vec_size,
          y + (first_m + 3) * out_vec_size,
          in_vec_size,
          tid,
          simd_gid,
          simd_lid);
      return;
    }
    // Claim adjacent rows in four active x-groups and let the remaining host
    // groups return. The established pair helper shares each packed-weight
    // load while preserving each row's qdot, K-loop, and simd_sum order.
    const int first_m = int(tid.x) * 2;
    if (first_m >= 8) {
      return;
    }
    qmv_affine4_g64_pair_impl<T, 64, 4>(
        w,
        scales,
        biases,
        x + first_m * in_vec_size,
        x + (first_m + 1) * in_vec_size,
        y + first_m * out_vec_size,
        y + (first_m + 1) * out_vec_size,
        in_vec_size,
        tid,
        simd_gid,
        simd_lid);
    return;
  }
  if (!batched && group_size == 64 && bits == 8 && ntg.x == 8 &&
      ntg.z == 1 && in_vec_size % 64 == 0 && out_vec_size >= 8 &&
      out_vec_size % 8 == 0) {
    // Dense decode projections use byte weights.
    if (out_vec_size >= 1024) {
      // WIDE-N tier -- the dense MLP of all 30 layers: gate_proj and up_proj
      // N = 2112 over K = 2816, down_proj N = 2816 over K = 2112. One
      // byte-weight stream feeds FOUR cohort rows in two active x-groups
      // (4+4); the remaining host groups return, and per-row qdot, K-loop and
      // simd_sum stay the stock qmv_impl sequence.
      //
      // Measured on the ranked box (M4 Pro, B = 8, streamed weight pool),
      // us/dispatch, incumbent pair -> this:
      //   N = 2112 (gate/up)  64.4 -> 56.2     N = 2816 (down)  67.4 -> 59.0
      // Same 1024 floor as the nibble tier, which keeps router.proj (N = 128)
      // on the promoted pair kernel byte-for-byte.
      const int first_m = int(tid.x) * 4;
      if (first_m >= 8) {
        return;
      }
      qmv_affine8_g64_quad_stream_impl<T, 64, 8>(
          w,
          scales,
          biases,
          x + first_m * in_vec_size,
          x + (first_m + 1) * in_vec_size,
          x + (first_m + 2) * in_vec_size,
          x + (first_m + 3) * in_vec_size,
          y + first_m * out_vec_size,
          y + (first_m + 1) * out_vec_size,
          y + (first_m + 2) * out_vec_size,
          y + (first_m + 3) * out_vec_size,
          in_vec_size,
          tid,
          simd_gid,
          simd_lid);
      return;
    }
    // Pair adjacent cohort rows so each weight byte feeds both exact per-row
    // dot-product streams.
    const int first_m = int(tid.x) * 2;
    if (first_m >= 8) {
      return;
    }
    qmv_affine8_g64_pair_impl<T, 64, 8>(
        w,
        scales,
        biases,
        x + first_m * in_vec_size,
        x + (first_m + 1) * in_vec_size,
        y + first_m * out_vec_size,
        y + (first_m + 1) * out_vec_size,
        in_vec_size,
        tid,
        simd_gid,
        simd_lid);
    return;
  }
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void affine_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      in_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, const int group_size, const int bits, int split_k = 32>
[[kernel]] void affine_qvm_split_k(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    const constant int& final_block_size [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      y,
      out_vec_size * M,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  // The in_vec_stride is the full K dimension, not the partition size
  int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;

  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size_adj,
      out_vec_size,
      in_vec_stride,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_partition_size [[buffer(8)]],
    const constant int& split_k_partition_stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int k_start = tid.z * k_partition_size;
  x += k_start;

  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      (const device uint32_t*)wl,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

// One affine-4 dot product against a packed weight WORD (32 values / 8
// nibbles) held in registers. Byte-for-byte the bits == 4 /
// values_per_thread == 8 arm of `qdot`: `ws[0]` is the low half-word and
// `ws[1]` the high half-word of the same aligned uint, so the four nibble
// masks, the two 4-term sums, the accumulation order over i and the
// `scale * accum + sum * bias` close are unchanged. Only the load shape
// differs -- one 4-byte load instead of two 2-byte loads.
inline float qdot_affine4_g64_word(
    uint v,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const uint lo = v & 0x0000FFFFu;
  const uint hi = v >> 16;
  float accum = 0;
  accum +=
      (x_thread[0] * float(lo & 0x000fu) + x_thread[1] * float(lo & 0x00f0u) +
       x_thread[2] * float(lo & 0x0f00u) + x_thread[3] * float(lo & 0xf000u));
  accum +=
      (x_thread[4] * float(hi & 0x000fu) + x_thread[5] * float(hi & 0x00f0u) +
       x_thread[6] * float(hi & 0x0f00u) + x_thread[7] * float(hi & 0xf000u));
  return scale * accum + sum * bias;
}
// EXPERT-SINGLES: the SINGLETON arm of the routed-expert gather QMV.
// Diverse decode routing (8 streams x top-8 over 128 experts, sorted into
// 64 assignments) leaves most runs at length ONE, so the RUN-QUAD leader
// rule above hands the majority of both expert planes to the stock
// `qmv_impl` -- the one arm of the hot expert path that had never been
// microbenched (the dequant-once / prefetch / unroll knobs were only ever
// tried on the tied-head quad_stream body, where they lost).
//
// This is that arm with LOADS-ONLY rescheduling. Identical lane -> K
// mapping, identical per-block `load_vector` transform, identical
// eight-term `qdot` expression evaluated in the identical 4 + 4 grouping,
// identical per-row accumulator, identical `simd_sum` and store: every
// output element's add sequence is byte-for-byte the sequence `qmv_impl`
// produces for it. Only the SHAPE of the loads changes.
//
//   WVEC : the two adjacent `uint16_t` loads the bits == 4 arm of `qdot`
//          emits per (row, K-block) become ONE aligned 4-byte load. The
//          packed row base is uint32-aligned at every block boundary
//          (in_vec_size_w = K / 2 with K in {2816, 704}, lane offset
//          simd_lid * 4, block stride 128), and `ws[0]` / `ws[1]` are the
//          low / high half-words of that word, so the four nibble masks
//          and their two 4-term sums are unchanged.
//   PF   : software prefetch of the NEXT block's four weight words. One
//          x row is live in the singleton arm, so the +13..+40% extra
//          live state that sank PF on the 4-row quad_stream body does not
//          apply here.
//   KFIX : in_vec_size as a compile-time constant. The gemma4 gate has
//          already proven in_vec_size is 2816 (gate/up) or 704 (down), so
//          the K-loop trip count and every stride fold constant-fold.
//
// Instantiated only under the gemma4 pair-geometry gate, which is
// compile-time false unless group_size == 64 && bits == 4; the affine-4 /
// g64 constants below are hardcoded exactly as `qmv_affine4_g64_pair_impl`
// hardcodes them.
template <typename T, int group_size, int bits, int KFIX, bool WVEC, bool PF>
METAL_FUNC void qmv_affine4_g64_singles_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size_rt,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;
  constexpr int block_bytes = 128;
  constexpr int qgroup = 64;

  const int in_vec_size = (KFIX > 0) ? KFIX : in_vec_size_rt;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / qgroup;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);
  if (out_row >= out_vec_size) {
    return;
  }

  ws += used_out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + used_out_row;

  const int nblocks = in_vec_size / block_size;
  const device uint8_t* ws0 = ws;

  thread uint wpf[results_per_simdgroup];
  if (PF) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      wpf[row] = *((const device uint*)(ws0 + row * in_vec_size_w));
    }
  }

  for (int blk = 0; blk < nblocks; blk++) {
    U sum = load_vector<T, U, values_per_thread, 4>(x, x_thread);

    thread uint wcur[results_per_simdgroup];
    if (PF) {
      for (int row = 0; row < results_per_simdgroup; row++) {
        wcur[row] = wpf[row];
      }
      const int nextblk = (blk + 1 < nblocks) ? (blk + 1) : blk;
      const device uint8_t* wsn = ws0 + nextblk * block_bytes;
      for (int row = 0; row < results_per_simdgroup; row++) {
        wpf[row] = *((const device uint*)(wsn + row * in_vec_size_w));
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      if (PF) {
        result[row] += qdot_affine4_g64_word(wcur[row], x_thread, s, b, sum);
      } else if (WVEC) {
        const uint v = *((const device uint*)(ws + row * in_vec_size_w));
        result[row] += qdot_affine4_g64_word(v, x_thread, s, b, sum);
      } else {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        result[row] += qdot<U, values_per_thread, 4>(wl, x_thread, s, b, sum);
      }
    }

    ws += block_bytes;
    scales += block_size / qgroup;
    biases += block_size / qgroup;
    x += block_size;
  }

  const int tail_values = in_vec_size - nblocks * block_size;
  if (tail_values > 0) {
    // Affine callers keep K a whole number of quantization groups and the
    // block loop advances by whole blocks, so the tail is a whole number of
    // values_per_thread lane packets (down_proj K = 704 leaves 192 = 24).
    // The dynamic safe tail below is kept verbatim from `qmv_impl` for the
    // genuinely partial packet no affine caller presents.
    if (tail_values % values_per_thread != 0) {
      const int remaining = clamp(
          static_cast<int>(tail_values - simd_lid * values_per_thread),
          0,
          values_per_thread);
      if (remaining > 0) {
        U sum = load_vector_safe<T, U, values_per_thread, 4>(
            x, x_thread, remaining);
        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[row] += qdot_safe<U, values_per_thread, 4>(
              wl, x_thread, s, b, sum, remaining);
        }
      }
    }
    const uint active_tail_lanes = uint(tail_values / values_per_thread);
    if (tail_values % values_per_thread == 0 && simd_lid < active_tail_lanes) {
      U sum = load_vector<T, U, values_per_thread, 4>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        if (WVEC || PF) {
          const uint v = *((const device uint*)(ws + row * in_vec_size_w));
          result[row] += qdot_affine4_g64_word(v, x_thread, s, b, sum);
        } else {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          result[row] += qdot<U, values_per_thread, 4>(wl, x_thread, s, b, sum);
        }
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

// KERN-DOWN-TILE: y-tile coarsening for the K = 704 expert down gather
// (the only pair-geometry plane at that K; out_vec_size = 2816). The
// frozen host launches grid (1, N/8 = 352, 64), so every 64-thread group
// amortizes its serial run_offset scan, gather offset arithmetic and
// eight simd_sums over only ~3 K-blocks of stream (704 = 2 * 256 + 192)
// -- measured ~390 GB/s while the K = 2816 gate/up gathers move the same
// unique bytes at 479-589 GB/s. Here only every span-th y-group survives
// (the rest return before the scan); the survivor elects ONCE and then
// walks its span consecutive 8-row y-tiles serially through the verbatim
// pair impl -- or, for a pairless run position, the verbatim stock
// qmv_impl -- with tid.y rewritten to the tile index (a strip-walk
// pattern). Tile u is served by survivor (u / span) * span
// at loop step u % span and by no other group, so every output row keeps
// the IDENTICAL qdot sequence, accumulator, simd_sum and store the
// untiled arm produces for it: loads-only rescheduling, registers stay
// pair-sized. 352 divides by both spans, so no ragged tail. The pairless
// arm is tile-walked HERE because the stock fall-through derives out_row
// from tid.y inside qmv_impl -- follower tiles of a pairless assignment
// would otherwise never be written. Verified uint16-exact vs the
// per-assignment quantized_matmul oracle and vs the untiled arm at
// K = 704, N = 2816, 64 assignments over 128 experts, M = 8, spans 4 and
// 2, 3 seeds, NaN-filled outputs (parity-down-tile, 2026-08-28).
template <typename T, int group_size, int bits>
METAL_FUNC void gather_qmv_gemma4_down_tile(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const uint lhs_stride,
    const uint rhs_stride,
    const int64_t x_stride,
    const int64_t w_stride,
    const int64_t s_stride,
    const int64_t b_stride,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int gemma4_down_tile_span = 4; // sweep alternate: 2
  if (tid.y % uint(gemma4_down_tile_span) != 0u) {
    return;
  }
  const uint assignment = tid.z;
  const uint32_t route_word = rhs_indices[assignment * rhs_stride];
  const bool expert_prefix_bounds = (route_word & 0x80000000u) != 0u;
  const uint32_t expert =
      expert_prefix_bounds ? (route_word & 0xffu) : route_word;
  uint run_offset = 0;
  if (expert_prefix_bounds) {
    run_offset = (route_word >> 8) & 0x3fu;
  } else {
    for (uint prior = assignment; prior > 0; --prior) {
      if (rhs_indices[(prior - 1) * rhs_stride] != expert) {
        break;
      }
      run_offset++;
    }
  }
  // Odd positions are produced by the immediately preceding pair leader.
  if ((run_offset & 1) != 0) {
    return;
  }
  const device uint32_t* tile_w = w + expert * w_stride;
  const device T* tile_scales = scales + expert * s_stride;
  const device T* tile_biases = biases + expert * b_stride;
  const device T* tile_x0 =
      x + lhs_indices[assignment * lhs_stride] * x_stride;
  device T* tile_y0 = y + assignment * out_vec_size;
  const bool has_pair = expert_prefix_bounds
      ? (((route_word >> 14) & 0x3fu) + 1u) > 1u
      : assignment + 1 < 64 &&
          rhs_indices[(assignment + 1) * rhs_stride] == expert;
  if (has_pair) {
    const device T* tile_x1 =
        x + lhs_indices[(assignment + 1) * lhs_stride] * x_stride;
    device T* tile_y1 = y + (assignment + 1) * out_vec_size;
    for (int t = 0; t < gemma4_down_tile_span; t++) {
      uint3 tile_tid = tid;
      tile_tid.y = tid.y + uint(t);
      qmv_affine4_g64_pair_impl<T, group_size, bits>(
          tile_w,
          tile_scales,
          tile_biases,
          tile_x0,
          tile_x1,
          tile_y0,
          tile_y1,
          in_vec_size,
          tile_tid,
          simd_gid,
          simd_lid);
    }
    return;
  }
  for (int t = 0; t < gemma4_down_tile_span; t++) {
    uint3 tile_tid = tid;
    tile_tid.y = tid.y + uint(t);
    qmv_impl<T, group_size, bits>(
        tile_w,
        tile_scales,
        tile_biases,
        tile_x0,
        tile_y0,
        in_vec_size,
        out_vec_size,
        tile_tid,
        simd_gid,
        simd_lid);
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  const bool gemma4_pair_geometry =
      group_size == 64 && bits == 4 && M == 1 && batch_ndims == 1 &&
      batch_shape[0] == 64 && x_batch_ndims == 1 && w_batch_ndims == 1 &&
      ((in_vec_size == 2816 && out_vec_size == 704) ||
       (in_vec_size == 704 && out_vec_size == 2816));
  if (gemma4_pair_geometry) {
    // KERN-DOWN-TILE gate (strip-walk pattern): compile-time flip; ON
    // here -- the K = 704 down plane takes the y-tile-coarsened arm above.
    // Flip to false to return every plane to the incumbent per-y-group
    // election below; the two arms are bit-identical by construction.
    constexpr bool gemma4_down_tile = true;
    if (gemma4_down_tile && in_vec_size == 704) {
      gather_qmv_gemma4_down_tile<T, group_size, bits>(
          w,
          scales,
          biases,
          x,
          lhs_indices,
          rhs_indices,
          y,
          in_vec_size,
          out_vec_size,
          (uint)lhs_strides[0],
          (uint)rhs_strides[0],
          x_strides[0],
          w_strides[0],
          s_strides[0],
          b_strides[0],
          tid,
          simd_gid,
          simd_lid);
      return;
    }
    const uint assignment = tid.z;
    const uint32_t route_word =
        rhs_indices[assignment * (uint)rhs_strides[0]];
    const bool expert_prefix_bounds = (route_word & 0x80000000u) != 0u;
    const uint32_t expert =
        expert_prefix_bounds ? (route_word & 0xffu) : route_word;
    uint run_offset = 0;
    if (expert_prefix_bounds) {
      run_offset = (route_word >> 8) & 0x3fu;
    } else {
      for (uint prior = assignment; prior > 0; --prior) {
        if (rhs_indices[(prior - 1) * (uint)rhs_strides[0]] != expert) {
          break;
        }
        run_offset++;
      }
    }

    // RUN-QUAD: leaders sit at run_offset % 4 == 0 and serve up to four
    // same-expert assignments from ONE weight stream. Positions 1..3 of each
    // aligned quartet are produced by their leader, so a run of two keeps the
    // incumbent pair arithmetic, a run of three takes the triple impl, and a
    // run of four takes the quad-stream impl -- each (output, input) pair
    // keeps its own accumulator, K-loop order, and qdot, so every output
    // element's add sequence is identical to the incumbent per-arm kernels.
    if ((run_offset & 3) != 0) {
      return;
    }
    uint run_len = 1;
    if (expert_prefix_bounds) {
      run_len = min(4u, ((route_word >> 14) & 0x3fu) + 1u);
    } else {
      while (run_len < 4 && assignment + run_len < 64 &&
             rhs_indices[(assignment + run_len) * (uint)rhs_strides[0]] ==
                 expert) {
        run_len++;
      }
    }
    if (run_len > 1) {
      const device uint32_t* run_w = w + expert * w_strides[0];
      const device T* run_scales = scales + expert * s_strides[0];
      const device T* run_biases = biases + expert * b_strides[0];
      const device T* run_x0 =
          x + lhs_indices[assignment * (uint)lhs_strides[0]] * x_strides[0];
      const device T* run_x1 = x +
          lhs_indices[(assignment + 1) * (uint)lhs_strides[0]] * x_strides[0];
      device T* run_y0 = y + assignment * out_vec_size;
      device T* run_y1 = y + (assignment + 1) * out_vec_size;
      if (run_len == 2) {
        qmv_affine4_g64_pair_impl<T, group_size, bits>(
            run_w,
            run_scales,
            run_biases,
            run_x0,
            run_x1,
            run_y0,
            run_y1,
            in_vec_size,
            tid,
            simd_gid,
            simd_lid);
        return;
      }
      const device T* run_x2 = x +
          lhs_indices[(assignment + 2) * (uint)lhs_strides[0]] * x_strides[0];
      device T* run_y2 = y + (assignment + 2) * out_vec_size;
      if (run_len == 3) {
        qmv_affine4_g64_triple_stream_impl<T, group_size, bits>(
            run_w,
            run_scales,
            run_biases,
            run_x0,
            run_x1,
            run_x2,
            run_y0,
            run_y1,
            run_y2,
            in_vec_size,
            tid,
            simd_gid,
            simd_lid);
        return;
      }
      const device T* run_x3 = x +
          lhs_indices[(assignment + 3) * (uint)lhs_strides[0]] * x_strides[0];
      device T* run_y3 = y + (assignment + 3) * out_vec_size;
      qmv_affine4_g64_quad_stream_impl<T, group_size, bits>(
          run_w,
          run_scales,
          run_biases,
          run_x0,
          run_x1,
          run_x2,
          run_x3,
          run_y0,
          run_y1,
          run_y2,
          run_y3,
          in_vec_size,
          tid,
          simd_gid,
          simd_lid);
      return;
    }

    // Singleton experts and odd-run tails still need one ordinary QMV, but
    // their one-dimensional offsets are already resolved by this guard.
    const uint32_t single_lhs =
        lhs_indices[assignment * (uint)lhs_strides[0]];
    const device T* single_x = x + single_lhs * x_strides[0];
    const device uint32_t* single_w = w + expert * w_strides[0];
    const device T* single_scales = scales + expert * s_strides[0];
    const device T* single_biases = biases + expert * b_strides[0];
    device T* single_y = y + assignment * (uint)out_vec_size;
    if (in_vec_size == 2816) {
      qmv_affine4_g64_singles_impl<
          T, group_size, bits, 2816, true, false>(
          single_w, single_scales, single_biases, single_x, single_y,
          in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
    } else {
      qmv_impl<T, group_size, bits>(
          single_w, single_scales, single_biases, single_x, single_y,
          in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
    }
    return;
  }
  uint32_t x_idx;
  uint32_t route_word;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    route_word = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    route_word = rhs_indices[idx.y];
  }
  if ((route_word & 0x80000000u) != 0u) {
    const uint32_t expert = route_word & 0xffu;
    if (x_batch_ndims == 1) {
      x += x_idx * x_strides[0];
    } else {
      x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
    }
    if (w_batch_ndims == 1) {
      w += expert * w_strides[0];
      scales += expert * s_strides[0];
      biases += expert * b_strides[0];
    } else {
      ulong3 idx = elem_to_loc_broadcast(
          expert, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
      w += idx.x;
      scales += idx.y;
      biases += idx.z;
    }
    y += tid.z * (out_vec_size * M);
  } else {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        y,
        out_vec_size * M,
        batch_ndims,
        batch_shape,
        lhs_strides,
        rhs_strides,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      in_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_gather_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
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
[[kernel]] void affine_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

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
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    // Matrices are all aligned check nothing
    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      // Tile aligned so check outside of the hot loop
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }

        // Store results to device memory
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    device T* scales [[buffer(2)]],
    device T* biases [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr float eps = 1e-7;
  constexpr int simd_size = 32;
  constexpr float n_bins = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = pack_factor / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / pack_factor;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;

  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t in_index = offset * values_per_reduce;
  size_t out_index = power_of_2_bits
      ? offset * writes_per_pack
      : offset * bytes_per_pack / writes_per_reduce;

  float w_thread[values_per_reduce];
  float w_min = Limits<T>::max;
  float w_max = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    float val = w[in_index + i];
    w_thread[i] = val;
    w_min = min(w_min, val);
    w_max = max(w_max, val);
  }

  w_min = simd_min(w_min);
  w_max = simd_max(w_max);

  float scale = max((w_max - w_min) / n_bins, eps);
  bool side = abs(w_min) > abs(w_max);
  scale = side ? scale : -scale;
  float edge = side ? w_min : w_max;
  float q0 = round(edge / scale);
  bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  float bias = at_zero ? 0 : edge;

  // Write out the scales and biases
  size_t gindex = in_index / group_size;
  if (in_index % group_size == 0) {
    scales[gindex] = static_cast<T>(scale);
    biases[gindex] = static_cast<T>(bias);
  }

  using OutType = metal::conditional_t<bits == 5, uint64_t, uint32_t>;
  OutType output = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
    if (bits == 8) {
      output = val;
    } else {
      output |= val << (bits * (i % pack_factor));
    }

    if (pack_factor < values_per_reduce && i % pack_factor == pack_factor - 1) {
      out[out_index + i / pack_factor] = output;
      output = 0;
    } else {
#pragma clang loop unroll(full)
      for (int j = 1; j < writes_per_reduce; j++) {
        uint8_t sval = simd_shuffle_down(val, j);
        output |= static_cast<OutType>(sval)
            << (bits * (j * values_per_reduce + i));
      }
    }
  }
  if (bits == 3 || bits == 6) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
    }
  } else if (bits == 5) {
    if (in_index % pack_factor == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
      out[out_index + 3] = (output & 0xff000000) >> 24;
      out[out_index + 4] = (output & 0xff00000000) >> 32;
    }
  } else {
    if (writes_per_reduce > 0 && out_index % writes_per_reduce == 0) {
      out[out_index / writes_per_reduce] = output;
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * pack_factor;
  size_t gindex = oindex / group_size;
  T scale = scales[gindex];
  T bias = biases[gindex];

  out += oindex;

  if (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x7) * scale + bias;
    out[1] = ((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
    out[3] = ((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = ((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
    out[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
  } else if (bits == 5) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x1f) * scale + bias;
    out[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
    out[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
    out[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
    out[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
    out[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
    out[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
    out[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
  } else if (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    uint val = w[offset];
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; i++) {
      uint8_t d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * d + bias;
    }
  }
}
