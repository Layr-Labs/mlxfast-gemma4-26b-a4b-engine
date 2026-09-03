// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

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

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct QuantizedBlockLoader<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    32,
    bits> {
  MLX_MTL_CONST short group_size = 32;

  static_assert(
      BCOLS % group_size == 0,
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
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

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
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        biases(biases_ + bi * src_ld / group_size + group_id) {}

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
      // if (group_steps > 1) {
      //   group_step_cnt++;
      //   if (group_step_cnt == group_steps) {
      //     group_step_cnt = 0;
      //     scales++;
      //     biases++;
      //   }
      // } else {
      scales += n_groups;
      biases += n_groups;
      // }
    } else {
      scales += n_groups * group_stride;
      biases += n_groups * group_stride;
    }
  }
};

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

// DARKBLOOM GEMMA4 NAX QMM-T ROW-STRIP TILING.
// qmm_t_nax_tgp_impl covers a BM x BN output tile with WM x WN simdgroups.
// The launch shape is fixed by the host (32, WN, WM) and the host is not
// editable, so the threadgroup is always 4 simdgroups over a 64 x 64 tile.
// Stock splits that tile 2 x 2, so each simdgroup owns 32 rows x 32 cols and
// the two simdgroups that share a row band each fetch the SAME 32 rows of the
// activation operand from device memory: A is read twice per threadgroup per
// K step. This constant instead lays the same 4 simdgroups out as 4 row
// strips of 16 rows x 64 cols. The strips are disjoint in M, so every A
// fragment is fetched exactly once, and the B operand -- which already lives
// in threadgroup memory as Ws -- is read wider instead.
//
// Nothing about the K loop moves. BK, SK and TK are untouched, the k and kk1
// loops keep their bounds and their order, and every output element still
// accumulates over exactly the same k values in exactly the same sequence.
// Only which simdgroup owns an element, and how the owner's fragments are
// shaped, change. The MMA op count per threadgroup is invariant as well:
// stock issues WM*WN * (TM * TN/2 * TK) = 4 * (2 * 1 * 2) = 16 ops per kk1
// step, the strip layout issues 4 * (1 * 2 * 2) = 16. Both shapes enter the
// same TN-even branch of tile_matmad_nax, so the per-element fragment
// accumulation chain is instruction-for-instruction the same.
//
// The kernel's template parameters, and therefore every kernel-name string
// the host builds, are untouched: BM, BN, BK, WM and WN all keep their
// values and only the interior mapping is re-derived from them.
//
// Kill switch: build with -DDARKBLOOM_GEMMA4_NAX_TILING=0 and SGM/SGN fold
// back to WM/WN, which reproduces the shipped expressions byte for byte.
#ifndef DARKBLOOM_GEMMA4_NAX_TILING
#define DARKBLOOM_GEMMA4_NAX_TILING 1
#endif

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
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

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

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

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  // Simdgroup grid over the BM x BN tile. Stock is WM x WN; the row-strip
  // layout stacks the same WM*WN simdgroups in the row direction only, so
  // no two of them share a row band. See the note on the enable above.
  constexpr int SGM = (DARKBLOOM_GEMMA4_NAX_TILING != 0) ? (WM * WN) : WM;
  constexpr int SGN = (DARKBLOOM_GEMMA4_NAX_TILING != 0) ? 1 : WN;
  static_assert(SGM * SGN == WM * WN, "simdgroup count must be preserved");
  static_assert(BM % (SGM * 16) == 0, "row strip must be a fragment multiple");
  static_assert(BN % (SGN * 16) == 0, "col strip must be a fragment multiple");

  constexpr short SM = BM / SGM;
  constexpr short SN = BN / SGN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  // tile_matmad_nax has no branch for an odd TN greater than one; it would
  // silently emit no MMA at all. Refuse to compile such a layout.
  static_assert(TN == 1 || TN % 2 == 0, "TN must be 1 or even for NAX MMA");

  const short tm = SM * (simd_gid / SGN);
  const short tn = SN * (simd_gid % SGN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_n_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;
  (void)M;

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
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
  // const short num_els = min(BM, M - y_row);
  // const short num_outs = min(BN, N - y_col);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TK, TN> Btile;

      volatile int compiler_barrier;

      Atile.load(x + kk1, K);
      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  Dtile.store(y + tm * N + tn, N);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 32,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_qmm_t_nax(
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
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_qmm_n_nax(
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

  constexpr int BN_padded = (BN + 16 / sizeof(T));

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

  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_t_nax(
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
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_n_nax(
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

  constexpr int BN_padded = (BN + 16 / sizeof(T));

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
  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

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

///////////////////////////////////////////////////////////////////////////////
// DARKBLOOM GEMMA4 NAX-EXACT (NAXEXACT-001): the exact-codes decode form of
// affine_gather_qmm_rhs_nax.
//
// The stock body dequantizes every weight tile to T through
// QuantizedBlockLoader (`scale * code + bias`, rounded to T) before the
// tensor op. The B=8 decode projections reach this kernel through the
// pseudo-expert gather road (the activation rows laid out once per view,
// M <= 16 rows, one row tile). For that shape the body below runs instead:
// the integer codes enter the tensor op as exactly representable T operands
// (0 .. 2^bits - 1), the products of one K group accumulate in fp32, and the
// group's scale and bias are applied in fp32 afterwards,
//
//     y[m][n] = sum_g ( s[n][g] * sum_{k in g} code[n][k] * x[m][k]
//                     + b[n][g] * rs[m][g] ),
//
// with rs the activation run sum in the form every quantized kernel of this
// tree (and upstream's load_vector) uses: for 2- and 4-bit, aligned 4-chunks
// of x summed in T with the chunk values accumulated in fp32; for 8-bit, the
// elements added to the fp32 sum one by one. This is the arithmetic of the tree's
// custom decode projections (integer codes, fp32 fold per group). No weight
// value is rounded to T anywhere.
//
// Structure: the K groups are split across the WM*WN simdgroups; each
// simdgroup streams its own weight words straight from device memory into
// the B fragments (one tile prefetched), so the K loop has no threadgroup
// staging and no barrier; the group's scales and biases travel by simd
// shuffle; the fp32 partial tiles of the simdgroups are summed through Ws
// once per expert segment and stored. Every other dispatch (M > 16, N or K
// not tile aligned, an instantiation outside {transpose, bits 2/4/8,
// group_size a multiple of BK}) runs the stock body unchanged; the prompt
// pass's expert projections have M >= 512 rows and never enter.
//
// Kill switch: -DDARKBLOOM_GEMMA4_NAX_EXACT_CODES=0 compiles the stock body
// alone. The Swift hosts' DARKBLOOM_GEMMA4_NAX_*=0 switches keep the road
// from being dispatched in the first place.
///////////////////////////////////////////////////////////////////////////////
#ifndef DARKBLOOM_GEMMA4_NAX_EXACT_CODES
#define DARKBLOOM_GEMMA4_NAX_EXACT_CODES 1
#endif

// The machine word holding four consecutive codes of one weight row.
template <int bits>
struct gather_rhs_exact_word {};
template <>
struct gather_rhs_exact_word<8> {
  using type = uint32_t;
};
template <>
struct gather_rhs_exact_word<4> {
  using type = uint16_t;
};
template <>
struct gather_rhs_exact_word<2> {
  using type = uint8_t;
};

// One tile's weight words for this lane: word ((step * TK + kk) * TN + nn)
// * 2 + i holds the codes k = step * SK + kk * 16 + fn .. + 3 of tile row
// nn * 16 + fm + 8 * i, which is exactly the B fragment element layout of
// BaseNAXFrag (rows fm, fm + 8; columns fn .. fn + 3).
template <
    typename word_t,
    int pack_factor,
    int BK,
    int SK,
    int TK,
    int TN,
    int kSteps>
METAL_FUNC void gather_rhs_exact_load_words(
    thread word_t* dst,
    const device uint8_t* wseg,
    const int K_w,
    const int tile,
    const short fm,
    const short fn) {
  STEEL_PRAGMA_UNROLL
  for (short step = 0; step < kSteps; ++step) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; ++kk) {
      const int kbyte = (tile * BK + step * SK + kk * 16 + fn) / pack_factor;
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
          const int row = nn * 16 + fm + i * 8;
          dst[((step * TK + kk) * TN + nn) * 2 + i] =
              *reinterpret_cast<const device word_t*>(
                  wseg + row * K_w + kbyte);
        }
      }
    }
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
    int WN>
METAL_FUNC void affine_gather_qmm_rhs_nax_exact_decode(
    const device T* x,
    const device uint8_t* wl,
    const device T* scales,
    const device T* biases,
    const device uint32_t* indices,
    device T* y,
    const int M,
    const int N,
    const int K,
    threadgroup T* Ws,
    const uint simd_group_id,
    const uint simd_lane_id) {
  constexpr int kSimdgroups = WM * WN;
  static_assert(kSimdgroups == 4, "the partial-tile reduction sums 4 simdgroups");
  constexpr int pack_factor = 8 / bits;
  constexpr int kTilesPerGroup = group_size / BK;
  constexpr short kFrag = 16;
  constexpr short SK = 32;
  constexpr short TK = SK / kFrag;
  constexpr short TN = BN / kFrag;
  constexpr short kSteps = BK / SK;
  constexpr short kWords = kSteps * TK * TN * 2;
  constexpr int kRedStride = kFrag * 4 + 4;
  constexpr int kRedTile = kFrag * kRedStride;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  static_assert(
      2 * kRedTile * sizeof(float) <= BN * BK_padded * sizeof(T),
      "the reduction scratch must fit inside Ws");
  using word_t = typename gather_rhs_exact_word<bits>::type;
  constexpr uint32_t kCodeMask = (1u << bits) - 1u;
  using acc_tile_t = NAXTile<float, 1, TN>;
  using a_tile_t = NAXTile<T, 1, TK>;
  using b_frag_t = typename BaseNAXFrag::dtype_frag_t<T>;
  using acc_frag_t = typename BaseNAXFrag::dtype_frag_t<float>;

  const int K_w = K / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * size_t(K_w);
  const size_t stride_s = size_t(N) * size_t(K_g);

  // Lane coordinates inside a 16 x 16 fragment (BaseNAXFrag::get_coord).
  const short qid = short(simd_lane_id >> 2);
  const short fm = short((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fn = short(((qid & 2) | (simd_lane_id & 1)) * 4);
  // This simdgroup's K groups (whole groups, contiguous), as tiles.
  const int G = K_g;
  const int t_lo = ((G * int(simd_group_id)) / kSimdgroups) * kTilesPerGroup;
  const int t_hi =
      ((G * (int(simd_group_id) + 1)) / kSimdgroups) * kTilesPerGroup;
  // Lane l carries the scale and bias of tile columns 2l and 2l + 1.
  const int sc0 = 2 * int(simd_lane_id);

  threadgroup float* red = (threadgroup float*)Ws;

  uint32_t index;
  short offset;
  uint32_t index_next = indices[0];
  short offset_next = 0;
  int n = 0;
  while (n < M) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = short(M);
    for (; n < M; n++) {
      if (indices[n] != index) {
        offset_next = short(n);
        index_next = indices[n];
        break;
      }
    }

    const device uint8_t* wseg = wl + index * stride_w;
    const device T* sseg = scales + index * stride_s;
    const device T* bseg = biases + index * stride_s;

    acc_tile_t Acc;
    Acc.clear();
    acc_tile_t D;
    D.clear();
    float rsp0 = 0.0f;
    float rsp1 = 0.0f;

    word_t wcur[kWords];
    word_t wnxt[kWords];
    T s_cur0 = T(0);
    T s_cur1 = T(0);
    T b_cur0 = T(0);
    T b_cur1 = T(0);
    T s_nxt0 = T(0);
    T s_nxt1 = T(0);
    T b_nxt0 = T(0);
    T b_nxt1 = T(0);

    if (t_lo < t_hi) {
      gather_rhs_exact_load_words<word_t, pack_factor, BK, SK, TK, TN, kSteps>(
          wcur, wseg, K_w, t_lo, fm, fn);
      const int g0 = t_lo / kTilesPerGroup;
      s_cur0 = sseg[sc0 * K_g + g0];
      s_cur1 = sseg[(sc0 + 1) * K_g + g0];
      b_cur0 = bseg[sc0 * K_g + g0];
      b_cur1 = bseg[(sc0 + 1) * K_g + g0];
    }

    for (int t = t_lo; t < t_hi; ++t) {
      const bool group_end = ((t + 1) % kTilesPerGroup) == 0;
      const bool has_next = (t + 1) < t_hi;
      if (has_next) {
        gather_rhs_exact_load_words<word_t, pack_factor, BK, SK, TK, TN, kSteps>(
            wnxt, wseg, K_w, t + 1, fm, fn);
        if (group_end) {
          const int g1 = (t + 1) / kTilesPerGroup;
          s_nxt0 = sseg[sc0 * K_g + g1];
          s_nxt1 = sseg[(sc0 + 1) * K_g + g1];
          b_nxt0 = bseg[sc0 * K_g + g1];
          b_nxt1 = bseg[(sc0 + 1) * K_g + g1];
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short step = 0; step < kSteps; ++step) {
        a_tile_t A;
        A.load_rows(x + t * BK + step * SK, K, short(M));
        // Activation run sums in the incumbents' form (upstream load_vector,
        // the decode kernels' mma8_runsum4 / mma8_runsum8): for 2- and 4-bit
        // each aligned 4-chunk of x is summed in T (`xt[0] + xt[1] + xt[2] +
        // xt[3]`, T-typed additions) and the chunk values accumulate in
        // fp32; for 8-bit every element is added to the fp32 sum directly.
        // A lane's four fragment columns are one aligned chunk
        // (k = 16 kk + fn .. + 3, fn a multiple of 4).
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          thread const b_frag_t& a = A.frag_at(0, kk);
          thread T xt[8];
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) {
            xt[e] = a[e];
          }
          if constexpr (bits == 8) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 4; ++e) {
              rsp0 += xt[e];
              rsp1 += xt[4 + e];
            }
          } else {
            rsp0 += xt[0] + xt[1] + xt[2] + xt[3];
            rsp1 += xt[4] + xt[5] + xt[6] + xt[7];
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < TN; nn += 2) {
            b_frag_t B0;
            b_frag_t B1;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
              const uint32_t w0 =
                  uint32_t(wcur[((step * TK + kk) * TN + nn) * 2 + i]);
              const uint32_t w1 =
                  uint32_t(wcur[((step * TK + kk) * TN + nn + 1) * 2 + i]);
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 4; ++j) {
                B0[i * 4 + j] =
                    static_cast<T>(float((w0 >> (bits * j)) & kCodeMask));
                B1[i * 4 + j] =
                    static_cast<T>(float((w1 >> (bits * j)) & kCodeMask));
              }
            }
            BaseNAXFrag::mma(
                D.frag_at(0, nn),
                D.frag_at(0, nn + 1),
                A.frag_at(0, kk),
                metal::bool_constant<false>{},
                B0,
                B1,
                metal::bool_constant<true>{});
          }
        }
      }

      if (group_end) {
        float rs0 = rsp0;
        float rs1 = rsp1;
        rs0 += simd_shuffle_xor(rs0, ushort(1));
        rs0 += simd_shuffle_xor(rs0, ushort(8));
        rs1 += simd_shuffle_xor(rs1, ushort(1));
        rs1 += simd_shuffle_xor(rs1, ushort(8));
        const float2 sl = float2(float(s_cur0), float(s_cur1));
        const float2 bl = float2(float(b_cur0), float(b_cur1));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < TN; ++nn) {
          const ushort src = ushort((nn * kFrag + fn) >> 1);
          const float2 sA = simd_shuffle(sl, src);
          const float2 sB = simd_shuffle(sl, ushort(src + 1));
          const float2 bA = simd_shuffle(bl, src);
          const float2 bB = simd_shuffle(bl, ushort(src + 1));
          const float4 s4 = float4(sA, sB);
          const float4 b4 = float4(bA, bB);
          thread acc_frag_t& acc = Acc.frag_at(0, nn);
          thread const acc_frag_t& d = D.frag_at(0, nn);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; ++j) {
            acc[j] += s4[j] * d[j] + rs0 * b4[j];
            acc[4 + j] += s4[j] * d[4 + j] + rs1 * b4[j];
          }
        }
        D.clear();
        rsp0 = 0.0f;
        rsp1 = 0.0f;
        s_cur0 = s_nxt0;
        s_cur1 = s_nxt1;
        b_cur0 = b_nxt0;
        b_cur1 = b_nxt1;
      }
      if (has_next) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kWords; ++i) {
          wcur[i] = wnxt[i];
        }
      }
    }

    // Sum the simdgroups' fp32 partial tiles through Ws (two scratch tiles,
    // pairwise), then simdgroup 0 stores the segment's rows.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id >= 2) {
      Acc.template store<float, kRedStride, 1>(
          red + (simd_group_id - 2) * kRedTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id < 2) {
      acc_tile_t P;
      P.template load<float, kRedStride, 1>(red + simd_group_id * kRedTile);
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TN; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          Acc.frag_at(0, f)[e] += P.frag_at(0, f)[e];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 1) {
      Acc.template store<float, kRedStride, 1>(red);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc_tile_t P;
      P.template load<float, kRedStride, 1>(red);
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TN; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          Acc.frag_at(0, f)[e] += P.frag_at(0, f)[e];
        }
      }
      Acc.store_slice(y, N, short2(0, offset), short2(BN, offset_next));
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
[[kernel]] void affine_gather_qmm_rhs_nax(
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

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

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
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  // NAXEXACT-001: the decode shape (one row tile of at most 16 activation
  // rows, tile-aligned N and K) takes the exact-codes body; everything else
  // continues into the stock body below unchanged.
  constexpr bool kExactCodesEligible = (DARKBLOOM_GEMMA4_NAX_EXACT_CODES != 0) &&
      transpose && (bits == 2 || bits == 4 || bits == 8) &&
      (group_size % BK == 0) && (BM == 64) && (BN == 64) && (BK == 64) &&
      (WM * WN == 4);
  if constexpr (kExactCodesEligible) {
    if (M <= 16 && (N % BN) == 0 && (K % BK) == 0) {
      affine_gather_qmm_rhs_nax_exact_decode<T, group_size, bits, BM, BN, BK, WM, WN>(
          x,
          wl,
          scales,
          biases,
          indices + y_row,
          y,
          M,
          N,
          K,
          Ws,
          simd_group_id,
          simd_lane_id);
      return;
    }
  }

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
    threadgroup_barrier(mem_flags::mem_none);

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

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

              volatile int compiler_barrier;

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
                  gather_rhs_load_frag_row(mm, Atile, xn + kk1, K);
                  gather_rhs_mma_frag_row(
                      mm,
                      Dtile,
                      Atile,
                      Btile,
                      metal::bool_constant<transpose>{});
                }
              }

              (void)compiler_barrier;
            }
          } else if (!seg_empty) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
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

              (void)compiler_barrier;
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

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (BK - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

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

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

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
