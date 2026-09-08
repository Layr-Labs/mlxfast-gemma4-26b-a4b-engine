import Foundation
import MLX

/// B8 affine-4/group-64 expert gate/up with explicit BF16 closes before GeGLU.
public enum Gemma4DecodeFusedGUV1 {
    static let enabled = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_DECODE_FUSED_GEGLU"] != "0"
    static let outputShape = [64, 1, 704]
    static let outputDType: DType = .bfloat16

    /// RUN-CAP SWEEP. The pair/triple/quad impls all inline into one kernel, so
    /// register allocation is worst-case across every path -- proven by RUN-OCT,
    /// where merely compiling an eight-stream path cost 9.1% even when unused.
    /// Runs of three or more are only 14% of runs at real top-8-of-128 routing,
    /// so the rarely-taken wide paths may be taxing the 86% that never run them.
    /// This compiles OUT every impl above the cap.
    /// DEFAULT 2. Measured single-worker under realistic top-8-of-128 routing:
    ///   cap4 (incumbent) 0.002927 s/token, cap2 0.002660 = **+9.12%**, zero
    ///   overlap across five alternating passes, tokens bit-identical.
    /// Runs of three or more are only 14% of runs, but the triple and quad
    /// impls inline into the same kernel, so their registers were charged to
    /// the 86% of threadgroups that never execute them.
    /// `DARKBLOOM_GEMMA4_GU_RUN_CAP=4` restores the incumbent.
    static let runCap: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_GU_RUN_CAP"] ?? "4"
        return Int(raw).map { min(max($0, 1), 4) } ?? 2
    }()

    static func call(x: MLXArray, storage: SwitchGateUpFusedStorage,
        lhs: MLXArray, rhs: MLXArray, taggedRoute: Bool = false) -> MLXArray {
        call([storage.weight, storage.scales, storage.biases, x, lhs, rhs],
            taggedRoute: taggedRoute)
    }

    /// Raw launch for callers that already passed the fused-GU contract.
    static func call(_ inputs: [MLXArray], taggedRoute: Bool = false) -> MLXArray {
        if packetSumsEnabled, runCap == 4, inputs.count == 6,
            inputs[3].dtype == .bfloat16,
            (inputs[3].shape == [8, 2816] || inputs[3].shape == [8, 1, 2816]),
            inputs[4].dtype == .uint32, inputs[4].shape == [64],
            inputs[5].dtype == .uint32, inputs[5].shape == [64]
        {
            let sums = packetSumKernel([inputs[3]],
                grid: (2816, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[8, 352]], outputDTypes: [.float32])[0]
            return (taggedRoute ? kernelTaggedSums : kernelGeneralSums)(inputs + [sums],
                grid: (32, 176 * 2, 64), threadGroup: (32, 2, 1),
                outputShapes: [outputShape], outputDTypes: [outputDType])[0]
        }
        return (taggedRoute ? kernelTagged : kernelGeneral)(inputs,
            grid: (32, 176 * 2, 64), threadGroup: (32, 2, 1),
            outputShapes: [outputShape], outputDTypes: [outputDType])[0]
    }

    /// GU-TAGGED-ROUTE. When the route producer emits prefix-bounds tagged
    /// words, `expert_run`'s untagged fallback -- a backward, data-dependent
    /// scan over `rhs` -- can never execute, but it still inlines into the
    /// body and its registers and its unhoistable loop are charged to every
    /// threadgroup. Same inline-path pruning that took the run cap from four
    /// to two. Chosen per call from the producer's own `hasExpertPrefixBounds`
    /// contract, so a fallback emitting raw keys keeps the general body; the
    /// two carry distinct kernel names so their pipeline-cache entries never
    /// alias. Only an already-unreachable branch is removed, so the output is
    /// bit-identical.
    private static func makeKernel(tagged: Bool, inputSums: Bool = false) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
        name: "gemma4_b8_decode_gateup_geglu_threadgroup_v3_packetsum1"
            + (tagged ? "_tagged_v1" : "") + (inputSums ? "_sum" : "_direct"),
        inputNames: ["w", "scales", "biases", "x", "lhs", "rhs"]
            + (inputSums ? ["inputSums"] : []),
        outputNames: ["y"],
        source: #"""
uint3 tid=threadgroup_position_in_grid;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;

    const uint linear=tid.y+tid.z*(176/guPairs);tid.y=linear/64;tid.z=linear%64;
    const uint assignment=tid.z;const ExpertRun run=expert_run(rhs,assignment);
    if(!run.leader)return;
    const uint localSg=sg&1u, localColumn=(sg/2)*4;
    const uint column=tid.y*(4*guPairs)+localColumn;
    const uint packedRow=(column/16)*32+column%16+(localSg==1 ? 16:0)-localSg*4;
    const uint expertBase=run.expert*1408;
    threadgroup bfloat tile[4*8*guPairs];
    threadgroup bfloat* scratch=tile+(localSg==1 ? 4*guPairs:0)+localColumn-localSg*4;
    uint3 mathTid=tid;mathTid.y=0;
    tg_execute_projection<bfloat>(w+(expertBase+packedRow)*352,scales+(expertBase+packedRow)*44,biases+(expertBase+packedRow)*44,
        x,lhs,scratch,8*guPairs,guSliceN,assignment,run.count,mathTid,localSg,lane GU_KERNEL_SUM_ARGS);
    // BF16 closes remain explicit; only the scratch address space changes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((sg&1u)==0 && lane<run.count*4){
        const uint localRow=lane/4, hidden=column+lane%4;
        const uint row=assignment+localRow, localHidden=localColumn+lane%4;
        const bfloat g=tile[localRow*8*guPairs+localHidden];
        const bfloat u=tile[localRow*8*guPairs+4*guPairs+localHidden];
        if(false){y[row*1408+hidden]=g;y[row*1408+704+hidden]=u;}
        y[(false ? 64*1408:0)+row*704+hidden]=gemma4_geglu_compiled_tape(g,u);
    }

"""#,
        header: "#define GU_RUN_CAP \(runCap)\n"
            + "#define GU_TAGGED_ROUTE \(tagged ? 1 : 0)\n"
            + "#define GU_INPUT_SUMS \(inputSums ? 1 : 0)\n"
            + kernelHeader,
        ensureRowContiguous: true)
    }

    private static let packetSumsEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_GU_PACKET_SUMS"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernelHeader = #"""
#if GU_INPUT_SUMS
#define GU_SUM_PARAMS , const device float* inputSums, const device T* inputBase
#define GU_DISPATCH_SUM_PARAMS , const device float* inputSums
#define GU_KERNEL_SUM_ARGS , inputSums
#define GU_HELPER_SUM_ARGS , inputSums, x
#define GU_VECTOR_LOAD(X, V) load_vector_sum_only<T, float, values_per_thread, 4>(X, V, inputSums, inputBase)
#else
#define GU_SUM_PARAMS
#define GU_DISPATCH_SUM_PARAMS
#define GU_KERNEL_SUM_ARGS
#define GU_HELPER_SUM_ARGS
#define GU_VECTOR_LOAD(X, V) load_vector<T, float, values_per_thread, 4>(X, V)
#endif

// Copyright © 2023-2024 Apple Inc. Canonical helpers from 093e716.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
static constant constexpr const int SIMD_SIZE=32;
static constant constexpr const int QUAD_SIZE=4;
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

template <typename U, int values_per_thread>
inline void qdot_affine4_pair_word(
    uint packedWord,
    const thread U* x0,
    const thread U* x1,
    U scale,
    U bias,
    U sum0,
    U sum1,
    thread U& out0,
    thread U& out1) {
  static_assert(values_per_thread == 8, "Word load expects eight 4-bit values");
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

template <typename T, int group_size, int bits>
METAL_FUNC void tg_qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    threadgroup T* y,
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


template<typename T,typename U,int values_per_thread,int bits>
METAL_FUNC U load_vector_sum_only(const device T* x,thread U* v,
    const device float* inputSums,const device T* inputBase) {
  static_assert(values_per_thread == 8 && bits == 4, "GU sum-only packet");
  for (int i=0; i<8; i+=4) {
    v[i]=x[i]; v[i+1]=x[i+1]/16.0f;
    v[i+2]=x[i+2]/256.0f; v[i+3]=x[i+3]/4096.0f;
  }
  const size_t packet=size_t(x-inputBase)/8u;
  return static_cast<U>(inputSums[packet]);
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void tg_qmv_affine4_g64_pair_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    threadgroup T* y0,
    threadgroup T* y1,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] GU_SUM_PARAMS) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread float x1_thread[values_per_thread];
  thread uint packed[results_per_simdgroup];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
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
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 = GU_VECTOR_LOAD(x0, x0_thread);
    float sum1 = GU_VECTOR_LOAD(x1, x1_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      float dot0;
      float dot1;
      qdot_affine4_pair_word<float, values_per_thread>(
          packed[row], x0_thread, x1_thread, scale_local[row], bias_local[row], sum0, sum1, dot0, dot1);
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
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 =
        GU_VECTOR_LOAD(x0, x0_thread);
    float sum1 =
        GU_VECTOR_LOAD(x1, x1_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      float dot0;
      float dot1;
      qdot_affine4_pair_word<float, values_per_thread>(
          packed[row], x0_thread, x1_thread, scale_local[row], bias_local[row], sum0, sum1, dot0, dot1);
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

template <typename T, const int group_size, const int bits>
METAL_FUNC void tg_qmv_affine4_g64_solo_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    threadgroup T* y0,
    const constant int& in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] GU_SUM_PARAMS) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_thread = 4;
  constexpr int scale_step_per_thread = 8;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x0_thread[values_per_thread];
  thread uint packed[results_per_simdgroup];
  thread float scale_local[results_per_simdgroup];
  thread float bias_local[results_per_simdgroup];
  thread float result0[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * bytes_per_thread;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x0 += simd_lid * values_per_thread;
  y0 += out_row;

  int k = 0;
  for (; k <= in_vec_size - block_size; k += block_size) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 = GU_VECTOR_LOAD(x0, x0_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x0_thread, scale_local[row], bias_local[row], sum0);
    }

    ws += block_size / 2;
    scales += block_size / 64;
    biases += block_size / 64;
    x0 += block_size;
  }

  // Same whole-packet tail contract as the pair path: the only caller enters
  // with K=guK=2816, a whole number of 256-value blocks, so the final block
  // holds complete eight-value lane packets and no lane takes this branch.
  const uint active_tail_lanes =
      uint((in_vec_size - k) / values_per_thread);
  if (simd_lid < active_tail_lanes) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      packed[row] = *((const device uint*)(ws + row * in_vec_size_w));
      scale_local[row] = scales[row * in_vec_size_g];
      bias_local[row] = biases[row * in_vec_size_g];
    }

    float sum0 =
        GU_VECTOR_LOAD(x0, x0_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x0_thread, scale_local[row], bias_local[row], sum0);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result0[row] = simd_sum(result0[row]);
    if (simd_lid == 0) {
      y0[row] = static_cast<T>(result0[row]);
    }
  }
}

#if GU_RUN_CAP >= 3
template <typename T, const int group_size, const int bits>
METAL_FUNC void tg_qmv_affine4_g64_triple_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    threadgroup T* y0,
    threadgroup T* y1,
    threadgroup T* y2,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] GU_SUM_PARAMS) {
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

    float sum = GU_VECTOR_LOAD(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = GU_VECTOR_LOAD(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = GU_VECTOR_LOAD(x2, x_thread);
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

#endif

#if GU_RUN_CAP >= 4
template <typename T, const int group_size, const int bits>
METAL_FUNC void tg_qmv_affine4_g64_quad_stream_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x0,
    const device T* x1,
    const device T* x2,
    const device T* x3,
    threadgroup T* y0,
    threadgroup T* y1,
    threadgroup T* y2,
    threadgroup T* y3,
    const int in_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]] GU_SUM_PARAMS) {
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

    float sum = GU_VECTOR_LOAD(x0, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result0[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = GU_VECTOR_LOAD(x1, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result1[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = GU_VECTOR_LOAD(x2, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      result2[row] += qdot_affine4_registered_word<float, values_per_thread>(
          packed[row], x_thread, scale_local[row], bias_local[row], sum);
    }
    sum = GU_VECTOR_LOAD(x3, x_thread);
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

#endif

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

#ifndef GU_PAIRS
#define GU_PAIRS 1
#endif
constant int guPairs=GU_PAIRS;
#ifndef GU_RUN_CAP
#define GU_RUN_CAP 4
#endif
constant uint guRunCap=GU_RUN_CAP;
constant int guK=2816,guN=704,guSliceN=8;
struct ExpertRun { uint expert; uint count; bool leader; };
METAL_FUNC ExpertRun expert_run(const device uint* rhs,uint assignment) {
    const uint word=rhs[assignment];
#if GU_TAGGED_ROUTE
    const uint expert=word&0xffu;
    const uint offset=(word>>8)&0x3fu;
    if(guRunCap>1u && (offset&(guRunCap-1u))!=0u)return {expert,0,false};
    const uint count=min(guRunCap,((word>>14)&0x3fu)+1u);
    return {expert,count,true};
#else
    const bool tagged=(word&0x80000000u)!=0u;
    const uint expert=tagged ? word&0xffu:word;
    uint offset=0;
    if(tagged)offset=(word>>8)&0x3fu;
    else for(uint p=assignment;p>0;--p){if(rhs[p-1]!=expert)break;++offset;}
    if(guRunCap>1u && (offset&(guRunCap-1u))!=0u)return {expert,0,false};
    uint count=1;
    if(tagged)count=min(guRunCap,((word>>14)&0x3fu)+1u);
    else while(count<guRunCap && assignment+count<64 && rhs[assignment+count]==expert)++count;
    return {expert,count,true};
#endif
}
template<typename T>
METAL_FUNC void tg_execute_projection(const device uint* w,const device T* scales,const device T* biases,
    const device T* x,const device uint* lhs,threadgroup T* y0,int rowStride,
    const constant int& outputN,uint assignment,uint count,uint3 tid,uint sg,uint lane GU_DISPATCH_SUM_PARAMS) {
    const device T* x0=x+lhs[assignment]*2816;
    if(count==1){tg_qmv_affine4_g64_solo_impl<T,64,4>(w,scales,biases,x0,y0,guK,tid,sg,lane GU_HELPER_SUM_ARGS);return;}
    const device T* x1=x+lhs[assignment+1]*2816;threadgroup T* y1=y0+rowStride;
    if(count==2){tg_qmv_affine4_g64_pair_impl<T,64,4>(w,scales,biases,x0,x1,y0,y1,guK,tid,sg,lane GU_HELPER_SUM_ARGS);return;}
#if GU_RUN_CAP >= 3
    const device T* x2=x+lhs[assignment+2]*2816;threadgroup T* y2=y1+rowStride;
    if(count==3){tg_qmv_affine4_g64_triple_stream_impl<T,64,4>(w,scales,biases,x0,x1,x2,y0,y1,y2,guK,tid,sg,lane GU_HELPER_SUM_ARGS);return;}
#endif
#if GU_RUN_CAP >= 4
    const device T* x3=x+lhs[assignment+3]*2816;threadgroup T* y3=y2+rowStride;
    tg_qmv_affine4_g64_quad_stream_impl<T,64,4>(w,scales,biases,x0,x1,x2,x3,y0,y1,y2,y3,guK,tid,sg,lane GU_HELPER_SUM_ARGS);
#endif
}

"""#

    private static let packetSumKernel = MLXFast.metalKernel(
        name: "gemma4_b8_gu_packet_sums_v1", inputNames: ["x"], outputNames: ["sums"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i >= 2816u) return;
            thread float terms[8];
            sums[i] = load_vector<bfloat, float, 8, 4>(x + i * 8u, terms);
            """,
        header: "#define GU_RUN_CAP 4\n#define GU_TAGGED_ROUTE 0\n#define GU_INPUT_SUMS 0\n" + kernelHeader,
        ensureRowContiguous: true)

    private static let kernelGeneralSums: MLXFast.MLXFastKernel = makeKernel(tagged: false, inputSums: true)
    private static let kernelTaggedSums: MLXFast.MLXFastKernel = makeKernel(tagged: true, inputSums: true)

    private static let kernelGeneral: MLXFast.MLXFastKernel = makeKernel(tagged: false)
    private static let kernelTagged: MLXFast.MLXFastKernel = makeKernel(tagged: true)
}
