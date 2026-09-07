import Foundation
import MLX
import MLXFast

enum CBv2ExpertPrefillTasksV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_EXPERT_PREFILL_TASKS"] else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let supportsNAX: Bool = {
        #if canImport(Metal)
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        let override = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        let architecture = override.isEmpty ? GPU.deviceInfo().architecture : override
        guard architecture.count >= 3,
            let generation = Int(architecture.dropLast().suffix(2))
        else { return false }
        return generation >= (architecture.last == "p" ? 18 : 17)
        #else
        return false
        #endif
    }()

    private static let descriptorKernel = MLXFast.metalKernel(
        name: "cbv2_expert_prefill_descriptors_v1",
        inputNames: ["indices"], outputNames: ["tasks", "passes"],
        source: #"""
        const uint e = thread_position_in_threadgroup.x;
        threadgroup uint starts[128];
        threadgroup uint counts[128];
        threadgroup uint ends[128];
        threadgroup uint old_passes[128];
        uint lo = 0, hi = M;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            if (indices[mid] < e) lo = mid + 1; else hi = mid;
        }
        const uint start = lo;
        hi = M;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2;
            if (indices[mid] <= e) lo = mid + 1; else hi = mid;
        }
        starts[e] = start;
        counts[e] = (lo - start + 63) / 64;
        old_passes[e] = lo == start ? 0 : (lo + 63) / 64 - start / 64;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint j = 0; j <= e; ++j) prefix += counts[j];
        ends[e] = prefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            uint original = 0;
            for (uint j = 0; j < 128; ++j) original += old_passes[j];
            passes[0] = original;
            passes[1] = ends[127];
        }
        for (uint slot = e; slot < SLOTS; slot += 128) {
            if (slot >= ends[127]) {
                tasks[slot * 3] = 0;
                tasks[slot * 3 + 1] = 0;
                tasks[slot * 3 + 2] = 0;
                continue;
            }
            uint a = 0, b = 128;
            while (a < b) {
                uint mid = a + (b - a) / 2;
                if (ends[mid] <= slot) a = mid + 1; else b = mid;
            }
            const uint prior = a == 0 ? 0 : ends[a - 1];
            const uint row = starts[a] + (slot - prior) * 64;
            const uint limit = a == 127 ? M : starts[a + 1];
            tasks[slot * 3] = a;
            tasks[slot * 3 + 1] = row;
            tasks[slot * 3 + 2] = min(64u, limit - row);
        }
        """, ensureRowContiguous: true)

    private static let projectionKernel = MLXFast.metalKernel(
        name: "cbv2_expert_prefill_nax_v1",
        inputNames: ["x", "weight", "scales", "biases", "tasks"],
        outputNames: ["output"],
        source: #"""
        using namespace mlx::steel;
        const uint slot = threadgroup_position_in_grid.y;
        const uint live = tasks[slot * 3 + 2];
        if (live == 0) return;
        const uint expert = tasks[slot * 3];
        const uint row = tasks[slot * 3 + 1];
        const uint col = threadgroup_position_in_grid.x * 64;
        const short tm = short(simdgroup_index_in_threadgroup) * 16;
        const short rows = short(min(16, max(0, int(live) - int(tm))));
        constexpr short BK = 64;
        constexpr short SK = 32;
        constexpr short PAD = 64 + 16 / sizeof(T);
        threadgroup T Ws[64 * PAD];
        using Loader = QuantizedBlockLoader<T, 64, 64, PAD, 1, 128, 64, 4>;
        thread Loader loader(
            (const device uint8_t*)weight +
                (size_t(expert) * N + col) * (K / 2),
            scales + (size_t(expert) * N + col) * (K / 64),
            biases + (size_t(expert) * N + col) * (K / 64),
            K, Ws, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        NAXTile<float, 1, 4> Dtile;
        Dtile.clear();
        const device T* xn = x + (size_t(row) + (rows > 0 ? tm : 0)) * K;
        dispatch_bool(rows == 16, [&](auto alignedM) {
            for (int k = 0; k < K / BK; ++k) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                loader.load_unsafe();
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (rows > 0) {
                    STEEL_PRAGMA_NO_UNROLL
                    for (int kk = 0; kk < BK; kk += SK) {
                        NAXTile<T, 1, 2> Atile;
                        NAXTile<T, 4, 2> Btile;
                        if constexpr (alignedM.value) {
                            Atile.load(xn + kk, K);
                        } else {
                            Atile.load_safe(xn + kk, K, short2(SK, rows));
                        }
                        Btile.template load<T, PAD, 1>(Ws + kk);
                        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{},
                            Btile, metal::bool_constant<true>{});
                    }
                }
                xn += BK;
                loader.next();
            }
        });
        if (rows == 0) return;
        if constexpr (GEGLU) {
            NAXTile<float, 1, 2> Otile;
            const_for_loop<0, 2, 1>([&](auto nn) {
                thread auto& gate = Dtile.frag_at(0, short(nn) * 2);
                thread auto& up = Dtile.frag_at(0, short(nn) * 2 + 1);
                thread auto& out = Otile.frag_at(0, short(nn));
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < Dtile.kElemsPerFrag; ++i) {
                    const T g = static_cast<T>(gate[i]);
                    const T u = static_cast<T>(up[i]);
                    out[i] = float(gemma4_geglu_compiled_tape(g, u));
                }
            });
            device T* y = output + (size_t(row) + tm) * (N / 2) + col / 2;
            if (rows == 16) Otile.store(y, N / 2);
            else Otile.store_slice(y, N / 2, short2(0, 0), short2(32, rows));
        } else {
            device T* y = output + (size_t(row) + tm) * N + col;
            if (rows == 16) Dtile.store(y, N);
            else Dtile.store_slice(y, N, short2(0, 0), short2(64, rows));
        }
        """, header: CBv2ExpertPrefillNAXSourceV1.header, ensureRowContiguous: true)

    static func project(
        x: MLXArray, indices: MLXArray, gateUp: SwitchGateUpFusedStorage,
        down: QuantizedSwitchLinear
    ) -> MLXArray? {
        guard enabled, supportsNAX,
            x.ndim == 3, x.dim(1) == 1, x.dim(2) == 2816,
            x.dim(0) >= 512, x.dim(0) <= Int(Int32.max) - 8192,
            x.dtype == .bfloat16,
            indices.dtype == .uint32, indices.size == x.dim(0),
            down.inputDims == 704, down.outputDims == 2816, down.numExperts == 128,
            down.groupSize == 64, down.bits == 4, down.mode == .affine, down.bias == nil,
            down.weight.dtype == .uint32, down.weight.shape == [128, 2816, 88],
            down.scales.dtype == .bfloat16, down.scales.shape == [128, 2816, 11],
            let downBiases = down.biases,
            downBiases.dtype == .bfloat16, downBiases.shape == [128, 2816, 11]
        else { return nil }
        let rows = x.dim(0)
        let slots = (rows + 63) / 64 + 127
        let descriptors = descriptorKernel(
            [indices], template: [("M", rows), ("SLOTS", slots)],
            grid: (128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[slots, 3], [2]], outputDTypes: [.uint32, .uint32])
        let tasks = descriptors[0]
        let activated = projectionKernel(
            [x, gateUp.weight, gateUp.scales, gateUp.biases, tasks],
            template: [("T", DType.bfloat16), ("N", 1408), ("K", 2816), ("GEGLU", true)],
            grid: (22 * 32, slots * 4, 1), threadGroup: (32, 4, 1),
            outputShapes: [[rows, 1, 704]], outputDTypes: [.bfloat16])[0]
        let output = projectionKernel(
            [activated, down.weight, down.scales, downBiases, tasks],
            template: [("T", DType.bfloat16), ("N", 2816), ("K", 704), ("GEGLU", false)],
            grid: (44 * 32, slots * 4, 1), threadGroup: (32, 4, 1),
            outputShapes: [[rows, 1, 2816]], outputDTypes: [.bfloat16])[0]
        CBv2EngageMark.once("prefill-expert-aligned-nax-tasks")
        return output
    }
}
