import Foundation
import MLX
import MLXFast

enum CBv2GroupedPrefillPVV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_GROUPED_PREFILL_PV"] else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let tightSwizzleEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_GROUPED_PV_TIGHT_SWIZZLE_V1"] else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private struct Geometry {
        let bm: Int
        let bk: Int
        let wm: Int
        let swizzle: Int
    }

    private static let geometry: Geometry? = {
        #if canImport(Metal)
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return nil
        }
        let override = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        let architecture = override.isEmpty ? GPU.deviceInfo().architecture : override
        let suffix = architecture.last
        guard architecture.count >= 3,
            let generation = Int(architecture.dropLast().suffix(2)),
            generation >= (suffix == "p" ? 18 : 17)
        else { return nil }
        if suffix == "s" || suffix == "c" || suffix == "d" {
            let swizzle = tightSwizzleEnabled ? 1 : 2
            return Geometry(bm: 64, bk: 256, wm: 2, swizzle: swizzle)
        }
        return Geometry(bm: 128, bk: 512, wm: 4, swizzle: 0)
        #else
        return nil
        #endif
    }()

    private static let source = #"""
        using namespace mlx::steel;
        const uint3 tid = threadgroup_position_in_grid;
        const int block = int(tid.z) / (BATCH * 16);
        const int batchHead = int(tid.z) % (BATCH * 16);
        const int b = batchHead / 16;
        const int h = batchHead % 16;
        const int tid_y = (int(tid.y) << SWIZZLE) +
            (int(tid.x) & ((1 << SWIZZLE) - 1));
        const int tid_x = int(tid.x) >> SWIZZLE;
        if (tid_y >= 128 / BM || tid_x >= D / 128) return;
        const int K = (block + 1) * 128;
        const device T* scores = nullptr;
        const device T* stats = nullptr;
        switch (block) {
            case 0: scores = s0; stats = r0; break;
            case 1: scores = s1; stats = r1; break;
            case 2: scores = s2; stats = r2; break;
            case 3: scores = s3; stats = r3; break;
            case 4: scores = s4; stats = r4; break;
            case 5: scores = s5; stats = r5; break;
            case 6: scores = s6; stats = r6; break;
            default: scores = s7; stats = r7; break;
        }
        threadgroup_barrier(mem_flags::mem_none);
        constexpr short SM = BM / WM;
        constexpr short SN = 128 / 4;
        const short tm = SM * (simdgroup_index_in_threadgroup / 4);
        const short tn = SN * (simdgroup_index_in_threadgroup % 4);
        const int row = tid_y * BM + tm;
        const int col = tid_x * 128 + tn;
        const device T* A = scores + (size_t(batchHead) * 128 + row) * K;
        const device T* C = stats + (size_t(batchHead) * 128 + row) * 4;
        const device T* V = values +
            (size_t(b) * KVHEADS + h / (16 / KVHEADS)) * 1024 * D + col;
        device T* O = output +
            ((size_t(b) * 1024 + block * 128 + row) * 16 + h) * D + col;
        dispatch_bool(K % BK == 0, [&](auto alignedK) {
            auto tile = gemm_loop_softmax<
                T, SM, SN, 32, BK, false, false, true, true, alignedK.value, float>(
                A, V, K, D, K, K / BK, SM, SN, C);
            tile.store(O, 16 * D);
        });
        """#

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_grouped_prefill_pv_nax_v1",
        inputNames: (0..<8).map { "s\($0)" } + (0..<8).map { "r\($0)" } + ["values"],
        outputNames: ["output"],
        source: source,
        header: CBv2GroupedPrefillPVNAXSourceV1.header,
        ensureRowContiguous: true)

    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, window: Int?, sinks: MLXArray?, softcap: Float?,
        queryPlane: MLXArray?
    ) -> MLXArray? {
        guard enabled, let geometry,
            CBv2PrefillAttnTrafficV1.enabled, !CBv2PrefillAttnTrafficV1.xcheck,
            scale == 1, sinks == nil, softcap == nil,
            (window ?? 1024) >= 1024,
            queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
            queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            queries.dim(0) >= 1, queries.dim(0) <= 8,
            queries.dim(1) == 16, queries.dim(2) == 1024,
            keys.dim(0) == queries.dim(0), values.shape == keys.shape,
            keys.dim(2) == 1024, keys.dim(3) == queries.dim(3),
            (keys.dim(1) == 8 && queries.dim(3) == 256)
                || (keys.dim(1) == 2 && queries.dim(3) == 512),
            let queryPlane
        else { return nil }

        var scores: [MLXArray] = []
        var stats: [MLXArray] = []
        scores.reserveCapacity(8)
        stats.reserveCapacity(8)
        for block in 0..<8 {
            let start = block * 128
            let end = start + 128
            guard let stage = CBv2ComposedPrefillSDPAV1.prepareScores(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys[0..., 0..., 0..<end, 0...],
                values: values[0..., 0..., 0..<end, 0...],
                scale: scale, L: 128, kL: end, window: window,
                bidirectional: false, sinks: sinks,
                queryPlaneSlice: queryPlane[0..., 0..., 0..., start..<end, 0...]),
                let statistics = CBv2PrefillAttnTrafficV1.statistics(
                    scores: stage.scores, values: stage.values)
            else { return nil }
            scores.append(stage.scores)
            stats.append(statistics)
        }
        let batch = queries.dim(0)
        let dim = values.dim(3)
        let swizzleTile = 1 << geometry.swizzle
        let tilesN = dim / 128 * swizzleTile
        let tilesM = (128 / geometry.bm + swizzleTile - 1) / swizzleTile
        CBv2EngageMark.once("prefill-grouped-pv-final")
        return kernel(
            scores + stats + [values],
            template: [("T", queries.dtype), ("BATCH", batch), ("D", dim),
                ("KVHEADS", keys.dim(1)), ("BM", geometry.bm), ("BK", geometry.bk),
                ("WM", geometry.wm), ("SWIZZLE", geometry.swizzle)],
            grid: (tilesN * 32, tilesM * 4, batch * 16 * 8 * geometry.wm),
            threadGroup: (32, 4, geometry.wm),
            outputShapes: [[batch, 1024, 16, dim]],
            outputDTypes: [.bfloat16])[0]
    }
}
