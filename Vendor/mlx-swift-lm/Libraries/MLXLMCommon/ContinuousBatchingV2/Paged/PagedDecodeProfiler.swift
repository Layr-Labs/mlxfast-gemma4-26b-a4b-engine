// PagedDecodeProfiler.swift
//
// Root-cause profiler for the paged decode path (kernel-opt track).
//
// The paged kernel passes all parity suites and the WS-C microbenchmark
// (PagedBackendBenchmark: ~1 ms/step at GPT-OSS shapes) yet decodes real
// GPT-OSS-20B at ~2 s/token — ~100x slower than the contiguous backend.
// The microbenchmark ran with a TINY pool (a few MB of slab); the real
// engine runs with 16 GiB. This profiler isolates the terms that could
// scale with slab size or layer count:
//
//   1. slab-size scaling: same shape, same work, pool capacity swept from
//      MBs to GiBs. Any ms/step growth is O(slab) work per step (e.g. a
//      slice-update write that stopped donating and degraded into a full
//      slab copy, or `ensureRowContiguous` copying a kernel input).
//   2. write-only vs dispatch-only vs both: attributes the O(slab) term to
//      the KV writes or to the kernel dispatch.
//   3. 24-layer GPT-OSS emulation: alternating sliding-window(128)/full
//      layers with sinks, all sharing ONE slab group (GPT-OSS reality),
//      with a value dependency chained through the step like a real model.
//      Paged vs per-row contiguous SDPA on the same structure.
//   4. dispatch overhead: many dispatch pairs per eval vs one per eval on
//      a tiny context — separates per-dispatch encode cost from submit.
//
// Run via the env-gated test:
//   DARKBLOOM_CBV2_PAGED_PROFILE=1 swift test --filter CBv2PagedProfileTests
//
// Wall-clock ablations by design (Metal counters need a capture context;
// varying one dimension at a time gives unambiguous attribution anyway).

import Foundation
import MLX
import MLXFast
import MLXRandom

public struct PagedDecodeProfiler {

    // MARK: - Config

    public static let gptossQueryHeads = 64
    public static let gptossKVHeads = 8
    public static let gptossHeadDim = 64
    public static let gptossWindow = 128

    public enum StepMode: String, CaseIterable, Sendable {
        case writeAndDispatch = "write+dispatch"
        case dispatchOnly = "dispatch-only"
        case writeOnly = "write-only"
    }

    public struct Result: Sendable {
        public var label: String
        public var mode: String
        public var batch: Int
        public var context: Int
        public var layers: Int
        public var slabGiB: Double
        public var msPerStep: Double
        public var stepPeakGiB: Double = 0
        public var stepTimesMs: [Double] = []

        public var markdownRow: String {
            let first = stepTimesMs.first.map { String(format: "%.2f", $0) } ?? "-"
            let last = stepTimesMs.last.map { String(format: "%.2f", $0) } ?? "-"
            return String(
                format: "| %@ | %@ | %d | %d | %d | %.2f | %.3f | %.2f | %@/%@ |",
                label, mode, batch, context, layers, slabGiB, msPerStep, stepPeakGiB,
                first, last)
        }
    }

    public static let markdownHeader = """
        | scenario | mode | B | ctx | layers | slab GiB | ms/step | step peak GiB | first/last ms |
        |----------|------|---|-----|--------|----------|---------|---------------|---------------|
        """

    public var warmupSteps: Int
    public var timedSteps: Int

    public init(warmupSteps: Int = 5, timedSteps: Int = 20) {
        self.warmupSteps = warmupSteps
        self.timedSteps = timedSteps
    }

    // MARK: - Scenario harness

    static func gptossLayerKinds(count: Int) -> [CBv2LayerKind] {
        (0 ..< count).map { i in
            CBv2LayerKind(
                attention: i % 2 == 0 ? .slidingWindow(gptossWindow) : .full,
                hasSinks: true,
                headDim: gptossHeadDim, kvHeads: gptossKVHeads, queryHeads: gptossQueryHeads)
        }
    }

    func measurePaged(
        label: String,
        layerCount: Int,
        capacityBytes: Int,
        batch: Int,
        context: Int,
        mode: StepMode
    ) throws -> Result {
        let kinds = Self.gptossLayerKinds(count: layerCount)
        let maxLength = context + warmupSteps + timedSteps + 8
        let config = PagedKVPoolConfig(
            capacityBytes: capacityBytes,
            maxPrefillChunk: 512,
            nominalMaxSequenceLength: maxLength)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        try backend.pool.materializeSlabs()
        let caches = backend.makeLayerCaches()

        var states: [[CBv2SequenceKV?]] = []
        for _ in 0 ..< batch {
            states.append(
                try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: context, maxLength: maxLength))
        }
        defer { states.forEach { backend.release($0) } }

        let d = Self.gptossHeadDim
        let kvh = Self.gptossKVHeads
        let qh = Self.gptossQueryHeads
        let scale = Float(1.0 / Double(d).squareRoot())
        let sinks = MLXRandom.normal([qh], dtype: .float32)

        for layer in 0 ..< layerCount {
            let rows = states.map { $0[layer]! as! PagedSequenceKV }
            for row in rows {
                var remaining = context
                while remaining > 0 {
                    let n = min(512, remaining)
                    row.write(
                        keys: MLXRandom.normal([kvh, n, d], dtype: .float16),
                        values: MLXRandom.normal([kvh, n, d], dtype: .float16))
                    remaining -= n
                }
            }
            caches[layer].setRows(rows)
        }
        let group = backend.pool.group(PagedKVGroupKey(kinds[0]))
        eval([group.writeFence])

        let qBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, qh, 1, d], dtype: .float16)
        }
        let kBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, kvh, 1, d], dtype: .float16)
        }
        let vBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, kvh, 1, d], dtype: .float16)
        }
        let zero = MLXArray(Float16(0))

        let step: () -> Void = {
            var carry = zero
            for layer in 0 ..< layerCount {
                let cache = caches[layer]
                let rows = cache.rows.map { $0 as! PagedSequenceKV }
                let q = qBase[layer] + carry
                let k = kBase[layer] + carry
                let v = vBase[layer] + carry
                switch mode {
                case .writeAndDispatch:
                    let out = cache.updateAndAttend(
                        queries: q, keys: k, values: v, scale: scale, sinks: sinks)
                    carry = out[0, 0, 0, 0] * zero
                case .writeOnly:
                    for (i, row) in rows.enumerated() {
                        row.write(keys: k[i], values: v[i])
                    }
                    carry = k[0, 0, 0, 0] * zero
                case .dispatchOnly:
                    let (seqinfo, maxAttend) = PagedAttentionKernel.seqinfo(
                        rows.map { $0.seqInfoRow(attending: $0.decodeAttendRange) })
                    let (out, _) = PagedAttentionKernel.decode(
                        queries: q,
                        kSlab: group.kSlab,
                        vSlab: group.vSlab,
                        tables: cache.deviceTables(rows: rows),
                        seqinfo: seqinfo,
                        maxAttendLength: maxAttend,
                        sinks: sinks,
                        params: MLXArray([Float(1), scale, 0, 0, 0, 0, 0, 0]),
                        softcap: false,
                        pageSize: backend.pool.config.pageSize,
                        writeFence: group.writeFence,
                        kernelSource: backend.pool.kernelSource)
                    carry = out[0, 0, 0] * zero
                }
            }
            switch mode {
            case .writeOnly:
                eval([group.writeFence])
            case .writeAndDispatch, .dispatchOnly:
                eval(carry)
            }
        }

        let (ms, peak, stepTimes) = time(step)
        let slabGiB =
            Double(group.pageCount) * Double(group.pageBytes) / Double(1 << 30)
        return Result(
            label: label, mode: mode.rawValue, batch: batch, context: context,
            layers: layerCount, slabGiB: slabGiB, msPerStep: ms,
            stepPeakGiB: peak, stepTimesMs: stepTimes)
    }

    func measureContiguousSDPA(
        label: String, layerCount: Int, batch: Int, context: Int
    ) -> Result {
        let kinds = Self.gptossLayerKinds(count: layerCount)
        let capacity = context + warmupSteps + timedSteps + 8
        let d = Self.gptossHeadDim
        let kvh = Self.gptossKVHeads
        let qh = Self.gptossQueryHeads
        let scale = Float(1.0 / Double(d).squareRoot())
        let sinks = MLXRandom.normal([qh], dtype: .float16)

        var ks: [[MLXArray]] = []
        var vs: [[MLXArray]] = []
        for _ in 0 ..< layerCount {
            var lk: [MLXArray] = []
            var lv: [MLXArray] = []
            for _ in 0 ..< batch {
                let k = MLXArray.zeros([1, kvh, capacity, d], dtype: .float16)
                let v = MLXArray.zeros([1, kvh, capacity, d], dtype: .float16)
                k[0..., 0..., 0 ..< context, 0...] =
                    MLXRandom.normal([1, kvh, context, d], dtype: .float16)
                v[0..., 0..., 0 ..< context, 0...] =
                    MLXRandom.normal([1, kvh, context, d], dtype: .float16)
                lk.append(k)
                lv.append(v)
            }
            ks.append(lk)
            vs.append(lv)
        }
        eval(ks.flatMap { $0 } + vs.flatMap { $0 })

        let qBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, qh, 1, d], dtype: .float16)
        }
        let kBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, kvh, 1, d], dtype: .float16)
        }
        let vBase = (0 ..< layerCount).map { _ in
            MLXRandom.normal([batch, kvh, 1, d], dtype: .float16)
        }
        let zero = MLXArray(Float16(0))
        var length = context

        let step: () -> Void = {
            var carry = zero
            for layer in 0 ..< layerCount {
                let q = qBase[layer] + carry
                let k = kBase[layer] + carry
                let v = vBase[layer] + carry
                let window: Int? = {
                    if case .slidingWindow(let w) = kinds[layer].attention { return w }
                    return nil
                }()
                for i in 0 ..< batch {
                    ks[layer][i][0..., 0..., length ..< (length + 1), 0...] =
                        k[i ..< (i + 1)]
                    vs[layer][i][0..., 0..., length ..< (length + 1), 0...] =
                        v[i ..< (i + 1)]
                    let t = length + 1
                    let start = window.map { max(0, t - $0) } ?? 0
                    let out = MLXFast.scaledDotProductAttention(
                        queries: q[i ..< (i + 1)],
                        keys: ks[layer][i][0..., 0..., start ..< t, 0...],
                        values: vs[layer][i][0..., 0..., start ..< t, 0...],
                        scale: scale, mask: .none, sinks: sinks)
                    carry = out[0, 0, 0, 0] * zero
                }
            }
            length += 1
            eval(carry)
        }

        let (ms, peak, stepTimes) = time(step)
        let slabGiB =
            Double(layerCount * batch * 2 * kvh * capacity * d * 2) / Double(1 << 30)
        return Result(
            label: label, mode: "contig-sdpa", batch: batch, context: context,
            layers: layerCount, slabGiB: slabGiB, msPerStep: ms,
            stepPeakGiB: peak, stepTimesMs: stepTimes)
    }

    private func time(_ step: () -> Void) -> (
        msPerStep: Double, stepPeakGiB: Double, stepTimesMs: [Double]
    ) {
        for _ in 0 ..< warmupSteps { step() }
        let baseline = Memory.activeMemory
        Memory.peakMemory = 0
        var stepTimes: [Double] = []
        stepTimes.reserveCapacity(timedSteps)
        let start = DispatchTime.now()
        for _ in 0 ..< timedSteps {
            let s0 = DispatchTime.now()
            step()
            stepTimes.append(
                Double(DispatchTime.now().uptimeNanoseconds - s0.uptimeNanoseconds) / 1e6)
        }
        let elapsed =
            Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        let peak = Double(max(0, Memory.peakMemory - baseline)) / Double(1 << 30)
        return (elapsed * 1000 / Double(timedSteps), peak, stepTimes)
    }

    // MARK: - Suites

    public func slabScalingSuite(
        capacitiesBytes: [Int] = [64 << 20, 512 << 20, 2 << 30, 8 << 30],
        context: Int = 512
    ) throws -> [Result] {
        var results: [Result] = []
        for capacity in capacitiesBytes {
            for mode in StepMode.allCases {
                results.append(
                    try measurePaged(
                        label: "slab-sweep",
                        layerCount: 1, capacityBytes: capacity, batch: 1,
                        context: context, mode: mode))
                Memory.clearCache()
            }
        }
        return results
    }

    public func layerScalingSuite(
        capacityBytes: Int = 2 << 30,
        layerCounts: [Int] = [1, 2, 4, 8, 16, 24],
        context: Int = 512,
        modes: [StepMode] = [.writeAndDispatch, .writeOnly]
    ) throws -> [Result] {
        var results: [Result] = []
        for layers in layerCounts {
            for mode in modes {
                results.append(
                    try measurePaged(
                        label: "layer-sweep",
                        layerCount: layers, capacityBytes: capacityBytes, batch: 1,
                        context: context, mode: mode))
                Memory.clearCache()
            }
        }
        return results
    }

    public func gptossEmulationSuite(
        capacityBytes: Int = 16 << 30, context: Int = 512, batches: [Int] = [1, 4]
    ) throws -> [Result] {
        var results: [Result] = []
        for batch in batches {
            results.append(
                try measurePaged(
                    label: "gptoss-24L",
                    layerCount: 24, capacityBytes: capacityBytes, batch: batch,
                    context: context, mode: .writeAndDispatch))
            Memory.clearCache()
            results.append(
                measureContiguousSDPA(
                    label: "gptoss-24L", layerCount: 24, batch: batch, context: context))
            Memory.clearCache()
        }
        return results
    }

    public func dispatchOverheadSuite() throws -> [Result] {
        var results: [Result] = []
        for layers in [1, 24] {
            results.append(
                try measurePaged(
                    label: "overhead",
                    layerCount: layers, capacityBytes: 64 << 20, batch: 1,
                    context: 16, mode: .dispatchOnly))
            Memory.clearCache()
        }
        return results
    }

    // MARK: - MTP verify-round attribution (gemma-4 26B geometry)

    public static func gemma4LayerKinds() -> [CBv2LayerKind] {
        (0 ..< 30).map { i in
            i % 6 == 5
                ? CBv2LayerKind(
                    attention: .full, headDim: 512, kvHeads: 2, queryHeads: 16)
                : CBv2LayerKind(
                    attention: .slidingWindow(1024), headDim: 256, kvHeads: 8,
                    queryHeads: 16)
        }
    }

    static func gemma4CaptureLayers(_ kinds: [CBv2LayerKind]) -> (full: Int, sliding: Int) {
        let full = kinds.lastIndex { $0.attention == .full && $0.sharesKVWithLayer == nil }!
        let sliding = kinds.lastIndex {
            if case .slidingWindow = $0.attention, $0.sharesKVWithLayer == nil { return true }
            return false
        }!
        return (full, sliding)
    }

    public enum MTPRoundPhase: String, CaseIterable, Sendable {
        case verify = "verify-1+k"
        case decodeSerial = "decode-x(1+k)"
        case capture = "capture+fence"
        case captureProbeFence = "capture+probe"
    }

    func measurePagedMTPRound(
        label: String, k: Int, context: Int, phase: MTPRoundPhase,
        capacityBytes: Int = 8 << 30
    ) throws -> Result {
        let kinds = Self.gemma4LayerKinds()
        let columns = 1 + k
        let maxLength = context + (warmupSteps + timedSteps + 2) * columns
        let config = PagedKVPoolConfig(
            capacityBytes: capacityBytes,
            maxPrefillChunk: 512,
            nominalMaxSequenceLength: maxLength)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        try backend.pool.materializeSlabs()
        let caches = backend.makeLayerCaches()
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: context, maxLength: maxLength)
        defer { backend.release(state) }

        var rows: [PagedSequenceKV] = []
        for (layer, kind) in kinds.enumerated() {
            let row = state[layer]! as! PagedSequenceKV
            var remaining = context
            while remaining > 0 {
                let n = min(config.maxPrefillChunk, remaining)
                row.write(
                    keys: MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16),
                    values: MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16))
                remaining -= n
            }
            caches[layer].setRows([row])
            rows.append(row)
        }
        for key in backend.pool.groupKeys { eval([backend.pool.group(key).writeFence]) }

        let capture = Self.gemma4CaptureLayers(kinds)
        let scale = Float(1.0 / Double(kinds[0].headDim).squareRoot())
        let qkv = Self.roundTiles(kinds: kinds, columns: columns)
        let zero = MLXArray(Float16(0))

        let step: () -> Void = {
            switch phase {
            case .capture, .captureProbeFence:
                var edges: [MLXArray] = []
                for layer in [capture.full, capture.sliding] {
                    let row = rows[layer]
                    let snap = row.snapshot()
                    let group = row.pool.group(row.groupKey)
                    let edge =
                        phase == .capture
                        ? (snap.keys.sum() + snap.values.sum())
                        : (snap.keys[0, 0, 0, 0] + snap.values[0, 0, 0, 0])
                    group.writeFence = group.writeFence + edge.asType(group.writeFence.dtype) * 0
                    edges.append(group.writeFence)
                }
                eval(edges)
            case .verify:
                for row in rows { row.beginSpeculativeWrite() }
                var carry = zero
                for layer in 0 ..< kinds.count {
                    caches[layer].mtpSerializesRectangularAttention = true
                    let out = caches[layer].updateAndAttend(
                        queries: qkv.q[layer] + carry, keys: qkv.k[layer] + carry,
                        values: qkv.v[layer] + carry, scale: scale, sinks: nil)
                    caches[layer].mtpSerializesRectangularAttention = false
                    carry = out[0, 0, 0, 0] * zero
                }
                eval(carry)
                for row in rows { row.commitSpeculativeWrite() }
            case .decodeSerial:
                var carry = zero
                for t in 0 ..< columns {
                    for layer in 0 ..< kinds.count {
                        let out = caches[layer].updateAndAttend(
                            queries: qkv.q[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            keys: qkv.k[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            values: qkv.v[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            scale: scale, sinks: nil)
                        carry = out[0, 0, 0, 0] * zero
                    }
                }
                eval(carry)
            }
        }

        let (ms, peak, stepTimes) = time(step)
        let slabGiB = Double(backend.pool.bytesPhysical) / Double(1 << 30)
        return Result(
            label: label, mode: "paged " + phase.rawValue, batch: 1, context: context,
            layers: kinds.count, slabGiB: slabGiB, msPerStep: ms,
            stepPeakGiB: peak, stepTimesMs: stepTimes)
    }

    func measureContiguousMTPRound(
        label: String, k: Int, context: Int, phase: MTPRoundPhase
    ) throws -> Result {
        let kinds = Self.gemma4LayerKinds()
        let columns = 1 + k
        let maxLength = context + (warmupSteps + timedSteps + 2) * columns
        let backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: 8 << 30, kvDType: .float16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: context, maxLength: maxLength)
        defer { backend.release(state) }
        let caches = kinds.enumerated().map { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind, rows: [state[index]!])
        }
        var rows: [CBv2SequenceKV] = []
        for (layer, kind) in kinds.enumerated() {
            let row = state[layer]!
            _ = row.update(
                keys: MLXRandom.normal([1, kind.kvHeads, context, kind.headDim], dtype: .float16),
                values: MLXRandom.normal([1, kind.kvHeads, context, kind.headDim], dtype: .float16)
            )
            caches[layer].setRows([row])
            rows.append(row)
        }
        eval(rows.flatMap { [$0.snapshot().keys, $0.snapshot().values] })

        let capture = Self.gemma4CaptureLayers(kinds)
        let scale = Float(1.0 / Double(kinds[0].headDim).squareRoot())
        let qkv = Self.roundTiles(kinds: kinds, columns: columns)
        let zero = MLXArray(Float16(0))

        let step: () -> Void = {
            switch phase {
            case .capture, .captureProbeFence:
                var probes: [MLXArray] = []
                for layer in [capture.full, capture.sliding] {
                    let snap = rows[layer].snapshot()
                    probes.append(snap.keys[0, 0, 0, 0] + snap.values[0, 0, 0, 0])
                }
                eval(probes)
            case .verify:
                for row in rows { row.beginSpeculativeWrite() }
                var carry = zero
                for layer in 0 ..< kinds.count {
                    caches[layer].mtpSerializesRectangularAttention = true
                    let out = caches[layer].updateAndAttend(
                        queries: qkv.q[layer] + carry, keys: qkv.k[layer] + carry,
                        values: qkv.v[layer] + carry, scale: scale, sinks: nil)
                    caches[layer].mtpSerializesRectangularAttention = false
                    carry = out[0, 0, 0, 0] * zero
                }
                eval(carry)
                for row in rows { row.commitSpeculativeWrite() }
            case .decodeSerial:
                var carry = zero
                for t in 0 ..< columns {
                    for layer in 0 ..< kinds.count {
                        let out = caches[layer].updateAndAttend(
                            queries: qkv.q[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            keys: qkv.k[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            values: qkv.v[layer][0..., 0..., t ..< (t + 1), 0...] + carry,
                            scale: scale, sinks: nil)
                        carry = out[0, 0, 0, 0] * zero
                    }
                }
                eval(carry)
            }
        }

        let (ms, peak, stepTimes) = time(step)
        return Result(
            label: label, mode: "contig " + phase.rawValue, batch: 1, context: context,
            layers: kinds.count, slabGiB: 0, msPerStep: ms,
            stepPeakGiB: peak, stepTimesMs: stepTimes)
    }

    private static func roundTiles(
        kinds: [CBv2LayerKind], columns: Int
    ) -> (q: [MLXArray], k: [MLXArray], v: [MLXArray]) {
        var q: [MLXArray] = []
        var k: [MLXArray] = []
        var v: [MLXArray] = []
        for kind in kinds {
            q.append(
                MLXRandom.normal([1, kind.queryHeads, columns, kind.headDim], dtype: .float16))
            k.append(MLXRandom.normal([1, kind.kvHeads, columns, kind.headDim], dtype: .float16))
            v.append(MLXRandom.normal([1, kind.kvHeads, columns, kind.headDim], dtype: .float16))
        }
        eval(q + k + v)
        return (q, k, v)
    }

    public func mtpSeqinfoUploadCost(k: Int, layers: Int = 30) -> [Result] {
        func measure(_ label: String, uploads: Int) -> Result {
            let step: () -> Void = {
                var arrays: [MLXArray] = []
                arrays.reserveCapacity(uploads)
                for i in 0 ..< uploads {
                    arrays.append(MLXArray([Int32(i), 1, 1, 0, 0, 0, 0, 0], [1, 8]))
                }
                eval(arrays)
            }
            let (ms, peak, times) = time(step)
            return Result(
                label: label, mode: "seqinfo x\(uploads)", batch: 1, context: 0,
                layers: layers, slabGiB: 0, msPerStep: ms, stepPeakGiB: peak,
                stepTimesMs: times)
        }
        return [
            measure("seqinfo-decode", uploads: layers),
            measure("seqinfo-round", uploads: (1 + k) * layers),
        ]
    }

    public func mtpRoundSuite(k: Int = 3, context: Int = 1024) throws -> [Result] {
        var results: [Result] = []
        for phase in MTPRoundPhase.allCases {
            results.append(
                try measurePagedMTPRound(
                    label: "gemma4-30L", k: k, context: context, phase: phase))
            Memory.clearCache()
            results.append(
                try measureContiguousMTPRound(
                    label: "gemma4-30L", k: k, context: context, phase: phase))
            Memory.clearCache()
        }
        results.append(contentsOf: mtpSeqinfoUploadCost(k: k))
        return results
    }
}
