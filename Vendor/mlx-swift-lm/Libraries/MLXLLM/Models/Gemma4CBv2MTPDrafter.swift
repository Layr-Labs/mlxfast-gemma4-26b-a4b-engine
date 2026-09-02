// Copyright © 2026 Apple Inc.

// ContinuousBatchingV2 drafter adapter: `CBv2MTPDrafter` over the v1
// `Gemma4AssistantDraftModel`. The engine hands per-row frozen-KV captures
// (`CBv2MTPRowCapture`); this adapter owns padding, mask construction,
// target-embedding lookup, and the greedy argmax chain — the drafter module
// itself is reused unchanged.
//
// Round invariants (must match the v1 `runGemma4MTPGreedyRound` semantics):
//  - The query RoPE position is CONSTANT for every draft step of a round
//    (each row's anchor), carried as a `.batch` offset.
//  - Full attention is bidirectional over the whole retained snapshot.
//  - Sliding attention uses ABSOLUTE retained positions and the target's
//    strict `(anchor - window, anchor + window)` rule. A full rotating ring
//    retains the boundary key at `anchor-window`; it must be masked even for
//    B == 1.
//  - B > 1 right-pads each row's capture to the batch max length and combines
//    the absolute sliding-window rule with the padding mask.
//    v1's `makeMasks` is NOT reused here: it reads row 0 of a `.batch`
//    offset (a shared-frontier assumption that is wrong for CBv2 rows at
//    genuinely different positions) and it keys the sliding mask to
//    0-based storage indices rather than absolute positions.
//  - No host syncs: lengths/anchors are host ints from capture metadata;
//    padding, masks, and the argmax chain all stay lazy on device.

import Foundation
import MLX
import MLXLMCommon

enum Gemma4MTPWarmRetirement {
    static let timeoutSeconds = 120
    static let engineShutdownTimeoutSeconds = timeoutSeconds + 5

    enum Stage: String, Equatable, Sendable {
        case drain
        case shutdown
    }

    enum State: Equatable, Sendable {
        case waitingForDrain
        case waitingForShutdown
        case retired
        case failed(Stage)
    }

    enum Event: Sendable {
        case completed
        case deadlineExpired
    }

    static func transition(_ state: State, on event: Event) -> State {
        switch (state, event) {
        case (.waitingForDrain, .completed):
            return .waitingForShutdown
        case (.waitingForDrain, .deadlineExpired):
            return .failed(.drain)
        case (.waitingForShutdown, .completed):
            return .retired
        case (.waitingForShutdown, .deadlineExpired):
            return .failed(.shutdown)
        case (.retired, _), (.failed, _):
            return state
        }
    }

    struct Failure: LocalizedError, Equatable, Sendable {
        let stage: Stage

        var errorDescription: String? {
            "Gemma 4 MTP warm engine \(stage.rawValue) did not complete within "
                + "\(timeoutSeconds) seconds; worker startup refused."
        }
    }

    static func wait(
        drainAlreadyCompleted: Bool,
        drained: DispatchSemaphore,
        stopped: DispatchSemaphore,
        deadline: DispatchTime
    ) -> State {
        var state: State = drainAlreadyCompleted ? .waitingForShutdown : .waitingForDrain
        if state == .waitingForDrain {
            state = transition(
                state,
                on: drained.wait(timeout: deadline) == .success
                    ? .completed : .deadlineExpired)
        }
        if state == .waitingForShutdown {
            state = transition(
                state,
                on: stopped.wait(timeout: deadline) == .success
                    ? .completed : .deadlineExpired)
        }
        return state
    }
}

public struct DrafterRoPETable: @unchecked Sendable {
    public let cos: MLXArray
    public let sin: MLXArray
    public let dims: Int
    public let startPosition: Int
    public let windowAhead: Int
    public let base: Float

    public init(
        cos: MLXArray, sin: MLXArray, dims: Int,
        startPosition: Int, windowAhead: Int, base: Float
    ) {
        self.cos = cos
        self.sin = sin
        self.dims = dims
        self.startPosition = startPosition
        self.windowAhead = windowAhead
        self.base = base
    }
}

public final class Gemma4CBv2MTPDrafter: CBv2MTPDrafter {

    public static let defaultRoPEWindowAhead = 16

    private final class Prepared: CBv2MTPPreparedCapture {
        let sharedKV: Gemma4SharedKV
        let masks: Gemma4DrafterMasks
        let positionOffset: Gemma4.PositionOffset
        let ropeTable: DrafterRoPETable?
        let mirror: Gemma4DrafterMirrorAttention.Context?

        init(
            sharedKV: Gemma4SharedKV, masks: Gemma4DrafterMasks,
            positionOffset: Gemma4.PositionOffset,
            ropeTable: DrafterRoPETable?,
            mirror: Gemma4DrafterMirrorAttention.Context? = nil
        ) {
            self.sharedKV = sharedKV
            self.masks = masks
            self.positionOffset = positionOffset
            self.ropeTable = ropeTable
            self.mirror = mirror
        }
    }

    private let drafter: Gemma4AssistantDraftModel
    private let target: any Gemma4MTPTarget

    private var cachedRoPETable: DrafterRoPETable?

    public init(drafter: Gemma4AssistantDraftModel, target: any Gemma4MTPTarget) throws {
        try drafter.bind(target: target)
        self.drafter = drafter
        self.target = target
        if let model = target as? Gemma4TextModel {
            try warmSpeculativeCohort(model: model)
        }
    }

    private func warmSpeculativeCohort(model: Gemma4TextModel) throws {
        let warmupEnabled: Bool = {
            guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_WARM"]
            else { return true }
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }()
        let gateApplies = model.expertQMMGeometryEligible
        let shouldWarm = gateApplies
            ? CBv2MTPDeviceGate.beginWarmup(warmupEnabled: warmupEnabled)
            : warmupEnabled && CBv2MTPSpeculationPolicy.speculationEnabled
        defer {
            if gateApplies {
                if CBv2MTPDeviceGate.mode == .auto,
                    !CBv2MTPDeviceGate.automaticMeasurementFinished
                {
                    CBv2MTPDeviceGate.failAutomaticMeasurement("measurement incomplete")
                }
                if let line = CBv2MTPDeviceGate.takeStderrLine() {
                    FileHandle.standardError.write(Data(line.utf8))
                }
            }
        }
        guard shouldWarm, CBv2MTPSpeculationPolicy.speculationEnabled else { return }
        let batch = 8
        let seedCount = 1024
        let depth = CBv2MTPSpeculationPolicy.draftDepth
        if gateApplies, CBv2MTPDeviceGate.mode == .auto,
            (depth != 2 || !CBv2MTPWideVerifyContext.enabled)
        {
            CBv2MTPDeviceGate.failAutomaticMeasurement("depth-2 wide path unavailable")
            return
        }
        let warmTokens = 4 * (depth + 1)
        let measuring = gateApplies && CBv2MTPDeviceGate.mode == .auto
        do {
            let caches = try model.newCacheV2 { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 4 << 30))
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds,
                backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: batch,
                    maxBatchedTokensPerStep: Swift.max(2048, batch * seedCount),
                    prefillChunkSize: Swift.max(512, seedCount),
                    maxWaiting: batch,
                    enablePrefixCache: false),
                loopConfig: CBv2EngineLoopConfig(
                    shutdownTimeout:
                        TimeInterval(Gemma4MTPWarmRetirement.engineShutdownTimeoutSeconds)),
                mtpDrafter: self,
                mtpConfig: CBv2MTPConfig(
                    enabled: true,
                    maxDraftTokens: depth,
                    maxSpeculativeBatch: batch,
                    fixedDraftTokens: depth,
                    verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 32))
            let drained = DispatchSemaphore(value: 0)
            let progress = WarmProgress(slots: batch, target: warmTokens)
            let consumer = Task {
                await withTaskGroup(of: Void.self) { group in
                    for slot in 0 ..< batch {
                        var seeds = Array(repeating: 2, count: seedCount)
                        for i in stride(from: 1, to: seedCount, by: 1) {
                            seeds[i] = 1000 + ((i * 7919 + slot * 104729) % 20000)
                        }
                        guard let stream = try? engine.submit(
                            CBv2Request(
                                id: CBv2RequestID(UInt64(slot)),
                                promptTokens: seeds,
                                sampling: CBv2SamplingParams(temperature: 0),
                                maxTokens: 4096,
                                stopTokens: [],
                                prefixCacheEnabled: false))
                        else { continue }
                        group.addTask {
                            for await event in stream {
                                if case .delta(_, let ids, _) = event {
                                    let finished = measuring
                                        ? CBv2MTPDeviceGate.automaticMeasurementFinished
                                        : progress.record(slot: slot, count: ids.count)
                                    if finished {
                                        for other in 0 ..< batch {
                                            engine.cancel(CBv2RequestID(UInt64(other)))
                                        }
                                    }
                                }
                                if case .finished = event { break }
                            }
                        }
                    }
                    await group.waitForAll()
                }
                drained.signal()
            }
            let retirementDeadline = DispatchTime.now()
                + .seconds(Gemma4MTPWarmRetirement.timeoutSeconds)
            let drainAlreadyCompleted = drained.wait(timeout: .now() + 15) == .success
            if !drainAlreadyCompleted {
                consumer.cancel()
                for slot in 0 ..< batch { engine.cancel(CBv2RequestID(UInt64(slot))) }
            }
            let stopped = DispatchSemaphore(value: 0)
            Task {
                await engine.shutdownSynchronously()
                stopped.signal()
            }
            let retirement = Gemma4MTPWarmRetirement.wait(
                drainAlreadyCompleted: drainAlreadyCompleted,
                drained: drained, stopped: stopped, deadline: retirementDeadline)
            if case .failed(let stage) = retirement {
                let failure = Gemma4MTPWarmRetirement.Failure(stage: stage)
                if measuring {
                    CBv2MTPDeviceGate.failAutomaticMeasurement(
                        "warm engine \(stage.rawValue) timeout")
                }
                throw failure
            }
            Memory.clearCache()
            CBv2EngageMark.once("mtp-warm-cohort")
        } catch let failure as Gemma4MTPWarmRetirement.Failure {
            throw failure
        } catch {
            if measuring {
                CBv2MTPDeviceGate.failAutomaticMeasurement("warm-up error")
            }
        }
    }

    // MARK: - CBv2MTPDrafter

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    public var currentRoPETable: DrafterRoPETable? { cachedRoPETable }

    public func prepare(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture?
    ) -> CBv2MTPPreparedCapture {
        guard let cohort else { return prepare(rows: rows) }
        precondition(
            rows.count == cohort.anchors.dim(0),
            "Gemma4CBv2MTPDrafter.prepare: \(rows.count) rows but \(cohort.anchors.dim(0)) anchors")
        return prepareDevice(rows: rows, cohort: cohort)
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        precondition(!rows.isEmpty, "Gemma4CBv2MTPDrafter.prepare: rows must be non-empty")
        let positionOffset = Gemma4.PositionOffset.batch(
            MLXArray(rows.map { Int32($0.anchor) }))

        let slidingWindow = drafter.config.textConfig.slidingWindow
        let anchorMin = rows.map(\.anchor).min() ?? 0
        let ropeTable = Self.materializeDrafterRoPETable(
            drafterConfig: drafter.config.textConfig,
            startPosition: anchorMin,
            windowAhead: Self.defaultRoPEWindowAhead)
        cachedRoPETable = ropeTable

        if rows.count == 1 {
            let row = rows[0]
            let slidingMask = Self.slidingMask(
                rows: rows, tMax: row.slidingKeys.dim(2), window: slidingWindow,
                dtype: row.slidingKeys.dtype)
            return Prepared(
                sharedKV: Gemma4SharedKV(
                    fullAttention: (row.fullKeys, row.fullValues),
                    slidingAttention: (row.slidingKeys, row.slidingValues)),
                masks: Gemma4DrafterMasks(
                    full: .none,
                    sliding: slidingMask.map { .array($0) } ?? .none),
                positionOffset: positionOffset,
                ropeTable: ropeTable)
        }

        let (fullKV, fullMask) = Self.padAndMask(
            keys: rows.map(\.fullKeys), values: rows.map(\.fullValues))
        let (slidingKV, slidingMask) = Self.padAndMask(
            keys: rows.map(\.slidingKeys), values: rows.map(\.slidingValues))
        let absoluteSlidingMask = Self.slidingMask(
            rows: rows, tMax: slidingKV.0.dim(2), window: slidingWindow,
            dtype: slidingKV.0.dtype)
        return Prepared(
            sharedKV: Gemma4SharedKV(fullAttention: fullKV, slidingAttention: slidingKV),
            masks: Gemma4DrafterMasks(
                full: .array(fullMask),
                sliding: .array(absoluteSlidingMask ?? slidingMask)),
            positionOffset: positionOffset,
            ropeTable: ropeTable)
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let prepared = prepared as? Prepared else {
            preconditionFailure(
                "Gemma4CBv2MTPDrafter.draftStep: prepared capture "
                    + "\(type(of: prepared)) was not built by prepare(rows:)")
        }
        let hidden = hidden.ndim == 2 ? hidden.expandedDimensions(axis: 1) : hidden
        let rotatedHidden: MLXArray
        if let table = prepared.ropeTable {
            rotatedHidden = Self.applyCachedDrafterRoPE(
                hidden: hidden, table: table, positionOffset: prepared.positionOffset)
        } else {
            rotatedHidden = hidden
        }
        let inputsEmbeds = concatenated(
            [target.embedTokensForDrafter(tokens), rotatedHidden], axis: -1)
        let (newHidden, logits) = Gemma4DrafterMirrorAttention.with(prepared.mirror) {
            drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: prepared.sharedKV,
                positionOffset: prepared.positionOffset,
                masks: prepared.masks)
        }
        let next = logits.squeezed(axis: 1).argMax(axis: -1).asType(.int32)
        return (next, newHidden)
    }

    private final class WarmProgress: @unchecked Sendable {
        private let lock = NSLock()
        private var counts: [Int]
        private let target: Int
        private var fired = false
        init(slots: Int, target: Int) {
            counts = Array(repeating: 0, count: slots)
            self.target = target
        }
        func record(slot: Int, count: Int) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            counts[slot] += count
            guard !fired, counts.allSatisfy({ $0 >= target }) else { return false }
            fired = true
            return true
        }
    }

    // MARK: - Device geometry (chained rounds)

    private func prepareDevice(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture
    ) -> CBv2MTPPreparedCapture {
        cachedRoPETable = nil
        let fullKV = cohort.pooledFull.map { ($0.keys, $0.values) }
            ?? Self.padStack(rows.map(\.fullKeys), rows.map(\.fullValues))
        let fullMask = Self.lengthMask(
            tMax: fullKV.0.dim(2), lengths: cohort.fullLengths, dtype: fullKV.0.dtype)

        let slidingWindow = drafter.config.textConfig.slidingWindow
        let slidingKV: (MLXArray, MLXArray)
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        let mirror = Gemma4DrafterMirrorAttention.context(
            rows: rows, cohort: cohort, window: slidingWindow)
        if mirror != nil {
            let empty = MLXArray.zeros(
                [1, drafter.config.textConfig.numKeyValueHeads, 0, drafter.config.textConfig.headDim],
                dtype: fullKV.0.dtype)
            slidingKV = (empty, empty)
            slidingMask = .none
        } else {
            slidingKV = Self.padStack(rows.map(\.slidingKeys), rows.map(\.slidingValues))
            slidingMask = .array(
                Self.slidingMaskDevice(
                    tMax: slidingKV.0.dim(2),
                    lengths: MLXArray(rows.map { Int32($0.slidingKeys.dim(2)) }),
                    starts: cohort.slidingStarts, anchors: cohort.anchors,
                    window: slidingWindow, dtype: slidingKV.0.dtype))
        }
        return Prepared(
            sharedKV: Gemma4SharedKV(fullAttention: fullKV, slidingAttention: slidingKV),
            masks: Gemma4DrafterMasks(full: .array(fullMask), sliding: slidingMask),
            positionOffset: .batch(cohort.anchors),
            ropeTable: nil,
            mirror: mirror)
    }

    private static func lengthMask(tMax: Int, lengths: MLXArray, dtype: DType) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let valid = positions .< lengths.reshaped([-1, 1, 1, 1])
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
    }

    private static func slidingMaskDevice(
        tMax: Int, lengths: MLXArray, starts: MLXArray, anchors: MLXArray,
        window: Int, dtype: DType
    ) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let insideLength = positions .< lengths.reshaped([-1, 1, 1, 1])
        let insideWindow =
            (positions + starts.reshaped([-1, 1, 1, 1])) .> (anchors - Int32(window)).reshaped([-1, 1, 1, 1])
        let valid = MLX.logicalAnd(insideLength, insideWindow)
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
    }

    // MARK: - Padding + masks (B > 1)

    private static func padAndMask(
        keys: [MLXArray], values: [MLXArray]
    ) -> (kv: (MLXArray, MLXArray), mask: MLXArray) {
        let lengths = keys.map { $0.dim(2) }
        let tMax = lengths.max() ?? 0
        precondition(tMax > 0, "Gemma4CBv2MTPDrafter: empty KV capture")

        let padded = padStack(keys, values)

        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let valid = positions .< MLXArray(lengths.map { Int32($0) }).reshaped([-1, 1, 1, 1])
        let dtype = keys[0].dtype
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return (padded, MLX.where(valid, zero, negInf))
    }

    static func slidingMask(
        rows: [CBv2MTPRowCapture], tMax: Int, window: Int, dtype: DType
    ) -> MLXArray? {
        let lengths = rows.map { $0.slidingKeys.dim(2) }
        let needsMask = rows.enumerated().contains { index, row in
            lengths[index] < tMax || row.slidingStart <= row.anchor - window
        }
        guard needsMask else { return nil }

        let positions = MLXArray(Int32(0) ..< Int32(tMax)).reshaped([1, 1, 1, tMax])
        let lengthsArray = MLXArray(lengths.map(Int32.init)).reshaped([-1, 1, 1, 1])
        let starts = MLXArray(rows.map { Int32($0.slidingStart) }).reshaped([-1, 1, 1, 1])
        let lowerBounds = MLXArray(rows.map { Int32($0.anchor - window) })
            .reshaped([-1, 1, 1, 1])
        let insideLength = positions .< lengthsArray
        let insideWindow = (positions + starts) .> lowerBounds
        let valid = MLX.logicalAnd(insideLength, insideWindow)
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        return MLX.where(valid, zero, negInf)
    }

    private static func padStack(_ keys: [MLXArray], _ values: [MLXArray]) -> (MLXArray, MLXArray) {
        let tMax = keys.map { $0.dim(2) }.max() ?? 0
        return (padStack(keys, to: tMax), padStack(values, to: tMax))
    }

    private static func padStack(_ rows: [MLXArray], to tMax: Int) -> MLXArray {
        if rows.allSatisfy({ $0.dim(2) == tMax }) {
            return concatenated(rows, axis: 0)
        }
        let out = MLXArray.zeros(
            [rows.count, rows[0].dim(1), tMax, rows[0].dim(3)], dtype: rows[0].dtype)
        for (i, row) in rows.enumerated() {
            out[i ..< (i + 1), 0..., 0 ..< row.dim(2), 0...] = row
        }
        return out
    }

    // MARK: - Drafter RoPE table (per-round precompute)

    static func materializeDrafterRoPETable(
        drafterConfig: Gemma4TextConfiguration,
        startPosition: Int,
        windowAhead: Int
    ) -> DrafterRoPETable? {
        precondition(
            windowAhead > 0, "windowAhead must be positive")
        let fullDim = drafterConfig.headDim
        let factor = drafterConfig.globalPartialRotaryFactor
        let rotatedDims = 2 * Int((factor * Float(fullDim) / 2.0).rounded(.down))
        guard rotatedDims >= 2 else { return nil }
        let halfDim = rotatedDims / 2
        let base = drafterConfig.fullRopeTheta

        let indices = MLXArray(stride(
            from: 0, to: halfDim, by: 1
        )).asType(.float32)
        let exponent = indices * (-2.0 / Float(rotatedDims))
        let freqs = MLX.pow(MLXArray(base), exponent)  // [halfDim]

        let positions = MLXArray(
            stride(from: startPosition, to: startPosition + windowAhead, by: 1)
        ).asType(.float32)
            .reshaped([windowAhead, 1])               // [windowAhead, 1]

        let angles = positions * freqs
        let cos = MLX.cos(angles)
        let sin = MLX.sin(angles)
        return DrafterRoPETable(
            cos: cos, sin: sin,
            dims: rotatedDims,
            startPosition: startPosition,
            windowAhead: windowAhead,
            base: base)
    }

    static func applyCachedDrafterRoPE(
        hidden: MLXArray, table: DrafterRoPETable,
        positionOffset: Gemma4.PositionOffset
    ) -> MLXArray {
        let rotaryPrefix = table.dims
        let featureDim = hidden.dim(-1)
        guard rotaryPrefix > 0, rotaryPrefix <= featureDim else {
            return hidden
        }
        let halfDim = rotaryPrefix / 2

        let perRow: [Int32]
        switch positionOffset {
        case .scalar(let v):
            perRow = [Int32(v)]
        case .batch(let arr):
            perRow = arr.asArray(Int32.self)
        case .graphArray(let arr):
            perRow = arr.asArray(Int32.self)
        }

        let steps = perRow.map { Int32($0) - Int32(table.startPosition) }
        let inRange = steps.allSatisfy { $0 >= 0 && $0 < Int32(table.windowAhead) }
        guard inRange else { return hidden }

        let indices = MLXArray(steps, [perRow.count])
        let leadShape = Array(hidden.shape.dropLast())  // [B] or [B, 1]
        let broadcastShape =
            [perRow.count] + Array(repeating: 1, count: max(0, leadShape.count - 1))
            + [halfDim]
        let cosB = table.cos[indices].reshaped(broadcastShape)  // [B, (1,) halfDim]
        let sinB = table.sin[indices].reshaped(broadcastShape)

        let prefix = hidden[.ellipsis, 0 ..< rotaryPrefix]
        let pairs = prefix.reshaped(leadShape + [halfDim, 2])
        let a = pairs[.ellipsis, 0]  // [..., halfDim]
        let b = pairs[.ellipsis, 1]
        let aRot = a * cosB - b * sinB
        let bRot = a * sinB + b * cosB
        let rotatedPrefix = MLX.stacked([aRot, bRot], axis: -1)
            .reshaped(leadShape + [rotaryPrefix])
        let tail = hidden[.ellipsis, rotaryPrefix ..< featureDim]
        return MLX.concatenated([rotatedPrefix, tail], axis: -1)
    }
}

enum Gemma4DrafterMirrorAttention {
    struct Context {
        let mirrors: [MLXArray]
        let slotBases: MLXArray
        let fence: MLXArray
        let window: Int
    }

    nonisolated(unsafe) private(set) static var current: Context?

    static func context(
        rows: [CBv2MTPRowCapture], cohort: CBv2MTPCohortCapture, window: Int
    ) -> Context? {
        guard rows.count == 8,
            let slotBases = cohort.slidingMirrorSlotBases,
            let fence = cohort.slidingMirrorFence
        else { return nil }
        let mirrors = rows.compactMap(\.slidingMirror)
        guard mirrors.count == rows.count else { return nil }
        return Context(
            mirrors: mirrors, slotBases: slotBases.asType(.uint32), fence: fence, window: window)
    }

    static func with<T>(_ context: Context?, _ body: () -> T) -> T {
        let previous = current
        current = context
        defer { current = previous }
        return body()
    }

    static func attend(queries: MLXArray, isSliding: Bool) -> MLXArray? {
        guard isSliding, let current else { return nil }
        guard let output = CBv2MTPMirrorAttention.attend(
                queries: queries, mirrors: current.mirrors, slotBases: current.slotBases,
                fence: current.fence, window: current.window)
        else {
            preconditionFailure(
                "Gemma4DrafterMirrorAttention: the q4 mirror kernel refused queries "
                    + "\(queries.shape) \(queries.dtype)")
        }
        return output
    }
}
