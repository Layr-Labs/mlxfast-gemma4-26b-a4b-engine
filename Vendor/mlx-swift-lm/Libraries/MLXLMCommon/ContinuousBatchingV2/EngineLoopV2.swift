// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the execution loop.
//
// Single engine thread (serial GCD queue, the EngineCore idiom): the loop
// does graph-build + `asyncEval` ONLY. Tokenization/detokenization state
// machines are pluggable (WS-E), prefix-cache donation and SSD I/O live
// elsewhere. Decode is rectangular [B, 1]; prefill runs per-request
// [1, chunk] under the shared token budget. There is no left padding, no
// shared frontier, and no batch-wide trim anywhere in this file.
//
// Chained async decode (SGLang-MLX pattern, report 09 §7): step N+1's [B, 1]
// forward is built ON TOP of step N's still-lazy sampled-token array and
// `asyncEval`ed BEFORE the loop blocks on step N's tokens for stop detection.
// Tokens are therefore inspected one step late; a finished request wastes at
// most one slot-step, and its extra token + KV tail are rolled back
// (`CBv2SequenceKV.rollback(1)`). The chain breaks on ANY membership change
// (prefill completion into a different set, finish, cancel, join, pause).

import Foundation
import MLX

// MARK: - Model interface (WS-F adapters / WS-G fixtures conform)

public protocol CBv2SteppableModel: AnyObject {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray

    func compactDecodeEvaluationRoots(
        forwardOutput: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> [MLXArray]?
}

extension CBv2SteppableModel {
    public func compactDecodeEvaluationRoots(
        forwardOutput: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> [MLXArray]? {
        nil
    }
}

@inline(__always)
internal func resolveCBv2CompactDecodeRootsEnabled(_ raw: String?) -> Bool {
    guard let raw else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}

private let cbv2CompactDecodeRootsEnabled = resolveCBv2CompactDecodeRootsEnabled(
    ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_COMPACT_DECODE_ROOTS"])

/// A prompt chunk's ring and mirror writes are not inputs of its logits, so
/// the sampled tokens complete while those writes are still queued and the
/// next step's forward waits behind them. Fencing the outputs the finalize
/// waits on keeps that tail inside the prefill step's own wait.
/// `DARKBLOOM_CBV2_PREFILL_WRITE_TAIL_FENCE=0` disables it.
let cbv2PrefillWriteTailFenceEnabled = resolveCBv2CompactDecodeRootsEnabled(
    ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PREFILL_WRITE_TAIL_FENCE"])

public protocol CBv2LayerCacheProvider: AnyObject {
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]
    var supportsMultimodalSpans: Bool { get }
    var supportsPackedPrefill: Bool { get }
    var supportsPackedMultimodalSpans: Bool { get }
}

extension CBv2LayerCacheProvider {
    public var supportsMultimodalSpans: Bool { false }
    public var supportsPackedPrefill: Bool { false }
    public var supportsPackedMultimodalSpans: Bool { false }
}

// MARK: - Sampler interface (WS-E's CBv2DefaultSampler is the production impl)

public protocol CBv2StepSampler: AnyObject {
    var supportsTokenConstraints: Bool { get }

    func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray

    func takeStepLogprobs() -> CBv2StepLogprobs?

    func requestDidFinish(_ id: CBv2RequestID)

    func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID])

    func tokenConstraintFailure(for id: CBv2RequestID) -> String?
}

extension CBv2StepSampler {
    public var supportsTokenConstraints: Bool { false }
    public func takeStepLogprobs() -> CBv2StepLogprobs? { nil }
    public func requestDidFinish(_ id: CBv2RequestID) {}
    public func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID]) {}
    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? { nil }
}

public final class CBv2GreedySampler: CBv2StepSampler {
    private let constraintSampler = CBv2TokenConstraintSampler()
    private var configuredIDs: [CBv2RequestID] = []

    public var supportsTokenConstraints: Bool { true }

    public init() {}
    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        if requestIDs != configuredIDs {
            constraintSampler.configure(rowContext())
            configuredIDs = requestIDs
        }
        return argMax(
            constraintSampler.mask(logits, requestIDs: requestIDs),
            axis: -1
        ).asType(.int32)
    }

    public func requestDidFinish(_ id: CBv2RequestID) {
        constraintSampler.requestDidFinish(id)
        if configuredIDs.contains(id) {
            configuredIDs = []
        }
    }

    public func confirmSampledTokens(
        _ tokens: [Int], requestIDs: [CBv2RequestID]
    ) {
        constraintSampler.confirm(tokens: tokens, requestIDs: requestIDs)
    }

    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
        constraintSampler.failure(for: id)
    }
}

// MARK: - Detokenizer interface (WS-E conforms; null stub until then)

public protocol CBv2IncrementalDetokenizer: AnyObject {
    func push(_ tokens: [Int]) -> String
    var matchedStopString: Bool { get }
    func flush() -> String
}

public protocol CBv2DetokenizerFactory: AnyObject {
    func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer
}

public final class CBv2NullDetokenizerFactory: CBv2DetokenizerFactory {
    final class NullDetokenizer: CBv2IncrementalDetokenizer {
        let matchedStopString = false
        func push(_ tokens: [Int]) -> String { "" }
        func flush() -> String { "" }
    }
    public init() {}
    public func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        NullDetokenizer()
    }
}

// MARK: - Loop configuration

public struct CBv2EngineLoopConfig: Sendable {
    public static let defaultShutdownTimeout: TimeInterval = {
        guard let cRaw = getenv("DARKBLOOM_CBV2_SHUTDOWN_TIMEOUT"),
            let value = TimeInterval(String(cString: cRaw).trimmingCharacters(in: .whitespaces)),
            value > 10
        else { return 10 }
        return value
    }()

    public static let defaultSafetyFloorTPS: Double = {
        guard let cRaw = getenv("DARKBLOOM_CBV2_SAFETY_FLOOR_TPS"),
            let value = Double(String(cString: cRaw).trimmingCharacters(in: .whitespaces)),
            value > 0, value < 5
        else { return 5 }
        return value
    }()

    public static let defaultStepTimeout: TimeInterval = {
        guard let cRaw = getenv("DARKBLOOM_CBV2_STEP_TIMEOUT"),
            let value = TimeInterval(String(cString: cRaw).trimmingCharacters(in: .whitespaces)),
            value > 30
        else { return 30 }
        return value
    }()

    public var requestTimeout: TimeInterval
    public var useLegacyRequestTimeout: Bool
    public var admissionLease: TimeInterval
    public var prefillProgressLease: TimeInterval
    public var decodeProgressLease: TimeInterval
    public var backpressureLease: TimeInterval
    public var safetyCeilingDecodeFloorTPS: Double
    public var clock: CBv2Clock
    public var stepTimeout: TimeInterval
    public var watchdogInterval: TimeInterval
    public var idleRecheckInterval: TimeInterval
    public var eventBufferCapacity: Int
    public var shutdownTimeout: TimeInterval

    public init(
        requestTimeout: TimeInterval = 120,
        stepTimeout: TimeInterval = CBv2EngineLoopConfig.defaultStepTimeout,
        watchdogInterval: TimeInterval = 0.25, idleRecheckInterval: TimeInterval = 0.001,
        eventBufferCapacity: Int = 256,
        shutdownTimeout: TimeInterval = CBv2EngineLoopConfig.defaultShutdownTimeout,
        useLegacyRequestTimeout: Bool = false,
        admissionLease: TimeInterval = 120,
        prefillProgressLease: TimeInterval = 120,
        decodeProgressLease: TimeInterval = 120,
        backpressureLease: TimeInterval = 120,
        safetyCeilingDecodeFloorTPS: Double = CBv2EngineLoopConfig.defaultSafetyFloorTPS,
        clock: CBv2Clock = .continuous
    ) {
        self.requestTimeout = requestTimeout
        self.stepTimeout = stepTimeout
        self.watchdogInterval = watchdogInterval
        self.idleRecheckInterval = idleRecheckInterval
        self.eventBufferCapacity = eventBufferCapacity
        self.shutdownTimeout = shutdownTimeout
        self.useLegacyRequestTimeout = useLegacyRequestTimeout
        self.admissionLease = admissionLease
        self.prefillProgressLease = prefillProgressLease
        self.decodeProgressLease = decodeProgressLease
        self.backpressureLease = backpressureLease
        self.safetyCeilingDecodeFloorTPS = safetyCeilingDecodeFloorTPS
        self.clock = clock
    }
}

// MARK: - In-flight step

final class CBv2InFlightStep {
    let participants: Set<CBv2RequestID>
    let sampledRows: [CBv2RequestID]
    let sampledTokens: MLXArray?
    let evalTargets: [MLXArray]
    var discard: Set<CBv2RequestID> = []
    var logprobSegments: [CBv2StepLogprobs] = []
    let wallStartedNanos: UInt64
    var mtpMeasurement: CBv2MTPStepMeasurement?
    fileprivate var deferredReleases:
        [(
            id: CBv2RequestID, state: [CBv2SequenceKV?], rollbackOne: Bool,
            donation: CBv2DonationIntent?
        )] = []
    var mtpRound: CBv2MTPRoundInFlight?

    init(
        participants: Set<CBv2RequestID>, sampledRows: [CBv2RequestID],
        sampledTokens: MLXArray?, evalTargets: [MLXArray],
        wallStartedNanos: UInt64
    ) {
        self.participants = participants
        self.sampledRows = sampledRows
        self.sampledTokens = sampledTokens
        self.evalTargets = evalTargets
        self.wallStartedNanos = wallStartedNanos
    }
}

// MARK: - Prefix adoption handoff

struct CBv2PrefixAdoption: @unchecked Sendable {
    let requestID: CBv2RequestID
    let tokens: [Int]
    let matched: Int
    let plan: CBv2PrefixReusePlan
    let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    let cacheSalt: String?
}

struct CBv2PrefixLookup: @unchecked Sendable {
    let adoption: CBv2PrefixAdoption?
    let outcome: CBv2PrefixCacheOutcome
    let matchedTokens: Int
}

struct CBv2PrefixUsage {
    var outcome: CBv2PrefixCacheOutcome
    var matchedTokens: Int
    var prefillTokensSaved: Int
    var strategy: CBv2PrefixReuseStrategy?
    var replayTokens: Int
    var boundarySplits: Int

    static let disabled = CBv2PrefixUsage(
        outcome: .disabled,
        matchedTokens: 0,
        prefillTokensSaved: 0,
        strategy: nil,
        replayTokens: 0,
        boundarySplits: 0)
}

struct CBv2DonationIntent {
    let requestID: CBv2RequestID
    let tokens: [Int]
    let cacheSalt: String?
}

private struct CBv2Handoff<Value>: @unchecked Sendable {
    let value: Value
}

private final class CBv2DrainWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?

    init(_ continuation: CheckedContinuation<Void, Never>) {
        self.continuation = continuation
    }

    @discardableResult
    func resume() -> Bool {
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume()
        return c != nil
    }
}

// MARK: - EngineLoopV2

public final class EngineLoopV2: @unchecked Sendable {
    let scheduler: SchedulerV2
    let capacity: CBv2StepCapacity?
    let backend: CBv2KVBackend
    let cacheProvider: CBv2LayerCacheProvider
    let model: CBv2SteppableModel
    let sampler: CBv2StepSampler
    let detokenizerFactory: CBv2DetokenizerFactory
    let layerKinds: [CBv2LayerKind]
    let prefixCache: CBv2PrefixCache?
    let mtp: CBv2MTPRoundDriver?
    let config: CBv2EngineLoopConfig
    let gauges: CBv2EngineGauges

    private let engineQueue = DispatchQueue(
        label: "com.eigen.cbv2.engine", qos: .userInitiated)
    private let watchdogQueue = DispatchQueue(
        label: "com.eigen.cbv2.watchdog", qos: .utility)
    private let donationQueue = DispatchQueue(
        label: "com.eigen.cbv2.donation", qos: .utility)
    let detokQueue = DispatchQueue(
        label: "com.eigen.cbv2.detok", qos: .userInitiated)

    private let stateLock = NSLock()
    private var streams: [CBv2RequestID: CBv2OutputStream] = [:]
    private var pendingCancels: Set<CBv2RequestID> = []
    private var stepStartedNanos: UInt64 = 0
    private var wedgeReported = false
    private var _healthy = true
    private var usageSnapshots: [CBv2RequestID: CBv2Usage] = [:]
    private var prefillFrontierCaptureHook: (@Sendable (CBv2RequestID, MLXArray) -> Void)?

    var detokenizers: [CBv2RequestID: CBv2IncrementalDetokenizer] = [:]
    var kvStates: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
    private var prefixHitTokens: [CBv2RequestID: Int] = [:]
    private var prefixUsageByID: [CBv2RequestID: CBv2PrefixUsage] = [:]
    private var pendingDonationReleaseCount = 0
    var multimodalByID: [CBv2RequestID: CBv2ResolvedMultimodal] = [:]
    var leasesByID: [CBv2RequestID: CBv2RequestLeaseState] = [:]
    var leasePreemptionsPendingFinalize: [CBv2RequestID] = []
    private var inFlight: CBv2InFlightStep?
    private var running = false
    private var draining = false

    // MARK: ADMIT-COALESCE-001 (bounded admission coalescing)

    static let admitCoalesceWindowCapMS = 25
    static let admitCoalesceWindowMS: Int = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_ADMIT_COALESCE_MS"],
            let value = Int(raw), value >= 0
        else { return 3 }
        return Swift.min(value, admitCoalesceWindowCapMS)
    }()

    private var admitBatchFirstEnqueue: ContinuousClock.Instant?

    private var drainWaiters: [CBv2DrainWaiter] = []
    var eagerCompositionStale = false

    public private(set) var stepCount = 0
    public private(set) var chainedStepCount = 0
    public private(set) var preemptionCount = 0
    public private(set) var packedPrefillRowsExecuted = 0
    public private(set) var packedPrefillGroupsExecuted = 0
    var packedPrefillSupported: Bool {
        cacheProvider.supportsPackedPrefill
            && (model as? CBv2PackedPrefillSteppableModel)?.supportsPackedPrefill == true
    }

    func packedPrefillActivity() -> CBv2PackedPrefillActivity {
        engineQueue.sync {
            CBv2PackedPrefillActivity(
                isSupported: packedPrefillSupported,
                rowsExecuted: packedPrefillRowsExecuted,
                groupsExecuted: packedPrefillGroupsExecuted)
        }
    }

    private(set) var teacherForcedPrefillChunks = 0
    private(set) var teacherForcedDecodeForwards = 0

    func teacherForcedScoringActivity() -> CBv2TeacherForcedScoringActivity {
        engineQueue.sync {
            CBv2TeacherForcedScoringActivity(
                prefillChunksExecuted: teacherForcedPrefillChunks,
                decodeForwardsExecuted: teacherForcedDecodeForwards)
        }
    }

    private(set) var capacityRequeueCount = 0
    private var capacityRequeues: [CBv2RequestID: Int] = [:]
    static let maxCapacityRequeues = 64
    public internal(set) var offsetChainEvalSteps = 0
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)?
    var enqueueStartDelayForTesting: TimeInterval = 0

    public var isHealthy: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _healthy
    }

    init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler,
        detokenizerFactory: CBv2DetokenizerFactory,
        scheduler: SchedulerV2,
        capacity: CBv2StepCapacity?,
        prefixCache: CBv2PrefixCache? = nil,
        mtp: CBv2MTPRoundDriver? = nil,
        config: CBv2EngineLoopConfig,
        gauges: CBv2EngineGauges
    ) {
        self.model = model
        self.layerKinds = layerKinds
        self.backend = backend
        self.cacheProvider = cacheProvider
        self.sampler = sampler
        self.detokenizerFactory = detokenizerFactory
        self.scheduler = scheduler
        self.capacity = capacity
        self.prefixCache = prefixCache
        self.mtp = mtp
        self.config = config
        self.gauges = gauges
        if self.mtp?.isTargetOnlyPolicy == false {
            scheduler.speculationPlanner = { [unowned self] rec in
                self.mtpPlanSpeculation(for: rec)
            }
        }
    }

    // MARK: Lifecycle

    func start() {
        engineQueue.async { [self] in
            guard !running else { return }
            running = true
            startWatchdog()
            engineQueue.async { [weak self] in self?.engineStep() }
        }
    }

    func drain() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let waiter = CBv2DrainWaiter(c)
            watchdogQueue.asyncAfter(deadline: .now() + config.shutdownTimeout) { [weak self] in
                guard let self else {
                    waiter.resume()
                    return
                }
                if waiter.resume() {
                    self.forceFinishStreamsOnShutdownTimeout()
                }
            }
            engineQueue.async { [self] in
                guard running else {
                    waiter.resume()
                    return
                }
                draining = true
                for rec in scheduler.waiting {
                    finishRequest(rec.id, reason: .cancelled)
                }
                publishGauges()
                drainWaiters.append(waiter)
                completeDrainIfReady()
            }
        }
    }

    private func forceFinishStreamsOnShutdownTimeout() {
        stateLock.lock()
        let liveStreams = streams
        pendingCancels.formUnion(liveStreams.keys)
        stateLock.unlock()
        for (_, stream) in liveStreams {
            stream.finish(
                reason: .error(
                    "engine shutdown timed out after \(Int(config.shutdownTimeout))s"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
    }

    private func completeStop() {
        mtp?.removeAllRequestState()
        running = false
        draining = false
        stopWatchdog()
        if CBv2StepProfiler.enabled {
            FileHandle.standardError.write(
                Data(("[cbv2-step-profile]\n" + CBv2StepProfiler.summaryTable()).utf8))
        }
    }

    private func completeDrainIfReady() {
        guard draining, !scheduler.hasWork, inFlight == nil,
            pendingDonationReleaseCount == 0
        else { return }
        completeStop()
        let waiters = drainWaiters
        drainWaiters = []
        for waiter in waiters { waiter.resume() }
    }

    // MARK: Submission (from EngineV2)

    func register(stream: CBv2OutputStream) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard streams[stream.id] == nil else { return false }
        streams[stream.id] = stream
        return true
    }

    func unregister(_ id: CBv2RequestID) {
        stateLock.lock()
        streams.removeValue(forKey: id)
        stateLock.unlock()
    }

    func setPrefillFrontierCapture(
        _ capture: (@Sendable (CBv2RequestID, MLXArray) -> Void)?
    ) {
        stateLock.lock()
        prefillFrontierCaptureHook = capture
        stateLock.unlock()
    }

    func prefillFrontierCapture() -> (@Sendable (CBv2RequestID, MLXArray) -> Void)? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return prefillFrontierCaptureHook
    }

    func onEngineQueueSync<T>(_ body: () throws -> T) rethrows -> T {
        try engineQueue.sync(execute: body)
    }

    func enqueue(
        _ request: CBv2Request,
        prefixLookup: CBv2PrefixLookup = CBv2PrefixLookup(
            adoption: nil, outcome: .disabled, matchedTokens: 0),
        multimodal: CBv2ResolvedMultimodal? = nil
    ) {
        engineQueue.async { [self] in
            defer {
                gauges.endSubmit()
                publishGauges()
            }
            if enqueueStartDelayForTesting > 0 {
                Thread.sleep(forTimeInterval: enqueueStartDelayForTesting)
            }
            prefixUsageByID[request.id] = CBv2PrefixUsage(
                outcome: prefixLookup.outcome,
                matchedTokens: prefixLookup.matchedTokens,
                prefillTokensSaved: 0,
                strategy: nil,
                replayTokens: 0,
                boundarySplits: 0)
            guard running, !draining else {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .error("engine is shutting down"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
                return
            }
            if consumeEarlyCancel(request.id) {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .cancelled,
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
                return
            }
            do {
                try scheduler.enqueue(request)
                if scheduler.runningCount == 0, scheduler.waitingCount == 1 {
                    admitBatchFirstEnqueue = config.clock.now()
                }
                armLease(for: request)
                detokenizers[request.id] =
                    detokenizerFactory.makeDetokenizer(stopStrings: request.stopStrings)
                if let multimodal {
                    multimodalByID[request.id] = multimodal
                }
                if let adoption = prefixLookup.adoption {
                    applyAdoption(adoption, requestID: request.id)
                }
            } catch let error as CBv2SchedulerError {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .error("scheduler rejected request: \(error)"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
            } catch {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .error("token_budget_exhausted: request queue full"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
            }
        }
    }

    // MARK: Prefix-cache adoption (engine thread)

    private func applyAdoption(_ adoption: CBv2PrefixAdoption, requestID: CBv2RequestID) {
        defer {
            prefixCache?.endAdoption(
                requestID: adoption.requestID, tokens: adoption.tokens, matched: adoption.matched,
                cacheSalt: adoption.cacheSalt)
        }
        guard let rec = scheduler.record(for: requestID), kvStates[requestID] == nil else {
            markPrefixAdoptionFailed(requestID, outcome: .adoptionFailed)
            return
        }
        if let capacity {
            do {
                try capacity.reserve(
                    id: requestID,
                    additionalTokens: adoption.plan.capacityReservationTokens,
                    additionalBytes: adoption.plan.initialAdditionalCapacityBytes)
            } catch {
                markPrefixAdoptionFailed(requestID, outcome: .skippedCapacity)
                return  // capacity tight — full prefill with the usual backstops
            }
        }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                adopting: adoption.prefix,
                plan: adoption.plan,
                layerKinds: layerKinds,
                maxLength: maxLength)
            kvStates[requestID] = state
            rec.numComputedTokens = adoption.plan.replayStart
            rec.prefixReusePlan = adoption.plan
            prefixHitTokens[requestID] = adoption.plan.prefillTokensSaved
            prefixUsageByID[requestID]?.outcome = .hit
            prefixUsageByID[requestID]?.prefillTokensSaved = adoption.plan.prefillTokensSaved
            prefixUsageByID[requestID]?.strategy = adoption.plan.strategy
            prefixUsageByID[requestID]?.replayTokens = adoption.plan.replayTokens
        } catch {
            capacity?.unreserve(
                id: requestID,
                tokens: adoption.plan.capacityReservationTokens,
                bytes: adoption.plan.initialAdditionalCapacityBytes)
            let outcome: CBv2PrefixCacheOutcome
            if let kvError = error as? CBv2KVError, case .capacityExhausted = kvError {
                outcome = .skippedCapacity
            } else {
                outcome = .adoptionFailed
            }
            markPrefixAdoptionFailed(requestID, outcome: outcome)
        }
    }

    private func markPrefixAdoptionFailed(
        _ requestID: CBv2RequestID, outcome: CBv2PrefixCacheOutcome
    ) {
        prefixHitTokens.removeValue(forKey: requestID)
        prefixUsageByID[requestID]?.outcome = outcome
        prefixUsageByID[requestID]?.prefillTokensSaved = 0
        prefixUsageByID[requestID]?.strategy = nil
        prefixUsageByID[requestID]?.replayTokens = 0
        prefixUsageByID[requestID]?.boundarySplits = 0
    }

    private func invalidateAdoptedPrefix(_ requestID: CBv2RequestID) {
        guard prefixUsageByID[requestID]?.outcome == .hit else { return }
        markPrefixAdoptionFailed(requestID, outcome: .adoptionFailed)
    }

    private func releaseAbandonedAdoption(_ adoption: CBv2PrefixAdoption?) {
        guard let adoption else { return }
        prefixCache?.endAdoption(
            requestID: adoption.requestID, tokens: adoption.tokens, matched: adoption.matched,
            cacheSalt: adoption.cacheSalt)
    }

    // MARK: Cancellation & backpressure (any thread)

    func requestCancel(_ id: CBv2RequestID) {
        stateLock.lock()
        pendingCancels.insert(id)
        stateLock.unlock()
    }

    func setPaused(_ id: CBv2RequestID, _ paused: Bool) {
        engineQueue.async { [self] in
            let now = config.clock.now()
            if paused {
                scheduler.pause(id)
                if var lease = leasesByID[id] {
                    lease.markPaused(now: now)
                    leasesByID[id] = lease
                }
            } else {
                scheduler.resume(id)
                if var lease = leasesByID[id] {
                    lease.markResumed(now: now)
                    leasesByID[id] = lease
                }
            }
        }
    }

    // MARK: The step loop

    private func engineStep() {
        guard running else { return }
        let stepStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        markStepStarted()
        defer {
            markStepEnded()
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.step.wall", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
            }
        }

        let stepNow = config.clock.now()
        processCancellations()
        processLeaseExpiry(now: stepNow)

        // Chained MTP round: build round n+1 on round n's lazy device
        // hand-off, then finalize round n one round late.
        if let previous = inFlight, let mtp,
            let (verify, ids) = mtpChainedRoundCandidate(previous),
            let next = launchChainedMTPRound(
                previous: previous, verify: verify, ids: ids, driver: mtp)
        {
            inFlight = next
            chainedStepCount += 1
            stepCount += 1
            finalize(previous, now: stepNow)
            publishGauges()
            scheduleNextStep()
            return
        }

        if let previous = inFlight,
            previous.mtpRound == nil,
            previous.sampledTokens != nil,
            let ids = scheduler.chainCandidateIDs(),
            ids == previous.sampledRows,
            ids.allSatisfy({ kvStates[$0] != nil }),
            ids.allSatisfy({ scheduler.record(for: $0)?.request.tokenConstraint == nil }),
            !mtpWantsStep(ids: ids),
            capacity?.hasHeadroom(additionalTokens: ids.count) ?? true
        {
            beginMTPPlan()
            let plan = scheduler.plan()
            if isPureDecodePlan(plan, matching: ids) {
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.boundary", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
                }
                let measurement = mtpMeasurement(for: plan)
                let next = launchChainedDecode(plan, feeding: previous.sampledTokens!)
                attachMTPMeasurement(measurement, to: next, chained: true)
                if var previousMeasurement = previous.mtpMeasurement {
                    previousMeasurement.chained = true
                    previous.mtpMeasurement = previousMeasurement
                }
                inFlight = next
                chainedStepCount += 1
                stepCount += 1
                finalize(previous, now: stepNow)
                publishGauges()
                scheduleNextStep()
                return
            }
            scheduler.rollback(plan)
            leasePreemptionsPendingFinalize.append(contentsOf: plan.preemptions)
            for id in plan.preemptions {
                preemptionCount += 1
                invalidateAdoptedPrefix(id)
                mtp?.invalidateCarry(id)
                guard let state = kvStates.removeValue(forKey: id) else { continue }
                if previous.participants.contains(id) {
                    previous.deferredReleases.append(
                        (
                            id: id, state: state, rollbackOne: false, donation: nil
                        ))
                } else {
                    backend.release(state)
                }
            }
        }

        if let previous = inFlight {
            inFlight = nil
            finalize(previous, now: stepNow)
        }
        if !leasePreemptionsPendingFinalize.isEmpty {
            for id in leasePreemptionsPendingFinalize {
                if var lease = leasesByID[id] {
                    lease.markPreempted(now: stepNow)
                    leasesByID[id] = lease
                }
            }
            leasePreemptionsPendingFinalize.removeAll(keepingCapacity: true)
        }

        guard scheduler.hasWork else {
            (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            admitBatchFirstEnqueue = nil
            publishGauges()
            if draining {
                completeDrainIfReady()
                if !running { return }
            }
            scheduleIdleRecheck()
            return
        }

        if Self.admitCoalesceWindowMS > 0,
            let batchStart = admitBatchFirstEnqueue,
            scheduler.runningCount == 0,
            scheduler.waitingCount > 0,
            scheduler.waitingCount < scheduler.config.maxConcurrentRequests,
            stepNow - batchStart < .milliseconds(Self.admitCoalesceWindowMS)
        {
            publishGauges()
            scheduleIdleRecheck()
            return
        }
        admitBatchFirstEnqueue = nil

        beginMTPPlan()
        let waitingBeforePlan = Set(scheduler.waiting.map(\.id))
        let plan = scheduler.plan()
        handlePreemptions(plan.preemptions)
        guard !plan.assignments.isEmpty else {
            publishGauges()
            scheduleIdleRecheck()
            return
        }
        markAdmitted(plan, now: stepNow, readmittedFrom: waitingBeforePlan)
        let measurement = mtpMeasurement(for: plan)
        inFlight = mtpRoundNeeded(plan) ? executeMTPRound(plan) : executeMixed(plan)
        attachMTPMeasurement(measurement, to: inFlight, chained: false)
        stepCount += 1
        publishGauges()
        scheduleNextStep()
    }

    private func scheduleNextStep() {
        engineQueue.async { [weak self] in self?.engineStep() }
    }

    private func scheduleIdleRecheck() {
        engineQueue.asyncAfter(deadline: .now() + config.idleRecheckInterval) { [weak self] in
            self?.engineStep()
        }
    }

    // MARK: Step execution

    private func isPureDecodePlan(_ plan: CBv2StepPlan, matching ids: [CBv2RequestID]) -> Bool {
        guard plan.preemptions.isEmpty, plan.assignments.count == ids.count else { return false }
        for (i, assignment) in plan.assignments.enumerated() {
            guard assignment.numTokens == 1, assignment.id == ids[i] else { return false }
        }
        return true
    }

    func eagerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        if eagerCompositionStale {
            (cacheProvider as? CBv2CompositionInvalidating)?.invalidateBoundComposition()
            eagerCompositionStale = false
        }
        let caches = cacheProvider.layerCaches(rowStates: rowStates)
        return caches
    }

    /// Order a prompt chunk's K/V ring and mirror writes (`tail`) ahead of the
    /// outputs the step's finalize waits on. `depends` aliases its input, so
    /// values and every other op are unchanged.
    func fencedOnPrefillWriteTail(
        sampled: MLXArray?, outputs: [MLXArray], tail: [MLXArray]
    ) -> (MLXArray?, [MLXArray]) {
        guard cbv2PrefillWriteTailFenceEnabled, !tail.isEmpty else { return (sampled, outputs) }
        let fencedSampled = sampled.map { MLX.depends(input: $0, dependencies: tail) }
        let fencedOutputs =
            outputs.isEmpty ? outputs : MLX.depends(inputs: outputs, dependencies: tail)
        CBv2EngageMark.once("prefill-write-tail-fence")
        return (fencedSampled, fencedOutputs)
    }

    func eagerCacheInnerState(_ caches: [CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] }
    }

    func eagerDecodeEvaluationRoots(
        _ caches: [CBv2AttendingLayerCache], logitsRoot: MLXArray
    ) -> [MLXArray] {
        if cbv2CompactDecodeRootsEnabled,
            let compact = model.compactDecodeEvaluationRoots(
                forwardOutput: logitsRoot, caches: caches)
        {
            return compact
        }
        return eagerCacheInnerState(caches)
    }

    private func decodeLogits(
        rowStates: [[CBv2SequenceKV?]], tokens: MLXArray
    ) -> (logits: MLXArray, cacheInnerState: [MLXArray]) {
        let caches = eagerCaches(rowStates: rowStates)
        let logits = model.forward(tokens: tokens, caches: caches)
        let last = logits[0..., -1, 0...]
        return (last, eagerDecodeEvaluationRoots(caches, logitsRoot: last))
    }

    func prefillOutput(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        if let prefillModel = model as? CBv2PrefillSteppableModel {
            return prefillModel.prefill(
                tokens: tokens,
                inputEmbeddings: inputEmbeddings,
                caches: caches,
                requirement: requirement)
        }

        let logits: MLXArray
        if let inputEmbeddings {
            guard let multimodalModel = model as? CBv2MultimodalSteppableModel else {
                preconditionFailure(
                    "CBv2 embedding prefill reached a model without embedding-forward support")
            }
            logits = multimodalModel.forward(
                tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
        } else {
            logits = model.forward(tokens: tokens, caches: caches)
        }
        switch requirement {
        case .evaluationOnly:
            return logits[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return logits[0..., -1, 0...]
        }
    }

    // MARK: - Teacher-forced top-1 scoring (backend parity measurement)

    func teacherForcedTop1(promptTokens: [Int], continuation: [Int]) throws -> [Int] {
        guard !promptTokens.isEmpty, !continuation.isEmpty else {
            throw CBv2TeacherForcingError.nothingToScore(
                promptTokens: promptTokens.count, continuation: continuation.count)
        }
        return try engineQueue.sync {
            try scoreTeacherForced(promptTokens: promptTokens, continuation: continuation)
        }
    }

    private func scoreTeacherForced(
        promptTokens: [Int], continuation: [Int]
    ) throws -> [Int] {
        guard running, !draining else { throw CBv2TeacherForcingError.engineNotRunning }
        guard !scheduler.hasWork else {
            throw CBv2TeacherForcingError.engineBusy(
                scheduledRequests: scheduler.running.count + scheduler.waiting.count)
        }

        let state = try backend.makeSequenceState(
            layerKinds: layerKinds,
            promptLength: promptTokens.count,
            maxLength: promptTokens.count + continuation.count)
        defer {
            (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            backend.release(state)
        }

        var top1: [MLXArray] = []
        top1.reserveCapacity(continuation.count)

        let chunkSize = max(
            1, min(scheduler.config.prefillChunkSize, scheduler.config.maxBatchedTokensPerStep))
        var index = 0
        while index < promptTokens.count {
            let count = min(chunkSize, promptTokens.count - index)
            let isFinalChunk = index + count == promptTokens.count
            let inputs = MLXArray(promptTokens[index ..< index + count].map(Int32.init))
                .reshaped([1, count])
            let caches = eagerCaches(rowStates: [state])
            let output = prefillOutput(
                tokens: inputs, inputEmbeddings: nil, caches: caches,
                requirement: isFinalChunk ? .lastPositionLogits : .evaluationOnly)
            teacherForcedPrefillChunks += 1
            var toEval = eagerCacheInnerState(caches)
            if isFinalChunk {
                let argmax = argMax(output, axis: -1)  // [1]
                top1.append(argmax)
                toEval.append(argmax)
            } else {
                toEval.append(output)
            }
            asyncEval(toEval)
            index += count
        }

        for forced in continuation.dropLast() {
            let inputs = MLXArray([Int32(forced)]).reshaped([1, 1])
            let caches = eagerCaches(rowStates: [state])
            let logits = model.forward(tokens: inputs, caches: caches)[0..., -1, 0...]
            teacherForcedDecodeForwards += 1
            let argmax = argMax(logits, axis: -1)  // [1]
            top1.append(argmax)
            var toEval = eagerCacheInnerState(caches)
            toEval.append(argmax)
            asyncEval(toEval)
        }

        let scored = top1.count == 1 ? top1[0] : concatenated(top1, axis: 0)
        return scored.asArray(Int32.self).map(Int.init)
    }

    private func launchChainedDecode(
        _ plan: CBv2StepPlan, feeding lazyTokens: MLXArray
    ) -> CBv2InFlightStep {
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let ids = plan.assignments.map(\.id)
        let rowStates = ids.map { kvStates[$0]! }  // presence pre-checked
        var params: [CBv2SamplingParams] = []
        params.reserveCapacity(ids.count)
        for id in ids { params.append(scheduler.record(for: id)!.request.sampling) }

        let inputs = lazyTokens.reshaped([ids.count, 1])
        let forwardStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let (last, cacheInnerState) = CBv2OrderOnlyLogits.withOrderOnly(params) {
            decodeLogits(rowStates: rowStates, tokens: inputs)  // [B, vocab]
        }
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.forward.build", seconds: CFAbsoluteTimeGetCurrent() - forwardStart)
        }
        let samplerStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let sampled = sampler.sample(
            logits: last, params: params, requestIDs: ids, stepIndex: stepCount,
            pendingSampledTokens: lazyTokens,
            rowContext: { [scheduler] in
                ids.map { Self.samplerRow(scheduler.record(for: $0)!) }
            })
        let stepLogprobs = sampler.takeStepLogprobs()
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.sampler.build", seconds: CFAbsoluteTimeGetCurrent() - samplerStart)
        }
        scheduler.markPendingSamples(ids: ids)
        var toEval = [sampled]
        if let stepLogprobs { toEval.append(contentsOf: stepLogprobs.evalTargets) }
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        let evalStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        asyncEval(toEval)
        if CBv2StepProfiler.enabled {
            let now = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("v2.asyncEval.submit", seconds: now - evalStart)
            CBv2StepProfiler.record("v2.launch.total", seconds: now - buildStart)
            FileHandle.standardError.write(
                Data(String(format: "[cbv2-plain] rows=%d build %.1f ms submit %.1f ms\n",
                    ids.count, (evalStart - buildStart) * 1000, (now - evalStart) * 1000).utf8))
        }
        let step = CBv2InFlightStep(
            participants: Set(ids), sampledRows: ids, sampledTokens: sampled, evalTargets: [],
            wallStartedNanos: wallStartedNanos)
        if let stepLogprobs { step.logprobSegments = [stepLogprobs] }
        return step
    }

    func executeMixed(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        struct RowWork {
            let rec: CBv2ScheduledRequest
            let start: Int
            let count: Int
            let samples: Bool
            let isDecode: Bool
        }

        var work: [RowWork] = []
        work.reserveCapacity(plan.assignments.count)
        for (id, n) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }  // error-finished
            let start = rec.numComputedTokens - n  // pre-optimistic-advance position
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                n == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                RowWork(rec: rec, start: start, count: n, samples: samples, isDecode: isDecode))
        }
        guard !work.isEmpty else { return nil }

        var cacheInnerState: [MLXArray] = []
        var prefillWriteTail: [MLXArray] = []

        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var logprobSegments: [CBv2StepLogprobs] = []
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let (last, decodeInnerState) = CBv2OrderOnlyLogits.withOrderOnly(
                decodeRows.map(\.rec.request.sampling)
            ) {
                decodeLogits(
                    rowStates: decodeRows.map { kvStates[$0.rec.id]! }, tokens: inputs)
            }
            cacheInnerState.append(contentsOf: decodeInnerState)
            decodeSampled = sampler.sample(
                logits: last,
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,  // finalize preceded: all confirmed
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
        }

        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var evalTargets: [MLXArray] = []
        var packedIDs = Set<CBv2RequestID>()

        if packedPrefillSupported,
            let packedModel = model as? CBv2PackedPrefillSteppableModel
        {
            let canPackMultimodal =
                packedModel.supportsPackedMultimodalPrefill
                && cacheProvider.supportsPackedMultimodalSpans
            struct PackedGroup {
                let count: Int
                let samples: Bool
                var rows: [RowWork]
            }
            var groups: [PackedGroup] = []
            for row in work where !row.isDecode {
                let hasSpan =
                    multimodalByID[row.rec.id]?.chunkContext(
                        start: row.start, count: row.count) != nil
                if hasSpan && !canPackMultimodal { continue }
                if let index = groups.firstIndex(where: {
                    $0.count == row.count && $0.samples == row.samples
                }) {
                    groups[index].rows.append(row)
                } else {
                    groups.append(
                        PackedGroup(count: row.count, samples: row.samples, rows: [row]))
                }
            }

            for group in groups where group.rows.count > 1 {
                var flatTokens: [Int32] = []
                flatTokens.reserveCapacity(group.rows.count * group.count)
                for row in group.rows {
                    flatTokens.append(
                        contentsOf: row.rec.tokens[row.start ..< row.start + row.count]
                            .map(Int32.init))
                }
                let inputs = MLXArray(flatTokens).reshaped([group.rows.count, group.count])
                let caches = eagerCaches(rowStates: group.rows.map { kvStates[$0.rec.id]! })
                let requirement: CBv2PrefillRequirement =
                    group.samples ? .lastPositionLogits : .evaluationOnly
                let spanContexts = group.rows.map {
                    multimodalByID[$0.rec.id]?.chunkContext(
                        start: $0.start, count: $0.count)
                }
                let output: MLXArray
                if spanContexts.contains(where: { $0 != nil }) {
                    output = packedMultimodalChunksForward(
                        tokens: inputs,
                        starts: group.rows.map(\.start),
                        multimodal: group.rows.map { multimodalByID[$0.rec.id] },
                        spanContexts: spanContexts,
                        caches: caches,
                        requirement: requirement)
                } else {
                    output = prefillOutput(
                        tokens: inputs, inputEmbeddings: nil, caches: caches,
                        requirement: requirement)
                }
                let written = eagerCacheInnerState(caches)
                cacheInnerState.append(contentsOf: written)
                prefillWriteTail.append(contentsOf: written)

                if group.samples {
                    if let capture = prefillFrontierCapture() {
                        for (index, row) in group.rows.enumerated() {
                            capture(row.rec.id, output[index])
                        }
                    }
                    let sampled = sampler.sample(
                        logits: output,
                        params: group.rows.map(\.rec.request.sampling),
                        requestIDs: group.rows.map(\.rec.id),
                        stepIndex: stepCount,
                        pendingSampledTokens: nil,
                        rowContext: { group.rows.map { Self.samplerRow($0.rec) } })
                    for (index, row) in group.rows.enumerated() {
                        prefillSampled[row.rec.id] = sampled[index ..< index + 1]
                    }
                    if let stepLogprobs = sampler.takeStepLogprobs() {
                        logprobSegments.append(stepLogprobs)
                    }
                } else {
                    evalTargets.append(output)
                }
                packedIDs.formUnion(group.rows.map(\.rec.id))
                packedPrefillRowsExecuted += group.rows.count
                packedPrefillGroupsExecuted += 1
            }
        }

        for row in work where !row.isDecode {
            let rec = row.rec
            if packedIDs.contains(rec.id) { continue }
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let requirement: CBv2PrefillRequirement =
                row.samples ? .lastPositionLogits : .evaluationOnly
            let output: MLXArray
            if let multimodal = multimodalByID[rec.id],
                let spanContext = multimodal.chunkContext(start: row.start, count: row.count)
            {
                output = multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    multimodal: multimodal, spanContext: spanContext, caches: caches,
                    requirement: requirement)
            } else {
                output = prefillOutput(
                    tokens: inputs, inputEmbeddings: nil, caches: caches,
                    requirement: requirement)
            }
            let written = eagerCacheInnerState(caches)
            cacheInnerState.append(contentsOf: written)
            prefillWriteTail.append(contentsOf: written)
            if row.samples {
                if let capture = prefillFrontierCapture() {
                    capture(rec.id, output[0])
                }
                prefillSampled[rec.id] = sampler.sample(
                    logits: output,
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                evalTargets.append(output)
            }
        }

        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIdx = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIdx ..< decodeIdx + 1])
                decodeIdx += 1
                sampledRows.append(row.rec.id)
            } else if let s = prefillSampled[row.rec.id] {
                pieces.append(s)
                sampledRows.append(row.rec.id)
            }
        }
        var sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))
        if !prefillWriteTail.isEmpty {
            (sampledTokens, evalTargets) = fencedOnPrefillWriteTail(
                sampled: sampledTokens, outputs: evalTargets, tail: prefillWriteTail)
        }

        scheduler.markPendingSamples(ids: sampledRows)
        var toEval = evalTargets
        if let sampledTokens { toEval.append(sampledTokens) }
        for segment in logprobSegments { toEval.append(contentsOf: segment.evalTargets) }
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        asyncEval(toEval)

        let step = CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            evalTargets: evalTargets,
            wallStartedNanos: wallStartedNanos)
        step.logprobSegments = logprobSegments
        return step
    }

    // MARK: Vision prefill (span-containing chunks only)

    func multimodalChunkForward(
        tokens: MLXArray, start: Int, count: Int,
        multimodal: CBv2ResolvedMultimodal, spanContext: CBv2SpanChunkContext,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            preconditionFailure(
                "CBv2 multimodal chunk reached a model without embedding-forward support")
        }
        let textEmbeddings = mmModel.embedPromptTokens(tokens)
        let spliced = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: textEmbeddings,
            chunkStart: start,
            spans: multimodal.spansInChunk(start: start, count: count))
        let bindables = count > 1 ? caches.compactMap { $0 as? CBv2SpanMaskBinding } : []
        for bindable in bindables { bindable.bindSpanContext(spanContext) }
        defer { for bindable in bindables { bindable.bindSpanContext(nil) } }
        return prefillOutput(
            tokens: tokens, inputEmbeddings: spliced, caches: caches,
            requirement: requirement)
    }

    func packedMultimodalChunksForward(
        tokens: MLXArray,
        starts: [Int],
        multimodal: [CBv2ResolvedMultimodal?],
        spanContexts: [CBv2SpanChunkContext?],
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            preconditionFailure(
                "CBv2 packed multimodal chunk reached a model without embedding-forward support")
        }
        let batch = tokens.dim(0)
        let count = tokens.dim(1)
        precondition(
            starts.count == batch && multimodal.count == batch
                && spanContexts.count == batch,
            "CBv2 packed multimodal metadata must match batch \(batch)")

        let textEmbeddings = mmModel.embedPromptTokens(tokens)
        var embeddingRows: [MLXArray] = []
        embeddingRows.reserveCapacity(batch)
        for index in 0 ..< batch {
            let textRow = textEmbeddings[index ..< index + 1]
            if spanContexts[index] != nil, let rowMultimodal = multimodal[index] {
                embeddingRows.append(
                    CBv2MultimodalPlan.spliceEmbeddings(
                        textEmbeddings: textRow,
                        chunkStart: starts[index],
                        spans: rowMultimodal.spansInChunk(
                            start: starts[index], count: count)))
            } else {
                embeddingRows.append(textRow)
            }
        }
        let spliced = concatenated(embeddingRows, axis: 0)

        let bindables =
            count > 1 ? caches.compactMap { $0 as? CBv2PackedSpanMaskBinding } : []
        precondition(
            count == 1 || bindables.count == caches.count,
            "CBv2 packed multimodal prefill requires per-row span binding on every cache")
        for bindable in bindables { bindable.bindSpanContexts(spanContexts) }
        defer { for bindable in bindables { bindable.bindSpanContexts(nil) } }
        return prefillOutput(
            tokens: tokens, inputEmbeddings: spliced, caches: caches,
            requirement: requirement)
    }

    // MARK: Finalization (deferred stop detection)

    private var stepLogOrigin: CFAbsoluteTime = 0

    private func finalize(_ step: CBv2InFlightStep, now: ContinuousClock.Instant) {
        let readbackStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        var host: [Int32] = []
        if let tokens = step.sampledTokens {
            host = tokens.asArray(Int32.self)
        } else if !step.evalTargets.isEmpty {
            eval(step.evalTargets)
        }
        if CBv2StepProfiler.enabled {
            let finalizeTime = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("v2.readback.wait", seconds: finalizeTime - readbackStart)
            if stepLogOrigin == 0 { stepLogOrigin = finalizeTime }
            FileHandle.standardError.write(
                Data(String(format: "[cbv2-step] t=%.0f ms k=%d sampled=%d running=%d\n",
                    (finalizeTime - stepLogOrigin) * 1000, step.mtpRound?.verify?.k ?? -1,
                    step.sampledRows.count, scheduler.running.count).utf8))
            if let round = step.mtpRound, round.verify == nil, !round.seedRows.isEmpty {
                Self.mtpSeedProfileLine("readback", seconds: finalizeTime - readbackStart)
            }
        }
        if !step.sampledRows.isEmpty {
            sampler.confirmSampledTokens(
                host.map(Int.init), requestIDs: step.sampledRows)
        }

        var logprobsByID: [CBv2RequestID: CBv2TokenLogprob] = [:]
        if !step.logprobSegments.isEmpty {
            var tokenByID: [CBv2RequestID: Int] = [:]
            for (i, id) in step.sampledRows.enumerated() { tokenByID[id] = Int(host[i]) }
            for segment in step.logprobSegments {
                let sampledTokens = segment.rows.map { tokenByID[$0] ?? -1 }
                let assembled = CBv2Logprobs.assemble(
                    segment.gathered, sampledTokens: sampledTokens,
                    topLogprobsPerRow: segment.topLogprobsPerRow)
                for (row, logprob) in zip(segment.rows, assembled) {
                    if let logprob { logprobsByID[row] = logprob }
                }
            }
        }

        var finalizedPlainWork = false
        for (i, id) in step.sampledRows.enumerated() {
            if step.discard.contains(id) { continue }
            guard let rec = scheduler.record(for: id) else { continue }
            finalizedPlainWork = true
            let token = Int(host[i])
            scheduler.recordSampled(id: id, token: token)
            if let constraintFailure = sampler.tokenConstraintFailure(for: id) {
                finishRequest(id, reason: .error(constraintFailure))
                continue
            }

            let isStopToken = rec.request.stopTokens.contains(token)
            let detokenizer = detokenizers[id]
            let logprobs = logprobsByID[id].map { [$0] }

            var matchedStopString = false
            if rec.request.stopStrings.isEmpty {
                let stream = stream(for: id)
                stream?.reserveEmission()
                detokQueue.async {
                    let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: [token], logprobs: logprobs),
                        consumingReservation: true)
                }
            } else {
                let detokStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.detok.push", seconds: CFAbsoluteTimeGetCurrent() - detokStart)
                }
                let emitStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                stream(for: id)?.emit(.delta(text: text, tokens: [token], logprobs: logprobs))
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.stream.emit", seconds: CFAbsoluteTimeGetCurrent() - emitStart)
                }
                matchedStopString = detokenizer?.matchedStopString ?? false
            }

            if isStopToken {
                finishRequest(id, reason: .stop)
            } else if matchedStopString {
                finishRequest(id, reason: .stop)
            } else if rec.generatedTokenCount >= rec.request.maxTokens {
                finishRequest(id, reason: .length)
            }
        }

        if step.mtpRound != nil {
            finalizeMTPRound(step)
        }

        for (_, state, rollbackOne, donation) in step.deferredReleases {
            if rollbackOne {
                for sequence in state { sequence?.rollback(1) }
            }
            retire(state: state, donating: donation)
        }
        if let measurement = step.mtpMeasurement {
            let elapsed = DispatchTime.now().uptimeNanoseconds &- step.wallStartedNanos
            mtp?.recordStepCost(
                measurement,
                wallTimeNanos: elapsed,
                finalizedPlainWork: finalizedPlainWork,
                finalizedSeedIDs: step.mtpRound?.finalizedSeedIDs ?? [],
                finalizedVerification: !(step.mtpRound?.finalizedVerifyIDs.isEmpty ?? true),
                claimedSeedCostNanos: step.mtpRound?.claimedSeedCostNanos ?? 0)
        }

        refreshProgressLeases(step, now: config.clock.now())
    }

    private func refreshProgressLeases(_ step: CBv2InFlightStep, now: ContinuousClock.Instant) {
        for id in step.participants {
            guard let rec = scheduler.record(for: id) else { continue }
            if var lease = leasesByID[id] {
                lease.recordProgress(
                    now: now,
                    computedTokens: rec.numComputedTokens,
                    generatedTokens: rec.generatedTokenCount)
                leasesByID[id] = lease
            }
            if rec.generatedTokenCount > 0 {
                setUsageSnapshot(
                    id,
                    CBv2Usage(
                        promptTokens: rec.request.promptTokens.count,
                        completionTokens: rec.generatedTokenCount))
            }
        }
    }

    // MARK: Request completion

    func finishRequest(_ id: CBv2RequestID, reason: CBv2FinishReason) {
        capacityRequeues.removeValue(forKey: id)
        multimodalByID.removeValue(forKey: id)
        leasesByID.removeValue(forKey: id)
        clearUsageSnapshot(id)
        mtp?.requestDidFinish(id)
        guard let rec = scheduler.finish(id: id, reason: reason) else {
            let usage = takePrefixUsage(requestID: id, promptTokens: 0, completionTokens: 0)
            takeStream(id)?.finish(
                reason: reason, usage: usage)
            return
        }
        capacity?.releaseAll(id: id)
        sampler.requestDidFinish(id)

        if let state = kvStates.removeValue(forKey: id) {
            let donation = donationIntent(for: rec, reason: reason, state: state)
            if let inFlight, inFlight.participants.contains(id) {
                inFlight.discard.insert(id)
                inFlight.deferredReleases.append(
                    (
                        id: id, state: state,
                        rollbackOne: inFlight.sampledRows.contains(id),
                        donation: donation
                    ))
            } else {
                retire(state: state, donating: donation)
            }
        }

        let detokenizer = detokenizers.removeValue(forKey: id)
        let stream = takeStream(id)
        prefixUsageByID[id]?.boundarySplits = rec.prefixReplayBoundarySplits
        let usage = takePrefixUsage(
            requestID: id, promptTokens: rec.request.promptTokens.count,
            completionTokens: rec.generatedTokenCount)

        if rec.request.stopStrings.isEmpty {
            detokQueue.async {
                let trailing = detokenizer?.flush() ?? ""
                if !trailing.isEmpty {
                    stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
                }
                stream?.finish(reason: reason, usage: usage)
            }
        } else {
            let trailing = detokenizer?.flush() ?? ""
            if !trailing.isEmpty {
                stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
            }
            stream?.finish(reason: reason, usage: usage)
        }
    }

    // MARK: Prefix-cache donation (engine thread → donation queue)

    private func donationIntent(
        for rec: CBv2ScheduledRequest, reason: CBv2FinishReason, state: [CBv2SequenceKV?]
    ) -> CBv2DonationIntent? {
        guard prefixCache != nil else { return nil }
        guard rec.request.prefixCacheEnabled else { return nil }
        guard cbv2LayerKindsAllowPrefixReuse(layerKinds) else { return nil }
        guard rec.request.multimodal == nil else { return nil }
        switch reason {
        case .stop, .length: break
        case .cancelled, .error, .terminal: return nil
        }
        guard rec.generatedTokenCount >= 1, rec.tokens.count > 1 else { return nil }
        return CBv2DonationIntent(
            requestID: rec.request.prefixCacheReceiptID ?? rec.id,
            tokens: Array(rec.tokens.dropLast()),
            cacheSalt: rec.request.cacheSalt)
    }

    private func retire(
        state: [CBv2SequenceKV?], donating donation: CBv2DonationIntent?
    ) {
        guard prefixCache != nil, let donation else {
            backend.release(state)
            return
        }
        let queued = enqueueDonation(state: state, intent: donation)
        if !queued {
            backend.release(state)
        }
    }

    private func enqueueDonation(
        state: [CBv2SequenceKV?], intent: CBv2DonationIntent
    ) -> Bool {
        guard let prefixCache else { return false }
        let tokenCount = intent.tokens.count
        guard stateCoversDonation(state, tokenCount: tokenCount) else { return false }
        let windowDonor = prefixCache as? any CBv2SlidingWindowDonating
        let donatesWindows = windowDonor?.wantsSlidingWindowDonation ?? false
        var built: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        var sliding: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        built.reserveCapacity(layerKinds.count)
        if donatesWindows { sliding.reserveCapacity(layerKinds.count) }
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            var isSliding = false
            if case .slidingWindow = kind.attention {
                cacheable = false
                isSliding = true
            }
            if donatesWindows {
                sliding.append(
                    isSliding && kind.sharesKVWithLayer == nil
                        ? state[i]?.snapshot()
                        : nil)
            }
            guard cacheable, let seq = state[i] else {
                built.append(nil)
                continue
            }
            let snap = seq.snapshot()
            if snap.offset == tokenCount {
                built.append(snap)
            } else {
                built.append(
                    (
                        keys: snap.keys[.ellipsis, 0 ..< tokenCount, 0...],
                        values: snap.values[.ellipsis, 0 ..< tokenCount, 0...],
                        offset: tokenCount
                    ))
            }
        }
        let layerKinds = self.layerKinds
        let handoff = CBv2Handoff(
            value: (state: state, snapshots: built, sliding: sliding, intent: intent))
        pendingDonationReleaseCount += 1
        donationQueue.async {
            if let windowDonor, donatesWindows {
                windowDonor.donate(
                    requestID: handoff.value.intent.requestID,
                    tokens: handoff.value.intent.tokens,
                    snapshots: handoff.value.snapshots,
                    slidingSnapshots: handoff.value.sliding,
                    layerKinds: layerKinds,
                    cacheSalt: handoff.value.intent.cacheSalt)
            } else {
                prefixCache.donate(
                    requestID: handoff.value.intent.requestID,
                    tokens: handoff.value.intent.tokens,
                    snapshots: handoff.value.snapshots, layerKinds: layerKinds,
                    cacheSalt: handoff.value.intent.cacheSalt)
            }
            self.releaseDonationStateOnEngineQueue(handoff.value.state)
        }
        return true
    }

    private func stateCoversDonation(
        _ state: [CBv2SequenceKV?], tokenCount: Int
    ) -> Bool {
        guard tokenCount > 1 else { return false }
        var cacheableLayers = 0
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            if case .slidingWindow = kind.attention { cacheable = false }
            guard cacheable else { continue }
            guard let sequence = state[i], sequence.absoluteOffset >= tokenCount else { return false }
            cacheableLayers += 1
        }
        return cacheableLayers > 0
    }

    private func takePrefixUsage(
        requestID: CBv2RequestID, promptTokens: Int, completionTokens: Int
    ) -> CBv2Usage {
        let prefix = prefixUsageByID.removeValue(forKey: requestID) ?? .disabled
        let saved = prefixHitTokens.removeValue(forKey: requestID) ?? prefix.prefillTokensSaved
        return CBv2Usage(
            promptTokens: promptTokens, completionTokens: completionTokens,
            prefixCacheHitTokens: saved,
            prefixCacheOutcome: prefix.outcome,
            prefixCacheMatchedTokens: prefix.matchedTokens,
            prefixCachePrefillTokensSaved: saved,
            prefixCacheStrategy: prefix.strategy,
            prefixCacheReplayTokens: prefix.replayTokens,
            prefixCacheBoundarySplits: prefix.boundarySplits)
    }

    private func releaseDonationStateOnEngineQueue(_ state: [CBv2SequenceKV?]) {
        let handoff = CBv2Handoff(value: state)
        engineQueue.async { [self] in
            backend.release(handoff.value)
            assert(pendingDonationReleaseCount > 0, "unbalanced donation release completion")
            pendingDonationReleaseCount = max(0, pendingDonationReleaseCount - 1)
            publishGauges()
            completeDrainIfReady()
        }
    }

    // MARK: Sampler row context

    static func samplerRow(_ rec: CBv2ScheduledRequest) -> CBv2SamplerRow {
        CBv2SamplerRow(
            id: rec.id,
            params: rec.request.sampling,
            promptTokens: rec.request.promptTokens,
            outputTokens: Array(rec.tokens.dropFirst(rec.request.promptTokens.count)),
            tokenConstraint: rec.request.tokenConstraint,
            maxTokens: rec.request.maxTokens)
    }

    // MARK: Boundary housekeeping

    private func processCancellations() {
        stateLock.lock()
        let cancels = pendingCancels
        stateLock.unlock()

        var resolved: Set<CBv2RequestID> = []
        for id in cancels {
            if scheduler.record(for: id) != nil {
                finishRequest(id, reason: .cancelled)
                resolved.insert(id)
            } else if !hasRegisteredStream(id) {
                resolved.insert(id)
            }
        }
        guard !resolved.isEmpty else { return }
        stateLock.lock()
        pendingCancels.subtract(resolved)
        stateLock.unlock()
    }

    private func hasRegisteredStream(_ id: CBv2RequestID) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id] != nil
    }

    private func consumeEarlyCancel(_ id: CBv2RequestID) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return pendingCancels.remove(id) != nil
    }

    // MARK: Deadline leases

    private func armLease(for request: CBv2Request) {
        let now = config.clock.now()
        let lease: CBv2RequestLeaseState
        if config.useLegacyRequestTimeout {
            lease = .legacy(now: now, wall: config.requestTimeout)
        } else {
            lease = CBv2RequestLeaseState(
                now: now,
                admissionLease: config.admissionLease,
                prefillLease: config.prefillProgressLease,
                decodeLease: config.decodeProgressLease,
                backpressureLease: config.backpressureLease,
                safety: CBv2SafetyCeiling.duration(
                    promptTokens: request.promptTokens.count,
                    maxTokens: request.maxTokens,
                    admissionLease: config.admissionLease,
                    decodeFloorTPS: config.safetyCeilingDecodeFloorTPS),
                computedTokens: 0,
                generatedTokens: 0)
        }
        leasesByID[request.id] = lease
        setUsageSnapshot(
            request.id,
            CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
    }

    private func markAdmitted(
        _ plan: CBv2StepPlan, now: ContinuousClock.Instant,
        readmittedFrom waitingBefore: Set<CBv2RequestID>
    ) {
        for (id, _) in plan.assignments {
            guard var lease = leasesByID[id] else { continue }
            if !lease.isAdmitted {
                lease.markAdmitted(now: now)
                leasesByID[id] = lease
            } else if waitingBefore.contains(id) {
                lease.markReadmitted(now: now)
                leasesByID[id] = lease
            }
        }
    }

    private func processLeaseExpiry(now: ContinuousClock.Instant) {
        var expired: [(CBv2RequestID, CBv2TerminalCause)] = []
        for rec in scheduler.running {
            if let cause = leasesByID[rec.id]?.expiredCause(
                now: now, isRunning: true, isPaused: rec.isPaused)
            {
                expired.append((rec.id, cause))
            }
        }
        for rec in scheduler.waiting {
            if let cause = leasesByID[rec.id]?.expiredCause(
                now: now, isRunning: false, isPaused: rec.isPaused)
            {
                expired.append((rec.id, cause))
            }
        }
        for (id, cause) in expired {
            finishRequest(id, reason: leaseFinishReason(for: cause))
        }
    }

    private func leaseFinishReason(for cause: CBv2TerminalCause) -> CBv2FinishReason {
        switch cause {
        case .legacyRequestTimeout:
            return .error("request exceeded \(Int(config.requestTimeout))s deadline")
        default:
            return .terminal(cause: cause, message: cause.diagnostic)
        }
    }

    private func handlePreemptions(_ ids: [CBv2RequestID]) {
        assert(inFlight == nil, "preemption with a step in flight")
        let now = config.clock.now()
        for id in ids {
            preemptionCount += 1
            invalidateAdoptedPrefix(id)
            mtp?.invalidateCarry(id)
            if var lease = leasesByID[id] {
                lease.markPreempted(now: now)
                leasesByID[id] = lease
            }
            if let state = kvStates.removeValue(forKey: id) {
                backend.release(state)
            }
        }
    }

    func ensureKVState(_ rec: CBv2ScheduledRequest) -> [CBv2SequenceKV?]? {
        if let state = kvStates[rec.id] { return state }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                layerKinds: layerKinds, promptLength: rec.tokens.count, maxLength: maxLength)
            kvStates[rec.id] = state
            capacityRequeues.removeValue(forKey: rec.id)
            return state
        } catch let kvError as CBv2KVError {
            if case .capacityExhausted = kvError {
                let attempts = capacityRequeues[rec.id, default: 0]
                if attempts < Self.maxCapacityRequeues, scheduler.requeueOnCapacity(rec.id) {
                    capacityRequeues[rec.id] = attempts + 1
                    capacityRequeueCount += 1
                    mtp?.invalidateCarry(rec.id)  // preempted-style restart
                    if var lease = leasesByID[rec.id] {
                        lease.markPreempted(now: config.clock.now())
                        leasesByID[rec.id] = lease
                    }
                    return nil
                }
                finishRequest(
                    rec.id,
                    reason: .error(
                        CBv2KVError.capacityExhaustedFinishPrefix + "\(kvError)"))
                return nil
            }
            finishRequest(rec.id, reason: .error("KV allocation failed: \(kvError)"))
            return nil
        } catch {
            finishRequest(rec.id, reason: .error("KV allocation failed: \(error)"))
            return nil
        }
    }

    // MARK: Streams

    func stream(for id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id]
    }

    private func takeStream(_ id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams.removeValue(forKey: id)
    }

    // MARK: Reconciled-usage snapshots (watchdog-readable)

    private func setUsageSnapshot(_ id: CBv2RequestID, _ usage: CBv2Usage) {
        stateLock.lock()
        usageSnapshots[id] = usage
        stateLock.unlock()
    }

    private func clearUsageSnapshot(_ id: CBv2RequestID) {
        stateLock.lock()
        usageSnapshots.removeValue(forKey: id)
        stateLock.unlock()
    }

    // MARK: Test/telemetry introspection

    func pausedIDsSnapshot() -> Set<CBv2RequestID> {
        engineQueue.sync { Set(scheduler.running.filter(\.isPaused).map(\.id)) }
    }

    // MARK: Gauges

    private func publishGauges() {
        gauges.update(
            CBv2CapacitySnapshot(
                activeRequests: scheduler.runningCount,
                waitingRequests: scheduler.waitingCount,
                kvBytesInUse: backend.bytesInUse,
                kvBytesCapacity: capacity?.bytesCapacity ?? backend.bytesCapacity,
                kvBytesBackendCapacity: backend.bytesCapacity,
                kvBytesReserved: backend.bytesReserved,
                activeTokens: scheduler.activeTokens,
                stepsExecuted: stepCount))
    }

    // MARK: Watchdog (engine health signal)

    private var watchdogTimer: DispatchSourceTimer?

    private func markStepStarted() {
        stateLock.lock()
        stepStartedNanos = DispatchTime.now().uptimeNanoseconds
        stateLock.unlock()
    }

    private func markStepEnded() {
        stateLock.lock()
        stepStartedNanos = 0
        wedgeReported = false
        _healthy = true
        stateLock.unlock()
    }

    deinit {
        watchdogTimer?.cancel()
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(
            deadline: .now() + config.watchdogInterval, repeating: config.watchdogInterval)
        timer.setEventHandler { [weak self] in self?.watchdogTick() }
        timer.resume()
        watchdogTimer = timer
    }

    private func stopWatchdog() {
        watchdogTimer?.cancel()
        watchdogTimer = nil
    }

    private func watchdogTick() {
        stateLock.lock()
        let started = stepStartedNanos
        guard started != 0 else {
            stateLock.unlock()
            return
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- started) / 1_000_000_000
        guard elapsed > config.stepTimeout, !wedgeReported else {
            stateLock.unlock()
            return
        }
        wedgeReported = true
        _healthy = false
        let liveStreams = streams
        let usageByID = usageSnapshots
        pendingCancels.formUnion(liveStreams.keys)
        stateLock.unlock()

        onStepWedge?(elapsed)
        let message = "engine step exceeded \(Int(config.stepTimeout))s watchdog"
        for (id, stream) in liveStreams {
            stream.finish(
                reason: .terminal(cause: .watchdog, message: message),
                usage: usageByID[id]
                    ?? CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
    }
}
