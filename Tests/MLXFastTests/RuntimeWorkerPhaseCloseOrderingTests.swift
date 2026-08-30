import Foundation
import Testing
@testable import MLXLMCommon

private final class RuntimeWorkerDrainJoinResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func store(_ newValue: Bool) {
        lock.lock()
        value = newValue
        lock.unlock()
    }

    func load() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

@Suite("Runtime worker phase-close ordering", .serialized)
struct RuntimeWorkerPhaseCloseOrderingTests {
    @Test
    func phaseCloseJoinsDetachedDrainsThenRetiresGPUBeforeAllocatorDrain() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let phaseStart = try #require(source.range(of: "        case \"phase_diagnostics\":"))
        let phaseEnd = try #require(source.range(
            of: "        default:",
            range: phaseStart.upperBound..<source.endIndex))
        let phase = String(source[phaseStart.lowerBound..<phaseEnd.lowerBound])

        let cohortTeardown = try #require(phase.range(
            of: "cohortSession.shutdownBlocking()"))
        let singleStreamTeardown = try #require(phase.range(
            of: "freeRunSession.shutdownBlocking()"))
        let recordingTeardown = try #require(phase.range(
            of: "recordingSession.shutdownBlocking()"))
        let detachedDrainJoin = try #require(phase.range(
            of: "guard CBv2DetachedDrainRegistry.joinAll(timeout: 15) else {"),
            "phase close must check a bounded join of every fast-ack engine drain")
        let detachedDrainRefusal = try #require(phase.range(
            of: "runtime worker phase close could not prove natural retirement of detached CBv2 drains"),
            "a detached-drain timeout or watchdog escape must fail the phase closed")
        let gpuRetirement = try #require(phase.range(
            of: "Stream.gpu.synchronize()"),
            "phase close must retire the process-global GPU stream used by CBv2")
        let preDrainActive = try #require(phase.range(
            of: "let mlxActiveMemoryBytes = Memory.activeMemory"))
        let preDrainCache = try #require(phase.range(
            of: "let mlxCacheMemoryBytes = Memory.cacheMemory"))
        let clear = try #require(phase.range(of: "Memory.clearCache()"))
        let drainedRead = try #require(phase.range(
            of: "let drainedCacheMemory = Memory.cacheMemory"))

        #expect(cohortTeardown.lowerBound < detachedDrainJoin.lowerBound)
        #expect(singleStreamTeardown.lowerBound < detachedDrainJoin.lowerBound)
        #expect(recordingTeardown.lowerBound < detachedDrainJoin.lowerBound)
        #expect(detachedDrainJoin.lowerBound < detachedDrainRefusal.lowerBound)
        #expect(detachedDrainRefusal.lowerBound < gpuRetirement.lowerBound)
        #expect(gpuRetirement.lowerBound < preDrainActive.lowerBound)
        #expect(preDrainActive.lowerBound < preDrainCache.lowerBound)
        #expect(preDrainCache.lowerBound < clear.lowerBound)
        #expect(clear.lowerBound < drainedRead.lowerBound)
    }

    @Test
    func detachedDrainContractDistinguishesNaturalRetirementFromWatchdogEscape() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let loopSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/EngineLoopV2.swift"),
            encoding: .utf8)
        let engineSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/EngineV2.swift"),
            encoding: .utf8)

        #expect(loopSource.contains(
            "enum CBv2DrainRetirement: Sendable, Equatable"))
        #expect(loopSource.contains(
            "func drain() async -> CBv2DrainRetirement"))
        #expect(loopSource.contains("waiter.resume(.watchdogEscaped)"))
        #expect(loopSource.contains("waiter.resume(.natural)"))
        #expect(engineSource.contains(
            "final class CBv2DetachedDrainStorage: @unchecked Sendable"))
        #expect(engineSource.contains(
            "private var drains: [Task<CBv2DrainRetirement, Never>]"))
        #expect(engineSource.contains("private let joinLock = NSLock()"))
        #expect(engineSource.contains(
            "guard outcomes.allSatisfy({ $0 == .natural }) else { return false }"))
        #expect(engineSource.contains("drains.removeFirst(pending.count)"))

        let registerStart = try #require(engineSource.range(
            of: "    func register(_ task: Task<CBv2DrainRetirement, Never>) {"))
        let registerEnd = try #require(engineSource.range(
            of: "\n    }",
            range: registerStart.upperBound..<engineSource.endIndex))
        let register = engineSource[registerStart.lowerBound..<registerEnd.upperBound]
        let joinLock = try #require(register.range(of: "joinLock.lock()"))
        let storageLock = try #require(register.range(of: "lock.lock()"))
        #expect(joinLock.lowerBound < storageLock.lowerBound)
    }

    @Test
    func naturalDrainRetirementCrossesAQueueLifetimeSentinel() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/EngineLoopV2.swift"),
            encoding: .utf8)
        let methodStart = try #require(source.range(
            of: "    private func completeDrainIfReady() {"))
        let methodEnd = try #require(source.range(
            of: "\n    }",
            range: methodStart.upperBound..<source.endIndex))
        let method = String(source[methodStart.lowerBound..<methodEnd.upperBound])

        let stop = try #require(method.range(of: "completeStop()"))
        let capture = try #require(method.range(of: "let waiters = drainWaiters"))
        let sentinel = try #require(method.range(
            of: "engineQueue.async {"))
        let resume = try #require(method.range(
            of: "waiter.resume(.natural)"))
        #expect(stop.lowerBound < capture.lowerBound)
        #expect(capture.lowerBound < sentinel.lowerBound)
        #expect(sentinel.lowerBound < resume.lowerBound)
        #expect(!method[method.startIndex..<sentinel.lowerBound].contains(
            "waiter.resume(.natural)"))
    }

    @Test
    func everyShutdownModeRegistersItsDrainBeforeAnyOptionalAwait() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let engineSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/EngineV2.swift"),
            encoding: .utf8)

        let shutdownStart = try #require(engineSource.range(
            of: "    public func shutdown() async {"))
        let shutdownEnd = try #require(engineSource.range(
            of: "    /// Always-awaiting variant",
            range: shutdownStart.upperBound..<engineSource.endIndex))
        let shutdown = String(
            engineSource[shutdownStart.lowerBound..<shutdownEnd.lowerBound])
        let synchronousStart = try #require(engineSource.range(
            of: "    public func shutdownSynchronously() async {"))
        let synchronousEnd = try #require(engineSource.range(
            of: "    /// Resolved once:",
            range: synchronousStart.upperBound..<engineSource.endIndex))
        let synchronous = String(
            engineSource[synchronousStart.lowerBound..<synchronousEnd.lowerBound])
        let registrationStart = try #require(engineSource.range(
            of: "    private func registerDrain() -> Task<CBv2DrainRetirement, Never> {"))
        let registrationEnd = try #require(engineSource.range(
            of: "\n    }",
            range: registrationStart.upperBound..<engineSource.endIndex))
        let registration = String(
            engineSource[registrationStart.lowerBound..<registrationEnd.upperBound])

        let shutdownBegin = try #require(shutdown.range(
            of: "beginRejectingSubmissions()"))
        let shutdownDrain = try #require(shutdown.range(
            of: "let drain = registerDrain()"))
        let fastAckBranch = try #require(shutdown.range(
            of: "if Self.fastAckShutdown"))
        let optionalAwait = try #require(shutdown.range(
            of: "_ = await drain.value"))
        #expect(shutdownBegin.lowerBound < shutdownDrain.lowerBound)
        #expect(shutdownDrain.lowerBound < fastAckBranch.lowerBound)
        #expect(fastAckBranch.lowerBound < optionalAwait.lowerBound)

        let synchronousBegin = try #require(synchronous.range(
            of: "beginRejectingSubmissions()"))
        let synchronousDrain = try #require(synchronous.range(
            of: "let drain = registerDrain()"))
        let synchronousAwait = try #require(synchronous.range(
            of: "_ = await drain.value"))
        #expect(synchronousBegin.lowerBound < synchronousDrain.lowerBound)
        #expect(synchronousDrain.lowerBound < synchronousAwait.lowerBound)

        let taskCreation = try #require(registration.range(
            of: "let drain = Task.detached"))
        let taskRegistration = try #require(registration.range(
            of: "CBv2DetachedDrainRegistry.register(drain)"))
        let taskReturn = try #require(registration.range(of: "return drain"))
        #expect(taskCreation.lowerBound < taskRegistration.lowerBound)
        #expect(taskRegistration.lowerBound < taskReturn.lowerBound)
    }

    @Test
    func libraryModelAccessorIsLockFreeAndConstructionBoundaryChecksDrainFence() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift"),
            encoding: .utf8)
        let accessorStart = try #require(source.range(
            of: "    public func requireLibraryModel() throws -> Gemma4TextModel {"))
        let boundaryStart = try #require(source.range(
            of: "    public func requireLibraryModelAtDrainFencedBoundary() throws -> Gemma4TextModel {"))
        let startupStart = try #require(source.range(
            of: "    /// Startup readiness check",
            range: boundaryStart.upperBound..<source.endIndex))
        let accessor = String(source[accessorStart.lowerBound..<boundaryStart.lowerBound])
        let boundary = String(source[boundaryStart.lowerBound..<startupStart.lowerBound])

        #expect(!accessor.contains("CBv2DetachedDrainRegistry"))
        #expect(boundary.contains(
            "guard CBv2DetachedDrainRegistry.joinAll(timeout: 5) else {"))
        #expect(boundary.contains(
            "Gemma library model construction boundary cannot proceed "))
        #expect(boundary.contains(
            "+ \"before all CBv2 drains retire naturally\""))
    }

    @Test
    func scoredStepCasesDoNotFenceOrTouchTheDrainRegistry() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)

        for (kind, nextKind) in [
            ("correctness_step", "prefill"),
            ("decode_step", "free_decode_begin"),
            ("free_decode_run", "record_reference_begin"),
        ] {
            let start = try #require(source.range(of: "        case \"\(kind)\":"))
            let end = try #require(source.range(
                of: "        case \"\(nextKind)\":",
                range: start.upperBound..<source.endIndex))
            let body = source[start.lowerBound..<end.lowerBound]
            #expect(!body.contains("requireLibraryModelAtDrainFencedBoundary"))
            #expect(!body.contains("CBv2DetachedDrainRegistry"))
            #expect(!body.contains("joinAll("))
        }

        let trustedSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastTrustedHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)
        for (kind, terminator) in [
            ("correctness_step", "        case \"prefill\":"),
            ("decode_step", "        default:"),
        ] {
            let start = try #require(trustedSource.range(
                of: "        case \"\(kind)\":"))
            let end = try #require(trustedSource.range(
                of: terminator,
                range: start.upperBound..<trustedSource.endIndex))
            let body = trustedSource[start.lowerBound..<end.lowerBound]
            #expect(!body.contains("requireLibraryModelAtDrainFencedBoundary"))
            #expect(!body.contains("CBv2DetachedDrainRegistry"))
            #expect(!body.contains("joinAll("))
        }
    }

    @Test
    func freeDecodeBeginFencesOnceAndPassesTheSameModelIntoEitherEngine() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let workerSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4RuntimeWorker.swift"),
            encoding: .utf8)
        let beginStart = try #require(workerSource.range(
            of: "        case \"free_decode_begin\":"))
        let beginEnd = try #require(workerSource.range(
            of: "        case \"free_decode_run\":",
            range: beginStart.upperBound..<workerSource.endIndex))
        let begin = String(workerSource[beginStart.lowerBound..<beginEnd.lowerBound])

        #expect(begin.components(
            separatedBy: "requireLibraryModelAtDrainFencedBoundary()").count == 2)
        #expect(begin.contains("currentTarget: freeDecodeModel"))
        #expect(begin.components(separatedBy: "model: freeDecodeModel").count == 3)

        let cohortSource = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4RuntimeCohortDriver.swift"),
            encoding: .utf8)
        let cohortStart = try #require(cohortSource.range(
            of: "    static func handleCohortFreeDecodeBegin("))
        let cohortEnd = try #require(cohortSource.range(
            of: "    /// v1.2 batched `free_decode_run`",
            range: cohortStart.upperBound..<cohortSource.endIndex))
        let cohort = cohortSource[cohortStart.lowerBound..<cohortEnd.lowerBound]
        #expect(cohort.contains("model: Gemma4TextModel"))
        #expect(!cohort.contains("requireLibraryModel"))
        #expect(!cohort.contains("CBv2DetachedDrainRegistry"))
    }

    @Test
    func detachedDrainRegistryRetainsTimedOutTaskForTheNextFence() throws {
        let registry = CBv2DetachedDrainStorage()
        let started = DispatchSemaphore(value: 0)
        let release = AsyncStream<Void>.makeStream()
        registry.register(Task.detached {
            started.signal()
            for await _ in release.stream { break }
            return .natural
        })
        defer {
            release.continuation.finish()
            _ = registry.joinAll(timeout: 1)
        }

        #expect(started.wait(timeout: .now() + 1) == .success)
        #expect(!registry.joinAll(timeout: 0.01))

        release.continuation.yield(())
        release.continuation.finish()
        #expect(registry.joinAll(timeout: 1))
    }

    @Test
    func detachedDrainRegistryRejectsAndRetainsWatchdogEscape() {
        let registry = CBv2DetachedDrainStorage()

        registry.register(Task.detached {
            CBv2DrainRetirement.watchdogEscaped
        })

        #expect(!registry.joinAll(timeout: 1))
        #expect(
            !registry.joinAll(timeout: 1),
            "a watchdog escape must remain registered and fail every later fence")
    }

    @Test
    func registrationLinearizesAfterAnActiveJoin() throws {
        let registry = CBv2DetachedDrainStorage()
        let firstRelease = AsyncStream<Void>.makeStream()
        registry.register(Task.detached {
            for await _ in firstRelease.stream { break }
            return .natural
        })
        let snapshotted = DispatchSemaphore(value: 0)
        let joinFinished = DispatchSemaphore(value: 0)
        let joinResult = RuntimeWorkerDrainJoinResultBox()
        DispatchQueue.global().async {
            joinResult.store(registry.joinAll(timeout: 1, snapshotObserverForTesting: {
                snapshotted.signal()
            }))
            joinFinished.signal()
        }
        #expect(snapshotted.wait(timeout: .now() + 1) == .success)

        let registrationFinished = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            registry.register(Task.detached { .watchdogEscaped })
            registrationFinished.signal()
        }
        #expect(registrationFinished.wait(timeout: .now() + 0.01) == .timedOut)

        firstRelease.continuation.finish()
        #expect(joinFinished.wait(timeout: .now() + 1) == .success)
        #expect(joinResult.load())
        #expect(registrationFinished.wait(timeout: .now() + 1) == .success)
        #expect(!registry.joinAll(timeout: 1))
    }
}
