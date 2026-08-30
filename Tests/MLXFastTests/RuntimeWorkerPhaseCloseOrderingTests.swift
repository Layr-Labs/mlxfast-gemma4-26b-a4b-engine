import Foundation
import Testing
@testable import MLXLMCommon

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
            "private static var drains: [Task<CBv2DrainRetirement, Never>]"))
        #expect(engineSource.contains("private static let joinLock = NSLock()"))
        #expect(engineSource.contains(
            "guard outcomes.allSatisfy({ $0 == .natural }) else { return false }"))
        #expect(engineSource.contains("drains.removeFirst(pending.count)"))
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
    func libraryModelConstructionBoundaryChecksDetachedDrainFence() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift"),
            encoding: .utf8)
        let methodStart = try #require(source.range(
            of: "    public func requireLibraryModel() throws -> Gemma4TextModel {"))
        let methodEnd = try #require(source.range(
            of: "    /// Startup readiness check",
            range: methodStart.upperBound..<source.endIndex))
        let method = String(source[methodStart.lowerBound..<methodEnd.lowerBound])

        #expect(method.contains(
            "guard CBv2DetachedDrainRegistry.joinAll(timeout: 5) else {"))
        #expect(method.contains(
            "Gemma library model construction boundary cannot proceed "))
        #expect(method.contains(
            "+ \"before all CBv2 drains retire naturally\""))
    }

    @Test
    func detachedDrainRegistryRetainsTimedOutTaskForTheNextFence() throws {
        CBv2DetachedDrainRegistry.resetForTesting()
        let started = DispatchSemaphore(value: 0)
        let release = AsyncStream<Void>.makeStream()
        CBv2DetachedDrainRegistry.register(Task.detached {
            started.signal()
            for await _ in release.stream { break }
            return .natural
        })
        defer {
            release.continuation.finish()
            _ = CBv2DetachedDrainRegistry.joinAll(timeout: 1)
            CBv2DetachedDrainRegistry.resetForTesting()
        }

        #expect(started.wait(timeout: .now() + 1) == .success)
        #expect(!CBv2DetachedDrainRegistry.joinAll(timeout: 0.01))

        release.continuation.yield(())
        release.continuation.finish()
        #expect(CBv2DetachedDrainRegistry.joinAll(timeout: 1))
    }

    @Test
    func detachedDrainRegistryRejectsAndRetainsWatchdogEscape() {
        CBv2DetachedDrainRegistry.resetForTesting()
        defer { CBv2DetachedDrainRegistry.resetForTesting() }

        CBv2DetachedDrainRegistry.register(Task.detached {
            CBv2DrainRetirement.watchdogEscaped
        })

        #expect(!CBv2DetachedDrainRegistry.joinAll(timeout: 1))
        #expect(
            !CBv2DetachedDrainRegistry.joinAll(timeout: 1),
            "a watchdog escape must remain registered and fail every later fence")
    }
}
