import Foundation
import Testing
@testable import MLXLMCommon

@Suite("Runtime worker phase-close ordering")
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
            of: "runtime worker phase close timed out waiting for detached CBv2 drains"),
            "a detached-drain timeout must fail the phase closed")
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
    func detachedDrainRegistryRetainsTimedOutTaskForTheNextFence() throws {
        let started = DispatchSemaphore(value: 0)
        let release = AsyncStream<Void>.makeStream()
        CBv2DetachedDrainRegistry.register(Task.detached {
            started.signal()
            for await _ in release.stream { break }
        })
        defer {
            release.continuation.finish()
            _ = CBv2DetachedDrainRegistry.joinAll(timeout: 1)
        }

        #expect(started.wait(timeout: .now() + 1) == .success)
        #expect(!CBv2DetachedDrainRegistry.joinAll(timeout: 0.01))

        release.continuation.yield(())
        release.continuation.finish()
        #expect(CBv2DetachedDrainRegistry.joinAll(timeout: 1))
    }
}
