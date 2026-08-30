import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2 detached-drain retirement fence", .serialized)
struct CBv2DetachedDrainRegistryTests {
    @Test
    func watchdogEscapeFailsClosedAndRemainsRegistered() {
        CBv2DetachedDrainRegistry.resetForTesting()
        defer { CBv2DetachedDrainRegistry.resetForTesting() }

        CBv2DetachedDrainRegistry.register(Task.detached {
            CBv2DrainRetirement.watchdogEscaped
        })

        #expect(!CBv2DetachedDrainRegistry.joinAll(timeout: 1))
        #expect(
            !CBv2DetachedDrainRegistry.joinAll(timeout: 1),
            "an unsuccessful retirement must remain registered")
    }

    @Test
    func naturalRetirementClearsTheJoinedPrefix() {
        CBv2DetachedDrainRegistry.resetForTesting()
        defer { CBv2DetachedDrainRegistry.resetForTesting() }

        CBv2DetachedDrainRegistry.register(Task.detached {
            CBv2DrainRetirement.natural
        })

        #expect(CBv2DetachedDrainRegistry.joinAll(timeout: 1))
        #expect(CBv2DetachedDrainRegistry.joinAll(timeout: 0))
    }

    @Test
    func deadlineRetainsPendingDrainForALaterFence() {
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
}
