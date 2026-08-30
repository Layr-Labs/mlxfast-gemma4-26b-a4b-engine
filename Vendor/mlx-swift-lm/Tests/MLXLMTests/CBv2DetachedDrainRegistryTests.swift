import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2 detached-drain retirement fence", .serialized)
struct CBv2DetachedDrainRegistryTests {
    @Test
    func watchdogEscapeFailsClosedAndRemainsRegistered() {
        let registry = CBv2DetachedDrainStorage()

        registry.register(Task.detached {
            CBv2DrainRetirement.watchdogEscaped
        })

        #expect(!registry.joinAll(timeout: 1))
        #expect(
            !registry.joinAll(timeout: 1),
            "an unsuccessful retirement must remain registered")
    }

    @Test
    func naturalRetirementClearsTheJoinedPrefix() {
        let registry = CBv2DetachedDrainStorage()

        registry.register(Task.detached {
            CBv2DrainRetirement.natural
        })
        // Models two idempotent shutdown calls: both registered drain results
        // must be joined and removed as one natural prefix.
        registry.register(Task.detached {
            CBv2DrainRetirement.natural
        })

        #expect(registry.joinAll(timeout: 1))
        #expect(registry.joinAll(timeout: 0))
    }

    @Test
    func deadlineRetainsPendingDrainForALaterFence() {
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
}
