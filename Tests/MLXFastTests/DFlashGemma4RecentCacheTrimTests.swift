import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("DFlash Gemma 4 recent-cache rollback")
struct DFlashGemma4RecentCacheTrimTests {
    private var runtimeTestsEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    private func values(_ start: Int, _ count: Int) -> (MLXArray, MLXArray) {
        let keys = MLXArray(
            (start ..< start + count).map { Float($0) }
        ).reshaped(1, 1, count, 1)
        return (keys, keys + 1_000)
    }

    @Test func saturatedRotatingCacheDropsRejectedTailInTemporalOrder() {
        guard runtimeTestsEnabled else { return }

        let cache = RotatingKVCache(maxSize: 5, keep: 0)
        for token in 0 ..< 8 {
            let (keys, values) = values(token, 1)
            _ = cache.update(keys: keys, values: values)
        }
        let (verifyKeys, verifyValues) = values(8, 3)
        _ = cache.update(keys: verifyKeys, values: verifyValues)
        eval(cache)

        #expect(cache.offset == 11)
        #expect(cache.trimRecent(2) == 2)
        eval(cache)

        #expect(cache.offset == 9)
        #expect(cache.metaState.last == "5")
        #expect(cache.state[0].asArray(Float.self) == [4, 5, 6, 7, 8])
        #expect(cache.state[1].asArray(Float.self) == [1004, 1005, 1006, 1007, 1008])
    }

    @Test func legacyTrimProviderRemainsButSessionInstallsCBv2Transaction() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let gemma = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift"),
            encoding: .utf8)
        let session = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift"),
            encoding: .utf8)

        #expect(gemma.contains(
            "Gemma4TextModel: DFlashTargetModel, DFlashTargetCacheRollbackProvider"))
        #expect(gemma.contains(
            "cache.forEach { _ = $0.trimRecent(rejectedTokenCount) }"))
        #expect(session.contains("Gemma4DFlashCBv2TargetCache("))
        #expect(session.contains("targetCacheTransaction: targetCacheTransaction"))
        #expect(!session.contains("targetCache.allSatisfy({ $0.isRecentTrimmable })"))
        #expect(!session.contains("target.newCache(parameters: nil)"))
    }

    @Test func greedyOrderOnlyScopeIsExactAndRestoresItsCaller() {
        CBv2OrderOnlyLogits.set(false)
        let outer = CBv2OrderOnlyLogits.withGreedyOrderOnly {
            #expect(CBv2OrderOnlyLogits.engaged)
            return CBv2OrderOnlyLogits.withGreedyOrderOnly {
                CBv2OrderOnlyLogits.engaged
            }
        }
        #expect(outer)
        #expect(!CBv2OrderOnlyLogits.engaged)

        CBv2OrderOnlyLogits.set(true)
        let nested = CBv2OrderOnlyLogits.withGreedyOrderOnly {
            CBv2OrderOnlyLogits.engaged
        }
        #expect(nested)
        #expect(CBv2OrderOnlyLogits.engaged)
        CBv2OrderOnlyLogits.set(false)
    }

    @Test func dflashGreedyTargetDeclaresOrderOnlyConsumption() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4TextDFlash.swift"),
            encoding: .utf8)

        let forwardStart = try #require(source.range(
            of: "    public func forwardGreedyTokensForDFlash("))
        let forwardEnd = try #require(source.range(
            of: "    public func embedTokensForDFlash(",
            range: forwardStart.upperBound..<source.endIndex))
        let forward = String(source[forwardStart.lowerBound..<forwardEnd.lowerBound])
        #expect(forward.contains("CBv2OrderOnlyLogits.withGreedyOrderOnly"))
        #expect(forward.contains("applyLMHead(forward.postNorm, verifier: verifier)"))
    }

    @Test func dflashGreedyDrafterSkipsOnlyTheMonotonicSoftcap() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift"),
            encoding: .utf8)

        #expect(source.contains("applyFinalLogitSoftcap: Bool"))
        #expect(source.contains("if applyFinalLogitSoftcap, let cap"))
        #expect(source.contains("applyFinalLogitSoftcap: false"))
        #expect(source.contains("applyFinalLogitSoftcap: true"))
    }
}
