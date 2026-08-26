import Dispatch
import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

// MARK: - Hybrid cache-position validation

/// Synthetic stand-in for `Gemma4TextModel.newCache(parameters: nil)`: one
/// cache per layer, `StandardKVCache` (unbounded) on the 5 global layers
/// (index % 6 == 5) and `RotatingKVCache(maxSize: slidingWindow)` on the 25
/// sliding ones. Constructing these allocates no MLXArray and loads no model,
/// so the hybrid contract is testable on any machine.
private func syntheticGemmaHybridCaches(
    offset: Int,
    slidingOffset: Int? = nil,
    layerCount: Int = MLXFastConstants.numHiddenLayers,
    homogeneous: Bool = false
) -> [BaseKVCache] {
    let interval = MLXFastConstants.fullAttentionInterval
    return (0..<layerCount).map { index in
        let isGlobal = homogeneous || index % interval == interval - 1
        let cache: BaseKVCache = isGlobal
            ? StandardKVCache()
            : RotatingKVCache(maxSize: MLXFastConstants.slidingWindow, keep: 0)
        cache.offset = isGlobal ? offset : (slidingOffset ?? offset)
        return cache
    }
}

/// RE-DERIVED 2026-08-22 for the Gemma tower. The Qwen form of this suite
/// asserted the OPPOSITE rule -- that the non-global caches stay pinned at
/// offset 0, because Qwen's gated-delta layers hold recurrent state. Gemma 4
/// has no recurrent layers: both of its cache classes count every position
/// written (the ring lives in `RotatingKVCache.idx`), so the stack is lockstep
/// and a sliding cache pinned at 0 after a prefill is now the ERROR case. The
/// original regression this suite guards is unchanged in spirit: the gate must
/// agree with what the model's own `newCache` produces, or the first forward
/// after a prefill dies.
@Test
func gemmaHybridCachePositionAcceptsTheLockstepStack() throws {
    // Fresh cache before the first forward.
    try Gemma4Runtime.verifyQwenCachePosition(
        positionOffset: 0,
        cache: syntheticGemmaHybridCaches(offset: 0)
    )
    // After a 512-token prefill: every one of the 30 caches reads 512.
    try Gemma4Runtime.verifyQwenCachePosition(
        positionOffset: 512,
        cache: syntheticGemmaHybridCaches(offset: 512)
    )
    // And after each subsequent teacher-forced decode step.
    for step in 1...4 {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 512 + step,
            cache: syntheticGemmaHybridCaches(offset: 512 + step)
        )
    }
    // Past the sliding window: `offset` keeps counting positions even once the
    // 1024-entry ring has wrapped, so the lockstep rule still holds. This is
    // exactly the case a rule written around `idx` would get wrong.
    try Gemma4Runtime.verifyQwenCachePosition(
        positionOffset: MLXFastConstants.slidingWindow + 37,
        cache: syntheticGemmaHybridCaches(
            offset: MLXFastConstants.slidingWindow + 37)
    )
}

@Test
func gemmaHybridCachePositionRejectsDivergentGlobalOffsets() {
    let caches = syntheticGemmaHybridCaches(offset: 512)
    // Layer 11 is global; make it lag the rest of the stack.
    caches[11].offset = 511
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(positionOffset: 512, cache: caches)
    }
}

@Test
func gemmaHybridCachePositionRejectsSlidingCachesThatDidNotAdvance() {
    // The Qwen tower's recurrent caches legitimately stayed at 0; Gemma's
    // sliding caches never do. A stack that looks like the old one is a stale
    // or half-advanced stack here.
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticGemmaHybridCaches(offset: 512, slidingOffset: 0)
        )
    }
}

@Test
func gemmaHybridCachePositionRejectsWrongTopologyOrLayerCount() {
    // An all-unbounded stack is lockstep-consistent and would satisfy the
    // offset rule, but it is not this tower: 25 layers must be windowed.
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticGemmaHybridCaches(offset: 512, homogeneous: true)
        )
    }
    // Right topology, wrong window on the sliding layers.
    let miswindowed = syntheticGemmaHybridCaches(offset: 512)
    let replaced = miswindowed.enumerated().map { index, cache -> BaseKVCache in
        guard index % MLXFastConstants.fullAttentionInterval
            != MLXFastConstants.fullAttentionInterval - 1
        else { return cache }
        let narrower = RotatingKVCache(
            maxSize: MLXFastConstants.slidingWindow / 2, keep: 0)
        narrower.offset = 512
        return narrower
    }
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 512, cache: replaced)
    }
    // Wrong layer count, right topology.
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticGemmaHybridCaches(offset: 512, layerCount: 40)
        )
    }
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(positionOffset: 0, cache: [])
    }
}

@Test
func gemmaHybridCachePositionStillRejectsAStaleOrReusedCache() {
    // The fail-loudly contract the Laguna check existed for is preserved: the
    // caller's position must equal what the KV stack actually holds.
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: 513,
            cache: syntheticGemmaHybridCaches(offset: 512)
        )
    }
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.verifyQwenCachePosition(
            positionOffset: -1,
            cache: syntheticGemmaHybridCaches(offset: 0)
        )
    }
}

/// The gate and the cache check must read the same schedule; a tower whose
/// interval changed would otherwise pass one and fail the other.
@Test
func qwenFullAttentionIntervalMatchesThePinnedLayerSchedule() {
    #expect(MLXFastConstants.fullAttentionInterval == 6)
    #expect(MLXFastConstants.numHiddenLayers % MLXFastConstants.fullAttentionInterval == 0)
    let fullAttentionLayers = (0..<MLXFastConstants.numHiddenLayers).filter {
        $0 % MLXFastConstants.fullAttentionInterval
            == MLXFastConstants.fullAttentionInterval - 1
    }
    // Gemma 4 26B A4B: 30 layers on a six-layer repeat, so the global layers
    // are 5/11/17/23/29 -- five of them, against Qwen's sixteen of sixty-four.
    #expect(fullAttentionLayers.count == 5)
    #expect(fullAttentionLayers.first == 5)
    #expect(fullAttentionLayers.last == 29)
    #expect(fullAttentionLayers == [5, 11, 17, 23, 29])
}

@Test
func gemma4CorrectnessSelectsGreedyTokenWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    #expect(try Gemma4Correctness.greedyToken(
        from: MLXArray([Float(0.1), 2.0, 1.0], [3])
    ) == 1)
    #expect(try Gemma4Correctness.greedyToken(
        from: MLXArray([Float(1), 2, 3, 2], [2, 2])
    ) == 0)
}

// The worker's orphan self-reaper is what frees the ~21.6 GB model residency
// when the harness parent dies while the worker is not blocked on protocol
// stdin (most importantly during the minutes-long model load).
@Test
func runtimeWorkerOrphanReaperFiresOnceTheParentIsGone() {
    let fired = DispatchSemaphore(value: 0)
    let thread = Gemma4Runtime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { true },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 2) == .success)
}

@Test
func runtimeWorkerOrphanReaperStaysQuietWhileTheParentIsAlive() {
    let fired = DispatchSemaphore(value: 0)
    let thread = Gemma4Runtime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { false },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 0.3) == .timedOut)
}

@Test
func phaseStartAllocatorResetLeavesExactlyEmptyCacheWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    Memory.cacheLimit = 32 << 30
    do {
        let scratch = MLXArray(Array(repeating: Float(1), count: 1 << 20), [1024, 1024])
        eval(scratch + scratch)
    }
    try Gemma4Runtime.resetRuntimeWorkerAllocatorForPhaseStart()
    #expect(Memory.cacheMemory == 0)
    #expect(Memory.cacheLimit == Gemma4Runtime.trustedRuntimeWorkerPhaseStartCacheLimitBytes)
}

@Test
func traceProtocolCarriesRequestedTopKAndExpectedTokenDiagnostics() throws {
    let request = RuntimeWorkerRequest(
        id: 4,
        kind: "correctness_step",
        token: 7,
        topK: 16,
        expectedToken: 23
    )
    let requestObject = try #require(
        JSONSerialization.jsonObject(
            with: JSONEncoder().encode(request)
        ) as? [String: Any]
    )
    #expect(requestObject["top_k"] as? Int == 16)
    #expect(requestObject["expected_token"] as? Int == 23)

    let response = RuntimeWorkerResponse(
        id: 4,
        nonce: "nonce",
        ok: true,
        token: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 9),
        ],
        expectedTokenLogit: 3.5,
        expectedTokenRank: 19,
        topLogitMargin: 0.25
    )
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self,
        from: JSONEncoder().encode(response)
    )
    #expect(decoded.expectedTokenLogit == 3.5)
    #expect(decoded.expectedTokenRank == 19)
    #expect(decoded.topLogitMargin == 0.25)
}

@Test
func workerComputesExpectedTokenDiagnosticsOutsideReturnedTopSubset() throws {
    let diagnostics = try Gemma4Runtime.correctnessLogitDiagnostics(
        values: (0..<20).map { Double(20 - $0) },
        topK: 12,
        expectedToken: 19
    )
    #expect(diagnostics.topLogits.count == 12)
    #expect(!diagnostics.topLogits.contains { $0.token == 19 })
    #expect(diagnostics.expectedTokenLogit == 1)
    #expect(diagnostics.expectedTokenRank == 20)
    #expect(diagnostics.topLogitMargin == 1)
}
