import MLX
import XCTest

@testable import MLXLMCommon

private final class SlowFrozenReplayModel: CBv2SteppableModel, @unchecked Sendable {
    private let model: TinyTestModel
    private let lock = NSLock()
    private var delay: TimeInterval = 0
    private var forwards = 0

    init(_ model: TinyTestModel) {
        self.model = model
    }

    func setDelay(_ value: TimeInterval) {
        lock.withLock { delay = value }
    }

    var forwardCount: Int {
        lock.withLock { forwards }
    }

    func forward(
        tokens: MLXArray,
        caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        let delay = lock.withLock { () -> TimeInterval in
            forwards += 1
            return self.delay
        }
        if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        return model.forward(tokens: tokens, caches: caches)
    }
}

final class CBv2FrozenReplayFullRowTests: XCTestCase {
    private func tensor(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped(1, 1, values.count, 1)
    }

    private func values(_ array: MLXArray) -> [Float] {
        eval(array)
        return array.reshaped([-1]).asArray(Float.self)
    }

    func testReplayReadsCachedRowsWithoutMutationThenTransitionsOnce() {
        let cachedKeys = (0 ..< 8).map(Float.init)
        let cachedValues = (0 ..< 8).map { Float(100 + $0) }
        let row = CBv2FrozenReplayFullSequenceKV(
            snapshot: (tensor(cachedKeys), tensor(cachedValues), 8),
            replayStart: 4,
            maxLength: 16,
            kvHeads: 1,
            headDim: 1)

        XCTAssertEqual(row.absoluteOffset, 4)
        XCTAssertEqual(row.retainedCount, 4)
        XCTAssertEqual(row.transitionCountForTesting, 0)

        let first = row.update(
            keys: tensor([900, 901]),
            values: tensor([1900, 1901]))
        XCTAssertEqual(row.absoluteOffset, 6)
        XCTAssertEqual(values(first.0), Array(cachedKeys.prefix(6)))
        XCTAssertEqual(values(first.1), Array(cachedValues.prefix(6)))

        let second = row.update(
            keys: tensor([902, 903]),
            values: tensor([1902, 1903]))
        XCTAssertEqual(row.absoluteOffset, 8)
        XCTAssertEqual(values(second.0), cachedKeys)
        XCTAssertEqual(values(second.1), cachedValues)
        XCTAssertFalse(row.didTransitionToAppendForTesting)

        let appended = row.update(keys: tensor([42]), values: tensor([142]))
        XCTAssertTrue(row.didTransitionToAppendForTesting)
        XCTAssertEqual(row.transitionCountForTesting, 1)
        XCTAssertEqual(values(appended.0), cachedKeys + [42])
        XCTAssertEqual(values(appended.1), cachedValues + [142])

        _ = row.update(keys: tensor([43]), values: tensor([143]))
        XCTAssertEqual(row.transitionCountForTesting, 1)
        row.rollback(1)
        XCTAssertEqual(row.absoluteOffset, 9)
        XCTAssertEqual(values(row.snapshot().keys), cachedKeys + [42])
    }

    func testFirstAppendUsesReservedSlackInsteadOfDoublingM() {
        let matched = 1_024
        let keys = MLXArray.zeros([1, 1, matched, 1], dtype: .float32)
        let values = MLXArray.zeros([1, 1, matched, 1], dtype: .float32)
        let row = CBv2FrozenReplayFullSequenceKV(
            snapshot: (keys, values, matched),
            replayStart: matched - 1,
            maxLength: 2_000,
            kvHeads: 1,
            headDim: 1)
        _ = row.update(
            keys: MLXArray.zeros([1, 1, 1, 1], dtype: .float32),
            values: MLXArray.zeros([1, 1, 1, 1], dtype: .float32))
        _ = row.update(
            keys: MLXArray.ones([1, 1, 1, 1], dtype: .float32),
            values: MLXArray.ones([1, 1, 1, 1], dtype: .float32))
        let bytesPerToken = 2 * MemoryLayout<Float>.size
        XCTAssertEqual(
            row.byteCount,
            (matched + CBv2FullSequenceKV.initialSlack) * bytesPerToken)
        XCTAssertLessThan(row.byteCount, matched * 2 * bytesPerToken)
    }
}

final class CBv2FrozenReplayCounterexampleTests: XCTestCase {
    private struct Run {
        let state: [CBv2SequenceKV?]
        var logits: MLXArray
    }

    private func process(
        model: TinyTestModel,
        state: [CBv2SequenceKV?],
        tokens: ArraySlice<Int>
    ) -> MLXArray {
        let bank = CBv2LayerCacheBank(layerKinds: model.layerKinds)
        var logits = MLXArray.zeros([1, 1, model.vocabularySize])
        for token in tokens {
            logits = model.forward(
                tokens: MLXArray([Int32(token)]).reshaped(1, 1),
                caches: bank.layerCaches(rowStates: [state]))
            eval(logits)
        }
        return logits
    }

    private func coldRun(model: TinyTestModel, prompt: [Int], extra: Int = 128) throws -> Run {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 30))
        let state = try backend.makeSequenceState(
            layerKinds: model.layerKinds,
            promptLength: prompt.count,
            maxLength: prompt.count + extra)
        return Run(
            state: state,
            logits: process(model: model, state: state, tokens: prompt[...]))
    }

    private func oldMutatingResume(
        model: TinyTestModel,
        prompt: [Int],
        donor: [CBv2SequenceKV?],
        matched: Int,
        replayStart: Int,
        extra: Int = 128
    ) throws -> Run {
        let state: [CBv2SequenceKV?] = try zip(model.layerKinds, donor).map {
            kind, donorRow in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return CBv2WindowedSequenceKV(
                    window: window,
                    kvHeads: kind.kvHeads,
                    headDim: kind.headDim,
                    initialOffset: replayStart)
            case .full:
                let snapshot = try XCTUnwrap(donorRow?.snapshot())
                let row = CBv2FullSequenceKV(
                    promptLength: replayStart,
                    maxLength: prompt.count + extra,
                    kvHeads: kind.kvHeads,
                    headDim: kind.headDim)
                _ = row.update(
                    keys: snapshot.keys[.ellipsis, 0 ..< replayStart, 0...],
                    values: snapshot.values[.ellipsis, 0 ..< replayStart, 0...])
                return row
            }
        }
        return Run(
            state: state,
            logits: process(model: model, state: state, tokens: prompt[replayStart...]))
    }

    private func frozenResume(
        model: TinyTestModel,
        prompt: [Int],
        donor: [CBv2SequenceKV?],
        matched: Int,
        extra: Int = 128
    ) throws -> Run {
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: model.layerKinds,
            backend: .contiguousUnquantized)
        let plan = try XCTUnwrap(capability.plan(matchedBoundary: matched))
        let prefix = zip(model.layerKinds, donor).map {
            kind, donorRow -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard kind.sharesKVWithLayer == nil,
                case .full = kind.attention,
                let snapshot = donorRow?.snapshot()
            else { return nil }
            return (
                snapshot.keys[.ellipsis, 0 ..< matched, 0...],
                snapshot.values[.ellipsis, 0 ..< matched, 0...],
                matched)
        }
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 30))
        let state = try backend.makeSequenceState(
            adopting: prefix,
            plan: plan,
            layerKinds: model.layerKinds,
            maxLength: prompt.count + extra)
        return Run(
            state: state,
            logits: process(model: model, state: state, tokens: prompt[plan.replayStart...]))
    }

    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        eval(lhs, rhs)
        return zip(
            lhs.reshaped([-1]).asArray(Float.self),
            rhs.reshaped([-1]).asArray(Float.self)
        ).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    private func token(_ logits: MLXArray) -> Int {
        eval(logits)
        return logits[0, -1].argMax().item(Int.self)
    }

    private func assertFullKVExact(
        _ lhs: [CBv2SequenceKV?],
        _ rhs: [CBv2SequenceKV?],
        kinds: [CBv2LayerKind],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        for (index, kind) in kinds.enumerated()
        where kind.sharesKVWithLayer == nil {
            guard case .full = kind.attention else { continue }
            let left = try XCTUnwrap(lhs[index]?.snapshot(), file: file, line: line)
            let right = try XCTUnwrap(rhs[index]?.snapshot(), file: file, line: line)
            XCTAssertEqual(left.offset, right.offset, file: file, line: line)
            XCTAssertEqual(maxAbsDiff(left.keys, right.keys), 0, accuracy: 0, file: file, line: line)
            XCTAssertEqual(
                maxAbsDiff(left.values, right.values), 0, accuracy: 0,
                file: file, line: line)
        }
    }

    func testFirstPersistentFullLayerDivergenceAndFrozenExactness() throws {
        let model = TinyTestModel.make(
            seed: 0xD00D_F00D,
            stackedSlidingFull: true,
            windowSize: 16)
        let prompt = makePromptTokens(length: 73, seed: 0x5EED)
        let cold = try coldRun(model: model, prompt: prompt)
        let matched = 72
        let replayStart = 40
        let old = try oldMutatingResume(
            model: model,
            prompt: prompt,
            donor: cold.state,
            matched: matched,
            replayStart: replayStart)
        let frozen = try frozenResume(
            model: model,
            prompt: prompt,
            donor: cold.state,
            matched: matched)

        XCTAssertGreaterThan(maxAbsDiff(cold.logits, old.logits), 0)
        XCTAssertEqual(maxAbsDiff(cold.logits, frozen.logits), 0, accuracy: 0)
        try assertFullKVExact(cold.state, frozen.state, kinds: model.layerKinds)

        // Layer 3 is the first storage-owning full layer after the two
        // sliding rows. Old replay permanently writes divergent K/V there.
        let coldLayer = try XCTUnwrap(cold.state[3]?.snapshot())
        let oldLayer = try XCTUnwrap(old.state[3]?.snapshot())
        XCTAssertGreaterThan(maxAbsDiff(coldLayer.keys, oldLayer.keys), 0)
        XCTAssertGreaterThan(maxAbsDiff(coldLayer.values, oldLayer.values), 0)
    }

    func testTenPrefixLengthsAround256TokenBoundariesAndDivergentTails() throws {
        let model = TinyTestModel.make(
            seed: 0xD00D_F00D,
            stackedSlidingFull: true,
            windowSize: 16)
        let lengths = [257, 258, 511, 512, 513, 767, 768, 769, 1024, 1025]
        for length in lengths {
            let prompt = makePromptTokens(
                length: length,
                seed: UInt64(10_000 + length))
            let cold = try coldRun(model: model, prompt: prompt)
            let matched = ((length - 1) / 256) * 256
            let frozen = try frozenResume(
                model: model,
                prompt: prompt,
                donor: cold.state,
                matched: matched)
            XCTAssertEqual(
                maxAbsDiff(cold.logits, frozen.logits), 0, accuracy: 0,
                "length \(length), M \(matched)")
            try assertFullKVExact(cold.state, frozen.state, kinds: model.layerKinds)
        }

        let common = makePromptTokens(length: 512, seed: 0xCAFE)
        let donorPrompt = common + makePromptTokens(length: 17, seed: 0xD0A0)
        let requestPrompt = common + makePromptTokens(length: 9, seed: 0xBEEF)
        let donor = try coldRun(model: model, prompt: donorPrompt)
        let cold = try coldRun(model: model, prompt: requestPrompt)
        let frozen = try frozenResume(
            model: model,
            prompt: requestPrompt,
            donor: donor.state,
            matched: 512)
        XCTAssertEqual(maxAbsDiff(cold.logits, frozen.logits), 0, accuracy: 0)
        try assertFullKVExact(cold.state, frozen.state, kinds: model.layerKinds)
    }

    func testMixedPositionBatchSizesTwoAndFourRemainExact() throws {
        let model = TinyTestModel.make(
            seed: 0xD00D_F00D,
            stackedSlidingFull: true,
            windowSize: 16)
        let lengths = [513, 769, 1025, 258]

        for batchSize in [2, 4] {
            var coldRuns: [Run] = []
            var frozenRuns: [Run] = []
            for index in 0 ..< batchSize {
                let prompt = makePromptTokens(
                    length: lengths[index],
                    seed: UInt64(0x9000 + index))
                let cold = try coldRun(model: model, prompt: prompt)
                let matched = ((prompt.count - 1) / 256) * 256
                coldRuns.append(cold)
                frozenRuns.append(try frozenResume(
                    model: model,
                    prompt: prompt,
                    donor: cold.state,
                    matched: matched))
            }

            let next = coldRuns.map { token($0.logits) }
            func batchStep(_ states: [[CBv2SequenceKV?]]) -> MLXArray {
                let bank = CBv2LayerCacheBank(layerKinds: model.layerKinds)
                let logits = model.forward(
                    tokens: MLXArray(next.map(Int32.init)).reshaped(batchSize, 1),
                    caches: bank.layerCaches(rowStates: states))
                eval(logits)
                return logits
            }
            let coldBatch = batchStep(coldRuns.map(\.state))
            let frozenBatch = batchStep(frozenRuns.map(\.state))
            XCTAssertEqual(maxAbsDiff(coldBatch, frozenBatch), 0, accuracy: 0)
        }
    }

    func testCancellationDuringReplayPublishesNoPartialStateOrReservation() async throws {
        let tiny = TinyTestModel.make(
            seed: 0xD00D_F00D,
            stackedSlidingFull: true,
            windowSize: 16)
        let slow = SlowFrozenReplayModel(tiny)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 30))
        let cache = PrefixCacheV2(
            config: .init(
                blockSize: 8,
                promptContractID: "cancel-frozen-replay",
                materializeOnDonate: true))
        let engine = EngineV2(
            model: slow,
            layerKinds: tiny.layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: tiny.layerKinds),
            schedulerConfig: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16,
                maxWaiting: 4,
                enablePrefixCache: true),
            prefixCache: cache)
        let prompt = makePromptTokens(length: 73, seed: 0xCACE)
        let donor = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1),
                    promptTokens: prompt,
                    sampling: .init(temperature: 0),
                    maxTokens: 8)))
        XCTAssertEqual(donor.finishReason, .length)
        let donated = await cbv2SchedWait { cache.stats().entryCount > 0 }
        XCTAssertTrue(donated)

        let previousForwards = slow.forwardCount
        slow.setDelay(0.2)
        let stream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(2),
                promptTokens: prompt,
                sampling: .init(temperature: 0),
                maxTokens: 64))
        let replayStarted = await cbv2SchedWait(timeoutSeconds: 2) {
            slow.forwardCount > previousForwards
        }
        XCTAssertTrue(replayStarted)
        engine.cancel(CBv2RequestID(2))
        let cancelled = await cbv2SchedCollect(stream)
        slow.setDelay(0)
        XCTAssertEqual(cancelled.finishReason, .cancelled)
        XCTAssertEqual(cancelled.usage?.prefixCachePrefillTokensSaved, 40)
        let drained = await cbv2SchedWait {
            engine.capacity().activeRequests == 0 && backend.bytesReserved == 0
        }
        XCTAssertTrue(drained)

        await engine.shutdown()
        cache.evict(toFit: 0)
        XCTAssertEqual(cache.bytesInUse, 0, "cancelled replay must release the lookup pin")
        XCTAssertEqual(backend.bytesReserved, 0)
    }
}
