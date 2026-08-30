import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Production-strategy seal for the certified physical-B1 Gemma verifier.
//
// The production artifact now installs immutable exact verifier contexts for
// B1/C2, B1/C3, and B1/C4 at construction. This suite therefore seals the
// engine-facing half of that contract: requested depths 1...3 select explicit
// rectangular verification at physical batch one, and a complete real-engine
// fixture stays lockstep with the serial verifier in every observable round
// dimension. The weight-free fixture below proves engine strategy selection,
// acceptance, commit, and KV-accounting semantics; it does not claim to
// exercise the artifact-bound exact projection kernels. The real-artifact
// route and digest gate belongs to the guarded benchmark task.

@Suite("MTPVerificationStrategySeal", .serialized)
struct MTPVerificationStrategySealTests {

    // MARK: - (1) GPU-free: the production seal selects certified B1 rectangles

    /// The same seam used by both free-run legs must select only the certified
    /// physical-B1 widths. Depth zero remains disabled and is not a production
    /// MTP route; depths 1...3 map directly to C2...C4.
    @Test
    func resolveConfigSealsCertifiedPhysicalB1Rectangles() throws {
        let disabled = try Gemma4MTPEnvelope.resolveConfig(depth: 0)
        #expect(!disabled.enabled)
        #expect(disabled.fixedDraftTokens == 0)

        for depth in 1 ... Gemma4MTPEnvelope.maxDraftTokens {
            let config = try Gemma4MTPEnvelope.resolveConfig(depth: depth)
            #expect(config.verificationMode == .rectangular)
            #expect(config.enabled)
            #expect(config.maxDraftTokens == Gemma4MTPEnvelope.maxDraftTokens)
            #expect(config.fixedDraftTokens == depth)
            #expect(config.maxSpeculativeBatch == 1)
        }
    }

    @Test
    func productionMTPRequestBoundaryRefusesPhysicalBatchAboveOne() throws {
        #expect(throws: Never.self) {
            try Gemma4Runtime.requireCertifiedMTPPhysicalBatch(1)
        }
        do {
            try Gemma4Runtime.requireCertifiedMTPPhysicalBatch(2)
            Issue.record("production Gemma MTP accepted uncertified physical B2")
        } catch {
            let message = String(describing: error)
            #expect(message.contains("physical batch 1"))
            #expect(message.contains("got 2"))
            #expect(message.contains("before engine construction"))
        }
    }

    // MARK: - Shared fixture (mirrors RuntimeWorkerMTPRoundExecutionTests)

    private let vocabSize = 64
    private let hiddenSize = 32
    /// Small window so decode + verify rounds wrap the sliding ring early —
    /// the staged-transaction regime the production 25-of-30 sliding layers
    /// live in (port-notes 5.3).
    private let slidingWindow = 12

    /// Cache-sensitive exactness fixture. A dominant token table fixes the
    /// greedy stream, while a small full/sliding attention term derived from
    /// token-valued K/V makes every raw logit row depend on committed cache
    /// contents. The serial-query rectangular attention contract therefore
    /// remains bit-exact to independent `[1, 1]` calls, while stale rollback
    /// state becomes directly observable in subsequent logits.
    private final class ExactCycleTarget: CBv2MTPSteppableModel {
        static let vocabularySize = 64
        private static let headDimension = 16
        private static let hiddenDimension = 8

        let layerKinds = [
            CBv2LayerKind(
                attention: .full, headDim: headDimension,
                kvHeads: 1, queryHeads: 1),
            CBv2LayerKind(
                attention: .slidingWindow(12), headDim: headDimension,
                kvHeads: 1, queryHeads: 1),
        ]
        let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 1)
        var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

        private let table: MLXArray
        private let recordLock = NSLock()
        private var recordedVerificationLogits: [MLXArray] = []

        init() {
            var values = [Float](
                repeating: -8,
                count: Self.vocabularySize * Self.vocabularySize)
            for token in 0 ..< Self.vocabularySize {
                values[token * Self.vocabularySize + token] = 8
            }
            table = MLXArray(
                values, [Self.vocabularySize, Self.vocabularySize])
        }

        func forward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache]
        ) -> MLXArray {
            exactForward(tokens: tokens, caches: caches, record: true).logits
        }

        func forwardWithHidden(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache]
        ) -> (logits: MLXArray, lastHidden: MLXArray) {
            exactForward(tokens: tokens, caches: caches, record: true)
        }

        func forwardRectangularVerificationWithHidden(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache]
        ) -> (logits: MLXArray, lastHidden: MLXArray) {
            exactForward(tokens: tokens, caches: caches, record: true)
        }

        private func exactForward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache], record: Bool
        ) -> (logits: MLXArray, lastHidden: MLXArray) {
            let batch = tokens.dim(0)
            let columns = tokens.dim(1)
            // Token-derived K/V makes every later logit row depend on the
            // actual committed full/sliding cache contents. A rollback that
            // leaves even one provisional token behind changes subsequent
            // attention and therefore breaks the bit-exact logit comparison.
            let qkv = broadcast(
                tokens.asType(.float32).reshaped([batch, 1, columns, 1])
                    / Float(Self.vocabularySize),
                to: [batch, 1, columns, Self.headDimension])
            var cacheDependency = MLXArray.zeros(
                [batch, columns, 1], dtype: .float32)
            for cache in caches {
                let attended = cache.updateAndAttend(
                    queries: qkv, keys: qkv, values: qkv,
                    scale: 0.25, sinks: nil)
                cacheDependency = cacheDependency
                    + attended.sum(axis: -1).squeezed(axis: 1)
                        .expandedDimensions(axis: -1)
            }
            let next = (tokens + 1) % Self.vocabularySize
            let logits = take(table, next, axis: 0) + cacheDependency * 0.001
            let hidden =
                MLXArray.zeros(
                    [batch, columns, Self.hiddenDimension], dtype: .float32)
                + cacheDependency * 0.001
            if record {
                recordLock.lock()
                recordedVerificationLogits.append(logits + 0)
                recordLock.unlock()
            }
            return (logits, hidden)
        }

        /// Flatten call grouping into one exact logit row per scored token.
        /// Serial produces C singleton calls while rectangular produces one
        /// C-column call; this normalization compares their actual values.
        func verificationLogitRows() -> [[Float]] {
            recordedLogitCalls().flatMap { $0 }
        }

        /// Preserve call grouping so committed verifier prefixes can be
        /// compared to the independent plain-AR one-call-per-token oracle.
        func recordedLogitCalls() -> [[[Float]]] {
            recordLock.lock()
            let arrays = recordedVerificationLogits
            recordLock.unlock()
            eval(arrays)
            return arrays.map { array in
                let values = array.reshaped([-1]).asArray(Float.self)
                return stride(
                    from: 0, to: values.count, by: Self.vocabularySize
                ).map {
                    Array(values[$0 ..< $0 + Self.vocabularySize])
                }
            }
        }
    }

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeTarget(seed: UInt64) throws -> Gemma4TextModel {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        eval(target)
        return target
    }

    private func promptTokens(length: Int, seed: Int) -> [Int] {
        var value = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
        return (0 ..< length).map { _ in
            value = value &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(value % UInt64(vocabSize))
        }
    }

    /// Deterministic scripted drafter: drafts the target's own recorded
    /// greedy stream, with per-(round, step) deliberate corruption to force
    /// exact acceptance patterns. The engine's finalize accept-walk compares
    /// drafts against the target's own verify argmaxes, so a correct draft is
    /// accepted and a corrupted one is rejected at exactly the chosen
    /// position — every boundary (full acceptance + bonus, partial, zero
    /// acceptance with rollback truncation) is reachable on demand.
    private final class ScriptedDrafter: CBv2MTPDrafter {
        final class Cursor: CBv2MTPPreparedCapture {
            let bases: [Int]
            var step = 0
            let round: Int
            init(bases: [Int], round: Int) {
                self.bases = bases
                self.round = round
            }
        }

        private let script: [Int]
        private let promptLength: Int
        private let vocabSize: Int
        private let wrong: (Int, Int) -> Bool
        private var roundCounter = 0
        let mtpTargetIdentity: ObjectIdentifier?

        init(
            script: [Int], promptLength: Int, vocabSize: Int,
            target: AnyObject, wrong: @escaping (Int, Int) -> Bool
        ) {
            self.script = script
            self.promptLength = promptLength
            self.vocabSize = vocabSize
            self.mtpTargetIdentity = ObjectIdentifier(target)
            self.wrong = wrong
        }

        func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
            defer { roundCounter += 1 }
            // anchor == prompt + generated - 1, so the first draft of a round
            // continues the stream at index (anchor - promptLength + 1) of
            // `script` (whose index 0 is the seed token).
            return Cursor(
                bases: rows.map { $0.anchor - promptLength + 1 },
                round: roundCounter)
        }

        func draftStep(
            tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
        ) -> (tokens: MLXArray, hidden: MLXArray) {
            let cursor = prepared as! Cursor
            defer { cursor.step += 1 }
            let ids = cursor.bases.map { base -> Int32 in
                let index = base + cursor.step
                let value = index < script.count ? script[index] : 0
                let offset = wrong(cursor.round, cursor.step) ? 1 : 0
                return Int32((value + offset) % vocabSize)
            }
            return (MLXArray(ids), hidden)
        }
    }

    /// Serial (drafter-free) reference stream through the same width-1 CBv2
    /// engine construction the worker uses — the losslessness baseline.
    private func serialReference(
        target: Gemma4TextModel, prompt: [Int], n: Int, maxTokens: Int
    ) throws -> [Int] {
        let engine = try Gemma4Runtime.makeCohortEngine(
            model: target, batchSize: 1, seedTokenCount: prompt.count,
            maxTokensPerStream: maxTokens)
        let session = try RuntimeWorkerCohortSession(
            engine: engine, seedTokensByStream: [prompt], maxTokensPerStream: maxTokens)
        let result = try session.runSerial(targetN: n)
        return [session.seedTokenByStream[0]] + result.tokensByStream[0]
    }

    // MARK: - (2) MLX-gated: serial and rectangular full rounds are lockstep

    private struct StrategyRun {
        let seedToken: Int
        let result: RuntimeWorkerFreeRunResult
        let metrics: CBv2MTPMetrics
        let logitRows: [[Float]]
        let logitCalls: [[[Float]]]

        /// The target's authoritative argmax for every verify column.
        var targetArgmaxes: [[Int]] { metrics.roundAudits.map(\.targetTokens) }
        var finalCacheLengths: [Int] {
            guard let last = metrics.roundAudits.last else { return [] }
            return [last.tokensCountAfter, last.numComputedAfter]
        }

        /// Logit rows that actually authored committed output tokens. The
        /// first call's last row emits the prompt bonus. Every later call is
        /// paired with the engine's committed width for that delta; rejected
        /// speculative suffix rows are deliberately excluded.
        var committedLogitRows: [[Float]] {
            guard let promptCall = logitCalls.first,
                let promptBonus = promptCall.last
            else { return [] }
            var rows = [promptBonus]
            for (call, committed) in zip(
                logitCalls.dropFirst(), result.acceptanceLengths)
            {
                rows.append(contentsOf: call.prefix(committed))
            }
            return rows
        }
    }

    private struct PlainExactRun {
        let stream: [Int]
        let committedLogitRows: [[Float]]
    }

    private func makeExactEngine(
        target: ExactCycleTarget,
        promptLength: Int,
        drafter: (any CBv2MTPDrafter)?,
        config: CBv2MTPConfig
    ) -> EngineV2 {
        let caches = target.layerKinds.enumerated().map {
            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
        }
        return EngineV2(
            model: target,
            layerKinds: target.layerKinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 24)),
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: CBv2DefaultSampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 2048,
                prefillChunkSize: max(512, promptLength),
                maxWaiting: 1,
                enablePrefixCache: false),
            mtpDrafter: drafter,
            mtpConfig: config)
    }

    private func runStrategy(
        _ mode: CBv2MTPVerificationMode,
        depth: Int,
        prompt: [Int],
        baseline: [Int],
        n: Int,
        maxTokens: Int,
        wrong: @escaping (Int, Int) -> Bool = { _, _ in false }
    ) throws -> StrategyRun {
        let target = ExactCycleTarget()
        let drafter = ScriptedDrafter(
            script: baseline, promptLength: prompt.count,
            vocabSize: ExactCycleTarget.vocabularySize,
            target: target, wrong: wrong)
        var config = try Gemma4MTPEnvelope.resolveConfig(depth: depth)
        config.enabled = true
        config.maxSpeculativeBatch = 1
        config.fixedDraftTokens = depth
        config.verificationMode = mode
        let engine = makeExactEngine(
            target: target, promptLength: prompt.count,
            drafter: drafter, config: config)
        try Gemma4Runtime.requireMTPActive(engine)
        let session = try RuntimeWorkerMTPSession(
            engine: engine, seedTokens: prompt, maxTokens: maxTokens, stopTokens: [])
        let result = try session.run(targetN: n)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        return StrategyRun(
            seedToken: session.seedToken, result: result, metrics: metrics,
            logitRows: target.verificationLogitRows(),
            logitCalls: target.recordedLogitCalls())
    }

    private func runPlainExactAR(
        prompt: [Int], n: Int, maxTokens: Int
    ) throws -> PlainExactRun {
        let target = ExactCycleTarget()
        let engine = makeExactEngine(
            target: target, promptLength: prompt.count,
            drafter: nil, config: CBv2MTPConfig())
        let session = try RuntimeWorkerFreeRunSession(
            engine: engine, mode: .serial, seedTokens: prompt,
            maxTokens: maxTokens, stopTokens: [])
        let result = try session.run(targetN: n)
        let calls = target.recordedLogitCalls()
        let promptBonus = try #require(calls.first?.last)
        var committedRows = [promptBonus]
        for (call, committed) in zip(calls.dropFirst(), result.acceptanceLengths) {
            committedRows.append(contentsOf: call.prefix(committed))
        }
        return PlainExactRun(
            stream: [session.seedToken] + result.tokens,
            committedLogitRows: committedRows)
    }

    /// Run every production depth once through the serial verifier and once
    /// through explicit rectangular verification. The target, prompt,
    /// scripted proposals, requested token budget, and physical batch are
    /// matched. `maxTokens == seed + N` prevents a cancelled in-flight tail,
    /// making strategy round counts and final KV accounting directly
    /// comparable.
    @Test
    func productionDepthsMatchSerialFullRoundEvidence() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let prompt = promptTokens(length: 14, seed: 11)
        let n = 36
        let maxTokens = n + 1

        for depth in 1 ... Gemma4MTPEnvelope.maxDraftTokens {
            let start = try #require(prompt.last)
            let baseline = (1 ... (n + 1)).map {
                (start + $0) % ExactCycleTarget.vocabularySize
            }
            let serial = try runStrategy(
                .serialTarget, depth: depth, prompt: prompt,
                baseline: baseline, n: n, maxTokens: maxTokens)
            let rectangular = try runStrategy(
                .rectangular, depth: depth, prompt: prompt,
                baseline: baseline, n: n, maxTokens: maxTokens)

            #expect(serial.seedToken == rectangular.seedToken, "depth \(depth)")
            #expect(serial.logitRows == rectangular.logitRows, "depth \(depth): logits")
            #expect(serial.targetArgmaxes == rectangular.targetArgmaxes, "depth \(depth)")
            #expect(
                serial.metrics.roundAudits.map(\.accepted)
                    == rectangular.metrics.roundAudits.map(\.accepted),
                "depth \(depth): accepted draft counts")
            #expect(
                serial.result.acceptanceLengths == rectangular.result.acceptanceLengths,
                "depth \(depth): committed widths")
            #expect(serial.result.tokens == rectangular.result.tokens, "depth \(depth)")
            #expect(serial.finalCacheLengths == rectangular.finalCacheLengths, "depth \(depth)")
            #expect([serial.seedToken] + serial.result.tokens == baseline, "depth \(depth)")

            #expect(serial.metrics.rounds > 0, "depth \(depth)")
            #expect(serial.metrics.rectangularVerificationRounds == 0, "depth \(depth)")
            #expect(
                serial.metrics.serialVerificationRounds == serial.metrics.rounds,
                "depth \(depth)")
            #expect(rectangular.metrics.rounds > 0, "depth \(depth)")
            #expect(
                rectangular.metrics.rectangularVerificationRounds
                    == rectangular.metrics.rounds,
                "depth \(depth)")
            #expect(rectangular.metrics.serialVerificationRounds == 0, "depth \(depth)")
            #expect(rectangular.result.draftedTotal > 0, "depth \(depth)")
            #expect(rectangular.result.committedTotal == n, "depth \(depth)")
            #expect(rectangular.result.acceptanceLengths.reduce(0, +) == n, "depth \(depth)")
        }
    }

    // MARK: - (3) MLX-gated: acceptance-pattern lockstep (hardening)

    /// The structural invariant the divergence hypothesis named, kept as a
    /// permanent regression net: across every acceptance pattern — full
    /// acceptance with bonus (commit k+1), partial acceptance (commit
    /// 1 < c <= k), zero acceptance with rollback truncation (commit 1),
    /// and mixes — the committed stream stays token-exact against the serial
    /// reference and the session's counters advance in lockstep with the
    /// commit count (sum(acceptance_lengths) == N == tokens.count,
    /// completed_work == rounds + 1). This held on the pre-fix tree too
    /// (the defect was the verify strategy's numerics, not round
    /// accounting); it pins the accounting so a future round-loop change
    /// cannot silently break it.
    @Test
    func acceptancePatternsStayLockstepAndTokenExact() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let patterns: [(String, (Int, Int) -> Bool, ClosedRange<Int>)] = [
            // (name, corrupt(round, step)?, expected max committed width)
            ("full-accept", { _, _ in false }, 3 ... 3),
            ("accept-1-of-2", { _, step in step == 1 }, 2 ... 2),
            ("reject-all", { _, _ in true }, 1 ... 1),
            ("mixed", { round, step in (round + step) % 3 == 2 }, 3 ... 3),
        ]
        for (name, wrong, expectedMaxWidth) in patterns {
            let target = try makeTarget(seed: 0x5EED_0001)
            let prompt = promptTokens(length: 14, seed: 11)
            let n = 96
            let maxTokens = n + 1

            let baseline = try serialReference(
                target: target, prompt: prompt, n: n, maxTokens: maxTokens)

            let drafter = ScriptedDrafter(
                script: baseline, promptLength: prompt.count, vocabSize: vocabSize,
                target: target, wrong: wrong)
            var config = try Gemma4MTPEnvelope.resolveConfig(depth: 2)
            config.fixedDraftTokens = 2
            let engine = try Gemma4Runtime.makeCohortEngine(
                model: target, batchSize: 1, seedTokenCount: prompt.count,
                maxTokensPerStream: maxTokens,
                mtpDrafter: drafter,
                mtpConfig: config)
            try Gemma4Runtime.requireMTPActive(engine)
            let session = try RuntimeWorkerMTPSession(
                engine: engine, seedTokens: prompt, maxTokens: maxTokens, stopTokens: [])
            let result = try session.run(targetN: n)
            let speculative = [session.seedToken] + result.tokens

            #expect(
                speculative == baseline,
                "pattern \(name): committed stream diverged from serial reference")
            #expect(result.committedTotal == n, "pattern \(name)")
            #expect(result.tokens.count == n, "pattern \(name)")
            #expect(
                result.acceptanceLengths.reduce(0, +) == n, "pattern \(name)")
            #expect(
                result.completedWork == result.rounds + 1, "pattern \(name)")
            #expect(
                result.acceptanceLengths.allSatisfy { (1 ... 3).contains($0) },
                "pattern \(name): committed width outside 1...depth+1")
            let maxWidth = result.acceptanceLengths.max() ?? 0
            #expect(
                expectedMaxWidth.contains(maxWidth),
                "pattern \(name): max committed width \(maxWidth), expected \(expectedMaxWidth)")
            #expect(result.draftedTotal >= result.acceptedTotal, "pattern \(name)")
            #expect(result.draftedTotal > 0, "pattern \(name)")

            // Bit-exact serial-vs-rectangular evidence with logits that are a
            // function of token-derived full/sliding KV contents. Partial,
            // zero, and mixed acceptance force rollback; stale provisional
            // cache state changes a subsequent authoritative logit bit.
            let exactPrompt = promptTokens(length: 14, seed: 11)
            let plainExact = try runPlainExactAR(
                prompt: exactPrompt, n: n, maxTokens: maxTokens)
            let exactBaseline = plainExact.stream
            let exactSerial = try runStrategy(
                .serialTarget, depth: 2, prompt: exactPrompt,
                baseline: exactBaseline, n: n, maxTokens: maxTokens, wrong: wrong)
            let exactRectangular = try runStrategy(
                .rectangular, depth: 2, prompt: exactPrompt,
                baseline: exactBaseline, n: n, maxTokens: maxTokens, wrong: wrong)
            #expect(
                exactSerial.logitRows == exactRectangular.logitRows,
                "pattern \(name): cache-dependent verifier logits diverged after rollback")
            #expect(
                exactSerial.result.tokens == exactRectangular.result.tokens,
                "pattern \(name): committed streams diverged after rollback")
            #expect(
                [exactRectangular.seedToken] + exactRectangular.result.tokens
                    == plainExact.stream,
                "pattern \(name): rectangular stream diverged from plain AR")
            #expect(
                exactRectangular.committedLogitRows == plainExact.committedLogitRows,
                "pattern \(name): cache-dependent committed logits diverged from plain AR")
        }
    }
}
