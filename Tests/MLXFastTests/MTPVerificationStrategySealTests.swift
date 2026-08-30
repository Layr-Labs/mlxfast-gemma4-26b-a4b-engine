import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// TOKEN-EXACTNESS SEAL for the MTP verify strategy (2026-08-25).
//
// The first live single-stream v1.1 MTP run (benchd FreeRunV1_1, depth 2,
// engine 8c0bfec2) failed token-exactness on 7 of 8 hidden prompts at a
// DETERMINISTIC prompt-specific step, committing text-equivalent
// boundary-shifted token variants of the serial oracle's tokens (147083
// 'chamber' vs 18782 ' chamber'; 1345 ' end' vs 643 'end'; 1018 '**' vs
// 236779 '_'), and committing an early stop token mid-window on the eighth.
// That signature is a near-tie argmax flip inside the drafting rounds'
// verify forward, not a KV/position accounting error: every structural
// boundary (full-acceptance + bonus, partial acceptance, zero-acceptance
// rollback truncation, ring-wrap staged transactions, stop-token clamp) was
// exercised token-exact through the REAL vendored engine while diagnosing
// this (see `acceptancePatternsStayLockstepAndTokenExact` below, which keeps
// that coverage), and the committed serial-shaped rounds of the live run
// matched the oracle for tens of steps before each divergence.
//
// The defect is the sealed verification strategy. The vendored contract is
// explicit (`CBv2MTPConfig.maxAutomaticRectangularTokens`,
// MTPContractsV2.swift): "a positive envelope is the integrator's explicit
// claim that rectangular target evaluation is argmax-exact for the deployed
// chip/OS/MLX/model tuple at every shape inside it" — and this repo's own
// port notes (docs/gemma4-port-notes.md section 3.1) record the darkbloom
// finding that `[B,1]` and `[B,L]` shapes select different quantized-matmul
// reduction paths on the production QAT checkpoint and "are not
// bit-identical". `Gemma4MTPEnvelope.resolveConfig` nevertheless sealed
// `.automatic` with a positive cap (32), so every drafting round scored its
// 1+k verify columns through the UNCERTIFIED rectangular `[B, 1+k]` forward
// instead of the vendored chip-independent serial oracle ("every column
// executes the same [B, 1] eager forward used by ordinary decode ... It
// works independently of chip-specific multi-position kernel numerics").
//
// These tests seal the strategy: until a rectangular exactness certification
// exists for the deployed tuple, the production config must select
// `.serialTarget`, and a production-config single-stream session must run
// ZERO rectangular verify rounds while still doing real draft/verify work.

@Suite("MTPVerificationStrategySeal", .serialized)
struct MTPVerificationStrategySealTests {

    // MARK: - (1) GPU-free: the sealed config selects the serial oracle

    /// The production seam both free-run legs construct their engine from
    /// (`free_decode_begin` mtp arm, `handleCohortFreeDecodeBegin`) must seal
    /// the chip-independent `.serialTarget` oracle — the only strategy whose
    /// verify columns are bit-identical to ordinary serial decode BY
    /// CONSTRUCTION — while keeping the speculative apparatus fully armed
    /// (enabled, request-fixed depth, the pinned batch gate). A `.automatic` seal
    /// with a positive rectangular cap is an argmax-exactness claim for the
    /// deployed chip/OS/MLX/model tuple that this engine cannot make (no
    /// certification artifact exists; the 2026-08-25 box run refuted it).
    @Test
    func resolveConfigSealsTheChipIndependentSerialOracle() throws {
        for depth in 0 ... Gemma4MTPEnvelope.maxDraftTokens {
            let config = try Gemma4MTPEnvelope.resolveConfig(depth: depth)
            #expect(
                config.verificationMode == .serialTarget,
                """
                depth \(depth): production MTP rounds must score verify \
                columns through the vendored serial oracle until rectangular \
                verification is exactness-certified for the deployed tuple \
                (got \(config.verificationMode))
                """)
            // The seal is a strategy choice, never a disarm: real speculative
            // work stays configured exactly as before.
            #expect(config.enabled)
            #expect(config.maxDraftTokens == depth)
            #expect(config.fixedDraftTokens == depth)
            #expect(config.maxSpeculativeBatch == 8)
        }
    }

    // MARK: - Shared fixture (mirrors RuntimeWorkerMTPRoundExecutionTests)

    private let vocabSize = 64
    private let hiddenSize = 32
    /// Small window so decode + verify rounds wrap the sliding ring early —
    /// the staged-transaction regime the production 25-of-30 sliding layers
    /// live in (port-notes 5.3).
    private let slidingWindow = 12

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
            target: Gemma4TextModel, wrong: @escaping (Int, Int) -> Bool
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

    // MARK: - (2) MLX-gated: production-config rounds use ONLY the oracle

    /// Drive the REAL single-stream v1.1 session (`RuntimeWorkerMTPSession`)
    /// with the PRODUCTION-SEALED config (`Gemma4MTPEnvelope.resolveConfig`,
    /// the exact seam `free_decode_begin`'s mtp arm calls) over the
    /// weight-free fixture, with an oracle drafter so every round performs
    /// the deepest verify (full acceptance + bonus — the widest rectangular
    /// candidate shape). The engine's own strategy accounting must show ZERO
    /// rectangular verify rounds: every drafting round scored through the
    /// chip-independent serial oracle, while real draft/verify work happened
    /// (draftedTotal > 0 — the seal is not a silent disarm).
    ///
    /// Gated behind `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` plus
    /// `tools/build-mlx-metallib.sh --all-build-roots` once per checkout —
    /// this repo's standing convention for MLX-runtime tests.
    @Test
    func productionConfigSingleStreamRoundsNeverUseRectangularVerification() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        let target = try makeTarget(seed: 0x5EED_0001)
        let prompt = promptTokens(length: 14, seed: 11)
        let n = 48
        let maxTokens = n + 8

        let baseline = try serialReference(
            target: target, prompt: prompt, n: n, maxTokens: maxTokens)

        let drafter = ScriptedDrafter(
            script: baseline, promptLength: prompt.count, vocabSize: vocabSize,
            target: target, wrong: { _, _ in false })
        // THE PRODUCTION SEAM — byte-for-byte the config free_decode_begin
        // seals for a `{"mode":"mtp","mtp":{"depth":2}}` spec. Only
        // `fixedDraftTokens` is overridden, to pin the round pattern: the
        // adaptive controller's depth choices are wall-time-driven and would
        // make the exercised pattern (and so this test) nondeterministic.
        // Strategy selection is independent of that knob
        // (`mtpBuildTargetVerification` reads only verificationMode, the cap,
        // and the round's width).
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
        let metrics = try #require(engine.mtpMetricsSnapshot())

        // Real speculative work happened — the seal is a strategy choice,
        // not a disarm (the envelope header's "silent-no-spec-work" hazard).
        #expect(result.draftedTotal > 0)
        #expect(metrics.rounds > 0)

        // THE SEAL: no verify round may score through the rectangular
        // [B, 1+k] forward — the strategy whose argmax-exactness on the
        // deployed chip/OS/MLX/model tuple is uncertified (port-notes 3.1;
        // the 2026-08-25 box run's 7-of-8-prompt divergence). Every drafting
        // round must use the vendored serial oracle.
        #expect(
            metrics.rectangularVerificationRounds == 0,
            """
            \(metrics.rectangularVerificationRounds) of \(metrics.rounds) \
            verify rounds scored through the uncertified rectangular \
            strategy; production rounds must use the chip-independent \
            serial oracle
            """)
        // Strategy is recorded at round LAUNCH (`recordVerificationStrategy`,
        // graph build) while `rounds` increments at FINALIZE (`recordRound`),
        // and the session's cancel at N can cut the last launched round
        // before its finalize — so launched-strategy counts can exceed
        // finalized rounds by the in-flight tail, never trail them.
        #expect(metrics.serialVerificationRounds >= metrics.rounds)

        // Greedy losslessness at fixture scale, plus the benchd §2.6 triple.
        #expect(speculative == baseline)
        #expect(result.committedTotal == n)
        #expect(result.acceptanceLengths.reduce(0, +) == n)
        #expect(result.completedWork == result.rounds + 1)
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
            let maxTokens = n + 8

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
        }
    }
}
