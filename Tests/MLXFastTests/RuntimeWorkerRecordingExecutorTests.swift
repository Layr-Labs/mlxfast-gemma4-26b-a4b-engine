import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// RECORDING-EXECUTOR IDENTITY for the CBv2-backed reference-tape recorder
// (`record-reference-tape --recording-backend cbv2`).
//
// The pool tapes gate the paired FreeRunV1_1 legs, and since the leg-identity
// fix (PR #29) BOTH legs run the width-1 CBv2 engine session. A tape recorded
// through any other implementation reintroduces the two-implementations-one-
// gate defect the fix removed (docs/gemma4-port-notes.md 3.1 / 5.1,
// within-backend), so the recording verbs must drive the SAME begin executor
// the wire legs dispatch to — `openSingleStreamFreeRunSession`, serial
// drafter-less configuration — with exactly one recording-only difference:
// the engine-side top-2 readout (`recordingTopLogprobs: 2`), which is
// observability over the RAW logits and must not move the argmax.
//
// Mirrors the LegIdentity test pattern (0640a2a/9debd0c): the MLX-gated test
// drives the REAL executors over a topology-conformant weight-free fixture;
// the pure tests cover the model-free row assembly.

@Suite("RuntimeWorkerRecordingExecutor", .serialized)
struct RuntimeWorkerRecordingExecutorTests {

    private let vocabSize = 64
    private let hiddenSize = 32

    private func topologyConformantTargetConfig() throws -> Gemma4TextConfiguration {
        let layerTypes = (0 ..< MLXFastConstants.numHiddenLayers).map { index in
            index % MLXFastConstants.fullAttentionInterval
                == MLXFastConstants.fullAttentionInterval - 1
                ? "\"full_attention\"" : "\"sliding_attention\""
        }.joined(separator: ", ")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": [\(layerTypes)],
                "sliding_window": \(MLXFastConstants.slidingWindow),
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

    private func promptTokens(length: Int, seed: Int) -> [Int] {
        var value = UInt64(bitPattern: Int64(seed)) &+ 0x9E3779B97F4A7C15
        return (0 ..< length).map { _ in
            value = value &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(value % UInt64(vocabSize))
        }
    }

    /// Drive the REAL recording executors — the shared begin executor with
    /// the recording readout enabled, then the session slot-move and drain
    /// the `record_reference_begin`/`record_reference_run` arms perform —
    /// and return the committed stream, the assembled per-row top-2 arrays,
    /// and the session diagnostics.
    private func runRecordingExecutors(
        model: Gemma4TextModel, prompt: [Int], n: Int
    ) throws -> (
        stream: [Int],
        top2Tokens: [[Int]],
        top2Logits: [[Double]],
        diagnostics: RuntimeWorkerFreeRunSessionDiagnostics
    ) {
        var state = RuntimeWorkerState()
        let seed = try Gemma4Runtime.openSingleStreamFreeRunSession(
            route: .serial,
            seedTokens: prompt,
            model: model,
            mtpDrafter: nil,
            requestedDepth: nil,
            stopTokens: [],
            state: &state,
            recordingTopLogprobs: 2)
        // The record_reference_begin arm's slot move.
        state.recordingSession = state.freeRunSession
        state.freeRunSession = nil
        let session = try #require(state.recordingSession)
        state.recordingSession = nil
        let result = try session.run(targetN: n)
        let rows = try Gemma4Runtime.assembleRecordingTopTwoRows(
            tokens: result.tokens,
            tokenLogprobs: session.tokenLogprobsSnapshot())
        let diagnostics = Gemma4Runtime.makeFreeRunSessionDiagnostics(
            route: .serial,
            session: session,
            state: state,
            result: result)
        return ([seed] + result.tokens, rows.top2Tokens, rows.top2Logits, diagnostics)
    }

    /// The recording executor IS the leg executor: same engine construction,
    /// same session type, same drain — and the recording readout must not
    /// move a single token relative to the wire-leg configuration
    /// (`topLogprobs` 0) on the same weights.
    ///
    /// Requires `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` plus
    /// `tools/build-mlx-metallib.sh --all-build-roots` (repo convention).
    @Test
    func recordingModeDrivesTheSharedEngineExecutorWithoutMovingTheStream() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        MLXRandom.seed(0x1E6_1D)
        let model = Gemma4TextModel(try topologyConformantTargetConfig())
        eval(model)
        let prompt = promptTokens(length: 14, seed: 11)
        let n = 32

        // The wire-leg configuration: the SAME executor with no recording
        // readout (what free_decode_begin opens for the serial leg).
        let legEngine = try Gemma4Runtime.makeCohortEngine(
            model: model, batchSize: 1, seedTokenCount: prompt.count,
            maxTokensPerStream: n + 8)
        let legSession = try RuntimeWorkerFreeRunSession(
            engine: legEngine, mode: .serial, seedTokens: prompt,
            maxTokens: n + 8, stopTokens: [])
        let legResult = try legSession.run(targetN: n)
        let legStream = [legSession.seedToken] + legResult.tokens

        let recording = try runRecordingExecutors(
            model: model, prompt: prompt, n: n)

        // GUARD: enabling the top-2 readout must not move the argmax —
        // token-identical streams on the same weights, or the recorded tape
        // would describe a third implementation.
        #expect(
            recording.stream == legStream,
            "recording readout moved the committed stream relative to the wire leg")

        // The per-row diagnostics are real and internally consistent: one
        // top-2 pair per row, argmax first, descending values.
        #expect(recording.top2Tokens.count == n)
        #expect(recording.top2Logits.count == n)
        for index in 0 ..< n {
            #expect(recording.top2Tokens[index].count == 2)
            #expect(recording.top2Tokens[index][0] == recording.stream[index + 1])
            #expect(recording.top2Logits[index].count == 2)
            #expect(recording.top2Logits[index][0] >= recording.top2Logits[index][1])
            #expect(recording.top2Logits[index].allSatisfy { $0.isFinite })
        }

        // EXECUTOR IDENTITY: the recording session reports the same
        // engine-backed executor as both wire legs — the identity the tape's
        // session evidence carries.
        #expect(
            recording.diagnostics.executor
                == RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine,
            """
            recording-session diagnostics report executor \
            '\(recording.diagnostics.executor)'; the tape must be produced by \
            '\(RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine)' — the \
            executor both measuring legs run
            """)
    }
}

// MARK: - Pure row assembly (model-free)

private func logprobReadout(
    token: Int, top: [(Int, Float)]
) -> CBv2TokenLogprob {
    CBv2TokenLogprob(
        token: token,
        logprob: top.first?.1 ?? 0,
        topLogprobs: top.map { (token: $0.0, logprob: $0.1) })
}

@Test
func recordingRowAssemblyMapsCollectorOrderToRows() throws {
    // Collector order: index 0 is the seed token (no tape row), rows follow.
    let rows = try Gemma4Runtime.assembleRecordingTopTwoRows(
        tokens: [5, 9],
        tokenLogprobs: [
            logprobReadout(token: 3, top: [(3, -0.5), (1, -4.0)]),  // seed
            logprobReadout(token: 5, top: [(5, -0.25), (2, -1.5)]),
            logprobReadout(token: 9, top: [(9, -0.125), (5, -2.25)]),
        ])
    #expect(rows.top2Tokens == [[5, 2], [9, 5]])
    #expect(rows.top2Logits == [[-0.25, -1.5], [-0.125, -2.25]])
}

@Test
func recordingRowAssemblyFailsClosedOnMissingOrMisalignedReadouts() {
    func expectRejected(
        tokens: [Int], readouts: [CBv2TokenLogprob?], naming needle: String
    ) {
        do {
            _ = try Gemma4Runtime.assembleRecordingTopTwoRows(
                tokens: tokens, tokenLogprobs: readouts)
            Issue.record("row assembly must reject: \(needle)")
        } catch {
            #expect("\(error)".contains(needle), "\(error)")
        }
    }
    let seed = logprobReadout(token: 3, top: [(3, -0.5), (1, -4.0)])
    let row = logprobReadout(token: 5, top: [(5, -0.25), (2, -1.5)])
    // Short coverage: the session did not observe every forward.
    expectRejected(
        tokens: [5, 9], readouts: [seed, row],
        naming: "did not observe every forward")
    // A row with no readout at all.
    expectRejected(
        tokens: [5], readouts: [seed, nil],
        naming: "no top-logprob readout")
    // A readout describing a different token than the engine committed.
    expectRejected(
        tokens: [7], readouts: [seed, row],
        naming: "collector misalignment")
    // Fewer than 2 alternatives cannot fill a tape row.
    expectRejected(
        tokens: [5],
        readouts: [seed, logprobReadout(token: 5, top: [(5, -0.25)])],
        naming: "needs the top 2")
}
