import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// LEG-IMPLEMENTATION-IDENTITY invariant for the v1.1 free-run pair
// (2026-08-25, exactness round two).
//
// benchd's paired FreeRunV1_1 measurement drives `free_decode_begin` /
// `free_decode_run` twice per pair: once with `{"mode":"serial"}` (the
// baseline leg) and once with `{"mode":"mtp",...}` (the candidate leg), and
// oracle-checks BOTH committed streams against the same pinned reference
// tape. A token-exactness gate over that triple is only sound if all three
// streams are computed by ONE decode implementation: the pinned tapes are
// recorded through the teacher-forced correctness verbs (legacy
// `model.newCache` single-token loop — Gemma4RuntimeReferenceTape.swift), the
// serial leg historically ran that same legacy loop, and the mtp leg runs
// the width-1 CBv2 engine. Near-tie argmaxes need not agree across two
// different implementations of the same model on the production
// chip/checkpoint tuple (docs/gemma4-port-notes.md section 3.1; the repo's
// own within-backend rule, section 5.1: "the speculative leg must be
// token-exact against a serial control running the same backend") — so a
// pairing that mixes implementations can fail the oracle deterministically
// with no acceptance/rollback defect at all, which is exactly the divergence
// signature the 2026-08-25 box runs sealed (byte-identical divergences at
// identical prompt-specific steps across two different verify-strategy
// builds).
//
// The invariant, enforced here against the REAL wire route executors (the
// exact functions `free_decode_begin`/`free_decode_run` dispatch to): both
// v1.1 free-run legs must compute their committed streams through the same
// engine-backed executor.
//
// The fixture is topology-conformant with the trusted cache-position gate
// (`verifyQwenCachePosition`): the pinned 30-layer schedule (full attention
// every 6th layer) at the pinned 1024 sliding window, with tiny widths —
// so the legacy serial executor is drivable weight-free.

@Suite("RuntimeWorkerFreeRunLegIdentity", .serialized)
struct RuntimeWorkerFreeRunLegIdentityTests {

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

    /// Drive the REAL v1.1 serial-leg executors — the begin executor
    /// (`openSingleStreamFreeRunSession`) then `runFreeDecode(route:
    /// .serial)`, the exact functions the wire arms dispatch to — and
    /// return the committed stream plus the state observed between the
    /// begin and the run (what an in-flight phase looks like) and the run's
    /// diagnostics.
    private func runSerialLegExecutors(
        model: Gemma4TextModel, prompt: [Int], n: Int
    ) throws -> (
        stream: [Int],
        sessionAfterBegin: RuntimeWorkerFreeRunSession?,
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
            state: &state)
        state.decodeRoute = .serial
        state.freeRunLastToken = seed
        let sessionAfterBegin = state.freeRunSession
        let result = try Gemma4Runtime.runFreeDecode(
            targetN: n, route: .serial, state: &state)
        let diagnostics = Gemma4Runtime.makeFreeRunSessionDiagnostics(
            route: .serial,
            session: sessionAfterBegin,
            state: state,
            result: result)
        return ([seed] + result.tokens, sessionAfterBegin, diagnostics)
    }

    /// FAILING-FIRST invariant (red on the observability commit, green with
    /// the leg-identity fix): the serial leg's committed stream must be
    /// computed by the SAME engine-backed executor as the mtp leg's —
    /// observable as (1) an engine-backed free-run session on the worker
    /// state after the serial begin executor runs, and (2) the session
    /// diagnostics reporting the engine executor identity for the serial
    /// route. Alongside it, the guard that makes the fix safe where a
    /// laptop can certify it: the engine-backed serial stream is
    /// token-identical to the legacy loop's on the same weights.
    ///
    /// Requires `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` plus
    /// `tools/build-mlx-metallib.sh --all-build-roots` (repo convention).
    @Test
    func serialLegRunsTheSameEngineBackedExecutorAsTheMTPLeg() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        MLXRandom.seed(0x1E6_1D)
        let model = Gemma4TextModel(try topologyConformantTargetConfig())
        eval(model)
        let prompt = promptTokens(length: 14, seed: 11)
        let n = 32

        // The reference the pairing is sound against: the same model through
        // the width-1 CBv2 engine (the mtp leg's executor), drafter-less.
        let engine = try Gemma4Runtime.makeCohortEngine(
            model: model, batchSize: 1, seedTokenCount: prompt.count,
            maxTokensPerStream: n + 8)
        let engineSession = try RuntimeWorkerFreeRunSession(
            engine: engine, mode: .serial, seedTokens: prompt,
            maxTokens: n + 8, stopTokens: [])
        let engineResult = try engineSession.run(targetN: n)
        let engineStream = [engineSession.seedToken] + engineResult.tokens

        let (serialStream, sessionAfterBegin, diagnostics) = try runSerialLegExecutors(
            model: model, prompt: prompt, n: n)

        // GUARD (green before and after the fix at this scale): the two
        // implementations agree token-for-token on the same weights, so
        // moving the serial leg onto the engine executor cannot change its
        // committed stream anywhere a laptop can certify.
        #expect(
            serialStream == engineStream,
            "serial-leg executor and width-1 engine disagree on the same weights")

        // INVARIANT (RED until the leg-identity fix): the serial leg must
        // be engine-backed — the same executor as the mtp leg — so the
        // paired legs' token-exactness comparison is within ONE
        // implementation, per the track's own within-backend rule
        // (port-notes section 5.1).
        #expect(
            sessionAfterBegin != nil,
            """
            the v1.1 serial free-run leg ran the legacy single-token loop \
            while the mtp leg runs the width-1 CBv2 engine; the paired \
            token-exactness gate compares streams from two different decode \
            implementations
            """)
        #expect(
            diagnostics.executor == RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine,
            """
            serial-leg diagnostics report executor '\(diagnostics.executor)'; \
            both legs of the paired measurement must report \
            '\(RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine)'
            """)

        // The wire counters the serial leg reports must be unchanged by the
        // executor: N committed, [1]*N acceptance, zero drafts, R+1
        // completed-work accounting.
        #expect(engineResult.committedTotal == n)
        #expect(engineResult.acceptanceLengths == Array(repeating: 1, count: n))
        #expect(engineResult.draftedTotal == 0)
        #expect(engineResult.acceptedTotal == 0)
    }
}
