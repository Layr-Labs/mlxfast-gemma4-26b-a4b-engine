import Foundation
import MLXFastCore
import MLXLLM
import MLXLMCommon

// v1.1 SINGLE-STREAM free-run session — real round execution via a closed,
// width-1 CBv2 engine. Two modes:
//
//   * `.mtp` — the engine is bound to the assistant-head drafter this worker
//     loaded at startup (`Gemma4A4BAssistantHead.swift`); rounds are REAL
//     draft/verify rounds through the vendored `CBv2MTPRoundDriver`. This is
//     the single-stream twin of `RuntimeWorkerCohortSession.runMTP` at B=1,
//     per merged PR #9's own "What's next" note.
//   * `.serial` — the same width-1 engine construction with NO drafter
//     bound: the engine free-runs plain decode, one token per round.
//     (Added 2026-08-25 for the leg-implementation-identity fix; see
//     RuntimeWorkerFreeRunDiagnostics.swift for why the two legs of a
//     paired measurement must share one decode implementation.)
//
// What is REAL here (see `RuntimeWorkerCohortSession.runMTP`'s header for
// the parallel accounting on the batched leg — everything stated there about
// what the public `CBv2Event` / `EngineV2.mtpMetricsSnapshot()` surface can
// and cannot see applies unchanged at B=1):
//
//   * `acceptanceLengths` — one entry per `.delta` this session's single
//     collector received AFTER its seed chunk, i.e. one entry per REAL
//     round's committed width (`EngineLoopV2+MTPFinalize.swift`'s
//     `kept.count` for verify rounds; always 1 for plain rounds).
//   * `draftedTotal` / `acceptedTotal` — `engine.mtpMetricsSnapshot()`
//     diffed against the baseline snapshot taken right after the seed
//     collected (before any round ran). Zero in `.serial` mode (a
//     drafter-less engine has no MTP driver and snapshots nil).
//
// At B=1 there is no cross-row `min`, so — unlike the cohort leg — there is
// no `natural_accepted_by_stream` gap here: the v1.1 wire shape carries no
// such field at all (it is a v1.2 cohort-only AUDIT vector).
final class RuntimeWorkerFreeRunSession {

    enum Mode {
        /// Drafter-bound engine; REAL MTP rounds; metrics snapshot REQUIRED.
        case mtp
        /// Plain width-1 engine; every round commits exactly one token;
        /// drafted/accepted are structurally zero.
        case serial
    }

    private let engine: EngineV2
    private let collector: RuntimeWorkerCohortStreamCollector
    private let requestID: CBv2RequestID
    private let mode: Mode
    private var didShutdown = false
    /// `engine.mtpMetricsSnapshot()` taken right after the seed token
    /// collected below, before any round has a chance to run. nil for a
    /// `.serial` (drafter-less) engine.
    private let baselineMetrics: CBv2MTPMetrics?
    /// The final `engine.mtpMetricsSnapshot()` taken by `run` at drain time,
    /// exposed for the session diagnostics (strategy counts + the per-round
    /// acceptance/rollback audit records). nil before `run`, and always nil
    /// in `.serial` mode.
    private(set) var finalMetrics: CBv2MTPMetrics?

    /// The seed token — the prefill-bonus argmax `free_decode_begin` returns
    /// on the wire. Numerically identical to what `plainSeedForward` would
    /// compute over the same prompt (same weights, same greedy argmax,
    /// offset 0); the cohort session's own `seedTokenByStream` already
    /// relies on the same equivalence instead of reusing `plainSeedForward`.
    let seedToken: Int

    /// Open the single-stream free-run window: submit one CBv2 request over
    /// the full seed prompt and block until the prefill-bonus token is
    /// produced.
    ///
    /// `recordingTopLogprobs` (default 0 — every wire leg) asks the engine
    /// to also report the top-k raw-logprob readout per emitted token
    /// (`CBv2SamplingParams.topLogprobs`; captured from the RAW pre-transform
    /// logits via log_softmax — LogitsPipelineV2 rule (0)). It is the
    /// CBv2-backed reference-tape recorder's observability channel: token
    /// SELECTION is untouched (the sampler's argmax reads the same sampling
    /// tensor either way; the gather is a separate lazy graph branch over
    /// the raw logprobs), which the recording executor-identity test pins by
    /// comparing streams with the readout on and off on the same weights.
    init(
        engine: EngineV2,
        mode: Mode,
        seedTokens: [Int],
        maxTokens: Int,
        stopTokens: Set<Int>,
        recordingTopLogprobs: Int = 0
    ) throws {
        self.engine = engine
        self.mode = mode
        let id = CBv2RequestID(0)
        self.requestID = id
        let stream: AsyncStream<CBv2Event>
        do {
            stream = try engine.submit(
                CBv2Request(
                    id: id,
                    promptTokens: seedTokens,
                    // Greedy: temperature 0 selects the samplers' all-greedy
                    // argmax fast path — the same determinism the cohort
                    // leg's requests use, and the ONLY sampling MTP
                    // activates under (`EngineV2.init`'s
                    // `samplerSupportsMTP` gate).
                    sampling: CBv2SamplingParams(
                        temperature: 0,
                        topLogprobs: recordingTopLogprobs),
                    maxTokens: maxTokens,
                    // The real free-run stop-token set (v1.1's — the SAME
                    // set for both legs of a paired measurement), so a
                    // legitimate EOS ends the row through the engine's own
                    // finalize path rather than decoding past it.
                    stopTokens: stopTokens,
                    prefixCacheEnabled: false))
        } catch {
            RuntimeWorkerFreeRunSession.shutdownEngineBlocking(engine)
            throw error
        }
        let collector = RuntimeWorkerCohortStreamCollector(slot: 0)
        self.collector = collector
        collector.consume(stream)
        do {
            self.seedToken = try collector.waitForTokenCount(1)[0]
        } catch {
            RuntimeWorkerFreeRunSession.shutdownEngineBlocking(engine)
            didShutdown = true
            throw error
        }
        self.baselineMetrics = engine.mtpMetricsSnapshot()
    }

    /// Free-run to `targetN` MORE committed tokens past the seed and
    /// assemble the v1.1 free-run result. Single-shot: opened by
    /// `free_decode_begin`, fully drained and torn down by ONE
    /// `free_decode_run` call — the same single-call shape the batched
    /// cohort session's `runMTP`/`runSerial` already use (open at begin,
    /// drain-and-shutdown at run).
    func run(targetN: Int) throws -> RuntimeWorkerFreeRunResult {
        defer { shutdownBlocking() }
        let (tokens, chunkSizes, finished) = collector.waitForRounds(atLeast: targetN + 1)
        engine.cancel(requestID)
        // Shut down BEFORE the final metrics poll so the loop's in-flight
        // finalize (and its audit record / strategy count) is included —
        // the driver retains cumulative metrics for exactly this
        // post-shutdown poll, and the deferred shutdown is idempotent.
        shutdownBlocking()
        let drained = engine.mtpMetricsSnapshot()
        finalMetrics = drained

        guard chunkSizes.first == 1 else {
            throw MLXFastError.invalidInput(
                "runtime worker \(routeName) free_decode_run: engine's first delta committed "
                    + "\(chunkSizes.first.map(String.init) ?? "0") token(s), expected "
                    + "exactly 1 (the prefill-bonus seed step already returned by "
                    + "free_decode_begin)"
            )
        }

        var builder = RuntimeWorkerFreeRunBuilder(targetN: targetN)
        var cursor = 1  // skip the seed token itself (already on the wire).
        for size in chunkSizes.dropFirst() {
            guard !builder.isComplete else { break }
            let end = Swift.min(cursor + size, tokens.count)
            guard cursor < end else { break }
            builder.addRound(
                committedTokens: Array(tokens[cursor ..< end]), drafted: 0, accepted: 0)
            cursor = end
        }

        // Symmetric early-EOS: a `.stop` finish that landed short of N is a
        // valid engine outcome (the same structured verdict on both legs —
        // `runtimeWorkerFreeRunEarlyStop`'s contract), reported the same
        // structured way rather than as a generic short-assembly error.
        if let finished, case .stop = finished, !builder.isComplete {
            throw RuntimeWorkerFreeRunError.stopTokenBeforeTarget(
                route: routeName,
                token: tokens.last ?? seedToken,
                position: builder.committedTotal,
                n: targetN)
        }

        let assembled = try builder.finish()
        switch mode {
        case .serial:
            // A drafter-less engine drafts nothing by construction; the
            // honest serial counters are structural zeros, exactly like the
            // legacy serial loop's.
            return RuntimeWorkerFreeRunResult(
                tokens: assembled.tokens,
                acceptanceLengths: assembled.acceptanceLengths,
                draftedTotal: 0,
                acceptedTotal: 0,
                committedTotal: assembled.committedTotal
            )
        case .mtp:
            guard let drained else {
                throw MLXFastError.invalidInput(
                    "runtime worker mtp free_decode_run: engine reported no MTP metrics "
                        + "snapshot at phase end despite an active MTP session (wiring bug)"
                )
            }
            let baselineDrafted = baselineMetrics?.draftedTokens ?? 0
            let baselineAccepted = baselineMetrics?.acceptedTokens ?? 0
            let draftedTotal = Swift.max(0, drained.draftedTokens - baselineDrafted)
            let acceptedTotal = Swift.max(0, drained.acceptedTokens - baselineAccepted)
            return RuntimeWorkerFreeRunResult(
                tokens: assembled.tokens,
                acceptanceLengths: assembled.acceptanceLengths,
                draftedTotal: draftedTotal,
                acceptedTotal: acceptedTotal,
                committedTotal: assembled.committedTotal
            )
        }
    }

    private var routeName: String {
        switch mode {
        case .mtp: return RuntimeWorkerDecodeRoute.mtp.rawValue
        case .serial: return RuntimeWorkerDecodeRoute.serial.rawValue
        }
    }

    /// Recording surface: the per-token top-logprob readouts this session's
    /// collector observed, 1:1 with the committed token order (index 0 is
    /// the seed token). Meaningful only when the session was opened with
    /// `recordingTopLogprobs > 0`; on every wire leg it is a vector of nils.
    /// Valid after `run` (the collector's snapshot survives engine
    /// shutdown — the data was already collected).
    func tokenLogprobsSnapshot() -> [CBv2TokenLogprob?] {
        collector.tokenLogprobsSnapshot()
    }

    /// Idempotent synchronous engine shutdown. Must complete before the
    /// phase-close barrier, same reason as the cohort session's twin.
    func shutdownBlocking() {
        guard !didShutdown else { return }
        didShutdown = true
        RuntimeWorkerFreeRunSession.shutdownEngineBlocking(engine)
    }

    private static func shutdownEngineBlocking(_ engine: EngineV2) {
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            await engine.shutdown()
            semaphore.signal()
        }
        semaphore.wait()
    }
}

/// The pre-generalization name (single-stream MTP only). Kept as an alias so
/// existing references and their documentation trails stay valid.
typealias RuntimeWorkerMTPSession = RuntimeWorkerFreeRunSession

extension RuntimeWorkerFreeRunSession {
    /// Pre-generalization initializer shape (`.mtp` mode) — the signature
    /// the round-execution increment shipped with.
    convenience init(
        engine: EngineV2,
        seedTokens: [Int],
        maxTokens: Int,
        stopTokens: Set<Int>
    ) throws {
        try self.init(
            engine: engine, mode: .mtp, seedTokens: seedTokens,
            maxTokens: maxTokens, stopTokens: stopTokens)
    }
}
