import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLLM
import MLXLMCommon
import MLXSpeculative

// Closed-cohort free-run driver. Serial requests retain the general B1...B8
// scheduler surface. Production Gemma exact MTP is deliberately narrower:
// physical B1 only, with installed C2...C4 verifier entrypoints. A public MTP
// cohort wider than one is refused before engine construction; the generic
// multi-row round helpers remain solely for explicit engine-level fixtures.
//
// The cohort runs through the vendored ContinuousBatchingV2 engine
// (`EngineV2` + `SchedulerV2`), not through the worker's single-stream
// decode loop: a batched measurement is only a batch if the B streams share
// one scheduler and one forward pass. The driver holds the D4 closed-cohort
// shape end to end:
//
//   * ALL B streams are submitted before any consumer starts (darkbloom's
//     own benchmark admission pattern — reproducible scheduler membership);
//   * `maxConcurrentRequests == batch_size`, no refill, no waiting queue use;
//   * fixed identical per-stream budget N, requested by benchd;
//   * NO EOS exit — stop tokens are deliberately not configured on the
//     cohort requests (darkbloom's raw-parity shape: fixed length, every row
//     runs to exactly its budget). This diverges from the v1.1 single-stream
//     route's early-stop symmetry check ON PURPOSE: the batched regime is
//     fixed-length by D4 ruling, and both legs of a batched pair share this
//     driver, so the legs remain symmetric.
//
// KV backend: CONTIGUOUS, by construction — `CBv2ContiguousKVBackend` +
// `CBv2LayerCache` are instantiated explicitly and no paged type appears in
// this file, so there is nothing to degrade to (the ruled pin; also the only
// backend on which a future batched rectangular MTP leg runs at all at the
// vendored revision). If the model cannot produce CBv2 caches, the phase
// REFUSES; it never falls back to another backend.
//
// The one cohort `spec` resolves to `serial` or `mtp`. `serial` builds a plain
// target-only `EngineV2`; `mtp` first enforces physical B1, then binds the
// assistant-head drafter and certified direct verifier. Anything else is
// refused rather than clamped or relabeled.
//
// Measurement lives in benchd: nothing here times anything. The driver
// returns raw committed tokens and the cohort AUDIT counters; benchd owns
// the clock, the oracle exact-match, and the consistency quadruple.

// MARK: - Per-slot stream collector (pure async bridging; no MLX)

/// Collects one cohort slot's `CBv2Event` stream into an append-only token
/// list, bridging the engine's async consumer surface to the synchronous
/// worker protocol thread. Thread-safe (NSCondition); the protocol thread
/// blocks in `waitForTokenCount` while the consumer task appends.
final class RuntimeWorkerCohortStreamCollector: @unchecked Sendable {
    private let condition = NSCondition()
    private var tokens: [Int] = []
    /// Per-delta chunk sizes, in arrival order — one entry per `.delta` event
    /// this collector has appended. For a plain (non-MTP) stream every entry
    /// is 1 (one token per decode step); for an MTP stream the FIRST entry is
    /// the prefill-bonus / seed step (always 1, the same single greedy
    /// argmax `plainSeedForward` would have produced) and every entry after
    /// it is one MTP verify round's committed width
    /// (`EngineLoopV2+MTPFinalize.swift`'s `kept.count` — the engine's own
    /// finalize call is exactly one `.delta` per round per stream), i.e. the
    /// REAL per-round `acceptance_lengths` this session observed. Kept
    /// alongside `tokens` under the SAME lock so a reader never sees a token
    /// count that disagrees with the chunk-size sum that produced it.
    private var chunkSizes: [Int] = []
    /// Per-token top-logprob readouts, 1:1 with `tokens` and grown under the
    /// SAME lock (recording surface — the CBv2-backed reference-tape
    /// recorder). An entry is non-nil only when the request asked the engine
    /// for logprobs (`CBv2SamplingParams.topLogprobs > 0`); every wire leg
    /// keeps `topLogprobs == 0`, so on the measured paths this is a vector
    /// of nils and nothing else changes. Defensive padding: a delta whose
    /// logprob array does not cover its tokens contributes nils rather than
    /// misaligning the 1:1 invariant — a recording consumer treats a nil
    /// row as fatal, never as data.
    private var tokenLogprobs: [CBv2TokenLogprob?] = []
    /// Per-stream timing instrumentation (spec step 1): one monotonic
    /// nanosecond sample per `.delta` event, 1:1 with `chunkSizes` — the
    /// wall time at which THIS collector's own consuming task (running
    /// concurrently with every other slot's) observed the append, i.e. the
    /// real per-slot commit point the driver already has (this file's
    /// header: "sampled at the existing per-slot commit points inside the
    /// driver"). Recorded here rather than by the caller polling
    /// `waitForTokenCount`/`waitForRounds`, because polling order across
    /// slots is NOT the physical commit order — this append call is.
    private var commitTimestampsNs: [UInt64] = []
    private var endReason: String?
    /// The typed finish reason, when the stream ended via a `.finished`
    /// event (nil for engine-shutdown-without-finished termination). Lets a
    /// caller distinguish a legitimate `.stop`/`.length` finish from a
    /// structurally broken stream, which `endReason`'s free-text description
    /// cannot.
    private var typedFinishReason: CBv2FinishReason?
    private var task: Task<Void, Never>?
    /// The cohort slot this collector serves (error identity only).
    let slot: Int

    init(slot: Int) {
        self.slot = slot
    }

    /// Start draining `stream` on a detached task. Called AFTER every cohort
    /// request has been submitted, so scheduler membership is fixed before
    /// any consumer exists (closed-cohort admission).
    func consume(_ stream: AsyncStream<CBv2Event>) {
        task = Task.detached { [self] in
            for await event in stream {
                switch event {
                case .delta(_, let newTokens, let logprobs):
                    if !newTokens.isEmpty {
                        append(newTokens, logprobs: logprobs)
                    }
                case .finished(let reason, _):
                    markEnded(reason: String(describing: reason), typed: reason)
                }
            }
            // Stream termination without a finished event (engine shutdown)
            // still releases any waiter.
            markEnded(reason: "stream ended", typed: nil)
        }
    }

    /// Test seam: feed tokens without an engine. One call == one delta ==
    /// one chunk (one decode step or one MTP verify round). `logprobs`, when
    /// given, is the delta's own per-token readout (recording surface);
    /// absent or short coverage pads nil per token.
    func append(_ newTokens: [Int], logprobs: [CBv2TokenLogprob]? = nil) {
        // Sampled BEFORE the lock: the commit point is this append's own
        // arrival, not whenever the lock happens to be free (a busy waiter
        // thread must never skew the recorded timestamp later).
        let now = DispatchTime.now().uptimeNanoseconds
        condition.lock()
        tokens.append(contentsOf: newTokens)
        chunkSizes.append(newTokens.count)
        commitTimestampsNs.append(now)
        for index in newTokens.indices {
            if let logprobs, index < logprobs.count {
                tokenLogprobs.append(logprobs[index])
            } else {
                tokenLogprobs.append(nil)
            }
        }
        condition.broadcast()
        condition.unlock()
    }

    /// Thread-safe snapshot of the per-token top-logprob readouts, 1:1 with
    /// the token list (recording surface). Valid once the caller has
    /// independently confirmed the token count it needs; the snapshot is a
    /// copy, safe to use after the lock is released.
    func tokenLogprobsSnapshot() -> [CBv2TokenLogprob?] {
        condition.lock()
        defer { condition.unlock() }
        return tokenLogprobs
    }

    /// Test seam: mark the stream ended. The first reason wins (a `finished`
    /// event followed by stream termination keeps the engine's reason).
    func markEnded(reason: String, typed: CBv2FinishReason? = nil) {
        condition.lock()
        if endReason == nil {
            endReason = reason
            typedFinishReason = typed
        }
        condition.broadcast()
        condition.unlock()
    }

    /// Block until this slot has collected at least `count` tokens, returning
    /// the first `count`. Throws `RuntimeWorkerCohortError.streamEndedEarly`
    /// if the stream ends first — a cohort stream that stops before its
    /// budget is a broken cohort, never a silent short rectangle. Liveness is
    /// benchd's: the parent's RunTimeout kills a wedged worker, exactly as it
    /// does for a wedged v1.1 forward.
    func waitForTokenCount(_ count: Int) throws -> [Int] {
        condition.lock()
        defer { condition.unlock() }
        while tokens.count < count, endReason == nil {
            condition.wait()
        }
        guard tokens.count >= count else {
            throw RuntimeWorkerCohortError.streamEndedEarly(
                slot: slot,
                committed: tokens.count,
                target: count,
                reason: endReason ?? "stream ended")
        }
        return Array(tokens.prefix(count))
    }

    /// Block until AT LEAST `count` tokens are collected, or the stream ends
    /// (typed finish or otherwise) — whichever comes first. Unlike
    /// `waitForTokenCount`, never throws: it hands back whatever the stream
    /// actually produced (which may be short of `count`) plus the chunk
    /// sizes covering it and the typed finish reason, so the caller — which
    /// knows the wire semantics (a `.stop` short of N is a valid, symmetric
    /// leg outcome; anything else short of N is a broken assembly) —
    /// decides what a short result means. `tokens`/`chunkSizes` are snapshots
    /// (copies), safe to use after the lock is released.
    func waitForRounds(atLeast count: Int) -> (
        tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?
    ) {
        condition.lock()
        defer { condition.unlock() }
        while tokens.count < count, endReason == nil {
            condition.wait()
        }
        return (tokens, chunkSizes, typedFinishReason)
    }

    /// Thread-safe snapshot of this collector's full per-append chunk-size
    /// and commit-timestamp histories, 1:1 and grown together under the same
    /// lock as `tokens` (per-stream timing instrumentation). Valid to call
    /// once the caller has independently confirmed (via
    /// `waitForTokenCount`/`waitForRounds`) that this slot committed the
    /// tokens it needs — the histories are copies, safe to use after the
    /// lock is released.
    func commitHistorySnapshot() -> (chunkSizes: [Int], commitTimestampsNs: [UInt64]) {
        condition.lock()
        defer { condition.unlock() }
        return (chunkSizes, commitTimestampsNs)
    }
}

// MARK: - Cohort session (engine + collectors)

/// The live state of one batched free-run phase: the CBv2 engine, the B slot
/// collectors, and the seed tokens the batched `free_decode_begin` returned.
/// Sealed on the begin, consumed by the run, and always shut down before the
/// phase-close barrier (the allocator-drain assertion needs an idle engine).
final class RuntimeWorkerCohortSession {
    let batchSize: Int
    /// Per-slot seed-forward argmaxes, in SLOT ORDER — the batched begin's
    /// `seed_token_by_stream`, each oracle-checked by benchd.
    let seedTokenByStream: [Int]
    /// Per-stream timing instrumentation (spec step 1): per-slot monotonic
    /// nanoseconds from cohort-prefill start (the top of this initializer,
    /// before any request is submitted) to that slot's seed commit — the
    /// batched begin's `prefill_ns_by_stream`. Raw elapsed-ns only, in SLOT
    /// ORDER, same as `seedTokenByStream`.
    let prefillNsByStream: [UInt64]
    /// The (equal) per-stream seed prompt length — the cumulative-position
    /// anchor the MTP assembler's audit reconciliation converts
    /// `tokensCountAfter` with.
    let seedTokenCount: Int

    private let engine: EngineV2
    private let collectors: [RuntimeWorkerCohortStreamCollector]
    private let requestIDs: [CBv2RequestID]
    private var didShutdown = false
    /// `engine.mtpMetricsSnapshot()` taken right after the seed collection
    /// below, before any round has a chance to run — the baseline `runMTP`
    /// diffs against so drafted/accepted totals cover exactly this session's
    /// generation window. nil on a plain (non-MTP) session.
    private let mtpBaselineMetrics: CBv2MTPMetrics?

    /// Open the batched window: submit ALL B cohort requests before starting
    /// any consumer, then block until every slot has produced its seed token
    /// (the argmax after that slot's seed prefill — CBv2's first greedy
    /// sample of the request). `stopTokens` defaults empty — the D4
    /// raw-parity shape both the serial and MTP cohort legs share: a fixed-
    /// length closed cohort never exits on EOS (this file's header), which is
    /// also what makes `active_streams_by_round == [B] * R` a checked
    /// identity rather than an assumption in `runMTP`.
    init(
        engine: EngineV2,
        seedTokensByStream: [[Int]],
        maxTokensPerStream: Int,
        stopTokens: Set<Int> = []
    ) throws {
        // Per-stream timing instrumentation (spec step 1): the cohort-prefill
        // clock starts here, before ANY request is submitted — the
        // `prefill_ns_by_stream` origin.
        let prefillPhaseStartNs = DispatchTime.now().uptimeNanoseconds
        self.engine = engine
        self.batchSize = seedTokensByStream.count
        self.seedTokenCount = seedTokensByStream.first?.count ?? 0
        var streams: [AsyncStream<CBv2Event>] = []
        var ids: [CBv2RequestID] = []
        streams.reserveCapacity(batchSize)
        do {
            // Closed-cohort admission: every stream is submitted before any
            // consumer starts, so scheduler membership is fixed up front.
            for (slot, seeds) in seedTokensByStream.enumerated() {
                let id = CBv2RequestID(UInt64(slot))
                ids.append(id)
                streams.append(
                    try engine.submit(
                        CBv2Request(
                            id: id,
                            promptTokens: seeds,
                            // Greedy: temperature 0 selects the samplers'
                            // all-greedy argmax fast path — deterministic and
                            // batch-composition invariant by CBv2 contract.
                            sampling: CBv2SamplingParams(temperature: 0),
                            // Budget ceiling only: N arrives on the run verb,
                            // so the request is bounded at the engine-wide
                            // count ceiling (+1 for the seed argmax) and the
                            // run cancels the cohort once every slot reached
                            // its actual N.
                            maxTokens: maxTokensPerStream,
                            // D4 raw-parity (serial): NO stop tokens — a
                            // fixed-length closed cohort never exits on EOS.
                            // An MTP cohort passes the real set instead.
                            stopTokens: stopTokens,
                            prefixCacheEnabled: false)))
            }
        } catch {
            RuntimeWorkerCohortSession.shutdownEngineBlocking(engine)
            throw error
        }
        self.requestIDs = ids
        let collectors = (0..<batchSize).map {
            RuntimeWorkerCohortStreamCollector(slot: $0)
        }
        self.collectors = collectors
        for (collector, stream) in zip(collectors, streams) {
            collector.consume(stream)
        }
        do {
            self.seedTokenByStream = try collectors.map {
                try $0.waitForTokenCount(1)[0]
            }
            // Per-stream timing instrumentation: each slot's seed commit is
            // its collector's FIRST append (cumulative count 1) — sample the
            // REAL commit timestamp the collector recorded when that append
            // landed, not when this (sequential, per-slot) loop happened to
            // observe it.
            self.prefillNsByStream = try collectors.map { collector in
                let (chunkSizes, commitTimestampsNs) = collector.commitHistorySnapshot()
                guard
                    let commitTs = commitTimestampNs(
                        chunkSizes: chunkSizes, commitTimestampsNs: commitTimestampsNs,
                        atCumulativeCount: 1)
                else {
                    throw RuntimeWorkerCohortError.commitTimestampMissing(
                        slot: collector.slot, atCumulativeCount: 1)
                }
                return commitTs >= prefillPhaseStartNs ? commitTs - prefillPhaseStartNs : 0
            }
        } catch {
            RuntimeWorkerCohortSession.shutdownEngineBlocking(engine)
            didShutdown = true
            throw error
        }
        self.mtpBaselineMetrics = engine.mtpMetricsSnapshot()
    }

    /// Free-run the whole cohort until EVERY slot has committed `targetN`
    /// tokens past its seed, then cancel the cohort, shut the engine down
    /// (idle before the phase barrier), and assemble the serial cohort
    /// counters. The engine may have raced a few tokens past N before the
    /// cancel lands; they were produced inside the timed window and are
    /// trimmed by the assembler, never emitted.
    func runSerial(targetN: Int) throws -> RuntimeWorkerCohortFreeRunResult {
        defer { shutdownBlocking() }
        // Per-stream timing instrumentation: the decode-phase clock starts
        // here, at the top of the free-run leg — the `decode_ns_by_stream`
        // origin.
        let decodePhaseStartNs = DispatchTime.now().uptimeNanoseconds
        var streamsWithSeed: [[Int]] = []
        streamsWithSeed.reserveCapacity(batchSize)
        var decodeNsByStream: [UInt64] = []
        decodeNsByStream.reserveCapacity(batchSize)
        for collector in collectors {
            streamsWithSeed.append(try collector.waitForTokenCount(targetN + 1))
            // The final-token commit is the append that first brought this
            // slot's cumulative count to targetN + 1 (seed + N) — a serial
            // cohort appends exactly 1 token per delta, so this is the
            // (targetN)-th post-seed append's real commit timestamp.
            let (chunkSizes, commitTimestampsNs) = collector.commitHistorySnapshot()
            guard
                let commitTs = commitTimestampNs(
                    chunkSizes: chunkSizes, commitTimestampsNs: commitTimestampsNs,
                    atCumulativeCount: targetN + 1)
            else {
                throw RuntimeWorkerCohortError.commitTimestampMissing(
                    slot: collector.slot, atCumulativeCount: targetN + 1)
            }
            decodeNsByStream.append(
                commitTs >= decodePhaseStartNs ? commitTs - decodePhaseStartNs : 0)
        }
        for id in requestIDs {
            engine.cancel(id)
        }
        try requireNaturalRetirementBlocking()
        return try assembleSerialCohortFreeRun(
            streamsWithSeed: streamsWithSeed,
            batchSize: batchSize,
            targetN: targetN,
            decodeNsByStream: decodeNsByStream)
    }

    /// Free-run the whole cohort through REAL MTP rounds until every slot has
    /// committed `targetN` tokens past its seed, then assemble the cohort
    /// AUDIT counters from what the vendored `EngineV2`/`CBv2MTPRoundDriver`
    /// actually produced — not a structural re-derivation.
    ///
    /// What is REAL, sourced from the public engine surface (`CBv2Event` +
    /// `EngineV2.mtpMetricsSnapshot()` — the only two seams a harness module
    /// outside `MLXLMCommon` can observe; see the PR body's "vendored-
    /// machinery gaps" section for what is NOT reachable this way):
    ///
    ///   * `acceptanceLengths` — the per-round COMMON committed width. Each
    ///     stream's collector records one chunk per `.delta`, and
    ///     `EngineLoopV2+MTPFinalize.swift` emits exactly ONE delta per row
    ///     per round carrying `commonEmitted` tokens (the min-across-rows
    ///     width, already clamped there) — so slot 0's chunk-size sequence
    ///     (after dropping its seed chunk) IS the real per-round histogram,
    ///     and every other slot's sequence is asserted to agree (closed
    ///     cohort, uniform budget, D4 no-EOS ⇒ no row can legitimately
    ///     diverge from another's chunk sizes; a divergence is a broken
    ///     assembly, refused rather than silently reconciled).
    ///   * `draftedTotal` / `acceptedTotal` — `engine.mtpMetricsSnapshot()`
    ///     diffed against `mtpBaselineMetrics` (captured right after the
    ///     seed collection above, before any round ran).
    ///   * `activeStreamsByRound` — `[B] * R`. Derived, not assumed blind:
    ///     the D4 closed cohort carries NO stop tokens (this file's header;
    ///     unchanged for MTP) and every row shares the same budget N, so no
    ///     row can finish before another under the engine's own
    ///     min-across-rows commit rule — every round is B-wide by
    ///     construction. This is CHECKED, not skipped: the per-slot
    ///     round-count/chunk-size agreement assertion below would fail loudly
    ///     if any row's observed round history diverged.
    ///   * `depthClampReasons` — the union of the engine's own
    ///     `skippedRows` (rows clamped OUT of a round) and
    ///     `controllerFallbacks` (the depth CONTROLLER's own selection
    ///     reasons, including e.g. `tail_depth` / `automatic_rectangular_limit`)
    ///     histograms, summed key-wise. Both are literally "reasons this
    ///     driver did not run full depth"; the wire carries one map, so they
    ///     are merged rather than arbitrarily picking one.
    ///
    /// What is a DOCUMENTED FLOOR, not an observation: `naturalAcceptedByStream`
    /// (AUDIT-only, per-row PRE-min accept walk). The vendored engine computes
    /// this value (`naturalEmitted`, `EngineLoopV2+MTPFinalize.swift`) but
    /// never surfaces it on any public seam — `CBv2Event.delta` carries only
    /// the ALREADY-min-clamped `kept` tokens, and `CBv2MTPMetrics` is a
    /// cumulative/unordered snapshot, not a per-round-per-row series. Every
    /// row is reported at exactly the committed common width for that round
    /// — the only value provably `>= committed` (equality) without fabricating
    /// a number this harness never measured. See the PR body.
    func runMTP(targetN: Int) throws -> RuntimeWorkerCohortFreeRunResult {
        defer { shutdownBlocking() }
        // Per-stream timing instrumentation: same decode-phase origin as the
        // serial leg — the top of the free-run call.
        let decodePhaseStartNs = DispatchTime.now().uptimeNanoseconds
        var perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)] = []
        perSlot.reserveCapacity(batchSize)
        var decodeNsByStream: [UInt64] = []
        decodeNsByStream.reserveCapacity(batchSize)
        for collector in collectors {
            perSlot.append(collector.waitForRounds(atLeast: targetN + 1))
            // Same walk as the serial leg, over the (possibly multi-token)
            // MTP chunk sizes: the append that first brings cumulative
            // committed count to targetN + 1 (seed + N), whatever round
            // width landed it.
            let (chunkSizes, commitTimestampsNs) = collector.commitHistorySnapshot()
            guard
                let commitTs = commitTimestampNs(
                    chunkSizes: chunkSizes, commitTimestampsNs: commitTimestampsNs,
                    atCumulativeCount: targetN + 1)
            else {
                throw RuntimeWorkerCohortError.commitTimestampMissing(
                    slot: collector.slot, atCumulativeCount: targetN + 1)
            }
            decodeNsByStream.append(
                commitTs >= decodePhaseStartNs ? commitTs - decodePhaseStartNs : 0)
        }
        for id in requestIDs {
            engine.cancel(id)
        }
        // Shut the engine down BEFORE the final metrics poll: the loop's
        // in-flight step finalizes during shutdown, and the audit records
        // the assembler reconciles against (`roundAudits`) are written at
        // that finalize boundary. Snapshotting first would race the last
        // round's record. The driver retains cumulative metrics for exactly
        // this post-shutdown poll (`removeAllRequestState`'s contract), and
        // `shutdownBlocking` is idempotent for the deferred teardown above.
        try requireNaturalRetirementBlocking()
        let finalMetrics = engine.mtpMetricsSnapshot()
        return try assembleMTPCohortFreeRun(
            perSlot: perSlot,
            batchSize: batchSize,
            targetN: targetN,
            seedTokenCount: seedTokenCount,
            baselineMetrics: mtpBaselineMetrics,
            finalMetrics: finalMetrics,
            decodeNsByStream: decodeNsByStream)
    }

    /// Idempotent synchronous engine shutdown. Must complete before the
    /// phase-close barrier: a live engine loop would repopulate the MLX
    /// allocator cache that `phase_diagnostics` asserts drained, and the
    /// engine's released KV must land in that cache before `clearCache()`.
    func shutdownBlocking() {
        guard !didShutdown else { return }
        didShutdown = true
        RuntimeWorkerCohortSession.shutdownEngineBlocking(engine)
    }

    /// Normal evidence teardown. A watchdog escape is bounded but is not a
    /// retirement fence, so no post-drain cohort result or metrics may be
    /// published from it.
    private func requireNaturalRetirementBlocking() throws {
        let retirement = runtimeWorkerShutdownReportingRetirementBlocking(engine)
        didShutdown = true
        guard retirement == .natural else {
            throw MLXFastError.invalidInput(
                "runtime worker cohort engine did not retire naturally")
        }
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

// MARK: - Engine construction

extension Gemma4Runtime {
    /// Build a fresh CBv2 engine over the resident Gemma 4 text tower for one
    /// closed cohort of `batchSize` streams: CONTIGUOUS KV (explicitly — no
    /// paged type is constructible from this path), greedy default sampler,
    /// prefix cache off, and a scheduler sized so multi-stream cohorts prefill
    /// together and decode in lockstep `[B, 1]` rounds:
    ///
    ///   * `maxConcurrentRequests = batchSize` — the cohort IS the capacity;
    ///   * multi-stream cohorts retain
    ///     `maxBatchedTokensPerStep >= batchSize * seedTokenCount` and
    ///     `prefillChunkSize >= seedTokenCount`, so one planned step admits
    ///     every stream's whole (equal-length) seed;
    ///   * the physical-B1 lane caps long prefill steps at 16K tokens. A
    ///     single stream has no cohort peer to keep in lockstep, and the
    ///     bounded step stays below the engine-health watchdog on the exact
    ///     64K/128K benchmark prompts.
    ///
    /// A nil drafter builds the unchanged serial engine. A production Gemma
    /// MTP drafter is legal only at physical B1 and arrives with the sealed
    /// explicit-rectangular config. `CBv2LayerCacheBank` validates and binds
    /// its contiguous cache controller once during construction; the round
    /// driver then invokes the certified rectangular entrypoint directly.
    /// Unsupported production widths fail before engine construction and
    /// never degrade to serial under an MTP label. Generic `.automatic`
    /// configs remain available only to explicit engine-level fixtures.
    static func makeCohortEngine(
        model: Gemma4TextModel,
        batchSize: Int,
        seedTokenCount: Int,
        maxTokensPerStream: Int,
        mtpDrafter: (any CBv2MTPDrafter)? = nil,
        mtpConfig: CBv2MTPConfig = CBv2MTPConfig()
    ) throws -> EngineV2 {
        if mtpDrafter != nil, mtpConfig.verificationMode == .rectangular {
            try requireCertifiedMTPPhysicalBatch(batchSize)
        }
        let layerKinds = model.cbv2LayerKinds
        // CONTIGUOUS pin: CBv2LayerCache is the contiguous per-layer cache.
        // `newCacheV2` throws (refuses) if the model cannot serve CBv2 — the
        // fail-closed half of "refuse, don't degrade".
        let caches = try model.newCacheV2 { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: cohortContiguousKVBytesBudget(
                    layerKinds: layerKinds,
                    batchSize: batchSize,
                    maxSequenceLength: seedTokenCount + maxTokensPerStream)))
        // `mtpDrafter` is nil for the plain (target-only) cohort/single-stream
        // engine — EngineV2 runs byte-identical plain decode in that case
        // (`mtpDriver` stays nil at construction). A non-nil drafter binds
        // the real B1/C2...C4 round loop. The vendored driver invokes the
        // construction-certified rectangular cache controller directly.
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model),
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: CBv2DefaultSampler(),
            schedulerConfig: cohortSchedulerConfig(
                batchSize: batchSize, seedTokenCount: seedTokenCount),
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig)
    }

    /// Construction-time scheduler geometry for the closed cohort. The B1
    /// cap is a fixed workload route, not a per-step eligibility decision.
    static func cohortSchedulerConfig(
        batchSize: Int, seedTokenCount: Int
    ) -> CBv2SchedulerConfig {
        let wholeSeedChunk = Swift.max(512, seedTokenCount)
        let prefillChunkSize =
            batchSize == 1 ? Swift.min(16_384, wholeSeedChunk) : wholeSeedChunk
        let maxBatchedTokensPerStep =
            batchSize == 1
            ? Swift.max(2048, prefillChunkSize)
            : Swift.max(2048, batchSize * seedTokenCount)
        return CBv2SchedulerConfig(
            maxConcurrentRequests: batchSize,
            maxBatchedTokensPerStep: maxBatchedTokensPerStep,
            prefillChunkSize: prefillChunkSize,
            maxWaiting: batchSize,
            enablePrefixCache: false)
    }

    /// Production Gemma installs exact verifier entrypoints only for B1/C2,
    /// B1/C3, and B1/C4. Refuse any wider production MTP request before model
    /// lookup, allocator reset, cache allocation, or engine construction.
    static func requireCertifiedMTPPhysicalBatch(_ batchSize: Int) throws {
        guard batchSize == 1 else {
            throw MLXFastError.invalidInput(
                "production Gemma exact MTP is certified only for physical batch 1; "
                    + "got \(batchSize), refusing before engine construction")
        }
    }

    /// Fail-closed check for a just-constructed MTP-bound engine: this
    /// harness resolved `mtp` mode (a drafter loaded, envelope pins are
    /// set — `Gemma4MTPEnvelope`), so a non-nil `mtpInactiveReason` here
    /// means the vendored engine's OWN activation gate (sampler proof,
    /// model/drafter target-identity match, capture-layer availability —
    /// `EngineV2.init`) refused to bind it. The vendored engine's own
    /// posture at that point is to log and silently run plain decode
    /// (`CBv2 MTP inactive despite a bound drafter ... plain decode`); this
    /// harness's posture, once it deliberately resolved `mtp`, is the
    /// opposite — refuse loudly by name rather than silently serve serial
    /// decode under an `mtp` label, the same rule the former "round
    /// execution not yet wired" refusal enforced.
    static func requireMTPActive(_ engine: EngineV2) throws {
        if let reason = engine.mtpInactiveReason {
            throw MLXFastError.invalidInput(
                "runtime worker resolved mtp mode but the vendored CBv2 engine "
                    + "could not activate MTP round execution: \(reason); refusing "
                    + "rather than silently running plain decode under an mtp label"
            )
        }
    }

    // MARK: - Batched verb handlers

    /// v1.2 batched `free_decode_begin`: open the cohort window — allocator
    /// reset, spec resolution (general serial or certified physical-B1 MTP),
    /// engine build, closed-cohort admission, and the B seed forwards. Counts
    /// ONE completed_work unit for the whole cohort seed prefill (a round is one
    /// engine forward regardless of B; the phase closes at R + 1).
    static func handleCohortFreeDecodeBegin(
        _ request: RuntimeWorkerRequest,
        cohort: RuntimeWorkerValidatedCohortBegin,
        effectiveSpec: RuntimeWorkerEffectiveSpec?,
        sessionNonce: String,
        model: Gemma4TextModel,
        mtpDrafter: Gemma4CBv2MTPDrafter?,
        dflashDrafter: DFlashDraftModel? = nil,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        if effectiveSpec?.mode == RuntimeWorkerDecodeRoute.mtp.rawValue {
            try requireCertifiedMTPPhysicalBatch(cohort.batchSize)
        }
        // Speculative round EXECUTION for the batched cohort. A resolved
        // `mtp` spec — an assistant head loaded + envelope pins set
        // (`RuntimeWorkerSpecRegistry` / `Gemma4MTPEnvelope`) — binds the
        // SAME drafter this worker loaded at startup into a cohort-shaped
        // `EngineV2` via `makeCohortEngine`'s `mtpDrafter`/`mtpConfig`
        // parameters.
        //
        // DFLASH IS REFUSED HERE, BY NAME, and that is a correction of #38
        // rather than a missing feature. #38 ran the batched dflash arm by
        // passing the DFlash drafter through this same CBv2 path as a
        // `Gemma4CBv2MTPDrafter`; with the REAL `DFlashDraftModel` that is
        // not a type this seam can accept, and even ignoring types a batched
        // DFlash round needs batched KV caches on BOTH sides —
        // `DFlashDraftModel.makeBatchedCache(leftPadding:)` was dropped in
        // the port because its `BatchKVCache` / `BatchRotatingKVCache` types
        // are the v1 batching stack upstream deleted at ffede00, which this
        // tree does not carry. The dflash arm is therefore single-stream for
        // now (`RuntimeWorkerDFlashFreeRunSession`); a batched cohort needs a
        // batched DFlash cache type first. Refusing by name beats silently
        // running some other arm's rounds under a `dflash` label.
        let selectedDrafter: Gemma4CBv2MTPDrafter?
        let requestedDepth: Int?
        let arm: String
        switch effectiveSpec?.mode {
        case nil, RuntimeWorkerDecodeRoute.serial.rawValue:
            selectedDrafter = nil
            requestedDepth = nil
            arm = "serial"
        case RuntimeWorkerDecodeRoute.mtp.rawValue:
            selectedDrafter = mtpDrafter
            requestedDepth = effectiveSpec?.mtp?.depth
            arm = "mtp"
        case RuntimeWorkerDecodeRoute.dflash.rawValue:
            throw MLXFastError.invalidInput(
                "batched free_decode_begin resolved dflash mode, which this "
                    + "engine runs single-stream only: a batched DFlash round "
                    + "requires batched KV caches on both the target and the "
                    + "drafter, and this tree carries no batched KV cache type "
                    + "(the v1 batching stack was deleted upstream at ffede00). "
                    + "Use the single-stream free-run path for the dflash arm"
                    + (dflashDrafter == nil
                        ? "; note this worker also has no bound DFlash drafter"
                        : "")
            )
        case let other:
            throw MLXFastError.invalidInput(
                "batched free_decode_begin resolved unsupported spec mode "
                    + "'\(other ?? "?")'; the cohort driver runs serial or mtp"
            )
        }
        let isSpeculative = arm != "serial"
        // Spec resolution already refused a speculative mode for a worker with
        // no matching bound drafter (`RuntimeWorkerSpecRegistry.gemma4Worker`'s
        // `runnableModes` guard), so a resolved-speculative request reaching
        // here with a nil drafter would be a wiring bug, not a caller error —
        // refuse loudly by name rather than silently falling back to the
        // serial cohort under a speculative label.
        if isSpeculative, selectedDrafter == nil {
            throw MLXFastError.invalidInput(
                "batched free_decode_begin resolved \(arm) mode but this worker "
                    + "has no bound \(arm) drafter; spec resolution should have "
                    + "refused \(arm) for a worker with no staged head (wiring bug)")
        }
        try resetRuntimeWorkerAllocatorForPhaseStart()
        // The MTP arm binds through the pinned B1 explicit-rectangular CBv2
        // envelope, with depth clamped to the installed C2...C4 table.
        let mtpConfig =
            isSpeculative
            ? try Gemma4MTPEnvelope.resolveConfig(
                depth: Gemma4MTPEnvelope.resolveDepth(requestedDepth))
            : CBv2MTPConfig()
        let engine = try makeCohortEngine(
            model: model,
            batchSize: cohort.batchSize,
            seedTokenCount: cohort.seedTokensByStream[0].count,
            maxTokensPerStream:
                MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens + 1,
            mtpDrafter: isSpeculative ? selectedDrafter : nil,
            mtpConfig: mtpConfig)
        if isSpeculative {
            try requireMTPActive(engine)
        }
        let session = try RuntimeWorkerCohortSession(
            engine: engine,
            seedTokensByStream: cohort.seedTokensByStream,
            maxTokensPerStream:
                MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens + 1
            // D4 raw-parity, unchanged by MTP: NO stop tokens for either
            // cohort leg — a fixed-length closed cohort never exits on EOS
            // (this file's header). This is also what keeps
            // `active_streams_by_round == [B] * R` a real, checked identity
            // rather than an assumption: with no per-row early exit, no row
            // can end before another under the engine's own min-across-rows
            // commit rule.
        )
        state.cohortSession = session
        // `cohortSessionIsMTP` selects the drafter-driven `runMTP` assembler
        // over the plain `runSerial` one. The batched cohort runs `mtp` as
        // its only speculative arm (dflash is refused above), so this is
        // exactly "an assistant-head drafter bound".
        state.cohortSessionIsMTP = isSpeculative
        state.declaredSpec = effectiveSpec
        // The cohort seed prefill is the phase's first completed_work unit.
        state.completedWork += 1
        return RuntimeWorkerResponse(
            id: request.id,
            nonce: sessionNonce,
            ok: true,
            // The batched path is only reachable gate-on (guard 4), so the
            // echo rides unconditionally: a request that carried no spec
            // echoes none, mirroring the v1.1 begin.
            effectiveSpec: effectiveSpec,
            seedTokenByStream: session.seedTokenByStream,
            // Never-ignored echo: the width the engine WILL actually run —
            // which is exactly the validated request width, or this handler
            // would have refused.
            effectiveBatchSize: session.batchSize,
            // Per-stream timing instrumentation (spec step 1): raw per-slot
            // cohort-prefill elapsed ns, untrusted for scoring until benchd's
            // attestation admits it (engine-reported-time-untrusted
            // doctrine) — always populated alongside the rest of the
            // batched begin response, since this verb family is only
            // reachable at all when the speculative surface (and with it
            // `per_stream_timing`) was advertised at spawn.
            prefillNsByStream: session.prefillNsByStream
        )
    }

    /// v1.2 batched `free_decode_run`: free-run the sealed cohort to N
    /// committed tokens per stream and return the B x N rectangle plus the
    /// cohort AUDIT counters. R rounds add R completed_work units (the seed
    /// added one already), keeping the SCALAR `completed_work == R + 1`
    /// barrier — one engine forward per round regardless of B.
    static func handleCohortFreeDecodeRun(
        _ request: RuntimeWorkerRequest,
        batchSize: Int,
        targetN: Int,
        sessionNonce: String,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard let session = state.cohortSession, session.batchSize == batchSize
        else {
            throw MLXFastError.invalidInput(
                "runtime worker batched free_decode_run has no matching cohort "
                    + "session (validation drift)")
        }
        let result =
            state.cohortSessionIsMTP
            ? try session.runMTP(targetN: targetN)
            : try session.runSerial(targetN: targetN)
        state.cohortSession = nil
        state.cohortSessionIsMTP = false
        // R rounds — the serial cohort's R == N single-width rounds.
        state.completedWork += result.rounds
        return RuntimeWorkerResponse(
            id: request.id,
            nonce: sessionNonce,
            ok: true,
            acceptanceLengths: result.acceptanceLengths,
            draftedTotal: result.draftedTotal,
            acceptedTotal: result.acceptedTotal,
            committedTotal: result.committedTotal,
            effectiveBatchSize: batchSize,
            tokensByStream: result.tokensByStream,
            naturalAcceptedByStream: result.naturalAcceptedByStream,
            rounds: result.rounds,
            activeStreamsByRound: result.activeStreamsByRound,
            depthClampReasons: result.depthClampReasons,
            // Per-stream timing instrumentation (spec step 1): raw per-slot
            // decode-phase elapsed ns, same untrusted-until-attested posture
            // as `prefillNsByStream` above.
            decodeNsByStream: result.decodeNsByStream
        )
    }
}
