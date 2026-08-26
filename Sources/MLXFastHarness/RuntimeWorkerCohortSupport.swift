import Foundation
import MLXFastCore
import MLXLMCommon

// Pure, GPU-free support for the v1.2 BATCHED (cohort) free-run surface —
// `batched_free_run_decode` (batch-8 measurement brief, D6; benchd branch
// `lane/gemma4-batch8-cohort-measure` is the normative wire reference).
//
// v1.2 is WIRE-ADDITIVE on top of v1.1 and adds NO verb family: the SAME
// `free_decode_begin` / `free_decode_run` kinds carry the cohort form, selected
// by the additive `seed_tokens_by_stream` + `batch_size` request fields. benchd
// refuses to issue the cohort form to an engine that does not advertise the
// capability, BEFORE the cool gate and BEFORE the clock — and this engine
// symmetrically refuses the cohort fields when the v1.1 surface was gated off
// at spawn (the existing guard 4 covers the kinds; the cross-kind guard here
// covers the fields).
//
// Everything in this file is a pure decision over already-decoded wire data or
// already-collected token streams: validation of the cohort request form, the
// serial-path cohort assembly whose counters benchd's consistency QUADRUPLE
// cross-checks, and the contiguous-KV byte budget. The MLX/CBv2-touching half
// lives in `Gemma4RuntimeCohortDriver.swift`.

/// Hello capability advertising the v1.2 BATCHED (cohort) form of the free-run
/// verbs. Wire value matches benchd's `CAPABILITY_BATCHED_FREE_RUN_DECODE`.
///
/// Advertised under the same spawn gate as `free_run_decode`: the cohort form
/// is v1.1-surface-plus-width (benchd spawns batched legs with the same
/// `--speculative-protocol v1.1` — "the cohort form adds width, not a different
/// spawn surface").
let runtimeWorkerBatchedFreeRunDecodeCapability = "batched_free_run_decode"

/// The largest cohort width B this engine serves — THE single constant the
/// batched verbs derive their width ceiling from, and the value the gate-on
/// hello advertises as `max_batch_size` so benchd can refuse an over-wide
/// cohort PRE-GPU (before the cool gate and before the clock). 8 is both the
/// ruled scored batch point (batch-8 brief D0/D8) and the ceiling of what the
/// vendored CBv2 scheduler supports at all
/// (`CBv2SchedulerConfig.maxConcurrentRequests`, product range "B=3–4
/// (max 8)").
///
/// The hello field is safe to emit ONLY because the benchd gitlink advanced
/// onto the release-branch tip that carries the MERGED v1.2 cohort wire
/// (`deny_unknown_fields` would hard-fail every gate-on session against a
/// pre-cohort benchmarker; the gitlink advance and this emission land
/// together). The engine additionally enforces the same ceiling itself at the
/// batched verbs — refuse, never clamp — so an over-wide cohort dies with a
/// named error on either side of the wire.
let runtimeWorkerMaxCohortBatchSize = 8

/// Hello capability advertising per-stream wall-clock instrumentation on the
/// v1.2 batched (cohort) free-run verbs — the additive
/// `prefill_ns_by_stream` (batched `free_decode_begin`) and
/// `decode_ns_by_stream` (batched `free_decode_run`) fields (per-stream
/// timing instrumentation spec, step 1). Advertise-before-use: benchd
/// refuses to request per-stream scoring against an engine that does not
/// carry this in its hello capability list. Gated at spawn under the SAME
/// flag as `batched_free_run_decode` (the cohort verbs this rides on are
/// only reachable at all when the speculative surface is advertised), so
/// this engine emits both together — there is no independent toggle for
/// this increment.
let runtimeWorkerPerStreamTimingCapability = "per_stream_timing"

/// The capability list a gate-on hello advertises: the v1.1 single-stream
/// free-run mode, its v1.2 batched (cohort) form, and the per-stream timing
/// instrumentation the cohort form now carries. Order is stable —
/// benchd's session only membership-tests the list, but the hello bytes are
/// captured surface.
let runtimeWorkerAdvertisedCapabilities = [
    runtimeWorkerFreeRunDecodeCapability,
    runtimeWorkerBatchedFreeRunDecodeCapability,
    runtimeWorkerPerStreamTimingCapability,
]

/// The cohort reference-replay oracle capability. The `cohort_reference_replay`
/// verb is NOT behind the `--speculative-protocol` spawn gate (always
/// dispatchable), so its capability is advertised UNGATED on every hello.
/// benchd's (b) admission gate spawns the trusted oracle on a PLAIN
/// runtime-worker and REFUSES the capability-gated request unless this is
/// advertised.
let runtimeWorkerCohortReferenceReplayCapability = "cohort_reference_replay"

/// THE hello's capability list, as a function of the spawn gate — the single
/// place that surface is assembled, so both the gate pin
/// (`RuntimeWorkerCohortTests`) and the captured wire fixture
/// (`EmitWireFixtureTests`) exercise the REAL emitter instead of a re-typed
/// literal that can drift from it silently.
///
/// `cohort_reference_replay` rides UNGATED and FIRST (its verb is dispatchable
/// on a plain worker, and benchd's (b) admission oracle spawns plain); the three
/// speculative capabilities ride only when benchd opted in at spawn with
/// `--speculative-protocol`. Order is stable — benchd only membership-tests the
/// list, but the hello bytes are captured, sha-pinned surface.
func runtimeWorkerHelloCapabilities(
    advertisesSpeculativeProtocol: Bool
) -> [String] {
    [runtimeWorkerCohortReferenceReplayCapability]
        + (advertisesSpeculativeProtocol ? runtimeWorkerAdvertisedCapabilities : [])
}

// MARK: - Cohort request validation (pure; called from validateGenericWorkerRequest)

/// The validated cohort form of a `free_decode_begin`: the explicit width B and
/// the B per-slot seed token lists, in SLOT ORDER, already shape-checked.
struct RuntimeWorkerValidatedCohortBegin: Equatable {
    let batchSize: Int
    let seedTokensByStream: [[Int]]
}

/// Reject the v1.2 cohort request fields anywhere they are not honored — the
/// same posture as the cross-kind `spec` guard: a silently ignored field is a
/// false acknowledgment. `seed_tokens_by_stream` rides ONLY on
/// `free_decode_begin`; `batch_size` rides ONLY on the two free-run kinds.
func rejectCohortFieldsOnWrongKinds(_ request: RuntimeWorkerRequest) throws {
    if request.seedTokensByStream != nil, request.kind != "free_decode_begin" {
        throw MLXFastError.invalidInput(
            "seed_tokens_by_stream is valid only on free_decode_begin; request "
                + "kind '\(request.kind)' must not carry it")
    }
    if request.batchSize != nil,
        request.kind != "free_decode_begin",
        request.kind != "free_decode_run"
    {
        throw MLXFastError.invalidInput(
            "batch_size is valid only on free_decode_begin / free_decode_run; "
                + "request kind '\(request.kind)' must not carry it")
    }
}

/// Whether a `free_decode_begin` request selects the COHORT form. Presence of
/// EITHER cohort field selects it (so a half-carried pair is validated and
/// refused rather than silently treated as single-stream).
func runtimeWorkerRequestSelectsCohortBegin(_ request: RuntimeWorkerRequest) -> Bool {
    request.batchSize != nil || request.seedTokensByStream != nil
}

/// Validate the COHORT form of `free_decode_begin`. The wire contract
/// (bench-protocol v1.2): `batch_size` is the EXPLICIT width, never inferred
/// from an array length alone, and the request carries `seed_tokens_by_stream`
/// INSTEAD OF `seed_tokens` — one or the other, never both.
///
/// This increment additionally requires the cohort to be RECTANGULAR in its
/// seeds (every slot the same seed length). That is not a wire requirement —
/// it is the D4 closed-cohort shape this SERIAL driver is built for (equal
/// seeds ⇒ the whole cohort finishes prefill together and decodes in lockstep
/// `[B,1]` rounds), and benchd's cohort is B pool goldens with identical
/// 1024-token seeds. A ragged cohort is refused, never reshaped.
func validateCohortFreeDecodeBegin(
    _ request: RuntimeWorkerRequest,
    context: RuntimeWorkerRequestContext
) throws -> RuntimeWorkerValidatedCohortBegin {
    guard context.cohortBatchSize == nil else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_begin while a cohort phase is "
                + "already open; one batched window per phase")
    }
    guard let batchSize = request.batchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_begin requires an explicit "
                + "batch_size (B is never inferred from seed_tokens_by_stream)")
    }
    guard let seedsByStream = request.seedTokensByStream else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_begin requires "
                + "seed_tokens_by_stream")
    }
    guard request.seedTokens == nil else {
        throw MLXFastError.invalidInput(
            "runtime worker free_decode_begin carries both seed_tokens and "
                + "seed_tokens_by_stream; a request is single-stream or cohort, "
                + "never both")
    }
    guard batchSize >= 1, batchSize <= runtimeWorkerMaxCohortBatchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_begin batch_size must be in "
                + "1...\(runtimeWorkerMaxCohortBatchSize), got \(batchSize)")
    }
    guard seedsByStream.count == batchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_begin declared batch_size "
                + "\(batchSize) but carries \(seedsByStream.count) seed streams")
    }
    for (slot, seeds) in seedsByStream.enumerated() {
        guard !seeds.isEmpty else {
            throw MLXFastError.invalidInput(
                "runtime worker batched free_decode_begin stream \(slot) seed "
                    + "must not be empty")
        }
        guard seeds.count == seedsByStream[0].count else {
            throw MLXFastError.invalidInput(
                "runtime worker batched free_decode_begin stream \(slot) has "
                    + "\(seeds.count) seed tokens but stream 0 has "
                    + "\(seedsByStream[0].count); this serial cohort driver "
                    + "requires a rectangular (equal-seed-length) cohort")
        }
    }
    return RuntimeWorkerValidatedCohortBegin(
        batchSize: batchSize, seedTokensByStream: seedsByStream)
}

/// Validate the COHORT form of `free_decode_run` (selected by a present
/// `batch_size`): the width must equal the one the batched begin sealed.
/// Returns the validated width; the caller bounds `count` with the shared
/// v1.1 bound.
func validateCohortFreeDecodeRun(
    _ request: RuntimeWorkerRequest,
    context: RuntimeWorkerRequestContext
) throws -> Int {
    guard let sealed = context.cohortBatchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_run before batched "
                + "free_decode_begin")
    }
    guard let batchSize = request.batchSize, batchSize == sealed else {
        throw MLXFastError.invalidInput(
            "runtime worker batched free_decode_run batch_size "
                + "\(request.batchSize.map(String.init) ?? "nil") does not match "
                + "the cohort width \(sealed) sealed by free_decode_begin")
    }
    return batchSize
}

// MARK: - Cohort reference-replay validation (PR-1 fidelity-gate; pure, model-free)

/// The validated `cohort_reference_replay` payload: B streams' seeds and the
/// candidate's committed journals in SLOT ORDER, already shape/vocab-checked,
/// plus the bounded MEASUREMENT characterization params.
struct RuntimeWorkerValidatedCohortReferenceReplay: Equatable {
    let seedsByStream: [[Int]]
    let committedByStream: [[Int]]
    let logitTopK: Int
    let relEnvelope: Double
    /// The reference's replay WIDTH: `cohort` (batch-B, the scored candidate's
    /// geometry, David-ruled default) or `canonical` (per-stream width-1
    /// diagnostic). Parsed from the `replay_width` field, defaulting to
    /// `cohortReferenceReplayDefaultWidth` when absent.
    let replayWidth: CohortReferenceReplayWidth
}

/// Reject the reference-replay request fields anywhere they are not honored —
/// same false-acknowledgment posture as `rejectCohortFieldsOnWrongKinds`. All
/// four ride ONLY on `cohort_reference_replay`. Deliberately a SEPARATE surface
/// from the free-run cohort's `seed_tokens_by_stream` (which stays
/// `free_decode_begin`-only), so the trusted-CLI reference verb and the
/// benchd-facing free-run cohort never share a field.
func rejectCohortReferenceReplayFieldsOnWrongKinds(
    _ request: RuntimeWorkerRequest
) throws {
    let carriesReplayFields =
        request.replaySeedsByStream != nil
        || request.committedByStream != nil
        || request.logitTopK != nil
        || request.relEnvelope != nil
        || request.replayWidth != nil
    if carriesReplayFields, request.kind != "cohort_reference_replay" {
        throw MLXFastError.invalidInput(
            "replay_seeds_by_stream / committed_by_stream / logit_top_k / "
                + "rel_envelope / replay_width are valid only on "
                + "cohort_reference_replay; request kind '\(request.kind)' must "
                + "not carry them")
    }
}

/// Validate the COHORT REFERENCE-REPLAY request. Pure: shape (B streams of
/// seeds AND committed journals, equal counts, none empty), vocab range on every
/// token id, a committed-journal length bound (the reference walks one width-1
/// forward per committed token), and bounds on the measurement characterization
/// params. Renders no verdict — this is a shape gate only.
func validateCohortReferenceReplay(
    _ request: RuntimeWorkerRequest
) throws -> RuntimeWorkerValidatedCohortReferenceReplay {
    guard let seedsByStream = request.replaySeedsByStream else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay requires replay_seeds_by_stream")
    }
    guard let committedByStream = request.committedByStream else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay requires committed_by_stream")
    }
    let batchSize = seedsByStream.count
    guard batchSize >= 1, batchSize <= runtimeWorkerMaxCohortBatchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay batch (seed stream count) "
                + "must be in 1...\(runtimeWorkerMaxCohortBatchSize), got "
                + "\(batchSize)")
    }
    guard committedByStream.count == batchSize else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay has \(batchSize) seed streams "
                + "but \(committedByStream.count) committed streams; one committed "
                + "journal per seed stream")
    }
    for (slot, seeds) in seedsByStream.enumerated() {
        guard !seeds.isEmpty else {
            throw MLXFastError.invalidInput(
                "runtime worker cohort_reference_replay stream \(slot) seed must "
                    + "not be empty")
        }
        try cohortReferenceReplayRejectOutOfVocab(
            seeds, slot: slot, label: "seed")
    }
    for (slot, committed) in committedByStream.enumerated() {
        guard !committed.isEmpty else {
            throw MLXFastError.invalidInput(
                "runtime worker cohort_reference_replay stream \(slot) committed "
                    + "journal must not be empty")
        }
        guard committed.count
            <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "runtime worker cohort_reference_replay stream \(slot) committed "
                    + "journal has \(committed.count) tokens; the bound is "
                    + "\(MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens)")
        }
        try cohortReferenceReplayRejectOutOfVocab(
            committed, slot: slot, label: "committed")
    }
    let logitTopK = request.logitTopK ?? cohortReferenceReplayDefaultLogitTopK
    guard logitTopK >= 1, logitTopK <= MLXFastConstants.vocabSize else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay logit_top_k must be in "
                + "1...\(MLXFastConstants.vocabSize), got \(logitTopK)")
    }
    let relEnvelope =
        request.relEnvelope ?? cohortReferenceReplayDefaultRelEnvelope
    guard relEnvelope.isFinite, relEnvelope >= 0, relEnvelope <= 1 else {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay rel_envelope must be a finite "
                + "value in 0...1, got \(relEnvelope)")
    }
    // Replay WIDTH: defaults to cohort (batch-B, the scored candidate's
    // geometry — David's ruling); `canonical` opts into the width-1 diagnostic.
    // An unrecognized value is refused, never silently coerced.
    let replayWidth: CohortReferenceReplayWidth
    if let raw = request.replayWidth {
        guard let parsed = CohortReferenceReplayWidth(rawValue: raw) else {
            throw MLXFastError.invalidInput(
                "runtime worker cohort_reference_replay replay_width must be one "
                    + "of \(CohortReferenceReplayWidth.allCases.map(\.rawValue)), "
                    + "got '\(raw)'")
        }
        replayWidth = parsed
    } else {
        replayWidth = cohortReferenceReplayDefaultWidth
    }
    // The cohort-width path replays the B streams BATCHED at [B, 1] in lockstep,
    // so it requires a RECTANGULAR cohort (equal seed lengths AND equal
    // committed-journal lengths). Refuse a ragged cohort here (shape gate),
    // pointing at `canonical` for the per-stream width-1 diagnostic which has no
    // such requirement. In benchd's real cohort the seeds and the fixed budget N
    // are uniform, so this always holds for the scored path.
    if replayWidth == .cohort {
        let seedLength = seedsByStream[0].count
        let committedLength = committedByStream[0].count
        for slot in seedsByStream.indices {
            guard seedsByStream[slot].count == seedLength else {
                throw MLXFastError.invalidInput(
                    "runtime worker cohort_reference_replay stream \(slot) has "
                        + "\(seedsByStream[slot].count) seed tokens but stream 0 "
                        + "has \(seedLength); replay_width 'cohort' requires a "
                        + "rectangular (equal-seed-length) cohort — use "
                        + "replay_width 'canonical' for a ragged cohort")
            }
            guard committedByStream[slot].count == committedLength else {
                throw MLXFastError.invalidInput(
                    "runtime worker cohort_reference_replay stream \(slot) "
                        + "committed journal has \(committedByStream[slot].count) "
                        + "tokens but stream 0 has \(committedLength); replay_width "
                        + "'cohort' requires an equal-committed-length cohort so the "
                        + "B streams step in lockstep — use replay_width 'canonical' "
                        + "for a ragged cohort")
            }
        }
    }
    return RuntimeWorkerValidatedCohortReferenceReplay(
        seedsByStream: seedsByStream,
        committedByStream: committedByStream,
        logitTopK: logitTopK,
        relEnvelope: relEnvelope,
        replayWidth: replayWidth)
}

private func cohortReferenceReplayRejectOutOfVocab(
    _ tokens: [Int], slot: Int, label: String
) throws {
    for token in tokens where token < 0 || token >= MLXFastConstants.vocabSize {
        throw MLXFastError.invalidInput(
            "runtime worker cohort_reference_replay stream \(slot) \(label) token "
                + "\(token) is outside the vocab [0, \(MLXFastConstants.vocabSize))")
    }
}

// MARK: - Serial cohort assembly (the counters benchd's QUADRUPLE cross-checks)

/// A way the cohort assembly can fail its own consistency checks — the same
/// fail-closed posture as `RuntimeWorkerFreeRunError`, checked worker-side so a
/// broken cohort is caught where it was produced, never shipped.
enum RuntimeWorkerCohortError: Error, CustomStringConvertible, Equatable {
    case streamCount(expected: Int, got: Int)
    case streamTokenCount(slot: Int, expected: Int, got: Int)
    case streamEndedEarly(slot: Int, committed: Int, target: Int, reason: String)
    /// MTP cohort only: a slot's first observed `.delta` did not commit
    /// exactly one token, so it cannot be the prefill-bonus seed step
    /// `free_decode_begin` already returned.
    case mtpSeedChunkMalformed(slot: Int, got: Int)
    /// MTP cohort only: a slot's multi-token committed chunk has no verify
    /// audit record at its cumulative position — the stream claims a
    /// speculative commit the engine's own finalize never recorded. A
    /// corrupt assembly, refused (a PLAIN round commits exactly one token,
    /// so every width-greater-than-1 chunk must be a recorded verify round).
    case mtpChunkWithoutVerifyAudit(slot: Int, cumulative: Int, width: Int)
    /// MTP cohort only: a slot's committed chunk and the engine's verify
    /// audit at the same cumulative position disagree on the committed
    /// width — the stream and the finalize record cannot both be right.
    case mtpAuditWidthDisagreement(
        slot: Int, cumulative: Int, auditConfirmed: Int, chunkWidth: Int)
    /// MTP cohort only: the engine's audit ring hit its retention cap, so
    /// the oldest records were dropped and stream/audit reconciliation can
    /// no longer prove coverage of the whole window. Refused (fail-closed);
    /// the cap is sized so this cannot happen for any admitted window.
    case mtpRoundAuditsTruncated(count: Int)
    /// MTP cohort only: the engine reported no MTP metrics snapshot at
    /// phase end despite an active MTP session — a wiring bug, not a
    /// measurement outcome.
    case mtpMetricsMissing
    /// Per-stream timing instrumentation: a slot's collected commit history
    /// (chunk sizes + commit timestamps) never reached the cumulative token
    /// count the caller already confirmed via `waitForTokenCount`/
    /// `waitForRounds` — the two histories fell out of sync, a driver wiring
    /// bug rather than a measurement outcome (the token wait already
    /// guaranteed the count is there).
    case commitTimestampMissing(slot: Int, atCumulativeCount: Int)
    /// Per-stream timing instrumentation: the driver produced a
    /// `decodeNsByStream` vector whose length does not equal the cohort
    /// width B — a wiring bug in the per-slot collection loop.
    case decodeNsByStreamCount(expected: Int, got: Int)

    var description: String {
        switch self {
        case let .streamCount(expected, got):
            return "batched free_decode_run assembled \(got) streams, expected B=\(expected)"
        case let .streamTokenCount(slot, expected, got):
            return
                "batched free_decode_run stream \(slot) committed \(got) tokens, "
                + "expected N=\(expected)"
        case let .streamEndedEarly(slot, committed, target, reason):
            return
                "batched free_decode_run stream \(slot) ended (\(reason)) after "
                + "\(committed) tokens, before reaching \(target)"
        case let .mtpSeedChunkMalformed(slot, got):
            return
                "batched mtp free_decode_run stream \(slot)'s first delta committed "
                + "\(got) token(s), expected exactly 1 (the prefill-bonus seed step)"
        case let .mtpChunkWithoutVerifyAudit(slot, cumulative, width):
            return
                "batched mtp free_decode_run stream \(slot) committed a "
                + "\(width)-token chunk ending at cumulative token \(cumulative) "
                + "with no verify-round audit record at that position; a plain "
                + "round commits exactly one token, so an unrecorded multi-token "
                + "commit is a corrupt assembly"
        case let .mtpAuditWidthDisagreement(slot, cumulative, auditConfirmed, chunkWidth):
            return
                "batched mtp free_decode_run stream \(slot)'s chunk ending at "
                + "cumulative token \(cumulative) committed \(chunkWidth) token(s) "
                + "but the engine's verify audit at that position confirmed "
                + "\(auditConfirmed); the stream and the finalize record cannot "
                + "both be right"
        case let .mtpRoundAuditsTruncated(count):
            return
                "batched mtp free_decode_run: the engine's verify-round audit "
                + "ring reached its retention cap (\(count) records), so "
                + "stream/audit reconciliation cannot prove coverage of the "
                + "whole window"
        case .mtpMetricsMissing:
            return
                "batched mtp free_decode_run: engine reported no MTP metrics "
                + "snapshot at phase end despite an active MTP session"
        case let .commitTimestampMissing(slot, atCumulativeCount):
            return
                "batched free_decode_run stream \(slot) has no commit timestamp at "
                + "cumulative token count \(atCumulativeCount); the collector's "
                + "chunk-size and commit-timestamp histories fell out of sync"
        case let .decodeNsByStreamCount(expected, got):
            return
                "batched free_decode_run produced \(got) decode_ns_by_stream "
                + "entries, expected B=\(expected)"
        }
    }
}

// MARK: - Per-stream timing instrumentation (pure)

/// Given one cohort slot's per-append chunk-size history and its 1:1 commit
/// timestamp history (`chunkSizes[i]` tokens landed at
/// `commitTimestampsNs[i]`, both grown together under the collector's lock),
/// return the wall commit timestamp of the append whose landing FIRST
/// brought this slot's cumulative committed token count to at least
/// `cumulativeCount`. `nil` if the histories never reach that count (a
/// wiring bug when the caller has already confirmed the token count via
/// `waitForTokenCount`/`waitForRounds`, since the two histories are supposed
/// to grow in lockstep).
///
/// Used for BOTH per-stream timing sample points the spec defines: the
/// prefill seed commit (`cumulativeCount == 1` — the first append always
/// carries the single seed token) and the decode final-token commit
/// (`cumulativeCount == targetN + 1`, i.e. seed + N committed decode
/// tokens) — a serial cohort's chunk sizes are always 1 (one append per
/// token) and an MTP cohort's vary per round, so this single walk handles
/// both without the caller needing to know which regime produced the
/// history.
func commitTimestampNs(
    chunkSizes: [Int], commitTimestampsNs: [UInt64], atCumulativeCount cumulativeCount: Int
) -> UInt64? {
    var cumulative = 0
    for (index, size) in chunkSizes.enumerated() {
        cumulative += size
        if cumulative >= cumulativeCount {
            guard index < commitTimestampsNs.count else { return nil }
            return commitTimestampsNs[index]
        }
    }
    return nil
}

/// The assembled result of a batched `free_decode_run(N)` phase — exactly what
/// the worker returns to benchd, whose cohort consistency QUADRUPLE
/// cross-checks every count here (bench-core `verify_cohort_consistency`):
///
///   1. `tokens_by_stream` is the B x N committed rectangle;
///   2. `committed_total == B * N`;
///   3. `sum(acceptance_lengths) == N` (the per-round COMMON width — a single
///      vector even at B > 1) and `natural_accepted_by_stream` is B x R with no
///      row below the committed width;
///   4. `completed_work == R + 1` (SCALAR — a round is one engine forward
///      regardless of B), `rounds == R`, `active_streams_by_round` length R,
///      in `1...B`, non-increasing.
struct RuntimeWorkerCohortFreeRunResult: Equatable {
    /// The B x N committed token rectangle, in SLOT ORDER.
    let tokensByStream: [[Int]]
    /// Per-round COMMON committed width (min across rows), length R, sum N.
    let acceptanceLengths: [Int]
    /// Each row's PRE-`min` natural accept-walk length per round, B x R.
    /// AUDIT-only on the wire, never scored.
    let naturalAcceptedByStream: [[Int]]
    /// Streams still generating at each round, length R. Non-increasing under
    /// the closed fixed-N cohort.
    let activeStreamsByRound: [Int]
    /// Cohort-sum draft counters (`drafted >= accepted`) and committed total
    /// (`== B * N`).
    let draftedTotal: Int
    let acceptedTotal: Int
    let committedTotal: Int
    /// Depth-clamp reason histogram over the window. AUDIT-only; the serial
    /// route clamps nothing (there is no speculative plan to clamp), so it is
    /// EMPTY — sealed as such, not omitted.
    let depthClampReasons: [String: Int]
    /// Per-stream timing instrumentation (spec step 1): per-slot monotonic
    /// nanoseconds from decode-phase start (the top of `runSerial`/`runMTP`)
    /// to that slot's final-token commit — the append whose landing brought
    /// its cumulative committed count to `targetN + 1` (seed + N). Raw
    /// elapsed-ns only; nothing further computed here (untrusted for scoring
    /// until benchd's attestation admits it — engine-reported-time-untrusted
    /// doctrine). Length == batchSize, in SLOT ORDER, same as
    /// `tokensByStream`.
    let decodeNsByStream: [UInt64]

    /// R — the round count (`acceptanceLengths.count`).
    var rounds: Int { acceptanceLengths.count }

    /// The phase-close `completed_work`: the cohort seed forward plus R rounds.
    /// SCALAR by normative ruling — a round is ONE engine forward regardless of
    /// B, so this counter does not scale with the cohort width.
    var completedWork: Int { rounds + 1 }
}

/// Assemble the SERIAL (target-only) cohort result from the collected streams.
///
/// `streamsWithSeed[slot]` is slot's collected generation INCLUDING its leading
/// seed token (the one `free_decode_begin` already returned); the N tokens
/// AFTER it are the committed window, mirroring the v1.1 begin/run seam
/// (PROTOCOL-v1.1 §2.2 — the seed is never re-emitted).
///
/// The serial regime's per-round accounting is structural, not observed: a
/// non-drafting cohort commits EXACTLY one token per stream per round (each
/// committed token is one target-only `[B,1]` forward row), so R == N, the
/// common width is `[1]*N`, every row's natural walk equals the committed
/// width, and — because the budget N is identical per stream, EOS exit is
/// suppressed, and the cohort is closed with no refill — every stream is
/// active in every round: `active_streams_by_round == [B]*N`. This is the
/// batched generalization of the v1.1 serial route's `[1]*N` histogram, and it
/// is exactly the shape benchd's batched serial control STRUCTURALLY asserts
/// (common width 1 every round, R == N).
///
/// Serial totals: nothing is drafted and nothing is accepted-from-a-draft
/// (`drafted_total == accepted_total == 0`, the same accounting as the v1.1
/// serial route's per-round `drafted: 0, accepted: 0`), and
/// `committed_total == B * N`.
func assembleSerialCohortFreeRun(
    streamsWithSeed: [[Int]],
    batchSize: Int,
    targetN: Int,
    decodeNsByStream: [UInt64]
) throws -> RuntimeWorkerCohortFreeRunResult {
    guard streamsWithSeed.count == batchSize else {
        throw RuntimeWorkerCohortError.streamCount(
            expected: batchSize, got: streamsWithSeed.count)
    }
    guard decodeNsByStream.count == batchSize else {
        throw RuntimeWorkerCohortError.decodeNsByStreamCount(
            expected: batchSize, got: decodeNsByStream.count)
    }
    var tokensByStream: [[Int]] = []
    tokensByStream.reserveCapacity(batchSize)
    for (slot, stream) in streamsWithSeed.enumerated() {
        // Seed + N committed tokens. A longer collection is legal (the engine
        // may have raced a few tokens past N before the cohort was cancelled;
        // they were produced inside the timed window and are discarded) — a
        // shorter one is a broken cohort.
        guard stream.count >= targetN + 1 else {
            throw RuntimeWorkerCohortError.streamTokenCount(
                slot: slot, expected: targetN, got: Swift.max(stream.count - 1, 0))
        }
        tokensByStream.append(Array(stream.dropFirst().prefix(targetN)))
    }
    return RuntimeWorkerCohortFreeRunResult(
        tokensByStream: tokensByStream,
        acceptanceLengths: Array(repeating: 1, count: targetN),
        naturalAcceptedByStream: Array(
            repeating: Array(repeating: 1, count: targetN), count: batchSize),
        activeStreamsByRound: Array(repeating: batchSize, count: targetN),
        draftedTotal: 0,
        acceptedTotal: 0,
        committedTotal: batchSize * targetN,
        depthClampReasons: [:],
        decodeNsByStream: decodeNsByStream
    )
}

/// Assemble the MTP cohort result from B collected `(tokens, chunkSizes,
/// finished)` triples (`RuntimeWorkerCohortStreamCollector.waitForRounds`)
/// plus the engine's own MTP metrics snapshots straddling the run. See
/// `RuntimeWorkerCohortSession.runMTP`'s header for the accounting: which
/// fields are real observations of the vendored `CBv2MTPRoundDriver` and
/// which (`naturalAcceptedByStream`) is a documented, invariant-safe floor
/// rather than an independently measured value — the public `EngineV2`
/// surface has no seam for the true per-row pre-min accept walk.
///
/// `chunkSizes[0]` for every slot is the seed step (always exactly 1 — the
/// prefill-bonus token `free_decode_begin` already returned as
/// `seed_token_by_stream[slot]`); every entry after it is one REAL MTP
/// verify round's committed width for that slot.
func assembleMTPCohortFreeRun(
    perSlot: [(tokens: [Int], chunkSizes: [Int], finished: CBv2FinishReason?)],
    batchSize: Int,
    targetN: Int,
    seedTokenCount: Int,
    baselineMetrics: CBv2MTPMetrics?,
    finalMetrics: CBv2MTPMetrics?,
    decodeNsByStream: [UInt64]
) throws -> RuntimeWorkerCohortFreeRunResult {
    guard perSlot.count == batchSize else {
        throw RuntimeWorkerCohortError.streamCount(expected: batchSize, got: perSlot.count)
    }
    guard decodeNsByStream.count == batchSize else {
        throw RuntimeWorkerCohortError.decodeNsByStreamCount(
            expected: batchSize, got: decodeNsByStream.count)
    }
    var tokensByStream: [[Int]] = []
    tokensByStream.reserveCapacity(batchSize)
    var postSeedRoundsByStream: [[Int]] = []
    postSeedRoundsByStream.reserveCapacity(batchSize)
    for (slot, entry) in perSlot.enumerated() {
        guard entry.tokens.count >= targetN + 1 else {
            throw RuntimeWorkerCohortError.streamTokenCount(
                slot: slot, expected: targetN, got: Swift.max(entry.tokens.count - 1, 0))
        }
        guard entry.chunkSizes.first == 1 else {
            throw RuntimeWorkerCohortError.mtpSeedChunkMalformed(
                slot: slot, got: entry.chunkSizes.first ?? 0)
        }
        tokensByStream.append(Array(entry.tokens.dropFirst().prefix(targetN)))
        postSeedRoundsByStream.append(Array(entry.chunkSizes.dropFirst()))
    }

    guard let finalMetrics else {
        throw RuntimeWorkerCohortError.mtpMetricsMissing
    }

    // Cross-row consistency, reconciled against the engine's OWN per-round
    // finalize records instead of forced chunk-sequence equality.
    //
    // The previous check demanded every slot's trimmed chunk history equal
    // slot 0's round-for-round. That assumption is TOO STRONG and refused
    // healthy cohorts: per-row round HISTORIES legally offset by plain
    // rounds — the engine schedules a row's carry-(re)establishing plain
    // round at slightly different round indices per row — while every
    // VERIFY round itself commits its participating rows' uniform
    // min-across-rows width (the accept walk and rollback are per-row
    // correct; `CBv2MTPRoundAuditRecord` is the finalize-boundary proof).
    // Discovered by the acceptance-rule audit
    // (Tests/MLXFastTests/MTPAcceptanceRuleAuditTests.swift,
    // crossRowMinCommitsStayTokenExactUnderDivergentRowAcceptance).
    //
    // What is now CHECKED, per slot, over the committed window:
    //   * every multi-token chunk is a RECORDED verify round at exactly its
    //     cumulative position, with the recorded confirmed width
    //     (`mtpChunkWithoutVerifyAudit` / `mtpAuditWidthDisagreement`);
    //   * every unrecorded chunk commits exactly one token (a plain round);
    //   * each slot's widths sum to exactly N (`trimmedRoundWidths` below);
    //   * the audit ring did not truncate (`mtpRoundAuditsTruncated`), so
    //     coverage is proven, not assumed.
    // What stays REFUSED (unchanged): short streams, malformed seed chunks,
    // missing metrics, stream-count/timing-vector mismatches.
    guard finalMetrics.roundAudits.count < CBv2MTPRoundAuditRecord.retainedRecordCap
    else {
        throw RuntimeWorkerCohortError.mtpRoundAuditsTruncated(
            count: finalMetrics.roundAudits.count)
    }
    for slot in 0..<batchSize {
        try reconcileSlotAgainstRoundAudits(
            slot: slot,
            postSeedChunks: postSeedRoundsByStream[slot],
            targetN: targetN,
            seedTokenCount: seedTokenCount,
            audits: finalMetrics.roundAudits.filter { $0.requestID == UInt64(slot) })
    }

    // The cohort's representative per-round committed-width profile: slot
    // 0's post-seed chunk sequence trimmed to sum exactly N, using the same
    // clamp-the-final-round discipline `RuntimeWorkerFreeRunBuilder` uses
    // for the single-stream leg. With legally-offset histories the slots'
    // profiles can differ in segmentation (never in total); slot 0's is the
    // deterministic representative the AUDIT fields report — benchd's
    // cohort audit is informational and never scored (bench-core
    // `CohortFreeRunAudit`'s own doc), and the per-slot reconciliation
    // above is what actually anchors every slot to the engine's records.
    let acceptanceLengths = try trimmedRoundWidths(
        postSeedRoundsByStream[0], targetN: targetN, slot: 0)
    let rounds = acceptanceLengths.count
    for slot in 1..<batchSize {
        // Every slot must still individually reach exactly N (its own
        // trimmed profile validates totals even though its segmentation is
        // no longer forced equal to slot 0's).
        _ = try trimmedRoundWidths(
            postSeedRoundsByStream[slot], targetN: targetN, slot: slot)
    }
    let baselineDrafted = baselineMetrics?.draftedTokens ?? 0
    let baselineAccepted = baselineMetrics?.acceptedTokens ?? 0
    let draftedTotal = Swift.max(0, finalMetrics.draftedTokens - baselineDrafted)
    let acceptedTotal = Swift.max(0, finalMetrics.acceptedTokens - baselineAccepted)

    // DOCUMENTED FLOOR, not an observation (see this function's header): the
    // true per-row PRE-min natural accept walk is not observable from the
    // public engine surface, so every row is reported at exactly the
    // committed common width for that round — the largest value provably
    // `>= committed` (equality) without fabricating a number this harness
    // never measured.
    let naturalAcceptedByStream = Array(repeating: acceptanceLengths, count: batchSize)
    // D4 closed cohort, no stop tokens, equal budget: every round is B-wide
    // (checked above, not assumed — the round-history agreement check would
    // have failed if any row dropped out early).
    let activeStreamsByRound = Array(repeating: batchSize, count: rounds)

    return RuntimeWorkerCohortFreeRunResult(
        tokensByStream: tokensByStream,
        acceptanceLengths: acceptanceLengths,
        naturalAcceptedByStream: naturalAcceptedByStream,
        activeStreamsByRound: activeStreamsByRound,
        draftedTotal: draftedTotal,
        acceptedTotal: acceptedTotal,
        committedTotal: batchSize * targetN,
        depthClampReasons: mtpDepthClampReasons(
            baseline: baselineMetrics, final: finalMetrics),
        decodeNsByStream: decodeNsByStream
    )
}

/// Trim a slot's raw post-seed chunk-size sequence to sum exactly `targetN`,
/// clamping the final round the same way `RuntimeWorkerFreeRunBuilder` does
/// (a round that would overshoot N commits only its remaining share). Throws
/// if the sequence runs out before reaching N — a broken assembly, since the
/// caller already checked `tokens.count >= targetN + 1` for this slot.
private func trimmedRoundWidths(
    _ chunkSizes: [Int], targetN: Int, slot: Int
) throws -> [Int] {
    var widths: [Int] = []
    var committed = 0
    for size in chunkSizes {
        guard committed < targetN else { break }
        let take = Swift.min(size, targetN - committed)
        widths.append(take)
        committed += take
    }
    guard committed == targetN else {
        throw RuntimeWorkerCohortError.streamTokenCount(
            slot: slot, expected: targetN, got: committed)
    }
    return widths
}

/// Reconcile one cohort slot's committed chunk stream against the engine's
/// own per-verify-round finalize records (`CBv2MTPRoundAuditRecord`) for that
/// slot's request id — the cohort session submits `CBv2RequestID(slot)`, so
/// slot and request id coincide by construction
/// (`RuntimeWorkerCohortSession.init`).
///
/// Walk both sequences in order over the committed window (cumulative
/// committed count, seed included, up to `targetN + 1`):
///
///   * an audit record whose position (`tokensCountAfter - seedTokenCount`,
///     i.e. seed + committed run tokens) matches the chunk's end must agree
///     on the width (`confirmed == chunk width`) — else
///     `mtpAuditWidthDisagreement`;
///   * a chunk with NO audit at its end position must be a plain round
///     committing exactly one token — else `mtpChunkWithoutVerifyAudit`;
///   * audits past the window (the engine legally races rounds beyond N
///     before the cancel lands) are ignored, as are chunks past it.
///
/// The final in-window chunk may STRADDLE `targetN + 1` (the engine committed
/// a full verify width whose tail the budget trims); its audit must still
/// agree on the RAW width — trimming is presentation, not engine state.
private func reconcileSlotAgainstRoundAudits(
    slot: Int,
    postSeedChunks: [Int],
    targetN: Int,
    seedTokenCount: Int,
    audits: [CBv2MTPRoundAuditRecord]
) throws {
    var cumulative = 1  // the seed chunk (validated == 1 by the caller)
    var auditCursor = 0
    for width in postSeedChunks {
        guard cumulative < targetN + 1 else { break }
        let end = cumulative + width
        // Skip any audit records the engine finalized at positions this
        // walk has already passed without matching — impossible for a
        // healthy stream (every verify commit IS a delta), so surfacing the
        // stale record as a width disagreement at its own position keeps
        // the refusal precise.
        while auditCursor < audits.count,
            audits[auditCursor].tokensCountAfter - seedTokenCount < end
                && audits[auditCursor].tokensCountAfter - seedTokenCount != end
        {
            let position = audits[auditCursor].tokensCountAfter - seedTokenCount
            if position <= cumulative {
                // Audit strictly behind the walk: recorded commit the stream
                // never carried.
                throw RuntimeWorkerCohortError.mtpAuditWidthDisagreement(
                    slot: slot,
                    cumulative: position,
                    auditConfirmed: audits[auditCursor].confirmed,
                    chunkWidth: 0)
            }
            // Audit strictly inside this chunk's span: the chunk and the
            // record disagree on segmentation.
            throw RuntimeWorkerCohortError.mtpAuditWidthDisagreement(
                slot: slot,
                cumulative: position,
                auditConfirmed: audits[auditCursor].confirmed,
                chunkWidth: width)
        }
        if auditCursor < audits.count,
            audits[auditCursor].tokensCountAfter - seedTokenCount == end
        {
            let audit = audits[auditCursor]
            guard audit.confirmed == width else {
                throw RuntimeWorkerCohortError.mtpAuditWidthDisagreement(
                    slot: slot,
                    cumulative: end,
                    auditConfirmed: audit.confirmed,
                    chunkWidth: width)
            }
            auditCursor += 1
        } else {
            // No finalize record ends here: this must be a plain
            // (non-drafting) round, which commits exactly one token.
            guard width == 1 else {
                throw RuntimeWorkerCohortError.mtpChunkWithoutVerifyAudit(
                    slot: slot, cumulative: end, width: width)
            }
        }
        cumulative = end
    }
}

/// The engine's real depth-clamp / skip reason histogram over an MTP
/// session's generation window: `finalMetrics` diffed against
/// `baselineMetrics` (captured right after the seed step, before any round
/// ran), merging the two cumulative reason dictionaries `CBv2MTPMetrics`
/// tracks — `skippedRows` (rows clamped OUT of a round entirely: batch
/// gate, KV headroom, invalid carry) and `controllerFallbacks` (the depth
/// controller's own selection reasons, including e.g. `tail_depth` /
/// `automatic_rectangular_limit`). Both are "reasons this driver did not run
/// full depth"; the wire carries one map, so they are summed key-wise rather
/// than arbitrarily picking one. Shared by the single-stream and cohort MTP
/// assemblers (the single-stream wire shape has no `depth_clamp_reasons`
/// field today, but the merge itself is leg-agnostic).
func mtpDepthClampReasons(
    baseline: CBv2MTPMetrics?, final: CBv2MTPMetrics
) -> [String: Int] {
    var result: [String: Int] = [:]
    func accumulate(_ finalCounts: [String: Int], _ baselineCounts: [String: Int]) {
        for (reason, finalCount) in finalCounts {
            let delta = finalCount - (baselineCounts[reason] ?? 0)
            guard delta > 0 else { continue }
            result[reason, default: 0] += delta
        }
    }
    accumulate(final.skippedRows, baseline?.skippedRows ?? [:])
    accumulate(final.controllerFallbacks, baseline?.controllerFallbacks ?? [:])
    return result
}

// MARK: - Contiguous KV byte budget

/// Worst-case contiguous-KV byte demand for a closed cohort of `batchSize`
/// streams, each at most `maxSequenceLength` tokens (seed + generation), plus
/// 2x headroom for allocator growth-by-doubling and reservation rounding.
///
/// Per non-shared layer: K and V, `kvHeads * headDim` fp16 elements per token.
/// A windowed layer's contiguous row owns its whole fixed ring from the first
/// write, so it is charged the full window regardless of sequence length;
/// KV-shared layers own no storage. The budget is an admission-ledger ceiling
/// (`CBv2ContiguousBackendConfig.bytesCapacity`), not an allocation — sizing it
/// to the closed cohort's worst case means truthful admission can never
/// reject a cohort this engine already agreed to run, while a runaway request
/// shape still fails loudly at submit.
func cohortContiguousKVBytesBudget(
    layerKinds: [CBv2LayerKind],
    batchSize: Int,
    maxSequenceLength: Int
) -> Int {
    let bytesPerElement = 2  // fp16 (CBv2ContiguousBackendConfig default kvDType)
    var perStreamBytes = 0
    for kind in layerKinds where kind.sharesKVWithLayer == nil {
        let rowTokens: Int
        switch kind.attention {
        case .full:
            rowTokens = maxSequenceLength
        case .slidingWindow(let window):
            rowTokens = window
        }
        perStreamBytes += 2 * kind.kvHeads * kind.headDim * bytesPerElement * rowTokens
    }
    return perStreamBytes * batchSize * 2
}
