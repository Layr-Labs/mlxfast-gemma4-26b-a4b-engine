import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLLM
import MLXLMCommon

// COHORT REFERENCE-REPLAY ORACLE — MEASUREMENT MODE (fidelity-gate PR-1).
//
// WHY THIS EXISTS. On this quantized MoE, forward outputs at near-ties depend
// on the batch/verify WIDTH a forward is invoked at (port-notes 3.1: ~6% of
// pool positions flip argmax across widths, and the near-tie pool can be
// >= 3 deep — measured). An exact-token gate against a purely sequential
// golden therefore rejects HONEST MTP candidates whose only "error" is
// width-induced numerics. The challenger fix is to validate a free-run
// candidate by having a PINNED REFERENCE replay the candidate's OWN emitted
// journal, teacher-forced, at the reference's CANONICAL WIDTH-1 — producing the
// reference's per-position argmax + logits GIVEN THE CANDIDATE'S CONTEXT. A
// STATIC tape cannot do this: it is anchored to the reference's own token
// chain, so every row after the first legitimate divergence compares the wrong
// contexts (the exact structural blindness `DFlashLiveReferenceOracle` was
// built to avoid — `Gemma4RuntimeDFlashDriver.swift:179-191`, and the
// self-consistency half of `Gemma4RuntimeDFlash.swift:validateJournalAgainstReference`).
//
// WHAT THIS FILE BUILDS, AND WHAT IT DOES NOT. It builds ONLY the reference-
// replay ORACLE and its MEASUREMENT output. It is INTERPRETATION-AGNOSTIC: it
// renders NO admit/reject verdict, defines NO admission RULE, and pins NO
// budget/envelope admission constant. The admission ladder, the David-set 10%
// budget, and the calibrated envelope arrive in LATER PRs and consume this
// report. It arms nothing — there is no scored path here,
// `official_scoring_enabled` stays false, and go-live is a separate David gate.
//
// TRUSTED-SIDE INVARIANT (the anti-gaming core). The logits ranked here are the
// PINNED reference's own (this worker's organizer-transformed weights, loaded
// through `weightCache.requireLibraryModel()`), NEVER the candidate's. Nothing
// candidate-authored enters the readout: the request carries only token ids
// (seeds + committed journals), and the reference model produces every logit.
// The wire verb runs AFTER candidate teardown (the trusted parent spawns this
// worker from the pinned baseline tree over organizer weights; the candidate is
// gone), so the two weight sets never coexist.
//
// REPLAY WIDTH — PARAMETERIZED (`cohort` width-B vs `canonical` width-1). The
// request selects the reference's replay WIDTH via `replay_width`
// (`"cohort" | "canonical"`), defaulting to `"cohort"`:
//
//   * `"cohort"` (default, David-ruled). The B streams are replayed BATCHED,
//     teacher-forced, at `[B, 1]` steps on ONE fresh batched KV — the SAME
//     batch-width geometry the scored candidate's cohort free-run runs at
//     (`Gemma4RuntimeCohortDriver.swift`: B streams through the CBv2 scheduler
//     decode in lockstep `[B, 1]` rounds). This is the LIKE-FOR-LIKE comparison
//     the gate needs. On this quantized MoE the layer-0 quantized Q/K
//     projection selects a DIFFERENT reduction path at `[B, 1]` than at `[1, 1]`
//     (port-notes 3.1: shape-dependent quantized-kernel divergence, first
//     divergence layer-0 Q/K), so a candidate scored at BATCH-B against a
//     WIDTH-1 reference shows a FALSE ~16% divergence that is pure batch
//     geometry — the honest stock MTP engine and the serial baseline both
//     exhibit it. A width-1 diagnostic confirmed the same candidate diverges
//     ~3.9% at width-1 vs ~16.4% at batch-8. Replaying the reference at the
//     cohort width prices that geometry out on BOTH sides, so what remains is
//     the candidate's REAL (few-percent) divergence — which is what the gate
//     must measure. The cohort path REQUIRES a RECTANGULAR cohort (equal seed
//     lengths AND equal committed-journal lengths) so the B streams step in
//     lockstep; a ragged cohort is refused, never reshaped.
//
//   * `"canonical"` (opt-in diagnostic). Each stream is replayed on ITS OWN
//     fresh width-1 KV cache, stepped `[1, 1]` one token at a time — the
//     original per-stream path, RETAINED (not destroyed) so the width-1
//     diagnostic that measured the geometry gap stays available and the change
//     is reviewable/reversible. This is also the falsifiable GPU-free seam the
//     anti-static-tape tests exercise without a model.
//
// BOTH modes teacher-force on the CANDIDATE's own committed journal (the
// anti-static-tape anchor below) and feed the SAME pure readout core
// (`cohortReferenceReplayReadout`), so the ranked/relative near-tie math and the
// per stream x position report shape are IDENTICAL across widths — only the
// forward's batch geometry differs. benchd's (b) gate therefore consumes the
// report unchanged; only the argmax/logit VALUES move to the like-for-like
// batch-B geometry.
//
// AND THE REPORT SAYS WHICH WIDTH RAN (`replay_width`). Because the two modes
// produce the same SHAPE with different VALUES, a report is not self-describing
// without the width: two reports over the same cohort can disagree entirely and
// both be correct. The width also defaults engine-side, so a request that omits
// `replay_width` used to leave no trace anywhere of the geometry that actually
// produced the numbers. The report therefore STAMPS the resolved width, taken
// from the branch that ran — the enforced reference geometry becomes a property
// of the sealed artifact instead of an unrecorded engine-side default.
//
// REUSE MAP (path@sha:line where load-bearing).
//   * The per-position ranked/relative near-tie math is the width probe's
//     already-merged, box-swept `rankedReferenceCharacterization`
//     (`Gemma4RuntimeWidthProbe.swift:238-253`) — reused verbatim, not re-derived.
//     This file ADDS only the committed-token relative gap the width probe does
//     not compute (it ranks the top-N, not a supplied token).
//   * The teacher-forcing anchor (row N+1 conditioned on the CANDIDATE's
//     committed token N, grown only after row N is read) mirrors
//     `validateJournalAgainstReference` (`Gemma4RuntimeDFlash.swift:688-725`) and
//     the prefix-equality proof of `DFlashLiveReferenceOracle.referenceBatch`
//     (`Gemma4RuntimeDFlashDriver.swift:179-191`) — adapted to a LIVE per-stream
//     width-1 replay, NOT the RETIRED `dflash_reference_rows` wire verb (that
//     verb is a client-side stub only and is unhandled by the worker dispatch
//     switch — `Gemma4RuntimeWorker.swift:913` default → "unknown request kind").
//   * The post-softcap logit provenance is derived from the model config, not
//     hardcoded: `applyLMHead` applies gemma's final-logit softcap iff
//     `config.finalLogitSoftcapping > 0`
//     (`Gemma4Text.swift:2001-2015`), so the provenance stamp reads that same
//     config value here.

// MARK: - Replay width mode (parameterized; canonical width-1 vs cohort width-B)

/// The width the pinned reference replays each cohort at. `cohort` batches the B
/// streams into `[B, 1]` lockstep steps (the scored candidate's geometry, the
/// David-ruled default); `canonical` replays each stream on its own width-1 KV
/// (`[1, 1]`, the retained diagnostic path). Wire value is the raw string on the
/// `replay_width` request field.
enum CohortReferenceReplayWidth: String, Equatable, CaseIterable {
    case cohort
    case canonical
}

/// The default replay width when a `cohort_reference_replay` request omits
/// `replay_width`. `cohort` (batch-B) by David's ruling: the reference must
/// replay at the scored candidate's cohort width so the comparison is
/// like-for-like and the false batch-geometry divergence is priced out on both
/// sides. `canonical` (width-1) stays selectable for the diagnostic.
let cohortReferenceReplayDefaultWidth: CohortReferenceReplayWidth = .cohort

// MARK: - Measurement defaults (characterization params, NOT admission constants)

/// Default ranked readout depth K. The task fixes the cohort readout at K=16;
/// the width probe sweeps the same knob (`Gemma4RuntimeWidthProbe.swift:85-88`).
/// This is a MEASUREMENT readout width — how many ranks are reported — not an
/// admission budget.
let cohortReferenceReplayDefaultLogitTopK = 16

/// Default within-envelope relative-ratio threshold used to compute the
/// informational `within_envelope_depth`. Identical to the width probe's
/// documented generous default (`Gemma4RuntimeWidthProbe.swift:90-93`). It is a
/// CHARACTERIZATION sweep parameter, NOT the admission envelope: the report also
/// emits the RAW `ranked_relative_gaps` and `committed_relative_gap`, so the
/// calibration PR re-derives the depth at whatever envelope it later pins. No
/// admit/reject decision reads this value in this PR.
let cohortReferenceReplayDefaultRelEnvelope = 0.05

// MARK: - Structured measurement output

/// One reference position's readout, teacher-forced on the candidate's own
/// committed prefix. Every field is a property of the PINNED reference's forward
/// at that position GIVEN the candidate's context. Renders no verdict.
struct CohortReferenceReplayPosition: Codable, Equatable {
    /// The candidate's committed token at this position (the teacher-forced
    /// input, echoed so a consumer can see it without the request).
    let committedToken: Int
    /// The reference's own width-1 argmax at this position (`ranked_tokens[0]`),
    /// GIVEN the candidate's committed prefix. May differ from `committedToken`
    /// at a width-divergence — recorded, never rejected.
    let sequentialArgmax: Int
    /// Top-K reference token ids, rank-0 first (descending post-softcap logit).
    let rankedTokens: [Int]
    /// Top-K reference logit values (post-softcap), rank-0 first.
    let rankedLogits: [Double]
    /// Relative ratio of each rank's gap from top-1: `(top1 - rank_i)/max(1,|top1|)`.
    let rankedRelativeGaps: [Double]
    /// The reference's post-softcap logit for the COMMITTED token.
    let committedTokenLogit: Double
    /// The committed token's relative gap: `(top1 - logit(committed))/max(1,|top1|)`.
    /// Zero when the committed token IS the reference argmax; grows as the
    /// committed token sits deeper in the reference's distribution.
    let committedRelativeGap: Double
    /// Count of ranks whose relative ratio <= the characterization envelope
    /// (top-1 included, so >= 1). Depth >= 3 == "top-3+ within-envelope near-ties
    /// occur here". Informational; recomputable from `rankedRelativeGaps`.
    let withinEnvelopeDepth: Int

    enum CodingKeys: String, CodingKey, CaseIterable {
        case committedToken = "committed_token"
        case sequentialArgmax = "sequential_argmax"
        case rankedTokens = "ranked_tokens"
        case rankedLogits = "ranked_logits"
        case rankedRelativeGaps = "ranked_relative_gaps"
        case committedTokenLogit = "committed_token_logit"
        case committedRelativeGap = "committed_relative_gap"
        case withinEnvelopeDepth = "within_envelope_depth"
    }
}

/// One stream's replay: its slot index and per-position readouts, in emission
/// order (position i is teacher-forced on seed + committed[0..<i]).
struct CohortReferenceReplayStreamReport: Codable, Equatable {
    let slot: Int
    let positions: [CohortReferenceReplayPosition]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case slot
        case positions
    }
}

/// The per-cohort measurement report. Self-describing (provenance + the
/// characterization params it was computed at) so a downstream consumer needs
/// no out-of-band context. Carries no verdict, no score, no admission constant.
struct CohortReferenceReplayReport: Codable, Equatable {
    /// `post_softcap` when the reference config carries a positive final-logit
    /// softcap, `raw_no_softcap` otherwise. Derived from config, not hardcoded.
    let logitProvenance: String
    /// The ranked readout depth K the report was produced at.
    let logitTopK: Int
    /// The characterization envelope `within_envelope_depth` was computed at.
    let relEnvelope: Double
    /// The width the reference ACTUALLY replayed at — `"cohort"` (batched
    /// `[B, 1]`) or `"canonical"` (per-stream `[1, 1]`), the raw value of
    /// `CohortReferenceReplayWidth`.
    ///
    /// STAMPED FROM THE RESOLVED WIDTH, inside the branch that ran — never from
    /// the request, which may have omitted `replay_width` and silently taken
    /// `cohortReferenceReplayDefaultWidth`. Without this the enforced reference
    /// GEOMETRY — the thing that decides whether a divergence is real or pure
    /// batch numerics (see the width discussion at the top of this file) — rode
    /// an engine-side default that no sealed artifact recorded, so a consumer
    /// could not tell from the report which geometry produced it. A `String`
    /// (not the enum) because the trusted-side mirror links no MLX and must stay
    /// byte-parallel; same choice `logit_provenance` makes.
    let replayWidth: String
    let streams: [CohortReferenceReplayStreamReport]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case logitProvenance = "logit_provenance"
        case logitTopK = "logit_topk"
        case relEnvelope = "rel_envelope"
        case replayWidth = "replay_width"
        case streams
    }
}

// MARK: - Pure measurement core (GPU-free)

/// Rank one reference position's post-softcap logit vector and derive the
/// per-position readout, INCLUDING the committed token's relative gap. Pure over
/// an already-materialized `[V]` vector — the GPU-free seam tests exercise this
/// directly, and the model-backed forward feeds it the `applyLMHead` output.
///
/// Reuses `rankedReferenceCharacterization` (`Gemma4RuntimeWidthProbe.swift`) for
/// the ranked top-N + relative gaps + within-envelope depth; adds only the
/// committed-token gap, which the width probe does not compute.
func cohortReferenceReplayReadout(
    referenceLogits flat: [Float],
    committedToken: Int,
    logitTopK: Int,
    relEnvelope: Double
) throws -> CohortReferenceReplayPosition {
    guard committedToken >= 0, committedToken < flat.count else {
        throw MLXFastError.invalidInput(
            "cohort reference-replay committed token \(committedToken) is outside "
                + "the reference vocab [0, \(flat.count))")
    }
    let characterization = rankedReferenceCharacterization(
        logits: flat, logitTopK: logitTopK, relEnvelope: relEnvelope)
    let top1 = characterization.logits.first ?? 0
    let denominator = Swift.max(1.0, abs(top1))
    let committedLogit = Double(flat[committedToken])
    let committedRelativeGap = (top1 - committedLogit) / denominator
    return CohortReferenceReplayPosition(
        committedToken: committedToken,
        sequentialArgmax: characterization.tokens.first ?? -1,
        rankedTokens: characterization.tokens,
        rankedLogits: characterization.logits,
        rankedRelativeGaps: characterization.relativeGaps,
        committedTokenLogit: committedLogit,
        committedRelativeGap: committedRelativeGap,
        withinEnvelopeDepth: characterization.withinEnvelopeDepth)
}

// MARK: - The width-1 replay seam (GPU-free mockable)

/// One stream's reference forward, stepped at CANONICAL WIDTH-1. The seam that
/// lets the replay be exercised GPU-free (a synthetic forward in tests) while
/// the model-backed conformer drives the real pinned reference.
///
/// Contract: `prefill` establishes the seed and returns the post-softcap logits
/// predicting the FIRST post-seed position; each `step` feeds exactly ONE
/// teacher-forced token at width-1 and returns the logits predicting the NEXT
/// position. Implementations advance a single continuous width-1 KV cache — they
/// never batch.
protocol CohortReferenceReplayForward {
    mutating func prefill(seed: [Int]) -> [Float]
    mutating func step(feeding token: Int) -> [Float]
}

/// Replay ONE stream teacher-forced on the candidate's OWN committed journal.
///
/// The anti-static-tape anchor lives in the loop: to reach row `i+1` the driver
/// feeds `committed[i]` (the CANDIDATE's token), never the reference's own
/// argmax at row `i`. So row `i+1`'s logits are conditioned on the candidate's
/// real context — which a static tape, anchored to the reference's own chain,
/// cannot reproduce after the first divergence. `logits` is only ever advanced
/// AFTER the current row is read, so the readout at row `i` sees the reference
/// forward over seed + committed[0..<i].
func replayCohortReferenceStream<Forward: CohortReferenceReplayForward>(
    seed: [Int],
    committed: [Int],
    forward: inout Forward,
    logitTopK: Int,
    relEnvelope: Double
) throws -> [CohortReferenceReplayPosition] {
    guard !seed.isEmpty else {
        throw MLXFastError.invalidInput(
            "cohort reference-replay stream seed must not be empty")
    }
    guard !committed.isEmpty else {
        throw MLXFastError.invalidInput(
            "cohort reference-replay stream committed journal must not be empty")
    }
    var positions: [CohortReferenceReplayPosition] = []
    positions.reserveCapacity(committed.count)
    var logits = forward.prefill(seed: seed)
    for (index, token) in committed.enumerated() {
        positions.append(
            try cohortReferenceReplayReadout(
                referenceLogits: logits, committedToken: token,
                logitTopK: logitTopK, relEnvelope: relEnvelope))
        if index < committed.count - 1 {
            // Teacher-force on the CANDIDATE's committed token — the anchor.
            logits = forward.step(feeding: token)
        }
    }
    return positions
}

// MARK: - Model-backed width-1 forward (the pinned reference)

/// The pinned reference's width-1 forward over one continuous KV cache. Prefill
/// bulk-forwards the seed `[1, S]` and reads the last position's post-softcap
/// logits; each step forwards a single token `[1, 1]` and reads position 0.
/// `model(_:cache:)` is `applyLMHead(model(...))`, so every returned vector is
/// the exact post-softcap tensor the sampler consumes.
struct Gemma4CohortReferenceForward: CohortReferenceReplayForward {
    let model: Gemma4TextModel
    private var cache: [any KVCache]

    init(model: Gemma4TextModel) {
        self.model = model
        // FRESH width-1 cache per stream — the anti-width-divergence property.
        self.cache = model.newCache(parameters: nil)
    }

    mutating func prefill(seed: [Int]) -> [Float] {
        let input = MLXArray(seed.map(Int32.init)).reshaped([1, seed.count])
        let logits = model(input, cache: cache)
        return cohortReferenceReplayMaterializePosition(
            logits, position: seed.count - 1)
    }

    mutating func step(feeding token: Int) -> [Float] {
        // Canonical width-1: a single-token [1, 1] forward, never [B, 1].
        let input = MLXArray([Int32(token)]).reshaped([1, 1])
        let logits = model(input, cache: cache)
        return cohortReferenceReplayMaterializePosition(logits, position: 0)
    }
}

/// Slice one `[V]` position out of a `[1, L, V]` logits tensor and materialize
/// it off-device as `float32` for the pure-Swift core.
private func cohortReferenceReplayMaterializePosition(
    _ logits: MLXArray, position: Int
) -> [Float] {
    let row = logits[0 ..< 1, position ..< position + 1, 0...]
    eval(row)
    return row.asType(.float32).flattened().asArray(Float.self)
}

// MARK: - The cohort-width (batch-B) replay seam (GPU-free mockable)

/// The B streams' reference forward, stepped BATCHED at the cohort width `[B, 1]`
/// (the scored candidate's geometry). The seam that lets the batched replay be
/// exercised GPU-free (a synthetic batched forward in tests) while the
/// model-backed conformer drives the real pinned reference.
///
/// Contract (mirror of `CohortReferenceReplayForward`, widened to B rows):
/// `prefill` bulk-forwards every stream's (equal-length) seed as one `[B, S]`
/// batch and returns each stream's post-softcap logits predicting its FIRST
/// post-seed position (`[B][V]`, slot order); each `step` feeds exactly ONE
/// teacher-forced token PER STREAM at `[B, 1]` and returns each stream's logits
/// predicting its next position. Implementations advance ONE batched KV cache
/// across all B rows in lockstep — the geometry that reproduces the batch-width
/// numerics (port-notes 3.1).
protocol CohortReferenceReplayBatchedForward {
    /// `seedsByStream` is `[B][S]`, rectangular (every stream the same seed
    /// length). Returns `[B][V]`.
    mutating func prefill(seedsByStream: [[Int]]) -> [[Float]]
    /// `tokens` is one teacher-forced token per stream (`[B]`, slot order).
    /// Returns `[B][V]`.
    mutating func step(feedingByStream tokens: [Int]) -> [[Float]]
}

/// Replay the WHOLE cohort BATCHED, teacher-forced on each stream's OWN committed
/// journal, at the cohort width `[B, 1]`. Returns per-stream position readouts
/// (`[B][positions]`, slot order).
///
/// The anti-static-tape anchor is the SAME as the width-1 path, applied per
/// stream in lockstep: to reach row `i+1` the driver feeds `committed[*][i]`
/// (each CANDIDATE's own token), never the reference's argmax at row `i`. Every
/// stream shares one batched forward, so its row `i` logits are the reference's
/// forward over `seed + committed[0..<i]` AT THE COHORT WIDTH — the like-for-like
/// geometry a per-stream width-1 replay cannot produce.
///
/// REQUIRES a rectangular cohort: all seeds equal length AND all committed
/// journals equal length (the lockstep `[B, 1]` walk feeds one token per stream
/// per step). A ragged cohort is refused — the caller validates this up front,
/// and this driver re-checks fail-closed.
func replayCohortReferenceBatched<Forward: CohortReferenceReplayBatchedForward>(
    seedsByStream: [[Int]],
    committedByStream: [[Int]],
    forward: inout Forward,
    logitTopK: Int,
    relEnvelope: Double
) throws -> [[CohortReferenceReplayPosition]] {
    let batchSize = seedsByStream.count
    guard batchSize >= 1 else {
        throw MLXFastError.invalidInput(
            "cohort-width reference-replay requires at least one stream")
    }
    guard committedByStream.count == batchSize else {
        throw MLXFastError.invalidInput(
            "cohort-width reference-replay has \(batchSize) seed streams but "
                + "\(committedByStream.count) committed streams")
    }
    let seedLength = seedsByStream[0].count
    let committedLength = committedByStream[0].count
    guard seedLength >= 1 else {
        throw MLXFastError.invalidInput(
            "cohort-width reference-replay stream 0 seed must not be empty")
    }
    guard committedLength >= 1 else {
        throw MLXFastError.invalidInput(
            "cohort-width reference-replay stream 0 committed journal must not be "
                + "empty")
    }
    for slot in seedsByStream.indices {
        guard seedsByStream[slot].count == seedLength else {
            throw MLXFastError.invalidInput(
                "cohort-width reference-replay stream \(slot) has "
                    + "\(seedsByStream[slot].count) seed tokens but stream 0 has "
                    + "\(seedLength); the batched replay requires a rectangular "
                    + "(equal-seed-length) cohort")
        }
        guard committedByStream[slot].count == committedLength else {
            throw MLXFastError.invalidInput(
                "cohort-width reference-replay stream \(slot) committed journal has "
                    + "\(committedByStream[slot].count) tokens but stream 0 has "
                    + "\(committedLength); the batched replay requires a rectangular "
                    + "(equal-committed-length) cohort so the B streams step in "
                    + "lockstep")
        }
    }

    var positionsByStream: [[CohortReferenceReplayPosition]] = Array(
        repeating: [], count: batchSize)
    for slot in positionsByStream.indices {
        positionsByStream[slot].reserveCapacity(committedLength)
    }
    var logitsByStream = forward.prefill(seedsByStream: seedsByStream)
    guard logitsByStream.count == batchSize else {
        throw MLXFastError.invalidInput(
            "cohort-width reference-replay prefill returned "
                + "\(logitsByStream.count) logit rows, expected B=\(batchSize)")
    }
    for index in 0 ..< committedLength {
        for slot in 0 ..< batchSize {
            positionsByStream[slot].append(
                try cohortReferenceReplayReadout(
                    referenceLogits: logitsByStream[slot],
                    committedToken: committedByStream[slot][index],
                    logitTopK: logitTopK, relEnvelope: relEnvelope))
        }
        if index < committedLength - 1 {
            // Teacher-force ONE candidate token per stream — the batched anchor.
            let feed = committedByStream.map { $0[index] }
            logitsByStream = forward.step(feedingByStream: feed)
            guard logitsByStream.count == batchSize else {
                throw MLXFastError.invalidInput(
                    "cohort-width reference-replay step returned "
                        + "\(logitsByStream.count) logit rows, expected "
                        + "B=\(batchSize)")
            }
        }
    }
    return positionsByStream
}

// MARK: - Model-backed cohort-width (batch-B) forward (the pinned reference)

/// The pinned reference's BATCHED forward over one contiguous KV cache holding
/// all B rows in lockstep — the cohort width. Prefill bulk-forwards the B
/// (equal-length) seeds as `[B, S]` and reads each row's last-position
/// post-softcap logits; each step forwards one token per row `[B, 1]` and reads
/// position 0 of every row. `model(_:cache:)` is `applyLMHead(model(...))`, so
/// every returned vector is the exact post-softcap tensor the sampler consumes —
/// AND, crucially, the projections/FFN are invoked at the SAME `[B, ...]` batch
/// width the scored candidate's cohort free-run uses, reproducing the batch-width
/// quantized-matmul reduction path (port-notes 3.1) instead of the width-1 one.
struct Gemma4CohortReferenceBatchedForward: CohortReferenceReplayBatchedForward {
    let model: Gemma4TextModel
    private var cache: [any KVCache]

    init(model: Gemma4TextModel) {
        self.model = model
        // ONE fresh batched cache for the whole cohort — the rows are added on
        // the first (prefill) update from `keys.dim(0)`, so the same per-layer
        // Standard/Rotating caches the width-1 path uses hold `[B, ...]` here.
        self.cache = model.newCache(parameters: nil)
    }

    mutating func prefill(seedsByStream: [[Int]]) -> [[Float]] {
        let batchSize = seedsByStream.count
        let seedLength = seedsByStream[0].count
        // Row-major [B, S] int32.
        let flat = seedsByStream.flatMap { $0.map(Int32.init) }
        let input = MLXArray(flat).reshaped([batchSize, seedLength])
        let logits = model(input, cache: cache)
        return cohortReferenceReplayMaterializeBatchPositions(
            logits, position: seedLength - 1, batchSize: batchSize)
    }

    mutating func step(feedingByStream tokens: [Int]) -> [[Float]] {
        let batchSize = tokens.count
        // Cohort width: one column, B rows — [B, 1], never [1, 1] per stream.
        let input = MLXArray(tokens.map(Int32.init)).reshaped([batchSize, 1])
        let logits = model(input, cache: cache)
        return cohortReferenceReplayMaterializeBatchPositions(
            logits, position: 0, batchSize: batchSize)
    }
}

/// Slice ONE position out of every row of a `[B, L, V]` logits tensor and
/// materialize the resulting `[B, V]` off-device as `float32`, returned as B
/// per-stream `[V]` vectors (slot order) for the pure-Swift core.
private func cohortReferenceReplayMaterializeBatchPositions(
    _ logits: MLXArray, position: Int, batchSize: Int
) -> [[Float]] {
    let rows = logits[0..., position ..< position + 1, 0...]
    eval(rows)
    let vocab = rows.dim(2)
    let flat = rows.asType(.float32).flattened().asArray(Float.self)
    var byStream: [[Float]] = []
    byStream.reserveCapacity(batchSize)
    for slot in 0 ..< batchSize {
        let start = slot * vocab
        byStream.append(Array(flat[start ..< start + vocab]))
    }
    return byStream
}

// MARK: - Driver

extension Gemma4Runtime {
    /// Build the per-cohort reference-replay MEASUREMENT report, teacher-forced
    /// on each candidate's own committed journal. The logits are the pinned
    /// reference's own (`model` is `weightCache.requireLibraryModel()` in the
    /// handler). Renders no verdict.
    ///
    /// `replayWidth` selects the reference's forward geometry (defaulting to the
    /// David-ruled `cohort`):
    ///
    ///   * `.cohort` — the B streams are replayed BATCHED at `[B, 1]` on ONE
    ///     fresh batched KV, reproducing the scored candidate's cohort geometry
    ///     so the comparison is like-for-like (the batch-width numerics are
    ///     priced out on both sides). Requires a rectangular cohort.
    ///   * `.canonical` — each stream is replayed independently on its own fresh
    ///     width-1 KV (`[1, 1]`), the retained diagnostic path.
    ///
    /// The report shape and the per-position ranked/relative math are IDENTICAL
    /// across widths — only the forward geometry (and thus the values) differ,
    /// which is exactly why the returned report STAMPS the width that ran in
    /// `replayWidth` (wire `replay_width`).
    static func cohortReferenceReplayReport(
        model: Gemma4TextModel,
        seedsByStream: [[Int]],
        committedByStream: [[Int]],
        logitTopK: Int,
        relEnvelope: Double,
        replayWidth: CohortReferenceReplayWidth = cohortReferenceReplayDefaultWidth
    ) throws -> CohortReferenceReplayReport {
        guard seedsByStream.count == committedByStream.count else {
            throw MLXFastError.invalidInput(
                "cohort reference-replay has \(seedsByStream.count) seed streams "
                    + "but \(committedByStream.count) committed streams")
        }
        // Provenance derived from config, not hardcoded: the softcap is applied
        // inside `applyLMHead` iff this value is positive.
        let provenance =
            model.configuration.finalLogitSoftcapping > 0
            ? "post_softcap" : "raw_no_softcap"

        let streams: [CohortReferenceReplayStreamReport]
        // Stamped INSIDE each branch, so the reported width is the geometry that
        // actually ran rather than a re-read of the parameter — a future early
        // return or fallback cannot leave the stamp describing a path not taken.
        let replayedWidth: CohortReferenceReplayWidth
        switch replayWidth {
        case .cohort:
            // BATCHED [B, 1] on one shared KV — the scored candidate's geometry.
            var forward = Gemma4CohortReferenceBatchedForward(model: model)
            let positionsByStream = try replayCohortReferenceBatched(
                seedsByStream: seedsByStream,
                committedByStream: committedByStream,
                forward: &forward,
                logitTopK: logitTopK,
                relEnvelope: relEnvelope)
            streams = positionsByStream.enumerated().map { slot, positions in
                CohortReferenceReplayStreamReport(slot: slot, positions: positions)
            }
            replayedWidth = .cohort
        case .canonical:
            // Per-stream width-1 [1, 1] on its own fresh KV — the diagnostic.
            var built: [CohortReferenceReplayStreamReport] = []
            built.reserveCapacity(seedsByStream.count)
            for slot in seedsByStream.indices {
                var forward = Gemma4CohortReferenceForward(model: model)
                let positions = try replayCohortReferenceStream(
                    seed: seedsByStream[slot],
                    committed: committedByStream[slot],
                    forward: &forward,
                    logitTopK: logitTopK,
                    relEnvelope: relEnvelope)
                built.append(
                    CohortReferenceReplayStreamReport(slot: slot, positions: positions))
            }
            streams = built
            replayedWidth = .canonical
        }

        return CohortReferenceReplayReport(
            logitProvenance: provenance,
            logitTopK: logitTopK,
            relEnvelope: relEnvelope,
            replayWidth: replayedWidth.rawValue,
            streams: streams)
    }
}
