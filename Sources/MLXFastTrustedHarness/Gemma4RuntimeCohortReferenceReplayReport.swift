import Foundation

// TRUSTED-SIDE MIRROR of the cohort reference-replay measurement report
// (fidelity-gate PR-1). The worker-support target
// (`MLXFastRuntimeWorkerSupport`) owns the PRODUCING side — it links MLX and
// runs the pinned reference forward. The trusted binary links no MLX, so it
// carries its OWN pure Codable mirror of the report types (the same pattern as
// `CorrectnessTraceLogit`, `RuntimeWorkerEffectiveSpec`, etc., which are
// duplicated across the two harness copies). The wire shape here MUST stay
// byte-parallel with the worker-support definitions so the trusted parent can
// DECODE a `cohort_reference_replay` response — the `trustedDecoderKeySetEquals
// TheWorkerKeySet` cross-decode parity test guards that.
//
// This is DECODE-side only for PR-1: the trusted parent parses the report a
// pinned reference worker returns. It renders no verdict; the admission ladder
// that consumes these fields arrives in a later PR.

/// One reference position's readout (trusted-side mirror). Every field is a
/// property of the pinned reference's forward at that position given the
/// candidate's committed prefix.
struct CohortReferenceReplayPosition: Codable, Equatable {
    let committedToken: Int
    let sequentialArgmax: Int
    let rankedTokens: [Int]
    let rankedLogits: [Double]
    let rankedRelativeGaps: [Double]
    let committedTokenLogit: Double
    let committedRelativeGap: Double
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

/// One stream's replay (trusted-side mirror).
struct CohortReferenceReplayStreamReport: Codable, Equatable {
    let slot: Int
    let positions: [CohortReferenceReplayPosition]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case slot
        case positions
    }
}

/// The per-cohort measurement report (trusted-side mirror).
struct CohortReferenceReplayReport: Codable, Equatable {
    let logitProvenance: String
    let logitTopK: Int
    let relEnvelope: Double
    /// The width the reference ACTUALLY replayed at — `"cohort"` (batched
    /// `[B, 1]`) or `"canonical"` (per-stream `[1, 1]`). The producing side
    /// stamps this from the RESOLVED width inside the branch that ran, so the
    /// enforced reference geometry is recorded in the report instead of riding
    /// an unrecorded engine-side default. A `String` here because this target
    /// links no MLX and so cannot see the producing side's
    /// `CohortReferenceReplayWidth` enum — the same reason `logitProvenance` is
    /// a `String`.
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
