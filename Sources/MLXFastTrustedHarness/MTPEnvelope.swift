#if !MLXFAST_TRUSTED_HARNESS
import Foundation
import MLXLMCommon

// The MTP arm's two envelope knobs, structured as pinned identities with an
// explicit DECLARATION, an explicit SEAL, and a fail-loud REFUSAL — never a
// silent default. This is the trust boundary a participant-editable-surface
// security review examines: both knobs are DECLARED here as named constants,
// SEALED into the vendored `CBv2MTPConfig` at the one call site that
// constructs it (`Gemma4MTPEnvelope.resolveConfig`), and REFUSED (thrown,
// naming the missing pin) if either is unset or non-positive, at `mtp` spec
// RESOLUTION — before any round is planned, never lazily at first use.
//
// Background on WHY this posture is required, not merely tidy:
//
//   * `CBv2MTPConfig.maxAutomaticRectangularTokens` (vendored,
//     MTPContractsV2.swift) defaults to 0. Per that file's own doc comment, 0
//     means `.automatic` verification mode can NEVER select rectangular
//     scoring — every round silently falls back to the `.serialTarget` oracle
//     with no diagnostic. That is exactly the "silent-0-no-spec-work" failure
//     this file exists to prevent: an engine that resolves `mtp` mode, echoes
//     a nonzero depth, and then quietly performs no rectangular verification
//     at all.
//
//     CORRECTION (2026-08-25, exactness defect): the same vendored doc
//     comment also states what a POSITIVE cap means — "a positive envelope
//     is the integrator's explicit claim that rectangular target evaluation
//     is argmax-exact for the deployed chip/OS/MLX/model tuple at every
//     shape inside it". No such certification exists for this track's tuple,
//     and this repo's own port notes record the opposite finding
//     (docs/gemma4-port-notes.md section 3.1: on M5 with the production QAT
//     checkpoint, `[B,1]` and `[B,L]` quantized-matmul shapes "are not
//     bit-identical", first divergence in the layer-0 Q/K projection). The
//     first live single-stream MTP run (2026-08-25, engine 8c0bfec2)
//     confirmed it: with `.automatic` + cap 32 sealed here, 7 of 8 hidden
//     prompts diverged from the serial oracle at deterministic
//     prompt-specific steps, committing text-equivalent boundary-shifted
//     re-tokenizations — the near-tie argmax signature of the rectangular
//     `[1, 1+k]` verify forward, not a round-accounting error (every
//     acceptance/rollback/stop boundary was separately proven token-exact;
//     see Tests/MLXFastTests/MTPVerificationStrategySealTests.swift).
//     `resolveConfig` therefore seals `.serialTarget` — the vendored
//     chip-independent oracle whose verify columns are bit-identical to
//     ordinary serial decode BY CONSTRUCTION — until a rectangular exactness
//     certification for the deployed tuple exists. The cap below stays
//     pinned as the DECLARED CANDIDATE envelope such a certification would
//     re-enable; while `.serialTarget` is sealed it is inert (vendored:
//     "Ignored by explicit serial/rectangular modes").
//   * The attention query-block width (`CBv2AttentionV1.queryBlockSize`,
//     AttentionV1.swift) is general v1-attention-dispatch infrastructure,
//     env-var-controlled (`DARKBLOOM_CBV2_ATTN_QUERY_BLOCK`, default 128,
//     shared by every model/mode this engine runs — not MTP-specific). This
//     file does NOT change that file's own default or its env-var control;
//     that is shared, general-purpose vendored infrastructure outside this
//     repo's editable/adoption scope.
//
// IMPORTANT CORRECTION, recorded so it is not re-made: an earlier draft of
// this file justified the query-block pin as gating the MTP RECTANGULAR
// VERIFY path's dispatch shape. That claim is FALSE and is corrected here
// with the evidence. `AttentionV1.swift`'s `attendSerialQueries` — the
// function the per-column-serialized MTP verify path calls
// (`mtpSerializesRectangularAttention == true` routes every attention call
// through it) — calls `attendQueryBlocks(..., blockSize: 1)` with a
// HARDCODED literal `1`, never reading `CBv2AttentionV1.queryBlockSize` at
// all (`AttentionV1.swift:588-665`; the function's own doc comment states
// "`blockSize == 1` reproduces the MTP serial-query path exactly"). So
// `attentionQueryBlockPin`'s value has **zero effect** on MTP verification
// correctness or dispatch shape, at any setting.
//
// The pin is retained anyway, with its TRUE justification: both parity legs
// of this track's within-backend token-exactness claim (port-notes §5) run
// the SAME 1024-token seed PREFILL — which DOES dispatch through
// `queryBlockSize`-gated blocking (`shouldBlockQueries(_:)`,
// `L > queryBlockSize`) — ahead of every decode/free-run window, `mtp` mode
// included. An unpinned, env-resolved query-block width is exactly the kind
// of silently-resolved-rather-than-explicit identity port-notes §5.1-§5.2
// warns against for the KV backend pin: a requested-vs-resolved gap that
// changes the answer without being visible in the result. Pinning it
// explicitly at `mtp` spec resolution extends that same refuse-not-degrade
// discipline to the one other env-driven numerics knob a speculative run's
// seed prefill depends on, even though — unlike `maxAutomaticRectangularTokens`
// — it has no leverage over the rectangular verify columns themselves
// (`PagedSeamContract.CBv2MTPRectangularSerializing`,
// `EngineLoopV2+MTPTargetVerification.swift` are the verify-path files this
// pin does NOT reach).
enum Gemma4MTPEnvelope {

    // MARK: - Declaration

    /// The rectangular-verification width cap: `batch * (1 + k)` target rows
    /// eligible for the vendored `.automatic` verification mode's rectangular
    /// (per-column-serialized, exact-by-construction) scoring path. Pinned to
    /// admit this track's full widened envelope: B=8 seeds x (1+k) for k in
    /// 1...3, i.e. rectangular widths 16, 24, 32 — so the cap is 32, the
    /// largest of the three admitted widths.
    ///
    /// NOT a per-chip draft DEPTH ceiling (a different knob, on a different
    /// axis; this engine's own depth ceiling is `maxDraftTokens` below).
    /// `docs/gemma4-port-notes.md` OQ-5 names a vendored
    /// `MTPAutomaticVerificationPolicy` returning "8 on M3/M4/M5, 4 on
    /// M1/M2" for that per-chip ceiling — that symbol does NOT exist in the
    /// adopted vendored tree (doc-only lineage from an earlier design pass;
    /// confirmed by whole-repo search), so it is cited here only as the
    /// provenance of the number, not as live code this file calls. Conflating
    /// the port-notes depth ceiling with this file's token cap was flagged
    /// explicitly as a hazard during this increment's design and is called
    /// out here so it is not re-made.
    ///
    /// Independently verified against the vendored planner rather than
    /// asserted: `CBv2MTPRoundDriver.maximumAutomaticDepth`
    /// (`CBv2MTPRoundDriver.swift:253-259`) computes
    /// `maxWidth = maxAutomaticRectangularTokens / plannedDecodeRows`, then
    /// clamps depth to `max(0, maxWidth - 1)`. At the planned decode-row
    /// bucket `plannedDecodeRows == 8` (this track's B=8), `32 / 8 == 4`, so
    /// `k` is pre-clamped to exactly 3 — this cap and `maxDraftTokens` below
    /// are the SAME envelope stated on two axes, not independently chosen
    /// numbers that happen to agree.
    static let maxAutomaticRectangularTokens = 32

    /// The MTP arm's own per-round draft depth ceiling (k), independent of the
    /// vendored per-chip tested maximum (7): this engine pins a fixed,
    /// task-scoped ceiling of 3 because the basic wide-verification kernel
    /// this increment ships is exercised only up to rectangular width 32
    /// (`8 * (1 + 3)`). Raising this past 3 without also raising
    /// `maxAutomaticRectangularTokens` would silently starve the widened
    /// depths back down to `.serialTarget` fallback — the relationship is
    /// checked directly in `RuntimeWorkerSpecConfigTests`.
    static let maxDraftTokens = 3

    /// The attention query-block width (`CBv2AttentionV1.queryBlockSize`) a
    /// speculative-mode session's shared 1024-token seed prefill is pinned
    /// against. Does NOT gate the MTP rectangular verify columns themselves
    /// — see the file header correction above; `attendSerialQueries` hardcodes
    /// `blockSize: 1` regardless of this value. `nil` means "not pinned" and
    /// is the refusal trigger below — deliberately not `0`, because `0` is a
    /// legitimate "blocking disabled" value on the vendored knob and using it
    /// as a sentinel here would make an intentional pin indistinguishable
    /// from an absent one.
    static let attentionQueryBlockPin: Int? = 128

    // MARK: - Refusal

    enum EnvelopeError: Error, CustomStringConvertible, Equatable {
        case rectangularCapUnset
        case queryBlockPinUnset

        var description: String {
            switch self {
            case .rectangularCapUnset:
                return
                    "mtp spec mode refused: Gemma4MTPEnvelope.maxAutomaticRectangularTokens "
                    + "is unset (<=0) — an explicit positive pin is required before any "
                    + "spec-mode work runs; unset must never silently mean \"no rectangular "
                    + "verification, fall back to the serial oracle\""
            case .queryBlockPinUnset:
                return
                    "mtp spec mode refused: Gemma4MTPEnvelope.attentionQueryBlockPin is unset "
                    + "— the MTP arm requires an explicit, reviewed attention query-block "
                    + "width pin before it will resolve; it must never run under whatever "
                    + "CBv2AttentionV1.queryBlockSize / DARKBLOOM_CBV2_ATTN_QUERY_BLOCK "
                    + "happens to resolve to"
            }
        }
    }

    /// Validate both pins are explicitly set. Called from `mtp` spec
    /// resolution (`RuntimeWorkerSpecRegistry.effectiveFor`) — BEFORE any
    /// round is planned — so a missing pin refuses the request that asked for
    /// `mtp`, by name, rather than degrading silently downstream.
    ///
    /// Takes the two pins as PARAMETERS, defaulted to the real constants
    /// above, rather than reading them directly — `static let` constants
    /// cannot be overridden from a test, so this is the seam
    /// `RuntimeWorkerSpecConfigTests` uses to exercise both refusal branches
    /// (an unset rectangular cap, an unset query-block pin) without touching
    /// production values. Every real call site uses the defaults; only tests
    /// pass explicit overrides.
    static func requirePinned(
        rectangularCap: Int = maxAutomaticRectangularTokens,
        queryBlockPin: Int? = attentionQueryBlockPin
    ) throws {
        guard rectangularCap > 0 else {
            throw EnvelopeError.rectangularCapUnset
        }
        guard let pin = queryBlockPin, pin > 0 else {
            throw EnvelopeError.queryBlockPinUnset
        }
    }

    // MARK: - Depth resolution (echo)

    /// Clamp a requested depth into `0...maxDraftTokens`. `nil` (no depth in
    /// the request's `mtp` block) resolves to the full pinned ceiling — `mtp`
    /// mode with no explicit depth means "use everything this arm supports",
    /// matching the vendored `CBv2MTPConfig.maxDraftTokens` default
    /// convention (defaults to the tested maximum, not to zero).
    static func resolveDepth(_ requested: Int?) -> Int {
        guard let requested else { return maxDraftTokens }
        return Swift.max(0, Swift.min(requested, maxDraftTokens))
    }

    // MARK: - Seal

    /// The vendored config this arm's round execution must construct with —
    /// the ONE place both pins are SEALED into the type the engine actually
    /// consumes (`EngineV2.init(mtpDrafter:mtpConfig:)`). Throws
    /// `requirePinned()`'s refusal rather than ever returning a config with an
    /// unset knob. Both free-run legs construct from exactly this seam: the
    /// single-stream v1.1 `free_decode_begin` mtp arm (Gemma4RuntimeWorker.swift)
    /// and the v1.2 batched cohort begin (Gemma4RuntimeCohortDriver.swift).
    ///
    /// VERIFICATION STRATEGY — `.serialTarget`, sealed 2026-08-25 (see the
    /// file-header CORRECTION for the full evidence trail). The serial oracle
    /// scores every verify column through the same `[B, 1]` eager forward
    /// ordinary decode uses, so a committed token is bit-identical to what
    /// the serial control would have committed BY CONSTRUCTION, on every
    /// chip. `.automatic` with this file's positive rectangular cap was an
    /// uncertified argmax-exactness claim for the deployed tuple; the first
    /// live single-stream run refuted it (7 of 8 hidden prompts diverged
    /// inside drafting rounds' rectangular `[1, 1+k]` verify forwards).
    /// Re-enabling rectangular scoring requires an on-box exactness
    /// certification at every admitted width FIRST — flip the mode back only
    /// together with that artifact, never on its own.
    static func resolveConfig(depth: Int) throws -> CBv2MTPConfig {
        try requirePinned()
        return CBv2MTPConfig(
            enabled: true,
            maxDraftTokens: depth,
            maxSpeculativeBatch: 8,
            fixedDraftTokens: depth,
            verificationMode: .serialTarget,
            maxAutomaticRectangularTokens: maxAutomaticRectangularTokens
        )
    }
}
#endif
