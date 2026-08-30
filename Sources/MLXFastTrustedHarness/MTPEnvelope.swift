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
//     A positive rectangular envelope is an exactness claim, not merely a
//     throughput knob. The production Gemma artifact now installs immutable
//     physical-B1 verifier contexts for C2, C3, and C4 at construction; each
//     fixed-width projection is exact against independent B1/L1 projection
//     calls. This seal therefore selects explicit `.rectangular` verification
//     only with `maxSpeculativeBatch == 1` and fixed depths 1...3. Unsupported
//     shapes fail at the installed model route instead of recovering to stock
//     execution inside the measured path.
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

    /// Existing positive rectangular-work pin. Explicit `.rectangular` mode
    /// does not use this value to choose a strategy, but keeping it nonzero
    /// preserves the construction-time refusal and records the reviewed
    /// envelope in engine metrics. The physical route itself is narrower and
    /// authoritative: B1 with columns 2...4 only.
    static let maxAutomaticRectangularTokens = 32

    /// The MTP arm's own per-round draft depth ceiling (k), independent of the
    /// vendored per-chip tested maximum (7): this engine pins a fixed,
    /// task-scoped ceiling of 3 because the installed physical-B1 verifier
    /// table contains exactly C2, C3, and C4 entrypoints. Raising this value
    /// requires construction and exactness certification for another width.
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
    /// VERIFICATION STRATEGY — explicit `.rectangular` over the immutable B1
    /// exact contexts installed by the production artifact. Depth is fixed per
    /// request so C2/C3/C4 select the corresponding prebound entrypoint; depth
    /// zero disables MTP rather than presenting target-only work as speculative.
    static func resolveConfig(depth: Int) throws -> CBv2MTPConfig {
        try requirePinned()
        return CBv2MTPConfig(
            enabled: depth > 0,
            maxDraftTokens: maxDraftTokens,
            maxSpeculativeBatch: 1,
            fixedDraftTokens: depth,
            verificationMode: .rectangular,
            maxAutomaticRectangularTokens: maxAutomaticRectangularTokens
        )
    }
}
#endif
