import Foundation
import MLXFastCore

// Pure, GPU-free dispatch + free-run assembly for the GENERIC benchd-facing
// runtime worker (`runtime-worker` verb, `Gemma4Runtime.runWorker`).
//
// benchd drives ONLY generic kinds — `decode_begin`(spec)/`decode_step` and
// `free_decode_begin`(spec)/`free_decode_run(N)` — and never the native
// `mtp_decode_*` kinds. This file owns the two pieces of that dispatch that are
// pure decisions over already-resolved data, so they are unit-testable without a
// model or a transport:
//
//   1. ROUTE — map a resolved `effective_spec.mode` to the decode path the worker
//      runs. `serial` (plain decode) is the only route since the MTP arm left
//      on 2026-08-22; dflash / dspark never reach here because
//      `RuntimeWorkerSpecRegistry.resolveEffectiveSpec` already throws
//      (capability-absent / not-implemented) before a route is chosen, so this
//      stays fail-closed on anything else.
//
//   2. FREE-RUN ASSEMBLY — accumulate a `free_decode_run(N)` phase round by round
//      into the AUDIT counters benchd's §2.6 consistency TRIPLE cross-checks:
//      `R == acceptance_lengths.len()`, `sum(acceptance_lengths) == N`,
//      `completed_work == R + 1` (seed forward + R verify rounds), plus
//      `committed_total == N == tokens.len()` and `drafted_total >= accepted_total`.

/// Hello capability advertising the oracle-verified free-run timed-decode mode
/// (`free_decode_begin` / `free_decode_run`). benchd REFUSES to issue the
/// free-run kinds unless the engine advertises this. Wire value matches benchd's
/// `CAPABILITY_FREE_RUN_DECODE`.
///
/// **ADJUDICATED (#10 item 3): the spawn gate governs SERVING, not only
/// ADVERTISING.** The capability is emitted only when the v1.1 surface was
/// gated on at spawn, and the engine now also REFUSES `free_decode_begin` /
/// `free_decode_run` when it is gated off — `validateGenericWorkerRequest`
/// guard 4. The two halves are one decision: a worker that serves a kind its
/// own hello never offered is asserting a mode it did not advertise, which is
/// the same false acknowledgment guard 2 refuses for a `spec`.
///
/// The alternative reading — "the gate keeps the trusted native CLI's v1
/// decoder from choking on unknown hello FIELDS, so it should not disable a
/// request KIND" — was considered and rejected. The trusted CLI never issues
/// the free-run kinds (nothing under `Sources/MLXFastTrustedHarness` mentions
/// them), so gating them costs it nothing, and fail-closed is the posture the
/// rest of this surface takes.
let runtimeWorkerFreeRunDecodeCapability = "free_run_decode"

// MARK: - Speculative-protocol spawn gate (v1.1 opt-in)

/// The worker speaks FIRST (the hello is unsolicited), so the v1.1 speculative
/// surface cannot be negotiated in a handshake — it is gated at SPAWN. This flag,
/// passed by benchd in the worker argv, opts the engine into advertising the v1.1
/// surface: `spec_modes` / `capabilities` / `head_provenance` on the hello and
/// `effective_spec` on the decode-begin echoes. Absent (the native trusted CLI
/// never passes it) → the worker emits ONLY the v1 fields the trusted decoder
/// accepts, so `mlxfast-swift`'s worker-spawning verbs handshake unchanged. The
/// three v1 identity fields (protocol_version / backend / device) are NOT gated —
/// they ride on every hello.
public let runtimeWorkerSpeculativeProtocolFlag = "--speculative-protocol"

/// The only value the spawn flag accepts today. The engine implements
/// `protocol_version` 1 with the v1.1 additive surface; the token names the
/// advertised surface, not a version bump (PROTOCOL.md keeps `protocol_version`
/// at 1). A different value is a fail-closed error, never a silent v1 fallback.
let runtimeWorkerSpeculativeProtocolVersionToken = "v1.1"

// MARK: - Assistant-head spawn flag (--mtp-head, restored 2026-08-25)

/// The spawn flag naming the directory the assistant head loads from. RESTORED
/// for the Gemma 4 MTP arm (2026-08-25): the Qwen-era flag of the same name
/// went with the harness-port deletion (port-notes §8.3), but benchd's
/// measure-job spawn contract kept it — every leg is spawned
/// `runtime-worker --weights <W> --mtp-head <H>` (benchd @ c2327d15,
/// `crates/benchctl/src/measure_job.rs` `timed_leg_base_args` /
/// `RUNTIME_WORKER_ACCEPTED_FLAGS`), the serial control with the PINNED head
/// and the candidate with the declared BYO head, so a verb refusing the flag
/// kills every measure-job leg pre-hello. Absent → the CWD `./mtp-head/`
/// staging default (the native trusted CLI's flow). Present → the head loads
/// from EXACTLY that directory, fail-closed if it is not a loadable head.
///
/// ARGV IS THE ONLY CHANNEL. benchd resolves its own `QMTP_HEAD_DIR` /
/// `QMTP_CANDIDATE_HEAD_DIR` env inputs INTO this flag's value; the engine
/// never reads those names, and could not — benchd builds the worker child env
/// from an allowlist (`sanitized_engine_env`, the Rust mirror of this repo's
/// `sanitizedRuntimeWorkerEnvironment`) that drops `QMTP_*` before the worker
/// starts.
public let runtimeWorkerMTPHeadFlag = "--mtp-head"

// MARK: - DFlash drafter spawn flag (--dflash-head, David ruling 2026-08-26)

/// The spawn flag naming the directory the DFlash drafter loads from — the
/// EXACT twin of `runtimeWorkerMTPHeadFlag`, added because DFlash became a
/// first-class SCORED mode on this track (David, 2026-08-26).
///
/// WHY IT HAD TO EXIST. Until now the DFlash drafter loaded ONLY from the CWD
/// `./dflash-head/` default, and the comment on `gemma4DFlashHeadDirectoryName`
/// said why that was tolerable: "benchd does not spawn a DFlash leg today — the
/// explicit staging channel is follow-up work". benchd now does, and the
/// tolerance is gone. benchd spawns both legs with NO `current_dir`
/// (`bench_runner` transport.rs `spawn_command` sets none), so both workers
/// inherit benchctl's own working directory — and with a CWD-only channel BOTH
/// LEGS would resolve THE SAME `./dflash-head/`, whichever workspace benchctl
/// happened to run from. The candidate's drafter resident on the SERIAL leg is
/// contamination of the scored DENOMINATOR, and it is silent: the run produces
/// a number, just not the one it claims.
///
/// Same semantics as the MTP flag, through the SAME resolver
/// (`resolveGemma4AssistantHeadStaging`): absent → the CWD `./dflash-head/`
/// default, so an MTP-only spawn behaves byte-identically to before; present →
/// the drafter loads from EXACTLY that directory, FAIL-CLOSED if it is not a
/// loadable head. Fail-closed on the explicit channel is the point — an
/// explicitly staged per-leg drafter that silently fell back to the CWD
/// default would reintroduce the very cross-leg collapse this flag closes.
///
/// ARGV IS THE ONLY CHANNEL, as for `--mtp-head`: benchd resolves its own
/// `QMTP_DFLASH_HEAD_DIR` / `QMTP_CANDIDATE_DFLASH_HEAD_DIR` env inputs INTO
/// this flag's value, and the engine never reads those names — it could not,
/// since benchd builds the worker child env from an allowlist that drops
/// `QMTP_*` before the worker starts.
public let runtimeWorkerDFlashHeadFlag = "--dflash-head"

/// The COMPLETE option surface of the generic `runtime-worker` verb — the
/// cross-repo spawn contract. benchd pins the identical set as
/// `measure_job::RUNTIME_WORKER_ACCEPTED_FLAGS` and fences every leg's argv
/// against it before spawning (`validate_spawn_argv`); the verb here refuses
/// anything outside it (`requireOnly`), exiting 1 BEFORE the hello. The two
/// constants must stay equal, and the engine-side test
/// (`Gemma4AssistantHeadStagingTests`) pins this one.
public let runtimeWorkerAcceptedOptionFlags: Set<String> = [
    "--weights",
    runtimeWorkerMTPHeadFlag,
    runtimeWorkerDFlashHeadFlag,
    runtimeWorkerSpeculativeProtocolFlag,
]

/// Resolve the spawn flag's value into "advertise the v1.1 surface?". Absent /
/// empty → false (v1 only). Present but not the supported token → throws.
public func runtimeWorkerAdvertisesSpeculativeProtocol(flagValue: String?) throws -> Bool {
    guard let flagValue, !flagValue.isEmpty else { return false }
    guard flagValue == runtimeWorkerSpeculativeProtocolVersionToken else {
        throw MLXFastError.invalidInput(
            "\(runtimeWorkerSpeculativeProtocolFlag) accepts only "
                + "'\(runtimeWorkerSpeculativeProtocolVersionToken)', got "
                + "'\(flagValue)'"
        )
    }
    return true
}

// MARK: - Route

/// The decode path the generic worker runs for a resolved spec. `serial` is
/// always runnable; `mtp` is runnable only on a worker that resolved it at
/// spec time (an assistant head loaded), and `dflash` only on a worker that
/// bound a real `DFlashDraftModel` from `dflash-head/` —
/// `RuntimeWorkerSpecRegistry.gemma4Worker(mtpAvailable:dflash:)` decides
/// both, once, at startup. `dspark` is a stub and is rejected at spec
/// resolution, never routed.
///
/// Kept as an enum (not collapsed to a Bool) because it is what
/// `state.decodeRoute` seals on a begin and what the free-run error messages
/// name.
enum RuntimeWorkerDecodeRoute: String, Equatable {
    case serial
    case mtp
    /// The z-lab DFlash drafter arm (2026-08-25). A REAL `DFlashDraftModel`
    /// (Libraries/MLXSpeculative) bound to this worker's Gemma 4 target,
    /// running `draftBlock` → target verify → accept-walk → KV-rollback
    /// rounds in `RuntimeWorkerDFlashFreeRunSession` — NOT the CBv2 MTP round
    /// driver, whose drafter seam cannot express DFlash's multi-tap
    /// conditioning, its own KV cache, or its block-shaped draft (see that
    /// file's header). It commits through the same free_decode_begin /
    /// free_decode_run wire surface and is verified by the same per-stream
    /// token-tolerance gate.
    case dflash
}

/// Decide the decode route from a resolved `effective_spec`'s mode string.
///
/// The effective spec has already passed `resolveEffectiveSpec`, which fails
/// closed for a stub (dspark) or a capability-absent real module — so only a
/// mode this worker can actually run legitimately arrives. Any other mode is
/// a programming error and throws rather than silently defaulting to serial.
///
/// `dflashAvailable` DEFAULTS TO FALSE, and that default is the fence: an
/// `effective_spec` naming `dflash` is accepted only by a caller that can
/// state this worker bound a DFlash drafter. #38 removed the fence outright
/// (dflash routed unconditionally), which made a `dflash` echo routable on a
/// worker that had never loaded a drafter — a wiring bug would then have
/// reached the begin handler instead of being refused here.
func runtimeWorkerDecodeRoute(
    forEffectiveMode mode: String,
    dflashAvailable: Bool = false
) throws -> RuntimeWorkerDecodeRoute {
    switch mode {
    case RuntimeWorkerDecodeRoute.serial.rawValue:
        return .serial
    case RuntimeWorkerDecodeRoute.mtp.rawValue:
        return .mtp
    case RuntimeWorkerDecodeRoute.dflash.rawValue where dflashAvailable:
        return .dflash
    default:
        throw MLXFastError.invalidInput(
            "generic decode has no route for effective spec mode '\(mode)'; "
                + "runnable modes on this worker are serial and "
                + "(drafter-permitting) mtp / dflash"
        )
    }
}

// MARK: - Free-run assembly

/// The assembled result of a `free_decode_run(N)` phase, carrying exactly what the
/// worker returns to benchd plus the phase-close `completed_work`.
struct RuntimeWorkerFreeRunResult: Equatable {
    /// The N committed token IDs, in commit order.
    let tokens: [Int]
    /// Per verify-round committed count; length is the round count R, sum is N.
    let acceptanceLengths: [Int]
    /// Total draft tokens proposed across all rounds (self-reported, `>= accepted`).
    let draftedTotal: Int
    /// Total drafts that passed verification and were committed (self-reported).
    let acceptedTotal: Int
    /// Total committed tokens; equals N and `tokens.count`.
    let committedTotal: Int

    /// R — the number of verify rounds (`acceptanceLengths.count`).
    var rounds: Int { acceptanceLengths.count }

    /// The phase-close `completed_work`: the seed forward plus R verify rounds.
    /// benchd's §2.6 triple asserts `completed_work == R + 1`.
    var completedWork: Int { rounds + 1 }
}

/// A way a `free_decode_run` assembly can fail its own consistency triple — the
/// same fail-closed posture benchd enforces on the wire, checked worker-side so a
/// broken round is caught where it was produced.
enum RuntimeWorkerFreeRunError: Error, CustomStringConvertible, Equatable {
    case tokenCount(expected: Int, got: Int)
    case committedTotal(n: Int, committedTotal: Int)
    case acceptanceSum(n: Int, sum: Int)
    case draftedLessThanAccepted(drafted: Int, accepted: Int)
    case emptyRound
    /// #109 W3 finding 6 — an MTP round opened on a token that is NOT the one already on the wire
    /// (`free_decode_begin`'s `seed_token` for round 1, the previous round's fallback afterwards).
    /// The session's committed stream and the stream benchd is oracle-checking have diverged;
    /// refused here rather than shipping a silently misaligned window.
    case seedSeamBroken(expected: Int, got: Int)
    /// The leg committed a stop token before reaching N. Structured and
    /// SYMMETRIC: every leg raises exactly this, at the same committed
    /// position, so a paired measurement fails on both sides or neither. It is
    /// deliberately NOT reported as a token-count mismatch — that
    /// message describes a broken assembly, and an early EOS is a valid engine
    /// outcome that simply makes the leg unusable as a timing sample.
    case stopTokenBeforeTarget(route: String, token: Int, position: Int, n: Int)

    var description: String {
        switch self {
        case let .tokenCount(expected, got):
            return "free_decode_run assembled \(got) committed tokens, expected N=\(expected)"
        case let .committedTotal(n, committedTotal):
            return "free_decode_run committed_total \(committedTotal) != N \(n)"
        case let .acceptanceSum(n, sum):
            return "free_decode_run sum(acceptance_lengths) \(sum) != N \(n)"
        case let .draftedLessThanAccepted(drafted, accepted):
            return "free_decode_run drafted_total \(drafted) < accepted_total \(accepted)"
        case .emptyRound:
            return "free_decode_run round committed zero tokens"
        case let .seedSeamBroken(expected, got):
            return
                "free_decode_run round opened on token \(got), but the token already on the "
                + "wire was \(expected); the engine's committed stream diverged from the one "
                + "benchd is verifying (PROTOCOL-v1.1 §2.2 begin/run seam)"
        case let .stopTokenBeforeTarget(route, token, position, n):
            return
                "free_decode_run \(route) leg committed stop token \(token) at "
                + "committed position \(position) of N=\(n); the leg is invalid "
                + "(a paired leg must reach N or both legs must stop)"
        }
    }
}

/// Accumulates a `free_decode_run(N)` phase one verify round at a time.
///
/// Each committed round contributes one `acceptance_lengths` entry (its committed
/// count) and appends its committed tokens. The builder CLAMPS the final round so
/// the running commit total never exceeds N — a block round that would overshoot
/// commits only its first `N - committed` tokens — which keeps
/// `sum(acceptance_lengths) == N == tokens.count` exactly, satisfying the triple.
struct RuntimeWorkerFreeRunBuilder {
    /// The requested committed-token count N (benchd's `free_decode_run.count`).
    let targetN: Int

    private(set) var tokens: [Int] = []
    private(set) var acceptanceLengths: [Int] = []
    private(set) var draftedTotal = 0
    private(set) var acceptedTotal = 0

    init(targetN: Int) {
        self.targetN = Swift.max(targetN, 0)
    }

    /// Tokens committed so far across all recorded rounds.
    var committedTotal: Int { tokens.count }

    /// Has the phase reached N committed tokens? The driver stops issuing rounds
    /// once this holds.
    var isComplete: Bool { committedTotal >= targetN }

    /// Record one verify round. `committedTokens` are the tokens the round
    /// committed (>= 1 for a well-formed round — a serial round commits exactly
    /// one; a speculative round commits `1 + acceptedDrafts`). `drafted` /
    /// `accepted` are
    /// the round's self-reported draft counters. A round arriving after the phase
    /// is already complete, or one clamped to zero remaining budget, is dropped.
    mutating func addRound(committedTokens: [Int], drafted: Int, accepted: Int) {
        guard !isComplete else { return }
        let remaining = targetN - committedTotal
        let take = Swift.min(committedTokens.count, remaining)
        guard take > 0 else { return }
        tokens.append(contentsOf: committedTokens.prefix(take))
        acceptanceLengths.append(take)
        draftedTotal += Swift.max(drafted, 0)
        acceptedTotal += Swift.max(accepted, 0)
    }

    /// Seal the phase, validating the consistency triple. Fails closed on any
    /// divergence (a round that committed nothing, an off-by-one histogram, or an
    /// impossible `drafted < accepted`), matching benchd's own `verify_consistency`.
    func finish() throws -> RuntimeWorkerFreeRunResult {
        if acceptanceLengths.contains(0) {
            throw RuntimeWorkerFreeRunError.emptyRound
        }
        let sum = acceptanceLengths.reduce(0, +)
        guard tokens.count == targetN else {
            throw RuntimeWorkerFreeRunError.tokenCount(
                expected: targetN, got: tokens.count)
        }
        guard sum == targetN else {
            throw RuntimeWorkerFreeRunError.acceptanceSum(n: targetN, sum: sum)
        }
        guard draftedTotal >= acceptedTotal else {
            throw RuntimeWorkerFreeRunError.draftedLessThanAccepted(
                drafted: draftedTotal, accepted: acceptedTotal)
        }
        return RuntimeWorkerFreeRunResult(
            tokens: tokens,
            acceptanceLengths: acceptanceLengths,
            draftedTotal: draftedTotal,
            acceptedTotal: acceptedTotal,
            committedTotal: tokens.count
        )
    }
}

/// **#109 W3 finding 6 — the `free_decode_begin` / `free_decode_run` SEAM.**
///
/// PROTOCOL-v1.1 §2.2 fixes the seam exactly: `free_decode_begin` returns `seed_token`, benchd
/// verifies it against `expected_decode_seed_token`, and then `free_decode_run(N)`'s `tokens[i]` is
/// matched against `expected_decode_tokens[i]` — the N tokens that come AFTER the seed. §2.1 says
/// `free_decode_begin` "establishes the last-committed state"; the run commits N MORE.
///
/// The MTP session's round result is in COMMIT order (`[primary] + acceptedDrafts`), which lags
/// PRODUCTION order by one: the round's opening primary was determined by the PREVIOUS target
/// forward, and the fallback this round's forward produced (`nextPrimary`) is not in its own
/// `tokens`. Emitting the commit list verbatim therefore re-sends the token `free_decode_begin`
/// already returned as `tokens[0]`, putting every following token one position late against the
/// golden — window 3's finding 6.
///
/// This converts one round to the wire's PRODUCTION order: the drafts this round's verify forward
/// confirmed, followed by the base-model fallback it produced. §3 defines the per-round count as
/// exactly that — "how many draft tokens survived internal verification (plus fallback) and were
/// committed in each MTP round" — so the count per round is UNCHANGED (`acceptedDrafts + 1`); only
/// the identities shift by one. It is also the shape the SERIAL free-run route already has: a serial
/// round forwards the last emitted token and emits what that forward produced.
///
/// FAIL-CLOSED on the seam itself: the round's opening primary MUST be the token already on the wire
/// (the seed for round 1, the previous round's fallback afterwards). A session that drifted from the
/// stream benchd verified is refused here, where it happened.
///
/// Returns an EMPTY emission for a round that determined nothing — a stop-token round, whose
/// opening primary was already emitted and which has no successor to predict.
func runtimeWorkerFreeRunRoundEmission(
    committedTokens: [Int],
    nextPrimary: Int?,
    lastEmittedToken: Int
) throws -> [Int] {
    guard let primary = committedTokens.first else {
        throw RuntimeWorkerFreeRunError.emptyRound
    }
    guard primary == lastEmittedToken else {
        throw RuntimeWorkerFreeRunError.seedSeamBroken(
            expected: lastEmittedToken, got: primary)
    }
    var emission = Array(committedTokens.dropFirst())
    if let nextPrimary {
        emission.append(nextPrimary)
    }
    return emission
}

/// The SYMMETRIC early-EOS verdict for one just-recorded free-run round.
///
/// `stopToken` is the stop token this round committed, or `nil` if it committed
/// none. Every leg feeds this the same way — the serial leg checks each
/// committed token against the session's stop-token set — so the legs of a
/// paired measurement reach the identical verdict at the identical committed
/// position.
///
/// Returns `nil` (leg is fine) when the round committed no stop token, or when
/// the phase already reached N: a stop token AT position N truncated nothing and
/// the leg delivered exactly what was asked for. Otherwise the leg is invalid as
/// a timing sample and this is the error to raise — a structured, named verdict,
/// not the token-count mismatch a short assembly would report.
func runtimeWorkerFreeRunEarlyStop(
    route: RuntimeWorkerDecodeRoute,
    stopToken: Int?,
    builder: RuntimeWorkerFreeRunBuilder
) -> RuntimeWorkerFreeRunError? {
    guard let stopToken, !builder.isComplete else { return nil }
    return .stopTokenBeforeTarget(
        route: route.rawValue,
        token: stopToken,
        position: builder.committedTotal,
        n: builder.targetN)
}

// MARK: - Stop tokens

/// The stop-token set both free-run legs halt on, read from the transformed
/// weights tree's own `config.json` / `generation_config.json`.
///
/// ONE SET FOR EVERY LEG, and that is the whole point of hoisting it out of the
/// decode loop: the legs of a paired measurement have to stop on the same
/// thing, because one side stopping early while the other runs on puts the
/// difference straight into the ratio those legs exist to produce.
///
/// `eos_token_id` and `pad_token_id` are each accepted as a scalar or an array,
/// because both spellings occur in published checkpoints. A file that is
/// missing or unparseable contributes nothing rather than throwing: the set is
/// a halting hint for the free-run window, and the pinned-config gate is what
/// refuses a malformed tree.
func resolveRuntimeWorkerStopTokens(directory: URL) -> Set<Int> {
    var ids = Set<Int>()
    for name in ["config.json", "generation_config.json"] {
        guard let data = try? Data(
            contentsOf: directory.appendingPathComponent(name)),
            let root = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
        else { continue }
        for key in ["eos_token_id", "pad_token_id"] {
            switch root[key] {
            case let value as Int:
                ids.insert(value)
            case let values as [Any]:
                ids.formUnion(values.compactMap { $0 as? Int })
            default:
                continue
            }
        }
    }
    return ids
}
