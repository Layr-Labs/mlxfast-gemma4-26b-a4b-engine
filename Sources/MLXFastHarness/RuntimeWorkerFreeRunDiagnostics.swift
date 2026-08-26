import Foundation
import MLXLMCommon

// Free-run session OBSERVABILITY (2026-08-25, exactness round two).
//
// The first live single-stream MTP window failed token-exactness on 7 of 8
// hidden prompts, and the #28 verify-strategy seal produced BYTE-IDENTICAL
// divergences at IDENTICAL steps — proof that the causal variable was being
// assumed, not observed. This module makes the v1.1 free-run legs report,
// per session, on stderr (never the wire):
//
//   * which EXECUTOR actually computed the leg's committed stream (the
//     legacy single-token loop vs the width-1 CBv2 engine) — the
//     implementation-identity variable the tape/serial-leg/mtp-leg pairing
//     silently mixes;
//   * where the MTP config came from and what it sealed (verification mode,
//     rectangular cap, depth), plus the engine's own per-strategy round
//     counts — so "the seal governs the wire path" is observed, not assumed;
//   * one line per VERIFY round (`CBv2MTPRoundAuditRecord`, populated at the
//     vendored finalize boundary): draft ids vs target argmaxes, the accept
//     boundary, rollback count, and post-round scheduler accounting — so an
//     on-box divergence at step S can be attributed:
//       - S inside a verify round's committed span, committed token from a
//         mismatching target column ⇒ per-token REFERENCE wrong (verify
//         forward not the serial model's output);
//       - accounting invariant broken (`num_computed_after !=
//         tokens_after - 1`, or committed span misaligned with the running
//         position) ⇒ ROLLBACK BOUNDARY wrong (accept/discard off-by-one);
//       - S outside every verify span ⇒ the PLAIN (non-drafting) decode
//         reference itself drifted from the oracle tape's implementation.
//
// stderr only: benchd's worker protocol is stdout JSON; stderr is captured
// into the session log (the same channel the cool-gate progress lines use),
// so these lines ride the sealed evidence without touching the wire shape.

/// The implementation that computes a free-run leg's committed stream.
/// These are DECLARED identities consumed by the real wire arms (the
/// begin/run executors stamp them into the session diagnostics); tests pin
/// the wire arms' stamps against them.
enum RuntimeWorkerFreeRunExecutor {
    /// `plainSeedForward`/`plainDecodeStep`: one whole-prompt forward plus
    /// one `[1,1]` forward per step through `model.newCache` — the same
    /// implementation the teacher-forced correctness verbs (and therefore
    /// the pinned reference tapes) run.
    static let legacySingleTokenLoop = "legacy-single-token-loop"
    /// The width-1 CBv2 engine (`makeCohortEngine(batchSize: 1)` +
    /// `RuntimeWorkerMTPSession`): CBv2 chunked prefill plus the engine's
    /// own decode/round loop.
    static let cbv2WidthOneEngine = "cbv2-width1-engine"
}

/// Everything one v1.1 free-run leg reports about itself at
/// `free_decode_run` completion.
struct RuntimeWorkerFreeRunSessionDiagnostics {
    let route: RuntimeWorkerDecodeRoute
    /// Which implementation computed the committed stream
    /// (`RuntimeWorkerFreeRunExecutor`).
    let executor: String
    /// The seed prompt length (token count) — lets a reader convert the
    /// audit records' absolute `tokensCountAfter` into benchd step indices.
    let seedTokenCount: Int
    /// Committed tokens this run (N).
    let committedTotal: Int
    /// Round count (acceptance_lengths.count).
    let rounds: Int
    let draftedTotal: Int
    let acceptedTotal: Int
    /// MTP-leg only: the config seam and what it sealed.
    let configSource: String?
    let verificationMode: String?
    let rectangularCap: Int?
    let requestedDepth: Int?
    /// MTP-leg only: engine strategy/step counters from
    /// `mtpMetricsSnapshot()` (launched counts — can exceed finalized
    /// `rounds` by the cancel-cut in-flight tail).
    let serialVerifyRounds: Int?
    let rectangularVerifyRounds: Int?
    let seedSteps: Int?
    /// MTP-leg only: per-VERIFY-round audit records, finalize order.
    let roundAudits: [CBv2MTPRoundAuditRecord]

    /// Human/grep-stable stderr rendering. First line is the session
    /// summary; one `free-run-verify-round` line per audit record.
    func stderrLines() -> [String] {
        var summary =
            "mlxfast-runtime-worker: free-run-session route=\(route.rawValue) "
            + "executor=\(executor) seed_tokens=\(seedTokenCount) "
            + "committed=\(committedTotal) rounds=\(rounds) "
            + "drafted=\(draftedTotal) accepted=\(acceptedTotal)"
        if let configSource {
            summary += " config_source=\(configSource)"
        }
        if let verificationMode {
            summary += " verification_mode=\(verificationMode)"
        }
        if let rectangularCap {
            summary += " rectangular_cap=\(rectangularCap)"
        }
        if let requestedDepth {
            summary += " depth=\(requestedDepth)"
        }
        if let serialVerifyRounds {
            summary += " serial_verify_rounds=\(serialVerifyRounds)"
        }
        if let rectangularVerifyRounds {
            summary += " rectangular_verify_rounds=\(rectangularVerifyRounds)"
        }
        if let seedSteps {
            summary += " seed_steps=\(seedSteps)"
        }
        var lines = [summary]
        for (index, audit) in roundAudits.enumerated() {
            // Benchd step index of this round's LAST committed token:
            // rec.tokens = prompt + [seed] + run tokens, and the run tokens
            // are what benchd indexes from 0 — so the last committed run
            // index is tokens_after - seed_tokens - 2 (the seed itself is
            // verified separately as expected_decode_seed_token).
            let lastStep = audit.tokensCountAfter - seedTokenCount - 2
            let boundaryOK = audit.numComputedAfter == audit.tokensCountAfter - 1
            var line =
                "mlxfast-runtime-worker: free-run-verify-round index=\(index) "
                + "request=\(audit.requestID) k=\(audit.k) "
                + "drafts=\(audit.draftTokens) targets=\(audit.targetTokens) "
                + "accepted=\(audit.accepted) confirmed=\(audit.confirmed) "
                + "rejected=\(audit.rejected) last_step=\(lastStep) "
                + "tokens_after=\(audit.tokensCountAfter) "
                + "num_computed_after=\(audit.numComputedAfter) "
                + "generated_after=\(audit.generatedAfter) "
                + "boundary_ok=\(boundaryOK)"
            if let finishReason = audit.finishReason {
                line += " finish=\(finishReason)"
            }
            lines.append(line)
        }
        return lines
    }

    /// Emit to stderr (the channel session logs capture).
    func emitToStandardError() {
        let text = stderrLines().joined(separator: "\n") + "\n"
        FileHandle.standardError.write(Data(text.utf8))
    }

    // MARK: - Sidecar transport (round three, 2026-08-25)

    /// The round-two stderr echo NEVER REACHED the sealed evidence: benchd
    /// drops worker stderr on success paths (it retains only a redacted
    /// stderr TAIL, and only inside failure diagnostics — bench-runner
    /// `transport.rs` / `read_response_line`'s post-mortem). The sidecar is
    /// the least-invasive benchd-compatible channel that survives:
    ///
    ///   * `DARKBLOOM_FREE_RUN_DIAGNOSTICS_DIR=<dir>` names an
    ///     operator-owned directory; the worker APPENDS one JSON line per
    ///     free-run leg to `free-run-diagnostics-pid<pid>.jsonl` in it.
    ///   * The `DARKBLOOM_` prefix is on benchd's STRICT child-env
    ///     allowlist (`ENGINE_ENV_ALLOWED_PREFIXES`, bench-runner
    ///     `transport.rs:107-109`, the byte-for-byte port of this repo's own
    ///     `sanitizedRuntimeWorkerEnvironment`), so the variable reaches the
    ///     worker on ranked/box paths with NO benchd change — the same
    ///     pass-through the documented `DARKBLOOM_CBV2_MTP` kill switch
    ///     already rides.
    ///   * No phase oracle: the allowlist builds the child env identically
    ///     for the unscored gates pass and the scored timed pass, so the
    ///     variable is either present in both or neither.
    ///   * Absent variable → no file I/O at all (the stderr echo remains).
    ///     Write failures are swallowed after a single stderr note —
    ///     diagnostics must never fail a leg.
    ///   * Nothing here touches the wire: the response shape is unchanged
    ///     and the fixture stays byte-identical.
    static let sidecarDirectoryEnvironmentName = "DARKBLOOM_FREE_RUN_DIAGNOSTICS_DIR"

    /// One machine-parsable JSON line for the sidecar: the session summary
    /// plus every verify-round audit record (drafts, targets, the accept
    /// boundary, rollback count, post-round accounting, and the benchd step
    /// alignment `last_step`). Deterministic key order (sortedKeys) so the
    /// line is diffable across runs.
    func sidecarJSONLine() -> String? {
        var object: [String: Any] = [
            "kind": "free_run_session",
            "pid": Int(ProcessInfo.processInfo.processIdentifier),
            "route": route.rawValue,
            "executor": executor,
            "seed_tokens": seedTokenCount,
            "committed": committedTotal,
            "rounds": rounds,
            "drafted": draftedTotal,
            "accepted": acceptedTotal,
        ]
        if let configSource { object["config_source"] = configSource }
        if let verificationMode { object["verification_mode"] = verificationMode }
        if let rectangularCap { object["rectangular_cap"] = rectangularCap }
        if let requestedDepth { object["depth"] = requestedDepth }
        if let serialVerifyRounds { object["serial_verify_rounds"] = serialVerifyRounds }
        if let rectangularVerifyRounds {
            object["rectangular_verify_rounds"] = rectangularVerifyRounds
        }
        if let seedSteps { object["seed_steps"] = seedSteps }
        object["verify_rounds"] = roundAudits.map { audit -> [String: Any] in
            [
                "request": Int(audit.requestID),
                "k": audit.k,
                "drafts": audit.draftTokens,
                "targets": audit.targetTokens,
                "accepted": audit.accepted,
                "confirmed": audit.confirmed,
                "rejected": audit.rejected,
                "last_step": audit.tokensCountAfter - seedTokenCount - 2,
                "tokens_after": audit.tokensCountAfter,
                "num_computed_after": audit.numComputedAfter,
                "generated_after": audit.generatedAfter,
                "boundary_ok": audit.numComputedAfter == audit.tokensCountAfter - 1,
                "finish": audit.finishReason as Any,
            ]
        }
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys])
        else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    /// Append the JSON line to the sidecar file when the operator-owned
    /// directory is configured. Injectable environment for tests; the
    /// production caller passes the process environment.
    func writeSidecar(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        guard let directory = environment[Self.sidecarDirectoryEnvironmentName],
            !directory.isEmpty
        else { return }
        guard let line = sidecarJSONLine() else { return }
        let url = URL(fileURLWithPath: directory).appendingPathComponent(
            "free-run-diagnostics-pid\(ProcessInfo.processInfo.processIdentifier).jsonl")
        let payload = Data((line + "\n").utf8)
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: payload)
            } else {
                try payload.write(to: url, options: [])
            }
        } catch {
            FileHandle.standardError.write(
                Data(
                    ("mlxfast-runtime-worker: free-run diagnostics sidecar write "
                        + "failed (\(error)); continuing\n").utf8))
        }
    }
}
