import CryptoKit
import Foundation
import MLXFastCore

/// DEFENCE-IN-DEPTH ONLY — this is a BELT, not the trust boundary.
///
/// Trust model (David's standing ruling, 2026-08-21): the engine repository is
/// COMPETITOR-CONTROLLED surface — a competitor edits the worker's own code — so
/// any check the worker performs on itself can be disabled by the very party it
/// guards against. It therefore CANNOT be the authoritative binding on which
/// executable runs. All AUTHORITATIVE validation lives on the benchd / gate side,
/// where the competitor cannot reach.
///
/// THE BRACES (the authoritative binding, behind this belt): the bench-side
/// `scripts/window-preflight.sh` gate byte-seals the worker executable's sha256
/// against the env-seam pin `WP_ENGINE_BIN_SHA256` (its `check_bin enginebin`
/// does a full sha256 seal + an identity RUN of the exact `mlxfast-runtime-worker`
/// binary), and freezes `MLXFAST_RUNTIME_WORKER_EXECUTABLE` so the spawn cannot be
/// redirected. That pin (#138) is the trust anchor. `run-paired-window.sh`
/// (bench#143 wire a) now REFUSES a scoring window unless that gate ran and
/// PASSED, so the authoritative check cannot be skipped — none of which is
/// reachable from this engine-editable surface.
///
/// WHAT THIS BELT ADDS (bench#143 wire b): the trusted binary resolves the worker
/// executable from `MLXFAST_RUNTIME_WORKER_EXECUTABLE` and otherwise runs whatever
/// it names with no sha check of its own. As a redundant, in-process check the
/// harness re-verifies the resolved executable against an env-seam sha pin
/// immediately before spawning it. Do NOT read this as the binding that makes the
/// worker trustworthy — the gate's `WP_ENGINE_BIN_SHA256` seal is that. This only
/// narrows the window in honest/misconfigured setups (e.g. a gate-skipped local
/// run); an adversary who can edit the engine can also delete this file.
///
/// DECIDE (policy, flagged not settled — see the PR): the pin this verifies
/// against is a SELF-DECLARED OPERATOR value today (an env var), not an
/// organizer-signed baseline — as is the gate's `WP_ENGINE_BIN_SHA256` itself.
/// This wires the mechanism against the pin AS IT EXISTS; whether the pins must be
/// organizer-signed, and where the signing authority/key would live, is a ruling
/// for David. No signing scheme is implemented here.
public enum RuntimeWorkerExecutablePin {
    /// The env-seam key naming the expected lowercase-hex sha256 of the
    /// participant runtime-worker executable. Parallel to
    /// `MLXFAST_RUNTIME_WORKER_EXECUTABLE`, and inside the same `MLXFAST_*` env
    /// namespace the window-preflight gate seals.
    public static let environmentKey = "MLXFAST_RUNTIME_WORKER_EXECUTABLE_SHA256"

    /// #148 activation — the env-seam flag by which the GATE declares that the pin above is
    /// MANDATORY for this run. When it is `1`, an absent/empty pin is fail-closed (the worker is
    /// NOT spawned) instead of the opt-in skip; a set pin is verified exactly as before. The gate
    /// sets this flag ONLY together with exporting `environmentKey` (window-preflight.sh seals
    /// `WP_ENGINE_BIN_SHA256` and the driver exports it) — so its presence means "the gate promised
    /// an identity pin", and the pin's absence then signals a broken/stripped env seam, not a
    /// participant that simply never opted in. Absent this flag the belt keeps its opt-in behaviour,
    /// so local/participant runs are unchanged. This flag carries only a REQUIREMENT, never a value:
    /// the honest identity value lives only at the gate, so the engine never manufactures one here.
    public static let requiredKey = "MLXFAST_RUNTIME_WORKER_EXECUTABLE_SHA256_REQUIRED"

    public enum Outcome: Equatable {
        case verified(sha256: String)
        case skipped(reason: String)
        case mismatch(expected: String, actual: String)
        case unreadable(reason: String)
    }

    /// Pure evaluation: hash the file at `executablePath` and compare it to
    /// `declaredPinSHA256`. No env reads, no throwing — the caller decides how
    /// to act. A nil/empty pin is `.skipped` (nothing declared to verify
    /// against; whether an OFFICIAL run must REQUIRE a pin is a policy question,
    /// see the DECIDE note above). A malformed pin is `.unreadable` and fails
    /// closed: an unverifiable pin must never be silently treated as absent.
    public static func evaluate(
        executablePath: String,
        declaredPinSHA256: String?
    ) -> Outcome {
        let pin = (declaredPinSHA256 ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pin.isEmpty else {
            return .skipped(reason: "no \(environmentKey) pin declared")
        }
        guard let normalizedPin = normalizedHexSHA256(pin) else {
            return .unreadable(
                reason: "\(environmentKey) is not a 64-character hex sha256 "
                    + "(a pin that cannot be parsed is refused, never ignored)"
            )
        }
        guard let actual = sha256HexOfFile(URL(fileURLWithPath: executablePath)) else {
            return .unreadable(
                reason: "could not read the runtime-worker executable at "
                    + executablePath + " to verify its sha256"
            )
        }
        if actual == normalizedPin {
            return .verified(sha256: actual)
        }
        return .mismatch(expected: normalizedPin, actual: actual)
    }

    /// Enforcement used on the live spawn path: reads the env pin and THROWS on
    /// a mismatch or an unreadable/malformed pin. Returns normally when the sha
    /// matches. When NO pin is declared the behaviour depends on `requiredKey`
    /// (#148): if the gate declared the pin mandatory (`requiredKey == "1"`) an
    /// absent pin is fail-closed; otherwise the pin stays opt-in (unchanged —
    /// local/participant runs are unaffected). This is the worker's OWN check:
    /// it binds which binary runs even when the bench-side window-preflight gate
    /// was skipped.
    public static func enforceBeforeSpawn(
        executablePath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let pinRequired = (environment[requiredKey] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines) == "1"
        switch evaluate(
            executablePath: executablePath,
            declaredPinSHA256: environment[environmentKey]
        ) {
        case .verified:
            return
        case .skipped(let reason):
            // #148 activation: when the gate has declared the pin MANDATORY (requiredKey == "1")
            // there is nothing to verify against yet a run that was promised one — fail closed
            // rather than spawn unverified. Without that declaration the pin stays opt-in, so
            // local/participant runs keep their existing behaviour.
            guard pinRequired else { return }
            throw MLXFastError.invalidInput(
                "refusing to spawn the participant runtime worker: this run declares the "
                    + environmentKey + " pin MANDATORY (" + requiredKey + "=1) but "
                    + reason + ". #148: the window-preflight gate seals the worker binary's "
                    + "sha256 and the driver must export it; its absence under a run that "
                    + "requires it is a broken env seam, refused fail-closed."
            )
        case .mismatch(let expected, let actual):
            throw MLXFastError.invalidInput(
                "refusing to spawn the participant runtime worker: its sha256 "
                    + actual + " does not match the pinned "
                    + environmentKey + " " + expected
                    + ". The env-seam pin binds which binary runs; a mismatch "
                    + "means the resolved executable is not the pinned one "
                    + "(defence-in-depth: this is the worker's own check, "
                    + "independent of the window-preflight gate)."
            )
        case .unreadable(let reason):
            throw MLXFastError.invalidInput(
                "refusing to spawn the participant runtime worker: " + reason
            )
        }
    }

    /// Accept exactly 64 ASCII hex characters; reject fullwidth hex and every
    /// other non-ASCII scalar so a pin stays verifiable and can never be
    /// silently coarsened. Returns the lowercased canonical form.
    private static func normalizedHexSHA256(_ value: String) -> String? {
        let scalars = Array(value.unicodeScalars)
        guard scalars.count == 64 else { return nil }
        for scalar in scalars {
            let v = scalar.value
            let isDigit = v >= 0x30 && v <= 0x39      // 0-9
            let isLower = v >= 0x61 && v <= 0x66      // a-f
            let isUpper = v >= 0x41 && v <= 0x46      // A-F
            guard isDigit || isLower || isUpper else { return nil }
        }
        return value.lowercased()
    }

    /// Streamed file digest — a worker binary is tens of megabytes and must not
    /// be read into memory whole. Lowercase hex.
    private static func sha256HexOfFile(_ url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            guard let chunk = (try? handle.read(upToCount: 4 * 1024 * 1024)) ?? nil,
                  !chunk.isEmpty
            else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
