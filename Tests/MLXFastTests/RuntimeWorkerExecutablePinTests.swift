import CryptoKit
import Foundation
import MLXFastCore
@testable import MLXFastHarness
import Testing

// bench#143 wire b — the worker's OWN sha self-verification of its executable
// against the env-seam pin. This is DEFENCE-IN-DEPTH (a belt), NOT the trust
// boundary: the authoritative binding is the bench gate's WP_ENGINE_BIN_SHA256
// seal (window-preflight.sh), which run-paired-window.sh now forces to run. This
// engine-side check is competitor-editable; these tests prove the belt bites in
// honest/misconfigured setups, not that it is the trust anchor.
//
// Revert-proof: an executable whose sha256 != the declared pin is REFUSED before
// it is spawned. The acceptance mutation is "remove the self-check": deleting the
// throw in `enforceBeforeSpawn` flips the mismatch/malformed rows below to green
// (no throw), and deleting the `enforceBeforeSpawn` CALL from `RuntimeWorkerClient`
// flips the call-site row (it would spawn the dummy and fail the hello handshake
// with a different error instead of the pin error).

private func writeTempExecutable(bytes: Data) throws -> String {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlxfast-pin-test-\(UUID().uuidString)")
    try bytes.write(to: url)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755], ofItemAtPath: url.path)
    return url.path
}

/// Independent one-shot sha256 (the production helper hashes incrementally, so
/// this is a distinct code path over the same bytes — they must agree).
private func oneShotSHA256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

@Test
func evaluateVerifiesAMatchingPin() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }
    let sha = oneShotSHA256Hex(bytes)

    let outcome = RuntimeWorkerExecutablePin.evaluate(
        executablePath: path, declaredPinSHA256: sha)
    #expect(outcome == .verified(sha256: sha))
    // Enforcement returns normally on a match.
    #expect(throws: Never.self) {
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: path,
            environment: [RuntimeWorkerExecutablePin.environmentKey: sha])
    }
}

@Test
func evaluateAcceptsAnUppercaseAndPaddedPin() throws {
    let bytes = Data("payload-\(UUID().uuidString)".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }
    let sha = oneShotSHA256Hex(bytes)

    // The pin is compared case-insensitively and trimmed — a pin copied with
    // surrounding whitespace or uppercased must still verify, not silently fail.
    let padded = "  \(sha.uppercased())\n"
    #expect(RuntimeWorkerExecutablePin.evaluate(
        executablePath: path, declaredPinSHA256: padded) == .verified(sha256: sha))
}

@Test
func evaluateRefusesAMismatchedPin() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }
    let sha = oneShotSHA256Hex(bytes)
    // Flip the first hex nibble → a well-formed but WRONG pin.
    let wrong = (sha.first == "0" ? "1" : "0") + sha.dropFirst()

    let outcome = RuntimeWorkerExecutablePin.evaluate(
        executablePath: path, declaredPinSHA256: wrong)
    #expect(outcome == .mismatch(expected: wrong, actual: sha))

    // Enforcement THROWS on a mismatch, before any spawn. This is the wire.
    var message = ""
    #expect(throws: MLXFastError.self) {
        do {
            try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
                executablePath: path,
                environment: [RuntimeWorkerExecutablePin.environmentKey: wrong])
        } catch { message = "\(error)"; throw error }
    }
    #expect(message.contains("does not match the pinned"))
    #expect(message.contains(RuntimeWorkerExecutablePin.environmentKey))
}

@Test
func noPinDeclaredIsSkippedNotEnforced() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }

    // Absent / empty / whitespace-only pin: nothing declared to verify against.
    for pin in [nil, "", "   "] as [String?] {
        let outcome = RuntimeWorkerExecutablePin.evaluate(
            executablePath: path, declaredPinSHA256: pin)
        if case .skipped = outcome {} else {
            Issue.record("expected .skipped for pin \(String(describing: pin)), got \(outcome)")
        }
    }
    // No env key → enforcement is a no-op (unchanged behaviour; opt-in pin).
    #expect(throws: Never.self) {
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: path, environment: [:])
    }
}

@Test
func malformedPinFailsClosed() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }

    // A pin that is not 64 hex chars, or carries a non-ASCII (fullwidth) hex
    // digit, is unverifiable — refused, never silently treated as "no pin".
    let short = String(repeating: "a", count: 63)
    let fullwidth = String(repeating: "a", count: 63) + "\u{FF10}"  // U+FF10 fullwidth 0
    for bad in [short, fullwidth] {
        if case .unreadable = RuntimeWorkerExecutablePin.evaluate(
            executablePath: path, declaredPinSHA256: bad) {} else {
            Issue.record("expected .unreadable for malformed pin '\(bad)'")
        }
        #expect(throws: MLXFastError.self) {
            try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
                executablePath: path,
                environment: [RuntimeWorkerExecutablePin.environmentKey: bad])
        }
    }
}

@Test
func unreadableExecutableFailsClosedWhenPinned() throws {
    // A pin is declared but the executable cannot be read → refuse, do not spawn.
    let missing = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlxfast-pin-absent-\(UUID().uuidString)").path
    let anySHA = String(repeating: "0", count: 64)
    if case .unreadable = RuntimeWorkerExecutablePin.evaluate(
        executablePath: missing, declaredPinSHA256: anySHA) {} else {
        Issue.record("expected .unreadable for an absent executable with a pin")
    }
    #expect(throws: MLXFastError.self) {
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: missing,
            environment: [RuntimeWorkerExecutablePin.environmentKey: anySHA])
    }
}

// #148 ACTIVATION — when the gate declares the pin MANDATORY (requiredKey == "1"), an ABSENT pin
// is fail-closed instead of the opt-in skip. Revert-proof: deleting the `guard pinRequired`/throw
// in `enforceBeforeSpawn` makes `gateRequiredWithoutPinFailsClosed` stop throwing → that row flips
// red. The required flag carries only a REQUIREMENT, never a value: the honest identity sha lives
// only at the gate (WP_ENGINE_BIN_SHA256, exported by the driver), never manufactured here.

@Test
func gateRequiredWithoutPinFailsClosed() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }

    // requiredKey=1 but NO pin declared → the gate promised an identity pin that never arrived;
    // refuse fail-closed rather than spawn unverified.
    var message = ""
    #expect(throws: MLXFastError.self) {
        do {
            try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
                executablePath: path,
                environment: [RuntimeWorkerExecutablePin.requiredKey: "1"])
        } catch { message = "\(error)"; throw error }
    }
    #expect(message.contains(RuntimeWorkerExecutablePin.requiredKey))
    #expect(message.contains("MANDATORY"))
}

@Test
func gateRequiredWithMatchingPinProceeds() throws {
    let bytes = Data("payload-\(UUID().uuidString)".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }
    let sha = oneShotSHA256Hex(bytes)

    // requiredKey=1 AND a matching pin → verified, spawn proceeds (the activation does not break a
    // correctly-pinned run).
    #expect(throws: Never.self) {
        try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
            executablePath: path,
            environment: [
                RuntimeWorkerExecutablePin.requiredKey: "1",
                RuntimeWorkerExecutablePin.environmentKey: sha,
            ])
    }
}

@Test
func gateRequiredStillRefusesAMismatch() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }
    let sha = oneShotSHA256Hex(bytes)
    let wrong = (sha.first == "0" ? "1" : "0") + sha.dropFirst()

    // required=1 with a WRONG pin still throws the MISMATCH error (not the missing-pin one): the
    // requirement flag adds the absent-pin case, it does not soften the existing mismatch throw.
    var message = ""
    #expect(throws: MLXFastError.self) {
        do {
            try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
                executablePath: path,
                environment: [
                    RuntimeWorkerExecutablePin.requiredKey: "1",
                    RuntimeWorkerExecutablePin.environmentKey: wrong,
                ])
        } catch { message = "\(error)"; throw error }
    }
    #expect(message.contains("does not match the pinned"))
}

@Test
func noRequirementWithoutPinStillSkips() throws {
    let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
    let path = try writeTempExecutable(bytes: bytes)
    defer { try? FileManager.default.removeItem(atPath: path) }

    // requiredKey absent or "0" and no pin → unchanged opt-in skip (local/participant runs are
    // unaffected by #148; only a gate that sets requiredKey=1 arms the fail-closed path).
    for env in [[:], [RuntimeWorkerExecutablePin.requiredKey: "0"]] as [[String: String]] {
        #expect(throws: Never.self) {
            try RuntimeWorkerExecutablePin.enforceBeforeSpawn(
                executablePath: path, environment: env)
        }
    }
}

// Call-site binding: prove the LIVE client actually invokes the check before it
// spawns. Serialized because it sets a process-global env pin for the duration
// of one construction. The mismatch throws BEFORE `process.run()`, so no worker
// is spawned; removing the call from `RuntimeWorkerClient` makes this spawn the
// dummy and fail the hello with a different error → this row flips red.
@Suite(.serialized)
struct RuntimeWorkerClientExecutablePinCallSite {
    @Test
    func clientRefusesAMismatchedExecutableBeforeSpawn() throws {
        let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-pin-client-\(UUID().uuidString)")
        try bytes.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
        defer { try? FileManager.default.removeItem(at: url) }
        let sha = oneShotSHA256Hex(bytes)
        let wrong = (sha.first == "0" ? "1" : "0") + sha.dropFirst()

        let key = RuntimeWorkerExecutablePin.environmentKey
        setenv(key, wrong, 1)
        defer { unsetenv(key) }

        let options = RuntimeWorkerOptions(executablePath: url.path)
        var message = ""
        #expect(throws: MLXFastError.self) {
            do {
                _ = try RuntimeWorkerClient(
                    options: options, weightsPath: "/nonexistent-weights")
            } catch { message = "\(error)"; throw error }
        }
        #expect(message.contains("does not match the pinned"))
        #expect(message.contains(key))
    }

    // The SECOND live spawn site: Gemma4Runtime.runPreflightWithWorker (reached by
    // the CLI `preflight` verb). It has its own enforceBeforeSpawn call; without
    // this test a revert of JUST that call passes the suite green (an uncovered
    // live spawn). Mismatch → throw BEFORE process.run at this site too.
    @Test
    func preflightWithWorkerRefusesAMismatchedExecutableBeforeSpawn() throws {
        let bytes = Data("#!/bin/bash\nexit 0\n".utf8)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-pin-preflight-\(UUID().uuidString)")
        try bytes.write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
        defer { try? FileManager.default.removeItem(at: url) }
        let sha = oneShotSHA256Hex(bytes)
        let wrong = (sha.first == "0" ? "1" : "0") + sha.dropFirst()

        let key = RuntimeWorkerExecutablePin.environmentKey
        setenv(key, wrong, 1)
        defer { unsetenv(key) }

        let options = RuntimeWorkerOptions(executablePath: url.path)
        var message = ""
        #expect(throws: MLXFastError.self) {
            do {
                try Gemma4Runtime.runPreflightWithWorker(
                    weightsPath: "/nonexistent-weights", worker: options)
            } catch { message = "\(error)"; throw error }
        }
        #expect(message.contains("does not match the pinned"))
        #expect(message.contains(key))
    }
}
