import Foundation
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Sidecar transport for the free-run session diagnostics (round three).
//
// The round-two stderr echo never reached the sealed evidence: benchd drops
// worker stderr on success paths (only a redacted TAIL rides failure
// diagnostics). The sidecar channel is `DARKBLOOM_FREE_RUN_DIAGNOSTICS_DIR`
// — chosen because the `DARKBLOOM_` prefix is on benchd's strict child-env
// allowlist (bench-runner transport.rs `ENGINE_ENV_ALLOWED_PREFIXES`, the
// byte-for-byte port of this repo's own `sanitizedRuntimeWorkerEnvironment`),
// so it reaches the worker on box paths with NO benchd change and no phase
// oracle (the allowlist builds the child env identically for the gates pass
// and the timed pass). These tests pin the channel contract GPU-free.

@Suite("FreeRunDiagnosticsSidecar")
struct FreeRunDiagnosticsSidecarTests {

    private func sampleDiagnostics() -> RuntimeWorkerFreeRunSessionDiagnostics {
        RuntimeWorkerFreeRunSessionDiagnostics(
            route: .mtp,
            executor: RuntimeWorkerFreeRunExecutor.cbv2WidthOneEngine,
            seedTokenCount: 14,
            committedTotal: 96,
            rounds: 90,
            draftedTotal: 12,
            acceptedTotal: 8,
            configSource: "Gemma4MTPEnvelope.resolveConfig(depth:)",
            verificationMode: "serial_target",
            rectangularCap: 32,
            requestedDepth: 2,
            serialVerifyRounds: 6,
            rectangularVerifyRounds: 0,
            seedSteps: 6,
            roundAudits: [
                CBv2MTPRoundAuditRecord(
                    requestID: 0, k: 2, draftTokens: [7, 9], targetTokens: [7, 9, 3],
                    accepted: 2, confirmed: 3, rejected: 0,
                    tokensCountAfter: 40, numComputedAfter: 39, generatedAfter: 25,
                    finishReason: nil)
            ])
    }

    @Test
    func sidecarJSONLineCarriesTheSessionAndPerRoundFields() throws {
        let line = try #require(sampleDiagnostics().sidecarJSONLine())
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        #expect(object["kind"] as? String == "free_run_session")
        #expect(object["route"] as? String == "mtp")
        #expect(object["executor"] as? String == "cbv2-width1-engine")
        #expect(object["verification_mode"] as? String == "serial_target")
        #expect(object["committed"] as? Int == 96)
        let verifyRounds = try #require(object["verify_rounds"] as? [[String: Any]])
        #expect(verifyRounds.count == 1)
        let round = verifyRounds[0]
        #expect(round["drafts"] as? [Int] == [7, 9])
        #expect(round["targets"] as? [Int] == [7, 9, 3])
        #expect(round["accepted"] as? Int == 2)
        #expect(round["confirmed"] as? Int == 3)
        // Benchd step alignment: tokens_after(40) - seed(14) - 2 = 24 — the
        // run-token index of the round's LAST committed token.
        #expect(round["last_step"] as? Int == 24)
        #expect(round["boundary_ok"] as? Bool == true)
    }

    @Test
    func sidecarWritesOneAppendedLinePerLegWhenConfigured() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("sidecar-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let environment = [
            RuntimeWorkerFreeRunSessionDiagnostics.sidecarDirectoryEnvironmentName:
                directory.path
        ]
        let diagnostics = sampleDiagnostics()
        diagnostics.writeSidecar(environment: environment)
        diagnostics.writeSidecar(environment: environment)

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(files.count == 1)
        let content = try String(
            contentsOf: directory.appendingPathComponent(try #require(files.first)),
            encoding: .utf8)
        let lines = content.split(separator: "\n")
        #expect(lines.count == 2)
        for line in lines {
            #expect(
                (try? JSONSerialization.jsonObject(with: Data(line.utf8))) != nil,
                "sidecar lines must each be standalone JSON")
        }
    }

    @Test
    func sidecarIsANoOpWithoutTheEnvironmentVariable() {
        // Must not throw, must not write anywhere observable — exercised by
        // passing an empty environment.
        sampleDiagnostics().writeSidecar(environment: [:])
        sampleDiagnostics().writeSidecar(environment: [
            RuntimeWorkerFreeRunSessionDiagnostics.sidecarDirectoryEnvironmentName: ""
        ])
    }

    /// The channel rides the engine's OWN env allowlist: the variable name
    /// must keep the `DARKBLOOM_` prefix benchd forwards (bench-runner
    /// transport.rs ENGINE_ENV_ALLOWED_PREFIXES; the Swift original is this
    /// repo's sanitizedRuntimeWorkerEnvironment). Renaming it off the
    /// allowlist would silently kill the transport on box paths.
    @Test
    func sidecarEnvironmentNameStaysOnTheForwardedPrefix() {
        #expect(
            RuntimeWorkerFreeRunSessionDiagnostics.sidecarDirectoryEnvironmentName
                .hasPrefix("DARKBLOOM_"))
    }
}
