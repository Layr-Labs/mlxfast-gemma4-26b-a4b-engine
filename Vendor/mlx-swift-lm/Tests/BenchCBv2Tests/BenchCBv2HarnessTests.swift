// BenchCBv2HarnessTests.swift
//
// Integrity gates for the BenchCBv2 harness itself. These numbers feed the
// v0.8.0 release gates, so the harness must never produce a plausible-looking
// report for a different experiment than the one requested:
//
//   * a measured row's engine label always names the path that produced it —
//     a `v2-compiled` cell cannot be served by some other backend,
//   * a malformed option is a hard error, not a silent fallback,
//   * the recorded invocation reproduces the original argv exactly,
//   * the recorded revision is the one the binary was built from.
//
// Model-free by construction: nothing here loads weights or touches Metal.

import Foundation
import Testing

@testable import BenchCBv2Core

// `.serialized`: one case chdirs to prove the build stamp is not derived from
// the process's working directory.
@Suite("BenchCBv2 harness integrity", .serialized)
struct BenchCBv2HarnessTests {

    // MARK: - Engine labelling

    /// Split a markdown table row into its cells the way a GFM renderer does:
    /// on *unescaped* pipes only, dropping the empty leading and trailing
    /// fields produced by the outer pipes.
    private func cells(_ row: String) -> [String] {
        let escapedPipe = "\u{FFFF}"
        return row
            .replacingOccurrences(of: "\\|", with: escapedPipe)
            .components(separatedBy: "|")
            .dropFirst().dropLast()
            .map {
                $0.replacingOccurrences(of: escapedPipe, with: "\\|")
                    .trimmingCharacters(in: .whitespaces)
            }
    }

    private var headerColumnCount: Int {
        cells(CellResult.markdownHeader.split(separator: "\n").first.map(String.init) ?? "").count
    }

    @Test("a v2-compiled cell is refused, not silently served by another backend")
    func compiledCellIsRefusedAtEveryPromptLength() throws {
        // The exact command the review calls out: a prompt far past the 4096
        // capacity the compiled executor used to fall back to eager decode at.
        for length in [100, 4_096, 10_000, 1_000_000] {
            let options = try BenchOptions.parse([
                "--model", "/tmp/model",
                "--engines", "v2-compiled",
                "--prompt-lengths", "\(length)",
            ])
            #expect(options.engines == ["v2-compiled"])

            guard case .refuse(let reason) = resolveEngine("v2-compiled") else {
                Issue.record("v2-compiled resolved to a runnable backend at \(length) tokens")
                return
            }
            #expect(reason.contains("removed in v0.8.0"))

            // The emitted row must be unmistakably a non-measurement: same
            // column count as the table header, and every metric column `-`.
            let row = cells(refusalRow(engine: "v2-compiled", reason: reason))
            #expect(row.count == headerColumnCount)
            #expect(row[0] == "v2-compiled")
            for metric in row.dropFirst() where metric != "-" {
                #expect(
                    Double(metric) == nil,
                    "refused cell carries a number in a metric column: \(metric)")
            }
        }
    }

    @Test("a measured row is always labelled with the backend that produced it")
    func measuredRowLabelMatchesBackend() {
        for name in ["v2", "v2-paged", "v2-compiled", "legacy", "v2-Paged", "", "v3"] {
            switch resolveEngine(name) {
            case .run(let backend):
                // `runV2Cell` labels the row `backend.rawValue`; if that ever
                // diverges from the requested name the report lies.
                #expect(backend.rawValue == name)
            case .refuse(let reason):
                #expect(reason.hasPrefix("refused: "))
            }
        }
    }

    @Test("retired and unknown engines both refuse rather than fall through")
    func retiredEnginesRefuse() {
        #expect(resolveEngine("legacy") == .refuse(
            reason: "refused: the legacy engine was removed in v0.8.0 — use v2"))
        #expect(resolveEngine("nonsense") == .refuse(reason: "refused: unknown engine"))
        #expect(resolveEngine("v2") == .run(.contiguous))
        #expect(resolveEngine("v2-paged") == .run(.paged))
    }

    @Test("a refusal reason cannot forge extra table columns")
    func refusalReasonIsEscaped() {
        let row = refusalRow(
            engine: "v2-compiled", reason: "skipped: bad | 999.9 | 999.9\nsecond line")
        #expect(cells(row).count == headerColumnCount)
        #expect(!row.contains("\n"))
    }

    // MARK: - Option parsing

    @Test("malformed prompt-length lists are rejected, never silently dropped")
    func malformedPromptLengthsAreRejected() {
        // `100O0` is the review's example: a capital O for a zero. The old
        // compactMap/filter pair dropped it and ran the default 500/1500 mix
        // under a header that still claimed the requested axis.
        for raw in ["100O0", "0", "-5", "", "500,", ",500", "500,,1500", "500,abc", "1 000", "1_000"] {
            #expect(throws: BenchOptionError.self) {
                _ = try BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths", raw])
            }
        }
    }

    @Test("a rejected prompt-length list never degrades to the default mix")
    func rejectedListDoesNotBecomeTheDefaultAxis() throws {
        let defaults = try BenchOptions.parse(["--model", "/tmp/m"])
        #expect(defaults.promptLengths.isEmpty)
        #expect(defaults.promptAxisDescription.hasPrefix("default mix"))

        // The failure mode being gated: parsing must not *return* the default
        // axis for a malformed request.
        let parsed = try? BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths", "100O0"])
        #expect(parsed == nil)
    }

    @Test("a valid prompt-length list is preserved and sizes the paged pool")
    func validPromptLengthsSizeThePool() throws {
        let options = try BenchOptions.parse([
            "--model", "/tmp/m", "--prompt-lengths", "500, 10000", "--steps", "128",
        ])
        #expect(options.promptLengths == [500, 10_000])
        #expect(options.longestPrompt == 10_000)
        #expect(options.pagedNominalMaxSequenceLength == 10_128)
        #expect(options.promptAxisDescription == "500,10000")

        // No new flags → the historical 4096 sizing, unchanged.
        let legacyDefault = try BenchOptions.parse(["--model", "/tmp/m"])
        #expect(legacyDefault.pagedNominalMaxSequenceLength == 4_096)
    }

    @Test("other numeric options are equally strict")
    func otherNumericOptionsAreStrict() {
        for argument in [
            ["--batches", "1,2,x"], ["--batches", "0"], ["--steps", "12S"],
            ["--steps", "0"], ["--kv-gb", "-1"], ["--max-seq-len", "abc"],
        ] {
            #expect(throws: BenchOptionError.self) {
                _ = try BenchOptions.parse(["--model", "/tmp/m"] + argument)
            }
        }
    }

    @Test("a missing option value is an error, not a silent default")
    func missingValueIsAnError() {
        #expect(throws: BenchOptionError.missingValue(option: "--prompt-lengths")) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths"])
        }
        #expect(throws: BenchOptionError.unknownOption("--engnies")) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--engnies", "v2"])
        }
        #expect(throws: BenchOptionError.missingModel) {
            _ = try BenchOptions.parse(["--steps", "8"])
        }
        // A mistyped mode used to run neither correctness nor perf and emit an
        // empty but successful-looking report.
        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--mode", "pref"])
        }
    }

    @Test("--print-revision needs no model")
    func printRevisionNeedsNoModel() throws {
        let options = try BenchOptions.parse(["--print-revision"])
        #expect(options.printRevisionOnly)
    }

    // MARK: - Invocation provenance

    /// Split the argv `/bin/sh` recovers from a recorded invocation line.
    private func shellSplit(_ commandLine: String) throws -> [String] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", "printf '%s\\0' \(commandLine)"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        var fields = data.split(separator: 0, omittingEmptySubsequences: false)
            .map { String(decoding: $0, as: UTF8.self) }
        // Every element is NUL-terminated, so the split leaves a trailing "".
        if fields.last == "" { fields.removeLast() }
        return fields
    }

    @Test("an argv containing whitespace round-trips through the recorded invocation")
    func invocationRoundTripsWhitespace() throws {
        let argv = [
            "/tmp/build dir/BenchCBv2",
            "--model", "/Volumes/My Disk/models/gemma 4 e4b",
            "--label", "paged vs contiguous",
            "--prompt-lengths", "500,10000",
            "--out", "report 2026-07-25.md",
        ]
        let recorded = shellQuotedInvocation(argv)
        #expect(try shellSplit(recorded) == argv)
        // The old `joined(separator: " ")` form is exactly what this replaces.
        #expect(try shellSplit(argv.joined(separator: " ")) != argv)
    }

    @Test("hostile argv elements round-trip too")
    func invocationRoundTripsHostileArguments() throws {
        let argv = [
            "BenchCBv2",
            "--label", "it's a \"quoted\" tag",
            "--label", "$HOME `id` $(rm -rf /) ${x}",
            "--label", "tab\there",
            "--label", "line\nbreak",
            "--label", "back\\slash",
            "--label", "glob*?[]",
            "--label", "",
            "--label", "semi;colon&amp|pipe",
        ]
        #expect(try shellSplit(shellQuotedInvocation(argv)) == argv)
    }

    @Test("plain arguments stay unquoted so the common report is readable")
    func plainArgumentsAreNotQuoted() {
        #expect(shellQuotedInvocation(["BenchCBv2", "--steps", "128"])
            == "BenchCBv2 --steps 128")
        #expect(shellQuote("/tmp/models/gemma-4-e4b") == "/tmp/models/gemma-4-e4b")
        #expect(shellQuote("") == "''")
    }

    @Test("the report records the shell-escaped invocation, not a flattened one")
    func headerRecordsEscapedInvocation() throws {
        let argv = ["BenchCBv2", "--model", "/m/gemma 4", "--label", "a b"]
        let options = try BenchOptions.parse(Array(argv.dropFirst()))
        let header = reportHeader(
            options: options, argv: argv, chip: "Apple M4 Max", ramGB: 128,
            osVersion: "Version 26.0", hostLine: "load avg (1m) 0.4 / 16 cores",
            revision: "abc1234", date: Date(timeIntervalSince1970: 0))
        let invocationRow = try #require(
            header.split(separator: "\n").first { $0.hasPrefix("| Invocation |") })
        let recorded = String(invocationRow)
            .replacingOccurrences(of: "| Invocation | `", with: "")
            .replacingOccurrences(of: "` |", with: "")
        #expect(try shellSplit(recorded) == argv)
        #expect(header.contains("| Label | a b |"))
    }

    // MARK: - Build revision provenance

    @Test("the report stamps the injected build revision verbatim")
    func headerStampsInjectedRevision() throws {
        let options = try BenchOptions.parse(["--model", "/tmp/m"])
        let header = reportHeader(
            options: options, argv: ["BenchCBv2", "--model", "/tmp/m"],
            chip: "Apple M4 Max", ramGB: 128, osVersion: "Version 26.0",
            hostLine: "load avg (1m) 0.4 / 16 cores", revision: "deadbee-dirty",
            date: Date(timeIntervalSince1970: 0))
        #expect(header.contains("| mlx-swift-lm (build) | deadbee-dirty |"))
        // Nothing in the header may re-derive a revision of its own.
        #expect(!header.contains("| mlx-swift-lm |"))
    }

    @Test("the build revision is a compile-time constant, not a runtime git query")
    func buildRevisionIsStampedAtBuildTime() {
        let revision = buildRevision()
        #expect(revision == BenchBuildRevision.value)
        #expect(!revision.isEmpty)

        // Well-formed: `unknown`, or a short SHA optionally marked dirty. The
        // stamp is produced by scripts/stamp-bench-revision.sh at build time.
        let pattern = /^(unknown|[0-9a-f]{7,40}(-dirty)?)$/
        #expect(
            revision.wholeMatch(of: pattern) != nil,
            "unexpected build revision stamp: \(revision)")

        // Invariant to the process's working directory: the value is baked in,
        // so it cannot pick up the git state of wherever the binary is run.
        let original = FileManager.default.currentDirectoryPath
        defer { FileManager.default.changeCurrentDirectoryPath(original) }
        #expect(FileManager.default.changeCurrentDirectoryPath("/tmp"))
        #expect(buildRevision() == revision)
    }
}
