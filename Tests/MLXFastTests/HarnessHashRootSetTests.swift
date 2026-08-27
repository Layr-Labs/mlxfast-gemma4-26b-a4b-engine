import CryptoKit
import Foundation
import Testing

import MLXFastCore
@testable import MLXFastHarness

// Revert-proof coverage for the root-level benchmark.sh proxy and the 9-root
// set Gemma4Runtime.harnessHash() hashes.
//
// harnessHash() (Sources/MLXFastTrustedHarness/Gemma4RuntimePreflight.swift and
// the byte-identical Sources/MLXFastHarness/ copy) hashes a FIXED 9-root set
// that names "benchmark.sh" at index 4, resolving each root RELATIVE TO THE
// PROCESS CWD and silently `continue`-ing past any it cannot find. Two failure
// modes these tests pin shut:
//
//   1. benchmark.sh deleted again (commit 92bdeccc's over-strip) -> the root is
//      skipped, only 8/9 roots contribute, and the harness hash goes quietly
//      dishonest with no error anywhere.
//   2. the benchmark process run from the wrong CWD -> the roots resolve
//      nowhere and the hash collapses toward the empty-set digest.
//
// `swift test` runs with CWD == the package root (every source-reading test in
// this suite depends on that -- e.g. DFlashStartupMemoryPolicyTests reads
// "Sources/..." by relative path), so the ambient-CWD assertions below observe
// exactly what the real benchmark process observes.
//
// Scope note: no MLX device work, no weights, no GPU, no network, and the
// channel benchctl is NOT required to be resolved (benchd-bin/ need not exist)
// -- the proxy is asserted by its static content, never executed.

private enum HarnessHashRootFixture {
    /// The 9-root set, taken from the REAL array Gemma4Runtime.harnessHash() hashes
    /// (Gemma4Runtime.harnessHashRoots) rather than hand-copied, so this fixture
    /// cannot drift from production: a root added or removed there flows straight
    /// in, and the "benchmark.sh at index 4" assertion below reds if that changes
    /// the ordering. index 4 is the restored benchmark.sh file.
    static let roots = Gemma4Runtime.harnessHashRoots

    static let benchmarkShellIndex = 4

    /// The package root, resolved from THIS source file so the location is
    /// known independent of the process CWD.
    static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests/MLXFastTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // repo root

    /// The two byte-identical copies of Gemma4RuntimePreflight.swift, addressed
    /// location-relative (independent of the process CWD). Package.swift maps
    /// target NAMES to the INVERTED directories: the "MLXFastTrustedHarness"
    /// directory compiles into the target these tests `@testable import` as
    /// MLXFastHarness, while the "MLXFastHarness" directory compiles into the
    /// MLXFastRuntimeWorkerSupport module inside the mlxfast-runtime-worker
    /// executable -- the copy that stamps harnessHash() into sealed score
    /// payloads. The tests above (A-D) and the `Gemma4Runtime.harnessHashRoots`
    /// they read bind ONLY the trusted copy; nothing but the parity check below
    /// binds the worker copy, which is competitor-reachable.
    static let trustedPreflight: URL = repoRoot
        .appendingPathComponent("Sources/MLXFastTrustedHarness/Gemma4RuntimePreflight.swift")
    static let workerPreflight: URL = repoRoot
        .appendingPathComponent("Sources/MLXFastHarness/Gemma4RuntimePreflight.swift")

    /// Lowercase-hex SHA256 of `data`, for byte-parity failure diagnostics.
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// SHA256 of zero bytes -- the digest harnessHash() returns when NONE of its
    /// roots resolve (empty file set means the hasher takes no updates).
    static let emptySetDigest =
        SHA256.hash(data: Data()).map { String(format: "%02x", $0) }.joined()

    /// How many of the 9 roots exist under `base`, using the exact predicate
    /// harnessHash() uses (`FileManager.fileExists`). `base == nil` resolves
    /// CWD-relative, exactly as harnessHash() does.
    static func rootsPresent(under base: URL?) -> Int {
        roots.reduce(into: 0) { count, root in
            let path = base.map { $0.appendingPathComponent(root).path } ?? root
            if FileManager.default.fileExists(atPath: path) {
                count += 1
            }
        }
    }
}

// Test A -- the root set resolves benchmark.sh at the TOP LEVEL (not benchd's
// facade), so all 9 harnessHash roots contribute. Guards the delete-again
// revert: with benchmark.sh gone this drops to 8/9 and every assertion binding
// on the file goes red.
@Test
func harnessHashRootSetResolvesBenchmarkShellAtTopLevel() {
    // Index 4 of the copied root set is the file this test guards.
    #expect(HarnessHashRootFixture.roots[HarnessHashRootFixture.benchmarkShellIndex] == "benchmark.sh")

    // CWD == package root under `swift test`: this is the exact predicate
    // harnessHash() applies to the benchmark.sh root. Deleting benchmark.sh
    // flips it false and the hash silently falls to 8/9 roots.
    #expect(FileManager.default.fileExists(atPath: "benchmark.sh"))

    // It must be the TOP-LEVEL file, not the relocated facade at
    // tools/benchmark.sh (nor, historically, the benchd submodule's copy at
    // benchd/scripts/benchmark.sh -- that submodule no longer exists).
    let topLevel = HarnessHashRootFixture.repoRoot.appendingPathComponent("benchmark.sh")
    #expect(FileManager.default.fileExists(atPath: topLevel.path))
    #expect(!topLevel.path.contains("/benchd/"))
    #expect(!topLevel.path.contains("/tools/"))

    // All 9 roots resolve from the package root: the hash is 9/9, not 8/9.
    #expect(HarnessHashRootFixture.rootsPresent(under: nil) == HarnessHashRootFixture.roots.count)
}

// Test B -- the restored benchmark.sh is a THIN PROXY that resolves the verified
// benchctl and execs the facade at tools/benchmark.sh forwarding all args.
// Static-content smoke only; the real benchmark is never run.
//
// UPDATED for the pin-removal ruling (David 2026-08-27). benchd was first a
// SHA-pinned SOURCE submodule (`git submodule update --init` + a gitlink
// compare), then a PREBUILT binary frozen by ./benchd.pin. The pin file is now
// GONE: tools/fetch-benchd.sh resolves the bench branch's dist channel and
// verifies the binary against the channel's benchctl.manifest.json
// ({branch, source_commit, sha256, bytes}) -- provenance is RECORDED per run
// rather than frozen in this repo. What this proxy must still never do is
// carry a harness identity of its own: no submodule revival, no from-source
// build, and no hardcoded commit/sha literal that would shadow the manifest.
@Test
func rootBenchmarkShellIsThinProxyToBenchd() throws {
    let proxy = HarnessHashRootFixture.repoRoot.appendingPathComponent("benchmark.sh")

    #expect(FileManager.default.fileExists(atPath: proxy.path))

    // Executable (owner-or-any execute bit set).
    let perms = try #require(
        FileManager.default.attributesOfItem(atPath: proxy.path)[.posixPermissions] as? NSNumber
    )
    #expect((perms.intValue & 0o111) != 0)

    let body = try String(contentsOf: proxy, encoding: .utf8)

    // Strict bash preamble.
    #expect(body.hasPrefix("#!/usr/bin/env bash"))
    #expect(body.contains("set -euo pipefail"))

    // Resolves its own directory, so it is robust to the caller's CWD.
    #expect(body.contains(#"SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)""#))
    #expect(body.contains(#"BENCHD_ENTRY="${SCRIPT_DIR}/tools/benchmark.sh""#))

    // Resolves the measurement binary through the manifest-verifying fetcher ...
    #expect(body.contains("tools/fetch-benchd.sh"))
    // ... and never reintroduces the submodule path it replaced ...
    #expect(!body.contains("submodule update --init"))
    #expect(!body.contains("submodule update --remote"))
    // ... nor the pin file the fetcher replaced (the pin-removal ruling deleted
    // ./benchd.pin; a proxy that re-grows a pin path re-freezes the harness) ...
    #expect(!body.contains("benchd.pin"))
    //
    // 98f44fa is the RETIRED submodule gitlink: a fixed historical value that
    // must never reappear, so it is spelled literally.
    #expect(!body.contains("98f44fa"))
    // With the pin file gone there is no repo-side identity to compare against,
    // so the anti-inlining guard generalizes: the proxy must contain NO
    // commit-or-sha-shaped hex literal AT ALL (40 hex chars covers a git commit;
    // a sha256 is 64 and matches the same scan). A proxy that inlines the
    // resolved identity would shadow the channel manifest -- the run would keep
    // "verifying" bytes the channel has since moved past -- and reds here
    // immediately.
    #expect(
        body.range(of: "[0-9a-fA-F]{40}", options: .regularExpression) == nil,
        "proxy hardcodes a commit/sha-shaped hex literal; harness identity lives in the channel manifest"
    )
    // ... and never falls back to building benchd from source: the ranked box has
    // no Rust toolchain, and a from-source fallback would silently defeat the pin.
    // (Spelled as a literal the proxy must not contain -- including in prose, which
    // is why this file says "Rust toolchain" wherever it means that build tool.)
    #expect(!body.contains("cargo"))

    // Execs benchd's facade forwarding every argument unchanged.
    #expect(body.contains(#"exec "${BENCHD_ENTRY}" "$@""#))

    // THIN: this proxy must not regrow into the 2181-line original benchmark.
    let lineCount = body.split(separator: "\n", omittingEmptySubsequences: false).count
    #expect(lineCount < 120)
}

// Test C -- the CWD invariant. harnessHash() resolves its roots relative to the
// PROCESS CWD; the benchmark process must run with CWD == repo root or the
// 9-root hash silently collapses. Proven without mutating the shared process
// CWD (swift-testing runs suites in parallel and other suites read relative
// paths), by exercising the real function in the ambient CWD and the
// location-relative resolution against a controlled base.
@Test
func harnessHashRootsAreCwdRelativeAndLoadBearing() throws {
    // (1) The REAL function, in the actual benchmark CWD (package root under
    // `swift test`), hashes a NON-EMPTY file set: its digest is not the
    // empty-set digest. Were the process run from a CWD where the roots do not
    // resolve, harnessHash() would return exactly the empty-set digest -- so
    // this inequality IS the observable that makes CWD load-bearing.
    let liveHash = Gemma4Runtime.harnessHash()
    #expect(!liveHash.isEmpty)
    #expect(liveHash != HarnessHashRootFixture.emptySetDigest)

    // (2) The roots are LOCATION-relative. From the package root all 9 resolve;
    // from an unrelated empty directory NONE do -- so a harnessHash() run there
    // would hash the empty set. That is the collapse the invariant guards.
    #expect(
        HarnessHashRootFixture.rootsPresent(under: HarnessHashRootFixture.repoRoot)
            == HarnessHashRootFixture.roots.count
    )

    let scratch = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("harnesshash-cwd-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }
    #expect(HarnessHashRootFixture.rootsPresent(under: scratch) == 0)
}

// Test D -- the proxy ESTABLISHES the CWD invariant Test C guards, rather than
// merely documenting it. harnessHash() collapses (Test C) from any CWD but the
// repo root; the benchd facade the proxy execs never cd's; so the proxy itself
// must `cd "${SCRIPT_DIR}"` before the exec, making the Swift benchmark process
// benchd launches inherit CWD == repo root no matter where the caller invoked
// the proxy. Static-content assertion (the benchd submodule need not be checked
// out and the real benchmark is never run): the proxy contains that exact cd and
// it PRECEDES the exec of benchd's facade. Deleting the cd line -- the revert
// this pins shut -- drops the #require and turns this red.
@Test
func rootBenchmarkShellCdsToScriptDirBeforeExec() throws {
    let proxy = HarnessHashRootFixture.repoRoot.appendingPathComponent("benchmark.sh")
    let body = try String(contentsOf: proxy, encoding: .utf8)
    let lines = body
        .split(separator: "\n", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespaces) }

    // The workspace root is established as the PROCESS CWD ...
    let cdIndex = try #require(
        lines.firstIndex(of: #"cd "${SCRIPT_DIR}""#),
        #"proxy must `cd "${SCRIPT_DIR}"` to establish the workspace root as the process CWD"#
    )
    // ... and only THEN is control handed to benchd's facade.
    let execIndex = try #require(
        lines.firstIndex { $0.hasPrefix(#"exec "${BENCHD_ENTRY}""#) },
        "proxy must exec benchd's facade"
    )
    #expect(cdIndex < execIndex)
}

// Test E -- copy parity (closes F-A). Gemma4RuntimePreflight.swift is duplicated:
// the trusted copy compiles into the harness these tests import, and the WORKER
// copy compiles into mlxfast-runtime-worker, which is what stamps harnessHash()
// into the sealed score. Tests A-D bind ONLY the trusted copy (they import it and
// read Gemma4Runtime.harnessHashRoots from it), so a reviewer proved that dropping a
// harnessHashRoots entry from the worker copy alone -- the one that ships in the
// competitor-reachable executable -- leaves the whole suite green on a clean
// rebuild. This reads BOTH files as raw bytes and asserts they are identical, so
// ANY divergence in EITHER copy (a dropped root, or any other edit) reds here and
// transitively binds the worker copy the other tests never touch. Enforcing parity
// by ASSERTION, not by merging the files: the trusted/worker split is a deliberate
// security boundary (only the worker is competitor-controllable) and must stay two
// separate compilation units.
@Test
func preflightWorkerAndTrustedCopiesAreByteIdentical() throws {
    let trustedBytes = try Data(contentsOf: HarnessHashRootFixture.trustedPreflight)
    let workerBytes = try Data(contentsOf: HarnessHashRootFixture.workerPreflight)

    #expect(
        trustedBytes == workerBytes,
        """
        Gemma4RuntimePreflight.swift copies have DRIFTED -- the trusted and worker \
        copies must stay byte-identical, but they differ. The worker copy \
        (Sources/MLXFastHarness/Gemma4RuntimePreflight.swift) is the one that stamps \
        harnessHash() into sealed scores and is NOT bound by any other test, so \
        this drift would otherwise go silent. \
        trusted(Sources/MLXFastTrustedHarness) sha256=\
        \(HarnessHashRootFixture.sha256Hex(trustedBytes)) bytes=\(trustedBytes.count); \
        worker(Sources/MLXFastHarness) sha256=\
        \(HarnessHashRootFixture.sha256Hex(workerBytes)) bytes=\(workerBytes.count)
        """
    )
}

// Test F -- independent membership pin (closes F-B). Test A's count assertion
// compares the live array to itself on disk, so a reviewer proved that dropping a
// root at index > 4 (e.g. "TASK.md") from even the TRUSTED harnessHashRoots leaves
// the suite green: the fixture re-derives its count from the same mutated array.
// This pins the roster with an INDEPENDENTLY hand-written literal (modeled on the
// bench crate's ROSTER_OF_EIGHT freeze): a future edit to the live array does NOT
// auto-update this literal, so any add, removal, or reorder must diverge from it
// and red. Keeping it as a literal -- never re-read from Gemma4Runtime -- is the
// whole point; do not "DRY" this against the production array.
@Test
func harnessHashRootsMatchesIndependentRoster() {
    // The frozen 9-root roster, in exact order, authored by hand. If a change to
    // Gemma4Runtime.harnessHashRoots is intended, update THIS literal in the same
    // commit -- that deliberate edit is the record that the roster changed.
    let expectedRoster: [String] = [
        "Package.swift",
        "Sources",
        "Tests",
        "benchmark.json",
        "benchmark.sh",
        "setup.sh",
        "tools",
        "README.md",
        "TASK.md",
    ]

    #expect(
        Gemma4Runtime.harnessHashRoots == expectedRoster,
        """
        Gemma4Runtime.harnessHashRoots has DRIFTED from the frozen roster. \
        Any add/removal/reorder of the harnessHash root set must be mirrored in \
        this independent literal in the same commit. \
        live=\(Gemma4Runtime.harnessHashRoots) expected=\(expectedRoster)
        """
    )
}

// Test G -- FAIL-CLOSED on a missing root (harnessHash-load-bearing ruling). Before
// the ruling, `harnessHashRootFiles()`'s predecessor silently `continue`d past any
// root FileManager could not find, so a dropped/renamed/wrong-CWD root produced a
// DIFFERENT-but-still-valid-looking digest (or, with every root gone, the empty-set
// digest) with NO error. The ruled behavior THROWS on the first missing root; the
// public `harnessHash()` turns that throw into a fatal (it is stamped from
// non-throwing benchmark entrypoints). This binds the throwing seam directly, off
// the shared process CWD, via the `baseDirectory` test hook.
//
// REVERT-PROOF (the exact mutation this kills): replace the
//   `throw MLXFastError.missingFile("harnessHash root missing from disk: \(root)")`
// in Gemma4Runtime.harnessHashRootFiles() with the pre-ruling `continue`. The empty
// scratch base then yields `[]` instead of throwing and the `#expect(throws:)`
// below goes RED. The baseline half (all 9 roots present under the repo root →
// NON-EMPTY, no throw) keeps the refusal non-vacuous: it proves the throw fires on
// the missing-root condition, not unconditionally.
@Test
func harnessHashRootFilesFailsClosedOnAMissingRoot() throws {
    // Baseline (non-vacuous): from the package root all 9 roots resolve, so the
    // helper returns a NON-EMPTY file list and does NOT throw. If the refusal below
    // fired unconditionally, this call would throw too and the test could not pass.
    let present = try Gemma4Runtime.harnessHashRootFiles(
        baseDirectory: HarnessHashRootFixture.repoRoot
    )
    #expect(!present.isEmpty)

    // An empty scratch directory has NONE of the 9 roots. The pre-ruling code
    // silently skipped each absent root and returned []/the empty-set digest; the
    // ruled fail-closed behavior THROWS on the FIRST missing root instead of
    // yielding a different-but-valid-looking hash.
    let scratch = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("harnesshash-missing-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }

    #expect(throws: MLXFastError.self) {
        _ = try Gemma4Runtime.harnessHashRootFiles(baseDirectory: scratch)
    }
}
