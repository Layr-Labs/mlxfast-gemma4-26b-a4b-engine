import Foundation
import MLXFastCore
@testable import MLXFastRuntimeWorkerSupport
import Testing

// The `--mtp-head <DIR>` spawn channel, restored 2026-08-25 to close the
// cross-repo spawn-contract mismatch with benchd: benchd's measure-job spawns
// EVERY worker leg `runtime-worker --weights <W> --mtp-head <H>
// [--speculative-protocol v1.1]` (benchd @ c2327d15,
// crates/benchctl/src/measure_job.rs `timed_leg_base_args` /
// `leg_spawn_args`, transport prefix `ChildStdioTransport::build_args`),
// while the verb's option surface had shrunk to
// {--weights, --speculative-protocol} — so every leg died pre-hello as
// "engine closed the stream before returning a response".
//
// Coverage here is the two halves of the fix, both GPU-free (no model, no
// transport):
//   1. the OPTION SURFACE — the engine's accepted-flag set is pinned equal to
//      benchd's spawn fence (`measure_job::RUNTIME_WORKER_ACCEPTED_FLAGS`),
//      and the CLI's `runtime-worker` case is wired to that shared constant
//      (source-text tripwire, the same style `MTPTimingRetirementTests`
//      uses — `WorkerOptions` is private to the executable target, so the
//      wiring is observed in source);
//   2. the STAGING RESOLUTION — explicit argv directory vs the CWD default,
//      with the explicit channel fail-closed (`resolveGemma4AssistantHeadStaging`)
//      and the resolved directory being what the real loader consumes
//      (`loadGemma4AssistantDraftModelSync` throws OUT OF THE GIVEN DIRECTORY
//      on a broken head, before any weights work).

// MARK: - Option surface (the cross-repo spawn contract)

// benchd's fence, verbatim: `RUNTIME_WORKER_ACCEPTED_FLAGS` in
// crates/benchctl/src/measure_job.rs (benchd @ c2327d15). benchd asserts its
// spawn argv ⊆ this set before spawning any worker; the engine must accept
// exactly this set, and refuse everything else, or the two repos disagree at
// the spawn boundary again.
//
// `--dflash-head` JOINED THE SET (David ruling 2026-08-26) when DFlash became a
// first-class SCORED mode on this track. The two constants move TOGETHER or the
// legs die pre-hello again: benchd's `validate_spawn_argv` refuses to spawn a
// flag outside its own fence, and `requireOnly` here refuses one outside this
// set, so a one-sided addition breaks the run in one direction or the other.
private let benchdRuntimeWorkerAcceptedFlags: Set<String> = [
    "--weights",
    "--mtp-head",
    "--dflash-head",
    "--speculative-protocol",
]

@Test
func acceptedOptionSurfaceMatchesBenchdSpawnFence() {
    #expect(runtimeWorkerAcceptedOptionFlags == benchdRuntimeWorkerAcceptedFlags)
    #expect(runtimeWorkerMTPHeadFlag == "--mtp-head")
    #expect(runtimeWorkerDFlashHeadFlag == "--dflash-head")
}

// The FULL argv benchd constructs, per leg regime (measure_job.rs
// `leg_spawn_args` over the transport's `runtime-worker --weights <W>`
// prefix): a teacher-forced leg adds `--mtp-head <H>`, a free-run leg adds
// `--mtp-head <H> --speculative-protocol v1.1`. Every flag-shaped token in
// both regimes must be in the engine's accepted set — this is the exact
// per-flag check benchd's `validate_spawn_argv` runs on its side.
@Test
func everyFlagInBenchdsFullSpawnArgvIsAccepted() {
    let teacherForcedLeg = [
        "runtime-worker", "--weights", "/w", "--mtp-head", "/h",
    ]
    let freeRunLeg = teacherForcedLeg + ["--speculative-protocol", "v1.1"]
    // David ruling 2026-08-26 — the DFLASH leg, the argv benchd's
    // `paired_leg_spawn_args` builds when a drafter is staged. Each leg carries
    // its OWN `--dflash-head` value (pinned to the serial control, BYO to the
    // candidate); what this test pins is that the FLAG is accepted at all.
    let dflashLeg = freeRunLeg + ["--dflash-head", "/d"]
    for argv in [teacherForcedLeg, freeRunLeg, dflashLeg] {
        for flag in argv.dropFirst() where flag.hasPrefix("--") {
            #expect(
                runtimeWorkerAcceptedOptionFlags.contains(flag),
                "benchd spawns \(flag); the runtime-worker verb must accept it"
            )
        }
    }
}

// The CLI wiring, observed in source (CWD == package root under `swift test`,
// the same invariant `HarnessHashRootSetTests` relies on): the
// `runtime-worker` case must (a) gate its options on the SHARED constant —
// not a re-typed local set that can drift from benchd's fence — and (b)
// actually thread the head flag's value through to `runWorker`, preserving
// present-vs-absent (`optionalValue`, not an empty-string default).
@Test
func runtimeWorkerVerbIsWiredToTheSharedOptionSurface() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastRuntimeWorkerCLI/main.swift",
        encoding: .utf8)
    #expect(source.contains("values: runtimeWorkerAcceptedOptionFlags"))
    #expect(source.contains("mtpHeadPath: options.optionalValue("))
    #expect(source.contains("for: runtimeWorkerMTPHeadFlag"))
    // …and the same for the DFlash drafter's per-leg channel. `optionalValue`,
    // not an empty-string default: present-vs-absent is the whole distinction
    // between "this leg was handed its own drafter" and "fall back to the CWD
    // default", and an empty string would resolve to the fail-closed branch
    // with a nonsense path instead of to the default.
    #expect(source.contains("dflashHeadPath: options.optionalValue("))
    #expect(source.contains("for: runtimeWorkerDFlashHeadFlag"))
}

// MARK: - DFlash per-leg staging (David ruling 2026-08-26)

// benchd spawns both legs with NO `current_dir`, so before `--dflash-head` both
// legs resolved the SAME CWD `./dflash-head/` — the candidate's drafter resident
// on the scored DENOMINATOR leg, silently. These cases pin the resolution half
// of the fix; the loader half (that the resolved directory is what gets loaded)
// is `Gemma4DFlashStagedHeadSizeGateTests`.

@Test
func dflashPerLegStagingResolvesEachLegToItsOwnDirectory() throws {
    let pinned = try makeTempDirectory()
    let byo = try makeTempDirectory()
    defer {
        try? FileManager.default.removeItem(at: pinned)
        try? FileManager.default.removeItem(at: byo)
    }
    for dir in [pinned, byo] {
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
    }

    // The serial CONTROL leg's argv value and the CANDIDATE leg's argv value
    // resolve to DIFFERENT directories. This is the property the whole flag
    // exists for: two workers, one process CWD, two drafters.
    var resolved: [URL] = []
    for path in [pinned.path, byo.path] {
        let staging = try resolveGemma4AssistantHeadStaging(
            explicitDirectoryPath: path,
            defaultDirectoryName: gemma4DFlashHeadDirectoryName,
            flagName: runtimeWorkerDFlashHeadFlag)
        guard case .staged(let url) = staging else {
            Issue.record("an explicitly staged drafter must resolve to .staged")
            return
        }
        resolved.append(url.standardizedFileURL)
    }
    #expect(resolved[0] != resolved[1])
    #expect(resolved[0].path == pinned.standardizedFileURL.path)
    #expect(resolved[1].path == byo.standardizedFileURL.path)
}

@Test
func dflashExplicitStagingFailsClosedAndNamesItsOwnFlag() throws {
    // FAIL-CLOSED is load-bearing here beyond the usual reason: a
    // declared-but-missing per-leg drafter that fell back to the CWD default
    // would put BOTH legs back on one shared directory — restoring the exact
    // cross-leg residency this flag closes, silently.
    let missing = FileManager.default.temporaryDirectory
        .appendingPathComponent("dflash-head-missing-\(UUID().uuidString)").path
    var thrown: (any Error)?
    do {
        _ = try resolveGemma4AssistantHeadStaging(
            explicitDirectoryPath: missing,
            defaultDirectoryName: gemma4DFlashHeadDirectoryName,
            flagName: runtimeWorkerDFlashHeadFlag)
    } catch {
        thrown = error
    }
    let message = String(describing: thrown)
    #expect(thrown != nil, "a declared-but-missing drafter must refuse, never degrade")
    // The refusal must name the flag the OPERATOR passed, not the MTP flag the
    // shared resolver defaults to.
    #expect(message.contains(runtimeWorkerDFlashHeadFlag))
    #expect(!message.contains(runtimeWorkerMTPHeadFlag))
}

@Test
func dflashAbsentFlagKeepsTheCWDDefaultChannel() throws {
    // The no-perturbation control: with no `--dflash-head`, resolution is the
    // pre-ruling CWD default. A checkout that never ran the staging step has no
    // `./dflash-head/config.json`, so this is `.none` — capability-absent, never
    // an error, exactly as before.
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: nil,
        defaultDirectoryName: "dflash-head-definitely-not-staged-\(UUID().uuidString)",
        flagName: runtimeWorkerDFlashHeadFlag)
    #expect(staging == .none)
}

// MARK: - Staging resolution: explicit argv channel (fail-closed)

private func makeTempDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mtp-head-staging-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Test
func explicitDirectoryWithConfigResolvesToThatDirectory() throws {
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))

    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: dir.path)
    guard case .staged(let resolved) = staging else {
        Issue.record("explicit staged directory must resolve to .staged")
        return
    }
    #expect(resolved.standardizedFileURL.path == dir.standardizedFileURL.path)
}

@Test
func explicitMissingDirectoryFailsClosed() {
    let missing = FileManager.default.temporaryDirectory
        .appendingPathComponent("mtp-head-missing-\(UUID().uuidString)").path
    #expect(throws: (any Error).self) {
        _ = try resolveGemma4AssistantHeadStaging(explicitDirectoryPath: missing)
    }
}

@Test
func explicitPathThatIsAFileFailsClosed() throws {
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    let file = dir.appendingPathComponent("not-a-directory")
    try Data().write(to: file)
    #expect(throws: (any Error).self) {
        _ = try resolveGemma4AssistantHeadStaging(explicitDirectoryPath: file.path)
    }
}

@Test
func explicitDirectoryWithoutConfigFailsClosed() throws {
    // The DEFAULT channel treats this same shape (a placeholder directory, no
    // config.json) as "not staged" — the explicit channel must refuse it: an
    // argv declaration that resolves to an unloadable head is
    // present-but-broken, never silently serial-only.
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    #expect(throws: (any Error).self) {
        _ = try resolveGemma4AssistantHeadStaging(explicitDirectoryPath: dir.path)
    }
}

@Test
func explicitEmptyPathFailsClosed() {
    #expect(throws: (any Error).self) {
        _ = try resolveGemma4AssistantHeadStaging(explicitDirectoryPath: "")
    }
}

// MARK: - Staging resolution: default CWD channel (lenient, unchanged)

@Test
func absentFlagWithNoStagedDirectoryIsNoneNotAnError() throws {
    let missingName = "mtp-head-default-missing-\(UUID().uuidString)"
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: nil, defaultDirectoryName: missingName)
    #expect(staging == .none)
}

@Test
func absentFlagWithPlaceholderDirectoryIsNoneNotAnError() throws {
    // An existing directory with no config.json — the checked-in
    // `mtp-head/README.md` placeholder shape — stays "not staged, not broken"
    // on the default channel.
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data("placeholder".utf8).write(to: dir.appendingPathComponent("README.md"))
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: nil, defaultDirectoryName: dir.path)
    #expect(staging == .none)
}

@Test
func absentFlagWithStagedDefaultDirectoryResolvesToIt() throws {
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: nil, defaultDirectoryName: dir.path)
    guard case .staged(let resolved) = staging else {
        Issue.record("staged default directory must resolve to .staged")
        return
    }
    #expect(resolved.standardizedFileURL.path == dir.standardizedFileURL.path)
}

@Test
func explicitChannelWinsOverTheDefaultName() throws {
    // When BOTH are plausible, the argv directory is the one that loads: the
    // default name is never consulted once an explicit path is given.
    let explicitDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: explicitDir) }
    try Data("{}".utf8).write(to: explicitDir.appendingPathComponent("config.json"))

    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: explicitDir.path,
        defaultDirectoryName: "mtp-head-never-consulted-\(UUID().uuidString)")
    guard case .staged(let resolved) = staging else {
        Issue.record("explicit directory must win over the default name")
        return
    }
    #expect(resolved.standardizedFileURL.path == explicitDir.standardizedFileURL.path)
}

// MARK: - The load seam: the resolved directory is what the loader consumes

@Test
func brokenHeadInExplicitDirectoryFailsAtLoadFromThatDirectory() throws {
    // A directory that RESOLVES (exists, has config.json) but does not load —
    // malformed config — throws out of the real loader, before any weights or
    // GPU work (`Gemma4AssistantDraftModel.load` reads config.json first).
    // This is the "present-but-broken head is a refusal" half of the startup
    // contract, exercised from the directory the argv named rather than CWD.
    let dir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data("not json".utf8).write(to: dir.appendingPathComponent("config.json"))

    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: dir.path)
    guard case .staged(let resolved) = staging else {
        Issue.record("directory with config.json must resolve to .staged")
        return
    }
    #expect(throws: (any Error).self) {
        _ = try loadGemma4AssistantDraftModelSync(from: resolved)
    }
}
