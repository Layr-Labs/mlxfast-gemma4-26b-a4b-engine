import Foundation
import MLXFastCore
@testable import MLXFastHarness
import Testing

// The DFlash drafter is the gemma4-26b-a4b-mlx-v1 track's SECOND speculative
// head, and per David's 2026-08-25 ruling it gets the SAME 2 GiB size-only cap
// the MTP head gets, through the SAME declaration mechanism:
// `spec-decoder-head.manifest.json` (`source` / optional `sha256` / `bytes` /
// `max_bytes`), parsed and size-gated by `Gemma4MTPHeadDeclaration` with
// `kind: .dflash`, absent-means-organizer-default, present-but-broken-means-
// refusal.
//
// REQUANT-ONLY (David ruling, 2026-08-26): `pinned` is the only source either
// head accepts. `remote` and `in_branch` are retired and are now named
// refusals, so every case below that used to DRIVE the size gate through
// `in_branch` or `remote` now drives it through `pinned` with a stated size.
// That is not a cosmetic rewrite: it is what keeps the size gate under test
// after the sources that used to reach it were retired.
//
// This suite is the DFlash mirror of Gemma4MTPHeadDeclarationSizeGateTests. What
// it pins:
//   * the gate BINDS for .dflash exactly as for .mtp (over-cap refused, at-cap
//     accepted, lowered cap honored, raised cap refused, digestless accepted,
//     declared digest carried but never a gate);
//   * the retired sources refuse for .dflash exactly as for .mtp;
//   * absence is NO CAPABILITY, not an error;
//   * present-but-broken is a refusal, never a silent fall back;
//   * refusal text names the cap AND the actual size;
//   * the mirror did not RENAME the MTP mechanism out from under itself --
//     .mtp still resolves mtp-head.manifest.json and still says "MTP".
//
// FIX-BAR / mutations to kill: giving DFlash its own (larger, or absent) cap
// turns `dflashHeadOneByteOverTheCapIsRefused` red; letting a broken DFlash
// declaration fall back to the pinned default turns
// `presentButBrokenDFlashManifestIsARefusal` red.
//
// Scope note: no MLX device work, no weights, no GPU, no network.

private let dflashDigest = String(repeating: "Cd", count: 32)

private func makeDFlashTempDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("dflash-head-decl-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

// --- The mechanism is shared, and the cap is ONE cap ------------------------

@Test
func dflashKindNamesItsOwnManifestAndDirectory() {
    #expect(ReplaceableHeadKind.dflash.manifestRelativePath == "spec-decoder-head.manifest.json")
    #expect(ReplaceableHeadKind.dflash.stagedDirectoryName == "dflash-head")
    // The mirror must not have renamed the MTP side.
    #expect(ReplaceableHeadKind.mtp.manifestRelativePath == "mtp-head.manifest.json")
    #expect(ReplaceableHeadKind.mtp.stagedDirectoryName == "mtp-head")
    #expect(Gemma4MTPHeadDeclaration.relativePath == "mtp-head.manifest.json")
    // ONE cap, not one per head: 2 GiB, the same constant both manifests
    // declare as `max_bytes`. It is NOT the contract's
    // editableSurfaceByteBudget.exemptPathMaxBytes -- that one bounded bytes
    // that RODE IN a submission, nothing rides in any more, and the two numbers
    // were never equal after the 2026-08-26 BYO-512 ruling anyway.
    #expect(Gemma4MTPHeadDeclaration.defaultMaxBytes == 2_147_483_648)
}

// --- Size enforcement for the DFlash head -----------------------------------

@Test
func dflashHeadAtTheCapIsAccepted() throws {
    // Exactly at the cap is INSIDE it (`bytes <= maxBytes`).
    let atCap = Gemma4MTPHeadDeclaration.defaultMaxBytes
    let json = #"{"source":"pinned","bytes":\#(atCap)}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(
        data: Data(json.utf8), origin: "test", kind: .dflash)
    #expect(decl.kind == .dflash)
    #expect(decl.source == .pinned)
    #expect(decl.bytes == atCap)
    #expect(decl.maxBytes == atCap)
}

@Test
func dflashHeadOneByteOverTheCapIsRefused() throws {
    let overCap = Gemma4MTPHeadDeclaration.defaultMaxBytes + 1
    let json = #"{"source":"pinned","bytes":\#(overCap)}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "spec-decoder-head.manifest.json", kind: .dflash)
    }
}

@Test
func dflashOverCapRefusalNamesTheCapAndTheActualSize() throws {
    let overCap = Gemma4MTPHeadDeclaration.defaultMaxBytes + 1
    // Driven through `pinned`: a `remote` declaration now refuses for being
    // remote, BEFORE the size gate is reached, so it can no longer prove the
    // size gate names anything.
    let json = #"{"source":"pinned","bytes":\#(overCap)}"#
    do {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "spec-decoder-head.manifest.json", kind: .dflash)
        Issue.record("an over-cap DFlash declaration must refuse")
    } catch let error as MLXFastError {
        let message = error.description
        #expect(message.contains("\(overCap)"), "refusal must name the actual size")
        #expect(
            message.contains("\(Gemma4MTPHeadDeclaration.defaultMaxBytes)"),
            "refusal must name the cap")
        #expect(message.contains("DFlash"), "refusal must name which head it is about")
        #expect(message.contains("spec-decoder-head.manifest.json"))
    }
}

@Test
func dflashHeadOverLoweredCapIsRefused() throws {
    // A declaration may LOWER the cap; bytes above the lowered max_bytes refuse.
    let json = #"""
    {"source":"pinned","max_bytes":1000,"bytes":1001}
    """#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "test", kind: .dflash)
    }
}

@Test
func dflashMaxBytesAboveTheTrackCapIsRefused() throws {
    // May lower, may not raise -- no per-head escape hatch above 2 GiB.
    let raised = Gemma4MTPHeadDeclaration.defaultMaxBytes + 1
    let json = #"""
    {"source":"pinned","max_bytes":\#(raised),"bytes":1}
    """#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "test", kind: .dflash)
    }
}

@Test
func dflashHeadWithANegativeStatedByteCountIsRefused() throws {
    // `bytes: 0` now means NOT STATED and is accepted -- that is what both
    // checked-in declarations say. A stated size must still be a positive
    // number, so a negative one fails closed rather than reading as "absent".
    let json = #"{"source":"pinned","bytes":-1}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "test", kind: .dflash)
    }
    let unstated = try Gemma4MTPHeadDeclaration.parse(
        data: Data(#"{"source":"pinned","bytes":0}"#.utf8), origin: "test", kind: .dflash)
    #expect(unstated.bytes == 0)
}

@Test
func dflashRetiredSourcesAreRefused() throws {
    // The DFlash half of the requant-only ruling. Both retired sources, in the
    // shapes that were ACCEPTED before it.
    for json in [
        #"{"source":"in_branch","path":"dflash-head/w","bytes":1024}"#,
        #"{"source":"remote","source_url":"hf:acme/dflash@abc","bytes":1024}"#,
    ] {
        #expect(throws: MLXFastError.self, "retired source must refuse: \(json)") {
            _ = try Gemma4MTPHeadDeclaration.parse(
                data: Data(json.utf8), origin: "spec-decoder-head.manifest.json", kind: .dflash)
        }
    }
}

// --- DECIDE-2 Q-B parity: SIZE only, the digest is carried not gated --------

@Test
func digestlessDFlashHeadAcceptedUnderCap() throws {
    let json = #"{"source":"pinned","bytes":4096}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(
        data: Data(json.utf8), origin: "test", kind: .dflash)
    #expect(decl.kind == .dflash)
    #expect(decl.source == .pinned)
    #expect(decl.sha256 == nil)
    #expect(decl.bytes == 4096)
}

@Test
func dflashHeadDeclaredDigestIsCarriedLowercasedNotGated() throws {
    let json = #"""
    {"source":"pinned","bytes":1024,"sha256":"\#(dflashDigest)"}
    """#
    let decl = try Gemma4MTPHeadDeclaration.parse(
        data: Data(json.utf8), origin: "test", kind: .dflash)
    #expect(decl.sha256 == dflashDigest.lowercased())
    #expect(decl.bytes == 1024)
}

// --- Absent = no DFlash capability; present-but-broken = refusal ------------

@Test
func absentDFlashManifestIsTheOrganizerDefaultNotAnError() throws {
    // An empty contract root carries no spec-decoder-head.manifest.json: that is the
    // NORMAL case (no DFlash capability declared), never a refusal.
    let root = try makeDFlashTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let decl = try Gemma4MTPHeadDeclaration.resolve(contractRoot: root, kind: .dflash)
    #expect(decl == Gemma4MTPHeadDeclaration.pinnedDefault(for: .dflash))
    #expect(decl.source == .pinned)
    #expect(decl.kind == .dflash)
}

@Test
func sourcelessDFlashManifestStaysTheOrganizerDefault() throws {
    let decl = try Gemma4MTPHeadDeclaration.parse(
        data: Data("{}".utf8), origin: "test", kind: .dflash)
    #expect(decl == Gemma4MTPHeadDeclaration.pinnedDefault(for: .dflash))
}

@Test
func presentButBrokenDFlashManifestIsARefusal() throws {
    // Every one of these is a manifest that EXISTS; none may silently select
    // the organizer default. Driven through `resolve` so the file-present path
    // (not just `parse`) is the thing under test. The `remote` / `in_branch`
    // entries refused for a MALFORMED url or path before the requant-only
    // ruling and refuse for the SOURCE itself after it -- either way the
    // property under test here is "present-but-broken never falls back".
    let broken = [
        "not json at all",
        #"{"source":"borrowed","bytes":10}"#,
        #"{"source":"remote","bytes":10}"#,
        #"{"source":"in_branch","bytes":10}"#,
        #"{"source":"in_branch","path":"../escape","bytes":10}"#,
        #"{"source":"in_branch","path":"dflash-head/w"}"#,
    ]
    for body in broken {
        let root = try makeDFlashTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(body.utf8).write(
            to: root.appendingPathComponent(ReplaceableHeadKind.dflash.manifestRelativePath))
        #expect(throws: MLXFastError.self, "broken DFlash manifest must refuse: \(body)") {
            _ = try Gemma4MTPHeadDeclaration.resolve(contractRoot: root, kind: .dflash)
        }
    }
}

@Test
func resolveReadsTheKindsOwnManifestOnly() throws {
    // A broken MTP manifest must not fail a DFlash resolve, and vice versa:
    // the two declarations are independent files.
    let root = try makeDFlashTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try Data("not json".utf8).write(
        to: root.appendingPathComponent(ReplaceableHeadKind.mtp.manifestRelativePath))
    let decl = try Gemma4MTPHeadDeclaration.resolve(contractRoot: root, kind: .dflash)
    #expect(decl == Gemma4MTPHeadDeclaration.pinnedDefault(for: .dflash))
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.resolve(contractRoot: root, kind: .mtp)
    }
}

// --- The checked-in declaration ---------------------------------------------

@Test
func checkedInDFlashManifestParsesAndSelectsTheOrganizerDefault() throws {
    // CWD == package root under `swift test` (the invariant
    // HarnessHashRootSetTests relies on), so the repo's own declaration is the
    // one parsed here.
    let decl = try Gemma4MTPHeadDeclaration.resolve(
        contractRoot: URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        kind: .dflash)
    #expect(decl.source == .pinned)
    #expect(decl.kind == .dflash)
    #expect(decl.maxBytes == Gemma4MTPHeadDeclaration.defaultMaxBytes)
}
