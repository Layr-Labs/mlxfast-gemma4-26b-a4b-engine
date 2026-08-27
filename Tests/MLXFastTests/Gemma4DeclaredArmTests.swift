import Foundation
import MLXFastCore
@testable import MLXFastHarness
import Testing

// THE SELECTION CHANNEL, parse side.
//
// `spec-decoder-head.manifest.json` gained an `arm` key on 2026-08-26: the channel a
// submission uses to say which speculative arm it wants measured. Until it
// existed, `benchmark.json`, `docs/participant-contract.md` section 5.1.1,
// `README.md` and `TASK.md` all promised "a submission runs whichever mode its
// own code drives" and nothing in the repository could deliver it -- the mode
// reached the benchmarker only through an operator's CLI flag.
//
// This suite pins the PARSE. The reader that ACTS on the arm is the trusted
// wrapper `tools/gemma4-measure-and-score.sh` (covered by
// `tools/test-gemma4-arm-selection.sh`, which also fails if this file's
// vocabulary and the wrapper's case list drift apart). What is pinned here is
// the property that makes the channel safe to have at all: the declaration
// parser used to read six keys and IGNORE every other one, so an `arm` key was
// accepted and inert. A key that selects which arm gets SCORED cannot be
// inert, and cannot fall back.
//
// What it pins:
//   * absent key => `DeclaredArm.absentDefault` (= .mtp), the one silent path;
//   * an unknown value REFUSES by name and names the vocabulary;
//   * a non-string value REFUSES (a bare `jq -r` would have stringified it);
//   * the vocabulary is CASE-SENSITIVE ("MTP" is a typo, not a synonym);
//   * `arm` on the manifest that does NOT declare the arm REFUSES -- the
//     mistyped-file tolerance, closed;
//   * the rest of the declaration is unchanged by any of it.
//
// FIX-BAR / mutations to kill: making the unknown-value branch return
// `.absentDefault` turns `unknownArmIsRefusedByName` red; dropping the
// wrong-manifest refusal turns `armOnTheMTPManifestIsRefused` red; reading the
// key with a plain `as? String` and a `?? .mtp` turns BOTH the non-string and
// the unknown case red.
//
// Scope note: no MLX device work, no weights, no GPU, no network.

private func parseDFlash(_ json: String) throws -> Gemma4MTPHeadDeclaration {
    try Gemma4MTPHeadDeclaration.parse(
        data: Data(json.utf8), origin: "test-dflash", kind: .dflash)
}

private func parseMTP(_ json: String) throws -> Gemma4MTPHeadDeclaration {
    try Gemma4MTPHeadDeclaration.parse(
        data: Data(json.utf8), origin: "test-mtp", kind: .mtp)
}

// --- The vocabulary ---------------------------------------------------------

@Test
func declaredArmVocabularyIsTheTwoScoredArms() {
    #expect(DeclaredArm.allCases.map(\.rawValue) == ["mtp", "dflash"])
    #expect(DeclaredArm.absentDefault == .mtp)
    #expect(DeclaredArm.declaringHeadKind == .dflash)
    #expect(
        DeclaredArm.declaringHeadKind.manifestRelativePath
            == "spec-decoder-head.manifest.json")
    #expect(Gemma4MTPHeadDeclaration.armKey == "arm")
}

// --- Absence is the default, and the ONLY silent path -----------------------

@Test
func absentArmKeySelectsTheDefaultArm() throws {
    let decl = try parseDFlash(#"{"source":"pinned"}"#)
    #expect(decl.arm == .mtp)
    #expect(decl.arm == DeclaredArm.absentDefault)
}

@Test
func absentManifestSelectsTheDefaultArm() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("declared-arm-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let decl = try Gemma4MTPHeadDeclaration.resolve(contractRoot: root, kind: .dflash)
    #expect(decl.arm == .mtp)
    #expect(decl.source == .pinned)
}

// --- The two declared values ------------------------------------------------

@Test
func declaringMTPSelectsMTP() throws {
    // The load-bearing NEGATIVE control's parse half: an explicit "mtp" must
    // land on `.mtp` and not be conflated with "the key was absent". A reader
    // that resolved every value to the default would pass this and fail the
    // next test -- which is why both are here.
    let decl = try parseDFlash(#"{"source":"pinned","arm":"mtp"}"#)
    #expect(decl.arm == .mtp)
}

@Test
func declaringDFlashSelectsDFlash() throws {
    let decl = try parseDFlash(#"{"source":"pinned","arm":"dflash"}"#)
    #expect(decl.arm == .dflash)
}

@Test
func theArmDoesNotDisturbTheRestOfTheDeclaration() throws {
    let withArm = try parseDFlash(
        #"{"source":"pinned","arm":"dflash","bytes":1024,"max_bytes":2048,"sha256":"AB"}"#)
    let withoutArm = try parseDFlash(
        #"{"source":"pinned","bytes":1024,"max_bytes":2048,"sha256":"AB"}"#)
    #expect(withArm.source == withoutArm.source)
    #expect(withArm.bytes == withoutArm.bytes)
    #expect(withArm.maxBytes == withoutArm.maxBytes)
    #expect(withArm.sha256 == withoutArm.sha256)
    #expect(withArm.kind == withoutArm.kind)
    // ...and the arm is the ONLY thing that differs.
    #expect(withArm.arm == .dflash)
    #expect(withoutArm.arm == .mtp)
}

// --- The refusals -----------------------------------------------------------

@Test
func unknownArmIsRefusedByName() {
    #expect(throws: MLXFastError.self) {
        _ = try parseDFlash(#"{"source":"pinned","arm":"dspark"}"#)
    }
    do {
        _ = try parseDFlash(#"{"source":"pinned","arm":"dspark"}"#)
        Issue.record("an unknown arm must refuse, never fall back to the default")
    } catch {
        let text = "\(error)"
        // The participant is entitled to know WHICH value was rejected and
        // WHAT the alternatives are; "invalid declaration" is not enough to
        // act on.
        #expect(text.contains("dspark"))
        #expect(text.contains("mtp"))
        #expect(text.contains("dflash"))
    }
}

@Test
func emptyArmIsRefused() {
    // `""` is the shape a hand-edited manifest most easily produces, and the
    // one an `as? String ?? default` read would silently swallow.
    #expect(throws: MLXFastError.self) {
        _ = try parseDFlash(#"{"source":"pinned","arm":""}"#)
    }
}

@Test
func armVocabularyIsCaseSensitive() {
    // benchd's spec modes are lowercase. Normalizing "MTP" here would mean the
    // engine and the benchmarker disagreed about what a valid mode string is.
    #expect(throws: MLXFastError.self) {
        _ = try parseDFlash(#"{"source":"pinned","arm":"MTP"}"#)
    }
    #expect(throws: MLXFastError.self) {
        _ = try parseDFlash(#"{"source":"pinned","arm":"DFlash"}"#)
    }
}

@Test
func nonStringArmIsRefused() {
    for value in ["1", "true", "null", "[\"dflash\"]", "{\"mode\":\"dflash\"}"] {
        #expect(throws: MLXFastError.self) {
            _ = try parseDFlash(#"{"source":"pinned","arm":\#(value)}"#)
        }
    }
}

@Test
func armOnTheMTPManifestIsRefused() {
    // The mistyped-FILE tolerance. A participant who writes the arm into
    // mtp-head.manifest.json would otherwise be scored on the other arm and
    // never told, because nothing reads that file's `arm`.
    #expect(throws: MLXFastError.self) {
        _ = try parseMTP(#"{"source":"pinned","arm":"dflash"}"#)
    }
    // Even the HARMLESS-looking value refuses: the defect is the location, not
    // the value, and a refusal that fired only for "dflash" would leave the
    // participant's real mistake in place.
    #expect(throws: MLXFastError.self) {
        _ = try parseMTP(#"{"source":"pinned","arm":"mtp"}"#)
    }
    do {
        _ = try parseMTP(#"{"source":"pinned","arm":"dflash"}"#)
        Issue.record("an arm on the MTP manifest must refuse")
    } catch {
        #expect("\(error)".contains("spec-decoder-head.manifest.json"))
    }
    // ...and the MTP manifest WITHOUT an arm is untouched by the rule.
    #expect(throws: Never.self) {
        _ = try parseMTP(#"{"source":"pinned"}"#)
    }
}
