import Foundation
import MLXFastCore
@testable import MLXFastHarness
import Testing

// REQUANT-ONLY (David ruling, 2026-08-26): `pinned` is the ONLY source this
// track accepts. The two speculative heads are the organizer's own weights. A
// participant may declare a re-quantization of them and may not substitute
// weights of their own.
//
// WHAT THIS SUITE USED TO SAY, AND WHY IT FLIPPED. Under DECIDE-2 Q-B a
// bring-your-own head was gated by SIZE ONLY: `in_branch` and `remote` were
// accepted flows, a digestless head was fine, and the byte cap was the whole
// gate. Every one of those accept-cases is now a REFUSAL case, because the
// mechanism they exercised is retired. The tests are flipped rather than
// deleted: an accept-case turned refusal-case is the assertion that goes red if
// someone restores the source, where a deleted test is just silence.
//
// TWO LAYERS, ONE RULING. `benchmark.json` drops `mtp-head/` and `dflash-head/`
// from `editablePaths`, so head weights cannot RIDE IN a submission; this parser
// refuses the declarations that would ask the runner to GO AND GET some. Neither
// half is sufficient alone, so both land together.
//
// FIX-BAR / mutation to kill: restoring `case .remote` or `case .inBranch` to a
// non-throwing branch in `Gemma4MTPHeadDeclaration.parse` must turn
// `inBranchSourceIsRefused` and `remoteSourceIsRefused` RED.

private let validDigest = String(repeating: "a", count: 64)

// 64 uppercase hex; parse lowercases a declared digest.
private let mixedCaseDigest = String(repeating: "Ab", count: 32)

private func refusalMessage(_ json: String) -> String {
    do {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
        return "<accepted>"
    } catch let error as MLXFastError {
        return "\(error)"
    } catch {
        return "\(error)"
    }
}

// --- The retired sources are now named refusals -----------------------------

@Test
func inBranchSourceIsRefused() throws {
    // The bring-your-own-weights mode, in the exact shape that was ACCEPTED
    // before this ruling (safe path, positive bytes, under the 2 GiB cap).
    let json = #"{"source":"in_branch","path":"heads/mtp","bytes":1024}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
    let message = refusalMessage(json)
    #expect(message.contains("in_branch"), "the refusal must name the retired source: \(message)")
    #expect(message.contains("'pinned' only"), "the refusal must name what IS accepted: \(message)")
    #expect(message.contains("mtp-head/"), "the refusal must explain why: \(message)")
}

@Test
func remoteSourceIsRefused() throws {
    // The redirect mode. `source_url` was constrained to an `hf:`/`r2:` PREFIX
    // and nothing else -- no repository allowlist, no revision pin -- so any
    // acceptance here is an arbitrary-weights primitive the moment a fetcher
    // exists. It is refused before `source_url` is looked at.
    let json = #"{"source":"remote","source_url":"hf:acme/head@abc","bytes":2048}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
    let message = refusalMessage(json)
    #expect(message.contains("remote"), "the refusal must name the retired source: \(message)")
    #expect(message.contains("'pinned' only"), "the refusal must name what IS accepted: \(message)")
}

@Test
func aWellFormedRemoteRedirectIsStillRefused() throws {
    // THE NEGATIVE CONTROL THAT MATTERS. The old parser refused a remote
    // declaration only for a MALFORMED source_url. Prove the refusal is now
    // about the SOURCE and not about the URL's shape: a perfectly well-formed
    // `hf:` URL naming someone else's repository is refused just the same.
    let json =
        #"{"source":"remote","source_url":"hf:attacker/better-head@deadbeef","bytes":2048}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
    // And the organizer's OWN url is refused too: the rule is "pinned only",
    // not "pinned or a url we happen to like". This is the checked-in
    // declaration's former content.
    let organizerURL = "hf:mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit@bb94eae1"
    let organizerJSON =
        #"{"source":"remote","source_url":"\#(organizerURL)","bytes":247463936}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(
            data: Data(organizerJSON.utf8), origin: "test")
    }
}

@Test
func retiredSourcesAreRefusedForBothHeadKinds() throws {
    // One mechanism, two heads. A rule that binds the MTP head and not the
    // DFlash drafter is not a rule.
    for kind in ReplaceableHeadKind.allCases {
        for json in [
            #"{"source":"in_branch","path":"heads/x","bytes":1024}"#,
            #"{"source":"remote","source_url":"hf:acme/head@abc","bytes":1024}"#,
        ] {
            #expect(throws: MLXFastError.self) {
                _ = try Gemma4MTPHeadDeclaration.parse(
                    data: Data(json.utf8), origin: "test", kind: kind)
            }
        }
    }
}

@Test
func anUnknownSourceIsStillRefusedAsUnknown() throws {
    // The retired sources must not swallow the unknown-source diagnostic: a
    // typo is a different mistake from using a retired mode, and it should read
    // differently.
    let message = refusalMessage(#"{"source":"in-branch","path":"heads/mtp","bytes":1}"#)
    #expect(message.contains("unknown source"), "got: \(message)")
}

// --- Pinned declarations still parse, and the size gate still binds ----------

@Test
func pinnedSourceIsAccepted() throws {
    let json = #"{"source":"pinned"}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    #expect(decl.source == .pinned)
}

@Test
func pinnedDeclarationMayStateTheStagedSize() throws {
    // A re-quantized head is SMALLER than the shipped one, and stating the
    // expected size is how a participant records what the box should produce.
    // A stated size is accepted under the cap.
    let json = #"{"source":"pinned","bytes":123456789}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    #expect(decl.source == .pinned)
    #expect(decl.bytes == 123_456_789)
}

@Test
func statedSizeOverTheCapIsStillRefused() throws {
    // THE SIZE GATE SURVIVED THE NARROWING. Before this change the bytes check
    // was reachable only for a non-pinned source, so retiring those sources
    // would have made it dead code. It now binds on any STATED size.
    let overCap = Gemma4MTPHeadDeclaration.defaultMaxBytes + 1
    let json = #"{"source":"pinned","bytes":\#(overCap)}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
}

@Test
func statedSizeOverALoweredCapIsStillRefused() throws {
    // A declaration may LOWER the cap; a stated size above the lowered
    // max_bytes still refuses.
    let json = #"{"source":"pinned","max_bytes":1000,"bytes":1001}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
}

@Test
func maxBytesAboveTrackCapStillRefused() throws {
    // The max_bytes ceiling (may lower, may not raise) is unchanged.
    let raised = Gemma4MTPHeadDeclaration.defaultMaxBytes + 1
    let json = #"{"source":"pinned","max_bytes":\#(raised),"bytes":1}"#
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    }
}

// --- Declared-digest passthrough is UNCHANGED (regression) ------------------

@Test
func pinnedHeadWithDeclaredDigestUnchanged() throws {
    // A declared digest is still PARSED AND CARRIED, and still verified against
    // nothing. That is not an oversight this change fixes -- see
    // docs/participant-contract.md section 4 and the PR that landed this
    // ruling: nothing in the repository binds staged head bytes to the
    // organizer's pinned digests at run time.
    let json = #"{"source":"pinned","sha256":"\#(validDigest)"}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    #expect(decl.source == .pinned)
    #expect(decl.sha256 == validDigest)
}

@Test
func declaredDigestIsLowercased() throws {
    let json = #"{"source":"pinned","sha256":"\#(mixedCaseDigest)","bytes":1024}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    #expect(decl.sha256 == mixedCaseDigest.lowercased())
    #expect(decl.bytes == 1024)
}

@Test
func absentManifestStaysPinnedDefault() throws {
    // A source-less manifest still means pinned, still requires no digest.
    let json = #"{}"#
    let decl = try Gemma4MTPHeadDeclaration.parse(data: Data(json.utf8), origin: "test")
    #expect(decl == Gemma4MTPHeadDeclaration.pinnedDefault)
}

@Test
func theCheckedInMTPDeclarationParsesAndSelectsThePinnedHead() throws {
    // The shipped declaration must itself be legal under the rule it states.
    // It named `source: "remote"` until this ruling, which the parser now
    // refuses -- so this assertion is what catches a half-landed change.
    let decl = try Gemma4MTPHeadDeclaration.parse(
        contentsOf: URL(fileURLWithPath: "mtp-head.manifest.json"))
    #expect(decl.source == .pinned)
}
