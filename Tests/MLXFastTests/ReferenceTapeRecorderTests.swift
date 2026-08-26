import CryptoKit
import Foundation
import Testing

import MLXFastCore
@testable import MLXFastHarness

// Unit coverage for the `record-reference-tape` producer (trusted harness,
// Gemma4RuntimeReferenceTape.swift): serialization exactness against benchd's
// TimedPromptTapeDocument shape, the self-consistency replay's fatal path,
// and the 1024-token seed gate. No model, no worker, no GPU: the recorder's
// forward seam is a protocol, and these tests bind scripted sessions.

/// Deterministic scripted forward session. Each `beginForward` starts a new
/// pass; steps are a pure function of the fed token, so a second pass over
/// the same chain reproduces — unless a divergence is scripted, which is the
/// nondeterministic-reference fault the replay must catch.
private final class ScriptedForwardSession: ReferenceTapeForwardSession {
    let seedArgmax: Int
    /// Seed argmax to return from the SECOND begin (the replay pass); nil
    /// means deterministic.
    let replaySeedArgmax: Int?
    /// Step index (within the replay pass) whose argmax flips; nil means
    /// deterministic.
    let replayDivergesAtStep: Int?

    private(set) var beginCount = 0
    private(set) var totalStepCount = 0
    private var stepsInCurrentPass = 0

    init(
        seedArgmax: Int,
        replaySeedArgmax: Int? = nil,
        replayDivergesAtStep: Int? = nil
    ) {
        self.seedArgmax = seedArgmax
        self.replaySeedArgmax = replaySeedArgmax
        self.replayDivergesAtStep = replayDivergesAtStep
    }

    func beginForward(seedTokens: [Int]) throws -> Int {
        beginCount += 1
        stepsInCurrentPass = 0
        if beginCount > 1, let replaySeedArgmax {
            return replaySeedArgmax
        }
        return seedArgmax
    }

    func stepForward(previousToken: Int) throws -> ReferenceTapeStepReadout {
        let stepIndex = stepsInCurrentPass
        stepsInCurrentPass += 1
        totalStepCount += 1
        var token = (previousToken &* 7 &+ 13) % 4_096
        if beginCount > 1, replayDivergesAtStep == stepIndex {
            token += 1
        }
        let runnerUp = token == 0 ? 1 : token - 1
        return ReferenceTapeStepReadout(
            token: token,
            top2Tokens: [token, runnerUp],
            top2Logits: [18.5, 17.25],
            top1Logit: 18.5
        )
    }
}

@Test
func recorderRecordsAndReplaysADeterministicSession() throws {
    let session = ScriptedForwardSession(seedArgmax: 111)
    let document = try Gemma4Runtime.recordReferenceTapeDocument(
        seedTokens: [5, 6, 7],
        steps: 4,
        session: session
    )

    #expect(document.seedTokens == [5, 6, 7])
    #expect(document.referenceSeedToken == 111)
    #expect(document.rows.count == 4)
    // The chain is teacher forcing on the recorder's own argmaxes.
    var expectedChain: [Int] = []
    var previous = 111
    for _ in 0..<4 {
        previous = (previous * 7 + 13) % 4_096
        expectedChain.append(previous)
    }
    #expect(document.rows.map(\.sequentialArgmax) == expectedChain)
    #expect(document.emittedTokens == expectedChain)
    #expect(document.referenceSelfConsistent)
    // The replay is an ACTUAL second pass, not a re-assertion: two begins
    // (fresh seed prefill each) and every row stepped twice.
    #expect(session.beginCount == 2)
    #expect(session.totalStepCount == 8)
}

@Test
func replayRowMismatchIsFatalAndNamesTheRow() {
    let session = ScriptedForwardSession(seedArgmax: 111, replayDivergesAtStep: 2)
    do {
        _ = try Gemma4Runtime.recordReferenceTapeDocument(
            seedTokens: [5, 6, 7],
            steps: 4,
            session: session
        )
        Issue.record("a diverging replay must throw, never return a document")
    } catch {
        let message = "\(error)"
        #expect(message.contains("self-consistency replay"))
        #expect(message.contains("row 2"))
    }
}

@Test
func replaySeedMismatchIsFatal() {
    let session = ScriptedForwardSession(seedArgmax: 111, replaySeedArgmax: 112)
    do {
        _ = try Gemma4Runtime.recordReferenceTapeDocument(
            seedTokens: [5, 6, 7],
            steps: 2,
            session: session
        )
        Issue.record("a diverging replay seed must throw, never return a document")
    } catch {
        let message = "\(error)"
        #expect(message.contains("self-consistency replay"))
        #expect(message.contains("seed forward"))
    }
}

@Test
func seedGateRequiresTheFullSeedAndPrefixesExactly() throws {
    let required = MLXFastConstants.correctnessPromptTokens
    // One token short: refused, naming the requirement.
    let short = Array(0..<(required - 1))
    do {
        _ = try Gemma4Runtime.referenceTapeSeed(fromEncodedPrompt: short)
        Issue.record("a short prompt must be refused")
    } catch {
        #expect("\(error)".contains("at least \(required)"))
    }
    // Longer than required: the seed is exactly the first `required` tokens.
    let long = Array(0..<(required + 37))
    let seed = try Gemma4Runtime.referenceTapeSeed(fromEncodedPrompt: long)
    #expect(seed.count == required)
    #expect(seed == Array(0..<required))
}

// The serialized document, pinned byte-for-byte. This is the shape benchd's
// TimedPromptTapeDocument loader (`deny_unknown_fields` at both levels)
// parses: exactly five top-level keys, exactly four row keys, nothing else —
// no `name`, no `declared_frame_argmax`, no extras. The encoder sorts keys,
// so the literal also pins the canonical key order.
@Test
func referenceTapeSerializesToTheExactPinnedShape() throws {
    let document = ReferenceTapeDocument(
        seedTokens: [1, 2],
        referenceSeedToken: 9,
        rows: [
            ReferenceTapeRow(
                sequentialArgmax: 11,
                top2Tokens: [11, 4],
                top2Logits: [18.5, 17.25],
                top1Logit: 18.5
            )
        ],
        referenceSelfConsistent: true,
        emittedTokens: [11]
    )
    let data = try Gemma4Runtime.encodeReferenceTapeDocument(document)
    let expected = """
    {
      "emitted_tokens" : [
        11
      ],
      "reference_seed_token" : 9,
      "reference_self_consistent" : true,
      "rows" : [
        {
          "sequential_argmax" : 11,
          "top1_logit" : 18.5,
          "top2_logits" : [
            18.5,
            17.25
          ],
          "top2_tokens" : [
            11,
            4
          ]
        }
      ],
      "seed_tokens" : [
        1,
        2
      ]
    }
    """
    #expect(String(decoding: data, as: UTF8.self) == expected)

    // Key-set exactness asserted structurally too, independent of formatting.
    let object = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(
        Set(object.keys) == [
            "seed_tokens", "reference_seed_token", "rows",
            "reference_self_consistent", "emitted_tokens",
        ]
    )
    let rows = try #require(object["rows"] as? [[String: Any]])
    #expect(
        Set(try #require(rows.first).keys) == [
            "sequential_argmax", "top2_tokens", "top2_logits", "top1_logit",
        ]
    )
}

private func recordedSampleData(steps: Int) throws -> Data {
    let session = ScriptedForwardSession(seedArgmax: 111)
    let document = try Gemma4Runtime.recordReferenceTapeDocument(
        seedTokens: Array(1_000..<1_016),
        steps: steps,
        session: session
    )
    return try Gemma4Runtime.encodeReferenceTapeDocument(document)
}

@Test
func strictValidatorAcceptsRecorderOutputAndRejectsDrift() throws {
    let data = try recordedSampleData(steps: 4)
    try Gemma4Runtime.validateReferenceTapeDocumentData(data, requiredRows: 4)

    func mutated(_ mutate: (inout [String: Any]) -> Void) throws -> Data {
        var object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        mutate(&object)
        return try JSONSerialization.data(withJSONObject: object)
    }
    func expectRejected(_ candidate: Data, requiredRows: Int = 4, naming needle: String) {
        do {
            try Gemma4Runtime.validateReferenceTapeDocumentData(
                candidate,
                requiredRows: requiredRows
            )
            Issue.record("validator must reject a tape whose defect is: \(needle)")
        } catch {
            #expect("\(error)".contains(needle), "\(error)")
        }
    }

    // Extra top-level key: exactly what benchd's deny_unknown_fields refuses.
    expectRejected(
        try mutated { $0["surprise_field"] = 1 },
        naming: "top-level keys"
    )
    // Extra per-row key (the Swift DFlash Row's optional extras must never
    // leak into this document).
    expectRejected(
        try mutated {
            var rows = $0["rows"] as? [[String: Any]] ?? []
            rows[0]["declared_frame_argmax"] = 11
            $0["rows"] = rows
        },
        naming: "rows[0] keys"
    )
    // A written false flag is forbidden outright — failure is a thrown error.
    expectRejected(
        try mutated { $0["reference_self_consistent"] = false },
        naming: "reference_self_consistent=true"
    )
    // The emitted chain must BE the row argmax chain.
    expectRejected(
        try mutated {
            var emitted = $0["emitted_tokens"] as? [Int] ?? []
            emitted[1] += 1
            $0["emitted_tokens"] = emitted
        },
        naming: "row argmax chain"
    )
    // Row-count mismatch against the requested window.
    expectRejected(data, requiredRows: 5, naming: "expected 5")
    // Out-of-vocab token.
    expectRejected(
        try mutated {
            var seed = $0["seed_tokens"] as? [Int] ?? []
            seed[0] = MLXFastConstants.vocabSize
            $0["seed_tokens"] = seed
        },
        naming: "outside vocab"
    )
}

// Emits a recorder-produced SYNTHETIC sample (scripted session, invented
// tokens — no organizer bytes, no real prompt content) for the cross-language
// round-trip check against benchd's real parser. Same operator pattern as
// emitEngineWireFixture: the path and sha are printed to stderr; feed the
// file to `bench_core::tape::load_timed_prompt_tape` (see the recorder
// runbook in docs/gemma4-port-notes.md).
@Test
func emitReferenceTapeRoundTripSample() throws {
    let data = try recordedSampleData(steps: 129)
    try Gemma4Runtime.validateReferenceTapeDocumentData(data, requiredRows: 129)
    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("reference-tape-roundtrip-sample.json")
    try data.write(to: outURL)
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    FileHandle.standardError.write(
        "EMIT_REFERENCE_TAPE_SAMPLE path=\(outURL.path) bytes=\(data.count) sha256=\(digest)\n"
            .data(using: .utf8)!
    )
}

// MARK: - CBv2 recording backend (pass-shaped seam)

/// Deterministic scripted ENGINE-PASS session (the CBv2 recorder's seam).
/// One begin == one fresh engine pass; the pass free-runs its own greedy
/// chain as a pure function of the seed argmax, so a second pass reproduces
/// — unless a divergence is scripted, which is the nondeterministic-reference
/// fault the replay must catch.
private final class ScriptedEnginePassSession: ReferenceTapeEnginePassSession {
    let seedArgmax: Int
    /// Seed argmax returned by the SECOND begin (the replay pass); nil means
    /// deterministic.
    let replaySeedArgmax: Int?
    /// Row index (within the replay pass) whose argmax flips; nil means
    /// deterministic.
    let replayDivergesAtRow: Int?

    private(set) var beginCount = 0
    private(set) var runCount = 0
    private(set) var totalRowsProduced = 0

    init(
        seedArgmax: Int,
        replaySeedArgmax: Int? = nil,
        replayDivergesAtRow: Int? = nil
    ) {
        self.seedArgmax = seedArgmax
        self.replaySeedArgmax = replaySeedArgmax
        self.replayDivergesAtRow = replayDivergesAtRow
    }

    func beginEnginePass(seedTokens: [Int]) throws -> Int {
        beginCount += 1
        if beginCount > 1, let replaySeedArgmax {
            return replaySeedArgmax
        }
        return seedArgmax
    }

    func runEnginePass(rows: Int) throws -> [ReferenceTapeStepReadout] {
        runCount += 1
        var previous = seedArgmax
        var readouts: [ReferenceTapeStepReadout] = []
        for row in 0..<rows {
            var token = (previous &* 7 &+ 13) % 4_096
            if beginCount > 1, replayDivergesAtRow == row {
                token += 1
            }
            let runnerUp = token == 0 ? 1 : token - 1
            // CBv2-scale diagnostics: raw log-softmax logprobs (<= 0), the
            // value scale the engine's topLogprobs readout reports.
            readouts.append(
                ReferenceTapeStepReadout(
                    token: token,
                    top2Tokens: [token, runnerUp],
                    top2Logits: [-0.125, -2.5],
                    top1Logit: -0.125
                )
            )
            previous = token
            totalRowsProduced += 1
        }
        return readouts
    }
}

@Test
func cbv2RecorderRecordsAndReplaysThroughTheEnginePassSeam() throws {
    let session = ScriptedEnginePassSession(seedArgmax: 111)
    let document = try Gemma4Runtime.recordReferenceTapeDocumentEnginePass(
        seedTokens: [5, 6, 7],
        steps: 4,
        session: session
    )

    #expect(document.seedTokens == [5, 6, 7])
    #expect(document.referenceSeedToken == 111)
    #expect(document.rows.count == 4)
    // The chain is the engine's own greedy free run.
    var expectedChain: [Int] = []
    var previous = 111
    for _ in 0..<4 {
        previous = (previous * 7 + 13) % 4_096
        expectedChain.append(previous)
    }
    #expect(document.rows.map(\.sequentialArgmax) == expectedChain)
    #expect(document.emittedTokens == expectedChain)
    #expect(document.referenceSelfConsistent)
    // Per-row top-2 diagnostics carried straight from the engine readout.
    for row in document.rows {
        #expect(row.top2Tokens[0] == row.sequentialArgmax)
        #expect(row.top2Logits == [-0.125, -2.5])
        #expect(row.top1Logit == -0.125)
    }
    // The replay is an ACTUAL second engine pass, not a re-assertion: two
    // begins (fresh seed prefill each) and every row produced twice.
    #expect(session.beginCount == 2)
    #expect(session.runCount == 2)
    #expect(session.totalRowsProduced == 8)

    // The emitted bytes survive the strict document validation with the
    // CBv2 (logprob-scale) top-2 values: the byte-format contract is
    // backend-neutral.
    let data = try Gemma4Runtime.encodeReferenceTapeDocument(document)
    try Gemma4Runtime.validateReferenceTapeDocumentData(data, requiredRows: 4)
}

@Test
func cbv2ReplayRowMismatchIsFatalAndNamesTheRow() {
    let session = ScriptedEnginePassSession(
        seedArgmax: 111, replayDivergesAtRow: 2)
    do {
        _ = try Gemma4Runtime.recordReferenceTapeDocumentEnginePass(
            seedTokens: [5, 6, 7],
            steps: 4,
            session: session
        )
        Issue.record("a diverging replay must throw, never return a document")
    } catch {
        let message = "\(error)"
        #expect(message.contains("self-consistency replay"))
        #expect(message.contains("row 2"))
    }
}

@Test
func cbv2ReplaySeedMismatchIsFatal() {
    let session = ScriptedEnginePassSession(
        seedArgmax: 111, replaySeedArgmax: 112)
    do {
        _ = try Gemma4Runtime.recordReferenceTapeDocumentEnginePass(
            seedTokens: [5, 6, 7],
            steps: 2,
            session: session
        )
        Issue.record("a diverging replay seed must throw, never return a document")
    } catch {
        let message = "\(error)"
        #expect(message.contains("self-consistency replay"))
        #expect(message.contains("seed forward"))
    }
}

@Test
func recordingBackendDefaultsToCBv2AndParsesFailClosed() throws {
    // cbv2 is the default backend for this track's oracle: the measuring
    // legs run the width-1 CBv2 engine, so the tape producer must too
    // (port-notes 5.1, within-backend).
    #expect(ReferenceTapeRecordingBackend.standard == .cbv2)
    #expect(try ReferenceTapeRecordingBackend(cliValue: "cbv2") == .cbv2)
    #expect(try ReferenceTapeRecordingBackend(cliValue: "legacy") == .legacy)
    do {
        _ = try ReferenceTapeRecordingBackend(cliValue: "fast")
        Issue.record("an unknown backend must be refused")
    } catch {
        #expect("\(error)".contains("cbv2|legacy"))
    }
}

@Test
func legacyBackendBytesAreUnchangedOnTheSyntheticFixture() throws {
    // The legacy path (teacher-forced correctness verbs) is retained for
    // provenance and must reproduce the pre-selector recorder byte-for-byte
    // on the same scripted forwards. The sha pins the exact emitted bytes of
    // the deterministic synthetic fixture recorded through the LEGACY
    // step-shaped seam; a drift here means the legacy mode changed.
    let data = try recordedSampleData(steps: 4)
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }
        .joined()
    #expect(
        digest
            == "a56a95c77703fec8e4ea56c123b168c6dfe87a9e03799f2425dd42739de7167e"
    )
}
