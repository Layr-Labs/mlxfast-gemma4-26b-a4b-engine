import Foundation
import MLXFastCore

// Serial reference tape recorder — the ARM-NEUTRAL producer of the
// "timed-prompt tape" documents the ranked track's `timed_prompt_pool`
// pins (benchd `bench-core/src/tape.rs`, `TimedPromptTapeDocument`).
//
// Lineage: the pool objects were historically produced by the Qwen-era
// `mtp-verify --generate` pass, and the DFlash track had its own
// `dflash-reference` producer. Neither is runnable on this engine — the MTP
// and DFlash worker verbs left with the OQ-3 adoption / harness port
// (docs/gemma4-port-notes.md OQ-3, 8.3). This recorder replaces them with a
// producer that carries NO arm-specific logic at all: both of its backends
// are pure serial (no drafter, no head, no spec), so the tape it emits is
// what every speculation arm is verified against rather than the product of
// any one of them. It produces a reference ARTIFACT; it measures and scores
// nothing — timing stays benchd's.
//
// TWO RECORDING BACKENDS (`ReferenceTapeRecordingBackend`), because the
// engine has two serial decode implementations and the oracle must be
// recorded on the one it gates (port-notes 5.1, within-backend):
//   * cbv2 (default) — the width-1 CBv2 engine session both post-#29 v1.1
//     free-run legs run, via the trusted-CLI-only recording verbs
//     (`record_reference_begin` / `record_reference_run`); worker-side the
//     verbs reuse the SAME begin executor the wire legs dispatch to.
//   * legacy — the teacher-forced correctness verbs (`correctness_begin` /
//     `correctness_step`, the legacy single-token loop); retained for
//     provenance of the legacy-recorded pool tapes and for the
//     teacher-forced decode verbs that deliberately keep this loop.

/// Which backend PRODUCES the tape rows.
///
/// * `.cbv2` (DEFAULT) — the width-1 CBv2 engine session the post-#29 v1.1
///   free-run legs run (`openSingleStreamFreeRunSession` worker-side), in
///   its pure serial configuration: no drafter, no head, no spec. This is
///   the correct backend for this track's oracle: benchd's paired
///   FreeRunV1_1 measurement oracle-checks BOTH legs' CBv2-committed
///   streams against the pinned tape, and the track's own within-backend
///   rule (docs/gemma4-port-notes.md 5.1) requires the reference to be
///   computed by the implementation it gates — the legacy-recorded pool
///   tapes are the two-implementations-one-gate defect the leg-identity fix
///   surfaced.
/// * `.legacy` — the teacher-forced correctness verbs (`correctness_begin`
///   / `correctness_step`, the legacy `model.newCache` single-token loop).
///   Retained for provenance: reproducing/attributing the existing pool
///   tapes, and the teacher-forced decode verbs benchd's TF mode still
///   runs. Not the backend any free-run leg executes.
///
/// The DOCUMENT is byte-format identical either way (same
/// TimedPromptTapeDocument shape, same strict validation); only the
/// producing backend — and therefore near-tie row content, and the top-2
/// value scale (raw logits legacy-side, raw log-softmax logprobs
/// engine-side; per-row margins are identical because log_softmax shifts
/// both entries by the same normalizer) — changes. Backend provenance
/// belongs in the tape-generation session evidence (stderr logs), never in
/// the document.
public enum ReferenceTapeRecordingBackend: String, CaseIterable, Sendable {
    case cbv2
    case legacy

    /// The default backend for this track's oracle.
    public static let standard = ReferenceTapeRecordingBackend.cbv2

    /// Parse a `--recording-backend` CLI value. Fail-closed on anything but
    /// the two spelled-out backends.
    public init(cliValue: String) throws {
        guard let parsed = ReferenceTapeRecordingBackend(rawValue: cliValue)
        else {
            throw MLXFastError.invalidInput(
                "--recording-backend must be one of "
                    + "\(ReferenceTapeRecordingBackend.allCases.map(\.rawValue).joined(separator: "|")); "
                    + "got '\(cliValue)'"
            )
        }
        self = parsed
    }
}

/// Inputs for one recording run. `seedTokens` is the full seed (the caller
/// applies the 1024-token seed contract via
/// ``Gemma4Runtime/referenceTapeSeed(fromEncodedPrompt:)``); `steps` is the
/// number of tape ROWS to record (the decode chain after the seed argmax).
public struct ReferenceTapeRecordingOptions {
    public let weightsPath: String
    public let seedTokens: [Int]
    public let steps: Int

    public init(weightsPath: String, seedTokens: [Int], steps: Int) {
        self.weightsPath = weightsPath
        self.seedTokens = seedTokens
        self.steps = steps
    }
}

/// One tape row — format-exact to benchd's `TimedPromptTapeRow`: exactly
/// these four fields, no `name`, no extra keys (the benchd loader is
/// `deny_unknown_fields` at both levels, so an extra field is a refusal at
/// measure time, not a tolerated decoration).
public struct ReferenceTapeRow: Codable, Equatable {
    /// The token the serial reference emitted for this row.
    public let sequentialArgmax: Int
    /// Top-2 token ids at this row's forward, argmax first.
    public let top2Tokens: [Int]
    /// The logits behind `top2Tokens`, same order.
    public let top2Logits: [Double]
    /// `top2Logits[0]`, carried separately because the wire schema does.
    public let top1Logit: Double

    public init(
        sequentialArgmax: Int,
        top2Tokens: [Int],
        top2Logits: [Double],
        top1Logit: Double
    ) {
        self.sequentialArgmax = sequentialArgmax
        self.top2Tokens = top2Tokens
        self.top2Logits = top2Logits
        self.top1Logit = top1Logit
    }

    enum CodingKeys: String, CodingKey {
        case sequentialArgmax = "sequential_argmax"
        case top2Tokens = "top2_tokens"
        case top2Logits = "top2_logits"
        case top1Logit = "top1_logit"
    }
}

/// The whole tape — format-exact to benchd's `TimedPromptTapeDocument`:
/// exactly these five top-level keys. `rows[i].sequentialArgmax` is the token
/// emitted at index `i + 1` (index 0 is `referenceSeedToken`), and
/// `emittedTokens` IS the row argmax chain — benchd refuses a tape where the
/// two disagree.
public struct ReferenceTapeDocument: Codable, Equatable {
    public let seedTokens: [Int]
    public let referenceSeedToken: Int
    public let rows: [ReferenceTapeRow]
    /// Always `true` on a written tape: the recorder throws on a failed
    /// replay instead of ever writing `false` (benchd refuses an explicit
    /// `false` as an operator fault, and a false flag on a tape that was
    /// never replayed would be worse — see the replay pass below).
    public let referenceSelfConsistent: Bool
    public let emittedTokens: [Int]

    public init(
        seedTokens: [Int],
        referenceSeedToken: Int,
        rows: [ReferenceTapeRow],
        referenceSelfConsistent: Bool,
        emittedTokens: [Int]
    ) {
        self.seedTokens = seedTokens
        self.referenceSeedToken = referenceSeedToken
        self.rows = rows
        self.referenceSelfConsistent = referenceSelfConsistent
        self.emittedTokens = emittedTokens
    }

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case referenceSeedToken = "reference_seed_token"
        case rows
        case referenceSelfConsistent = "reference_self_consistent"
        case emittedTokens = "emitted_tokens"
    }
}

/// What one serial decode step reads out of the worker.
struct ReferenceTapeStepReadout: Equatable {
    let token: Int
    let top2Tokens: [Int]
    let top2Logits: [Double]
    let top1Logit: Double
}

/// The recorder's forward seam. `beginForward` starts a FRESH teacher-forced
/// pass (new cache, full seed prefill) and returns the seed argmax;
/// `stepForward` advances that pass by one teacher-forced token. Production
/// binds this to the runtime worker's existing serial correctness verbs; the
/// tests bind scripted sessions so the record/replay logic is provable
/// without a model.
protocol ReferenceTapeForwardSession {
    func beginForward(seedTokens: [Int]) throws -> Int
    func stepForward(previousToken: Int) throws -> ReferenceTapeStepReadout
}

/// The production session: the EXISTING `correctness_begin` /
/// `correctness_step` worker verbs, nothing else. Zero worker-side changes —
/// the per-step top-2 readout is the same `top_logits` diagnostic the
/// correctness trace path already requests.
struct RuntimeWorkerReferenceTapeSession: ReferenceTapeForwardSession {
    let client: RuntimeWorkerClient

    func beginForward(seedTokens: [Int]) throws -> Int {
        let response = try client.beginTeacherForcedCorrectness(
            promptTokens: seedTokens
        )
        guard let token = response.token else {
            throw MLXFastError.invalidInput(
                "runtime worker reference-tape seed forward returned no token"
            )
        }
        try requireReferenceTapeVocabToken(token, field: "seed argmax")
        return token
    }

    func stepForward(previousToken: Int) throws -> ReferenceTapeStepReadout {
        // The worker accepts trace diagnostics only as a topK+expectedToken
        // pair; the recorder has no golden expectation, so it passes the
        // step's own input token and ignores the expected-token fields.
        let response = try client.teacherForcedCorrectnessStep(
            previousToken: previousToken,
            topK: 2,
            expectedToken: previousToken
        )
        guard let token = response.token else {
            throw MLXFastError.invalidInput(
                "runtime worker reference-tape step returned no token"
            )
        }
        let topLogits = try Gemma4Runtime.validatedWorkerTopLogits(
            response.topLogits,
            actualToken: token,
            maximumCount: 2
        )
        guard topLogits.count == 2 else {
            throw MLXFastError.invalidInput(
                "runtime worker returned \(topLogits.count) top logits; "
                    + "a reference tape row needs the top 2"
            )
        }
        return ReferenceTapeStepReadout(
            token: token,
            top2Tokens: topLogits.map(\.token),
            top2Logits: topLogits.map(\.logit),
            top1Logit: topLogits[0].logit
        )
    }
}

/// The CBv2 recorder's forward seam: one PASS = one fresh width-1 engine
/// session (full seed prefill, then a free-run drain to N rows). Unlike the
/// step-wise legacy seam, the engine commits its own greedy chain — feeding
/// each step its previous argmax is what the engine's free run IS — so the
/// seam is pass-shaped, not step-shaped. Production binds the trusted worker
/// client's `record_reference_begin` / `record_reference_run` verbs (the
/// shared post-#29 begin executor in-process on the worker); the tests bind
/// scripted sessions so the record/replay logic is provable without a model.
protocol ReferenceTapeEnginePassSession {
    /// Open a fresh engine pass over the seed; returns the seed argmax.
    func beginEnginePass(seedTokens: [Int]) throws -> Int
    /// Free-run the open pass to exactly `rows` rows; returns one readout
    /// per row (argmax + top-2 diagnostics), in emitted order.
    func runEnginePass(rows: Int) throws -> [ReferenceTapeStepReadout]
}

/// The production CBv2 pass session: the trusted worker client's recording
/// verbs, nothing else. Shape validation happens HERE (counts, per-row
/// alignment, top-2 arity); the semantic tape invariants are re-enforced by
/// the strict document validation before any byte lands on the output path.
struct RuntimeWorkerEngineReferenceTapeSession: ReferenceTapeEnginePassSession {
    let client: RuntimeWorkerClient

    func beginEnginePass(seedTokens: [Int]) throws -> Int {
        let response = try client.beginRecordReference(seedTokens: seedTokens)
        guard let token = response.seedToken else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_begin returned no seed token"
            )
        }
        try requireReferenceTapeVocabToken(token, field: "seed argmax")
        return token
    }

    func runEnginePass(rows: Int) throws -> [ReferenceTapeStepReadout] {
        let response = try client.runRecordReference(rowCount: rows)
        guard let tokens = response.tokens, tokens.count == rows else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_run returned "
                    + "\(response.tokens?.count ?? 0) tokens; expected \(rows)"
            )
        }
        guard let top2Tokens = response.perRowTop2Tokens,
              let top2Logits = response.perRowTop2Logits,
              top2Tokens.count == rows,
              top2Logits.count == rows
        else {
            throw MLXFastError.invalidInput(
                "runtime worker record_reference_run returned "
                    + "\(response.perRowTop2Tokens?.count ?? 0)/"
                    + "\(response.perRowTop2Logits?.count ?? 0) per-row top-2 "
                    + "readouts; expected \(rows) of each"
            )
        }
        return try (0..<rows).map { index in
            guard top2Tokens[index].count == 2,
                  top2Logits[index].count == 2
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker record_reference_run rows[\(index)] top-2 "
                        + "arrays must carry exactly 2 entries"
                )
            }
            return ReferenceTapeStepReadout(
                token: tokens[index],
                top2Tokens: top2Tokens[index],
                top2Logits: top2Logits[index],
                top1Logit: top2Logits[index][0]
            )
        }
    }
}

func requireReferenceTapeVocabToken(_ token: Int, field: String) throws {
    guard token >= 0, token < MLXFastConstants.vocabSize else {
        throw MLXFastError.invalidInput(
            "reference tape \(field) token \(token) is outside vocab "
                + "0..<\(MLXFastConstants.vocabSize)"
        )
    }
}

extension Gemma4Runtime {
    /// The seed contract shared with `generate-golden` /
    /// `attach-free-run-gate`: the encoded prompt must reach
    /// `correctnessPromptTokens` (1024) and exactly that prefix is the seed.
    public static func referenceTapeSeed(
        fromEncodedPrompt encoded: [Int]
    ) throws -> [Int] {
        let required = MLXFastConstants.correctnessPromptTokens
        guard encoded.count >= required else {
            throw MLXFastError.invalidInput(
                "--prompt-file tokenized to \(encoded.count) tokens; "
                    + "reference tapes need at least \(required)"
            )
        }
        return Array(encoded.prefix(required))
    }

    /// Record a tape against the participant runtime worker. Serial path in
    /// BOTH backends: no drafter loaded, no head, no speculative verb
    /// touched. The backend selects the PRODUCING implementation only —
    /// `.cbv2` (default) drives the shared post-#29 width-1 CBv2 free-run
    /// session, `.legacy` the teacher-forced correctness verbs — the
    /// document contract is identical.
    public static func recordReferenceTape(
        _ options: ReferenceTapeRecordingOptions,
        worker workerOptions: RuntimeWorkerOptions?,
        backend: ReferenceTapeRecordingBackend = .standard,
        progress: ((String) -> Void)? = nil
    ) throws -> ReferenceTapeDocument {
        guard let workerOptions else {
            throw MLXFastError.invalidInput(
                "record-reference-tape requires the participant runtime worker"
            )
        }
        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.weightsPath
        )
        defer {
            client.close()
        }
        switch backend {
        case .cbv2:
            return try recordReferenceTapeDocumentEnginePass(
                seedTokens: options.seedTokens,
                steps: options.steps,
                session: RuntimeWorkerEngineReferenceTapeSession(client: client),
                progress: progress
            )
        case .legacy:
            return try recordReferenceTapeDocument(
                seedTokens: options.seedTokens,
                steps: options.steps,
                session: RuntimeWorkerReferenceTapeSession(client: client),
                progress: progress
            )
        }
    }

    /// The recording itself: one generation pass, then an ACTUAL replay pass.
    ///
    /// Generation is greedy decode expressed as teacher forcing on the
    /// recorder's own chain: each step is fed the previous step's argmax, so
    /// the recorded rows ARE the serial reference trajectory. The replay then
    /// starts a second fresh teacher-forced pass (new cache, full seed
    /// re-prefill) and feeds the RECORDED chain, verifying the seed argmax
    /// and every row argmax reproduce. Only a run whose every row survives
    /// that replay may write `reference_self_consistent: true`; any mismatch
    /// is a thrown error naming the position — the recorder never writes a
    /// tape carrying `false`, because a nondeterministic reference is an
    /// operator fault that must stop the pipeline, not a value to record.
    static func recordReferenceTapeDocument(
        seedTokens: [Int],
        steps: Int,
        session: ReferenceTapeForwardSession,
        progress: ((String) -> Void)? = nil
    ) throws -> ReferenceTapeDocument {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "reference tape seed_tokens must not be empty"
            )
        }
        guard steps > 0 else {
            throw MLXFastError.invalidInput(
                "reference tape steps must be positive"
            )
        }
        for (index, token) in seedTokens.enumerated() {
            try requireReferenceTapeVocabToken(
                token,
                field: "seed_tokens[\(index)]"
            )
        }

        let seedToken = try session.beginForward(seedTokens: seedTokens)
        progress?("seed forward complete reference_seed_token=\(seedToken)")
        var rows: [ReferenceTapeRow] = []
        rows.reserveCapacity(steps)
        var previousToken = seedToken
        for step in 0..<steps {
            let readout = try session.stepForward(previousToken: previousToken)
            try requireReferenceTapeVocabToken(
                readout.token,
                field: "rows[\(step)].sequential_argmax"
            )
            rows.append(
                ReferenceTapeRow(
                    sequentialArgmax: readout.token,
                    top2Tokens: readout.top2Tokens,
                    top2Logits: readout.top2Logits,
                    top1Logit: readout.top1Logit
                )
            )
            previousToken = readout.token
            if (step + 1) % 16 == 0 || step + 1 == steps {
                progress?("recorded \(step + 1)/\(steps) rows")
            }
        }

        progress?("replaying \(rows.count) rows teacher-forced")
        let replaySeedToken = try session.beginForward(seedTokens: seedTokens)
        guard replaySeedToken == seedToken else {
            throw MLXFastError.invalidInput(
                "reference tape failed its self-consistency replay at the "
                    + "seed forward: recorded argmax \(seedToken), replay "
                    + "produced \(replaySeedToken); the reference build is "
                    + "nondeterministic and this tape must not be written"
            )
        }
        var replayInput = seedToken
        for (index, row) in rows.enumerated() {
            let readout = try session.stepForward(previousToken: replayInput)
            guard readout.token == row.sequentialArgmax else {
                throw MLXFastError.invalidInput(
                    "reference tape failed its self-consistency replay at "
                        + "row \(index): recorded argmax "
                        + "\(row.sequentialArgmax), replay produced "
                        + "\(readout.token); the reference build is "
                        + "nondeterministic and this tape must not be written"
                )
            }
            replayInput = row.sequentialArgmax
            if (index + 1) % 16 == 0 || index + 1 == rows.count {
                progress?("replayed \(index + 1)/\(rows.count) rows")
            }
        }

        return ReferenceTapeDocument(
            seedTokens: seedTokens,
            referenceSeedToken: seedToken,
            rows: rows,
            // Set ONLY here, after the full replay above proved every row.
            referenceSelfConsistent: true,
            emittedTokens: rows.map(\.sequentialArgmax)
        )
    }

    /// The CBv2 recording: one engine PASS, then an ACTUAL second engine
    /// pass as the self-consistency replay.
    ///
    /// Generation: the width-1 CBv2 session free-runs greedily — each step's
    /// input IS its own previous argmax, so the recorded rows ARE the serial
    /// reference trajectory of the engine implementation the measuring legs
    /// run. The replay then opens a SECOND fresh session (new engine, full
    /// seed re-prefill) and verifies the seed argmax and EVERY row argmax
    /// reproduce, in order. This is teacher-forced verification expressed
    /// through the engine's own free run: at each compared row the replay's
    /// input prefix equals the recorded chain (every earlier row was just
    /// proven equal, and the first inequality throws naming its position),
    /// so each comparison checks exactly "given the recorded prefix, the
    /// engine's argmax equals the recorded argmax" — the same predicate the
    /// legacy replay checks by feeding the chain explicitly, computed by the
    /// SAME CBv2 executor rather than the legacy verbs. It is also the 5.4
    /// A≡B determinism tripwire run as part of every recording. Only a run
    /// whose every row survives may write `reference_self_consistent: true`;
    /// any mismatch is a thrown error naming the position — the recorder
    /// never writes a tape carrying `false`.
    static func recordReferenceTapeDocumentEnginePass(
        seedTokens: [Int],
        steps: Int,
        session: ReferenceTapeEnginePassSession,
        progress: ((String) -> Void)? = nil
    ) throws -> ReferenceTapeDocument {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "reference tape seed_tokens must not be empty"
            )
        }
        guard steps > 0 else {
            throw MLXFastError.invalidInput(
                "reference tape steps must be positive"
            )
        }
        for (index, token) in seedTokens.enumerated() {
            try requireReferenceTapeVocabToken(
                token,
                field: "seed_tokens[\(index)]"
            )
        }

        progress?(
            "recording backend: cbv2 — width-1 CBv2 free-run engine session "
                + "(the post-#29 leg executor), serial drafter-less "
                + "configuration")
        let seedToken = try session.beginEnginePass(seedTokens: seedTokens)
        progress?("seed forward complete reference_seed_token=\(seedToken)")
        let readouts = try session.runEnginePass(rows: steps)
        guard readouts.count == steps else {
            throw MLXFastError.invalidInput(
                "reference tape engine pass produced \(readouts.count) rows; "
                    + "expected \(steps)"
            )
        }
        var rows: [ReferenceTapeRow] = []
        rows.reserveCapacity(steps)
        for (step, readout) in readouts.enumerated() {
            try requireReferenceTapeVocabToken(
                readout.token,
                field: "rows[\(step)].sequential_argmax"
            )
            rows.append(
                ReferenceTapeRow(
                    sequentialArgmax: readout.token,
                    top2Tokens: readout.top2Tokens,
                    top2Logits: readout.top2Logits,
                    top1Logit: readout.top1Logit
                )
            )
        }
        progress?("recorded \(rows.count)/\(steps) rows")

        progress?(
            "replaying \(rows.count) rows through a second fresh CBv2 "
                + "engine pass (same executor)")
        let replaySeedToken = try session.beginEnginePass(seedTokens: seedTokens)
        guard replaySeedToken == seedToken else {
            throw MLXFastError.invalidInput(
                "reference tape failed its self-consistency replay at the "
                    + "seed forward: recorded argmax \(seedToken), replay "
                    + "produced \(replaySeedToken); the reference build is "
                    + "nondeterministic and this tape must not be written"
            )
        }
        let replayReadouts = try session.runEnginePass(rows: steps)
        guard replayReadouts.count == steps else {
            throw MLXFastError.invalidInput(
                "reference tape replay pass produced \(replayReadouts.count) "
                    + "rows; expected \(steps)"
            )
        }
        for (index, replay) in replayReadouts.enumerated() {
            guard replay.token == rows[index].sequentialArgmax else {
                throw MLXFastError.invalidInput(
                    "reference tape failed its self-consistency replay at "
                        + "row \(index): recorded argmax "
                        + "\(rows[index].sequentialArgmax), replay produced "
                        + "\(replay.token); the reference build is "
                        + "nondeterministic and this tape must not be written"
                )
            }
        }
        progress?("replayed \(rows.count)/\(rows.count) rows")

        return ReferenceTapeDocument(
            seedTokens: seedTokens,
            referenceSeedToken: seedToken,
            rows: rows,
            // Set ONLY here, after the full replay above proved every row.
            referenceSelfConsistent: true,
            emittedTokens: rows.map(\.sequentialArgmax)
        )
    }

    /// Canonical serialization for a tape (repo-idiom encoder settings).
    /// Pool pins bind sha256 + byte count of the EXACT emitted bytes, so
    /// callers must pin what this returns, unmodified.
    public static func encodeReferenceTapeDocument(
        _ document: ReferenceTapeDocument
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [
            .prettyPrinted, .sortedKeys, .withoutEscapingSlashes,
        ]
        return try encoder.encode(document)
    }

    /// Strict re-validation of tape BYTES before they may land on the output
    /// path — the local mirror of the benchd loader this document must
    /// survive (`deny_unknown_fields` at both levels plus its semantic
    /// checks), so a malformed emit fails HERE, not at measure time on the
    /// box.
    public static func validateReferenceTapeDocumentData(
        _ data: Data,
        requiredRows: Int
    ) throws {
        let topLevelKeys: Set<String> = [
            "seed_tokens", "reference_seed_token", "rows",
            "reference_self_consistent", "emitted_tokens",
        ]
        let rowKeys: Set<String> = [
            "sequential_argmax", "top2_tokens", "top2_logits", "top1_logit",
        ]
        guard
            let object = try JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "reference tape must serialize to a JSON object"
            )
        }
        guard Set(object.keys) == topLevelKeys else {
            throw MLXFastError.invalidInput(
                "reference tape top-level keys \(Set(object.keys).sorted()) "
                    + "must be exactly \(topLevelKeys.sorted())"
            )
        }
        guard let rawRows = object["rows"] as? [[String: Any]] else {
            throw MLXFastError.invalidInput(
                "reference tape rows must be an array of objects"
            )
        }
        for (index, row) in rawRows.enumerated() where Set(row.keys) != rowKeys {
            throw MLXFastError.invalidInput(
                "reference tape rows[\(index)] keys "
                    + "\(Set(row.keys).sorted()) must be exactly "
                    + "\(rowKeys.sorted())"
            )
        }

        let document = try JSONDecoder().decode(
            ReferenceTapeDocument.self,
            from: data
        )
        guard !document.seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "reference tape seed_tokens must not be empty"
            )
        }
        guard document.rows.count == requiredRows else {
            throw MLXFastError.invalidInput(
                "reference tape carries \(document.rows.count) rows; "
                    + "expected \(requiredRows)"
            )
        }
        guard document.referenceSelfConsistent else {
            throw MLXFastError.invalidInput(
                "reference tape must carry reference_self_consistent=true; "
                    + "a failed replay is a thrown error, never a written flag"
            )
        }
        guard document.emittedTokens == document.rows.map(\.sequentialArgmax)
        else {
            throw MLXFastError.invalidInput(
                "reference tape emitted_tokens must equal the row argmax chain"
            )
        }
        for (index, token) in document.seedTokens.enumerated() {
            try requireReferenceTapeVocabToken(
                token,
                field: "seed_tokens[\(index)]"
            )
        }
        try requireReferenceTapeVocabToken(
            document.referenceSeedToken,
            field: "reference_seed_token"
        )
        for (index, row) in document.rows.enumerated() {
            try requireReferenceTapeVocabToken(
                row.sequentialArgmax,
                field: "rows[\(index)].sequential_argmax"
            )
            guard row.top2Tokens.count == 2, row.top2Logits.count == 2 else {
                throw MLXFastError.invalidInput(
                    "reference tape rows[\(index)] top-2 arrays must carry "
                        + "exactly 2 entries"
                )
            }
            for token in row.top2Tokens {
                try requireReferenceTapeVocabToken(
                    token,
                    field: "rows[\(index)].top2_tokens"
                )
            }
            guard row.top2Logits.allSatisfy(\.isFinite),
                  row.top1Logit.isFinite
            else {
                throw MLXFastError.invalidInput(
                    "reference tape rows[\(index)] logits must be finite"
                )
            }
            guard row.top2Tokens[0] == row.sequentialArgmax,
                  row.top1Logit == row.top2Logits[0],
                  row.top2Logits[0] >= row.top2Logits[1]
            else {
                throw MLXFastError.invalidInput(
                    "reference tape rows[\(index)] top-2 readout is "
                        + "internally inconsistent with its argmax"
                )
            }
        }
    }
}
