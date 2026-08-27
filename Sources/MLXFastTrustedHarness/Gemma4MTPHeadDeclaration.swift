import Foundation
import MLXFastCore

// Head delivery for the Qwen 3.6 native-MTP track (qwen3.8-27b-mtp-v1).
//
// OPERATOR-RATIFIED 2026-08-14. The MTP head is part of the competitive
// surface: a submission may bring its own. It declares one in
// `mtp-head.manifest.json`, which is an editable path, and the RUNNER resolves
// that declaration pre-sandbox the way it resolves a hidden golden -- fetch and
// refuse on oversize. A declared digest is parsed and carried when present but
// is NOT verified against the head bytes and is not a gate (head_provenance
// records the harness's own recomputed tree digest, not the declared value).
// Per DECIDE-2 (Q-B) a bring-your-own head is NOT required to declare one, and a
// digestless head and a wrong-digest head are treated identically: both are
// bounded by SIZE ONLY.
//
// THE SAFETY ARGUMENT, in one line: a head only PROPOSES tokens. The
// organizer-pinned target decides every emitted token and the trusted parent
// re-checks the whole stream against a hidden serial trajectory after the clock
// stops, so a substituted head moves the accept rate -- which is the game --
// and cannot move the output.
//
// THIS FILE IS TRUSTED CODE. It parses the declaration and applies the size
// gate; it never decides whether a run passes.
//
// SCOPE AFTER THE GEMMA 4 HARNESS PORT (2026-08-22). The MTP arm itself is
// deferred to the follow-up increment, and the tree-digest helper that used to
// live here (`computeQwenMTPHeadProvenance`) went with it: its only consumer
// was the `mtp-verify` evidence payload. This half STAYS because it is neither
// Qwen-specific nor MLX-linked -- it is the implementation of the ratified
// DECIDE-2 (Q-B) ruling that a bring-your-own head is bounded by SIZE ONLY,
// and that ruling survives the change of target tower.

// BOTH REPLACEABLE HEADS, ONE MECHANISM (David ruling, 2026-08-25). The
// gemma4-26b-a4b-mlx-v1 track carries a SECOND replaceable head -- the z-lab
// DFlash drafter staged at `dflash-head/` -- and it is declared exactly the
// way the MTP head is: `spec-decoder-head.manifest.json`, same
// `source`/`sha256`/`bytes` shape, the SAME 2 GiB size-only cap, the same
// absent-means-organizer-default and present-but-broken-means-refusal
// posture. Rather than a second copy of this parser, the head KIND is a
// parameter: one gate, two manifests, no way for the two to drift apart.

// THE DECLARATION ALSO CARRIES THE ARM (2026-08-26). `dflash-head.manifest.
// json` gained an `arm` key, and this parser gained a KNOWN key to match. The
// parser used to read six keys and ignore every other one, which is the right
// posture for a key nothing consumes and the wrong one for a key that decides
// which arm is scored. `arm` is therefore parsed by `parseArm` and REFUSES on
// an unknown value, a non-string value, and presence on the manifest that does
// not declare the arm -- see `DeclaredArm`.
//
// The reader that ACTS on the arm is `tools/gemma4-measure-and-score.sh`: it is
// the trusted wrapper that composes the benchmarker's argv, so it is the only
// place a declaration can become a `--candidate-spec`. This file owns the
// VOCABULARY and the fail-closed parse; `tools/test-gemma4-arm-selection.sh`
// holds the two in step.

/// Which replaceable head a declaration is about. The kind decides only the
/// manifest filename, the staged directory, and how refusals name the head --
/// never the gate, which is byte-for-byte identical for both.
public enum ReplaceableHeadKind: String, Equatable, CaseIterable, Sendable {
    case mtp
    case dflash

    /// The declaration file, resolved against the contract root.
    public var manifestRelativePath: String {
        switch self {
        case .mtp: return "mtp-head.manifest.json"
        case .dflash: return "spec-decoder-head.manifest.json"
        }
    }

    /// The staged head directory -- an `editablePaths` entry held out of the
    /// code byte budget by `editableSurfaceByteBudget.exemptPaths` and charged
    /// against `exemptPathMaxBytes` (the same 2 GiB) instead.
    public var stagedDirectoryName: String {
        switch self {
        case .mtp: return "mtp-head"
        case .dflash: return "dflash-head"
        }
    }

    /// How a refusal names this head.
    public var declarationNoun: String {
        switch self {
        case .mtp: return "MTP"
        case .dflash: return "DFlash"
        }
    }
}

/// Kind-neutral spelling of the declaration type. The type is still named for
/// the head it was written for; this alias is what non-MTP call sites should
/// read as.
public typealias ReplaceableHeadDeclaration = Gemma4MTPHeadDeclaration

/// THE SELECTION CHANNEL (2026-08-26). Which speculative arm the submission
/// declares it wants MEASURED.
///
/// `benchmark.json`, `docs/participant-contract.md` section 5.1.1, `README.md`
/// and `TASK.md` all say "a submission runs whichever mode its own code
/// drives". Until this type existed there was no way for a submission to say
/// which one: the mode reached the benchmarker only through an operator's
/// `--candidate-spec` / `--mtp-depth` flag, and `tools/gemma4-measure-and-
/// score.sh` -- the ONLY thing `benchmark.json` runs -- passed neither. Every
/// ranked run therefore measured the MTP default no matter what the submission
/// contained. This enum is the vocabulary of the channel that closes that gap.
///
/// WHERE IT IS DECLARED: the `arm` key of `spec-decoder-head.manifest.json`, which is
/// an `editablePaths` AND `optionalEditablePaths` entry -- a file a submission
/// can actually write. (`benchmark.json` itself is trusted-side, so the ruled
/// design's literal "a field on the workspace manifest" is unimplementable as
/// written on this track.)
///
/// WHERE IT IS ENFORCED: `tools/gemma4-measure-and-score.sh`, the TRUSTED
/// wrapper, is the reader that turns a declared arm into a `--candidate-spec`
/// argument. Participant code never composes that argument and cannot reach
/// the benchmarker's CLI at all. This type is the SHARED VOCABULARY that reader
/// is drift-checked against (`tools/test-gemma4-arm-selection.sh`), and the
/// fail-closed parse any other reader of a declaration inherits.
///
/// WHY IT IS NOT `ReplaceableHeadKind`: the two enums happen to spell the same
/// two words today, but they answer different questions. `ReplaceableHeadKind`
/// asks "which head is this manifest about"; this asks "which spec mode should
/// the candidate leg run". The mode vocabulary is the benchmarker's
/// (`serial` / `mtp` / `dflash`), and a track that later admits a declared
/// `serial` arm would extend this and not that.
public enum DeclaredArm: String, Equatable, CaseIterable, Sendable {
    /// The stateless assistant MTP arm. THE DEFAULT, and what an absent `arm`
    /// key -- and an absent manifest -- mean.
    case mtp
    /// The z-lab DFlash drafter arm. Single-stream only.
    case dflash

    /// What an absent declaration selects. Stated once, here, so no reader
    /// spells the default itself.
    public static let absentDefault = DeclaredArm.mtp

    /// The manifest that carries the declaration. ONE file, named once: a
    /// track with two head manifests must not leave a participant guessing
    /// which of them the arm goes in, and the other one REFUSES an `arm` key
    /// rather than ignoring it (see `parse`).
    public static let declaringHeadKind = ReplaceableHeadKind.dflash
}

/// One parsed head declaration (`mtp-head.manifest.json` or, for the same
/// mechanism aimed at the DFlash drafter, `spec-decoder-head.manifest.json`).
public struct Gemma4MTPHeadDeclaration: Equatable, Sendable {
    public enum Source: String, Equatable, CaseIterable, Sendable {
        /// The operator's section-9d pinned head. The default, and what an
        /// ABSENT manifest means.
        case pinned
        /// Fetched by the runner from `sourceURL` (`hf:<repo>@<rev>` or
        /// `r2:<key>`) into the run's private directory.
        case remote
        /// Shipped in the submission under the editable weights directory
        /// named by `path`.
        case inBranch = "in_branch"
    }

    public let source: Source
    public let sourceURL: String?
    public let path: String?
    public let sha256: String?
    public let bytes: Int
    public let maxBytes: Int
    /// Which replaceable head this declaration is about.
    public let kind: ReplaceableHeadKind
    /// THE DECLARED ARM. Meaningful only on `DeclaredArm.declaringHeadKind`'s
    /// manifest; on every other kind an `arm` key is a REFUSAL, so this is
    /// always `absentDefault` there rather than quietly something else.
    public let arm: DeclaredArm

    public init(
        source: Source,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int = Gemma4MTPHeadDeclaration.defaultMaxBytes,
        kind: ReplaceableHeadKind = .mtp,
        arm: DeclaredArm = .absentDefault
    ) {
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes
        self.kind = kind
        self.arm = arm
    }

    /// 2 GiB. Mirrored in `mtp-head.manifest.json` and
    /// `spec-decoder-head.manifest.json` (`max_bytes`) and in the contract
    /// manifest's `editableSurfaceByteBudget.exemptPathMaxBytes`; a
    /// declaration may lower it and may not raise it. ONE cap for BOTH
    /// replaceable heads -- there is no per-kind ceiling.
    public static let defaultMaxBytes = 2_147_483_648

    /// The default when no manifest exists at all.
    public static let pinnedDefault = Gemma4MTPHeadDeclaration(source: .pinned)

    /// The organizer-pinned default for one head kind -- what an ABSENT
    /// manifest of that kind means. For a kind the organizer pins no head for
    /// (DFlash today), "pinned" is the worker's "no capability staged", which
    /// is a normal outcome and never an error.
    public static func pinnedDefault(
        for kind: ReplaceableHeadKind
    ) -> Gemma4MTPHeadDeclaration {
        Gemma4MTPHeadDeclaration(source: .pinned, kind: kind)
    }

    public static let relativePath = ReplaceableHeadKind.mtp.manifestRelativePath

    /// Parse a declaration, FAILING CLOSED on anything malformed.
    ///
    /// Only two things select the pinned head: the file being absent, and an
    /// explicit `source: "pinned"`. A manifest that is present but unreadable,
    /// unparseable, or internally inconsistent is a REFUSAL -- never a silent
    /// fall back -- because "your head declaration was broken so we quietly
    /// scored you on the pinned head" is exactly the failure mode that makes a
    /// leaderboard number unattributable.
    public static func parse(
        contentsOf url: URL,
        kind: ReplaceableHeadKind = .mtp
    ) throws -> Gemma4MTPHeadDeclaration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw MLXFastError.invalidInput(
                "the \(kind.declarationNoun) head declaration at \(url.path) "
                    + "exists but could not be read; refusing to fall back to "
                    + "the pinned head")
        }
        return try parse(data: data, origin: url.path, kind: kind)
    }

    public static func parse(
        data: Data,
        origin: String,
        kind: ReplaceableHeadKind = .mtp
    ) throws -> Gemma4MTPHeadDeclaration {
        let noun = kind.declarationNoun
        guard let root = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) is not a JSON object")
        }
        let rawSource = (root["source"] as? String) ?? Source.pinned.rawValue
        guard let source = Source(rawValue: rawSource) else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) names an unknown source "
                    + "'\(rawSource)'; expected one of "
                    + Source.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let maxBytes = (root["max_bytes"] as? NSNumber)?.intValue
            ?? defaultMaxBytes
        guard maxBytes > 0, maxBytes <= defaultMaxBytes else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) sets max_bytes "
                    + "\(maxBytes); it must be positive and may not exceed the "
                    + "track cap \(defaultMaxBytes)")
        }
        let sourceURL = root["source_url"] as? String
        let path = root["path"] as? String
        let sha256 = (root["sha256"] as? String)?.lowercased()
        let bytes = (root["bytes"] as? NSNumber)?.intValue ?? 0
        let arm = try parseArm(root: root, origin: origin, kind: kind)

        // REQUANT-ONLY (David ruling, 2026-08-26). `pinned` is the ONLY source
        // this track accepts. Both speculative heads are the organizer's own
        // weights; a participant may declare a re-quantization of them and may
        // not substitute weights of their own. `remote` and `in_branch` are the
        // two spellings of "load bytes the participant chose", so both are now
        // named refusals rather than gated flows.
        //
        // WHY THE CASES STAY IN THE ENUM. Deleting them would make a manifest
        // that names one fail as "unknown source", which reads like a typo. The
        // participant did not typo; they used a mode this track retired, and the
        // refusal should say so and say what replaced it.
        //
        // ATOMICITY. This is the parse-time half of the same ruling
        // `benchmark.json` implements by dropping `mtp-head/` and `dflash-head/`
        // from `editablePaths`. The surface gate stops weights that ride in the
        // submission; this stops a declaration that asks the runner to go and
        // get some. Neither is complete without the other, so they land together.
        switch source {
        case .pinned:
            break
        case .remote:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'remote'; this track accepts source 'pinned' only. The "
                    + "\(noun) head is the organizer's pinned checkpoint "
                    + "(fixtures/gemma4_26b_a4b_track.json), and custom head "
                    + "weights are not accepted. Declare a re-quantization of "
                    + "the pinned head instead")
        case .inBranch:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'in_branch'; this track accepts source 'pinned' only. A "
                    + "submission carries no head weights -- \(kind.stagedDirectoryName)/ "
                    + "is not an editable path -- so there is nothing an "
                    + "in-branch path could name. Declare a re-quantization of "
                    + "the pinned head instead")
        }

        // THE SIZE GATE SURVIVES THE NARROWING. It used to be reachable only
        // for a non-pinned source, which after the ruling would make it dead
        // code -- and deleting it would silently drop the one numeric bound a
        // declaration still carries. A `pinned` declaration may state the
        // `bytes` of the artifact the box stages (the re-quantized head is
        // smaller than the shipped one, and stating it is how a participant
        // records what they expect), so the bound is now "if you state a size,
        // it must fit the cap" rather than "non-pinned sources must state one".
        // `bytes: 0` means "not stated", which is what both checked-in
        // declarations say.
        if bytes != 0 {
            guard bytes > 0 else {
                throw MLXFastError.invalidInput(
                    "the \(noun) head declaration at \(origin) states bytes "
                        + "\(bytes); a stated byte count must be positive")
            }
            guard bytes <= maxBytes else {
                throw MLXFastError.invalidInput(
                    "the declared \(noun) head is \(bytes) bytes, above the "
                        + "\(maxBytes)-byte cap in \(origin)")
            }
        }
        return Gemma4MTPHeadDeclaration(
            source: source,
            sourceURL: sourceURL,
            path: path,
            sha256: sha256,
            bytes: bytes,
            maxBytes: maxBytes,
            kind: kind,
            arm: arm
        )
    }

    /// Parse the `arm` selection key, FAILING CLOSED.
    ///
    /// `arm` is the SIXTH-key-plus-one. The five keys above it (`source`,
    /// `source_url`, `path`, `sha256`, `bytes`, `max_bytes`) are read with
    /// `as?` casts that treat an absent key and a wrong-typed key alike, and
    /// every OTHER key in the object is simply not read -- the parser's
    /// unknown-key tolerance. That tolerance is harmless for a key nothing
    /// consumes and is NOT harmless for a key that selects which arm gets
    /// scored: a participant who writes `"arm": "dsparkk"`, or writes `arm`
    /// into the wrong manifest, would have been scored on the other arm and
    /// never told. So `arm` is handled by its own function with its own
    /// refusals rather than by a sixth `as?`.
    ///
    /// The three refusals, all pre-GPU and all named:
    ///   * present on a manifest that does not declare the arm;
    ///   * present but not a JSON string;
    ///   * a string outside `DeclaredArm`'s vocabulary (case-sensitive: the
    ///     benchmarker's mode strings are lowercase, so `"MTP"` is a typo and
    ///     is told so, not silently normalized).
    ///
    /// ABSENT is the ONLY silent path, and it selects
    /// `DeclaredArm.absentDefault` -- today's behaviour, unchanged.
    private static func parseArm(
        root: [String: Any],
        origin: String,
        kind: ReplaceableHeadKind
    ) throws -> DeclaredArm {
        guard let raw = root[armKey] else {
            return .absentDefault
        }
        let declaringKind = DeclaredArm.declaringHeadKind
        guard kind == declaringKind else {
            throw MLXFastError.invalidInput(
                "the \(kind.declarationNoun) head declaration at \(origin) sets "
                    + "'\(armKey)', but the arm is declared in "
                    + "\(declaringKind.manifestRelativePath) only; move the key "
                    + "there rather than leaving it here, where nothing reads it")
        }
        guard let text = raw as? String else {
            throw MLXFastError.invalidInput(
                "the \(kind.declarationNoun) head declaration at \(origin) sets "
                    + "'\(armKey)' to a non-string value; it must be one of "
                    + DeclaredArm.allCases.map { "'\($0.rawValue)'" }
                    .joined(separator: ", "))
        }
        guard let arm = DeclaredArm(rawValue: text) else {
            throw MLXFastError.invalidInput(
                "the \(kind.declarationNoun) head declaration at \(origin) "
                    + "declares \(armKey) '\(text)', which is not a mode this "
                    + "track admits; expected one of "
                    + DeclaredArm.allCases.map { "'\($0.rawValue)'" }
                    .joined(separator: ", ")
                    + ", or omit the key to run "
                    + "'\(DeclaredArm.absentDefault.rawValue)'")
        }
        return arm
    }

    /// The declaration key that selects the arm. Named once; the trusted
    /// wrapper reads the same string and is drift-checked against this file.
    public static let armKey = "arm"

    /// Read the declaration of one head kind next to a contract root, treating
    /// ABSENCE as the pinned default and everything else as parse-or-refuse.
    public static func resolve(
        contractRoot: URL,
        kind: ReplaceableHeadKind = .mtp
    ) throws -> Gemma4MTPHeadDeclaration {
        let url = contractRoot.appendingPathComponent(kind.manifestRelativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return pinnedDefault(for: kind)
        }
        return try parse(contentsOf: url, kind: kind)
    }
}

