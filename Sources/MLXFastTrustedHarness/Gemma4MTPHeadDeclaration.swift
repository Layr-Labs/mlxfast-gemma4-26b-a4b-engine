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
// way the MTP head is: `dflash-head.manifest.json`, same
// `source`/`sha256`/`bytes` shape, the SAME 2 GiB size-only cap, the same
// absent-means-organizer-default and present-but-broken-means-refusal
// posture. Rather than a second copy of this parser, the head KIND is a
// parameter: one gate, two manifests, no way for the two to drift apart.

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
        case .dflash: return "dflash-head.manifest.json"
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

/// One parsed head declaration (`mtp-head.manifest.json` or, for the same
/// mechanism aimed at the DFlash drafter, `dflash-head.manifest.json`).
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

    public init(
        source: Source,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int = Gemma4MTPHeadDeclaration.defaultMaxBytes,
        kind: ReplaceableHeadKind = .mtp
    ) {
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes
        self.kind = kind
    }

    /// 2 GiB. Mirrored in `mtp-head.manifest.json` and
    /// `dflash-head.manifest.json` (`max_bytes`) and in the contract
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
            kind: kind
        )
    }

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

