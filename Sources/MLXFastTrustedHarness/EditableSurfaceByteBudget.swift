import Foundation

/// Deterministic byte budget over the editable submission surface, enforced
/// immediately before a participant worker launch.
///
/// The full kernel-bypass review POLICY is a judge question and lives outside
/// this repository; these mechanical caps are the launch-time backstop that
/// binds every ranked worker launch path, including dispatches that never pass
/// through a review step.
///
/// PARITY NOTE. This is a re-implementation of the challenger's
/// `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift`
/// (Layr-Labs/qwen-3.8-mtp-challenge@bfab0de:1-157). Two deliberate
/// divergences, both documented in `docs/submission-restriction-spec.md`:
///
///   1. The caps are read from the CONTRACT, not from compiled-in constants.
///      The constants below are the fallback for a contract that declares no
///      caps, so the manifest is the single source of truth and a
///      manifest/enforcer drift is a test failure rather than a silent
///      divergence.
///   2. A cap key that is PRESENT but not a positive integer fails closed.
///      The original coerced a malformed `exemptPathMaxBytes` to its default
///      (`bfab0de:72-73`), i.e. a typo widened nothing but silently replaced
///      the operator's intent.
public enum EditableSurfaceByteBudget {
    /// RAISED 2026-08-24 (David ruling: the vendored ContinuousBatchingV2
    /// batching/round-driving engine joins the gemma4-26b-a4b-mlx-v1
    /// editablePaths surface -- docs/participant-contract.md section 2). The
    /// unmodified full-stack surface for that track is 3,358,640 bytes at
    /// rest, of which 2,629 bytes are the EXEMPT mtp-head/README.md (weights
    /// path, bounded separately by defaultExemptPathMaxBytes below, not by
    /// this constant) -- leaving 3,356,011 bytes of ENFORCED (non-exempt)
    /// surface this cap actually governs, already exceeding the previous
    /// 3.0 MB cap before any participant edit.
    ///
    /// MINIMALITY (re-derived 2026-08-24 after review, corrected once more
    /// on a second review pass): the cap is the ENFORCED at-rest surface
    /// (3,356,011 B) plus a STATED margin of exactly 1 MiB (1,048,576 B =
    /// 4x defaultMaxGrowthBytes) = 4,404,587. Two earlier revisions of this
    /// constant used less precise derivations: a ~40%-headroom-RATIO version
    /// (matching the original 3.0 MB / 2,139,781 B pair, never a documented
    /// design rule) landed on 4,700,000, and a first minimum-plus-margin pass
    /// double-counted the exempt mtp-head/ bytes into the "at-rest" figure
    /// (using 3,358,640 instead of the correct enforced-only 3,356,011),
    /// landing on 4,407,216. Both are superseded by the value below. 1 MiB
    /// of margin still fits a handful of near-term merged edits at the per-
    /// submission growth cap before this budget needs raising again, while
    /// keeping the cap tight enough to bind fail-closed rather than leave
    /// exploitable slack for smuggled content.
    public static let defaultMaxTotalBytes = 4_404_587
    public static let defaultMaxFileBytes = 524_288
    /// Bound on the bytes a submission may ADD to the editable surface versus
    /// its review base. Not consumed by the launch-time walk (there is no base
    /// commit at launch); resolved here so the pre-dispatch static-review
    /// tooling and this enforcer cannot disagree about its value.
    public static let defaultMaxGrowthBytes = 262_144
    public static let defaultContractRelativePath = "benchmark.json"

    /// Aggregate fallback cap over every EXEMPT editable path when the contract
    /// names exemptions without a cap. 512 MB decimal (David ruling
    /// 2026-08-26).
    ///
    /// SCOPE: this bounds SUBMITTED bytes -- the exempt-path content that rides
    /// in the submission archive -- and nothing else. Yukon's platform-wide
    /// expanded-archive cap is 512 MiB (536,870,912) and `maxSubmissionBytes`
    /// can only lower it, so an exempt surface above this could never pass
    /// submission validation. 512,000,000 plus `defaultMaxTotalBytes`
    /// (4,404,587) leaves ~32.4 MB of headroom under the platform cap.
    ///
    /// NOT the staged-head cap. An organizer-pinned head fetched ON BOX by
    /// `setup-gemma4-assistant.sh` / `setup-gemma4-dflash.sh` is gitignored,
    /// never archived, and never walked by this enforcer (the exempt walk runs
    /// on the submission archive, before any stager). Staged heads stay bounded
    /// at 2 GiB by `Gemma4MTPHeadDeclaration.defaultMaxBytes` and
    /// `gemma4DFlashStagedHeadMaxBytes`, which this ruling leaves untouched --
    /// the pinned DFlash drafter is ~859 MB and must keep loading.
    public static let defaultExemptPathMaxBytes = 512_000_000

    /// Per-FILE fallback cap for a SUBMITTED file under an exempt path. 100 MB
    /// decimal (David ruling 2026-08-26).
    ///
    /// Promotion commits the exempt head directories into the repository, so a
    /// submitted head file also has to survive a git push. GitHub refuses a
    /// single blob above 100 MB outright, so a head shipped as one monolithic
    /// shard would be unpushable even while fitting the aggregate. Both head
    /// loaders glob and merge every `*.safetensors` in the directory
    /// (`Gemma4MTP.swift`, `DFlashDraftModel.swift`), so sharding to satisfy
    /// this is a repack, not a capability loss.
    ///
    /// Same SUBMITTED-bytes scope as `defaultExemptPathMaxBytes`: an on-box
    /// staged head is never walked and is not bounded by this.
    public static let defaultExemptPathMaxFileBytes = 100_000_000
}

/// The caps in force for one contract.
public struct EditableSurfaceBudgetLimits: Equatable {
    public let maxTotalBytes: Int
    public let maxFileBytes: Int
    public let maxGrowthBytes: Int
    public let exemptPathMaxBytes: Int
    public let exemptPathMaxFileBytes: Int

    public init(
        maxTotalBytes: Int,
        maxFileBytes: Int,
        maxGrowthBytes: Int,
        exemptPathMaxBytes: Int,
        exemptPathMaxFileBytes: Int
            = EditableSurfaceByteBudget.defaultExemptPathMaxFileBytes
    ) {
        self.maxTotalBytes = maxTotalBytes
        self.maxFileBytes = maxFileBytes
        self.maxGrowthBytes = maxGrowthBytes
        self.exemptPathMaxBytes = exemptPathMaxBytes
        self.exemptPathMaxFileBytes = exemptPathMaxFileBytes
    }
}

public enum EditableSurfaceBudgetLimitsResolution: Equatable {
    case resolved(EditableSurfaceBudgetLimits)
    /// Nothing to read (no contract on disk). Official runs treat this as
    /// fatal; local invocations from outside a checkout proceed.
    case missingContract(reason: String)
    /// A contract that exists but cannot be trusted to state its own caps.
    case invalid(reason: String)
}

public enum EditableSurfaceBudgetVerification: Equatable {
    case verified(totalBytes: Int, fileCount: Int)
    /// Nothing to check (no contract on disk). Official runs treat this as
    /// fatal; local invocations from outside a checkout proceed.
    case skipped(reason: String)
    case exceeded(reason: String)
}

/// The subset of the track manifest this enforcer reads. Decoding (rather
/// than `JSONSerialization` + `as?` casts) is what makes a malformed cap
/// fail closed: `JSONDecoder` refuses `true`, `"2000000"` and `3.5` for an
/// `Int`.
///
/// The hand-written initializer exists for one reason: the synthesized one
/// uses `decodeIfPresent`, which reports an explicit JSON `null` as ABSENT and
/// would therefore coerce `"maxTotalBytes": null` to the fallback cap. Only a
/// key that is genuinely missing may take the fallback (divergence D2), and
/// `.github/scripts/submission-static-review-checks.sh` fails closed on the
/// same input — the two enforcers must not disagree about what a malformed cap
/// means.
private struct ContractBudgetPolicy: Decodable {
    var exemptPaths: [String]?
    var exemptPathMaxBytes: Int?
    var exemptPathMaxFileBytes: Int?
    var maxTotalBytes: Int?
    var maxFileBytes: Int?
    var maxGrowthBytes: Int?

    private enum CodingKeys: String, CodingKey {
        case exemptPaths
        case exemptPathMaxBytes
        case exemptPathMaxFileBytes
        case maxTotalBytes
        case maxFileBytes
        case maxGrowthBytes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        exemptPaths = try container.decodeIfPresent([String].self, forKey: .exemptPaths)
        exemptPathMaxBytes = try Self.decodeCap(container, .exemptPathMaxBytes)
        exemptPathMaxFileBytes = try Self.decodeCap(container, .exemptPathMaxFileBytes)
        maxTotalBytes = try Self.decodeCap(container, .maxTotalBytes)
        maxFileBytes = try Self.decodeCap(container, .maxFileBytes)
        maxGrowthBytes = try Self.decodeCap(container, .maxGrowthBytes)
    }

    /// Absent -> `nil` (take the fallback). Present -> must decode as `Int`,
    /// `null` included, so a malformed cap throws instead of vanishing.
    private static func decodeCap(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) throws -> Int? {
        guard container.contains(key) else { return nil }
        return try container.decode(Int.self, forKey: key)
    }
}

private struct ContractDocument: Decodable {
    var editablePaths: [String]?
    var editableSurfaceByteBudget: ContractBudgetPolicy?

    private enum CodingKeys: String, CodingKey {
        case editablePaths
        case editableSurfaceByteBudget
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        editablePaths = try container.decodeIfPresent([String].self, forKey: .editablePaths)
        // Same rule one level up: a PRESENT `editableSurfaceByteBudget` must be
        // an object. An explicit `null` there is a malformed budget, not a
        // contract that declares no caps.
        editableSurfaceByteBudget = container.contains(.editableSurfaceByteBudget)
            ? try container.decode(ContractBudgetPolicy.self, forKey: .editableSurfaceByteBudget)
            : nil
    }
}

private enum ContractLoad {
    case loaded(ContractDocument)
    case failed(EditableSurfaceBudgetLimitsResolution)
}

private func loadContract(at contractPath: String) -> ContractLoad {
    guard let data = FileManager.default.contents(atPath: contractPath) else {
        return .failed(.missingContract(
            reason: "no benchmark contract at \(contractPath)"
        ))
    }
    do {
        return .loaded(try JSONDecoder().decode(ContractDocument.self, from: data))
    } catch {
        return .failed(.invalid(
            reason: "benchmark contract at \(contractPath) is not readable as a "
                + "track manifest: \(error)"
        ))
    }
}

private enum CapResolution {
    case cap(Int)
    case failed(EditableSurfaceBudgetLimitsResolution)
}

private func positiveCap(
    _ declared: Int?,
    named name: String,
    fallback: Int,
    contractPath: String
) -> CapResolution {
    guard let declared else {
        return .cap(fallback)
    }
    guard declared > 0 else {
        return .failed(.invalid(
            reason: "editableSurfaceByteBudget.\(name) in \(contractPath) is "
                + "\(declared); it must be a positive integer"
        ))
    }
    return .cap(declared)
}

/// Resolve the caps this contract declares. An absent key takes the pinned
/// default; a present key must be a positive integer or the whole resolution
/// fails closed.
public func resolveEditableSurfaceBudgetLimits(
    contractPath: String
) -> EditableSurfaceBudgetLimitsResolution {
    let contract: ContractDocument
    switch loadContract(at: contractPath) {
    case .loaded(let document):
        contract = document
    case .failed(let resolution):
        return resolution
    }
    let policy = contract.editableSurfaceByteBudget
    var resolvedCaps: [Int] = []
    let declared: [(String, Int?, Int)] = [
        ("maxTotalBytes", policy?.maxTotalBytes, EditableSurfaceByteBudget.defaultMaxTotalBytes),
        ("maxFileBytes", policy?.maxFileBytes, EditableSurfaceByteBudget.defaultMaxFileBytes),
        ("maxGrowthBytes", policy?.maxGrowthBytes, EditableSurfaceByteBudget.defaultMaxGrowthBytes),
        (
            "exemptPathMaxBytes",
            policy?.exemptPathMaxBytes,
            EditableSurfaceByteBudget.defaultExemptPathMaxBytes
        ),
        (
            "exemptPathMaxFileBytes",
            policy?.exemptPathMaxFileBytes,
            EditableSurfaceByteBudget.defaultExemptPathMaxFileBytes
        ),
    ]
    for (name, value, fallback) in declared {
        switch positiveCap(value, named: name, fallback: fallback, contractPath: contractPath) {
        case .cap(let cap):
            resolvedCaps.append(cap)
        case .failed(let resolution):
            return resolution
        }
    }
    let maxTotalBytes = resolvedCaps[0]
    let maxFileBytes = resolvedCaps[1]
    let maxGrowthBytes = resolvedCaps[2]
    let exemptPathMaxBytes = resolvedCaps[3]
    let exemptPathMaxFileBytes = resolvedCaps[4]
    guard maxFileBytes <= maxTotalBytes else {
        return .invalid(
            reason: "editableSurfaceByteBudget.maxFileBytes (\(maxFileBytes)) in "
                + "\(contractPath) exceeds maxTotalBytes (\(maxTotalBytes)); the "
                + "per-file cap can never bind"
        )
    }
    // DELIBERATELY NO `exemptPathMaxFileBytes <= exemptPathMaxBytes` guard,
    // unlike the maxFileBytes/maxTotalBytes relation above. Those two share a
    // consistent pair of defaults, so a contract that lowers one and not the
    // other is stating something incoherent. The exempt pair does not: a
    // contract may legitimately declare a small aggregate and say nothing about
    // the per-file bound, and it would then be refused for a default it never
    // wrote. An unbindable per-file cap is harmless anyway -- the aggregate is
    // the tighter bound and refuses first -- so this fails OPEN on the
    // incoherent pair and closed on the bytes, which is the safe direction.
    return .resolved(EditableSurfaceBudgetLimits(
        maxTotalBytes: maxTotalBytes,
        maxFileBytes: maxFileBytes,
        maxGrowthBytes: maxGrowthBytes,
        exemptPathMaxBytes: exemptPathMaxBytes,
        exemptPathMaxFileBytes: exemptPathMaxFileBytes
    ))
}

/// Resolve the contract's own caps and enforce them. This is the entry point
/// a launch path should use: the manifest states the budget, nothing else.
public func verifyEditableSurfaceByteBudget(
    contractPath: String
) -> EditableSurfaceBudgetVerification {
    switch resolveEditableSurfaceBudgetLimits(contractPath: contractPath) {
    case .missingContract(let reason):
        return .skipped(reason: reason)
    case .invalid(let reason):
        return .exceeded(reason: reason)
    case .resolved(let limits):
        return verifyEditableSurfaceByteBudget(contractPath: contractPath, limits: limits)
    }
}

/// Walk the contract's `editablePaths` under the contract's directory and
/// enforce the per-file and total byte caps over every regular file. A symlink
/// at or under an editable path fails the surface closed rather than being
/// skipped (issue #20 Q2: a skipped link is a byte-count bypass), matching the
/// overlay tool's symlink refusal; a surface where every editable path is absent
/// or empty also fails closed (issue #20 Q3); a malformed contract fails closed.
///
/// EXEMPT PATHS. The contract may name `editableSurfaceByteBudget.exemptPaths`.
/// Those paths are held out of the CODE budget and charged against their own
/// `exemptPathMaxBytes` cap instead. The one path this exists for is the
/// editable MTP head weights directory (`mtp-head/`): head weights are part of
/// the competitive surface, they are not source, and a 3 MB source budget would
/// make the in-branch delivery mode unusable. The lookup-table defence the code
/// budget provides is untouched for every other path, and what bounds an exempt
/// path instead is the digest + byte-count verification of
/// `mtp-head.manifest.json`, done before the sandbox opens.
public func verifyEditableSurfaceByteBudget(
    contractPath: String,
    limits: EditableSurfaceBudgetLimits
) -> EditableSurfaceBudgetVerification {
    let fileManager = FileManager.default
    let contract: ContractDocument
    switch loadContract(at: contractPath) {
    case .loaded(let document):
        contract = document
    case .failed(.missingContract(let reason)):
        return .skipped(reason: reason)
    case .failed(.invalid(let reason)):
        return .exceeded(reason: reason)
    case .failed(.resolved):
        return .exceeded(reason: "unreachable contract load state for \(contractPath)")
    }
    guard let editablePaths = contract.editablePaths, !editablePaths.isEmpty else {
        return .exceeded(
            reason: "benchmark contract at \(contractPath) has no usable editablePaths"
        )
    }

    let exemptPaths = Set(contract.editableSurfaceByteBudget?.exemptPaths ?? [])
    let contractRoot = URL(fileURLWithPath: contractPath).deletingLastPathComponent()
    var totalBytes = 0
    var fileCount = 0
    var exemptBytes = 0

    func account(fileAt url: URL, sizeBytes: Int) -> EditableSurfaceBudgetVerification? {
        if sizeBytes > limits.maxFileBytes {
            return .exceeded(
                reason: "editable file \(url.path) is \(sizeBytes) bytes, above the "
                    + "per-file static review limit \(limits.maxFileBytes)"
            )
        }
        totalBytes += sizeBytes
        fileCount += 1
        if totalBytes > limits.maxTotalBytes {
            return .exceeded(
                reason: "editable surface is at least \(totalBytes) bytes, above the "
                    + "static review limit \(limits.maxTotalBytes)"
            )
        }
        return nil
    }

    /// An exempt path pays into its own cap and never into the code budget.
    func accountExempt(sizeBytes: Int, path: String) -> EditableSurfaceBudgetVerification? {
        // PER-FILE first, so a single oversize blob is named as itself rather
        // than surfacing later as an aggregate overshoot that points at the
        // whole exempt surface. Exempt files are held out of `maxFileBytes`
        // (that is the source-review cap), so without this an exempt path had
        // no per-file bound at all -- and a head shipped as one monolithic
        // shard above GitHub's 100 MB blob limit would pass the budget and
        // then fail the promotion push.
        if sizeBytes > limits.exemptPathMaxFileBytes {
            return .exceeded(
                reason: "exempt editable file \(path) is \(sizeBytes) bytes, "
                    + "above the exempt per-file limit "
                    + "\(limits.exemptPathMaxFileBytes)"
            )
        }
        exemptBytes += sizeBytes
        fileCount += 1
        if exemptBytes > limits.exemptPathMaxBytes {
            // The cap is an AGGREGATE over every exempt path, so the diagnostic
            // names the aggregate, not whichever path happened to tip it over
            // (issue #20 Q4: two 900 B exempt paths under a 1000 B cap were
            // refused naming only the second, implying it alone was oversize).
            return .exceeded(
                reason: "exempt editable paths total at least \(exemptBytes) bytes "
                    + "across \(exemptPaths.count) exempt path(s), above the "
                    + "exempt-path limit \(limits.exemptPathMaxBytes)"
            )
        }
        return nil
    }

    for editablePath in editablePaths {
        let isExempt = exemptPaths.contains(editablePath)
        let rootURL = contractRoot.appendingPathComponent(editablePath)
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDirectory) else {
            // A missing editable path contributes nothing to the walk. That is
            // safe on its own -- the fileCount backstop after the loop fails the
            // whole surface closed if EVERY editable path is absent, so absence
            // reads as an accusation rather than a clean zero-byte pass (issue
            // #20 Q3).
            continue
        }
        // Q2 (issue #20): a symlink AT an editable path is a byte-count bypass.
        // fileExists() above follows the link, so without this guard the walk
        // would size the target through it, or -- for a link the directory
        // enumerator declines to follow -- count it as zero. attributesOfItem()
        // does NOT follow the link, so it sees the link itself.
        if let rootType = (try? fileManager.attributesOfItem(atPath: rootURL.path))?[.type]
            as? FileAttributeType, rootType == .typeSymbolicLink
        {
            return .exceeded(
                reason: "editable path \(editablePath) is a symlink; refusing to "
                    + "byte-count the editable surface through a link"
            )
        }
        if !isDirectory.boolValue {
            guard let attributes = try? fileManager.attributesOfItem(atPath: rootURL.path) else {
                continue
            }
            // Not a symlink (refused above) and not a directory, so a non-regular
            // entry here is a FIFO / socket / device. It carries no payload, but
            // the in-directory branch below and the shell whole-surface gate both
            // refuse ANY non-regular entry, so the root branch refuses too rather
            // than skipping it silently -- all three gates symmetric on a
            // non-regular file AT a root editable path (issue #20 Q2, F1).
            if attributes[.type] as? FileAttributeType != .typeRegular {
                return .exceeded(
                    reason: "editable path \(editablePath) is a non-regular file; "
                        + "refusing a surface that is not plain files and directories"
                )
            }
            guard let sizeBytes = (attributes[.size] as? NSNumber)?.intValue else {
                continue
            }
            if let verdict = isExempt
                ? accountExempt(sizeBytes: sizeBytes, path: editablePath)
                : account(fileAt: rootURL, sizeBytes: sizeBytes)
            {
                return verdict
            }
            continue
        }
        guard let enumerator = fileManager.enumerator(atPath: rootURL.path) else {
            return .exceeded(
                reason: "could not enumerate editable path \(rootURL.path)"
            )
        }
        while let entry = enumerator.nextObject() as? String {
            guard let type = enumerator.fileAttributes?[.type] as? FileAttributeType else {
                continue
            }
            // Directories are descended into, not counted.
            if type == .typeDirectory { continue }
            // Q2 (issue #20): the enumerator does not follow symlinks, so a link
            // inside an editable directory used to be silently skipped and its
            // (possibly multi-megabyte) target went uncounted. Any non-regular
            // entry now fails the surface closed, matching the overlay tool's
            // validate_overlay_tree, which admits only regular files and
            // directories.
            if type != .typeRegular {
                return .exceeded(
                    reason: "editable path \(editablePath) contains a non-regular "
                        + "entry \(entry); refusing to byte-count the editable "
                        + "surface through a link or special file"
                )
            }
            guard let sizeBytes = (enumerator.fileAttributes?[.size] as? NSNumber)?.intValue
            else {
                continue
            }
            if let verdict = isExempt
                ? accountExempt(
                    sizeBytes: sizeBytes,
                    path: "\(editablePath)/\(entry)"
                )
                : account(
                    fileAt: rootURL.appendingPathComponent(entry),
                    sizeBytes: sizeBytes
                )
            {
                return verdict
            }
        }
    }
    // Q3 (issue #20): a surface where every editable path is absent (or every
    // present path is an empty directory) walked to totalBytes=0 fileCount=0 and
    // returned .verified -- absence read as a clean pass. It is a refusal: a real
    // submission always carries at least one editable source file. Mirrors the
    // shell gate's `file_count == 0` guard in whole-surface mode.
    guard fileCount > 0 else {
        return .exceeded(
            reason: "no editable file found under any editablePath at "
                + "\(contractRoot.path); every editable path is absent or empty "
                + "(absence is a refusal, not a zero-byte pass)"
        )
    }
    return .verified(totalBytes: totalBytes, fileCount: fileCount)
}
