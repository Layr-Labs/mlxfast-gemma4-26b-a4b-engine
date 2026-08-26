import CryptoKit
import Foundation
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative

// Assistant-head loading for the Gemma 4 26B A4B MTP arm.
//
// The assistant is a HEAD ATTACHED TO THE TARGET (port-notes §4.2), not a
// second model-factory entry: a standalone 4-layer drafter with its own
// `config.json` + safetensors, bound to the running target instance via
// `Gemma4CBv2MTPDrafter.init(drafter:target:)`, which calls
// `Gemma4AssistantDraftModel.bind(target:)` internally and THAT runs
// `Gemma4MTPCompatibilityValidator.validate` (Gemma4MTPConfigurationValidation
// .swift) as part of binding — geometry mismatch throws
// `Gemma4MTPError.incompatibleDrafter` there, not in a separate check this
// file has to remember to call. This is the corrected mechanism: an earlier
// design pass for this file assumed the Qwen-era `AdditionalWeightSource` /
// `Load.swift` pre-`sanitize` merge path applied here. It does not — this
// engine's `Gemma4A4BRuntimeWeightCache.loadLibraryModel` never calls
// `loadWeights` at all (it hand-builds `Gemma4TextModel` directly), so the
// `_additionalWeightSources` merge point has zero consumers on this path.
//
// The drafter reuses the TARGET's tokenizer at generation time (verified fork
// behavior, port-notes §4.0a) — this file never loads or references the
// assistant's own tokenizer files, and neither does anything downstream of
// `Gemma4CBv2MTPDrafter` (`Gemma4MTPConfigurationValidation.swift` reads only
// `config.json`).
//
// STAGING. Two channels select the head directory, and which one is in play
// decides the failure posture:
//
// * DEFAULT (no `--mtp-head` argv): the staged head lives at `mtp-head/`,
//   resolved relative to the current working directory — the same
//   CWD-is-package-root invariant `EmitWireFixtureTests` and
//   `HarnessHashRootSetTests` already rely on, and the flow the native
//   trusted CLI and `setup-gemma4-assistant.sh` use. Absent directory (or a
//   placeholder directory with no `config.json`) = no head, not an error —
//   the DECIDE-2 "absent declaration = no head" posture.
// * EXPLICIT (`--mtp-head <DIR>`, restored 2026-08-25): benchd's measure-job
//   spawns every worker leg with the head directory on the argv
//   (`timed_leg_base_args`, benchd @ c2327d15 — the serial control gets the
//   PINNED head, the candidate its declared BYO head; both legs load one so
//   residency charges the denominator). An explicit argv directory is a
//   DECLARATION: it must exist and carry a loadable head, and anything less
//   fails the worker at startup — the same present-but-broken refusal
//   `Gemma4MTPHeadDeclaration` states for a broken manifest, never a silent
//   downgrade to serial-only.
//
// Fetching a REMOTE declaration (`mtp-head.manifest.json` `source: "remote"`)
// into a staging directory is TRUSTED, pre-sandbox, runner-side work
// (`Gemma4MTPHeadDeclaration`, Sources/MLXFastTrustedHarness) that stays
// upstream of both channels. This file's job starts AFTER staging: resolve
// which directory (if any) is staged, decide `mtpAvailable`, and load if so.

/// Where the staged assistant head lives, relative to the process CWD.
let gemma4AssistantHeadDirectoryName = "mtp-head"

/// Where the staged z-lab DFlash drafter lives, relative to the process CWD.
///
/// Separate directory AND separate load machinery. The DFlash drafter
/// (z-lab/gemma-4-26B-A4B-it-DFlash) is EAGLE-style in the loose sense — it
/// conditions on target hidden states and borrows the target's embedding and
/// LM head — but it is NOT the MTP assistant head's architecture and its
/// `config.json` is not the MTP assistant's schema: it declares
/// `architectures: ["DFlashDraftModel"]`, `block_size`, `num_target_layers`
/// and a `dflash_config { target_layer_ids, mask_token_id }` block, it
/// conditions on the CONCATENATION of several tapped layers rather than the
/// single last hidden, and it owns a KV cache of its own. An earlier revision
/// of this comment claimed "IDENTICAL load machinery … loaded and bound by
/// exactly the same code (`Gemma4AssistantDraftModel` +
/// `Gemma4CBv2MTPDrafter`)"; that alias could never have bound a real z-lab
/// head — `Gemma4AssistantConfiguration` cannot decode that `config.json` at
/// all. It is now loaded as a real `DFlashDraftModel` (Vendor/mlx-swift-lm
/// Libraries/MLXSpeculative) bound through `DFlashTargetModel`.
///
/// This is the DEFAULT channel only. An earlier revision of this comment said
/// "DFlash is loaded ONLY from this CWD default (no `--dflash-head` argv flag
/// yet … the explicit staging channel is follow-up work)", on the premise that
/// "benchd does not spawn a DFlash leg today". David's 2026-08-26 ruling made
/// DFlash a first-class SCORED mode, so benchd does, and that follow-up work is
/// `runtimeWorkerDFlashHeadFlag` — the per-leg `--dflash-head` channel, added to
/// `runtimeWorkerAcceptedOptionFlags` on BOTH sides of the cross-repo spawn
/// fence in the same change. An absent/placeholder directory = no DFlash
/// capability, never an error, exactly like the default MTP channel; a
/// PRESENT-but-unloadable one is capability-absent plus a named stderr warning,
/// refused later at spec resolution (`loadGemma4DFlashHeadIfStaged`).
///
/// SIZE. The DFlash drafter is the track's SECOND replaceable head and it is
/// held to the SAME 2 GiB per-head cap as the MTP head (David's 2026-08-25
/// both-replaceable-heads-one-cap ruling). That cap binds at TWO layers, and
/// this arm needs both: `dflash-head.manifest.json`, parsed and size-gated by
/// `Gemma4MTPHeadDeclaration` with `kind: .dflash`
/// (Sources/MLXFastTrustedHarness), gates what the trusted, pre-sandbox
/// stager is allowed to DECLARE and fetch; `gemma4DFlashStagedHeadMaxBytes`
/// below gates what this loader is allowed to LOAD off disk. See that
/// constant for why one layer is not enough.
let gemma4DFlashHeadDirectoryName = "dflash-head"

/// The 2 GiB per-head cap on the BYTES ACTUALLY STAGED at `dflash-head/`.
///
/// WHY THIS EXISTS SEPARATELY FROM THE DECLARATION GATE. The declaration
/// mechanism (`Gemma4MTPHeadDeclaration` + `ReplaceableHeadKind.dflash`,
/// Sources/MLXFastTrustedHarness) enforces the same 2 GiB on
/// `dflash-head.manifest.json`'s `bytes` — but that gate sits on the MANIFEST
/// layer, upstream of the sandbox, and it binds the DECLARED size of a head
/// the stager fetches. The MTP arm can lean on it alone because its loader
/// runs behind that same stager. This arm cannot: `loadGemma4DFlashHeadIfStaged`
/// loads a REAL `DFlashDraftModel` straight off the CWD `dflash-head/`
/// default and never routes through the MTP head loader or the declaration
/// parser (the alias that used to make that true was deleted when the real
/// port landed). Without this constant, a `dflash-head/` that is simply
/// PRESENT on disk — placed by any path that did not go through a
/// declaration — would be loaded at whatever size it happens to be, and the
/// 2 GiB ruling would hold for the MTP head and not for the DFlash one.
///
/// ONE SOURCE. The value is `Gemma4MTPHeadDeclaration.defaultMaxBytes` and must
/// stay equal to it. It is MIRRORED rather than imported because
/// `Sources/MLXFastHarness` (the sandboxed participant runtime-worker target)
/// deliberately does not depend on the trusted-harness target — the two trees
/// carry ~45 twin symbols and the trust split is the point, so a dependency
/// edge here would compile the trusted build into the worker binary. That is
/// the same constraint the existing `MTPEnvelope.swift` twins live under, and
/// it is answered the same way: a tripwire test
/// (`stagedDFlashHeadCapMirrorsTheDeclarationCap`) imports BOTH modules and
/// fails the moment the two numbers disagree, so the declaration mechanism
/// stays the single authority for the value.
let gemma4DFlashStagedHeadMaxBytes = 2_147_483_648

/// The result of attempting to load and bind the assistant head at startup.
struct Gemma4AssistantHeadLoadResult {
    let drafter: Gemma4CBv2MTPDrafter
    /// The harness's OWN recomputed tree digest over the staged directory's
    /// files, in stable path order — never the manifest's declared value
    /// (`Gemma4MTPHeadDeclaration`'s own header comment: "head_provenance
    /// records the harness's own recomputed tree digest, not the declared
    /// value"). This is what rides on `RuntimeWorkerHeadProvenance`.
    let provenance: RuntimeWorkerHeadProvenance
}

/// Where a worker's assistant head is staged, decided BEFORE any load work.
/// `.none` is the normal serial-only outcome of the default channel; the
/// explicit channel can never produce it (see
/// `resolveGemma4AssistantHeadStaging`).
enum Gemma4AssistantHeadStaging: Equatable {
    /// Nothing staged — the worker runs serial-only. Never an error.
    case none
    /// A head is staged at this directory; from here on every failure to
    /// load or bind it is a refusal.
    case staged(URL)
}

/// Resolve which directory (if any) a replaceable head loads from.
///
/// `explicitDirectoryPath` is the argv value of the head's spawn flag
/// (benchd's spawn channel) — `--mtp-head` for the MTP assistant,
/// `--dflash-head` for the DFlash drafter. `flagName` names WHICH, so the
/// fail-closed refusals below quote the flag the operator actually passed;
/// everything else about the two channels is identical, which is why they
/// share this one resolver rather than growing a second copy of the rule.
///
/// When non-nil the directory is a DECLARATION and this fails
/// closed — `MLXFastError.invalidInput` — unless it exists, is a directory,
/// and carries the head's `config.json`. A missing or placeholder explicit
/// directory is *present-but-broken*, never "not staged": benchd
/// existence-checks the pinned head dir before spawning (die 8), so an
/// explicit path that resolves to nothing here is a real staging fault the
/// worker must surface pre-hello, not quietly serial-only its way past.
///
/// FOR THE DFLASH ARM the fail-closed half is load-bearing in a second way
/// (David ruling 2026-08-26). A per-leg explicit path that silently fell back
/// to the CWD default would put BOTH legs back on one shared `./dflash-head/`
/// — the cross-leg residency the flag exists to close, restored by a
/// tolerance. So an explicitly declared drafter that is not there refuses;
/// it does not degrade.
///
/// When nil, the default CWD channel applies: `defaultDirectoryName` relative
/// to the process CWD, where an absent directory — or one with no
/// `config.json` (an empty/placeholder directory is a normal outcome of a
/// checkout that never ran the staging step) — is `.none`, not an error.
func resolveGemma4AssistantHeadStaging(
    explicitDirectoryPath: String?,
    defaultDirectoryName: String = gemma4AssistantHeadDirectoryName,
    flagName: String = runtimeWorkerMTPHeadFlag
) throws -> Gemma4AssistantHeadStaging {
    let fm = FileManager.default
    if let explicitDirectoryPath {
        let directory = URL(fileURLWithPath: explicitDirectoryPath, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard
            !explicitDirectoryPath.isEmpty,
            fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw MLXFastError.invalidInput(
                "\(flagName) names '\(explicitDirectoryPath)', which is not "
                    + "an existing directory — an explicitly declared head directory is "
                    + "loaded fail-closed, never skipped")
        }
        let configURL = directory.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configURL.path) else {
            throw MLXFastError.invalidInput(
                "\(flagName) directory '\(explicitDirectoryPath)' carries no "
                    + "config.json — an explicitly declared head directory must contain a "
                    + "loadable assistant head (fail-closed, never a silent serial-only "
                    + "downgrade)")
        }
        return .staged(directory)
    }

    let directory = URL(fileURLWithPath: defaultDirectoryName, isDirectory: true)
    var isDirectory: ObjCBool = false
    guard
        fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        return .none
    }
    let configURL = directory.appendingPathComponent("config.json")
    guard fm.fileExists(atPath: configURL.path) else {
        // A directory exists but carries no config.json — not staged, not
        // broken (default channel only; the explicit channel above refuses
        // this same shape).
        return .none
    }
    return .staged(directory)
}

/// Attempt to load and bind the assistant head against `target`, resolving
/// the staging directory per `resolveGemma4AssistantHeadStaging`:
/// `explicitDirectoryPath` (the `--mtp-head` argv value) when given, else the
/// CWD `directoryName` default. Returns `nil` — NOT a thrown error — only
/// when the DEFAULT channel finds nothing staged: "no head staged" is the
/// normal, expected case for a worker spawned without one (DECIDE-2: absent
/// declaration selects the pinned default, which for a worker with nothing
/// staged is "no MTP capability", not a refusal). A staged head that fails to
/// load or fails compatibility validation DOES throw — as does an explicit
/// directory that is missing or unloadable — a present-but-broken head is a
/// refusal, never a silent downgrade to serial-only (the same posture
/// `Gemma4MTPHeadDeclaration`'s doc comment states for a broken declaration).
func loadGemma4AssistantHeadIfStaged(
    explicitDirectoryPath: String? = nil,
    directoryName: String = gemma4AssistantHeadDirectoryName,
    target: Gemma4TextModel
) throws -> Gemma4AssistantHeadLoadResult? {
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: explicitDirectoryPath,
        defaultDirectoryName: directoryName)
    guard case .staged(let directory) = staging else {
        return nil
    }

    let drafterModel = try loadGemma4AssistantDraftModelSync(from: directory)
    let drafter = try Gemma4CBv2MTPDrafter(drafter: drafterModel, target: target)
    let provenance = try computeGemma4AssistantHeadProvenance(directory: directory)
    return Gemma4AssistantHeadLoadResult(drafter: drafter, provenance: provenance)
}

/// A bound z-lab DFlash drafter plus the harness's own tree digest over the
/// directory it came from.
struct Gemma4DFlashHeadLoadResult {
    let drafter: DFlashDraftModel
    /// Same recomputed tree digest the MTP arm seals, over `dflash-head/`.
    let provenance: RuntimeWorkerHeadProvenance

    /// The largest draft depth this drafter can actually propose
    /// (`gemma4DFlashMaxDepth`). Both the `effective_spec` echo and the round
    /// loop read this one value, so the echoed depth is the depth that runs.
    var maxDepth: Int { gemma4DFlashMaxDepth(for: drafter) }
}

/// What startup found at `dflash-head/`. Three outcomes, not two, because
/// "nothing staged" and "staged but unloadable" carry different obligations
/// (see `loadGemma4DFlashHeadIfStaged`).
enum Gemma4DFlashHeadLoadOutcome {
    /// No `dflash-head/` (or a placeholder directory with no `config.json`).
    /// Capability-absent, silent — the normal case for every worker that is
    /// not running the DFlash arm.
    case absent
    /// A drafter loaded and bound.
    case loaded(Gemma4DFlashHeadLoadResult)
    /// A directory IS staged and carries a `config.json`, but the drafter
    /// could not be loaded or bound. Capability-absent for this worker's
    /// lifetime, with `reason` retained so DFlash SPEC RESOLUTION can name
    /// the real failure instead of the generic "not runnable" shape.
    case incompatible(directory: URL, reason: String)
}

/// Load and bind the staged z-lab DFlash drafter, if any, from the CWD
/// `dflash-head/` default.
///
/// This loads a REAL `DFlashDraftModel` (Vendor/mlx-swift-lm
/// Libraries/MLXSpeculative) bound to the running Gemma 4 target through
/// `DFlashTargetModel`. It is NOT the MTP assistant-head loader: a z-lab
/// DFlash checkpoint's `config.json` declares `architectures:
/// ["DFlashDraftModel"]`, `block_size`, `num_target_layers` and a
/// `dflash_config { target_layer_ids, mask_token_id }` block that
/// `Gemma4AssistantConfiguration` cannot decode at all, so the alias this
/// replaces (gemma4-dflash-arm, #38) could never have bound a real drafter —
/// it would have failed at the first `config.json` read. `bind(target:)`
/// runs `validateCompatibility`, which checks hidden size, vocab size and
/// `num_target_layers == target.dFlashLayerCount`, then installs the target's
/// embedding and LM head as the drafter's borrowed input/output.
///
/// FAIL-SOFT, AUDIT-REQUIRED (2026-08-25). This function NEVER throws. An
/// absent directory is `.absent`, silently. A present-but-unloadable
/// directory is `.incompatible` — capability-absent plus a retained reason —
/// because the worker MUST still reach `hello` with its serial and mtp arms
/// intact: a DFlash head that fails to decode is a DFlash problem, and
/// killing the process before hello would take the serial CONTROL leg down
/// with it. The present-but-broken REFUSAL still happens, just later and
/// scoped to the arm that is actually broken: resolving `{"mode":"dflash"}`
/// on such a worker refuses by name and quotes `reason`.
///
/// SIZE-GATED BEFORE THE LOAD. A staged tree over the per-head
/// `gemma4DFlashStagedHeadMaxBytes` cap is `.incompatible` — the SAME
/// fail-soft outcome as an undecodable one, not a throw. This is deliberate
/// and it is the only posture consistent with the paragraph above: an
/// oversized DFlash head is still a DFlash-only problem, and refusing it by
/// killing the worker before hello would take the serial CONTROL leg down
/// with it. So the cap refusal lands exactly where every other DFlash load
/// refusal lands — named on stderr, capability-absent for this worker's
/// lifetime, quoted back at `{"mode":"dflash"}` resolution, serial and mtp
/// untouched. Note the ORDER: the cap is checked BEFORE
/// `loadGemma4DFlashDraftModelSync`, so an over-cap tree is never read into
/// memory in the first place.
/// PER-LEG STAGING (David ruling 2026-08-26). `explicitDirectoryPath` is the
/// `--dflash-head` argv value benchd passes PER LEG — the PINNED drafter to
/// the serial control, the candidate's own to the candidate leg. It resolves
/// through the SAME `resolveGemma4AssistantHeadStaging` the MTP head uses, so
/// the two channels cannot drift: explicit ⇒ fail-closed on that exact
/// directory, nil ⇒ the CWD `./dflash-head/` default, unchanged.
///
/// The fail-closed half is what makes the per-leg guarantee real. Both legs
/// inherit benchctl's CWD (benchd's spawn sets no `current_dir`), so a
/// declared-but-missing drafter that fell back to the default would put both
/// legs back on ONE shared directory — the candidate's drafter resident on
/// the scored DENOMINATOR leg, silently. A declaration that cannot be honoured
/// is therefore an error, and it is the ONE DFlash failure mode that is not
/// fail-soft: the fail-soft posture below exists so a BROKEN DFlash head
/// cannot take the serial control leg down, and a MISWIRED per-leg staging is
/// the opposite case — the serial control leg is exactly what it endangers.
func loadGemma4DFlashHeadIfStaged(
    explicitDirectoryPath: String? = nil,
    directoryName: String = gemma4DFlashHeadDirectoryName,
    target: Gemma4TextModel
) throws -> Gemma4DFlashHeadLoadOutcome {
    let staging = try resolveGemma4AssistantHeadStaging(
        explicitDirectoryPath: explicitDirectoryPath,
        defaultDirectoryName: directoryName,
        flagName: runtimeWorkerDFlashHeadFlag)
    guard case .staged(let directory) = staging else {
        return .absent
    }

    // The size gate runs on the STAGED bytes, after "is anything staged at
    // all" and before anything is loaded. An unmeasurable tree is treated as
    // a failed gate rather than a passed one: this is a cap, so it fails
    // closed on the DFlash arm (which, being fail-soft, means
    // capability-absent — never a throw).
    let stagedBytes: Int
    do {
        stagedBytes = try measureGemma4StagedHeadBytes(directory: directory)
    } catch {
        return .incompatible(
            directory: directory,
            reason: "the staged DFlash head could not be measured against the "
                + "\(gemma4DFlashStagedHeadMaxBytes)-byte per-head cap — "
                + describeGemma4DFlashLoadFailure(error))
    }
    guard stagedBytes <= gemma4DFlashStagedHeadMaxBytes else {
        return .incompatible(
            directory: directory,
            reason: "the staged DFlash head is \(stagedBytes) bytes, above the "
                + "\(gemma4DFlashStagedHeadMaxBytes)-byte per-head cap that "
                + "the DFlash head declaration (dflash-head.manifest.json) "
                + "enforces on the manifest layer")
    }

    do {
        let drafter = try loadGemma4DFlashDraftModelSync(from: directory, target: target)
        let provenance = try computeGemma4AssistantHeadProvenance(directory: directory)
        return .loaded(
            Gemma4DFlashHeadLoadResult(drafter: drafter, provenance: provenance))
    } catch {
        return .incompatible(
            directory: directory,
            reason: describeGemma4DFlashLoadFailure(error))
    }
}

/// Bridge `DFlashDraftModel.load(from:)` (async) into the worker's
/// synchronous startup path, exactly once, outside every timed window — the
/// same shape the MTP head's `loadGemma4AssistantDraftModelSync` uses — then
/// bind on THIS thread.
///
/// The bind is deliberately split out of the load rather than using the
/// fork's `load(from:bindTo:)` convenience: that overload would have to
/// capture the ~21.6 GB `Gemma4TextModel` in the detached `Task`'s closure,
/// and the target is not `Sendable`. Binding after the weights land is
/// equivalent — `bind(target:)` runs `validateCompatibility` (hidden size,
/// vocab size, `num_target_layers`, tap-id range) off the CONFIG, which is
/// already decoded, and then installs the borrowed embedding / LM-head
/// closures. `DFlashTokenIterator` binds in the same after-the-fact order.
func loadGemma4DFlashDraftModelSync(
    from directory: URL,
    target: Gemma4TextModel
) throws -> DFlashDraftModel {
    let semaphore = DispatchSemaphore(value: 0)
    let box = DFlashLoadResultBox()
    Task {
        do {
            box.result = .success(try await DFlashDraftModel.load(from: directory))
        } catch {
            box.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "DFlash drafter load task at \(directory.path) completed without a result")
    }
    let drafter = try result.get()
    try drafter.bind(target: target)
    return drafter
}

private final class DFlashLoadResultBox: @unchecked Sendable {
    var result: Result<DFlashDraftModel, Error>?
}

/// Render a DFlash load failure as something an operator can act on.
///
/// `DecodingError.localizedDescription` is the useless "The data couldn't be
/// read because it isn't in the correct format." — which hides exactly the
/// information the audit asked for, because every one of
/// `DFlashConfiguration`'s validity checks (`block_size >= 2`,
/// `mask_token_id` in range, non-empty / unique / in-range
/// `target_layer_ids`, `layer_types` count, sliding-window presence) throws
/// `DecodingError.dataCorruptedError` whose message lives in the CONTEXT's
/// `debugDescription`. This digs that out, with the key path, so the stderr
/// warning names the actual defect.
func describeGemma4DFlashLoadFailure(_ error: Error) -> String {
    guard let decoding = error as? DecodingError else {
        // DFlashError already carries a specific `errorDescription`
        // (incompatible drafter field, missing safetensors, duplicate weight
        // key, …), as does MLXFastError.
        return (error as? LocalizedError)?.errorDescription ?? "\(error)"
    }
    func render(_ label: String, _ context: DecodingError.Context) -> String {
        let path = context.codingPath.map(\.stringValue).joined(separator: ".")
        let where_ = path.isEmpty ? "" : " at '\(path)'"
        let underlying = context.underlyingError.map { " (\($0))" } ?? ""
        return "\(label)\(where_): \(context.debugDescription)\(underlying)"
    }
    switch decoding {
    case .dataCorrupted(let context):
        return render("invalid value", context)
    case .keyNotFound(let key, let context):
        return render("missing key '\(key.stringValue)'", context)
    case .typeMismatch(let type, let context):
        return render("type mismatch (expected \(type))", context)
    case .valueNotFound(let type, let context):
        return render("null value (expected \(type))", context)
    @unknown default:
        return "\(decoding)"
    }
}

/// Bridge `Gemma4AssistantDraftModel.load(from:)` (async) into the worker's
/// synchronous startup path, exactly once, outside every timed window — the
/// same shape `weightCache.requireLibraryModel()` already uses to load the
/// ~21.6 GB target synchronously at startup.
func loadGemma4AssistantDraftModelSync(
    from directory: URL
) throws -> Gemma4AssistantDraftModel {
    let semaphore = DispatchSemaphore(value: 0)
    let box = LoadResultBox()
    Task {
        do {
            let model = try await Gemma4AssistantDraftModel.load(from: directory)
            box.result = .success(model)
        } catch {
            box.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "assistant head load task at \(directory.path) completed without a result")
    }
    return try result.get()
}

/// Plain reference box carrying the async load's result across the
/// semaphore hand-off; a local `var` captured by the `Task` closure above
/// would not be safely readable from this function's thread without one.
private final class LoadResultBox: @unchecked Sendable {
    var result: Result<Gemma4AssistantDraftModel, Error>?
}

/// The harness's own tree digest over the staged head directory, computed by
/// the SAME algorithm `mtp-head/README.md`'s "TREE DIGEST RULE" documents
/// (mirrored there specifically so a declarer can recompute the number this
/// function will produce): SHA-256 over the concatenation, in `LC_ALL=C`
/// sorted relative-path order, of `"<hex file sha256>  <relative path>\n"`
/// for every regular file in the tree EXCEPT a top-level `README.md` (the
/// inert placeholder that keeps the directory archivable when no weights are
/// shipped there — excluding it is what makes it invisible to verification).
/// Plus the total byte count and file count of that same file set. Mirrors
/// the shape `RuntimeWorkerHeadProvenance` already declares on the wire
/// (`RuntimeWorkerSpecConfig.swift`); this is what makes the field non-nil
/// again now that a head can load (it was "RETAINED, ALWAYS NIL" per that
/// type's header comment because the digest helper "went with the MTP arm").
/// Total bytes of a staged head tree, over EXACTLY the file set
/// `computeGemma4AssistantHeadProvenance` digests: every regular file except
/// a top-level `README.md`. Keeping the two file sets identical is what makes
/// the size gate and the sealed `head_provenance.bytes` describe the same
/// tree — a cap measured over a different file set than the one that gets
/// sealed would be a cap on a number nobody can reproduce.
///
/// Reads SIZES, never contents: this runs before the load precisely so an
/// over-cap tree is refused without being pulled into memory, and reading it
/// to measure it would defeat that.
func measureGemma4StagedHeadBytes(directory: URL) throws -> Int {
    let fm = FileManager.default
    guard
        let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles])
    else {
        throw MLXFastError.invalidInput(
            "could not enumerate staged head directory \(directory.path)")
    }
    let base = directory.standardizedFileURL.path
    var totalBytes = 0
    for case let url as URL in enumerator {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else { continue }
        let full = url.standardizedFileURL.path
        guard full.hasPrefix(base) else { continue }
        let relative = String(full.dropFirst(base.count))
        guard relative != "README.md", relative != "/README.md" else { continue }
        guard let fileSize = values.fileSize else {
            throw MLXFastError.invalidInput(
                "staged head file at \(full) reported no size")
        }
        totalBytes += fileSize
    }
    return totalBytes
}

func computeGemma4AssistantHeadProvenance(
    directory: URL
) throws -> RuntimeWorkerHeadProvenance {
    let fm = FileManager.default
    guard
        let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])
    else {
        throw MLXFastError.invalidInput(
            "could not enumerate assistant head directory \(directory.path)")
    }
    var relativePaths: [String] = []
    let base = directory.standardizedFileURL.path
    for case let url as URL in enumerator {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else { continue }
        let full = url.standardizedFileURL.path
        guard full.hasPrefix(base) else { continue }
        let relative = String(full.dropFirst(base.count))
        guard relative != "README.md", relative != "/README.md" else { continue }
        relativePaths.append(relative)
    }
    // `LC_ALL=C` sorted order == plain Unicode-scalar string ordering, which
    // is what Swift's default `<` on String already gives for the ASCII
    // paths a checkpoint tree contains.
    relativePaths.sort()

    var treeDigest = SHA256()
    var totalBytes = 0
    for relative in relativePaths {
        let fileURL = directory.appendingPathComponent(relative)
        guard let data = fm.contents(atPath: fileURL.path) else {
            throw MLXFastError.invalidInput(
                "assistant head file at \(fileURL.path) could not be read")
        }
        let fileHex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let relativePath = relative.hasPrefix("/") ? String(relative.dropFirst()) : relative
        treeDigest.update(data: Data("\(fileHex)  \(relativePath)\n".utf8))
        totalBytes += data.count
    }
    let sha256 = treeDigest.finalize().map { String(format: "%02x", $0) }.joined()
    return RuntimeWorkerHeadProvenance(
        sha256: sha256, bytes: totalBytes, fileCount: relativePaths.count)
}
