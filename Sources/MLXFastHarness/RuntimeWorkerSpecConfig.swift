import Foundation
import MLXFastCore

// Per-module speculative configuration for the benchd-facing worker protocol.
//
// David-ruled 2026-08-19 (mlxfast-bench/docs/spec-config-design.md): depth is a
// MODULE field, not a benchmarker concept. `decode_begin` gains an optional
// tagged-union `spec` `{ "mode": "serial" | "mtp":{depth} | "dflash":{…} |
// "dspark":{…} }`; the module parses its own block, fills its own defaults,
// bounds it itself, and the engine echoes the resolved block as
// `effective_spec`. `hello.spec_modes` advertises only the modes THIS engine can
// actually run. The benchmarker forwards bytes and seals the echo; it never
// interprets a module block.
//
// ALL measurement lives in benchd (Model 2): this surface only PARSES the spec,
// RESOLVES defaults, and ECHOES the result. It times nothing and scores nothing.
//
// Fail-closed posture, non-negotiable:
//   * unknown fields (top level or inside any module block) → reject;
//   * a block key that does not match the declared mode → reject (config drift);
//   * a hostile / out-of-range depth → reject with a thrown error, never a
//     fatalError and never a silent clamp-to-serial;
//   * a sha256 that is not 64 ASCII hex characters (fullwidth-hex included) →
//     reject at parse time;
//   * a mode this engine cannot run → reject (real module: "not runnable on this
//     engine"; stub module: "not implemented on this engine"), never a silent
//     fall back to serial.
//
// MTP RETURNED 2026-08-23 with the Gemma 4 26B A4B MTP arm. `mtp` is a REAL
// module again (`RuntimeWorkerSpecModuleTable.all`), runnable exactly when this
// worker successfully loaded an assistant head (mtp-head.manifest.json /
// Gemma4MTPHeadDeclaration) at startup — see RuntimeWorkerSpecRegistry.gemma4Worker
// below. A worker that loaded no head advertises only `serial`, the same
// fail-closed shape dflash/dspark already use for a capability-absent real
// module: the mode is registered in the wire vocabulary (so a caller gets "not
// runnable on THIS worker", never "unknown mode"), but is not in
// `advertisedModeStrings` and is refused at resolution.
//
// NOTE ON EXECUTION STATUS (2026-08-23): `mtp` is wired end-to-end — the SPEC
// layer (parse, validate, echo, envelope-knob refusal via
// `Gemma4MTPEnvelope.requirePinned`/`resolveConfig` in MTPEnvelope.swift), the
// assistant-head LOAD layer, and now CBv2 round EXECUTION for both the
// single-stream (`RuntimeWorkerMTPSession`, Gemma4RuntimeMTPDriver.swift) and
// batched cohort (`RuntimeWorkerCohortSession.runMTP`,
// Gemma4RuntimeCohortDriver.swift) free-run paths, driving `EngineV2`'s real
// `mtpDrafter`/`mtpConfig` binding and assembling the free-run counters from
// the engine's own `.delta` events and `mtpMetricsSnapshot()`. One AUDIT-only
// field (the cohort's per-row `natural_accepted_by_stream`) is a documented
// invariant-safe floor rather than an independent observation — the public
// engine surface has no seam for it; see the PR body.

/// The modes every engine registers. Which are *runnable* is per-engine
/// (`RuntimeWorkerSpecRegistry.runnableModes`); which are *implemented anywhere*
/// vs stub is `RuntimeWorkerSpecModule.kind`.
enum RuntimeWorkerSpecMode: String, Codable, CaseIterable {
    case serial
    case mtp
    case dflash
    case dspark
}

/// A module is `real` (implemented on some engine; runnable wherever a worker
/// lists it) or `stub` (parses+validates its block, then fails closed). The
/// table is TOTAL — both engines register all four — so enablement is fill-in,
/// never restructure.
enum RuntimeWorkerSpecModuleKind: Equatable {
    case real
    case stub
}

struct RuntimeWorkerSpecModule: Equatable {
    let mode: RuntimeWorkerSpecMode
    let kind: RuntimeWorkerSpecModuleKind
}

enum RuntimeWorkerSpecModuleTable {
    /// The total module table. `dspark` is the only stub on this Metal engine;
    /// `dflash` is a real module (implemented by the Laguna DFlash worker) that
    /// is simply capability-absent on this worker — the two error differently on
    /// purpose.
    ///
    /// `mtp` RETURNED TO THE TABLE 2026-08-23 with the Gemma 4 26B A4B MTP arm,
    /// as `.real`: this engine's model family genuinely implements MTP (unlike
    /// `dflash`, whose implementation lives in a different worker family
    /// entirely). Whether THIS worker instance can run it is a runtime fact —
    /// did an assistant head load? — handled the same way `dflash`
    /// capability-absence is: `.real` modules are always well-formed spec, and
    /// `RuntimeWorkerSpecRegistry.runnableModes` is what actually gates
    /// resolution.
    static let all: [RuntimeWorkerSpecModule] = [
        RuntimeWorkerSpecModule(mode: .serial, kind: .real),
        RuntimeWorkerSpecModule(mode: .mtp, kind: .real),
        RuntimeWorkerSpecModule(mode: .dflash, kind: .real),
        RuntimeWorkerSpecModule(mode: .dspark, kind: .stub),
    ]

    static func module(for mode: RuntimeWorkerSpecMode) -> RuntimeWorkerSpecModule {
        // Total by construction, so this force-unwrap cannot fail; kept as a
        // guard-throw would hide a table-drift bug behind a wire error.
        all.first { $0.mode == mode }!
    }
}

// MARK: - Wildcard key for strict unknown-field detection

/// A `CodingKey` that accepts any string, so a decoder can enumerate every key
/// actually present and reject the ones a struct does not declare. `Decodable`
/// otherwise ignores unknown keys, which would let a config carry a
/// behaviour-bearing field the module never validated.
private struct RuntimeWorkerSpecWireKey: CodingKey {
    let stringValue: String
    let intValue: Int?
    init?(stringValue: String) { self.stringValue = stringValue; intValue = nil }
    init?(intValue: Int) { stringValue = String(intValue); self.intValue = intValue }
}

private func rejectUnknownSpecKeys(
    _ decoder: Swift.Decoder,
    allowed: Set<String>,
    field: String
) throws {
    let container = try decoder.container(keyedBy: RuntimeWorkerSpecWireKey.self)
    if let unknown = container.allKeys.first(
        where: { !allowed.contains($0.stringValue) }
    ) {
        throw MLXFastError.invalidInput(
            "spec \(field) has unknown field '\(unknown.stringValue)'"
        )
    }
}

/// Validate a wire sha256: exactly 64 ASCII hex characters. Rejects fullwidth
/// hex (e.g. U+FF10) and every other non-ASCII scalar, so a head/artifact
/// identity stays verifiable and can never be silently coarsened. Returns the
/// lowercased canonical form.
func validateRuntimeWorkerWireSHA256(_ value: String, field: String) throws -> String {
    let scalars = Array(value.unicodeScalars)
    guard scalars.count == 64 else {
        throw MLXFastError.invalidInput(
            "spec \(field) sha256 must be 64 hex characters, got \(scalars.count)"
        )
    }
    for scalar in scalars {
        let v = scalar.value
        let isDigit = v >= 0x30 && v <= 0x39      // 0-9
        let isLower = v >= 0x61 && v <= 0x66      // a-f
        let isUpper = v >= 0x41 && v <= 0x46      // A-F
        guard isDigit || isLower || isUpper else {
            throw MLXFastError.invalidInput(
                "spec \(field) sha256 has a non-ASCII or non-hex character "
                    + "(U+\(String(format: "%04X", v)))"
            )
        }
    }
    return value.lowercased()
}

// MARK: - Request-side module blocks (parse + validate their OWN config)

struct RuntimeWorkerDFlashDraftBlock: Codable, Equatable {
    let artifact: String
    let sha256: String

    private enum CodingKeys: String, CodingKey { case artifact, sha256 }

    init(artifact: String, sha256: String) {
        self.artifact = artifact
        self.sha256 = sha256
    }

    init(from decoder: Swift.Decoder) throws {
        try rejectUnknownSpecKeys(
            decoder, allowed: ["artifact", "sha256"], field: "dflash.draft")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        artifact = try c.decode(String.self, forKey: .artifact)
        // sha256 hex is validated at PARSE time so a hostile digest is rejected
        // even for a mode this engine will not run (capability-absent dflash on
        // the Metal engine still parses and validates its full block).
        let raw = try c.decode(String.self, forKey: .sha256)
        sha256 = try validateRuntimeWorkerWireSHA256(raw, field: "dflash.draft")
    }
}

/// The `mtp` block on a `decode_begin` / `free_decode_begin` `spec`. `depth` is
/// OPTIONAL and, when present, an upper bound the module clamps into its own
/// tested range (`Gemma4MTPEnvelope.resolveDepth`, MTPEnvelope.swift) — never a
/// value that reaches the vendored `CBv2MTPConfig` unclamped. A negative or
/// absurd depth is not a parse error (unlike a malformed sha256): it is a
/// request for "as much speculation as this module allows", clamped, not
/// refused, because depth is advisory and the wire's `effective_spec` is the
/// authority on what actually ran.
struct RuntimeWorkerMTPBlock: Codable, Equatable {
    let depth: Int?

    private enum CodingKeys: String, CodingKey { case depth }

    init(depth: Int?) {
        self.depth = depth
    }

    init(from decoder: Swift.Decoder) throws {
        try rejectUnknownSpecKeys(decoder, allowed: ["depth"], field: "mtp")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decodeIfPresent(Int.self, forKey: .depth)
    }
}

struct RuntimeWorkerDFlashBlock: Codable, Equatable {
    let depth: Int?
    let draft: RuntimeWorkerDFlashDraftBlock?

    private enum CodingKeys: String, CodingKey { case depth, draft }

    init(depth: Int?, draft: RuntimeWorkerDFlashDraftBlock?) {
        self.depth = depth
        self.draft = draft
    }

    init(from decoder: Swift.Decoder) throws {
        try rejectUnknownSpecKeys(
            decoder, allowed: ["depth", "draft"], field: "dflash")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decodeIfPresent(Int.self, forKey: .depth)
        draft = try c.decodeIfPresent(
            RuntimeWorkerDFlashDraftBlock.self, forKey: .draft)
    }
}

/// DSpark's schema is RESERVED pending cudafast#26. The stub still parses and
/// validates its block so schema enforcement is uniform across engines: today
/// the block has no known fields, so any field is unknown and rejected, and an
/// empty block is accepted — then resolution fails closed with
/// "not implemented on this engine".
struct RuntimeWorkerDSparkBlock: Codable, Equatable {
    init() {}

    init(from decoder: Swift.Decoder) throws {
        try rejectUnknownSpecKeys(decoder, allowed: [], field: "dspark")
    }
}

// MARK: - Request-side tagged union

/// The `spec` object on a `decode_begin` request: exactly one mode, the config
/// nested under the mode key. Cross-module keys and unknown keys are rejected
/// here, before any block is interpreted.
struct RuntimeWorkerSpecRequest: Codable, Equatable {
    let mode: RuntimeWorkerSpecMode
    let mtp: RuntimeWorkerMTPBlock?
    let dflash: RuntimeWorkerDFlashBlock?
    let dspark: RuntimeWorkerDSparkBlock?

    private enum CodingKeys: String, CodingKey {
        case mode, mtp, dflash, dspark
    }

    init(
        mode: RuntimeWorkerSpecMode,
        mtp: RuntimeWorkerMTPBlock? = nil,
        dflash: RuntimeWorkerDFlashBlock? = nil,
        dspark: RuntimeWorkerDSparkBlock? = nil
    ) {
        self.mode = mode
        self.mtp = mtp
        self.dflash = dflash
        self.dspark = dspark
    }

    init(from decoder: Swift.Decoder) throws {
        try rejectUnknownSpecKeys(
            decoder,
            allowed: ["mode", "mtp", "dflash", "dspark"],
            field: "root"
        )
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let rawMode = try c.decode(String.self, forKey: .mode)
        guard let parsedMode = RuntimeWorkerSpecMode(rawValue: rawMode) else {
            throw MLXFastError.invalidInput(
                "spec mode '\(rawMode)' is not one of "
                    + RuntimeWorkerSpecMode.allCases.map(\.rawValue)
                    .joined(separator: ", ")
            )
        }
        mode = parsedMode

        // Cross-module rejection: the ONLY block key allowed is the one that
        // matches the declared mode. `{"mode":"serial","dflash":{…}}` is an error,
        // not an ignored field — this is checked BEFORE the matching block is
        // decoded so config drift never reaches a module's validator.
        let wire = try decoder.container(keyedBy: RuntimeWorkerSpecWireKey.self)
        for key in wire.allKeys where key.stringValue != "mode" {
            guard key.stringValue == parsedMode.rawValue else {
                throw MLXFastError.invalidInput(
                    "spec declares mode '\(parsedMode.rawValue)' but carries a "
                        + "'\(key.stringValue)' block; a mode may carry only its "
                        + "own block"
                )
            }
        }

        mtp = parsedMode == .mtp
            ? try c.decodeIfPresent(RuntimeWorkerMTPBlock.self, forKey: .mtp)
            : nil
        dflash = parsedMode == .dflash
            ? try c.decodeIfPresent(RuntimeWorkerDFlashBlock.self, forKey: .dflash)
            : nil
        dspark = parsedMode == .dspark
            ? try c.decodeIfPresent(RuntimeWorkerDSparkBlock.self, forKey: .dspark)
            : nil
    }
}

// MARK: - Effective spec (the echo)

/// The module-parsed, default-filled block the engine will actually run. This is
/// the ONLY thing provenance seals — provenance is what the engine acknowledged,
/// not what the caller asked. Optional blocks are omitted when absent, so
/// `serial` echoes exactly `{"mode":"serial"}`.
struct RuntimeWorkerEffectiveSpec: Codable, Equatable {
    let mode: String
    let mtp: EffectiveMTP?
    let dflash: EffectiveDFlash?

    /// The resolved `mtp` echo: `{"mtp":{"depth":<clamped>}}`. Depth is always
    /// present and always the ACTUAL clamped value the module will run — never
    /// the caller's raw request — because this echo is what a security reviewer
    /// (and benchd's own oracle replay) reads to know what the worker
    /// acknowledged, not what it was asked.
    struct EffectiveMTP: Codable, Equatable {
        let depth: Int
    }

    struct EffectiveDFlash: Codable, Equatable {
        let depth: Int
        let draft: EffectiveDraft?

        struct EffectiveDraft: Codable, Equatable {
            let artifact: String
            let sha256: String
        }
    }

    static func serial() -> RuntimeWorkerEffectiveSpec {
        RuntimeWorkerEffectiveSpec(mode: "serial", mtp: nil, dflash: nil)
    }

    static func mtp(depth: Int) -> RuntimeWorkerEffectiveSpec {
        RuntimeWorkerEffectiveSpec(
            mode: "mtp", mtp: EffectiveMTP(depth: depth), dflash: nil)
    }
}

// MARK: - Per-engine registry + resolver

/// The set of modes ONE worker can run, plus resolution of a request into the
/// effective echo. `advertisedModeStrings` is `hello.spec_modes` — runnable
/// modes only; a stub (dspark) is visible in the module table but never here.
struct RuntimeWorkerSpecRegistry {
    let runnableModes: [RuntimeWorkerSpecMode]
    /// The engine default when a request carries no `spec` at all.
    let defaultMode: RuntimeWorkerSpecMode
    /// This worker's DFlash capability. `.available` is what puts `.dflash`
    /// in `runnableModes`; the other two cases shape the refusal.
    var dflash: DFlashCapability = .absent

    /// This engine's worker with NO assistant head loaded: serial is the only
    /// runnable mode. `dflash` is a real module but capability-absent here (its
    /// model-side worker went with the vendored adoption); `dspark` is a stub;
    /// `mtp` is registered (`.real`, see `RuntimeWorkerSpecModuleTable`) but not
    /// runnable without a head — resolving it throws the same
    /// "not runnable on this engine" error `dflash` already produces. Default is
    /// serial. Retained under its historical name (some call sites and tests
    /// predate the MTP arm and construct a headless worker deliberately) —
    /// prefer `gemma4Worker(mtpAvailable:)` for new call sites so the choice is
    /// explicit rather than implied by the name.
    static let serialOnlyWorker = RuntimeWorkerSpecRegistry(
        runnableModes: [.serial],
        defaultMode: .serial
    )

    /// The registry this worker actually runs with, decided ONCE at startup by
    /// whether an assistant head loaded (`Gemma4AssistantHeadLoad`,
    /// AssistantHead.swift). `mtpAvailable` is never re-derived per-request —
    /// the hello's `spec_modes` and every subsequent resolution must agree with
    /// what was decided before the first byte was read, or a caller could see
    /// `mtp` advertised and then refused mid-session.
    ///
    /// Default mode stays `serial` even when a head loaded: MTP is opt-in per
    /// request (`{"mode":"mtp"}`), never a silent default — an absent `spec`
    /// must keep meaning "plain decode" regardless of what this worker is
    /// capable of, because that is the serial CONTROL leg's whole contract.
    ///
    /// DFLASH ARM (2026-08-25). `dflash` is runnable iff this worker bound a
    /// REAL `DFlashDraftModel` from `dflash-head/` at startup
    /// (`loadGemma4DFlashHeadIfStaged`), a runtime fact decided once, exactly
    /// like `mtpAvailable`. Passing the capability as a VALUE rather than a
    /// Bool is what lets the echo be honest: the resolved depth is clamped by
    /// the bound drafter's own block width, and the echoed draft digest is
    /// the harness's recomputed tree digest of the head that will actually
    /// draft — neither is knowable from a Bool.
    ///
    /// `unavailableReason` carries a PRESENT-BUT-BROKEN `dflash-head/`
    /// (`Gemma4DFlashHeadLoadOutcome.incompatible`). Startup is fail-soft
    /// there — the worker must reach hello with serial + mtp intact — so the
    /// refusal lands HERE instead, at spec-resolution time, quoting the real
    /// load failure rather than the generic "not runnable on this engine".
    static func gemma4Worker(
        mtpAvailable: Bool,
        dflash: DFlashCapability = .absent
    ) -> RuntimeWorkerSpecRegistry {
        var modes: [RuntimeWorkerSpecMode] = [.serial]
        if mtpAvailable { modes.append(.mtp) }
        if case .available = dflash { modes.append(.dflash) }
        return RuntimeWorkerSpecRegistry(
            runnableModes: modes,
            defaultMode: .serial,
            dflash: dflash
        )
    }

    /// This worker's DFlash capability, decided once at startup.
    enum DFlashCapability: Equatable {
        /// Nothing staged at `dflash-head/`.
        case absent
        /// Staged but unloadable; `reason` names the actual defect.
        case broken(reason: String)
        /// A drafter is bound. `maxDepth` is its own block ceiling
        /// (`Gemma4DFlashHeadLoadResult.maxDepth`); `provenanceSHA256` is the
        /// harness's recomputed tree digest over `dflash-head/`.
        case available(maxDepth: Int, provenanceSHA256: String)
    }

    var advertisedModeStrings: [String] {
        runnableModes.map(\.rawValue)
    }

    /// Resolve a request's spec into the effective echo, or throw fail-closed.
    ///
    /// `nil` request → the engine default, default-filled (v1 callers that never
    /// send a spec are unchanged and valid). A present spec that names a stub,
    /// or a real mode this worker cannot run, throws — never a silent serial
    /// fall back, never a silently-different echo.
    func resolveEffectiveSpec(
        _ request: RuntimeWorkerSpecRequest?
    ) throws -> RuntimeWorkerEffectiveSpec {
        guard let request else {
            return try effectiveFor(mode: defaultMode, request: nil)
        }
        let module = RuntimeWorkerSpecModuleTable.module(for: request.mode)
        if module.kind == .stub {
            // The block was already parsed+validated by decoding; now fail
            // closed with the distinct stub error.
            throw MLXFastError.invalidInput(
                "spec mode '\(request.mode.rawValue)' is not implemented on this "
                    + "engine"
            )
        }
        guard runnableModes.contains(request.mode) else {
            // PRESENT-BUT-BROKEN DFLASH HEAD. Startup is fail-soft (the
            // worker must reach hello with serial + mtp intact), so this is
            // where the refusal for a staged-but-unloadable `dflash-head/`
            // actually lands. Name the real defect — the caller asked to run
            // DFlash and is entitled to know why the staged head could not
            // be loaded, not just that the mode is missing from the list.
            if request.mode == .dflash, case .broken(let reason) = dflash {
                throw MLXFastError.invalidInput(
                    "spec mode 'dflash' is not runnable on this engine: a "
                        + "dflash-head/ IS staged but could not be loaded — "
                        + "\(reason)"
                )
            }
            throw MLXFastError.invalidInput(
                "spec mode '\(request.mode.rawValue)' is not runnable on this "
                    + "engine; runnable modes are "
                    + advertisedModeStrings.joined(separator: ", ")
            )
        }
        return try effectiveFor(mode: request.mode, request: request)
    }

    private func effectiveFor(
        mode: RuntimeWorkerSpecMode,
        request: RuntimeWorkerSpecRequest?
    ) throws -> RuntimeWorkerEffectiveSpec {
        switch mode {
        case .serial:
            return .serial()
        case .mtp:
            // Envelope-knob refusal FIRST, before any depth is resolved or
            // echoed: `Gemma4MTPEnvelope.requirePinned()` (MTPEnvelope.swift)
            // throws by name if `maxAutomaticRectangularTokens` or the
            // attention query-block pin is unset. This registry separately
            // decided RUNNABILITY (was a head loaded? — `runnableModes`
            // above), so the two refusal reasons ("no head" vs "envelope pin
            // missing") stay distinguishable: an unrunnable-mode request never
            // reaches this function at all (caught by the `runnableModes`
            // guard in `resolveEffectiveSpec`), while a runnable-but-unpinned
            // request reaches here and is refused by name.
            try Gemma4MTPEnvelope.requirePinned()
            let requestedDepth = request?.mtp?.depth
            let depth = Gemma4MTPEnvelope.resolveDepth(requestedDepth)
            return .mtp(depth: depth)
        case .dflash:
            // Reached only on a worker that already decided `dflash` RUNNABLE
            // (`runnableModes`), i.e. one that bound a real DFlashDraftModel.
            //
            // NO `Gemma4MTPEnvelope.requirePinned()` HERE, unlike `mtp`
            // above, and that is deliberate rather than an omission. Both of
            // that envelope's pins are CBv2 knobs —
            // `maxAutomaticRectangularTokens` gates
            // `CBv2MTPRoundDriver`'s rectangular verify, and
            // `attentionQueryBlockPin` gates `CBv2AttentionV1.queryBlockSize`
            // — and the DFlash arm runs neither: its rounds execute in
            // `RuntimeWorkerDFlashFreeRunSession` over plain `[KVCache]`, not
            // in `EngineV2`. #38 required the pins because it (incorrectly)
            // ran DFlash through the CBv2 MTP driver; with the real drafter
            // that coupling is gone, and asserting an unrelated pin would
            // refuse by a reason that does not describe this arm. DFlash's
            // own precondition — a bound drafter — is `runnableModes`.
            guard case .available(let maxDepth, let provenanceSHA256) = dflash else {
                throw MLXFastError.invalidInput(
                    "spec mode 'dflash' resolved on a worker with no bound "
                        + "DFlash drafter; runnability and capability disagree "
                        + "(wiring bug)"
                )
            }
            let block = request?.dflash
            // The echo must be the depth the round loop will ACTUALLY run.
            // `RuntimeWorkerDFlashFreeRunSession` is opened with exactly this
            // value and runs `blockSize = depth + 1`, so the two cannot
            // drift. (#38 echoed `experimentalDFlashMaxBlockSize` = 16 while
            // executing an envelope-clamped 3.)
            let depth = Self.resolveDFlashDepth(block?.depth, maxDepth: maxDepth)
            // Draft identity is the BOUND drafter's, never the caller's.
            // A caller-declared `sha256` is VERIFIED against the harness's
            // own recomputed tree digest over `dflash-head/` and refused on
            // mismatch; the echo then always carries the digest of the head
            // that will actually draft. Echoing an unverified caller string
            // would let a run claim a drafter identity it is not using.
            if let declared = block?.draft?.sha256,
                declared.lowercased() != provenanceSHA256.lowercased()
            {
                throw MLXFastError.invalidInput(
                    "spec dflash.draft.sha256 declares '\(declared)' but the "
                        + "drafter bound from \(gemma4DFlashHeadDirectoryName)/ "
                        + "hashes to '\(provenanceSHA256)'; refusing rather than "
                        + "echoing a draft identity this run is not using"
                )
            }
            return RuntimeWorkerEffectiveSpec(
                mode: "dflash",
                mtp: nil,
                dflash: RuntimeWorkerEffectiveSpec.EffectiveDFlash(
                    depth: depth,
                    draft: RuntimeWorkerEffectiveSpec.EffectiveDFlash.EffectiveDraft(
                        artifact: gemma4DFlashHeadDirectoryName,
                        sha256: provenanceSHA256))
            )
        case .dspark:
            throw MLXFastError.invalidInput(
                "spec mode 'dspark' is not implemented on this engine"
            )
        }
    }

    /// Clamp a requested DFlash draft depth (k, tokens PROPOSED per round —
    /// the round's block is `1 + k`) into `1...maxDepth`, where `maxDepth`
    /// is the bound drafter's own ceiling
    /// (`Gemma4DFlashHeadLoadResult.maxDepth`: its
    /// `recommendedBlockSize - 1`, itself capped by
    /// `MLXFastConstants.experimentalDFlashMaxBlockSize - 1`). An absent
    /// depth means "use everything this drafter supports", matching the mtp
    /// arm's convention.
    ///
    /// The ONE resolver: `effectiveFor(.dflash)` echoes its result and
    /// `RuntimeWorkerDFlashFreeRunSession` runs it, so the echoed depth is
    /// the executed depth by construction.
    static func resolveDFlashDepth(_ requested: Int?, maxDepth: Int) -> Int {
        let ceiling = Swift.max(1, maxDepth)
        guard let requested else { return ceiling }
        return Swift.max(1, Swift.min(requested, ceiling))
    }
}

// MARK: - Head provenance (sha256 / bytes / file_count) on the worker path

/// Head identity a worker seals so benchd can verify which head drafted. The
/// sha256 is the FULL-WIDTH 64-hex tree digest, never coarsened.
///
/// RETAINED, ALWAYS NIL on this engine (2026-08-22). It is a WIRE-SCHEMA type:
/// benchd decodes `head_provenance` off the v1.1 hello, so the field stays
/// declared and an engine that loads no head sends `null` — which is a true
/// statement about the run. The tree-digest helper that used to fill it went
/// with the MTP arm, because a digest of a head nothing loads is not evidence.
struct RuntimeWorkerHeadProvenance: Codable, Equatable {
    let sha256: String
    let bytes: Int
    let fileCount: Int

    enum CodingKeys: String, CodingKey {
        case sha256
        case bytes
        case fileCount = "file_count"
    }
}

