import Foundation
import Testing

// Structural checks on the Yukon track manifest (benchmark.json) for track
// gemma4-26b-a4b-mlx-v1. This is NOT a re-implementation of Yukon's zod
// schema (src/benchmark/manifest.ts, RawBenchmarkConfigSchema /
// parseRepositoryBenchmarkManifest) -- that schema was validated directly by
// running `bun run` against the real yukon checkout
// ($HOME/projects/layr-labs/.worktrees/yukon-gemma-onboarding),
// importing parseRepositoryBenchmarkManifest and feeding it this file's own
// benchmark.json at authoring time, which returned PARSE OK with
// schemaVersion 1 and the expected editablePathsCount. That is a one-time,
// external verification method (recorded here so it can be re-run, not
// because this suite re-implements it) and is out of this repository's swift
// test loop because it needs the yukon checkout and its own bun/npm
// dependency tree, neither of which this repository depends on or vendors.
//
// What THIS suite pins, so a manifest edit cannot silently drift from the
// tree it describes or from tools/lint-benchmark-manifest.py's own rules
// without a swift test failure calling it out:
//   - benchmark.json parses as JSON and required top-level keys are present
//     with the right JSON type (a lightweight mirror of the zod schema's
//     shape requirements, not a full re-implementation).
//   - every editablePaths / optionalEditablePaths entry exists in this repo's
//     tree (repo-root relative, same as the Yukon importer resolves them).
//   - optionalEditablePaths is a subset of editablePaths.
//   - no two editablePaths entries are duplicates or nest inside each other.
//   - scorePath is not inside any editablePaths entry.
//   - runner.workflow names a real file under .github/workflows/.
//   - contractPath exists, parses as JSON, and its track_id equals trackId.
//   - editableSurfaceByteBudget's four numeric caps equal the compiled-in
//     EditableSurfaceByteBudget defaults (manifest/enforcer drift guard,
//     mirroring tools/lint-benchmark-manifest.py check 3b).
//
// Scope note: no MLX device work, no weights, no GPU, no network, and the pinned
// benchctl is NOT required to be resolved (benchd-bin/ need not exist) -- every
// assertion below is static content / filesystem existence, exactly like
// HarnessHashRootSetTests's own scope note.

private func repoRootRelative(_ path: String) -> String { path }

private func loadJSONObject(atRepoPath path: String) throws -> [String: Any] {
    let data = try Data(contentsOf: URL(fileURLWithPath: repoRootRelative(path)))
    let object = try JSONSerialization.jsonObject(with: data)
    guard let dict = object as? [String: Any] else {
        Issue.record("\(path) did not decode to a JSON object")
        return [:]
    }
    return dict
}

@Suite("Gemma4 benchmark.json manifest")
struct Gemma4BenchmarkManifestTests {

    @Test
    func manifestParsesAndCarriesRequiredKeys() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")

        // Mirrors RawBenchmarkManifestV1Schema's required keys (yukon
        // src/benchmark/manifest.ts) plus this repository's own convention
        // of always carrying a "scoring" object (tools/lint-benchmark-
        // manifest.py SCHEMA, line ~113: "scoring": dict).
        let requiredStringKeys = [
            "name", "description", "category", "direction", "scorePath", "contractPath",
            "staticReviewTrackId", "trackId",
        ]
        for key in requiredStringKeys {
            #expect(manifest[key] is String, "benchmark.json[\(key)] must be a string")
        }

        #expect(manifest["schemaVersion"] as? Int == 1)
        #expect(manifest["direction"] as? String == "+")
        #expect(manifest["category"] as? String == "swift")
        #expect(manifest["trackId"] as? String == "gemma4-26b-a4b-mlx-v1")

        #expect(manifest["editablePaths"] is [String])
        #expect((manifest["editablePaths"] as? [String])?.isEmpty == false)
        #expect(manifest["setupCommand"] is [String])
        #expect(manifest["benchmarkCommand"] is [String])
        #expect(manifest["scoring"] is [String: Any])

        let runner = try #require(manifest["runner"] as? [String: Any])
        #expect(runner["provider"] as? String == "github-actions")
        #expect(runner["workflow"] as? String == "benchmark.yml")
    }

    @Test
    func everyEditablePathExistsInTree() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        // 94 -> 92 on 2026-08-26 (David requant-only ruling): the two head
        // WEIGHTS DIRECTORIES leave the editable surface. `mtp-head/` and
        // `dflash-head/` are organizer-staged, so a submission has no reason to
        // carry bytes there and no way to. The two declaration files
        // (`mtp-head.manifest.json`, `spec-decoder-head.manifest.json`) STAY editable:
        // they are how a participant declares a re-quantization of the pinned
        // head. 94 was the 2026-08-25 count, when the DFlash drafter joined as a
        // second REPLACEABLE head; replacement is retired, re-quantization is
        // what remains.
        //
        // 92 -> 93 on 2026-08-26 (David: "dflash needs to be editable"): the
        // DFlash drafter MODEL file
        // (`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`)
        // joins, so the contract's promise that either head may be re-quantized
        // is reachable for BOTH heads. It carries `load(from:bindTo:)` and the
        // `quantize(model:)` call that binds the drafter's geometry, which is
        // the same seam `Gemma4MTP.swift` already gives the MTP head. One model
        // file, exactly as for MTP: the loader in `Sources/MLXFastHarness` stays
        // trusted, and the head WEIGHTS directory `dflash-head/` stays
        // non-editable -- this adds CODE, never an upload surface.
        #expect(editablePaths.count == 93, "editablePaths count drifted; update this pin deliberately if the surface changed")

        let fm = FileManager.default
        for path in editablePaths {
            #expect(fm.fileExists(atPath: path), "editablePaths entry does not exist in tree: \(path)")
        }
    }

    @Test
    func optionalEditablePathsIsSubsetOfEditablePaths() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = Set(try #require(manifest["editablePaths"] as? [String]))
        let optionalPaths = try #require(manifest["optionalEditablePaths"] as? [String])
        #expect(!optionalPaths.isEmpty)
        for path in optionalPaths {
            #expect(editablePaths.contains(path), "optionalEditablePaths entry \(path) is not also an editablePaths entry")
        }
    }

    @Test
    func editablePathsHaveNoDuplicatesOrNesting() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])

        #expect(Set(editablePaths).count == editablePaths.count, "editablePaths contains a duplicate entry")

        func isEqualOrDescendant(_ candidate: String, of other: String) -> Bool {
            candidate == other || candidate.hasPrefix(other + "/")
        }

        for i in editablePaths.indices {
            for j in editablePaths.indices where i != j {
                let a = editablePaths[i]
                let b = editablePaths[j]
                #expect(
                    !isEqualOrDescendant(a, of: b),
                    "editablePaths entries overlap/nest: \(a) is inside \(b)"
                )
            }
        }
    }

    @Test
    func scorePathIsNotInsideAnyEditablePath() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let scorePath = try #require(manifest["scorePath"] as? String)

        for editable in editablePaths {
            let insideEditable = scorePath == editable || scorePath.hasPrefix(editable + "/")
            #expect(!insideEditable, "scorePath \(scorePath) is inside editablePaths entry \(editable)")
        }
    }

    @Test
    func runnerWorkflowFileExists() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let runner = try #require(manifest["runner"] as? [String: Any])
        let workflow = try #require(runner["workflow"] as? String)
        let path = ".github/workflows/\(workflow)"
        #expect(FileManager.default.fileExists(atPath: path), "runner.workflow does not resolve to a real file: \(path)")
    }

    @Test
    func contractPathExistsAndTrackIdMatches() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let contractPath = try #require(manifest["contractPath"] as? String)
        #expect(FileManager.default.fileExists(atPath: contractPath))

        let contract = try loadJSONObject(atRepoPath: contractPath)
        let contractTrackId = try #require(contract["track_id"] as? String)
        let manifestTrackId = try #require(manifest["trackId"] as? String)
        #expect(contractTrackId == manifestTrackId)
    }

    @Test
    func byteBudgetMatchesTrustedEnforcerDefaults() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])

        // Mirrors Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift's
        // defaultMaxTotalBytes / defaultMaxFileBytes / defaultMaxGrowthBytes /
        // defaultExemptPathMaxBytes -- read directly off that file at authoring
        // time (values verified there, not re-derived here). tools/lint-
        // benchmark-manifest.py check 3b enforces the same equality; this test
        // gives the same guard a swift-test-visible failure.
        //
        // maxTotalBytes raised 2026-08-24 (3,000,000 -> 4,404,587) in the same
        // commit that added the ContinuousBatchingV2 batching/round-driving
        // engine to editablePaths (David ruling). It is a computed
        // minimum-plus-margin over the ENFORCED (non-exempt) at-rest surface,
        // with a stated margin of at least 1 MiB (1,048,576 B = 4x
        // maxGrowthBytes). Two earlier values are superseded: a
        // ~40%-headroom-ratio derivation (4,700,000) and a first minimum-
        // plus-margin pass that double-counted the then-exempt mtp-head/ bytes
        // into the at-rest figure (4,407,216).
        //
        // UNCHANGED by the 2026-08-26 requant-only ruling, and that is the
        // point: `mtp-head/` and `dflash-head/` were already EXEMPT, so they
        // paid into exemptPathMaxBytes and contributed exactly zero to this
        // cap. Dropping them from editablePaths therefore moves the enforced
        // at-rest total by zero bytes. `enforcedSurfaceStaysUnderTotalCap`
        // below re-derives that from the tree rather than restating it.
        #expect(budget["maxTotalBytes"] as? Int == 4_404_587)
        #expect(budget["maxFileBytes"] as? Int == 524_288)
        #expect(budget["maxGrowthBytes"] as? Int == 262_144)

        // THE TWO EXEMPT CAPS ARE KEPT, DELIBERATELY, WITH NOTHING EXEMPT.
        // `exemptPaths` is GONE (asserted below), so neither cap can bind
        // today. They stay DECLARED because both enforcers still carry them as
        // compiled-in fallbacks --
        // EditableSurfaceByteBudget.defaultExemptPathMaxBytes /
        // .defaultExemptPathMaxFileBytes, and benchd's
        // DEFAULT_EXEMPT_PATH_MAX_BYTES / DEFAULT_EXEMPT_PATH_MAX_FILE_BYTES --
        // and the manifest is what pins those constants to a reviewed number
        // (tools/lint-benchmark-manifest.py check 3b, BUDGET_CAP_KEYS).
        // Deleting the declaration would delete that drift check and leave the
        // constants as the cap with nothing checking them, which is the
        // single-source design inverted. Keeping an inert-but-checked cap costs
        // nothing and refuses nothing.
        #expect(budget["exemptPathMaxBytes"] as? Int == 512_000_000)
        #expect(budget["exemptPathMaxFileBytes"] as? Int == 100_000_000)

        // NOTHING IS EXEMPT (David requant-only ruling, 2026-08-26). An exempt
        // path only ever existed to let head WEIGHTS ride in a submission
        // outside the 4.4 MB source budget. A submission no longer carries head
        // weights at all, so an exemption would be an exemption from nothing --
        // and a stale `exemptPaths` naming a path that is no longer editable is
        // a hard failure in tools/lint-benchmark-manifest.py (exemptPaths must
        // be a subset of editablePaths), not a harmless leftover.
        #expect(budget["exemptPaths"] == nil, "editableSurfaceByteBudget.exemptPaths must be absent: nothing rides in a submission that is exempt from the code budget")
    }

    @Test
    func headWeightDirectoriesAreNotEditable() throws {
        // THE ENFORCEMENT HALF OF THE REQUANT-ONLY RULING (David, 2026-08-26).
        // The two speculative heads are the organizer's pinned weights. A
        // participant may declare a re-quantization of them; a participant may
        // not upload head weights. What makes that real is this list: with the
        // directories out of editablePaths,
        // .github/scripts/enforce-modifiable-surface.sh refuses any changed file
        // under them, and .github/scripts/overlay-editable-paths.sh never copies
        // one into the measured tree.
        //
        // The DECLARATION files stay editable. They are the recipe surface, and
        // they carry no weights.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let optionalPaths = try #require(manifest["optionalEditablePaths"] as? [String])
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])
        let exemptPaths = (budget["exemptPaths"] as? [String]) ?? []

        let headWeightDirectories = ["mtp-head", "dflash-head"]
        for bucket in [
            ("editablePaths", editablePaths),
            ("optionalEditablePaths", optionalPaths),
            ("editableSurfaceByteBudget.exemptPaths", exemptPaths),
        ] {
            for entry in bucket.1 {
                for directory in headWeightDirectories {
                    #expect(
                        entry != directory && !entry.hasPrefix(directory + "/"),
                        "\(bucket.0) entry \(entry) reaches head weights directory \(directory); a submission must not be able to carry head weights"
                    )
                }
            }
        }

        // POSITIVE DISCRIMINATOR. The rule is "no head WEIGHTS", not "no head
        // declaration": a requant submission has to have something to edit.
        for declaration in ["mtp-head.manifest.json", "spec-decoder-head.manifest.json"] {
            #expect(editablePaths.contains(declaration), "\(declaration) must stay editable: it is how a re-quantization is declared")
            #expect(optionalPaths.contains(declaration), "\(declaration) must stay optional: an absent declaration selects the organizer-pinned head")
        }
    }

    @Test
    func enforcedSurfaceStaysUnderTotalCap() throws {
        // Re-derives the maxTotalBytes margin from the tree instead of trusting
        // the comment above it. With `exemptPaths` gone, every editable path is
        // now ENFORCED, so this is the whole walk -- and it must still clear the
        // declared cap with the stated 1 MiB margin. If the head directories had
        // been left editable-but-no-longer-exempt, this is the assertion that
        // would have caught it the moment a staged head landed in the tree.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])
        let maxTotalBytes = try #require(budget["maxTotalBytes"] as? Int)

        let fm = FileManager.default
        var totalBytes = 0
        for path in editablePaths {
            var isDirectory = ObjCBool(false)
            guard fm.fileExists(atPath: path, isDirectory: &isDirectory) else { continue }
            if !isDirectory.boolValue {
                totalBytes += (try fm.attributesOfItem(atPath: path)[.size] as? Int) ?? 0
                continue
            }
            guard let enumerator = fm.enumerator(atPath: path) else { continue }
            // The entry NAME is not used -- `fileAttributes` describes whatever
            // the enumerator last returned -- so the call advances the walk and
            // its result is deliberately discarded rather than bound.
            while enumerator.nextObject() != nil {
                guard enumerator.fileAttributes?[.type] as? FileAttributeType == .typeRegular else {
                    continue
                }
                totalBytes += (enumerator.fileAttributes?[.size] as? NSNumber)?.intValue ?? 0
            }
        }

        #expect(totalBytes < maxTotalBytes, "enforced at-rest editable surface (\(totalBytes) B) is above maxTotalBytes (\(maxTotalBytes) B)")
        #expect(
            maxTotalBytes - totalBytes >= 1_048_576,
            "maxTotalBytes leaves \(maxTotalBytes - totalBytes) B of margin over the \(totalBytes) B at-rest surface, below the stated 1 MiB minimum"
        )
    }

    @Test
    func benchdGitlinkPathIsNeverEditable() throws {
        // Load-bearing exclusion (tools/lint-benchmark-manifest.py check 3):
        // no editablePaths entry may cover the pinned measurement harness at any
        // prefix depth -- an editable entry there would let a submission repoint
        // the thing that scores it at its own code.
        //
        // benchd used to be a SOURCE submodule (`benchd` + `.gitmodules`); it is
        // now a pinned PREBUILT binary (`benchd.pin` naming {branch, commit,
        // sha256, bytes}, resolved into `benchd-bin/` by tools/fetch-benchd.sh),
        // so the pin and the resolved-binary directory are the LIVE entries. The
        // two submodule spellings are kept so a reintroduced gitlink is covered
        // on arrival rather than after someone remembers to re-add it.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        for path in editablePaths {
            #expect(path != "benchd" && !path.hasPrefix("benchd/"), "editablePaths entry reaches the benchd gitlink: \(path)")
            #expect(path != ".gitmodules", "editablePaths entry reaches .gitmodules: \(path)")
            #expect(path != "benchd.pin", "editablePaths entry reaches the benchd pin: \(path)")
            #expect(path != "benchd-bin" && !path.hasPrefix("benchd-bin/"), "editablePaths entry reaches the resolved benchctl: \(path)")
        }
    }
}
