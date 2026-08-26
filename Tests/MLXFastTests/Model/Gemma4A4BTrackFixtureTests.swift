import Foundation
import MLXFastCore
@testable import MLXFastModel
import Testing

// Contract tests for `fixtures/gemma4_26b_a4b_track.json` (lane/gemma4-track-fixture).
// Mirrors the coverage the qwen contract fixture has no dedicated Swift test for today
// (fixtures/qwen3_8_27b_mtp_track.json is read only incidentally, by
// `Qwen35ArtifactFixtureSupport.swift`, not asserted field-by-field in this repository) --
// this file is written to the same standard the CLAUDE.md task brief asks for: the
// fixture parses, its pinned fields are exact, its sentinels are exact-match detected as
// unarmed, `scored_exponents` is present and equal to the ruled pair, and
// `official_scoring_enabled` is false.
//
// This is a laptop-side JSON-shape test only. It does NOT exercise benchd's Rust
// `Contract`/`ScoredExponents`/`golden` parsers directly -- those live in the separate
// `benchd` submodule/repository and are cross-checked out-of-band (see the PR
// description for the `cargo test` verification run against both the pinned gitlink
// dfd801f9 and the release-branch tip ea12c02e).

@Suite("Gemma 4 26B A4B track contract fixture")
struct Gemma4A4BTrackFixtureTests {

    /// The sentinel PENDING-ORGANIZER marker every unarmed identity slot in this fixture
    /// carries, per this repository's standing convention (mirrors `QWEN38-PENDING-RELEASE`
    /// / `QWEN-MTP-CUDA-PENDING-ORGANIZER`). EXACT-MATCH ONLY -- never a prefix check.
    static let pendingOrganizerSentinel = "GEMMA4-MLX-PENDING-ORGANIZER"

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try gemma4A4BTrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the release-branch / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try gemma4A4BTrackContractObject()
        #expect(object["track_id"] as? String == "gemma4-26b-a4b-mlx-v1")
    }

    @Test("track_id is substring-clean against every retired Gemma-era MTP name (AGENTS.md:1008-1009, OQ-4)")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try gemma4A4BTrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(!trackId.contains(name), "track_id must not substring-collide with retired name \(name)")
            #expect(!name.contains(trackId), "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// ARMED. This test asserted `false` until PR #65 flipped the fixture on
    /// David's "arm now" ruling and did not update it, which left `main` red.
    /// Repaired here rather than left for a separate lane: this PR edits the
    /// same fixture, so its own suite cannot be green while this is not.
    @Test("official_scoring_enabled is true — the track is armed (David ruling, PR #65)")
    func officialScoringEnabled() throws {
        let object = try gemma4A4BTrackContractObject()
        #expect(object["official_scoring_enabled"] as? Bool == true)
    }

    /// MODE FENCE (David ruling 2026-08-26) — the track declares the modes a
    /// submission may run, and `dflash` is one of them. The benchmarker's fence
    /// is contract-driven: absent, it falls back to `[serial, mtp]` and refuses
    /// dflash, so this declaration is the whole of what arms the DFlash arm.
    ///
    /// `serial` must be present because the baseline leg is pinned serial and is
    /// validated against this same list.
    @Test("allowed_modes declares the scored modes, including dflash")
    func allowedModesDeclaresDFlash() throws {
        let object = try gemma4A4BTrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp", "dflash"])
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try gemma4A4BTrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    @Test("scored_batch_size is pinned to the ruled B=8 cohort width")
    func scoredBatchSizeIsEight() throws {
        let object = try gemma4A4BTrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 8)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try gemma4A4BTrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct exactly
        // (crates/benchctl/src/measure_job.rs @ ea12c02e) -- NOT the shorthand
        // "prefill"/"decode" spelling, which `ScoredExponents::certify` would treat as
        // absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        // Bit-exact, matching `ScoredExponents::certify`'s `f64::to_bits` comparison
        // (both values are exactly representable, so `==` above is already bit-exact;
        // asserted a second way so a future refactor that introduces float formatting
        // cannot silently reintroduce drift).
        #expect(prefill.bitPattern == Double(0.25).bitPattern)
        #expect(decode.bitPattern == Double(0.75).bitPattern)
        #expect(exponents.count == 2, "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    @Test("timed_prompt_pool has exactly 8 slots, matching scored_batch_size")
    func timedPromptPoolHasEightSlots() throws {
        let object = try gemma4A4BTrackContractObject()
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8)
    }

    @Test("every timed_prompt_pool entry is a real armed pin with anti-lottery distinctness")
    func timedPromptPoolEntriesAreArmedPins() throws {
        let object = try gemma4A4BTrackContractObject()
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8, "the cohort IS the whole pool: exactly 8 entries")
        var shas = Set<String>()
        var paths = Set<String>()
        for entry in pool {
            let sha256 = try #require(entry["sha256"] as? String)
            // A real pin, never the pre-arming sentinel (or any near-miss of it).
            #expect(!sha256.hasPrefix(Self.pendingOrganizerSentinel))
            #expect(isSixtyFourLowercaseHex(sha256))
            let bytes = try #require(entry["bytes"] as? Int)
            #expect(bytes > 0, "pins bind sha256 AND byte count together; 0 is the sentinel placeholder")
            let r2Path = try #require(entry["r2_path"] as? String)
            #expect(r2Path.hasPrefix("correctness_prompts/gemma4-26b-a4b-mlx-v1/gemma4-26b-a4b-pool-"))
            #expect(r2Path.hasSuffix(".json"))
            #expect(!r2Path.contains("PENDING-ORGANIZER"))
            shas.insert(sha256)
            paths.insert(r2Path)
        }
        // Anti-lottery floor counts DISTINCTNESS, not length: one object listed
        // eight times has cardinality 8 and anti-lottery value zero.
        #expect(shas.count == 8)
        #expect(paths.count == 8)
    }

    @Test("hidden_correctness_golden is a root-level sibling of timed_prompt_pool, armed with a real pin")
    func hiddenCorrectnessGoldenIsRootLevelArmedPin() throws {
        let object = try gemma4A4BTrackContractObject()
        // ROOT level -- the CUDA lesson: benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root, never a nested wrapper.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        let sha256 = try #require(golden["sha256"] as? String)
        #expect(!sha256.hasPrefix(Self.pendingOrganizerSentinel))
        #expect(isSixtyFourLowercaseHex(sha256))
        let bytes = try #require(golden["bytes"] as? Int)
        #expect(bytes > 0)
        // Distinct from every pool pin: the oracle is a SIBLING, never a ninth pool entry.
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(!pool.contains { ($0["sha256"] as? String) == sha256 })
        // It must NOT appear nested under any "hidden_material"-style wrapper.
        #expect(object["hidden_material"] == nil)
    }

    @Test("target reference-model pin is a real 40-hex revision, matching the compiled constants")
    func targetPinIsFortyHexAndMatchesConstants() throws {
        let object = try gemma4A4BTrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let modelId = try #require(target["upstream_model_id"] as? String)
        let revision = try #require(target["upstream_revision"] as? String)
        #expect(modelId == gemma4A4BRepository)
        #expect(revision == gemma4A4BRevision)
        #expect(isFortyLowercaseHex(revision))
    }

    @Test("assistant pin is a real 40-hex revision, distinct from the target")
    func assistantPinIsFortyHexAndDistinctFromTarget() throws {
        let object = try gemma4A4BTrackContractObject()
        let assistant = try #require(object["assistant"] as? [String: Any])
        let modelId = try #require(assistant["upstream_model_id"] as? String)
        let revision = try #require(assistant["upstream_revision"] as? String)
        #expect(modelId == gemma4A4BAssistantRepository)
        #expect(revision == gemma4A4BAssistantRevision)
        #expect(isFortyLowercaseHex(revision))
        #expect(revision != gemma4A4BRevision)
    }

    @Test("scoring_semantics records the ruled composite formula and its current gated-off state")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try gemma4A4BTrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
        let ruling = try #require(semantics["ruling_verbatim"] as? [String])
        #expect(ruling.contains { $0.contains("prefill gains ^ .25 * decode ^ .75") })
    }
}

private func isSixtyFourLowercaseHex(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { byte in
        (byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9"))
            || (byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "f"))
    }
}

private func isFortyLowercaseHex(_ value: String) -> Bool {
    value.utf8.count == 40 && value.utf8.allSatisfy { byte in
        (byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9"))
            || (byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "f"))
    }
}
