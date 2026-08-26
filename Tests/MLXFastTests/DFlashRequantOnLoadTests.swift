import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Testing

@testable import MLXFastRuntimeWorkerSupport

// RE-QUANTIZING THE DFLASH DRAFTER ON LOAD — the DFlash half of the same
// 2026-08-26 ruling `HeadRequantOnLoadTests` pins for the MTP head: "a requant
// is when the user requantizes ON LOAD. Nothing happens on disk."
//
// WHY THIS SUITE EXISTS SEPARATELY. The participant contract has promised a
// DFlash re-quantization since section 3.4 was written, and until now that
// promise was not reachable: `DFlashDraftModel` lives in
// `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/`, its worker-side loader lives
// in `Sources/MLXFastHarness/`, and neither was an `editablePaths` entry, so no
// participant code could decide the drafter's geometry. David ruled on
// 2026-08-26 that "dflash needs to be editable", and
// `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift` joined
// the editable surface — ONE MODEL FILE, the same shape the MTP head already
// had, where `Gemma4MTP.swift` is editable and its harness loader is not. The
// head WEIGHTS directory `dflash-head/` is untouched and stays non-editable:
// this change adds CODE, never an upload surface.
//
// WHERE THE PARTICIPANT'S EDIT GOES. `DFlashDraftModel.load(from:bindTo:)`
// (DFlashDraftModel.swift:503) already calls `applyDeclaredQuantization`
// (:541), which calls `quantize(model:)` at :562 off the checkpoint's own
// declared per-layer policy. That is the same seam
// `Gemma4AssistantDraftModel.load(from:)` gives the MTP head at
// Gemma4MTP.swift:1275, and a participant edits the quantization decision
// inside it. This suite reproduces that edit FROM THE OUTSIDE — load the
// organizer's staged drafter, then quantize the loaded module before anything
// uses it — because a test may not edit the editable file it is testing.
//
// THE TWO PROPERTIES, both tested rather than asserted:
//
//  1. Nothing on disk changes, so benchd's write-divergence gate — a disk
//     comparison — has nothing to see.
//  2. Nothing is written, so the ranked worker's Seatbelt profile
//     (`(deny file-write*)`, bench-runner sandbox.rs@dc7712ca:71-72) is not in
//     the picture. `withReadOnlyStagedHead` strips every write bit from the
//     staged directory for the duration of the load, so a load that tried to
//     write would fail there.
//
// GPU-free at fixture scale, forced onto `.cpu`, same as
// `DFlashQuantizedLoadingTests`.
@Suite("DFlash requant on load")
struct DFlashRequantOnLoadTests {

    // MARK: - Fixtures

    /// A target whose geometry matches `drafterConfigJSON`. Copied from
    /// `DFlashQuantizedLoadingTests` so both suites drive the same shape.
    private func tinyTargetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": [
                    "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention"
                ],
                "sliding_window": 16,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// The drafter config the ORGANIZER stages: no `quantization` block at all,
    /// so the checkpoint is full precision and the loader binds it as-is. That
    /// is the starting point a re-quantization has to work from.
    private func drafterConfigJSON() -> String {
        """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": 4,
            "layer_types": ["full_attention", "full_attention"],
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [0, 3],
                "mask_token_id": 4
            }
        }
        """
    }

    /// Stage a FULL-PRECISION DFlash drafter at `<workspace>/dflash-head/` —
    /// the organizer's shipped shape, declaring no quantization. A `README.md`
    /// rides along because the real directory carries one and the tree digest
    /// excludes it by rule.
    @discardableResult
    private func stageHead(in workspace: URL) throws -> URL {
        let directory = workspace.appendingPathComponent("dflash-head", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let json = drafterConfigJSON()
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
        let drafter = DFlashDraftModel(config: config)
        eval(drafter)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: drafter.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        try Data(json.utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data("placeholder\n".utf8)
            .write(to: directory.appendingPathComponent("README.md"))
        return directory
    }

    /// The same tree digest the worker seals as head identity. It is a
    /// directory-generic walk — sorted relative paths, per-file sha256, top-level
    /// `README.md` excluded — so reusing it for the DFlash tree asserts
    /// "unchanged" through the number a run actually reports, not through a
    /// second hand-rolled hash that could drift from it.
    private func digest(_ directory: URL) throws -> RuntimeWorkerHeadProvenance {
        try computeGemma4AssistantHeadProvenance(directory: directory)
    }

    /// Run `body` with every write bit stripped from the staged head, then put
    /// the modes back. The honest stand-in for the ranked worker's Seatbelt
    /// profile: under it a write to the staged tree fails, so a body that
    /// completes provably did not write one.
    private func withReadOnlyStagedHead<R>(
        _ directory: URL, _ body: () async throws -> R
    ) async rethrows -> R {
        let fm = FileManager.default
        let entries =
            (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for entry in entries {
            try? fm.setAttributes([.posixPermissions: 0o444], ofItemAtPath: entry.path)
        }
        try? fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: directory.path)
        defer {
            try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: directory.path)
            for entry in entries {
                try? fm.setAttributes([.posixPermissions: 0o644], ofItemAtPath: entry.path)
            }
        }
        return try await body()
    }

    private func withTemporaryWorkspace<R>(_ body: (URL) async throws -> R) async throws -> R {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        return try await body(directory)
    }

    private func quantizedModulePaths(_ drafter: DFlashDraftModel) -> Set<String> {
        Set(
            drafter.leafModules().flattened().compactMap { path, module in
                module is Quantized ? path : nil
            })
    }

    // MARK: - The mechanism

    /// THE DFLASH HALF OF THE RULING, IN ONE TEST — the exact counterpart of
    /// `HeadRequantOnLoadTests.aHeadIsQuantizedOnLoadAndTheStagedBytesNeverChange`.
    ///
    /// A full-precision drafter is staged. The staged tree is made read-only,
    /// standing in for the ranked sandbox. Participant-style code loads it,
    /// binds it to the target, and quantizes it to 4 bits IN MEMORY. The drafter
    /// then produces finite logits of the right shape — and the staged bytes are
    /// byte-for-byte what they were before.
    @Test func theDrafterIsQuantizedOnLoadAndTheStagedBytesNeverChange() async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryWorkspace { workspace in
                let head = try stageHead(in: workspace)
                let before = try digest(head)

                let target = Gemma4TextModel(try tinyTargetConfig())
                eval(target)

                let drafter = try await withReadOnlyStagedHead(head) {
                    // THE PARTICIPANT'S EDIT, reproduced from outside the
                    // editable file: load the staged drafter, then quantize the
                    // module before anything uses it. In a submission these two
                    // steps are one, inside `DFlashDraftModel.load(from:bindTo:)`
                    // at the `quantize(model:)` call the loader already makes.
                    let drafter = try await DFlashDraftModel.load(
                        from: head, bindTo: target)
                    quantize(model: drafter, groupSize: 32, bits: 4) { _, _ in true }
                    eval(drafter)
                    return drafter
                }

                // It really is quantized, at the geometry the participant chose.
                let quantized = quantizedModulePaths(drafter)
                #expect(quantized.contains("fc"))
                #expect(quantized.contains("layers.0.self_attn.q_proj"))
                #expect(quantized.contains("layers.1.mlp.down_proj"))
                let fc = try #require(drafter.contextProjection as? QuantizedLinear)
                #expect(fc.bits == 4)
                #expect(fc.groupSize == 32)

                // It still drafts. A re-quantization that cannot produce logits
                // is not an optimization, it is a broken head.
                let forward = try target.forwardForDFlash(
                    MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis],
                    cache: target.newCache(parameters: nil),
                    targetLayerIds: drafter.config.targetLayerIds)
                let lastHidden = forward.targetHidden[0..., (-1)..., 0...]
                let logits = try drafter(
                    MLXArray([Int32(1), 4, 4, 4])[.newAxis, .ellipsis],
                    targetHidden: lastHidden,
                    cache: try drafter.makeCache(),
                    logitsStart: 1)
                eval(logits)
                #expect(logits.shape == [1, 3, 32])
                #expect(isFinite(logits).all().item(Bool.self))

                // NOTHING ON DISK CHANGED. This is the claim the ruling rests
                // on, and it is checked on the bytes, not inferred.
                let after = try digest(head)
                #expect(after == before)
                #expect(after.sha256 == before.sha256)
            }
        }
    }

    /// THE DO-NOTHING DEFAULT, the DFlash mirror of the MTP suite's. A
    /// participant who re-quantizes NOTHING gets the organizer's pinned drafter
    /// loaded exactly as staged: it binds, it drafts, it carries no quantized
    /// module, and the staged tree digest is unmoved.
    @Test func withNoRequantTheOrganizerDrafterLoadsUnchanged() async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryWorkspace { workspace in
                let head = try stageHead(in: workspace)
                let before = try digest(head)

                let target = Gemma4TextModel(try tinyTargetConfig())
                eval(target)

                let drafter = try await withReadOnlyStagedHead(head) {
                    try await DFlashDraftModel.load(from: head, bindTo: target)
                }

                // The organizer stages a full-precision drafter and declares no
                // quantization, so the default load must quantize nothing.
                #expect(
                    quantizedModulePaths(drafter).isEmpty,
                    "the default path must not quantize anything")

                // It still drafts. The default is the case every submission
                // starts from, so "unchanged" has to mean "unchanged and
                // working", not merely "untouched".
                let forward = try target.forwardForDFlash(
                    MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis],
                    cache: target.newCache(parameters: nil),
                    targetLayerIds: drafter.config.targetLayerIds)
                let lastHidden = forward.targetHidden[0..., (-1)..., 0...]
                let logits = try drafter(
                    MLXArray([Int32(1), 4, 4, 4])[.newAxis, .ellipsis],
                    targetHidden: lastHidden,
                    cache: try drafter.makeCache(),
                    logitsStart: 1)
                eval(logits)
                #expect(logits.shape == [1, 3, 32])
                #expect(isFinite(logits).all().item(Bool.self))

                let after = try digest(head)
                #expect(after == before)
                #expect(after.sha256 == before.sha256)
            }
        }
    }

    /// MUTATION CONTROL for the test above. The digest comparison must be able
    /// to FAIL: if it could not, "the bytes did not change" would be vacuous.
    /// A deliberate one-byte edit to the staged tree must move the digest.
    @Test func theUnchangedBytesAssertionIsNotVacuous() async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryWorkspace { workspace in
                let head = try stageHead(in: workspace)
                let before = try digest(head)
                try Data("tampered\n".utf8)
                    .write(to: head.appendingPathComponent("config.json"))
                #expect(try digest(head).sha256 != before.sha256)
            }
        }
    }

    /// The staged drafter really was unwritable while the load ran. Without
    /// this the read-only stand-in could be silently ineffective — a chmod that
    /// did not take would make the sandbox proxy prove nothing.
    @Test func theReadOnlyStandInActuallyBlocksAWrite() async throws {
        try await withTemporaryWorkspace { workspace in
            let head = try stageHead(in: workspace)
            let blocked = await withReadOnlyStagedHead(head) {
                let probe = head.appendingPathComponent("requant-probe.bin")
                return !FileManager.default.createFile(atPath: probe.path, contents: Data("x".utf8))
            }
            #expect(blocked, "a write into the staged drafter succeeded under the read-only stand-in")
        }
    }
}
