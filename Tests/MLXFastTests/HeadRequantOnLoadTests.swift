import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastRuntimeWorkerSupport

// RE-QUANTIZING A SPECULATIVE HEAD ON LOAD — David's 2026-08-26 ruling, as
// clarified 2026-08-26: "a requant is when the user requantizes ON LOAD.
// Nothing happens on disk."
//
// That clarification is what makes the mechanism simple. A participant does not
// produce a re-quantized checkpoint. Participant code quantizes the head's
// parameters IN MEMORY, in the same pass that binds them, and the staged bytes
// are only ever READ. Two things follow, and this suite pins both:
//
//  1. Nothing on disk changes, so the benchmarker's write-divergence gate --
//     which compares the candidate workspace against the trusted baseline --
//     has nothing to see. It is a disk comparison and this is not a disk
//     operation.
//  2. Nothing is written, so the official Seatbelt profile the ranked worker
//     runs under -- `(deny file-write*)` with only `/dev/null` allowed,
//     bench-runner sandbox.rs@dc7712ca:71-72 -- is not in the picture either.
//
// THE SECOND POINT IS TESTED, NOT ASSERTED. `withReadOnlyStagedHead` strips
// every write bit from the staged directory and its files for the duration of
// the load. A load that tried to write would fail there, so the read-only run
// standing in for the sandbox is what makes "nothing is written" a result
// rather than a claim.
//
// WHERE THE PARTICIPANT'S EDIT GOES. `Gemma4AssistantDraftModel` and its
// `load(from:)` live in `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/
// Gemma4MTP.swift`, which IS a `benchmark.json` `editablePaths` entry, and that
// loader ALREADY calls `quantize(model:)` off the checkpoint's declared
// per-layer policy (Gemma4MTP.swift:1275). So the seam is not something this
// change invents; it is the existing shape of the loader, and a participant
// edits the quantization decision inside it. This suite reproduces that edit
// from the outside -- quantize the loaded module before it is used -- because a
// test may not edit the editable file it is testing.
//
// THE DFLASH HALF LIVES NEXT DOOR, in
// `Tests/MLXFastTests/DFlashRequantOnLoadTests.swift`. When this suite was first
// written there was no DFlash seam to test: `DFlashDraftModel` sat in a
// non-editable vendored directory, its worker-side loader sat in non-editable
// `Sources/MLXFastHarness`, and no editable file referenced it at all. That was
// reported as a finding rather than worked around, and David ruled on it the
// same day -- `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`
// is an `editablePaths` entry as of this branch, and the DFlash suite mirrors
// every test here against `DFlashDraftModel.load(from:bindTo:)`.
@Suite("Head requant on load")
struct HeadRequantOnLoadTests {

    private let hiddenSize = 32
    private let vocabSize = 32

    private func drafterConfigJSON() -> String {
        """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": \(hiddenSize),
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 16,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
        }
        """
    }

    /// Stage a FULL-PRECISION assistant head at `<workspace>/mtp-head/` — the
    /// organizer's shipped shape, declaring no quantization.
    @discardableResult
    private func stageHead(in workspace: URL) throws -> URL {
        let directory = workspace.appendingPathComponent("mtp-head", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let json = drafterConfigJSON()
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
        let drafter = try Gemma4AssistantDraftModel(config: config)
        eval(drafter)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: drafter.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        try Data(json.utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data("placeholder\n".utf8)
            .write(to: directory.appendingPathComponent("README.md"))
        return directory
    }

    /// The same tree digest the worker seals as `head_provenance`. Reused here
    /// so "the staged bytes did not change" is asserted through the number the
    /// run actually reports as head identity.
    private func digest(_ directory: URL) throws -> RuntimeWorkerHeadProvenance {
        try computeGemma4AssistantHeadProvenance(directory: directory)
    }

    /// Run `body` with every write bit stripped from the staged head, then put
    /// the modes back. This is the honest stand-in for the ranked worker's
    /// Seatbelt profile: under it a write to the staged tree fails, so a body
    /// that completes provably did not write one.
    private func withReadOnlyStagedHead<R>(
        _ directory: URL, _ body: () throws -> R
    ) rethrows -> R {
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
        return try body()
    }

    private func withTemporaryWorkspace<R>(_ body: (URL) async throws -> R) async throws -> R {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        return try await body(directory)
    }

    private func quantizedModulePaths(_ drafter: Gemma4AssistantDraftModel) -> Set<String> {
        Set(
            drafter.leafModules().flattened().compactMap { path, module in
                module is Quantized ? path : nil
            })
    }

    // MARK: - The mechanism

    /// THE WHOLE RULING, IN ONE TEST. A full-precision head is staged. The
    /// staged tree is made read-only, standing in for the ranked sandbox.
    /// Participant-style code loads it and quantizes it to 4 bits IN MEMORY.
    /// The modules really are quantized at the chosen geometry — and the staged
    /// bytes are byte-for-byte what they were before.
    ///
    /// NO FORWARD IS RUN HERE, deliberately, and the docstring does not claim
    /// one. `Gemma4AssistantDraftModel.callAsFunction` needs `inputsEmbeds` plus
    /// a bound target's `Gemma4SharedKV` (Gemma4MTP.swift:1050-1054), so a
    /// standalone drafter cannot produce logits without a target instance to
    /// borrow from. That path is exercised by the MLX-runtime-gated
    /// `RuntimeWorkerMTPRoundExecutionTests`. The DFlash counterpart CAN run its
    /// forward from a tiny target and does — see
    /// `DFlashRequantOnLoadTests.theDrafterIsQuantizedOnLoadAndTheStagedBytesNeverChange`.
    @Test func aHeadIsQuantizedOnLoadAndTheStagedBytesNeverChange() async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryWorkspace { workspace in
                let head = try stageHead(in: workspace)
                let before = try digest(head)

                let drafter = try withReadOnlyStagedHead(head) {
                    // THE PARTICIPANT'S EDIT, reproduced from outside the
                    // editable file: load the staged head, then quantize the
                    // module before anything uses it. In a submission these two
                    // steps are one, inside `Gemma4AssistantDraftModel.load`.
                    let config = try JSONDecoder.json5().decode(
                        Gemma4AssistantConfiguration.self,
                        from: try Data(contentsOf: head.appendingPathComponent("config.json")))
                    let drafter = try Gemma4AssistantDraftModel(config: config)
                    try drafter.update(
                        parameters: ModuleParameters.unflattened(
                            try loadArrays(url: head.appendingPathComponent("model.safetensors"))),
                        verify: [.all])
                    quantize(model: drafter, groupSize: 32, bits: 4) { _, _ in true }
                    eval(drafter)
                    return drafter
                }

                // It really is quantized, at the geometry the participant chose.
                let quantized = quantizedModulePaths(drafter)
                #expect(!quantized.isEmpty)
                for path in quantized {
                    guard
                        let module = drafter.leafModules().flattened()
                            .first(where: { $0.0 == path })?.1 as? QuantizedLinear
                    else { continue }
                    #expect(module.bits == 4)
                    #expect(module.groupSize == 32)
                }

                // NOTHING ON DISK CHANGED. This is the claim the ruling rests
                // on, and it is checked on the bytes, not inferred.
                let after = try digest(head)
                #expect(after == before)
                #expect(after.sha256 == before.sha256)
            }
        }
    }

    /// THE DO-NOTHING DEFAULT. A participant who re-quantizes NOTHING must get
    /// the organizer's pinned head, loaded exactly as staged. This is the case
    /// every submission starts from and the one a requant lane is most likely to
    /// break silently, so it is asserted on its own rather than left implied by
    /// the requant test above: the head loads, it carries NO quantized module,
    /// and the staged tree digest is unmoved.
    @Test func withNoRequantTheOrganizerHeadLoadsUnchanged() async throws {
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryWorkspace { workspace in
                let head = try stageHead(in: workspace)
                let before = try digest(head)

                let drafter = try withReadOnlyStagedHead(head) {
                    let config = try JSONDecoder.json5().decode(
                        Gemma4AssistantConfiguration.self,
                        from: try Data(contentsOf: head.appendingPathComponent("config.json")))
                    let drafter = try Gemma4AssistantDraftModel(config: config)
                    try drafter.update(
                        parameters: ModuleParameters.unflattened(
                            try loadArrays(url: head.appendingPathComponent("model.safetensors"))),
                        verify: [.all])
                    eval(drafter)
                    return drafter
                }

                // The organizer stages a full-precision head and declares no
                // quantization, so a load that changed nothing must produce no
                // quantized module at all.
                #expect(
                    quantizedModulePaths(drafter).isEmpty,
                    "the default path must not quantize anything")
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

    /// The staged head really was unwritable while the load ran. Without this
    /// the read-only stand-in could be silently ineffective — a chmod that did
    /// not take would make the sandbox proxy prove nothing.
    @Test func theReadOnlyStandInActuallyBlocksAWrite() async throws {
        try await withTemporaryWorkspace { workspace in
            let head = try stageHead(in: workspace)
            let blocked = withReadOnlyStagedHead(head) {
                let probe = head.appendingPathComponent("requant-probe.bin")
                return !FileManager.default.createFile(atPath: probe.path, contents: Data("x".utf8))
            }
            #expect(blocked, "a write into the staged head succeeded under the read-only stand-in")
        }
    }
}
