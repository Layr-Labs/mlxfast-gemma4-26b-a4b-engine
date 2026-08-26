import Foundation
import MLX
import MLXFastCore
import MLXFastHarness
import MLXLLM
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

// The IN-LOADER half of the DFlash head's 2 GiB per-head cap.
//
// Gemma4DFlashHeadDeclarationSizeGateTests covers the MANIFEST half: the cap
// as `Gemma4MTPHeadDeclaration` with `kind: .dflash` enforces it on
// `dflash-head.manifest.json`, upstream of the sandbox, on the size a stager
// is allowed to DECLARE. That gate was written when the DFlash arm was an
// ALIAS over the MTP assistant-head loader, so the declaration layer was the
// only layer there was.
//
// The real-loader port removed the alias: `loadGemma4DFlashHeadIfStaged`
// loads a real `DFlashDraftModel` straight off the CWD `dflash-head/`
// default and never touches the declaration parser. That opened a gap — a
// tree PRESENT on disk without having gone through a declaration would be
// loaded at any size — and `gemma4DFlashStagedHeadMaxBytes` closes it by
// re-asserting the same cap on the bytes actually staged, before the load.
// This suite is the tripwire for that second layer.
//
// FIX-BAR / mutations to kill: deleting the in-loader gate, or moving it
// AFTER the load, turns `overCapStagedDFlashHeadIsIncompatibleAndNamesTheCap`
// red; making the gate throw instead of returning `.incompatible` turns it
// red too (a throw would crash the worker before hello and take the serial
// CONTROL leg with it). Making the gate fire on trees UNDER the cap turns
// `atCapStagedDFlashHeadProceedsToTheLoadAttempt` red. Giving the loader its
// own, drifted cap number turns `stagedDFlashHeadCapMirrorsTheDeclarationCap`
// red.
//
// Scope note: no weights, no network, no GPU — the target is a 16-hidden
// fixture model on `.cpu`, and the over/at-cap trees are SPARSE files
// (`ftruncate` on APFS), so "2 GiB staged" costs no disk and no time.
@Suite("Gemma4 DFlash staged-head size gate")
struct Gemma4DFlashStagedHeadSizeGateTests {

    /// The smallest target `loadGemma4DFlashHeadIfStaged` will accept as a
    /// bind partner. Its geometry is irrelevant to every test here: the cap
    /// is checked before the drafter is loaded, so no test in this suite ever
    /// reaches `bind(target:)`.
    private func tinyTarget() throws -> Gemma4TextModel {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 8,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention"],
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
        return Gemma4TextModel(
            try JSONDecoder.json5().decode(
                Gemma4TextConfiguration.self, from: Data(json.utf8)))
    }

    /// A staged `dflash-head/` whose measured tree is EXACTLY `totalBytes`.
    ///
    /// `config.json` has to be real and present or the loader short-circuits
    /// to `.absent` before the gate is ever reached, so the padding file is
    /// sized to make `config.json` + padding land on the requested total. The
    /// padding is sparse — `truncate(atOffset:)` allocates no blocks on APFS
    /// — which is the only reason a 2 GiB fixture is a reasonable unit test.
    private func makeStagedDirectory(totalBytes: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash-head-staged-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        // Deliberately a config the DFlash decoder REJECTS, so that any test
        // reaching the load attempt fails on the config and not on missing
        // weights -- the failure reason then names a decode defect, which is
        // exactly how a test tells "the gate did not fire" from "the gate
        // fired".
        let config = Data(#"{"architectures": ["DFlashDraftModel"]}"#.utf8)
        try config.write(to: directory.appendingPathComponent("config.json"))

        let padding = totalBytes - config.count
        #expect(padding >= 0, "fixture total must leave room for config.json")
        let paddingURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.createFile(atPath: paddingURL.path, contents: nil) else {
            throw MLXFastError.invalidInput("could not create \(paddingURL.path)")
        }
        let handle = try FileHandle(forWritingTo: paddingURL)
        try handle.truncate(atOffset: UInt64(padding))
        try handle.close()

        // The measured set must match what the loader will measure.
        #expect(try measureGemma4StagedHeadBytes(directory: directory) == totalBytes)
        return directory
    }

    // --- The gate binds ------------------------------------------------------

    /// One byte over the per-head cap is capability-absence, named, NOT a
    /// throw and NOT a load.
    @Test func overCapStagedDFlashHeadIsIncompatibleAndNamesTheCap() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = try makeStagedDirectory(
                totalBytes: gemma4DFlashStagedHeadMaxBytes + 1)
            defer { try? FileManager.default.removeItem(at: directory) }

            let outcome = try loadGemma4DFlashHeadIfStaged(
                directoryName: directory.path, target: try tinyTarget())

            guard case .incompatible(let refusedDirectory, let reason) = outcome else {
                Issue.record("expected .incompatible, got \(outcome)")
                return
            }
            #expect(refusedDirectory.standardizedFileURL == directory.standardizedFileURL)
            // The refusal names the cap AND the actual size -- the same
            // reporting bar the declaration-layer refusal is held to.
            #expect(reason.contains("\(gemma4DFlashStagedHeadMaxBytes)"))
            #expect(reason.contains("\(gemma4DFlashStagedHeadMaxBytes + 1)"))
            #expect(reason.lowercased().contains("cap"))
        }
    }

    /// AT the cap is not over it. The tree is admitted past the gate and the
    /// loader goes on to attempt the real load, which then fails on this
    /// fixture's deliberately-undecodable `config.json`. The distinguishing
    /// assertion is that the reason is a DECODE complaint and says nothing
    /// about a cap.
    @Test func atCapStagedDFlashHeadProceedsToTheLoadAttempt() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = try makeStagedDirectory(
                totalBytes: gemma4DFlashStagedHeadMaxBytes)
            defer { try? FileManager.default.removeItem(at: directory) }

            let outcome = try loadGemma4DFlashHeadIfStaged(
                directoryName: directory.path, target: try tinyTarget())

            guard case .incompatible(_, let reason) = outcome else {
                Issue.record("expected .incompatible from the load attempt, got \(outcome)")
                return
            }
            #expect(!reason.lowercased().contains("cap"))
            #expect(!reason.contains("\(gemma4DFlashStagedHeadMaxBytes)"))
        }
    }

    // --- One cap, one source -------------------------------------------------

    /// `gemma4DFlashStagedHeadMaxBytes` is a MIRROR of the declaration
    /// mechanism's cap, not a second opinion about it. The participant
    /// runtime-worker target cannot import the trusted-harness target (the
    /// trust split is the point, and the two trees carry ~45 twin symbols),
    /// so the mirror is held honest here instead: this test target imports
    /// both and fails the moment the two numbers disagree.
    @Test func stagedDFlashHeadCapMirrorsTheDeclarationCap() {
        #expect(gemma4DFlashStagedHeadMaxBytes == Gemma4MTPHeadDeclaration.defaultMaxBytes)
        // ONE cap for BOTH replaceable heads -- the MTP head's cap is the
        // same number, and the DFlash kind does not get its own ceiling.
        #expect(
            Gemma4MTPHeadDeclaration.pinnedDefault(for: .dflash).maxBytes
                == gemma4DFlashStagedHeadMaxBytes)
    }
}
