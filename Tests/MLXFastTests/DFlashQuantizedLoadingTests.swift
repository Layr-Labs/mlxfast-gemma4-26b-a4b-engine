import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Testing

// DFlash drafter QUANTIZED-WEIGHT BINDING — David's 2026-08-26 ruling.
//
// The participant contract (docs/participant-contract.md §3.4) promises that a
// DFlash drafter may be re-quantized, and under the 512 MB submitted-bytes cap
// (§3.2) requant is the only realistic way to bring your own DFlash head. The
// loader did not honor that promise: `DFlashConfiguration` does not model
// quantization, so a `quantization` block landed in its `ignoredConfigKeys`
// diagnostic list and never reached the loader, the drafter was constructed at
// full precision, and `update(parameters:verify: [.all])` then refused the
// first packed tensor with
//
//     UpdateError.mismatchedSize(path: ["fc", "weight"],
//                                modules: ["DFlashDraftModel", "Linear"],
//                                expectedShape: [64, 128], actualShape: [64, 16])
//
// (a 4-bit affine weight packs eight values per uint32, so the last axis is
// 1/8 the width). Upstream that surfaced as the fail-soft `.incompatible`
// outcome in `loadGemma4DFlashHeadIfStaged` — DFlash capability absent for the
// worker's lifetime — with the shape error quoted back at spec resolution.
//
// These tests pin the fix from both sides: a quantized drafter binds and
// drafts, an unquantized one loads exactly as it did before, and every way of
// getting the declaration wrong refuses by name instead of quietly running a
// drafter nobody asked for.
//
// Forced onto `.cpu` and fixture-scale like `Gemma4DFlashForwardTests`, but
// constructing the models still needs the built MLX runtime, which hosted CI
// does not have: box-only, every test gated behind MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
@Suite("DFlash quantized weight binding")
struct DFlashQuantizedLoadingTests {

    // MARK: - Fixtures

    /// A target whose geometry matches `drafterConfigJSON`. Widened from the
    /// 16-wide config in `Gemma4DFlashForwardTests` to 32 so the drafter's
    /// narrowest projection input (`hidden_size`) is a legal MLX affine group
    /// size; nothing else about the shape matters here.
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

    /// `quantization` is a raw JSON fragment (including its leading comma) so a
    /// test can declare a well-formed block, a malformed one, or none at all
    /// over otherwise identical bytes.
    private func drafterConfigJSON(quantization: String = "") -> String {
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
            }\(quantization)
        }
        """
    }

    private static let declaredQuantization =
        ",\n    \"quantization\": { \"group_size\": 32, \"bits\": 4 }"

    /// Build a drafter checkpoint on disk: `config.json` carrying `configJSON`
    /// plus one `model.safetensors` holding the parameters of a freshly
    /// constructed drafter, optionally quantized first.
    ///
    /// Returns the directory and the exact tensors written, so a caller can
    /// compare what the loader bound against what was on disk.
    @discardableResult
    private func stageCheckpoint(
        configJSON: String,
        quantizeModules: ((String) -> Bool)? = nil,
        groupSize: Int = 32,
        bits: Int = 4,
        in directory: URL
    ) throws -> [String: MLXArray] {
        let config = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(configJSON.utf8))
        let drafter = DFlashDraftModel(config: config)
        if let quantizeModules {
            quantize(model: drafter, groupSize: groupSize, bits: bits) { path, _ in
                quantizeModules(path)
            }
        }
        eval(drafter)

        let weights = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
        try save(
            arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        try Data(configJSON.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        return weights
    }

    private func withTemporaryDirectory<R>(_ body: (URL) async throws -> R) async throws -> R {
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

    // MARK: - (a) A quantized drafter binds and drafts

    /// The whole point of the lane: a 4-bit repack of the drafter loads,
    /// binds to a matching target, and produces finite logits of the right
    /// shape. Before this change the load threw `UpdateError.mismatchedSize`
    /// on `fc.weight`.
    @Test func quantizedDrafterBindsAndProducesFiniteLogits() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: Self.declaredQuantization),
                    quantizeModules: { _ in true },
                    in: directory)

                let target = Gemma4TextModel(try tinyTargetConfig())
                eval(target)
                let drafter = try await DFlashDraftModel.load(
                    from: directory, bindTo: target)

                // Every Linear came back quantized at the declared geometry.
                let quantized = quantizedModulePaths(drafter)
                #expect(quantized.contains("fc"))
                #expect(quantized.contains("layers.0.self_attn.q_proj"))
                #expect(quantized.contains("layers.1.mlp.down_proj"))
                let fc = try #require(drafter.contextProjection as? QuantizedLinear)
                #expect(fc.groupSize == 32)
                #expect(fc.bits == 4)

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
            }
        }
    }

    /// The per-layer plumbing is the SAME `BaseConfiguration.PerLayerQuantization`
    /// the MTP head loader drives, so a `false` entry must skip that module and
    /// leave it at full precision rather than refusing the head.
    @Test func perLayerSkipLeavesThatModuleUnquantized() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let skipped = "layers.0.mlp.down_proj"
                let quantization = """
                    ,
                        "quantization": {
                            "group_size": 32,
                            "bits": 4,
                            "\(skipped)": false
                        }
                    """
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: quantization),
                    quantizeModules: { $0 != skipped },
                    in: directory)

                let drafter = try await DFlashDraftModel.load(from: directory)
                let quantized = quantizedModulePaths(drafter)
                #expect(quantized.contains("fc"))
                #expect(!quantized.contains(skipped))
            }
        }
    }

    // MARK: - (b) The unquantized control is unchanged

    /// The organizer's shipped drafter is full precision, so this is the
    /// regression that matters most. A checkpoint that declares no
    /// quantization must load exactly as it did before: no module quantized,
    /// every parameter bound bit-for-bit as written, and the same logits a
    /// drafter updated straight from those tensors produces.
    @Test func unquantizedDrafterLoadIsUnchanged() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let configJSON = drafterConfigJSON()
                let staged = try stageCheckpoint(configJSON: configJSON, in: directory)

                let drafter = try await DFlashDraftModel.load(from: directory)
                #expect(quantizedModulePaths(drafter).isEmpty)
                #expect(drafter.config.ignoredConfigKeys.isEmpty)

                let bound = Dictionary(
                    uniqueKeysWithValues: drafter.parameters().flattened())
                #expect(Set(bound.keys) == Set(staged.keys))
                for (key, value) in staged {
                    let loaded = try #require(bound[key])
                    #expect(loaded.shape == value.shape, "\(key)")
                    #expect(loaded.dtype == value.dtype, "\(key)")
                    #expect(
                        arrayEqual(loaded, value).all().item(Bool.self),
                        "\(key) did not bind bit-for-bit")
                }

                // And the loaded drafter computes what an in-memory drafter
                // updated from the same tensors computes, exactly.
                let reference = DFlashDraftModel(
                    config: try JSONDecoder.json5().decode(
                        DFlashConfiguration.self, from: Data(configJSON.utf8)))
                try reference.update(
                    parameters: ModuleParameters.unflattened(staged), verify: [.all])
                eval(reference)

                let target = Gemma4TextModel(try tinyTargetConfig())
                eval(target)
                try drafter.bind(target: target)
                try reference.bind(target: target)

                let forward = try target.forwardForDFlash(
                    MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis],
                    cache: target.newCache(parameters: nil),
                    targetLayerIds: drafter.config.targetLayerIds)
                let lastHidden = forward.targetHidden[0..., (-1)..., 0...]
                let block = MLXArray([Int32(1), 4, 4, 4])[.newAxis, .ellipsis]

                let loadedLogits = try drafter(
                    block, targetHidden: lastHidden,
                    cache: try drafter.makeCache(), logitsStart: 1)
                let referenceLogits = try reference(
                    block, targetHidden: lastHidden,
                    cache: try reference.makeCache(), logitsStart: 1)
                eval(loadedLogits, referenceLogits)
                #expect(arrayEqual(loadedLogits, referenceLogits).all().item(Bool.self))
            }
        }
    }

    // MARK: - (c) Unsupported geometry refuses, it does not fall back

    @Test func refusesUnsupportedQuantizationBits() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let quantization =
                    ",\n    \"quantization\": { \"group_size\": 32, \"bits\": 9 }"
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: quantization),
                    quantizeModules: { _ in true },
                    in: directory)

                await #expect(
                    throws: DFlashError.unsupportedQuantization(
                        field: "quantization.bits", reason: "must be between 2 and 8")
                ) {
                    _ = try await DFlashDraftModel.load(from: directory)
                }
            }
        }
    }

    @Test func refusesUnsupportedQuantizationGroupSize() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let quantization =
                    ",\n    \"quantization\": { \"group_size\": 0, \"bits\": 4 }"
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: quantization),
                    quantizeModules: { _ in true },
                    in: directory)

                await #expect(
                    throws: DFlashError.unsupportedQuantization(
                        field: "quantization.groupSize", reason: "must be positive")
                ) {
                    _ = try await DFlashDraftModel.load(from: directory)
                }
            }
        }
    }

    /// A per-layer override is bounded the same way the default is, and the
    /// refusal names the layer.
    @Test func refusesUnsupportedPerLayerQuantization() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let quantization = """
                    ,
                        "quantization": {
                            "group_size": 32,
                            "bits": 4,
                            "fc": { "group_size": 32, "bits": 1 }
                        }
                    """
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: quantization),
                    quantizeModules: { _ in true },
                    in: directory)

                await #expect(
                    throws: DFlashError.unsupportedQuantization(
                        field: "quantization.fc.bits", reason: "must be between 2 and 8")
                ) {
                    _ = try await DFlashDraftModel.load(from: directory)
                }
            }
        }
    }

    /// A malformed declaration is not "no declaration". Loading such a
    /// checkpoint at full precision would report a drafter that never ran.
    @Test func refusesUndecodableQuantizationDeclaration() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let quantization =
                    ",\n    \"quantization\": { \"group_size\": \"wide\", \"bits\": 4 }"
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(),
                    in: directory)
                try Data(drafterConfigJSON(quantization: quantization).utf8)
                    .write(to: directory.appendingPathComponent("config.json"))

                var thrown: Error?
                do {
                    _ = try await DFlashDraftModel.load(from: directory)
                } catch {
                    thrown = error
                }
                let error = try #require(thrown as? DFlashError)
                guard case .undecodableQuantization = error else {
                    Issue.record("expected .undecodableQuantization, got \(error)")
                    return
                }
            }
        }
    }

    /// Quantized bytes with nothing declaring them. This used to reach
    /// `update(verify:)` and die on a shape; now it is named at the seam that
    /// knows what went wrong.
    @Test func refusesQuantizedWeightsWithoutDeclaration() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(),
                    quantizeModules: { _ in true },
                    in: directory)

                await #expect(
                    throws: DFlashError.quantizedWeightsWithoutDeclaration("fc")
                ) {
                    _ = try await DFlashDraftModel.load(from: directory)
                }
            }
        }
    }

    /// The silent-fp16-fallback case: a declaration with nothing to apply it
    /// to. Binding this at full precision would run a drafter the config says
    /// is 4-bit.
    @Test func refusesDeclarationWithoutQuantizedWeights() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(),
                    in: directory)
                try Data(
                    drafterConfigJSON(quantization: Self.declaredQuantization).utf8
                ).write(to: directory.appendingPathComponent("config.json"))

                await #expect(
                    throws: DFlashError.declaredQuantizationWithoutQuantizedWeights
                ) {
                    _ = try await DFlashDraftModel.load(from: directory)
                }
            }
        }
    }

    // MARK: - (d) The silent drop is now impossible

    /// The regression that names the old bug. `DFlashConfiguration` still does
    /// not model quantization — the key is still reported in
    /// `ignoredConfigKeys` — but the LOADER no longer decides anything from
    /// that list: `DFlashConfigurationDocument` re-reads the same bytes for the
    /// declaration, and a drafter loaded from a config that carries one comes
    /// back quantized.
    ///
    /// MUTATION PROOF: drop the `quantize(model:)` call in
    /// `DFlashDraftModel.applyDeclaredQuantization` and this fails — the load
    /// throws `UpdateError.mismatchedSize` on `fc.weight`, which is exactly the
    /// pre-fix behavior.
    @Test func declaredQuantizationIsNeverSilentlyDropped() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                let configJSON = drafterConfigJSON(quantization: Self.declaredQuantization)
                try stageCheckpoint(
                    configJSON: configJSON,
                    quantizeModules: { _ in true },
                    in: directory)

                // The config decoder's own view is unchanged: it never learned
                // about quantization, and that is precisely why the loader must
                // not read the declaration through it.
                let config = try JSONDecoder.json5().decode(
                    DFlashConfiguration.self, from: Data(configJSON.utf8))
                #expect(config.ignoredConfigKeys == ["quantization"])

                // The loader's view is the one that changed.
                let document = try DFlashConfigurationDocument.decode(Data(configJSON.utf8))
                let declared = try #require(document.quantization?.quantization)
                #expect(declared.bits == 4)
                #expect(declared.groupSize == 32)

                let drafter = try await DFlashDraftModel.load(from: directory)
                #expect(!quantizedModulePaths(drafter).isEmpty)
            }
        }
    }

    /// The DFlash geometry gate is config-only (`hiddenSize`,
    /// `vocabularySize`, `numTargetLayers`, tap-id range), so quantized
    /// storage shapes cannot make it false-refuse a valid head. Pinned here
    /// because that is easy to break by "improving" the check into a
    /// tensor-shape comparison.
    @Test func compatibilityValidationIgnoresQuantizedStorageShapes() async throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try await Device.withDefaultDevice(.cpu) {
            try await withTemporaryDirectory { directory in
                try stageCheckpoint(
                    configJSON: drafterConfigJSON(quantization: Self.declaredQuantization),
                    quantizeModules: { _ in true },
                    in: directory)

                let target = Gemma4TextModel(try tinyTargetConfig())
                eval(target)
                // Binding BEFORE the weights land and binding after both pass:
                // neither reads a tensor.
                let bound = try await DFlashDraftModel.load(from: directory, bindTo: target)
                #expect(bound.config.numTargetLayers == 4)

                let afterwards = try await DFlashDraftModel.load(from: directory)
                try afterwards.bind(target: target)
            }
        }
    }
}
