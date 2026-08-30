import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastRuntimeWorkerSupport

// THE LOADED-STATE TARGET QUANTIZATION BIND — David's 2026-08-26 ruling.
//
// The target's quantization format used to be enforced entirely on DISK: the
// trusted gate reads `config.json` and pins bits/group_size/mode plus the
// 120-entry per-tensor override table, and nothing ever read back the LIVE
// modules. The step from that declaration to the live modules --
// `Gemma4A4BRuntimeWeights.quantizeWithPerPathWidths` -- is in
// `Sources/MLXFastModel`, a `benchmark.json` `editablePaths` entry. So a
// participant could quantize the target IN MEMORY at any geometry, change
// nothing on disk, and pass every gate. `validateLoadedTargetQuantization`
// reads the loaded modules instead.
//
// WHAT THIS SUITE PINS, in the order the tests appear:
//
//  1. The pinned geometry PASSES. A gate that refuses the shipped model is
//     worse than no gate.
//  2. A participant-style in-memory requant of the TARGET is REFUSED, by a
//     message that names the module and both geometries.
//  3. NON-VACUITY. The same requantized target runs and produces finite
//     logits, and its declared config is byte-identical to the pinned model's
//     -- so nothing else in the stack objects, and the bind is the only thing
//     that catches it.
//  4. SCOPE. A lawfully re-quantized HEAD does not trip the bind. Heads are an
//     explicit exception to the freeze; catching one would be a bug.
//  5. THE POST-STARTUP RESIDUAL IS CLOSED. A startup-only bind verifies a model
//     that editable request-path code can still mutate in place afterwards, via
//     the same `Module.update(modules:)` seam `quantize(model:)` uses. The
//     pre-measure re-check runs the FULL validation again at the top of every
//     window that gets measured. These tests show the attack CAUGHT, show that
//     without the re-check nothing else objects (identity survives, the model
//     still runs), show a substituted instance refused by its own message, and
//     show the head exception still silent through BOTH checks.
//
// GPU-free at fixture scale, forced onto `.cpu`, same as
// `Gemma4DFlashForwardTests` and `HeadRequantOnLoadTests`. The real 26B target
// is never loaded.
// `.serialized` — a MITIGATION, and named as one rather than left to look like
// a default. Every test here builds a `Gemma4TextModel`, and MLX module
// initializers draw their weights from the GLOBAL RNG stream
// (`MLXNN/Linear.swift:89` calls `MLXRandom.uniform` with no key). So each of
// these tests consumes from a stream that other suites are also using.
//
// That matters to one neighbour in particular.
// `RuntimeWorkerFreeRunLegIdentityTests
// .serialLegRunsTheSameEngineBackedExecutorAsTheMTPLeg` does
// `MLXRandom.seed(0x1E6_1D)` and THEN constructs its model, and those two
// statements are not atomic under swift-testing's parallel execution: a model
// built concurrently in another suite advances the stream in between, so the
// "seeded" weights are not actually deterministic. Random weights make near-tie
// argmaxes likely, and that test compares TWO implementations token-for-token,
// which is exactly the divergence class `docs/gemma4-port-notes.md` section 5.1
// and CLAUDE.md's near-tie caveat describe. It was observed failing once in
// three full `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` runs.
//
// Serializing THIS suite does not fix that hazard — the fix belongs in the
// fragile test, which should not compare two implementations on weights it
// cannot guarantee — but it stops these eight model constructions from piling
// into the shared stream concurrently, which is this suite's own contribution
// to it. Reported upward rather than patched here.
// Box-only: needs the built Metal runtime, gated behind MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
@Suite("Target quantization bind", .serialized)
struct TargetQuantizationBindTests {

    private let layerCount = 2
    private let hiddenSize = 32
    private let vocabSize = 32

    @Test
    func runtimeLoaderInstallsVerifierOnlyAfterStrictUpdateAndEvaluation() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let update = try #require(source.range(of: "try model.update("))
        let verifyAll = try #require(
            source.range(of: "verify: [.all]", range: update.lowerBound..<source.endIndex))
        let evaluated = try #require(
            source.range(of: "eval(model)", range: verifyAll.lowerBound..<source.endIndex))
        let install = try #require(
            source.range(
                of: "try model.installCBv2MTPVerifier()",
                range: evaluated.lowerBound..<source.endIndex))
        let returned = try #require(
            source.range(of: "return model", range: install.lowerBound..<source.endIndex))

        #expect(update.lowerBound < verifyAll.lowerBound)
        #expect(verifyAll.lowerBound < evaluated.lowerBound)
        #expect(evaluated.lowerBound < install.lowerBound)
        #expect(install.lowerBound < returned.lowerBound)
    }

    // MARK: - Fixtures

    /// The pins the fixture is checked against.
    ///
    /// Derived from `.production` so every value except one is the shipped
    /// pin. The exception is `groupSize`: MLX accepts only 32/64/128 and the
    /// contracted axis must be a whole number of groups, and this fixture's
    /// narrowest projection input is `hidden_size == 32`. A 64-wide group
    /// cannot be formed over 32 elements, so the fixture runs at 32 and the
    /// production run at 64. Nothing else about the check varies with the
    /// number.
    private var pins: Gemma4TargetQuantizationPins {
        let production = Gemma4TargetQuantizationPins.production
        return Gemma4TargetQuantizationPins(
            fallbackBits: production.fallbackBits,
            overrideBits: production.overrideBits,
            groupSize: 32,
            modeName: production.modeName,
            packedWeightDType: production.packedWeightDType,
            scaleDType: production.scaleDType,
            packedWordBits: production.packedWordBits,
            overrideFamilies: production.overrideFamilies
        )
    }

    /// A tiny MoE target. `enable_moe_block` is ON deliberately: it is what
    /// puts `router.proj` (one of the four promoted families) and the stacked
    /// `experts.switch_glu.*` `QuantizedSwitchLinear` projections into the
    /// walk, and the switch layers are the 3-D packed case the shape relation
    /// has to handle. Shaped after `WidthProbeMachineryTests.moeTargetConfig`.
    private func targetConfigJSON() -> String {
        """
        {
            "model_type": "gemma4_text",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": \(layerCount),
            "intermediate_size": 64,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 32,
            "num_attention_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 16,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": \(vocabSize),
            "vocab_size_per_layer_input": \(vocabSize),
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0,
            "use_double_wide_mlp": false
        }
        """
    }

    private func targetConfig() throws -> Gemma4TextConfiguration {
        try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(targetConfigJSON().utf8))
    }

    /// A staged MTP assistant head, sized to `targetConfigJSON`'s backbone.
    /// Shared by the two scope tests so both drive the same head shape.
    private func assistantDrafterConfigJSON() -> String {
        """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": \(hiddenSize),
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
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

    /// Build a target and quantize it with `geometry`, which stands in for the
    /// per-path decision `quantizeWithPerPathWidths` makes on the real load.
    ///
    /// The parameters are cast to BF16 FIRST. MLX's affine conversion keeps
    /// the `scales`/`biases` companions at the source element type, and the
    /// pinned companions are BF16 (the checkpoint's own `dtype`), so a
    /// fixture left at the constructor's F32 default would produce F32
    /// companions and fail the dtype half of the check for a reason that has
    /// nothing to do with what is being tested.
    private func makeTarget(
        _ geometry: (String) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
    ) throws -> Gemma4TextModel {
        let model = Gemma4TextModel(try targetConfig())
        model.update(parameters: model.mapParameters(map: { $0.asType(.bfloat16) }))
        quantize(model: model) { path, _ in geometry(path) }
        eval(model)
        return model
    }

    /// The pinned per-path geometry: the four promoted families on every layer
    /// at `overrideBits`, everything else at `fallbackBits`.
    private func pinnedGeometry(
        overridePaths: Set<String>
    ) -> (String) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
        let pins = self.pins
        return { path in
            (
                groupSize: pins.groupSize,
                bits: overridePaths.contains(path) ? pins.overrideBits : pins.fallbackBits,
                mode: .affine
            )
        }
    }

    private func quantizedModule(_ model: Module, at path: String) -> Quantized? {
        model.leafModules().flattened()
            .first { $0.0 == path }?.1 as? Quantized
    }

    // MARK: - 1. The pinned geometry passes

    @Test func aTargetAtThePinnedGeometryPasses() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let model = try makeTarget(pinnedGeometry(overridePaths: overrides))

            // The fixture really does exercise both sides of the table and the
            // 3-D switch-layer case; otherwise "it passed" would be cheap.
            #expect(quantizedModule(model, at: "model.layers.1.mlp.gate_proj")?.bits == 8)
            #expect(quantizedModule(model, at: "model.layers.1.router.proj")?.bits == 8)
            #expect(quantizedModule(model, at: "model.embed_tokens")?.bits == 4)
            #expect(
                model.leafModules().flattened().contains {
                    $0.0 == "model.layers.0.experts.switch_glu.down_proj"
                        && $0.1 is QuantizedSwitchLinear
                },
                "the fixture must exercise the 3-D stacked expert case")

            #expect(throws: Never.self) {
                try validateLoadedTargetQuantization(
                    model: model, numHiddenLayers: layerCount, pins: pins)
            }
        }
    }

    // MARK: - 2. The negative control

    /// THE LOAD-BEARING TEST. This is the participant's edit: the same load,
    /// the same disk, one different number handed to `quantize(model:)`. It
    /// must refuse, and the refusal must be readable enough to act on.
    @Test func anInMemoryRequantOfTheTargetIsRefused() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            // `quantize(model: target, groupSize: 32, bits: 2)`, expressed
            // through the same per-path seam the real loader uses.
            let requantized = try makeTarget { _ in
                (groupSize: pins.groupSize, bits: 2, mode: .affine)
            }

            do {
                try validateLoadedTargetQuantization(
                    model: requantized, numHiddenLayers: layerCount, pins: pins)
                Issue.record("a 2-bit in-memory requant of the target must be refused")
            } catch let error as MLXFastError {
                let message = error.description
                #expect(message.contains("target quantization is frozen"))
                #expect(
                    message.contains(
                        "is bits=2 group_size=32 mode=affine, "
                            + "pinned bits=4 group_size=32 mode=affine"),
                    "refusal must name the found geometry and the pinned one")
                #expect(
                    message.contains("model."),
                    "refusal must name the offending runtime module path")
                #expect(
                    message.contains("violation(s) total"),
                    "a whole-model requant must report a count, not just one path")
            }
        }
    }

    /// The same refusal at single-module resolution, so the "names the
    /// offending path" claim is pinned exactly rather than through a substring
    /// that any of several modules could have produced. One promoted family
    /// entry is left at the 4-bit fallback -- the silent demotion the override
    /// table exists to prevent -- and everything else is pinned.
    @Test func theRefusalNamesTheExactDemotedOverridePath() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let demoted = "model.layers.1.mlp.gate_proj"
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
                .subtracting([demoted])
            let model = try makeTarget(pinnedGeometry(overridePaths: overrides))

            do {
                try validateLoadedTargetQuantization(
                    model: model, numHiddenLayers: layerCount, pins: pins)
                Issue.record("a demoted override path must be refused")
            } catch let error as MLXFastError {
                #expect(
                    error.description
                        == "target quantization is frozen: \(demoted) is bits=4 "
                            + "group_size=32 mode=affine, pinned bits=8 group_size=32 "
                            + "mode=affine")
            }
        }
    }

    // MARK: - 3. Non-vacuity

    /// MUTATION CONTROL. The bind is only worth its bytes if nothing else
    /// catches the requant, so this shows the requantized target behaving like
    /// a perfectly good model everywhere the rest of the stack looks:
    ///
    ///   * it RUNS -- finite logits out of a real forward -- so a numerics or
    ///     correctness smoke test would not flag it;
    ///   * its DECLARED config is byte-identical to the pinned model's, so the
    ///     disk-side gate (`validateRuntimeWorkerPinnedConfiguration`, which
    ///     reads exactly this JSON) cannot tell the two apart, and neither can
    ///     the benchmarker's write-divergence comparison, because nothing was
    ///     written.
    ///
    /// The last two lines are the control: same declaration, same runnability,
    /// and the bind is what separates them.
    @Test func nothingButTheBindObjectsToARequantizedTarget() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let pinned = try makeTarget(pinnedGeometry(overridePaths: overrides))
            let requantized = try makeTarget { _ in
                (groupSize: pins.groupSize, bits: 2, mode: .affine)
            }

            // It runs. A 2-bit target is a degraded model, not a broken one --
            // which is exactly why a gate is needed rather than a crash.
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = requantized(tokens, cache: requantized.newCache(parameters: nil))
            eval(logits)
            #expect(logits.shape == [1, 3, vocabSize])
            #expect(
                logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite },
                "the requantized target must still produce finite logits")

            // The declaration is unchanged. Both models were built from the
            // same `config.json` bytes and neither wrote anything, so every
            // disk-side gate sees one identical artifact.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let pinnedDeclaration = try encoder.encode(pinned.configuration)
            let requantizedDeclaration = try encoder.encode(requantized.configuration)
            #expect(
                pinnedDeclaration == requantizedDeclaration,
                "the two models must be indistinguishable on disk")

            // And the bind is what separates them.
            #expect(throws: Never.self) {
                try validateLoadedTargetQuantization(
                    model: pinned, numHiddenLayers: layerCount, pins: pins)
            }
            #expect(throws: MLXFastError.self) {
                try validateLoadedTargetQuantization(
                    model: requantized, numHiddenLayers: layerCount, pins: pins)
            }
        }
    }

    // MARK: - 4. Head scope

    /// The MTP assistant head and the DFlash drafter MAY be re-quantized on
    /// load (David ruling 2026-08-26; `docs/participant-contract.md` section
    /// 4.4). The bind walks the TARGET it is handed and nothing else, so a
    /// head at some other geometry sitting next to a pinned target is fine.
    /// Refusing one would revoke a granted exception.
    @Test func aLawfullyRequantizedHeadDoesNotTripTheBind() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let target = try makeTarget(pinnedGeometry(overridePaths: overrides))

            let drafterJSON = assistantDrafterConfigJSON()
            let drafter = try Gemma4AssistantDraftModel(
                config: try JSONDecoder.json5().decode(
                    Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8)))
            // A geometry the target could never carry, so "the bind ignored
            // the head" is a real result and not an accident of the head
            // happening to match.
            quantize(model: drafter, groupSize: 32, bits: 2) { _, _ in true }
            eval(drafter)
            // Hoisted rather than inlined into `#expect`: the macro rewrites a
            // trailing `.contains(_:)` into an optional-chained call, which
            // does not type-check against a non-optional `[Int]`.
            let headBits = Set(
                drafter.leafModules().flattened()
                    .compactMap { ($0.1 as? Quantized)?.bits })
            #expect(
                headBits.contains(2),
                "the head fixture must really be re-quantized")

            #expect(throws: Never.self) {
                try validateLoadedTargetQuantization(
                    model: target, numHiddenLayers: layerCount, pins: pins)
            }

            // The scoping is a CHOICE, not an accident: handing the drafter to
            // the bind does refuse. This is why the call site passes the target
            // instance explicitly and never walks a drafter.
            #expect(throws: MLXFastError.self) {
                try validateLoadedTargetQuantization(
                    model: drafter, numHiddenLayers: layerCount, pins: pins)
            }
        }
    }

    // MARK: - 5. The pre-measure re-check closes the post-startup residual

    /// Splice a COARSER quantized module into `model` at `path`, in place.
    ///
    /// This is the residual attack in its exact shape:
    /// `Module.update(modules:)` is the very seam `quantize(model:)` itself
    /// uses (`Vendor/mlx-swift/Source/MLXNN/Quantized.swift:78`), it is
    /// reachable from editable request-path code, and it mutates the ALREADY
    /// VERIFIED instance rather than replacing it.
    private func spliceCoarserModule(
        into model: Gemma4TextModel, at path: String, bits: Int
    ) throws {
        let pins = self.pins
        let coarse = try makeTarget { _ in
            (groupSize: pins.groupSize, bits: bits, mode: .affine)
        }
        let replacement = try #require(
            coarse.leafModules().flattened().first { $0.0 == path }?.1,
            "the coarse fixture must carry the path being spliced")

        // The FULL leaf list is rebuilt with one entry swapped, rather than a
        // one-entry update. That is not a stylistic choice: `unflattened`
        // reconstructs `layers` as an ARRAY, and a sparse list carrying only
        // `model.layers.1....` cannot describe a two-element array — MLXNN
        // raises `UpdateError.unexpectedStructure(key: "layers")` and takes the
        // process down. Passing every leaf back is exactly what
        // `quantize(model:)` does (Quantized.swift:64-78), which also makes this
        // splice the same operation a participant's own requant would perform.
        let updates = model.leafModules().flattened().map { entry -> (String, Module) in
            entry.0 == path ? (entry.0, replacement) : entry
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        eval(model)
    }

    /// (i) CAUGHT. A target that passed the startup bind is mutated in place
    /// afterwards, exactly as editable request-path code could do between the
    /// hello and the first timed forward. The pre-measure re-check refuses it,
    /// with the named refusal and the phase that fired.
    ///
    /// (ii) MUTATION CONTROL, in the same test so the two cannot drift: WITHOUT
    /// the re-check the attack lands. Three things are asserted about the
    /// mutated model, and each is a way the attack would otherwise survive:
    ///
    ///   * the STARTUP verdict was already taken and was green — it verified a
    ///     model that no longer exists in that form, and nothing re-derives it;
    ///   * the instance IDENTITY is unchanged, so the startup bind's own
    ///     `===` stability guard does not see this at all. This is why the
    ///     re-check re-runs the FULL validation instead of only comparing
    ///     identity;
    ///   * the mutated model still RUNS and produces finite logits, so no
    ///     numerics or smoke check flags it either.
    ///
    /// Remove the re-check and every one of those three stays true. That is the
    /// red half.
    @Test func anInPlaceMutationAfterStartupIsCaughtOnlyByTheReCheck() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            let demoted = "model.layers.1.mlp.gate_proj"
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let target = try makeTarget(pinnedGeometry(overridePaths: overrides))

            // The state a worker is in after `runWorker`'s startup bind.
            #expect(throws: Never.self) {
                try validateLoadedTargetQuantization(
                    model: target, numHiddenLayers: layerCount, pins: pins)
            }
            let verified: Module = target

            // THE ATTACK, after that verdict was taken.
            try spliceCoarserModule(into: target, at: demoted, bits: 2)

            // (ii) CONTROL — nothing but the re-check objects.
            #expect(
                verified === target,
                "an in-place mutation leaves identity intact, so the startup stability guard cannot see it")
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = target(tokens, cache: target.newCache(parameters: nil))
            eval(logits)
            #expect(
                logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite },
                "the mutated target must still run; a crash would be its own alarm")

            // (i) CAUGHT — the pre-measure re-check refuses, naming when and what.
            do {
                try revalidateTargetForMeasuredWindow(
                    phase: "free_decode_begin",
                    verifiedTarget: verified,
                    currentTarget: target,
                    numHiddenLayers: layerCount,
                    pins: pins)
                Issue.record("an in-place post-startup requant must be refused")
            } catch let error as MLXFastError {
                let message = error.description
                #expect(message.contains("pre-measure re-check (free_decode_begin)"))
                #expect(message.contains("target quantization is frozen"))
                #expect(
                    message.contains(
                        "\(demoted) is bits=2 group_size=32 mode=affine, "
                            + "pinned bits=8 group_size=32 mode=affine"),
                    "the refusal must name the mutated module and both geometries")
            }
        }
    }

    /// A SUBSTITUTED instance is refused too, and by its own message. The
    /// identity comparison and the geometry walk are separate failures with
    /// separate diagnostics, so a rejection says which one happened.
    @Test func aSubstitutedTargetInstanceIsRefusedByIdentity() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let verified = try makeTarget(pinnedGeometry(overridePaths: overrides))
            // A second, perfectly PINNED model: the geometry walk would pass it.
            // Only identity refuses it.
            let substitute = try makeTarget(pinnedGeometry(overridePaths: overrides))

            do {
                try revalidateTargetForMeasuredWindow(
                    phase: "prefill",
                    verifiedTarget: verified,
                    currentTarget: substitute,
                    numHiddenLayers: layerCount,
                    pins: pins)
                Issue.record("a substituted target instance must be refused")
            } catch let error as MLXFastError {
                #expect(
                    error.description.contains(
                        "the prefill window is about to run a different target instance"))
            }
        }
    }

    /// (iii) THE HEAD STAYS SILENT THROUGH BOTH CHECKS. A lawful head requant
    /// must not be caught at startup and must not be caught at the pre-measure
    /// re-check either — a gate that tightened only at the second one would
    /// revoke the granted exception halfway through a run.
    @Test func aLawfullyRequantizedHeadIsSilentThroughBothChecks() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
            return
        }
        try Device.withDefaultDevice(.cpu) {
            let pins = self.pins
            let overrides = pins.runtimeOverridePaths(layerCount: layerCount)
            let target = try makeTarget(pinnedGeometry(overridePaths: overrides))
            let verified: Module = target

            let drafterJSON = assistantDrafterConfigJSON()
            let drafter = try Gemma4AssistantDraftModel(
                config: try JSONDecoder.json5().decode(
                    Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8)))
            quantize(model: drafter, groupSize: 32, bits: 2) { _, _ in true }
            eval(drafter)

            // Startup bind: silent.
            #expect(throws: Never.self) {
                try validateLoadedTargetQuantization(
                    model: target, numHiddenLayers: layerCount, pins: pins)
            }
            // Pre-measure re-check, for each window the worker guards: silent.
            for phase in ["prefill", "decode_begin", "free_decode_begin"] {
                #expect(throws: Never.self) {
                    try revalidateTargetForMeasuredWindow(
                        phase: phase,
                        verifiedTarget: verified,
                        currentTarget: target,
                        numHiddenLayers: layerCount,
                        pins: pins)
                }
            }
        }
    }
}
