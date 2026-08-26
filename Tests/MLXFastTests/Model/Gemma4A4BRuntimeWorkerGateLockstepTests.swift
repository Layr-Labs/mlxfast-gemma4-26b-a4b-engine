import Foundation
import MLXFastCore
import Testing
@testable import MLXFastHarness
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
@testable import MLXFastTransform

/// Lockstep coverage for the runtime-worker pinned-configuration gate's two
/// independent implementations.
///
/// `validateRuntimeWorkerPinnedConfigurationData` exists TWICE in this
/// package: once in `Sources/MLXFastHarness/Gemma4RuntimeWorker.swift` (the
/// `MLXFastRuntimeWorkerSupport` SwiftPM target, which links `MLXFastModel`
/// and therefore single-sources its check through `Gemma4A4BConfig`), and
/// once in `Sources/MLXFastTrustedHarness/Gemma4RuntimeWorker.swift` (the
/// `MLXFastHarness` SwiftPM target, which the trusted `mlxfast-swift` binary
/// builds from and which deliberately does NOT link `MLXFastModel` -- see
/// that function's doc comment). The two module names are swapped relative
/// to their source directories on purpose (`Package.swift`); this file
/// disambiguates every call with the module name for exactly that reason.
///
/// Because the trusted twin cannot share `Gemma4A4BConfig`'s code, it shares
/// only the static key manifest (`MLXFastCore.Gemma4A4BConfigKeys`) and the
/// frozen geometry (`MLXFastCore.MLXFastConstants`), and re-derives the rest
/// of the check locally. This file is what proves the two independent
/// derivations still agree: every fixture below is run through BOTH
/// validators, and the test fails if they ever disagree on accept vs.
/// reject.
private func runsBothGates(_ data: Data) -> (participant: Bool, trusted: Bool) {
    let participantAccepted: Bool
    do {
        try MLXFastRuntimeWorkerSupport.validateRuntimeWorkerPinnedConfigurationData(data)
        participantAccepted = true
    } catch {
        participantAccepted = false
    }
    let trustedAccepted: Bool
    do {
        try MLXFastHarness.validateRuntimeWorkerPinnedConfigurationData(data)
        trustedAccepted = true
    } catch {
        trustedAccepted = false
    }
    return (participantAccepted, trustedAccepted)
}

private func mutatedGemma4RuntimeConfig(
    _ mutate: (inout [String: Any]) -> Void
) throws -> Data {
    let root = try gemma4A4BConfigObject()
    let runtimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: root,
        family: .gemma4A4B
    )
    var mutable = try #require(
        try JSONSerialization.jsonObject(with: runtimeConfig) as? [String: Any]
    )
    mutate(&mutable)
    return try JSONSerialization.data(withJSONObject: mutable)
}

@Test
func gemma4A4BTrustedGateAgreesWithConfigLoaderAcrossFixtures() throws {
    // (a) The real, unmutated transform output: both must ACCEPT.
    let validGemma4 = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: try gemma4A4BConfigObject(),
        family: .gemma4A4B
    )
    let validResult = runsBothGates(validGemma4)
    #expect(validResult.participant)
    #expect(validResult.trusted)

    // (b) A Qwen 3.6-shaped runtime config -- the exact box refusal scenario:
    // both must REJECT.
    let qwenShaped = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: try qwen36ConfigObject(),
        family: .qwen35
    )
    let qwenResult = runsBothGates(qwenShaped)
    #expect(!qwenResult.participant)
    #expect(!qwenResult.trusted)

    // (c) One required key dropped: both must REJECT.
    let missingKey = try mutatedGemma4RuntimeConfig {
        $0.removeValue(forKey: "attention_k_eq_v")
    }
    let missingKeyResult = runsBothGates(missingKey)
    #expect(!missingKeyResult.participant)
    #expect(!missingKeyResult.trusted)

    // (d) One forbidden key added: both must REJECT.
    let forbiddenKey = try mutatedGemma4RuntimeConfig {
        $0["moe_router_logit_softcapping"] = 0.0
    }
    let forbiddenKeyResult = runsBothGates(forbiddenKey)
    #expect(!forbiddenKeyResult.participant)
    #expect(!forbiddenKeyResult.trusted)

    // (e) One unrelated unexpected key added: both must REJECT.
    let unexpectedKey = try mutatedGemma4RuntimeConfig {
        $0["extra_unpinned_field"] = 1
    }
    let unexpectedKeyResult = runsBothGates(unexpectedKey)
    #expect(!unexpectedKeyResult.participant)
    #expect(!unexpectedKeyResult.trusted)

    // (f) One of the 120 per-tensor quantization overrides dropped: both
    // must REJECT. This is the mixed-precision trap the port notes call out
    // (section 1.3) -- a gate that only checked the three scalar keys would
    // miss it.
    let droppedOverride = try mutatedGemma4RuntimeConfig {
        guard var quantization = $0["quantization"] as? [String: Any] else { return }
        quantization.removeValue(forKey: "language_model.model.layers.0.mlp.gate_proj")
        $0["quantization"] = quantization
    }
    let droppedOverrideResult = runsBothGates(droppedOverride)
    #expect(!droppedOverrideResult.participant)
    #expect(!droppedOverrideResult.trusted)

    // (g) One override demoted from 8 bits to 4: both must REJECT -- the
    // exact silent-numerics failure mode section 1.3 warns about, now
    // structural instead of silent.
    let wrongOverrideWidth = try mutatedGemma4RuntimeConfig {
        guard var quantization = $0["quantization"] as? [String: Any] else { return }
        quantization["language_model.model.layers.0.mlp.gate_proj"] = [
            "group_size": 64, "bits": 4,
        ]
        $0["quantization"] = quantization
    }
    let wrongOverrideWidthResult = runsBothGates(wrongOverrideWidth)
    #expect(!wrongOverrideWidthResult.participant)
    #expect(!wrongOverrideWidthResult.trusted)

    // (h) Wrong model_type: both must REJECT.
    let wrongModelType = try mutatedGemma4RuntimeConfig {
        $0["model_type"] = "gemma4_text_v2"
    }
    let wrongModelTypeResult = runsBothGates(wrongModelType)
    #expect(!wrongModelTypeResult.participant)
    #expect(!wrongModelTypeResult.trusted)

    // (i) Wrong num_hidden_layers: both must REJECT.
    let wrongLayerCount = try mutatedGemma4RuntimeConfig {
        $0["num_hidden_layers"] = 29
    }
    let wrongLayerCountResult = runsBothGates(wrongLayerCount)
    #expect(!wrongLayerCountResult.participant)
    #expect(!wrongLayerCountResult.trusted)
}

/// The trusted gate's own error text, independent of the participant side,
/// itemizes the same missing/unexpected split the box printed verbatim --
/// not just "some rejection happened".
@Test
func gemma4A4BTrustedGateRejectsQwenShapedConfigWithTheExactBoxSplit() throws {
    let qwenShaped = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: try qwen36ConfigObject(),
        family: .qwen35
    )

    var rejection: MLXFastError?
    do {
        try MLXFastHarness.validateRuntimeWorkerPinnedConfigurationData(qwenShaped)
    } catch let error as MLXFastError {
        rejection = error
    }
    let message = try #require(rejection?.description)

    // See `gemma4A4BRuntimeWorkerGateRejectsQwenShapedConfigWithTheExactBoxSplit`
    // (Qwen35ArtifactContractTests.swift) for why this direction inverts the
    // box's own missing/unexpected labels.
    for key in ["attention_k_eq_v", "enable_moe_block", "num_experts"] {
        #expect(message.contains("missing required key \(key)"), "\(key) not reported missing")
    }
    for key in ["attn_output_gate", "mtp_num_hidden_layers", "partial_rotary_factor"] {
        #expect(message.contains("unexpected key \(key)"), "\(key) not reported unexpected")
    }
}
