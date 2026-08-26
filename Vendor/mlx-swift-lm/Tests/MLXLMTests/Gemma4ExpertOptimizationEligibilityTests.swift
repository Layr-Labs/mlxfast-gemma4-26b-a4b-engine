// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

/// The retained production tuple pairs CBv2-prefill weighted unsort with the safe
/// Gemma 4 expert-QMM (R1) kernel. Weighted unsort measured materially slower
/// on its own, so both must gate on exactly one configuration contract: the
/// exact production expert topology AND the quantization contract that
/// `classify_gemma4_expert_qmm` enforces before it inspects topology
/// (`group_size == 64 && bits == 4 && mode == affine`; anything else is
/// `fallback_quantization`).
@Suite("Gemma4 coupled expert-optimization eligibility")
struct Gemma4ExpertOptimizationEligibilityTests {

    @Test("weighted expert reduction is scoped to scheduled CBv2 prefill")
    func weightedReductionScope() {
        #expect(!gemma4AllowsWeightedExpertUnsort(schedulePrefill: false))
        #expect(gemma4AllowsWeightedExpertUnsort(schedulePrefill: true))
    }

    /// The measured checkpoint's expert topology, parameterized only on the
    /// declared weight quantization and on whether the topology matches.
    private func config(
        bits: Int? = 4,
        groupSize: Int? = 64,
        productionTopology: Bool = true
    ) throws -> Gemma4TextConfiguration {
        var fields = [String]()
        if let bits { fields.append("\"bits\": \(bits)") }
        if let groupSize { fields.append("\"group_size\": \(groupSize)") }
        let quantizationJSON =
            fields.isEmpty ? "" : ", \"quantization\": {\(fields.joined(separator: ", "))}"
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(productionTopology ? 2816 : 2048),
                "num_hidden_layers": 30,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "use_bidirectional_attention": "vision"
                \(quantizationJSON)
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// One row per checkpoint shape the pair can plausibly meet, with the
    /// single expected eligibility answer for BOTH weighted unsort and safe R1.
    private typealias Row = (label: String, config: Gemma4TextConfiguration, eligible: Bool)

    private func matrix() throws -> [Row] {
        try [
            ("4-bit/64 production", config(), true),
            ("8-bit/64", config(bits: 8), false),
            ("6-bit/64", config(bits: 6), false),
            ("4-bit/32", config(groupSize: 32), false),
            ("4-bit/128", config(groupSize: 128), false),
            ("bits only", config(groupSize: nil), false),
            ("group size only", config(bits: nil), false),
            ("unquantized", config(bits: nil, groupSize: nil), false),
            ("4-bit/64 off-topology", config(productionTopology: false), false),
            ("8-bit/64 off-topology", config(bits: 8, productionTopology: false), false),
        ]
    }

    @Test func exactFourBitGroup64ProductionCheckpointIsEligibleForBoth() throws {
        let c = try config()

        #expect(gemma4SupportsProductionExpertTopology(c))
        #expect(gemma4SupportsSafeExpertQMMQuantization(c))
        // Safe-R1 geometry eligibility, i.e. `expertQMMGeometryEligible`.
        #expect(gemma4SupportsCoupledExpertOptimizations(c))
        // Weighted-unsort effectiveness once the process requests it.
        #expect(gemma4ShouldFuseWeightedUnsort(c, requested: true))
        #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: false))
    }

    @Test func eightBitCheckpointIsIneligibleForBoth() throws {
        let c = try config(bits: 8)

        // The topology is still exact; only the selector's quantization
        // contract fails, which is what previously left weighted unsort
        // engaged with safe R1 categorically unable to hit.
        #expect(gemma4SupportsProductionExpertTopology(c))
        #expect(!gemma4SupportsSafeExpertQMMQuantization(c))
        #expect(!gemma4SupportsCoupledExpertOptimizations(c))
        #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))

    }

    @Test func nonSixtyFourGroupSizeIsIneligibleForBoth() throws {
        for groupSize in [32, 128] {
            let c = try config(groupSize: groupSize)
            #expect(gemma4SupportsProductionExpertTopology(c))
            #expect(!gemma4SupportsSafeExpertQMMQuantization(c))
            #expect(!gemma4SupportsCoupledExpertOptimizations(c))
            #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))
        }
    }

    /// Unquantized and half-declared checkpoints cannot satisfy the selector,
    /// so neither half of the pair may engage on them.
    @Test func unquantizedOrPartialQuantizationMetadataIsIneligibleForBoth() throws {
        for c in [
            try config(bits: nil, groupSize: nil),
            try config(bits: 4, groupSize: nil),
            try config(bits: nil, groupSize: 64),
        ] {
            #expect(!gemma4SupportsSafeExpertQMMQuantization(c))
            #expect(!gemma4SupportsCoupledExpertOptimizations(c))
            #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))
        }
    }

    /// The quantization contract is additive: it never rescues a checkpoint
    /// whose expert topology already fails.
    @Test func quantizationContractDoesNotRelaxTheTopologyRequirement() throws {
        let c = try config(productionTopology: false)

        #expect(gemma4SupportsSafeExpertQMMQuantization(c))
        #expect(!gemma4SupportsProductionExpertTopology(c))
        #expect(!gemma4SupportsCoupledExpertOptimizations(c))
        #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))
    }

    /// The release blocker: weighted unsort effective while safe R1 is
    /// ineligible is an unmeasured partial state. Across every supported
    /// checkpoint shape and both request values, weighted effectiveness is
    /// exactly the request AND safe-R1 eligibility — so that state is
    /// unreachable in either direction.
    @Test func noWeightedOnlyStateIsReachableOnSupportedCheckpoints() throws {
        for (label, c, expected) in try matrix() {
            let safeR1Eligible = gemma4SupportsCoupledExpertOptimizations(c)
            #expect(safeR1Eligible == expected, "\(label) safe-R1 eligibility")

            for requested in [false, true] {
                let weightedEffective = gemma4ShouldFuseWeightedUnsort(c, requested: requested)
                #expect(
                    weightedEffective == (requested && safeR1Eligible),
                    "\(label) weighted effectiveness (requested=\(requested))")
                #expect(
                    !weightedEffective || safeR1Eligible,
                    "\(label) reached weighted-only state (requested=\(requested))")
            }
        }
    }

    /// VLM checkpoints carry quantization beside `text_config`; the overlay
    /// must reach the coupled predicate, not just the nested spelling.
    @Test func rootLevelQuantizationOverlayDrivesCoupledEligibility() throws {
        var c = try config(bits: nil, groupSize: nil)
        #expect(!gemma4SupportsCoupledExpertOptimizations(c))

        c.mergeQuantization(BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        #expect(gemma4SupportsCoupledExpertOptimizations(c))
        #expect(gemma4ShouldFuseWeightedUnsort(c, requested: true))

        c.mergeQuantization(BaseConfiguration.Quantization(groupSize: 64, bits: 8))
        #expect(!gemma4SupportsCoupledExpertOptimizations(c))
        #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))
    }

    @Test func expertPerLayerQuantizationOverridesFailClosed() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "use_bidirectional_attention": "vision",
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "model.layers.0.experts.switch_glu.gate_proj": false
                }
            }
            """
        let c = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        #expect(c.hasExpertQuantizationOverrides)
        #expect(!gemma4SupportsSafeExpertQMMQuantization(c))
        #expect(!gemma4SupportsCoupledExpertOptimizations(c))
        #expect(!gemma4ShouldFuseWeightedUnsort(c, requested: true))

        let roundTripped = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: JSONEncoder().encode(c))
        #expect(roundTripped.hasExpertQuantizationOverrides)
        #expect(!gemma4SupportsCoupledExpertOptimizations(roundTripped))
    }

    @Test func unrelatedPerLayerQuantizationOverrideDoesNotDisableExperts() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "use_bidirectional_attention": "vision",
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "model.embed_tokens": false
                }
            }
            """
        let c = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        #expect(!c.hasExpertQuantizationOverrides)
        #expect(gemma4SupportsCoupledExpertOptimizations(c))
    }

    @Test func textWrapperMergesRootQuantizationModeAndExpertOverrides() throws {
        let json = """
            {
                "model_type": "gemma4",
                "vocab_size": 262144,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2816,
                    "num_hidden_layers": 30,
                    "enable_moe_block": true,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704,
                    "use_bidirectional_attention": "vision"
                },
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "mxfp4",
                    "model.layers.0.experts.switch_glu.gate_proj": false
                }
            }
            """
        let wrapper = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(json.utf8))
        #expect(wrapper.textConfig.quantizationMode == .mxfp4)
        #expect(wrapper.textConfig.hasExpertQuantizationOverrides)
        #expect(!gemma4SupportsSafeExpertQMMQuantization(wrapper.textConfig))
        #expect(!gemma4ShouldFuseWeightedUnsort(wrapper.textConfig, requested: true))
    }

    /// A tiny MoE checkpoint still keeps the generic SwitchGLU reduction: it
    /// carries the 4-bit/group-64 contract but not the production topology, so
    /// the model's reported eligibility must be false on both surfaces and
    /// must agree with the shared predicates that drive real dispatch.
    @Test func modelReportingMatchesTheSharedPredicates() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 16,
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "enable_moe_block": true,
                "num_experts": 4,
                "top_k_experts": 2,
                "moe_intermediate_size": 16,
                "use_bidirectional_attention": "vision",
                "quantization": {"bits": 4, "group_size": 64}
            }
            """
        let c = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(0x4E45)
        let model = Gemma4TextModel(c)

        #expect(gemma4SupportsSafeExpertQMMQuantization(c))
        #expect(!gemma4SupportsProductionExpertTopology(c))
        #expect(model.expertQMMGeometryEligible == gemma4SupportsCoupledExpertOptimizations(c))
        #expect(model.weightedExpertUnsortEffective == gemma4ShouldFuseWeightedUnsort(c))
        #expect(!model.expertQMMGeometryEligible)
        #expect(!model.weightedExpertUnsortEffective)
    }
}
