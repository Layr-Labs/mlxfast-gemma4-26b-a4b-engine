// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

@Suite("Gemma4 MTP automatic policy")
struct Gemma4MTPPolicyTests {
    private func config(
        hiddenSize: Int,
        layers: Int,
        enableMoe: Bool = false,
        quantizationBits: Int? = nil
    ) throws -> Gemma4TextConfiguration {
        let quantizationJSON =
            if let quantizationBits {
                """
                ,
                            "quantization": {"bits": \(quantizationBits), "group_size": 64}
                """
            } else {
                ""
            }
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": \(layers),
            "enable_moe_block": \(enableMoe),
            "num_experts": \(enableMoe ? 128 : 0),
            "top_k_experts": \(enableMoe ? 8 : 0),
            "moe_intermediate_size": \(enableMoe ? 704 : 0)
            \(quantizationJSON)
        }
        """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    @Test func e2bUsesBatchedMTPForBatchedRequests() throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 1536, layers: 35))

        #expect(policy.family == .e2b)
        #expect(policy.supportsBatchedMTP)
        #expect(policy.strategy(forBatchSize: 1) == .singleStream(blockSize: 5))
        #expect(policy.strategy(forBatchSize: 4) == .batched(blockSize: 3))
        #expect(policy.singleStreamAction(
            generatedTokens: 32, maxTokens: nil) == .mtp(blockSize: 5))
    }

    @Test func quantizedE2BUsesTokenRangeAwareSingleStreamPolicy() throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 1536, layers: 35, quantizationBits: 4))

        #expect(policy.family == .e2b)
        #expect(!policy.supportsBatchedMTP)
        #expect(policy.strategy(forBatchSize: 1) == .singleStream(blockSize: 3))
        #expect(policy.strategy(forBatchSize: 4) == .singleStream(blockSize: 3))
        #expect(policy.singleStreamAction(
            generatedTokens: 0, maxTokens: 12) == .mtp(blockSize: 4))
        #expect(policy.singleStreamAction(
            generatedTokens: 15, maxTokens: 12) == .mtp(blockSize: 4))
        #expect(policy.singleStreamAction(
            generatedTokens: 0, maxTokens: 64) == .targetOnly)
        #expect(policy.singleStreamAction(
            generatedTokens: 64, maxTokens: 1024) == .targetOnly)
        #expect(policy.singleStreamAction(
            generatedTokens: 255, maxTokens: 1024) == .targetOnly)
        #expect(policy.singleStreamAction(
            generatedTokens: 256, maxTokens: 1024) == .mtp(blockSize: 4))
        #expect(policy.singleStreamAction(
            generatedTokens: 0, maxTokens: nil) == .targetOnly)
    }

    @Test func e4bDisablesAutomaticMTPUntilCorrectPathBeatsBaseline() throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 2560, layers: 42))

        #expect(policy.family == .e4b)
        #expect(!policy.enablesAutomaticMTP)
        #expect(!policy.supportsBatchedMTP)
        #expect(policy.strategy(forBatchSize: 1) == .singleStream(blockSize: 3))
        #expect(policy.strategy(forBatchSize: 4) == .singleStream(blockSize: 3))
        #expect(policy.singleStreamAction(
            generatedTokens: 64, maxTokens: nil) == .targetOnly)
    }

    @Test func moeA4BFallsBackToSingleStreamForBatchedRequests() throws {
        let policy = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 2816, layers: 30, enableMoe: true))

        #expect(policy.family == .moeA4B)
        #expect(!policy.supportsBatchedMTP)
        #expect(policy.strategy(forBatchSize: 1) == .singleStream(blockSize: 3))
        #expect(policy.strategy(forBatchSize: 4) == .singleStream(blockSize: 3))
    }

    @Test func unknownModelsStayConservative() throws {
        let dense = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 4096, layers: 48))
        let moe = Gemma4MTPAutomaticPolicy.automatic(
            for: try config(hiddenSize: 4096, layers: 48, enableMoe: true))

        #expect(dense.family == .unknownDense)
        #expect(dense.strategy(forBatchSize: 4) == .singleStream(blockSize: 4))
        #expect(moe.family == .unknownMoE)
        #expect(moe.strategy(forBatchSize: 4) == .singleStream(blockSize: 3))
    }
}
