// CBv2MTPDeterministicFixture.swift
//
// Wide-margin target weights for strict parity tests whose verification
// shapes intentionally differ from ordinary one-token decode shapes.

import Foundation
import MLX
import MLXNN

@testable import MLXLLM

/// Make the target predict `(inputToken + 1) % vocabularySize` with a wide
/// logit margin. Attention and MLP outputs are zeroed at their final
/// projections, so the residual carries a deterministic token code through
/// every real Gemma layer while KV projection and cache paths still execute.
func stabilizeCBv2MTPGreedyCycleTarget(_ target: Gemma4TextModel) {
    let vocabularySize = target.vocabularySize
    let hiddenSize = target.configuration.hiddenSize
    let bitCount = max(1, Int(ceil(log2(Double(vocabularySize)))))
    precondition(bitCount <= hiddenSize)

    var codebook = [Float](repeating: 0, count: vocabularySize * hiddenSize)
    for token in 0 ..< vocabularySize {
        for column in 0 ..< hiddenSize {
            let bit = column % bitCount
            codebook[token * hiddenSize + column] =
                token & (1 << bit) == 0 ? -0.25 : 0.25
        }
    }

    var cycleHead = [Float](repeating: 0, count: codebook.count)
    for outputToken in 0 ..< vocabularySize {
        let inputToken = (outputToken - 1 + vocabularySize) % vocabularySize
        for column in 0 ..< hiddenSize {
            cycleHead[outputToken * hiddenSize + column] =
                codebook[inputToken * hiddenSize + column]
        }
    }

    let parameters = target.parameters().flattened()
    let embeddingKey = "model.embed_tokens.weight"
    let headKey = "lm_head.weight"
    guard let embedding = parameters.first(where: { $0.0 == embeddingKey })?.1,
        let head = parameters.first(where: { $0.0 == headKey })?.1
    else {
        preconditionFailure("deterministic MTP fixture requires an untied Gemma target")
    }

    var replacements: [String: MLXArray] = [
        embeddingKey: MLXArray(codebook, [vocabularySize, hiddenSize]).asType(embedding.dtype),
        headKey: MLXArray(cycleHead, [vocabularySize, hiddenSize]).asType(head.dtype),
    ]
    var zeroedProjections = 0
    var fixedLayerScalars = 0
    for (key, value) in parameters {
        if key.hasSuffix(".self_attn.o_proj.weight")
            || key.hasSuffix(".mlp.down_proj.weight")
        {
            replacements[key] = MLXArray.zeros(value.shape, dtype: value.dtype)
            zeroedProjections += 1
        } else if key.hasSuffix(".layer_scalar") {
            replacements[key] = MLXArray.ones(value.shape, dtype: value.dtype)
            fixedLayerScalars += 1
        } else if key == "model.norm.weight" {
            replacements[key] = MLXArray.ones(value.shape, dtype: value.dtype)
        }
    }
    precondition(zeroedProjections == target.configuration.numHiddenLayers * 2)
    precondition(fixedLayerScalars == target.configuration.numHiddenLayers)
    target.update(parameters: ModuleParameters.unflattened(replacements))
}

func cbv2MTPExpectedGreedyCycle(
    after token: Int, count: Int, vocabularySize: Int
) -> [Int] {
    (0 ..< count).map { (token + $0 + 1) % vocabularySize }
}
