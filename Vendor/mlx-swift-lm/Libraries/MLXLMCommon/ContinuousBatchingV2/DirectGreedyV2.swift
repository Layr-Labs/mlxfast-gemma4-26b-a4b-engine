import MLX

/// Opt-in model seam for decode paths that can produce the final greedy token
/// without materializing an otherwise-unused transformed vocabulary tensor.
/// The engine uses this only with its production sampler after proving that
/// every row is untransformed greedy sampling.
public protocol CBv2DirectGreedySteppableModel: CBv2SteppableModel {
    func supportsDirectGreedy(batchSize: Int) -> Bool
    func directGreedyTokens(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray
}

/// LanguageModel-level twin consumed by the standard steppable adapter.
public protocol CBv2LanguageModelDirectGreedyForwardable {
    func cbv2SupportsDirectGreedy(batchSize: Int) -> Bool
    func cbv2DirectGreedyTokens(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray
}

/// Exactly the state in which `LogitsPipelineV2` performs no arithmetic or
/// constraint transform and `SamplerV2` returns plain argmax. Top-k/p/min-p
/// are intentionally absent: the established pipeline ignores them on greedy
/// rows. A disabled repetition window likewise performs no transform.
@inline(__always)
func cbv2IsUntransformedGreedy(_ params: CBv2SamplingParams) -> Bool {
    params.temperature < LogitsPipelineV2.greedyEpsilon
        && params.logitBias.isEmpty
        && (params.repetitionPenalty == 1 || params.repetitionContextSize <= 0)
        && params.frequencyPenalty == 0
        && params.presencePenalty == 0
        && params.topLogprobs == 0
}
