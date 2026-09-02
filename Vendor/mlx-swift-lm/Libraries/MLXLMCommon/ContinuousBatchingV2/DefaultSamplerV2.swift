// DefaultSamplerV2.swift
//
// The production `CBv2StepSampler`: WS-E's `LogitsPipelineV2` (bias →
// penalties → temperature → top-k/top-p/min-p) composed with `SamplerV2`
// (all-greedy argmax fast path; keyed Gumbel-max otherwise). This is
// EngineV2's default sampler; `CBv2GreedySampler` remains the deterministic
// stub for scheduler tests.
//
// Statefulness & exactness:
//  - Reconfiguration happens ONLY when the row-ID order changes between
//    `sample` calls. `rowContext()` (confirmed history) rebuilds the
//    per-row tensors; when `pendingSampledTokens` is present (chained
//    decode reconfiguring mid-chain), the pending [B] tokens are folded in
//    ON-DEVICE via the pipelines' own `commit`, so penalty counts and RNG
//    step indices are exact — a pure function of each request's history,
//    never of batch composition or host visibility timing.
//  - Between reconfigurations, per-step `commit(sampledTokens:)` maintains
//    the state incrementally (device scatter-adds; no host syncs anywhere
//    on this path — `sample` builds graph nodes only).
//
// Batch-composition invariance: every pipeline transform is row-independent
// and the RNG is keyed (seed, requestID, per-request step), so a row's
// tokens cannot depend on its batchmates (research report 12 item 5).

import Foundation
import MLX

public final class CBv2DefaultSampler: CBv2StepSampler {

    private var pipeline: LogitsPipelineV2?
    private let sampler: SamplerV2
    private let constraintSampler = CBv2TokenConstraintSampler()
    private var configuredIDs: [CBv2RequestID] = []
    /// The lazy logprob gather built by the most recent `sample` call, until
    /// the loop consumes it via `takeStepLogprobs` (take semantics).
    private var pendingStepLogprobs: CBv2StepLogprobs?
    /// Number of `sample` calls that built logprob gather nodes
    /// (telemetry/test hook — must stay 0 when no row asks for logprobs).
    public private(set) var logprobGatherCount = 0
    /// Test hook: the composed pipeline's logprob-capture counter.
    var pipelineLogprobBuildCount: Int { pipeline?.logprobBuildCount ?? 0 }
    public var supportsTokenConstraints: Bool { true }

    /// - Parameter fallbackSeed: engine-level seed for rows without a
    ///   per-request seed (fixed at init so nil-seed rows stay
    ///   batch-invariant within a process; random by default).
    public init(fallbackSeed: UInt64? = nil) {
        self.sampler = SamplerV2(fallbackSeed: fallbackSeed)
    }

    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        let vocab = logits.dim(-1)
        if pipeline?.vocabSize != vocab {
            pipeline = LogitsPipelineV2(vocabSize: vocab)
            configuredIDs = []
        }
        let pipeline = self.pipeline!

        if requestIDs != configuredIDs {
            let rows = rowContext()
            pipeline.setRows(rows)
            sampler.setRows(rows)
            constraintSampler.configure(rows)
            configuredIDs = requestIDs
            if let pendingSampledTokens {
                // Fold the chained in-flight tokens into the fresh state so
                // penalty counts and per-row RNG step indices include them.
                pipeline.commit(sampledTokens: pendingSampledTokens)
                sampler.commit()
            }
        }
        // Same IDs ⇒ the incremental commits below already covered every
        // token sampled since configuration (including any pending ones).

        // Apply the grammar exactly once, after arithmetic transforms and
        // before top-k/top-p/min-p. This preserves the valid language even
        // for malformed-but-decodable penalties that can resurrect
        // -infinity, without rebuilding the dense mask twice. Raw logprobs
        // still come from the original unmasked distribution by contract.
        let output = pipeline.process(
            logits,
            rawLogprobsFrom: logits,
            hardMask: constraintSampler.hasRows ? { [constraintSampler] transformed in
                constraintSampler.mask(
                    transformed, requestIDs: requestIDs)
            } : nil)
        let tokens = sampler.sample(from: output.sampling)
        pipeline.commit(sampledTokens: tokens)
        sampler.commit()

        // Lazy logprob gather from the RAW (pre-transform) logprobs — graph
        // nodes only, no host sync; the loop materializes at finalization.
        // Rows with topLogprobs == 0 pay nothing beyond riding the batch's
        // shared capture (and the whole branch is skipped when NO row asks).
        if let rawLogprobs = output.rawLogprobs {
            let k = params.reduce(0) { max($0, $1.topLogprobs) }
            pendingStepLogprobs = CBv2StepLogprobs(
                rows: requestIDs,
                topLogprobsPerRow: params.map(\.topLogprobs),
                gathered: CBv2Logprobs.gather(
                    rawLogprobs: rawLogprobs, sampledTokens: tokens, k: k))
            logprobGatherCount += 1
        } else {
            pendingStepLogprobs = nil
        }
        return tokens
    }

    public func takeStepLogprobs() -> CBv2StepLogprobs? {
        defer { pendingStepLogprobs = nil }
        return pendingStepLogprobs
    }

    /// Invalidate the configured fingerprint when a finished request's id
    /// was part of it: a FUTURE request may legally reuse that id, and an
    /// identical `requestIDs` array must then reconfigure (fresh penalties,
    /// RNG step 0) instead of inheriting the retired request's state
    /// (PR#62 review). Forcing a full `setRows` on the next `sample` is
    /// exact — reconfiguration is a pure function of `rowContext()`.
    public func requestDidFinish(_ id: CBv2RequestID) {
        constraintSampler.requestDidFinish(id)
        if configuredIDs.contains(id) {
            configuredIDs = []
        }
    }

    public func confirmSampledTokens(
        _ tokens: [Int], requestIDs: [CBv2RequestID]
    ) {
        constraintSampler.confirm(tokens: tokens, requestIDs: requestIDs)
    }

    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
        constraintSampler.failure(for: id)
    }
}

// MARK: - Logitsless greedy head (LGH-001)

/// `sample` collapses to one `argMax` exactly when every row is greedy and no
/// stateful or shape-changing transform is armed --- the same condition
/// `LogitsPipelineV2.process` uses for its identity fast path, evaluated on
/// host data before the step's logits exist. The engine may then take the
/// tokens straight from a fused head.
extension CBv2DefaultSampler: CBv2FusedGreedySampler {

    public func admitsFusedGreedy(params: [CBv2SamplingParams]) -> Bool {
        guard !params.isEmpty else { return false }
        return params.allSatisfy { p in
            p.temperature < LogitsPipelineV2.greedyEpsilon
                && p.topLogprobs == 0
                && p.logitBias.isEmpty
                && !(p.repetitionPenalty != 1 && p.repetitionContextSize > 0)
                && p.frequencyPenalty == 0
                && p.presencePenalty == 0
        }
    }

    /// A step the sampler never saw leaves its per-row penalty counts and RNG
    /// step indices one behind. Dropping the configured fingerprint makes the
    /// next `sample` rebuild both from confirmed history plus the in-flight
    /// token, which is exactly what a membership change already does --- so
    /// the state stays a pure function of each request's own history.
    public func noteFusedGreedySample() {
        configuredIDs = []
        pendingStepLogprobs = nil
    }
}
