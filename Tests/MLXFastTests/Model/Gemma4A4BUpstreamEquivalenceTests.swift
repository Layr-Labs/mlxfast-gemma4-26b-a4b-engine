import Foundation
import MLX
import MLXFastCore
@testable import MLXFastModel
import Testing

// Gemma 4 26B A4B upstream-equivalence gate. Ported from
// `LagunaUpstreamEquivalence` / `Gemma4CorrectnessTests.lagunaRuntimeMatchesVendoredUpstreamOnM5WhenEnabled`
// -- see docs/gemma4-port-notes.md section 6.4 and
// `Sources/MLXFastModel/Gemma4A4BUpstreamEquivalence.swift`'s header for what
// changed in the port and why.
//
// This file has two kinds of coverage, deliberately:
//
// 1. `gemma4RuntimeMatchesVendoredUpstreamOnM5WhenEnabled` -- the real gate.
//    Env-gated and skipped by default, so it is CI/laptop-safe: it loads the
//    full ~production Gemma 4 26B A4B checkpoint and is only meaningful on a
//    machine that has it staged (the M5 box).
//
// 2. The synthetic-scale tests below it -- an unit test of the comparator
//    plumbing that needs no config, no checkpoint, and none of this gate's own
//    env vars. `Gemma4A4BConfig.load(from:)` hard-pins the exact production
//    geometry (`validateFrozenInvariants`: 30 layers, hidden 2816, vocab
//    262144, ...), so unlike a from-scratch Swift model type there is no way
//    to build a "tiny Gemma4A4BConfig" the way a shrunk end-to-end model would
//    need. The synthetic-scale test therefore targets
//    `Gemma4A4BUpstreamEquivalence.compareLastLogitRow` directly -- the exact
//    per-step comparator `compare(weightsPath:...)` calls at every prefill and
//    decode step -- with hand-built toy MLXArrays, bypassing config/weight
//    loading entirely. That still proves the two things that matter: the
//    comparator agrees when logits and argmaxes match, and it DETECTS a
//    mismatch (both a logit-error mismatch and an argmax mismatch) rather than
//    silently passing. Reported in the PR description as a place the Laguna
//    shape did not transfer cleanly.
//
//    These still construct and `eval()` real MLXArrays, which needs the MLX
//    Metal backend available (`tools/build-mlx-metallib.sh`) -- the same
//    machine-capability constraint every other MLX-computation test in this
//    target observes (see `MLXTensorBridgeTests.swift`), unrelated to this
//    gate's own weights/checkpoint gating above. So they follow the same
//    established `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` convention rather than
//    running unconditionally.

/// Deterministic, fully public token sequence for the gated gate below. This
/// gate proves RUNTIME-vs-VENDORED agreement given IDENTICAL weights and
/// IDENTICAL input tokens; it never compares against a golden or expected
/// continuation (there is no golden file in this gate at all -- see the
/// header comment), so token IDENTITY carries no information of its own, only
/// validity (`0..<vocabSize`) and determinism. A fixed linear-congruential
/// sequence seeded by a small public constant keeps the prompt reproducible
/// across runs without depending on a tokenizer, an external file, or any
/// hidden fixture.
private func gemma4SyntheticTokens(count: Int, vocabSize: Int, seed: UInt64) -> [Int] {
    var state = seed
    var tokens: [Int] = []
    tokens.reserveCapacity(count)
    for _ in 0..<count {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        tokens.append(Int(state % UInt64(vocabSize)))
    }
    return tokens
}

@Test
func gemma4RuntimeMatchesVendoredUpstreamOnM5WhenEnabled() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_GEMMA4_UPSTREAM_EQUIVALENCE"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_GEMMA4_EQUIVALENCE_WEIGHTS_PATH"]
    )
    let tolerance = Float(
        environment["MLXFAST_GEMMA4_EQUIVALENCE_MAX_ABS_ERROR"] ?? "0"
    ) ?? 0

    let config = try Gemma4A4BConfig.load(from: weightsPath)
    let promptTokens = gemma4SyntheticTokens(
        count: MLXFastConstants.correctnessPromptTokens,
        vocabSize: config.vocabSize,
        seed: 0x67656d6d6134
    )
    let decodeTokens = gemma4SyntheticTokens(
        count: 8,
        vocabSize: config.vocabSize,
        seed: 0x646563_6f6465
    )

    let report = try Gemma4A4BUpstreamEquivalence.compare(
        weightsPath: weightsPath,
        promptTokens: promptTokens,
        decodeTokens: decodeTokens
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let encoded = try encoder.encode(report)
    print(String(decoding: encoded, as: UTF8.self))
    #expect(report.passes(maximumAbsoluteLogitError: tolerance))
}

// MARK: - Synthetic-scale comparator plumbing
// (gated on MLXFAST_RUN_MLX_RUNTIME_TESTS, this target's standing convention
// for any test that evaluates a real MLXArray -- see the file header.)

@Test
func gemma4UpstreamEquivalenceComparatorPassesOnIdenticalLogits() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    let runtimeLogits = MLXArray([Float(1.0), 2.0, 3.0, 0.5], [1, 4])
    let upstreamLogits = MLXArray([Float(1.0), 2.0, 3.0, 0.5], [1, 4])

    let step = try Gemma4A4BUpstreamEquivalence.compareLastLogitRow(
        label: "prefill",
        runtime: runtimeLogits,
        upstream: upstreamLogits,
        vocabularySize: 4
    )

    #expect(step.tokensMatch)
    #expect(step.runtimeToken == 2)
    #expect(step.upstreamToken == 2)
    #expect(step.maximumAbsoluteLogitError == 0)
    #expect(step.meanAbsoluteLogitError == 0)

    let report = Gemma4A4BUpstreamEquivalenceReport(
        promptTokenCount: 4, decodeTokenCount: 0, steps: [step]
    )
    #expect(report.passes(maximumAbsoluteLogitError: 0))
}

/// Proves the gate can FAIL: a small numeric divergence that leaves the
/// argmax unchanged is caught by the logit-error bound, and a divergence that
/// flips the argmax is caught by `tokensMatch` even at a loose tolerance.
/// Neither is a hypothetical -- this is exactly the failure shape trap 1.3 in
/// docs/gemma4-port-notes.md warns about: right shapes, wrong numerics, only
/// visible as a small per-element logit delta or a flipped decode token.
@Test
func gemma4UpstreamEquivalenceComparatorDetectsAMismatch() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    let runtimeLogits = MLXArray([Float(1.0), 2.0, 3.0, 0.5], [1, 4])

    // Same argmax (index 2), but a real numeric divergence at index 3.
    let upstreamCloseButDiverged = MLXArray([Float(1.0), 2.0, 3.0, 1.7], [1, 4])
    let closeStep = try Gemma4A4BUpstreamEquivalence.compareLastLogitRow(
        label: "decode-0",
        runtime: runtimeLogits,
        upstream: upstreamCloseButDiverged,
        vocabularySize: 4
    )
    #expect(closeStep.tokensMatch)
    #expect(closeStep.maximumAbsoluteLogitError == 1.2)
    let closeReport = Gemma4A4BUpstreamEquivalenceReport(
        promptTokenCount: 4, decodeTokenCount: 0, steps: [closeStep]
    )
    // Exact-zero default tolerance: caught even though the argmax agrees.
    #expect(!closeReport.passes(maximumAbsoluteLogitError: 0))
    // A loosened diagnostic tolerance that covers this delta still passes --
    // proving the gate only fails when a real divergence exceeds tolerance,
    // not unconditionally.
    #expect(closeReport.passes(maximumAbsoluteLogitError: 1.5))

    // A divergence that flips the argmax outright: runtime's max is index 2
    // (value 3.0), upstream's is index 1 (value 6.0).
    let upstreamFlippedArgmax = MLXArray([Float(1.0), 6.0, 3.0, 0.5], [1, 4])
    let flippedStep = try Gemma4A4BUpstreamEquivalence.compareLastLogitRow(
        label: "decode-1",
        runtime: runtimeLogits,
        upstream: upstreamFlippedArgmax,
        vocabularySize: 4
    )
    #expect(!flippedStep.tokensMatch)
    #expect(flippedStep.runtimeToken == 2)
    #expect(flippedStep.upstreamToken == 1)
    let flippedReport = Gemma4A4BUpstreamEquivalenceReport(
        promptTokenCount: 4, decodeTokenCount: 0, steps: [flippedStep]
    )
    // No tolerance rescues a flipped greedy token -- `tokensMatch` is checked
    // unconditionally by `passes(maximumAbsoluteLogitError:)`.
    #expect(!flippedReport.passes(maximumAbsoluteLogitError: 1_000))
}

@Test
func gemma4UpstreamEquivalenceComparatorRejectsNonFiniteLogits() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    let runtimeLogits = MLXArray([Float(1.0), 2.0, 3.0], [1, 3])
    let upstreamLogits = MLXArray([Float.nan, 2.0, 3.0], [1, 3])
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4A4BUpstreamEquivalence.compareLastLogitRow(
            label: "prefill",
            runtime: runtimeLogits,
            upstream: upstreamLogits,
            vocabularySize: 3
        )
    }
}
