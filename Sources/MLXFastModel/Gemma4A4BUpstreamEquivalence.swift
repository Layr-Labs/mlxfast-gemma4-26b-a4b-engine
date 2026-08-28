import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

public struct Gemma4A4BUpstreamEquivalenceStep: Codable, Equatable {
    public let label: String
    public let maximumAbsoluteLogitError: Float
    public let meanAbsoluteLogitError: Float
    public let runtimeToken: Int
    public let upstreamToken: Int

    public var tokensMatch: Bool {
        runtimeToken == upstreamToken
    }
}

public struct Gemma4A4BUpstreamEquivalenceReport: Codable, Equatable {
    public let promptTokenCount: Int
    public let decodeTokenCount: Int
    public let steps: [Gemma4A4BUpstreamEquivalenceStep]

    public func passes(maximumAbsoluteLogitError: Float) -> Bool {
        steps.allSatisfy {
            $0.tokensMatch
                && $0.maximumAbsoluteLogitError.isFinite
                && $0.maximumAbsoluteLogitError <= maximumAbsoluteLogitError
        }
    }
}

/// Development-only cross-check between this engine's Gemma 4 26B A4B
/// weight-loading pipeline and a standalone-vendored load of the SAME
/// checkpoint. Ported from `LagunaUpstreamEquivalence` (docs/gemma4-port-notes.md
/// section 6.4); read that file's header before changing this one, because two
/// things about the Gemma 4 port do not carry over from the Laguna shape
/// unchanged and are recorded here rather than silently "fixed":
///
/// 1. **There is no separate runtime model type.** Laguna has
///    `LagunaRuntimeModel` (this engine's reimplementation) vs `LagunaModel`
///    (the vendored reference) -- two different Swift types. Gemma 4 does not:
///    `Gemma4A4BRuntimeWeightCache` hand-builds the VENDORED `Gemma4TextModel`
///    directly (see that file's header comment) and never reimplements the
///    forward pass. So both legs of this comparison construct the SAME
///    `Gemma4TextModel` type. What differs -- and what this gate actually
///    tests -- is the WEIGHT-LOADING AND QUANTIZATION-WIRING PIPELINE: this
///    engine's own per-path override application
///    (`Gemma4A4BRuntimeWeightCache.quantizeWithPerPathWidths`, called here
///    directly so the "runtime" leg is the real scored-path code) versus the
///    vendored generic mechanism (`BaseConfiguration.PerLayerQuantization` +
///    `quantize(model:filter:)`, the same primitives
///    `MLXLMCommon.loadWeights`/`LLMModelFactory` use). Given section 1.3 of
///    the port notes, that pipeline is the single most likely source of a
///    silent numerical divergence on this checkpoint, so this is still a
///    load-bearing check despite comparing two instances of one type rather
///    than two independent implementations.
///
/// 2. **The quantization override keys need the same prefix restoration on
///    BOTH legs, not just the runtime one.** The checkpoint's `quantization`
///    block overrides are keyed by CHECKPOINT paths, e.g.
///    `language_model.model.layers.0.mlp.gate_proj`
///    (`Gemma4A4BQuantization` doc comment) -- because the transform passes
///    the block through verbatim, unchanged from the source VLM checkpoint's
///    own config. `quantize(model:)` walks the plain `Gemma4TextModel` module
///    tree this engine loads standalone (no VLM wrapper), so it reports paths
///    like `model.layers.0.mlp.gate_proj` -- no `language_model.` prefix. This
///    engine's own runtime path restores the prefix before every lookup
///    (`quantizeWithPerPathWidths`'s "NAME-SPACE MAPPING" comment). The
///    vendored `BaseConfiguration.PerLayerQuantization` type has no
///    corresponding restoration anywhere upstream -- `MLXLMCommon.loadWeights`
///    strips `language_model.` from the WEIGHT dictionary
///    (`_primaryWeightKeyPrefixStrip`) before quantizing, but never touches the
///    quantization-override dictionary's own keys. Calling the vendored
///    mechanism unmodified against this engine's standalone-text-tower config
///    would therefore miss every one of the 120 overrides on the "vendored"
///    leg -- not because of a real behavioral difference, but because of a
///    naming mismatch specific to extracting Gemma 4's text tower out of a
///    VLM checkpoint, which the vendored generic loader was never asked to do
///    before. This file performs the SAME `language_model.` strip on the
///    vendored `PerLayerQuantization`'s override keys that
///    `MLXLMCommon.loadWeights` already performs on the weight keys, so the
///    "vendored" leg is a meaningful standalone load rather than one that
///    always fails for a reason unrelated to whether this port is correct.
///    This is reported here, not fixed upstream -- see the PR description.
///
/// Both legs receive the SAME `[String: MLXArray]` object references (loaded
/// once, prefix-stripped once), each model's own `sanitize(weights:)` is
/// still called separately (as Laguna's harness does), and comparison is
/// exact by default: the benchmark protocol and scored path never call this
/// type.
public enum Gemma4A4BUpstreamEquivalence {
    public static func compare(
        weightsPath: String,
        promptTokens: [Int],
        decodeTokens: [Int]
    ) throws -> Gemma4A4BUpstreamEquivalenceReport {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B upstream equivalence requires at least one prompt token"
            )
        }

        let runtimeConfig = try Gemma4A4BConfig.load(from: weightsPath)
        let allTokens = promptTokens + decodeTokens
        guard allTokens.allSatisfy({ $0 >= 0 && $0 < runtimeConfig.vocabSize }) else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B upstream equivalence token is outside the model vocabulary"
            )
        }

        let loader = try Gemma4A4BWeightLoader(weightsPath: weightsPath)
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: runtimeConfig)

        let directory = URL(fileURLWithPath: weightsPath)
        let configData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let textConfiguration = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: configData)

        // `Gemma4TextConfiguration.init(from:)` already decodes a
        // `BaseConfiguration` from the same bytes internally and exposes it as
        // `perLayerQuantization` -- reusing that property (rather than
        // decoding `BaseConfiguration` a second time) keeps this the literal
        // vendored decode, not a parallel one that could read the JSON
        // differently.
        guard let vendoredPerLayerQuantization = textConfiguration.perLayerQuantization
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B upstream equivalence: config.json carries no vendored "
                    + "quantization block to compare against"
            )
        }
        let prefix = RuntimeWeightNameTracker.languageModelPrefix
        var vendoredOverrides: [String: BaseConfiguration.QuantizationOption] = [:]
        vendoredOverrides.reserveCapacity(
            vendoredPerLayerQuantization.perLayerQuantization.count)
        for (path, option) in vendoredPerLayerQuantization.perLayerQuantization {
            let strippedPath =
                path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
            vendoredOverrides[strippedPath] = option
        }
        let vendoredQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: vendoredPerLayerQuantization.quantization,
            perLayerQuantization: vendoredOverrides
        )

        // Load the checkpoint's raw arrays ONCE and strip the checkpoint's
        // "language_model." wrapper prefix ONCE -- the SAME MLXArray object
        // references then feed both models' own `sanitize(weights:)`,
        // mirroring the Laguna harness's "load once, install into both"
        // shape.
        let rawWeights = try loadArrays(directory: directory)
        let sharedWeights = try strippingWeightKeyPrefix(prefix, from: rawWeights)

        let runtimeModel = Gemma4TextModel(textConfiguration)
        let runtimeSanitized = runtimeModel.sanitize(weights: sharedWeights)
        try Gemma4A4BRuntimeWeightCache.quantizeWithPerPathWidths(
            model: runtimeModel, sanitized: runtimeSanitized, config: runtimeConfig)
        try runtimeModel.update(
            parameters: ModuleParameters.unflattened(runtimeSanitized), verify: [.all])
        eval(runtimeModel)

        let upstreamModel = Gemma4TextModel(textConfiguration)
        let upstreamSanitized = upstreamModel.sanitize(weights: sharedWeights)
        quantize(model: upstreamModel) { path, _ in
            guard upstreamSanitized["\(path).scales"] != nil else { return nil }
            return vendoredQuantization.quantization(layer: path)?.asTuple
        }
        try upstreamModel.update(
            parameters: ModuleParameters.unflattened(upstreamSanitized), verify: [.all])
        eval(upstreamModel)

        let runtimeCache = runtimeModel.newCache(parameters: nil)
        let upstreamCache = upstreamModel.newCache(parameters: nil)
        let prompt = MLXArray(promptTokens.map(Int32.init), [1, promptTokens.count])

        var steps: [Gemma4A4BUpstreamEquivalenceStep] = []
        steps.reserveCapacity(1 + decodeTokens.count)
        steps.append(
            try compareLastLogitRow(
                label: "prefill",
                runtime: runtimeModel(prompt, cache: runtimeCache),
                upstream: upstreamModel(prompt, cache: upstreamCache),
                vocabularySize: runtimeConfig.vocabSize
            )
        )

        for (index, token) in decodeTokens.enumerated() {
            let input = MLXArray([Int32(token)], [1, 1])
            steps.append(
                try compareLastLogitRow(
                    label: "decode-\(index)",
                    runtime: runtimeModel(input, cache: runtimeCache),
                    upstream: upstreamModel(input, cache: upstreamCache),
                    vocabularySize: runtimeConfig.vocabSize
                )
            )
        }

        return Gemma4A4BUpstreamEquivalenceReport(
            promptTokenCount: promptTokens.count,
            decodeTokenCount: decodeTokens.count,
            steps: steps
        )
    }

    /// Internal rather than private so `Gemma4A4BUpstreamEquivalenceTests` can
    /// exercise the real per-step comparator directly, at synthetic MLXArray
    /// scale, without loading `Gemma4A4BConfig` -- which hard-pins the exact
    /// production geometry (30 layers, hidden 2816, vocab 262144, ...) and so
    /// cannot represent a "tiny" checkpoint the way this method's Laguna
    /// counterpart's callers could. See the PR description and
    /// `Gemma4A4BUpstreamEquivalenceTests.swift` for why the synthetic-scale
    /// unit test targets this function instead of a shrunk end-to-end model.
    static func compareLastLogitRow(
        label: String,
        runtime runtimeLogits: MLXArray,
        upstream upstreamLogits: MLXArray,
        vocabularySize: Int
    ) throws -> Gemma4A4BUpstreamEquivalenceStep {
        let runtimeLast = runtimeLogits.reshaped([-1, vocabularySize])[-1]
        let upstreamLast = upstreamLogits.reshaped([-1, vocabularySize])[-1]
        guard runtimeLast.shape == upstreamLast.shape else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B upstream equivalence \(label) logit shapes differ: "
                    + "\(runtimeLast.shape) vs \(upstreamLast.shape)"
            )
        }

        let runtimeFinite = all(isFinite(runtimeLast))
        let upstreamFinite = all(isFinite(upstreamLast))
        eval(runtimeLast, upstreamLast, runtimeFinite, upstreamFinite)
        let runtimeLogitsAreFinite = runtimeFinite.item(Bool.self)
        let upstreamLogitsAreFinite = upstreamFinite.item(Bool.self)
        guard runtimeLogitsAreFinite, upstreamLogitsAreFinite else {
            throw MLXFastError.invalidInput(
                "Gemma 4 26B A4B upstream equivalence \(label) produced non-finite logits: "
                    + "runtime_finite=\(runtimeLogitsAreFinite) "
                    + "upstream_finite=\(upstreamLogitsAreFinite)"
            )
        }

        let difference = abs(
            runtimeLast.asType(.float32) - upstreamLast.asType(.float32)
        )
        eval(difference)
        return Gemma4A4BUpstreamEquivalenceStep(
            label: label,
            maximumAbsoluteLogitError: difference.max().item(Float.self),
            meanAbsoluteLogitError: difference.mean().item(Float.self),
            runtimeToken: Int(runtimeLast.argMax().item(Int32.self)),
            upstreamToken: Int(upstreamLast.argMax().item(Int32.self))
        )
    }
}
