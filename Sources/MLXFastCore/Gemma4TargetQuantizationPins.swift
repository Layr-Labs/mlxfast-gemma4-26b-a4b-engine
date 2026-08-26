import Foundation

/// The tensor element types the pinned target quantization uses, spelled
/// without an MLX dependency.
///
/// `MLXFastCore` links into the trusted `mlxfast-swift` binary and therefore
/// carries no MLX/model dependency, so it cannot name `MLX.DType`. The two
/// spellings here are the only ones the affine contract needs, and the harness
/// maps them onto the real `DType` at the one place it reads a live tensor.
/// A raw value is the MLX `DType` spelling so the mapping is by name rather
/// than by an ordering that could drift.
public enum Gemma4PinnedTensorDType: String, Equatable, Sendable {
    case uint32
    case bfloat16
}

/// The Gemma 4 26B A4B TARGET's frozen quantization geometry.
///
/// WHY THIS TYPE EXISTS. Until now the target's quantization format was
/// enforced ENTIRELY DECLARATIVELY: `validateRuntimeWorkerPinnedConfiguration`
/// reads the on-disk `config.json` and checks `bits`/`group_size`/`mode` plus
/// the 120-entry per-tensor override table, and nothing ever read back the
/// LIVE `QuantizedLinear` / `QuantizedEmbedding` / `QuantizedSwitchLinear`
/// modules. The step from the declaration to the live modules is
/// `Gemma4A4BRuntimeWeights.quantizeWithPerPathWidths`, which lives in
/// `Sources/MLXFastModel` -- a `benchmark.json` `editablePaths` entry. So
/// participant code could quantize the target IN MEMORY at any geometry it
/// liked while every disk-side gate still passed, and a lossier target would
/// substitute a degraded model for the accepted one (CLAUDE.md, "the target
/// quantization is frozen as shipped"). `validateLoadedTargetQuantization`
/// closes that gap, and these are the numbers it closes it against.
///
/// WHY IT LIVES HERE. The check is only worth its bytes if the EXPECTED side
/// cannot be moved by the code being checked. `Sources/MLXFastModel`
/// (`Gemma4A4BConfig`) and `Sources/MLXFastTransform` (`PinnedGeometry`) are
/// both editable, so reading the pins from either would make the gate
/// circular: a candidate that re-quantized the target would simply re-declare
/// the geometry it re-quantized to and pass. `MLXFastCore` is trusted scope
/// (Package.swift: the trusted-harness source scope is the manifest,
/// `Package.resolved`, `Sources/MLXFastCLI`, `Sources/MLXFastTrustedHarness`
/// and `Sources/MLXFastCore`), so these values are outside the candidate's
/// reach.
///
/// The values are the hand-derived ones the compiled-out trusted twin
/// (`Sources/MLXFastTrustedHarness/Gemma4RuntimeWorker.swift`,
/// `validateRuntimeWorkerPinnedConfigurationData`) already enforces against
/// the on-disk config: fallback `bits=4`, `group_size=64`, `mode=affine`, and
/// four projection families per layer promoted to `bits=8`, `group_size=64`.
/// The two gates are deliberately the same numbers read from the same place --
/// one against the DECLARATION, one against the LOADED MODULES.
public struct Gemma4TargetQuantizationPins: Equatable, Sendable {
    /// Bit width every quantized target module carries unless its runtime path
    /// is one of the promoted override paths.
    public let fallbackBits: Int

    /// Bit width the four promoted projection families carry, on every layer.
    public let overrideBits: Int

    /// Group size. One value: the checkpoint uses the same group size for the
    /// fallback and for every override, and the trusted config gate pins both.
    public let groupSize: Int

    /// `QuantizationMode.affine`'s raw value. Compared as a string so this
    /// module stays MLX-free.
    public let modeName: String

    /// Element type of the packed code tensor (`weight`). Affine codes are
    /// bit-packed into 32-bit words.
    public let packedWeightDType: Gemma4PinnedTensorDType

    /// Element type of the `scales` and `biases` companions. The checkpoint's
    /// own `dtype` is `bfloat16` and the MLX affine conversion keeps the
    /// companions at the source element type
    /// (`Sources/MLXFastTransform/Gemma4A4BCheckpointValidation.swift`'s
    /// `addAffine` pins exactly this pair: `U32` codes, `BF16` companions).
    public let scaleDType: Gemma4PinnedTensorDType

    /// Bits per packed word. Not a free parameter -- it is `UInt32`'s width --
    /// but it is named rather than spelled `32` inline so the shape relation
    /// below reads as an identity instead of as a magic number.
    public let packedWordBits: Int

    /// The four projection families promoted to `overrideBits` on every layer.
    public let overrideFamilies: [String]

    public init(
        fallbackBits: Int,
        overrideBits: Int,
        groupSize: Int,
        modeName: String,
        packedWeightDType: Gemma4PinnedTensorDType,
        scaleDType: Gemma4PinnedTensorDType,
        packedWordBits: Int,
        overrideFamilies: [String]
    ) {
        self.fallbackBits = fallbackBits
        self.overrideBits = overrideBits
        self.groupSize = groupSize
        self.modeName = modeName
        self.packedWeightDType = packedWeightDType
        self.scaleDType = scaleDType
        self.packedWordBits = packedWordBits
        self.overrideFamilies = overrideFamilies
    }

    /// The shipped Gemma 4 26B A4B target's geometry.
    public static let production = Gemma4TargetQuantizationPins(
        fallbackBits: 4,
        overrideBits: 8,
        groupSize: 64,
        modeName: "affine",
        packedWeightDType: .uint32,
        scaleDType: .bfloat16,
        packedWordBits: 32,
        overrideFamilies: Gemma4A4BConfigKeys.quantizationOverrideFamilies
    )

    /// The RUNTIME module paths that must carry `overrideBits`.
    ///
    /// Built by CONSTRUCTION from the layer count and the four families, the
    /// same discipline the trusted config gate uses for the checkpoint-side
    /// key set, so a table that lost entries shows up as a missing path rather
    /// than as a count that still happens to match.
    ///
    /// NAME SPACE. The trusted config gate builds CHECKPOINT keys
    /// (`language_model.model.layers.7.mlp.gate_proj`) because it reads
    /// `config.json`. This builds RUNTIME module paths
    /// (`model.layers.7.mlp.gate_proj`) because it is compared against
    /// `Module.leafModules()`, and the runtime strips the `language_model.`
    /// prefix on load (`RuntimeWeightNameTracker`). The two differ by exactly
    /// that prefix; the prefix is spelled here rather than imported because
    /// `RuntimeWeightNameTracker` lives in the editable `Sources/MLXFastModel`
    /// and the whole point of this type is that the expected side is not
    /// reachable from there.
    public func runtimeOverridePaths(layerCount: Int) -> Set<String> {
        var paths: Set<String> = []
        for layer in 0..<layerCount {
            for family in overrideFamilies {
                paths.insert("\(Self.runtimeModelPrefix).layers.\(layer).\(family)")
            }
        }
        return paths
    }

    /// The runtime module path of the text tower's root, i.e.
    /// `Gemma4A4BWeightNames.modelPrefix` (`language_model.model`) with the
    /// `language_model.` prefix the loader strips already removed.
    public static let runtimeModelPrefix = "model"
}
