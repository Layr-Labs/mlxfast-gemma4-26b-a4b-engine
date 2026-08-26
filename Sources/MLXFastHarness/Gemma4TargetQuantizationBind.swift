import Foundation
import MLX
import MLXFastCore
import MLXNN

// THE LOADED-STATE TARGET QUANTIZATION BIND (David ruling 2026-08-26).
//
// THE HOLE THIS CLOSES. The target's quantization format was enforced entirely
// DECLARATIVELY. `validateRuntimeWorkerPinnedConfiguration` reads the on-disk
// `config.json` and checks `bits=4` / `group_size=64` / `mode=affine` plus the
// 120-entry per-tensor 8-bit override table. Nothing ever read back the LIVE
// `QuantizedLinear` / `QuantizedEmbedding` / `QuantizedSwitchLinear` modules.
// The single step from the declaration to the live modules is
// `Gemma4A4BRuntimeWeights.quantizeWithPerPathWidths`, and it lives in
// `Sources/MLXFastModel`, which IS a `benchmark.json` `editablePaths` entry.
// So participant code could hand `quantize(model:)` any geometry it liked --
// 2 bits, a different group size, a different mode -- and every disk-side gate
// would still pass, because the disk never changed. A re-quantized target is
// not an optimization of the accepted model; it is a different, degraded model
// wearing the accepted model's config (CLAUDE.md, "the target quantization is
// frozen as shipped").
//
// WHERE THIS FILE LIVES AND WHY. `Sources/MLXFastHarness` (SwiftPM target
// `MLXFastRuntimeWorkerSupport`, Package.swift:67-88) is NOT in
// `benchmark.json`'s `editablePaths`; `Sources/MLXFastModel` and
// `Sources/MLXFastTransform` are. A gate a candidate can edit is not a gate.
// The pinned numbers come from `MLXFastCore` for the same reason -- see
// `Gemma4TargetQuantizationPins`.
//
// SCOPE: TARGET ONLY. The MTP assistant head and the DFlash drafter are
// EXPLICITLY allowed to be re-quantized on load (David ruling 2026-08-26, and
// `docs/participant-contract.md` section 4.4). This function therefore takes
// the model to walk as a parameter and is only ever called on the target
// instance. It must never be handed a drafter -- doing so would refuse a
// re-quantization the contract grants.

/// Verify that the LOADED target model carries exactly the pinned
/// quantization geometry.
///
/// Walks every leaf module of `model` that conforms to `MLXNN.Quantized` and
/// checks it against `pins`. Modules whose runtime path is one of the promoted
/// override paths must be at `pins.overrideBits`; every other quantized module
/// must be at `pins.fallbackBits`. Group size and mode are pinned everywhere.
///
/// NON-CIRCULARITY IS THE POINT OF THE WHOLE FUNCTION. The EXPECTED side comes
/// only from `pins` and `numHiddenLayers`, both of which are non-editable
/// constants at the production call site. Nothing about the expectation is
/// read from `config.json`, from `Gemma4A4BConfig`, or from anything else a
/// candidate can move. The ACTUAL side is read from the loaded module, and it
/// is read TWICE, from two independent places:
///
///   (a) the module's own `bits` / `groupSize` / `mode` properties, and
///   (b) the module's REAL materialized tensors, off `parameters()`.
///
/// WHY (b) EXISTS. `Quantized` is a protocol, and `bits` / `groupSize` /
/// `mode` are ordinary properties. A hand-written conformer -- or a subclass
/// of `QuantizedLinear` in one of the editable vendored model files -- can
/// return any triple it likes from those three properties while the tensors it
/// actually multiplies are something else entirely. (a) alone is therefore a
/// self-report, and this whole file exists because self-reports were the
/// problem. (b) reads the packed code tensor and its scale/bias companions:
/// their dtypes, their agreeing leading axes, and the packing identity
///
///     weight.last * packedWordBits == scales.last * groupSize * bits
///
/// which says the codes really are `bits`-wide and really are grouped
/// `groupSize` at a time. Those tensors are what the kernels read, so they
/// cannot lie about the arithmetic that will run.
///
/// WHY (a) STILL EXISTS -- the honest residual. (b) alone does NOT pin the
/// geometry: the identity only constrains the PRODUCT `groupSize * bits`, so
/// it admits the whole family `{ (g, b) : g * b == pinnedGroupSize *
/// pinnedBits }` -- (32, 8) and (128, 2) satisfy it just as well as (64, 4).
/// (a) is what selects the single member of that family, and (a) is what the
/// kernels are actually dispatched with. Neither check subsumes the other, so
/// both run on every module and a module passes only when they agree with each
/// other and with the pins.
///
/// - Parameters:
///   - model: the TARGET text tower. Never a drafter -- see the scope note at
///     the top of this file.
///   - numHiddenLayers: layer count used to build the expected override-path
///     set by construction.
///   - pins: the frozen geometry. Defaults to the shipped target's.
func validateLoadedTargetQuantization(
    model: Module,
    numHiddenLayers: Int,
    pins: Gemma4TargetQuantizationPins = .production
) throws {
    guard numHiddenLayers > 0 else {
        throw MLXFastError.invalidInput(
            "target quantization is frozen: cannot verify a loaded target "
                + "against \(numHiddenLayers) layers"
        )
    }

    let expectedOverridePaths = pins.runtimeOverridePaths(layerCount: numHiddenLayers)
    let expectedPackedDType = mlxDType(pins.packedWeightDType)
    let expectedScaleDType = mlxDType(pins.scaleDType)

    var errors: [String] = []
    var foundOverridePaths: Set<String> = []

    for (path, module) in model.leafModules().flattened() {
        guard let quantized = module as? Quantized else { continue }

        let isOverride = expectedOverridePaths.contains(path)
        if isOverride { foundOverridePaths.insert(path) }
        let expectedBits = isOverride ? pins.overrideBits : pins.fallbackBits

        // (a) The module's self-reported geometry -- the triple the MLX
        // quantized kernels are dispatched with.
        guard quantized.bits == expectedBits,
              quantized.groupSize == pins.groupSize,
              quantized.mode.rawValue == pins.modeName
        else {
            errors.append(
                "\(path) is bits=\(quantized.bits) "
                    + "group_size=\(quantized.groupSize) "
                    + "mode=\(quantized.mode.rawValue), pinned "
                    + "bits=\(expectedBits) group_size=\(pins.groupSize) "
                    + "mode=\(pins.modeName)"
            )
            // One violation per module. The tensor checks below would restate
            // the same divergence in shape terms and bury the readable cause.
            continue
        }

        // (b) The real materialized tensors.
        let parameters = Dictionary(
            module.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        guard let weight = parameters["weight"] else {
            errors.append("\(path) reports bits=\(quantized.bits) but carries no packed weight tensor")
            continue
        }
        guard let scales = parameters["scales"] else {
            errors.append("\(path) reports bits=\(quantized.bits) but carries no scales tensor")
            continue
        }
        // `mode` is pinned to affine above, and affine is exactly the mode that
        // requires a bias companion; the mxfp modes carry none. So a missing
        // `biases` here is a real divergence, not a mode-dependent absence.
        guard let biases = parameters["biases"] else {
            errors.append("\(path) is pinned mode=\(pins.modeName) but carries no biases tensor")
            continue
        }

        guard weight.dtype == expectedPackedDType else {
            errors.append(
                "\(path) packed weight is \(weight.dtype), pinned "
                    + "\(pins.packedWeightDType.rawValue)"
            )
            continue
        }
        guard scales.dtype == expectedScaleDType, biases.dtype == expectedScaleDType else {
            errors.append(
                "\(path) scales/biases are \(scales.dtype)/\(biases.dtype), pinned "
                    + "\(pins.scaleDType.rawValue)/\(pins.scaleDType.rawValue)"
            )
            continue
        }

        // Leading axes must agree. `QuantizedSwitchLinear` (the MoE expert
        // projections) packs a 3-D `[experts, out, packed_in]` weight against
        // 3-D `[experts, out, groups]` companions, so the relation below is
        // written on the LAST axis and the leading axes are compared
        // separately -- that covers the 2-D `QuantizedLinear` /
        // `QuantizedEmbedding` case and the stacked expert case with one rule.
        guard weight.shape.count == scales.shape.count,
              weight.shape.count >= 2,
              scales.shape == biases.shape,
              weight.shape.dropLast() == scales.shape.dropLast()
        else {
            errors.append(
                "\(path) packed weight shape \(weight.shape) does not agree with "
                    + "scales \(scales.shape) / biases \(biases.shape)"
            )
            continue
        }

        // The packing identity. `packedIn * 32 == groups * groupSize * bits`:
        // both sides are the contracted axis's element count times `bits`.
        let packedBitCount = weight.shape[weight.shape.count - 1] * pins.packedWordBits
        let declaredBitCount =
            scales.shape[scales.shape.count - 1] * pins.groupSize * expectedBits
        guard packedBitCount == declaredBitCount else {
            errors.append(
                "\(path) packs \(weight.shape[weight.shape.count - 1]) words for "
                    + "\(scales.shape[scales.shape.count - 1]) group(s), which is "
                    + "\(packedBitCount) bits against the \(declaredBitCount) bits "
                    + "pinned bits=\(expectedBits) group_size=\(pins.groupSize) requires"
            )
            continue
        }
    }

    // COMPLETENESS. Every promoted path must be PRESENT as a quantized module.
    // Without this, dropping a promoted tensor to some other representation --
    // dequantizing it, or replacing the module with something that is not
    // `Quantized` at all -- would pass, because the loop above only sees
    // modules that are still quantized.
    let missingOverrides = expectedOverridePaths.subtracting(foundOverridePaths).sorted()
    if !missingOverrides.isEmpty {
        errors.append(
            "\(missingOverrides.count) pinned bits=\(pins.overrideBits) override path(s) "
                + "are not quantized modules on the loaded target, first: "
                + missingOverrides.prefix(3).joined(separator: ", ")
        )
    }

    guard errors.isEmpty else {
        // House style, matching the trusted config gate: the first few named
        // violations plus a total, so a systematic re-quantization reads as one
        // line instead of 400.
        let summary = errors.prefix(3).joined(separator: "; ")
        let tail = errors.count > 3 ? "; ... (\(errors.count) violation(s) total)" : ""
        throw MLXFastError.invalidInput(
            "target quantization is frozen: " + summary + tail
        )
    }
}

/// Re-verify the target IMMEDIATELY BEFORE a measured window opens.
///
/// WHY A SECOND CHECK EXISTS. The startup bind in `Gemma4Runtime.runWorker`
/// verifies the instance the weight cache built, and checks that the cache
/// keeps handing back that same instance. Neither of those stops the SAME
/// instance from being mutated in place afterwards: `Module.update(modules:)`
/// is the seam `quantize(model:)` itself uses, and editable request-path code
/// can call it on the loaded target between the hello and the first timed
/// forward. A startup-only gate therefore verifies a model that no longer has
/// to be the model that runs.
///
/// So this runs again at the top of every verb that opens a measured window,
/// and it re-runs the FULL check rather than only comparing identity. Identity
/// is deliberately not sufficient and the tests pin that: an in-place
/// `update(modules:)` leaves `===` perfectly intact while replacing the
/// arithmetic underneath it.
///
/// WHAT IT COSTS. One traversal of the module tree with shape and dtype reads.
/// No GPU work, no allocation of tensors, no forward. It lands inside the
/// measured window because that is the only place it can be trustworthy, and it
/// lands identically on BOTH legs -- the same non-editable code ships in the
/// baseline workspace and the candidate workspace -- so it is a constant added
/// to both sides of every ratio, once per phase, not once per token.
///
/// - Parameters:
///   - phase: the window being opened, named into the refusal so a rejection
///     says WHEN it fired, not only what it found.
///   - verifiedTarget: the instance the startup bind accepted.
///   - currentTarget: the instance this request is about to run. Same object as
///     `verifiedTarget` on any honest path; compared, not assumed.
///   - numHiddenLayers: from the trusted constants, never from the candidate's
///     config.
///   - pins: the frozen geometry.
func revalidateTargetForMeasuredWindow(
    phase: String,
    verifiedTarget: Module,
    currentTarget: Module,
    numHiddenLayers: Int,
    pins: Gemma4TargetQuantizationPins = .production
) throws {
    guard currentTarget === verifiedTarget else {
        throw MLXFastError.invalidInput(
            "target quantization is frozen: the \(phase) window is about to run a "
                + "different target instance than the one verified at startup"
        )
    }
    do {
        try validateLoadedTargetQuantization(
            model: verifiedTarget, numHiddenLayers: numHiddenLayers, pins: pins)
    } catch let error as MLXFastError {
        // Keep the inner diagnostic verbatim -- it names the module and both
        // geometries -- and add only WHEN it fired. A caller that sees this
        // knows the model passed at startup and was changed afterwards, which
        // is a different and more serious fact than failing at load.
        throw MLXFastError.invalidInput(
            "pre-measure re-check (\(phase)): " + error.description
        )
    }
}

/// Map a pinned dtype spelling onto the MLX type. Total by construction --
/// `Gemma4PinnedTensorDType` has exactly these two cases -- so there is no
/// fallback branch to fail open through.
private func mlxDType(_ pinned: Gemma4PinnedTensorDType) -> DType {
    switch pinned {
    case .uint32:
        return .uint32
    case .bfloat16:
        return .bfloat16
    }
}
