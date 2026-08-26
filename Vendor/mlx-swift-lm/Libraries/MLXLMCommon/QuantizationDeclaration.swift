// Copyright © 2026 Apple Inc.

import Foundation

// A checkpoint's `quantization` block is a DECLARATION: participant-supplied
// bytes that decide which modules a loader quantizes, and with what geometry,
// before any weight is bound. This engine carries two replaceable heads whose
// loaders each read one — the MTP assistant head
// (`Gemma4AssistantConfigurationDocument`, Libraries/MLXLLM) and the DFlash
// drafter (`DFlashConfigurationDocument`, Libraries/MLXSpeculative). Both need
// the same bounds on that declaration and neither module can see the other's
// error type, so the bounds and their wording live here and each arm raises
// its own error from the reported violation.

/// The affine-quantization geometries a checkpoint declaration may name.
///
/// These are DoS bounds on untrusted input, not a reference pin: a head is
/// free to ship any geometry in range, and MLX itself rejects the combinations
/// it cannot pack. Nothing here asserts what a head "should" be quantized to.
public enum QuantizationGeometry {

    /// Bit widths a declared quantization may use.
    public static let supportedBits: ClosedRange<Int> = 2 ... 8

    /// Upper bound on a declared group size.
    public static let maximumGroupSize = 65_536

    /// Upper bound on how many per-layer entries a declaration may carry.
    public static let maximumPerLayerEntries = 8_192

    /// Upper bound on the length of a per-layer path, in UTF-8 bytes.
    public static let maximumLayerPathBytes = 1_024

    /// One out-of-bounds field, reported rather than thrown.
    ///
    /// `fieldSuffix` is appended to whatever the caller calls the declaration
    /// (`"quantization"`, `"textConfig.quantization"`, …), so both arms report
    /// the same field path and the same reason for the same defect.
    public struct Violation: Sendable, Equatable {
        public let fieldSuffix: String
        public let reason: String

        public init(fieldSuffix: String, reason: String) {
            self.fieldSuffix = fieldSuffix
            self.reason = reason
        }
    }

    /// `nil` when this geometry is one a loader will apply.
    public static func violation(
        in quantization: BaseConfiguration.Quantization
    ) -> Violation? {
        guard quantization.groupSize > 0 else {
            return Violation(fieldSuffix: ".groupSize", reason: "must be positive")
        }
        guard quantization.groupSize <= maximumGroupSize else {
            return Violation(
                fieldSuffix: ".groupSize",
                reason: "exceeds supported maximum \(maximumGroupSize)")
        }
        guard supportedBits.contains(quantization.bits) else {
            return Violation(
                fieldSuffix: ".bits",
                reason:
                    "must be between \(supportedBits.lowerBound) and \(supportedBits.upperBound)")
        }
        return nil
    }

    /// `nil` when every geometry in `policy` — the default one and each
    /// per-layer override — is one a loader will apply, and the declaration
    /// stays within its size bounds.
    ///
    /// Per-layer paths are walked in sorted order so the reported violation is
    /// the same one on every run over the same declaration.
    public static func violation(
        in policy: BaseConfiguration.PerLayerQuantization
    ) -> Violation? {
        if let base = policy.quantization, let violation = violation(in: base) {
            return violation
        }
        guard policy.perLayerQuantization.count <= maximumPerLayerEntries else {
            return Violation(fieldSuffix: "", reason: "has too many per-layer entries")
        }
        for path in policy.perLayerQuantization.keys.sorted() {
            guard path.utf8.count <= maximumLayerPathBytes else {
                return Violation(fieldSuffix: ".\(path)", reason: "layer path is too long")
            }
            guard case .quantize(let value)? = policy.perLayerQuantization[path] else {
                continue
            }
            if let violation = violation(in: value) {
                return Violation(
                    fieldSuffix: ".\(path)\(violation.fieldSuffix)",
                    reason: violation.reason)
            }
        }
        return nil
    }
}

extension BaseConfiguration {

    /// Decode ONLY the quantization declaration out of a `config.json`.
    ///
    /// `BaseConfiguration` itself requires `model_type`, which is right for a
    /// model-factory entry and wrong for a head whose own schema does not
    /// require it (`DFlashConfiguration` defaults it). This reads the same
    /// `quantization` / `quantization_config` keys through the same container
    /// the full decode uses, so a declaration means exactly what it means on
    /// the MTP path, without dragging in a field neither the geometry nor the
    /// policy depends on.
    ///
    /// - Returns: `nil` when the config declares no quantization at all.
    /// - Throws: a `DecodingError` when a declaration is present but malformed.
    ///   A malformed declaration is never "no declaration": silently loading
    ///   such a checkpoint at full precision would misreport what ran.
    public static func declaredPerLayerQuantization(
        in data: Data,
        using decoder: JSONDecoder = JSONDecoder.json5()
    ) throws -> PerLayerQuantization? {
        try decoder.decode(QuantizationDeclarationDocument.self, from: data)
            .container?
            .perLayerQuantization
    }
}

/// The `quantization` / `quantization_config` slice of a `config.json`.
struct QuantizationDeclarationDocument: Decodable {
    let container: BaseConfiguration.QuantizationContainer?

    enum CodingKeys: String, CodingKey {
        case quantization
        case quantizationConfiguration = "quantization_config"
    }

    init(from decoder: any Decoder) throws {
        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        container =
            try keyed.decodeIfPresent(
                BaseConfiguration.QuantizationContainer.self, forKey: .quantization)
            ?? keyed.decodeIfPresent(
                BaseConfiguration.QuantizationContainer.self, forKey: .quantizationConfiguration)
    }
}
