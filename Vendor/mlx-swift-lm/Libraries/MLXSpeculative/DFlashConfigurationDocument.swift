// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// One parse of a DFlash drafter's `config.json` that yields BOTH the model
/// geometry and the quantization declaration its weights were packed under.
///
/// This mirrors `Gemma4AssistantConfigurationDocument` (the MTP head's loader,
/// Libraries/MLXLLM): the same bytes produce the config and the per-layer
/// quantization policy, so the loader cannot construct a drafter from one and
/// then bind weights packed under the other.
///
/// It exists because `DFlashConfiguration` deliberately does not model
/// quantization — a `quantization` key lands in its `ignoredConfigKeys`
/// diagnostic list, which is exactly the silent drop this document layer
/// closes. Keeping the policy OUT of `DFlashConfiguration` keeps the config's
/// own decoding, and therefore every unquantized checkpoint, untouched.
public struct DFlashConfigurationDocument {
    public let config: DFlashConfiguration

    /// `nil` when the checkpoint declares no quantization. A malformed or
    /// out-of-bounds declaration is a throw, never a `nil`.
    public let quantization: BaseConfiguration.PerLayerQuantization?

    public static func read(from directory: URL) throws -> Self {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashError.missingConfig(directory.path)
        }
        return try decode(try Data(contentsOf: configURL))
    }

    public static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder.json5()
        let config = try decoder.decode(DFlashConfiguration.self, from: data)

        let quantization: BaseConfiguration.PerLayerQuantization?
        do {
            quantization = try BaseConfiguration.declaredPerLayerQuantization(
                in: data, using: decoder)
        } catch {
            // A present-but-malformed declaration is refused by name. The old
            // behavior — drop it and carry on at full precision — is what this
            // lane exists to remove.
            throw DFlashError.undecodableQuantization(
                (error as? LocalizedError)?.errorDescription ?? "\(error)")
        }

        if let quantization, let violation = QuantizationGeometry.violation(in: quantization) {
            throw DFlashError.unsupportedQuantization(
                field: "quantization\(violation.fieldSuffix)",
                reason: violation.reason)
        }

        return Self(config: config, quantization: quantization)
    }
}
