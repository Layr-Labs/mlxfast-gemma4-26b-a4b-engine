import CryptoKit
import Foundation
import MLXFastCore
import Testing
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
@testable import MLXFastTransform

/// Read an integer shape pin the manifest declares in its OWN header, e.g.
/// `#   MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS: 10`. Returns `nil` when the
/// key is absent or its value is not an integer, so callers couple to the
/// manifest's declared shape rather than a decoupled external literal.
func manifestHeaderPin(_ lines: [String], _ key: String) -> Int? {
    for line in lines {
        guard line.hasPrefix("#"), let colon = line.range(of: "\(key):") else {
            continue
        }
        let token = line[colon.upperBound...]
            .split(whereSeparator: { $0 == " " || $0 == "\t" })
            .first
        return token.flatMap { Int($0) }
    }
    return nil
}

/// The public `config.json` of `mlx-community/Qwen3.6-27B-4bit`, checked in at
/// `fixtures/qwen3_6_27b_config.json` normalized only for JSON formatting.
///
/// Note the artifact-snake prefix `qwen3_6_27b` deliberately omits a vendor
/// segment where the Laguna fixtures carry `poolside_`: mlx-community is the
/// distributor of an Alibaba model, not its author, so a `mlx_community_`
/// prefix would name the wrong party. See docs/qwen3.6-weight-contract.md.
@Test(.disabled("Gemma 4 port: pins the public config fixture against the geometry constants; fixtures/qwen3_6_27b_config.json is Qwen geometry. Unblocks with the box-extracted Gemma config fixture -- docs/gemma4-port-notes.md section 6.1."))
func qwen36PublicConfigFixturePinsExactArtifactSemantics() throws {
    let root = try qwen36ConfigObject()
    #expect(try qwen36ConfigData().count < 10_000)
    #expect(Set(root.keys) == [
        "architectures",
        "eos_token_id",
        "image_token_id",
        "language_model_only",
        "model_type",
        "quantization",
        "quantization_config",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
    ])
    #expect(root["model_type"] as? String == "qwen3_5")
    #expect(root["architectures"] as? [String] == ["Qwen3_5ForConditionalGeneration"])
    #expect(root["tie_word_embeddings"] as? Bool == false)

    // The multimodal wrapper is present but out of scope: the transform
    // selects `language_model.*` only, and `vision_config` never reaches the
    // runtime config.
    let vision = try #require(root["vision_config"] as? [String: Any])
    #expect(vision["depth"] as? Int == 27)
    #expect(vision["out_hidden_size"] as? Int == MLXFastConstants.hiddenSize)

    let text = try #require(root["text_config"] as? [String: Any])
    #expect(text["model_type"] as? String == "qwen3_5_text")
    #expect(text["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(text["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
    #expect(text["intermediate_size"] as? Int == MLXFastConstants.intermediateSize)
    #expect(text["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(text["num_attention_heads"] as? Int == MLXFastConstants.attentionHeads)
    #expect(text["num_key_value_heads"] as? Int == 4)
    #expect(text["head_dim"] as? Int == 256)
    #expect(text["full_attention_interval"] as? Int == 4)
    #expect(text["tie_word_embeddings"] as? Bool == false)
    #expect(text["mtp_num_hidden_layers"] as? Int == 1)
    #expect(text["mtp_use_dedicated_embeddings"] as? Bool == false)
    #expect(text["pad_token_id"] is NSNull)

    let expectedLayerTypes = (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 3 ? "full_attention" : "linear_attention"
    }
    #expect(text["layer_types"] as? [String] == expectedLayerTypes)

    // Partial RoPE: 0.25 of head_dim 256 = 64 rotary dimensions, theta 1e7.
    let rope = try #require(text["rope_parameters"] as? [String: Any])
    #expect(rope["rope_type"] as? String == "default")
    #expect(rope["rope_theta"] as? Double == 10_000_000)
    #expect(rope["partial_rotary_factor"] as? Double == 0.25)
    #expect(rope["mrope_interleaved"] as? Bool == true)
    #expect(rope["mrope_section"] as? [Int] == [11, 11, 10])
    #expect(text["partial_rotary_factor"] as? Double == 0.25)

    // The same affine spec is published twice; both must agree.
    let spec = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: root)
    #expect(spec.groupSize == 64)
    #expect(spec.bits == 4)
    #expect(spec.mode == "affine")
}

/// End-to-end pin: the runtime config this checkpoint transforms into is
/// exactly what the runtime worker's pinned-configuration gate accepts.
///
/// RE-DERIVED FOR GEMMA 4 26B A4B (2026-08-23), now that
/// `fixtures/gemma4_26b_a4b_config.json` exists (laptop-generated from the
/// PUBLIC pinned HF revision's own `config.json` -- see
/// `Gemma4A4BArtifactFixtureSupport.swift` for provenance and the "not
/// box-verified" caveat) and `validateRuntimeWorkerPinnedConfigurationData`
/// has been ported off the Qwen `qwen3_5_text` schema onto
/// `Gemma4A4BConfig`. This used to be
/// `qwen36TransformOutputSatisfiesTheRuntimeWorkerPinnedConfigGate`,
/// `.disabled` since the gate stopped matching the tower this repository
/// actually drives; it is not restorable in its old Qwen form because the
/// worker gate now rejects Qwen configs by construction (see
/// `gemma4A4BRuntimeWorkerGateRejectsQwenShapedConfigWithTheExactBoxSplit`
/// below), so this is the Gemma-shaped replacement rather than a re-enable.
///
/// This is the interface the weight contract documents, exercised in one step
/// -- transform emitter on the left, worker gate on the right -- so a change
/// to either side that breaks the other fails here rather than on box 3.
@Test
func gemma4A4BTransformOutputSatisfiesTheRuntimeWorkerPinnedConfigGate() throws {
    let root = try gemma4A4BConfigObject()
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: root) == .gemma4A4B)

    let runtimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: root,
        family: .gemma4A4B
    )
    try validateRuntimeWorkerPinnedConfigurationData(runtimeConfig)

    // The emitted config is the source `text_config` plus one `quantization`
    // block -- no vision/audio config, no duplicated `quantization_config`,
    // no `text_config` wrapper.
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: runtimeConfig) as? [String: Any]
    )
    #expect(emitted["vision_config"] == nil)
    #expect(emitted["audio_config"] == nil)
    #expect(emitted["quantization_config"] == nil)
    #expect(emitted["text_config"] == nil)
    let quantization = try #require(emitted["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["mode"] as? String == "affine")
    // The 120 per-tensor 8-bit overrides (port notes section 1.3) ride along
    // verbatim -- the three scalar keys above are not the whole block, and a
    // gate or emitter that rebuilt this block from just the fallback triple
    // (the way the Qwen branch legitimately does, because the Qwen block only
    // HAS three keys) would silently drop all 120 of them.
    #expect(quantization.count == 123)

    // And it loads as the model target's own config type, closing the loop
    // between the transform emitter, the worker gate, and the config loader
    // that discriminates this target -- all three now agree on one schema.
    let config = try Gemma4A4BConfig.load(data: runtimeConfig)
    #expect(config.modelType == "gemma4_text")
    #expect(config.numHiddenLayers == MLXFastConstants.numHiddenLayers)
    #expect(config.quantization.overrides.count == 120)
}

/// The mirror image of the exact box refusal this whole port responds to,
/// reproduced as a regression fixture.
///
/// The box observed a real TRANSFORMED GEMMA 4 config refused by the OLD
/// (pre-port) Qwen-shaped gate: from that gate's point of view, the Qwen
/// keys it expected were "missing" and the Gemma keys the config actually
/// carried were "unexpected". This test runs the OPPOSITE artifact through
/// the NEW (ported) gate -- a Qwen 3.6-shaped runtime config -- which
/// inverts which half of the split each key set falls into: the Gemma keys
/// this gate requires and the Qwen config lacks are now "missing", and the
/// Qwen-only keys the Gemma schema does not recognize are now "unexpected".
/// Both directions are refused; a gate that only checked "does this throw"
/// could not tell a gate that moved to the wrong schema from one that
/// correctly rejects every foreign artifact, which is exactly the failure
/// mode this port fixes.
@Test
func gemma4A4BRuntimeWorkerGateRejectsQwenShapedConfigWithTheExactBoxSplit() throws {
    let qwenRoot = try qwen36ConfigObject()
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: qwenRoot) == .qwen35)
    let qwenRuntimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: qwenRoot,
        family: .qwen35
    )

    var rejection: MLXFastError?
    do {
        try validateRuntimeWorkerPinnedConfigurationData(qwenRuntimeConfig)
    } catch let error as MLXFastError {
        rejection = error
    }
    let message = try #require(rejection?.description)

    // Missing: pinned Gemma 4 keys no Qwen config carries (the box's
    // "unexpected" list, inverted for this direction).
    for key in [
        "attention_k_eq_v", "enable_moe_block", "final_logit_softcapping",
        "global_head_dim", "hidden_activation", "hidden_size_per_layer_input",
        "moe_intermediate_size", "num_experts", "num_global_key_value_heads",
        "num_kv_shared_layers", "sliding_window", "top_k_experts",
        "use_bidirectional_attention", "use_double_wide_mlp",
        "vocab_size_per_layer_input",
    ] {
        #expect(message.contains("missing required key \(key)"), "\(key) not reported missing")
    }
    // Unexpected: Qwen-only keys the Gemma schema does not know (the box's
    // "missing" list, inverted for this direction).
    for key in [
        "attn_output_gate", "full_attention_interval", "hidden_act",
        "linear_conv_kernel_dim", "linear_key_head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_value_head_dim", "mamba_ssm_dtype", "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings", "partial_rotary_factor",
    ] {
        #expect(message.contains("unexpected key \(key)"), "\(key) not reported unexpected")
    }
    // pad_token_id is required on BOTH schemas but with a different shape --
    // Qwen carries it as null, Gemma requires a non-null int -- so it lands
    // in neither bucket above; assert its own, third, failure mode instead.
    #expect(message.contains("required key pad_token_id must not be null"))
}

/// A single dropped required key is refused, not silently accepted.
@Test
func gemma4A4BRuntimeWorkerGateRejectsAConfigMissingOneRequiredKey() throws {
    let root = try gemma4A4BConfigObject()
    let runtimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: root,
        family: .gemma4A4B
    )
    var mutable = try #require(
        try JSONSerialization.jsonObject(with: runtimeConfig) as? [String: Any]
    )
    mutable.removeValue(forKey: "attention_k_eq_v")
    let mutated = try JSONSerialization.data(withJSONObject: mutable)

    var rejection: MLXFastError?
    do {
        try validateRuntimeWorkerPinnedConfigurationData(mutated)
    } catch let error as MLXFastError {
        rejection = error
    }
    #expect(rejection?.description.contains("missing required key attention_k_eq_v") == true)
}

/// A single added unknown key is refused, not silently ignored -- the point
/// of the exact-key-set check documented on `Gemma4A4BConfig
/// .validateKeyPresence`: `Decodable`-style parsing would let an unexpected
/// behaviour-bearing field through unnoticed.
@Test
func gemma4A4BRuntimeWorkerGateRejectsAConfigWithOneUnexpectedKey() throws {
    let root = try gemma4A4BConfigObject()
    let runtimeConfig = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: root,
        family: .gemma4A4B
    )
    var mutable = try #require(
        try JSONSerialization.jsonObject(with: runtimeConfig) as? [String: Any]
    )
    mutable["moe_router_logit_softcapping"] = 0.0
    let mutated = try JSONSerialization.data(withJSONObject: mutable)

    var rejection: MLXFastError?
    do {
        try validateRuntimeWorkerPinnedConfigurationData(mutated)
    } catch let error as MLXFastError {
        rejection = error
    }
    #expect(
        rejection?.description.contains("forbidden key moe_router_logit_softcapping is present")
            == true
    )
}

@Test
func qwen36TensorInventoryFixturePinsAllPublicHeaders() throws {
    let fixtureData = try Data(contentsOf: qwen36InventoryFixtureURL)
    let fixture = try qwen36InventoryFixture()

    #expect(fixtureData.count < 250_000)
    #expect(fixture.schemaVersion == 1)
    #expect(fixture.source.repository == qwen36Repository)
    #expect(fixture.source.revision == qwen36Revision)
    #expect(fixture.source.configSHA256 == qwen36ConfigSHA256)
    #expect(
        fixture.source.indexSHA256
            == "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
    )
    #expect(fixture.source.indexTotalSize == 16_054_262_240)
    #expect(fixture.canonicalization.contains("[name,dtype,shape,shard_name]"))

    // The published artifact carries the text tower AND the vision tower; the
    // transform selects only the former.
    #expect(fixture.tensors.count == 2_180)
    #expect(fixture.summary.tensorCount == fixture.tensors.count)
    #expect(fixture.summary.dtypeCounts == ["BF16": 1_682, "U32": 498])

    let textTower = fixture.tensors.filter {
        SwiftTransform.isSelectedTextTowerKey($0.name, family: .qwen35)
    }
    #expect(textTower.count == Qwen35CheckpointValidation.expectedTensorCount)
    #expect(textTower.count == 1_847)
    #expect(
        fixture.tensors.count - textTower.count
            == fixture.tensors.filter { $0.name.hasPrefix("vision_tower.") }.count
    )

    // Every name is unique and every shard index is in range.
    #expect(Set(fixture.tensors.map(\.name)).count == fixture.tensors.count)
    #expect(fixture.shards.count == 3)
    for record in fixture.tensors {
        #expect((1...fixture.shards.count).contains(record.shard))
        #expect(!record.shape.isEmpty)
        #expect(record.shape.allSatisfy { $0 > 0 })
    }

    var perShardCounts = [Int](repeating: 0, count: fixture.shards.count)
    for record in fixture.tensors {
        perShardCounts[record.shard - 1] += 1
    }
    for (offset, shard) in fixture.shards.enumerated() {
        #expect(shard.tensorCount == perShardCounts[offset], "\(shard.name)")
        #expect(shard.headerLength > 0)
        #expect(shard.headerSHA256.count == 64)
        #expect(
            shard.dtypeCounts.values.reduce(0, +) == shard.tensorCount,
            "\(shard.name)"
        )
    }

    let canonical = qwen36CanonicalInventoryData(fixture.tensors, shards: fixture.shards)
    let digest = qwen36SHA256(canonical)
    #expect(
        digest == fixture.summary.canonicalSHA256,
        Comment(rawValue: "actual canonical digest: \(digest)")
    )
    #expect(
        fixture.summary.canonicalSHA256
            == "f17f15bb2d498ab22478bea86b8e1a3e7fd7d939c65104338b04367cd11e3f54"
    )

    let byName = Dictionary(uniqueKeysWithValues: fixture.tensors.map { ($0.name, $0) })
    for representative in fixture.representative {
        #expect(byName[representative.name] == representative)
    }
}

/// The two checked-in canonical digests differ BY DESIGN: the `fixtures/`
/// record embeds the shard name, the `Tests/Fixtures/` record does not.
@Test
func qwen36HeaderInventoryContractPinsTheShardIndependentDigest() throws {
    let fixture = try qwen36InventoryFixture()
    let contract = try qwen36HeaderInventoryContract()

    #expect(contract.schemaVersion == 1)
    #expect(contract.source.repository == qwen36Repository)
    #expect(contract.source.revision == qwen36Revision)
    #expect(contract.canonicalRecordFormat == "UTF-8 name<TAB>dtype<TAB>comma-separated-shape<LF>, sorted by name")
    #expect(contract.tensorCount == fixture.tensors.count)
    #expect(contract.dtypeCounts == fixture.summary.dtypeCounts)

    let digest = qwen36SHA256(qwen36CanonicalHeaderInventoryData(fixture.tensors))
    #expect(
        digest == contract.canonicalInventorySHA256,
        Comment(rawValue: "actual header-inventory digest: \(digest)")
    )
    #expect(contract.canonicalInventorySHA256 != fixture.summary.canonicalSHA256)

    let byName = Dictionary(uniqueKeysWithValues: fixture.tensors.map { ($0.name, $0) })
    #expect(!contract.representativeTensors.isEmpty)
    for representative in contract.representativeTensors {
        let record = try #require(
            byName[representative.name],
            Comment(rawValue: representative.name)
        )
        #expect(record.dtype == representative.dtype)
        #expect(record.shape == representative.shape)
    }
}

@Test
func qwen36ConfigContractCarriesThePublicConfigAndItsPublishedDigest() throws {
    let data = try Data(contentsOf: qwen36ConfigContractURL)
    let contract = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    let source = try #require(contract["source"] as? [String: Any])
    #expect(source["repository"] as? String == qwen36Repository)
    #expect(source["revision"] as? String == qwen36Revision)
    #expect(source["config_sha256"] as? String == qwen36ConfigSHA256)

    // The contract's embedded config must be the same object the public
    // fixture carries; only the surrounding envelope differs.
    let embedded = try #require(contract["config"] as? [String: Any])
    let canonicalEmbedded = try JSONSerialization.data(
        withJSONObject: embedded,
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    let canonicalFixture = try JSONSerialization.data(
        withJSONObject: try qwen36ConfigObject(),
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    #expect(canonicalEmbedded == canonicalFixture)
}

/// The `config_sha256` both 3.6 fixtures pin is the digest of the checkpoint's
/// own `config.json` bytes; the checked-in reference manifest carries the same
/// kind of record for the pinned 3.8 target. This binds all three artifacts
/// that have to agree about a `config.json`.
///
/// UN-SUSPENDED 2026-08-20. This test was HALF-SUSPENDED on 2026-08-14 while
/// `fixtures/reference_qwen3_8_27b_4bit.sha256` was a header-only stub, and it
/// asserted the stub -- no `config.json` record, a `QWEN38-PENDING-RELEASE`
/// marker in the file -- so that the stub's disappearance would be loud. It
/// disappeared: the manifest's own header now reads "GENERATED FROM THE
/// PUBLISHED SNAPSHOT 2026-08-14", it carries ten generated records, and
/// AGENTS.md documents the target as "published and public" at the same
/// revision `MLXFastConstants.referenceModelRevision` compiles in. The stub
/// assertions were therefore asserting a state that no longer exists, which is
/// exactly what they were built to report.
///
/// The suspension note asked for two things on restoration. Both are settled:
///
///   * "re-scope the 3.6 config/inventory fixtures" -- already recorded, in
///     `Sources/MLXFastCore/Constants.swift`: "Tests/Fixtures/Qwen3627B4bit/
///     and fixtures/qwen3_6_27b_* keep their 3.6 names and 3.6 content
///     deliberately -- they describe the checkpoint they were captured from,
///     and the geometry they record is now known to describe both towers."
///     This file's own fixture-support header says the same. So the 3.6 leg
///     stays 3.6 by decision, and is pinned here as such.
///   * "a real three-way comparison against the 3.8 config.json digest" -- the
///     three legs cannot be an equality chain, because leg one describes 3.6
///     and legs two and three describe 3.8. What is asserted instead is what
///     is actually true and actually load-bearing: the 3.6 fixtures agree with
///     each other; the manifest names the compiled 3.8 pin and carries a real
///     `config.json` record for it; and the track contract's
///     `target.manifest_sha256` / `expected_source_bytes` agree with the
///     manifest's own bytes and record body, so a tampered manifest is caught
///     before any weight byte is read.
///
/// The 3.6 and 3.8 `config.json` digests are pinned as DIFFERENT below. That is
/// the deliberate divergence, not a drift: if they ever become equal, the 3.6
/// fixtures were re-captured and this test's 3.6 leg needs re-reading.
@Test(.disabled("Gemma 4 port: asserts the reference sha256 manifest header names the pinned repo/revision; fixtures/reference_qwen3_8_27b_4bit.sha256 is still the Qwen manifest. Unblocks with the box-generated Gemma manifest -- docs/gemma4-port-notes.md section 6.1."))
func qwen36ConfigContractDigestMatchesTheReferenceManifest() throws {
    // Leg 1 -- the two 3.6 capture fixtures still agree with each other.
    #expect(try qwen36InventoryFixture().source.configSHA256 == qwen36ConfigSHA256)

    let manifestData = try Data(contentsOf: qwen36ReferenceManifestURL)
    let manifest = try #require(String(data: manifestData, encoding: .utf8))
    let lines = manifest.split(separator: "\n").map(String.init)

    // Leg 2 -- the manifest names the target the binary compiles against, and
    // is no longer the pending-release stub.
    #expect(
        lines.contains(
            "# SHA256 manifest for \(MLXFastConstants.referenceModelRepository)."
        )
    )
    #expect(lines.contains("# Revision: \(MLXFastConstants.referenceModelRevision)"))
    #expect(
        !manifest.contains("QWEN38-PENDING-RELEASE"),
        """
        the reference manifest went back to a pending-release stub. The target \
        pin cannot be a placeholder while MLXFastConstants compiles a real \
        revision in.
        """
    )

    // ... and carries a real config.json record for it.
    let records = lines
        .filter { !$0.hasPrefix("#") && !$0.isEmpty }
        .map { $0.split(separator: " ", maxSplits: 2).map(String.init) }
    #expect(records.allSatisfy { $0.count == 3 })
    let configRecord = try #require(
        records.first { $0.count == 3 && $0[2] == "config.json" },
        """
        the reference manifest lost its config.json record. It had one at \
        \(qwen38ManifestConfigSHA256) / \(qwen38ManifestConfigBytes) bytes.
        """
    )
    #expect(configRecord[0] == qwen38ManifestConfigSHA256)
    #expect(Int(configRecord[1]) == qwen38ManifestConfigBytes)

    // The deliberate divergence, pinned rather than assumed.
    #expect(configRecord[0] != qwen36ConfigSHA256)

    // Leg 3 -- the manifest's OWN header pins its shape, and the track contract
    // pins the manifest. Read the record count and byte total from the header
    // pins the manifest declares about itself
    // (`MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS` / `_BYTES`) rather than
    // hardcoding them, so the coupling is to the manifest's declared shape, not
    // a decoupled external literal. Reading `_BYTES` here also gives that pin a
    // reader it did not have before.
    let recordsPin = try #require(
        manifestHeaderPin(lines, "MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS"),
        "the reference manifest lost its MANIFEST_RECORDS header pin"
    )
    let bytesPin = try #require(
        manifestHeaderPin(lines, "MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES"),
        "the reference manifest lost its MANIFEST_BYTES header pin"
    )
    let contract = try #require(
        try JSONSerialization.jsonObject(
            with: try Data(contentsOf: qwen38MTPTrackContractURL)
        ) as? [String: Any]
    )
    let target = try #require(contract["target"] as? [String: Any])
    let manifestDigest = SHA256.hash(data: manifestData)
        .map { String(format: "%02x", $0) }
        .joined()
    #expect(target["manifest_sha256"] as? String == manifestDigest)
    #expect(records.count == recordsPin)
    let totalBytes = records.compactMap { Int($0[1]) }.reduce(0, +)
    #expect(totalBytes == bytesPin)
    #expect(target["expected_source_bytes"] as? Int == totalBytes)
}

/// RECORD SCOPE, resolved 2026-08-22 by operator ruling: the backbone manifest
/// pins exactly the ten LOAD-BEARING records and deliberately does NOT pin the
/// published repository's generated `.gitattributes`. That file is never
/// engine-loaded; the per-shard digests are the real defense, and tree-equality
/// over generated platform metadata is purity, not protection. `verify_cache`
/// verifies the load-bearing set's pins and ignores non-loaded platform
/// metadata rather than requiring the cache tree to equal the manifest.
///
/// This pins the ruling in BOTH directions: every load-bearing record carries a
/// full `<sha256> <byte_count> <path>` pin, and no eleventh record is present
/// -- neither a `.gitattributes` record nor the eleven-record `15153238687`
/// byte total that adding one would force -- so nobody re-introduces
/// tree-equality by "resolving" the open item the other way.
@Test
func referenceManifestPinsLoadBearingSetAndNotNonLoadedPlatformMetadata() throws {
    let manifestData = try Data(contentsOf: qwen36ReferenceManifestURL)
    let manifest = try #require(String(data: manifestData, encoding: .utf8))
    let lines = manifest.split(separator: "\n").map(String.init)
    let records = lines
        .filter { !$0.hasPrefix("#") && !$0.isEmpty }
        .map { $0.split(separator: " ", maxSplits: 2).map(String.init) }

    // Every load-bearing record is a full <sha256> <byte_count> <path> pin.
    #expect(!records.isEmpty)
    for record in records {
        try #require(record.count == 3, Comment(rawValue: record.joined(separator: " ")))
        #expect(record[0].count == 64, Comment(rawValue: record[2]))
        #expect(Int(record[1]) != nil, Comment(rawValue: record[2]))
    }

    // The pinned set is exactly the load-bearing files, each named once, and
    // `.gitattributes` (non-loaded platform metadata) is NOT among them.
    let pinnedPaths = records.map { $0[2] }
    #expect(Set(pinnedPaths).count == pinnedPaths.count)
    #expect(!pinnedPaths.contains(".gitattributes"))

    // The count agrees with the manifest's own MANIFEST_RECORDS header pin,
    // and the byte total is the load-bearing sum the header declares -- never
    // the eleven-record total an added `.gitattributes` record would force.
    let recordsPin = try #require(
        manifestHeaderPin(lines, "MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS")
    )
    #expect(records.count == recordsPin)
    let totalBytes = records.compactMap { Int($0[1]) }.reduce(0, +)
    #expect(totalBytes == manifestHeaderPin(lines, "MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES"))
    #expect(totalBytes != 15_153_238_687)
}

/// The transform's hardcoded inventory is only worth having if it is the real
/// checkpoint's inventory. Compare it, tensor for tensor, against the public
/// header fixture.
@Test
func qwen35CheckpointValidationInventoryMatchesThePublicHeaders() throws {
    let fixture = try qwen36InventoryFixture()
    let expected = Qwen35CheckpointValidation.expectedTensorInventory()
    #expect(expected.count == 1_847)

    let actual = Dictionary(
        uniqueKeysWithValues: fixture.tensors
            .filter { SwiftTransform.isSelectedTextTowerKey($0.name, family: .qwen35) }
            .map { ($0.name, $0) }
    )
    #expect(Set(expected.keys) == Set(actual.keys))
    for (name, metadata) in expected.sorted(by: { $0.key < $1.key }) {
        let record = try #require(actual[name], Comment(rawValue: name))
        #expect(record.dtype == metadata.dtype, Comment(rawValue: name))
        #expect(record.shape == metadata.shape, Comment(rawValue: name))
    }

    // Per-layer counts. These used to be cross-checked against
    // `Qwen35WeightLoader`'s independently-pinned constants; that loader was
    // deleted with the Qwen model surface (2026-08-22), so the counts are
    // spelled here from this validator's own header comment ("7 top level plus
    // 48 linear-attention layers of 30 and 16 full-attention layers of 25",
    // 7 + 48*30 + 16*25 = 1,847). That is one fewer independent witness and it
    // is stated rather than hidden.
    //
    // The loop is also WIDER than it was: it used to run
    // `0..<MLXFastConstants.numHiddenLayers`, which is now the GEMMA layer count
    // (30) and silently skipped 34 of the Qwen tower's 64 layers.
    #expect(expected.keys.filter { !$0.contains(".layers.") }.count == 7)
    let interval = Qwen35CheckpointValidation.PinnedGeometry.fullAttentionInterval
    for layerIndex in 0..<Qwen35CheckpointValidation.PinnedGeometry.layerCount {
        let prefix = "language_model.model.layers.\(layerIndex)."
        let count = expected.keys.filter { $0.hasPrefix(prefix) }.count
        let wanted = layerIndex % interval == interval - 1 ? 25 : 30
        #expect(count == wanted, "layer \(layerIndex)")
    }
}

@Test
func qwen35TransformAcceptsTheExactPublicTextTower() throws {
    let fixture = try qwen36InventoryFixture()
    let metadata = qwen36ValidationMetadata(
        records: fixture.tensors,
        shards: fixture.shards
    )
    #expect(metadata.selectedKeys.count == 1_847)
    try Qwen35CheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: try exactQwen36Quantization()
    )
}

@Test
func qwen35TransformRejectsInventoryAndPackingDrift() throws {
    let fixture = try qwen36InventoryFixture()
    let quantization = try exactQwen36Quantization()

    func expectRejected(
        _ name: String,
        _ mutate: (inout [Qwen36TensorRecord]) -> Void
    ) throws {
        var records = fixture.tensors
        mutate(&records)
        let metadata = qwen36ValidationMetadata(records: records, shards: fixture.shards)
        #expect(throws: MLXFastError.self, "case \(name)") {
            try Qwen35CheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
        }
    }

    let template = try #require(
        fixture.tensors.first { $0.name == "language_model.model.norm.weight" }
    )
    func record(
        _ name: String,
        dtype: String? = nil,
        shape: [Int]? = nil
    ) -> Qwen36TensorRecord {
        var copy = template
        copy.name = name
        if let dtype { copy.dtype = dtype }
        if let shape { copy.shape = shape }
        return copy
    }

    try expectRejected("missing-tensor") { records in
        records.removeAll { $0.name == "language_model.model.norm.weight" }
    }
    try expectRejected("extra-text-tensor") { records in
        records.append(record("language_model.model.unexpected.weight"))
    }
    try expectRejected("wrong-shape") { records in
        for index in records.indices
        where records[index].name == "language_model.model.norm.weight" {
            records[index].shape = [MLXFastConstants.hiddenSize + 1]
        }
    }
    try expectRejected("wrong-dtype") { records in
        for index in records.indices
        where records[index].name == "language_model.model.layers.3.self_attn.q_proj.weight" {
            records[index].dtype = "BF16"
        }
    }
    try expectRejected("missing-affine-biases") { records in
        records.removeAll { $0.name == "language_model.lm_head.biases" }
    }
    try expectRejected("mtp-head-smuggled-in") { records in
        records.append(record("language_model.model.mtp.layers.0.norm.weight"))
    }
    try expectRejected("compressed-tensors-alias") { records in
        records.append(
            record("language_model.model.layers.3.self_attn.q_proj.weight_packed")
        )
    }
    try expectRejected("fp8-kv-scale") { records in
        records.append(record("language_model.model.layers.3.self_attn.k_scale"))
    }
    try expectRejected("repacked-at-a-different-group-size") { records in
        // group_size 32 would double the scale columns; the packed U32 width
        // no longer matches group_count * group_size * bits / 32.
        for index in records.indices
        where records[index].name.hasPrefix("language_model.lm_head.")
            && records[index].name.hasSuffix("s")
        {
            records[index].shape = [MLXFastConstants.vocabSize, 160]
        }
    }
}

@Test
func qwen35TransformRejectsAlternateQuantizationSpecs() throws {
    var root = try qwen36ConfigObject()

    for (name, block) in [
        ("bits", ["group_size": 64, "bits": 8, "mode": "affine"] as [String: Any]),
        ("group", ["group_size": 32, "bits": 4, "mode": "affine"]),
        ("mode", ["group_size": 64, "bits": 4, "mode": "nvfp4"]),
        ("extra-field", ["group_size": 64, "bits": 4, "mode": "affine", "skip_modules": []]),
        ("missing-mode", ["group_size": 64, "bits": 4]),
    ] {
        var mutated = root
        mutated["quantization"] = block
        mutated["quantization_config"] = block
        #expect(throws: MLXFastError.self, "case \(name)") {
            _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: mutated)
        }
    }

    // Present but disagreeing blocks are rejected rather than silently
    // preferring one of them.
    root["quantization_config"] = ["group_size": 32, "bits": 4, "mode": "affine"]
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: root)
    }

    var neither = try qwen36ConfigObject()
    neither.removeValue(forKey: "quantization")
    neither.removeValue(forKey: "quantization_config")
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35CheckpointValidation.quantizationSpec(fromConfigRoot: neither)
    }
}
