import CryptoKit
import Foundation
import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastTransform

/// The identity of the checkpoint `fixtures/gemma4_26b_a4b_config.json` was
/// captured from. This IS the current `MLXFastConstants.referenceModelRepository`
/// / `_Revision` pin (`Sources/MLXFastCore/Constants.swift`,
/// `docs/gemma4-port-notes.md` section 1) -- unlike the retained Qwen 3.6
/// fixtures, which record a superseded backbone on purpose, this fixture was
/// captured specifically to validate the CURRENT target.
let gemma4A4BRepository = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
let gemma4A4BRevision = "0e3cbab38ce568cf6e23543010d08d03b731910c"

/// SHA256 of the checkpoint's own `config.json` bytes exactly as published at
/// the pinned revision (fetched 2026-08-23 from
/// `https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit/raw/0e3cbab38ce568cf6e23543010d08d03b731910c/config.json`,
/// revision confirmed via the HF API to resolve to this exact commit, not a
/// branch alias) -- NOT of the normalized `fixtures/gemma4_26b_a4b_config.json`
/// re-render checked into this repository, which differs only in whitespace
/// and key order (`JSONSerialization`-equivalent re-encoding: 2-space indent,
/// keys sorted, trailing newline -- reproduced with
/// `json.dumps(d, indent=2, sort_keys=True) + "\n"`, verified byte-identical
/// to this repository's existing `fixtures/qwen3_6_27b_config.json` round trip
/// before being applied here).
///
/// THIS IS A LAPTOP-SIDE PUBLIC FETCH, NOT A BOX-VERIFIED ARTIFACT. It proves
/// the fixture matches what the pinned HF revision publishes today; it does
/// NOT prove it matches the byte the ranked box's transform actually reads
/// (docs/gemma4-port-notes.md section 6.1 still calls that box-only). Do not
/// promote this digest to a manifest pin without a box-side re-verification.
let gemma4A4BConfigSHA256 =
    "29910322dd085f45c8f95c6c0f1611b20f722d6f6c8394321b34817e98a972fa"

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let gemma4A4BArtifactRepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let gemma4A4BConfigFixtureURL = gemma4A4BArtifactRepositoryRoot
    .appendingPathComponent("fixtures/gemma4_26b_a4b_config.json")

/// The track contract fixture (lane/gemma4-track-fixture). Mirrors how
/// `Qwen35ArtifactFixtureSupport.swift`'s `qwen38MTPTrackContractURL` locates
/// `fixtures/qwen3_8_27b_mtp_track.json`.
let gemma4A4BTrackContractURL = gemma4A4BArtifactRepositoryRoot
    .appendingPathComponent("fixtures/gemma4_26b_a4b_track.json")

/// The assistant (spec-decode) head pin the track contract fixture carries
/// under `assistant.{upstream_model_id,upstream_revision}` -- RULED
/// 2026-08-22 (docs/gemma4-port-notes.md section 4.0). Kept here, alongside
/// the target pin above, so a test can assert the contract and the compiled
/// constants never drift apart without re-parsing the fixture by hand.
let gemma4A4BAssistantRepository = "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit"
let gemma4A4BAssistantRevision = "bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c"

func gemma4A4BConfigData() throws -> Data {
    try Data(contentsOf: gemma4A4BConfigFixtureURL)
}

func gemma4A4BTrackContractData() throws -> Data {
    try Data(contentsOf: gemma4A4BTrackContractURL)
}

func gemma4A4BTrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: gemma4A4BTrackContractData()
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Gemma 4 26B A4B track contract fixture must be a JSON object")
    }
    return object
}

func gemma4A4BConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: gemma4A4BConfigData()
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Gemma 4 26B A4B config fixture must be a JSON object")
    }
    return object
}
