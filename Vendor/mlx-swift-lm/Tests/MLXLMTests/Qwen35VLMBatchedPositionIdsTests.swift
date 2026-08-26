// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Regression test for d-inference issue #513: under continuous batching the
/// provider crashed with
///
///     [broadcast_shapes] Shapes (3,1) and (6,1) cannot be broadcast
///
/// serving mlx-community/Qwen3.5-0.8B-MLX-4bit (a `vision_config`-bearing
/// checkpoint, so the provider loads MLXVLM's `Qwen35`).
///
/// Root cause: `Qwen35Language.LanguageModel` keeps `ropeDeltas` as
/// module-level state, written by the most recent prefill (shape
/// `[prefillBatch]`). The continuous-batching scheduler interleaves cold
/// prefill of newly-admitted requests between the live batch's decode steps,
/// so at decode time `ropeDeltas.dim(0)` can differ from the decoding batch
/// size. The decode-path "adjustment" for that mismatch used
/// `repeated(delta, count: batchSize, axis: 0)`, which repeats every element
/// `batchSize` times — producing `prefillBatch * batchSize` rows — and then
/// `base + delta[0..., .newAxis]` trapped with
/// `(batchSize,1)` vs `(prefillBatch*batchSize,1)`.
///
/// This test distills the scheduler interleaving to three raw model calls:
/// prefill batch A (B=3), prefill batch B (B=2, fresh caches — overwrites
/// `ropeDeltas` with a 2-row array), then a decode step of batch A (B=3,
/// caches at offset > 0). Without the fix the third call fatal-errors inside
/// MLX (`[broadcast_shapes] (3,1) vs (6,1)`), killing the test process; with
/// the fix the stale deltas are ignored (they belong to batch B's rows; text
/// rows always have delta 0) and the decode step returns [3, 1, vocab] logits.
final class Qwen35VLMBatchedPositionIdsTests: XCTestCase {

    /// Minimal 2-layer config: layer 0 GDN (linear attention), layer 1 full
    /// attention (`full_attention_interval` 2), so `faIdx == 1` and the
    /// language model's position-id path sees a real attention cache.
    /// GDN dims mirror Qwen35VLMGatedDeltaTests (Hk=2, Dk=32, Hv=4, Dv=16)
    /// so the gated-delta kernel dispatches with sane thread counts.
    private func makeConfig() throws -> Qwen35Configuration {
        let json = """
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 32,
                    "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 64,
                    "full_attention_interval": 2,
                    "num_experts": 0,
                    "num_experts_per_tok": 0
                },
                "vision_config": {
                    "model_type": "qwen3_5",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": 16,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 8
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
    }

    private func tokens(batch: Int, length: Int, vocab: Int) -> MLXArray {
        MLXArray((0 ..< (batch * length)).map { Int32($0 % vocab) }).reshaped(batch, length)
    }

    func testDecodeSurvivesStaleRopeDeltasFromInterleavedPrefill() throws {
        MLXRandom.seed(0)
        let config = try makeConfig()
        let model = Qwen35(config)
        let vocab = config.textConfiguration.vocabularySize

        // 1. Prefill batch A (B=3) on fresh caches — the recompute branch
        //    runs (fa cache offset == 0) and stores ropeDeltas of shape [3].
        let cachesA = model.newCache(parameters: nil)
        let prefillA = model(tokens(batch: 3, length: 4, vocab: vocab), cache: cachesA)
        eval(prefillA)
        XCTAssertEqual(prefillA.shape, [3, 4, vocab])

        // 2. Prefill batch B (B=2) on separate fresh caches — exactly what the
        //    continuous-batching scheduler does when new requests are admitted
        //    while batch A is mid-decode. This overwrites the module-level
        //    ropeDeltas with a [2]-shaped array.
        let cachesB = model.newCache(parameters: nil)
        let prefillB = model(tokens(batch: 2, length: 4, vocab: vocab), cache: cachesB)
        eval(prefillB)
        XCTAssertEqual(prefillB.shape, [2, 4, vocab])

        // 3. Decode step of batch A (B=3, fa cache offset 4 > 0). ropeDeltas
        //    is now the stale [2] from batch B. Pre-fix this fatal-errored:
        //    repeated([2 rows], count: 3) -> [6] rows, then
        //    base(3,1) + delta(6,1) -> [broadcast_shapes] (3,1) vs (6,1).
        let decodeA = model(tokens(batch: 3, length: 1, vocab: vocab), cache: cachesA)
        eval(decodeA)
        XCTAssertEqual(
            decodeA.shape, [3, 1, vocab],
            "decode step after an interleaved smaller-batch prefill must not "
                + "apply the other batch's ropeDeltas")

        // Also cover the grown-batch continuation at matching offsets and the
        // stale-LARGER-deltas direction (pre-fix the `>` branch silently
        // applied the first rows of another batch's deltas; post-fix the
        // mismatched array is ignored entirely).
        let decodeB = model(tokens(batch: 2, length: 1, vocab: vocab), cache: cachesB)
        eval(decodeB)
        XCTAssertEqual(decodeB.shape, [2, 1, vocab])
    }
}
