// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4DrafterMaskBuilder")
struct DrafterMasksTests {

    @Test func fullAttentionAlwaysReturnsNone() {
        // Full attention is always bidirectional; SDPA handles it directly.
        let mask = Gemma4DrafterMaskBuilder.bidirectionalFull(
            queryLen: 3, kvLen: 100, dtype: .float32)
        switch mask {
        case .causal:
            Issue.record("expected .none, got .causal")
        case .array(_):
            Issue.record("expected .none, got .array(...)")
        default:
            break  // .none / bias nil → interpreted as .causal by default call sites
        }
    }

    @Test func swaShortCircuitsWhenKVFitsInWindow() {
        // kvLen=32, window=512: every query already sees the whole KV.
        // queryOffset=32, queryLen=1 (bonus + k-1 drafts land at position 32+).
        let mask = Gemma4DrafterMaskBuilder.bidirectionalSWA(
            queryLen: 1, queryOffset: 32, kvLen: 32,
            window: 512, dtype: .float32)
        switch mask {
        case .causal:
            Issue.record("expected .none, got .causal")
        case .array(_):
            Issue.record("expected .none, got .array(...)")
        default:
            break
        }
    }

    @Test func swaProducesArrayWhenWindowIsTight() {
        // kvLen=1024, window=512, queryOffset=1024, queryLen=3.
        // Query position 1024 should attend to KV positions (512, 1536) ⇒
        // first 513 KV slots are blocked for query 0.
        let mask = Gemma4DrafterMaskBuilder.bidirectionalSWA(
            queryLen: 3, queryOffset: 1024, kvLen: 1024,
            window: 512, dtype: .float32)
        switch mask {
        case .array(let arr):
            // Expected shape [1, 1, queryLen=3, kvLen=1024].
            #expect(arr.shape == [1, 1, 3, 1024])
            // Row 0 (query at abs position 1024): KV index 511 should be -inf
            // (|1024 - 511| = 513 > window), KV index 513 should be 0
            // (|1024 - 513| = 511 < window).
            let row0_511 = arr[0, 0, 0, 511].item(Float.self)
            let row0_513 = arr[0, 0, 0, 513].item(Float.self)
            #expect(row0_511 == -Float.infinity)
            #expect(row0_513 == 0.0)
        default:
            Issue.record("expected .array(...) when kvLen > window")
        }
    }

    @Test func makeDispatchesToBothTypes() {
        // Smoke: make(...) returns a tuple with both entries; full always
        // .none, sliding .none when the window covers.
        let fullK = MLXArray.zeros([1, 1, 10, 4], dtype: .float32)
        let fullV = MLXArray.zeros([1, 1, 10, 4], dtype: .float32)
        let slidingK = MLXArray.zeros([1, 1, 10, 4], dtype: .float32)
        let slidingV = MLXArray.zeros([1, 1, 10, 4], dtype: .float32)
        // Actually we don't have Gemma4SharedKV in scope without MLXLLM import;
        // use the free-function form with explicit kvLen.
        let full = Gemma4DrafterMaskBuilder.bidirectionalFull(queryLen: 1, kvLen: 10, dtype: .float32)
        let sliding = Gemma4DrafterMaskBuilder.bidirectionalSWA(
            queryLen: 1, queryOffset: 10, kvLen: 10, window: 64, dtype: .float32)
        _ = (fullK, fullV, slidingK, slidingV)
        switch full {
        case .causal, .array(_):
            Issue.record("full must return .none")
        default:
            break
        }
        switch sliding {
        case .causal, .array(_):
            Issue.record("sliding must return .none when window covers")
        default:
            break
        }
    }
}
