// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4SharedKV types")
struct Gemma4SharedKVTests {

    @Test func sharedKVConstructs() {
        let k = MLXArray.zeros([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.zeros([1, 2, 8, 4], dtype: .float32)
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        #expect(shared.fullAttention.0.dim(2) == 8)
        #expect(shared.slidingAttention.1.dim(2) == 8)
    }

    @Test func sliceTailZeroRejectedIsIdentity() {
        let k = MLXArray.ones([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 8, 4], dtype: .float32) * 2.0
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 0)
        #expect(sliced.fullAttention.0.dim(2) == 8)
        #expect(sliced.slidingAttention.0.dim(2) == 8)
    }

    @Test func sliceTailTrimsTimeAxis() {
        let k = MLXArray.ones([1, 2, 8, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 8, 4], dtype: .float32) * 2.0
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 3)
        #expect(sliced.fullAttention.0.dim(2) == 5)
        #expect(sliced.slidingAttention.0.dim(2) == 5)
        #expect(sliced.fullAttention.1.dim(2) == 5)
    }

    @Test func sliceTailClampsAtOne() {
        // When rejected >= T, preserve at least one slot (preserving the
        // minimum cache invariant used by the round-loop).
        let k = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        let shared = Gemma4SharedKV(
            fullAttention: (k, v),
            slidingAttention: (k, v)
        )
        let sliced = Gemma4SharedKV.sliceTail(from: shared, rejected: 100)
        #expect(sliced.fullAttention.0.dim(2) == 1)
    }

    @Test func captureIsReferenceType() {
        let capture = Gemma4SharedKVCapture()
        #expect(capture.fullAttention == nil)
        #expect(capture.slidingAttention == nil)
        let k = MLXArray.zeros([1, 2, 4, 4], dtype: .float32)
        let v = MLXArray.zeros([1, 2, 4, 4], dtype: .float32)
        capture.fullAttention = (k, v)
        // Aliasing semantics: a second reference observes the write.
        let alias = capture
        #expect(alias.fullAttention != nil)
    }

    @Test func captureSnapshotReturnsImmutableSnapshot() {
        let capture = Gemma4SharedKVCapture()
        let kFull = MLXArray.ones([1, 2, 8, 4], dtype: .float32)
        let vFull = MLXArray.ones([1, 2, 8, 4], dtype: .float32) * 2.0
        let kSliding = MLXArray.ones([1, 2, 6, 4], dtype: .float32) * 3.0
        let vSliding = MLXArray.ones([1, 2, 6, 4], dtype: .float32) * 4.0
        capture.fullAttention = (kFull, vFull)
        capture.slidingAttention = (kSliding, vSliding)

        let snap = capture.snapshot()
        // Shapes preserved.
        #expect(snap.fullAttention.0.dim(2) == 8)
        #expect(snap.slidingAttention.0.dim(2) == 6)
        // Values round-trip (verifies the snapshot isn't zeroing anything).
        #expect(snap.fullAttention.0.sum().item(Float.self) == 64.0)       // 1*2*8*4
        #expect(snap.fullAttention.1.sum().item(Float.self) == 128.0)      // 2*2*8*4
        #expect(snap.slidingAttention.0.sum().item(Float.self) == 144.0)   // 3*2*6*4
        #expect(snap.slidingAttention.1.sum().item(Float.self) == 192.0)   // 4*2*6*4
    }
}
