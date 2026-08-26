// CBv2PrefillLogitDigestTests.swift — the prefill logit digest as EVIDENCE.
//
// `CBv2Engine.prefillLogitDigest(_:)` answers the question a token stream
// cannot: did two arms compute the SAME final-position prefill logits, or
// did they merely happen to sample the same id out of two different vectors?
// `submit` yields tokens, text and top-k logprobs; a top-k slice cannot
// reconstruct a full vector, and hashing a vector obtained by calling the
// model directly would measure the MODEL — the one thing two backends share
// — instead of the engine.
//
// So the digest is taken at the engine's own prompt frontier, BEFORE the
// sampler, and this suite pins the properties a parity harness has to trust:
//
//  1. testTwoEnginesOverTheSameWeightsAgreeAndAPerturbedOneDoesNot — the
//     discrimination claim in both directions. Two independently constructed
//     engines over the SAME weights agree on every field; one weight nudged
//     by 1e-3 breaks the hash while `count` and `maxAbs` stay comparable,
//     which is what lets a harness SIZE a mismatch instead of flagging it.
//  2. testFailClosedDefaultThrowsInsteadOfFabricatingADigest /
//     testEmptyPromptIsRefusedRatherThanDigested — refusal, not fabrication.
//     A digest of nothing compares EQUAL across two arms and reads as parity.
//  3. testDigestIsTakenInTheModelDType — a bf16 model reports "bfloat16"
//     over 2 bytes per element and hashes the bf16 bytes; hashing an upcast
//     copy would be a different digest, and the suite pins that it is not
//     the one produced.
//  4. testDigestIsTakenOnTheEnginePrefillPath /
//     testReportedDTypeFollowsTheFrontierOutputNotTheWeights — the digest
//     rides the engine's real chunked prefill, and the dtype it reports is
//     the RESOLVED width it actually hashed, not the weights' width.
//
// Tiny seeded-random model (vocab 128, 2 layers: full + sliding(16)). No
// checkpoints, no downloads.

import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

// MARK: - Fixtures

/// Passes every prompt forward through to `CBv2PrefillNarrowingModel` while
/// recording the requirement and the OUTPUT dtype/shape as the model handed
/// them to the engine. The digest's `dtype`/`count` are cross-checked
/// against this recording: a seam that recomputed the vector anywhere else
/// would still produce a plausible-looking digest, but the recorder would
/// disagree with it — or stay empty.
final class CBv2DigestRecordingModel: CBv2PrefillSteppableModel, @unchecked Sendable {
    struct Output {
        let requirement: CBv2PrefillRequirement
        let dtype: DType
        let shape: [Int]
    }

    private let inner: any CBv2PrefillSteppableModel
    private let lock = NSLock()
    private var _outputs: [Output] = []
    /// Optional dtype coercion of the FRONTIER output only, so the logits the
    /// engine sees can be driven to a chosen width without touching what the
    /// KV backend stores.
    private let coerceFrontierTo: DType?

    init(_ inner: any CBv2PrefillSteppableModel, coerceFrontierTo: DType? = nil) {
        self.inner = inner
        self.coerceFrontierTo = coerceFrontierTo
    }

    var outputs: [Output] {
        lock.lock()
        defer { lock.unlock() }
        return _outputs
    }

    var frontierOutputs: [Output] { outputs.filter { $0.requirement == .lastPositionLogits } }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        inner.forward(tokens: tokens, caches: caches)
    }

    func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        var output = inner.prefill(
            tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
            requirement: requirement)
        if let coerceFrontierTo, requirement == .lastPositionLogits {
            output = output.asType(coerceFrontierTo)
        }
        lock.lock()
        _outputs.append(
            Output(requirement: requirement, dtype: output.dtype, shape: output.shape))
        lock.unlock()
        return output
    }
}

/// Minimal `CBv2Engine` conformer with NO digest witness: it exercises the
/// protocol's fail-closed default. Everything else is inert on purpose —
/// this type exists to be asked exactly one question.
final class CBv2NoDigestEngine: CBv2Engine, @unchecked Sendable {
    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        AsyncStream { $0.finish() }
    }
    func cancel(_ id: CBv2RequestID) {}
    func capacity() -> CBv2CapacitySnapshot {
        CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0, kvBytesCapacity: 0,
            activeTokens: 0)
    }
    func shutdown() async {}
}

// MARK: - Suite

final class CBv2PrefillLogitDigestTests: XCTestCase {

    /// 24 tokens: three chunks at `prefillChunkSize: 8`, one at 64.
    private let prompt = makePromptTokens(length: 24, seed: 0x1D16E57)

    // MARK: 1. Discrimination

    func testTwoEnginesOverTheSameWeightsAgreeAndAPerturbedOneDoesNot() async throws {
        let baseline = TinyTestModel.make(seed: 0xD16E57)
        let candidate = TinyTestModel.make(seed: 0xD16E57)

        let engineA = makeEngine(model: baseline, layerKinds: baseline.layerKinds)
        let engineB = makeEngine(model: candidate, layerKinds: candidate.layerKinds)

        let digestA = try engineA.prefillLogitDigest(prompt)
        let digestB = try engineB.prefillLogitDigest(prompt)

        XCTAssertEqual(
            digestA, digestB,
            "two engines over the same weights must agree on every field of the digest")
        XCTAssertEqual(digestA.count, baseline.config.vocabSize)
        XCTAssertEqual(digestA.sha256.count, 64, "sha256 is 32 bytes of lowercase hex")
        XCTAssertTrue(
            digestA.sha256.allSatisfy { $0.isHexDigit && !$0.isUppercase },
            "digest hex must be lowercase")
        XCTAssertGreaterThan(digestA.maxAbs, 0, "a real logit vector is not identically zero")

        // Re-asking the SAME engine must not move the answer: the digest is a
        // property of (weights, prompt, backend), not of call order.
        XCTAssertEqual(
            try engineA.prefillLogitDigest(prompt), digestA,
            "the digest must be stable across repeated calls on one engine")

        // One output-head weight nudged by 1e-3 — the smallest change that is
        // unambiguously a change.
        let perturbed = TinyTestModel.make(seed: 0xD16E57)
        perturbed.lmHead.weight[0, 0] =
            perturbed.lmHead.weight[0, 0] + MLXArray(Float(1e-3))
        eval(perturbed)
        let enginePerturbed = makeEngine(model: perturbed, layerKinds: perturbed.layerKinds)
        let digestPerturbed = try enginePerturbed.prefillLogitDigest(prompt)

        XCTAssertNotEqual(
            digestPerturbed.sha256, digestA.sha256,
            "one perturbed weight must break the digest — otherwise it cannot discriminate")
        // ... and the mismatch is SIZEABLE, not merely flagged.
        XCTAssertEqual(
            digestPerturbed.count, digestA.count,
            "same vocabulary: the mismatch is numerical, not a shape defect")
        XCTAssertEqual(digestPerturbed.dtype, digestA.dtype)
        XCTAssertEqual(
            digestPerturbed.maxAbs, digestA.maxAbs, accuracy: 0.05 * digestA.maxAbs,
            "a 1e-3 weight nudge is drift, not a scale defect — maxAbs stays comparable")

        await engineA.shutdown()
        await engineB.shutdown()
        await enginePerturbed.shutdown()
    }

    // MARK: 2. Refusal, not fabrication

    func testFailClosedDefaultThrowsInsteadOfFabricatingADigest() throws {
        let engine: any CBv2Engine = CBv2NoDigestEngine()
        do {
            let fabricated = try engine.prefillLogitDigest(prompt)
            XCTFail("an engine with no digest path must throw, not return \(fabricated)")
        } catch let error as CBv2PrefillLogitDigestError {
            XCTAssertEqual(error, .unsupported(engine: "CBv2NoDigestEngine"))
        }
    }

    func testEmptyPromptIsRefusedRatherThanDigested() async throws {
        let model = TinyTestModel.make(seed: 0xE09B71)
        let engine = makeEngine(model: model, layerKinds: model.layerKinds)
        XCTAssertThrowsError(try engine.prefillLogitDigest([])) { error in
            XCTAssertEqual(error as? CBv2PrefillLogitDigestError, .emptyPrompt)
        }
        await engine.shutdown()
    }

    // MARK: 3. Model dtype

    /// A bf16 model must report `"bfloat16"` over 2 bytes per element, and an
    /// fp32 model `"float32"` over 4 — never one silently reported as the
    /// other. The width claim is checked against an INDEPENDENT SHA-256 of
    /// the same vector, so "taken in the model dtype" is a statement about
    /// the hashed BYTES, not about a label the seam attached to them.
    ///
    /// Chunk size 64 > prompt 24, so the engine's frontier forward is the
    /// same single `[1, 24]` pass the cross-check runs.
    func testDigestIsTakenInTheModelDType() async throws {
        let fp32Model = TinyTestModel.make(seed: 0xBF1600)
        let bf16Model = TinyTestModel.make(seed: 0xBF1600)
        castParameters(of: bf16Model, to: .bfloat16)

        let fp32Engine = makeEngine(
            model: fp32Model, layerKinds: fp32Model.layerKinds, prefillChunkSize: 64)
        let bf16Engine = makeEngine(
            model: bf16Model, layerKinds: bf16Model.layerKinds, prefillChunkSize: 64)

        let fp32 = try fp32Engine.prefillLogitDigest(prompt)
        let bf16 = try bf16Engine.prefillLogitDigest(prompt)

        XCTAssertEqual(fp32.dtype, "float32", "an fp32 model's digest is an fp32 digest")
        XCTAssertEqual(
            bf16.dtype, "bfloat16", "a bf16 model must NOT silently report an fp32 digest")
        XCTAssertEqual(fp32.count, fp32Model.config.vocabSize)
        XCTAssertEqual(bf16.count, bf16Model.config.vocabSize)
        XCTAssertNotEqual(
            bf16.sha256, fp32.sha256,
            "different byte widths over different values cannot share a hash")

        // The digested width is the MODEL's width. Same vector, hashed at
        // both widths: the seam produced the 2-byte one.
        let bf16Vector = try frontierVectorOutsideTheEngine(of: bf16Model)
        XCTAssertEqual(bf16Vector.dtype, .bfloat16, "fixture check: the model really is bf16")
        XCTAssertEqual(bf16Vector.nbytes, 2 * bf16.count, "bfloat16 is 2 bytes per element")
        XCTAssertEqual(
            bf16.sha256, sha256Hex(of: bf16Vector),
            "the digest must hash the bf16 bytes the model produced")
        XCTAssertNotEqual(
            bf16.sha256, sha256Hex(of: bf16Vector.asType(.float32)),
            "hashing an upcast copy would be a different — and wrong — digest")

        await fp32Engine.shutdown()
        await bf16Engine.shutdown()
    }

    // MARK: 4. Taken on the engine's own prefill path

    /// The digest call drives a real `.lastPositionLogits` prompt forward on
    /// the engine and digests exactly what that forward returned. The
    /// engine's chunking is visible in the recorder: a 24-token prompt at
    /// chunk 8 is three chunks, only the last of which samples.
    func testDigestIsTakenOnTheEnginePrefillPath() async throws {
        let base = TinyTestModel.make(seed: 0x9EA111)
        let recorder = CBv2DigestRecordingModel(CBv2PrefillNarrowingModel(base))
        let engine = makeEngine(model: recorder, layerKinds: base.layerKinds)

        XCTAssertTrue(recorder.outputs.isEmpty, "no forward has run yet")
        let digest = try engine.prefillLogitDigest(prompt)

        XCTAssertEqual(
            recorder.outputs.map(\.requirement),
            [.evaluationOnly, .evaluationOnly, .lastPositionLogits],
            "the digest rode the engine's own chunked prefill, frontier chunk last")
        let frontier = try XCTUnwrap(recorder.frontierOutputs.last)
        XCTAssertEqual(
            frontier.shape, [1, base.config.vocabSize],
            "the frontier forward returned [B, vocab] — the vector that was digested")
        XCTAssertEqual(digest.count, frontier.shape[1])
        XCTAssertEqual(digest.dtype, EngineV2.dtypeName(frontier.dtype))

        await engine.shutdown()
    }

    /// The reported dtype tracks what the ENGINE actually hashed, not what
    /// the weights are: coerce only the frontier output to fp16 and the
    /// digest follows it. Resolved, never requested.
    func testReportedDTypeFollowsTheFrontierOutputNotTheWeights() async throws {
        let base = TinyTestModel.make(seed: 0x9EA111)
        let recorder = CBv2DigestRecordingModel(
            CBv2PrefillNarrowingModel(base), coerceFrontierTo: .float16)
        let engine = makeEngine(model: recorder, layerKinds: base.layerKinds)

        let digest = try engine.prefillLogitDigest(prompt)
        let frontier = try XCTUnwrap(recorder.frontierOutputs.last)
        XCTAssertEqual(frontier.dtype, .float16, "fixture check: the frontier really is fp16")
        XCTAssertEqual(
            digest.dtype, "float16",
            "fp32 weights but an fp16 frontier ⇒ the digest reports fp16, the width it hashed")
        XCTAssertEqual(digest.count, base.config.vocabSize)

        await engine.shutdown()
    }

    // MARK: - Helpers

    private func makeEngine(
        model: CBv2SteppableModel, layerKinds: [CBv2LayerKind], prefillChunkSize: Int = 8
    ) -> EngineV2 {
        EngineV2(
            model: model,
            layerKinds: layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxBatchedTokensPerStep: 256, prefillChunkSize: prefillChunkSize,
                maxWaiting: 4))
    }

    /// Cast every parameter of `model` in place.
    private func castParameters(of model: Module, to dtype: DType) {
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        eval(model)
    }

    /// The final-position logit vector computed OUTSIDE the engine, used for
    /// byte-level cross-checks only — never as the digest's source. That
    /// would make this suite the second prefill the seam refuses to be.
    private func frontierVectorOutsideTheEngine(of model: TinyTestModel) throws -> MLXArray {
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26))
        let rowState = try backend.makeSequenceState(
            layerKinds: model.layerKinds, promptLength: prompt.count,
            maxLength: prompt.count + 1)
        let caches = CBv2LayerCacheBank(layerKinds: model.layerKinds)
            .layerCaches(rowStates: [rowState])
        let tokens = MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count])
        let logits = model.forward(tokens: tokens, caches: caches)
        return logits[0, -1, 0...]
    }

    /// Deliberately NOT `EngineV2.digest` — an independent hash of the same
    /// bytes, so the cross-check cannot agree with the seam by construction.
    private func sha256Hex(of vector: MLXArray) -> String {
        var hasher = SHA256()
        hasher.update(data: vector.asData(access: .copy).data)
        return hasher.finalize().reduce(into: "") { $0 += String(format: "%02x", $1) }
    }
}
