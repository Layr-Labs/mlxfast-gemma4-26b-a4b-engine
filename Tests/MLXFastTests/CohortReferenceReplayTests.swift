import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXRandom
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Coverage for the COHORT REFERENCE-REPLAY ORACLE (fidelity-gate PR-1;
// Gemma4RuntimeCohortReferenceReplay.swift). MEASUREMENT MODE ONLY — nothing here
// asserts an admit/reject verdict, because the oracle renders none.
//
// The load-bearing property is the ANTI-STATIC-TAPE anchor: the reference's row
// N+1 must be teacher-forced on the CANDIDATE's committed token N, never on the
// reference's own argmax. A static tape (anchored to the reference's own chain)
// gets this wrong after the first divergence, which is exactly why an exact-tape
// gate rejects honest MTP candidates. The GPU-free `ScriptedReferenceForward`
// makes that property falsifiable without a model; the gated model-backed suite
// re-proves it on a real (synthetic) Gemma4 forward.

// MARK: - GPU-free seam: a scripted width-1 reference forward

/// A deterministic width-1 reference forward whose logits depend ONLY on the
/// last token it conditioned on, and which records every token fed via `step`.
/// The recorded `fedTokens` is the replay's teacher-forcing trace — the direct
/// evidence of WHAT the replay conditioned on.
private struct ScriptedReferenceForward: CohortReferenceReplayForward {
    let vocabSize: Int
    /// Tokens fed via `step` (NOT the seed), in order.
    private(set) var fedTokens: [Int] = []

    init(vocabSize: Int) { self.vocabSize = vocabSize }

    mutating func prefill(seed: [Int]) -> [Float] {
        logits(after: seed.last ?? 0)
    }

    mutating func step(feeding token: Int) -> [Float] {
        fedTokens.append(token)
        return logits(after: token)
    }

    /// argmax after conditioning on `t` is `(t + 1) % V`; a strictly-decreasing
    /// ring ramp away from it, so the ranking and every gap are deterministic.
    private func logits(after t: Int) -> [Float] {
        let peak = ((t % vocabSize) + vocabSize + 1) % vocabSize
        return (0 ..< vocabSize).map { v in
            let raw = abs(v - peak)
            let ringDistance = Swift.min(raw, vocabSize - raw)
            return Float(-ringDistance)
        }
    }
}

/// The token `(t + 1) % V` the scripted forward peaks at after conditioning on t.
private func scriptedPeak(after t: Int, vocabSize: Int) -> Int {
    ((t % vocabSize) + vocabSize + 1) % vocabSize
}

/// GPU-free BATCHED (cohort-width) reference forward: B rows, each whose logits
/// depend ONLY on the last token that row conditioned on (the same ring ramp the
/// width-1 `ScriptedReferenceForward` uses), and which records every token fed
/// PER STREAM. Because each row's logits depend only on its own last token, the
/// scripted mock is deliberately WIDTH-INVARIANT — a stream's readout is
/// identical whether replayed batched or width-1 — so these tests isolate the
/// DRIVER WIRING (anchor, per-row independence, slot order) from the real
/// model's batch-geometry numerics, which only the model-backed suite exercises.
private struct ScriptedBatchedReferenceForward: CohortReferenceReplayBatchedForward {
    let vocabSize: Int
    /// Tokens fed via `step`, per stream, in order (NOT the seed).
    private(set) var fedByStream: [[Int]]

    init(vocabSize: Int, batchSize: Int) {
        self.vocabSize = vocabSize
        self.fedByStream = Array(repeating: [], count: batchSize)
    }

    mutating func prefill(seedsByStream: [[Int]]) -> [[Float]] {
        seedsByStream.map { logits(after: $0.last ?? 0) }
    }

    mutating func step(feedingByStream tokens: [Int]) -> [[Float]] {
        for (slot, token) in tokens.enumerated() {
            fedByStream[slot].append(token)
        }
        return tokens.map { logits(after: $0) }
    }

    private func logits(after t: Int) -> [Float] {
        let peak = ((t % vocabSize) + vocabSize + 1) % vocabSize
        return (0 ..< vocabSize).map { v in
            let raw = abs(v - peak)
            let ringDistance = Swift.min(raw, vocabSize - raw)
            return Float(-ringDistance)
        }
    }
}

@Suite("CohortReferenceReplay")
struct CohortReferenceReplayTests {

    // MARK: - The anti-static-tape property (failing-test-first)

    @Test
    func replayFollowsTheCandidatePrefixNotTheReferenceOwnChain() throws {
        // A scripted candidate journal that DIVERGES from the reference's own
        // greedy chain at every row: with the scripted forward, the reference
        // would emit (lastConditioned + 1); the candidate committed something
        // else. The oracle must teacher-force on the CANDIDATE's tokens.
        let vocabSize = 16
        let seed = [0]
        let committed = [3, 5, 2]
        var forward = ScriptedReferenceForward(vocabSize: vocabSize)
        let positions = try replayCohortReferenceStream(
            seed: seed, committed: committed, forward: &forward,
            logitTopK: 4, relEnvelope: 0.05)

        #expect(positions.count == committed.count)
        // The teacher-forcing trace: the replay stepped on the candidate's own
        // committed tokens (all but the last, which is never stepped past), and
        // NEVER on the reference's own argmaxes.
        #expect(forward.fedTokens == Array(committed.dropLast()))

        // Row 0 is conditioned on the seed only: argmax = (0 + 1) = 1, while the
        // candidate committed 3 — a divergence the oracle RECORDS, not rejects.
        #expect(positions[0].committedToken == 3)
        #expect(positions[0].sequentialArgmax == scriptedPeak(after: 0, vocabSize: vocabSize))
        #expect(positions[0].sequentialArgmax == 1)
        #expect(positions[0].sequentialArgmax != positions[0].committedToken)

        // Row 1 is the discriminator. Anchored to the CANDIDATE's committed[0]=3
        // ⇒ argmax = 4. A static tape anchored to the reference's OWN row-0
        // argmax (1) would give argmax = 2. The oracle must produce 4.
        #expect(positions[1].sequentialArgmax == scriptedPeak(after: committed[0], vocabSize: vocabSize))
        #expect(positions[1].sequentialArgmax == 4)
        let staticTapeRow1 = scriptedPeak(after: positions[0].sequentialArgmax, vocabSize: vocabSize)
        #expect(staticTapeRow1 == 2)
        #expect(positions[1].sequentialArgmax != staticTapeRow1)

        // Row 2 anchored to committed[1]=5 ⇒ argmax = 6, vs a static tape's 5.
        #expect(positions[2].sequentialArgmax == scriptedPeak(after: committed[1], vocabSize: vocabSize))
        #expect(positions[2].sequentialArgmax == 6)
        let staticTapeRow2 = scriptedPeak(after: positions[1].sequentialArgmax, vocabSize: vocabSize)
        #expect(positions[2].sequentialArgmax != staticTapeRow2)

        // The whole produced argmax chain differs from the reference's own
        // self-chain from row 1 on — the exact thing a static tape gets wrong.
        let referenceSelfChain = [1, 2, 3]
        #expect(positions.map(\.sequentialArgmax) == [1, 4, 6])
        #expect(positions.map(\.sequentialArgmax) != referenceSelfChain)
    }

    // MARK: - Trusted-side invariant: readout is the reference's, not the candidate's

    @Test
    func referenceRankedReadoutIsIndependentOfTheCommittedTokenIdentity() throws {
        // The ranked reference tokens/logits are a property of the REFERENCE
        // forward given the (identical) prefix, never of the committed token id.
        // Changing only the committed token must leave the ranked reference
        // readout untouched and move ONLY the committed-token gap.
        let vocabSize = 16
        let seed = [7]
        var forwardA = ScriptedReferenceForward(vocabSize: vocabSize)
        var forwardB = ScriptedReferenceForward(vocabSize: vocabSize)
        let a = try replayCohortReferenceStream(
            seed: seed, committed: [2], forward: &forwardA,
            logitTopK: 5, relEnvelope: 0.05)[0]
        let b = try replayCohortReferenceStream(
            seed: seed, committed: [9], forward: &forwardB,
            logitTopK: 5, relEnvelope: 0.05)[0]

        #expect(a.rankedTokens == b.rankedTokens)
        #expect(a.rankedLogits == b.rankedLogits)
        #expect(a.rankedRelativeGaps == b.rankedRelativeGaps)
        #expect(a.sequentialArgmax == b.sequentialArgmax)
        #expect(a.withinEnvelopeDepth == b.withinEnvelopeDepth)
        // Only the committed-token readout differs.
        #expect(a.committedToken == 2)
        #expect(b.committedToken == 9)
        #expect(a.committedTokenLogit != b.committedTokenLogit)
    }

    // MARK: - Pure core math

    @Test
    func pureReadoutMatchesHandComputedRankedAndCommittedGap() throws {
        // flat sorted desc: 9(1) 8(5) 7(3) 4(6) 3(2) 2(4) 1(0) 0(7).
        let flat: [Float] = [1, 9, 3, 7, 2, 8, 4, 0]
        let position = try cohortReferenceReplayReadout(
            referenceLogits: flat, committedToken: 3, logitTopK: 3, relEnvelope: 0.05)

        #expect(position.sequentialArgmax == 1)
        #expect(position.rankedTokens == [1, 5, 3])
        #expect(position.rankedLogits == [9, 8, 7])
        // Relative gaps: (9-9)/9, (9-8)/9, (9-7)/9 with denom = max(1, |9|) = 9.
        #expect(position.rankedRelativeGaps[0] == 0)
        #expect(abs(position.rankedRelativeGaps[1] - 1.0 / 9.0) < 1e-9)
        #expect(abs(position.rankedRelativeGaps[2] - 2.0 / 9.0) < 1e-9)
        // Committed token 3 has logit 7 ⇒ gap (9-7)/9.
        #expect(position.committedTokenLogit == 7)
        #expect(abs(position.committedRelativeGap - 2.0 / 9.0) < 1e-9)
        // Only rank 0 is within a 0.05 envelope.
        #expect(position.withinEnvelopeDepth == 1)
    }

    @Test
    func readoutRejectsACommittedTokenOutsideTheReferenceVocab() {
        let flat: [Float] = [1, 2, 3, 4]
        #expect(throws: MLXFastError.self) {
            _ = try cohortReferenceReplayReadout(
                referenceLogits: flat, committedToken: 4, logitTopK: 2,
                relEnvelope: 0.05)
        }
    }

    // MARK: - Canonical width-1, per-stream independence

    @Test
    func eachStreamSteppsOneTokenAtATimeAndStreamsAreIndependent() throws {
        // The mock records exactly the tokens it was stepped on; a width-1 walk
        // steps ONE token per row (never a [B,1] batched forward), and two
        // streams replayed with fresh forwards do not see each other's journal.
        let vocabSize = 32
        var forwardA = ScriptedReferenceForward(vocabSize: vocabSize)
        let a = try replayCohortReferenceStream(
            seed: [1, 2], committed: [10, 11, 12, 13], forward: &forwardA,
            logitTopK: 4, relEnvelope: 0.05)
        #expect(a.count == 4)
        #expect(forwardA.fedTokens == [10, 11, 12])

        var forwardB = ScriptedReferenceForward(vocabSize: vocabSize)
        let b = try replayCohortReferenceStream(
            seed: [1, 2], committed: [20, 21], forward: &forwardB,
            logitTopK: 4, relEnvelope: 0.05)
        // Same seed, different journal ⇒ independent readouts (stream B's row 0
        // sees only the seed, exactly as stream A's did).
        #expect(a[0].sequentialArgmax == b[0].sequentialArgmax)
        #expect(forwardB.fedTokens == [20])
    }

    // MARK: - Cohort-width (batch-B) replay driver (GPU-free)

    @Test
    func batchedReplayTeacherForcesEachStreamOnItsOwnCandidatePrefix() throws {
        // Two streams with DIFFERENT committed journals that both diverge from
        // the reference's own greedy chain. The batched driver must teacher-force
        // each stream on ITS OWN committed tokens (never the reference's argmax,
        // never another stream's journal) and preserve slot order.
        let vocabSize = 32
        let seedsByStream = [[0], [5]]
        let committedByStream = [[3, 7, 2], [9, 1, 4]]
        var forward = ScriptedBatchedReferenceForward(
            vocabSize: vocabSize, batchSize: 2)
        let positionsByStream = try replayCohortReferenceBatched(
            seedsByStream: seedsByStream,
            committedByStream: committedByStream,
            forward: &forward,
            logitTopK: 4, relEnvelope: 0.05)

        #expect(positionsByStream.count == 2)
        // Each stream stepped on its own committed tokens (all but the last).
        #expect(forward.fedByStream == [[3, 7], [9, 1]])

        for slot in 0 ..< 2 {
            let positions = positionsByStream[slot]
            let committed = committedByStream[slot]
            #expect(positions.count == committed.count)
            // Row 0 conditioned on the seed only.
            #expect(
                positions[0].sequentialArgmax
                    == scriptedPeak(after: seedsByStream[slot][0], vocabSize: vocabSize))
            // Row i (i>0) anchored to THIS stream's committed[i-1].
            for i in 1 ..< committed.count {
                #expect(
                    positions[i].sequentialArgmax
                        == scriptedPeak(after: committed[i - 1], vocabSize: vocabSize))
                #expect(positions[i].committedToken == committed[i])
            }
        }
    }

    @Test
    func batchedReplayMatchesTheWidthOneReadoutOnTheWidthInvariantMock() throws {
        // On the width-invariant scripted mock (each row's logits depend only on
        // its own last token), the cohort-width batched replay and the canonical
        // width-1 per-stream replay must produce byte-identical readouts — the
        // driver wiring is the same anchor, only the forward geometry differs.
        // (On the real quantized model the geometries diverge; that is the
        // model-backed suite's job, and the whole point of the fix.)
        let vocabSize = 24
        let seedsByStream = [[1, 2], [3, 4]]
        let committedByStream = [[10, 11, 12], [20, 21, 22]]

        var batched = ScriptedBatchedReferenceForward(
            vocabSize: vocabSize, batchSize: 2)
        let batchedByStream = try replayCohortReferenceBatched(
            seedsByStream: seedsByStream, committedByStream: committedByStream,
            forward: &batched, logitTopK: 5, relEnvelope: 0.05)

        for slot in 0 ..< 2 {
            var widthOne = ScriptedReferenceForward(vocabSize: vocabSize)
            let canonical = try replayCohortReferenceStream(
                seed: seedsByStream[slot], committed: committedByStream[slot],
                forward: &widthOne, logitTopK: 5, relEnvelope: 0.05)
            #expect(batchedByStream[slot] == canonical)
        }
    }

    @Test
    func batchedReplayRejectsARaggedCohort() {
        let vocabSize = 16
        // Ragged committed journals (lengths 3 and 2).
        var forward = ScriptedBatchedReferenceForward(vocabSize: vocabSize, batchSize: 2)
        #expect(throws: MLXFastError.self) {
            _ = try replayCohortReferenceBatched(
                seedsByStream: [[1], [1]],
                committedByStream: [[2, 3, 4], [5, 6]],
                forward: &forward, logitTopK: 4, relEnvelope: 0.05)
        }
        // Ragged seeds (lengths 2 and 1).
        var forward2 = ScriptedBatchedReferenceForward(vocabSize: vocabSize, batchSize: 2)
        #expect(throws: MLXFastError.self) {
            _ = try replayCohortReferenceBatched(
                seedsByStream: [[1, 2], [3]],
                committedByStream: [[4, 5], [6, 7]],
                forward: &forward2, logitTopK: 4, relEnvelope: 0.05)
        }
    }

    @Test
    func emptySeedOrJournalIsRejected() {
        var forward = ScriptedReferenceForward(vocabSize: 8)
        #expect(throws: MLXFastError.self) {
            _ = try replayCohortReferenceStream(
                seed: [], committed: [1], forward: &forward,
                logitTopK: 2, relEnvelope: 0.05)
        }
        var forward2 = ScriptedReferenceForward(vocabSize: 8)
        #expect(throws: MLXFastError.self) {
            _ = try replayCohortReferenceStream(
                seed: [1], committed: [], forward: &forward2,
                logitTopK: 2, relEnvelope: 0.05)
        }
    }

    // MARK: - Wire validation (pure; the trusted-CLI verb is not spawn-gated)

    private func request(_ json: String) throws -> RuntimeWorkerRequest {
        try JSONDecoder().decode(RuntimeWorkerRequest.self, from: Data(json.utf8))
    }

    private func rejection(
        _ json: String,
        context: RuntimeWorkerRequestContext = RuntimeWorkerRequestContext()
    ) -> String? {
        do {
            _ = try validateGenericWorkerRequest(
                try request(json), context: context,
                specRegistry: .serialOnlyWorker)
            return nil
        } catch {
            return "\(error)"
        }
    }

    private let validJSON = #"""
        {"id":1,"kind":"cohort_reference_replay",
         "replay_seeds_by_stream":[[1,2],[3,4]],
         "committed_by_stream":[[5,6],[7,8]],
         "logit_top_k":16,"rel_envelope":0.05}
        """#

    @Test
    func acceptsAWellFormedReplayRequestWithoutTheSpawnGate() throws {
        // Trusted-CLI-only: accepted on a gate-OFF context (the trusted CLI
        // spawns this worker without --speculative-protocol, like the recorder).
        let validated = try validateGenericWorkerRequest(
            try request(validJSON), context: RuntimeWorkerRequestContext(),
            specRegistry: .serialOnlyWorker)
        let payload = try #require(validated.cohortReferenceReplay)
        #expect(payload.seedsByStream == [[1, 2], [3, 4]])
        #expect(payload.committedByStream == [[5, 6], [7, 8]])
        #expect(payload.logitTopK == 16)
        #expect(payload.relEnvelope == 0.05)
    }

    @Test
    func defaultsTheCharacterizationParamsWhenAbsent() throws {
        let json = #"""
            {"id":1,"kind":"cohort_reference_replay",
             "replay_seeds_by_stream":[[1,2]],"committed_by_stream":[[5]]}
            """#
        let validated = try validateGenericWorkerRequest(
            try request(json), context: RuntimeWorkerRequestContext(),
            specRegistry: .serialOnlyWorker)
        let payload = try #require(validated.cohortReferenceReplay)
        #expect(payload.logitTopK == cohortReferenceReplayDefaultLogitTopK)
        #expect(payload.relEnvelope == cohortReferenceReplayDefaultRelEnvelope)
        // replay_width defaults to the David-ruled cohort (batch-B) width.
        #expect(payload.replayWidth == cohortReferenceReplayDefaultWidth)
        #expect(payload.replayWidth == .cohort)
    }

    @Test
    func parsesTheReplayWidthModeAndDefaultsToCohort() throws {
        func validate(_ json: String) throws -> RuntimeWorkerValidatedCohortReferenceReplay {
            try #require(
                try validateGenericWorkerRequest(
                    try request(json), context: RuntimeWorkerRequestContext(),
                    specRegistry: .serialOnlyWorker
                ).cohortReferenceReplay)
        }
        // Explicit canonical (width-1) parses — a ragged cohort is allowed there.
        let canonical = try validate(#"""
            {"id":1,"kind":"cohort_reference_replay",
             "replay_seeds_by_stream":[[1,2],[3,4]],
             "committed_by_stream":[[5,6,7],[8]],
             "replay_width":"canonical"}
            """#)
        #expect(canonical.replayWidth == .canonical)
        // Explicit cohort parses on a rectangular cohort.
        let cohort = try validate(#"""
            {"id":1,"kind":"cohort_reference_replay",
             "replay_seeds_by_stream":[[1,2],[3,4]],
             "committed_by_stream":[[5,6],[7,8]],
             "replay_width":"cohort"}
            """#)
        #expect(cohort.replayWidth == .cohort)
    }

    @Test
    func rejectsAnUnknownReplayWidth() {
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1]],"committed_by_stream":[[2]],
                 "replay_width":"wide"}
                """#) != nil)
    }

    @Test
    func cohortWidthRequiresARectangularCohortButCanonicalDoesNot() {
        // Ragged committed journals under cohort width ⇒ refused at validation.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1,2],[3,4]],
                 "committed_by_stream":[[5,6],[7]],
                 "replay_width":"cohort"}
                """#) != nil)
        // Ragged seed lengths under cohort width ⇒ refused.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1,2],[3]],
                 "committed_by_stream":[[5,6],[7,8]],
                 "replay_width":"cohort"}
                """#) != nil)
        // The DEFAULT is cohort, so the same ragged cohort with NO replay_width
        // is refused too (the default carries the rectangular requirement).
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1,2],[3,4]],
                 "committed_by_stream":[[5,6],[7]]}
                """#) != nil)
        // The SAME ragged cohort under canonical width is accepted.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1,2],[3,4]],
                 "committed_by_stream":[[5,6],[7]],
                 "replay_width":"canonical"}
                """#) == nil)
    }

    @Test
    func replayWidthIsRejectedOnOtherKinds() {
        #expect(
            rejection(
                #"{"id":1,"kind":"correctness","prompt_tokens":[1],"steps":1,"replay_width":"cohort"}"#)
                != nil)
    }

    @Test
    func rejectsMalformedReplayRequests() {
        // Missing committed_by_stream.
        #expect(
            rejection(#"{"id":1,"kind":"cohort_reference_replay","replay_seeds_by_stream":[[1]]}"#)
                != nil)
        // Mismatched stream counts.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1],[2]],"committed_by_stream":[[3]]}
                """#) != nil)
        // Empty seed.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[]],"committed_by_stream":[[3]]}
                """#) != nil)
        // Empty committed journal.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1]],"committed_by_stream":[[]]}
                """#) != nil)
        // Out-of-vocab committed token.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1]],"committed_by_stream":[[999999999]]}
                """#) != nil)
        // Over-wide cohort (B > 8).
        let wideSeeds = (0 ..< 9).map { "[\($0)]" }.joined(separator: ",")
        let wideCommitted = (0 ..< 9).map { "[\($0)]" }.joined(separator: ",")
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[\#(wideSeeds)],
                 "committed_by_stream":[\#(wideCommitted)]}
                """#) != nil)
        // logit_top_k must be >= 1.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1]],"committed_by_stream":[[2]],
                 "logit_top_k":0}
                """#) != nil)
        // rel_envelope must be in 0...1.
        #expect(
            rejection(#"""
                {"id":1,"kind":"cohort_reference_replay",
                 "replay_seeds_by_stream":[[1]],"committed_by_stream":[[2]],
                 "rel_envelope":2.0}
                """#) != nil)
    }

    @Test
    func replayFieldsAreRejectedOnOtherKinds() {
        // committed_by_stream on a correctness request is a cross-kind field.
        #expect(
            rejection(
                #"{"id":1,"kind":"correctness","prompt_tokens":[1],"steps":1,"committed_by_stream":[[2]]}"#)
                != nil)
        // replay_seeds_by_stream is NOT the free-run cohort seed field — it is
        // rejected on free_decode_begin.
        #expect(
            rejection(
                #"{"id":1,"kind":"free_decode_begin","replay_seeds_by_stream":[[1]]}"#,
                context: RuntimeWorkerRequestContext(advertisesSpeculativeProtocol: true))
                != nil)
    }

    // MARK: - Wire round-trip

    @Test
    func requestDecodesTheReplayFieldsWithBenchdWireNames() throws {
        let decoded = try request(validJSON)
        #expect(decoded.replaySeedsByStream == [[1, 2], [3, 4]])
        #expect(decoded.committedByStream == [[5, 6], [7, 8]])
        #expect(decoded.logitTopK == 16)
        #expect(decoded.relEnvelope == 0.05)
    }

    @Test
    func requestEnvelopeStaysClosedOverUnknownFields() {
        #expect(throws: (any Error).self) {
            _ = try self.request(
                #"{"id":1,"kind":"cohort_reference_replay","not_a_field":1}"#)
        }
    }

    @Test
    func responseCarriesTheReportUnderTheBenchdWireName() throws {
        let report = CohortReferenceReplayReport(
            logitProvenance: "post_softcap",
            logitTopK: 16,
            relEnvelope: 0.05,
            replayWidth: CohortReferenceReplayWidth.cohort.rawValue,
            streams: [
                CohortReferenceReplayStreamReport(
                    slot: 0,
                    positions: [
                        CohortReferenceReplayPosition(
                            committedToken: 5,
                            sequentialArgmax: 5,
                            rankedTokens: [5, 9],
                            rankedLogits: [3.5, 2.0],
                            rankedRelativeGaps: [0, 0.4285714285714286],
                            committedTokenLogit: 3.5,
                            committedRelativeGap: 0,
                            withinEnvelopeDepth: 1)
                    ])
            ])
        let response = RuntimeWorkerResponse(
            id: 1, nonce: "n", ok: true, cohortReferenceReplay: report)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(response)
        let json = String(decoding: data, as: UTF8.self)
        #expect(json.contains("\"cohort_reference_replay\""))
        #expect(json.contains("\"logit_provenance\":\"post_softcap\""))
        #expect(json.contains("\"sequential_argmax\":5"))
        #expect(json.contains("\"committed_relative_gap\":0"))
        #expect(json.contains("\"within_envelope_depth\":1"))
        // The replay-width stamp rides under benchd's wire name too.
        #expect(json.contains("\"replay_width\":\"cohort\""))

        // Round-trips back to an equal report.
        let decoded = try JSONDecoder().decode(RuntimeWorkerResponse.self, from: data)
        #expect(decoded.cohortReferenceReplay == report)
    }

    // MARK: - Model-backed (gated): the oracle over a real synthetic Gemma4 forward

    private let mbVocabSize = 64
    private let mbHiddenSize = 32

    private func syntheticConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(mbHiddenSize),
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "enable_moe_block": true,
                "num_experts": 8,
                "top_k_experts": 2,
                "moe_intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 1,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": 12,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "vocab_size": \(mbVocabSize),
                "vocab_size_per_layer_input": \(mbVocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// An independent width-1 teacher-forced walk over `committed`, returning the
    /// per-position argmax and the committed token's logit — the ground truth the
    /// oracle must reproduce, computed here without the oracle under test.
    private func groundTruthWalk(
        model: Gemma4TextModel, seed: [Int], committed: [Int]
    ) -> (argmax: [Int], committedLogit: [Float]) {
        let cache = model.newCache(parameters: nil)
        var argmax: [Int] = []
        var committedLogit: [Float] = []
        var logits = model(
            MLXArray(seed.map(Int32.init)).reshaped([1, seed.count]), cache: cache)
        var current = logits[0 ..< 1, (seed.count - 1) ..< seed.count, 0...]
        for (index, token) in committed.enumerated() {
            let flat = current.asType(.float32).flattened()
            argmax.append(flat.argMax().item(Int.self))
            committedLogit.append(flat[token].item(Float.self))
            if index < committed.count - 1 {
                logits = model(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
                current = logits[0 ..< 1, 0 ..< 1, 0...]
            }
        }
        return (argmax, committedLogit)
    }

    @Test
    func modelBackedReplayReproducesTheCandidateAnchoredReferenceWalk() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0xCAFE_F00D)
        let model = Gemma4TextModel(try syntheticConfig())
        eval(model)

        let seed = [1, 2, 3, 4, 5]
        // Two streams; stream 0's journal is DELIBERATELY not the reference's own
        // greedy chain (arbitrary tokens), so the replay must follow the
        // candidate's prefix, not a self-chain.
        let committedByStream = [[10, 20, 30, 40], [7, 8, 9]]

        let report = try Gemma4Runtime.cohortReferenceReplayReport(
            model: model,
            seedsByStream: [seed, seed],
            committedByStream: committedByStream,
            logitTopK: 8,
            relEnvelope: 0.05,
            // CANONICAL width-1: this test uses RAGGED committed journals and a
            // width-1 ground-truth walk, so it exercises the per-stream path.
            replayWidth: .canonical)

        #expect(report.logitProvenance == "post_softcap")
        #expect(report.replayWidth == "canonical")
        #expect(report.streams.count == 2)
        for (slot, stream) in report.streams.enumerated() {
            #expect(stream.slot == slot)
            let committed = committedByStream[slot]
            #expect(stream.positions.count == committed.count)
            let truth = groundTruthWalk(
                model: model, seed: seed, committed: committed)
            for (index, position) in stream.positions.enumerated() {
                #expect(position.committedToken == committed[index])
                // The reference argmax matches a width-1 walk teacher-forced on
                // the CANDIDATE's own tokens — the anti-static-tape property, on
                // a real forward.
                #expect(position.sequentialArgmax == truth.argmax[index])
                #expect(
                    abs(position.committedTokenLogit - Double(truth.committedLogit[index]))
                        < 1e-3)
                #expect(position.rankedTokens.count == 8)
                #expect(position.rankedTokens.first == position.sequentialArgmax)
                #expect(position.committedRelativeGap >= 0)
            }
        }
    }

    /// An independent BATCHED [B, 1] teacher-forced walk over a RECTANGULAR
    /// cohort, returning per-stream per-position argmax and committed-token
    /// logit — the cohort-width ground truth, computed here without the oracle
    /// under test. Mirrors `groundTruthWalk` widened to B rows on one shared KV,
    /// exactly the geometry `Gemma4CohortReferenceBatchedForward` drives.
    private func groundTruthBatchedWalk(
        model: Gemma4TextModel, seedsByStream: [[Int]], committedByStream: [[Int]]
    ) -> (argmax: [[Int]], committedLogit: [[Float]]) {
        let batchSize = seedsByStream.count
        let seedLength = seedsByStream[0].count
        let committedLength = committedByStream[0].count
        let cache = model.newCache(parameters: nil)
        var argmax: [[Int]] = Array(repeating: [], count: batchSize)
        var committedLogit: [[Float]] = Array(repeating: [], count: batchSize)

        let seedFlat = seedsByStream.flatMap { $0.map(Int32.init) }
        var logits = model(
            MLXArray(seedFlat).reshaped([batchSize, seedLength]), cache: cache)
        var current = logits[0..., (seedLength - 1) ..< seedLength, 0...]
        for index in 0 ..< committedLength {
            for slot in 0 ..< batchSize {
                let row = current[slot ..< slot + 1, 0..., 0...]
                let flat = row.asType(.float32).flattened()
                argmax[slot].append(flat.argMax().item(Int.self))
                committedLogit[slot].append(
                    flat[committedByStream[slot][index]].item(Float.self))
            }
            if index < committedLength - 1 {
                let feed = committedByStream.map { Int32($0[index]) }
                logits = model(MLXArray(feed).reshaped([batchSize, 1]), cache: cache)
                current = logits[0..., 0 ..< 1, 0...]
            }
        }
        return (argmax, committedLogit)
    }

    @Test
    func modelBackedCohortWidthReplayMatchesTheBatchedGroundTruthWalk() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0xCAFE_F00D)
        let model = Gemma4TextModel(try syntheticConfig())
        eval(model)

        // RECTANGULAR cohort (equal seeds, equal committed lengths) — the scored
        // cohort's shape. Journals deliberately diverge from the reference's own
        // chain so the batched replay must follow each candidate's prefix.
        let seed = [1, 2, 3, 4, 5]
        let seedsByStream = [seed, seed, seed]
        let committedByStream = [[10, 20, 30, 40], [7, 8, 9, 11], [3, 5, 61, 62]]

        let report = try Gemma4Runtime.cohortReferenceReplayReport(
            model: model,
            seedsByStream: seedsByStream,
            committedByStream: committedByStream,
            logitTopK: 8,
            relEnvelope: 0.05,
            replayWidth: .cohort)

        #expect(report.logitProvenance == "post_softcap")
        #expect(report.replayWidth == "cohort")
        #expect(report.streams.count == 3)
        let truth = groundTruthBatchedWalk(
            model: model, seedsByStream: seedsByStream,
            committedByStream: committedByStream)
        for (slot, stream) in report.streams.enumerated() {
            #expect(stream.slot == slot)
            let committed = committedByStream[slot]
            #expect(stream.positions.count == committed.count)
            for (index, position) in stream.positions.enumerated() {
                #expect(position.committedToken == committed[index])
                // The reference argmax matches a BATCH-B walk teacher-forced on
                // the CANDIDATE's own tokens — the like-for-like cohort geometry,
                // on a real forward.
                #expect(position.sequentialArgmax == truth.argmax[slot][index])
                #expect(
                    abs(position.committedTokenLogit
                        - Double(truth.committedLogit[slot][index])) < 1e-3)
                #expect(position.rankedTokens.count == 8)
                #expect(position.rankedTokens.first == position.sequentialArgmax)
                #expect(position.committedRelativeGap >= 0)
            }
        }
    }

    /// The STAMP, end to end: the report records the width the reference
    /// ACTUALLY replayed at.
    ///
    /// Before this, the enforced reference GEOMETRY — the thing that decides
    /// whether a measured divergence is the candidate's real error or pure batch
    /// numerics — rode `cohortReferenceReplayDefaultWidth`, an engine-side
    /// default that no sealed artifact recorded. A consumer holding a report
    /// could not tell which geometry produced it, and a change to that default
    /// would silently re-base every archived measurement. The third leg below is
    /// the load-bearing one: a request that OMITS `replay_width` (the shape
    /// benchd actually sends) must still come back stamped with the width the
    /// default resolved to.
    @Test
    func reportStampsTheWidthActuallyReplayedIncludingOnTheDefaultPath() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0xCAFE_F00D)
        let model = Gemma4TextModel(try syntheticConfig())
        eval(model)

        // Rectangular, so the same cohort is legal at BOTH widths and the stamp
        // is the only thing that varies.
        let seedsByStream = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        let committedByStream = [[10, 20, 30], [7, 8, 9]]

        func report(
            _ width: CohortReferenceReplayWidth?
        ) throws -> CohortReferenceReplayReport {
            if let width {
                return try Gemma4Runtime.cohortReferenceReplayReport(
                    model: model, seedsByStream: seedsByStream,
                    committedByStream: committedByStream,
                    logitTopK: 8, relEnvelope: 0.05, replayWidth: width)
            }
            // No `replayWidth:` argument at all — the DEFAULT path.
            return try Gemma4Runtime.cohortReferenceReplayReport(
                model: model, seedsByStream: seedsByStream,
                committedByStream: committedByStream,
                logitTopK: 8, relEnvelope: 0.05)
        }

        // A cohort-width replay's report carries "cohort".
        #expect(try report(.cohort).replayWidth == "cohort")
        // A canonical request's report carries "canonical".
        #expect(try report(.canonical).replayWidth == "canonical")
        // And an omitted `replay_width` is stamped with what the default
        // RESOLVED to, not left blank or guessed — stated against the constant
        // so re-ruling the default moves this expectation with it.
        #expect(try report(nil).replayWidth == cohortReferenceReplayDefaultWidth.rawValue)
        #expect(try report(nil).replayWidth == "cohort")
        // The stamp is always a legal wire value.
        #expect(
            CohortReferenceReplayWidth(rawValue: try report(.canonical).replayWidth)
                == .canonical)
    }

    @Test
    func modelBackedCohortWidthRefusesARaggedCohort() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
        else { return }
        MLXRandom.seed(0xCAFE_F00D)
        let model = Gemma4TextModel(try syntheticConfig())
        eval(model)
        // Ragged committed journals under replay_width cohort ⇒ refusal.
        #expect(throws: MLXFastError.self) {
            _ = try Gemma4Runtime.cohortReferenceReplayReport(
                model: model,
                seedsByStream: [[1, 2, 3], [1, 2, 3]],
                committedByStream: [[10, 11, 12], [20, 21]],
                logitTopK: 8, relEnvelope: 0.05, replayWidth: .cohort)
        }
    }
}
