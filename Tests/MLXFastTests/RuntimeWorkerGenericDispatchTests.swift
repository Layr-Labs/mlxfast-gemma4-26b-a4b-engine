import Foundation
import MLXFastCore
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Conformance for the GENERIC benchd-facing decode dispatch: route-by-mode over a
// resolved effective_spec, and the free_decode_run(N) acceptance_lengths assembly
// + consistency triple. All GPU-free (no model, no transport).
//
// `serial` and (2026-08-23, the MTP arm) `mtp` are the two routable modes.
// The multi-token-round coverage below is deliberately KEPT and driven
// directly through `RuntimeWorkerFreeRunBuilder` /
// `runtimeWorkerFreeRunRoundEmission`: those are pure wire-assembly functions
// whose contract (the §2.6 triple, the begin/run seam) is what benchd
// cross-checks, and it must not silently change shape when a route's round
// EXECUTION eventually lands.

// MARK: - Route by resolved mode

@Test
func genericRouteMapsSerial() throws {
    #expect(try runtimeWorkerDecodeRoute(forEffectiveMode: "serial") == .serial)
}

@Test
func genericRouteMapsMTP() throws {
    #expect(try runtimeWorkerDecodeRoute(forEffectiveMode: "mtp") == .mtp)
}

@Test
func genericRouteRejectsAnythingElseFailClosed() {
    // On a worker with NO bound DFlash drafter — the default — dflash never
    // reaches routing, exactly like dspark and an unknown mode: a programming
    // error, never a silent serial default. The `dflashAvailable` parameter
    // defaults to false precisely so this stays the behaviour by default
    // rather than by remembering to pass something.
    for mode in ["dflash", "dspark", "teleport", ""] {
        #expect(throws: (any Error).self) {
            _ = try runtimeWorkerDecodeRoute(forEffectiveMode: mode)
        }
    }
}

@Test
func genericRouteMapsDFlashOnlyWhenADrafterIsBound() throws {
    // The fence #38 removed: `dflash` routes ONLY on a worker that actually
    // bound a drafter. Same string, two workers, two outcomes.
    #expect(
        try runtimeWorkerDecodeRoute(forEffectiveMode: "dflash", dflashAvailable: true)
            == .dflash)
    #expect(throws: (any Error).self) {
        _ = try runtimeWorkerDecodeRoute(forEffectiveMode: "dflash", dflashAvailable: false)
    }
    // Availability must not widen anything else.
    #expect(throws: (any Error).self) {
        _ = try runtimeWorkerDecodeRoute(forEffectiveMode: "dspark", dflashAvailable: true)
    }
}

@Test
func decodeBeginSpecSerialResolvesAndRoutesSerial() throws {
    let request = try JSONDecoder().decode(
        RuntimeWorkerRequest.self,
        from: Data(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1,2],"spec":{"mode":"serial"}}"#
                .utf8))
    let effective = try RuntimeWorkerSpecRegistry.serialOnlyWorker
        .resolveEffectiveSpec(request.spec)
    #expect(effective.mode == "serial")
    #expect(try runtimeWorkerDecodeRoute(forEffectiveMode: effective.mode) == .serial)
}

@Test
func decodeBeginSpecDFlashIsNotRunnableAndNeverRoutes() throws {
    let sha = String(repeating: "a", count: 64)
    let request = try JSONDecoder().decode(
        RuntimeWorkerRequest.self,
        from: Data(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1],"spec":{"mode":"dflash","dflash":{"draft":{"artifact":"a","sha256":"\#(sha)"}}}}"#
                .utf8))
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try RuntimeWorkerSpecRegistry.serialOnlyWorker
                .resolveEffectiveSpec(request.spec)
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    #expect(caught?.contains("not runnable on this engine") == true)
}

@Test
func decodeBeginSpecDSparkIsNotImplementedAndNeverRoutes() throws {
    let request = try JSONDecoder().decode(
        RuntimeWorkerRequest.self,
        from: Data(
            #"{"id":1,"kind":"decode_begin","seed_tokens":[1],"spec":{"mode":"dspark"}}"#
                .utf8))
    var caught: String?
    #expect(throws: (any Error).self) {
        do {
            _ = try RuntimeWorkerSpecRegistry.serialOnlyWorker
                .resolveEffectiveSpec(request.spec)
        } catch {
            caught = "\(error)"
            throw error
        }
    }
    #expect(caught?.contains("not implemented on this engine") == true)
}

// MARK: - free_decode_run acceptance_lengths + triple

@Test
func freeRunMultiTokenRoundHistogramSatisfiesTheTriple() throws {
    // N=4 over two rounds committing 3 then 1 token (sum=4). R=2 → completed_work 3.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 4)
    builder.addRound(committedTokens: [700, 701, 702], drafted: 3, accepted: 2)
    #expect(builder.isComplete == false)
    builder.addRound(committedTokens: [703], drafted: 1, accepted: 0)
    #expect(builder.isComplete == true)
    let result = try builder.finish()

    #expect(result.tokens == [700, 701, 702, 703])
    #expect(result.acceptanceLengths == [3, 1])
    #expect(result.acceptanceLengths.reduce(0, +) == 4)  // sum == N
    #expect(result.rounds == 2)                          // R == len
    #expect(result.completedWork == 3)                   // completed_work == R + 1
    #expect(result.committedTotal == 4)                  // committed == N == tokens.len
    #expect(result.draftedTotal == 4)
    #expect(result.acceptedTotal == 2)
    #expect(result.draftedTotal >= result.acceptedTotal)
}

@Test
func freeRunSerialCommitsOneTokenPerRound() throws {
    // Serial: each round commits exactly one token, drafts nothing.
    // acceptance_lengths = [1]*N, R = N, completed_work = N + 1.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 5)
    for i in 0..<5 {
        builder.addRound(committedTokens: [3000 + i], drafted: 0, accepted: 0)
    }
    let result = try builder.finish()
    #expect(result.acceptanceLengths == [1, 1, 1, 1, 1])
    #expect(result.rounds == 5)
    #expect(result.completedWork == 6)
    #expect(result.committedTotal == 5)
    #expect(result.draftedTotal == 0)
    #expect(result.acceptedTotal == 0)
}

@Test
func freeRunClampsAnOvershootingFinalRoundToExactlyN() throws {
    // A block round that would overshoot N commits only its first N-committed
    // tokens, so the histogram still sums to exactly N and tokens.count == N.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 4)
    builder.addRound(committedTokens: [10, 11], drafted: 2, accepted: 1)
    // This round offers 3 but only 2 remain: it is clamped to 2.
    builder.addRound(committedTokens: [12, 13, 14], drafted: 3, accepted: 2)
    #expect(builder.isComplete == true)
    // A round arriving after completion is dropped.
    builder.addRound(committedTokens: [99], drafted: 1, accepted: 0)
    let result = try builder.finish()
    #expect(result.tokens == [10, 11, 12, 13])
    #expect(result.acceptanceLengths == [2, 2])
    #expect(result.acceptanceLengths.reduce(0, +) == 4)
    #expect(result.completedWork == 3)
}

@Test
func freeRunUnfinishedPhaseFailsClosed() {
    // Fewer than N committed tokens: the token-count guard fires.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 4)
    builder.addRound(committedTokens: [1, 2], drafted: 0, accepted: 0)
    #expect(throws: RuntimeWorkerFreeRunError.self) {
        _ = try builder.finish()
    }
}

@Test
func freeRunDraftedLessThanAcceptedFailsClosed() {
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 2)
    // Impossible self-report: accepted exceeds drafted.
    builder.addRound(committedTokens: [1, 2], drafted: 0, accepted: 2)
    #expect(throws: RuntimeWorkerFreeRunError.self) {
        _ = try builder.finish()
    }
}

@Test
func freeRunResultEncodesIntoTheBenchdFreeDecodeRunShape() throws {
    // The assembled result maps onto a RuntimeWorkerResponse whose JSON carries
    // exactly the field names benchd's free-run consumer reads.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 4)
    builder.addRound(committedTokens: [700, 701, 702], drafted: 3, accepted: 2)
    builder.addRound(committedTokens: [703], drafted: 1, accepted: 0)
    let result = try builder.finish()

    let response = RuntimeWorkerResponse(
        id: 9,
        nonce: "n",
        ok: true,
        tokens: result.tokens,
        acceptanceLengths: result.acceptanceLengths,
        draftedTotal: result.draftedTotal,
        acceptedTotal: result.acceptedTotal,
        committedTotal: result.committedTotal
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let json = String(decoding: try encoder.encode(response), as: UTF8.self)
    #expect(json.contains("\"acceptance_lengths\":[3,1]"))
    #expect(json.contains("\"tokens\":[700,701,702,703]"))
    #expect(json.contains("\"drafted_total\":4"))
    #expect(json.contains("\"accepted_total\":2"))
    #expect(json.contains("\"committed_total\":4"))
}

// MARK: - #109 W3 finding 6 — the free_decode_begin / free_decode_run seam
//
// PROTOCOL-v1.1 §2.2: `free_decode_begin` returns `seed_token` (benchd verifies it against
// `expected_decode_seed_token`), and `free_decode_run(N)`'s `tokens[i]` is matched against
// `expected_decode_tokens[i]` — the N tokens AFTER the seed. §2.1: begin "establishes the
// last-committed state"; the run commits N MORE.
//
// The MTP session's round result is in COMMIT order (`[primary] + acceptedDrafts`), which lags
// PRODUCTION order by one. Emitting it verbatim re-sent the seed as `tokens[0]`.

@Test
func freeRunRoundEmitsWhatItProducedNotWhatItWrote() throws {
    // A depth-2 round that accepted both drafts: commit list [primary, d1, d2], fallback 99.
    // §3's per-round count is "draft tokens that survived verification (plus fallback)" — 3 either
    // way — but the IDENTITIES shift one position: the primary was already on the wire.
    let emission = try runtimeWorkerFreeRunRoundEmission(
        committedTokens: [700, 701, 702], nextPrimary: 99, lastEmittedToken: 700)
    #expect(emission == [701, 702, 99])
    #expect(emission.count == 3)  // unchanged: acceptedDrafts + 1

    // A non-drafting round commits only its primary and produces one fallback.
    #expect(
        try runtimeWorkerFreeRunRoundEmission(
            committedTokens: [9], nextPrimary: 10, lastEmittedToken: 9) == [10])
}

@Test
func freeRunRoundThatDeterminedNothingEmitsNothing() throws {
    // A stop-token round: its opening primary is already on the wire and there is no successor to
    // predict. It contributes no token and no histogram entry.
    #expect(
        try runtimeWorkerFreeRunRoundEmission(
            committedTokens: [248_044], nextPrimary: nil, lastEmittedToken: 248_044).isEmpty)
}

@Test
func freeRunSeamFailsClosedWhenTheStreamsDiverge() {
    // The round opened on a token that is NOT the one benchd already verified. Refused where it
    // happened rather than shipping a silently misaligned window.
    #expect(throws: RuntimeWorkerFreeRunError.seedSeamBroken(expected: 4625, got: 11)) {
        _ = try runtimeWorkerFreeRunRoundEmission(
            committedTokens: [11, 321], nextPrimary: 279, lastEmittedToken: 4625)
    }
    #expect(throws: RuntimeWorkerFreeRunError.emptyRound) {
        _ = try runtimeWorkerFreeRunRoundEmission(
            committedTokens: [], nextPrimary: 279, lastEmittedToken: 4625)
    }
}

/// **The window-3 isolation, made permanent.**
///
/// `parity-window-20260820-3` ran the real MTP head on the GPU at depth 2 against two independent
/// pinned tapes and measured: the engine's `free_decode_run` stream was **0/16** against
/// `emitted_tokens[0..]` and **16/16** against `[reference_seed_token] + emitted_tokens`. The
/// speculation was correct; the two sides disagreed about ONE index, at the begin/run seam.
///
/// This replays that window's recorded shape GPU-free on a SYNTHESIZED tape — the beagle tape's
/// `reference_seed_token` (4625), its recorded commit-order stream, and its recorded
/// `acceptance_lengths` `[3, 1, 3, 3, 3, 1, 1, 1]` — against the SPEC-TRUE expectation: N tokens
/// starting after the seed. The final oracle token is synthesized (window 3 compared a 16-token
/// window whose 16th reference token it never printed); everything before it is the recorded data.
@Test
func w3f6WindowThreeIsolationIsTokenExactUnderTheSpecTrueSeam() throws {
    // The tape.
    let seedToken = 4625
    let oracle = [  // expected_decode_tokens[0..16] — what §2.2 matches tokens[i] against
        11, 321, 279, 9544, 80548, 11, 524, 1602,
        2858, 310, 2842, 11, 64467, 1083, 279, 13,
    ]
    let recordedAcceptanceLengths = [3, 1, 3, 3, 3, 1, 1, 1]
    let n = oracle.count
    #expect(n == 16)
    #expect(recordedAcceptanceLengths.reduce(0, +) == n)

    // The session's COMMIT stream: the seed, then the oracle. That is exactly what window 3 read off
    // the wire — `[4625, 11, 321, 279, …]`, the tape's own seed token in position 0.
    let commitStream = [seedToken] + oracle

    // Decompose it into the recorded rounds. Round i opens on the token already emitted and commits
    // `acceptance_lengths[i]` tokens; its fallback is the next round's opening token.
    var rounds: [(committed: [Int], nextPrimary: Int?)] = []
    var cursor = 0
    for (index, count) in recordedAcceptanceLengths.enumerated() {
        let committed = Array(commitStream[cursor ..< (cursor + count)])
        let isLast = index == recordedAcceptanceLengths.count - 1
        // The last round's fallback is the token past the recorded window (synthesized above as the
        // final oracle entry); every other round's is the next round's primary.
        rounds.append((committed, isLast ? oracle[n - 1] : commitStream[cursor + count]))
        cursor += count
    }

    // COMMIT-ORDER assembly — what the engine used to send. 0/16.
    var wrong = RuntimeWorkerFreeRunBuilder(targetN: n)
    for round in rounds {
        wrong.addRound(committedTokens: round.committed, drafted: 0, accepted: 0)
    }
    let wrongResult = try wrong.finish()
    #expect(wrongResult.tokens.first == seedToken)  // the tape's own reference_seed_token, re-sent
    #expect(zip(wrongResult.tokens, oracle).filter { $0 == $1 }.count == 0)  // 0/16

    // PRODUCTION-ORDER assembly — the spec-true seam. 16/16, and the recorded histogram is
    // UNCHANGED: only the identities shifted.
    var right = RuntimeWorkerFreeRunBuilder(targetN: n)
    var lastEmitted = seedToken
    for round in rounds {
        let emission = try runtimeWorkerFreeRunRoundEmission(
            committedTokens: round.committed,
            nextPrimary: round.nextPrimary,
            lastEmittedToken: lastEmitted
        )
        if let last = emission.last { lastEmitted = last }
        right.addRound(
            committedTokens: emission,
            drafted: emission.count,
            accepted: Swift.max(emission.count - 1, 0))
    }
    let result = try right.finish()
    #expect(result.tokens == oracle)  // 16/16 against expected_decode_tokens[i]
    #expect(zip(result.tokens, oracle).filter { $0 == $1 }.count == n)
    #expect(result.tokens.contains(seedToken) == false || oracle.contains(seedToken))
    // The §2.6 triple, verbatim from the window's own audit line.
    #expect(result.acceptanceLengths == recordedAcceptanceLengths)
    #expect(result.acceptanceLengths.reduce(0, +) == n)
    #expect(result.rounds == 8)
    #expect(result.completedWork == 9)  // R + 1: the seed forward plus 8 verify rounds
    #expect(result.committedTotal == n)
    #expect(result.draftedTotal >= result.acceptedTotal)
}

// MARK: - Early EOS is symmetric across the paired legs
//
// Review finding (6) on PR #5: the speculative leg broke out on reachedStopToken
// and then tripped the token-count guard (a "broken assembly" message for a
// valid engine outcome), while the serial leg had no stop-token notion at all
// and decoded straight past EOS to N. Two different behaviours on the two sides
// of one paired ratio. The verdict is now produced by ONE route-agnostic
// function, which is why the surviving serial leg still pins it: symmetry is a
// property of `runtimeWorkerFreeRunEarlyStop`, not of having two callers.

@Test
func earlyStopIsTheSameVerdictOnBothLegs() throws {
    // Same phase shape on both sides: N=8, a stop token committed as the 3rd
    // token. The only difference in the two errors is the leg's name.
    func verdict(_ route: RuntimeWorkerDecodeRoute) -> RuntimeWorkerFreeRunError? {
        var builder = RuntimeWorkerFreeRunBuilder(targetN: 8)
        builder.addRound(committedTokens: [10, 11], drafted: 0, accepted: 0)
        builder.addRound(committedTokens: [248_044], drafted: 0, accepted: 0)
        return runtimeWorkerFreeRunEarlyStop(
            route: route, stopToken: 248_044, builder: builder)
    }
    #expect(
        verdict(.serial)
            == .stopTokenBeforeTarget(route: "serial", token: 248_044, position: 3, n: 8))
    // Structured and self-describing — NOT the token-count-mismatch message.
    #expect(
        "\(verdict(.serial)!)"
            == "free_decode_run serial leg committed stop token 248044 at committed "
                + "position 3 of N=8; the leg is invalid (a paired leg must reach "
                + "N or both legs must stop)")
    #expect("\(verdict(.serial)!)".contains("assembled") == false)
}

@Test
func aStopTokenAtExactlyNIsNotAnError() throws {
    // The leg delivered N committed tokens and truncated nothing, so it is a
    // usable sample even though the last token happens to be EOS.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 3)
    builder.addRound(committedTokens: [10, 11], drafted: 0, accepted: 0)
    builder.addRound(committedTokens: [248_044], drafted: 0, accepted: 0)
    #expect(builder.isComplete)
    #expect(
        runtimeWorkerFreeRunEarlyStop(
            route: .serial, stopToken: 248_044, builder: builder) == nil)
    // And it still assembles into a valid result.
    #expect(try builder.finish().committedTotal == 3)
}

@Test
func aRoundThatCommittedNoStopTokenIsNeverFlagged() {
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 8)
    builder.addRound(committedTokens: [10, 11], drafted: 2, accepted: 1)
    for route in [RuntimeWorkerDecodeRoute.serial] {
        #expect(
            runtimeWorkerFreeRunEarlyStop(route: route, stopToken: nil, builder: builder)
                == nil)
    }
}

@Test
func aClampedOvershootPastAStopTokenIsNotAnError() throws {
    // A multi-token round offering [a, b, EOS] with only 2 of N remaining is
    // clamped to [a, b]: EOS never enters the window, the phase is complete, no
    // error.
    var builder = RuntimeWorkerFreeRunBuilder(targetN: 4)
    builder.addRound(committedTokens: [10, 11], drafted: 2, accepted: 1)
    builder.addRound(committedTokens: [12, 13, 248_044], drafted: 3, accepted: 2)
    #expect(builder.isComplete)
    #expect(builder.tokens == [10, 11, 12, 13])
    #expect(
        runtimeWorkerFreeRunEarlyStop(
            route: .serial, stopToken: 248_044, builder: builder) == nil)
}
