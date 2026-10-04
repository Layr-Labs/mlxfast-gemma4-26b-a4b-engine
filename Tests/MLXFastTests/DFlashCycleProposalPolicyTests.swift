import Foundation
import MLXSpeculative
import Testing

// Coverage for the target-verified CYCLE PROPOSAL route added 2026-09-02:
// `DFlashCycleProposalPolicy` (the pure decision function) and the
// `proposal:` seam it drives in `runDFlashGreedyRound`.
//
// The policy tests are exhaustive and unconditional: the type touches no MLX,
// reads no environment, and holds no clock, so every rule it enforces is
// reachable on a host with no Metal and no weights. That is the whole reason
// the detection rule lives in its own type instead of inside the session.
//
// The ROUND tests are structural rather than executed. `runDFlashGreedyRound`
// takes a concrete `DFlashDraftModel`, not a protocol, so a spy cannot be
// substituted for the drafter, and constructing a real one allocates MLX
// arrays — which is why every existing DFlash forward test is gated behind
// `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` and skipped on the CPU suite. The two
// facts that actually matter for this change are therefore pinned against the
// source text, the same way this suite already pins the harness-twin files:
// that a proposal round cannot reach `draftBlock`, and that the session repays
// the drafter-context debt a skipped `draftBlock` creates. Both are silent
// failures if reverted — the run still produces correct tokens, just from a
// drafter whose cache has drifted off the committed chain — so a compile-time
// signal is worth more here than nothing.

// MARK: - Detection

@Test
func policyPromotesThePeriod18LoopOnlyAfterTwentyThreeCommittedTokens() {
    // A real Gemma 4 stanza loop: 18 distinct positions, two of which repeat
    // values from earlier in the same period, so the suffix is not a trivial
    // ramp.
    let cycle = [
        100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 101, 102,
    ]
    let leadIn = [900, 901]
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 32)

    // One full period plus TWO confirmations is not enough.
    policy.record(
        roundTokens: leadIn + cycle + Array(cycle.prefix(2)),
        proposed: 3,
        accepted: 0,
        terminal: false)
    #expect(policy.installedCycle == nil)

    // The third confirmation installs it.
    policy.record(roundTokens: [cycle[2]], proposed: 3, accepted: 0, terminal: false)

    // The installed cycle is the committed suffix, so it is the period rotated
    // to start at the token that comes NEXT — not the caller's phase of it.
    let nextAligned = Array(cycle[3...] + cycle[..<3])
    #expect(policy.installedCycle == nextAligned)
    #expect(policy.installedCycleOffset == 0)
    #expect(
        policy.nextAction(remaining: 64)
            == .cycle(tokens: Array((cycle + cycle)[3 ..< 34])))
}

@Test
func policyRejectsThePeriod3CoincidenceInsideThePeriod18Loop() {
    let cycle = [
        100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 101, 102,
    ]
    let observed = [900, 901] + cycle + Array(cycle.prefix(3))

    // The trap this stream sets: at lag 3 the last TWO tokens agree, so a
    // two-token confirmation would install a period-3 cycle. The third token
    // disagrees.
    #expect(Array(observed.suffix(2)) == Array(observed.dropLast(3).suffix(2)))
    #expect(Array(observed.suffix(3)) != Array(observed.dropLast(3).suffix(3)))

    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 32)
    policy.record(roundTokens: observed, proposed: 3, accepted: 0, terminal: false)

    #expect(policy.installedCycle == Array(cycle[3...] + cycle[..<3]))
    #expect(policy.installedCycle != Array(observed.suffix(3)))
}

@Test
func policyInstallsNeitherAConstantRunNorAPeriodAboveThirtyTwo() {
    var constant = DFlashCycleProposalPolicy(ordinaryBlockSize: 2, wideBlockSize: 16)
    constant.record(
        roundTokens: Array(repeating: 7, count: 64),
        proposed: 1,
        accepted: 0,
        terminal: false)
    // Fundamental period one. Proposing a block of identical tokens on the
    // evidence of a single distinct value is exactly the case the rule bars.
    #expect(constant.installedCycle == nil)

    var tooLong = DFlashCycleProposalPolicy(ordinaryBlockSize: 2, wideBlockSize: 16)
    let period33 = Array(0 ..< 33)
    tooLong.record(
        roundTokens: period33 + period33,
        proposed: 1,
        accepted: 0,
        terminal: false)
    #expect(tooLong.installedCycle == nil)
}

@Test
func policyDoesNotReadARepeatedTailAsAConstantCycle() {
    // [1, 2, 3, 3] ends in a repeated token but is NOT constant, so the
    // non-constant rule must not reject it.
    let cycle = [1, 2, 3, 3]
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 2, wideBlockSize: 16)
    policy.record(roundTokens: cycle + cycle, proposed: 1, accepted: 0, terminal: false)
    #expect(policy.installedCycle == cycle)
}

@Test
func policyTakesTheShortestConfirmedPeriodAndSpansTheTokensAcrossRounds() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    // Evidence arrives one round at a time, exactly as the session feeds it.
    policy.record(roundTokens: [1, 2, 3, 1, 2], proposed: 3, accepted: 0, terminal: false)
    #expect(policy.installedCycle == nil)
    policy.record(roundTokens: [3], proposed: 3, accepted: 0, terminal: false)
    #expect(policy.installedCycle == [1, 2, 3])
    #expect(policy.nextAction(remaining: 32) == .cycle(tokens: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]))
}

// MARK: - Block sizing

@Test
func policyNarrowsBothArmsToWhatTheRunCanStillCommit() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    // Drafter arm: a round always emits at least the bonus column, so a block
    // never needs to be wider than remaining + 1.
    #expect(policy.nextAction(remaining: 128) == .drafter(blockSize: 4))
    #expect(policy.nextAction(remaining: 2) == .drafter(blockSize: 3))
    #expect(policy.nextAction(remaining: 1) == .drafter(blockSize: 2))

    policy.record(roundTokens: [4, 5, 6, 4, 5, 6], proposed: 3, accepted: 0, terminal: false)
    #expect(policy.installedCycle == [4, 5, 6])
    // Wide arm: the same narrowing, so the tail of a run never verifies a
    // rectangle whose extra columns cannot be committed.
    #expect(policy.nextAction(remaining: 128) == .cycle(tokens: [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]))
    #expect(policy.nextAction(remaining: 4) == .cycle(tokens: [4, 5, 6, 4]))
    #expect(policy.nextAction(remaining: 1) == .cycle(tokens: [4]))
}

// MARK: - Judging a wide round

@Test
func fullyAcceptedWideRoundsWalkTheCycleOffsetForward() {
    let cycle = Array(100 ..< 118)
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 32)
    policy.record(roundTokens: cycle + cycle, proposed: 3, accepted: 0, terminal: false)
    #expect(policy.nextAction(remaining: 128) == .cycle(tokens: Array((cycle + cycle).prefix(31))))

    // 31 proposed tokens all verified, plus the target's own bonus token on
    // top: 32 committed, so the next proposal resumes 32 positions along.
    let committed = Array((cycle + cycle).prefix(32))
    policy.record(roundTokens: committed, proposed: 31, accepted: 31, terminal: false)
    #expect(policy.installedCycleOffset == 32 % 18)
    #expect(policy.nextAction(remaining: 4) == .cycle(tokens: [114, 115, 116, 117]))

    // And again from the updated offset.
    let fromFourteen = Array(cycle[14...] + cycle + cycle)
    policy.record(
        roundTokens: Array(fromFourteen.prefix(32)),
        proposed: 31,
        accepted: 31,
        terminal: false)
    #expect(policy.nextAction(remaining: 4) == .cycle(tokens: [110, 111, 112, 113]))
}

@Test
func aPartiallyAcceptedWideRoundDemotesTheNextRoundToTheDrafter() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    policy.record(roundTokens: [4, 5, 6, 4, 5, 6], proposed: 3, accepted: 0, terminal: false)
    #expect(policy.installedCycle == [4, 5, 6])

    // The target left the loop: one proposed token verified, the rest did not.
    policy.record(roundTokens: [4, 99], proposed: 15, accepted: 1, terminal: false)
    #expect(policy.installedCycle == nil)
    #expect(policy.nextAction(remaining: 128) == .drafter(blockSize: 4))
}

@Test
func aTerminalWideRoundIsNeverReadAsARejection() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    policy.record(roundTokens: [4, 5, 6, 4, 5, 6], proposed: 3, accepted: 0, terminal: false)

    // The run hit its token target (or a stop token) inside the block, so the
    // emission was clipped and `accepted < proposed` says nothing about the
    // cycle. Demoting here would be a verdict on evidence that does not exist.
    policy.record(roundTokens: [4], proposed: 15, accepted: 0, terminal: true)
    #expect(policy.installedCycle == [4, 5, 6])
}

@Test
func reArmingAfterADemotionNeedsFreshEvidenceNotTheDiscardedHistory() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    policy.record(roundTokens: [4, 5, 6, 4, 5, 6], proposed: 3, accepted: 0, terminal: false)
    policy.record(roundTokens: [4, 99], proposed: 15, accepted: 1, terminal: false)
    #expect(policy.installedCycle == nil)

    // The evidence that installed [4, 5, 6] is gone. One more period of the
    // same loop is NOT enough to re-arm, because the detection window now
    // starts at the rejected round.
    policy.record(roundTokens: [4, 5, 6], proposed: 3, accepted: 3, terminal: false)
    #expect(policy.installedCycle == nil)

    // A full period plus three confirmations of freshly committed tokens does
    // re-arm it.
    policy.record(roundTokens: [4, 5, 6], proposed: 3, accepted: 3, terminal: false)
    #expect(policy.installedCycle == [4, 5, 6])
}

@Test
func anEmptyRoundChangesNothing() {
    var policy = DFlashCycleProposalPolicy(ordinaryBlockSize: 4, wideBlockSize: 16)
    policy.record(roundTokens: [1, 2, 3, 1, 2, 3], proposed: 3, accepted: 0, terminal: false)
    let armed = policy
    policy.record(roundTokens: [], proposed: 15, accepted: 0, terminal: false)
    #expect(policy == armed)
}

// MARK: - The round seam and the session's half of the cache contract

private let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()  // Tests/MLXFastTests
    .deletingLastPathComponent()  // Tests
    .deletingLastPathComponent()  // package root

private func sourceText(_ relativePath: String) throws -> String {
    try String(
        contentsOf: repoRoot.appendingPathComponent(relativePath),
        encoding: .utf8)
}

/// Collapse runs of whitespace so these pins survive reformatting but not a
/// change of meaning.
private func normalized(_ text: String) -> String {
    text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
}

@Test
func aProposalRoundCannotReachTheNeuralDrafter() throws {
    let source = try sourceText(
        "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift")

    // Exactly one call site, and it is inside the `else` arm of the proposal
    // branch. Hoisting it back out would make a cycle round pay for a draft
    // block it discards AND advance the drafter cache the session is already
    // accounting for by hand — double-counting the committed frontier.
    let calls = source.components(separatedBy: "drafter.draftBlock(").count - 1
    #expect(calls == 1)

    let branch = try #require(source.range(of: "if let proposal {"))
    let call = try #require(source.range(of: "drafter.draftBlock("))
    #expect(branch.lowerBound < call.lowerBound)
    #expect(source[branch.upperBound ..< call.lowerBound].contains("} else {"))

    // The proposal itself is the block, verbatim, one row wide.
    #expect(
        normalized(source).contains(
            "draftTokens = MLXArray(proposal.map(Int32.init))[.newAxis, .ellipsis]"))
    // A proposal whose length disagrees with the round's block size is
    // refused, not silently verified at a different width.
    #expect(normalized(source).contains("guard proposal.count == blockSize - 1 else {"))
}

@Test
func theSessionRepaysTheDrafterContextASkippedDraftBlockDidNotCache() throws {
    let source = try sourceText(
        "Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift")
    let flat = normalized(source)

    // The whole correctness argument for cycle rounds in one line: a drafter
    // round consumes the pending context, a cycle round adds to it. Drop the
    // append and the drafter silently proposes from a cache that has fallen
    // behind the committed chain — no error, just a collapsed acceptance rate.
    #expect(flat.contains("pendingDraftContext = [round.targetHidden]"))
    #expect(flat.contains("pendingDraftContext.append(round.targetHidden)"))
    #expect(flat.contains("concatenated(pendingDraftContext, axis: 1)"))

    // The round is driven through the shared seam, not a second round
    // function, and the policy is consulted once per round.
    #expect(flat.contains("proposal: proposal)"))
    #expect(source.components(separatedBy: "runDFlashGreedyRound(").count - 1 == 1)
    #expect(source.components(separatedBy: "cycleProposals.nextAction(").count - 1 == 1)

    // Stop tokens are resolved BEFORE the policy trains on the round, so a
    // token the verifier produced past the stop can never become evidence.
    let stopBranch = try #require(source.range(of: "let upToStop = Array(round.tokens[...stopIndex])"))
    let stopRecord = try #require(source.range(of: "roundTokens: upToStop,"))
    #expect(stopBranch.lowerBound < stopRecord.lowerBound)
}
