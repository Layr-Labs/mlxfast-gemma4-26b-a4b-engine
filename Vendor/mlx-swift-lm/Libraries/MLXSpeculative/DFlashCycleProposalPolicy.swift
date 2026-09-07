// Copyright © 2026 Apple Inc.

import Foundation

/// Round-source policy for a single-stream DFlash free run: it decides, from
/// the TARGET-COMMITTED token stream alone, whether the next round's block
/// should come from the neural drafter or from a detected output cycle.
///
/// WHY THIS EXISTS. A DFlash round is `[bonus, d1 … dk] -> target verify ->
/// accept walk`. Nothing in that round cares WHERE `d1 … dk` came from: the
/// target's own greedy argmax is what gets committed at every emitted
/// position, and a wrong proposal only costs the rejected tail of one block.
/// So when the committed output has fallen into a repeating cycle -- the
/// failure mode that makes a model emit the same 18-token stanza forever --
/// the cheapest possible drafter is the cycle itself: continuing it costs no
/// forward at all, and the stock rectangular target verify
/// (`DFlashTargetModel.forwardGreedyTokensForDFlash`) accepts or rejects the
/// proposal exactly as it would a drafter block. There is no hand-bound
/// verifier, no second kernel, and no new fidelity posture: a cycle round is
/// an ordinary DFlash round with a different draft source.
///
/// WHAT IT IS NOT. This is not loop SUPPRESSION. The policy never changes a
/// committed token; it only changes what gets proposed. If the target leaves
/// the cycle, the round that proposed the continuation is partially rejected,
/// the target's own token is committed, and the policy hands the next round
/// back to the drafter.
///
/// PURE. No MLX, no environment reads, no clocks, no counters. The session
/// owns all of that; this type is a decision function over the committed
/// token stream so it can be tested exhaustively on a host with no Metal.
///
/// DETECTION RULE (ported from the measured Gemma 4 recurrence policy).
/// Candidate periods are 2 through 32. A period `p` is installed when the
/// last three committed tokens repeat the three tokens `p` positions earlier
/// -- i.e. one full period is present plus three confirming tokens -- and the
/// `p`-token suffix is NOT constant. The shortest such period wins. Period 1
/// is never installed: a run of one repeated token has fundamental period 1
/// and reading it as a cycle would propose a block of identical tokens on the
/// strength of a single distinct value. The 32 ceiling and the three-token
/// confirmation are what keep a period-3 coincidence inside a period-18 loop
/// from being installed as a period-3 cycle (two tokens can agree by accident
/// where three do not).
public struct DFlashCycleProposalPolicy: Equatable, Sendable {
    /// What the session should run for the next round.
    public enum Action: Equatable, Sendable {
        /// Run the neural drafter at this block size (`blockSize - 1`
        /// proposed tokens), exactly as an unmodified DFlash round does.
        case drafter(blockSize: Int)
        /// Skip `draftBlock` and verify these tokens instead. The round's
        /// block size is `tokens.count + 1`.
        case cycle(tokens: [Int])
    }

    /// The longest cycle the policy will install. A period above this is left
    /// to the drafter: the confirmation evidence for a long period is weak
    /// relative to the block a round can propose anyway.
    public static let maximumPeriod = 32
    /// Committed tokens that must repeat the earlier copy before a period is
    /// installed, and again before a demoted policy may re-arm.
    public static let confirmationTokenCount = 3

    /// Block size for an ordinary (drafter) round: the session's resolved
    /// `depth + 1`.
    public let ordinaryBlockSize: Int
    /// Block size a cycle round proposes into when the run has room for it.
    public let wideBlockSize: Int

    private enum Phase: Equatable {
        case drafter
        case cycle(tokens: [Int])
    }

    private var phase: Phase = .drafter
    /// Position inside the installed cycle that the NEXT proposal starts at.
    private var cycleOffset = 0
    /// Bounded suffix of the committed stream, the only thing detection reads.
    /// Capped at `2 * maximumPeriod` because a period `p <= 32` needs at most
    /// `p + 3` tokens of evidence.
    private var recentTokens: [Int] = []

    public init(ordinaryBlockSize: Int, wideBlockSize: Int) {
        precondition(
            ordinaryBlockSize >= 2,
            "a DFlash round cannot be narrower than a 2-wide block")
        precondition(
            wideBlockSize >= ordinaryBlockSize,
            "the wide (cycle) block must be at least as wide as an ordinary "
                + "round, otherwise a cycle round proposes fewer tokens than "
                + "the drafter it replaces")
        self.ordinaryBlockSize = ordinaryBlockSize
        self.wideBlockSize = wideBlockSize
    }

    /// The installed cycle, if the policy is currently proposing one. Read by
    /// tests and by the session's evidence line; not used in the round path.
    public var installedCycle: [Int]? {
        if case .cycle(let tokens) = phase { return tokens }
        return nil
    }

    /// Where in `installedCycle` the next proposal starts.
    public var installedCycleOffset: Int? {
        if case .cycle = phase { return cycleOffset }
        return nil
    }

    /// The next round's draft source. `remaining` is how many MORE tokens the
    /// run still has to commit (>= 1); a round always emits at least the
    /// verified bonus column, so a block never needs to be wider than
    /// `remaining + 1`.
    ///
    /// `mutating` by contract: the round decision belongs to the policy, and
    /// the session calls this exactly once per round before running it. The
    /// current body only reads phase state.
    public mutating func nextAction(remaining: Int) -> Action {
        precondition(
            remaining >= 1,
            "nextAction(remaining:) is only meaningful while the run still "
                + "has a token to commit")
        switch phase {
        case .drafter:
            return .drafter(blockSize: Swift.min(ordinaryBlockSize, remaining + 1))
        case .cycle(let cycle):
            let count = Swift.min(wideBlockSize, remaining + 1) - 1
            return .cycle(
                tokens: (0 ..< count).map { cycle[(cycleOffset + $0) % cycle.count] })
        }
    }

    /// Train on (or judge) one completed round.
    ///
    /// - Parameters:
    ///   - roundTokens: the tokens the round actually COMMITTED, already
    ///     truncated at a stop token by the caller. The policy must never
    ///     train on a token the run did not commit.
    ///   - proposed: draft tokens the round put in front of the verifier
    ///     (`blockSize - 1`).
    ///   - accepted: how many of them the accept walk kept.
    ///   - terminal: the run ended on this round (target count reached, or a
    ///     stop token committed). A terminal round is never read as a
    ///     rejection: a wide block clipped by `maxEmitCount` reports
    ///     `accepted < proposed` for a reason that says nothing about the
    ///     cycle.
    public mutating func record(
        roundTokens: [Int],
        proposed: Int,
        accepted: Int,
        terminal: Bool
    ) {
        guard !roundTokens.isEmpty else { return }

        if case .cycle(let cycle) = phase {
            guard accepted == proposed else {
                guard !terminal else { return }
                // The target left the cycle. Hand the next round back to the
                // drafter and throw the detection history away, so re-arming
                // needs a full period plus three confirmations of FRESH
                // committed tokens rather than the stale evidence that
                // installed the cycle we just lost.
                phase = .drafter
                cycleOffset = 0
                recentTokens = Array(roundTokens.suffix(Self.maximumPeriod * 2))
                return
            }
            // Every proposed token verified, and the target added its own
            // bonus token on top; the next proposal continues from there.
            cycleOffset = (cycleOffset + roundTokens.count) % cycle.count
            return
        }

        recentTokens.append(contentsOf: roundTokens)
        if recentTokens.count > Self.maximumPeriod * 2 {
            recentTokens.removeFirst(recentTokens.count - Self.maximumPeriod * 2)
        }

        let upper = Swift.min(
            Self.maximumPeriod,
            recentTokens.count - Self.confirmationTokenCount)
        guard upper >= 2 else { return }
        let confirmationStart = recentTokens.count - Self.confirmationTokenCount
        for period in 2 ... upper {
            let priorStart = confirmationStart - period
            guard
                recentTokens[confirmationStart ..< recentTokens.count]
                    .elementsEqual(
                        recentTokens[
                            priorStart ..< priorStart + Self.confirmationTokenCount])
            else { continue }
            let cycle = Array(recentTokens.suffix(period))
            // A constant suffix has fundamental period one. Skip it and keep
            // looking, so a longer non-constant period can still be found in
            // the same pass.
            guard cycle.contains(where: { $0 != cycle[0] }) else { continue }
            phase = .cycle(tokens: cycle)
            cycleOffset = 0
            return
        }
    }
}
