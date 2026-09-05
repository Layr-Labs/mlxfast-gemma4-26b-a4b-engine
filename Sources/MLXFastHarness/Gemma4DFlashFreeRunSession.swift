import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative

// v1.1 SINGLE-STREAM free-run session for the DFLASH arm.
//
// WHY THIS IS NOT A `RuntimeWorkerFreeRunSession` (the CBv2 one). The
// gemma4-dflash-arm lane (#38) ran the dflash arm through the CBv2 MTP round
// driver by binding the drafter as a `Gemma4CBv2MTPDrafter`. With the REAL
// `DFlashDraftModel` that is not merely suboptimal, it is unrepresentable,
// for three independent reasons visible in the vendored seam
// (`MLXLMCommon/ContinuousBatchingV2/MTP/MTPContractsV2.swift`):
//
//   1. CONDITIONING WIDTH. `CBv2MTPDrafter.draftStep(tokens:hidden:prepared:)`
//      hands the drafter `hidden` = `[B, 1, H]`, the target's pre-norm LAST
//      decoder-layer hidden. A DFlash drafter conditions on the CONCATENATION
//      of the target's hidden states at `dflash_config.target_layer_ids` —
//      `[B, 1, taps * H]`, six taps on the z-lab A4B head. The engine's MTP
//      forward (`CBv2MTPForwardable.cbv2ForwardWithHidden`) captures one
//      hidden, not a tap set, so the value the drafter needs does not exist
//      anywhere on that path. `DFlashDraftModel` refuses the mismatch by
//      construction (`DFlashError.targetHiddenSizeMismatch`).
//   2. DRAFTER KV. The CBv2 seam is explicitly a FROZEN-KV drafter contract:
//      "the DRAFTER writes no KV", it attends a per-round snapshot of the
//      TARGET's KV at two layers (`CBv2MTPRowCapture`). A DFlash drafter owns
//      a real KV cache of its own (`makeCache()`, trimmed each round), which
//      the seam has nowhere to live.
//   3. DRAFT SHAPE. `draftStep` is a CHAINED single-token step. DFlash drafts
//      a whole block in ONE forward over `[bonus, mask, mask, …]`
//      (`draftBlock(bonus:targetHidden:cache:blockSize:)`).
//
// So the draft call is implemented at the seam #38 created instead: this
// session is what the `.dflash` route's `free_decode_begin` opens, and it
// commits through the SAME `free_decode_begin` / `free_decode_run` wire
// surface, the same `RuntimeWorkerFreeRunBuilder`, the same consistency
// triple and the same symmetric early-EOS verdict as the serial and mtp
// legs. benchd's per-stream token-tolerance gate and the 10% budget are
// untouched; nothing about the wire shape moved.
//
// WHAT REMAINS (honest, not hidden):
//   * The BATCHED (cohort) dflash path is refused by name rather than run —
//     see `Gemma4RuntimeCohortDriver`. A batched DFlash round needs batched KV
//     caches on BOTH sides; `DFlashDraftModel.makeBatchedCache(leftPadding:)`
//     was dropped in the port because its `BatchKVCache` /
//     `BatchRotatingKVCache` types are the v1 batching stack upstream deleted
//     at ffede00 and this tree does not carry.
//   * Committed tokens here are the TARGET's own greedy argmax at every
//     emitted position (the drafter only proposes), but the target forward is
//     block-shaped `[1, 1+k]` rather than the serial `[1, 1]`, so this leg is
//     NOT bit-identical to serial the way the CBv2 mtp arm's
//     `.serialTarget`-verified rounds are. That is the ordinary DFlash
//     fidelity posture and is exactly what the per-stream token-tolerance
//     gate exists to price.
//   * Not ported, therefore not run: the fork's verify-path fusions and its
//     sequential / mixed / auto verify selectors.
//
// CYCLE PROPOSALS (2026-09-02). Each round's draft block comes from
// `DFlashCycleProposalPolicy`, which reads only the tokens this run has
// COMMITTED. Normally it says "run the drafter" and this session is what it
// was before. When the committed output has fallen into a repeating cycle —
// the degenerate loop a decoder can enter and never leave — it instead hands
// over the continuation of that cycle as the block, and the session passes it
// to the SAME `runDFlashGreedyRound` through its `proposal:` seam. There is no
// second verifier: the block is checked by the stock rectangular target
// forward, the accept walk is the same walk, and every emitted token is still
// the target's own greedy argmax. A cycle that has ended costs one partially
// rejected round and the policy goes back to the drafter. The only thing a
// cycle round buys is that a wide block costs no drafter forward, so the run
// can verify 16 positions per target forward instead of `depth + 1` while the
// loop lasts.
//
// Nothing here is keyed on the REQUEST's tokens across requests: the policy is
// constructed per session and dies with it, and it only ever reads tokens the
// target committed inside this one free run.

/// The largest draft depth `drafter` can propose per round: its own trained
/// block minus the bonus column — `recommendedBlockSize` already narrows the
/// trained `block_size` to the sliding window when the drafter carries
/// sliding-attention layers — clamped by the engine's configured block
/// ceiling. The single source for the dflash depth ceiling.
func gemma4DFlashMaxDepth(for drafter: DFlashDraftModel) -> Int {
    // The participant's editable draft-depth lever (DFlashDraftModel.swift), the
    // DFlash counterpart of MTP's CBv2MTPRoundDriver.submissionDraftDepth,
    // clamped to the drafter's trained ceiling and the engine ceiling. The pure
    // clamp lives on DFlashDraftModel so it is unit-testable without this model.
    DFlashDraftModel.clampDepth(
        requested: DFlashDraftModel.submissionDraftDepth,
        drafterCeiling: drafter.config.recommendedBlockSize - 1,
        engineCeiling: MLXFastConstants.experimentalDFlashMaxBlockSize - 1)
}

/// Single-stream DFlash free-run session: real `draftBlock` → target verify →
/// accept-walk → KV rollback rounds over the resident Gemma 4 target and the
/// bound z-lab drafter.
final class RuntimeWorkerDFlashFreeRunSession {
    /// The prefill-bonus argmax `free_decode_begin` puts on the wire.
    let seedToken: Int
    /// The draft depth (`blockSize - 1`) this session actually runs. The
    /// `effective_spec` echo reads the SAME resolved value — see
    /// `RuntimeWorkerSpecRegistry.resolveDFlashDepth`.
    let depth: Int

    private let target: Gemma4TextModel
    private let drafter: DFlashDraftModel
    private var targetCache: [KVCache]
    private let draftCache: [KVCache]
    private let promptTokenCount: Int
    private let stopTokens: Set<Int>

    private var bonus: Int
    /// Target tap-hidden context the drafter has NOT yet cached, oldest first.
    ///
    /// On an unmodified DFlash run this is always exactly one element — the
    /// hidden of the tokens the previous round committed — and passing it is
    /// what main does. It becomes a list only because a CYCLE round skips
    /// `draftBlock`, which is the one call that writes drafter context KV (see
    /// the alignment note on `runDFlashGreedyRound`). The skipped rounds'
    /// committed hidden is held here and handed to the next drafter round as
    /// one concatenated context, so the drafter caches the same vectors at the
    /// same positions it would have cached round by round. Nothing is
    /// approximated and nothing is replayed through the drafter twice.
    private var pendingDraftContext: [MLXArray]
    /// Tokens produced past the prompt, INCLUDING the seed — the counter
    /// `runDFlashGreedyRound` uses to locate the drafter cache's committed
    /// frontier (`DFlashTokenIterator.tokenCount` has the same meaning).
    private var generatedTokenCount = 1
    /// Chooses each round's draft source from the committed token stream.
    private var cycleProposals: DFlashCycleProposalPolicy

    private(set) var roundsRun = 0
    private(set) var draftedTotal = 0
    private(set) var acceptedTotal = 0
    /// Rounds whose block came from the cycle policy rather than the drafter.
    /// Diagnostic only; nothing in the round path reads it.
    private(set) var cycleRoundsRun = 0

    /// Block a cycle round proposes into. 16 is the engine's declared DFlash
    /// block ceiling (`MLXFastConstants.experimentalDFlashMaxBlockSize`), the
    /// same rectangle the drafter's own trained `block_size` allows, so a
    /// cycle round verifies a shape the target already runs.
    static let cycleProposalBlockSize = MLXFastConstants.experimentalDFlashMaxBlockSize

    /// Open the window: prefill the seed prompt through the target's DFlash
    /// forward (which captures the tap hidden states in the SAME pass) and
    /// keep the bonus token plus that hidden for the first round.
    init(
        target: Gemma4TextModel,
        drafter: DFlashDraftModel,
        seedTokens: [Int],
        depth: Int,
        stopTokens: Set<Int>
    ) throws {
        guard depth >= 1 else {
            throw MLXFastError.invalidInput(
                "runtime worker dflash free_decode_begin resolved draft depth "
                    + "\(depth); a DFlash round needs at least one proposed token "
                    + "(block size >= 2)")
        }
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "runtime worker dflash free_decode_begin received an empty seed")
        }
        self.target = target
        self.drafter = drafter
        self.depth = depth
        self.stopTokens = stopTokens
        self.promptTokenCount = seedTokens.count

        let targetCache = target.newCache(parameters: nil)
        let draftCache = try drafter.makeCache()
        guard canTrimPromptCache(draftCache) else {
            throw MLXFastError.invalidInput(
                "runtime worker dflash free_decode_begin: the drafter's own KV "
                    + "cache is not trimmable, so a rejected block cannot be "
                    + "rolled back (refusing rather than drifting the drafter's "
                    + "context away from the committed chain)")
        }
        self.targetCache = targetCache
        self.draftCache = draftCache
        // The wide block never narrows below an ordinary round: a cycle
        // proposal that is no wider than the drafter's own block would trade
        // a trained proposal for a guessed one and win nothing.
        self.cycleProposals = DFlashCycleProposalPolicy(
            ordinaryBlockSize: depth + 1,
            wideBlockSize: Swift.max(depth + 1, Self.cycleProposalBlockSize))

        let prompt = MLXArray(seedTokens.map { Int32($0) })[.newAxis, .ellipsis]
        let prefill = try target.forwardGreedyTokensForDFlash(
            prompt,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds)
        let bonusArray = prefill.tokens[0..., -1]
        eval(bonusArray, prefill.targetHidden)
        self.bonus = Int(bonusArray.item(Int32.self))
        self.pendingDraftContext = [prefill.targetHidden]
        self.seedToken = self.bonus
    }

    /// Free-run to `targetN` MORE committed tokens past the seed and assemble
    /// the v1.1 free-run result. Single-shot, same as the CBv2 sessions: the
    /// caller drops the session afterwards.
    func run(targetN: Int) throws -> RuntimeWorkerFreeRunResult {
        var builder = RuntimeWorkerFreeRunBuilder(targetN: targetN)
        while !builder.isComplete {
            let remaining = targetN - builder.committedTotal
            // A round always emits at least the verified bonus column, so the
            // last round only needs `remaining` emissions; narrowing the block
            // there avoids drafting tokens the phase cannot commit. Both arms
            // of the policy narrow the same way.
            let roundBlockSize: Int
            let proposal: [Int]?
            switch cycleProposals.nextAction(remaining: remaining) {
            case .drafter(let size):
                roundBlockSize = size
                proposal = nil
            case .cycle(let tokens):
                roundBlockSize = tokens.count + 1
                proposal = tokens
            }
            guard roundBlockSize >= 2 else {
                throw MLXFastError.invalidInput(
                    "runtime worker dflash free_decode_run: \(remaining) token(s) "
                        + "remain but a DFlash round cannot be narrower than a "
                        + "2-wide block (wiring bug)")
            }
            // A drafter round gets every context vector the drafter has not
            // cached yet; a cycle round calls no drafter, so this argument is
            // unread and the pending list keeps growing. On a run where the
            // policy never fires the list is always one element and this is
            // main's own `targetHidden`, passed unchanged.
            let draftContext =
                pendingDraftContext.count == 1
                ? pendingDraftContext[0]
                : concatenated(pendingDraftContext, axis: 1)
            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: bonus,
                targetHidden: draftContext,
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                blockSize: roundBlockSize,
                maxEmitCount: remaining,
                proposal: proposal)
            guard !round.tokens.isEmpty else {
                throw RuntimeWorkerFreeRunError.emptyRound
            }
            roundsRun += 1
            if proposal != nil { cycleRoundsRun += 1 }
            bonus = round.bonus

            // `roundBlockSize - 1` tokens went in front of the verifier this
            // round; `round.accepted` of them survived the walk. The counters
            // mean "proposed to the target" and are reported the same way for
            // both draft sources, so `accepted_total / drafted_total` stays
            // the acceptance rate of the rectangle that was actually run.
            let proposed = roundBlockSize - 1
            draftedTotal += proposed
            acceptedTotal += round.accepted

            // Symmetric early-EOS, identical verdict to the other two legs: a
            // stop token committed before N makes this leg unusable as a
            // timing sample, and it must be reported the same structured way
            // rather than as a short assembly.
            //
            // The policy is trained on `upToStop`, never on `round.tokens`: a
            // token the verifier produced past the stop was never committed,
            // and a cycle detected on uncommitted tokens would be evidence the
            // run does not have.
            if let stopIndex = round.tokens.firstIndex(where: { stopTokens.contains($0) }) {
                let upToStop = Array(round.tokens[...stopIndex])
                builder.addRound(
                    committedTokens: upToStop,
                    drafted: proposed,
                    accepted: round.accepted)
                generatedTokenCount += upToStop.count
                cycleProposals.record(
                    roundTokens: upToStop,
                    proposed: proposed,
                    accepted: round.accepted,
                    terminal: true)
                if !builder.isComplete {
                    throw RuntimeWorkerFreeRunError.stopTokenBeforeTarget(
                        route: RuntimeWorkerDecodeRoute.dflash.rawValue,
                        token: round.tokens[stopIndex],
                        position: builder.committedTotal,
                        n: targetN)
                }
                break
            }

            builder.addRound(
                committedTokens: round.tokens,
                drafted: proposed,
                accepted: round.accepted)
            generatedTokenCount += round.tokens.count

            // `round.targetHidden` is the accepted prefix's hidden, so its
            // length is exactly this round's committed token count — which is
            // what the drafter cache must advance by. A drafter round has just
            // consumed everything pending; a cycle round adds to the debt and
            // materialises the slice so the unevaluated verify graph behind it
            // is not held alive across the whole loop.
            if proposal == nil {
                pendingDraftContext = [round.targetHidden]
            } else {
                asyncEval(round.targetHidden)
                pendingDraftContext.append(round.targetHidden)
            }

            // `builder.isComplete` is the terminal signal: it is the only way
            // `maxEmitCount` can have clipped this round's emission, and a
            // clipped round reports `accepted < proposed` for a reason that
            // says nothing about whether the cycle still holds.
            cycleProposals.record(
                roundTokens: round.tokens,
                proposed: proposed,
                accepted: round.accepted,
                terminal: builder.isComplete)
        }
        return try builder.finish()
    }
}
