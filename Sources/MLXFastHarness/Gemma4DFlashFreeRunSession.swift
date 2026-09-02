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

/// Session-owned width selection for the measured Gemma 4 DFlash crossover.
/// D7 starts on the installed C4 verifier route, promotes only after two fully
/// accepted C4 rounds, and returns after a partial C8 round. D15 uses its
/// separately construction-bound C4/C16 recurrence loop. Other requested
/// depths remain fixed controls.
struct Gemma4DFlashWidthPolicy {
    let requestedDepth: Int
    private(set) var currentDepth: Int
    private var consecutiveFullC4Rounds = 0

    init(requestedDepth: Int) {
        self.requestedDepth = requestedDepth
        self.currentDepth = requestedDepth == 7 || requestedDepth == 15 ? 3 : requestedDepth
    }

    mutating func record(
        roundBlockSize: Int,
        maxEmitCount: Int,
        committed: Int
    ) {
        let fullWidth = currentDepth + 1
        guard requestedDepth == 7,
            roundBlockSize == fullWidth,
            maxEmitCount >= fullWidth
        else { return }

        let fullyAccepted = committed == fullWidth
        if currentDepth == 3 {
            consecutiveFullC4Rounds = fullyAccepted ? consecutiveFullC4Rounds + 1 : 0
            if consecutiveFullC4Rounds == 2 {
                currentDepth = 7
                consecutiveFullC4Rounds = 0
            }
        } else if !fullyAccepted {
            currentDepth = 3
            consecutiveFullC4Rounds = 0
        }
    }
}

/// Construction-installed D15 proposal source. The phase itself is the only
/// measured-loop route decision: all metadata and fixed verifier widths are
/// proven before the session begins.
enum Gemma4DFlashProposalPhase: Equatable {
    case exactDFlashC4
    case periodicExactWide(cycle: [Int])
}

enum Gemma4DFlashRecurrenceAction: Equatable {
    case none
    case replayWideDrafterAndDemote
}

/// Target-token-derived recurrence policy for the D15 single-stream lane.
/// Detection requires one complete copy plus three target-confirmed tokens of
/// the shortest period in the bounded committed-token suffix. Period one and
/// periods above 32 are never installed. A wide rejection demotes the
/// following round to exact C4.
struct Gemma4DFlashRecurrencePolicy {
    let exactVerifierWidth: Int
    let verifierWidth: Int
    private(set) var phase: Gemma4DFlashProposalPhase = .exactDFlashC4
    private var recentTokens: [Int] = []
    private var cycleOffset = 0
    private let maximumPeriod = 32
    private let confirmationTokenCount = 3

    init(exactVerifierWidth: Int, verifierWidth: Int) {
        precondition(exactVerifierWidth >= 2)
        precondition(verifierWidth >= 2)
        precondition(exactVerifierWidth <= verifierWidth)
        self.exactVerifierWidth = exactVerifierWidth
        self.verifierWidth = verifierWidth
    }

    func verifierBlockSize(remaining: Int) -> Int {
        switch phase {
        case .exactDFlashC4:
            // D15 installs exactly C4 and C16 target forwards. Preserve the
            // physical C4 graph for a short terminal tail; maxEmitCount caps
            // how many verified tokens are committed from that rectangle.
            return exactVerifierWidth
        case .periodicExactWide:
            return verifierWidth
        }
    }

    func makeDraftTokens(count: Int) -> [Int]? {
        guard count > 0 else { return nil }
        guard case .periodicExactWide(let cycle) = phase else { return nil }
        return (0 ..< count).map {
            cycle[(cycleOffset + $0) % cycle.count]
        }
    }

    @discardableResult
    mutating func record(
        committedTokens: [Int],
        accepted: Int,
        proposed: Int,
        isTerminalRound: Bool = false
    ) -> Gemma4DFlashRecurrenceAction {
        guard !committedTokens.isEmpty else { return .none }
        if case .periodicExactWide(let cycle) = phase {
            guard accepted == proposed else {
                guard !isTerminalRound else { return .none }
                phase = .exactDFlashC4
                cycleOffset = 0
                recentTokens = Array(
                    committedTokens.suffix(maximumPeriod * 2))
                return .replayWideDrafterAndDemote
            }
            cycleOffset = (cycleOffset + committedTokens.count) % cycle.count
            return .none
        }

        recentTokens.append(contentsOf: committedTokens)
        if recentTokens.count > maximumPeriod * 2 {
            recentTokens.removeFirst(recentTokens.count - maximumPeriod * 2)
        }

        let upper = Swift.min(
            maximumPeriod,
            recentTokens.count - confirmationTokenCount)
        guard upper >= 2 else { return .none }
        for period in 2 ... upper {
            let confirmationStart = recentTokens.count - confirmationTokenCount
            let priorStart = confirmationStart - period
            guard recentTokens[
                confirmationStart ..< recentTokens.count
            ].elementsEqual(
                recentTokens[
                    priorStart ..< priorStart + confirmationTokenCount])
            else {
                continue
            }
            let cycle = Array(recentTokens.suffix(period))
            // A constant cycle has fundamental period one; continue looking
            // only so a later nonconstant suffix can prove a valid period.
            guard cycle.contains(where: { $0 != cycle[0] }) else { continue }
            phase = .periodicExactWide(cycle: cycle)
            cycleOffset = 0
            return .none
        }
        return .none
    }
}

private enum Gemma4DFlashWideReplayContext {
    case projected(MLXArray)
    case targetHidden(MLXArray)
}

struct Gemma4DFlashReplayFrame<Context> {
    let bonus: Int
    let context: Context
    let blockSize: Int
    let generatedTokenCountBeforeRound: Int
}

struct Gemma4DFlashReplayStep<Context> {
    let bonus: Int
    let context: Context
    let blockSize: Int
    let committedDraftOffset: Int
}

func gemma4DFlashReplayPlan<Context>(
    frames: [Gemma4DFlashReplayFrame<Context>],
    promptTokenCount: Int,
    action: Gemma4DFlashRecurrenceAction
) -> [Gemma4DFlashReplayStep<Context>] {
    guard action == .replayWideDrafterAndDemote else { return [] }
    return frames.map { frame in
        Gemma4DFlashReplayStep(
            bonus: frame.bonus,
            context: frame.context,
            blockSize: frame.blockSize,
            committedDraftOffset: Swift.max(
                0,
                promptTokenCount
                    + frame.generatedTokenCountBeforeRound - 1))
    }
}

private typealias Gemma4DFlashWideReplayFrame =
    Gemma4DFlashReplayFrame<Gemma4DFlashWideReplayContext>
private typealias Gemma4DFlashWideReplayStep =
    Gemma4DFlashReplayStep<Gemma4DFlashWideReplayContext>

/// Holds construction ownership of newly allocated row state until every
/// failable cache-bank preparation step has completed. A successful return
/// transfers that ownership to the published cache object's `deinit`.
func withDFlashCBv2RowStateOwnership<State, Prepared>(
    _ state: State,
    release: (State) -> Void,
    prepare: (State) throws -> Prepared
) rethrows -> Prepared {
    do {
        return try prepare(state)
    } catch {
        release(state)
        throw error
    }
}

/// One construction-certified, physical-B1 contiguous target-cache stack for
/// the DFlash session. Prefill binds and fills it once. Decode then uses the
/// row objects' speculative-write transaction directly; the enabled path has
/// no legacy-cache eligibility check and no fallback route.
final class Gemma4DFlashCBv2TargetCache: DFlashTargetCacheRoundTransaction {
    private let backend: CBv2ContiguousKVBackend
    private let bank: CBv2LayerCacheBank
    private let rowState: [CBv2SequenceKV?]
    private let storageRows: [CBv2SequenceKV]
    private let certifiedRectangularVerification: Bool
    private let pendingCertifiedInstaller: () throws -> ((MLXArray) ->
        DFlashGreedyTargetForward)
    private var certifiedForward: (MLXArray) -> DFlashGreedyTargetForward

    let cache: [KVCache]

    init(
        target: Gemma4TextModel,
        promptTokenCount: Int,
        maxLength: Int,
        targetLayerIds: [Int] = [],
        requiredVerifierColumns: Set<Int> = [],
        certifiedRectangularVerification: Bool = false
    ) throws {
        let layerKinds = target.cbv2LayerKinds
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: cohortContiguousKVBytesBudget(
                    layerKinds: layerKinds,
                    batchSize: 1,
                    maxSequenceLength: maxLength)))
        let rowState = try backend.makeSequenceState(
            layerKinds: layerKinds,
            promptLength: promptTokenCount,
            maxLength: maxLength)
        let prepared = try withDFlashCBv2RowStateOwnership(
            rowState,
            release: { backend.release($0) },
            prepare: { rowState in
                let unboundCaches = try target.newCacheV2 { index, kind in
                    CBv2LayerCache(layerIndex: index, kind: kind)
                }
                let bank = CBv2LayerCacheBank(caches: unboundCaches)
                guard !certifiedRectangularVerification
                    || bank.supportsCertifiedMTPRectangularVerification
                else {
                    throw MLXFastError.invalidInput(
                        "runtime worker dflash certified rectangular verification "
                            + "requires a construction-certified CBv2 attention bank")
                }
                let layerCaches = bank.layerCaches(rowStates: [rowState])
                let storageRows = rowState.compactMap { $0 }
                guard !storageRows.isEmpty,
                    storageRows.allSatisfy(\.supportsSpeculativeWrites)
                else {
                    throw MLXFastError.invalidInput(
                        "runtime worker dflash target cache cannot guarantee exact "
                            + "speculative rollback for every owning layer")
                }
                let cache = try layerCaches.map { layer -> KVCache in
                    guard let cache = layer as? KVCache else {
                        throw MLXFastError.invalidInput(
                            "runtime worker dflash CBv2 layer cache does not conform "
                                + "to the Gemma target KVCache forward surface")
                    }
                    return cache
                }
                let prefillOnlyForward: (MLXArray) -> DFlashGreedyTargetForward = { _ in
                    preconditionFailure(
                        "prefill-only DFlash CBv2 cache cannot verify a round")
                }
                let pendingCertifiedInstaller: () throws ->
                    ((MLXArray) -> DFlashGreedyTargetForward)
                if requiredVerifierColumns.isEmpty {
                    pendingCertifiedInstaller = { prefillOnlyForward }
                } else {
                    guard requiredVerifierColumns == Set([4, 16]) else {
                        throw MLXFastError.invalidInput(
                            "certified DFlash CBv2 cache requires exactly C4 and C16")
                    }
                    pendingCertifiedInstaller = {
                        let c4 = try target.bindCertifiedDFlashGreedyForward(
                            cache: cache,
                            targetLayerIds: targetLayerIds,
                            columns: 4)
                        let c16 = try target.bindCertifiedDFlashGreedyForward(
                            cache: cache,
                            targetLayerIds: targetLayerIds,
                            columns: 16)
                        return { input in
                            if input.dim(1) == 4 {
                                return c4(input)
                            }
                            return c16(input)
                        }
                    }
                }
                return (
                    bank: bank,
                    storageRows: storageRows,
                    cache: cache,
                    prefillOnlyForward: prefillOnlyForward,
                    pendingCertifiedInstaller: pendingCertifiedInstaller)
            })

        self.backend = backend
        self.bank = prepared.bank
        self.rowState = rowState
        self.storageRows = prepared.storageRows
        self.certifiedRectangularVerification = certifiedRectangularVerification
        self.pendingCertifiedInstaller = prepared.pendingCertifiedInstaller
        self.certifiedForward = prepared.prefillOnlyForward
        self.cache = prepared.cache
    }

    deinit {
        backend.release(rowState)
    }

    /// Commit the prompt graph before enabling the construction-certified
    /// rectangular attention lane. Decode then leaves that installed mode on
    /// for the lifetime of this cache rather than toggling it per round.
    func finishPrefill(evaluating outputs: [MLXArray]) throws {
        eval(outputs + evaluationRoots)
        let installed = try pendingCertifiedInstaller()
        certifiedForward = installed
        if certifiedRectangularVerification {
            bank.setCertifiedMTPRectangularVerification(true)
        }
    }

    var evaluationRoots: [MLXArray] {
        cache.flatMap { $0.innerState() }
    }

    var storageOffsets: [Int] {
        storageRows.map(\.absoluteOffset)
    }

    func committedSnapshots() -> [(
        keys: MLXArray, values: MLXArray, offset: Int
    )] {
        storageRows.map { row in
            let snapshot = row.snapshot()
            return (snapshot.keys, snapshot.values, snapshot.offset)
        }
    }

    func beginRound() {
        for row in storageRows {
            row.beginSpeculativeWrite()
        }
    }

    func forwardGreedy(
        target: any DFlashTargetModel,
        verifyInput: MLXArray,
        cache: [KVCache],
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        return certifiedForward(verifyInput)
    }

    func evaluationRoots(cache: [KVCache]) -> [MLXArray] {
        evaluationRoots
    }

    func finishRound(
        target: any DFlashTargetModel,
        cache: inout [KVCache],
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        if rejectedTokenCount > 0 {
            for row in storageRows {
                row.rollback(rejectedTokenCount)
            }
        }
        for row in storageRows {
            row.commitSpeculativeWrite()
        }
        if rejectedTokenCount > 0 {
            // The rectangular graph advanced the shared on-device position
            // chain through every provisional column. Rebind once after a
            // rollback so the next graph starts from the host-authoritative
            // committed offsets.
            bank.invalidateBoundComposition()
            _ = bank.layerCaches(rowStates: [rowState])
        }
        return verifiedTargetHidden[0..., 0 ..< acceptedTokenCount + 1, 0...]
    }
}

/// Persistent whole-verifier compile bank for the physical-B1 DFlash lane.
///
/// A session still prefills through the proven contiguous CBv2 cache. At the
/// construction boundary its committed K/V is copied into these fixed-shape
/// caches. Their MLXArray identities survive across sessions, so the worker's
/// discarded warmup sample compiles each used width once and measured samples
/// reuse the same graph and state objects.
final class Gemma4DFlashCompiledVerifierBank: DFlashTargetCacheRoundTransaction {
    private let maxLength: Int
    private let targetLayerIds: [Int]
    private let requiredVerifierColumns: Set<Int>
    private let fixedCaches: [CompilableKVCache]
    private let forwards: [Int: Gemma4DFlashGreedyForwardBinding]

    let cache: [KVCache]

    init(
        target: Gemma4TextModel,
        prefill: Gemma4DFlashCBv2TargetCache,
        maxLength: Int,
        targetLayerIds: [Int],
        requiredVerifierColumns: Set<Int>
    ) throws {
        let snapshots = prefill.committedSnapshots()
        precondition(
            snapshots.count == target.dFlashLayerCount,
            "compiled DFlash bank requires one storage-owning row per target layer")
        let fixedCaches = snapshots.map { snapshot -> CompilableKVCache in
            let cache = CompilableKVCache(maxLength: maxLength)
            cache.installCommittedPrefix(
                keys: snapshot.keys,
                values: snapshot.values,
                offset: snapshot.offset)
            return cache
        }
        eval(fixedCaches.flatMap { $0.innerState() })
        let cache: [KVCache] = fixedCaches

        var forwards: [Int: Gemma4DFlashGreedyForwardBinding] = [:]
        for columns in requiredVerifierColumns.sorted() {
            forwards[columns] = try target.bindCompiledDFlashGreedyForward(
                cache: cache,
                targetLayerIds: targetLayerIds,
                columns: columns)
        }

        self.maxLength = maxLength
        self.targetLayerIds = targetLayerIds
        self.requiredVerifierColumns = requiredVerifierColumns
        self.fixedCaches = fixedCaches
        self.cache = cache
        self.forwards = forwards
    }

    func supports(
        maxLength: Int,
        targetLayerIds: [Int],
        requiredVerifierColumns: Set<Int>
    ) -> Bool {
        self.maxLength == maxLength
            && self.targetLayerIds == targetLayerIds
            && self.requiredVerifierColumns == requiredVerifierColumns
    }

    func installPrefillState(from prefill: Gemma4DFlashCBv2TargetCache) {
        let snapshots = prefill.committedSnapshots()
        precondition(snapshots.count == fixedCaches.count)
        for (cache, snapshot) in zip(fixedCaches, snapshots) {
            cache.installCommittedPrefix(
                keys: snapshot.keys,
                values: snapshot.values,
                offset: snapshot.offset)
        }
        eval(fixedCaches.flatMap { $0.innerState() })
    }

    func beginRound() {}

    func forwardGreedy(
        target: any DFlashTargetModel,
        verifyInput: MLXArray,
        cache: [KVCache],
        targetLayerIds: [Int]
    ) throws -> DFlashGreedyTargetForward {
        let columns = verifyInput.dim(1)
        guard let forward = forwards[columns] else {
            preconditionFailure("compiled DFlash bank has no C\(columns) route")
        }
        return forward(verifyInput)
    }

    func evaluationRoots(cache: [KVCache]) -> [MLXArray] {
        fixedCaches.flatMap { $0.innerState() }
    }

    func finishRound(
        target: any DFlashTargetModel,
        cache: inout [KVCache],
        verifyInput: MLXArray,
        acceptedTokenCount: Int,
        rejectedTokenCount: Int,
        targetLayerIds: [Int],
        verifiedTargetHidden: MLXArray
    ) throws -> MLXArray {
        if rejectedTokenCount > 0 {
            for cache in fixedCaches {
                cache.rewindCompiledOffset(by: rejectedTokenCount)
            }
        }
        return verifiedTargetHidden[0..., 0 ..< acceptedTokenCount + 1, 0...]
    }
}

/// Single-stream DFlash free-run session: real `draftBlock` → target verify →
/// accept-walk → KV rollback rounds over the resident Gemma 4 target and the
/// bound z-lab drafter.
final class RuntimeWorkerDFlashFreeRunSession {
    static let d15FidelityVerifierWidth = 16
    static let d15ExactC4VerifierWidth = 4
    static let d15RequiredVerifierColumns: Set<Int> = [4, 16]

    static func targetCacheMaximumTokens(
        promptTokenCount: Int,
        depth: Int
    ) -> Int {
        let provisionalTail = depth == 15
            ? d15FidelityVerifierWidth - 1
            : 1
        return promptTokenCount
            + MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
            + provisionalTail
    }

    static func inheritedCompiledVerifierBank<Bank>(_ bank: Bank?) -> Bank? {
        bank
    }

    /// The prefill-bonus argmax `free_decode_begin` puts on the wire.
    let seedToken: Int
    /// The requested draft-depth ceiling. Fixed controls run this depth; D7
    /// uses C4-to-C8, and D15 retains its logical echo while running the
    /// construction-installed exact C4 and periodic C16 verifiers.
    /// The `effective_spec` echo reads this SAME resolved ceiling — see
    /// `RuntimeWorkerSpecRegistry.resolveDFlashDepth`.
    let depth: Int

    private let target: Gemma4TextModel
    private let drafter: DFlashDraftModel
    private var prefillTargetCacheTransaction: Gemma4DFlashCBv2TargetCache?
    private let reusableCompiledVerifierBank: Gemma4DFlashCompiledVerifierBank?
    private var targetCache: [KVCache] = []
    private let draftCache: [KVCache]
    private let promptTokenCount: Int
    private let stopTokens: Set<Int>
    private var widthPolicy: Gemma4DFlashWidthPolicy
    private var proposalPhasePolicy: Gemma4DFlashRecurrencePolicy
    private var wideReplayFrames: [Gemma4DFlashWideReplayFrame] = []
    private var priorWideTargetHidden: MLXArray?

    private var bonus: Int
    private var draftContext: MLXArray
    /// Tokens produced past the prompt, INCLUDING the seed — the counter
    /// `runDFlashGreedyRound` uses to locate the drafter cache's committed
    /// frontier (`DFlashTokenIterator.tokenCount` has the same meaning).
    private var generatedTokenCount = 1

    /// Retained by worker state after this session is consumed so later
    /// samples reuse its compiled graphs and stable cache-array identities.
    private(set) var compiledVerifierBank: Gemma4DFlashCompiledVerifierBank?
    private(set) var maximumPhysicalVerifierWidth: Int

    private(set) var roundsRun = 0
    private(set) var draftedTotal = 0
    private(set) var acceptedTotal = 0

    /// Open the window: prefill the seed prompt through the target's DFlash
    /// forward (which captures the tap hidden states in the SAME pass) and
    /// keep the bonus token plus that hidden for the first round.
    init(
        target: Gemma4TextModel,
        drafter: DFlashDraftModel,
        seedTokens: [Int],
        depth: Int,
        stopTokens: Set<Int>,
        compiledVerifierBank existingCompiledVerifierBank:
            Gemma4DFlashCompiledVerifierBank? = nil
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
        self.widthPolicy = Gemma4DFlashWidthPolicy(requestedDepth: depth)
        self.proposalPhasePolicy = Gemma4DFlashRecurrencePolicy(
            exactVerifierWidth: Self.d15ExactC4VerifierWidth,
            verifierWidth: Self.d15FidelityVerifierWidth)
        let maximumPhysicalVerifierWidth =
            depth == 15 ? Self.d15FidelityVerifierWidth : depth + 1
        self.maximumPhysicalVerifierWidth = maximumPhysicalVerifierWidth

        let prefillCacheTransaction = try Gemma4DFlashCBv2TargetCache(
            target: target,
            promptTokenCount: seedTokens.count,
            maxLength: Self.targetCacheMaximumTokens(
                promptTokenCount: seedTokens.count,
                depth: depth),
            targetLayerIds: drafter.config.targetLayerIds,
            requiredVerifierColumns: depth == 15
                ? Self.d15RequiredVerifierColumns : [],
            certifiedRectangularVerification: depth == 15)
        let prefillTargetCache = prefillCacheTransaction.cache
        let draftCache = try drafter.makeCache()
        guard canTrimPromptCache(draftCache) else {
            throw MLXFastError.invalidInput(
                "runtime worker dflash free_decode_begin: the drafter's own KV "
                    + "cache is not trimmable, so a rejected block cannot be "
                    + "rolled back (refusing rather than drifting the drafter's "
                    + "context away from the committed chain)")
        }
        let prompt = MLXArray(seedTokens.map { Int32($0) })[.newAxis, .ellipsis]
        let prefill = try target.forwardGreedyTokensForDFlash(
            prompt,
            cache: prefillTargetCache,
            targetLayerIds: drafter.config.targetLayerIds)
        let bonusArray = prefill.tokens[0..., -1]
        let draftContext = try drafter.projectTargetHidden(prefill.targetHidden)
        try prefillCacheTransaction.finishPrefill(
            evaluating: [bonusArray, draftContext])

        self.prefillTargetCacheTransaction = prefillCacheTransaction
        self.reusableCompiledVerifierBank = existingCompiledVerifierBank
        self.compiledVerifierBank = Self.inheritedCompiledVerifierBank(
            existingCompiledVerifierBank)
        self.draftCache = draftCache
        self.bonus = Int(bonusArray.item(Int32.self))
        self.draftContext = draftContext
        self.seedToken = self.bonus
    }

    /// Free-run to `targetN` MORE committed tokens past the seed and assemble
    /// the v1.1 free-run result. Single-shot, same as the CBv2 sessions: the
    /// caller drops the session afterwards.
    func run(targetN: Int) throws -> RuntimeWorkerFreeRunResult {
        guard let prefillTargetCacheTransaction else {
            preconditionFailure("DFlash compiled verifier was prepared twice")
        }
        if depth == 15 {
            self.targetCache = prefillTargetCacheTransaction.cache
            self.prefillTargetCacheTransaction = nil
            return try runD15ProposalRounds(
                targetN: targetN,
                targetCacheTransaction: prefillTargetCacheTransaction)
        }

        let compiledMaxLength = promptTokenCount + targetN + 1
        let requiredVerifierColumns = Set(
            2 ... MLXFastConstants.experimentalDFlashMaxBlockSize)
        let targetCacheTransaction: Gemma4DFlashCompiledVerifierBank
        if let reusableCompiledVerifierBank,
            reusableCompiledVerifierBank.supports(
                maxLength: compiledMaxLength,
                targetLayerIds: drafter.config.targetLayerIds,
                requiredVerifierColumns: requiredVerifierColumns)
        {
            reusableCompiledVerifierBank.installPrefillState(
                from: prefillTargetCacheTransaction)
            targetCacheTransaction = reusableCompiledVerifierBank
        } else {
            targetCacheTransaction = try Gemma4DFlashCompiledVerifierBank(
                target: target,
                prefill: prefillTargetCacheTransaction,
                maxLength: compiledMaxLength,
                targetLayerIds: drafter.config.targetLayerIds,
                requiredVerifierColumns: requiredVerifierColumns)
        }
        self.compiledVerifierBank = targetCacheTransaction
        self.targetCache = targetCacheTransaction.cache
        self.prefillTargetCacheTransaction = nil

        return try runStandardRounds(
            targetN: targetN,
            targetCacheTransaction: targetCacheTransaction)
    }

    /// The unchanged fixed/adaptive DFlash loop. D1-D14 never execute or
    /// inspect the separately installed D15 exact-C4 path.
    private func runStandardRounds(
        targetN: Int,
        targetCacheTransaction: Gemma4DFlashCompiledVerifierBank
    ) throws -> RuntimeWorkerFreeRunResult {
        var builder = RuntimeWorkerFreeRunBuilder(targetN: targetN)
        while !builder.isComplete {
            let remaining = targetN - builder.committedTotal
            let roundBlockSize = Swift.min(widthPolicy.currentDepth + 1, remaining + 1)
            guard roundBlockSize >= 2 else {
                throw MLXFastError.invalidInput(
                    "runtime worker dflash free_decode_run: \(remaining) token(s) "
                        + "remain but a DFlash round cannot be narrower than a "
                        + "2-wide block (wiring bug)")
            }
            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: bonus,
                projectedContext: draftContext,
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                blockSize: roundBlockSize,
                maxEmitCount: remaining,
                targetCacheTransaction: targetCacheTransaction)
            guard !round.tokens.isEmpty else {
                throw RuntimeWorkerFreeRunError.emptyRound
            }
            roundsRun += 1
            bonus = round.bonus
            draftContext = try drafter.projectTargetHidden(round.targetHidden)

            let proposed = roundBlockSize - 1
            draftedTotal += proposed
            acceptedTotal += round.accepted
            widthPolicy.record(
                roundBlockSize: roundBlockSize,
                maxEmitCount: remaining,
                committed: round.tokens.count)

            if let stopIndex = round.tokens.firstIndex(where: { stopTokens.contains($0) }) {
                let upToStop = Array(round.tokens[...stopIndex])
                builder.addRound(
                    committedTokens: upToStop,
                    drafted: proposed,
                    accepted: round.accepted)
                generatedTokenCount += upToStop.count
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
        }
        return try builder.finish()
    }

    /// The logical-D15 lane starts on exact C4 DFlash. Once two complete
    /// copies prove a bounded recurrence, it switches to the single wider
    /// verifier installed at construction. A rejected wide proposal demotes
    /// the following round to exact C4; neither route falls back in-round.
    private func runD15ProposalRounds(
        targetN: Int,
        targetCacheTransaction: Gemma4DFlashCBv2TargetCache
    ) throws -> RuntimeWorkerFreeRunResult {
        var builder = RuntimeWorkerFreeRunBuilder(targetN: targetN)
        while !builder.isComplete {
            let remaining = targetN - builder.committedTotal
            let phase = proposalPhasePolicy.phase
            let roundBlockSize = proposalPhasePolicy.verifierBlockSize(
                remaining: remaining)
            guard roundBlockSize >= 2 else {
                throw MLXFastError.invalidInput(
                    "runtime worker dflash free_decode_run: \(remaining) token(s) "
                        + "remain but a DFlash round cannot be narrower than a "
                        + "2-wide block (wiring bug)")
            }
            let round: DFlashGreedyRoundResult
            let wideReplayFrame: Gemma4DFlashWideReplayFrame?
            switch phase {
            case .exactDFlashC4:
                wideReplayFrame = nil
                round = try runDFlashGreedyRound(
                    target: target,
                    drafter: drafter,
                    targetCache: &targetCache,
                    draftCache: draftCache,
                    bonus: bonus,
                    projectedContext: draftContext,
                    promptTokenCount: promptTokenCount,
                    generatedTokenCount: generatedTokenCount,
                    blockSize: roundBlockSize,
                    maxEmitCount: remaining,
                    targetCacheTransaction: targetCacheTransaction)
            case .periodicExactWide:
                let replayContext: Gemma4DFlashWideReplayContext
                if let priorWideTargetHidden {
                    replayContext = .targetHidden(priorWideTargetHidden)
                } else {
                    replayContext = .projected(draftContext)
                }
                wideReplayFrame = Gemma4DFlashWideReplayFrame(
                    bonus: bonus,
                    context: replayContext,
                    blockSize: roundBlockSize,
                    generatedTokenCountBeforeRound: generatedTokenCount)
                let proposal = proposalPhasePolicy.makeDraftTokens(
                    count: roundBlockSize - 1)!
                round = try runDFlashGreedyProposalRound(
                    target: target,
                    targetCache: &targetCache,
                    bonus: bonus,
                    draftTokens: MLXArray(proposal.map(Int32.init))[
                        .newAxis, .ellipsis],
                    targetLayerIds: drafter.config.targetLayerIds,
                    blockSize: roundBlockSize,
                    maxEmitCount: remaining,
                    targetCacheTransaction: targetCacheTransaction)
            }
            guard !round.tokens.isEmpty else {
                throw RuntimeWorkerFreeRunError.emptyRound
            }
            roundsRun += 1
            let proposed = roundBlockSize - 1
            draftedTotal += proposed
            acceptedTotal += round.accepted

            // Stop is a hard committed-chain boundary. Neither post-stop
            // tokens nor their hidden state may train recurrence or mutate
            // lazy replay state before the structured early-stop verdict.
            if let stopIndex = round.tokens.firstIndex(where: { stopTokens.contains($0) }) {
                let upToStop = Array(round.tokens[...stopIndex])
                builder.addRound(
                    committedTokens: upToStop,
                    drafted: proposed,
                    accepted: round.accepted)
                generatedTokenCount += upToStop.count
                if !builder.isComplete {
                    throw RuntimeWorkerFreeRunError.stopTokenBeforeTarget(
                        route: RuntimeWorkerDecodeRoute.dflash.rawValue,
                        token: round.tokens[stopIndex],
                        position: builder.committedTotal,
                        n: targetN)
                }
                break
            }

            bonus = round.bonus
            let isTerminalRound = round.tokens.count == remaining
            if case .exactDFlashC4 = phase {
                draftContext = try drafter.projectTargetHidden(
                    round.targetHidden)
            } else if !isTerminalRound, let wideReplayFrame {
                wideReplayFrames.append(wideReplayFrame)
                priorWideTargetHidden = round.targetHidden
            }

            let recurrenceAction = proposalPhasePolicy.record(
                committedTokens: round.tokens,
                accepted: round.accepted,
                proposed: proposed,
                isTerminalRound: isTerminalRound)
            let replayPlan = gemma4DFlashReplayPlan(
                frames: wideReplayFrames,
                promptTokenCount: promptTokenCount,
                action: recurrenceAction)
            switch recurrenceAction {
            case .none:
                break
            case .replayWideDrafterAndDemote:
                try replaySkippedWideDrafterBlocks(replayPlan)
                draftContext = try drafter.projectTargetHidden(
                    round.targetHidden)
                wideReplayFrames.removeAll(keepingCapacity: true)
                priorWideTargetHidden = nil
            }

            builder.addRound(
                committedTokens: round.tokens,
                drafted: proposed,
                accepted: round.accepted)
            generatedTokenCount += round.tokens.count
        }
        return try builder.finish()
    }

    /// Lazily materialize the drafter KV work skipped by synthetic wide
    /// proposal rounds only when a nonterminal rejection needs exact C4 next.
    /// Each frame owns the context that existed before its target round, so
    /// replay preserves round order without projecting hidden state eagerly.
    private func replaySkippedWideDrafterBlocks(
        _ plan: [Gemma4DFlashWideReplayStep]
    ) throws {
        for step in plan {
            let projectedContext: MLXArray
            switch step.context {
            case .projected(let context):
                projectedContext = context
            case .targetHidden(let hidden):
                projectedContext = try drafter.projectTargetHidden(hidden)
            }
            let replayedDraftTokens = try drafter.draftBlock(
                bonus: step.bonus,
                projectedContext: projectedContext,
                cache: draftCache,
                blockSize: step.blockSize)
            eval(replayedDraftTokens)

            if let draftOffset = draftCache.first?.offset {
                let extraDraftContext =
                    draftOffset - step.committedDraftOffset
                if extraDraftContext > 0 {
                    let trimmed = trimPromptCache(
                        draftCache, numTokens: extraDraftContext)
                    if trimmed != extraDraftContext {
                        throw DFlashError.untrimmableCache
                    }
                }
            }
        }
    }
}
