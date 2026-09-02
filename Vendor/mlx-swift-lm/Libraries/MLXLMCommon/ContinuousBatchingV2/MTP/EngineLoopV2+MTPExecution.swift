// EngineLoopV2+MTPExecution.swift
//
// MTP row classification and lazy MLX graph construction.

import Foundation
import MLX

/// MTP-SETUP-CACHE. Every round hands the step's `asyncEval` a list of
/// evaluation roots for the caches the round mutated. The non-speculative
/// `[B, 1]` decode step builds that list through
/// `CBv2SteppableModel.compactDecodeEvaluationRoots`: the forward's own
/// output, the one shared position-offset chain, and one ring-write fence per
/// layer -- 32 handles for a 30-layer, 8-row bank, and the census mark
/// `compact-decode-roots rows=8 layers=30 roots=32` is that door firing on the
/// crown's decode road. The MTP round never took it. It called
/// `eagerCacheInnerState` instead, which walks every layer and every row and
/// rebuilds the FULL per-row inner state -- for this bank 25 sliding layers x
/// (2 + 8 rows x 3 buffers) + 5 full layers x (2 + 8 rows x 2) = roughly 740
/// `MLXArray` handles, constructed and then handed to `asyncEval`, once per
/// round, every round.
///
/// This switch routes the round's decode-shaped groups through the SAME door,
/// with the same model-side proof obligation (`forwardOutput` dominates every
/// cache mutation it represents) and the same fail-closed guard: the compact
/// builder returns nil unless the model affirms the contract over an
/// all-owning, all-contiguous bank with one shared position state, and nil
/// keeps the established full list verbatim. It is a HOST-side change only:
/// which arrays are NAMED as roots, never a value that any of them holds.
///
/// `DARKBLOOM_GEMMA4_MTP_SETUP_CACHE=0` restores `eagerCacheInnerState` at
/// every call site in the same executable.
enum CBv2MTPSetupCache {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MTP_SETUP_CACHE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()
    /// Census only. When `MLXFAST_ENGAGE_MARKS` is set the mark carries BOTH
    /// counts, so the census is a measurement of the mechanism and not just
    /// evidence that a branch was taken (R13): the full list is built once,
    /// beside the compact one, purely to be counted. Never on a timing run.
    static let marksArmed =
        ProcessInfo.processInfo.environment["MLXFAST_ENGAGE_MARKS"] != nil
}

struct CBv2MTPRowWork {
    let rec: CBv2ScheduledRequest
    let start: Int
    let count: Int
    let samples: Bool
    let isDecode: Bool
    let isSeed: Bool
    /// Non-nil for verify rows: the consumed carry.
    let carry: CBv2MTPCarry?
}

struct CBv2MTPGraphBuild {
    let sampledRows: [CBv2RequestID]
    let sampledTokens: MLXArray?
    /// Non-sampling prefill handles retained by the in-flight step.
    let prefillEvalTargets: [MLXArray]
    let asyncEvalTargets: [MLXArray]
    let logprobSegments: [CBv2StepLogprobs]
    let verify: CBv2MTPRoundInFlight.Verify?
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    let seedHidden: MLXArray?
}

extension EngineLoopV2 {

    func mtpPrepareRoundWork(
        _ plan: CBv2StepPlan,
        driver mtp: CBv2MTPRoundDriver,
        demoteAllRounds: Bool
    ) -> [CBv2MTPRowWork] {
        var work: [CBv2MTPRowWork] = []
        work.reserveCapacity(plan.assignments.count)

        for (id, assignedTokens) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }
            var count = assignedTokens

            if let k = mtp.roundMark(for: id) {
                if demoteAllRounds {
                    if count == 1 + k { scheduler.rollbackComputed(id: id, tokens: k) }
                    count = 1
                    mtp.invalidateCarry(id)
                } else if count == 1 + k, let carry = mtp.consumeCarry(for: id) {
                    work.append(
                        CBv2MTPRowWork(
                            rec: rec, start: rec.numComputedTokens - count, count: count,
                            samples: true, isDecode: false, isSeed: false, carry: carry))
                    continue
                } else if count == 1 + k {
                    // A marked assignment without a consumable carry demotes
                    // exactly like the scheduler's headroom retry.
                    scheduler.rollbackComputed(id: id, tokens: k)
                    count = 1
                } else {
                    switch plan.speculationFallbacks[id] {
                    case .tokenBudget: mtp.recordSkip("token_budget")
                    case .kvHeadroom: mtp.recordSkip("kv_headroom")
                    case nil: mtp.recordSkip("planner_clamp")
                    }
                }
            }

            // Mirror executeMixed's row classification. MTP decode-shaped
            // work stays eager; final-token image spans remain prefill work.
            let start = rec.numComputedTokens - count
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                count == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                CBv2MTPRowWork(
                    rec: rec, start: start, count: count, samples: samples,
                    isDecode: isDecode, isSeed: isDecode && mtp.isSeedMarked(id), carry: nil))
        }
        return work
    }

    func mtpBuildRoundGraph(
        _ work: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> CBv2MTPGraphBuild {
        var cacheInnerState: [MLXArray] = []
        var logprobSegments: [CBv2StepLogprobs] = []

        // Plain and seed rows share one eager [B, 1] target batch. Seed rows
        // retain the pre-norm hidden; logits remain identical to plain eager.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var seedRows: [(id: CBv2RequestID, decodeIndex: Int)] = []
        var seedHidden: MLXArray?
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = eagerCaches(rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let (logits, hidden) = mtp.model.forwardWithHidden(tokens: inputs, caches: caches)
            // MTP-SETUP-CACHE: this group IS the `[B, 1]` decode cell, so it
            // takes the decode step's own compact-root door.
            cacheInnerState.append(
                contentsOf: mtpEvaluationRoots(caches, forwardOutput: logits))
            decodeSampled = sampler.sample(
                logits: logits[0..., -1, 0...],
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
            for (index, row) in decodeRows.enumerated() where row.isSeed {
                seedRows.append((id: row.rec.id, decodeIndex: index))
            }
            if !seedRows.isEmpty { seedHidden = hidden }
        }

        // Chunked prefills remain per-request [1, chunk], matching executeMixed.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var prefillEvalTargets: [MLXArray] = []
        for row in work where !row.isDecode && row.carry == nil {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let requirement: CBv2PrefillRequirement =
                row.samples ? .lastPositionLogits : .evaluationOnly
            let output: MLXArray
            if let multimodal = multimodalByID[rec.id],
                let spanContext = multimodal.chunkContext(start: row.start, count: row.count)
            {
                output = multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    multimodal: multimodal, spanContext: spanContext, caches: caches,
                    requirement: requirement)
            } else {
                output = prefillOutput(
                    tokens: inputs, inputEmbeddings: nil, caches: caches,
                    requirement: requirement)
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                prefillSampled[rec.id] = sampler.sample(
                    logits: output,
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                prefillEvalTargets.append(output)
            }
        }

        let verifyRows = work.filter { $0.carry != nil }
        let verify = mtpBuildVerifyGraph(
            verifyRows, driver: mtp, cacheInnerState: &cacheInnerState)

        // Plain sampled tokens stay in plan order. Verify rows are finalized
        // from the target-authoritative acceptance packet instead.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIndex = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIndex ..< decodeIndex + 1])
                decodeIndex += 1
                sampledRows.append(row.rec.id)
            } else if let sampled = prefillSampled[row.rec.id] {
                pieces.append(sampled)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        var asyncEvalTargets = prefillEvalTargets
        if let sampledTokens { asyncEvalTargets.append(sampledTokens) }
        for segment in logprobSegments {
            asyncEvalTargets.append(contentsOf: segment.evalTargets)
        }
        if let verify {
            asyncEvalTargets.append(verify.acceptancePacket)
            asyncEvalTargets.append(verify.lastHidden)
        }
        if let seedHidden { asyncEvalTargets.append(seedHidden) }
        if !cacheInnerState.isEmpty {
            asyncEvalTargets.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }

        return CBv2MTPGraphBuild(
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            prefillEvalTargets: prefillEvalTargets,
            asyncEvalTargets: asyncEvalTargets,
            logprobSegments: logprobSegments,
            verify: verify,
            seedRows: seedRows,
            seedHidden: seedHidden)
    }

    private func mtpBuildVerifyGraph(
        _ verifyRows: [CBv2MTPRowWork],
        driver mtp: CBv2MTPRoundDriver,
        cacheInnerState: inout [MLXArray]
    ) -> CBv2MTPRoundInFlight.Verify? {
        guard !verifyRows.isEmpty else { return nil }
        let draftStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let depths = Set(verifyRows.compactMap { mtp.roundMark(for: $0.rec.id) })
        precondition(depths.count == 1, "CBv2 MTP: one plan must use one uniform depth")
        let k = depths.first!
        let batch = verifyRows.count
        var captures: [CBv2MTPRowCapture] = []
        var rowMetadata: [CBv2MTPRoundInFlight.VerifyRow] = []
        var seedTokens: [Int32] = []
        var carryHiddens: [MLXArray] = []
        // Each capture paired with the row it was gathered from, so
        // `mtpFreezeCaptures` can fence it against that row's own storage.
        var captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)] = []
        captures.reserveCapacity(batch)
        captured.reserveCapacity(2 * batch)

        for row in verifyRows {
            let state = kvStates[row.rec.id]!
            let carry = row.carry!
            // Captured BEFORE the target forward writes the round's
            // speculative columns. On a backend whose snapshots are lazy
            // reads of in-place-mutated storage that ordering is not free —
            // `mtpFreezeCaptures` below is what actually enforces it.
            let fullRow = state[mtp.captureLayers.full]!
            let slidingRow = state[mtp.captureLayers.sliding]!
            (slidingRow as? CBv2WindowedSequenceKV)?.drafterCaptureRow = true
            precondition(
                fullRow.absoluteOffset == carry.kvOffset,
                "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
            )
            let fullSnapshot = fullRow.snapshot()
            // DRAFTER-RING-ATTN: a full sliding ring is handed to the drafter as
            // its retained allocation + oldest slot (no temporal-order gather);
            // the drafter's sliding layers attend it through the decode ring
            // kernel. DARKBLOOM_GEMMA4_DRAFTER_RING_ATTN=0 restores the snapshot.
            if Self.drafterRingAttnEnabled,
                let ringRow = slidingRow as? CBv2WindowedSequenceKV,
                let ring = ringRow.decodeRingView
            {
                CBv2EngageMark.once("drafter-ring-attn")
                captures.append(
                    CBv2MTPRowCapture(
                        fullKeys: fullSnapshot.keys,
                        fullValues: fullSnapshot.values,
                        slidingKeys: ring.keys,
                        slidingValues: ring.values,
                        slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                        anchor: fullRow.absoluteOffset,
                        slidingRing: (ring.keys, ring.values, ring.start)))
                captured.append((fullRow, fullSnapshot.keys, fullSnapshot.values))
            } else {
                let slidingSnapshot = slidingRow.snapshot()
                captures.append(
                    CBv2MTPRowCapture(
                        fullKeys: fullSnapshot.keys,
                        fullValues: fullSnapshot.values,
                        slidingKeys: slidingSnapshot.keys,
                        slidingValues: slidingSnapshot.values,
                        slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                        anchor: fullRow.absoluteOffset))
                captured.append((fullRow, fullSnapshot.keys, fullSnapshot.values))
                captured.append((slidingRow, slidingSnapshot.keys, slidingSnapshot.values))
            }
            rowMetadata.append(
                CBv2MTPRoundInFlight.VerifyRow(
                    id: row.rec.id, storageRows: state.compactMap { $0 }))
            seedTokens.append(Int32(carry.token))
            carryHiddens.append(carry.hidden)
        }

        mtpFreezeCaptures(captured)

        let prepared = mtp.drafter.prepare(rows: captures)
        let seedColumn = MLXArray(seedTokens).reshaped([batch, 1])
        var draftInput = seedColumn
        var draftHidden = concatenated(carryHiddens, axis: 0)
        var draftSteps: [MLXArray] = []
        draftSteps.reserveCapacity(k)
        // Diagnostic split of the round into its draft and its verify forward.
        // Armed by `DARKBLOOM_MTP1_TIMING=1`; the evals it adds are what make
        // the split measurable, so it never shares a run with a decode-window
        // measurement and is inert on the scored path.
        let mtp1Timing = ProcessInfo.processInfo.environment["DARKBLOOM_MTP1_TIMING"] == "1"
        let mtp1T0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< k {
            let (next, nextHidden) = mtp.drafter.draftStep(
                tokens: draftInput, hidden: draftHidden, prepared: prepared)
            if mtp1Timing { eval(next, nextHidden) }
            draftSteps.append(next)
            draftInput = next.reshaped([batch, 1])
            draftHidden = nextHidden
        }
        let draftIDs = stacked(draftSteps, axis: 1)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.draft.build", seconds: CFAbsoluteTimeGetCurrent() - draftStart)
        }

        // Windowed rows stage provisional writes; other supported storage
        // backends implement the transaction hooks as exact no-ops/rollback.
        let verifyStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        for metadata in rowMetadata {
            for sequence in metadata.storageRows { sequence.beginSpeculativeWrite() }
        }
        let targetColumns = [seedColumn] + draftSteps.map { $0.reshaped([batch, 1]) }
        let mtp1T1 = CFAbsoluteTimeGetCurrent()
        let target = mtpBuildTargetVerification(
            columns: targetColumns, rows: verifyRows, driver: mtp)
        if mtp1Timing {
            eval(target.argmax, target.hidden)
            let mtp1T2 = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(
                Data("MTP1 timing: k=\(k) draft_ms=\(Int((mtp1T1 - mtp1T0) * 1000)) verify_ms=\(Int((mtp1T2 - mtp1T1) * 1000))\n".utf8))
        }
        cacheInnerState.append(contentsOf: target.cacheInnerState)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.verify.build", seconds: CFAbsoluteTimeGetCurrent() - verifyStart)
        }
        let acceptancePacket = concatenated(
            [draftIDs.reshaped([-1]), target.argmax.reshaped([-1])], axis: 0)
        return CBv2MTPRoundInFlight.Verify(
            k: k,
            rows: rowMetadata,
            acceptancePacket: acceptancePacket,
            lastHidden: target.hidden)
    }

    /// MTP-SETUP-CACHE: the round's evaluation roots for one decode-shaped
    /// group. `forwardOutput` is the root the group's own forward produced;
    /// the compact builder is the same one the non-speculative decode step
    /// uses and it fails closed to the established full inner state.
    func mtpEvaluationRoots(
        _ caches: [CBv2AttendingLayerCache], forwardOutput: MLXArray
    ) -> [MLXArray] {
        guard CBv2MTPSetupCache.enabled,
            let compact = model.compactDecodeEvaluationRoots(
                forwardOutput: forwardOutput, caches: caches)
        else { return eagerCacheInnerState(caches) }
        if CBv2MTPSetupCache.marksArmed {
            CBv2EngageMark.once(
                "mtp-setup-cache roots=\(compact.count) "
                    + "full=\(eagerCacheInnerState(caches).count) "
                    + "layers=\(caches.count)")
        } else {
            CBv2EngageMark.once("mtp-setup-cache")
        }
        return compact
    }

    /// Freeze the round's pre-write KV captures against the in-place writes
    /// the very same graph is about to perform. See `CBv2MTPCaptureFence`
    /// for the hazard and the mechanism.
    static let drafterRingAttnEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_DRAFTER_RING_ATTN"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private func mtpFreezeCaptures(
        _ captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)]
    ) {
        // Contiguous rows are ARC-owned by their views and need nothing.
        // `requiresMaterializedSnapshots` is the bit that already documents
        // exactly this recyclable-storage hazard: true for `PagedKVBackend`,
        // false everywhere else, so contiguous stays byte-identical.
        guard backend.requiresMaterializedSnapshots else { return }
        let unfenceable = CBv2MTPCaptureFence.publish(captured)
        if !unfenceable.isEmpty { eval(unfenceable) }
    }

}
