// EngineLoopV2+MTPExecution.swift
//
// MTP row classification and lazy MLX graph construction.

import Foundation
import MLX

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
            cacheInnerState.append(contentsOf: eagerDecodeEvaluationRoots(caches, logitsRoot: logits))
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
            precondition(
                fullRow.absoluteOffset == carry.kvOffset,
                "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
            )
            let fullSnapshot = fullRow.snapshot()
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
        for _ in 0 ..< k {
            let (next, nextHidden) = mtp.drafter.draftStep(
                tokens: draftInput, hidden: draftHidden, prepared: prepared)
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
        let target = mtpBuildTargetVerification(
            columns: targetColumns, rows: verifyRows, driver: mtp)
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

    /// Freeze the round's pre-write KV captures against the in-place writes
    /// the very same graph is about to perform. See `CBv2MTPCaptureFence`
    /// for the hazard and the mechanism.
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
