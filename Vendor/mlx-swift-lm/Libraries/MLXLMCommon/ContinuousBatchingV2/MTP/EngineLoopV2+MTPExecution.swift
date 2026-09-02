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

    /// Diagnostic only (DARKBLOOM_CBV2_MTP_PHASE_EVAL=1): force an eval at
    /// each round phase so wall time attributes to that phase's GPU work.
    static let mtpPhaseEvalTiming: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP_PHASE_EVAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Profiler-gated (`CBv2StepProfiler.enabled`) stage line for the seed
    /// round, the same gate and shape as `[cbv2-plain]`.
    static func mtpSeedProfileLine(_ stage: String, seconds: Double) {
        FileHandle.standardError.write(
            Data(String(format: "[mtp-seed] %@ %.1f ms\n", stage, seconds * 1000).utf8))
    }

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
            let profile = CBv2StepProfiler.enabled
            let composeStart = profile ? CFAbsoluteTimeGetCurrent() : 0
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = eagerCaches(rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let composeEnd = profile ? CFAbsoluteTimeGetCurrent() : 0
            if Self.mtpPhaseEvalTiming {
                // The previous step's still-running K/V writes are this
                // forward's inputs; waiting for them here attributes their
                // tail separately from the forward's own cost.
                let t = CFAbsoluteTimeGetCurrent()
                eval(eagerCacheInnerState(caches))
                let waited = CFAbsoluteTimeGetCurrent() - t
                CBv2StepProfiler.record("v2.mtp.phase.seed_input_eval", seconds: waited)
                if profile { Self.mtpSeedProfileLine("input-tail-wait", seconds: waited) }
            }
            let (logits, hidden) = mtp.model.forwardWithHidden(tokens: inputs, caches: caches)
            let lastLogits = logits[0..., -1, 0...]
            // Same evaluation roots as the plain decode step: the logits
            // reach every K/V mutation of this forward, so the per-row
            // storage arrays need not be enumerated.
            cacheInnerState.append(
                contentsOf: eagerDecodeEvaluationRoots(caches, logitsRoot: lastLogits))
            let forwardEnd = profile ? CFAbsoluteTimeGetCurrent() : 0
            decodeSampled = sampler.sample(
                logits: lastLogits,
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
            if !seedRows.isEmpty { seedHidden = Self.mtpRank3Hidden(hidden) }
            if profile, !seedRows.isEmpty {
                let samplerEnd = CFAbsoluteTimeGetCurrent()
                Self.mtpSeedProfileLine("compose", seconds: composeEnd - composeStart)
                Self.mtpSeedProfileLine("forward-build", seconds: forwardEnd - composeEnd)
                Self.mtpSeedProfileLine("sampler-build", seconds: samplerEnd - forwardEnd)
            }
        }

        // Chunked prefills remain per-request [1, chunk], matching executeMixed.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var prefillEvalTargets: [MLXArray] = []
        var prefillWriteTail: [MLXArray] = []
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
            let written = eagerCacheInnerState(caches)
            cacheInnerState.append(contentsOf: written)
            prefillWriteTail.append(contentsOf: written)
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
        var sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))
        if !prefillWriteTail.isEmpty {
            (sampledTokens, prefillEvalTargets) = fencedOnPrefillWriteTail(
                sampled: sampledTokens, outputs: prefillEvalTargets, tail: prefillWriteTail)
        }

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

    /// One verify round's inputs, independent of how the seed reached us:
    /// from host carries (the vendored round) or from the previous round's
    /// lazy device outputs (a chained round).
    struct CBv2MTPVerifyInputs {
        let ids: [CBv2RequestID]
        let states: [[CBv2SequenceKV?]]
        /// `[B, 1]` int32 seed tokens (column 0 of the rectangle).
        let seedColumn: MLXArray
        /// `[B, 1, H]` pre-norm carry hidden the drafter consumes.
        let carryHidden: MLXArray
        /// Chained rounds: `[B]` int32 device position of column 0 per row
        /// (the host counters may run ahead by the previous round's
        /// unconfirmed columns). nil = host counters are exact.
        let base: MLXArray?
    }

    private func mtpBuildVerifyGraph(
        _ verifyRows: [CBv2MTPRowWork],
        driver mtp: CBv2MTPRoundDriver,
        cacheInnerState: inout [MLXArray]
    ) -> CBv2MTPRoundInFlight.Verify? {
        guard !verifyRows.isEmpty else { return nil }
        let depths = Set(verifyRows.compactMap { mtp.roundMark(for: $0.rec.id) })
        precondition(depths.count == 1, "CBv2 MTP: one plan must use one uniform depth")
        let k = depths.first!
        let batch = verifyRows.count
        for row in verifyRows {
            let carry = row.carry!
            let fullRow = kvStates[row.rec.id]![mtp.captureLayers.full]!
            precondition(
                fullRow.absoluteOffset == carry.kvOffset,
                "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
            )
        }
        let inputs = CBv2MTPVerifyInputs(
            ids: verifyRows.map(\.rec.id),
            states: verifyRows.map { kvStates[$0.rec.id]! },
            seedColumn: MLXArray(verifyRows.map { Int32($0.carry!.token) }).reshaped([batch, 1]),
            carryHidden: concatenated(verifyRows.map { Self.mtpRank3Hidden($0.carry!.hidden) }, axis: 0),
            base: nil)
        return mtpBuildVerifyRound(inputs, k: k, driver: mtp, cacheInnerState: &cacheInnerState)
    }

    func mtpBuildVerifyRound(
        _ inputs: CBv2MTPVerifyInputs, k: Int,
        driver mtp: CBv2MTPRoundDriver,
        cacheInnerState: inout [MLXArray]
    ) -> CBv2MTPRoundInFlight.Verify? {
        let draftStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let batch = inputs.ids.count
        var captures: [CBv2MTPRowCapture] = []
        var rowMetadata: [CBv2MTPRoundInFlight.VerifyRow] = []
        var captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)] = []
        captures.reserveCapacity(batch)
        captured.reserveCapacity(2 * batch)

        // MIRROR ROAD (MTP/CBv2MTPMirrorOps.swift): every verify column is an
        // ordinary [B, 1] decode forward whose fused pass-A writes the live
        // q4 mirror slot in place; the k slots columns 1...k overwrite are
        // copied out per layer by a fenced capture BEFORE the columns run and
        // the rejected suffix is restored from that log in-graph.
        let verifyStates = inputs.states
        let roundCaches = eagerCaches(rowStates: verifyStates)
        let mirrorRoad = mtpMirrorRoadAdmits(states: verifyStates, caches: roundCaches)
        let base = inputs.base
        precondition(base == nil || mirrorRoad, "CBv2 MTP: a chained round needs the mirror road")
        var mirrorRestore: [CBv2MTPMirrorRestoreLayer] = []
        var deviceSlotBases: MLXArray? = nil
        // One `[8]` slot-base upload per distinct host value set per round;
        // the unified position chain makes every sliding layer share it.
        var hostSlotBases: [[Int32]: MLXArray] = [:]
        let captureStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        if mirrorRoad {
            for (layerIndex, cache) in roundCaches.enumerated() {
                guard let layerCache = cache as? CBv2LayerCache,
                    layerCache.kind.sharesKVWithLayer == nil
                else { continue }
                let windowed = verifyStates.compactMap {
                    $0[layerIndex] as? CBv2WindowedSequenceKV
                }
                guard windowed.count == batch else { continue }
                let mirrors = windowed.compactMap(\.mtpQuantMirror)
                precondition(mirrors.count == batch, "CBv2 MTP mirror road: admitted row without a mirror")
                let slotBases: MLXArray
                if let base {
                    if deviceSlotBases == nil {
                        deviceSlotBases = base % Int32(windowed[0].window)
                    }
                    slotBases = deviceSlotBases!
                } else {
                    let bases = windowed.map { Int32($0.mtpSlotBase) }
                    if let shared = hostSlotBases[bases] {
                        slotBases = shared
                    } else {
                        slotBases = MLXArray(bases)
                        hostSlotBases[bases] = slotBases
                    }
                }
                let captured = CBv2MTPMirrorOps.capture(
                    mirrors: mirrors, slotBases: slotBases, depth: k,
                    kvHeads: windowed[0].kvHeads, headDim: windowed[0].headDim,
                    window: windowed[0].window, fence: layerCache.mtpWriteFence)
                layerCache.mtpWriteFence = captured.fence
                mirrorRestore.append(
                    CBv2MTPMirrorRestoreLayer(
                        cache: layerCache, rows: windowed, mirrors: mirrors,
                        slotBases: slotBases, undo: captured.undo, depth: k,
                        kvHeads: windowed[0].kvHeads, headDim: windowed[0].headDim,
                        window: windowed[0].window))
            }
            CBv2EngageMark.once("mtp-mirror-road")
            if CBv2StepProfiler.enabled, base == nil {
                FileHandle.standardError.write(
                    Data(String(format: "[mtp-round] first mirror-capture %.1f ms layers=%d\n",
                        (CFAbsoluteTimeGetCurrent() - captureStart) * 1000,
                        mirrorRestore.count).utf8))
            }
        }

        let slidingLayer = mtp.captureLayers.sliding
        // One `[B]` mirror slot-base vector per round; each row's view takes its slice.
        let slidingSlotBases: MLXArray? = base.flatMap { base in
            (verifyStates[0][slidingLayer] as? CBv2WindowedSequenceKV).map {
                base % Int32($0.window)
            }
        }
        for rowIndex in 0 ..< batch {
            let state = verifyStates[rowIndex]
            let fullRow = state[mtp.captureLayers.full]!
            let slidingRow = state[slidingLayer]!
            let fullSnapshot = fullRow.snapshot()
            let slidingSnapshot: (keys: MLXArray, values: MLXArray)
            var slidingMirror: MLXArray? = nil
            if mirrorRoad,
                let windowed = slidingRow as? CBv2WindowedSequenceKV,
                let slidingCache = roundCaches[slidingLayer] as? CBv2LayerCache,
                let dequantized = slidingSlotBases.map({
                    windowed.mtpDequantizedRetainedViews(
                        startDevice: $0[rowIndex ..< (rowIndex + 1)],
                        fence: slidingCache.mtpWriteFence)
                }) ?? windowed.mtpDequantizedRetainedViews(fence: slidingCache.mtpWriteFence)
            {
                // The drafter attends the bytes the target attends: the q4
                // mirror, dequantized, ordered after this round's capture. A
                // chained round also hands the mirror itself so the drafter
                // can read it in place and leave these views unevaluated.
                slidingSnapshot = dequantized
                slidingMirror = windowed.mtpQuantMirror
            } else {
                let snapshot = slidingRow.snapshot()
                slidingSnapshot = (snapshot.keys, snapshot.values)
            }
            captures.append(
                CBv2MTPRowCapture(
                    fullKeys: fullSnapshot.keys,
                    fullValues: fullSnapshot.values,
                    slidingKeys: slidingSnapshot.keys,
                    slidingValues: slidingSnapshot.values,
                    slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                    anchor: fullRow.absoluteOffset, slidingMirror: slidingMirror))
            captured.append((fullRow, fullSnapshot.keys, fullSnapshot.values))
            captured.append((slidingRow, slidingSnapshot.keys, slidingSnapshot.values))
            rowMetadata.append(
                CBv2MTPRoundInFlight.VerifyRow(
                    id: inputs.ids[rowIndex], storageRows: state.compactMap { $0 }))
        }

        mtpFreezeCaptures(captured)

        if Self.mtpPhaseEvalTiming {
            let t = CFAbsoluteTimeGetCurrent()
            eval(captures.flatMap { [$0.fullKeys, $0.fullValues, $0.slidingKeys, $0.slidingValues] })
            CBv2StepProfiler.record("v2.mtp.phase.capture_eval", seconds: CFAbsoluteTimeGetCurrent() - t)
        }
        // Host counters bound the captured views; the device base is the truth.
        let cohort: CBv2MTPCohortCapture? = base.map { base in
            let retained = MLXArray(verifyStates.map { Int32($0[slidingLayer]!.retainedCount) })
            return CBv2MTPCohortCapture(
                anchors: base, slidingStarts: base - retained, fullLengths: base,
                pooledFull: CBv2MTPCohortCapture.pooledFullViews(
                    states: verifyStates, layer: mtp.captureLayers.full),
                slidingMirrorSlotBases: slidingSlotBases,
                slidingMirrorFence: (roundCaches[slidingLayer] as? CBv2LayerCache)?.mtpWriteFence)
        }
        let prepared = mtp.drafter.prepare(rows: captures, cohort: cohort)
        let seedColumn = inputs.seedColumn
        var draftInput = seedColumn
        var draftHidden = inputs.carryHidden
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
        if Self.mtpPhaseEvalTiming {
            let t = CFAbsoluteTimeGetCurrent()
            eval(draftIDs)
            CBv2StepProfiler.record("v2.mtp.phase.draft_eval", seconds: CFAbsoluteTimeGetCurrent() - t)
        }

        // Windowed rows on the mirror road are NOT armed: their columns write
        // the live mirror in place and the rejected suffix is undone from the
        // captured log. Every other storage row keeps the staging contract.
        let verifyStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        for metadata in rowMetadata {
            for sequence in metadata.storageRows {
                if mirrorRoad, sequence is CBv2WindowedSequenceKV { continue }
                sequence.beginSpeculativeWrite()
            }
        }
        let targetColumns = [seedColumn] + draftSteps.map { $0.reshaped([batch, 1]) }
        let target = mtpBuildTargetVerification(
            columns: targetColumns, rowStates: verifyStates, driver: mtp,
            lazyColumns: mirrorRoad, deviceBase: base)
        cacheInnerState.append(contentsOf: target.cacheInnerState)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.verify.build", seconds: CFAbsoluteTimeGetCurrent() - verifyStart)
        }
        if Self.mtpPhaseEvalTiming {
            let t = CFAbsoluteTimeGetCurrent()
            eval([target.argmax] + target.cacheInnerState)
            CBv2StepProfiler.record("v2.mtp.phase.verify_eval", seconds: CFAbsoluteTimeGetCurrent() - t)
        }
        let acceptancePacket = concatenated(
            [draftIDs.reshaped([-1]), target.argmax.reshaped([-1])], axis: 0)

        var verify = CBv2MTPRoundInFlight.Verify(
            k: k,
            rows: rowMetadata,
            acceptancePacket: acceptancePacket,
            lastHidden: target.hidden,
            mirrorRestore: mirrorRestore)
        verify.rowStates = verifyStates

        // DEVICE ACCEPT WALK: the count of leading draft/target matches per
        // row, computed in the round's own graph so the rejected mirror slots
        // are restored, the device position chain rewound and the next
        // round's seed/carry selected without a host round trip. The host
        // finalize re-derives the same numbers from the packet for
        // bookkeeping.
        if mirrorRoad, batch == 8 {
            let matches = (draftIDs .== target.argmax[0..., 0 ..< k]).asType(.int32)
            let accepted = cumprod(matches, axis: 1).sum(axis: 1)
            verify.acceptedDevice = accepted
            // Column `accepted + 1` is the first rejected one and is
            // overwritten by the next seed before anything reads it; the
            // slots after it are restored from the undo log.
            let firstRestored = accepted + Int32(2)
            for layer in mirrorRestore {
                layer.cache.mtpWriteFence = CBv2MTPMirrorOps.restore(
                    mirrors: layer.mirrors, undo: layer.undo, slotBases: layer.slotBases,
                    firstRestored: firstRestored, depth: k,
                    kvHeads: layer.kvHeads, headDim: layer.headDim, window: layer.window,
                    fence: layer.cache.mtpWriteFence)
            }
            // The rectangle advanced the unified position chain by 1 + k;
            // committed width is accepted + 1.
            let rewind = Int32(k) - accepted
            for cache in roundCaches {
                (cache as? CBv2LayerCache)?.mtpRewindPositionOffsets(by: rewind)
            }
            // Chained hand-off: the last committed column is the next seed,
            // its pre-norm hidden the next carry.
            let column = accepted.reshaped([batch, 1])
            verify.seedNext = takeAlong(target.argmax, column, axis: 1).reshaped([batch])
            verify.carryHiddenNext = takeAlong(
                target.hidden, column.reshaped([batch, 1, 1]), axis: 1)
            // Column 0's position: the captures recorded it BEFORE the
            // columns advanced the host counters.
            let hostBase = MLXArray(captures.map { Int32($0.anchor) })
            verify.nextBase = (base ?? hostBase) + accepted + Int32(1)
        }
        return verify
    }

    /// Mirror-road admission for one round: every verify row's storage-owning
    /// sliding layer is a full-ring windowed row with a live q4 mirror and no
    /// staged transaction, every storage-owning cache is the contiguous
    /// `CBv2LayerCache` (the fence owner), and the kill switch is not set.
    /// Anything else keeps the vendored staging path for the whole round —
    /// the two roads never mix within a round.
    private func mtpMirrorRoadAdmits(
        states: [[CBv2SequenceKV?]], caches: [CBv2AttendingLayerCache]
    ) -> Bool {
        guard CBv2MTPMirrorOps.enabled, states.count == 8 else { return false }
        var sawWindowed = false
        for (layerIndex, cache) in caches.enumerated() {
            guard cache.kind.sharesKVWithLayer == nil else { continue }
            guard cache is CBv2LayerCache else { return false }
            for state in states {
                guard let sequence = state[layerIndex] else { return false }
                if let windowed = sequence as? CBv2WindowedSequenceKV {
                    guard windowed.mtpMirrorRoadAvailable else { return false }
                    sawWindowed = true
                }
            }
        }
        return sawWindowed
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

extension CBv2MTPCohortCapture {
    /// The full-attention pool every row of `states` is bound to, as one
    /// row-ordered view pair over the longest retained length. nil when any
    /// row is unpooled; this never forms a pool (the lockstep decode path
    /// owns migration).
    static func pooledFullViews(
        states: [[CBv2SequenceKV?]], layer: Int
    ) -> (keys: MLXArray, values: MLXArray)? {
        let rows = states.compactMap { $0[layer] as? CBv2FullSequenceKV }
        guard rows.count == states.count, rows.first?.cohortPool != nil,
            let pool = CBv2FullSequenceKV.cohortPool(binding: rows)
        else { return nil }
        return pool.batchViews(upTo: rows.map(\.retainedCount).max()!)
    }
}
