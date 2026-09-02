// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Foundation
import MLX

extension EngineLoopV2 {

    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        if let seedHidden = round.seedHidden {
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                round.finalizedSeedIDs.insert(id)
            }
        }
        if CBv2StepProfiler.enabled, let launch = round.launchMemory {
            let now = CBv2MTPRoundInFlight.MemorySnapshot()
            let mb = 1.0 / Double(1 << 20)
            FileHandle.standardError.write(
                Data(String(
                    format: "[%@] alloc active %+.1f MB cache %+.1f MB peak %+.1f MB\n",
                    round.verify == nil ? "mtp-seed" : "mtp-round",
                    Double(now.active - launch.active) * mb,
                    Double(now.cache - launch.cache) * mb,
                    Double(now.peak - launch.peak) * mb).utf8))
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        let waitStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let host = verify.acceptancePacket.asArray(Int32.self)
        if CBv2MTPDeviceGate.measurementActive, step.launchedByChain {
            step.gateFencedNanos = DispatchTime.now().uptimeNanoseconds
        }
        if CBv2StepProfiler.enabled {
            FileHandle.standardError.write(
                Data(String(format: "[mtp-round] readback wait %.1f ms\n",
                    (CFAbsoluteTimeGetCurrent() - waitStart) * 1000).utf8))
        }
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            let targets: [Int]
            let accepted: Int
            let naturalEmitted: Int
        }

        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)
        var commonEmitted = targetWidth

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                anyRejected = true
                continue
            }
            let rec = scheduler.record(for: id)!
            let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
            let targets = (0 ..< targetWidth).map {
                Int(host[draftCount + batchIndex * targetWidth + $0])
            }

            var accepted = 0
            while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
            var naturalEmitted = accepted + 1
            naturalEmitted = min(
                naturalEmitted,
                rec.request.maxTokens - rec.generatedTokenCount)
            if let stopIndex = targets[..<naturalEmitted].firstIndex(where: {
                rec.request.stopTokens.contains($0)
            }) {
                naturalEmitted = stopIndex + 1
            }
            commonEmitted = min(commonEmitted, naturalEmitted)
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    accepted: accepted,
                    naturalEmitted: naturalEmitted))
        }
        let mirrorRoadRows = Set(
            verify.mirrorRestore.flatMap { $0.rows.map(ObjectIdentifier.init) })
        var confirmedByBatchIndex = [Int](repeating: targetWidth, count: verify.rows.count)

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs)

        for outcome in outcomes {
            let rowAccepted = min(outcome.accepted, outcome.naturalEmitted)
            let observedDrafts =
                outcome.naturalEmitted <= rowAccepted
                ? outcome.naturalEmitted : min(k, rowAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: rowAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket)
        }
        _ = commonEmitted

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            let emitted = Array(outcome.targets.prefix(outcome.naturalEmitted))

            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for token in emitted {
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }
                if hasStopStrings {
                    textPieces.append(detokenizer?.push([token]) ?? "")
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            let confirmed = kept.count
            let rejected = (1 + k) - confirmed
            confirmedByBatchIndex[batchIndex] = confirmed
            if rejected > 0 {
                for sequence in metadata.storageRows {
                    if let windowed = sequence as? CBv2WindowedSequenceKV,
                        mirrorRoadRows.contains(ObjectIdentifier(windowed))
                    {
                        windowed.mtpRollbackMirrorRoad(rejected)
                    } else {
                        sequence.rollback(rejected)
                    }
                }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }

            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let pushTokens = endsWithStopToken ? Array(kept.dropLast()) : kept
                let allTokens = kept
                detokQueue.async {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: allTokens, logprobs: nil),
                        consumingReservation: true)
                }
            }

            let committedAccepted = min(accepted, confirmed)
            mtp.recordRound(
                drafted: k, accepted: committedAccepted, emitted: confirmed,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw,
                    k: k,
                    draftTokens: Array(
                        host[batchIndex * k ..< (batchIndex + 1) * k].map(Int.init)),
                    targetTokens: outcome.targets,
                    accepted: accepted,
                    confirmed: confirmed,
                    rejected: rejected,
                    tokensCountAfter: rec.tokens.count,
                    numComputedAfter: rec.numComputedTokens,
                    generatedAfter: rec.generatedTokenCount,
                    finishReason: finishReason.map { String(describing: $0) }))

            if let finishReason {
                finishRequest(id, reason: finishReason)
            } else {
                let hiddenColumn = CBv2MTPHiddenIndex.carryColumn(
                    targetOutputIndex: confirmed - 1, draftDepth: k)
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: verify.lastHidden[
                        batchIndex ..< (batchIndex + 1),
                        hiddenColumn ..< (hiddenColumn + 1), 0...],
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
            }
        }

        if verify.acceptedDevice == nil {
            for layer in verify.mirrorRestore {
                var firstRestored = [Int](repeating: k + 1, count: layer.rows.count)
                for outcome in outcomes where outcome.batchIndex < firstRestored.count {
                    firstRestored[outcome.batchIndex] = confirmedByBatchIndex[outcome.batchIndex] + 1
                }
                guard firstRestored.contains(where: { $0 <= k }) else { continue }
                layer.cache.mtpWriteFence = CBv2MTPMirrorOps.restore(
                    mirrors: layer.mirrors, undo: layer.undo, slotBases: layer.slotBases,
                    firstRestored: MLXArray(firstRestored.map { Int32($0) }), depth: k,
                    kvHeads: layer.kvHeads, headDim: layer.headDim, window: layer.window,
                    fence: layer.cache.mtpWriteFence)
            }
        }

        if anyRejected, verify.acceptedDevice == nil {
            eagerCompositionStale = true
        }
        if CBv2StepProfiler.enabled, mtp.roundsForProfile % 16 == 0 {
            FileHandle.standardError.write(
                Data(("[cbv2-step-profile] rounds=\(mtp.roundsForProfile)\n"
                    + CBv2StepProfiler.summaryTable()).utf8))
        }
    }
}
