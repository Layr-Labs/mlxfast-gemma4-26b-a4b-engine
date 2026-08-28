// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Foundation
import MLX

extension EngineLoopV2 {

    /// Runs at the step's existing host-sync boundary after ordinary sampled
    /// rows finalize and before deferred KV releases.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        // The ordinary finalize loop has confirmed each seed row's bonus.
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

        guard let verify = round.verify else { return }
        let k = verify.k
        let host = verify.acceptancePacket.asArray(Int32.self)
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            let targets: [Int]
            let accepted: Int
            let emitted: Int
        }

        // Resolve and commit each row's own target-authoritative prefix.
        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            // A departed row is fenced by deferred release. Its device offset
            // is stale, so force the next eager composition to rebuild.
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
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    accepted: accepted,
                    emitted: naturalEmitted))
        }

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs)

        if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, $0.emitted) }.min() ?? 0
            let observedDrafts =
                outcomes.allSatisfy { $0.emitted <= stepAccepted }
                ? outcomes.map(\.emitted).min() ?? 0 : min(k, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: stepAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket)
        }

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            let emitted = Array(outcome.targets.prefix(outcome.emitted))

            // Confirm in order with the same stop and length semantics as the
            // ordinary finalize loop.
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

            // Correct KV and scheduler state before any terminal release.
            let confirmed = kept.count
            let rejected = (1 + k) - confirmed
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
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
            // Acceptance/rollback audit record (observability, 2026-08-25):
            // every value is already on the host at this boundary. The
            // scheduler fields are read AFTER recordSampled/rollbackComputed
            // above, so the record states the row's post-round accounting —
            // the boundary invariant a consumer checks is
            // `numComputedAfter == tokensCountAfter - 1`.
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
                // No inline deadline check: an MTP row that just confirmed
                // tokens is making progress and its decode lease is refreshed
                // in `refreshProgressLeases` (run at the end of the enclosing
                // `finalize`, after this round). Lease expiry is evaluated
                // centrally in `processLeaseExpiry` — identical typed-terminal
                // semantics to the ordinary decode path.
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

        // Rejected suffixes advanced eager device offsets past host truth.
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}
