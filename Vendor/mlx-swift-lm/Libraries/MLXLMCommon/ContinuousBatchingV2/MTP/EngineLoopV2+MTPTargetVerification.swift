// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import Foundation
import MLX

/// Model-family hook for a target that can build the sealed serial semantics
/// as one lazy graph.  MLXLM registers the concrete Gemma target when its MTP
/// drafter binds; keeping the hook here avoids widening the ordinary model
/// protocol for one production-only execution strategy.
public enum CBv2MTPBatchedExactTargetRegistry {
    public typealias Forward = (
        _ columns: [MLXArray], _ caches: [CBv2AttendingLayerCache]
    ) -> (argmax: MLXArray, lastHidden: MLXArray)?

    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var forwards: [ObjectIdentifier: Forward] = [:]
    }

    private static let state = State()

    public static func register(targetID: ObjectIdentifier, forward: @escaping Forward) {
        state.lock.lock()
        state.forwards[targetID] = forward
        state.lock.unlock()
    }

    static func forward(
        targetID: ObjectIdentifier, columns: [MLXArray],
        caches: [CBv2AttendingLayerCache]
    ) -> (argmax: MLXArray, lastHidden: MLXArray)? {
        state.lock.lock()
        let forward = state.forwards[targetID]
        state.lock.unlock()
        return forward?(columns, caches)
    }
}

/// Default ON; `0` restores the original blocking serial-column loop below.
private let cbv2MTPBatchedVerifyEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_CBV2_MTP_BATCHED_VERIFY"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// Deliberately default OFF.  The current exact Gemma expert fast path is
/// specialized to one 64-assignment `[8, 8]` route plane; widening it across
/// verify columns is not an order-identity proof.  A request for the unproven
/// mode fails closed to the blocking serial oracle.
private let cbv2MTPMoEDedupRequested: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_CBV2_MTP_MOE_DEDUP"]
    else { return false }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

extension EngineLoopV2 {

    /// Serial semantics are the chip-independent authority: every plane of
    /// every column executes with the same `[B, 1]` shape ordinary decode
    /// uses, while one surrounding speculative KV transaction defers commit
    /// until the accept walk.  A capable B=8 Gemma target may construct those
    /// order-identical calls in one lazy graph; every failed capability check
    /// and the kill switch retain the blocking column oracle below.
    /// Rectangular mode remains a separate, explicitly selected strategy.
    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        let argmax: MLXArray
        let hidden: MLXArray

        var useRectangular = switch mtp.config.verificationMode {
        case .serialTarget: false
        case .rectangular: true
        case .automatic:
            columns.count * columns[0].dim(0) <= mtp.config.maxAutomaticRectangularTokens
        }

        // Rectangular verification obliges every layer cache in the bank to
        // serialise its attention one query position at a time for the
        // duration of the round. That capability is the opt-in marker
        // `CBv2MTPRectangularSerializing` (Paged/PagedSeamContract.swift),
        // NOT a concrete type: `CBv2LayerCache` conforms by extension, and a
        // paged bank conforms only once `PagedLayerCache.updateAndAttend`
        // grows the per-column loop (WS-3.4).
        //
        // This was `as? CBv2LayerCache` behind a `preconditionFailure`.
        // `CBv2LayerCache` is `final` and `PagedLayerCache` is a SIBLING
        // conformer of `CBv2AttendingLayerCache`, never a subclass, so that
        // cast could not succeed for a paged bank — and `preconditionFailure`
        // is a `fatalError`: daemon death, every co-resident model's
        // in-flight requests lost, and not one line of telemetry. A bank that
        // cannot serialise MUST degrade to the serial oracle above and MUST
        // NOT trap (PagedSeamContract: "Callers MUST degrade to serial
        // verification for a cache that does not conform, and MUST NOT trap").
        var serializingCaches: [CBv2MTPRectangularSerializing] = []
        if useRectangular {
            serializingCaches = caches.compactMap { $0 as? CBv2MTPRectangularSerializing }
            if serializingCaches.count != caches.count {
                mtp.recordControllerFallback("rectangular_cache_unsupported")
                useRectangular = false
            }
        }
        mtp.recordVerificationStrategy(rectangular: useRectangular)

        if !useRectangular {
            let batch = columns[0].ndim == 2 ? columns[0].dim(0) : 0
            let exactOutput: (argmax: MLXArray, lastHidden: MLXArray)?
            if cbv2MTPBatchedVerifyEnabled,
                !cbv2MTPMoEDedupRequested,
                mtp.config.verificationMode == .serialTarget,
                batch == 8,
                columns.count > 1,
                columns.allSatisfy({
                    $0.ndim == 2 && $0.dim(0) == batch && $0.dim(1) == 1
                }),
                let targetID = mtp.model.mtpTargetIdentity
            {
                exactOutput = CBv2MTPBatchedExactTargetRegistry.forward(
                    targetID: targetID,
                    columns: columns,
                    caches: caches)
            } else {
                exactOutput = nil
            }

            if let output = exactOutput {
                argmax = output.argmax
                hidden = output.lastHidden
                CBv2EngageMark.once("mtp-batched-verify")
            } else {
                var argmaxColumns: [MLXArray] = []
                var hiddenColumns: [MLXArray] = []
                argmaxColumns.reserveCapacity(columns.count)
                hiddenColumns.reserveCapacity(columns.count)
                for column in columns {
                    precondition(
                        column.dim(1) == 1,
                        "CBv2 MTP: serial target column must have L=1")
                    let output = mtp.model.forwardWithHidden(tokens: column, caches: caches)
                    let columnArgmax = argMax(output.logits, axis: -1).asType(.int32)
                    // Building several eager decode calls in one lazy graph can
                    // let mutable KV buffers observe a later version. Complete
                    // each canonical target step before constructing the next.
                    eval([columnArgmax, output.lastHidden] + eagerCacheInnerState(caches))
                    argmaxColumns.append(columnArgmax)
                    hiddenColumns.append(output.lastHidden)
                }
                argmax = concatenated(argmaxColumns, axis: 1)
                hidden = concatenated(hiddenColumns, axis: 1)
            }

        } else {
            for cache in serializingCaches { cache.mtpSerializesRectangularAttention = true }
            defer {
                for cache in serializingCaches { cache.mtpSerializesRectangularAttention = false }
            }
            let tokens = concatenated(columns, axis: 1)
            let output = mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
            argmax = argMax(output.logits, axis: -1).asType(.int32)
            hidden = output.lastHidden
        }

        return (argmax, hidden, eagerCacheInnerState(caches))
    }
}
