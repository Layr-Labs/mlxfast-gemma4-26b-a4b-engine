// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import MLX

extension EngineLoopV2 {

    /// Serial mode is the chip-independent authority path: every column
    /// executes the same `[B, 1]` eager forward used by ordinary decode,
    /// while one surrounding speculative KV transaction defers commit until
    /// the accept walk. Rectangular mode is an explicit optimized strategy.
    ///
    /// **Serial is an oracle, not a performance mode.** Production ALWAYS
    /// selects rectangular: `CBv2MTPRoundDriver.maximumAutomaticDepth`
    /// pre-clamps depth so `(1 + k) * B <= maxAutomaticRectangularTokens`
    /// always holds, so the `.automatic` arm below can never pick serial,
    /// and `MTPAutomaticVerificationPolicy` returns 8 on M3/M4/M5 and 4 on
    /// M1/M2 — never 0. Serial has never executed in the shipping provider.
    /// It is also strictly slower than MTP-off: `1 + k` target forwards emit
    /// at most `1 + k` tokens where plain decode emits one per forward.
    /// Degrading to it is a SAFETY NET, never a plan.
    /// `lazyColumns` (mirror road, see MTP/CBv2MTPMirrorOps.swift): the
    /// serial columns are built into ONE lazy graph. The per-column blocking
    /// `eval` below exists to keep mutable KV buffers from observing a later
    /// version; on the mirror road every in-place ring store is fence-chained
    /// (pass-A consumes the previous column's fence) and every other KV
    /// mutation is functional, so the chain is already ordered by data edges
    /// and the host never has to wait between columns.
    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver,
        lazyColumns: Bool = false
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
        // WIDE VERIFY on the mirror road: the same per-column-exact
        // arithmetic as the serial loop below, expressed as one [B, 1+k]
        // forward whose attention is serialized per column through the fused
        // decode road (see CBv2MTPWideVerifyContext). Opt-in until every
        // token-local plane carries its widened body.
        let wideVerify = lazyColumns && CBv2MTPWideVerifyContext.enabled && columns.count > 1
        if wideVerify { useRectangular = true }

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
            var argmaxColumns: [MLXArray] = []
            var hiddenColumns: [MLXArray] = []
            argmaxColumns.reserveCapacity(columns.count)
            hiddenColumns.reserveCapacity(columns.count)
            for column in columns {
                precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
                let output = mtp.model.forwardWithHidden(tokens: column, caches: caches)
                let columnHidden = Self.mtpRank3Hidden(output.lastHidden)
                let columnArgmax = argMax(output.logits, axis: -1).asType(.int32)
                // Building several eager decode calls in one lazy graph can
                // let mutable KV buffers observe a later version. Complete
                // each canonical target step before constructing the next —
                // except on the fence-ordered mirror road (`lazyColumns`).
                if !lazyColumns {
                    eval([columnArgmax, columnHidden] + eagerCacheInnerState(caches))
                }
                argmaxColumns.append(columnArgmax)
                hiddenColumns.append(columnHidden)
            }
            argmax = concatenated(argmaxColumns, axis: 1)
            hidden = concatenated(hiddenColumns, axis: 1)

        } else {
            for cache in serializingCaches { cache.mtpSerializesRectangularAttention = true }
            defer {
                for cache in serializingCaches { cache.mtpSerializesRectangularAttention = false }
            }
            let tokens = concatenated(columns, axis: 1)
            let output: (logits: MLXArray, lastHidden: MLXArray)
            if wideVerify {
                CBv2EngageMark.once("mtp-wide-verify")
                output = CBv2MTPWideVerifyContext.with(columns: columns.count) {
                    mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
                }
            } else {
                output = mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
            }
            var logits = output.logits
            var lastHidden = Self.mtpRank3Hidden(output.lastHidden)
            if wideVerify {
                // The trunk carried the rectangle as B * (1+k) rows; restore
                // the [B, 1+k, ...] geometry the accept walk indexes.
                let batch = columns[0].dim(0)
                let width = columns.count
                if logits.dim(0) == batch * width {
                    logits = logits.reshaped([batch, width, logits.dim(-1)])
                }
                if lastHidden.dim(0) == batch * width {
                    lastHidden = lastHidden.reshaped([batch, width, lastHidden.dim(-1)])
                }
            }
            argmax = argMax(logits, axis: -1).asType(.int32)
            hidden = lastHidden
        }

        return (argmax, hidden, eagerCacheInnerState(caches))
    }

    /// The pre-norm hidden as `[B, L, H]`. The promoted fused decode tail
    /// hands back `[B, H]` for an `L == 1` forward; every carry slice and the
    /// per-column concat below index a rank-3 tensor.
    static func mtpRank3Hidden(_ hidden: MLXArray) -> MLXArray {
        hidden.ndim == 2 ? hidden.expandedDimensions(axis: 1) : hidden
    }
}
