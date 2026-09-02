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
    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        let argmax: MLXArray
        let hidden: MLXArray
        // MTP-SETUP-CACHE: non-nil only on the rectangular road, where the
        // round's `[B, 1 + k]` forward is one decode-shaped graph whose logits
        // dominate every cache mutation it made. The serial oracle evaluates
        // each column separately and keeps the established root list.
        var rectangularForwardOutput: MLXArray?

        var useRectangular = switch mtp.config.verificationMode {
        // DARKBLOOM_GEMMA4_MTP_RECT_PROBE: under the sealed serial_target mode the
        // round's 1+k columns are verified by ONE [B, 1+k] forward instead of
        // 1+k serial [B, 1] forwards; `false` restores the serial oracle verbatim.
        // The sealed mode's 1+k separate target forwards are the oracle, never a
        // plan: they emit at most 1+k tokens for 1+k forwards, which is strictly
        // worse than plain decode. Every column of the rectangle is computed with
        // the canonical `L == 1` attention (`mtpSerializesRectangularAttention`),
        // so a column's attention is the same arithmetic the serial oracle would
        // run for it; only the weight-bound body batches across columns.
        case .serialTarget: CBv2MTPDepthController.DARKBLOOM_GEMMA4_MTP_RECT_PROBE
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
            var argmaxColumns: [MLXArray] = []
            var hiddenColumns: [MLXArray] = []
            argmaxColumns.reserveCapacity(columns.count)
            hiddenColumns.reserveCapacity(columns.count)
            for column in columns {
                precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
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

        } else {
            for cache in serializingCaches { cache.mtpSerializesRectangularAttention = true }
            defer {
                for cache in serializingCaches { cache.mtpSerializesRectangularAttention = false }
            }
            let tokens = concatenated(columns, axis: 1)
            let output = mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
            argmax = argMax(output.logits, axis: -1).asType(.int32)
            hidden = output.lastHidden
            rectangularForwardOutput = output.logits
        }

        guard let rectangularForwardOutput else {
            return (argmax, hidden, eagerCacheInnerState(caches))
        }
        return (
            argmax, hidden,
            mtpEvaluationRoots(caches, forwardOutput: rectangularForwardOutput)
        )
    }
}
