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
            // MTP-BATCHED-VERIFY: one layer-major pass over all columns —
            // per-column [B,1] roads everywhere, one merged expert gather.
            // The model declines geometries it cannot serve exactly and the
            // serial column loop below remains the road.
            if columns.count > 1,
                let batched = mtp.model.verifyColumns(
                    tokens: concatenated(columns, axis: 1), caches: caches)
            {
                argmax = concatenated(batched.argmax, axis: 1)
                hidden = concatenated(batched.hidden, axis: 1)
                return (argmax, hidden, mtpCompactVerifyRoots(caches))
            }
            var argmaxColumns: [MLXArray] = []
            var hiddenColumns: [MLXArray] = []
            argmaxColumns.reserveCapacity(columns.count)
            hiddenColumns.reserveCapacity(columns.count)
            for column in columns {
                precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
                let output = mtp.model.forwardWithHidden(tokens: column, caches: caches)
                let columnArgmax = argMax(output.logits, axis: -1).asType(.int32)
                // MTP-LAZY-COLUMNS: the predecessor completed each column
                // with a blocking `eval` before building the next, citing
                // mutable KV buffers observing a later version. On this
                // engine's verify roads that hazard has the same shape as
                // ordinary CHAINED DECODE, which already builds step N+1 on
                // step N's lazy tokens with no eval between: sliding rows
                // STAGE functionally (and the exact verify road writes
                // nothing in place), full rows' WRITE-022 in-kernel stores
                // are ordered by the write-fence value threaded through the
                // layer caches as graph edges, and every other cache
                // transition is a functional rebind. Profiled at depth 3,
                // the per-column eval made `v2.mtp.verify.build` 111.6 ms of
                // a ~140 ms round; the whole round now evaluates once at its
                // asyncEval boundary. Committed-token parity vs the serial
                // arm stays the gate for this change (8/8 bit-identical).
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
        }

        return (argmax, hidden, mtpCompactVerifyRoots(caches))
    }
}
