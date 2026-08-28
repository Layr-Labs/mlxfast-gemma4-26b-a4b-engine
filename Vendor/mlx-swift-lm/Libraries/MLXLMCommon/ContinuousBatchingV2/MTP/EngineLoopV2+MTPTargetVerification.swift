// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import MLX

extension EngineLoopV2 {

    /// Pack one row per equal-history group into the single causal target
    /// rectangle used by representative verification.
    static func mtpGroupedRepresentativeTokens(
        columns: [MLXArray], representativeIndices: [Int]
    ) -> MLXArray {
        precondition(!columns.isEmpty)
        let representatives = MLXArray(representativeIndices.map(Int32.init))
        return concatenated(columns.map { $0[representatives] }, axis: 1)
    }

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
        if let grouped = mtpBuildGroupedHistoryVerification(
            columns: columns, rows: rows, driver: mtp)
        {
            return grouped
        }
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })

        let installedDepth = columns.count - 1
        let usesInstalledVerifier =
            columns[0].dim(0) == 8
            && mtp.supportsInstalledVerification(
                batchSize: 8, draftDepth: installedDepth)
        if usesInstalledVerifier {
            // Both production cache backends certify this phase route. The
            // flag keeps each attention column identical to standalone L=1
            // decode while the weight-bound trunk runs once at M16.
            cacheProvider.setMTPRectangularVerification(true)
            defer {
                cacheProvider.setMTPRectangularVerification(false)
            }
            let tokens = concatenated(columns, axis: 1)
            let output = mtp.forwardInstalledMTPVerification(
                tokens: tokens, caches: caches)
            return (
                argMax(output.logits, axis: -1).asType(.int32),
                output.lastHidden,
                eagerCacheInnerState(caches))
        }

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
        }

        return (argmax, hidden, eagerCacheInnerState(caches))
    }

    /// Equal confirmed histories produce equal target activations and K/V
    /// tails. Score one representative per distinct history together, then
    /// append each representative's projected tail to only its equal-history
    /// peers. Every request keeps independent storage and finalize-time
    /// longest-prefix rollback.
    private func mtpBuildGroupedHistoryVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray])? {
        guard rows.count == 8 else { return nil }

        let groupIndices = mtp.draftGroupIndices(rows.map(\.rec))
        let groupCount = (groupIndices.max() ?? -1) + 1
        guard groupCount > 0, groupCount < rows.count else { return nil }
        let representativeIndices = (0..<groupCount).map { group in
            groupIndices.firstIndex(of: group)!
        }

        let states = rows.map { kvStates[$0.rec.id]! }
        let representativeStates = representativeIndices.map { states[$0] }
        let representativeCaches = eagerCaches(rowStates: representativeStates)
        let tokens = Self.mtpGroupedRepresentativeTokens(
            columns: columns, representativeIndices: representativeIndices)
        cacheProvider.setMTPRectangularVerification(true)
        defer { cacheProvider.setMTPRectangularVerification(false) }
        let output = mtp.model.forwardWithHidden(
            tokens: tokens, caches: representativeCaches)
        let representativeArgmax = argMax(output.logits, axis: -1).asType(.int32)
        let representativeHidden = output.lastHidden
        eval([representativeArgmax, representativeHidden]
            + eagerCacheInnerState(representativeCaches))

        let width = columns.count
        for layer in states[0].indices {
            for group in 0..<groupCount {
                let representativeIndex = representativeIndices[group]
                guard let source = states[representativeIndex][layer] else { continue }
                let snapshot = source.snapshot()
                let retained = snapshot.keys.dim(2)
                precondition(retained >= width)
                let newKeys =
                    snapshot.keys[0..., 0..., (retained - width)..<retained, 0...]
                let newValues =
                    snapshot.values[0..., 0..., (retained - width)..<retained, 0...]
                for peer in rows.indices
                where groupIndices[peer] == group && peer != representativeIndex {
                    guard let destination = states[peer][layer] else { continue }
                    _ = destination.update(keys: newKeys, values: newValues)
                }
            }
        }

        let gather = MLXArray(groupIndices.map(Int32.init))
        let argmax = representativeArgmax[gather]
        let hidden = representativeHidden[gather]
        let allCaches = eagerCaches(rowStates: states)
        return (argmax, hidden, eagerCacheInnerState(allCaches))
    }
}
