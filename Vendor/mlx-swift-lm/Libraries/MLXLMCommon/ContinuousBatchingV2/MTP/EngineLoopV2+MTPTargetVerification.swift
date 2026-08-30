// EngineLoopV2+MTPTargetVerification.swift
//
// Construction-bound target-authoritative scoring strategies for one known
// MTP draft chain.

import MLX

extension EngineLoopV2 {

    /// Explicit serial oracle. This is bound only for `.serialTarget`; it is
    /// never the recovery path of the certified rectangular production lane.
    func mtpBuildSerialTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        mtp.recordVerificationStrategy(rectangular: false)

        var argmaxColumns: [MLXArray] = []
        var hiddenColumns: [MLXArray] = []
        argmaxColumns.reserveCapacity(columns.count)
        hiddenColumns.reserveCapacity(columns.count)
        for column in columns {
            precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
            let output = mtp.model.forwardWithHidden(tokens: column, caches: caches)
            let columnArgmax = argMax(output.logits, axis: -1).asType(.int32)
            // Building several eager decode calls in one lazy graph can let
            // mutable KV buffers observe a later version. Complete each
            // canonical target step before constructing the next.
            eval([columnArgmax, output.lastHidden] + eagerCacheInnerState(caches))
            argmaxColumns.append(columnArgmax)
            hiddenColumns.append(output.lastHidden)
        }
        return (
            concatenated(argmaxColumns, axis: 1),
            concatenated(hiddenColumns, axis: 1),
            eagerCacheInnerState(caches))
    }

    /// Certified explicit rectangular lane. Engine construction has already
    /// proven and captured an all-cache controller, and the driver's callable
    /// is bound directly to this entrypoint. There is no eligibility check,
    /// allocation, fallback, or fallback accounting in this path.
    func mtpBuildCertifiedRectangularVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        mtp.recordVerificationStrategy(rectangular: true)
        cacheProvider.setCertifiedMTPRectangularVerification(true)
        defer { cacheProvider.setCertifiedMTPRectangularVerification(false) }
        let tokens = concatenated(columns, axis: 1)
        let output = mtp.model.forwardRectangularVerificationWithHidden(
            tokens: tokens, caches: caches)
        return (
            argMax(output.logits, axis: -1).asType(.int32),
            output.lastHidden,
            eagerCacheInnerState(caches))
    }

    /// Legacy/generic automatic integration lane. Its work envelope changes
    /// with the planned row count, so it retains runtime strategy selection
    /// and safe serial degradation. Production Gemma never binds this lane:
    /// its sealed B1 configs use explicit `.rectangular` verification.
    func mtpBuildGenericAutomaticVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let rectangularWithinEnvelope =
            columns.count * columns[0].dim(0)
            <= mtp.config.maxAutomaticRectangularTokens
        if rectangularWithinEnvelope,
            cacheProvider.supportsCertifiedMTPRectangularVerification
        {
            return mtpBuildCertifiedRectangularVerification(
                columns: columns, rows: rows, driver: mtp)
        }
        if rectangularWithinEnvelope {
            mtp.recordControllerFallback("rectangular_cache_unsupported")
        }
        return mtpBuildSerialTargetVerification(
            columns: columns, rows: rows, driver: mtp)
    }
}
