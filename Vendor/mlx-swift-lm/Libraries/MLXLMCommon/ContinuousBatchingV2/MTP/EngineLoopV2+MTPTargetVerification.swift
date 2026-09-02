// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import MLX

extension EngineLoopV2 {

    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver,
        lazyColumns: Bool = false
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        mtpBuildTargetVerification(
            columns: columns, rowStates: rows.map { kvStates[$0.rec.id]! }, driver: mtp,
            lazyColumns: lazyColumns, deviceBase: nil)
    }

    func mtpBuildTargetVerification(
        columns: [MLXArray], rowStates: [[CBv2SequenceKV?]], driver mtp: CBv2MTPRoundDriver,
        lazyColumns: Bool, deviceBase: MLXArray?
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rowStates)
        let argmax: MLXArray
        let hidden: MLXArray

        var useRectangular = switch mtp.config.verificationMode {
        case .serialTarget: false
        case .rectangular: true
        case .automatic:
            columns.count * columns[0].dim(0) <= mtp.config.maxAutomaticRectangularTokens
        }
        let wideVerify = lazyColumns && CBv2MTPWideVerifyContext.enabled && columns.count > 1
        if wideVerify { useRectangular = true }

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
                output = CBv2MTPWideVerifyContext.with(columns: columns.count, base: deviceBase) {
                    mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
                }
            } else {
                output = mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
            }
            var logits = output.logits
            var lastHidden = Self.mtpRank3Hidden(output.lastHidden)
            if wideVerify {
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

    static func mtpRank3Hidden(_ hidden: MLXArray) -> MLXArray {
        hidden.ndim == 2 ? hidden.expandedDimensions(axis: 1) : hidden
    }
}
