// LayerCacheBankV2.swift
//
// The engine's `CBv2LayerCacheProvider` over PERSISTENT per-layer
// attending caches (WS-A `CBv2LayerCache` or WS-C `PagedLayerCache`).
//
// The loop asks for layer caches every step (including every chained
// decode step). Rebinding rows costs a host-integer positionOffsets
// rebuild on the contiguous backend, so the bank fingerprints the batch
// composition by row-object identity and calls the contract's canonical
// `setRows(_:)` ONLY when composition actually changed — the chained
// pure-decode hot path performs zero host rebuilds (the caches advance
// their offsets on-device), preserving WS-A's membership-change-only
// discipline (`CBv2CoreInstrumentation.positionOffsetsHostRebuilds`).
//
// Identity safety: the previous batch's row objects stay retained by the
// caches' `rows` arrays until the next `setRows`, so an ObjectIdentifier
// can never alias a deallocated row.
//
// Engine-thread-confined (no locking) — the loop is the only caller.

import Foundation

public protocol CBv2CompositionInvalidating: AnyObject {
    func invalidateBoundComposition()

    func releaseBoundRows()
}

public protocol CBv2PackedPrefillCapableCache: AnyObject {
    var keepsRowsIndependentWhenPacked: Bool { get }
}

public protocol CBv2MultimodalSpanCapableCache: CBv2SpanMaskBinding {
    var honorsSpanMaskContexts: Bool { get }
}

extension CBv2LayerCache: CBv2PackedPrefillCapableCache, CBv2MultimodalSpanCapableCache {
    public var keepsRowsIndependentWhenPacked: Bool { true }

    public var honorsSpanMaskContexts: Bool { true }
}

public protocol CBv2KVSourceChunkRetaining: AnyObject {
    func setRetainsChunkForBorrowers(_ retains: Bool)
}

public final class CBv2LayerCacheBank: CBv2LayerCacheProvider, CBv2CompositionInvalidating {

    private let caches: [any CBv2AttendingLayerCache]
    private var boundRowIdentity: [ObjectIdentifier] = []
    private var hasBound = false
    private var unifiedPositionLayerIndex: Int?

    public init(caches: [any CBv2AttendingLayerCache]) {
        self.caches = caches
        let contiguous = caches.compactMap { $0 as? CBv2LayerCache }
        if contiguous.count == caches.count,
            let canonical = contiguous.last(where: { $0.kind.sharesKVWithLayer == nil })
        {
            let state = CBv2PositionOffsetsState(rows: canonical.rows)
            unifiedPositionLayerIndex = canonical.layerIndex
            for cache in contiguous {
                cache.unifyPositionOffsets(with: state, advances: cache === canonical)
            }
        }
        var borrowedSources = Set<Int>()
        for cache in caches {
            guard let source = cache.kind.sharesKVWithLayer else { continue }
            borrowedSources.insert(source)
        }
        for cache in caches {
            guard let retaining = cache as? CBv2KVSourceChunkRetaining else { continue }
            retaining.setRetainsChunkForBorrowers(borrowedSources.contains(cache.layerIndex))
        }
    }

    public convenience init(layerKinds: [CBv2LayerKind], attentionSoftcap: Float? = nil) {
        self.init(
            caches: layerKinds.enumerated().map { index, kind in
                CBv2LayerCache(
                    layerIndex: index, kind: kind, attentionSoftcap: attentionSoftcap)
            })
    }

    public func invalidateBoundComposition() {
        hasBound = false
        boundRowIdentity = []
    }

    public func releaseBoundRows() {
        guard hasBound else { return }
        for cache in caches where cache.kind.sharesKVWithLayer == nil {
            bindRows([], to: cache)
        }
        hasBound = false
        boundRowIdentity = []
    }

    public var supportsMultimodalSpans: Bool {
        caches.allSatisfy {
            ($0 as? CBv2MultimodalSpanCapableCache)?.honorsSpanMaskContexts ?? false
        }
    }

    public var supportsPackedPrefill: Bool {
        caches.allSatisfy {
            ($0 as? CBv2PackedPrefillCapableCache)?.keepsRowsIndependentWhenPacked ?? false
        }
    }

    public var supportsPackedMultimodalSpans: Bool {
        caches.allSatisfy { $0 is CBv2PackedSpanMaskBinding }
    }

    public func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        let identity = rowStates.map { row -> ObjectIdentifier in
            guard let anchor = row.compactMap({ $0 }).first else {
                preconditionFailure("CBv2LayerCacheBank: row owns no storage at any layer")
            }
            return ObjectIdentifier(anchor)
        }
        if !hasBound || identity != boundRowIdentity {
            validateUnifiedPositionInvariant(rowStates)
            for (layer, cache) in caches.enumerated() {
                guard cache.kind.sharesKVWithLayer == nil else { continue }
                bindRows(
                    rowStates.map { states in
                        guard let state = states[layer] else {
                            preconditionFailure(
                                "CBv2LayerCacheBank: missing sequence state for layer \(layer)")
                        }
                        return state
                    },
                    to: cache)
            }
            boundRowIdentity = identity
            hasBound = true
        }
        return caches
    }

    private func validateUnifiedPositionInvariant(
        _ rowStates: [[CBv2SequenceKV?]]
    ) {
        guard let canonicalLayer = unifiedPositionLayerIndex else { return }
        for states in rowStates {
            guard let canonical = states[canonicalLayer] else {
                preconditionFailure("CBv2LayerCacheBank: missing canonical position state")
            }
            for cache in caches where cache.kind.sharesKVWithLayer == nil {
                guard let state = states[cache.layerIndex],
                    state.absoluteOffset == canonical.absoluteOffset
                else {
                    preconditionFailure(
                        "CBv2LayerCacheBank: layer positions diverged inside a unified bank")
                }
            }
        }
    }

    private func bindRows(
        _ rows: [CBv2SequenceKV], to cache: any CBv2AttendingLayerCache
    ) {
        guard let unifiedPositionLayerIndex,
            let contiguous = cache as? CBv2LayerCache
        else {
            cache.setRows(rows)
            return
        }
        contiguous.setRows(
            rows, rebuildPositionOffsets: cache.layerIndex == unifiedPositionLayerIndex)
    }
}
