// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — paged storage policy for the admission ledger.
//
// PR#87 made admission charge every sliding-window layer its whole fixed
// ring. That is exactly right for the CONTIGUOUS backend, whose windowed
// rows allocate `MLXArray.zeros([1, kvHeads, window, headDim])` on their
// first write, and exactly wrong for the PAGED backend, which reserves
// `min(ceil(maxLength / pageSize), ringPageCount)` pages and therefore
// never commits a whole ring for a short row. This file is the paged half
// of that distinction; the seam itself is `CBv2KVResidencyPolicy`.

import Foundation

public struct CBv2PagedKVResidency: CBv2KVResidencyPolicy {
    public let config: PagedKVPoolConfig

    public init(config: PagedKVPoolConfig) {
        self.config = config
    }

    public var rowGranularity: Int { max(1, config.pageSize) }

    public func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int? {
        guard tokens >= 0, config.pageSize > 0 else { return nil }
        if case .slidingWindow(let window) = kind.attention {
            guard window > 0,
                let exposed = Self.add(
                    PagedSequenceKV.maxWindowExposure(window: window),
                    CBv2PagedSpeculation.maxSpeculativeSpan),
                Self.add(max(exposed, config.maxPrefillChunk), config.pageSize - 1) != nil
            else { return nil }
        }
        guard Self.add(tokens, config.pageSize - 1) != nil else { return nil }
        let pages = PagedKVPool.pageDemand(
            kind: kind, maxLength: tokens, config: config)
        let (rows, overflow) = pages.multipliedReportingOverflow(by: config.pageSize)
        return overflow ? nil : rows
    }

    private static func add(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }
}

extension PagedKVBackend {
    public var kvResidency: any CBv2KVResidencyPolicy {
        CBv2PagedKVResidency(config: pool.config)
    }
}
