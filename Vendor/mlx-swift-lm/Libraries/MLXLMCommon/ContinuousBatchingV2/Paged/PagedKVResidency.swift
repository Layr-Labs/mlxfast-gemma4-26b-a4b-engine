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

/// Shared paged slabs (`PagedKVBackend`): a row's storage is whatever
/// `PagedKVPool.pageDemand` reserves for it, in whole pages.
///
/// The charge is derived from `pageDemand` rather than restated, so the
/// ledger cannot drift from the reservation the pool actually takes. That
/// matters in both directions:
///
/// * UNDER-charging is a free-list underflow — `PagedKVGroup.allocatePage`
///   traps, which is a daemon abort under load, not a rejected request.
/// * OVER-charging rejects requests the pool can serve. A one-token request
///   on gemma-4 needs ONE 16-token page per windowed layer, not the 1,024
///   rows of the window (nor the full width of the ring). Do not restate the
///   ring's width here and do not assume one: it is `ringPageCount`'s to
///   define and it has already moved once under this file's feet (WS-1.2
///   took gemma-4's from 97 pages to 65 once `PagedLayerCache` gathered
///   BEFORE writing). Deriving the charge from `pageDemand` is what kept
///   this file correct across that change without an edit, and is why it
///   needs none for the next one.
public struct CBv2PagedKVResidency: CBv2KVResidencyPolicy {
    public let config: PagedKVPoolConfig

    public init(config: PagedKVPoolConfig) {
        self.config = config
    }

    /// Pages come out of the free list whole, so a row's occupancy always
    /// lands on a page boundary.
    public var rowGranularity: Int { max(1, config.pageSize) }

    public func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int? {
        guard tokens >= 0, config.pageSize > 0 else { return nil }
        // `pageDemand` and `ringPageCount` do UNCHECKED arithmetic;
        // `PagedKVPool.init` refuses geometries that overflow it, but this
        // policy must stay total for callers that build it without a pool
        // (probes, tests). Mirror exactly what `ringPageCount` computes —
        // `max(maxWindowExposure + maxSpeculativeSpan, maxPrefillChunk)`
        // rounded up to a page — because a guard that checks a term the
        // formula no longer has protects nothing. It used to check
        // `window - 1 + maxPrefillChunk`, which let `window == Int.max` with
        // a one-token chunk through to a trap inside `window + span`.
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
    /// Paged rows are page-capped, never ring-committed — see
    /// `CBv2PagedKVResidency`.
    public var kvResidency: any CBv2KVResidencyPolicy {
        CBv2PagedKVResidency(config: pool.config)
    }
}
