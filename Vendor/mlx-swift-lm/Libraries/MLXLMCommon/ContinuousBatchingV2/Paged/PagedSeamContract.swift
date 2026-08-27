// PagedSeamContract.swift
// Declarations for PagedKVPool <-> PagedSequenceKV <-> PagedLayerCache.
// Ranked KV is contiguous; this file is not on the scored path.

import Foundation
import MLX

// MARK: - Speculative span

/// Max token positions one speculative round can write past the confirmed frontier.
public enum CBv2PagedSpeculation {

    static let declaredSpan = 8

    public static let maxSpeculativeSpan: Int = {
        assertSpanCoversMTPBound()
        return declaredSpan
    }()

    static var spanCoversMTPBound: Bool {
        declaredSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1
    }

    static func assertSpanCoversMTPBound() {
        precondition(spanCoversMTPBound, spanDriftMessage)
    }

    static var spanDriftMessage: String {
        """
        CBv2PagedSpeculation.maxSpeculativeSpan (\(declaredSpan)) no longer \
        covers CBv2MTPConfig.testedMaxDraftTokens + 1 \
        (\(CBv2MTPConfig.testedMaxDraftTokens + 1)). Raise the span and re-check \
        PagedKVPool.ringPageCount before raising the MTP draft bound.
        """
    }
}

// MARK: - Windowed ring geometry

/// Same arithmetic as `PagedKVPool.ringPageCount`; tests bind the two copies.
public enum CBv2PagedRingGeometry {

    public static func attendableTokens(window: Int) -> Int {
        PagedSequenceKV.maxWindowExposure(window: window)
    }

    public static func requiredTokens(window: Int, maxPrefillChunk: Int) -> Int {
        max(
            attendableTokens(window: window) + CBv2PagedSpeculation.maxSpeculativeSpan,
            maxPrefillChunk)
    }

    public static func ringPageCount(window: Int, pageSize: Int, maxPrefillChunk: Int) -> Int {
        let required = requiredTokens(window: window, maxPrefillChunk: maxPrefillChunk)
        return (required + pageSize - 1) / pageSize
    }
}

// MARK: - Rectangular MTP verification

protocol CBv2MTPRectangularSerializing: AnyObject {
    var mtpSerializesRectangularAttention: Bool { get set }
}

extension CBv2LayerCache: CBv2MTPRectangularSerializing {}

// MARK: - Row-side speculative transaction

protocol CBv2PagedSpeculativeRow: AnyObject {
    var speculativeHeadroom: Int { get }
}

// MARK: - Windowed prefix adoption (WS-4.1)

public enum CBv2PagedWindowRestoreRefusal: Error, Equatable, CustomStringConvertible {

    case boundaryMismatch(snapshotEnd: Int, requested: Int)
    case inexactWindow(tokens: Int, required: Int)
    case notWindowed(requested: Int)

    public var description: String {
        switch self {
        case .boundaryMismatch(let snapshotEnd, let requested):
            return
                "windowed prefix refused: snapshot ends at absolute \(snapshotEnd) but the "
                + "adoption boundary is \(requested); installing it would place the donor's "
                + "keys at the wrong absolute positions"
        case .inexactWindow(let tokens, let required):
            return
                "windowed prefix refused: snapshot carries \(tokens) positions, the boundary "
                + "needs exactly \(required); a partial window is not an exact restore"
        case .notWindowed(let requested):
            return "windowed prefix refused: row at boundary \(requested) has no sliding window"
        }
    }
}

public struct CBv2PagedWindowSnapshot {

    public let keys: MLXArray
    public let values: MLXArray
    public let base: Int
    public let tokens: Int

    public var endBoundary: Int { base + tokens }

    public init?(keys: MLXArray, values: MLXArray, base: Int) {
        guard base >= 0,
            keys.ndim == 4, values.ndim == 4,
            keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(1) == values.dim(1),
            keys.dim(2) == values.dim(2),
            keys.dim(3) == values.dim(3),
            keys.dim(2) > 0
        else { return nil }
        self.keys = keys
        self.values = values
        self.base = base
        self.tokens = keys.dim(2)
    }

    public func requireAdmissible(at matchedBoundary: Int, window: Int?) throws {
        guard let window, window > 0 else {
            throw CBv2PagedWindowRestoreRefusal.notWindowed(requested: matchedBoundary)
        }
        guard endBoundary == matchedBoundary else {
            throw CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: endBoundary, requested: matchedBoundary)
        }
        let required = min(matchedBoundary, window)
        guard tokens == required else {
            throw CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: tokens, required: required)
        }
    }
}
