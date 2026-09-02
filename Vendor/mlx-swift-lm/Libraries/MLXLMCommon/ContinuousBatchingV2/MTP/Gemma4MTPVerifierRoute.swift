/// Projection families admitted by the exact Gemma 4 B8 verifier.
public enum CBv2Gemma4MTPVerifierProjection: Sendable {
    case qkv
    case attentionOutput
    case denseGateUp
    case denseDown
    case expert
    case router
    case tiedHead
}

/// Every admitted projection either shares immutable weights across position-
/// major B8 cohorts or invokes the prebound ordinary B8 projection per cohort.
public enum CBv2Gemma4MTPVerifierProjectionStrategy: Sendable, Equatable {
    case combined
    case independentB8
    case sharedSerialReduction
}

/// Construction-time attention implementation. The shared-K/V case remains a
/// candidate until its exactness and width-specific performance gates pass.
public enum CBv2Gemma4MTPVerifierAttentionStrategy: Sendable, Equatable {
    case sharedKVExact
    case serializedDecode
}

public struct CBv2Gemma4MTPVerifierShape: Sendable, Hashable {
    public let batch: Int
    public let columns: Int

    public init(batch: Int, columns: Int) {
        self.batch = batch
        self.columns = columns
    }
}

/// Fixed route for the only certified physical widths, C2, C3, C4, C8, and C16.
public struct CBv2Gemma4MTPVerifierRoute: Sendable {
    public static let certifiedColumns: Set<Int> = [2, 3, 4, 8, 16]

    public static let production = Self(
        gateUpUsesMMA8: CBv2DenseMLPQMVV1.mma8GateUpEnabled,
        tiedHeadVersion: Gemma4MMAQuantizedGEMV.activeVersion)

    private let gateUpUsesMMA8: Bool
    private let tiedHeadVerifierCompatible: Bool

    private init(gateUpUsesMMA8: Bool, tiedHeadVersion: Int) {
        self.gateUpUsesMMA8 = gateUpUsesMMA8
        tiedHeadVerifierCompatible = Gemma4MMAQuantizedGEMV.isVerifierCompatible(
            version: tiedHeadVersion)
    }

    /// Pure construction-policy factory used by CPU-only contract tests.
    static func testing(
        gateUpUsesMMA8: Bool, tiedHeadVersion: Int
    ) -> Self {
        Self(gateUpUsesMMA8: gateUpUsesMMA8, tiedHeadVersion: tiedHeadVersion)
    }

    public func strategy(
        for projection: CBv2Gemma4MTPVerifierProjection,
        columns: Int
    ) -> CBv2Gemma4MTPVerifierProjectionStrategy? {
        guard Self.certifiedColumns.contains(columns) else { return nil }
        if columns == 8 || columns == 16 {
            return .sharedSerialReduction
        }
        switch projection {
        case .qkv, .attentionOutput, .denseDown, .expert:
            return .combined
        case .denseGateUp:
            return gateUpUsesMMA8 || columns == 4 ? .independentB8 : .combined
        case .router:
            return .independentB8
        case .tiedHead:
            return !tiedHeadVerifierCompatible || columns == 2
                ? .independentB8 : .combined
        }
    }

    public func attentionStrategy(
        columns: Int
    ) -> CBv2Gemma4MTPVerifierAttentionStrategy? {
        Self.certifiedColumns.contains(columns) ? .serializedDecode : nil
    }

    /// Explicit diagnostic candidate route. Normal serving continues to call
    /// `attentionStrategy(columns:)`, so an unmeasured kernel cannot silently
    /// replace the serial control.
    public func candidateAttentionStrategy(
        kind: CBv2LayerKind.Attention, columns: Int
    ) -> CBv2Gemma4MTPVerifierAttentionStrategy? {
        guard (2...4).contains(columns) else { return nil }
        switch kind {
        case .full:
            return .sharedKVExact
        case .slidingWindow:
            return .serializedDecode
        }
    }

    /// Preserve the exact four-row gate/up reduction geometry.
    public func gateUpRows(
        columns: Int
    ) -> CBv2DenseMLPQMVV1.GateUpVerifierRows? {
        guard (2...4).contains(columns) else { return nil }
        return .four
    }

    /// Preserve the existing two-way K split and two-tile output geometry.
    public func qkvGeometry(
        outputWidth: Int,
        columns: Int
    ) -> CBv2AttentionQKVMMA8V1.VerifierGeometry? {
        guard [1024, 2048, 4096, 8192].contains(outputWidth),
            (2...4).contains(columns)
        else { return nil }
        return .ks2Tile2
    }
}

/// Pure causal geometry shared by the binder and CPU-only route tests.
public enum Gemma4B1MTPFullAttentionGeometry {
    public static func visibleKeyLengths(
        historyLength: Int, columns: Int
    ) -> [Int] {
        precondition(historyLength >= 0)
        precondition(CBv2Gemma4MTPVerifierRoute.certifiedColumns.contains(columns))
        return (0..<columns).map { historyLength + $0 + 1 }
    }
}

public extension CBv2Gemma4MTPVerifierRoute {
    func supports(_ shape: CBv2Gemma4MTPVerifierShape) -> Bool {
        shape.batch == 1 && Self.certifiedColumns.contains(shape.columns)
    }
}
