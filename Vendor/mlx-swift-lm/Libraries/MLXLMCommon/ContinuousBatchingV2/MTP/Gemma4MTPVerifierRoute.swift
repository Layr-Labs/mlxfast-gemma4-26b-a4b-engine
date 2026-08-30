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
}

/// C2...C4 attention always retains ordinary decode geometry and ownership.
public enum CBv2Gemma4MTPVerifierAttentionStrategy: Sendable, Equatable {
    case serializedDecode
}

/// Fixed route for the only certified physical widths, C2 through C4.
public struct CBv2Gemma4MTPVerifierRoute: Sendable {
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
        guard (2...4).contains(columns) else { return nil }
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
        (2...4).contains(columns) ? .serializedDecode : nil
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
