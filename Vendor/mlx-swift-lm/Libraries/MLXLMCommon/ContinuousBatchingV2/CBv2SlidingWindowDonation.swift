// Copyright © 2026 Eigen Labs.
//
// Opt-in donor seam for a prefix cache that persists the donor's SLIDING
// rows as well as its full-attention blocks (the provider's WS-4.2 windowed
// sidecar).
//
// Why this is a SEPARATE protocol rather than another `CBv2PrefixCache`
// overload:
//
//   * Cost. A sliding snapshot is `windowCount × window` positions of real
//     K/V — 200 MiB on gemma-4 — and the feature is default-OFF. Folding it
//     into the frozen donation contract would make EVERY donation pay for a
//     capability almost no deployment has enabled. `wantsSlidingWindowDonation`
//     is asked once, on the engine thread, BEFORE anything is built, so a
//     conformer with the feature off costs exactly one existential cast.
//   * Contract. `CBv2PrefixCache`'s `snapshots` array is documented as nil at
//     every windowed / KV-shared layer, and several conformers rely on that.
//     Sliding rows travel in their OWN array here, so that contract is
//     untouched.

import Foundation
import MLX

/// A `CBv2PrefixCache` that can also persist the donor's sliding-window K/V.
///
/// The engine probes for this conformance on the donation path only. A cache
/// that does not conform, or that answers `false`, sees the ordinary
/// request-aware `donate(requestID:tokens:snapshots:layerKinds:cacheSalt:)`
/// and no behaviour change of any kind.
public protocol CBv2SlidingWindowDonating: AnyObject, Sendable {

    /// Whether this cache will actually persist sliding rows right now.
    ///
    /// Read ONCE per donation on the engine thread, before any snapshot is
    /// built, so answering `false` costs nothing beyond the property read.
    /// Implementations must answer from cheap, already-resolved state (an
    /// operator knob folded into construction), never from I/O.
    var wantsSlidingWindowDonation: Bool { get }

    /// Request-correlated donation carrying the donor's sliding rows beside
    /// the ordinary full-attention snapshots.
    ///
    /// `slidingSnapshots` is indexed exactly like `layerKinds` and is non-nil
    /// ONLY at storage-owning `.slidingWindow` layers: a KV-shared sliding
    /// layer borrows its source's storage, so donating it would double-write
    /// the same bytes. Entries are graph-built on the engine thread — same
    /// discipline as `snapshots` — and are NOT truncated to the donated token
    /// count: a sliding ring holds the last `window` positions ending at the
    /// row's own absolute offset, which is exactly what the cache persists.
    ///
    /// Called on the engine's donation queue, never the step thread.
    func donate(
        requestID: CBv2RequestID,
        tokens: [Int],
        snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        slidingSnapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind],
        cacheSalt: String?)
}
