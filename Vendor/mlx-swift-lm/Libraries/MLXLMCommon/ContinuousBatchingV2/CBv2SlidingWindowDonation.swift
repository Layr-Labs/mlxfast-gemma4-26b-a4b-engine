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

public protocol CBv2SlidingWindowDonating: AnyObject, Sendable {

    var wantsSlidingWindowDonation: Bool { get }

    func donate(
        requestID: CBv2RequestID,
        tokens: [Int],
        snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        slidingSnapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind],
        cacheSalt: String?)
}
