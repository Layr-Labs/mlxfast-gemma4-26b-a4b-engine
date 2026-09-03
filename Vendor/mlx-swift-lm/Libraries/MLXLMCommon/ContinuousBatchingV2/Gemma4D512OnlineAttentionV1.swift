// Copyright © 2026 Apple Inc.

import Foundation

/// Admission gate for an experimental one-dispatch D512 full-attention kernel.
///
/// The experiment is intentionally limited to the ranked unit-decode topology.
/// The incumbent three-dispatch path remains the unconditional fallback until
/// the fused kernel proves bit-for-bit output parity.
enum Gemma4D512OnlineAttentionV1 {
    static let environmentKey = "DARKBLOOM_GEMMA4_D512_ONLINE_ATTENTION_V1"

    @inline(__always)
    static func isEnabled(
        batchSize: Int,
        sequenceLength: Int,
        headDimension: Int,
        queryHeadCount: Int,
        keyValueHeadCount: Int
    ) -> Bool {
        guard ProcessInfo.processInfo.environment[environmentKey] == "1" else {
            return false
        }
        return batchSize == 8
            && sequenceLength == 1
            && headDimension == 512
            && queryHeadCount == 16
            && keyValueHeadCount == 2
    }
}
