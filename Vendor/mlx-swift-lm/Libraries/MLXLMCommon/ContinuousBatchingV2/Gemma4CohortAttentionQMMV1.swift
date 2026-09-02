// Copyright © 2026 Apple Inc.

/// Exact ranked-decode gate for the cohort-wide attention-projection experiment.
///
/// The existing attention path projects eight decode rows as independent GEMVs.
/// This experiment instead keeps the cohort intact so one quantized matrix
/// operation can amortize each projection's packed weight plane across all rows.
enum Gemma4CohortAttentionQMMV1 {
    static let batchSize = 8
    static let sequenceLength = 1

    @inline(__always)
    static func shouldUse(batchSize: Int, sequenceLength: Int) -> Bool {
        batchSize == Self.batchSize && sequenceLength == Self.sequenceLength
    }
}
