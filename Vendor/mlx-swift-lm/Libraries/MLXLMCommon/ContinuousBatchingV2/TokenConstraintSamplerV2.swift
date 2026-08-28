// Copyright © 2026 Eigen Labs.
//
// Row-local host state and dense logit masking for CBv2 token constraints.
// Ordinary batches never enter this file's mask construction path.

import MLX

final class CBv2TokenConstraintSampler {
    private struct RowState {
        let constraint: any CBv2TokenConstraint
        var state: Int
        var consumed: Int
        var failure: String?
    }

    private var rows: [CBv2RequestID: RowState] = [:]

    var hasRows: Bool { !rows.isEmpty }

    func configure(_ descriptions: [CBv2SamplerRow]) {
        for row in descriptions {
            guard let constraint = row.tokenConstraint else {
                rows.removeValue(forKey: row.id)
                continue
            }
            if let existing = rows[row.id],
                ObjectIdentifier(existing.constraint) == ObjectIdentifier(constraint),
                existing.consumed == row.outputTokens.count
            {
                continue
            }

            var rebuilt = RowState(
                constraint: constraint,
                state: constraint.initialState,
                consumed: 0,
                failure: nil)
            for token in row.outputTokens {
                guard let next = constraint.nextState(
                    state: rebuilt.state, tokenID: token)
                else {
                    rebuilt.failure = CBv2TokenConstraintFailure.impossibleState
                    break
                }
                rebuilt.state = next
                rebuilt.consumed += 1
            }
            rows[row.id] = rebuilt
        }
    }

    /// Apply a per-row boolean mask to `[B, vocab]` logits. The common path
    /// returns the original array without allocating a mask.
    func mask(_ logits: MLXArray, requestIDs: [CBv2RequestID]) -> MLXArray {
        guard requestIDs.contains(where: { rows[$0] != nil }) else { return logits }

        let vocab = logits.dim(-1)
        var maskedRows: [MLXArray] = []
        maskedRows.reserveCapacity(requestIDs.count)
        for (rowIndex, id) in requestIDs.enumerated() {
            let rowLogits = logits[rowIndex ..< rowIndex + 1, 0...]
            guard var row = rows[id] else {
                maskedRows.append(rowLogits)
                continue
            }
            let remaining = max(0, row.constraint.maxTokens - row.consumed)
            var allowed = row.failure == nil
                ? row.constraint.allowedTokenIDs(
                    state: row.state, remainingTokens: remaining)
                : []
            if allowed.isEmpty || allowed.contains(where: { $0 < 0 || $0 >= vocab }) {
                row.failure = CBv2TokenConstraintFailure.impossibleState
                allowed = [row.constraint.fallbackTokenID]
                rows[id] = row
            }
            guard allowed.allSatisfy({ $0 >= 0 && $0 < vocab }) else {
                maskedRows.append(
                    MLXArray.full(
                        [1, vocab], values: MLXArray(-Float.infinity)))
                continue
            }

            if allowed.count <= vocab / 2 {
                let indices = MLXArray(allowed.map(Int32.init))
                    .reshaped([1, allowed.count])
                let values = takeAlong(rowLogits, indices, axis: -1)
                let base = MLXArray.full(
                    [1, vocab], values: MLXArray(-Float.infinity))
                maskedRows.append(
                    putAlong(base, indices, values: values, axis: -1))
            } else {
                var disallowed: [Int32] = []
                disallowed.reserveCapacity(vocab - allowed.count)
                var allowedIndex = 0
                for token in 0 ..< vocab {
                    while allowedIndex < allowed.count, allowed[allowedIndex] < token {
                        allowedIndex += 1
                    }
                    if allowedIndex == allowed.count || allowed[allowedIndex] != token {
                        disallowed.append(Int32(token))
                    }
                }
                guard !disallowed.isEmpty else {
                    maskedRows.append(rowLogits)
                    continue
                }
                let indices = MLXArray(disallowed)
                    .reshaped([1, disallowed.count])
                let values = MLXArray.full(
                    [1, disallowed.count], values: MLXArray(-Float.infinity))
                maskedRows.append(
                    putAlong(rowLogits, indices, values: values, axis: -1))
            }
        }
        return maskedRows.count == 1
            ? maskedRows[0]
            : concatenated(maskedRows, axis: 0)
    }

    func confirm(tokens: [Int], requestIDs: [CBv2RequestID]) {
        precondition(tokens.count == requestIDs.count)
        for (token, id) in zip(tokens, requestIDs) {
            guard var row = rows[id], row.failure == nil else { continue }
            guard let next = row.constraint.nextState(state: row.state, tokenID: token) else {
                row.failure = CBv2TokenConstraintFailure.impossibleState
                rows[id] = row
                continue
            }
            row.state = next
            row.consumed += 1
            rows[id] = row
        }
    }

    func failure(for id: CBv2RequestID) -> String? {
        rows[id]?.failure
    }

    func requestDidFinish(_ id: CBv2RequestID) {
        rows.removeValue(forKey: id)
    }
}
