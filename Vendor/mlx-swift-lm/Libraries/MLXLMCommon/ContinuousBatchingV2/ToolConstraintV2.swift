// Copyright © 2026 Eigen Labs.
//
// Per-request inference-time token constraints for ContinuousBatchingV2.
// The engine owns only this format-neutral contract. Model families compile
// their actual output grammar (Gemma tool envelopes, JSON schema, and so on)
// before submission and hand the immutable machine to the sampler.

import Foundation

public enum CBv2TokenConstraintMode: String, Sendable {
    case none
    case required
    case named
}

public protocol CBv2TokenConstraint: AnyObject, Sendable {
    var mode: CBv2TokenConstraintMode { get }
    var initialState: Int { get }
    var maxTokens: Int { get }

    var fallbackTokenID: Int { get }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int]

    func nextState(state: Int, tokenID: Int) -> Int?
}

enum CBv2TokenConstraintFailure {
    static let impossibleState = "tool_constraint_impossible_state"
}
