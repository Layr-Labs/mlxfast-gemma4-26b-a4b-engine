// BlockHasher.swift
//
// SHA-256 chain hashing at fixed token-block granularity for the v2 prefix
// cache (workstream D).
//
// The chain scheme has one architecture-independent, versioned binary
// representation:
//
//     h_i = SHA256(domain
//                 ‖ len(contract) ‖ contract
//                 ‖ len(scope) ‖ scope
//                 ‖ parent32
//                 ‖ u32be(blockIndex)
//                 ‖ u32be(token0) ‖ ...)
//
// Because each hash folds in its parent, h_i fingerprints the ENTIRE prefix
// up to and including block i. The versioned SSD layout deliberately does not
// read hashes produced by the retired textual tuple format.

import CryptoKit
import Foundation

/// Stateless block chain hasher. Configure once per (model, tenant scope)
/// and reuse; all methods are pure.
public struct CBv2BlockHasher: Sendable, Equatable {
    public static let version = "darkbloom-block-chain-v1"
    public static let domain = Data("darkbloom.prefix-block-chain.v1".utf8)

    /// Tokens per block. 256 matches the legacy checkpoint tier and the
    /// coarsest common divisor of the fleet's checkpoint ladders.
    public static let defaultBlockSize = 256

    public let blockSize: Int
    /// Prompt-contract identity derived from immutable tokenizer/template/config
    /// artifacts and semantic versions.
    public let promptContractID: String
    /// Authenticated cache-scope identity.
    public let scopeID: String

    public init(
        blockSize: Int = CBv2BlockHasher.defaultBlockSize,
        promptContractID: String = "",
        scopeID: String = ""
    ) {
        precondition(
            blockSize > 0 && UInt64(blockSize) <= UInt64(UInt32.max),
            "blockSize must fit UInt32")
        self.blockSize = blockSize
        self.promptContractID = promptContractID
        self.scopeID = scopeID
    }

    // MARK: - Single block

    /// Hash one block. `parent == nil` emits the all-zero 32-byte root.
    public func blockHash(
        parent: Data?,
        blockTokens: some Collection<Int>,
        blockIndex: Int
    ) -> Data {
        precondition(blockIndex >= 0 && blockIndex <= Int(UInt32.max))
        precondition(blockTokens.count == blockSize, "blockTokens must contain one full block")
        let contractID = Data(promptContractID.utf8)
        let scope = Data(scopeID.utf8)
        precondition(contractID.count <= Int(UInt32.max))
        precondition(scope.count <= Int(UInt32.max))
        var input = Data()
        input.reserveCapacity(
            Self.domain.count
                + MemoryLayout<UInt32>.size + contractID.count
                + MemoryLayout<UInt32>.size + scope.count
                + SHA256.Digest.byteCount
                + MemoryLayout<UInt32>.size
                + blockTokens.count * MemoryLayout<UInt32>.size)
        input.append(Self.domain)
        appendLengthPrefixed(contractID, to: &input)
        appendLengthPrefixed(scope, to: &input)
        if let parent {
            precondition(parent.count == 32)
            input.append(parent)
        } else {
            input.append(Data(repeating: 0, count: 32))
        }
        appendUInt32(UInt32(blockIndex), to: &input)
        for token in blockTokens {
            precondition(token >= 0 && UInt64(token) <= UInt64(UInt32.max))
            appendUInt32(UInt32(token), to: &input)
        }
        return Data(SHA256.hash(data: input))
    }

    // MARK: - Chains

    /// Number of whole blocks covered by `tokenCount` tokens.
    public func fullBlockCount(tokenCount: Int) -> Int {
        max(0, tokenCount) / blockSize
    }

    /// Largest number of whole blocks a LOOKUP over `tokenCount` tokens may
    /// match: at least one token must remain uncached so the engine always
    /// recomputes the final position for logits (vLLM last-token rule,
    /// report 08 §2). Equivalent to reserving the last block when the count
    /// is an exact block multiple.
    public func maxLookupBlocks(tokenCount: Int) -> Int {
        max(0, tokenCount - 1) / blockSize
    }

    /// Chain hashes `[h_1, …, h_k]` for the first `k` whole blocks of
    /// `tokens`, where `k = min(fullBlockCount, maxBlocks ?? ∞)`.
    /// `h_j` (index `j-1`) fingerprints the entire prefix `tokens[0 ..< j·bs]`.
    public func chainHashes(tokens: [Int], maxBlocks: Int? = nil) -> [Data] {
        var count = fullBlockCount(tokenCount: tokens.count)
        if let maxBlocks { count = min(count, max(0, maxBlocks)) }
        guard count > 0 else { return [] }

        var hashes: [Data] = []
        hashes.reserveCapacity(count)
        var parent: Data? = nil
        for i in 0 ..< count {
            let start = i * blockSize
            let hash = blockHash(
                parent: parent,
                blockTokens: tokens[start ..< start + blockSize],
                blockIndex: i)
            hashes.append(hash)
            parent = hash
        }
        return hashes
    }

    private func appendLengthPrefixed(_ value: Data, to output: inout Data) {
        precondition(value.count <= Int(UInt32.max))
        appendUInt32(UInt32(value.count), to: &output)
        output.append(value)
    }

    private func appendUInt32(_ value: UInt32, to output: inout Data) {
        var bigEndian = value.bigEndian
        withUnsafeBytes(of: &bigEndian) { output.append(contentsOf: $0) }
    }
}
