// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

/// Sendable wrapper around the non-Sendable ``ToolCallProcessor`` for
/// capture in a streaming-completion Task closure. Only touched from
/// that single Task.
///
/// Extracted from the deleted `MLXBatchedEngineServerEngine+ToolCallParser`
/// (the legacy continuous-batching server adapter died with the v1 engine —
/// v0.7.5 one-engine): the handler itself is engine-agnostic and is the
/// tool-call stream parser the Darkbloom provider drives over its
/// ContinuousBatchingV2 generation events.
public final class BatchedToolStreamHandler: @unchecked Sendable {
    private let processor: ToolCallProcessor
    private var residualText: String?

    public init(format: ToolCallFormat, tools: [[String: any Sendable]]?) {
        self.processor = ToolCallProcessor(format: format, tools: tools)
    }

    public func processChunk(_ chunk: String) -> String? {
        processor.processChunk(chunk)
    }

    public func finish() -> [ToolCall] {
        residualText = processor.processEOS(returnBufferedText: true)
        return processor.toolCalls
    }

    public func takeResidualText() -> String? {
        defer { residualText = nil }
        return residualText
    }

    public var parseFailureCount: Int {
        processor.parseFailureCount
    }
}
