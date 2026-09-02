// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

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
