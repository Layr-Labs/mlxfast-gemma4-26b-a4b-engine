// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

/// Errors thrown by ``MLXModelContainerEngine``.
public enum MLXModelContainerEngineError: Error, LocalizedError, Equatable {
    /// A request carried `image_url` or `video_url` content. This engine
    /// flattens chat content to text (see `OpenAIChatMessage.chatMessage()`),
    /// so media cannot be served and is rejected instead of silently dropped.
    case mediaUnsupported

    /// A per-request `tool_call_parser` override resolved to a format that
    /// differs from the server-pinned one. The tool-call format is a
    /// server-level setting (pinned once at load); conflicting per-request
    /// overrides are rejected rather than applied by mutating shared model
    /// state (which raced across concurrent requests).
    case unsupportedToolCallParser(pinned: ToolCallFormat, requested: ToolCallFormat)

    public var errorDescription: String? {
        switch self {
        case .mediaUnsupported:
            return
                "This model server is text-only and cannot process image_url or video_url content. Send text-only messages."
        case .unsupportedToolCallParser(let pinned, let requested):
            return
                "This model server is pinned to the '\(pinned.rawValue)' tool-call parser; a per-request override to '\(requested.rawValue)' is not supported. Start the server with the desired --tool-call-parser instead."
        }
    }
}
