// Copyright © 2026 Eigen Labs Inc.
//
// OpenAI wire-type → chat-template translation helpers. (Formerly
// `MLXBatchedEngineServerEngine+Translation`; the extensions are pure
// wire-type translation and survive the legacy engine's deletion.)

import Foundation
import MLXLMCommon

extension OpenAIChatMessage {
    /// Chat-template-ready dictionary (the shape `applyChatTemplate`
    /// consumes): role/content plus `name`, `tool_call_id`, `tool_calls`
    /// (arguments decoded to a mapping), and `reasoning_content`. Gemma4's
    /// renderer reads `tool_calls` from assistant history. Used by
    /// `MLXModelContainerEngine` for tool-history chats and
    /// `/apply-template`.
    func templateMessage() -> [String: any Sendable] {
        var entry: [String: any Sendable] = [
            "role": role.rawValue,
            "content": textContent,
        ]
        if let name {
            entry["name"] = name
        }
        if let toolCallID {
            entry["tool_call_id"] = toolCallID
        }
        if let toolCalls, !toolCalls.isEmpty {
            entry["tool_calls"] = toolCalls.map { call -> [String: any Sendable] in
                [
                    "id": call.id,
                    "type": call.type,
                    "function": [
                        "name": call.function.name,
                        // Decode the JSON-string arguments into an object so the
                        // chat template renders the `is mapping` branch. See
                        // `decodeToolCallArguments` for why a raw string corrupts
                        // Gemma multi-turn tool calls (double-brace rendering).
                        "arguments": decodeToolCallArguments(call.function.arguments),
                    ] as [String: any Sendable],
                ]
            }
        }
        if let reasoningContent {
            entry["reasoning_content"] = reasoningContent
        }
        return entry
    }
}
