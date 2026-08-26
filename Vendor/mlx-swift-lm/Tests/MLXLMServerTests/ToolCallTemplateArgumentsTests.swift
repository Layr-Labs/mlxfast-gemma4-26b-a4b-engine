// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

/// Regression tests for issue #249 at the MLXLMServer translation boundary.
///
/// `templateMessage()` must hand the Gemma chat template a tool-call
/// `arguments` *mapping* (which renders the `is mapping` branch → single brace
/// `command:<|"|>ls -la<|"|>`) rather than the raw OpenAI JSON *string* (which
/// renders the `is string` branch → corrupting double brace
/// `{{"command":"ls -la"}}`). The OpenAI wire contract is unchanged; only the
/// template-input shape is normalized.
struct ToolCallTemplateArgumentsTests {
    private func assistantWithToolCall(arguments: String) -> OpenAIChatMessage {
        OpenAIChatMessage(
            role: .assistant,
            content: .text(""),
            toolCalls: [
                OpenAIToolCall(
                    id: "call_1",
                    function: .init(name: "run_terminal", arguments: arguments)
                )
            ]
        )
    }

    private func renderedArguments(_ message: OpenAIChatMessage) throws -> any Sendable {
        let dict = message.templateMessage()
        let toolCalls = try #require(dict["tool_calls"] as? [[String: any Sendable]])
        let function = try #require(toolCalls.first?["function"] as? [String: any Sendable])
        return try #require(function["arguments"])
    }

    @Test("templateMessage decodes JSON-string arguments into a mapping")
    func decodesArgumentsIntoMapping() throws {
        let arguments = try renderedArguments(
            assistantWithToolCall(arguments: #"{"command":"ls -la"}"#))
        let mapping = try #require(arguments as? [String: any Sendable])
        #expect(mapping["command"] as? String == "ls -la")
        // Guards the regression: a String here re-enables the double-brace bug.
        #expect(!(arguments is String))
    }

    @Test("templateMessage keeps non-JSON arguments as a string")
    func keepsNonJSONArgumentsAsString() throws {
        let arguments = try renderedArguments(
            assistantWithToolCall(arguments: "not json"))
        #expect(arguments as? String == "not json")
    }
}

/// v0.7.5 one-engine: the CLI's single-request container engine must not
/// flatten tool-use history. `requiresTemplateMessages` decides when the
/// prompt renders through full `templateMessage()` dictionaries (tool
/// fields preserved) instead of role+content `Chat.Message`s; and
/// `templateMessage()` must carry every history field the chat template
/// reads.
struct ContainerEngineToolHistoryTests {

    @Test("plain chats stay on the structured chat path")
    func plainChatNeedsNoTemplateMessages() {
        let messages: [OpenAIChatMessage] = [
            .init(role: .system, content: .text("be brief")),
            .init(role: .user, content: .text("hi")),
            .init(role: .assistant, content: .text("hello")),
        ]
        #expect(!MLXModelContainerEngine.requiresTemplateMessages(messages))
    }

    @Test("assistant tool_calls history selects the template-dict path")
    func toolCallsSelectTemplateMessages() {
        let messages: [OpenAIChatMessage] = [
            .init(role: .user, content: .text("list files")),
            .init(
                role: .assistant, content: .text(""),
                toolCalls: [
                    OpenAIToolCall(
                        id: "call_1",
                        function: .init(name: "ls", arguments: "{}"))
                ]),
        ]
        #expect(MLXModelContainerEngine.requiresTemplateMessages(messages))
    }

    @Test("tool role responses (tool_call_id) select the template-dict path")
    func toolCallIDSelectsTemplateMessages() {
        let messages: [OpenAIChatMessage] = [
            .init(role: .tool, content: .text("file1 file2"), toolCallID: "call_1")
        ]
        #expect(MLXModelContainerEngine.requiresTemplateMessages(messages))
    }

    @Test("reasoning_content selects the template-dict path")
    func reasoningContentSelectsTemplateMessages() {
        let messages: [OpenAIChatMessage] = [
            .init(role: .assistant, content: .text("done"), reasoningContent: "chain")
        ]
        #expect(MLXModelContainerEngine.requiresTemplateMessages(messages))
    }

    @Test("the OpenAI name field (no tool fields) selects the template-dict path")
    func nameSelectsTemplateMessages() {
        // Regression (Codex, PR #68 review): a plain history whose only
        // non-role/content field is `name` (a named participant/function)
        // must NOT take the `chatMessage()` path — `Chat.Message` has no
        // `name` slot, so the template would never see it. templateMessage()
        // carries `name`, so the predicate must route here.
        let messages: [OpenAIChatMessage] = [
            .init(role: .user, content: .text("hi"), name: "alice")
        ]
        #expect(MLXModelContainerEngine.requiresTemplateMessages(messages))
        // And the field actually survives the translation.
        #expect(messages[0].templateMessage()["name"] as? String == "alice")
    }

    @Test("per-request tool_call_parser: nil / auto / matching are accepted")
    func toolParserOverrideAccepted() throws {
        // No override, or "auto", trusts the server-pinned format.
        try MLXModelContainerEngine.validateToolParserOverride(
            requested: nil, pinned: .harmony, modelType: "gpt_oss")
        try MLXModelContainerEngine.validateToolParserOverride(
            requested: "auto", pinned: .harmony, modelType: "gpt_oss")
        // An override that resolves to the SAME pinned format is fine.
        try MLXModelContainerEngine.validateToolParserOverride(
            requested: "harmony", pinned: .harmony, modelType: "gpt_oss")
    }

    @Test("per-request tool_call_parser: a conflicting override is REJECTED, not applied")
    func toolParserOverrideConflictRejected() {
        // Regression (Codex, PR #68 review): the CLI container engine must
        // not honor a per-request parser by mutating shared ModelContext
        // state (that raced across concurrent requests — request B's parser
        // could overwrite the config before request A's stream snapshotted
        // it). A conflicting override is rejected outright, exactly as the
        // deleted v1 batched adapter did.
        #expect(throws: MLXModelContainerEngineError.self) {
            try MLXModelContainerEngine.validateToolParserOverride(
                requested: "json", pinned: .harmony, modelType: "gpt_oss")
        }
    }

    @Test("templateMessage preserves every history field the template reads")
    func templateMessageCarriesAllHistoryFields() throws {
        let message = OpenAIChatMessage(
            role: .assistant,
            content: .text("calling a tool"),
            name: "helper",
            toolCallID: "call_9",
            toolCalls: [
                OpenAIToolCall(
                    id: "call_9",
                    function: .init(name: "run", arguments: #"{"cmd":"ls"}"#))
            ],
            reasoningContent: "the user wants a listing"
        )
        let dict = message.templateMessage()
        #expect(dict["role"] as? String == "assistant")
        #expect(dict["content"] as? String == "calling a tool")
        #expect(dict["name"] as? String == "helper")
        #expect(dict["tool_call_id"] as? String == "call_9")
        #expect(dict["reasoning_content"] as? String == "the user wants a listing")
        let calls = try #require(dict["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.first?["id"] as? String == "call_9")
    }
}
