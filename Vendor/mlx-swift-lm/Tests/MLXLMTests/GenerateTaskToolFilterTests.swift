// Copyright © 2026 Eigen Labs Inc.
//
// Regression (Codex, PR #68 review — declared-tool filtering in CLI
// streams): when a chat request supplies `tools`, the generation loop's
// streaming `ToolCallProcessor` must receive those schemas. The deleted v1
// batched adapter did this via `BatchedToolStreamHandler(format:tools:)`;
// the container path built its processor WITHOUT the request tools, so
// parsers (e.g. JSON) could surface arbitrary/undeclared function names as
// tool calls for tool-enabled requests. These tests drive the REAL
// `generateTask` loop with a scripted token iterator (no model weights, no
// MLX compute) and assert the declared-tool allowlist is enforced.

import Foundation
import MLXLMCommon
import Testing

// MARK: - Scripted fixtures

/// Token iterator that replays a fixed token script (the "model output").
private struct ScriptedTokenIterator: TokenIteratorProtocol {
    var tokens: [Int]
    var index = 0

    var maxTokens: Int? { nil }
    var tokenCount: Int { index }
    var promptPrefillTime: TimeInterval { 0 }

    mutating func next() -> Int? {
        guard index < tokens.count else { return nil }
        defer { index += 1 }
        return tokens[index]
    }
}

/// Tokenizer whose decode is a pure fragment concatenation — each token id
/// maps to a fixed text fragment, so `NaiveStreamingDetokenizer`'s
/// incremental diffing works exactly.
private struct FragmentTokenizer: MLXLMCommon.Tokenizer {
    let fragments: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { fragments[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { fragments[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { 999 }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { 998 }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

struct GenerateTaskToolFilterTests {

    /// Model output: a JSON-format tool call to `other_fn`.
    private static let fragments: [Int: String] = [
        1: "<tool_call>",
        2: #"{"name": "other_fn", "arguments": {}}"#,
        3: "</tool_call>",
    ]
    private static let script = [1, 2, 3]

    /// OpenAI-shaped tool spec declaring ONLY `declared_fn`.
    private static let declaredTools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "declared_fn",
                "description": "the only declared tool",
                "parameters": ["type": "object"] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    private func run(
        tools: [[String: any Sendable]]?
    ) async -> (toolCalls: [ToolCall], text: String) {
        let (stream, _) = generateTask(
            promptTokenCount: 1,
            modelConfiguration: ModelConfiguration(id: "test/scripted"),
            tokenizer: FragmentTokenizer(fragments: Self.fragments),
            iterator: ScriptedTokenIterator(tokens: Self.script),
            tools: tools
        )
        var toolCalls: [ToolCall] = []
        var text = ""
        for await generation in stream {
            switch generation {
            case .toolCall(let call): toolCalls.append(call)
            case .chunk(let chunk): text += chunk
            case .info: break
            }
        }
        return (toolCalls, text)
    }

    @Test("undeclared function name is NOT surfaced as a tool call when tools are declared")
    func undeclaredToolCallFiltered() async {
        // Pre-fix: `generateTask` built its ToolCallProcessor without the
        // request tools, so `other_fn` parsed as a real ToolCall despite the
        // request declaring only `declared_fn`.
        let (toolCalls, _) = await run(tools: Self.declaredTools)
        #expect(
            toolCalls.isEmpty,
            "a call to an undeclared function must not surface as a tool call")
    }

    @Test("declared function calls still parse as tool calls")
    func declaredToolCallStillParses() async {
        let declared: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": ["name": "other_fn"] as [String: any Sendable],
            ]
        ]
        let (toolCalls, _) = await run(tools: declared)
        #expect(toolCalls.count == 1)
        #expect(toolCalls.first?.function.name == "other_fn")
    }

    @Test("no declared tools (nil) keeps the historical parse-anything behavior")
    func nilToolsUnchanged() async {
        let (toolCalls, _) = await run(tools: nil)
        #expect(toolCalls.count == 1)
        #expect(toolCalls.first?.function.name == "other_fn")
    }
}
