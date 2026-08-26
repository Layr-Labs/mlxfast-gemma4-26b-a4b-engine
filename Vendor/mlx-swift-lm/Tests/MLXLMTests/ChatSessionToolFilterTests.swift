// Copyright © 2026 Eigen Labs Inc.
//
// Regression (Codex, PR #68 follow-up round): `ChatSession` includes its
// declared tools in the `UserInput` (prompt template) but used to call
// `generateTask` WITHOUT passing them, so its streaming `ToolCallProcessor`
// stayed on the parse-anything path — a model emitting a JSON call to an
// UNDECLARED function name surfaced it as a real `.toolCall` (and, with
// `toolDispatch` set, would try to dispatch it). These tests drive a REAL
// `ChatSession` over a scripted one-hot model (deterministic output, tiny
// MLX compute) and assert the declared-tool allowlist reaches the parser.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// MARK: - Scripted fixtures

/// One token id per output fragment; decode is pure concatenation so
/// `NaiveStreamingDetokenizer` diffing is exact.
private let toolScriptFragments: [Int: String] = [
    1: "<tool_call>",
    2: #"{"name": "other_fn", "arguments": {}}"#,
    3: "</tool_call>",
]
/// The model's scripted output: a JSON-format tool call to `other_fn`.
private let toolScript = [1, 2, 3]
private let scriptVocabSize = 8

/// Deterministic model: emits `toolScript` via one-hot logits (argmax under
/// temperature 0 reproduces the script exactly).
private final class ScriptedToolCallModel: Module, LanguageModel {
    private var nextIndex = 0

    private func oneHotLogits(for token: Int) -> MLXArray {
        var row = [Float](repeating: 0, count: scriptVocabSize)
        row[token] = 10
        return MLXArray(row).reshaped([1, 1, scriptVocabSize])
    }

    private func nextLogits() -> MLXArray {
        let token = toolScript[min(nextIndex, toolScript.count - 1)]
        nextIndex += 1
        return oneHotLogits(for: token)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .logits(LMOutput(logits: nextLogits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nextLogits()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private struct ScriptFragmentTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [0] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { toolScriptFragments[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { toolScriptFragments[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [0] }
}

private struct ScriptInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        LMInput(tokens: MLXArray([0]))
    }
}

struct ChatSessionToolFilterTests {

    private func makeContext() -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: "test/scripted-tools"),
            model: ScriptedToolCallModel(),
            processor: ScriptInputProcessor(),
            tokenizer: ScriptFragmentTokenizer())
    }

    private func toolCalls(declaredTools: [ToolSpec]) async throws -> [ToolCall] {
        let session = ChatSession(
            makeContext(),
            generateParameters: GenerateParameters(maxTokens: toolScript.count, temperature: 0),
            tools: declaredTools)
        var calls: [ToolCall] = []
        for try await item in session.streamDetails(
            to: "call something", images: [], videos: [])
        {
            if let call = item.toolCall {
                calls.append(call)
            }
        }
        return calls
    }

    @Test("ChatSession does not surface a call to an UNDECLARED function as a tool call")
    func undeclaredToolNameFiltered() async throws {
        // Pre-fix: ChatSession passed its tools to the prompt template but
        // NOT to generateTask, so the streaming parser accepted `other_fn`
        // despite the session declaring only `declared_fn`.
        let declared: [ToolSpec] = [
            [
                "type": "function",
                "function": ["name": "declared_fn"] as [String: any Sendable],
            ]
        ]
        let calls = try await toolCalls(declaredTools: declared)
        #expect(
            calls.isEmpty,
            "an undeclared function name must not surface as a ChatSession tool call")
    }

    @Test("ChatSession still surfaces calls to declared functions")
    func declaredToolNameParses() async throws {
        let declared: [ToolSpec] = [
            [
                "type": "function",
                "function": ["name": "other_fn"] as [String: any Sendable],
            ]
        ]
        let calls = try await toolCalls(declaredTools: declared)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "other_fn")
    }
}
