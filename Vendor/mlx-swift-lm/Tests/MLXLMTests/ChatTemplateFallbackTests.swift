// Copyright © 2026 Eigen Labs Inc.
//
// Regression (Codex, PR #68 review — CLI chat-template fallback cascade):
// for models whose tokenizer config lacks an inline `chat_template`, the
// deleted v1 engine's `buildPrompt` tried a `chat_template.jinja` file next
// to the weights before degrading to the generic content-join. Routing the
// CLI through `MLXModelContainerEngine` (container prepare path) lost that
// leg — `LLMUserInputProcessor` fell straight through to the content-join,
// handing chat-tuned models a degenerate plain-text prompt. These tests pin
// the restored cascade: inline template → chat_template.jinja → content-join.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// The render failure a malformed shipped template produces (a stand-in for
/// a swift-jinja parse/render error).
private struct MalformedTemplateError: Error {}

/// Tokenizer with NO embedded chat template (plain `applyChatTemplate`
/// throws `missingChatTemplate`) that DOES support rendering an explicitly
/// provided template string — the exact shape of a checkpoint that ships
/// its template as `chat_template.jinja` instead of tokenizer_config.json.
/// Templates without the fixture marker fail to render (malformed).
private struct NoInlineTemplateTokenizer: MLXLMCommon.Tokenizer {
    static let jinjaTokens = [42, 43, 44]
    static let encodeTokens = [7]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { Self.encodeTokens }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        chatTemplate: String
    ) throws -> [Int] {
        guard chatTemplate.contains("fixture-template") else {
            throw MalformedTemplateError()
        }
        return Self.jinjaTokens
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        chatTemplate: String,
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }
}

struct ChatTemplateFallbackTests {

    private func makeModelDir(jinja: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("template-fallback-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let jinja {
            try Data(jinja.utf8)
                .write(to: dir.appendingPathComponent("chat_template.jinja"))
        }
        return dir
    }

    private func prepare(dir: URL) throws -> [Int] {
        let processor = LLMUserInputProcessor(
            tokenizer: NoInlineTemplateTokenizer(),
            configuration: ModelConfiguration(directory: dir),
            messageGenerator: DefaultMessageGenerator())
        let input = try processor.prepare(
            input: UserInput(chat: [.user("hello")]))
        return input.text.tokens.asArray(Int.self)
    }

    @Test("missing inline template falls back to chat_template.jinja, not the content-join")
    func jinjaFileFallback() throws {
        let dir = try makeModelDir(jinja: "{# fixture-template #}{{ messages }}")
        defer { try? FileManager.default.removeItem(at: dir) }
        // Pre-fix: prepare() skipped the jinja file and content-joined
        // (encode-based tokens). With the cascade restored the explicit
        // template renders.
        #expect(try prepare(dir: dir) == NoInlineTemplateTokenizer.jinjaTokens)
    }

    @Test("no inline template and no jinja file still degrades to the content-join")
    func contentJoinLastResort() throws {
        let dir = try makeModelDir(jinja: nil)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try prepare(dir: dir) == NoInlineTemplateTokenizer.encodeTokens)
    }

    @Test("a PRESENT but malformed chat_template.jinja surfaces the render error, never the content-join")
    func malformedJinjaSurfacesError() throws {
        // Regression (Codex, follow-up round): with `try?` around the jinja
        // render, a present-but-malformed template was silently treated as
        // "no template" and degraded to the content-join — masking a real
        // model-packaging bug behind degenerate output. The render error
        // must propagate.
        let dir = try makeModelDir(jinja: "{% broken")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: MalformedTemplateError.self) {
            _ = try prepare(dir: dir)
        }
    }
}
