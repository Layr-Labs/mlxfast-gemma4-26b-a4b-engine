// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    private struct ArgumentSchema {
        let names: Set<String>
        let allowsAdditional: Bool
    }

    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"
    public let alternateStartTags: [String] = ["<start_function_call>"]
    public let alternateEndTags: [String] = ["<end_function_call>"]

    private let quoteMarkers = ["<|\"|>", "<escape>"]
    private let wrapperTags = [
        "<|tool_call>",
        "<tool_call|>",
        "<start_function_call>",
        "<end_function_call>",
    ]

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present. Gemma 4 emits the newer `<|tool_call>` form,
        // while older Gemma-family templates use `<start_function_call>`.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        for tag in wrapperTags {
            text = text.replacingOccurrences(of: tag, with: "")
        }

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name (word characters until {)
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let rawFuncName = String(remaining[..<braceStart])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let funcName = resolveFunctionName(rawFuncName, tools: tools) else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        let argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])
        guard let arguments = parseArguments(argsStr, funcName: funcName, tools: tools)
        else { return nil }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    private func resolveFunctionName(
        _ rawName: String,
        tools: [[String: any Sendable]]?
    ) -> String? {
        guard !rawName.isEmpty else { return nil }
        guard let tools, !tools.isEmpty else { return rawName }

        let declaredNames = tools.compactMap { tool in
            (tool["function"] as? [String: any Sendable])?["name"] as? String
        }
        guard !declaredNames.isEmpty else { return rawName }
        if declaredNames.contains(rawName) { return rawName }

        // Repair only a unique, nearby declared name. This covers observed
        // Gemma token glitches without turning arbitrary text into a tool call.
        let likelyMatches = declaredNames.filter { declaredName in
            let maxDistance = declaredName.count < 5 ? 0 : max(1, declaredName.count / 4)
            return (declaredName.count >= 5 && rawName.hasPrefix(declaredName + " "))
                || editDistance(rawName, declaredName) <= maxDistance
        }
        return likelyMatches.count == 1 ? likelyMatches[0] : nil
    }

    private func editDistance(_ lhs: String, _ rhs: String) -> Int {
        let left = Array(lhs)
        let right = Array(rhs)
        var previous = Array(0...right.count)

        for (leftIndex, leftCharacter) in left.enumerated() {
            var current = [leftIndex + 1]
            current.reserveCapacity(right.count + 1)
            for (rightIndex, rightCharacter) in right.enumerated() {
                current.append(min(
                    current[rightIndex] + 1,
                    previous[rightIndex + 1] + 1,
                    previous[rightIndex] + (leftCharacter == rightCharacter ? 0 : 1)
                ))
            }
            previous = current
        }
        return previous[right.count]
    }

    private func parseArguments(
        _ text: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: JSONValue]? {
        let schema = argumentSchema(funcName: funcName, tools: tools)
        var strict = GemmaArgumentValueParser(text)
        if let parsed = strict.parseTopLevelArguments(),
            strict.isAtEnd,
            schema.allowsAdditional || parsed.keys.allSatisfy(schema.names.contains)
        {
            return parsed
        }

        // Historical unconstrained auto output is intentionally accepted by
        // the bounded compatibility parser below (missing markers, comma-
        // bearing bare strings). Required/named output never needs this path:
        // its sampler emits only the strict template language above.
        var arguments: [String: JSONValue] = [:]
        // Gemma occasionally omits its string markers. The schema lets us keep
        // a comma in `Boston, MA` while still splitting before `unit:`.
        for pair in splitTopLevel(text, separator: ",") {
            guard let colon = pair.firstIndex(of: ":") else { continue }
            let key = parseArgumentKey(pair[..<colon])
            let rawValue = pair[pair.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            guard schema.allowsAdditional || schema.names.contains(key) else { return nil }
            arguments[key] = JSONValue.from(parseValue(rawValue))
        }
        return arguments
    }

    private func argumentSchema(
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> ArgumentSchema {
        guard let function = tools?.lazy.compactMap({ tool in
            tool["function"] as? [String: any Sendable]
        }).first(where: { $0["name"] as? String == funcName }),
              let parameters = function["parameters"] as? [String: any Sendable]
        else {
            return ArgumentSchema(names: [], allowsAdditional: true)
        }
        let properties = parameters["properties"] as? [String: any Sendable]
        return ArgumentSchema(
            names: Set(properties?.keys.map { $0 } ?? []),
            allowsAdditional: parameters["additionalProperties"] as? Bool != false)
    }

    private func parseArgumentKey(_ rawKey: Substring) -> String {
        let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return decodeJSONString(key) ?? key
    }

    private func parseValue(_ rawValue: String) -> any Sendable {
        for marker in quoteMarkers where rawValue.hasPrefix(marker) && rawValue.hasSuffix(marker) {
            let start = rawValue.index(rawValue.startIndex, offsetBy: marker.count)
            let end = rawValue.index(rawValue.endIndex, offsetBy: -marker.count)
            return String(rawValue[start..<end])
        }
        if let string = decodeJSONString(rawValue) {
            return string
        }
        if let data = rawValue.data(using: .utf8),
           let json = deserializeJSON(data)
        {
            return json
        }
        return rawValue
    }

    private func decodeJSONString(_ rawValue: String) -> String? {
        guard rawValue.first == "\"", rawValue.last == "\"",
              let data = rawValue.data(using: .utf8)
        else { return nil }
        return try? JSONDecoder().decode(String.self, from: data)
    }

    private func splitTopLevel(
        _ text: String,
        separator: Character
    ) -> [String] {
        var result: [String] = []
        var start = text.startIndex
        var index = text.startIndex
        var depth = 0
        var inJSONString = false
        var isEscaped = false

        while index < text.endIndex {
            if !inJSONString,
               let marker = quoteMarkers.first(where: { text[index...].hasPrefix($0) })
            {
                let afterStart = text.index(index, offsetBy: marker.count)
                if let end = text.range(of: marker, range: afterStart..<text.endIndex) {
                    index = end.upperBound
                    continue
                }
            }

            if inJSONString {
                if isEscaped {
                    isEscaped = false
                } else if text[index] == "\\" {
                    isEscaped = true
                } else if text[index] == "\"" {
                    inJSONString = false
                }
                index = text.index(after: index)
                continue
            }

            if text[index] == "\"" {
                inJSONString = true
                index = text.index(after: index)
                continue
            }

            switch text[index] {
            case "{", "[":
                depth += 1
            case "}", "]":
                depth = max(0, depth - 1)
            case separator where depth == 0:
                let next = text.index(after: index)
                if startsArgument(in: text, at: next) {
                    result.append(String(text[start..<index]))
                    start = next
                }
            default:
                break
            }
            index = text.index(after: index)
        }

        result.append(String(text[start..<text.endIndex]))
        return result
    }

    private func startsArgument(
        in text: String,
        at start: String.Index
    ) -> Bool {
        var keyStart = start
        while keyStart < text.endIndex, text[keyStart].isWhitespace {
            keyStart = text.index(after: keyStart)
        }

        var index = keyStart
        var inJSONString = false
        var isEscaped = false
        while index < text.endIndex {
            let character = text[index]
            if inJSONString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    inJSONString = false
                }
            } else if character == "\"" {
                inJSONString = true
            } else if character == ":" {
                let key = parseArgumentKey(text[keyStart..<index])
                return !key.isEmpty
            } else if character == "," || character == "{" || character == "}" {
                return false
            }
            index = text.index(after: index)
        }
        return false
    }
}

/// Strict recursive parser for the exact value syntax produced by Gemma's
/// `format_argument(..., escape_keys=False)` macro. The older permissive
/// parser remains as a fallback only for historical unconstrained auto output.
private struct GemmaArgumentValueParser {
    private let text: String
    private var index: String.Index

    init(_ text: String) {
        self.text = text
        self.index = text.startIndex
    }

    var isAtEnd: Bool {
        var copy = self
        copy.skipWhitespace()
        return copy.index == text.endIndex
    }

    mutating func parseTopLevelArguments() -> [String: JSONValue]? {
        skipWhitespace()
        if index == text.endIndex { return [:] }
        var output: [String: JSONValue] = [:]
        while index < text.endIndex {
            guard let key = parseKey(), consume(":"),
                let value = parseValue()
            else { return nil }
            output[key] = value
            skipWhitespace()
            if index == text.endIndex { return output }
            guard consume(",") else { return nil }
        }
        return output
    }

    private mutating func parseValue() -> JSONValue? {
        skipWhitespace()
        for marker in ["<|\"|>", "<escape>"] where text[index...].hasPrefix(marker) {
            advance(marker.count)
            guard let range = text.range(of: marker, range: index ..< text.endIndex) else {
                return nil
            }
            let value = String(text[index ..< range.lowerBound])
            index = range.upperBound
            return .string(value)
        }
        if consume("{") {
            var object: [String: JSONValue] = [:]
            skipWhitespace()
            if consume("}") { return .object(object) }
            while true {
                guard let key = parseKey(), consume(":"),
                    let value = parseValue()
                else { return nil }
                object[key] = value
                if consume("}") { return .object(object) }
                guard consume(",") else { return nil }
            }
        }
        if consume("[") {
            var array: [JSONValue] = []
            skipWhitespace()
            if consume("]") { return .array(array) }
            while true {
                guard let value = parseValue() else { return nil }
                array.append(value)
                if consume("]") { return .array(array) }
                guard consume(",") else { return nil }
            }
        }
        if consume("true") { return .bool(true) }
        if consume("false") { return .bool(false) }
        if consume("null") { return .null }
        if text[index...].hasPrefix("\""), let string = parseJSONString() {
            return .string(string)
        }
        return parseNumber()
    }

    private mutating func parseKey() -> String? {
        skipWhitespace()
        if text[index...].hasPrefix("\"") { return parseJSONString() }
        let start = index
        while index < text.endIndex {
            let character = text[index]
            if character == ":" { break }
            if character == "," || character == "{" || character == "}" || character == "[" || character == "]" {
                return nil
            }
            index = text.index(after: index)
        }
        let key = text[start ..< index].trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    private mutating func parseJSONString() -> String? {
        let start = index
        guard consume("\"") else { return nil }
        var escaped = false
        while index < text.endIndex {
            let character = text[index]
            index = text.index(after: index)
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                let raw = String(text[start ..< index])
                return try? JSONDecoder().decode(String.self, from: Data(raw.utf8))
            }
        }
        return nil
    }

    private mutating func parseNumber() -> JSONValue? {
        let start = index
        while index < text.endIndex {
            let character = text[index]
            guard character.isNumber || character == "-" || character == "+"
                || character == "." || character == "e" || character == "E"
            else { break }
            index = text.index(after: index)
        }
        guard start != index else { return nil }
        let raw = String(text[start ..< index])
        if let integer = Int(raw) { return .int(integer) }
        if let number = Double(raw), number.isFinite { return .double(number) }
        return nil
    }

    private mutating func consume(_ literal: String) -> Bool {
        skipWhitespace()
        guard text[index...].hasPrefix(literal) else { return false }
        advance(literal.count)
        return true
    }

    private mutating func skipWhitespace() {
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
    }

    private mutating func advance(_ count: Int) {
        index = text.index(index, offsetBy: count)
    }
}
