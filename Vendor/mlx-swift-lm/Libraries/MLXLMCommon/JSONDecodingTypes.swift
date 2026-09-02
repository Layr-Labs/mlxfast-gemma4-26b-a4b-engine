// Copyright © 2024 Apple Inc.

import Foundation

// MARK: - IntOrIntArray

public struct IntOrIntArray: Codable, Sendable, Equatable {
    public let values: [Int]

    public init(_ values: [Int]) {
        self.values = values
    }

    public init(_ value: Int) {
        self.values = [value]
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let array = try? container.decode([Int].self) {
            self.values = array
        } else if let single = try? container.decode(Int.self) {
            self.values = [single]
        } else {
            throw DecodingError.typeMismatch(
                IntOrIntArray.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected Int or [Int]"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if values.count == 1 {
            try container.encode(values[0])
        } else {
            try container.encode(values)
        }
    }
}

// MARK: - StringOrNumber

public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else if let v = try? values.decode(Bool.self) {
            self = .bool(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        }
    }

    public func asInts() -> [Int]? {
        switch self {
        case .string(_): nil
        case .int(let v): [v]
        case .float(_): nil
        case .ints(let array): array
        case .floats(_): nil
        case .bool(_): nil
        }
    }

    public func asInt() -> Int? {
        switch self {
        case .string(_): nil
        case .int(let v): v
        case .float(_): nil
        case .ints(let array): array.count == 1 ? array[0] : nil
        case .floats(_): nil
        case .bool(let bool): bool ? 1 : 0
        }
    }

    public func asFloats() -> [Float]? {
        switch self {
        case .string(_): nil
        case .int(let v): [Float(v)]
        case .float(let float): [float]
        case .ints(let array): array.map { Float($0) }
        case .floats(let array): array
        case .bool(let bool): [bool ? 1.0 : 0.0]
        }
    }

    public func asFloat() -> Float? {
        switch self {
        case .string(_): nil
        case .int(let v): Float(v)
        case .float(let float): float
        case .ints(let array): array.count == 1 ? Float(array[0]) : nil
        case .floats(let array): array.count == 1 ? array[0] : nil
        case .bool(let bool): bool ? 1.0 : 0.0
        }
    }
}
