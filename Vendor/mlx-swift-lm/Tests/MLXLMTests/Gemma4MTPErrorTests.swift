// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

@Suite("Gemma4MTPError")
struct Gemma4MTPErrorTests {

    @Test func unsupportedTargetHasNonEmptyDescription() {
        let err = Gemma4MTPError.unsupportedTarget("FooModel")
        #expect(err.errorDescription != nil)
        #expect(err.errorDescription!.contains("FooModel"))
    }

    @Test func rebindForbiddenHasDescription() {
        let err = Gemma4MTPError.rebindForbidden
        #expect(err.errorDescription != nil)
        #expect(!(err.errorDescription ?? "").isEmpty)
    }

    @Test func incompatibleDrafterMentionsField() {
        let err = Gemma4MTPError.incompatibleDrafter(
            field: "backboneHiddenSize", drafter: "2560", target: "1536")
        let desc = err.errorDescription ?? ""
        #expect(desc.contains("backboneHiddenSize"))
        #expect(desc.contains("2560"))
        #expect(desc.contains("1536"))
    }

    @Test func invalidBlockSizeMentionsValue() {
        let err = Gemma4MTPError.invalidBlockSize(1)
        #expect((err.errorDescription ?? "").contains("1"))
    }

    @Test func drafterNotBoundHasDescription() {
        let err = Gemma4MTPError.drafterNotBound
        #expect(err.errorDescription != nil)
        #expect(!(err.errorDescription ?? "").isEmpty)
    }
}
