// Thin driver over the real enforcer in
// Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift so the
// hostile-archive suite exercises THAT source, not a copy of its rules.
//
// Built by tools/test-submission-security.sh with:
//   swiftc -O Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift \
//          tools/editable-surface-budget-cli/main.swift -o <bin>
// The enforcer imports Foundation only, so this compiles in ~a second without
// the MLX dependency graph. `swift build` still compiles the same file into
// MLXFastTrustedHarness; there is one implementation, two ways in.
//
// Usage:
//   editable-surface-budget verify CONTRACT_PATH
//   editable-surface-budget limits CONTRACT_PATH
//
// Exit: 0 verified / limits resolved, 1 exceeded or invalid (fail closed),
//       2 no contract on disk (the caller decides whether that is fatal).

import Foundation

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data(("editable-surface-budget: " + message + "\n").utf8))
    exit(code)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    fail("usage: editable-surface-budget verify|limits CONTRACT_PATH", code: 64)
}
let command = arguments[1]
let contractPath = arguments[2]

switch command {
case "limits":
    switch resolveEditableSurfaceBudgetLimits(contractPath: contractPath) {
    case .resolved(let limits):
        print("maxTotalBytes=\(limits.maxTotalBytes)")
        print("maxFileBytes=\(limits.maxFileBytes)")
        print("maxGrowthBytes=\(limits.maxGrowthBytes)")
        print("exemptPathMaxBytes=\(limits.exemptPathMaxBytes)")
        print("exemptPathMaxFileBytes=\(limits.exemptPathMaxFileBytes)")
    case .missingContract(let reason):
        fail("skipped: " + reason, code: 2)
    case .invalid(let reason):
        fail("invalid: " + reason, code: 1)
    }
case "verify":
    switch verifyEditableSurfaceByteBudget(contractPath: contractPath) {
    case .verified(let totalBytes, let fileCount):
        print("verified totalBytes=\(totalBytes) fileCount=\(fileCount)")
    case .skipped(let reason):
        fail("skipped: " + reason, code: 2)
    case .exceeded(let reason):
        fail("exceeded: " + reason, code: 1)
    }
default:
    fail("unknown command '\(command)'; expected verify or limits", code: 64)
}
