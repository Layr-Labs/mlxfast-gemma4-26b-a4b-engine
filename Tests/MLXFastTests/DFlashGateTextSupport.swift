import Foundation
import Testing

// SHARED TEXT-ASSERTION SUPPORT for the checked-in source/fixture suites.
//
// Originally extracted from DFlashTrackTests.swift at the 2026-08-13 repo split
// to host the workflow/script/manifest assertion helpers. After the
// 2026-08-13 strip to the engine tree (removal of `.github/**`,
// `benchmark*.json`, and the `benchmark*.sh` layer, which now live with the
// benchd benchmarker), the workflow/manifest suites that used those helpers
// were excised with the machinery they tested. What remains is the pair of
// generic checked-in-file readers still used by the surviving source-layout
// suites (e.g. Gemma4MTPBackboneLayoutTests). The `S = DFlashGateTextSupport`
// typealias is kept at those call sites, so the name is retained.
//
// Nothing here needs a model, hidden material, or network access: every helper
// reads a checked-in file relative to the repository root.

enum DFlashGateTextSupport {
    static func text(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    static func json(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }
}
