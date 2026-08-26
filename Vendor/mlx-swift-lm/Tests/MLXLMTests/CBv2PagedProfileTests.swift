// CBv2PagedProfileTests.swift
//
// Env-gated runner for the paged-decode root-cause profiler (kernel-opt
// track). Run with:
//   DARKBLOOM_CBV2_PAGED_PROFILE=1 swift test --filter CBv2PagedProfileTests
//
// Prints markdown ablation tables (slab-size sweep, write/dispatch split,
// 24-layer GPT-OSS emulation, dispatch-overhead probe). Numbers land in
// docs/engine-v2/kernel-research.md.

import Foundation
import Testing
import XCTest

@testable import MLXLMCommon

@Suite("CBv2PagedProfile", .serialized)
struct CBv2PagedProfileTests {

    static let enabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PAGED_PROFILE"] == "1"

    @Test(.enabled(if: enabled))
    func slabScaling() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.slabScalingSuite()
        print("\n## Paged decode profile: slab-size sweep (1 layer, GPT-OSS shape)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func layerScaling() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.layerScalingSuite()
        print("\n## Paged decode profile: layer-count sweep (2 GiB pool)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func gptossEmulation() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.gptossEmulationSuite()
        print("\n## Paged decode profile: 24-layer GPT-OSS emulation (16 GiB pool)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    @Test(.enabled(if: enabled))
    func dispatchOverhead() throws {
        let profiler = PagedDecodeProfiler()
        let results = try profiler.dispatchOverheadSuite()
        print("\n## Paged decode profile: dispatch-overhead probe (ctx=16)\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }

    /// Always-on smoke: one tiny scenario per mode exercises the profiler
    /// plumbing in CI without meaningful wall time.
    @Test func profilerSmoke() throws {
        let profiler = PagedDecodeProfiler(warmupSteps: 1, timedSteps: 2)
        for mode in PagedDecodeProfiler.StepMode.allCases {
            let r = try profiler.measurePaged(
                label: "smoke", layerCount: 2, capacityBytes: 8 << 20, batch: 2,
                context: 32, mode: mode)
            #expect(r.msPerStep > 0)
        }
        let sdpa = profiler.measureContiguousSDPA(
            label: "smoke", layerCount: 2, batch: 2, context: 32)
        #expect(sdpa.msPerStep > 0)
    }
}

/// MTP verify-round attribution at gemma-4 26B geometry: paged vs contiguous,
/// per round phase. Run with
///   DARKBLOOM_CBV2_PAGED_PROFILE=1 swift test --filter CBv2PagedMTPRoundProfile
/// and knobs `DARKBLOOM_MTP_K` (default 3), `DARKBLOOM_MTP_CTX` (default
/// 1,024), `DARKBLOOM_MTP_STEPS` (default 20).
///
/// XCTest, not `@Test`, for historical reasons only. Every swift-testing case
/// in this package — including the four `CBv2PagedProfileTests` entries above
/// — used to be unreachable: `BenchCBv2Tests` depended on the `BenchCBv2`
/// EXECUTABLE target, so SwiftPM pointed the swift-testing pass at that
/// binary, which exits on `--test-bundle-path`; only XCTest classes ran. The
/// bench driver now lives in the `BenchCBv2Core` library and the tests import
/// that, so the swift-testing pass runs again and this can fold into the suite
/// above as one more `@Test(.enabled(if: enabled))`.
///
/// The bundle also needs `mlx.metallib` beside the xctest runner
/// (`.build/<config>/*PackageTests.xctest/Contents/MacOS/`) or every MLX op
/// traps on "Failed to load the default metallib" — CI does this
/// (.github/workflows/ci.yml, "Extract mlx.metallib"), a local checkout
/// does not.
final class CBv2PagedMTPRoundProfile: XCTestCase {

    private func knob(_ key: String, _ fallback: Int) -> Int {
        Int(ProcessInfo.processInfo.environment[key] ?? "") ?? fallback
    }

    func testMTPRoundAttribution() throws {
        try XCTSkipUnless(CBv2PagedProfileTests.enabled, "set DARKBLOOM_CBV2_PAGED_PROFILE=1")
        let k = knob("DARKBLOOM_MTP_K", 3)
        let context = knob("DARKBLOOM_MTP_CTX", 1024)
        let profiler = PagedDecodeProfiler(timedSteps: knob("DARKBLOOM_MTP_STEPS", 20))
        let results = try profiler.mtpRoundSuite(k: k, context: context)
        print("\n## MTP verify-round attribution (gemma4-30L, B=1, k=\(k), ctx=\(context))\n")
        print(PagedDecodeProfiler.markdownHeader)
        for r in results { print(r.markdownRow) }
        print("")
    }
}
