import Foundation
@testable import MLXFastModel
import Testing

@Test
func startupMemoryPolicyProtectsDocumented36GiBLocalMachine() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    #expect(policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 6 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 128)
    #expect(policy.maxOperationsPerCommandBuffer == 64)
    #expect(policy.clearAllocatorCacheAfterWarmup)
    // Consistency contract: the low-memory profile is pure memory
    // management and must not default any feature flag off. A memory-gated
    // feature-disable default makes small machines silently skip code paths
    // the ranked box runs (the old profile defaulted the compiled-decode
    // flags off below 64 GiB exactly this way, so <64 GiB machines never
    // executed the compile(shapeless:) decode closures the ranked M5
    // exercises). If this expectation fails, a feature divergence has been
    // reintroduced; read the policy's doc comment before shipping it.
    #expect(policy.environmentOverrides.isEmpty)
}

// The low-memory profile no longer disables the compiled-decode flags, so
// both must be unread by the policy and remain default-on where the model
// code consults them (MLXLMCommon CompiledDecode/MLXHardwareInfo). This
// pins the flag spellings so a vendored rename cannot silently orphan the
// documented opt-out names.
@Test
func compiledDecodeFlagsStayReadableByModelSources() throws {
    let vendoredRuntimeDirectory = "Vendor/mlx-swift-lm/Libraries/MLXLMCommon"
    let sourceFiles = try FileManager.default
        .contentsOfDirectory(atPath: vendoredRuntimeDirectory)
        .filter { $0.hasSuffix(".swift") }
    #expect(!sourceFiles.isEmpty)
    let combinedSources = try sourceFiles
        .map {
            try String(
                contentsOfFile: "\(vendoredRuntimeDirectory)/\($0)",
                encoding: .utf8
            )
        }
        .joined(separator: "\n")
    // DARKBLOOM_COMPILED_DECODE was DROPPED from this list by the 2026-08-22
    // adoption of mlx-swift-lm main: upstream removed that darkbloom-specific
    // alias entirely (it lived in the v1 CompiledDecode.swift, deleted at
    // ffede00), and it is now read nowhere in the vendored tree. MLX_COMPILED_DECODE
    // survives, in MLXLMCommon/MLXHardwareInfo.swift, so the guard this test
    // exists to provide is still live for the flag that still exists.
    //
    // This test CAUGHT that removal rather than being loosened to accommodate
    // it -- which is the behaviour it was written for. The orphaned mention in
    // Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift is annotated in
    // place; see docs/gemma4-port-notes.md section 2.5.
    for name in ["MLX_COMPILED_DECODE"] {
        #expect(
            combinedSources.contains("\"\(name)\""),
            "\(name) is not read anywhere in \(vendoredRuntimeDirectory)"
        )
    }
}

// The full-profile 512 MiB post-wire command-buffer budget (and the serial
// path's 32 GiB allocator cap) is behind a profile gate. Exercise the gate
// decision directly: before this, `Qwen35RuntimeWeightCache.init` force-set
// the budget for every >= 16-layer config with no gate and no test, so a
// sub-64 GiB serial machine silently ran a command-buffer profile it was never
// measured for. The gate opens only for the ranked full profile.
@Test
func gemma4MTPFullProfileCommandBufferGateOpensOnlyForRankedFullProfile() {
    // Ranked box (128 GiB), no explicit request, kill switch unset -> open,
    // and the 96 GiB floor itself is open.
    #expect(
        RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: nil,
            killSwitchValue: nil
        )
    )
    #expect(
        RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(96) << 30,
            requestedProfile: "",
            killSwitchValue: nil
        )
    )
    // The previously-ungated hole: a sub-64 GiB serial box silently got 512.
    // The gate now closes it.
    #expect(
        !RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: nil,
            killSwitchValue: nil
        )
    )
    // 64 GiB is the full/low boundary for the STARTUP profile, but still below
    // the 96 GiB command-buffer floor, so the budget stays closed there.
    #expect(
        !RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(64) << 30,
            requestedProfile: nil,
            killSwitchValue: nil
        )
    )
    // An explicit low-memory request closes the gate even on a large box...
    #expect(
        !RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "low",
            killSwitchValue: nil
        )
    )
    #expect(
        !RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "LOW",
            killSwitchValue: nil
        )
    )
    // ...and so does the benchmark-forwarded kill switch.
    #expect(
        !RuntimeStartupMemoryPolicy.gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: nil,
            killSwitchValue: "0"
        )
    )
}

// Revert guard for the serial path: `Gemma4A4BRuntimeWeightCache.init` must
// route its 512 MiB / 32 GiB full-profile install through the gate, never
// force-set it unconditionally the way the submission commit 0c90733 did. If
// the gate call disappears, or the budget setenv moves out from behind it, this
// reddens. (Repointed from `Qwen35RuntimeWeights.swift` with the 2026-08-22
// harness port; the guard is on the model-cache constructor, whichever tower it
// loads.)
@Test
func gemmaWeightCacheGatesTheFullProfileCommandBufferInstall() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift",
        encoding: .utf8
    )
    let budgetLiteral = #"setenv("MLX_MAX_MB_PER_BUFFER", "512", 1)"#
    let gate = try #require(
        source.range(of: "gemma4MTPFullProfileCommandBufferGateIsOpen(")
    )
    let setBudget = try #require(source.range(of: budgetLiteral))
    // Exactly one place still sets the 512 MiB budget, and the gate is consulted
    // before it -- the setenv lives inside the gated branch.
    #expect(source.components(separatedBy: budgetLiteral).count - 1 == 1)
    #expect(gate.lowerBound < setBudget.lowerBound)
}

@Test
func startupMemoryPolicyKeepsRanked128GiBProfile() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(128) << 30
    )

    #expect(!policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 32 << 30)
    // 512 / 50, NOT the 320 / 128 this test pinned until 2026-08-19.
    // Accepted submission 0cd0a6b4-b539-4705-a1c7-cb271c1f9d3b (commit
    // 0c90733, `Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift`) moved
    // the full-profile command-buffer constants onto the promoted Laguna
    // M5-Max post-wire budget, which is the budget
    // `installGemma4MTPFullProfileCommandBufferDefaults` already force-sets on
    // every machine at or above 96 GiB -- so before that commit the struct
    // reported 320 / 128 while the ranked box actually ran 512 / 50, and the
    // policy contradicted itself inside one `resolve` call. The same
    // submission stopped `Qwen35RuntimeWeightCache.init` clobbering the
    // 512 MiB budget with the older 128 MiB serial default.
    //
    // These two fields have NO production reader on the full profile: every
    // consumer -- `LagunaRuntimeWeightCache.init`, both
    // `Gemma4RuntimeMTPWorker.applyQwenMTPStartupMemoryProfile`s, both
    // `Gemma4RuntimeDFlashWorker` startup paths -- reads them only inside an
    // `isLowMemory` branch. So the move changed no measured ranked behavior;
    // it made the reported full profile equal the environment the ranked
    // process actually gets. If these ever gain a full-profile reader, that
    // reader is the thing to review, not this pin.
    #expect(policy.maxMegabytesPerCommandBuffer == 512)
    #expect(policy.maxOperationsPerCommandBuffer == 50)
    #expect(!policy.clearAllocatorCacheAfterWarmup)
    #expect(policy.environmentOverrides.isEmpty)

    // The ranked path must stay silent and write no feature defaults.
    let plan = policy.environmentPlan { _ in nil }
    #expect(plan.defaultsToApply.isEmpty)
    #expect(plan.preservedUserValues.isEmpty)
    #expect(plan.noticeLines.isEmpty)
}

@Test
func startupMemoryPolicyUses64GiBAsFullProfileBoundary() {
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: (UInt64(64) << 30) - 1
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
}

@Test
func startupMemoryPolicyHonorsExplicitProfileRequest() {
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "full"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "low"
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "FULL"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "auto"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: ""
        ).isLowMemory
    )
    // The override must keep its DARKBLOOM_ prefix: the trusted worker
    // environment allowlist forwards that family, so renaming it (e.g. to
    // MLXFAST_*) would silently stop it from ever reaching the worker.
    #expect(
        RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
            == "DARKBLOOM_STARTUP_MEMORY_PROFILE"
    )
}

// The low-memory plan writes no environment defaults -- compiled decode and
// every other ranked code path stay enabled -- and its stderr notice states
// exactly what it does (cap + warmup clear), that ranked code paths stay
// enabled, and that a genuinely-too-small machine fails loudly with an
// out-of-memory error instead of silently skipping ranked code paths (the
// silent divergence the old feature-disable defaults caused).
@Test
func lowMemoryPlanWritesNoFeatureDefaultsAndAnnouncesRankedParity() throws {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    // Even with the historical compiled-decode flag names exported, the plan
    // neither writes nor tracks them: the profile no longer touches feature
    // flags at all.
    let plan = policy.environmentPlan { name in
        ["MLX_COMPILED_DECODE", "DARKBLOOM_COMPILED_DECODE"].contains(name)
            ? "1" : nil
    }

    #expect(plan.defaultsToApply.isEmpty)
    #expect(plan.preservedUserValues.isEmpty)
    try #require(plan.noticeLines.count == 2)
    #expect(plan.noticeLines[0].contains("low-memory startup profile active"))
    #expect(plan.noticeLines[0].contains("capping the MLX allocator cache at 6 GiB"))
    #expect(plan.noticeLines[0].contains(
        "compiled decode and every other ranked code path stay enabled"
    ))
    #expect(plan.noticeLines[0].contains("DARKBLOOM_STARTUP_MEMORY_PROFILE=full"))
    #expect(plan.noticeLines[1].contains("out-of-memory"))
    #expect(plan.noticeLines[1].contains("rely on the ranked run"))
    // Deliberately NOT asserting the machine-size phrasing in this line.
    // RuntimeStartupMemoryPolicy.swift is an editable path, so a submission
    // ships its own copy of this notice string. Pinning cosmetic guidance
    // wording here fails every submission whose snapshot predates a wording
    // change in trusted main, for no contract reason -- observed 2026-07-28,
    // when rewording "64 GiB+ machine" to "machine with more unified memory"
    // reddened in-flight submissions cb418c4c and others. The assertions that
    // remain (out-of-memory guidance, ranked-run fallback) are stable across
    // both wordings, and the noticeLines[0] assertions above are kept because
    // they encode the actual contract (no feature-disable defaults; compiled
    // decode stays enabled), not phrasing.
}
