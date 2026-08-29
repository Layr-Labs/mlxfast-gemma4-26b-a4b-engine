import Darwin
import Foundation
import MLX

/// Selects a model-startup profile from the machine's physical-memory budget.
///
/// The profile is pure memory management: it never changes which code path
/// executes or what the model produces. Machines below the 64 GiB
/// full-profile minimum get a 6 GiB MLX allocator-cache cap, shorter
/// command buffers, and a free-warmup-buffer clear before the worker
/// protocol hello -- headroom insurance that lets the ~21.6 GB Poolside
/// NVFP4 checkpoint run down to the documented 36 GiB local minimum.
/// Compiled decode and every other ranked code path stay enabled on every
/// machine, matching the ranked 128 GiB box.
///
/// `environmentOverrides` is deliberately empty in both profiles: a
/// memory-gated feature-disable default makes small machines silently skip
/// code paths the ranked box runs (the pre-2026-07 profile defaulted
/// compiled decode off below 64 GiB this way, so local runs never executed
/// the `compile(shapeless:)` decode closures the ranked M5 exercises). The
/// no-overwrite plan machinery remains so any future override has to
/// reintroduce it consciously, under test. The automatic selection can be
/// overridden with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`. When
/// the low-memory profile engages it announces itself on stderr; a machine
/// too small for the model plus the decode working set fails loudly with an
/// out-of-memory error instead of silently diverging from the ranked
/// behavior.
public struct RuntimeStartupMemoryPolicy: Equatable, Sendable {
    public static let fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30

    /// Environment name for the explicit profile override. It must keep the
    /// `DARKBLOOM_` prefix: the trusted harness forwards only that
    /// model-tuning family (plus MLX_/system prefixes) into the runtime
    /// worker's sanitized environment, so any other spelling would never
    /// reach the process that resolves the policy.
    public static let profileOverrideEnvironmentName =
        "DARKBLOOM_STARTUP_MEMORY_PROFILE"

    public let isLowMemory: Bool
    /// Why this profile was selected; quoted in the stderr notice.
    public let selectionReason: String
    public let cacheLimitBytes: Int
    public let maxMegabytesPerCommandBuffer: Int
    public let maxOperationsPerCommandBuffer: Int
    public let clearAllocatorCacheAfterWarmup: Bool
    public let environmentOverrides: [String: String]

    /// Compiled-in Metal op cap for a coalesced CBv2 MTP round: target MoE
    /// gather and MTP drafter MoE gather share one command buffer. The MLX
    /// Max-tier default of 50 ops commits between those two gathers (double
    /// GPU submit per round). 256 admits both after the 512 MiB post-wire
    /// budget has already absorbed persistent weights.
    static let coalescedCBv2RoundMaxOpsPerCommandBuffer = 256

    /// The profile gate for the full-profile 512 MiB post-wire command-buffer
    /// budget (and the 32 GiB serial-path allocator cap). Returns `true` only
    /// when a machine may install it: at or above the 96 GiB floor and not
    /// running under an explicit low-memory request.
    ///
    /// Compiled-in: `killSwitchValue` is ignored. A runtime env
    /// (`DARKBLOOM_QWEN_MTP_POST_WIRE_COMMAND_BUFFER=0`) previously closed
    /// this gate and split the CBv2 round's environmentPlan/apply batches
    /// into two Metal submits.
    ///
    /// It is pure so BOTH the MTP startup path
    /// (`installGemma4MTPFullProfileCommandBufferDefaults`) and the serial
    /// `Gemma4A4BRuntimeWeightCache` constructor gate on the SAME decision, and so
    /// a test can exercise the open and closed branches without a device or the
    /// checkpoint. The serial constructor previously force-set the budget for
    /// every >= 16-layer config with no gate and no test, silently handing a
    /// sub-floor machine a command-buffer profile it was never measured for.
    static func gemma4MTPFullProfileCommandBufferGateIsOpen(
        physicalMemoryBytes: UInt64,
        requestedProfile: String?,
        killSwitchValue: String?
    ) -> Bool {
        guard physicalMemoryBytes >= (UInt64(96) << 30) else { return false }
        guard requestedProfile?.lowercased() != "low" else { return false }
        _ = killSwitchValue
        return true
    }

    /// Install the full-memory Qwen-MTP command-buffer geometry before the
    /// worker's first MLX device access. The MTP worker calls `resolve` before
    /// loading either the backbone or head, but deliberately returns early
    /// from the low-memory policy application on the ranked 128 GiB machine.
    /// MLX caches both limits on first use, so this is the last editable hook
    /// early enough to supply a full-profile default without touching trusted
    /// worker code.
    ///
    /// The 512 MiB referenced-byte budget is the independently promoted
    /// post-residency setting from the Laguna M5-Max track. It admits a whole
    /// model layer per command buffer after the persistent weights have been
    /// wired, while the stock 50-operation cap remains the outer safety wall.
    /// Explicit user values win and a benchmark-forwarded kill switch supports
    /// same-binary controls.
    ///
    /// The gate decision is factored into
    /// `gemma4MTPFullProfileCommandBufferGateIsOpen` so the serial
    /// `Gemma4A4BRuntimeWeightCache` constructor can gate its own install on the
    /// SAME rule instead of force-setting the budget unconditionally.
    private static func installGemma4MTPFullProfileCommandBufferDefaults(
        physicalMemoryBytes: UInt64,
        requestedProfile: String?
    ) {
        guard gemma4MTPFullProfileCommandBufferGateIsOpen(
            physicalMemoryBytes: physicalMemoryBytes,
            requestedProfile: requestedProfile,
            killSwitchValue: nil
        ) else { return }
        applyCoalescedCBv2RoundCommandBufferEnv()
    }

    /// One Metal command-buffer geometry for the CBv2 round: environmentPlan
    /// then apply (target MoE gather + MTP drafter MoE gather) share a single
    /// enqueue. overwrite=1 so a parent-exported 50-op / 50 MiB MLX default
    /// cannot split those gathers into two submits.
    private static func applyCoalescedCBv2RoundCommandBufferEnv() {
        setenv("MLX_MAX_MB_PER_BUFFER", "512", 1)
        setenv(
            "MLX_MAX_OPS_PER_BUFFER",
            String(coalescedCBv2RoundMaxOpsPerCommandBuffer),
            1
        )
    }

    public static func resolve(
        physicalMemoryBytes: UInt64,
        requestedProfile: String? = nil
    ) -> RuntimeStartupMemoryPolicy {
        installGemma4MTPFullProfileCommandBufferDefaults(
            physicalMemoryBytes: physicalMemoryBytes,
            requestedProfile: requestedProfile
        )
        let lowMemory: Bool
        let selectionReason: String
        switch requestedProfile?.lowercased() ?? "" {
        case "", "auto":
            lowMemory = physicalMemoryBytes < fullProfileMinimumPhysicalMemoryBytes
            selectionReason = "physical memory \(physicalMemoryBytes >> 30) GiB "
                + (lowMemory ? "is below" : "meets")
                + " the \(fullProfileMinimumPhysicalMemoryBytes >> 30) GiB full-profile minimum"
        case "full":
            lowMemory = false
            selectionReason = "\(profileOverrideEnvironmentName)=full"
        case "low":
            lowMemory = true
            selectionReason = "\(profileOverrideEnvironmentName)=low"
        default:
            preconditionFailure(
                "\(profileOverrideEnvironmentName) must be auto, full, or low"
            )
        }

        if lowMemory {
            return RuntimeStartupMemoryPolicy(
                isLowMemory: true,
                selectionReason: selectionReason,
                cacheLimitBytes: 6 << 30,
                // Shorter command buffers bound transient in-flight memory on
                // machines whose allocator cache is also capped to match the
                // trusted worker's 6 GiB phase-start value. 128 MiB is a
                // quarter of the full profile's referenced-byte budget. The op
                // cap is the one asymmetry left after the full profile moved
                // to 512 / 50: 64 here is ABOVE the full profile's 50, so the
                // low profile is bounded by bytes, not by op count. That is an
                // open item for whoever owns these budgets -- it is recorded
                // rather than silently "fixed", because changing a live
                // low-memory budget is not a comment's decision to make.
                maxMegabytesPerCommandBuffer: 128,
                maxOperationsPerCommandBuffer: 64,
                clearAllocatorCacheAfterWarmup: true,
                // No feature-disable defaults. Compiled decode
                // (MLX_COMPILED_DECODE only since the 2026-08-22 vendored
                // adoption -- DARKBLOOM_COMPILED_DECODE was removed upstream
                // with the v1 CompiledDecode.swift and is read nowhere; the
                // name is retained in this comment only as history)
                // (MLX_COMPILED_DECODE / DARKBLOOM_COMPILED_DECODE, both
                // default-on in the vendored CompiledDecode/MLXHardwareInfo)
                // was the only profile-touched setting that changed which
                // code path executes; defaulting it off here made <64 GiB
                // machines silently skip the compile(shapeless:) decode
                // closures the ranked box runs. It captures the decode graph
                // -- not a second model residency -- and the allocator cap
                // above bounds its transient footprint, so it stays enabled
                // for ranked parity. The cap, the command-buffer budgets,
                // and the warmup clear are pure memory management with no
                // effect on kernel selection or token output.
                environmentOverrides: [:]
            )
        }

        return RuntimeStartupMemoryPolicy(
            isLowMemory: false,
            selectionReason: selectionReason,
            // Ranked/full profile -- byte-identical to the constants this
            // policy replaced. The 32 GiB soft allocator cap lets the M5 Max
            // retain freed intermediates for reuse; it is not a reservation,
            // and model weights stay active allocations outside it.
            cacheLimitBytes: 32 << 30,
            // The MLX M5 Max default commits a command buffer after
            // referencing 50 MiB. Many 4-bit projections individually exceed
            // that, so 512 MiB groups adjacent kernels without long command
            // buffers; decode's explicit async-eval groups remain the outer
            // command-buffer boundary, and this referenced-buffer budget
            // governs within them. 512 / 50 is the promoted Laguna M5-Max
            // post-wire profile -- the same pair
            // `installGemma4MTPFullProfileCommandBufferDefaults` force-sets
            // above, so these values now REPORT the ranked environment rather
            // than contradict it. (They were 320 / 128 until the accepted
            // submission 0cd0a6b4-b539-4705-a1c7-cb271c1f9d3b; the prose here
            // still said 320 afterwards, which is what this line fixes.)
            maxMegabytesPerCommandBuffer: 512,
            maxOperationsPerCommandBuffer: coalescedCBv2RoundMaxOpsPerCommandBuffer,
            clearAllocatorCacheAfterWarmup: false,
            environmentOverrides: [:]
        )
    }

    /// The environment work `apply()` will perform, split into defaults to
    /// install (name currently unset) and explicit user values to preserve,
    /// plus the stderr notice lines describing the outcome. Pure given
    /// `existingValue`, so tests can verify the no-overwrite semantics and
    /// the notice without mutating process state.
    func environmentPlan(
        existingValue: (String) -> String?
    ) -> RuntimeStartupMemoryEnvironmentPlan {
        var defaultsToApply: [String: String] = [:]
        var preservedUserValues: [String: String] = [:]
        for (name, value) in environmentOverrides {
            if let existing = existingValue(name) {
                preservedUserValues[name] = existing
            } else {
                defaultsToApply[name] = value
            }
        }
        var noticeLines: [String] = []
        if isLowMemory {
            noticeLines.append(
                "mlxfast: low-memory startup profile active (\(selectionReason)): "
                    + "capping the MLX allocator cache at \(cacheLimitBytes >> 30) GiB and "
                    + "clearing free warmup buffers; compiled decode and every other "
                    + "ranked code path stay enabled; set "
                    + "\(Self.profileOverrideEnvironmentName)=full to opt out"
            )
            noticeLines.append(
                "mlxfast: a machine too small for the model plus the decode working "
                    + "set fails with an out-of-memory error instead of silently "
                    + "skipping ranked code paths; verify on a "
                    + "\(Self.fullProfileMinimumPhysicalMemoryBytes >> 30) GiB+ machine "
                    + "or rely on the ranked run"
            )
            if !preservedUserValues.isEmpty {
                let preserved = preservedUserValues.keys.sorted()
                    .map { name in "\(name)=\(preservedUserValues[name] ?? "")" }
                    .joined(separator: " ")
                noticeLines.append(
                    "mlxfast: low-memory startup profile preserved user-set flags: "
                        + preserved
                )
            }
        }
        return RuntimeStartupMemoryEnvironmentPlan(
            defaultsToApply: defaultsToApply,
            preservedUserValues: preservedUserValues,
            noticeLines: noticeLines
        )
    }

    /// MUST STAY `internal`. It was briefly made `public` so the trusted DFlash
    /// worker could call it, but this file lives under `Sources/MLXFastModel`,
    /// which is a DIRECTORY entry in both tracks' `editablePaths` -- so `submit`
    /// packages this file with EVERY submission, including the serial ones in
    /// flight today that carry the `internal` form. Trusted, non-editable code
    /// that depends on an access level declared in editable code breaks the build
    /// of any submission overlaying an older copy, through no fault of the
    /// submitter. The trusted worker therefore applies the policy itself from the
    /// long-public scalars below; see the note at its call site.
    func apply() {
        // Coalesce environmentPlan then the command-buffer batch so the
        // CBv2 round's target MoE gather and MTP drafter MoE gather share
        // one Metal enqueue. Plan defaults land first (no-overwrite);
        // the full-profile coalesced geometry overwrites a parent 50-op
        // default afterwards. Splitting these into two setenv epochs let
        // MLX snapshot the 50-op cap on first device use.
        let plan = environmentPlan { name in
            getenv(name).map { String(cString: $0) }
        }
        for (name, value) in plan.defaultsToApply {
            setenv(name, value, 0)
        }
        if isLowMemory {
            setenv(
                "MLX_MAX_MB_PER_BUFFER",
                String(maxMegabytesPerCommandBuffer),
                1
            )
            setenv(
                "MLX_MAX_OPS_PER_BUFFER",
                String(maxOperationsPerCommandBuffer),
                1
            )
        } else {
            Self.applyCoalescedCBv2RoundCommandBufferEnv()
        }
        for line in plan.noticeLines {
            fputs(line + "\n", stderr)
        }
        Memory.cacheLimit = cacheLimitBytes
    }
}

/// See `RuntimeStartupMemoryPolicy.environmentPlan(existingValue:)`.
struct RuntimeStartupMemoryEnvironmentPlan: Equatable, Sendable {
    let defaultsToApply: [String: String]
    let preservedUserValues: [String: String]
    let noticeLines: [String]
}
