// CBv2WiredResidency.swift
//
// Installs a Metal residency-set capacity so the wired model weights are
// actually held resident by the queue's residency set.

import Cmlx
import Foundation
import MLX

/// MLX attaches one `MTLResidencySet` to its command queue and inserts every
/// non-heap buffer it allocates into it. The set's capacity starts at zero and
/// only `set_wired_limit` ever raises it, so with no caller the set stays empty
/// and every allocation lands in the unwired side table instead.
public enum CBv2WiredResidency {

    /// Default capacity on a machine large enough to hold the full profile.
    ///
    /// Deliberately UNDER the 13.8 GiB of transformed target weights. Every
    /// insert and erase that lands in the wired set costs a
    /// `MTLResidencySet.commit()`, so a capacity with headroom left over after
    /// load makes every per-round transient wire and unwire again. Sizing the
    /// set so it saturates on the immortal weight buffers means the weights
    /// are wired, the set is full for the rest of the process, and every
    /// transient afterwards takes the cheap unwired path with no commit.
    public static let fullProfileBytes = 13 << 30

    /// Machines below this floor keep the stock behaviour untouched.
    public static let physicalMemoryFloorBytes = 96 << 30

    /// Resolve the capacity to install, or nil to leave the set alone.
    ///
    /// `override` is a byte count. It exists so the capacity can be pinned on a
    /// machine below the floor for A/B measurement; "0" disables the install.
    public static func resolve(
        physicalMemoryBytes: UInt64,
        override: String?,
        maxRecommendedWorkingSetBytes: Int?
    ) -> Int? {
        var requested: Int
        if let override, let parsed = Int(override.trimmingCharacters(in: .whitespaces)) {
            guard parsed > 0 else { return nil }
            requested = parsed
        } else {
            guard physicalMemoryBytes >= UInt64(physicalMemoryFloorBytes) else { return nil }
            requested = fullProfileBytes
        }
        // `metal::set_wired_limit` refuses a capacity above the device's
        // recommended working set, so clamp rather than risk the throw.
        if let ceiling = maxRecommendedWorkingSetBytes, ceiling > 0 {
            requested = min(requested, ceiling / 2)
        }
        return requested > 0 ? requested : nil
    }

    /// Raise the residency-set capacity. Returns the previous capacity, or nil
    /// if nothing was installed.
    @discardableResult
    public static func install(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        override: String? = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_WIRED_RESIDENCY_BYTES"]
    ) -> Int? {
        guard
            let limit = resolve(
                physicalMemoryBytes: physicalMemoryBytes,
                override: override,
                maxRecommendedWorkingSetBytes: GPU.maxRecommendedWorkingSetBytes())
        else { return nil }
        var previous = size_t(0)
        guard mlx_set_wired_limit(&previous, size_t(limit)) == 0 else { return nil }
        return Int(previous)
    }
}
