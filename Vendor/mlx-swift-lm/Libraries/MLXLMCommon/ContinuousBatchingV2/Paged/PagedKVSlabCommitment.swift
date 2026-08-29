// PagedKVSlabCommitment.swift
//
// WHEN a paged pool's slabs become MLX-resident (D1) — and what happens
// when the commitment no longer fits the box.
//
// The problem
// -----------
// `PagedKVGroup` builds its two slabs with `MLXArray.zeros(...)`, which is a
// LAZY `Full` primitive: no Metal buffer exists until something evaluates it.
// The only thing that forced them resident at engine-construction time was an
// explicit `PagedKVPool.materializeSlabs()` call in the provider's production
// factory. That single call is what broke two-model co-residency on a 36 GiB
// box: the FIRST model's pool — idle, holding no KV for anyone — landed in
// `MLX.GPU.activeMemory` before the SECOND model's post-load headroom guard
// re-measured, so the second model measured 0.15 GiB against a 1 GiB
// serveable-KV minimum and was unloaded with a 503. An all-contiguous pair
// measured 2.40 GiB and served, because a contiguous grant is an admission
// CEILING that allocates as it goes, never a preallocation.
//
// So paged was being charged its worst case at construction while contiguous
// was charged as it went. This file removes that asymmetry.
//
// What is deferred, and what is NOT
// ---------------------------------
// Deferred: the ALLOCATION. `commitSlabs()` is the one place that evaluates
// them, and it runs at the pool's first admission rather than at construction.
//
// NOT deferred: the GUARANTEE. Every page a row can reach must be backed
// before the row exists, or `PagedSequenceKV.ensurePage` reaches a page whose
// buffer was never allocated. `PagedKVBackend.reserve` and
// `makeSequenceState` therefore commit the WHOLE pool — every group, both
// slabs — immediately after the admission charge succeeds and before any
// `PagedSequenceKV` is minted. An admission that throws `capacityExhausted`
// commits nothing, and an admission whose COMMIT throws is unwound — the
// page charge is released, the pool is left idle and unwired, and the next
// admission retries the commit from a clean ledger.
//
// Nothing else changes: `pageCount` is still fixed at `PagedKVPool.init`, the
// slabs are still `let`, and there is still no resize primitive. Lazy FIRST
// commitment is not a resize — it is the same buffer, allocated later.
//
// Why the ORDERING is correctness-neutral
// ---------------------------------------
// The slabs are passed to the write/decode Metal kernels as INPUTS, never as
// declared outputs. MLX's `eval_impl` walks the tape leaf-first, so a slab's
// `Full` primitive runs (allocating and zero-filling the buffer) BEFORE the
// custom kernel that const-casts and writes into it, with a real
// `memoryBarrier(BarrierScopeBuffers)` between them because the encoder sees
// the slab as a prior output and then as an input. `eval_impl` then detaches
// the primitive, so the buffer is never recomputed or re-zeroed and its
// identity is stable for the pool's lifetime — which is the only property the
// in-place write design actually requires (see pagedattention.metal,
// "In-place slab writes"). The one hazard, a `Full` surviving inside a
// `compile` tracer and re-zeroing on a later eval, cannot arise here: there
// is no tracing path at all. The compiled [B, 1] decode graph was deleted
// with the v0.8.0 migration, so nothing in the engine puts a slab — or any
// other array — inside a `compile` tracer.
//
// Ordering, however, is only HALF of what the old eager commit provided.
// The other half was TIMING: an eager commit ran while the just-measured
// load headroom was still true. A deferred commit runs at first admission —
// possibly minutes later, after a co-resident model has consumed the very
// headroom this pool was deferred to yield (the provider's physical-capacity
// policy deliberately reports an uncommitted pool as ZERO bytes, so nothing
// holds the pool's bytes in escrow between load and first admission). The
// deferred eval can therefore genuinely fail, and MLX's answer to a Metal
// allocation failure is the installed error handler — which is `fatalError`
// when nobody bound one: a daemon abort, on a machine whose watchdog
// restarts it into the same state.
//
// Why a failed commitment is a REFUSAL, not an abort
// --------------------------------------------------
// The commit's capacity test is THE ALLOCATION ATTEMPT ITSELF, nothing
// else. Each slab's eval runs under MLX's SCOPED error handler
// (`withError`, task-local — never a process-global handler swap): an
// allocation failure inside the C++ layer is caught at the mlx-c boundary
// after a clean C++ unwind and surfaces as a thrown Swift error. Nothing
// throws across C++ frames. See `PagedKVPool.materializeSlabs`.
//
// There is deliberately NO proactive pre-check, because no MLX counter
// predicts whether the allocator will refuse. What actually makes
// `MetalAllocator::malloc` throw (vendored `allocator.cpp`) is: a single
// buffer above `maxBufferLength` (enforced per-slab at `PagedKVPool.init`,
// before any pool exists), the Metal resource-COUNT limit, or the OS
// returning a null buffer. `Memory.memoryLimit` gates NONE of these — the
// allocator GCs its cache and allocates straight past the byte limit; the
// limit's real semantics are a THROTTLE (`transforms.cpp`: active >
// limit → serialize work; docs/engine-v2/kernel-research.md). And the
// embedding provider RELIES on those semantics: its MLXMemoryGuard pins
// `memoryLimit = physical − reserve` precisely so MLX throttles instead
// of jetsam-killing the process, while its UnifiedMemoryCap doc states
// the capacity cap "lives in the admission layer, not in an MLX setting".
// A pre-check that hard-refused at `memoryLimit` therefore rejected
// serveable pools in exactly the deployment that lowers the limit on
// purpose — a permanently capacity-erroring backend on a healthy box —
// while never predicting a real failure the eval attempt would not
// surface anyway.
//
// A failed attempt surfaces as `capacityExhausted` — the engine's RETRYABLE
// capacity class: `EngineLoopV2.ensureKVState` requeues the request while
// the pool waits for room, then finish-errors with
// `capacityExhaustedFinishPrefix`, which bridges map to a retryable
// capacity rejection (429-class), never a server error and never an abort.
// The pool itself stays UNWIRED and IDLE (`slabsAreWired` remains false),
// the admission charge is unwound by the caller, and the next admission
// retries the whole commit — `guard !slabsAreWired` is the idempotence
// that makes the call free once it finally succeeds.
//
// A PARTIAL commit (some slabs evaluated, a later one failed) retries
// exactly the REMAINDER. `materializeSlabs` evaluates slab-by-slab and
// records each slab's residency the moment its blocking eval returns
// (`PagedKVGroup.kSlabMaterialized`/`vSlabMaterialized`), so a retry
// re-attempts only the missing slabs and never re-evals a resident one.
// The same flags make an ALREADY-materialized pool (the profiler calls
// `pool.materializeSlabs()` directly before minting rows) commit for
// free: nothing left to eval, the pool is simply marked wired.
//
// Byte accounting
// ---------------
// Lazy commitment does NOT make any existing figure time-varying.
// `bytesInUse`, `bytesReserved`, `bytesCapacity` and `bytesPhysical` are all
// host-side arithmetic over page bookkeeping fixed at `PagedKVPool.init`;
// none of them observes evaluation state. In particular `bytesPhysical`
// remains the allocation CEILING (`pageCount * pageBytes`, poison pages
// included) and stays the right input for sizing and wired-limit consumers.
// The time-varying figures are `PagedKVPool.bytesUnmaterialized` (what a
// commit still has to allocate — it only ever shrinks) and its
// complements `bytesMaterialized` / `PagedKVBackend.bytesWired`. All are
// deliberately diagnostic: nothing admits or refuses on them — admission
// is decided by the allocation attempt.

import Foundation
import MLX

/// When a paged pool's slabs are evaluated into real Metal residency.
public enum PagedKVSlabCommitment: String, Sendable, Equatable, CaseIterable {
    /// Wire the slabs during `PagedKVBackend.init`. First-token latency never
    /// pays the allocation, at the cost of an idle pool occupying unified
    /// memory that a co-resident model's headroom measurement will see. Use
    /// for microbenchmarks and profilers that want allocation out of the
    /// timed region, and for single-slot deployments that will never share a
    /// box.
    case atConstruction

    /// Wire the slabs at the pool's first admission — the moment it stops
    /// being idle. The production default: an unused pool contributes zero
    /// bytes to a co-resident model's post-load headroom measurement, exactly
    /// as an unused contiguous grant does.
    case atFirstAdmission
}

extension PagedKVBackend {
    /// Bytes the slabs have ACTUALLY committed to MLX right now: zero until
    /// the pool's first admission under `.atFirstAdmission`, `bytesPhysical`
    /// once wired, and the honest resident amount in the (transient)
    /// partially-committed state between a failed commit and its retry.
    ///
    /// TIME-VARYING BY CONSTRUCTION. Diagnostics and telemetry only — never
    /// an admission input, never a sizing input, never a wired-limit input.
    /// Anything that must not move under this backend's feet wants
    /// `bytesCapacity` (the budgeted, admission-relevant figure) or
    /// `bytesPhysical` (the allocation ceiling), both of which are fixed at
    /// pool construction.
    public var bytesWired: Int { pool.bytesMaterialized }

    /// Evaluate every group's slabs, making the pool's pages physically
    /// resident. Idempotent after success: once wired this is a bool test,
    /// so the admission path can call it unconditionally.
    ///
    /// REFUSES rather than traps when the box can no longer take the pool:
    /// throws `CBv2KVError.capacityExhausted` — the engine's retryable
    /// capacity class — and leaves the pool unwired so a later admission
    /// retries the commit.
    ///
    /// The capacity test is the ALLOCATION ATTEMPT itself — there is
    /// deliberately no headroom pre-check here. Nothing the engine can
    /// read predicts the allocator: `MetalAllocator::malloc` throws only
    /// on per-buffer `maxBufferLength` (already enforced per-slab at
    /// `PagedKVPool.init`), the Metal resource-count limit, or the OS
    /// refusing the buffer — never on `Memory.memoryLimit`, whose real
    /// semantics are a work-serializing THROTTLE that the embedding
    /// provider deliberately pins low (`MLXMemoryGuard`: throttle instead
    /// of jetsam; capacity enforcement lives in the provider's admission
    /// layer). Rejecting on `activeMemory + demand > memoryLimit` refused
    /// serveable pools in exactly that deployment. See the file header.
    ///
    /// An already-materialized pool (`pool.materializeSlabs()` called
    /// directly, as the profiler does) commits for free regardless of any
    /// memory condition: nothing is left to eval, so the pool is simply
    /// marked wired.
    ///
    /// Thread-affinity is the pool's: the engine loop thread, no locking.
    public func commitSlabs() throws {
        guard !slabsAreWired else { return }
        do {
            try pool.materializeSlabs()
        } catch {
            // `needed` is what is STILL missing after the partial progress
            // this attempt made; `available` is a diagnostic-only reading
            // of the throttle limit's remaining headroom (it is NOT what
            // admission decided on — the failed attempt is).
            throw CBv2KVError.capacityExhausted(
                needed: pool.bytesUnmaterialized,
                available: max(0, Memory.memoryLimit - Memory.activeMemory))
        }
        markSlabsWired()
    }
}
