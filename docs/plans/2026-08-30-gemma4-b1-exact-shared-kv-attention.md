# Gemma 4 B1 Exact Shared-KV Attention Implementation Plan

> **Execution note:** Implement inline with the `executing-plans`,
> `systematic-debugging`, `test-driven-development`, and
> `performance-investigation` workflows. Do not run any MLX/Metal command until
> the user approves the prepared test window.

**Goal:** Collapse each Gemma 4 target full-attention verifier round from C
independent K/V traversals to one physical-B1, C2-C4 shared traversal without
changing any column's ordinary B1/L1 arithmetic, visible key range, cache
ownership, or tokens.

**Architecture:** Add an unpromoted `Gemma4B1MTPFullAttentionV1` candidate beside
the frozen D512 composed attention implementation. It reuses the ordinary QK,
precise-softmax, and AV reduction order, but keeps C independent accumulators
while loading each K/V packet once. Construction binds exact C2/C3/C4 callables
for full-attention layers; sliding-window layers retain the existing serialized
decode callable. The real verifier route is not promoted until the candidate
passes bit equality and beats unchanged serial attention for that width.

**Tech stack:** Swift 6.3, MLX Swift, Metal kernels through
`MLXFast.metalKernel`, Swift Testing, CBv2 contiguous/paged KV caches, guarded
macOS MLX lifecycle.

**Hard constraints:**

- Physical batch is exactly one; columns are exactly 2, 3, or 4.
- Q/K/V are BF16 with QH16, KVH2, GQA8, D512, scale 1.0, no sinks, no
  attention softcap, and full attention.
- Column `j` sees exactly `historyLength + j + 1` keys.
- No hot-path metadata validation, environment read, fallback, or proof
  counter is added.
- The existing serial implementation remains the explicit control, not a
  fallback inside an enabled candidate.
- No MLX model load, kernel compilation, GPU self-check, or benchmark runs
  before the user approves the prepared guarded command.

---

## Task 1: Freeze the construction and causal-visibility contracts

**Files:**

- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`
- Create: `Tests/MLXFastTests/Gemma4B1MTPFullAttentionContractTests.swift`

- [x] **Step 1: Add CPU-only failing route tests**

Add an explicit layer kind to the attention route and prove that full and
sliding layers cannot alias:

```swift
@Test(arguments: [2, 3, 4])
func fullAttentionWidthsBindTheSharedKVCandidate(columns: Int) throws {
    let route = CBv2Gemma4MTPVerifierRoute.production
    #expect(route.attentionStrategy(kind: .full, columns: columns) == .sharedKVExact)
    #expect(route.attentionStrategy(kind: .sliding(window: 1_024), columns: columns)
        == .serializedDecode)
}

@Test
func unsupportedAttentionShapesPublishNoStrategy() {
    let route = CBv2Gemma4MTPVerifierRoute.production
    #expect(route.attentionStrategy(kind: .full, columns: 1) == nil)
    #expect(route.attentionStrategy(kind: .full, columns: 5) == nil)
}
```

Add pure visible-length tests:

```swift
@Test(arguments: [2, 3, 4], [0, 1_024, 4_095, 4_096, 16_384, 65_536, 131_072])
func eachColumnHasItsOwnCausalEnd(columns: Int, history: Int) {
    #expect(Gemma4B1MTPFullAttentionGeometry.visibleKeyLengths(
        historyLength: history, columns: columns
    ) == (0..<columns).map { history + $0 + 1 })
}
```

- [ ] **Step 2: Run only the CPU route tests** (deferred with the guarded
  operator test at the user's requested test boundary)

```bash
swift test --filter Gemma4MTPVerifierRouteTests
swift test --filter Gemma4B1MTPFullAttentionContractTests
```

Expected: FAIL because `.sharedKVExact`, the layer-kind route argument, and
`Gemma4B1MTPFullAttentionGeometry` do not yet exist. If Swift package loading
would initialize MLX/Metal on this machine, do not run these commands; stop at
the prepared command instead.

- [x] **Step 3: Implement immutable construction policy**

Use construction-only types:

```swift
public enum CBv2Gemma4MTPVerifierAttentionKind: Sendable, Equatable {
    case full
    case sliding(window: Int)
}

public enum CBv2Gemma4MTPVerifierAttentionStrategy: Sendable, Equatable {
    case sharedKVExact
    case serializedDecode
}

public enum Gemma4B1MTPFullAttentionGeometry {
    static func visibleKeyLengths(historyLength: Int, columns: Int) -> [Int] {
        precondition(historyLength >= 0)
        precondition((2...4).contains(columns))
        return (0..<columns).map { historyLength + $0 + 1 }
    }
}
```

`attentionStrategy(kind:columns:)` returns `.sharedKVExact` only for `.full`
and C2-C4. `.sliding(window: 1_024)` returns `.serializedDecode`. Any other
width returns nil at construction.

- [ ] **Step 4: Re-run the CPU-only tests** (next action)

Expected: PASS without compiling or executing a Metal kernel.

---

## Task 2: Add bit-exact operator tests before the candidate kernel

**Files:**

- Create: `Tests/MLXFastTests/Gemma4B1MTPFullAttentionTests.swift`
- Reference unchanged oracle: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/RaggedTwoPassDecodeAttentionV1.swift`

- [x] **Step 1: Write the GPU operator tests without running them**

For C2, C3, and C4, create deterministic BF16 Q/K/V with distinct columns and
compare the candidate to concatenated independent ordinary B1/L1 calls:

```swift
let reference = concatenated((0..<columns).map { column in
    ordinaryB1D512Attention(
        query: queries[0..., 0..., column..<(column + 1), 0...],
        keys: keys[0..., 0..., 0..<(history + column + 1), 0...],
        values: values[0..., 0..., 0..<(history + column + 1), 0...])
}, axis: 2)

let candidate = try #require(
    Gemma4B1MTPFullAttentionV1.bind(columns: columns)
)(queries, keys, values, history)

eval(reference, candidate)
#expect(candidate.shape == [1, 16, columns, 512])
#expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
```

Cover key lengths 1, 1,024, 4,095, 4,096, 16,384, 65,536, and 131,072,
subject to the runbook's static memory admission. Add adversarial score fixtures
containing large positive, large negative, tied, and per-column-distinct QK
rows. Assert byte equality, not only tolerance equality.

- [ ] **Step 2: Add cache-transaction tests without running them** (requires
  candidate promotion into the cache seam after the operator gate)

The tests append C K/V entries once, invoke the candidate, commit each possible
accepted prefix, and compare cache lengths/tensors with the unchanged serial
oracle. Exercise every accepted count from zero through C.

Expected when eventually run before production code: FAIL because
`Gemma4B1MTPFullAttentionV1` is absent.

---

## Task 3: Implement the exact C2-C4 shared-KV attention candidate

**Files:**

- Create: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B1MTPFullAttentionV1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/RaggedTwoPassDecodeAttentionV1.swift`

- [x] **Step 1: Extract only reusable frozen ordinary primitives** (transcribed
  into the isolated candidate; the established ordinary entrypoint is unchanged)

Expose package-internal kernel source fragments or helper factories for the
ordinary D512 QK, precise BF16 softmax, and AV arithmetic. Do not change the
ordinary entrypoint, launch geometry, or existing tests. Preserve:

- QK packet/lane mapping and accumulation order;
- BF16 score storage;
- `softmax_single_row` for visible length up to 4,096;
- MLX's `softmax_looped` online max/normalizer order above 4,096;
- AV packet/lane mapping, probability conversion, accumulation, and reduction.

- [x] **Step 2: Implement one fixed-column kernel family**

Publish only construction-bound callables:

```swift
public enum Gemma4B1MTPFullAttentionV1 {
    public typealias Attention = (
        _ queries: MLXArray,
        _ keys: MLXArray,
        _ values: MLXArray,
        _ historyLength: Int
    ) -> MLXArray

    public static func bind(columns: Int) -> Attention? {
        switch columns {
        case 2: return bindFixedColumns(2)
        case 3: return bindFixedColumns(3)
        case 4: return bindFixedColumns(4)
        default: return nil
        }
    }
}
```

The returned closure assumes its construction-proven shapes. It contains no
eligibility branch or stock fallback.

QK and AV use fixed C-specific Metal functions so each threadgroup loads an
immutable K or V packet once, then updates C independent ordinary accumulators.
Each column has a fixed `visibleEnd = historyLength + column + 1`; packets at or
beyond that end contribute the ordinary identity value. Never pad a shorter
column into another column's softmax reduction.

- [x] **Step 3: Transcribe both precise softmax regimes**

Use the frozen MLX source as authority:

- `axisSize <= 4_096`: threadgroup size
  `32 * ceil(ceil(axisSize / 4) / 32)` and the existing precise BF16 block
  reduction.
- `axisSize > 4_096`: pipeline maximum threadgroup size and the exact
  `softmax_looped<bf16,float,4>` recurrence from MLX, independently for each
  visible column length.

The candidate may issue one softmax dispatch per column. This preserves each
row's exact reduction length while the bandwidth-critical QK/AV stages share
K/V traversal.

- [x] **Step 4: Prepare, but do not run, the focused operator gate**

```bash
/opt/homebrew/bin/python3 \
  ~/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist ~/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --timeout-seconds 1800 \
  --child-timeout-seconds 3600 \
  -- /bin/zsh -lc 'cd ~/projects/OpenSourceWTF/.worktrees/mlxfast-gemma4-mtp-depth3-tip && MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_RUN_MLX_LONG_RUNTIME_TESTS=1 swift test --filter Gemma4B1MTPFullAttentionTests'
```

Expected after implementation: every admitted length and C2-C4 is byte-exact;
the guard restores and verifies `mtplx-flash-next-optimized-speed` before
releasing `/tmp/mtplx-gpu-exclusive.lock`.

---

## Task 4: Wire an explicit diagnostic seam without promoting production

**Files:**

- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/LayerCacheV2.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/PagedLayerCache.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/LayerCacheBankV2.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/PagedSeamContract.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionV1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`

- [ ] **Step 1: Replace the Boolean attention mode with an installed strategy**

Introduce a construction-installed cache value with two explicit cases:

```swift
enum CBv2MTPRectangularAttentionBinding {
    case serializedDecode
    case sharedKVExact(Gemma4B1MTPFullAttentionV1.Attention)
}
```

The bank binds `.sharedKVExact` only to the five full-attention caches and
`.serializedDecode` to the 25 sliding caches. The per-round attention call
switches only on this already-installed phase binding. It does not re-check
model metadata, dtype, head geometry, or feature flags.

- [ ] **Step 2: Keep the production verifier route unpromoted**

The candidate is exposed to focused tests and the isolated benchmark probe,
but normal serving keeps the unchanged `.serializedDecode` production context
until C2/C3/C4 independently pass exactness and speed gates. Do not add a
runtime environment toggle or silent fallback.

- [ ] **Step 3: Add CPU-only source/route seals**

Prove that:

- full and sliding cache bindings are installed at construction;
- no per-token eligibility check or fallback closure exists;
- the ordinary serial route remains directly selectable as the control;
- no engagement counter is added to the measured path.

- [ ] **Step 4: Run only non-MLX contract tests if safe**

```bash
swift test --filter Gemma4MTPVerifierRouteTests
swift test --filter MTPVerificationStrategySealTests
```

Expected: PASS. Do not run if package initialization reaches Metal.

---

## Task 5: Prepare the isolated performance gate and halt

**Files:**

- Modify: `Sources/MLXFastHarness/Gemma4RuntimeWidthProbe.swift`
- Modify: `Sources/MLXFastRuntimeWorkerCLI/main.swift`
- Modify: `Tests/MLXFastTests/RuntimeWorkerCohortTests.swift`
- Create: `docs/benchmarks/gemma4-b1-shared-kv-attention-gate.md`

- [ ] **Step 1: Add explicit control/candidate probe modes**

The probe accepts a construction-time attention implementation identifier:
`serial-control` or `shared-kv-candidate`. Each produces a receipt containing
the git SHA, model revision, physical batch, C, history length, exact output
digest, wall time, and tokens/s. This selector is diagnostic-only and never
read inside the attention hot path.

- [ ] **Step 2: Define the isolated width gate**

For C2/C3/C4 at the real full-attention geometry and 16K history:

- one unmeasured primer per implementation;
- three interleaved measured samples per implementation;
- unchanged serial implementation is the control;
- byte-exact output digest is mandatory;
- winner is lower mean `wall_s`; decode TPS remains displayed;
- a losing width is not promoted.

- [ ] **Step 3: Define the real-model follow-up gate**

Only after the operator gate passes, run exact token/cache parity at 0, 16K,
and 64K for D1-D3, then the exact 16K input / 1,024-token Python coding decode
mean-of-three against unchanged AR. Do not retune acceptance/depth until the
attention result is attributable.

- [ ] **Step 4: Halt ready for testing**

Before any GPU execution, report:

- exact changed files and diff summary;
- CPU-only checks actually run;
- the prepared guarded operator command;
- static memory admission for the largest operator fixture;
- current production service identity and health from read-only checks;
- current lock-owner status from the runbook's diagnostic command.

Do not acquire the lock, unload the service, compile a Metal kernel, load the
model, run the operator gate, or claim the candidate works. Wait for explicit
user approval.
