# Gemma 4 B1 Exact MTP Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gemma 4 fixed-depth MTP verification token-exact and faster than the unchanged autoregressive control for one physical prompt at C2, C3, and C4.

**Architecture:** Start from the three clean fixed-width verifier commits, but key every installed context by physical batch and column count. Add B1/C2-C4 quantized projection kernels whose per-position accumulation order matches an independent B1/L1 call while sharing each weight traversal across positions. Bind those entrypoints once after strict model loading, route only certified B1 shapes through one rectangular target forward, and retain serial verification as an explicit unmodified control rather than a hot-path fallback.

**Tech Stack:** Swift 6.3, MLX Swift, Metal kernels through `MLXFast.metalKernel`, Swift Testing, CBv2 MTP runtime, guarded macOS MLX lifecycle.

**Assumptions:**

- Assumes the source remains based on `origin/main@8fbf2f3` plus the committed depth/cache fixes and design commits — will NOT apply cleanly to the older grouped-verifier branch.
- Assumes the real single-prompt worker preserves physical batch one, as proved by the receipt — will NOT treat a logical B1 request padded to B8 as equivalent.
- Assumes exact greedy output is mandatory — will NOT accept tolerance-only logits or a different token stream.
- Assumes affine group-size-64 target weights and the pinned Gemma artifact topology — will NOT install the lane for another quantization or model geometry.

---

## File structure

- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift` — immutable `(batch, columns)` route key and construction policy.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B1MTPQuantizedProjection.swift` — shared B1/C2-C4 affine QMV implementation derived from the pinned ordinary B1 reduction.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionQKVMMA8V1.swift` — Q/K/V bindings; existing B8 behavior stays intact, B1 delegates to the new projection.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionOQMVV1.swift` — attention-output binding for B1.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/DenseMLPQMVV1.swift` — dense gate/up/down bindings for B1.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4MMAQuantizedGEMV.swift` — tied-head B1 binding with the artifact vocabulary width.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift` — router and expert projection bindings for B1.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift` — all-or-nothing B1 verifier installation and direct model route.
- `Sources/MLXFastHarness/MTPEnvelope.swift` and `Sources/MLXFastTrustedHarness/MTPEnvelope.swift` — twin construction-time selection of certified rectangular verification.
- `Tests/MLXFastTests/Gemma4MTPVerifier*Tests.swift` — isolated B1/C2-C4 bit-parity and installation tests.
- `Tests/MLXFastTests/MTPVerificationStrategySealTests.swift` — engine route and no-fallback seal.
- `Tests/MLXFastTests/RuntimeWorkerMTPRoundExecutionTests.swift` — full-round token, acceptance, and cache parity.

### Task 1: Integrate the reviewed clean verifier base

**Files:**
- Create/modify: the exact files recorded by commits `3128eb5`, `5eea619`, and `764e9eb`

**Security flag:** none

- [x] **Step 1: Verify the isolated branch contract**

Run:

```bash
git status --short --branch
git merge-base --is-ancestor 8fbf2f3 HEAD
git log -7 --oneline
```

Expected: clean `bench/gemma4-mtp-depth3-tip`, the ancestry command exits zero, and HEAD includes `8f0365c`.

- [x] **Step 2: Cherry-pick only the three clean commits**

```bash
git cherry-pick 3128eb5 5eea619 764e9eb
```

Expected: three new commits; no content is copied from `mlxfast-gemma4-mtp-tip-qmm` or another dirty worktree.

- [x] **Step 3: Run the existing fixed-width verifier suites under the real GPU lock**

```bash
/opt/homebrew/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --timeout-seconds 900 \
  --child-timeout-seconds 1800 \
  -- /bin/zsh -lc 'cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/mlxfast-gemma4-mtp-depth3-tip && swift test --filter Gemma4MTPVerifier'
```

Expected: all clean B8 verifier tests pass; the guard restores `mtplx-flash-next-optimized-speed` before returning.

### Task 2: Add a physical verifier-shape key and failing B1 route tests

**Files:**
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`

**Security flag:** none

**Does NOT cover:** B2-B8 promotion, C1, or C5+; those shapes remain outside the B1 performance lane.

- [x] **Step 1: Write failing route-key tests**

Add:

```swift
@Test
func singlePromptShapesAreExplicitAndCannotAliasB8() throws {
    for columns in 2...4 {
        let b1 = CBv2Gemma4MTPVerifierShape(batch: 1, columns: columns)
        let b8 = CBv2Gemma4MTPVerifierShape(batch: 8, columns: columns)
        #expect(CBv2Gemma4MTPVerifierRoute.production.supports(b1))
        #expect(b1 != b8)
    }
    #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
        .init(batch: 1, columns: 1)))
    #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
        .init(batch: 1, columns: 5)))
    #expect(!CBv2Gemma4MTPVerifierRoute.production.supports(
        .init(batch: 2, columns: 2)))
}
```

- [x] **Step 2: Run the test and verify the new type is absent**

Run the guarded Swift command from Task 1 with `--filter Gemma4MTPVerifierRouteTests`.

Expected: FAIL because `CBv2Gemma4MTPVerifierShape` and `supports(_:)` do not exist.

- [x] **Step 3: Add the immutable shape key**

Implement:

```swift
public struct CBv2Gemma4MTPVerifierShape: Sendable, Hashable {
    public let batch: Int
    public let columns: Int

    public init(batch: Int, columns: Int) {
        self.batch = batch
        self.columns = columns
    }
}

public extension CBv2Gemma4MTPVerifierRoute {
    func supports(_ shape: CBv2Gemma4MTPVerifierShape) -> Bool {
        shape.batch == 1 && (2...4).contains(shape.columns)
    }
}
```

Keep the existing B8 projection-policy methods for their clean tests, but do not let them imply that B8 is promoted by this branch.

- [x] **Step 4: Run the focused test**

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/Gemma4MTPVerifierRoute.swift Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift
git commit -m 'test: distinguish Gemma B1 verifier shapes'
```

### Task 3: Implement the exact shared B1 quantized projection core

**Files:**
- Create: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B1MTPQuantizedProjection.swift`
- Create: `Tests/MLXFastTests/Gemma4MTPVerifierB1ProjectionTests.swift`

**Security flag:** none

**Does NOT cover:** gather/expert routing; this task handles a single immutable affine weight plane shared by C positions.

- [ ] **Step 1: Write failing affine-4 and affine-8 parity tests**

Use the real production input widths and deterministic synthetic weights. The core assertion is bit equality, not tolerance:

```swift
private struct B1AffineProjectionFixture {
    let x: MLXArray
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let ordinaryB1: (MLXArray) -> MLXArray

    init(columns: Int, bits: Int, groupSize: Int, inDim: Int, outDim: Int) {
        let x = MLXArray((0..<(columns * inDim)).map {
            Float(($0 * 37 + columns * 11) % 257 - 128) / 128
        }).reshaped([1, columns, inDim]).asType(.bfloat16)
        let packed = inDim * bits / 32
        let weight = MLXArray((0..<(outDim * packed)).map {
            UInt32(truncatingIfNeeded: $0 * 2_654_435_761)
        }).reshaped([outDim, packed])
        let scales = MLXArray((0..<(outDim * inDim / groupSize)).map {
            Float(($0 % 31) + 1) / 64
        }).reshaped([outDim, inDim / groupSize]).asType(.bfloat16)
        let biases = MLXArray((0..<(outDim * inDim / groupSize)).map {
            Float(($0 % 17) - 8) / 128
        }).reshaped([outDim, inDim / groupSize]).asType(.bfloat16)
        self.x = x
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.ordinaryB1 = { input in
            quantizedMM(
                input, weight, scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits,
                mode: .affine)
        }
    }
}

private func independentColumns(
    _ x: MLXArray,
    project: (MLXArray) -> MLXArray
) -> MLXArray {
    concatenated((0..<x.dim(1)).map { column in
        project(x[0..., column..<(column + 1), 0...])
    }, axis: 1)
}

@Test(arguments: [2, 3, 4], [4, 8])
func b1FixedWidthProjectionMatchesIndependentB1(
    columns: Int, bits: Int
) throws {
    let fixture = B1AffineProjectionFixture(
        columns: columns, bits: bits, groupSize: 64,
        inDim: 2_816, outDim: 2_112)
    let reference = independentColumns(fixture.x, project: fixture.ordinaryB1)
    let candidate = try #require(Gemma4B1MTPQuantizedProjection.bind(
        columns: columns, inDim: 2_816, outDim: 2_112,
        weight: fixture.weight, scales: fixture.scales,
        biases: fixture.biases, groupSize: 64, bits: bits))(fixture.x)
    eval(reference, candidate)
    #expect(candidate.shape == [1, columns, 2_112])
    #expect(allClose(candidate, reference, rtol: 0, atol: 0).item(Bool.self))
}
```

The fixture's `ordinaryB1` must call the same pinned `QuantizedLinear`/`quantizedMatmul` path used by the model, not another custom kernel.

- [ ] **Step 2: Run the focused test under the guard**

Expected: FAIL because `Gemma4B1MTPQuantizedProjection` does not exist.

- [ ] **Step 3: Implement the fixed-C Metal entrypoint**

Create a construction-bound API:

```swift
public enum Gemma4B1MTPQuantizedProjection {
    public typealias Projection = (MLXArray) -> MLXArray

    public static func bind(
        columns: Int, inDim: Int, outDim: Int,
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int
    ) -> Projection? {
        guard (2...4).contains(columns), groupSize == 64,
              bits == 4 || bits == 8, let biases,
              weight.dtype == .uint32,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              weight.shape == [outDim, inDim * bits / 32],
              scales.shape == [outDim, inDim / 64],
              biases.shape == scales.shape else { return nil }
        let kernel = bits == 4 ? affine4Kernel : affine8Kernel
        return { x in
            precondition(x.shape == [1, columns, inDim])
            let flat = x.reshaped([columns, inDim])
            return kernel(
                [flat, weight, scales, biases],
                template: [("T", DType.bfloat16), ("COLUMNS", columns)],
                grid: (32, outDim, 1), threadGroup: (32, 1, 1),
                outputShapes: [[columns, outDim]],
                outputDTypes: [.bfloat16])[0]
                .reshaped([1, columns, outDim])
        }
    }
}
```

The Metal body must copy the literal K-loop, dequantization, bias contribution, and `simd_sum` order from the pinned ordinary `qmv_impl<T, 64, bits>` in `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`. Replace its single accumulator with `thread float acc[COLUMNS]`; load each packed weight/scales/bias value once, then update `acc[c]` for `c = 0..<COLUMNS` in the same statement order as the ordinary B1 body. Do not use matrix-multiply reductions or `affine_qmv_fast`.

- [ ] **Step 4: Run bit-parity tests**

Expected: all six B1/C2-C4 affine-4/8 cells pass bit-for-bit.

- [ ] **Step 5: Commit**

```bash
git add Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B1MTPQuantizedProjection.swift Tests/MLXFastTests/Gemma4MTPVerifierB1ProjectionTests.swift
git commit -m 'perf: add exact shared Gemma B1 MTP projection'
```

### Task 4: Bind B1 QKV, attention-output, dense, and tied-head projections

**Files:**
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionQKVMMA8V1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionOQMVV1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/DenseMLPQMVV1.swift`
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4MMAQuantizedGEMV.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierQKVKernelTests.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierAttentionOKernelTests.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierDenseMLPKernelTests.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierHeadKernelTests.swift`

**Security flag:** none

**Does NOT cover:** B8 route behavior; the existing B8 binders remain byte-for-byte callable and separately tested.

- [ ] **Step 1: Add failing B1 cases to every projection suite**

For C2-C4, build `x` as `[1, columns, K]`, compare the new B1 binder to concatenated ordinary B1 calls, and require exact BF16 storage equality. For the head, use `outDim = 262_144` in the artifact-width test.

The expected binder surface is:

```swift
bindB1Verifier(
    columns: Int,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> ((MLXArray) -> MLXArray)?
```

Dense and attention-output variants retain their existing `inDim`/`outDim` arguments.

- [ ] **Step 2: Run all four focused suites**

Expected: FAIL because the B1 binders are absent.

- [ ] **Step 3: Implement direct B1 bindings**

Each `bindB1Verifier` performs its existing weight/topology checks once, then returns the closure from `Gemma4B1MTPQuantizedProjection.bind`. No enabled-path closure may call `matmul(...)`, inspect environment variables, or fall back to stock.

Example:

```swift
public static func bindB1Verifier(
    columns: Int, inDim: Int,
    weight: MLXArray, scales: MLXArray, biases: MLXArray?,
    groupSize: Int, bits: Int, mode: QuantizationMode
) -> ((MLXArray) -> MLXArray)? {
    guard mode == .affine, liveInputWidth(inDim) else { return nil }
    return Gemma4B1MTPQuantizedProjection.bind(
        columns: columns, inDim: inDim, outDim: outputWidth,
        weight: weight, scales: scales, biases: biases,
        groupSize: groupSize, bits: bits)
}
```

- [ ] **Step 4: Run the projection suites and the original B8 suites**

Expected: B1 and B8 tests all pass.

- [ ] **Step 5: Commit**

```bash
git add Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionQKVMMA8V1.swift Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/AttentionOQMVV1.swift Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/DenseMLPQMVV1.swift Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4MMAQuantizedGEMV.swift Tests/MLXFastTests/Gemma4MTPVerifierQKVKernelTests.swift Tests/MLXFastTests/Gemma4MTPVerifierAttentionOKernelTests.swift Tests/MLXFastTests/Gemma4MTPVerifierDenseMLPKernelTests.swift Tests/MLXFastTests/Gemma4MTPVerifierHeadKernelTests.swift
git commit -m 'perf: bind exact Gemma B1 verifier projections'
```

### Task 5: Bind exact B1 router and expert projections

**Files:**
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierExpertKernelTests.swift`

**Security flag:** none

**Does NOT cover:** changing routing, top-k selection, expert ordering, weighted-unsort behavior, or sharing weights across different expert IDs.

- [ ] **Step 1: Write failing B1 router and expert tests**

Add C2-C4 cases with `[1, C, H]` inputs, `[1, C, 8]` expert indices/weights, and deterministic repeated plus disjoint expert assignments. Require exact router scores and exact expert output against C independent B1 calls.

```swift
#expect(allClose(candidateRouter, referenceRouter, rtol: 0, atol: 0)
    .item(Bool.self))
#expect(allClose(candidateExpert, referenceExpert, rtol: 0, atol: 0)
    .item(Bool.self))
```

- [ ] **Step 2: Run the expert suite**

Expected: FAIL because B1 bindings are absent.

- [ ] **Step 3: Implement construction-bound B1 bindings**

The router delegates to `Gemma4B1MTPQuantizedProjection`. The expert binder captures the immutable quantized expert planes once, groups assignments by existing expert ID without changing stable order, and invokes the same pinned gather-QMV arithmetic per assignment. It may share a plane only among positions whose expert IDs are equal.

Expose the same complete weight surface as the existing `bindVerifier`, with a
B1-specific name:

```swift
public static func bindB1Verifier(
    columns: Int,
    gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray?,
    upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray?,
    downWeight: MLXArray, downScales: MLXArray, downBiases: MLXArray?,
    groupSize: Int, bits: Int, mode: QuantizationMode
) -> Projection?
```

The returned `Projection` accepts `(x, topKIndices, topKWeights)` and has no topology validation or fallback inside the call.

- [ ] **Step 4: Run repeated and disjoint expert cases**

Expected: exact pass for C2-C4; disjoint assignments prove no accidental cross-expert sharing.

- [ ] **Step 5: Commit**

```bash
git add Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift Tests/MLXFastTests/Gemma4MTPVerifierExpertKernelTests.swift
git commit -m 'perf: bind exact Gemma B1 MTP experts'
```

### Task 6: Install all-or-nothing B1 model contexts

**Files:**
- Modify: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`
- Modify: `Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift`
- Modify: `Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift`
- Modify: `Tests/MLXFastTests/TargetQuantizationBindTests.swift`

**Security flag:** none

**Does NOT cover:** runtime eligibility checks or fallback; installation either publishes all three B1 contexts or throws before warmup.

- [ ] **Step 1: Write failing construction tests**

Require these semantics:

```swift
let shapes = [2, 3, 4].map {
    CBv2Gemma4MTPVerifierShape(batch: 1, columns: $0)
}
let installed = try fixture.install()
#expect(installed.shapes == shapes)
for shape in shapes {
    #expect(installed.entrypointCount(shape: shape)
        == fixture.topology.requiredProjectionEntrypoints)
}
#expect(installed.context(batch: 8, columns: 2) == nil)
```

Also test that one missing B1 projection closure throws and leaves `cbv2MTPVerifierInstalled == false`.

- [ ] **Step 2: Run the construction suites**

Expected: FAIL because contexts are keyed only by columns and B1 binding is not installed.

- [ ] **Step 3: Replace the context key and bind B1 closures**

Use:

```swift
private var installedMTPVerifierContexts:
    [CBv2Gemma4MTPVerifierShape: Gemma4MTPVerifierContext]?

public func cbv2MTPVerifierContext(
    batch: Int, columns: Int
) -> Gemma4MTPVerifierContext? {
    installedMTPVerifierContexts?[.init(batch: batch, columns: columns)]
}
```

Build a local dictionary for B1/C2-C4, verify every layer and tied-head closure, then publish it once. Preserve `try model.installCBv2MTPVerifier()` after strict update and `eval(model)` in the runtime loader.

- [ ] **Step 4: Route the model boundary by exact shape**

```swift
let shape = CBv2Gemma4MTPVerifierShape(
    batch: tokens.dim(0), columns: tokens.dim(1))
let verifier: Gemma4MTPVerifierContext?
if tokens.ndim == 2, tokens.dim(1) > 1 {
    guard let installed = installedMTPVerifierContexts?[shape] else {
        preconditionFailure("Gemma 4 MTP verifier has no installed B\(shape.batch)/C\(shape.columns) route")
    }
    verifier = installed
} else {
    verifier = nil
}
```

- [ ] **Step 5: Run construction and projection suites**

Expected: all pass; failure cases prove no partial installation.

- [ ] **Step 6: Commit**

```bash
git add Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift Sources/MLXFastModel/Gemma4A4BRuntimeWeights.swift Tests/MLXFastTests/Gemma4MTPVerifierRouteTests.swift Tests/MLXFastTests/TargetQuantizationBindTests.swift
git commit -m 'perf: install exact Gemma B1 verifier contexts'
```

### Task 7: Enable certified rectangular verification and prove full rounds

**Files:**
- Modify: `Tests/MLXFastTests/MTPVerificationStrategySealTests.swift`
- Modify: `Tests/MLXFastTests/RuntimeWorkerMTPRoundExecutionTests.swift`
- Modify: `Sources/MLXFastHarness/MTPEnvelope.swift`
- Modify: `Sources/MLXFastTrustedHarness/MTPEnvelope.swift`

**Security flag:** none

**Does NOT cover:** automatic shape fallback, batch greater than one, or silent serial recovery after a verifier error.

- [ ] **Step 1: Replace the old serial seal with failing B1 exact-route tests**

For depths 1, 2, and 3 require:

```swift
let config = try Gemma4MTPEnvelope.resolveConfig(depth: depth)
#expect(config.verificationMode == .rectangular)
#expect(config.fixedDraftTokens == depth)
#expect(config.maxSpeculativeBatch == 1)
```

Run the fixture once with serial verification and once with rectangular verification. Require identical logits/argmax tokens, acceptance lengths, committed sequence, and final cache lengths; require `rectangularVerificationRounds == rounds` for the candidate and zero for the serial control.

- [ ] **Step 2: Run the two suites**

Expected: FAIL because the envelope remains `.serialTarget`.

- [ ] **Step 3: Change both envelope twins identically**

```swift
return CBv2MTPConfig(
    enabled: depth > 0,
    maxDraftTokens: maxDraftTokens,
    maxSpeculativeBatch: 1,
    fixedDraftTokens: depth,
    verificationMode: .rectangular,
    maxAutomaticRectangularTokens: maxAutomaticRectangularTokens)
```

Keep the twin files byte-identical and remove comments that claim the production seal is serial.

- [ ] **Step 4: Run twin equality, strategy seal, acceptance audit, and round execution**

Use one guarded child:

```bash
swift test --filter 'MTPWorkerTwinEquality|MTPVerificationStrategySeal|MTPAcceptanceRuleAudit|RuntimeWorkerMTPRoundExecution'
```

Expected: all tests pass, including the existing single and batch token-losslessness cases; batch tests keep their explicit test config and do not imply B2+ production installation.

- [ ] **Step 5: Commit**

```bash
git add Sources/MLXFastHarness/MTPEnvelope.swift Sources/MLXFastTrustedHarness/MTPEnvelope.swift Tests/MLXFastTests/MTPVerificationStrategySealTests.swift Tests/MLXFastTests/RuntimeWorkerMTPRoundExecutionTests.swift
git commit -m 'perf: enable exact Gemma B1 MTP verification'
```

### Task 8: Run the real-model correctness and performance gate

**Files:**
- Modify only if a measured B1 component fails: the file owning that component
- Create: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-b1-exact-verifier-gate-20260830.json`

**Security flag:** none

- [ ] **Step 1: Build the release worker without executing MLX**

```bash
swift build -c release --product mlxfast-runtime-worker --build-path .build-worker
shasum -a 256 .build-worker/arm64-apple-macosx/release/mlxfast-runtime-worker
```

Expected: build succeeds and the worker hash is recorded before the guarded window.

- [ ] **Step 2: Run bounded B1/C2-C4 real-model parity under the canonical guard**

Use the pinned weights, assistant head, and prompt matrix. Run exactly three
prompt cells per depth: 0 prefix + 1,024 coding tokens, 16,384 prefix + 1,024
coding tokens, and 65,536 prefix + 1,024 coding tokens. Decode 128 committed
tokens per cell with a serial control and exact rectangular candidate from the
same worker. Record token digests, acceptance lengths, final cache lengths,
verifier round counts, peak MLX memory, source SHA, worker SHA, artifact hashes,
and guard lifecycle.

Expected: exact candidate/control token and cache parity at depths 1, 2, and 3; positive rectangular rounds; no B8 route.

- [ ] **Step 3: Profile the matched 16K + 1K / 1K-decode cell**

Run one discarded primer and three measured samples for serial plus each depth. Record separate prefill/decode wall time and existing MTP acceptance/round statistics. Do not add counters to model or kernel hot paths.

Expected first performance gate: the best exact MTP depth has a higher arithmetic-mean decode TPS than serial; otherwise retain the receipt, identify the top measured boundary, and iterate only that component before promotion.

- [ ] **Step 4: Verify service and lock postflight**

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
lsof /tmp/mtplx-gpu-exclusive.lock
```

Expected: health reports zero active requests and understood warmup state; model ID is exactly `mtplx-flash-next-optimized-speed`; `lsof` shows no lock owner.

- [ ] **Step 5: Close the gate only on complete evidence**

Run `git status --short` and hash the receipt. Expected: the source worktree is
clean; the receipt status is complete; no raw service log is staged. If either
correctness or the serial-throughput gate fails, retain the raw receipt and stop
without promoting the rectangular route.
