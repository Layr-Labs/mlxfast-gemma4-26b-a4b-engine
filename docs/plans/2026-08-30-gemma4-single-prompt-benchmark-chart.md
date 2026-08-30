# Gemma 4 Single-Prompt Benchmark and Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified, attractive chart of Gemma 4 batch-1 prefill and decode performance from three-run 0/64K/128K context measurements and the winning exact MTP depth.

**Architecture:** Keep the two timing domains physically separate. A direct prefix harness measures only the fixed 1,024-token Python coding suffix after an unmeasured 0/64K/128K prefix build; a worker-protocol harness measures only 1,024 committed decode tokens for matched serial and exact-MTP cells and performs the 16K depth sweep. A receipt assembler validates both raw receipts into one immutable chart receipt, and a standard-library renderer produces the SVG without transcribing numbers by hand.

**Tech Stack:** Swift 6.3, MLX Swift, Python 3 standard library, JSON, SVG, macOS `qlmanage` or `sips` for optional PNG rasterization, canonical guarded GPU lifecycle.

**Assumptions:**

- Assumes the exact-verifier plan has passed its real-model correctness and 16K performance gate — will NOT chart an unverified rectangular path as exact.
- Assumes the fixed input matrix remains `/tmp/gemma-prefill-tokenizer.jeuqvl/gemma-python-prefix-matrix.json` with SHA-256 `3c59b96ca9f1d81e2aed48b1a916fc42c81847fa8bec847d48b48305f3dc4cd7` — will NOT mix prompts from another tokenizer or task.
- Assumes prefill means the fixed 1,024-token coding suffix only and excludes prefix construction — will NOT divide the 64K/128K prefix setup by the coding-token count.
- Assumes decode means exactly 1,024 committed tokens and excludes all prompt work — will NOT report end-to-end or aggregate TPS in the prefill/decode fields.

---

## File structure

- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Package.swift` — standalone prefix-benchmark package bound to the selected worktree.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Sources/PrefixBenchCore/Receipt.swift` — prompt splitting, progressive summaries, and receipt validation.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Sources/GemmaPrefixBench/main.swift` — direct fixed-1K suffix prefill harness, derived from the proved `/tmp/gemma-prefix-bench` harness.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Tests/PrefixBenchCoreTests/ReceiptTests.swift` — CPU-only prefix receipt tests.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/gemma_decode_matrix.py` — worker-protocol serial/MTP depth and context decoder.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/test_gemma_decode_matrix.py` — protocol and summary tests.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/build_chart_receipt.py` — joins only complete, compatible raw receipts.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/render_chart.py` — deterministic two-panel SVG renderer.
- `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/test_chart.py` — schema, data-binding, and SVG structure tests.
- `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-*.json` — raw and assembled receipts.
- `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-performance-20260830.svg` — canonical chart.
- `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-performance-20260830.png` — optional raster rendering of the same SVG.

### Task 1: Make the fixed-1K prefix benchmark durable and tested

**Files:**
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Package.swift`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Sources/PrefixBenchCore/Receipt.swift`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Sources/GemmaPrefixBench/main.swift`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/Tests/PrefixBenchCoreTests/ReceiptTests.swift`

**Security flag:** none

**Does NOT cover:** MTP decode or prefix-build throughput; this executable reports only the fixed coding-suffix prefill cell.

- [ ] **Step 1: Write failing CPU receipt tests**

Port the proved helpers from `/tmp/gemma-prefix-bench` and add:

```swift
private enum PrefixBenchFixture {
    static func completeRows(prefixes: [Int], repeats: Int) -> [PrefixSample] {
        prefixes.flatMap { prefix in rows(prefix: prefix, repeats: repeats) }
    }

    static func rows(prefix: Int, repeats: Int) -> [PrefixSample] {
        (1...repeats).map { repeatIndex in
            PrefixSample(
                prefixTokens: prefix, codingTokens: 1_024,
                repeatIndex: repeatIndex,
                promptDigest: "prompt-\(prefix)",
                prefillWallSeconds: 0.25 + Double(repeatIndex) / 100,
                prefillTPS: 1_024 / (0.25 + Double(repeatIndex) / 100))
        }
    }
}

@Test
func summaryRequiresThreeSamplesPerPrefixAndSeparatePhaseNames() throws {
    let rows = PrefixBenchFixture.completeRows(prefixes: [0, 65_536, 131_072], repeats: 3)
    let result = try PrefixReceipt.summary(rows, repeats: 3, requireComplete: true)
    #expect(result["0"]?.prefill.sampleCount == 3)
    #expect(result["65536"]?.prefill.tokensPerSample == 1_024)
    #expect(result["131072"]?.decode == nil)
}

@Test
func incompleteOrMixedPromptRowsAreRejected() {
    #expect(throws: PrefixReceiptError.incompleteCell(prefix: 65_536)) {
        _ = try PrefixReceipt.summary(
            PrefixBenchFixture.rows(prefix: 65_536, repeats: 2),
            repeats: 3, requireComplete: true)
    }
}
```

- [ ] **Step 2: Run the tests and verify the package is absent**

```bash
cd /Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt
swift test
```

Expected: FAIL before package creation, then compile failures until the receipt types exist.

- [ ] **Step 3: Create the package and prefix-only schema**

The manifest depends on the active worktree and its vendored MLX packages. The receipt schema must contain:

```swift
struct PrefixSample: Sendable, Equatable {
    let prefixTokens: Int
    let codingTokens: Int
    let repeatIndex: Int
    let promptDigest: String
    let prefillWallSeconds: Double
    let prefillTPS: Double
}

struct PrefixPhaseSummary: Sendable, Equatable {
    let sampleCount: Int
    let tokensPerSample: Int
    let wallSamples: [Double]
    let tpsSamples: [Double]
    let wallMean: Double
    let tpsMean: Double
    let minimumTPS: Double
    let maximumTPS: Double
}

struct PrefixCellSummary: Sendable, Equatable {
    let prefill: PrefixPhaseSummary
    let decode: PrefixPhaseSummary? = nil
}
```

```json
{
  "schema_version": 1,
  "benchmark": "gemma4_single_prompt_fixed_1k_prefill",
  "status": "complete",
  "workload": {
    "batch_size": 1,
    "concurrency": 1,
    "coding_tokens": 1024,
    "prefix_tokens": [0, 65536, 131072]
  },
  "measurement_protocol": {
    "prefix_setup": "unmeasured 1024-token chunks",
    "prefill": "fixed 1024-token Python suffix only"
  },
  "samples": [],
  "summary_by_prefix": {}
}
```

Copy the working direct-prefill sequence from `/tmp/gemma-prefix-bench/Sources/GemmaPrefixBench/main.swift`: build prefix chunks with `.evaluationOnly`, cool to 40 C, time exactly the coding suffix with `.lastPositionLogits`, and do not execute autoregressive decode in this executable.

- [ ] **Step 4: Enforce memory admission and guard attestation**

Retain the existing projected-128K rule:

```swift
let projected = peak64 + Int(1.25 * Double(max(peak64 - peak0, 0)))
guard UInt64(projected) <= physical - (UInt64(32) << 30) else {
    fail("128K memory admission refused at projected \(projected) bytes")
}
```

Require both `MTPLX_GUARD_ATTEST_FD` and `MTPLX_GUARD_ATTEST_NONCE` before model loading.

- [ ] **Step 5: Run CPU tests**

Expected: all `PrefixBenchCoreTests` pass without MLX execution.

- [ ] **Step 6: Commit the durable harness in the workspace repository that owns `bench/gemma`**

```bash
git add bench/gemma/single_prompt
git commit -m 'bench: add Gemma single-prompt prefill harness'
```

If `/Users/davidtai/projects/OpenSourceWTF` is not itself a Git worktree, record the files and hashes in the final receipt instead of creating a commit in another repository.

### Task 2: Build the matched serial/MTP decode matrix runner

**Files:**
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/gemma_decode_matrix.py`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/test_gemma_decode_matrix.py`

**Security flag:** none

**Does NOT cover:** prefill TPS; `free_decode_begin` is setup and its elapsed time is retained only as unscored diagnostics.

- [ ] **Step 1: Write failing protocol and sample-validation tests**

Start from the proved `/tmp/gemma_mtp_tune.py` worker client and require:

```python
def serial_run_fixture(count):
    return {
        "ok": True, "tokens": list(range(count)),
        "acceptance_lengths": [1] * count,
        "drafted_total": 0, "accepted_total": 0,
        "committed_total": count,
    }

def mtp_run_fixture(count):
    lengths = [3] * 341 + [1]
    return {
        "ok": True, "tokens": list(range(count)),
        "acceptance_lengths": lengths,
        "drafted_total": 682, "accepted_total": 600,
        "committed_total": count,
    }

def diagnostics_fixture():
    return {"ok": True, "completed_work": 1025, "cache_memory": 0}

def mtp_diagnostics_fixture(rectangular_rounds):
    return {
        "ok": True, "completed_work": 343, "cache_memory": 0,
        "rectangular_verify_rounds": rectangular_rounds,
    }

def test_serial_and_mtp_samples_have_separate_decode_only_rates(self):
    serial = validate_sample(
        mode="serial", depth=None, decode_tokens=1024,
        setup_s=10.0, decode_s=8.0,
        begin={"ok": True, "effective_spec": {"mode": "serial"}},
        run=serial_run_fixture(1024), diagnostics=diagnostics_fixture())
    mtp = validate_sample(
        mode="mtp", depth=2, decode_tokens=1024,
        setup_s=10.0, decode_s=4.0,
        begin={"ok": True, "effective_spec": {"mode": "mtp", "mtp": {"depth": 2}}},
        run=mtp_run_fixture(1024), diagnostics=mtp_diagnostics_fixture(rectangular_rounds=400))
    self.assertEqual(serial["decode_tps"], 128.0)
    self.assertEqual(mtp["decode_tps"], 256.0)
    self.assertNotIn("prefill_tps", serial)

def test_mtp_rejects_zero_rectangular_rounds_at_batch_one(self):
    with self.assertRaisesRegex(ValueError, "rectangular verification"):
        validate_sample(
            mode="mtp", depth=2, decode_tokens=1024,
            setup_s=10.0, decode_s=4.0,
            begin={"ok": True, "effective_spec": {"mode": "mtp", "mtp": {"depth": 2}}},
            run=mtp_run_fixture(1024),
            diagnostics=mtp_diagnostics_fixture(rectangular_rounds=0))
```

Also reject wrong `effective_spec`, non-1 batch/concurrency, fewer than 1,024 committed tokens, unstable digests within a cell, missing acceptance totals, and unexpected worker teardown codes.

- [ ] **Step 2: Run tests and verify the runner is absent**

```bash
python3 -m unittest -v /Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/test_gemma_decode_matrix.py
```

Expected: FAIL because `gemma_decode_matrix.py` does not exist.

- [ ] **Step 3: Implement depth tuning and final matrix schedules**

Use these immutable schedules:

```python
DEPTH_PROMPT_TOKENS = 16_384 + 1_024
DEPTHS = (1, 2, 3)
PREFIXES = (0, 65_536, 131_072)
REPEATS = 3
DECODE_TOKENS = 1_024

depth_order = [
    ("serial", None), ("mtp", 1), ("mtp", 2), ("mtp", 3),
    ("mtp", 1), ("mtp", 2), ("mtp", 3), ("serial", None),
    ("mtp", 2), ("mtp", 3), ("serial", None), ("mtp", 1),
]
```

After selecting the highest mean decode TPS that beats serial, rotate the six final cells so each mode/context appears once in each ordinal position across the three repetitions.

- [ ] **Step 4: Emit an atomic progressive receipt**

The receipt includes exact source/worker/artifact hashes, the full schedule, discarded primers, all measured samples, arithmetic means, min/max values, token digests, acceptance, committed tokens per round, rectangular verification rounds, peak memory, and winner depth. Write `status: running` after every sample and `status: complete` only after all cells validate.

- [ ] **Step 5: Run Python tests**

Expected: all protocol, order, summary, and failure tests pass.

- [ ] **Step 6: Commit**

```bash
git add bench/gemma/single_prompt/gemma_decode_matrix.py bench/gemma/single_prompt/test_gemma_decode_matrix.py
git commit -m 'bench: add Gemma serial and exact-MTP decode matrix'
```

Apply the same workspace-repository caveat from Task 1.

### Task 3: Validate and render the publication chart

**Files:**
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/build_chart_receipt.py`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/render_chart.py`
- Create: `/Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/test_chart.py`

**Security flag:** none

**Does NOT cover:** inventing or interpolating missing benchmark cells; incomplete receipts are rejected.

- [ ] **Step 1: Write failing receipt-join and SVG tests**

Use synthetic complete receipts and assert:

```python
def prefill_fixture():
    return {
        "schema_version": 1, "status": "complete",
        "source": {"revision": "a" * 40},
        "artifacts": {"matrix_sha256": "b" * 64},
        "workload": {"batch_size": 1, "concurrency": 1,
                     "coding_tokens": 1024,
                     "prefix_tokens": [0, 65536, 131072], "repeats": 3},
        "summary_by_prefix": {
            str(prefix): {"samples": [1000.0, 1010.0, 1020.0]}
            for prefix in (0, 65536, 131072)
        },
    }

def decode_fixture():
    cells = []
    for prefix in (0, 65536, 131072):
        cells += [
            {"mode": "serial", "depth": None, "prefix_tokens": prefix,
             "samples": [100.0, 101.0, 102.0]},
            {"mode": "mtp", "depth": 2, "prefix_tokens": prefix,
             "samples": [140.0, 141.0, 142.0],
             "rectangular_verify_rounds": [340, 341, 339]},
        ]
    return {
        "schema_version": 1, "status": "complete",
        "source": {"revision": "a" * 40},
        "artifacts": {"matrix_sha256": "b" * 64},
        "workload": {"batch_size": 1, "concurrency": 1,
                     "coding_tokens": 1024, "decode_tokens": 1024,
                     "prefix_tokens": [0, 65536, 131072], "repeats": 3},
        "winner_depth": 2, "cells": cells,
    }

chart = build_chart_receipt(prefill_fixture(), decode_fixture())
self.assertEqual(chart["workload"]["prefix_tokens"], [0, 65536, 131072])
self.assertEqual(chart["workload"]["repeats"], 3)
self.assertEqual(len(chart["prefill"]), 3)
self.assertEqual(len(chart["decode"]), 6)

svg = render(chart)
self.assertIn("Gemma 4 26B A4B", svg)
self.assertIn("Prefill · fixed 1K Python input", svg)
self.assertIn("Decode · 1K generated tokens", svg)
self.assertIn("Batch 1 · concurrency 1 · mean of 3", svg)
self.assertNotIn("aggregate TPS", svg)
```

Reject mixed source revisions, matrix hashes, coding/decode counts, batch sizes, incomplete cells, non-finite values, or a decode receipt whose MTP winner does not have positive rectangular rounds.

- [ ] **Step 2: Run tests and verify both modules are absent**

Expected: FAIL on import.

- [ ] **Step 3: Implement the canonical chart receipt**

The assembler stores, for every point, the three samples, arithmetic mean, minimum, and maximum. Prefill has one control series; decode has `Serial AR` and `Exact MTP depth N`. Store SHA-256 hashes of both raw receipts and the renderer source.

- [ ] **Step 4: Implement deterministic SVG rendering**

Render a 1600×900 SVG with:

- warm off-white background `#F6F2EA`;
- navy text `#12263A`;
- serial bars/line `#2F6BFF`;
- exact-MTP bars/line `#FF6B4A`;
- three context groups labeled `0 prefix`, `64K prefix`, `128K prefix`;
- two independently zero-based panels, prefill on the left and decode on the right;
- mean labels above each mark and min-to-max whiskers;
- a small 16K depth-sweep callout with winning depth, acceptance, and committed tokens per round;
- footer containing short source SHA, artifact SHA, receipt SHA, and `mean of 3`.

Attach `data-series`, `data-prefix`, `data-mean`, `data-min`, and `data-max` attributes to every plotted element so tests can prove the SVG values came from the receipt.

- [ ] **Step 5: Run chart tests**

Expected: all schema and SVG tests pass.

- [ ] **Step 6: Commit**

```bash
git add bench/gemma/single_prompt/build_chart_receipt.py bench/gemma/single_prompt/render_chart.py bench/gemma/single_prompt/test_chart.py
git commit -m 'bench: render Gemma single-prompt performance chart'
```

### Task 4: Run the guarded benchmark and verify the final artifacts

**Files:**
- Create: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-prefill-0-64k-128k-3x-20260830.json`
- Create: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-decode-ar-mtp-0-64k-128k-3x-20260830.json`
- Create: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-chart-receipt-20260830.json`
- Create: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-performance-20260830.svg`
- Create if rasterization succeeds: `/Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-performance-20260830.png`

**Security flag:** none

- [ ] **Step 1: Build and hash the exact candidate worker and prefix executable**

Builds do not execute MLX. Record the source SHA and executable hashes in a shell transcript before the guarded runs.

- [ ] **Step 2: Run the prefix benchmark under one canonical guarded window**

Use exactly three repeats, prefixes `0,65536,131072`, the pinned weights/matrix, 1,024 prefix chunks, the 40 C thermal gate, and the 32 GiB admission reserve. Do not run the executable unless the guard attestation is present.

Expected: complete receipt with nine measured fixed-1K prefill samples and three discarded primers.

- [ ] **Step 3: Run the depth tune plus matched decode matrix under one canonical guarded window**

Invoke `gemma_decode_matrix.py` with the release worker, pinned weights, `/tmp/gemma-mtp-head`, pinned matrix, three repeats, and 1,024 decode tokens. The runner performs the 16K depth sweep, selects the exact winner only if it beats serial, then runs serial and winner at 0/64K/128K.

Expected: complete receipt with three samples per depth/control tune cell and three samples per final mode/context cell; all B1 MTP samples have positive rectangular rounds and valid digests.

- [ ] **Step 4: Verify postflight after each guarded window**

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
lsof /tmp/mtplx-gpu-exclusive.lock
```

Expected: zero active requests, understood background warmup state, exactly `mtplx-flash-next-optimized-speed`, and no lock owner.

- [ ] **Step 5: Assemble and render**

```bash
python3 /Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/build_chart_receipt.py \
  --prefill /Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-prefill-0-64k-128k-3x-20260830.json \
  --decode /Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-decode-ar-mtp-0-64k-128k-3x-20260830.json \
  --output /Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-chart-receipt-20260830.json
python3 /Users/davidtai/projects/OpenSourceWTF/bench/gemma/single_prompt/render_chart.py \
  --receipt /Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-chart-receipt-20260830.json \
  --output /Users/davidtai/projects/OpenSourceWTF/benchmark-results/gemma4-single-prompt-performance-20260830.svg
```

- [ ] **Step 6: Prove chart-data identity and inspect the rendered image**

Run the chart tests against the real receipt, parse every SVG `data-*` field back into numeric values, and compare with the receipt. Rasterize with `qlmanage -t -s 1600` or `sips` if available, then inspect the SVG/PNG for clipped text, overlapping labels, misleading scales, and low contrast.

Expected: numeric identity passes; both panels are readable at 1600×900; prefill and decode scales are visibly separate.
