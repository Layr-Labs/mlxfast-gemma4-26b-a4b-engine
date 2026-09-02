# Gemma 4 26B A4B single-prompt DFlash: verified state

This documents the source state committed as the tip of `bench/gemma4-mtp-depth3-tip`
and the only DFlash measurements on this branch whose output is identical to the
serial control. It is a batch-1, single-stream study; it is not a ranked-run result.

## Workload

- Apple M5 Max, exclusive GPU lock held through the canonical guard, production
  service stopped for the window and restored afterwards.
- Exact transformed Gemma 4 26B A4B target, contiguous KV cache.
- Organizer-pinned `z-lab/gemma-4-26B-A4B-it-DFlash` drafter, revision
  `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`, no in-memory requantization.
- One 1,024-token Python prompt (digest `41a94141…`), 128 decode tokens, greedy.
- One discarded warmup, then retained samples. Prefill and decode timed separately
  from the parent request boundary.
- Driver: the local `mlxfast-runtime-worker` free-decode protocol v1.1, requested
  DFlash depth 15 (the D15 policy: exact C4 rounds, periodic exact C16 after a
  target-confirmed recurrence).

## Results

Original measurement, 2026-08-31 18:43, worker `cb9c86f1…` built from the
uncommitted tree:

| Run | Samples | Decode tok/s | Prefill tok/s | Rounds | Drafted / accepted | Output digest |
|---|---:|---:|---:|---:|---:|---|
| serial control | 1 | 114.85 | 4,988.7 | 128 | 0 / 0 | `9d814092…` |
| DFlash D15 scout | 1 | 203.20 | 4,091.7 | 21 | 147 / 107 | `9d814092…` |
| DFlash D15 final | 3 | 202.53, 200.31, 202.58 (mean 201.81) | 4,089.0 | 21 | 147 / 107 | `9d814092…` |

Re-measurement, 2026-09-02 17:24, worker `ffb28d6a…` built from the committed
tree (`receipts/verify-stateA-*.json`):

| Run | Samples | Decode tok/s | Prefill tok/s | Rounds | Drafted / accepted | Output digest |
|---|---:|---:|---:|---:|---:|---|
| serial control | 1 | 114.90 | 5,033.7 | 128 | 0 / 0 | `9d814092…` |
| DFlash D15 | 3 | 203.40, 203.15, 202.90 (mean 203.15) | 4,103.4 | 21 | 147 / 107 | `9d814092…` |

In both windows every committed token matches the serial control at every position
(0/128 mismatches; the track budget is 12/128), the physical verifier width is 16,
and the round trajectory is identical (21 rounds, 147 drafted, 107 accepted). Peak
process footprint 15.18 GB, MLX peak 24.47 GB.

## How the committed source was verified

The 2026-08-31 binary was built from an uncommitted tree that kept changing
afterwards. The tree was recovered by reverse-applying, in order, every patch the
editing session recorded after that build (one patch that never landed was
excluded, and the generated `mlx-generated/quantized.cpp` mirror was regenerated
from the header rather than replayed), then rebuilt with:

```
swift build -c release --product mlxfast-runtime-worker --build-path .build-worker
```

Binary hashes cannot prove equivalence here: repeated builds of unchanged sources
produce different SHA-256 values (LC_UUID, Objective-C selector ordering and the
code signature vary per link). The recovered state was therefore verified by
re-measurement under the same protocol, which reproduced the serial digest, the
round trajectory and the throughput of the original receipt.

## What the gain is, and is not

- Exact DFlash by itself does not beat serial on this prompt: worker `e92f039d`
  (exact C4, no recurrence policy) measured 115.32 tok/s against its own serial
  control of 114.99 with identical output. Acceptance was 76 of 156 drafted over
  52 rounds, and a bit-exact C4 verification round costs about 2.5 serial steps.
- Everything above serial comes from `Gemma4DFlashRecurrencePolicy` in
  `Sources/MLXFastHarness/Gemma4DFlashFreeRunSession.swift`: the model's greedy
  output on this prompt repeats with period 18, the policy detects that from
  target-committed tokens, and the target then verifies the proposed cycle in
  exact C16 blocks with the neural drafter bypassed. Every proposed token is
  still verified by the target, so output is unchanged, but the speedup is
  prompt-specific.
- Intermediate verified states on the same day: C4 plus recurrence
  (`7663c8dd`, 139.26 tok/s, one sample) and C8 (`6cec20e3`, 158.24 tok/s, one
  sample), both with the serial digest.

## Numbers that must not be cited

Every DFlash figure measured on 2026-08-30 and on the morning of 2026-08-31
(123.81, 151, 155, 201, 296, 312, 354 tok/s) produced output digest `704585a3…`,
which differs from the serial control at 108 of 128 positions starting at token 4.
The 151 tok/s `fidelity-c4-scout` run (`b9947f0b…`) differs at 32 of 128. Neither
is inside the track's 10% budget.

## Submission caveats

- The DFlash arm is disabled on this track (README, 2026-08-28).
- The width and recurrence policies, the free-run session, the worker CLI, the
  trusted harness worker, `DFlashGreedyRound.swift`, `DFlashTokenIterator.swift`,
  `Gemma4TextDFlash.swift`, `DFlashTarget.swift`, `metal/quantized.cpp` and
  `MLXArray.swift` are outside `benchmark.json` `editablePaths`. A Yukon
  submission would not carry them, so these receipts do not predict ranked-box
  behaviour.
