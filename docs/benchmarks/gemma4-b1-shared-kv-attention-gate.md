# Gemma 4 B1 Shared-KV Attention Gate

Date prepared: 2026-08-30

## Candidate

`Gemma4B1MTPFullAttentionV1` is an unpromoted physical-B1/C2-C4 full-attention
candidate. Production remains on the unchanged serial-query control. The first
test must establish byte equality before any cache integration or real-model
performance claim.

The focused suite compares every column with an independent ordinary B1/L1
`MLXFast.scaledDotProductAttention` call at key lengths:

```text
4, 1024, 4095, 4096, 4097, 8191, 8192, 8193, 16384, 65536, 131072
```

These cover the block/looped precise-softmax boundary and verify that Gemma's
512-wide AV output retains MLX's SM8/SN4 reduction through 8,192 and beyond.
The frozen MLX SM4/SN8 branch also requires output width at least 2,048 and is
therefore not part of this operator.

## Static memory admission

Largest fixture: B1, C4, QH16, KVH2, D512, BF16, key length 131,072.

| Allocation | Bound |
|---|---:|
| Q | 0.0625 MiB |
| K | 256 MiB |
| V | 256 MiB |
| candidate scores | <16 MiB |
| candidate probabilities | <16 MiB |
| candidate output | 0.0625 MiB |
| ordinary oracle scores/probabilities/output | <32.125 MiB |
| **Operator tensors total** | **<576.25 MiB** |

Use 640 MiB as the conservative tensor bound. The test clears the MLX cache
after every fixture. This admits only the model-free operator test; it is not a
memory admission for loading the Gemma model or running the real-model gate.

## Guarded command

```bash
/opt/homebrew/bin/python3 \
  ~/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist ~/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --timeout-seconds 1800 \
  --child-timeout-seconds 3600 \
  -- /bin/zsh -lc 'cd ~/projects/OpenSourceWTF/.worktrees/mlxfast-gemma4-mtp-depth3-tip && MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_RUN_MLX_LONG_RUNTIME_TESTS=1 swift test --filter Gemma4B1MTPFullAttentionTests'
```

The guard must acquire and retain `/tmp/mtplx-gpu-exclusive.lock` before it
unloads the service. After the child exits it must restore the exact captured
service, then the operator must verify `/health`, `/v1/models`, process state,
memory/swap state, and lock release.

## Pass criteria

- C2, C3, and C4 output shapes are `[1, 16, C, 512]`.
- Every candidate element equals the independent ordinary B1/L1 element.
- No test is skipped.
- The guard restores `mtplx-flash-next-optimized-speed` and releases the lock.

Only after this passes may the candidate be installed into the contiguous and
paged cache diagnostic seams and benchmarked against the unchanged serial
control. Production promotion and depth/acceptance tuning remain later gates.

## Result

Run completed 2026-08-30 under the canonical guard at the current dirty
worktree tip.

- Guarded child exit: 0.
- Swift build: passed.
- Focused suite: 3 tests passed in 0.884 seconds.
- Exact boundary test: passed for C2-C4 at every listed length.
- Long-context test: passed for C2-C4 at 65,536 and 131,072.
- Binder test: passed; only C2-C4 are published.
- Postflight: `mtplx-flash-next-optimized-speed` healthy and idle, warmup done,
  exact model endpoint restored, no lock owner, no test child remaining.

During diagnosis, an incorrect proposed AV geometry transition at 8,192 was
removed. Frozen MLX selects SM4/SN8 only when both the reduction length is at
least 8,192 and the output width is at least 2,048; Gemma AV output width is
512, so the exact ordinary route remains SM8/SN4 at all tested contexts.

## Isolated performance gate

The candidate was then measured against the unchanged control: C independent
ordinary physical-B1 `scaledDotProductAttention` calls, one per verification
column. Graph construction and first-use compilation were excluded. Each cell
used three synchronized, interleaved control/candidate samples; the arithmetic
mean is reported. `column_tps` is isolated attention-column throughput, not
model decode TPS.

```bash
/opt/homebrew/bin/python3 \
  ~/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist ~/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --timeout-seconds 1800 \
  --child-timeout-seconds 3600 \
  -- /bin/zsh -lc 'cd ~/projects/OpenSourceWTF/.worktrees/mlxfast-gemma4-mtp-depth3-tip && MLXFAST_RUN_MLX_BENCHMARKS=1 swift test --filter Gemma4B1MTPFullAttentionTests.benchmarkSharedKVAgainstSerialWidthOneControl'
```

| Key length | C | Serial mean ms | Candidate mean ms | Serial column_tps | Candidate column_tps | Serial/candidate |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 2 | 1.350 | 2.147 | 1,481.47 | 931.51 | 0.629x |
| 16,384 | 3 | 1.326 | 1.987 | 2,262.94 | 1,509.59 | 0.667x |
| 16,384 | 4 | 1.561 | 5.113 | 2,562.64 | 782.30 | 0.305x |
| 65,536 | 2 | 3.055 | 4.478 | 654.63 | 446.62 | 0.682x |
| 65,536 | 3 | 4.486 | 6.880 | 668.70 | 436.02 | 0.652x |
| 65,536 | 4 | 5.888 | 18.782 | 679.36 | 212.97 | 0.313x |
| 131,072 | 2 | 5.727 | 8.832 | 349.20 | 226.45 | 0.648x |
| 131,072 | 3 | 8.529 | 13.688 | 351.75 | 219.17 | 0.623x |
| 131,072 | 4 | 11.310 | 38.441 | 353.67 | 104.05 | 0.294x |

All timed cells retained exact output parity. The candidate lost every cell:
it took 1.47-1.59x the serial wall time at C2, 1.50-1.60x at C3, and
3.19-3.40x at C4. The isolated performance gate therefore fails, and the
production `.serializedDecode` route remains unchanged.
