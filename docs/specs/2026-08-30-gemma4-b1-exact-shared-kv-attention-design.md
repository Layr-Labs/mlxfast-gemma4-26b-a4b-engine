# Gemma 4 B1 Exact Shared-KV MTP Attention Design

Date: 2026-08-30  
Status: approved in conversation; pending written-spec review  
Worktree: `mlxfast-gemma4-mtp-depth3-tip`

## Goal

Replace the long-context target verifier's repeated full-K/V attention scans
with one construction-bound physical-B1/C2-C4 attention implementation that
shares each immutable K/V traversal across verification columns while keeping
every column bit-identical to an independent ordinary B1/L1 decode attention
call.

This is the first fix. Long-context draft-depth/acceptance policy is retuned
only after the attention change passes exactness and end-to-end performance
gates.

## Premise and measured cause

The issue is established on the exact 1,024-token Python suffix workload:

- At 64K extra prefix, MTP D1/D2/D3 rounds cost 2.54/3.52/4.91 ordinary AR
  token times while committing only 1.64/2.29/2.46 tokens per round.
- The target has five full-attention layers; the drafter has one.
- The installed verifier sets `mtpSerializesRectangularAttention`, which sends
  every target column through `attendSerialQueries(... blockSize: 1)`.
- For depth `k`, the resulting full-history scans are approximately
  `5 * (k + 1) + k`. Relative to AR's five scans, this predicts
  2.20/3.40/4.60 round-cost multiples, closely matching the measurements.

The cost of not building this lane is that exact MTP remains slower than AR at
16K and loses roughly half of AR decode throughput at 64K D3. A shape-specific
kernel is proportional because the problem is the dominant cost and the repo
already contains exact frozen-MLX D512 attention transcriptions and exact
multi-column projection patterns.

## Scope

This change will:

1. Add an exact shared-K/V attention implementation for:
   - physical batch 1;
   - verification columns 2, 3, or 4;
   - BF16 Q/K/V;
   - 16 query heads, 2 KV heads, GQA 8, head dimension 512;
   - scale 1.0, no sinks, no attention softcap;
   - full-attention layers only.
2. Preserve the existing serialized attention path for the 25 sliding-window
   layers, whose retained history is bounded at 1,024 tokens.
3. Install the full-attention entrypoints once after the existing Gemma
   topology and verifier checks succeed.
4. Prove operator, cache, token, and performance behavior before promotion.

## Non-goals

- Approximate/tolerance-only attention or accepting different output tokens.
- Changing target or drafter weights.
- Claiming that code can improve the assistant model's intrinsic long-context
  draft accuracy.
- Batches above one, verification widths outside C2-C4, other head dimensions,
  fp16, sinks, attention softcaps, or vision-span attention.
- Hiding an AR execution path under an MTP label.
- Optimizing the drafter's one full-attention layer in this first change.

## Architecture

### Construction boundary

Add `Gemma4B1MTPFullAttentionV1` beside the existing CBv2 attention helpers.
Its binder accepts the fixed layer geometry and C2-C4 width and returns a typed
callable only when every invariant matches. Gemma verifier installation binds
one callable per supported width for full-attention caches. Failure to bind
prevents installation before warmup; the enabled lane has no stock fallback,
environment reads, eligibility checks, or proof counters.

The cache bank retains its explicit ordinary-versus-MTP phase switch. Within
the MTP phase, each cache has a construction-installed strategy:

- full attention: exact shared-K/V C2-C4 callable;
- sliding attention: existing exact serial-query callable.

Layer kind and geometry are construction invariants, not revalidated per
token, layer, or round.

### Per-round data flow

The existing cache transaction remains authoritative:

1. Append all C speculative K/V positions once.
2. Form the same updated K/V views used by the serial-query oracle.
3. Let `history = keyLength - C`.
4. Verification column `j` attends exactly keys `0 ..< history + j + 1`.
5. Return `[1, 16, C, 512]` attention output to the unchanged verifier body.
6. Existing acceptance and rollback commit the accepted prefix.

The new callable does not own cache writes, positions, acceptance, or rollback.

### Exact shared-K/V attention

The implementation follows the frozen ordinary D512 composed attention
arithmetic in three stages:

1. **QK stage:** load each K packet once and apply it independently to all C
   query columns. Each column retains the ordinary L1 lane mapping, dot-product
   accumulation order, BF16 score store, and its own visible key length.
2. **Softmax stage:** normalize every score row independently with the exact
   ordinary long-row reduction geometry for that visible length. Softmax may
   remain one dispatch per column because it does not reread K/V; padding a
   shorter column to another column's reduction width is forbidden.
3. **AV stage:** load each V packet once and apply the C independent probability
   rows. Each output column retains the ordinary L1 accumulation and reduction
   order and ignores values outside its visible length.

QK and AV share K/V loads across columns. Mathematical equivalence is
insufficient: every stage is compared bit-for-bit with concatenated independent
L1 calls at the exact MLX kernel-selection boundaries used by the workload.

## Interfaces and contracts

`Gemma4B1MTPFullAttentionV1.bind(columns:geometry:)` returns an immutable
callable with this logical interface:

```swift
(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    historyLength: Int
) -> MLXArray
```

Required shapes are:

- queries: `[1, 16, C, 512]`;
- keys and values: `[1, 2, historyLength + C, 512]`;
- output: `[1, 16, C, 512]`.

For every `j in 0..<C`, output column `j` must equal the ordinary decode
attention result over query `j` and the K/V prefix ending at
`historyLength + j + 1`.

No callable is published unless all C2-C4 entrypoints self-check successfully.

## Error handling

- Unsupported construction geometry: fail once with the violated invariant;
  publish no partial verifier contexts.
- Kernel compile or exact self-check failure: do not install the shared-K/V
  lane; fail the experimental worker before measured generation.
- Cache/token mismatch: fail the benchmark gate and retain the prior explicit
  serial-attention implementation in source control; do not add runtime
  fallback.
- GPU interruption: the guard terminates only its owned child, restores and
  verifies the exact production service, then releases the lock.

## Testing strategy

Implementation is test-driven:

1. CPU-only route tests first prove that only full D512 B1/C2-C4 shapes can
   install and that sliding layers remain serialized.
2. GPU operator tests compare C2/C3/C4 output bit-for-bit against C independent
   ordinary L1 calls at short, 1,024, 4,095/4,096, 16K, 64K, and 128K key
   lengths where memory admission permits.
3. Tests use distinct query columns and adversarial score ranges so a kernel
   cannot pass by duplicating one column or relying on benign softmax values.
4. Cache tests prove per-column visible ends, speculative append ownership,
   rollback lengths, and final cache tensors.
5. An isolated full-attention benchmark compares unchanged serialized C2-C4
   against the candidate at the exact real geometry, with one primer and three
   measured samples.
6. The real-model gate repeats exact token/cache parity at 0/16K/64K for depths
   1-3, then the 16K 1,024-token mean-of-three decode tune.
7. Promotion requires a statistically attributable attention win and an
   end-to-end decode improvement versus unchanged AR. Correct-but-slower code
   is not installed.

All MLX/Metal tests and benchmarks run only while the parent holds
`/tmp/mtplx-gpu-exclusive.lock` through the canonical service guard.

## Failure-mode check

### Reduction-order drift changes tokens — critical

Using a conventional multi-query or online-softmax kernel could change QK,
softmax, or AV reduction tiling. The design instead transcribes the ordinary
L1 stages and requires bit equality at kernel-selection boundaries and in the
real model before installation.

### A column sees the wrong speculative K/V suffix — critical

All C K/V entries exist in the updated cache view, but column `j` may see only
through `history + j`. The callable receives the shared history length and
uses an independent visible length for every stage; cache tests exercise every
column and rollback boundary.

### K/V sharing loses to register pressure or occupancy — critical to speed

C4 independent accumulators can reduce occupancy enough to erase bandwidth
savings. The isolated benchmark gates C2, C3, and C4 separately. A losing width
is not installed, and no claimed MTP win may depend on it.

### Drafter attention becomes the next bottleneck — minor for this change

After target K/V scans collapse, the drafter still performs one full-history
scan per draft step. This is expected and measured after the target fix; it is
a separate follow-up rather than bundled into an unattributable change.

## Rollout

1. Add failing route and operator tests.
2. Implement and self-check the C2 entrypoint, then C3 and C4 independently.
3. Benchmark each width against unchanged serialized attention.
4. Install only exact winning widths at construction.
5. Run the real-model correctness and performance gates.
6. Retune fixed depth and then evaluate the existing adaptive cost/acceptance
   controller. Do not alter acceptance policy before the kernel result is
   attributable.
7. Resume the final 0/64K/128K prefill/decode matrix only after this gate.
