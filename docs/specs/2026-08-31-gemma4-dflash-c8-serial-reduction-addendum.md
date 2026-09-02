# Gemma 4 DFlash C8 serial-reduction addendum

## Status

Proposed correction to Task 4 of
`2026-08-31-gemma4-dflash-fidelity-200-tps-design.md`.

## Why Task 4 must change

The current same-worker evidence invalidates the planned C8/C16/C32
`stockModule` route:

- worker SHA `e92f039db19ff8fef039d7a0197ed256027792c927b9f68a430b899b0960182a`;
- the installed C2 verifier is bit-identical to two serial B1/L1 forwards at
  every captured layer and at final logits;
- the stock rectangular C2 control first differs at layer-zero K/V because
  the dirty vendored MLX runtime selects `qmv_wide` for `M >= 2`;
- replacing the attention-output rectangle with construction-bound M1
  columns repaired exact C4 from 32/128 mismatches to 0/128;
- recurrence worker SHA
  `7663c8dd3c9b4b2947645cd976287556ea40281defc94ac6be7ab5b48bcabfd6`
  retains 0/128 mismatches and reaches 139.264382 decode tok/s, versus its
  same-SHA serial control at 114.843865 tok/s.

Therefore a stock rectangular C8 context would knowingly reinstall the
reduction-tree regression that the C4 gate just removed. It is not an
admissible candidate.

## Decision

Implement and measure only fixed C8 first. C8 is installed at model
construction alongside C2-C4, but every quantized projection must use an
explicit serial-reduction-preserving binding. No stock rectangular projection,
runtime eligibility check, fallback, environment read, or engagement counter
is allowed in the enabled C8 path.

If C8 is correct but below 200 tok/s, C16/C32 require a new geometry decision
from the C8 profile; they are not aliases of C8 and are not silently admitted
by a numeric range.

## Fixed route

The certified shape set becomes exactly B1/C2, B1/C3, B1/C4, and B1/C8.
Construction assembles every layer, glue, target-hidden capture, and tied-head
binding in a local table and publishes the table only after all four contexts
succeed.

For C8:

1. Q/K/V, dense gate/up/down, router, and tied head use the existing
   `Gemma4B1MTPQuantizedProjection` arithmetic, extended only to the fixed C8
   specialization. Its per-column accumulator retains ordinary QMV lane
   arithmetic while immutable packed weights and affine metadata are shared.
2. Attention remains serialized decode. Attention output uses a separately
   named C8 shared serial-reduction binding; C2-C4 keep their independent M1
   binder. The C8 binding is not promoted unless the full-model token gate
   passes.
3. The expert route uses the existing B1 combined arithmetic with the existing
   64-assignment route kernel for exactly `8 columns * topK 8`. It does not use
   the generic stock `SwitchGLU` rectangle.
4. The existing rectangular glue is extended only where its row-generic
   arithmetic and output shapes are unchanged.
5. The D15 target cache prebinds C4 and C8 after prefill. The measured loop
   performs only the runtime-varying phase/width lookup and invokes the
   prebound closure directly.

No arbitrary width, C16, C32, C64, or C96 route is introduced by this step.

## Tests before runtime

CPU/source tests must prove:

- the certified set is exactly `[2, 3, 4, 8]`;
- C2-C4 bindings and arithmetic remain unchanged;
- C8 never returns a stock-module strategy;
- C8 uses the serial-reduction projection binding for every ordinary
  quantized projection and the 64-assignment expert route;
- missing C8 projection, glue, expert, attention-output, or head bindings fail
  before context publication;
- the D15 cache requires both C4 and C8 and has capacity for the full C8
  provisional tail;
- the enabled D15 loop has no ordinary target fallback.

Focused suites and the full CPU suite must introduce no new failure. The known
dirty-tree wire-fixture digest mismatch may remain the sole unrelated failure
and must be reported rather than rewritten.

## Guarded evidence sequence

After release build and worker SHA capture:

1. Run a fresh exact 1K Python / 128-token serial control with token capture.
2. Run one warmup plus one retained D15/C8 candidate under the canonical
   parent-held GPU guard.
3. Require physical verifier width 8, 128 committed tokens, identical worker
   SHA, safe memory, exact production-service restoration, and no more than
   12 positional mismatches.
4. If C8 exceeds 12 mismatches, stop and localize the first differing
   operation; do not widen.
5. If C8 is within fidelity but below 200 tok/s, retain its phase timings and
   projection evidence, then design C16 from those measurements.
6. Only a C8 scout at or above 200 tok/s advances to the final warmup plus
   three retained samples, where every retained sample and the mean must be at
   least 200 decode tok/s.

Prefill TPS remains reported separately and is never mixed with decode TPS.

