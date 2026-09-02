# Gemma 4 DFlash early-recurrence C16 implementation plan

## Task 1: Early recurrence policy

- Add CPU tests that prove period-18 promotion after 23 tokens, reject the
  period-3 false positive at confirmation two, preserve shortest-period and
  nonconstant rules, and preserve mismatch replay/demotion semantics.
- Run the focused tests and confirm they fail for the current strict policy.
- Implement the fixed three-target-token confirmation rule without adding hot
  instrumentation or invariant checks.
- Rerun the focused tests.

## Task 2: Fixed C16 primitives

- Add CPU route/source tests for explicit C16 projection, attention, stable
  rank-128 expert ordering, glue, and tied-head entrypoints.
- Confirm the tests fail before production changes.
- Implement fixed C16 shared serial-reduction projection and attention-output
  bindings, rank-128 expert routing, and exact glue/head bindings.
- Run focused CPU tests. Runtime-only parity remains gated off until guarded.

## Task 3: Atomic model and cache installation

- Add failing tests for immutable C2/C3/C4/C8/C16 installation, D15 exact
  C4/C16 dispatch, and C16-minus-one cache tail capacity.
- Implement construction-time C16 binding and atomic publication after the
  prefill evaluation barrier. Keep D1-D14 unchanged.
- Run focused installation/cache tests.

## Task 4: CPU and release verification

- Run `MLXFAST_RUN_MLX_RUNTIME_TESTS=0 swift test --filter Gemma4`.
- Run the full CPU suite and classify only pre-existing unrelated failures.
- Build the release worker and record its SHA-256 digest.
- Perform independent requirements and quality reviews; resolve all findings.

## Task 5: Guarded runtime selection and parity

- Use the canonical guard and exclusive GPU lock for every MLX/Metal action.
- Establish C16 memory safety from the existing C96/C8 peak plus static delta.
- Bit-compare C16 primitives with their unchanged controls.
- Compare direct C16 projection with fixed two-C8 chunking if the direct kernel
  exhibits register-pressure loss; install only the measured winner.
- Verify the production service is restored and healthy after each run.

## Task 6: Exact performance gate

- Run the exact 1K Python prompt, batch one, concurrency one, 128-token scout
  with one discarded warmup and one retained sample.
- Require the serial digest and zero positional mismatches.
- If decode is at least 200 tokens/s, run one warmup plus three retained samples
  and require every sample and their mean to be at least 200 tokens/s.
- If it misses, profile the measured path and iterate from evidence without
  weakening the acceptance gate.
