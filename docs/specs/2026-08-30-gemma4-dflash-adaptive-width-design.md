# Gemma 4 DFlash Adaptive Width Design

## Goal

Raise exact single-prompt Gemma 4 DFlash decode throughput above 150 tok/s on
the pinned 1,024-token Python prompt and 128-token decode while preserving the
retained C4 token digest.

## Evidence

- Fixed C4 averages 126.843 tok/s and produces digest
  `704585a3a78c96932198c35e98e9c9e9018e16951e5ccb40a9bd7f3b4941c37d`.
- Fixed C8 is slower overall because early acceptance is poor, but its final
  full-width rounds accept every proposed token.
- A guarded late-prefix probe ran the first 50 tokens at C4 and the remaining
  78 at C8. Every measured C8 suffix accepted `[8,8,8,8,8,8,8,8,8,6]`, the
  combined digest matched C4, and the three-sample reconstructed-inline mean
  was 150.255 tok/s.
- The generic `dflash-mlx` adaptive policy is not suitable here: its exact
  full-width reference run averaged only 106.619 tok/s.

## Design

The session receives a requested depth exactly as it does today. Requested
depths other than D7 remain fixed controls. A D7 request constructs a small
value-type policy with D3 as its initial depth and D7 as its installed upper
depth.

After each round, the policy observes only values already required by the
round loop: the physical block width, remaining-token emission limit, and
committed token count.

- At D3/C4, two consecutive fully committed C4 rounds promote the next round
  to D7/C8.
- A partial full-width C4 round resets the promotion streak.
- At D7/C8, a partial full-width C8 round immediately demotes the next round to
  D3/C4.
- A final block whose input or emission is narrowed by the remaining-token
  budget does not alter policy state.

The requested D7 value remains the public/effective-spec ceiling. The current
depth is private session state. C4 uses the already installed B1/C2-C4 verifier
table; C8 follows the explicit stock rectangular route already selected by the
immutable verifier table. There is no failed-custom fallback.

## Performance-path constraints

- No environment reads, topology checks, timing probes, eligibility checks, or
  fallback accounting enter the round loop.
- The policy stores only one decision-bearing streak integer and the current
  depth.
- No engagement counter or width-trace telemetry is added. Existing
  `acceptance_lengths` demonstrate C8 engagement because values greater than
  four cannot come from C4.
- Construction-time depth resolution continues to enforce the drafter and
  engine ceilings.

## Correctness and acceptance gates

1. Pure policy tests cover fixed controls, promotion, streak reset, demotion,
   and narrowed-tail behavior.
2. The existing DFlash and runtime-worker suites pass.
3. The release worker is rebuilt from the changed source.
4. A guarded warmup-plus-three exact benchmark at requested D7 reports:
   - mean decode throughput greater than 150 tok/s;
   - the retained C4 digest in all three samples;
   - acceptance lengths containing full C8 rounds;
   - exactly 128 committed tokens per sample.
5. The production service is restored healthy and the GPU lock is released.

## Exclusions

This design does not change D1-D3 fixed controls, batching, prefill, sampling,
the DFlash arithmetic, cache rollback, verifier kernels, or the generic
`dflash-mlx` policy. Requested depths D4-D6 and D8-D11 remain fixed.
