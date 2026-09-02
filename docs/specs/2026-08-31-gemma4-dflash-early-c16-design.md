# Gemma 4 DFlash early-recurrence C16 design

## Objective and measured evidence

The acceptance target is at least 200 decode tokens/s on the exact batch-one,
1,024-token Python prompt and 128-token decode workload. Prefill and decode are
reported separately. One discarded warmup precedes retained samples.

The unchanged serial control produced 114.381087 decode tokens/s. The exact
C8 DFlash route produced 158.235574 decode tokens/s, the same output digest,
and zero positional mismatches. Its 128-token trace spent 22 rounds discovering
an exact period-18 target recurrence and 12 C8 rounds after discovery.

The measured timing decomposition is 10.796684 ms per verifier round plus
2.142226 ms per verified column. It projects approximately 185.0 tokens/s for
early confirmation with C8, 172.0 tokens/s for strict detection with C16, and
207.7 tokens/s only when both changes are combined. C32 adds padded columns and
does not improve the projection.

## Construction-bound design

The D15 path installs exactly two physical verifier widths before measured
generation: C4 and C16. D1-D14 retain their existing routes. Ordinary C16
quantized projections and attention output use a fixed shared-weight serial
reduction entrypoint. Attention remains serialized decode. The C16 expert route
uses an exact stable rank-128 assignment ordering and the existing weighted
expert arithmetic. All model topology, storage, dtype, quantization, kernel,
and entrypoint checks happen at installation; the enabled path has no fallback,
environment read, eligibility check, or proof counter.

The D15 proposal phase has one runtime-varying decision: exact C4 or periodic
C16. The terminal tail remains physical C4 while `maxEmitCount` limits commits.
The model and persistent cache bank install immutable C2, C3, C4, C8, and C16
contexts atomically; the D15 dispatcher directly selects only its C4/C16 pair.

## Early target-confirmed recurrence

For candidate periods 2 through 32, choose the shortest nonconstant period for
which at least one complete period plus three confirmation tokens exists and
the final three committed target tokens equal the tokens one candidate period
earlier. This detects the measured period-18 trace after 23 committed tokens.
Two confirmation tokens are rejected because they falsely select period 3 on
the measured trace; three is the first safe observed window.

This is speculation, not a correctness assumption. Every proposed token is
verified by the target model. A nonterminal mismatch commits only the verified
prefix, replays the wide drafter cache from saved target state, and demotes to
exact C4. A terminal mismatch does not replay because generation is complete.
Stop conditions are checked before recurrence training or cache mutation.

## Geometry selection and gates

The direct C16 projection specialization is the primary candidate. A focused
guarded runtime microbenchmark must compare it with two prebound C8 chunks if
register pressure makes direct C16 slower; only the measured winner remains in
the installed route. There is no runtime fallback between them. The rank-128
expert route must be bit-exact against the incumbent stable ordering before it
can be installed.

Required gates are:

1. CPU policy, route, installation, cache, and source-contract tests pass.
2. Guarded runtime C16 kernels are bit-exact to their unchanged controls.
3. A guarded exact-workload scout uses one warmup and one retained sample,
   reports separate prefill/decode rates, and has zero positional mismatches.
4. If the scout reaches 200 decode tokens/s, the final run uses one warmup plus
   three retained samples; every retained sample and their mean must reach 200.
5. The exact production service is restored and healthy before the GPU lock is
   released after every guarded run.

If the full scout misses 200, profiling must identify the measured remaining
bottleneck before another geometry change. The target is not redefined around
a partial speedup.
