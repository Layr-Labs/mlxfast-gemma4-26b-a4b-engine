# Agent guide — Gemma 4 26B A4B MLX engine

This file is the working contract for coding agents in this repository.
`CLAUDE.md` is a symbolic link to this file.

Read [README.md](README.md) first. It states what this repository is, how to
set it up, and what the structure is. This file adds the operational rules that
a person or an agent needs while iterating here.

The ranked track is `gemma4-26b-a4b-mlx-v1`.

## Goal

Make the Gemma 4 26B A4B text tower decode and prefill faster on Apple Silicon.
Do not change the observable model behavior beyond what the token-tolerance gate
allows.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/gemma4_26b_a4b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering log for this port | `docs/gemma4-port-notes.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchctl.manifest.json`) |

`benchmark.json` and `fixtures/gemma4_26b_a4b_track.json` carry pure
configuration. They hold values, paths, commands, and pins. They carry no prose.
Where this file disagrees with the fixture, the fixture wins.

## Lineage

This repository descends from `Layr-Labs/mlxfast-qwen-38-27b-mtp-engine`, which
descends from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank
different models under different rules. Only this track's rules apply here.

The engine's own sources are Gemma-named. A few fixtures and transform
validators still carry `Qwen` or `Laguna` in their names, and those are not
leftovers: they name real foreign checkpoints that this track's gates are proven
against. `Qwen35CheckpointValidation` and `fixtures/qwen3_6_27b_config.json`
build the Qwen-shaped config the Gemma trusted-config gate must REJECT;
`LagunaConfig` is the fixture substrate the generic transform tests run on.
Renaming either would make the name lie about what it holds.

## Current state

> **WARNING — official scoring is not armed.**
> `fixtures/gemma4_26b_a4b_track.json` sets `official_scoring_enabled` to
> `false`. The benchmarker seals `composite` as `None` on every batched run. Its
> gate has no per-stream aggregate source yet.
> `tools/gemma4-measure-and-score.sh` refuses rather than emit a score.

> **WARNING — the ranked runner is not registered yet.**
> `.github/workflows/benchmark.yml` is the real ranked pipeline: hosted
> surface check, then a self-hosted ranked job on
> `[self-hosted, macOS, gemma4-26b-a4b-mlx-v1]`. The label is the ruled one,
> not a placeholder, but no runner advertises it yet and the box is not staged,
> so a dispatch queues or refuses. It holds no credential by design: the hidden
> tapes are staged onto the box and pin-verified by
> `tools/ranked-box-preflight.sh`, which refuses rather than fetch or
> substitute.

The DFlash arm runs single-stream only. The batched DFlash path is refused by
name. The cohort DFlash work is deferred, not deleted. The repositories stay
private until launch.

## Notes for autonomous agents

These behaviors are expected. They are not bugs.

### The cool-down gate

The benchmarker waits for the GPU to cool before it starts a timed run. The
local modes pass `--cool-gate` to the benchmarker automatically. The gate reads
the GPU temperature through `macmon`.

**The gate lives in the benchmarker, and only `./benchmark.sh` arms it.**
`./benchmark.sh` passes `--cool-gate` to `benchctl`, and `benchctl` runs the gate
itself before each timed phase. Prefill and decode are gated separately.

> **WARNING — driving the Swift CLI directly runs UNGATED.**
> `mlxfast-swift --local-iterate` and `--local-submit` do not go through
> `./benchmark.sh`. The Swift harness dispatches its per-phase gate to an
> external helper named by `MLXFAST_LOCAL_COOL_GATE_HELPER`
> (`Sources/MLXFastHarness/Gemma4RuntimeLocalIterate.swift:520`, and the trusted
> copy at `:530`). Nothing sets that variable. Unset, the gate returns
> immediately and times a hot GPU without saying so.

Set the variable to the pinned benchmarker to arm that path.

```bash
MLXFAST_LOCAL_COOL_GATE_HELPER="$PWD/benchd-bin/benchctl" mlxfast-swift --local-iterate
```

Prefer `./benchmark.sh`. It is the measured path and it needs no such variable.

`./benchmark.sh --local-cool-gate-only` exits 0 without probing anything. The
bare probe is the benchmarker's own entry point.

```bash
benchd-bin/benchctl --local-cool-gate-only
```

> **WARNING — a run that pauses on a cool-down message is working, not hung.**
> Do not kill it. Do not treat the wait as a failure.

The gate aborts with a non-zero exit when the GPU stays hot and is not trending
down. That abort means something else is loading the GPU. Free the GPU and
retry. The abort does not mean your change is wrong.

`./setup.sh` installs `macmon` as a pinned, hash-verified release binary. The
gate warns and skips when `macmon` is absent. Skip the install with
`MLXFAST_SKIP_MACMON_INSTALL=1`.

> **WARNING — a skipped gate still produces a number.**
> Locally, no reader means no gate, and the run times whatever temperature the
> GPU happens to be at. Treat a timing taken without `macmon` as unmeasured.

The ranked box does the opposite. A missing or frozen reader is a hard refusal
there, before any measurement (`tools/ranked-box-preflight.sh`, sections 2b and
2c). A ranked run never proceeds without thermal control.

The gate mirrors the ranked runner's fixed 40 C thermal contract. That contract
is operator-owned. The benchmarker owns the exact thresholds; this repository
does not set them. The threshold is a fixed constant inside the benchmarker and
no fixture can move it.

### Fan control for a stalled cool-down

Use the fan helper when the local gate sits hot with no cooling progress. The
helper is manual only. No gate, script, or workflow invokes it. Nothing boosts
the fans on your behalf, and a stalled cool-down will not fix itself.

```bash
tools/fan-control.sh boost
```

This command forces every fan to 70% of its maximum speed.

```bash
tools/fan-control.sh normal
```

This command returns the fans to macOS's automatic curve.

```bash
tools/fan-control.sh status
```

This command prints `manual`, `auto`, or `none`.

Fan targets are SMC keys. macOS accepts SMC writes only from root. The helper
therefore runs its writes under `sudo`. `sudo` prompts for the password itself.
The helper never reads, stores, echoes, or logs the password. It drops the
cached credential with `sudo -k` right after the writes. The helper needs an
`smc` CLI. It refuses cleanly on a fanless Mac.

### Measurement discipline

Trust a timing number only from a cool, quiescent machine. Back-to-back runs
heat the GPU and throttle it. A 2-minute to 3-minute pause between local runs is
normal.

> **WARNING — a local score is directional.**
> The local test runs a single stream. The ranked run runs a batch-8 cohort. The
> forward pass takes structurally different kernel paths at cohort width. The
> MoE expert-sort path engages at batch 8 and not at width 1. Do not read a
> local score as a prediction of the ranked composite.

Record a same-machine baseline before you optimize. Sync to the latest tip
first. Do not compare a change against a stale branch or an old local run. Rerun
the baseline whenever the base commit changes.

### One model-holding run at a time

The target model is RAM-resident. Two model residencies at once can exhaust a
local machine's memory.

> **WARNING — run one model-holding command at a time.**
> Do not start a second local run while the first is alive. Do not run a
> model-holding `mlxfast-swift` command next to a local test. These commands are
> `correctness`, `correctness-trace`, `generate-golden`, `generate-gpqa-answers`,
> `dflash-benchmark`, `dflash-probe`, and `dflash-reference`.

No run lock enforces this. The discipline is yours to keep.

`swift test` never loads the real model. It is safe to run alongside.

Check for an orphaned worker when a run aborts. A worker whose parent process
identifier is 1 is usually an orphan. Verify it, then kill it.

### The startup memory profile

The runtime selects a low-memory profile automatically below 64 GiB of physical
memory. The profile caps the MLX allocator cache at 6 GiB, shortens command
buffers, and releases free warmup buffers before the worker serves requests.

The profile is pure memory management. It disables no code path and no
output-affecting feature. It announces itself on stderr. Force it either way
with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`.

A machine that is too small fails loudly with an out-of-memory error. It does
not diverge silently from ranked behavior.

### The non-M5 near-tie caveat

The checked-in public goldens are M5-generated greedy continuations. A near-tie
argmax can diverge on another Apple Silicon generation, even for correct code.

> **WARNING — a local gate failure on non-M5 hardware may not be your bug.**
> Check whether an unmodified `main` fails at the same token position on your
> machine. Do that before you treat a local failure as a regression.

Rerun with `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` when unmodified `main` fails the
same way. The local mode then still publishes its timing estimate.

The override is local-only. It hides nothing. The score keeps
`passed_correctness: false`, records the diverging tokens, and explains itself in
`metrics.error`.

> **WARNING — never use the override to paper over a real regression.**
> The mismatch is yours when unmodified `main` passes on your machine.

### One ranked machine, one queue

Ranked runs execute serially on a single runner. Duplicate dispatches queue
behind the run in flight. They do not cancel it. Expect delays. Do not dispatch
several ranked runs in parallel and expect concurrent results.

### Know the runnable surface

Only the `benchmark.json` `editablePaths` entries ship in a submission. A change
anywhere else does not upload, even when it helps locally. Official ranking
needs hidden organizer goldens. It is not runnable locally.

## Building

Two build trees exist. Keep them straight.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker
```

This command builds the scored worker into `.build-worker/release`.

The worker builds under its own scratch root so a participant compile can never
write into the trusted tree.

```bash
tools/stage-runtime-worker.sh
```

This command copies the finished worker and its `mlx.metallib` into
`.build/release`.

> **WARNING — a bare `swift build -c release` is not enough.**
> The scored binary is `.build-worker/release/mlxfast-runtime-worker`. Metal
> loads `mlx.metallib` from the directory of the running binary. The staging
> step puts the pair where the benchmarker resolves them. `./setup.sh` runs that
> step for you.

### Kernel edits

The vendored MLX package builds in JIT mode. Two forms matter.

Families with an `mlx-generated/*.cpp` twin compile at runtime from the C++
source strings inside those files. The twin is the runtime-effective source.
Edit the twin. Keep the readable `.metal` and `.h` pair in step.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`.

```bash
tools/build-mlx-metallib.sh
```

This command rebuilds `mlx.metallib` from the vendored `.metal` sources. Run it
after you edit an ahead-of-time source. `./setup.sh` runs it for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

Rebuild both binaries after any kernel edit. Then re-measure through the
benchmarker.

### The frozen dependency graph

> **WARNING — pass `--force-resolved-versions` on every direct `swift build`
> and `swift test`.**
> The dependency graph is frozen. A bare invocation can rewrite
> `Package.resolved` silently. `./setup.sh` then refuses to run. The flag makes
> SwiftPM fail closed instead.

Avoid bare `swift package resolve` and `swift package update`. They can rewrite
`Package.resolved` and there is no fail-closed flag for `resolve`. Restore the
file with `git checkout -- Package.resolved` when it shows as modified.

## Common commands

```bash
swift test --force-resolved-versions
```

This command runs the cheap contract tests.

```bash
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
```

This command also runs the MLX runtime tests. Use it when a change touches MLX
runtime behavior and the machine can run those tests.

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh && ./setup-gemma4-assistant.sh
```

This command provisions the target model and then stages the MTP head.
`./setup.sh` provisions the target only; head staging is always separate.

```bash
./setup-gemma4-dflash.sh
```

This command is optional. It stages the DFlash drafter into `./dflash-head/`.
Nothing else invokes it, so a failure here cannot break the MTP path.

## Swift tooling

Use the Swift toolchain that `./setup.sh` validates. `sourcekit-lsp` is the
standard Swift language server. Xcode or the Swift toolchain usually installs
it. Point your editor at the repository root. SourceKit-LSP then reads
`Package.swift` and resolves the SwiftPM targets.

Prefer SourceKit-LSP symbol navigation over string-only edits when you change
Swift model code.

## Where to spend effort

Good changes improve one or more of these.

- Kernel-level work inside the vendored Metal sources. Prioritize kernels the
  cohort prefill and the timed decode window reach.
- The batching engine. Admission, scheduling, round driving, and stream drain
  are competitive surface.
- Attention dispatch. The sliding-window and full-attention layer types use
  different masks and different head dimensions.
- The quantized matmul and the MoE gather-GEMM for the routed experts.
- KV-cache handling. The sliding-window cache only ever needs the last 1024
  positions.
- Weight loading and reuse. Prepare eagerly at init. Warm kernels before the
  first scored forward. Avoid redundant conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets the runtime skip work safely.

## Wrong strategies

Do not specialize for the public correctness prompt. Keep every change
prompt-independent and model-general. The hidden prompts differ from the public
fixtures.

Do not assume the ranked box has your local machine's memory budget. A strategy
tuned on one Apple Silicon generation can move differently on another.

Do not treat a local-only environment override as proof of a valid improvement.
Disabling the sandbox, skipping the transform without verifying `weights/`, and
pointing at a user-specific reference path are debugging aids. They do not
establish a rankable optimization.

Do not draw a conclusion from a tiny local run alone. A local run is a smoke
test. It is especially weak for sequence-length-dependent changes, because it
may not exercise the ranked sequence lengths or the ranked memory pressure.

Be conservative with numeric reassociation. A changed accumulation order can
flip a near-tie greedy argmax.

> **WARNING — the target quantization is frozen as shipped.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. `Sources/MLXFastTransform/` is editable, but that does not
> license a change of target format: a lossier target substitutes a degraded
> model instead of optimizing the accepted one. The MTP head and the DFlash
> drafter are a narrow exception, and the exception is RE-QUANTIZATION ONLY
> (David ruling 2026-08-26) — re-quantize either one within its 2 GiB
> declaration cap, but do not replace either one and do not upload head weights.
> `mtp-head/` and `dflash-head/` are not editable paths, and a head declaration
> accepts `"source": "pinned"` only. A head re-quantization happens ON LOAD, in
> memory: the two head loaders already call `quantize(model:)` while they bind
> the checkpoint, and both files that hold that call are editable
> (`Gemma4MTP.swift` for the MTP assistant, `DFlashDraftModel.swift` for the
> DFlash drafter). Nothing on disk changes
> (`docs/participant-contract.md` section 4.4).
> They only propose tokens; the pinned target decides every emitted token.
> The target's own quantization is verified on the LOADED model TWICE: once at
> worker startup, and again at the top of every window that gets measured,
> immediately before the measured work starts. The second check is there because
> the first alone verifies a model that later code can still change in place. An
> in-memory re-quantization of the target is refused by name, and the refusal
> stops the worker before any measurement.

> **WARNING — do not add a cache keyed on a request's input tokens whose only
> possible hit is the harness repeating one identical computation.**
> Bit-identical output does not make it legitimate. The benchmark measures
> single-pass inference. An optimization must save work that recurs in
> single-pass production inference. The harness never legitimately issues the
> same whole-prompt forward twice to one worker process. Any such repetition is
> a harness bug, never a contract to rely on. Input-independent caching stays
> fine. Within-request KV reuse stays fine. A change in this category fails the
> static review as bypass behavior.

Do not hardcode hidden prompts, hidden token identifiers, or answers. Do not use
timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

## Before submitting

Run at least these commands.

```bash
swift test --force-resolved-versions
```

This command runs the contract tests.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI.

```bash
./setup.sh && ./setup-gemma4-assistant.sh
```

This command provisions the model and the head.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchmarker. Run a local test afterwards.

Check the non-M5 near-tie caveat above when local correctness fails. Prefer a
more conservative optimization when performance improves but correctness turns
fragile.

Use the Yukon CLI for every account operation and every submission operation.
README.md holds the submission commands. Python is not part of the challenge
runtime.
