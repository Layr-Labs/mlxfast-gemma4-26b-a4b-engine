# mlxfast — Gemma 4 26B A4B MLX

This repository is the engine for the Gemma 4 26B A4B MLX speedup benchmark.
The track identifier is `gemma4-26b-a4b-mlx-v1`.

## What this repository is

This repository holds the engine. The engine is the Swift and Metal code that
runs the target model on Apple Silicon. You optimize the engine. You make the
model do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchctl`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run measures a cohort of 8 prompts. The 8 prompts run at the same
time as one batch. The ranked run measures your engine and a serial control
engine in the same session. The score compares the two.

> **NOTE — official scoring is ARMED.**
> `fixtures/gemma4_26b_a4b_track.json` sets `official_scoring_enabled` to
> `true`. Section [Scoring and gates](#scoring-and-gates) states what that
> means for a run today.

### Lineage

This repository descends from `Layr-Labs/mlxfast-qwen-38-27b-mtp-engine`, which
descends from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank
different models under different rules. Only this track's rules apply here.

A few fixtures and transform validators still carry `Qwen` or `Laguna` in their
names. Those names point at real foreign checkpoints on purpose. They are the
negative controls and the fixture substrate that this track's own gates are
tested against.

## Requirements

- An Apple Silicon Mac.
- Enough unified memory for the target model and its working set. The target
  checkpoint is 15,641,239,658 bytes across 11 pinned files, of which 3 are
  safetensors shards. Add the KV cache and the decode buffers on top of that.
- At least 40 GiB of free disk space before the download starts. Change this
  limit with `MLXFAST_REFERENCE_MIN_FREE_GIB`.
- macOS 14 or later. `Package.swift` sets that platform floor. CI builds on a
  `macos-26` runner, and the Metal toolchain policy in `setup.sh` treats
  macOS 26 as its own case.
- Swift 6, through Xcode or through the Xcode Command Line Tools.
  `Package.swift` declares `swift-tools-version: 6.3`.
- The Xcode Metal Toolchain. `./setup.sh` tries to download it. Some users need
  full Xcode. Install it, open it once, and accept the license with
  `sudo xcodebuild -license accept`.
- CMake. `./setup.sh` installs it through Homebrew when it is missing.
- Git.

You do not need Rust. The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` verifies that binary against `benchd.pin`.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone <repository-url> mlxfast-gemma4-26b-a4b-engine
```

This command copies the repository to your machine.

```bash
cd mlxfast-gemma4-26b-a4b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchmarker binary into `benchd-bin/` and
verifies its sha256 and its byte count against `benchd.pin`.

```bash
./setup.sh
```

This command checks your toolchain, builds the two Swift binaries, builds
`mlx.metallib`, and downloads and verifies the target model.

```bash
./setup-gemma4-assistant.sh
```

This command stages the MTP head into `./mtp-head/`. The engine reads the head
from that exact directory name.

```bash
./setup-gemma4-dflash.sh
```

This command is **optional**. It stages the DFlash drafter into
`./dflash-head/`. Skip it unless you work on the DFlash arm.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/gemma4-26b-a4b-qat4bit \
  --output weights
```

This command converts the downloaded checkpoint into the `weights/` tree that
the engine loads.

```bash
MLXFAST_ENGINE_BIN=.build/release/mlxfast-runtime-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

This command runs the local test against the checked-in public golden.

> **WARNING — set `MLXFAST_CORRECTNESS_GOLDEN_PATH` yourself.**
> The local test has no default golden. It stops with an error when the
> variable is empty.

> **WARNING — do not pass `--golden`, `--weights`, or `--score-path` to
> `./benchmark.sh`.**
> The script rejects these flags. Use the environment variables instead.
> `MLXFAST_WEIGHTS_PATH` defaults to `weights`.

### The two public goldens

| File | Purpose |
|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` | The drift tripwire. 1024 prompt tokens and 256 expected tokens. |
| `correctness_prompts/public_longcopy_gate_english_1024_1024.json` | The local-submit golden. 1024 prompt tokens and 1024 expected tokens. |
| `correctness_prompts/public_longcopy_gate_english_1024.txt` | The prompt text the two goldens tokenize. |

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `Sources/MLXFastModel/` | The participant runtime. Weight loading, attention, the MoE MLP, the KV caches, prefill, and decode. | Editable |
| `Sources/MLXFastTransform/` | The offline transform that writes `weights/`. | Editable |
| `Sources/MLXFastCLI/` | The trusted CLI, `mlxfast-swift`. | Trusted |
| `Sources/MLXFastCore/` | Shared constants and contracts. | Trusted |
| `Sources/MLXFastTrustedHarness/` | Correctness, provenance, and the head declaration reader. | Trusted |
| `Sources/MLXFastHarness/` | Worker-side runtime support. | Trusted |
| `Sources/MLXFastRuntimeWorkerCLI/` | The sandboxed worker, `mlxfast-runtime-worker`. | Trusted |
| `Vendor/mlx-swift/` | The pinned MLX fork. The listed Metal kernel sources are editable. | Mixed |
| `Vendor/mlx-swift-lm/` | The pinned model fork. The `Gemma4*` model files, the batching engine, and the listed `MLXLMCommon` files are editable. | Mixed |
| `fixtures/` | The track contract and the pinned checkpoint manifests. | Trusted |
| `tools/` | Setup, build, lint, and measurement scripts. | Trusted |
| `benchd.pin` | The pin for the benchmarker binary. It names a branch, a commit, a sha256, and a byte count. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head/` | The slot for your own MTP head. | Editable, optional |
| `dflash-head/` | The slot for your own DFlash drafter. | Editable, optional |
| `correctness_prompts/` | The public prompt and the two public goldens. | Trusted |
| `weights/` | The transformed weights the engine loads. | Generated |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

### Staging the heads

`./setup.sh` provisions the target model only. Head staging is a separate
command by design.

`./setup-gemma4-assistant.sh` stages the organizer-pinned MTP head into
`./mtp-head/`. It verifies the download against
`fixtures/gemma4_assistant.sha256`.

`./setup-gemma4-dflash.sh` stages the pinned DFlash drafter
(`z-lab/gemma-4-26B-A4B-it-DFlash`) into `./dflash-head/`. It verifies the
download against `fixtures/gemma4_dflash_drafter.sha256`. The DFlash arm is a
first-class scored mode: `allowed_modes` declares it, and a submission runs it
by driving it. Nothing else runs this script. A failure here therefore cannot
break the MTP path.

Both stagers delegate to `./setup.sh`'s downloader. All three artifacts share
one download, verification, resume, and stall path.

These two scripts are how the pinned heads reach a machine. A participant who
iterates locally needs the heads on disk, and these scripts provide them. The
ranked box stages them the same way. No other source is accepted: a head
declaration accepts `"source": "pinned"` only.

### The head directories

`mtp-head/` and `dflash-head/` each hold one `README.md` and nothing else.

> **NOTE — keep both `README.md` files in place.**
> Each one documents what the organizer stages in that directory, and it keeps
> the directory present in a fresh clone. Each file is inert for scoring: the
> head tree digest excludes a top-level `README.md`. Neither directory is an
> editable path, so neither travels in a submission.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 93 entries. The
rule behind the list is simple. Code that **proposes** tokens or computes the
forward pass is editable. Code that **verifies**, **measures**, or **ledgers**
stays trusted.

The editable surface has five groups.

1. The two head declarations. `mtp-head.manifest.json` and
   `dflash-head.manifest.json`. The declaration files only. The weights
   directories `mtp-head/` and `dflash-head/` are not editable.
2. The participant runtime. `Sources/MLXFastModel/` and
   `Sources/MLXFastTransform/`.
3. The six vendored `Gemma4*.swift` model files, the DFlash drafter model file
   `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`, plus
   12 `MLXLMCommon` runtime support files.
4. The batching engine. The whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2` directory,
   plus `MLXLMServer/Runtime/ToolStreamHandler.swift`. How the cohort admits,
   schedules, and drains streams is an ordinary optimization target.
5. The vendored MLX Metal kernels the forward pass dispatches. These are the
   quantized matmul, the MoE gather-GEMM, SDPA and steel attention, RoPE,
   RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and gather
   indexing.

### The two organizer-pinned heads

Both speculative heads are the organizer's pinned weights. You may re-quantize
either one. You may not replace either one, and you may not upload head weights
of your own. Custom head weights are not accepted on this track.

`mtp-head/` and `dflash-head/` are **not** editable paths, so a submission
carries no head weight file. A submission that carries one is refused before any
measurement.

The declaration files stay editable: `mtp-head.manifest.json` and
`dflash-head.manifest.json`. Each accepts `"source": "pinned"` only.
`"source": "remote"` and `"source": "in_branch"` are refused by name. Each head
has a 2 GiB declaration cap (`max_bytes` = 2147483648); a declaration may lower
it and may not raise it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes. Each
head loader already calls `quantize(model:)` while it binds the checkpoint, and
both files that hold that call are editable:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4MTP.swift` for the MTP
assistant, and
`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift` for the
DFlash drafter. Change the geometry that call selects.
`docs/participant-contract.md` section 4.4 is the authority.

The size cap is the only gate on a declaration. A declared `sha256` is optional,
and the runner does not verify it against the head bytes. Nothing in this
repository binds the staged head bytes to the organizer's pinned digests at run
time; `docs/participant-contract.md` section 4.3 states that limit plainly.

An absent declaration selects the organizer-pinned head. That is the normal
case. A declaration that is present but broken is a refusal. The runner never
falls back silently.

A head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. The serial control leg always runs the organizer-pinned
head.

### Batch size and draft depth

> **NOTE — two levers are locked.**
> The batch size stays 8. The draft depth stays 1 (`fixed_depth = 1` for
> stateless Gemma). You may not tune either one. Make the forward passes
> faster instead.

The rectangular cap is `B * (1 + k) <= 8` on M3 and later.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 4404587 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent since 2026-08-26. The exemption existed to let head
weights ride in a submission outside the source budget. A submission carries no
head weights any more, so there is nothing to exempt. The two exempt caps stay
declared because both enforcers carry the same numbers as compiled-in fallbacks
and this manifest is what holds them to a reviewed value.

An organizer-pinned head staged on-box is not walked by this budget at all. The
head directories are not editable paths, so the walk never visits them.
`max_bytes` bounds what the runner **loads**, and stays at 2 GiB.

### The target quantization is frozen

The target model's quantization is frozen as shipped. Do not re-quantize a
target weight. Do not re-represent one. Do not change the numerical format of
one. This holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model instead of
optimizing the accepted one.

The two speculative decoders are a narrow exception, and the exception is
re-quantization only. You may re-quantize the MTP head. You may re-quantize the
DFlash drafter. You may not replace either one. Each stays within its own 2 GiB
cap. A decoder only proposes tokens, and the pinned target decides every emitted
token.

### What you must not change

- Everything in `Sources/` that `editablePaths` does not list.
- `Package.swift` and `Package.resolved`. The dependency graph is frozen.
- Everything in `Vendor/` that `editablePaths` does not list.
- `fixtures/`, `benchd.pin`, `benchmark.json`, the scripts, the tests, and the
  documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The local test and the ranked run are different by design. Read this section
before you tune.

The local test runs a **single stream**. It uses a public golden. It prints a
single-stream estimate.

### What each local mode checks

Both local modes run one fused checked-timing pass. The pass teacher-forces the
golden's expected tokens and times the wall clock. It judges correctness from
that same pass. A mismatch is reported as a teacher-forced token mismatch.

A correctness failure does not discard the timing. The benchmarker reruns the
timing phase in a mismatch-tolerant form. It then reports the correctness
failure together with real timing numbers.

| Mode | Decode steps | Expected tokens the golden must hold | Cool gate |
|---|---|---|---|
| `--local-iterate` | 128 | 129 | On, because `./benchmark.sh` always passes `--cool-gate` |
| `--local-submit` | 1023 | 1024 | On |

The gate is on because `./benchmark.sh` arms it. Driving the Swift CLI directly
skips it and times a hot GPU. Use `./benchmark.sh`. See AGENTS.md, "The
cool-down gate".

The two public goldens differ in length for this reason. Use the 256-token
golden for `--local-iterate`. Use the 1024-token golden for `--local-submit`.

> **NOTE — both local modes check correctness and speed.**
> Neither local mode is a speed-only signal. Both apply the teacher-forced
> check. Neither one runs the ranked cohort gates.

The ranked run runs **8 streams at the same time** as one batch-8 cohort.

Batch-8 is not simply more load. The forward pass takes structurally different
kernel paths at cohort width. The MoE expert-sort path engages at batch 8. It
does not engage at width 1. Performance can therefore differ from the local
test. Near-tie token behavior can also differ.

> **WARNING — a local score is directional, not predictive.**
> Treat a local score as a smoke signal for speed and correctness. Do not treat
> it as a prediction of the ranked composite. The ranked M5 cohort run is the
> authority.

Local testing stays single-stream. There is no local cohort mode.

## Scoring and gates

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the **per-stream sum**. Add each of the 8 concurrent streams'
own elapsed time together. Do this for prefill and for decode separately. Do it
on the baseline leg and on the candidate leg, over the same cohort.

> **NOTE — the aggregate is not the shared window.**
> The aggregate is not the single elapsed time of the concurrent cohort window.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 prompt tokens and 129 expected tokens |
| Streams per cohort | 8 |
| Prefill tokens per cohort | 8 x 1024 |

### The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 8 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 4 |
| `minPairsPerCohort` | 4 |
| `decodeSpeedupFloor` | 0.90 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The cohort width is fixed. A width other than 8 has no certified series tag.
The benchmarker refuses that width rather than run it.

The median over the 4 cohort ratios is the mean of the two central
values, so the fastest and slowest windows do not enter the score.

`kvBackend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

### Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10% budget**.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require your output to match the serial trajectory token
> for token. The block-shaped forward pass diverges from the serial forward
> pass at near-tie argmaxes. The gate prices that divergence against the 10%
> budget. Do not read the gate as lossless.

The checked-in public goldens are M5-generated. A near-tie argmax can diverge
on another Apple Silicon generation, even for correct code. Before you treat a
local failure as your own regression, check whether an unmodified `main` fails
at the same token position on your machine.

### Current status

> **NOTE — the track is ARMED; an official score is produced on the ranked box.**
> Three statements are true right now.

1. `fixtures/gemma4_26b_a4b_track.json` sets `official_scoring_enabled` to
   `true`. That flag is the single authority on the arm state, and the pinned
   benchmarker enforces it: it refuses to seal an official scoring artifact
   while the flag is `false` or absent, and seals one while it is `true`.
2. The score path produces `per_cohort[].composite`, from the benchmarker's own
   parent-clocked prefill and decode windows. A composite is absent only when
   the cohort accepted no pair or a window is degenerate, and every absent one
   carries a `composite_absent_reason`.
3. `tools/gemma4-measure-and-score.sh` emits a score on the ranked box. It
   still refuses **locally**, because the 8 hidden goldens it needs are
   box-only. It does not substitute the shared-window diagnostic for the ruled
   formula.

The timed prompt pool **is** armed. `timed_prompt_pool[]` carries 8 entries.
Each entry holds a real path, a real `sha256`, a real byte count, and a
measured `noop_decode_speedup`. `hidden_correctness_golden` also holds a real
`sha256` and byte count.

> **NOTE — armed does not mean visible.**
> The 8 prompts stay hidden. The organizer holds them on the ranked box. You
> get the pins, not the prompts.

The DFlash arm is a first-class scored mode. `allowed_modes` declares `serial`,
`mtp` and `dflash`, and a submission runs whichever its own code drives.

It runs single-stream only — the engine refuses the batched DFlash path by name
— so a DFlash submission is measured in the single-stream series and is not
scored by the B=8 cohort composite above. The two arms are separate values and
are never pooled. The cohort DFlash work is deferred, not deleted.

The repositories stay private until launch.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api <url>
```

This command authenticates you.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` @ `0e3cbab38ce568cf6e23543010d08d03b731910c` |
| Target manifest | `fixtures/reference_gemma4_26b_a4b_qat4bit.sha256` (11 records, 15,641,239,658 bytes) |
| MTP head | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` @ `bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c` |
| MTP head manifest | `fixtures/gemma4_assistant.sha256` (8 records) |
| DFlash drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` |
| DFlash drafter manifest | `fixtures/gemma4_dflash_drafter.sha256` (2 records) |
| Benchmarker | branch `gemma4-26b-a4b-mlx-v1`, commit `6dc978b7fd2e246859e965a9e9a470265698c693`, sha256 `e044e1f43f21a2efd6ae5ee7799c912c2723da600d6e0a3adca95e5629fde4f4`, 2528464 bytes (`benchd.pin` is the authority; `tools/fetch-benchd.sh` enforces the sha256) |
| Model fork revision | `ed55bee83beb0623152f4c2e70f0cf99ad379e35` |

Both model repositories are public. They download without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

### The target model

| Property | Value |
|---|---|
| Model type | `gemma4_text` |
| Hidden layers | 30 |
| Full-attention layers | 5, at indices 5, 11, 17, 23, and 29 |
| Attention pattern | A six-layer repeat. The other layers use a sliding window. |
| Sliding window | 1024 |
| Routed experts | 128 |
| Experts per token | 8 |
| Hidden size | 2816 |
| Head dimension | 256. The full-attention layers use 512. |
| Vocabulary | 262144 |
| Embeddings | Tied. There is no `lm_head` tensor. |
| Quantization | Affine, group size 64, 4 bits, mixed precision |
| Raw tensors | 1697 across 3 shards |
| Text tower tensors | 1339 |

## Building after an edit

Two build forms matter, because the vendored MLX package builds in JIT mode.

Kernel families with an `mlx-generated/*.cpp` twin compile at runtime from the
C++ source strings inside those files. For these families the twin is the
runtime-effective source. Edit the twin. Keep the readable `.metal` and `.h`
pair in step with it.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`. After you edit one of those sources, run
`tools/build-mlx-metallib.sh`. `./setup.sh` runs that script for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker
```

This command builds the scored worker into `.build-worker/release`.

```bash
tools/stage-runtime-worker.sh
```

This command copies the worker and its `mlx.metallib` into `.build/release`,
where the benchmarker resolves them.

> **WARNING — always pass `--force-resolved-versions`.**
> The dependency graph is frozen. A bare `swift build` or `swift test` can
> rewrite `Package.resolved`. `./setup.sh` then refuses to run. Restore the
> file with `git checkout -- Package.resolved`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks on `ubuntu-latest`. It runs
`swift build --build-tests` and `swift test` on a hosted `macos-26` runner. It
treats first-party warnings as errors.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` lists them, and CI fails when that list drifts.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, macOS, gemma4-26b-a4b-mlx-v1]`. Before it measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/gemma4_26b_a4b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering log for this port | `docs/gemma4-port-notes.md` |
| The measured window and the decode target | `docs/timed-decode-evaluation.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked M5 run is the authority on any score. The contract fixture
> `fixtures/gemma4_26b_a4b_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The pinned
checkpoints carry the `gemma` license tag. The terms are at
<https://ai.google.dev/gemma/terms>. This repository distributes no model
weights. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full
third-party attribution.
