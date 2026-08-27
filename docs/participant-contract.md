# Gemma 4 26B A4B — participant contract

This document states the terms that bind a submission to track
`gemma4-26b-a4b-mlx-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/gemma4_26b_a4b_track.json` is the track contract fixture. Both files
carry pure configuration: values, paths, commands, and pins. They carry no
prose. This document explains those files. It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/gemma4_26b_a4b_track.json` and `benchmark.json`.
3. This document.
4. `README.md` and `TASK.md`.

If either configuration file disagrees with this document on a plain value, the
configuration file wins. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchctl` binary resolved from the bench
repository's release channel (the track branch's `dist/`). The channel publishes
`benchctl.manifest.json` (`{branch, source_commit, sha256, bytes}`) beside the
binary; `./tools/fetch-benchd.sh` verifies the binary against that manifest,
installs both into `benchd-bin/`, and logs the resolved identity. The harness is
trusted-side: a submission cannot change what measures it. This repository
carries no submodule and no sha pin.

## 2. What the track measures

The track measures Gemma 4 26B A4B MLX text-tower inference speed.

You optimize the MLX runner, the offline transform, the batching engine, and
the vendored MLX Metal kernel families that the forward pass dispatches. You
also optimize the speculative-decode arm.

The target model is `mlx-community/gemma-4-26B-A4B-it-qat-4bit` at revision
`0e3cbab38ce568cf6e23543010d08d03b731910c`. It is a sparse MoE model with 30
hidden layers, 128 routed experts, 8 experts per token, and tied embeddings.
Five layers use full attention, at indices 5, 11, 17, 23, and 29 of a six-layer
repeat. The other layers use a 1024-token sliding window. Quantization is
affine, group size 64, 4 bits, with mixed precision.

The speculative-decode arm is a fixed-depth-1 stateless assistant. The
assistant is a 4-layer Q-only drafter. It borrows the target's embeddings,
hidden state, and frozen KV.

`kv_backend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority. It lists 93 entries.

The rule behind the list: anything that only **proposes** tokens or computes
the forward pass is editable. Anything that **verifies**, **measures**, or
**ledgers** stays trusted.

The editable surface has five groups.

1. **The two head declarations.** `mtp-head.manifest.json` and
   `spec-decoder-head.manifest.json`. The declaration files only. The weights
   directories `mtp-head/` and `dflash-head/` are **not** editable. See
   section 4.
2. **The participant runtime.** `Sources/MLXFastModel/` and
   `Sources/MLXFastTransform/`.
3. **The vendored model files.** The six `Gemma4*.swift` files
   (`Gemma4.swift`, `Gemma4Text.swift`, `Gemma4MTP.swift`,
   `Gemma4MTPTarget.swift`, `Gemma4MTPConfigurationValidation.swift`,
   `Gemma4CBv2MTPDrafter.swift`), the DFlash drafter model file
   (`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`),
   plus 12 `MLXLMCommon` runtime-support files. Each of these files holds a
   model that PROPOSES tokens or computes the forward pass. The head WEIGHTS
   directories are not here. See section 4.4.
4. **The batching engine.** The whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2` directory,
   plus `Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`.
   How the cohort admits, schedules, drives rounds, and drains streams is an
   ordinary optimization target.
5. **The vendored kernels.** The 68 `Vendor/mlx-swift` files the forward pass
   dispatches: quantized matmul, the MoE gather-GEMM, SDPA and steel attention,
   RoPE, RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and
   gather indexing.

### 3.1 Optional paths

`optionalEditablePaths` lists `mtp-head.manifest.json` and
`spec-decoder-head.manifest.json`.

`spec-decoder-head.manifest.json` carries one more key than the MTP declaration
does: `arm`, which selects the speculative arm the ranked run measures. See
section 5.1.1.

A submission archive has REPLACE semantics over `editablePaths`. An absent head
declaration means the organizer-pinned head, and an absent `arm` means the MTP
arm. The overlay therefore skips a missing optional path instead of failing
closed.
`.github/scripts/overlay-editable-paths.sh` reads this list from the trusted
contract, never from the submission.

### 3.2 The byte budget

`editableSurfaceByteBudget` caps the editable surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 9647467 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is **absent** since 2026-08-26. The exemption existed for one
reason: to let head weights ride in a submission outside the source budget. A
submission carries no head weights any more, so there is nothing to exempt.

The two exempt caps stay declared. They cannot bind while `exemptPaths` is
absent. They stay because both enforcers carry the same two numbers as
compiled-in fallbacks, and this manifest is what holds those constants to a
reviewed value. `tools/lint-benchmark-manifest.py` check 3b enforces that
equality.

An organizer-pinned head that a setup script stages on-box is **not** walked by
this budget at all. Those bytes are gitignored and `mtp-head/` and
`dflash-head/` are not editable paths, so the walk never visits them. They are
bounded instead by the 2 GiB declaration cap in section 4.

### 3.3 What you may not edit

You may not edit anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing and telemetry code. `fixtures/` is outside
the editable surface. The scoring step reads the contract from the trusted
checkout for that reason.

### 3.4 The target quantization is frozen

The target model's quantization is frozen as shipped.

A submission must not re-quantize any target weight. It must not re-represent a
target weight. It must not change the numerical format of a target weight. This
holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model. It does not
optimize the accepted one.

The two speculative decoders are a narrow exception, and the exception is
re-quantization only.

You may re-quantize the MTP head. You may re-quantize the DFlash drafter. You
may **not** replace either one. You may **not** upload head weights of your own.
Custom head weights are not accepted on this track.

This is the 2026-08-26 ruling. It replaces the earlier bring-your-own-head
design, under which a participant could declare and ship a head of their own
choosing. That design is retired.

Both heads are the organizer's pinned weights:

- The MTP head is `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` at
  revision `bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c`.
- The DFlash drafter is `z-lab/gemma-4-26B-A4B-it-DFlash` at revision
  `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`.

Both pins live in `fixtures/`, which is outside the editable surface.
`fixtures/gemma4_26b_a4b_track.json` names the repository and revision.
`fixtures/gemma4_assistant.sha256` and `fixtures/gemma4_dflash_drafter.sha256`
carry the per-file digests that `setup-gemma4-assistant.sh` and
`setup-gemma4-dflash.sh` verify every downloaded byte against.

Three things enforce this, and section 4 states each one:

1. `mtp-head/` and `dflash-head/` are not editable paths. A submission that
   carries a file under either one is refused.
2. A head declaration accepts `"source": "pinned"` only. `"remote"` and
   `"in_branch"` are refused by name.
3. A re-quantization happens on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission. Section 4.4 states the mechanism.

Both loaders read the head's own `config.json`. A `quantization` block there
selects which modules load quantized and at what geometry, in the shape an MLX
conversion writes. The accepted parameters are `group_size` (positive, at most
65536), `bits` between 2 and 8, and optional per-layer overrides (at most 8192
entries). A value outside those bounds is refused by name.

The two loaders differ on a declare-versus-carry mismatch, and the difference
is real:

- The **DFlash** loader fails closed in both directions. A drafter that declares
  a quantization its tensors do not carry is refused. A drafter that carries
  packed tensors it does not declare is also refused. Neither is ever loaded at
  full precision instead.
- The **MTP** head loader does neither check. An absent declaration skips
  quantization, and packed weights then fail later inside the weight bind with a
  shape error. A declaration with no packed tensor quantizes nothing, silently.

An earlier revision of this document claimed both refusals for both loaders.
That was wrong for the MTP head, and it is corrected here rather than promised.

The reason for the whole exception is the propose-and-decide split. A decoder
only proposes tokens. The pinned target model decides every emitted token.

## 4. The two organizer-pinned heads

The track carries two speculative heads. Both are the organizer's weights. Both
use the same declaration mechanism.

| Head | Declaration | Staged at | Organizer pin |
|---|---|---|---|
| The MTP head | `mtp-head.manifest.json` | `mtp-head/` | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit@bb94eae1` |
| The DFlash drafter | `spec-decoder-head.manifest.json` | `dflash-head/` | `z-lab/gemma-4-26B-A4B-it-DFlash@77d42027` |

The declaration file is editable. The staged directory is not.

### 4.1 What you may declare

`"source": "pinned"` is the only accepted source. A declaration may also state
`max_bytes` (it may lower the 2 GiB track cap and may not raise it), a `bytes`
count for the artifact you expect the box to stage, and an optional `sha256`.

`"source": "remote"` is refused by name. `"source": "in_branch"` is refused by
name. Both were accepted before the 2026-08-26 ruling and both meant "load
weights the participant chose". The refusal names the retired source and names
`pinned` as what replaced it.

An absent declaration selects the organizer-pinned head. A declaration that is
present but broken is a refusal. The runner never falls back silently.

### 4.2 What you may not do

You may not put a weight file under `mtp-head/` or `dflash-head/`. Neither
directory is an editable path, so a submission that carries one is refused
before any measurement. `.github/scripts/enforce-modifiable-surface.sh` names
the file and refuses. `.github/scripts/overlay-editable-paths.sh` never copies
it. The benchmarker's own write-divergence gate refuses any content that
differs from the trusted baseline outside the editable surface.

You may not edit the organizer's staged head bytes. That is the same refusal:
the path is not editable, so any change to it is outside the surface.

### 4.3 What the size cap does and does not do

The 2 GiB declaration cap (`max_bytes` = 2147483648) bounds what the runner
loads. A separate 2 GiB cap bounds the bytes actually staged at `dflash-head/`.

The size cap is the only gate on the declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes. It treats a
wrong digest and an absent digest alike. That is stated here plainly because it
is a real limit, not a detail: **nothing in this repository binds the staged
head bytes to the organizer's pinned digests at run time.** The digests in
`fixtures/` are verified by the setup scripts when a person runs them. The
ranked pipeline does not run those scripts, and the harness computes a head
tree digest that it reports and never compares.

What does bind at run time is relative, not absolute: the benchmarker compares
the candidate workspace against the trusted baseline workspace and refuses any
divergence outside the editable surface. A correctly staged baseline is
therefore load-bearing for the whole property.

The ranked job stages both workspaces itself. `tools/stage-ranked-heads.sh`
runs after `./setup.sh` and before the measurement. It puts the same
organizer-pinned bytes in the candidate workspace and in the baseline
workspace. It then compares the two head trees file by file and refuses on any
difference.

The same step names the MTP head directory for each leg. The benchmarker reads
`QMTP_HEAD_DIR` for the serial control leg and `QMTP_CANDIDATE_HEAD_DIR` for
the candidate leg, and it refuses a measure run when the first one is unset.
Each leg therefore loads the head out of its own workspace. The two trees hold
the same bytes, because the step above proved it.

### 4.4 How a re-quantization reaches the box

A re-quantization happens ON LOAD, in memory. Nothing on disk changes.

You do not make a re-quantized checkpoint. Your code quantizes the head's
parameters in memory, in the same pass that binds them. The staged bytes are
only read.

#### What to edit

Each head loader already calls `quantize(model:)` while it binds the
checkpoint. That call is the seam. Both files that hold it are editable paths.

| Head | File | The call |
|---|---|---|
| MTP assistant | `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4MTP.swift` | `Gemma4AssistantDraftModel.load(from:)` |
| DFlash drafter | `Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift` | `DFlashDraftModel.load(from:bindTo:)`, through `applyDeclaredQuantization` |

Change the geometry that call selects. The default reads the head's own
`config.json`. Your code may select a different geometry instead.

Section 3.4 states the bounds the loaders accept: `group_size` positive and at
most 65536, `bits` between 2 and 8, and at most 8192 per-layer overrides. A
value outside those bounds is refused by name.

The DFlash loader also refuses a mismatch between what the checkpoint declares
and what its tensors carry, in both directions. The MTP loader does not make
that check. Section 3.4 states the difference.

#### Why nothing is written

Two properties follow from the in-memory rule, and both are why this mechanism
is the safe one.

1. The benchmarker compares the candidate workspace against the trusted
   baseline workspace and refuses any change outside the editable surface. That
   comparison reads the disk. A re-quantization on load is not a disk
   operation, so there is nothing for the gate to see.
2. The ranked worker runs under a sandbox profile that denies file writes. Code
   that tried to rewrite a staged head would fail there.

Do not write into `./mtp-head/` or `./dflash-head/`. Do not rewrite a head from
`setup.sh` or from the `mlxfast-swift transform` command. Each of those runs
before the workspace comparison, and the benchmarker refuses the change.

`Tests/MLXFastTests/HeadRequantOnLoadTests.swift` and
`Tests/MLXFastTests/DFlashRequantOnLoadTests.swift` hold this contract for the
two heads. Each one quantizes a staged head on load with the staged directory
made read-only, and then asserts that the head tree digest did not move.

#### What changes in the record

The worker reports the tree digest of the head it loaded. That digest is the
digest of the ORGANIZER's staged bytes, before and after a re-quantization,
because the bytes do not change. The geometry you selected is not visible in
that digest.

#### What this does not permit

The exception is for the two heads only. The target model's quantization stays
frozen, as section 3.4 states.

The target is verified TWICE, and both checks read the loaded model, not the
declaration:

1. At worker startup, immediately after the target is loaded.
2. Again at the top of each window that gets measured, immediately before the
   measured work starts.

The second check exists because the first one alone verifies a model that code
can still change afterwards. Both refuse by name, and a refusal stops the worker
before any measurement.

The head tree digest excludes a top-level `README.md`. Keep the checked-in
`README.md` in each head directory. It documents what the organizer stages
there.

A head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. The serial control leg always runs the organizer-pinned
head.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the **per-stream sum**. Add each of the 8 concurrent cohort
streams' own elapsed time together. Do this for prefill and for decode
separately. Do it on the baseline leg and on the candidate leg, over the same
B=8 cohort.

`aggregate` is **not** the shared concurrent cohort window's single elapsed
time.

`scoring.mode` is `batched-cohort-paired-decode-only`. It names the measurement
methodology, not the formula.

The benchmarker certifies the exponent pair bit-exact against its own
`ScoredExponents::certify` when it selects the batched cohort regime.

### 5.1.1 The DFlash arm is scored in the single-stream series

DFlash became a first-class scored mode on this track on 2026-08-26. The track
fixture declares it: `allowed_modes` is `["serial", "mtp", "dflash"]`, and a
submission runs whichever of those modes its own code drives.

**Declare the arm in `spec-decoder-head.manifest.json`, key `arm`.**

| `arm` | What runs |
|---|---|
| absent, or the file is absent | The MTP arm |
| `"mtp"` | The MTP arm |
| `"dflash"` | The DFlash arm |

Any other value is refused by name before any measurement. So is a value that
is not a string, and so is an `arm` key in `mtp-head.manifest.json`, which is
the wrong file. A broken declaration never falls back to the other arm.

`tools/gemma4-measure-and-score.sh` reads the key and passes the matching
candidate spec to the benchmarker. That script is trusted side and is not an
editable path, so a submission states an arm and never states a spec. Declaring
`"mtp"`, or declaring nothing, produces the exact invocation this track ran
before the key existed.

DFlash is **single-stream only**. The cohort driver refuses it by name, so a
DFlash submission cannot run the B=8 cohort and is therefore **not** scored by
the composite above. It is measured in the single-stream v1.1 free-run series,
and its score is that series' aggregation over the same pinned 8-prompt pool.

#### The DFlash series, precisely

| Quantity | Value |
|---|---|
| Series tag (`results.timed_mode`) | `free_run_v1_1` |
| Published value | `aggregate.raw_decode_speedup_median` |
| Aggregation | Even-n median of the per-prompt raw ratio-of-means |
| Anchor | Serial = 1.0 |
| Floor | 0.90 |
| Ceiling | 5.0 |
| Pool | The same pinned 8 prompts |

The composite is a cohort quantity, and a single-stream run has no cohort:
`per_cohort` is absent from its results file entirely. The two series are
never pooled, averaged, or compared. `.github/scripts/emit-gemma4-score.sh`
selects the field by the sealed `results.timed_mode` and refuses a record whose
series and shape disagree, so a DFlash number cannot be emitted under a cohort
field name.

The benchmarker enforces this rather than trusting it. A DFlash candidate keeps
the single-stream regime even though this fixture pins `scored_batch_size: 8`,
the regime it actually ran is sealed in `results.timed_mode`, and the
benchmarker's series fence refuses to pool or compare a single-stream file with
a batched-cohort one. The two arms are therefore separate values, never two
numbers averaged into one.

Both arms are measured through the SAME verbs, on the SAME parent clock: the
benchmarker brackets `free_decode_begin` for prefill and `free_decode_run` for
decode on every route, and the worker opens one width-1 session for serial, MTP
and DFlash alike. There is no DFlash-specific timing path.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 `prompt_tokens` and 129 `expected_tokens` |
| Streams per cohort | 8 |
| Prefill tokens per cohort | 8 x 1024 |

`MLXFastConstants.correctnessPromptTokens`, `benchmarkPrefillPromptTokens`, and
`benchmarkDecodeSeedTokens` all equal 1024. `benchmarkDecodeSteps` is 128.

Every timed leg runs on a cool, quiescent box. The ranked job waits for the
machine to go idle before the clock starts, and the benchmarker holds each timed
phase behind the fixed 40 C cool-down gate. The job refuses to measure at all
when the box has no GPU temperature reader, or when that reader returns a frozen
or implausible value. Only pairs accepted under that gate feed the composite.

### 5.3 The parameters

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

The whole pinned 8-prompt pool runs concurrently as one cohort. There is no
sweep and no per-run choice of width. A width other than 8 has no certified
series tag. The benchmarker refuses that width rather than run it.

The even-n median over the 4 cohort ratios is the mean of the two central
order statistics -- the fastest and the slowest of the four scored windows do
not enter the published number.

### 5.4 Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10%
budget**.

This track does not require token-for-token equality with the serial
trajectory. The block-shaped forward pass diverges from the serial forward pass
at near-tie argmaxes. The gate prices that divergence against the 10% budget.
The gate accepts similar output. It does not certify lossless output.

### 5.5 Arming

`fixtures/gemma4_26b_a4b_track.json` sets `official_scoring_enabled` to
`false`. That flag is the SINGLE authority on this track's arm state, and it is
load-bearing. The pinned benchmarker reads it from the `--contract` fixture. It
refuses to seal an official scoring artifact while the flag is `false`. It also
refuses while the flag is absent, because it treats an absent flag as unarmed
rather than armed. No submission can publish an official score until that flag
flips.

The pinned benchmarker DOES produce the composite. It computes
`per_cohort[].composite` from benchd's own parent-clocked prefill and decode
windows, summed over the accepted pairs, at the certified exponent pair. No
engine-reported value feeds it, and it does not depend on per-stream
instrumentation. Every cohort seals exactly one of `composite` and
`composite_absent_reason`. A composite is absent only when the cohort accepted
no pair, or when a window is degenerate, and the reason names which.

What is missing is therefore the arm state and the goldens, not the score path.
This track's `timed_prompt_pool[]` is box-only, and the 8 hidden goldens it
needs do not exist publicly. `tools/gemma4-measure-and-score.sh` therefore
still refuses with a non-zero exit rather than emit a score. It does not
substitute the
shared-window `raw_ratio_of_means` diagnostic for the ruled composite formula.
Refuse, not degrade, is the standing posture for this track. The `kv_backend`
check and the byte-budget check use it too.

The timed prompt pool is armed. `timed_prompt_pool[]` carries 8 entries. Each
entry holds an `r2_path`, a `sha256`, a byte count, and a measured
`noop_decode_speedup`. `hidden_correctness_golden` holds a `sha256` and a byte
count.

### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` and `..._1024_1024.json` | Checked into git | **Yes.** They are already in your clone. |
| `timed_prompt_pool[]`, 8 tapes | R2, pinned by `r2_path` | **No.** The GETs are credentialed. The objects are benchd tape documents, not `--golden` files, so `--golden` could not load them. |
| `hidden_correctness_golden` | R2, pinned by digest only | **No.** It is the token-fidelity oracle. |

`tools/fetch-goldens.sh` is the organizer-side, pin-verified fetcher for R2
objects. It reads the R2 base from the environment variable
`R2_BUCKET_ENDPOINT` only. That value is secret-tier and is absent from this
repository. The script verifies the byte count first, then the sha256, and
deletes the file on either mismatch. It refuses to fetch anything the contract
declares hidden, and that guard fails closed when it cannot read the contract.

> **NOTE — this repository pins no public golden for that tool to fetch.**
> A participant has nothing to fetch with it today. Whether to publish a public
> local-calibration golden is an organizer decision.

## 6. Running the benchmark

`benchmarkCommand` targets `benchctl measure-job` through
`tools/gemma4-measure-and-score.sh`. That script is trusted-side tooling. It is
not an editable path, so a submission cannot rewrite the measurement pipeline
from inside its own archive.

The wrapped invocation is:

```text
benchctl measure-job --contract fixtures/gemma4_26b_a4b_track.json \
  --candidate . --baseline $MLXFAST_GEMMA4_BASELINE_WORKSPACE \
  --golden <each *.json in $MLXFAST_GEMMA4_GOLDEN_DIR> \
  --min-pairs 4 --target-pairs 4 \
  --tag <run tag> --out <results dir>
```

The script then reads `results.json` `per_cohort[0].composite` and converts it
to the `{score, metrics}` shape the scorer requires.

`preSubmitCommand` runs `./tools/gemma4-measure-and-score.sh --preflight-only`.
That runs the measure-job prerequisite and quiesce checks and exits without
measuring.

The ranked pipeline is `.github/workflows/benchmark.yml`, which `benchmark.json`
`runner.workflow` names. It triggers on `workflow_dispatch` only. Its hosted
surface-check job gates its ranked job, which runs on the self-hosted labels
`[self-hosted, macOS, gemma4-26b-a4b-mlx-v1]` — the third label is the track id.
The ranked job holds no credential. The organizer stages the hidden timed-pool
tapes onto the box. Before `./setup.sh` runs, `tools/ranked-box-preflight.sh`
verifies each tape against this track's `{sha256, bytes}` pins. One ranked run
occupies
the box at a time. A second dispatch queues rather than cancelling the first.

`setupCommand` is `./tools/fetch-benchd.sh && ./setup.sh`. It does not chain the
assistant stager. The checked-in `mtp-head.manifest.json` declares
`source=remote`, so the runner fetches the head out of band.
`./setup-gemma4-assistant.sh` is the local convenience path.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` @ `0e3cbab38ce568cf6e23543010d08d03b731910c` |
| Target manifest | `fixtures/reference_gemma4_26b_a4b_qat4bit.sha256` |
| MTP head | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` @ `bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c` |
| MTP head manifest | `fixtures/gemma4_assistant.sha256` |
| DFlash drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` |
| DFlash drafter manifest | `fixtures/gemma4_dflash_drafter.sha256` |
| Model fork revision | `ed55bee83beb0623152f4c2e70f0cf99ad379e35` |

`fixtures/reference_gemma4_26b_a4b_qat4bit.sha256` pins 11 files totalling
15,641,239,658 bytes, of which 3 are safetensors shards. The checkpoint holds
1697 raw tensors; the text tower holds 1339 of them.
`Sources/MLXFastCore/Constants.swift` mirrors the same repository and revision
pin.

Both model repositories are public and download without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

Participants never supply the target weights. Substituting or re-deriving the
target is a failure.

The rectangular cap is `B * (1 + k) <= 8` on M3 and later. Batch size is locked
at 8.

You set the draft depth in code. Each arm reads one editable constant,
`submissionDraftDepth`. The two constants are written the same way.

- MTP: `CBv2MTPRoundDriver.submissionDraftDepth`
  (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/CBv2MTPRoundDriver.swift`).
- DFlash: `DFlashDraftModel.submissionDraftDepth`
  (`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`).

Both constants default to 1. Edit the constant for the arm that your submission
runs. A value at or below the ceiling of the arm runs as set. A larger value
clamps to the ceiling. The engine does not refuse it. benchd adds no depth of
its own. The engine resolves the depth and reports the depth that ran.

The two arms use the constant differently.

MTP adapts. The constant is a ceiling. The depth controller selects a depth from
0 to this ceiling each round. A higher constant makes the adaptive range wider.
It does not force speculation. The controller is editable
(`CBv2MTPDepthController.swift`). You can change the adaptive policy. The MTP
envelope limits the ceiling to 3. `maxDraftTokens` is 3.
`maxAutomaticRectangularTokens` is 32 at batch 8. The effective ceiling is
`min(3, submissionDraftDepth)`.

DFlash is fixed. Block diffusion drafts one whole block each round. The constant
sets a fixed block depth for the run. The drafter proposes this many speculative
tokens each round. The block that it emits is `depth + 1`. The extra column is a
bonus token. The ceiling is the recommended block size of the drafter minus one.
The engine limits this ceiling to 15. `experimentalDFlashMaxBlockSize` is 16. On
DFlash the acceptance varies, not the drafted depth. Each round the target
verifies the block. The target commits the longest correct prefix.

You select the arm in `spec-decoder-head.manifest.json`, key `arm`. The values
are `mtp` and `dflash`. An absent key means `mtp`. See section 5.1.1. The depth
is code. The arm is the manifest.

Each run seals the depth that ran. Read `effective_spec` for the arm. Read
`effective_mean_draft_len` for the draft length that the run realized. On MTP the
two values can differ because the depth adapts.

## 8. Prohibited techniques

A submission that uses any of these fails the static review.

- A cache or memo keyed on a request's input tokens whose only possible hit is
  the harness repeating one identical computation. Bit-identical output does
  not make it legitimate. The benchmark measures single-pass inference. An
  optimization must save work that recurs in single-pass production inference.
- Hardcoded hidden prompts, hidden token identifiers, or answers.
- Timing shortcuts, protocol injection, network access, and filesystem
  exfiltration.
- Any change outside `editablePaths`.

Input-independent caching stays legal. This covers weights, dequantized
tensors, and RoPE or mask tables keyed on shapes and offsets. Within-request KV
reuse stays legal.

Keep every change prompt-independent and model-general. The hidden prompts
differ from the public fixtures.

## 9. Submitting

Use the Yukon CLI for every account operation and every submission operation.
`README.md` holds the commands.

A submission archive packages only `editablePaths`. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first,
and no local run blocks the upload.

The ranked run on the official runner is the gate that ranks a submission.

## 10. License

The pinned checkpoints carry the `gemma` license tag. That is the Hugging Face
tag spelling, not an SPDX identifier. This repository records whatever the
checkpoint's own tag says. The terms are at
<https://ai.google.dev/gemma/terms>.

The tag is confirmed on the assistant repository. The target repository carries
no explicit license tag in its own card data at the pinned revision. Both are
`mlx-community` derivatives of the same Google Gemma 4 base.

This repository distributes no model weights.
