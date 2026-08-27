# The task — Gemma 4 26B A4B MLX

Make the Gemma 4 26B A4B text tower run faster on Apple Silicon.

The ranked track is `gemma4-26b-a4b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the batching engine and the speculative-decode arm.

The target model is `mlx-community/gemma-4-26B-A4B-it-qat-4bit`. It is a sparse
MoE model. It has 30 layers, 128 routed experts, 8 experts per token, and tied
embeddings. Five layers use full attention. The other layers use a 1024-token
sliding window.

The MTP head proposes tokens. The target model decides every emitted token.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 93 entries in five
groups.

| Group | Paths |
|---|---|
| The head declarations | `mtp-head.manifest.json`, `dflash-head.manifest.json` (the declaration files only) |
| The participant runtime | `Sources/MLXFastModel/`, `Sources/MLXFastTransform/` |
| The vendored model | The six `Gemma4*.swift` files, `DFlashDraftModel.swift`, and 12 `MLXLMCommon` support files |
| The batching engine | `MLXLMCommon/ContinuousBatchingV2/` and `MLXLMServer/Runtime/ToolStreamHandler.swift` |
| The vendored kernels | The MLX Metal families the forward pass dispatches |

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted.

Both speculative heads are the organizer's pinned weights. You may re-quantize
either one. You may not replace either one, and you may not upload head weights
of your own.

`mtp-head/` and `dflash-head/` are not editable paths, so a submission carries
no head weight file. The declaration files `mtp-head.manifest.json` and
`dflash-head.manifest.json` stay editable, and each accepts `"source": "pinned"`
only. `"source": "remote"` and `"source": "in_branch"` are refused by name.

Each head has a 2 GiB declaration cap. The size cap is the only gate on a
declaration. A declared `sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

Each head loader already calls `quantize(model:)` while it binds the
checkpoint. That call is the seam, and both files that hold it are editable
paths: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4MTP.swift` for the
MTP assistant, and
`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift` for the
DFlash drafter. Change the geometry that call selects. Do not write into
`mtp-head/` or `dflash-head/`: the ranked worker runs under a sandbox that
denies file writes, and the benchmarker refuses a changed head tree.
`docs/participant-contract.md` section 4.4 is the authority.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The two
> speculative decoders are a narrow exception, and the exception is
> re-quantization only. You may re-quantize either one, within its 2 GiB
> declaration cap. You may not replace either one.

> **NOTE — batch size is locked. Draft depth is not.**
> The batch size stays 8. You may not tune it.
>
> The draft depth is a free lever on BOTH speculative arms, set from your own
> drafter code, which is editable. Neither arm is pinned at 1.
>
> MTP: select 1, 2 or 3; the non-editable envelope clamps at 3, and benchd
> measures at depth 2 when the invocation names no depth.
> DFlash: select `spec.dflash.depth` up to the drafter's ceiling (engine cap 15);
> an absent dflash depth means the full ceiling, not 1.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the Swift binaries and downloads the target model.

```bash
./setup-gemma4-assistant.sh
```

This command stages the MTP head into `./mtp-head/`.

```bash
./setup-gemma4-dflash.sh
```

This command is optional. It stages the DFlash drafter into `./dflash-head/`.
Skip it unless you work on the DFlash arm.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/gemma4-26b-a4b-qat4bit \
  --output weights
```

This command writes the `weights/` tree that the engine loads.

```bash
MLXFAST_ENGINE_BIN=.build/release/mlxfast-runtime-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

This command runs the local test against the checked-in public golden.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the per-stream sum. Add each of the 8 concurrent streams' own
elapsed time together. Do this for prefill and for decode separately, on both
legs.

The ranked run measures a batch-8 cohort over a 1024-token seed and a 128-step
decode window. It runs 2 pairs per cohort. The floor is 0.90. The ceiling is
5.0. The KV backend is pinned `contiguous`.

The benchmarker applies a per-stream token-tolerance gate with a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require token-for-token equality with the serial
> trajectory. The gate prices divergence against the 10% budget.

## The current state

> **NOTE — official scoring is ARMED.**
> `fixtures/gemma4_26b_a4b_track.json` sets `official_scoring_enabled` to
> `true`. The benchmarker seals `per_cohort[].composite` from its own
> parent-clocked prefill and decode windows, and
> `tools/gemma4-measure-and-score.sh` emits a score.
>
> A composite is absent only when the cohort accepted no pair or a window is
> degenerate, and every absent one carries a `composite_absent_reason`.

The DFlash arm is a first-class scored mode. `allowed_modes` declares
`serial`, `mtp` and `dflash`, and a submission runs whichever its own code
drives. Declare the arm in `dflash-head.manifest.json`, key `arm`, value
`"dflash"` or `"mtp"`. An absent key and an absent file both mean `"mtp"`.
Any other value is refused by name before any measurement.

It runs single-stream only — the engine refuses the batched DFlash path by
name — so a DFlash submission is measured in the single-stream series and is
not scored by the B=8 cohort composite. Its score is the even-n median of the
per-prompt raw ratio-of-means, serial-anchored, floor 0.90, ceiling 5.0. The
two arms are separate values and are never pooled. See
`docs/participant-contract.md` section 5.1.1.

The repositories stay private until launch.

## Local runs are directional

The local test runs a single stream. The ranked run runs 8 streams at once. The
forward pass takes structurally different kernel paths at cohort width. Treat a
local score as a smoke signal, not as a prediction. The ranked M5 run is the
authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/gemma4_26b_a4b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchctl.manifest.json`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
