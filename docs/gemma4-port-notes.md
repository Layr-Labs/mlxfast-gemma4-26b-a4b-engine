# Gemma 4 26B A4B port notes — DRAFT

Track `gemma4-26b-a4b-mlx-v1`. Working notes for the port of this engine from
the Qwen 3.8 27B native-MTP track to the Gemma 4 26B A4B MoE target.

**Status: DRAFT, and deliberately incomplete.** This document records what was
established laptop-side, what was measured off the pinned checkpoints' own
metadata, and what is blocked. It is not a completed migration record. Every
claim below is either cited to a file and line, derived from a named pinned
revision's own `config.json` / `model.safetensors.index.json`, or explicitly
marked as unverified.

Citations use `path@sha:line`. Where a sha is not meaningful (a pinned Hugging
Face revision, or a file in this branch), the pin or path is named instead.

---

## 1. Model pins

| Role | Repository | Revision |
|---|---|---|
| Target checkpoint | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` | `0e3cbab38ce568cf6e23543010d08d03b731910c` |
| Spec-decode arm 1 (assistant head) | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` | `bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c` |
| Spec-decode arm 2 (DFlash drafter) | `z-lab/gemma-4-26B-A4B-it-DFlash` | not pinned here; BF16 MLX conversion is a separate work item |

The target is the checkpoint darkbloom (`Layr-Labs/d-inference`) serves.

Recorded in code at `Sources/MLXFastCore/Constants.swift`
(`referenceModelRepository` / `referenceModelRevision` /
`referenceModelName` / `defaultReferencePath` / `defaultReferenceCachePath`),
which is the same set the Qwen pins occupied.

### 1.1 Target geometry, read off the pinned revision's own `config.json`

Top-level `model_type` is `gemma4`; the nested `text_config.model_type` is
`gemma4_text`. Text-tower values:

| Field | Value |
|---|---|
| `num_hidden_layers` | 30 |
| `hidden_size` | 2816 |
| `intermediate_size` (dense MLP) | 2112 |
| `vocab_size` | 262144 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 2 |
| `head_dim` / `global_head_dim` | 256 / 512 |
| `sliding_window` | 1024 |
| `enable_moe_block` | true |
| `num_experts` / `top_k_experts` | 128 / 8 |
| `moe_intermediate_size` | 704 |
| `attention_k_eq_v` | true |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |
| `max_position_embeddings` | 262144 |

`layer_types` has 30 entries; the full-attention layers are exactly indices
**5, 11, 17, 23, 29** — a six-layer repeat where the last layer of each group
is global. So **5 global layers, 25 sliding layers**. This is the same 25/1024
split the darkbloom exactness report uses to derive its prefix-donation floor.

**Config-authoritative confirmation (independent, 2026-08-22):** the seat
re-read `intermediate_size = 2112` and the `{5, 11, 17, 23, 29}` global-layer
schedule directly against the pinned `config.json @ 0e3cbab3`. Both agree with
the values in this section and with `MLXFastConstants`. These two are worth
the second reading because they are the two most load-bearing and most
easily-assumed numbers in the port: `intermediate_size` looks like a typo next
to `hidden_size` 2816 (the dense MLP is *narrower* than the residual stream,
which is unusual and correct), and the six-layer repeat is the value most
likely to be carried over as Qwen's four by reflex.

RoPE is per-layer-type: `full_attention` is `rope_type: "proportional"`,
`rope_theta: 1e6`, `partial_rotary_factor: 0.25`; `sliding_attention` is
`rope_type: "default"`, `rope_theta: 1e4` with no partial factor. There is no
single top-level RoPE block, unlike the Qwen config the old
`Qwen35Config.load` parsed.

### 1.2 Tensor inventory, read off the pinned `model.safetensors.index.json`

1697 tensors total across 3 shards, `total_size` 15,608,614,044 bytes.
By prefix: `language_model.` 1339, `vision_tower.` 355, `embed_vision.` 3.
The text-tower selection is therefore 1339 tensors and the transform must drop
358.

Two structural facts that any hardcoded inventory must encode:

- **No `lm_head.*` at all.** `tie_word_embeddings` is true, so the head is the
  embedding. Do not port the Qwen inventory's untied-head expectation.
- **`v_proj` exists on 25 layers, not 30.** The five full-attention layers
  (5, 11, 17, 23, 29) ship no `self_attn.v_proj.{weight,scales,biases}`,
  because `attention_k_eq_v` makes values reuse the keys on global layers.
  An inventory that assumes 30 uniform attention blocks is wrong by exactly
  5 layers x 3 quantization components = 15 tensors.

Per-layer text-tower pattern (all 30 layers unless noted): `input_layernorm`,
`post_attention_layernorm`, `pre_feedforward_layernorm`,
`pre_feedforward_layernorm_2`, `post_feedforward_layernorm`,
`post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`,
`layer_scalar`, `router.{proj.{weight,scales,biases}, scale, per_expert_scale}`,
`experts.switch_glu.{gate,up,down}_proj.{weight,scales,biases}`,
`mlp.{gate,up,down}_proj.{weight,scales,biases}`,
`self_attn.{q,k,o}_proj.{weight,scales,biases}`,
`self_attn.{q,k}_norm.weight`, and `self_attn.v_proj.*` on the 25 sliding
layers only. Plus `model.embed_tokens.{weight,scales,biases}` and
`model.norm.weight`.

The experts arrive **already split** into
`experts.switch_glu.{gate,up,down}_proj` on this conversion. The vendored
runtime also carries a fused `experts.gate_up_proj` unpacking path
(`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`, `sanitize`)
for checkpoints that ship the fused form; it will not fire on this pin.

### 1.3 TRAP — the quantization block is mixed-precision

This is the sharpest divergence from every model this engine has carried, and
it breaks the config-load contract that `NEW-MODEL-BRINGUP.md` section 3.2
describes.

`quantization` is **not** a three-key object. It is:

```
{ "group_size": 64, "bits": 4, "mode": "affine",
  "<tensor path>": { "group_size": 64, "bits": 8 },   x 120
  ... }
```

120 per-tensor overrides, all promoting to **8 bits**, in four families of 30
(one per layer): `mlp.gate_proj`, `mlp.down_proj`, `mlp.up_proj`, and
`router.proj`. `quantization_config` is byte-equal to `quantization`.

Consequences, each of which is a real code change and not a comment:

1. `Qwen35Config`'s quantization parse
   (`Sources/MLXFastModel/Qwen35Config.swift:416-454`, `qwenQuantization`)
   reads exactly `{group_size, bits, mode}` and would silently succeed here
   while discarding all 120 overrides. The Gemma config type must carry the
   override table, not a scalar triple.
2. The runbook's "key-set exactly `{group_size, bits, mode}`" check (section
   3.2 item 4) must be replaced by "the three scalars are present AND every
   remaining key is a tensor path mapping to a `{group_size, bits}` object".
3. The library-model quantize predicate
   (`Sources/MLXFastModel/Qwen35RuntimeWeights.swift:186-193`) hardcodes one
   `(groupSize, bits, mode)` for every path that has `.scales`. On this
   checkpoint that is **wrong for 120 tensors** — it would quantize the dense
   MLP and router projections at 4 bits when the shards were written at 8.
   The predicate must consult the override table by path.
4. `validateStructuralValues`'s divisibility checks must run against the
   effective per-path group size, not the default.

**This is the single most likely source of a silent numerical divergence in
this port.** It is not caught by anything that only checks tensor names and
shapes.

---

## 2. Vendored tree (`Vendor/mlx-swift-lm`) — model side

### 2.1 FINDING: the fork checkout was not on `main`

The brief and the seat's follow-up both name `Layr-Labs/mlx-swift-lm` **main @
`ed55bee`** as the model-side base. The sibling working checkout of that fork
(`../mlx-swift-lm`, alongside this repository in the workspace) is on branch
**`feat/qwen35-mtp` @ `e0c1d54`** ("Qwen 3.6 MTP: expose post_norm hidden + MTP
head hidden for depth>1 drafting"), not `main`.

Any diff taken against that working tree — including the initial patch
inventory produced for this port — is a diff against a Qwen-MTP feature
branch, not against `main @ ed55bee`, and its "engine-local patch" attribution
is not reliable. `main` does contain `ed55bee` as its tip; the correct base was
obtained with `git -C mlx-swift-lm archive ed55bee` into a scratch tree, and
every statement in section 2.2 below is against that.

The inventory in 2.2-2.4 below was subsequently re-derived against `ed55bee`
and against the vendored tree's true base, and is correct as stated.

### 2.1a The vendored tree's true base

`Package.swift` declares the pins in a comment:

```
// Exact vendored revisions:
// mlx-swift    df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc
// mlx-swift-lm bc1c0ee67d15798343be17c9f8f61f7c0d977149
```

`bc1c0ee` ("fix: PR #66 review follow-ups…", 2026-07-05) is a true ancestor of
`main` and was independently confirmed by scoring all 821 reachable commits on
exact `(path, blob)` matches against the vendored manifest — it scores 425 of
461 files, the highest of any commit. The vendored tree is **65 commits behind
`ed55bee`** (21 first-parent).

But it is not a clean `bc1c0ee` snapshot. It is `bc1c0ee` **grafted** with the
laguna-dflash lineage (`origin/laguna-dflash-mlx`) — `Libraries/MLXSpeculative/`,
`DFlashTarget.swift`, `DFlashVerifyLinear.swift`, `Models/Laguna.swift`,
`Sources/mlx-bench/`, `docs/laguna-dflash.md`,
`scripts/convert_laguna_dflash.py` — and with the **Gemma 4 MTP layer
deliberately deleted**: `Gemma4MTP.swift`, `Gemma4MTPTarget.swift`, 18 test
files, 4 resources (including
`Resources/gemma4-26B-A4B-assistant-config.json`), 5 benchmark reports and
`scripts/benchmark-mtp.sh`.

The 21 first-parent commits between `bc1c0ee` and `ed55bee` are the real
surface area, and three of them matter directly:

| Commit | What it does |
|---|---|
| `ffede00` | **deletes the v1 engine** (one-engine consolidation) |
| `201dbed` | **adds production Gemma 4 MTP** (`#74`) — the target of this port |
| `15c8857` | accelerates exact MTP verification on Apple silicon |
| `b07b5f7` / `edc6036` / `802398d` | PagedAttention migration v0.8.0, simplify, recoverable slab commit |
| `ed55bee` | Gemma 4 v0.8.2 — shared tower, layer-18 submission, coupled weighted-unsort+R1 |

`ffede00` is the one that makes this a migration rather than a copy. See 2.2.

### 2.2 FINDING: adopting `main @ ed55bee` wholesale is not a drop-in

`main @ ed55bee` has **no `Libraries/MLXSpeculative/`** and no
`Libraries/MLXLLM/DFlashTarget.swift` / `DFlashVerifyLinear.swift`. Those are
genuinely engine-local (the laguna-dflash graft).

More significantly, the vendored tree carries an entire **pre-CBv2 batching and
compiled-decode stack that upstream deleted at `ffede00`**. Note the direction
carefully: the vendored copy is **not** missing CBv2 — it already has 38 CBv2
files, because `bc1c0ee` is well past the CBv2 landing. Only 27 CBv2 files are
fork-only additions. The gap is the opposite way round — the vendored tree
still carries the v1 engine that `main` has since removed, and also keeps two
CBv2 files `main` has since dropped
(`ContinuousBatchingV2/Compiled/CompiledDecode*V2.swift`,
`SequenceKV/QuantizedSequenceKV.swift`):

```
Libraries/MLXLMCommon/BatchGenerator.swift
Libraries/MLXLMCommon/BatchKVCache.swift
Libraries/MLXLMCommon/CompilableBatchKVCache.swift
Libraries/MLXLMCommon/CompilableBatchRotatingKVCache.swift
Libraries/MLXLMCommon/CompiledDecode.swift
Libraries/MLXLMCommon/ContinuousBatching/            (v1 tree)
Libraries/MLXLMCommon/ContinuousBatchingV2/Compiled/
Libraries/MLXLMCommon/ContinuousBatchingV2/SequenceKV/QuantizedSequenceKV.swift
Libraries/MLXLMCommon/GenerationBatch.swift
Libraries/MLXLMCommon/PromptProcessingBatch.swift
Libraries/MLXLMCommon/QuantizedBatchKVCache.swift
Libraries/MLXLMCommon/RowSamplers.swift
Libraries/MLXLMServer/Runtime/…BatchedEngineServerEngine*.swift  (6 files)
Sources/mlx-bench/, Sources/BenchCBv2/BenchCBv2RealModel.swift
```

77 vendored paths have no counterpart on `main`, and the engine's own
`Package.swift`, `MLXFastModel` and `MLXFastRuntimeWorkerSupport` targets
depend on that v1 surface. Separately, **80 of the 384 common paths differ in
content**, including exactly the CBv2 seams the Gemma 4 MTP round driver binds
to: `EngineLoopV2.swift`, `CBv2Contracts.swift`, `SchedulerV2.swift`,
`LayerCacheV2.swift`, `PagedKVBackend.swift`, `PagedKVPool.swift`,
`SteppableAdapterV2.swift`.

Therefore "replace `Vendor/mlx-swift-lm` with fork main" is a **migration, not
a copy**: it deletes the substrate the engine's speculative runtime and the
`AGENTS.md`-documented editable DFlash surface compile against. Landing it
requires, in the same change, either

- **(a)** porting `MLXSpeculative` + `DFlashTarget` + `DFlashVerifyLinear` onto
  CBv2, or
- **(b)** carrying the removed v1 files forward alongside CBv2 as an explicit
  engine-local retention, or
- **(c)** deleting the DFlash arm from the engine harness as well —
  `Sources/MLXFast{,Trusted}Harness/QwenRuntimeDFlash*.swift` (6 files),
  `Sources/MLXFastModel/LagunaDFlash*.swift`, the DFlash tests, and the
  `dflash` spec-mode wiring in `RuntimeWorkerSpecConfig.swift`.

Option (c) is the smallest and is consistent with the track: `dflash` is
already never runnable in this worker
(`Sources/MLXFastHarness/QwenRuntimeWorker.swift:72-83` — runnable modes are
`serial` always plus `mtp` when a head is supplied; "dflash/dspark are never
runnable here and fail closed at resolution"). But it is a deletion of a
documented editable surface and needs an explicit ruling, not an implementer's
judgement call.

**This is the reason the vendored swap is not in this branch.** Sizing it
honestly was the deliverable; guessing at it was not.

### 2.2b EXECUTED 2026-08-22 — the adoption landed

`Vendor/mlx-swift-lm` is now fork `main @ ed55bee`, with a small, explicit
engine-local carry. `Vendor/mlx-swift` is **unchanged at `df1fdc5f`** — the
version-gap risk flagged in 2.4 did not materialise: fork main compiles against
the frozen kernel layer.

**Patch dispositions**, re-validated against the corrected base (`bc1c0ee`, not
`feat/qwen35-mtp`). The decisive measurement: upstream moved **zero lines** in
five of the six carried/considered files between `bc1c0ee` and `ed55bee`, so
those carries are pure additions onto an unchanged base — there is nothing
upstream could have subsumed, and no merge judgement was required.

| File | Upstream movement | Disposition |
|---|---|---|
| `MLXLMCommon/Load.swift` | 0 lines | **CARRIED** — `AdditionalWeightSource` / BYO-head merge ordering; the mechanism the 26B assistant head loads through |
| `MLXLMCommon/KVCache.swift` | 0 lines | **CARRIED** — causal-mask memo + rollback checkpoints / replay tape |
| `MLXLMCommon/AttentionUtils.swift` | 0 lines | **CARRIED** — wide-decode exactness chunk; same defect class as trap 3.1, and the MTP rectangular cap (OQ-5) sits on the same width wall |
| `MLXLLM/Models/Qwen35.swift` | 0 lines | **CARRIED** — the MTP hidden-state exposure API the Qwen harness's `Qwen36MTPTarget` conformance requires |
| `MLXLLM/Models/Qwen35MTP.swift` | 0 lines | **CARRIED** — same reason |
| `MLXLMCommon/BatchKVCache.swift` | file deleted upstream | **DROPPED** with v1; `zeroTailPerRow` goes with it |
| `MLXLLM/Models/Laguna.swift` | 592 lines | **UPSTREAM TAKEN** — engine-local perf patches dropped with the DFlash arm |
| `Package.swift` (vendored) | n/a | **ONE engine-local edit retained**: upstream's floating `branch: "main"` mlx-swift URL → `.package(path: "../mlx-swift")`, so the kernel layer stays frozen and out of `Package.resolved` |

Carrying `Qwen35.swift` forward is the load-bearing scoping decision. Without
it the adoption would have forced the entire Qwen speculative apparatus out in
the same change — `Qwen36MTPTarget`'s conformance is defined against that API —
merging the vendored adoption and the harness port into one unreviewable
commit. It costs nothing: upstream never touched the file.

### 2.2c What LEFT with the adoption

**Vendored (53 top-level entries).** The v1 batching/compiled-decode engine
upstream deleted at `ffede00` — `ContinuousBatching/` (v1 tree),
`BatchGenerator`, `BatchKVCache`, `CompilableBatchKVCache`,
`CompilableBatchRotatingKVCache`, `QuantizedBatchKVCache`, `CompiledDecode`,
`GenerationBatch`, `PromptProcessingBatch`, `RowSamplers`, the six
`MLXLMServer/Runtime/MLXBatchedEngineServerEngine*` files, `Sources/mlx-bench`,
`Sources/BenchCBv2/BenchCBv2RealModel.swift`, two CBv2 files upstream has since
dropped (`ContinuousBatchingV2/Compiled/`,
`SequenceKV/QuantizedSequenceKV.swift`), ~20 associated test files — plus the
DFlash runtime that depends on it: `Libraries/MLXSpeculative/`,
`MLXLLM/DFlashTarget.swift`, `MLXLLM/DFlashVerifyLinear.swift`,
`docs/laguna-dflash.md`, `scripts/convert_laguna_dflash.py`.

DFlash's coupling to v1 turned out to be **narrow — only `BatchKVCache`,
referenced in three files.** Retaining that one file would have kept the arm
alive. It was not retained: that is v1 substrate, and the adoption ruling
excluded carrying it forward. Recording the narrowness because it materially
lowers the cost estimate for the DFlash-onto-CBv2 lane — that lane is
re-homing three references, not rewriting against a foreign engine.

**Engine-side (5 files).** Only the model-side DFlash surface, i.e. exactly the
code that cannot compile without `MLXSpeculative`:

- `Sources/MLXFastModel/LagunaDFlashBlockSession.swift`
- `Sources/MLXFastModel/LagunaDFlashReference.swift`
- `Sources/MLXFastHarness/QwenRuntimeDFlashWorker.swift`
- `Sources/MLXFastTrustedHarness/QwenRuntimeDFlashWorker.swift`
- `Tests/MLXFastTests/RuntimeWorkerDFlashWireSurfaceTests.swift` (unit tests for
  pure functions that lived in the deleted worker; `.disabled` cannot help when
  the symbols are gone)

**What deliberately STAYED.** The MLX-free DFlash driver and protocol logic —
`QwenRuntimeDFlash.swift` and `QwenRuntimeDFlashDriver.swift` in **both**
harness trees (~2,500 lines each tree) — imports only `Foundation` and
`MLXFastCore`, so it compiles without the model layer and was left untouched.
The DFlash correctness contract, track fixture, and `MLXFastConstants`
DFlash block are likewise untouched. So the follow-up lane inherits the
protocol, driver, ledger and contract intact and needs to supply a
CBv2-backed worker.

The `dflash-runtime-worker` CLI verb is **retained and fails closed** with a
message naming the deferral and pointing at OQ-3, rather than being deleted.
A deferred arm should refuse audibly; deleting the verb would have made a
reserved name look like it never existed.

### 2.2d Finding: `DARKBLOOM_COMPILED_DECODE` was removed upstream

`RuntimeStartupMemoryPolicyTests.compiledDecodeFlagsStayReadableByModelSources`
exists to catch a vendored rename silently orphaning a documented opt-out flag.
It fired on this adoption, correctly: `MLX_COMPILED_DECODE` survives (now in
`MLXLMCommon/MLXHardwareInfo.swift`), but **`DARKBLOOM_COMPILED_DECODE` is read
nowhere in the vendored tree** — it lived in the v1 `CompiledDecode.swift`.

The test was **narrowed to the surviving flag rather than loosened**, with the
removal recorded at the assertion and the now-orphaned mention annotated in
place at `Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift`. Any operator
documentation still offering `DARKBLOOM_COMPILED_DECODE` as an opt-out is stale
and should be corrected — that is outside this repository.

### 2.3 Engine-local patch inventory — 20 files

Method: build the set of every `(path, blob)` pair across all 821 commits
reachable in the fork, then subtract the vendored manifest. **441 of the
vendored tree's 461 files are byte-identical to some commit in the fork's
history.** The remaining 20 contain content that exists nowhere in it, and are
the true engine-local patches. Disposition per class:

**PORT FORWARD — load-bearing for this track:**

| File | Patch |
|---|---|
| `MLXLMCommon/Load.swift` (+114) | `AdditionalWeightSource(directory:keyPrefix:)`, `strippingWeightKeyPrefix`, `loadArrays(directory:)`. Merges a separately-pinned head tree into `loadWeights` **before `sanitize` and before the quantize wiring**. The ordering is the whole point and its comment says why: `quantize(model:)` decides which submodules become quantized by asking whether `weights["<path>.scales"]` exists, so a later merge arrives as quantized tensors addressed to unquantized layers. **This is the mechanism the 238 MB assistant head must be attached with** (section 4.2). |
| `MLXLMCommon/KVCache.swift` (+116) | 32-entry locked memo for `createCausalMask` keyed on `(n, offset, windowSize)`, applied only when `lengths`/`leftPadding` are nil; plus `rollbackCheckpoints` and `prefixReplayTape` on the rollback cache type. |
| `MLXLMCommon/AttentionUtils.swift` (+39) | "WIDE-DECODE EXACTNESS CHUNK". For `B == 1`, causal, `6 <= qL <= 9`, splits the queries at row 5 into two SDPA calls so both halves stay on the fused sdpa-vector path (which serves `qL * gqa <= 32`). Documented as the measured source of the MTP width wall's top-2 value drift. **Generic**, and the Gemma MTP arm will hit the same wall — this is the same class of defect as trap 3.1. |
| `MLXLMCommon/BatchKVCache.swift` (+17) | `BatchRotatingKVCache.zeroTailPerRow(keepLengths:)`. |

**PORT FORWARD — DFlash arm (a coherent 5-file fix), conditional on OQ-3:**

`DFlashTarget.swift` (`makeDefaultDFlashCacheRollbackState` gains
`plannedWriteCount`, snapshots when `entry.offset + plannedWriteCount >=
maxSize`, and no longer throws `untrimmableCache` on a short trim),
`MLXSpeculative/DFlashBatchedTokenGenerator.swift`,
`MLXSpeculative/DFlashBenchmark.swift`,
`MLXSpeculative/DFlashGreedyRound.swift` (adds `DFlashRoundWorkBinding` and
`dflashFloatRowDigest`, the Criterion-E anti-elision work binding), and the new
`Tests/MLXLMTests/DFlashRollbackSeamTests.swift`.

The `DFlashTarget` comment is worth reading before any decision to drop it:
Laguna's sliding window is 512 and the ranked window is a 512-token seed plus
128 decode steps, so **every scored run crosses the seam**; it went unnoticed
only because bring-up used 26-68 token prompts. Gemma 4's sliding window is
1024 and the same arithmetic applies.

**DO NOT PORT — Qwen-specific:**

`Models/Qwen35.swift` (916L → 3412L, ~2600 net added: the
`callWithHiddenAndNormed` / `mtpHeadHiddenForward` /
`mtpHeadLastHiddenWithKVOnlyHistory` exposure API, fused-kernel argmax, compact
draft vocabulary, `Qwen35FusedMLP` and a large perf layer) and
`Models/Qwen35MTP.swift` (+40/−1). The Gemma equivalent is fork main's
`Gemma4MTP.swift` + the CBv2 MTP surface. These stay in the Qwen repo's history
as reference for later Gemma perf work.

**CASE-BY-CASE, per OQ-3:** `Models/Laguna.swift` (+~340: compiled softplus
gate, router tail, normalized expert combine, env-gated INT8 attention
requant), the `MLXSpeculative` tree, `Sources/mlx-bench/{BenchCommands,
DFlashBenchCommand,main}.swift`, `Tests/.../dflash-laguna-xs-2.1-config.json`.

**Wiring:** `LLMModelFactory.swift` (+1, registers `laguna` — redundant, `main`
already has it), `ContinuousBatching/BatchedMTP.swift` (+2/−2, comment only),
and `Package.swift` (mlx-swift dep → path; adds `MLXSpeculative` library and
`mlx-bench` executable; drops `DSV4Smoke`; removes the Gemma 4 MTP test
resources).

### 2.3a TRAP — two concrete hazards in `Gemma4Text.swift`

The vendored `Gemma4.swift` is 127L vs main's 131L and differs by **+2/−2,
comment only** (de-MTP-ized doc text). Take main's wholesale.

The vendored `Gemma4Text.swift` is 1483L against a 1484L base — **+31/−32**,
i.e. nearly pure-older, with exactly two substantive local changes. Both are
port hazards:

1. **Added:** a `public enum Gemma4 { public enum PositionOffset { case
   scalar(Int); case batch(MLXArray); case graphArray(MLXArray) } }` block at
   the top of the file. On fork main this type lives in
   `Gemma4MTP.swift:67-83`, not in `Gemma4Text.swift`; the engine copied it
   here precisely because it deleted `Gemma4MTP.swift`. The two definitions are
   identical apart from one comment word. **Porting `Gemma4MTP.swift` in will
   produce a duplicate declaration unless the engine's copy is deleted from
   `Gemma4Text.swift` first.**
2. **Removed:** `Gemma4TextModel.embedTokensForDrafter(_:)` and
   `_testCallInner(_:cache:captureHook:)`. The first is **required** by the
   `Gemma4MTPTarget` protocol (`Gemma4MTPTarget.swift`), which relies on
   `Gemma4TextModel` satisfying it directly. It must be restored; it lives at
   `Gemma4Text.swift:2000` on main. (`forwardForMTP` and
   `rollbackSpeculativeCache` are declared in an `extension Gemma4TextModel`
   inside `Gemma4MTP.swift:381+`, so those arrive with the file.)

**RESOLVED by the wholesale swap.** Both hazards were properties of a
*selective* file-by-file port; taking the whole tree removed them. Verified in
the adopted tree: exactly one `enum PositionOffset` definition
(`Models/Gemma4MTP.swift:75`), and `embedTokensForDrafter` is present at both
its protocol declaration (`Models/Gemma4MTPTarget.swift:27`) and its
implementation (`Models/Gemma4Text.swift:2000`).

Surviving MTP-relevant public API already in the vendored copy:
`Gemma4TextModelInner.callCapturingPreNorm(...)` (the pre-norm hidden-state
exposure seam), `Gemma4TextModel.configuration`, the `captureHook` parameter,
and `firstKvSharedLayerIdx` / `lastFullAttentionNonSharedIdx` /
`lastSlidingAttentionNonSharedIdx`.

### 2.3b Fork-only closure needed to compile the Gemma 4 MTP set

The six `Gemma4*.swift` files on main are `Gemma4.swift` (131L),
`Gemma4Text.swift` (2263L), `Gemma4MTP.swift` (2318L),
`Gemma4MTPTarget.swift` (51L), `Gemma4MTPConfigurationValidation.swift` (651L),
`Gemma4CBv2MTPDrafter.swift` (186L). Their transitive closure is 211 files; 16
are fork-only. Direct symbol dependencies:

- `MLXLMCommon/ContinuousBatchingV2/MTP/MTPContractsV2.swift` —
  `CBv2MTPForwardable`, `CBv2MTPDrafter`, `CBv2MTPSteppableModel`,
  `CBv2MTPConfig`, `CBv2MTPCaptureLayers`, `CBv2MTPPreparedCapture`,
  `CBv2MTPRowCapture`, `CBv2MTPMetrics`, `CBv2MTPCostInput`,
  `CBv2MTPVerificationMode`
- `MLXLMCommon/ContinuousBatchingV2/MTP/CBv2MTPRoundDriver.swift` —
  `CBv2MTPRoundDriver`, `CBv2MTPCarry`, `CBv2MTPHiddenIndex`,
  `CBv2MTPRoundInFlight`, `CBv2MTPSeedCostLedger`, `VerifyRow`
- `MLXLMCommon/ContinuousBatchingV2/PrefillOutputV2.swift` —
  `CBv2LanguageModelPrefillForwardable`, `CBv2PrefillSteppableModel`,
  `CBv2PackedPrefillSteppableModel`, `CBv2PrefillRequirement`
- `MLXLMCommon/ContinuousBatchingV2/LastQueryPrefillV2.swift` —
  `CBv2LastQueryPrefillLayerCache`

Indirect: `MTP/CBv2MTPCaptureFence.swift`, `MTP/CBv2MTPDepthController.swift`,
`CBv2DeadlineLeases.swift`, `CBv2SlidingWindowDonation.swift`,
`PrefixReusePlan.swift`, `TokenConstraintSamplerV2.swift`,
`ToolConstraintV2.swift`, `SequenceKV/FrozenReplayFullSequenceKV.swift`,
`Paged/{PagedAttentionResources,PagedKVSlabCommitment,PagedSeamContract}.swift`,
`MLXLMServer/Runtime/ToolStreamHandler.swift`.

Not in the symbol closure but part of the same feature — the engine loop drives
*into* the Gemma 4 code, so no edge points back — and needed in practice:
`MTP/EngineLoopV2+MTP{,Execution,Finalize,Measurement,Planning,
TargetVerification}.swift` (928L total).

Useful confirmation from the fork's own VLM factory: it already registers
`gemma4_26BA4B_it_4bit` → `mlx-community/gemma-4-26b-a4b-it-4bit`, with
`extraEOSTokens: ["<end_of_turn>"]`.

### 2.4 `Vendor/mlx-swift`

Untouched in this branch, and there is **no mechanical version conflict to
resolve**: the engine consumes both vendored packages as path dependencies, so
neither appears in `Package.resolved` at all, and the vendored
`mlx-swift-lm/Package.swift` already points at `.package(path: "../mlx-swift")`.

The substantive risk is different. Fork `main` declares
`.package(url: "…/Layr-Labs/mlx-swift.git", branch: "main")` — a **floating
branch**, no version or revision — and that branch has drifted alongside the 65
mlx-swift-lm commits, while the engine hard-freezes `df1fdc5f`. Fork-main
Gemma 4 / CBv2-MTP code may reference mlx-swift APIs that `df1fdc5f` does not
have. Check that against `Vendor/mlx-swift` directly **before** the port, since
relaxing the freeze is not free here: `setup.sh` refuses to build over a
modified `Package.swift`/`Package.resolved`, and every build passes
`--force-resolved-versions` specifically to fail closed on re-resolution.

**RESOLVED 2026-08-22: no bump needed.** Fork `main @ ed55bee` builds clean
against the frozen `df1fdc5f`, so `Vendor/mlx-swift` is untouched and the
authorisation to advance it was not exercised. The floating-branch risk was
neutralised instead of inherited: the vendored `Package.swift` keeps one
engine-local edit turning upstream's `branch: "main"` URL back into
`.package(path: "../mlx-swift")`. Both revisions are recorded in the engine's
own `Package.swift` comment in the existing format.

---

## 3. Correctness traps from the darkbloom contract

Source: `docs/architecture/gemma4-cbv2-mtp.md` in `Layr-Labs/d-inference`
(fetched read-only) and `docs/reports/2026-07-26-gemma-26b-adoption-exactness.md`.
These are darkbloom's findings on this model family and this checkpoint; they
are recorded here because this engine will hit the same physics.

### 3.1 Shape-dependent quantized-kernel divergence

The flagship trap. On M5 with the production QAT checkpoint, rectangular
verification first diverged **in the layer-0 quantized Q/K projection**:
`[B,1]` and `[B,L]` shapes select different quantized-matmul reduction paths in
MLX and are not bit-identical. Narrower shapes additionally exposed
shape-dependent SDPA. Corroborated upstream by MLX issues #3553
(quantized-matmul cost ramp at M=3..9) and #3573 (causal SDPA numerics change
across the query-length 8/9 dispatch boundary).

Darkbloom's fix has two halves: bound projection geometry to the tested kernel
regime, and use **decode-shaped rectangular attention** — projections and FFN
batched `[B,L]`, but every provisional query evaluated through canonical `L=1`
SDPA over its exact visible KV prefix, so the attention dispatch is
byte-identical to ordinary decode.

Relevance here: **any** widening of a forward's row count — the MTP verify
block, a batched free-run leg, a wider prefill chunk — is a candidate for
silent numerical drift on this checkpoint, independent of whether the model
code is "correct". Token-fidelity gates that assume shape-invariance will
mis-attribute the failure.

### 3.2 Step-global eligibility

A quantized MoE's numerics depend on the batch shape it is invoked with,
because a broader batch activates different expert routing and reduction. Two
distinct failures were found upstream: mixed terminal depths changing the
quantized-MoE target batch cadence, and mixed eligible/ineligible rows
splitting one decode batch into different shapes — which changes the output of
the **non-speculating** rows too.

The invariant: eligibility and draft depth are **step-global**. One uniform
depth for every speculating row in a verification cohort; if the cohort cannot
speculate together, the whole cohort goes to depth zero. Per-row partial
speculation and per-row variable depth are excluded.

Relevance here: this engine's worker is single-sequence today, so the cohort is
trivially uniform. It becomes binding the moment a batched free-run leg is
added, and it is a constraint on the decoder seam's API — a per-row depth
policy is the wrong shape even though it looks more general.

### 3.3 KV rollback storage classes — CORRECTED 2026-08-22

**An earlier revision of this document stated that paged sliding-window KV
"cannot take speculative writes at all", and derived from that a hard conflict
with the then-standing paged parity pin (25 of 30 layers being sliding). That
was STALE and is retracted** — and the pin it referred to has since been
superseded too: the track pins **contiguous** (section 5). It described upstream behaviour before 2026-07-25. The correct
picture, as shipped in the tree this branch adopts:

- **Contiguous full / quantized KV:** rewind the offset; a zero-reservation
  transaction is exact.
- **Contiguous windowed rings:** genuinely alias at `window` — `CBv2WindowedSequenceKV`
  holds exactly `window` slots — so a real **staged transaction** is required;
  the speculative K/V tile is held outside the ring until finalize.
- **Paged rings (any attention kind):** **rollback-safe.** The ring holds
  `ringPages * pageSize` slots, always at least one whole speculative span more
  than the window, so nothing a round overwrites is still readable. The paged
  transaction is **bookkeeping only** — a pure cursor move, no staging buffer.

The retraction is explicit in the adopted source. `mlx-swift-lm` commit
`5e36ed2` ("derive paged speculation from ring headroom, and make the round a
transaction", an ancestor of `ed55bee`) replaced the old `windowSize == nil`
gate, and
`Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/PagedSequenceKV.swift:253-273`
records why in the code itself: the old gate "transplanted the CONTIGUOUS
ring's problem onto the paged ring", and "That claim was false."

Eligibility is now **derived from ring headroom**, not attention kind:

```swift
public var supportsSpeculativeWrites: Bool {
    speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan
}
```

That comparison is what keeps ring sizing and the MTP draft bound coupled:
raise the span past what the ring reserves and rows go **ineligible** rather
than silently corrupting confirmed history.
`Paged/PagedSeamContract.swift:216-222` shows the sizing rule
(`attendableTokens(window:) + maxSpeculativeSpan`, floored by the prefill
chunk).

So MTP verification writes land **in place in the serving KV across all 30
layers**, and rows that cannot speculate fall back **per row** to target-only
decode — never corruption. This is also consistent with 3.2: per-row fallback
is an eligibility decision, and eligibility remains step-global for the cohort
that does speculate.

Verified byte-identical between `ed55bee` and darkbloom's production pin
`ab73a827` for the `Paged/` and `SequenceKV/` directories — so this is the same
arrangement darkbloom runs, not a divergent branch.

### 3.4 Sliding-window masks must use absolute positions

The retained window is half-open: the key at exactly `anchor - window` is
outside it and must be masked. Upstream shipped an off-by-one where that key
was visible to the assistant. Masks must be computed from **absolute
positions**, never from ring indices or relative offsets, because the assistant
holds a constant position across a draft round while the frozen KV view is a
ring.

### 3.5 Other traps recorded for the box session

- **Mutating replay poisons the prefix cache permanently** — a replay path that
  writes must not exist; upstream removed the write path entirely.
- **Emit the target's correction, not the rejected draft**, at first
  divergence. Full acceptance emits the target bonus token.
- **Pre- vs post-final-norm hidden state** is decided by tensor parity against
  official fixtures, not by inherited code or comments.
- **Depth-controller timing must be like-for-like** — depth-zero timing that
  includes chained neighbour work while speculative timing does not
  systematically biases the comparison.
- **A validation harness that cannot fail is not a gate** — reports must record
  whether speculation was actually active, because zero draft rounds is a
  legitimate outcome (on M5 the `B*L <= 8` envelope leaves B8 with no positive
  safe depth).

---

## 4. Spec-decode seam

### 4.0 RESOLVED — the assistant pin, and what it clears

**Ruling 2026-08-22:** the assistant arm pins to
`mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit @
bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c` — darkbloom's production artifact
(~236 MB). This supersedes 4.1 below, which is retained because the failure
mode it documents is worth not re-acquiring.

Verified against the new revision's own `config.json`, field by field, against
every check `Gemma4MTPCompatibilityValidator.validate` performs:

| Check | Assistant | Target | |
|---|---|---|---|
| `backbone_hidden_size` vs `hidden_size` | 2816 | 2816 | pass |
| `vocab_size` | 262144 | 262144 | pass |
| sliding `sliding_window` | 1024 | 1024 | pass |
| sliding `head_dim` | 256 | 256 | pass |
| sliding `num_key_value_heads` | 8 | 8 | pass |
| full `global_head_dim` | 512 | 512 | pass |
| full `attention_k_eq_v` | true | true | pass |
| full effective KV heads | 2 | 2 | pass |
| `model_type` vs `supportedModelType` | `gemma4_assistant` | `gemma4_assistant` | pass |

Shape: 4 layers, `num_kv_shared_layers: 4` (so it owns no KV and borrows the
target's entirely), `layer_types` = 3 sliding + 1 full.

**The "no loader exists" work item dissolves.** Fork main `@ ed55bee` already
carries the `gemma4_assistant` loader:
`Gemma4MTPConfigurationValidation.swift:36` declares
`supportedModelType = "gemma4_assistant"`, and
`Gemma4AssistantConfiguration` / `Gemma4AssistantDraftModel` are defined
across `Gemma4MTP.swift` and `Gemma4CBv2MTPDrafter.swift`. Nothing needs
registering; adopting fork main brings the loader with it.

### 4.0a Tokenizer delta — inert on this path, verified

Box staging attestation records that the 26B **assistant**'s `tokenizer.json`
is 32,169,440 bytes while the 26B **target**'s is 32,169,626 bytes — different
files, a 186-byte delta. (Both assistant repos, 12B and 26B, ship the
same-sized 32,169,440 file.)

**The fork never reads the assistant's tokenizer.** `Gemma4MTP.swift:1291-1292`
on `ed55bee` documents the drafter download as "Downloads `config.json` and
`*.safetensors` files (tokenizer files are skipped — drafters reuse the
target's tokenizer at generation time)", and the generation path takes
`let tokenizer = target.tokenizer` (same file, ~:2231).
`Gemma4MTPConfigurationValidation.swift` reads only `config.json` — it has no
tokenizer reference at all.

So the shared-tokenizer assumption holds, and by a stronger mechanism than
embedding reuse: the assistant's tokenizer files are never loaded, never
validated, and never staged, so the byte delta cannot reach the runtime. The
delta is not resolved — it is *unreachable*, which is a different and better
guarantee.

Two consequences worth keeping: (1) if any future path ever tokenizes with the
assistant's own copy, this becomes live and the delta must be characterised
(metadata-only vs actual vocab difference) before that path ships; (2) a
staging step that copies the assistant tree wholesale will bring an unused
32 MB tokenizer along — harmless, but it should not be mistaken for a
required artifact or pinned as one.

### 4.1 SUPERSEDED (retained) — the previous assistant pin did not match

`mlx-community/gemma-4-12B-it-qat-assistant-4bit @ 37ae18bb…` is an assistant
**head** (~238 MB; the seat reports `model.safetensors` at 237,894,178 bytes at
this revision), not a standalone 12B model — that much is consistent with the
darkbloom design, where the assistant is a four-layer Q-only drafter that
borrows the target's embeddings, hidden state and frozen KV.

But its own `config.json` at the pinned revision does not describe this target.
Three independent mismatches, each read off the two pinned configs:

| Field | Assistant | Target | Verdict |
|---|---|---|---|
| `backbone_hidden_size` | 3840 | `hidden_size` 2816 | mismatch |
| `text_config.num_global_key_value_heads` | 1 | 2 | mismatch |
| `model_type` | `gemma4_unified_assistant` | — | unsupported |

The fork's own validator makes the first two hard failures:
`Gemma4MTPCompatibilityValidator.validate` requires
`drafter.backboneHiddenSize == target.hiddenSize` and
`effectiveFullKVHeads(drafter) == effectiveFullKVHeads(target)`
(`Libraries/MLXLLM/Models/Gemma4MTPConfigurationValidation.swift:545-548,591-594`
in the fork). And `Gemma4MTPAssistantConfiguration.supportedModelType` is
`"gemma4_assistant"` (same file, line 36), not `gemma4_unified_assistant`.

For contrast, the fork's own 26B-A4B assistant fixture
(`Tests/MLXLMTests/Resources/gemma4-26B-A4B-assistant-config.json`) has
`backbone_hidden_size: 2816`, `num_global_key_value_heads: 2`, `model_type:
"gemma4_assistant"` — i.e. it matches this target on all three, and the pinned
12B assistant does not.

**Reading:** `gemma-4-12B-it-qat-assistant-4bit` is the assistant for the
**12B** backbone (hidden 3840), and the `_unified_` model type is a newer
family the fork has no loader for. This needs an operator ruling before any
MTP arm work: either a different assistant pin, or an explicit decision to
support `gemma4_unified_assistant` and re-derive the compatibility contract.
See OQ-1.

Resolved by the repin in 4.0. Recorded because the failure was silent in every
respect that usually catches this — the artifact is the right size, the right
family, and loads as a Gemma 4 assistant; only three geometry fields say it
belongs to a different backbone. Check the compatibility fields against the
target's config whenever an assistant pin moves.

### 4.2 Assistant loading shape

The assistant is a **head attached to the target**, not a second entry in the
model factory. That makes it structurally the same problem the Qwen track
solved with `mtp-head/` + `mtp-head.manifest.json` + the vendored
`Load.swift` `AdditionalWeightSource` machinery (bare-named head checkpoint
merged pre-`sanitize`). Any `gemma4_unified_assistant` registration work
belongs in the head-attachment path, not `LLMModelFactory`.

This is the argument for **keeping** `mtp-head/` and
`mtp-head.manifest.json` rather than deleting them — see OQ-2.

**`mtp-head.manifest.json`'s own re-aim (2026-08-23).** `bytes: 247463936`
(~236 MiB) is a declared *estimate* for the byte-cap check only, per DECIDE-2
Q-B: a bring-your-own head is bounded by size only. `sha256` is deliberately
omitted rather than fabricated — this manifest was authored without network
access to the artifact to compute a real digest, and a declared-but-
unverified digest is not a gate (`head_provenance` on the wire is the
harness's own recomputed tree digest, never this file's claim). The box
session is expected to fetch, verify the real byte count against this
declaration's `max_bytes` cap, and may add a real `sha256` at that point; an
absent or approximate declared byte count never widens the hard 2 GiB
`max_bytes` cap itself.

> **SUPERSEDED 2026-08-26 (David requant-only ruling).** Bring-your-own heads
> are retired. Both speculative heads are the organizer's pinned weights, a
> participant may only re-quantize them, and `mtp-head.manifest.json` now
> declares `"source": "pinned"` with no `source_url` and no `bytes` — the
> declaration parser refuses `"remote"` and `"in_branch"` by name. The pinned
> repository and revision are recorded in `fixtures/gemma4_26b_a4b_track.json`
> (`assistant.upstream_model_id` / `upstream_revision`) and the per-file digests
> in `fixtures/gemma4_assistant.sha256`, both outside the editable surface. The
> paragraph above is kept as the record of the retired design.
>
> The 236 MiB figure above is also where the box bricked on 2026-08-26: with
> `mtp-head/` still an editable path, the organizer's own monolithic staged
> `model.safetensors` (236,124,704 B) tripped the new `exemptPathMaxFileBytes`
> cap of 100,000,000 in the pinned benchd's budget walk. Dropping the head
> directories from `editablePaths` makes the staged head invisible to that walk
> and unbricks the box without a benchd rebuild.

### 4.3 DFlash arm

`z-lab/gemma-4-26B-A4B-it-DFlash` is a 0.4B block-diffusion drafter published
for vLLM/SGLang only. David ruled the MLX conversion target is **BF16, no
quant**. The conversion is a separate work item and is not attempted here.

That ruling fixes the ORGANIZER-pinned drafter's format. It does not bound
what a participant may stage: §3.4 of the participant contract allows a
re-quantized drafter, and since 2026-08-26 `DFlashDraftModel.load(from:)`
honours a `quantization` block in the drafter's own `config.json` —
`DFlashConfigurationDocument` decodes it through the same
`BaseConfiguration.PerLayerQuantization` plumbing the MTP head loader uses, and
the geometry bounds both arms apply live in
`MLXLMCommon.QuantizationGeometry`. Before that the block was dropped into
`DFlashConfiguration.ignoredConfigKeys` and a packed checkpoint died in
`update(parameters:verify:)` on a shape mismatch.

The seam to preserve is the DFlash-era track contract surface already in the
tree (`fixtures/laguna_xs_2_1_dflash_track.json`,
`docs/dflash-track-correctness-contract.md`, the `dflash` spec-mode that fails
closed at resolution). Note the tension with section 2.2 option (c): deleting
the DFlash runtime to unblock the fork adoption would remove the arm's
substrate. Sequence the two decisions together.

---

## 5. KV backend — CONTIGUOUS, pinned on both legs

**RULED 2026-08-22. This supersedes the paged parity pin carried in from the
bring-up handoff.**

David: *"go with the paged/contiguous decision that darkbloom uses."*
Darkbloom's production decision is **contiguous** — the v0.8.1 capacity revert
made it the serving arm, with paged available explicit-only. So:

- **KV backend = contiguous**, pinned **explicitly on BOTH legs** — the serial
  control and the speculative leg. Not `.auto`, on either.
- **Refusal, not degradation.** If the pinned backend cannot be honoured, the
  run refuses. It must never silently resolve to something else and publish a
  number, which is precisely the failure mode the exactness report caught
  (see 5.2).
- **Parity target = darkbloom's contiguous serving arm.** The thing this engine
  is trying to reproduce is what the fleet actually runs, not an abstract
  most-exact configuration.

### 5.1 Why this is coherent despite the exactness report

The report (5.2) found contiguous diverging on this checkpoint, so pinning
contiguous looks like knowingly choosing the worse arm. It is not, and the
reason is worth stating precisely, because a later reader who half-remembers
the report will try to "fix" this back to paged.

What the report measured is a **cross-condition** comparison: prefix-cache
adoption versus a cold prefill of the same prompt. Our parity claim is
**within-backend**: the speculative leg must be **token-exact against a serial
control running the same backend**. Both legs are contiguous, both legs are
cold, and neither leg adopts a prefix cache. The adoption divergence therefore
does not bind our claim — it is a statement about a code path this engine's
parity legs do not exercise.

Pinning paged would have bought exactness against a condition we do not test,
at the cost of diverging from the configuration we are trying to reproduce.

This is why the pin must be **explicit on both legs**. A same-backend claim is
only meaningful if both legs provably ran the same backend; `.auto` on either
leg makes the claim unfalsifiable even when it happens to resolve correctly.

### 5.2 The exactness report, and what it does and does not prove

`docs/reports/2026-07-26-gemma-26b-adoption-exactness.md` in
`Layr-Labs/d-inference`. Retained because its scope is narrower than
"contiguous decode is wrong", and both over- and under-applying it are easy:

- Measured: **prefix-cache adoption vs a cold prefill** of the same prompt, on
  the end-to-end slot path including the SSD tier.
- On this 26B checkpoint all three paged arms reproduced the cold answer
  byte-identically; explicit `contiguous`, `.auto`, and a paged-requested but
  kill-switch-degraded arm all diverged at byte 4 (cold 32 tokens vs adopted
  12). Rows split on the **resolved** backend, not the requested one.
- Cold-vs-cold was byte-identical on every arm — not run-to-run noise. An
  fp16-vs-fp32 control was token-exact — not precision.
- Checkpoint-dependent: on `gemma-4-e2b-it-4bit` the polarity reverses (paged
  diverged).

The load-bearing lesson for us is **not** "paged is better". It is that a
**requested backend silently degrading to a resolved one changed the answer and
was not visible in the result** — which is exactly why this track pins
explicitly and refuses rather than degrades.

### 5.3 What contiguous means mechanically on this target

25 of 30 layers are sliding-window, 5 are global. On contiguous:

- **Global layers** use `CBv2FullSequenceKV` — `supportsSpeculativeWrites` is
  `true`; rollback rewinds the offset.
- **Sliding layers** use `CBv2WindowedSequenceKV`, which holds exactly `window`
  slots and therefore genuinely aliases. It supports speculation through a
  **staged transaction**, not the paged ring's bookkeeping-only cursor move:
  `beginSpeculativeWrite()` arms the row, every intervening update stages into
  a side buffer instead of touching the ring, and the destructive ring write is
  deferred to `commitSpeculativeWrite()`. The vendored comment states the
  guarantee — *"the ring defers its destructive writes to
  `commitSpeculativeWrite()`, so speculative rollback is value-exact"*
  (`ContinuousBatchingV2/SequenceKV/WindowedSequenceKV.swift:181-186`).

So **all 30 layers speculate on contiguous**, with no per-row demotion. The
cost difference against paged is a real staging buffer on 25 of 30 layers
(`staged.keys.nbytes + staged.values.nbytes` enters the row's memory
accounting) — a memory/bandwidth cost, not a correctness one. Do not
"optimise" it away by switching backends; the backend is a pinned identity.

### 5.4 A≡B determinism tripwire at golden authoring

Before any golden is pinned on this track, run the A≡B check: the same prompt,
same backend, same build, twice, must produce byte-identical output. Author the
golden only if it holds.

It is cheap and it is the check that would have caught the report's finding at
authoring time rather than in production. It matters more here than it did on
the Qwen track because the staged-transaction path (5.3) is exercised on 25 of
30 layers and is newly load-bearing for us.

Per the standing naming/pinning convention, A≡B holds **before** pinning, never
after.

### 5.5 `kv_backend` becomes a pinned identity

There is no track-config or fixture field for the KV backend today. When one
materialises in a later increment it is a **pinned identity**, not a tunable:
it goes through the two-verdict review and the naming check via the
orchestrator, the same as any other pinned identity on this track.

Until that field exists, the pin lives in whatever code path selects the
backend, and the same rule applies — explicit on both legs, refuse on
mismatch.

## 6. Fixtures and pins the box session must regenerate

Nothing in this list can be produced on the laptop. Per
`NEW-MODEL-BRINGUP.md` sections 2.2-2.3 and 4, all of it is generated on the
ranked box against the transformed `weights/` of the new reference, using the
trusted CLI built from the migration branch. Local generation on other Apple
Silicon is invalid.

**The Qwen fixtures currently in this tree are STALE FOR THIS TRACK.** They are
left in place deliberately (they are hardware-generated and the laptop must not
fabricate replacements), and they now fail closed against the new constants —
`Sources/MLXFastCore/Golden.swift` validates every golden's provenance block
against `referenceModel{Repository,Revision}`, so every checked-in Qwen golden
is rejected. That is the correct direction and is not a regression to "fix"
locally.

### 6.1 Checkpoint pins (section 2.2-2.3)

| Artifact | Note |
|---|---|
| `fixtures/reference_gemma4_26b_a4b_qat4bit.sha256` | **LANDED 2026-08-24 (`lane/gemma4-setup-repoint`), laptop-side.** 11 load-bearing records, 15641239658 bytes, sourced directly from the pinned revision's own published tree via the Hugging Face API (same laptop-side-legitimate class as `gemma4_26b_a4b_config.json` below -- public, immutable, content-addressed HF metadata, independent of a box download). Replaces `fixtures/reference_qwen3_8_27b_4bit.sha256` as what `setup.sh`'s `MLXFAST_REFERENCE_MANIFEST_PATH` points at. This is the RAW DOWNLOAD manifest (whole published tree, vision tower bytes included inside the safetensors shards); the TEXT-TOWER-ONLY selection byte count (`fixtures/gemma4_26b_a4b_track.json` `target.text_tower_expected_source_bytes_status`) still needs real safetensors headers and stays box-only. |
| `fixtures/gemma4_assistant.sha256` | **LANDED 2026-08-24, laptop-side, same footing.** 8 records, 268325817 bytes, for `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit @ bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c`. Staged by the new `setup-gemma4-assistant.sh` (replaces `setup-qwen-mtp.sh`, section 8.2a item 2) directly into `./mtp-head/`, the fixed CWD-relative directory `Gemma4A4BAssistantHead.swift` reads. |
| `fixtures/gemma4_26b_a4b_config.json` | **LANDED 2026-08-23, laptop-side, partially.** This one row is narrower than "box-only": the checkpoint's `config.json` is public HF content, independent of the box's download or transform. Fetched directly from the pinned revision's own published `config.json` and re-rendered only for JSON formatting (2-space indent, sorted keys, trailing newline -- matching this repository's existing `fixtures/qwen3_6_27b_config.json` convention byte-for-byte). See `Tests/MLXFastTests/Model/Gemma4A4BArtifactFixtureSupport.swift` for the exact fetch provenance (URL, revision, sha256 of the raw published bytes) and its explicit caveat: this proves the fixture matches what HF publishes today, NOT that it matches the byte the box's transform actually reads. Do not promote its digest to a manifest pin without a box-side re-verification. |
| `fixtures/gemma4_26b_a4b_tensor_inventory.json` | public tensor-inventory fixture; still box-only (needs real safetensors headers, including the one laptop-unverifiable field noted below in 6.3a) |
| `Tests/Fixtures/Gemma426BA4BQat4bit/config-contract.json` | replaces `Tests/Fixtures/Qwen3627B4bit/`; still box-only |
| `Tests/Fixtures/Gemma426BA4BQat4bit/header-inventory-contract.json` | as above; still box-only |

### 6.2 Public correctness fixtures (section 4.1)

**RESOLVED 2026-08-25.** Regenerated Gemma-native at the 1024-seed contract
on box 3 at engine `b71e5c02` (== merged `e59a57c2`), A/B-verified, and landed
in-tree. All three under `correctness_prompts/`:

- `public_longcopy_gate_english_1024.txt` — prompt text (5602 bytes, 1046
  Gemma tokens; sha256
  `606968b5ee8b8c763057aab66cfac6f048e8bfac8a769a7330f3d7c5c9f0e290`)
- `public_longcopy_gate_english_1024_256.json` — drift-tripwire golden
  (1024 prompt + 256 expected tokens, case `longcopy-gate-english-1024`,
  `model_type` `gemma4_text`; sha256
  `d5c6e4edc05c95e784168ec1a90d649dd7e7d0df7bd66d095e89177152e70ab9`, 18700 B)
- `public_longcopy_gate_english_1024_1024.json` — local-submit golden
  (1024 prompt + 1024 expected tokens, same case name; sha256
  `36290b93b1445f354b9b8e3d5ba592976830b40dd924324f822ec55a87140be4`, 29576 B)

The Qwen-era `public_longcopy_gate_english_512.txt` / `_512_256.json` /
`_512_1024.json` are DELETED: the tokenizer-length trap below fired — the
2735-byte prompt tokenizes to only **506** Gemma tokens, short of the required
1024 — so the old prompt could not be reused and a longer prompt was authored.
The regenerated goldens' reference continuation echoes the prompt's closing
instruction for ~18 tokens before copying the passage faithfully; that
preamble is the true greedy reference behavior and is captured as-is.

Regenerate both goldens with `mlxfast-swift generate-golden` on the box, then
repin, in the same commit: the prompt sha256 and byte count, both fixture
sha256s, and every workflow/test literal that mirrors them. (Done for the
1024-seed regeneration: `Sources/MLXFastCore/Constants.swift` default paths,
`Tests/MLXFastTests/GoldenTests.swift` pins, and
`Tests/MLXFastTests/NGramSelfSimilarityTests.swift` thresholds all moved in
the landing commit, and both disabled tests were re-enabled.)

**SEED-LENGTH RULING (2026-08-24).** David: "Seed becomes 1024" — the seed
(`correctnessPromptTokens` / `benchmarkPrefillPromptTokens` /
`benchmarkDecodeSeedTokens`) is now **1024** tokens; the decode window stays
128 steps (`benchmarkDecodeSteps` unchanged), so the golden shape is 1024
prompt_tokens + 129 expected_tokens (seed next-token + 128 checked steps).
The 1024-token versions of the hidden pool prompts must be uploaded and
referenced from this track's benchmark branch; every checked-in and hidden
golden below regenerates at the 1024-token seed. The checked-in
`*_512*` fixtures were the pre-ruling Qwen-vintage captures, rejected
fail-closed (provenance) pending regeneration; as of 2026-08-25 the
regenerated `*_1024*` fixtures (named for the new seed length, as required)
have replaced them and the `*_512*` files are deleted.

**TOKENIZER-LENGTH TRAP (runbook 4.1).** The generator requires the prompt to
tokenize to at least 1024 tokens under
`tokenizer.encode(text:, addSpecialTokens: false)` then `prefix(1024)`. Note
the zero-margin consequence for prompts authored at exactly 1024 tokens:
`prefix(1024)` of a 1024-token prompt is the ENTIRE prompt — the whole prompt
is the seed, and no authoring margin remains. That is coherent and intended
under the ruling, not an authoring accident. The Qwen
tokenizer has vocab 248320; Gemma 4's has **262144** and is a different
tokenizer entirely. **TRAP CONFIRMED AND RESOLVED (2026-08-25, box 3):** the
existing 2735-byte English prompt tokenizes to only **506** Gemma tokens —
half the required 1024 — exactly as this flag suspected. The prompt was
extended to the 5602-byte `public_longcopy_gate_english_1024.txt`, which
tokenizes to **1046** Gemma tokens (22 tokens of authoring margin over
`prefix(1024)`), and the prompt sha and byte pins moved with it (new values
in the 6.2 artifact list above).

`Tests/MLXFastTests/NGramSelfSimilarityTests.swift` was re-checked as
instructed: the regenerated 256 fixture is still highly self-similar while
failing the 0.03 prompt-lookup threshold, but the tokenizer did move the
aggregate — `0.9224806201550387` (119/129) under the Gemma tokenizer versus
`0.45736` under Qwen 3.6. The test's thresholds were re-asserted around the
measured value and the test re-enabled.

### 6.3 Hidden artifacts (sections 4.2-4.3, 5)

Hidden teacher-forced correctness golden, hidden GPQA reference cases, and the
timed decode prompt selection are organizer material and are regenerated per
the runbook on the box. Not enumerated further here.

### 6.3a Types the box session now touches

The harness port renamed the model-facing surface, so the runbook's file names
resolve differently. Current spellings, for the box session's commands and for
the fixture regeneration in 6.1-6.2:

| Role | Type / file |
|---|---|
| Config contract | `Gemma4A4BConfig` (`Sources/MLXFastModel/Gemma4A4BConfig.swift`) |
| Weight names + metadata validation | `Gemma4A4BWeightNames`, `Gemma4A4BWeightLoader` |
| Runtime weight cache (scored path) | `Gemma4A4BRuntimeWeightCache` -> vendored `Gemma4TextModel` |
| KV backend pin | `Gemma4A4BKVBackend` (contiguous; `MLXFAST_GEMMA4_KV_BACKEND` asserts, never selects) |
| Transform family | `TransformModelFamily.gemma4A4B` |
| Transform-side inventory | `Gemma4A4BCheckpointValidation` |

Fixture naming follows: `fixtures/reference_gemma4_26b_a4b_qat4bit.sha256`,
`fixtures/gemma4_26b_a4b_config.json`,
`fixtures/gemma4_26b_a4b_tensor_inventory.json`,
`Tests/Fixtures/Gemma426BA4BQat4bit/`.

**One inventory field is NOT laptop-verifiable: tensor DTYPES.** Names, shapes
and the packing arithmetic were read off the pinned checkpoint's real
safetensors headers, but the dtype column in
`Gemma4A4BCheckpointValidation` encodes the MLX affine-conversion convention
(`U32` codes, `BF16` scales/biases and unquantized parameters) rather than a
per-tensor reading. If the box finds any parameter at F32 — `layer_scalar` and
`router.per_expert_scale` are the plausible candidates — that is a repin of the
dtype table, not a checkpoint fault. Check it first; it is cheap and it gates
the transform.

Also note `Tests/MLXFastTests/Model/Gemma4A4BCheckpointValidationTests.swift`
is entirely SYNTHETIC — it derives its fixtures from the validator's own pinned
geometry, because no `fixtures/gemma4_26b_a4b_*` artifact exists yet. It pins
internal consistency and rejection behaviour, not the artifact. It becomes a
real artifact gate only once 6.1 lands.

### 6.4 Upstream-equivalence gate

**IMPLEMENTED 2026-08-23** on `lane/gemma4-upstream-equivalence`, ported from
the retired-Laguna harness (`Sources/MLXFastModel/LagunaUpstreamEquivalence.swift`,
still in tree, untouched, own deletion is a separate call). The section-3.7
reference this note used to carry names the shape in the
`NEW-MODEL-BRINGUP.md` runbook that this gate satisfies, not a section of this
document -- there is no local section 3.7 here. This subsection is now the
authority for this track's gate.

**Type and test:**
`Sources/MLXFastModel/Gemma4A4BUpstreamEquivalence.swift`
(`Gemma4A4BUpstreamEquivalence.compare(weightsPath:promptTokens:decodeTokens:)`,
`Gemma4A4BUpstreamEquivalenceReport`/`Step`), exercised by the env-gated test
`gemma4RuntimeMatchesVendoredUpstreamOnM5WhenEnabled` in
`Tests/MLXFastTests/Model/Gemma4A4BUpstreamEquivalenceTests.swift`. Read that
file's and the source file's header comments before touching either --
between them they record two places the Laguna shape did not transfer
cleanly (no separate runtime model type for Gemma 4; the quantization
override keys need the `language_model.` prefix restored on the vendored leg
too, which nothing upstream does today).

Required **before** generating any golden: load the runtime model and a
standalone-vendored load of the same checkpoint, install the same
`[String: MLXArray]` into both, and compare one 1024-token prefill plus 8
serial teacher-forced decode steps on max/mean abs logit error and both greedy
argmaxes. Default tolerance is exact (`0`). Do not proceed until it is exact.
There is no golden file anywhere in this gate -- both legs are live runtime
forwards on the same in-memory weights, and the prompt/decode tokens are a
deterministic public synthetic sequence generated in the test (documented
there), not read from any fixture.

**Box invocation** (weights path is this track's transformed `weights/` tree,
the same layout staged at `~/gemma4-bringup/engine/weights` per the bring-up
handoff):

```bash
MLXFAST_RUN_GEMMA4_UPSTREAM_EQUIVALENCE=1 \
MLXFAST_GEMMA4_EQUIVALENCE_WEIGHTS_PATH=<path-to-transformed-weights> \
swift test --force-resolved-versions \
  --filter gemma4RuntimeMatchesVendoredUpstreamOnM5WhenEnabled
```

(Swift Testing's `--filter` matches the bare function name, not a
`File.function` qualifier -- verified laptop-side; the latter silently
matches zero tests instead of failing loudly.)

`MLXFAST_GEMMA4_EQUIVALENCE_MAX_ABS_ERROR` is available for diagnostics only
(loosens the printed pass/fail judgement to investigate a failure) and must
never be set to publish a passing result -- the gate's own default is exact
`0`, matching section 6.4's requirement above.

Given section 1.3, this gate is the one most likely to catch a mis-applied
per-tensor quantization override -- on EITHER leg, since as of this
implementation both legs apply the checkpoint's per-tensor overrides through
independent code paths (this engine's `Gemma4A4BQuantization` override table
vs the vendored `BaseConfiguration.PerLayerQuantization`), not just the
runtime one.

### 6.5 Timed-prompt tape recorder runbook (`record-reference-tape`)

**LANDED laptop-side (`lane/gemma4-reference-tape-recorder`).** The ranked
track's `timed_prompt_pool[]` (currently `PENDING-ORGANIZER`) pins
teacher-forcing TAPES — benchd's `TimedPromptTapeDocument`
(`bench-core/src/tape.rs`): top level `{seed_tokens, reference_seed_token,
rows, reference_self_consistent, emitted_tokens}`, each row
`{sequential_argmax, top2_tokens, top2_logits, top1_logit}`, strict
`deny_unknown_fields` at both levels. The Qwen-era producer (`mtp-verify
--generate`) and the DFlash one (`dflash-reference`) are both unrunnable on
this engine — their worker verbs left with the OQ-3 adoption / harness port.
`record-reference-tape` is the replacement, and it is ARM-NEUTRAL by
construction: both of its recording backends are pure serial (no drafter, no
head, no speculative code), so the tape is the serial reference trajectory
every speculation arm is verified against.

**RECORDING BACKENDS (CBv2 default — the within-backend fix).** The engine
has two serial decode implementations, and the leg-identity fix (PR #29) put
BOTH v1.1 free-run legs on the width-1 CBv2 engine while the pool tapes were
recorded through the legacy teacher-forced correctness verbs — a
two-implementations-one-gate pairing rule 5.1 forbids. The recorder therefore
carries `--recording-backend cbv2|legacy` (default `cbv2`):

- `cbv2` drives the trusted-CLI-only worker kinds `record_reference_begin` /
  `record_reference_run`, which reuse the SAME shared begin executor the wire
  legs dispatch to (`openSingleStreamFreeRunSession`, serial drafter-less
  configuration) plus an engine-side top-2 readout
  (`CBv2SamplingParams.topLogprobs = 2`, raw log-softmax over the
  pre-transform logits; selection-neutral, pinned by the recording
  executor-identity test). The self-consistency replay is a second fresh CBv2
  pass through the same executor — also the 5.4 A≡B tripwire run per
  recording. Note the top-2 VALUES on this backend are logprobs, not raw
  logits: per-row margins are identical (log_softmax shifts both entries by
  one normalizer) and benchd treats the fields as diagnostics.
- `legacy` keeps the original `correctness_begin` / `correctness_step` path
  byte-for-byte (pinned by test) for provenance/reproduction of the
  legacy-recorded pool tapes; it is the implementation the teacher-forced
  decode verbs still deliberately run, but NOT the one any free-run leg
  executes.

Backend provenance is session evidence (stderr progress lines + the worker's
session-diagnostics executor line + the stdout summary), never a document
field — the tape byte format is backend-neutral and benchd's parse is
unchanged.

Box invocation, one tape per pool prompt, against this track's transformed
`weights/`:

```bash
mlxfast-swift record-reference-tape \
  --prompt-file <prompt.txt> \
  --weights <transformed-weights> \
  --output <tape.json>            # --steps defaults to 129 rows
```

Mechanics and gates:

- **Seed contract:** the prompt is tokenized with the weights-dir tokenizer
  (`addSpecialTokens: false`) and must reach `correctnessPromptTokens`
  (1024); exactly that prefix is the seed (same convention and same
  zero-margin trap as `generate-golden`, section 6.2).
- **Rows:** `--steps` (default `benchmarkDecodeSteps + 1` = 129) greedy rows
  after the seed argmax. A shorter tape cannot oracle the timed window and is
  refused unless `--allow-short` is passed.
- **Self-consistency is a replay, not a flag:** after recording, the recorder
  runs a second fresh teacher-forced pass over the recorded chain (full seed
  re-prefill included) and verifies the seed argmax and every row argmax
  reproduce. Only then does it write `reference_self_consistent: true`; any
  mismatch aborts with the failing position — a tape carrying `false` is
  never written (benchd refuses an explicit `false` as an operator fault).
- **Sandbox:** the worker runs under the same blocked-output profile as
  `generate-golden` — when regenerating a tape in place, submitted-surface
  model code cannot read the artifact it is reproducing.
- **Write discipline:** staged to a temp sibling and strict-validated
  (exact key sets at both levels, emitted chain == row chain, vocab range,
  row count) before it may replace the destination.
- **Pinning:** pool pins are `sha256` + `bytes` of the exact emitted file;
  pin what the recorder wrote, unmodified.

**Cross-repo constant lag, report-only:** the pinned benchd submodule's
`bench-core` still carries `VOCAB_SIZE = 248_320` (Qwen) while this engine's
vocab is 262 144, so a real Gemma tape whose tokens land in
`248320..<262144` would be refused by the pinned benchd loader. Bench-side
repin needed before arming the pool; same sequencing class as the wire-sha
crosscheck note in 8.2a.

---

## 7. Open questions

Each of these needs a ruling; none should be resolved by an implementer.

**OQ-10 — restore true `natural_accepted_by_stream` (tracking, per the round-execution
review's accept-condition).** The cohort mtp leg reports the committed COMMON WIDTH per
row as a DOCUMENTED FLOOR, not the true per-row pre-min accept walk — the vendored
`EngineV2`/`CBv2Event` surface exposes no per-row pre-min seam, though the engine
computes `naturalEmitted` internally (`EngineLoopV2+MTPFinalize`). The field is
AUDIT-only and never scored, so the floor is safe for measurement; the cost is that
per-row straggler-throttling visibility is reduced (partially covered meanwhile by the
observed `depth_clamp_reasons`), and benchd's `CohortNaturalBelowCommitted` refusal is
unfireable from this engine. RESTORATION PATH: surface `naturalEmitted` on a public
per-row seam in our in-house `Layr-Labs/mlx-swift-lm` fork (small, plannable vendored
addition), then replace the floor with the observation and delete this OQ. Until then
the wire value must keep its in-code "documented floor" labeling.

**OQ-1 — assistant pin. RESOLVED 2026-08-22.** Pin moves to
`mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit @ bb94eae1`. All nine
compatibility fields verified against the pinned configs (section 4.0), and
fork main already carries the `gemma4_assistant` loader, so the
"no loader exists" item dissolves. `gemma4_unified_assistant` is not involved
and needs no support. No open work; the MTP arm is unblocked once the vendored
adoption (OQ-3) lands.

**OQ-2 — `mtp-head/` and `mtp-head.manifest.json` disposition.** Not deleted
(per brief). Given 4.2, the recommendation is to **keep and re-aim** them at
the Gemma assistant head rather than retire them: the assistant is a head
attached to the target, which is exactly the contract these two already
encode (declared `source`/`sha256`/`bytes`, size-cap gate, absent-declaration
means organizer-pinned). Confirm, and confirm whether the 2 GiB cap and the
"declared sha256 carried but not verified" posture carry over unchanged.

**OQ-3 — fork adoption sequencing. RESOLVED and EXECUTED 2026-08-22: DFlash
DEFERRED.** Landed on `lane/gemma4-fork-adoption`; see 2.2b-2.2d for what was
carried, what left, and what deliberately stayed.
Adopt fork `main @ ed55bee` clean — the v1 engine goes with it. Serial parity
and the MTP arm come first; **DFlash-onto-CBv2 is its own follow-up lane**, and
the #56 switchable-decoder seam is kept so that lane slots in without
re-litigating the abstraction. This is option (c) of the three below, scoped as
a deferral rather than a deletion of intent.

Two things this ruling makes the adopting change responsible for, neither of
which is optional:

1. **The sliding-window rollback seam fix (2.3) leaves the tree with DFlash.**
   It is a real defect fix — Laguna's 512 window vs a 512-seed + 128-decode
   ranked window means every scored run crossed the seam. Gemma 4's window is
   1024 and the same arithmetic applies, so the DFlash lane must re-derive it
   on CBv2 rather than assume the CBv2 rollback path never had the bug. Record
   it as a known-defect-to-recheck, not as deleted work.
2. **The four load-bearing patches in 2.3 are NOT part of the deferral.**
   `Load.swift` `AdditionalWeightSource` (the head-merge ordering the assistant
   depends on), `KVCache.swift`, `AttentionUtils.swift` (the wide-decode
   exactness chunk — same defect class as trap 3.1, and the MTP arm will hit
   it), and `BatchKVCache.swift` all carry forward.

The adopting change is therefore: swap to `ed55bee`; re-apply those four
patches; delete the duplicated `Gemma4.PositionOffset` from `Gemma4Text.swift`
and restore `embedTokensForDrafter` (2.3a); take the 16-file fork-only closure
plus the six `EngineLoopV2+MTP*` files (2.3b); resolve the mlx-swift freeze
(2.4); drop the DFlash harness surface
(`Sources/MLXFast{,Trusted}Harness/QwenRuntimeDFlash*.swift`,
`Sources/MLXFastModel/LagunaDFlash*.swift`, the DFlash tests) behind the
retained decoder seam; and re-point every engine target that referenced the
deleted v1 surface. One atomic change — it cannot be landed incrementally while
keeping the tree building.

**OQ-4 — the retired-names rule.** `AGENTS.md:1008-1009` says "The retired
Gemma-era MTP names stay retired — a Qwen track id must not substring-collide
with them." The retired list is `MLXFAST_MTP_`, `mtp-ranked`,
`measure-mtp-job`, `mtp-weights`, `laguna-xs-2.1-mtp`, `gemma4-31b-it`. This
track id `gemma4-26b-a4b-mlx-v1` is substring-clean against all six, and this
port introduces no `MLXFAST_MTP_`-prefixed environment name. But the rule was
written when a Gemma track was the retired thing and a Qwen track was the live
one; that premise is now inverted. Confirm the rule survives as-written
(a name-collision tripwire) rather than as "no Gemma-4 track may exist", and
confirm the guard suite should be re-pointed rather than relaxed.

**OQ-5 — paged KV vs speculation. RESOLVED 2026-08-22: THE CONFLICT DOES NOT
EXIST.** The darkbloom investigation reported, and the finding that raised this
question was stale (see 3.3). Paged rings are over-provisioned by a whole
speculative span, MTP verification writes land in place across all 30 layers,
paged rollback is a pure cursor move, contiguous windowed rows use a real
staged transaction, and ineligible rows degrade per-row to target-only. **The
arrangement ships in the tree this branch adopts** — no design work, no
backend re-selection, and no blocking of the MTP arm.

Citations: `mlx-swift-lm@5e36ed2`;
`Paged/PagedSequenceKV.swift:253-273`; `Paged/PagedSeamContract.swift:216-222`.
`Paged/` and `SequenceKV/` are byte-identical between `ed55bee` and darkbloom's
production pin `ab73a827`.

**The follow-on question — WHICH backend the parity legs pin — is also now
RULED: contiguous, explicitly on both legs, refuse-not-degrade.** See section 5.
Nothing in this branch depends on it; the branch pins no backend in code.

**MTP envelope constants to adopt verbatim when the arm lands** (from
`MTPAutomaticVerificationPolicy`, recorded here so they are not re-derived by
guess): fixed **depth 1** for stateless Gemma, and the rectangular cap
**`B * (1 + k) <= 8`** on M3 and later. Note the cap interacts directly with
trap 3.1 — it is the same width wall the `AttentionUtils` exactness chunk
addresses.

**OQ-6 — scope of the KV pin. RESOLVED 2026-08-22, and the question was
reframed by the ruling.** It asked whether the engine's pin should be written
as a backend-identity rule or as a narrower claim tied to prefix-cache
adoption. The ruling settles both halves at once: the pin is a **backend
identity** (`contiguous`, explicit on both legs, refusal not degradation), and
the adoption question is out of scope for our parity claim because that claim
is within-backend — see 5.1.

The one part of the original question that survives is the warning attached to
it: the report's Route B harness is structurally blind to this bug class and
will report "adoption exact" on both backends. That is now covered by the A≡B
tripwire at golden authoring (5.4) rather than by backend choice.

**OQ-7 — the legacy `.gemma4` transform family.** `Transform.swift:47-52`
retains a `.gemma4` family for the archived dense 31B multimodal layout,
deliberately, as the only family exercisable with small synthetic fixtures
(runbook 3.4 says do not delete it). This track's target also has a nested
`text_config` and a top-level `model_type` of `gemma4`, so
`detectModelFamily` (`Transform.swift:638-663`) would route it to the legacy
family and emit the projection + tied-head packed13 sidecars. The new family
must be detected **before** the `text_config`-means-Gemma fallthrough — the
proposed discriminator is `text_config.enable_moe_block == true` together with
`num_experts == 128`, which the legacy dense 31B does not carry. Confirm the
discriminator, and confirm the new family emits **no** sidecars (the vendored
`Gemma4TextModel` reads the checkpoint's own tensors and handles tying
itself).

**OQ-8 — mixed-precision quantization envelope.** Section 1.3: the checkpoint
ships 4-bit affine g64 with 120 per-tensor 8-bit promotions. The retired serial
track's "frozen quantization envelope" prose (`AGENTS.md`, "Retired Serial
Non-Speculative Track Rules") describes a single uniform representation plus
one sanctioned re-quant. What is the envelope for this track — "exactly as
shipped, including every per-tensor override" is the obvious answer, but it
should be stated, because a submission that applied the default 4-bit spec
uniformly would be *lossier* than shipped and the existing prose does not
name that case.

---

## 8. What is in this branch, and what is not

**In (branch `lane/gemma4-kv-contiguous-ruling`, stacked on
`lane/gemma4-fork-adoption`):** documentation only — the KV backend ruling in
section 5, and the OQ-5 / OQ-6 / trap-3.3 cross-references it supersedes. No
code change; the branch is stacked rather than based on `main` so it neither
disturbs the adoption PR under review nor conflicts with it.

**In (branch `lane/gemma4-fork-adoption`):** `Vendor/mlx-swift-lm` advanced to
fork `main @ ed55bee` with five engine-local files carried forward and one
vendored `Package.swift` edit retained (2.2b); the model-side DFlash surface
removed and the `dflash-runtime-worker` verb made to fail closed (2.2c); the
compiled-decode flag guard narrowed to the surviving flag (2.2d); the engine
`Package.swift` pin comment updated and its `MLXSpeculative` product references
dropped.

Cumulatively with the previous increment, the branch also carries the
trusted-core track identity and geometry, and this document.

**Not in, and why:**

- The harness Qwen→Gemma type port (`Gemma4A4BConfig` /
  `Gemma4A4BRuntimeWeights` / `Gemma4A4BCheckpointValidation`, the transform
  family, the ~22-file harness rename) — held for the next increment
  deliberately, and made separable by carrying `Qwen35.swift` forward (2.2b).
- The MTP arm — unblocked (OQ-1 resolved, loader verified present in the
  adopted tree, OQ-5 conflict dissolved), and it lands with the Gemma harness
  port rather than ahead of it.
- The DFlash arm — deferred per OQ-3; the MLX-free driver, protocol, ledger and
  contract stayed in tree so the follow-up lane re-homes three `BatchKVCache`
  references rather than rewriting.
- Any fixture or golden regeneration — box-only by construction (section 6).

The tree builds and its tests pass. It does not yet load a Gemma checkpoint:
the runtime still parses the Qwen config contract, so this branch changes what
the engine is *built against*, not yet what it can *run*.

### 8.2a Two coordination items this increment created

**1. `EmitWireFixtureTests` sha moved, and benchd pins the same constant.**
The emitted wire FIELD SET is unchanged; two VALUES moved because the worker no
longer advertises an MTP mode — `spec_modes` is now `["serial"]` and the
free-run begin echoes `{"mode":"serial"}`. The engine-side pin was updated to
`9305aae4…` (was `34d5babc…`).

benchd holds the same constant for a cross-repo crosscheck, and **it was not
touched** — the gitlink is pinned at `98f44fa5` and bench-side changes are out
of scope for this lane. Until benchd is repinned, that crosscheck compares two
different surfaces. This needs sequencing bench-side; it is not a defect in
this branch, but it is not self-resolving either.

**2. `setup-qwen-mtp.sh` is now dead.** It stages an MTP head and passes
`--mtp-head`, and both the `mtp-runtime-worker` verb and that flag now refuse.
The script is Qwen-era and was left untouched deliberately (out of scope), but
it should be re-aimed or retired alongside the MTP arm increment. It is
referenced from `AGENTS.md` and `README.md` setup prose.

**RESOLVED 2026-08-24 (`lane/gemma4-setup-repoint`).** `setup-qwen-mtp.sh` is
removed and replaced by `setup-gemma4-assistant.sh`, which stages the pinned
Gemma 4 assistant (`mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit @
bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c`) directly into `./mtp-head/` --
the fixed CWD-relative directory `Gemma4A4BAssistantHead.swift` reads, not
a cache-dir-plus-compat-symlink pair, since at the time there was no
`--mtp-head` flag to delegate a path through. [UPDATE 2026-08-25: the flag is
restored — see the `--mtp-head RESTORED` note in §8.3; the CWD `./mtp-head/`
staging this script performs remains the default channel and keeps
working unchanged.] `AGENTS.md` and `README.md` still name the
old script (both are owned by `lane/gemma4-participant-contract`, out of
scope for this lane); their setup prose needs the same swap this note
records.

### 8.3 The harness port increment (`lane/gemma4-harness-port`, 2026-08-22)

This lands the item 8 listed as "not in": the harness Qwen→Gemma type port. The
paragraph above ("the runtime still parses the Qwen config contract") no longer
holds — the serial path now parses `Gemma4A4BConfig` end to end.

**In:**

- `Sources/MLXFastTransform/Gemma4A4BCheckpointValidation.swift` and the
  `.gemma4A4B` transform family: detected on `text_config.enable_moe_block` +
  `num_experts == 128` **before** the unconditional `text_config`-means-legacy-
  `.gemma4` fallthrough, selecting `language_model.*` only, emitting the
  flattened tower plus the checkpoint's `quantization` block **verbatim** —
  all 120 per-tensor overrides preserved, not reduced to a triple (section 1.3)
  — and emitting no sidecars.
- The model-facing rename across both harness trees and both CLIs:
  `Qwen35Config` / `Qwen35WeightLoader` / `Qwen35RuntimeWeightCache` →
  `Gemma4A4B*`, and `Qwen35TextModel` → `Gemma4TextModel`.
- The dead Qwen model surface deleted: the 14 `Sources/MLXFastModel/Qwen35*`
  files (config, weights, runtime weights, and the whole custom fast path).
- `verifyQwenCachePosition` **re-derived** for the Gemma cache stack. This was
  not merely stale: it demanded that non-global caches be RECURRENT and pinned
  at offset 0, which is a fact about Qwen's gated-delta layers. Gemma has no
  recurrent layers — `StandardKVCache` on the 5 global layers and
  `RotatingKVCache` on the 25 sliding ones, both counting every position
  written (the ring lives in `RotatingKVCache.idx`, not `offset`) — so the rule
  is lockstep plus a windowed/unbounded topology check. Left alone it would
  have passed the first prefill and refused the second forward on real weights.

**Out, and how it refuses:** the MTP arm. It was written against the Qwen tower
end to end (`Qwen36MTPTarget` conforms `Qwen35TextModel` and
`MLXLLM.Qwen35Model` and nothing else; the head attachment merges a Qwen head
into a Qwen backbone; the block session's cache reasoning is the gated-delta
tower's), so renaming the model types would have produced code that compiles
and cannot run. The four `Qwen36MTP*` model files, the six `QwenRuntimeMTP*`
harness twins, the `mtp` spec mode and the `--mtp-head` plumbing are deleted;
`mtp-runtime-worker` and `mtp-verify` are RETAINED and REFUSE, pointing here.
`QwenMTPHeadDeclaration` stays — it is the DECIDE-2 (Q-B) size gate, neither
Qwen-specific nor MLX-linked.

**`--mtp-head` RESTORED 2026-08-25 (SUPERSEDES the deletion above for that
one flag).** The MTP arm returned 2026-08-23 with CWD-only head staging
(`./mtp-head/`, `Gemma4A4BAssistantHead.swift`), but benchd's measure-job
spawn contract kept the flag the whole time: every worker leg is spawned
`runtime-worker --weights <W> --mtp-head <H> [--speculative-protocol v1.1]`
(benchd @ `c2327d15`, `crates/benchctl/src/measure_job.rs`
`timed_leg_base_args` / `leg_spawn_args`, fenced by
`RUNTIME_WORKER_ACCEPTED_FLAGS = {--weights, --mtp-head,
--speculative-protocol}`), the serial control with the pinned head and the
candidate with its declared BYO head — so the shrunken
`{--weights, --speculative-protocol}` verb surface killed every measure-job
leg pre-hello. The generic `runtime-worker` verb accepts the flag again and
the CBv2 head loader now takes the directory:
`loadGemma4AssistantHeadIfStaged(explicitDirectoryPath:)` /
`resolveGemma4AssistantHeadStaging` load from the argv-named directory when
the flag is present (fail-closed if it is not a loadable head — an explicit
argv directory is a declaration, same posture as a broken
`mtp-head.manifest.json`), and keep the CWD `./mtp-head/` default with its
lenient absent-is-serial-only semantics when it is absent (the native
trusted CLI and `setup-gemma4-assistant.sh` flow, unchanged). ARGV-ONLY:
`QMTP_HEAD_DIR` / `QMTP_CANDIDATE_HEAD_DIR` are benchd's own env inputs,
resolved benchd-side into the flag value; the engine does not read them, and
benchd's allowlisted child env drops them before the worker starts. The
engine-side option surface is the shared
`runtimeWorkerAcceptedOptionFlags` constant, pinned equal to benchd's fence
by `Gemma4AssistantHeadStagingTests`. The wire protocol is untouched — this
is spawn-argv surface only (`emitEngineWireFixture` pin `e79fd829…`
unchanged).

**Return obligations of the MTP arm (explicit, so they cannot be silently
dropped):** two tripwire suites were deleted WITH their subjects and must come
back re-derived for the Gemma arm, not merely be remembered —
`MTPWorkerTwinEqualityTests` (the trusted/untrusted harness-twin drift
protection: both trees must carry byte-equivalent MTP driver logic) and
`QwenMTPTimingRetirementTests` (the retired-timing protection: the engine's
self-measured timing surface stays retired; benchd owns the clock). The
DFlash-side deletions carry their restore pointers in their `.disabled`
reasons; these two are the MTP-side equivalents and this list is their
pointer. An MTP-arm increment that does not restore both protections is
incomplete.

**Still box-gated:** everything in section 6, unchanged. The dtype column of
the new inventory is the MLX affine-conversion convention (`U32` codes, `BF16`
companions and unquantized parameters), not a reading of the pinned headers;
shapes and names are read off the artifact. A different dtype on one of the
small unquantized tensors is a repin of that table, not a relaxation.

### 8.2 Test state after the vendored adoption

`swift build --force-resolved-versions`: **green**.
`swift test`: **green — 339 tests in 8 suites, 0 failures, exit 0.**

The count moved 347 → 339 because eight tests left with the DFlash arm: the
seven in `RuntimeWorkerDFlashWireSurfaceTests.swift` (deleted — they exercise
pure functions that lived in the removed worker, so `.disabled` cannot apply
when the symbols no longer resolve) and one net from the suites above.

Three further tests moved to `.disabled` with reasons, all in
`DFlashStartupMemoryPolicyTests.swift`:
`dflashWorkerAppliesStartupMemoryPolicyBeforeItsModelLoads`,
`dflashWorkerTwinsDifferOnlyByTheTrustedHarnessGuard`, and
`trustedWorkerDoesNotCallTheEditableApply`. Each reads
`QwenRuntimeDFlashWorker.swift` from disk as a source-level assertion; each
reason names the deferral and points at OQ-3. They compile fine — they are
skipped, not broken.

`compiledDecodeFlagsStayReadableByModelSources` was **fixed, not skipped**, by
narrowing it to the flag that still exists (2.2d).

The six skips from the previous increment are unchanged and still box-gated.
Running total: **9 skipped with reasons, 0 hard-red.**

### 8.1 Test state: `swift build` green, `swift test` zero hard-red

`swift build --force-resolved-versions` is clean. `swift test` runs 347 tests
in 8 suites. Ten failed on the initial repin — every one a Qwen model-fact
assertion the identity/geometry change correctly invalidated, i.e. fail-closed
rather than breakage. (An earlier revision of this document said nine; that was
an undercount, corrected here.)

All ten are now **green or explicitly skipped with a reason**. Nothing is
hard-red.

**FIXED — literals that merely echoed the constants (4 tests, 8 assertions):**

| Test | Was |
|---|---|
| `goldenModelIdentityFailsClosedWhenRequired` | `GoldenTests.swift` pinned `requiredGoldenModelType == "qwen3_5_text"` plus four rejection-message and document literals |
| `qwenFullAttentionIntervalMatchesThePinnedLayerSchedule` | `RuntimeWorkerSupportTests.swift` pinned interval 4, 16 full layers, first 3, last 63 — now 6 / 5 / 5 / 29, with the full `[5, 11, 17, 23, 29]` schedule asserted outright |
| `rankedWorkerBenchmarkRejectsAForeignModelGoldenBeforeTheWorkerStarts` | `TrustedWorkerProtocolTests.swift` pinned the expected-model-type string in a rejection message |
| `trustedTracePreservesLargeTopKAndOutOfSubsetExpectedDiagnostics` | same file, a test-fixture golden built with `model_type: "qwen3_5_text"` |

These were safe to fix now precisely because they assert *nothing of their own*
— they restate constants this branch already repinned. Note the third and
fourth were originally mis-classified here as needing the harness rename; they
did not.

**SKIPPED with `.disabled(reason)` — blocked on box-regenerated fixtures
(4 tests):** each reason names the blocking artifact and points at section 6.

| Test | Blocked on |
|---|---|
| `checkedInPublicCorrectnessGoldenIsValid` | `correctness_prompts/*_256.json` provenance (6.2) — **RESOLVED 2026-08-25** (`lane/gemma4-local-goldens`): re-enabled against the regenerated `*_1024*` fixtures |
| `currentLongcopyFixtureDemonstratesHighSelfSimilarity` | same golden; thresholds are tokenizer-dependent and must be re-asserted after regeneration (6.2) — **RESOLVED 2026-08-25**: re-asserted at the measured `0.9224806201550387` and re-enabled |
| `qwen36ConfigContractDigestMatchesTheReferenceManifest` | `fixtures/reference_qwen3_8_27b_4bit.sha256` (6.1) |
| `qwen36PublicConfigFixturePinsExactArtifactSemantics` | `fixtures/qwen3_6_27b_config.json` (6.1) |

**RESOLVED 2026-08-23 (`lane/gemma4-worker-config-gate`) --
`qwen36TransformOutputSatisfiesTheRuntimeWorkerPinnedConfigGate` is no longer
disabled.** It was the runtime-worker pinned-configuration gate's own test:
`validateRuntimeWorkerPinnedConfigurationData`
(`Sources/MLXFastHarness/QwenRuntimeWorker.swift` and its
`Sources/MLXFastTrustedHarness` twin) was still pinned to the Qwen
`qwen3_5_text` schema, so it rejected every real transformed Gemma 4 config
before weight loading -- the box-observed blocker this whole port responds to.
Both twins are now ported to the Gemma 4 26B A4B schema (`Gemma4A4BConfig`,
`MLXFastCore.Gemma4A4BConfigKeys`); the participant-side gate single-sources
through `Gemma4A4BConfig.load(data:)`, while the trusted-side gate re-derives
the same contract locally (it does not, and must not, link `MLXFastModel`)
and is kept in lockstep by
`Tests/MLXFastTests/Model/Gemma4A4BRuntimeWorkerGateLockstepTests.swift`. The
re-derived test is
`gemma4A4BTransformOutputSatisfiesTheRuntimeWorkerPinnedConfigGate` in
`Tests/MLXFastTests/Model/Qwen35ArtifactContractTests.swift`, unblocked by the
new `fixtures/gemma4_26b_a4b_config.json` -- a LAPTOP-SIDE fetch of the
pinned revision's public `config.json` (not box-generated; see that fixture's
companion `Gemma4A4BArtifactFixtureSupport.swift` for provenance and the
explicit "not box-verified" caveat this leaves open).

**SKIPPED — needs rewriting, not renumbering (1 test):**
`rollbackTrimsTheWholeVerifyWindowAndNeverTrimsRecurrentCaches` expects 48
recurrent caches from Qwen's gated-delta tower. The Gemma tower has **no
recurrent layers at all** — 25 sliding + 5 global KV caches — so this contract
must be re-derived against the Gemma cache stack alongside the MTP arm. Merely
changing 48 to 25 would assert a shape that is not what the model does.

Five of the ten are waiting on artifacts only the ranked box can produce, and
the tokenizer-length check in 6.2 gates the largest of them. That ratio is the
honest measure of how much of this migration is box-gated.

## 9. Benchmark manifest & contract fixture — engineering context

`benchmark.json` and `fixtures/gemma4_26b_a4b_track.json` are pure
configuration (values, paths, commands, pins); their participant-facing
rationale lives in `docs/participant-contract.md`. This section is the
internal-engineering half of that same split: the benchd Rust source
citations, verification methodology, and derivations that do not belong in
either the manifest or the participant doc.

### 9.1 Composite scoring gate and the `pairs_per_cohort` pin history

Composite cohort scoring lives entirely in benchd's `measure_job` module
(`benchd/crates/benchctl/src/measure_job.rs`), landed by bench PR #182 ("Add
composite cohort scoring"). The relevant shape, current as of gitlink
`047e2183`:

- `PerCohort.composite: Option<CompositeCohortScore>` — the published
  composite. Its type shape is settled; its values are never populated in
  this build.
- `PerCohort.composite_absent_reason: Option<String>` — sealed whenever
  `composite` is `None` (always, today). The two are never both `None`: an
  absent composite always carries a reason, a present one carries none.
- `fn per_stream_aggregate_source() -> Result<std::convert::Infallible, String>`
  — the ONE gate a sequel PR's per-stream instrumentation plugs into. Its
  `Ok` type is `std::convert::Infallible`, which makes "this function can
  never succeed" a type-checked fact today, not a runtime posture that might
  silently change. It always refuses (there is no per-stream aggregate
  source yet — today's clock covers only the SHARED window across all
  streams, kept as an honest diagnostic on `CohortPhaseWindows`, never a
  composite input). `build_cohort_results` treats a refusal as "seal
  `composite: None`, `composite_absent_reason: Some(<this message>)`."
- `PerCohort.composite_scored_exponents: ScoredExponents` — sealed
  unconditionally on every batched run regardless of whether `composite` is
  populated, via `ScoredExponents::certify`, which bit-exact matches
  (`f64::to_bits`) the contract fixture's declared `scored_exponents`
  against the one ruled pair `{prefill_gain_exponent: 0.25,
  decode_gain_exponent: 0.75}`.
- Reaching a real per-stream aggregate source (spec step 4) requires spec
  step 3's box calibration to land first — tolerances are measured, not
  invented, and nothing may score on an unpinned tolerance. A real source
  will also need to reconcile engine-reported per-stream sub-clocks against
  benchd's standing "parent clock is the only trusted clock" doctrine
  (`bench_runner::timing`) — at minimum parent-window bounding and a
  sum-consistency check, not a bare trust of self-timed per-stream
  durations. The gate function does not attempt that reconciliation; it has
  nothing to reconcile yet.

**`pairs_per_cohort` pin history.** Three rulings, each superseding the one
before it. The value is currently **4**.

1. **Batch-8 brief D2** — the original default, 4.
2. **David, 2026-08-24: 2.** Verbatim "do 2" — "chose ~20-minute scored
   windows over 4-sample medians from the presented trade table". Landed in
   benchd as commit `bb1a6216655912b8a57967bb9cd45cff973a82df`
   ("pairs_per_cohort: 4 -> 2 (David ruling 2026-08-24)"), merged as bench PR
   #184 at `047e21833a66264310307e1cb86ae3a290b0fc27` on the
   `gemma4-26b-a4b-mlx-v1` release branch.
3. **David, 2026-08-26: 4.** Verbatim: "you run it using 4 pairs instead of 2
   of 8 batches". 8 prompts × 4 pairs is challenger-grade sample mass; the
   8/24 ruling is superseded, not reinterpreted, and the trade it made
   (shorter scored windows) is reversed. This is the current value in
   `benchmark.json`, `tools/lint-benchmark-manifest.py`'s
   `EXPECTED_SCORING_BY_TRACK` and both `--target-pairs` occurrences in
   `tools/gemma4-measure-and-score.sh`.

**The ruled-ahead-of-pin state recurs at each ruling, and it is load-bearing
that it fails LOUD.** Between a ruling and the pin advance that carries it,
the pinned benchd compiles the PREVIOUS target, and an official run declaring
the NEW one is refused before any GPU work — `run_cohort_measure_job` checks
`cfg.target_pairs != PAIRS_PER_COHORT_TARGET` and returns a named error. This
happened once already under the 8/24 ruling: the gitlink sat at `dfd801f9`,
which predated it and still compiled `PAIRS_PER_COHORT_TARGET = 4`, so a run
declaring `target_pairs=2` was refused until the advance to `047e2183`
resolved it (verified then by reading the checked-out submodule source
directly — `grep 'PAIRS_PER_COHORT_TARGET.*usize.*=' benchd/crates/benchctl/src/measure_job.rs`
returned `= 2` — and by containment,
`git -C benchd branch -r --contains 047e2183` returning exactly
`origin/gemma4-26b-a4b-mlx-v1`).

The 8/26 ruling is in that same state **as of this commit**: the benchd side
(`PAIRS_PER_COHORT_TARGET` 2 → 4) must merge and publish before the served
channel build compiles 4, so the served build
advance is a SEPARATE follow-up commit rather than part of this change. Until
it lands, an official-shaped invocation declaring `--target-pairs 4` is
refused at the pin, by name and pre-measurement. Verify the same way the 8/24
advance was verified — read the pinned source, do not assume — and record the
resolution here.

`minPairsPerCohort` moves with the target (now 4), and — since the
trusted-layer floor gate landed — it is **independently benchd-enforced, the
same way `pairsPerCohort` is**. benchd refuses an OFFICIAL batched cohort run
whose `min_pairs != PAIRS_PER_COHORT_TARGET`, by name, at the same pre-GPU
seam as the target refusal (`--local-dev` still explores other floors).

This corrects an earlier claim in this section. Until that gate landed, benchd's
only floor rule was the parse-time `target_pairs >= min_pairs`, and the ruled
floor rode entirely on the wrapper script's `--min-pairs` argv. A run declaring
min 2 / target 4 satisfied every trusted-side check, passed the target refusal
untouched, and then published a median over half the support the ruling
bought — with nothing downstream saying so. That is the gap the gate closes,
and it is why floor == target is now a property of the published median rather
than of the wrapper's command line.

The wrapper's own `--min-pairs 4` is kept as a belt-and-suspenders
**declaration** of the ruled floor: it states the value where an operator reads
it, and `tools/` is organizer-controlled and outside `editablePaths`, so a
submission cannot rewrite it. But it is no longer the thing doing the enforcing.

Note the division of labour, since the two are easy to conflate:
`tools/lint-benchmark-manifest.py` pins the VALUE of
`scoring.minPairsPerCohort` in `benchmark.json` against its registry; it does
not read the wrapper, so it would not catch a wrapper edited to `--min-pairs 2`
against a manifest still saying 4. **That** drift is caught at the pin, by
benchd's refusal.

Cross-checking `pairsPerCohort`/`PAIRS_PER_COHORT_TARGET` agreement against
the live `benchd` source is not implemented in
`tools/lint-benchmark-manifest.py` — CI runs that linter without a `benchd`
checkout (`--gitlink-targets report`), so a check binding this drift could
only ever run locally, for no CI-visible benefit. This section, plus the
linter's own `EXPECTED_SCORING_BY_TRACK` registry comment, is the load-bearing
instrument for that class of drift going forward.

### 9.2 Editable-surface byte budget arithmetic

`editableSurfaceByteBudget.maxTotalBytes` was raised from 3,000,000 (the
initial qwen-derived surface) to 4,404,587 in the same commit that added the
`ContinuousBatchingV2` surface to `editablePaths` (see §2 above /
`docs/participant-contract.md` §2 group 4). The number is computed, not
ratio-guessed, over the ENFORCED (non-exempt) surface specifically: the
at-rest 92-path editable surface totals 3,358,640 bytes, of which 2,629 bytes
are the exempt `mtp-head/README.md`, leaving 3,356,011 bytes of enforced
surface — already over the old 3.0 MB cap before any participant edit. The
cap is that enforced total plus a stated margin of exactly 1 MiB (1,048,576
bytes = 4× `maxGrowthBytes`, i.e. room for a handful of near-term merged
edits at the per-submission growth cap before the budget needs raising
again): `3,356,011 + 1,048,576 = 4,404,587`.

Two earlier derivations were corrected on review before landing on that
number: a ~40%-headroom-ratio version (matching the original
3.0 MB / 2,139,781 B pair) landed on 4,700,000, and a first
minimum-plus-margin pass double-counted the exempt `mtp-head/` bytes into the
"at-rest" figure (3,358,640 instead of the correct enforced-only 3,356,011),
landing on 4,407,216. Both are superseded by 4,404,587.

The cap was raised again on 2026-08-27, from 4,404,587 to 9,647,467 (David
ruling: add 5 MiB, 5,242,880 bytes). Promoted submissions had grown the
enforced at-rest surface on main to 3,367,645 bytes. That left 1,036,942
bytes of margin, under the stated 1 MiB minimum, so
`enforcedSurfaceStaysUnderTotalCap` failed on main and on every submission
PR. The margin at the raise is 6,279,822 bytes. The manifest declaration, the
trusted enforcer default, and the static-review shell fallback all moved in
the same commit. The 1 MiB minimum-margin assertion is unchanged.

`maxFileBytes` (524,288) is unchanged: the largest single file in the new
surface (`EngineLoopV2.swift`, 126,460 bytes) is well under the existing
512 KiB per-file cap, so only the total needed to move. The manifest's
numbers are verified equal to the trusted enforcer's own compiled-in
defaults (`Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift`:
`defaultMaxTotalBytes=9_647_467`, `defaultMaxFileBytes=524_288`,
`defaultMaxGrowthBytes=262_144`, `defaultExemptPathMaxBytes=512_000_000`,
`defaultExemptPathMaxFileBytes=100_000_000`) —
`tools/lint-benchmark-manifest.py` check 3b fails on manifest/enforcer drift,
so keeping the manifest in sync with the compiled defaults is load-bearing,
not cosmetic.

### 9.3 Engine wire-sha cross-check

benchd's `ENGINE_WIRE_V1_SHA256` (`benchd/crates/bench-runner/src/
wire_crosscheck.rs`) and this engine's own pin
(`Tests/MLXFastTests/EmitWireFixtureTests.swift`) must agree byte-for-byte —
both sides deserialize the same captured engine-wire fixture under their own
schema, and a mismatch there is a real cross-repo incoherence, not a
formatting difference. At gitlink `047e2183` both pin
`e79fd82920f6b0046ee08d3869834ca653979ff5d519965b5ec179a5db34247f`
(3005 bytes) — the 2026-08-23 per-stream-timing repin (line count stays 11;
the gate-on hello and the batched cohort begin/run lines each gain new
per-stream fields), confirmed stable across the `dfd801f9` → `047e2183`
gitlink advance, so no engine-side fixture repin was required for that
advance. Reproduce via engine
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
--force-resolved-versions --filter emitEngineWireFixture`.
