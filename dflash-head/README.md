# dflash-head/ — the organizer's staged DFlash drafter weights

This directory holds the organizer-pinned DFlash drafter. It is the twin of
`mtp-head/` and it is **not** an editable path. A submission carries nothing
here.

`setup-gemma4-dflash.sh` stages the pinned drafter into this directory and
verifies every downloaded byte against `fixtures/gemma4_dflash_drafter.sha256`.
The staged bytes are gitignored. The pin is `z-lab/gemma-4-26B-A4B-it-DFlash` at
revision `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`.

## Custom drafter weights are not accepted

This is the David ruling of 2026-08-26. It replaces the earlier
bring-your-own-head design.

You may re-quantize this drafter. You may not replace it, and you may not upload
drafter weights of your own.

Three things enforce that:

1. `dflash-head` is not in `benchmark.json` `editablePaths`. A submission that
   carries a file here is refused by
   `.github/scripts/enforce-modifiable-surface.sh`, which names the file and
   says it is outside the modifiable surface. The overlay never copies it, and
   the benchmarker's write-divergence gate refuses it as content that diverges
   from the trusted baseline outside the editable surface.
2. `../dflash-head.manifest.json` accepts `"source": "pinned"` only.
   `"source": "in_branch"` and `"source": "remote"` are refused by name.
3. A re-quantization happens on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission.

## How to declare a re-quantization

`../dflash-head.manifest.json` stays editable. It is the declaration surface.
Keep `"source": "pinned"`. State the `bytes` you expect the staged artifact to
have, if you want the record. A declaration may lower `max_bytes` and may not
raise it above the 2 GiB track cap, which is the SAME cap the MTP head gets, per
the 2026-08-25 both-heads-one-cap ruling.

A re-quantized drafter loads from its own `config.json`. A `quantization` block
there selects which modules load quantized and at what geometry. The block holds
`group_size` (positive, at most 65536), `bits` between 2 and 8, and optional
per-layer overrides (at most 8192 entries).

The weights and the declaration must agree. The loader refuses a drafter that
declares a quantization its tensors do not carry. It also refuses a drafter that
ships packed tensors it does not declare. It never loads such a drafter at full
precision instead. See `docs/participant-contract.md` section 3.4.

## How to produce a re-quantization

A re-quantization happens ON LOAD, in memory. Nothing in this directory
changes.

The loader for this drafter is `DFlashDraftModel.load(from:bindTo:)` in
`Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift`, which is
an editable path. It already calls `quantize(model:)` through
`applyDeclaredQuantization` while it binds the checkpoint. That call is the
seam. Change the geometry it selects.

Do not write into this directory. The ranked worker runs under a sandbox that
denies file writes, and the benchmarker refuses a head tree that changed.

The target model is not in scope. See `docs/participant-contract.md` section
4.4.

> **NOTE — this two-way check is the DFlash loader's, not the MTP loader's.**
> The MTP head loader does neither check. Section 3.4 of the participant
> contract states the difference.

## Why this README is checked in

It documents what the organizer stages here, and it keeps the directory present
in a fresh clone.

The head tree digest excludes a top-level `README.md`, so this file is invisible
to head verification.

## The tree digest rule

This rule is identical to the rule in `mtp-head/README.md`, because the same
`computeGemma4AssistantHeadProvenance` seals both.

The digest is a SHA-256 over a concatenation. The concatenation holds one
`"<hex file sha256>  <relative path>\n"` entry for every regular file in the
tree, except a top-level `README.md`. The entries are in `LC_ALL=C` sorted
relative-path order. `bytes` is the byte total of the same file set.

Run this equivalent shell inside the head directory:

```sh
find . -type f ! -name README.md \
  | sed 's|^\./||' \
  | LC_ALL=C sort \
  | while read -r f; do
      printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "$f"
    done | shasum -a 256
```

The harness computes this digest and reports it. It does not compare it against
the organizer's pinned digests. `docs/participant-contract.md` section 4.3
states that limit plainly.

A drafter only PROPOSES tokens. The organizer-pinned target decides every
emitted token. The baseline leg of every ranked pair runs the pinned head.
