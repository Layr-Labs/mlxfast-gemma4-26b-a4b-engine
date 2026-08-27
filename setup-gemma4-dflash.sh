#!/usr/bin/env bash
# Provision the organizer-pinned Gemma 4 26B A4B DFlash drafter for the
# gemma4-26b-a4b-mlx-v1 DFlash arm, into ./dflash-head/.
#
# THIS SCRIPT IS OPT-IN AND NOTHING CALLS IT. That is deliberate and it is the
# whole reason the DFlash staging is a second script instead of another leg of
# ./setup.sh or a second delegation inside setup-gemma4-assistant.sh: the
# DFlash arm is still in bring-up -- docs/gemma4-port-notes.md section 4.3 says
# the BF16 MLX conversion is a separate work item and its inventory row still
# reads "not pinned here", which this script and its manifest are what change --
# so a failure to stage it must not be able to break an
# MTP-only setup. Because no other script invokes this one, that property is
# structural rather than a tolerated error path: `./setup.sh` and
# `./setup-gemma4-assistant.sh` cannot fail because of anything in here.
#
# Run it only if you are working the DFlash arm:
#
#   ./setup.sh && ./setup-gemma4-assistant.sh    # MTP arm -- unaffected by this file
#   ./setup-gemma4-dflash.sh                     # DFlash arm -- additionally
#
# WHERE THE OUTPUT IS CONSUMED. ./dflash-head/ is NOT merely a convention this
# script invents: it is the fixed, CWD-relative directory the runtime worker
# loads the drafter from, exactly as ./mtp-head/ is for the assistant. The name
# is `gemma4DFlashHeadDirectoryName` in
# Sources/MLXFastHarness/Gemma4A4BAssistantHead.swift, and
# `loadGemma4DFlashHeadIfStaged` reads that default with no argv flag to
# override it. Staging anywhere else leaves the worker's DFlash arm unbound.
#
# The standalone probe CLIs are the other consumer and they DO take an explicit
# path -- `mlxfast-swift dflash-benchmark|dflash-probe|dflash-reference
# --drafter PATH` (Sources/MLXFastCLI/main.swift) -- so the same staged tree
# serves both. Point --drafter at it for a manual probe:
#
#   .build/release/mlxfast-swift dflash-probe --drafter dflash-head --golden ...
#
# WHAT LOADS IT. A real `DFlashDraftModel` (Vendor/mlx-swift-lm
# Libraries/MLXSpeculative), bound through `DFlashTargetModel`. That matters
# for this script because it fixes WHICH BYTES are the right ones to stage:
# the drafter's `config.json` declares `architectures: ["DFlashDraftModel"]`,
# `block_size`, `num_target_layers` and a `dflash_config { target_layer_ids,
# mask_token_id }` block, and `DFlashConfiguration` decodes exactly that
# upstream z-lab schema. So the UPSTREAM published bytes this script pins are
# the loadable artifact, not a placeholder awaiting a repack. (An earlier
# revision of the port notes described the arm as loading through the
# `Gemma4AssistantDraftModel` / `Gemma4CBv2MTPDrafter` alias; that alias could
# never have decoded this config.json and was deleted when the real port
# landed. Do not restore that description here.)
#
# The loader globs `*.safetensors` + `config.json` in the directory, so the
# checked-in ./dflash-head/README.md is ignored by the load path, and the
# head tree digest (`computeGemma4AssistantHeadProvenance`) EXCLUDES a
# top-level README.md by rule. Staging alongside that file is therefore
# correct, and this script must not remove it: setup.sh writes only the files
# named in the manifest and never prunes the directory, so the README
# survives a stage. Before staging, a dflash-head/ carrying only the README is
# the "placeholder, no config.json" state -- capability absent, not an error.
#
# THAT SURVIVAL IS THE MANIFEST'S DOING, so keep it that way: fixtures/gemma4_dflash_drafter.sha256
# deliberately pins neither the upstream README.md nor .gitattributes, and
# adding a README.md record would make this stage overwrite the checked-in one.
# The assistant's fixture DOES pin an upstream README.md, which is why
# setup-gemma4-assistant.sh has to drop that record from the manifest it hands
# the downloader. Nothing here needs that treatment while the pin stays clean.
#
# COMMITTABILITY. dflash-head/ IS a benchmark.json `editablePaths` /
# `optionalEditablePaths` entry (the track's SECOND replaceable head, twin of
# mtp-head/), so a participant MAY ship an in-branch drafter by declaring
# `"source": "in_branch"` in spec-decoder-head.manifest.json. What .gitignore
# excludes is only the ORGANIZER-PINNED bytes this script stages -- 859 MB
# that are downloaded and hash-verified, never authored, and that carry no
# authority the pin does not already carry. The committed README.md is
# re-included so archiving an optional editable path still finds the
# directory. An in-branch drafter is opted in explicitly (`git add -f`).
#
# SIZE. Both layers of the 2 GiB per-head cap apply to what lands here: the
# declaration gate (`Gemma4MTPHeadDeclaration`, kind `.dflash`) and the loader's
# own `gemma4DFlashStagedHeadMaxBytes` over the bytes actually staged. The
# pinned drafter is ~859 MB, comfortably under both.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER, exactly as
# setup-gemma4-assistant.sh explains for the assistant: setup.sh's downloader
# is fully parameterised by environment (MLXFAST_REFERENCE_MODEL_REPO /
# _REVISION / _MANIFEST_PATH / _BASE_URL / _DIR / _COMPAT_LINK), so this script
# points those at the drafter and delegates. One downloader, one verification
# path, one set of stall and resume semantics -- and, critically, one place
# where sha256+byte verification is implemented.
#
# PIN PROVENANCE. fixtures/gemma4_dflash_drafter.sha256 pins config.json and
# model.safetensors by sha256 AND byte count; setup.sh verifies every
# downloaded byte against it. See that fixture's header for how the two records
# were obtained and why they are upstream bytes rather than an MLX repack.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="gemma4-26b-a4b-mlx-v1"

# z-lab/gemma-4-26B-A4B-it-DFlash, a 0.4B block-diffusion drafter (block_size
# 16) published alongside the target -- docs/gemma4-port-notes.md section 4.3.
# The revision is the full 40-hex form of the 77d42027 short sha that section
# cites; setup.sh's URL builder wants the whole thing.
DFLASH_MODEL_ID="${MLXFAST_GEMMA4_DFLASH_REPO:-z-lab/gemma-4-26B-A4B-it-DFlash}"
DFLASH_REVISION="${MLXFAST_GEMMA4_DFLASH_REVISION:-77d4202772dfe50b2396ec7bac9cfffc7b9e7057}"
DFLASH_MANIFEST="${MLXFAST_GEMMA4_DFLASH_MANIFEST_PATH:-${ROOT_DIR}/fixtures/gemma4_dflash_drafter.sha256}"
DFLASH_HEAD_DIR="${MLXFAST_GEMMA4_DFLASH_HEAD_DIR:-${ROOT_DIR}/dflash-head}"

DEFAULT_DFLASH_BASE_URL="https://huggingface.co/${DFLASH_MODEL_ID}/resolve/${DFLASH_REVISION}"
DFLASH_BASE_URL="${MLXFAST_GEMMA4_DFLASH_BASE_URL:-${DEFAULT_DFLASH_BASE_URL}}"
# An explicitly overridden primary has no implicit fallback (the rule setup.sh
# and setup-gemma4-assistant.sh both follow): a mirror the operator chose must
# not silently fall back to a source they did not.
if [[ -n "${MLXFAST_GEMMA4_DFLASH_FALLBACK_BASE_URL+x}" ]]; then
  DFLASH_FALLBACK_BASE_URL="${MLXFAST_GEMMA4_DFLASH_FALLBACK_BASE_URL}"
else
  DFLASH_FALLBACK_BASE_URL=""
fi

VERIFY_ONLY=0

usage() {
  cat <<EOF
Usage: ./setup-gemma4-dflash.sh [--verify-only]

Provision the organizer-pinned Gemma 4 26B A4B DFlash drafter into
./dflash-head/. The download is anonymous (the pinned repository is public),
resumable, and every byte is checked against the checked-in SHA256/size
manifest, because the download itself is ./setup.sh's downloader driven at
this artifact.

OPT-IN: the DFlash arm is still in bring-up. Nothing else runs this script,
so a failure here cannot break the MTP-only path
(./setup.sh && ./setup-gemma4-assistant.sh).

The TARGET is not provisioned here: it is the Gemma 4 26B A4B reference
checkpoint ./setup.sh downloads and verifies against
fixtures/reference_gemma4_26b_a4b_qat4bit.sha256. Run ./setup.sh first
(without MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1), then this script.

Pinned artifact:
  repository ${DFLASH_MODEL_ID}
  revision   ${DFLASH_REVISION}
  manifest   ${DFLASH_MANIFEST}

Staging path -- the fixed CWD-relative directory the runtime worker loads the
drafter from (gemma4DFlashHeadDirectoryName; there is no argv flag to point
it elsewhere), and also what you pass to the standalone probe CLIs:
  ${DFLASH_HEAD_DIR}

  .build/release/mlxfast-swift dflash-probe --drafter dflash-head --golden PATH

The checked-in dflash-head/README.md is preserved by a stage: only the files
named in the manifest are written, and the loader ignores a top-level README.

Overrides:
  MLXFAST_GEMMA4_DFLASH_REPO
  MLXFAST_GEMMA4_DFLASH_REVISION
  MLXFAST_GEMMA4_DFLASH_MANIFEST_PATH
  MLXFAST_GEMMA4_DFLASH_HEAD_DIR
  MLXFAST_GEMMA4_DFLASH_BASE_URL
  MLXFAST_GEMMA4_DFLASH_FALLBACK_BASE_URL

This command does not alter or replace the normal base-track reference cache,
and does not touch ./mtp-head/.
EOF
}

while (( "$#" > 0 )); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "setup-gemma4-dflash.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -x "${ROOT_DIR}/setup.sh" ]]; then
  echo "setup-gemma4-dflash.sh: ${ROOT_DIR}/setup.sh is missing or not executable;" >&2
  echo "setup-gemma4-dflash.sh: this script delegates its downloader rather than reimplementing it" >&2
  exit 1
fi

if [[ ! -s "${DFLASH_MANIFEST}" ]]; then
  echo "setup-gemma4-dflash.sh: pinned byte manifest missing or empty: ${DFLASH_MANIFEST}" >&2
  echo "setup-gemma4-dflash.sh: refusing to download an unpinned drafter checkpoint" >&2
  exit 1
fi

# --- delegate ---------------------------------------------------------------
# setup.sh is driven at the drafter instead of the backbone.
# MLXFAST_REFERENCE_COMPAT_LINK is set equal to REFERENCE_DIR so
# ensure_reference_compat_link's early-exit ("reference_dir == link_absolute")
# makes the compat-link step an immediate no-op -- the same guard
# setup-gemma4-assistant.sh relies on, and for the same reason: left unset,
# DEFAULT_REFERENCE_DIR would resolve to the BACKBONE's compat link and this
# run would repoint it at an 859 MB drafter.
delegated_env=(
  MLXFAST_REFERENCE_MODEL_REPO="${DFLASH_MODEL_ID}"
  MLXFAST_REFERENCE_REVISION="${DFLASH_REVISION}"
  MLXFAST_REFERENCE_MANIFEST_PATH="${DFLASH_MANIFEST}"
  MLXFAST_REFERENCE_BASE_URL="${DFLASH_BASE_URL}"
  MLXFAST_REFERENCE_FALLBACK_BASE_URL="${DFLASH_FALLBACK_BASE_URL}"
  MLXFAST_REFERENCE_DIR="${DFLASH_HEAD_DIR}"
  MLXFAST_REFERENCE_COMPAT_LINK="${DFLASH_HEAD_DIR}"
  # The drafter is ~860 MB, not ~15 GB: the backbone's free-space floor would
  # refuse on machines that can comfortably hold it.
  MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_GEMMA4_DFLASH_MIN_FREE_GIB:-4}"
  # The closing summary must not tell this reader to transform REFERENCE_DIR
  # into weights/: REFERENCE_DIR is the drafter for the delegated run.
  MLXFAST_SETUP_SUMMARY_ROLE=mtp-head
  # Label the delegated run's output with the command the reader typed, the
  # same reason setup-gemma4-assistant.sh does: the downloader is a subprocess
  # here, and a wall of "setup.sh:" lines names a command this run never made.
  MLXFAST_SETUP_LOG_LABEL=setup-gemma4-dflash.sh
  # Reuse the Swift products ./setup.sh already built. Fails open in setup.sh:
  # if a product is missing (a standalone run on a fresh clone) the build
  # happens anyway.
  MLXFAST_SKIP_SWIFT_BUILD=1
  # ./setup.sh owns global tool state and the metallib; do not mutate either a
  # second time from here.
  MLXFAST_SKIP_HOMEBREW_INSTALL=1
  MLXFAST_SKIP_CMAKE_INSTALL=1
  MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1
  MLXFAST_SKIP_MACMON_INSTALL=1
  MLXFAST_SKIP_MLX_METALLIB=1
)

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  if [[ ! -d "${DFLASH_HEAD_DIR}" ]]; then
    echo "setup-gemma4-dflash.sh: no drafter cache at ${DFLASH_HEAD_DIR}; run ./setup-gemma4-dflash.sh first" >&2
    exit 1
  fi
  delegated_env+=(MLXFAST_SKIP_WEIGHTS_DOWNLOAD=0)
  echo "setup-gemma4-dflash.sh: verifying the pinned DFlash drafter at ${DFLASH_HEAD_DIR}" >&2
else
  echo "setup-gemma4-dflash.sh: provisioning the pinned DFlash drafter into ${DFLASH_HEAD_DIR}" >&2
fi

env "${delegated_env[@]}" "${ROOT_DIR}/setup.sh"

if [[ ! -s "${DFLASH_HEAD_DIR}/config.json" ]]; then
  echo "setup-gemma4-dflash.sh: the delegated run left no ${DFLASH_HEAD_DIR}/config.json; the drafter is not usable" >&2
  exit 1
fi

echo "setup-gemma4-dflash.sh: DFlash drafter ready at ${DFLASH_HEAD_DIR} (${DFLASH_MODEL_ID} @ ${DFLASH_REVISION}, track ${TRACK_ID})"
echo "setup-gemma4-dflash.sh: the runtime worker binds it from this CWD-relative default; the probe CLIs take it explicitly, e.g. --drafter ${DFLASH_HEAD_DIR}"
