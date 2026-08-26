#!/usr/bin/env bash
# Provision the organizer-pinned Gemma 4 26B A4B assistant (spec-decode
# drafter) checkpoint for the gemma4-26b-a4b-mlx-v1 MTP arm.
#
# REPLACES setup-qwen-mtp.sh (lane/gemma4-setup-repoint, 2026-08-24), which
# staged the retired Qwen 3.8 MTP head and is Qwen-era dead code per
# docs/gemma4-port-notes.md section 8.2a item 2: the mtp-runtime-worker verb
# and its --mtp-head flag both refuse post-harness-port, so nothing consumes
# that script's cache-dir-plus-compat-symlink output any more.
#
# THE CONSUMPTION MODEL CHANGED, not just the pinned identity. The Qwen head
# was read through an explicit `--mtp-head <dir>` CLI flag pointed at a cache
# directory. The Gemma 4 assistant is read by
# Sources/MLXFastHarness/Gemma4A4BAssistantHead.swift
# (`loadGemma4AssistantHeadIfStaged`), which hardcodes a CWD-relative
# directory name -- "mtp-head", no flag, no environment override -- because
# that is also the fixed location the TRUSTED runner stages a remote
# `mtp-head.manifest.json` declaration into pre-sandbox
# (Sources/MLXFastTrustedHarness/Gemma4MTPHeadDeclaration.swift). So this
# script provisions the SAME directory the checked-in mtp-head/README.md
# already documents as the in-branch declaration slot: ./mtp-head/, at the
# repository root, not a cache path with a compatibility symlink. There is
# nothing to "compat-link" any more; the one real directory the harness reads
# is the one this script writes.
#
# This script provisions the ASSISTANT and NOTHING else. The track's TARGET
# is the same Gemma 4 26B A4B checkpoint ./setup.sh already pins, downloads
# and verifies (fixtures/reference_gemma4_26b_a4b_qat4bit.sha256) -- so there
# is no second target download here and no way for this script to switch the
# base challenge onto a different checkpoint.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER, unchanged from the script it
# replaces: setup.sh's downloader is fully parameterised by environment
# (MLXFAST_REFERENCE_MODEL_REPO / _REVISION / _MANIFEST_PATH / _BASE_URL /
# _CACHE_DIR / _DIR / _COMPAT_LINK), so this script points those at the
# assistant and delegates. One downloader, one verification path, one set of
# stall and resume semantics.
#
# The delegated run does NOT rebuild the Swift binaries: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair of products is reused. That
# rebuild was never a no-op -- SwiftPM still relinks both products, ~25
# seconds on the ordinary `./setup.sh && ./setup-gemma4-assistant.sh` path --
# and nothing about them can have changed between the two commands. The knob
# fails open: on a fresh clone where ./setup.sh has not run, setup.sh builds
# anyway rather than leaving this script without a harness. The
# tool-installing legs (Homebrew, cmake, the Metal toolchain, macmon) and the
# metallib build are skipped for the same reason in the other direction:
# ./setup.sh owns those and this script must not mutate global state a
# second time.
#
# STATUS: provisioning only. The MTP/speculative-decode apparatus itself
# (drafting, verify, accept-walk, KV rollback) has not landed for Gemma 4 in
# this repository yet (fixtures/gemma4_26b_a4b_track.json `protocol.status`).
# Staging the assistant here makes it loadable by
# `loadGemma4AssistantHeadIfStaged` for local development; it does not by
# itself enable a ranked spec-decode run.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="gemma4-26b-a4b-mlx-v1"

# The assistant is a SEPARATELY pinned artifact, ruled 2026-08-22
# (docs/gemma4-port-notes.md section 4.0, superseding an earlier 12B pin that
# failed three independent geometry checks against this 26B-A4B target).
# CONFIRMED 2026-08-23 via the Hugging Face API against the repository's own
# `sha`, and cross-checked against mtp-head.manifest.json's `source_url`,
# which names the same repository and revision for the trusted runner's own
# remote-fetch path.
ASSISTANT_MODEL_ID="${MLXFAST_GEMMA4_ASSISTANT_REPO:-mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit}"
ASSISTANT_REVISION="${MLXFAST_GEMMA4_ASSISTANT_REVISION:-bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c}"
ASSISTANT_MANIFEST="${MLXFAST_GEMMA4_ASSISTANT_MANIFEST_PATH:-${ROOT_DIR}/fixtures/gemma4_assistant.sha256}"

# THE STAGING DIRECTORY IS THE CONSUMED DIRECTORY. Unlike the Qwen-era
# cache-root-plus-compat-symlink layout, there is no indirection here: this
# IS where Gemma4A4BAssistantHead.swift's `gemma4AssistantHeadDirectoryName`
# ("mtp-head", CWD-relative, not env-overridable) reads from. Overriding
# MLXFAST_GEMMA4_ASSISTANT_HEAD_DIR stages the download somewhere else for
# inspection or caching, but only a checkout at ./mtp-head itself (the
# default) is what the runtime worker actually loads -- move or symlink it
# into place by hand if you override this.
ASSISTANT_HEAD_DIR="${MLXFAST_GEMMA4_ASSISTANT_HEAD_DIR:-${ROOT_DIR}/mtp-head}"

DEFAULT_ASSISTANT_BASE_URL="https://huggingface.co/${ASSISTANT_MODEL_ID}/resolve/${ASSISTANT_REVISION}"
ASSISTANT_BASE_URL="${MLXFAST_GEMMA4_ASSISTANT_BASE_URL:-${DEFAULT_ASSISTANT_BASE_URL}}"
# An explicitly overridden primary has no implicit fallback (same rule
# setup.sh follows for the backbone): a mirror the operator chose must not
# silently fall back to a source they did not.
if [[ -n "${MLXFAST_GEMMA4_ASSISTANT_FALLBACK_BASE_URL+x}" ]]; then
  ASSISTANT_FALLBACK_BASE_URL="${MLXFAST_GEMMA4_ASSISTANT_FALLBACK_BASE_URL}"
else
  ASSISTANT_FALLBACK_BASE_URL=""
fi

VERIFY_ONLY=0

usage() {
  cat <<EOF
Usage: ./setup-gemma4-assistant.sh [--verify-only]

Provision the organizer-pinned Gemma 4 26B A4B assistant (spec-decode
drafter) checkpoint into ./mtp-head/. The download is anonymous (the pinned
repository is public), resumable, and every byte is checked against the
checked-in SHA256/size manifest, because the download itself is
./setup.sh's downloader driven at this artifact.

The TARGET is not provisioned here: it is the Gemma 4 26B A4B reference
checkpoint ./setup.sh downloads and verifies against
fixtures/reference_gemma4_26b_a4b_qat4bit.sha256. Run ./setup.sh first
(without MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1), then this script.

Pinned artifact:
  repository ${ASSISTANT_MODEL_ID}
  revision   ${ASSISTANT_REVISION}
  manifest   ${ASSISTANT_MANIFEST}

Staging path (this IS the directory the runtime worker reads -- see
Sources/MLXFastHarness/Gemma4A4BAssistantHead.swift):
  ${ASSISTANT_HEAD_DIR}

The checked-in mtp-head/README.md is preserved by a stage: the upstream
repository's own README.md is the one pinned record this script excludes from
the download, because the staging directory is a checked-in repository
directory. Nothing that loads or verifies the head reads a top-level README.md.

Overrides:
  MLXFAST_GEMMA4_ASSISTANT_REPO
  MLXFAST_GEMMA4_ASSISTANT_REVISION
  MLXFAST_GEMMA4_ASSISTANT_MANIFEST_PATH
  MLXFAST_GEMMA4_ASSISTANT_HEAD_DIR
  MLXFAST_GEMMA4_ASSISTANT_BASE_URL
  MLXFAST_GEMMA4_ASSISTANT_FALLBACK_BASE_URL

This command does not alter or replace the normal base-track reference cache.
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
      echo "setup-gemma4-assistant.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -x "${ROOT_DIR}/setup.sh" ]]; then
  echo "setup-gemma4-assistant.sh: ${ROOT_DIR}/setup.sh is missing or not executable;" >&2
  echo "setup-gemma4-assistant.sh: this script delegates its downloader rather than reimplementing it" >&2
  exit 1
fi

if [[ ! -s "${ASSISTANT_MANIFEST}" ]]; then
  echo "setup-gemma4-assistant.sh: pinned byte manifest missing or empty: ${ASSISTANT_MANIFEST}" >&2
  echo "setup-gemma4-assistant.sh: refusing to download an unpinned assistant checkpoint" >&2
  exit 1
fi

# --- the staged manifest ----------------------------------------------------
# THE STAGING DIRECTORY IS A CHECKED-IN REPOSITORY DIRECTORY, which is what
# makes this different from every other artifact setup.sh downloads. ./mtp-head/
# is not a cache path: it carries a tracked mtp-head/README.md -- the
# archive-mechanism placeholder that keeps an `optionalEditablePaths` entry
# present on disk so `yukon submit` can tar it -- and the upstream assistant
# repository happens to publish a README.md of its own. fixtures/gemma4_assistant.sha256
# pins that upstream file like every other published byte, so the delegated run
# wrote Hugging Face's 1957-byte model card straight over the repository's
# 2629-byte doc and left the tree dirty after a documented setup. That was
# observed end to end on box 3.
#
# EXCLUDE ON WRITE rather than restore afterwards. Restoring the tracked bytes
# after the delegated run would leave setup.sh's own `README.md.complete`
# verification marker describing a file that no longer matches it, so the very
# next `--verify-only` would refuse with "reference file README.md changed after
# download verification". Never fetching the record is the only form of the fix
# that stays consistent with the downloader's markers and cache lock.
#
# SAFE BY CONSTRUCTION, not by luck. The staged head is loaded by
# `Gemma4A4BAssistantHead.swift` from config.json plus *.safetensors, and the
# head tree digest (`computeGemma4AssistantHeadProvenance`) EXCLUDES a top-level
# README.md by rule -- both stated in mtp-head/README.md itself. Nothing that
# loads, verifies or scores the head can observe which README.md sits there.
# setup.sh writes only the files its selected manifest names and never prunes
# the directory, so dropping the record removes the file from the download list,
# from the metadata verification pass and from the cache lock's inventory in one
# move, exactly as fixtures/gemma4_dflash_drafter.sha256 achieves for
# ./dflash-head/ by not pinning its upstream README in the first place.
#
# The fixture is NOT edited here: the checked-in manifest stays the full record
# of the published tree (8 records, 268325817 bytes, as docs and README cite),
# and what is dropped is dropped by the stager that owns the collision, at the
# one place that knows the destination is a repository directory.
ASSISTANT_STAGED_MANIFEST=""
cleanup_staged_manifest() {
  if [[ -n "${ASSISTANT_STAGED_MANIFEST}" ]]; then
    rm -f "${ASSISTANT_STAGED_MANIFEST}"
    ASSISTANT_STAGED_MANIFEST=""
  fi
}
trap cleanup_staged_manifest EXIT

if ! ASSISTANT_STAGED_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/mlxfast-gemma4-assistant-manifest.XXXXXX")"; then
  echo "setup-gemma4-assistant.sh: could not create a temporary manifest" >&2
  exit 1
fi

{
  echo "# DERIVED AT RUN TIME by setup-gemma4-assistant.sh from"
  echo "#   ${ASSISTANT_MANIFEST}"
  echo "# with the top-level README.md record removed, because the staging"
  echo "# directory (${ASSISTANT_HEAD_DIR}) is a checked-in repository"
  echo "# directory whose own README.md must survive a stage. Nothing that loads"
  echo "# or verifies the head reads a top-level README.md. Every other record is"
  echo "# copied through byte for byte and is verified by setup.sh as usual."
  echo "#"
  echo "# Format: <sha256> <byte_count> <relative_path>"
  # Comment and blank lines are dropped rather than copied: setup.sh ignores
  # them, and carrying the source header over would leave its RECORDS/BYTES
  # pins describing a record set this file no longer has. Malformed records are
  # copied through untouched so setup.sh's own validation stays the single
  # place that rules on manifest shape.
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    $3 == "README.md" { next }
    { print }
  ' "${ASSISTANT_MANIFEST}"
} > "${ASSISTANT_STAGED_MANIFEST}"

ASSISTANT_STAGED_RECORDS="$(
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { count += 1 }
    END { print count + 0 }
  ' "${ASSISTANT_STAGED_MANIFEST}"
)"
if [[ "${ASSISTANT_STAGED_RECORDS}" -lt 1 ]]; then
  echo "setup-gemma4-assistant.sh: ${ASSISTANT_MANIFEST} names no records other than README.md;" >&2
  echo "setup-gemma4-assistant.sh: refusing to provision an empty assistant checkpoint" >&2
  exit 1
fi

# --- delegate ---------------------------------------------------------------
# setup.sh is driven at the assistant instead of the backbone.
# MLXFAST_REFERENCE_COMPAT_LINK is set equal to REFERENCE_DIR so
# ensure_reference_compat_link's early-exit ("reference_dir == link_absolute")
# makes the compat-link step an immediate no-op: there is no separate
# canonical-path-plus-symlink pair here, only the one real directory
# Gemma4A4BAssistantHead.swift reads, and the delegated run must not create or
# repoint any symlink -- least of all the backbone's own compat link at
# reference_weights/gemma4-26b-a4b-qat4bit, which DEFAULT_REFERENCE_DIR would
# resolve to if this were left unset.
delegated_env=(
  MLXFAST_REFERENCE_MODEL_REPO="${ASSISTANT_MODEL_ID}"
  MLXFAST_REFERENCE_REVISION="${ASSISTANT_REVISION}"
  # The DERIVED manifest, not the fixture: see "the staged manifest" above for
  # why the upstream README.md record must not be written into ./mtp-head/.
  MLXFAST_REFERENCE_MANIFEST_PATH="${ASSISTANT_STAGED_MANIFEST}"
  MLXFAST_REFERENCE_BASE_URL="${ASSISTANT_BASE_URL}"
  MLXFAST_REFERENCE_FALLBACK_BASE_URL="${ASSISTANT_FALLBACK_BASE_URL}"
  MLXFAST_REFERENCE_DIR="${ASSISTANT_HEAD_DIR}"
  MLXFAST_REFERENCE_COMPAT_LINK="${ASSISTANT_HEAD_DIR}"
  # The assistant is ~256 MB, not ~15 GB: the backbone's free-space floor
  # would refuse on machines that can comfortably hold it.
  MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_GEMMA4_ASSISTANT_MIN_FREE_GIB:-2}"
  # The closing summary must not tell this reader to transform REFERENCE_DIR
  # into weights/: REFERENCE_DIR is the assistant for the delegated run, and
  # transforming an 8-tensor drafter over the target weights is exactly the
  # wrong next step. Prose only -- nothing about the download or verification
  # changes.
  MLXFAST_SETUP_SUMMARY_ROLE=mtp-head
  # Label the delegated run's output with the command the reader typed. The
  # downloader is a subprocess here; a transcript of "setup.sh:" lines under
  # ./setup-gemma4-assistant.sh names a command this invocation never ran.
  MLXFAST_SETUP_LOG_LABEL=setup-gemma4-assistant.sh
  # Reuse the Swift products ./setup.sh already built instead of relinking
  # both of them for a download. Fails open in setup.sh: if a product is
  # missing (a standalone run on a fresh clone) the build happens anyway.
  MLXFAST_SKIP_SWIFT_BUILD=1
  # ./setup.sh owns global tool state and the metallib; do not mutate either
  # a second time from here.
  MLXFAST_SKIP_HOMEBREW_INSTALL=1
  MLXFAST_SKIP_CMAKE_INSTALL=1
  MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1
  MLXFAST_SKIP_MACMON_INSTALL=1
  MLXFAST_SKIP_MLX_METALLIB=1
)

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  if [[ ! -d "${ASSISTANT_HEAD_DIR}" ]]; then
    echo "setup-gemma4-assistant.sh: no assistant cache at ${ASSISTANT_HEAD_DIR}; run ./setup-gemma4-assistant.sh first" >&2
    exit 1
  fi
  delegated_env+=(MLXFAST_SKIP_WEIGHTS_DOWNLOAD=0)
  echo "setup-gemma4-assistant.sh: verifying the pinned assistant checkpoint at ${ASSISTANT_HEAD_DIR}" >&2
else
  echo "setup-gemma4-assistant.sh: provisioning the pinned assistant checkpoint into ${ASSISTANT_HEAD_DIR}" >&2
fi

env "${delegated_env[@]}" "${ROOT_DIR}/setup.sh"

if [[ ! -s "${ASSISTANT_HEAD_DIR}/config.json" ]]; then
  echo "setup-gemma4-assistant.sh: the delegated run left no ${ASSISTANT_HEAD_DIR}/config.json; the assistant is not usable" >&2
  exit 1
fi

echo "setup-gemma4-assistant.sh: assistant ready at ${ASSISTANT_HEAD_DIR} (${ASSISTANT_MODEL_ID} @ ${ASSISTANT_REVISION}, track ${TRACK_ID})"
