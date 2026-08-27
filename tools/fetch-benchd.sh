#!/usr/bin/env bash
#
# fetch-benchd.sh -- resolve the `benchctl` measurement harness from its release
# channel.
#
# THE PIN IS REMOVED (David ruling 2026-08-27). benchd used to be frozen here by
# ./benchd.pin ({branch, commit, sha256, bytes}), which made every benchd bug
# fix require an engine-repo pin-advance commit before it could reach a box --
# the depth/echo fix sat merged in the bench repo while the served pin still
# carried the bug. That coupling is gone: this repository names only the CHANNEL
# (the bench repo + branch), and the channel's tip is what runs. Fixing a
# measurement bug is now: merge it bench-side, republish dist, done -- zero
# engine commits.
#
# WHAT REPLACES THE PIN'S GUARANTEES, stated honestly:
#
#   * INTEGRITY (the bytes are what the organizer published): the dist channel
#     publishes `benchctl.manifest.json` next to the binary ({branch,
#     source_commit, sha256, bytes}, written by the bench repo's
#     scripts/build-dist.sh). The binary is verified against THAT manifest --
#     fetched from the same organizer-controlled channel -- and nothing
#     unverified is ever installed or returned. What this no longer defends
#     against is the channel itself moving, which is the point: the channel is
#     the organizer's bench repo, participants cannot write to it, and its tip
#     is now the intended source of truth.
#   * PROVENANCE (knowing which benchd measured a run): recorded, not pinned.
#     The resolved {branch, source_commit, sha256} is logged loudly on every
#     resolve and the manifest is installed beside the binary
#     (benchd-bin/benchctl.manifest.json), so any run's harness identity can be
#     read off the box afterwards.
#   * SUBMISSION-PROOFNESS: unchanged. This script and the channel constants
#     live under tools/, outside editablePaths, and the modifiable-surface
#     guard still forbids `benchd.pin`/`benchd-bin` spellings in submissions,
#     so a submission can neither redirect the fetch nor resurrect a pin.
#
# WHAT IT DOES, in order:
#   1. If benchd-bin/benchctl AND benchd-bin/benchctl.manifest.json exist and
#      agree (sha256 + bytes), use them, never touching the network -- the
#      OFFLINE path: the ranked box gets both files placed together. A binary
#      with no manifest, or a pair that disagrees, refuses loudly (a bare
#      unattributable binary is exactly what this script must never run).
#      Set BENCHD_REFRESH=1 to discard the local pair and re-resolve the
#      channel tip.
#   2. Otherwise obtain the PAIR -- from BENCHD_DIST_LOCAL (file/dir, for
#      air-gapped boxes) or by downloading manifest-then-binary from the
#      channel -- verify the binary against its manifest, and install both.
#
# Prints the absolute path of the verified binary on STDOUT (diagnostics go to
# stderr), so callers can do:  BENCHCTL="$(./tools/fetch-benchd.sh)"
#
# Env:
#   BENCHD_BRANCH       channel branch. Default: gemma4-26b-a4b-mlx-v1 (this
#                       track's bench branch).
#   BENCHD_BIN_DIR      install directory. Default: <repo>/benchd-bin
#                       (gitignored; a fetched artifact, not repository content).
#   BENCHD_REFRESH      set to 1 to discard an already-installed pair and
#                       re-resolve the channel tip.
#   BENCHD_DIST_LOCAL   path to an already-obtained dist (a directory holding
#                       benchctl + benchctl.manifest.json, or the benchctl file
#                       with the manifest beside it). Verified, never trusted
#                       bare.
#   BENCHD_DIST_BASE_URL
#                       raw host + repo prefix. Default:
#                       https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench
#   BENCHD_DIST_TOKEN / GITHUB_TOKEN
#                       bearer token for a private channel repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${BENCHD_BRANCH:-gemma4-26b-a4b-mlx-v1}"
DEST_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
DEST="${DEST_DIR}/benchctl"
DEST_MANIFEST="${DEST_DIR}/benchctl.manifest.json"
BASE_URL="${BENCHD_DIST_BASE_URL:-https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench}"

die() {
  echo "fetch-benchd.sh: $*" >&2
  exit 1
}

command -v shasum >/dev/null 2>&1 || die "shasum is required to verify benchctl."

# -- manifest -----------------------------------------------------------------
# benchctl.manifest.json is values-only JSON, one key per line (build-dist.sh).
# Parsed with sed rather than jq/python3 so the OFFLINE path on the ranked box
# needs nothing but a shell and shasum.
manifest_field() {
  sed -n "s/^[[:space:]]*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",]*\)\"\{0,1\}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "$1"
}

# Load + validate a manifest file; sets MF_BRANCH/MF_COMMIT/MF_SHA256/MF_BYTES.
# An unparseable manifest must refuse, not degrade: empty fields would make
# every comparison below trivially "match".
load_manifest() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  MF_BRANCH="$(manifest_field "${path}" branch)"
  MF_COMMIT="$(manifest_field "${path}" source_commit)"
  MF_SHA256="$(manifest_field "${path}" sha256)"
  MF_BYTES="$(manifest_field "${path}" bytes)"
  if [[ "${#MF_SHA256}" -ne 64 || -n "${MF_SHA256//[0-9a-f]/}" ]]; then
    echo "fetch-benchd.sh:   manifest sha256 is not 64 lowercase hex characters: '${MF_SHA256}' (${path})" >&2
    return 1
  fi
  if [[ -z "${MF_BYTES}" || -n "${MF_BYTES//[0-9]/}" ]]; then
    echo "fetch-benchd.sh:   manifest bytes is not a positive integer: '${MF_BYTES}' (${path})" >&2
    return 1
  fi
  if [[ "${MF_BRANCH}" != "${BRANCH}" ]]; then
    echo "fetch-benchd.sh:   manifest names branch '${MF_BRANCH}', expected '${BRANCH}' -- wrong channel (${path})" >&2
    return 1
  fi
  return 0
}

# Verify a binary against the LOADED manifest. Bytes first (cheap, and a length
# mismatch is already disqualifying), then the digest. Reports on stderr so
# callers see WHICH half failed.
matches_manifest() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${MF_BYTES}" ]]; then
    echo "fetch-benchd.sh:   byte count mismatch: manifest=${MF_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${MF_SHA256}" ]]; then
    echo "fetch-benchd.sh:   sha256 mismatch: manifest=${MF_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

announce() {
  echo "fetch-benchd.sh: benchctl identity: branch=${MF_BRANCH} source_commit=${MF_COMMIT} sha256=${MF_SHA256} bytes=${MF_BYTES}" >&2
}

# -- 1. already in place (the offline path) -----------------------------------
if [[ "${BENCHD_REFRESH:-0}" == "1" && -f "${DEST}" ]]; then
  echo "fetch-benchd.sh: BENCHD_REFRESH=1 -- discarding the installed pair to re-resolve the channel tip" >&2
  rm -f "${DEST}" "${DEST_MANIFEST}"
fi

if [[ -f "${DEST}" ]]; then
  load_manifest "${DEST_MANIFEST}" \
    || die "${DEST} exists but ${DEST_MANIFEST} is missing or malformed; a binary with no manifest is unattributable and will not be run. Place the channel's benchctl.manifest.json beside it, or delete the binary and re-run."
  matches_manifest "${DEST}" \
    || die "${DEST} does NOT match its manifest (see the mismatch above); refusing to run or replace it. Delete the pair deliberately, then re-run."
  chmod 755 "${DEST}"
  announce
  printf '%s\n' "${DEST}"
  exit 0
fi

# -- 2. obtain the pair -------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGED="${TMP_DIR}/benchctl"
STAGED_MANIFEST="${TMP_DIR}/benchctl.manifest.json"

if [[ -n "${BENCHD_DIST_LOCAL:-}" ]]; then
  src="${BENCHD_DIST_LOCAL}"
  [[ -d "${src}" ]] && src="${src}/benchctl"
  [[ -f "${src}" ]] || die "BENCHD_DIST_LOCAL does not resolve to a benchctl file: ${src}"
  src_manifest="$(dirname "${src}")/benchctl.manifest.json"
  [[ -f "${src_manifest}" ]] || die "BENCHD_DIST_LOCAL has no benchctl.manifest.json beside the binary (${src_manifest}); an unattributable binary will not be installed."
  echo "fetch-benchd.sh: taking benchctl from ${src}" >&2
  cp "${src}" "${STAGED}"
  cp "${src_manifest}" "${STAGED_MANIFEST}"
  load_manifest "${STAGED_MANIFEST}" \
    || die "BENCHD_DIST_LOCAL manifest is malformed or names the wrong channel; refusing."
  # One named source, so a mismatch is a hard refusal: silently continuing past
  # the file the operator explicitly pointed at would hide the real problem.
  matches_manifest "${STAGED}" \
    || die "BENCHD_DIST_LOCAL (${src}) does NOT match its manifest (see the mismatch above); refusing to install or run it."
else
  command -v curl >/dev/null 2>&1 || die "curl is required to download benchctl (or set BENCHD_DIST_LOCAL)."

  # The channel: dist/ on the bench branch tip. The manifest is fetched FIRST,
  # then the binary from the same directory, and the binary must match the
  # manifest -- so a half-updated publish (one file at the old build) refuses
  # rather than installing a pair that disagrees. The `refs/heads/` prefix is
  # required, not cosmetic: branch names contain slashes, and without the
  # explicit ref namespace raw.githubusercontent cannot tell where the ref ends
  # and the path begins.
  DIST_DIR_URL="${BASE_URL}/refs/heads/${BRANCH}/dist"
  auth_token="${BENCHD_DIST_TOKEN:-${GITHUB_TOKEN:-}}"
  fetch() {
    local url="$1" out="$2"
    if [[ -n "${auth_token}" ]]; then
      curl -fsSL --retry 3 --retry-delay 2 -H "Authorization: Bearer ${auth_token}" -o "${out}" "${url}"
    else
      curl -fsSL --retry 3 --retry-delay 2 -o "${out}" "${url}"
    fi
  }

  echo "fetch-benchd.sh: resolving the ${BRANCH} channel tip (${DIST_DIR_URL})" >&2
  fetch "${DIST_DIR_URL}/benchctl.manifest.json" "${STAGED_MANIFEST}" || {
    {
      echo "fetch-benchd.sh: could not fetch the channel manifest (${DIST_DIR_URL}/benchctl.manifest.json)."
      if [[ -z "${auth_token}" ]]; then
        echo "  if the channel repo is private, set BENCHD_DIST_TOKEN/GITHUB_TOKEN."
      fi
      echo "  offline alternative: place benchctl + benchctl.manifest.json at ${DEST_DIR}/,"
      echo "  or point BENCHD_DIST_LOCAL at a directory holding both."
    } >&2
    exit 1
  }
  load_manifest "${STAGED_MANIFEST}" \
    || die "the channel manifest is malformed or names the wrong branch; refusing (this is a publish problem, not resolved by re-running)."
  fetch "${DIST_DIR_URL}/benchctl" "${STAGED}" \
    || die "the channel manifest exists but the binary download failed (${DIST_DIR_URL}/benchctl)."
  matches_manifest "${STAGED}" \
    || die "the downloaded benchctl does NOT match the channel manifest (see the mismatch above) -- a half-updated publish; refusing. Republish dist bench-side, then re-run."
fi

# -- 3. install ONLY what was verified ----------------------------------------
mkdir -p "${DEST_DIR}"
chmod 755 "${STAGED}"
mv "${STAGED}" "${DEST}"
mv "${STAGED_MANIFEST}" "${DEST_MANIFEST}"
echo "fetch-benchd.sh: installed benchctl at ${DEST} (manifest beside it)" >&2
announce

printf '%s\n' "${DEST}"
