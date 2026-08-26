#!/usr/bin/env bash
#
# fetch-benchd.sh -- resolve the pinned `benchctl` measurement harness.
#
# This replaces the benchd SOURCE submodule. benchd is no longer checked out or
# built here: the organizer publishes a prebuilt per-branch `benchctl` (see the
# bench repo's scripts/build-dist.sh and .github/workflows/dist.yml) and this
# repository pins ONE of those binaries by {branch, commit, sha256, bytes} in
# ./benchd.pin. Two reasons the pin is a binary and not source:
#
#   1. NO CARGO ON THE RANKED BOX. The M5 box has no Rust toolchain, so a source
#      submodule could never be built where it actually matters -- the binary had
#      to be produced elsewhere and copied in, undocumented and unverified.
#   2. SOURCE-THE-USER-BUILDS IS WEAK TAMPER-EVIDENCE. The measurement harness is
#      immutable to the participant by contract; only the ENGINE is editable. A
#      participant who compiles the scorer can weaken it first. A pinned sha256
#      makes that concrete and, locally, evident.
#
# Honest limit, stated so nobody over-claims: a participant owns their machine
# and can patch this check out. What it buys LOCALLY is that the default path
# runs the organizer's exact binary and that drift is evident. The actual
# enforcement is the official run on organizer infrastructure.
#
# WHAT IT DOES, in order:
#   1. If benchd-bin/benchctl exists and its sha256 AND byte count match the
#      pin, use it and never touch the network. This is the OFFLINE path: on the
#      box the binary is scp'd into place and accepted here without any
#      download.
#   2. Otherwise try each download candidate in turn, verifying EACH against the
#      pin, and install the first one that matches -- a candidate that responds
#      with the wrong bytes is skipped, not fatal, so one stale publish cannot
#      hide a good binary at the next location.
#
# It NEVER runs, installs, or returns an unverified binary. An unparseable pin,
# a pre-placed binary that fails the pin, a named BENCHD_DIST_LOCAL that fails
# the pin, and an exhausted candidate list (whether nothing responded or nothing
# that responded matched) are all hard refusals -- there is no fallback to
# "whatever benchctl is on PATH" and no fallback to building from source,
# because both would silently score a submission against unpinned measurement
# code.
#
# Note the one asymmetry, which is deliberate: a mismatch is skipped only where
# there is a NEXT candidate to try. Where the caller named a single source --
# benchd-bin/benchctl already in place, or BENCHD_DIST_LOCAL -- a mismatch is
# fatal, because silently moving past the exact file the operator pointed at
# would hide the problem rather than route around it.
#
# Prints the absolute path of the verified binary on STDOUT (diagnostics go to
# stderr), so callers can do:  BENCHCTL="$(./tools/fetch-benchd.sh)"
#
# Env:
#   BENCHD_BIN_DIR      install directory. Default: <repo>/benchd-bin (gitignored;
#                       it is a fetched artifact, not repository content).
#   BENCHD_DIST_LOCAL   path to an already-obtained benchctl (file) or to a
#                       directory containing one. Copied in instead of
#                       downloading -- still hash-verified. For air-gapped boxes.
#   BENCHD_DIST_URL     exact URL to fetch, overriding the derived candidates.
#   BENCHD_DIST_BASE_URL
#                       raw host + repo prefix. Default:
#                       https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench
#   BENCHD_DIST_TOKEN / GITHUB_TOKEN
#                       bearer token; mlxfast-bench is private, so raw fetches
#                       need one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="${REPO_ROOT}/benchd.pin"
DEST_DIR="${BENCHD_BIN_DIR:-${REPO_ROOT}/benchd-bin}"
DEST="${DEST_DIR}/benchctl"
BASE_URL="${BENCHD_DIST_BASE_URL:-https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench}"

die() {
  echo "fetch-benchd.sh: $*" >&2
  exit 1
}

# -- pin ----------------------------------------------------------------------
# benchd.pin is values-only JSON, one key per line, authored by this repository
# from the dist manifest. Parsed with sed rather than jq/python3 so the OFFLINE
# path on the ranked box needs nothing but a shell and shasum.
pin_field() {
  sed -n "s/^[[:space:]]*\"$1\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",]*\)\"\{0,1\}[[:space:]]*,\{0,1\}[[:space:]]*\$/\1/p" "${PIN_FILE}"
}

[[ -f "${PIN_FILE}" ]] || die "${PIN_FILE} is missing; there is no pin to resolve."

PIN_BRANCH="$(pin_field branch)"
PIN_COMMIT="$(pin_field commit)"
PIN_SHA256="$(pin_field sha256)"
PIN_BYTES="$(pin_field bytes)"

# An unparseable pin must refuse, not degrade: empty fields would make every
# comparison below trivially "match".
if [[ "${#PIN_SHA256}" -ne 64 || -n "${PIN_SHA256//[0-9a-f]/}" ]]; then
  die "benchd.pin sha256 is not 64 lowercase hex characters: '${PIN_SHA256}'"
fi
if [[ -z "${PIN_BYTES}" || -n "${PIN_BYTES//[0-9]/}" ]]; then
  die "benchd.pin bytes is not a positive integer: '${PIN_BYTES}'"
fi
[[ -n "${PIN_BRANCH}" ]] || die "benchd.pin branch is empty."
[[ -n "${PIN_COMMIT}" ]] || die "benchd.pin commit is empty."

command -v shasum >/dev/null 2>&1 || die "shasum is required to verify the pinned benchctl."

# -- verify -------------------------------------------------------------------
# Bytes first (cheap, and a length mismatch is already disqualifying), then the
# digest. Reports on stderr so callers see WHICH half failed.
matches_pin() {
  local path="$1" actual_bytes actual_sha
  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${PIN_BYTES}" ]]; then
    echo "fetch-benchd.sh:   byte count mismatch: pin=${PIN_BYTES} actual=${actual_bytes} (${path})" >&2
    return 1
  fi
  actual_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${PIN_SHA256}" ]]; then
    echo "fetch-benchd.sh:   sha256 mismatch: pin=${PIN_SHA256} actual=${actual_sha} (${path})" >&2
    return 1
  fi
  return 0
}

# -- 1. already in place (the offline path) -----------------------------------
if [[ -f "${DEST}" ]]; then
  if matches_pin "${DEST}"; then
    chmod 755 "${DEST}"
    printf '%s\n' "${DEST}"
    exit 0
  fi
  # Present but WRONG. Do not silently overwrite an operator-placed binary that
  # fails the pin: that is exactly the condition worth surfacing loudly. Set
  # BENCHD_BIN_DIR elsewhere, or delete it deliberately, then re-run.
  die "${DEST} exists but does NOT match benchd.pin (see the mismatch above); refusing to run or replace it."
fi

# -- 2. obtain ----------------------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
STAGED="${TMP_DIR}/benchctl"

verified=0

if [[ -n "${BENCHD_DIST_LOCAL:-}" ]]; then
  src="${BENCHD_DIST_LOCAL}"
  [[ -d "${src}" ]] && src="${src}/benchctl"
  [[ -f "${src}" ]] || die "BENCHD_DIST_LOCAL does not resolve to a file: ${src}"
  echo "fetch-benchd.sh: taking the pinned benchctl from ${src}" >&2
  cp "${src}" "${STAGED}"
  # One named source, so a mismatch here is a hard refusal rather than a skip:
  # there is no next candidate to fall through to, and silently continuing past
  # the file the operator explicitly pointed at would hide the real problem.
  matches_pin "${STAGED}" \
    || die "BENCHD_DIST_LOCAL (${src}) does NOT match benchd.pin (see the mismatch above); refusing to install or run it."
  verified=1
else
  command -v curl >/dev/null 2>&1 || die "curl is required to download the pinned benchctl (or set BENCHD_DIST_LOCAL)."

  # Candidate URLs, tried in order. Trying more than one is NOT a weakening:
  # every candidate is hash-verified against the same pin before it is installed,
  # so a wrong or hostile response cannot be accepted from any of them.
  #
  #   (a) BENCHD_DIST_URL, when the caller names one exactly.
  #   (b) The CI layout published by the bench repo's .github/workflows/dist.yml:
  #       an orphan `dist/<branch>` branch with one directory per source commit.
  #       This is the steady state.
  #   (c) The HAND-PUBLISHED layout: dist/ committed on the source branch itself.
  #       Org Actions are currently disabled, so (b) does not exist yet for this
  #       pin and (c) is what actually resolves today. Drop (c) once CI publishes.
  #
  # The `refs/heads/` prefix is required, not cosmetic: these branch names
  # contain slashes, and without the explicit ref namespace raw.githubusercontent
  # cannot tell where the ref ends and the path begins.
  candidates=""
  if [[ -n "${BENCHD_DIST_URL:-}" ]]; then
    candidates="${BENCHD_DIST_URL}"
  else
    candidates="${BASE_URL}/refs/heads/dist/${PIN_BRANCH}/${PIN_COMMIT}/benchctl
${BASE_URL}/refs/heads/${PIN_BRANCH}/dist/benchctl"
  fi

  auth_token="${BENCHD_DIST_TOKEN:-${GITHUB_TOKEN:-}}"
  # A candidate is accepted only by PASSING THE PIN, not by returning 200. The
  # hash check lives inside this loop for availability: candidate (b) can exist
  # and serve the wrong bytes -- a dist branch left behind at an older commit is
  # the ordinary way that happens -- and verifying only after the loop would let
  # that one stale hit mask a later candidate holding the correct binary. A
  # mismatch therefore continues to the next URL instead of aborting.
  #
  # This does not loosen anything. Every candidate is measured against the SAME
  # pin, the loop can only end in "one candidate matched exactly" or "none did",
  # and the latter still refuses below. Trying more places to find the ONE
  # accepted byte sequence is not the same as accepting more byte sequences.
  # A MISSING candidate is the ORDINARY case, not a failure: candidate (b) does
  # not exist at all until CI publishes, so on every current run the first URL
  # 404s and the second one serves the binary. That normal resolve used to read
  # as a broken one -- an unqualified "fetching <url>", then curl's own
  # "curl: (22) The requested URL returned error: 404" on its own line, then
  # "not available here", none of which said that a next candidate was about to
  # be tried and that this was expected.
  #
  # So: fold the miss into ONE line that ends in what happens next, and keep
  # curl's exit code AND its message inside it. Nothing is lost -- the exit code
  # is what separates "published nothing here" (22) from a DNS (6) or TLS (35)
  # problem worth acting on, and curl's own text rides along after it. curl
  # keeps --show-error; its stderr is captured rather than silenced so it can be
  # folded in instead of landing mid-transcript.
  candidate_total="$(printf '%s\n' "${candidates}" | grep -c '[^[:space:]]' || true)"
  candidate_index=0
  curl_err="${TMP_DIR}/curl.err"
  downloaded_any=0
  while IFS= read -r url; do
    [[ -n "${url}" ]] || continue
    candidate_index=$((candidate_index + 1))
    if [[ "${candidate_index}" -lt "${candidate_total}" ]]; then
      next_step="trying the next candidate"
    else
      next_step="no candidates left"
    fi
    echo "fetch-benchd.sh: candidate ${candidate_index}/${candidate_total}: ${url}" >&2
    rm -f "${STAGED}"
    : > "${curl_err}"
    if [[ -n "${auth_token}" ]]; then
      curl_ok=0
      curl -fsSL --retry 3 --retry-delay 2 \
        -H "Authorization: Bearer ${auth_token}" \
        -o "${STAGED}" "${url}" 2>"${curl_err}" || curl_ok=$?
    else
      curl_ok=0
      curl -fsSL --retry 3 --retry-delay 2 -o "${STAGED}" "${url}" 2>"${curl_err}" || curl_ok=$?
    fi
    if [[ "${curl_ok}" != "0" || ! -f "${STAGED}" ]]; then
      # curl's own leading "curl: (<code>) " is dropped because the code is
      # already in the line; the rest of its text is kept verbatim, flattened to
      # one line.
      curl_detail="$(tr '\n' ' ' < "${curl_err}" | sed -e 's/^curl: ([0-9]*) //' -e 's/[[:space:]]*$//')"
      if [[ -n "${curl_detail}" ]]; then
        echo "fetch-benchd.sh:   nothing published here (curl exit ${curl_ok}: ${curl_detail}); ${next_step}" >&2
      else
        echo "fetch-benchd.sh:   nothing published here (curl exit ${curl_ok}); ${next_step}" >&2
      fi
      continue
    fi
    downloaded_any=1
    if matches_pin "${STAGED}"; then
      verified=1
      break
    fi
    echo "fetch-benchd.sh:   these bytes do not match the pin; ${next_step}" >&2
  done <<EOF
${candidates}
EOF

  if [[ "${verified}" != "1" ]]; then
    {
      if [[ "${downloaded_any}" == "1" ]]; then
        # Reached a candidate but the bytes were wrong. Nearly always a stale
        # publish rather than an attack, and worth naming that way so the reader
        # checks the dist before assuming the worst.
        echo "fetch-benchd.sh: every candidate that responded FAILED the pin (see the mismatches above)."
        echo "  pin: branch=${PIN_BRANCH} commit=${PIN_COMMIT}"
        echo "  usually this means the dist for this pin has not been published yet, or the"
        echo "  branch is serving an older build. It is NOT resolved by re-running."
      else
        echo "fetch-benchd.sh: could not download the pinned benchctl."
        echo "  pin: branch=${PIN_BRANCH} commit=${PIN_COMMIT}"
        if [[ -z "${auth_token}" ]]; then
          echo "  no BENCHD_DIST_TOKEN/GITHUB_TOKEN was set, and mlxfast-bench is a PRIVATE repository."
        fi
      fi
      echo "  offline alternative: place the binary at ${DEST}, or point BENCHD_DIST_LOCAL at it."
      echo "  it is accepted without any network access as long as it matches the pin."
    } >&2
    exit 1
  fi
fi

# -- 3. install ONLY what was verified ----------------------------------------
# Belt-and-braces: every path above sets verified=1 only immediately after a
# matches_pin() success on these exact staged bytes, and nothing has been made
# runnable yet. This re-asserts it so that a future edit adding a fourth way to
# obtain the binary cannot reach the install below without passing the pin.
if [[ "${verified}" != "1" ]]; then
  die "internal: reached install with unverified bytes; refusing."
fi

mkdir -p "${DEST_DIR}"
chmod 755 "${STAGED}"
mv "${STAGED}" "${DEST}"
echo "fetch-benchd.sh: installed the pinned benchctl at ${DEST}" >&2
echo "fetch-benchd.sh:   branch=${PIN_BRANCH} commit=${PIN_COMMIT} sha256=${PIN_SHA256} bytes=${PIN_BYTES}" >&2

printf '%s\n' "${DEST}"
