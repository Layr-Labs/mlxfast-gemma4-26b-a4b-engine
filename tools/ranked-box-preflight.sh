#!/usr/bin/env bash
#
# ranked-box-preflight.sh -- the fail-closed gate the ranked job runs BEFORE
# ./setup.sh and before any measurement, for track gemma4-26b-a4b-mlx-v1.
#
# WHAT IT IS FOR. The ranked job holds NO CREDENTIAL by design (bundles-no-keys:
# a box receives staged bundles, never keys), so the hidden material this track
# measures against cannot be fetched by the job itself -- it is staged onto the
# box out of band by the organizer. That moves the whole question from "can we
# download it" to "is what is on this box the pinned material". This script
# answers that question, and refuses when the answer is anything other than
# "yes, exactly".
#
# Every check below aborts before a score.json could exist. There is no
# degraded mode: the alternative to a verified staged asset is a non-zero exit,
# never a substituted, defaulted, or re-fetched one.
#
# THE STAGING CONVENTION IS THE EXISTING ONE, not a new one. The two paths come
# from the runner process environment under the names
# tools/gemma4-measure-and-score.sh already reads:
#
#   MLXFAST_GEMMA4_GOLDEN_DIR          directory holding the 8 timed-pool tapes
#   MLXFAST_GEMMA4_BASELINE_WORKSPACE  the pinned serial-control workspace
#
# On a self-hosted runner those are set in the runner service environment by
# whoever stages the box; a `run:` step inherits them. Nothing here reads a
# GitHub secret, and nothing here reaches the network.
#
# WHAT THE PINS ARE. fixtures/gemma4_26b_a4b_track.json is trusted-side (not an
# editable path), and its timed_prompt_pool[] carries {r2_path, sha256, bytes}
# per tape plus hidden_correctness_golden's {sha256, bytes}. A staged file is
# accepted only when its byte count AND its sha256 equal the contract's -- byte
# count first, because a truncated stage is the common failure and naming it
# precisely is worth one `wc -c` (the same order tools/fetch-goldens.sh and
# tools/fetch-benchd.sh verify in).
#
# Usage:  tools/ranked-box-preflight.sh
# Exit:   0 every check passed
#         1 a check failed (message on stderr names which)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${REPO_ROOT}/fixtures/gemma4_26b_a4b_track.json"

fail() {
  echo "ranked-box-preflight: REFUSING -- $*" >&2
  exit 1
}

ok() {
  echo "ok    $*"
}

# --- 0. tools ---------------------------------------------------------------
# jq is already a hard requirement of .github/scripts/emit-gemma4-score.sh, so
# a box that cannot run this cannot finish a ranked run either.
command -v jq >/dev/null 2>&1 || fail "jq is required to read the track contract"
command -v shasum >/dev/null 2>&1 || fail "shasum is required to verify staged assets"
[[ -r "${CONTRACT}" ]] || fail "cannot read the track contract at ${CONTRACT}"
jq -e . >/dev/null 2>&1 < "${CONTRACT}" || fail "the track contract is not valid JSON: ${CONTRACT}"
ok "track contract readable and parses: fixtures/gemma4_26b_a4b_track.json"

# --- 1. the job holds no credential -----------------------------------------
# The workflow references no secret (tools/ci-workflow-egress-scan.sh is the
# static half of that). This is the runtime half: a credential reaching the job
# through the RUNNER's environment would defeat the same invariant without ever
# appearing in the workflow file. R2 keys and a signer are what would let this
# job pull hidden material itself instead of measuring the staged bundle.
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_GEMMA4_R2_DOWNLOADER BENCHD_DIST_TOKEN; do
  eval "value=\${${var}:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; this job must hold no credential (boxes get staged bundles, never keys)"
done
ok "no R2 credential, signer, or dist token in the job environment"

# --- 2. no measurement-weakening override -----------------------------------
# Each of these is a real, local-debugging-only switch in this tree. On a
# ranked box any of them would silently change what is measured or what is
# accepted, so their presence is a refusal rather than a warning.
#
#   BENCHCTL                          tools/gemma4-measure-and-score.sh honours a
#                                     caller-supplied benchctl WITHOUT the
#                                     benchd.pin hash check (deliberate, for
#                                     benchd development) -- so on a ranked run
#                                     it is a way to measure against unpinned
#                                     scoring code.
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD /   setup.sh builds tools only and never
#   SKIP_MODEL_DOWNLOAD               obtains or verifies the checkpoint.
#   MLXFAST_LOCAL_COOL_GATE           disables the thermal gate.
#   MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT  publishes a timing estimate past a failed
#                                     public correctness gate.
for var in BENCHCTL MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT; do
  eval "value=\${${var}:-}"
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; it weakens or bypasses what the ranked run measures"
done
ok "no measurement-weakening override in the job environment"

# --- 3. the timed pool is armed ---------------------------------------------
# "Armed" is a property of the CONTRACT, checked before anything on disk is
# looked at: a sentinel entry has no digest to verify a staged file against, so
# a staged file would be accepted on its name alone.
if grep -q 'PENDING-ORGANIZER' "${CONTRACT}"; then
  fail "the track contract still carries PENDING-ORGANIZER sentinels; the timed pool is unarmed and nothing can be pin-verified against it"
fi

pool_count="$(jq -r '.timed_prompt_pool | length' "${CONTRACT}")"
[[ "${pool_count}" == "8" ]] || fail "timed_prompt_pool has ${pool_count} entries, expected 8 (the cohort size this track scores)"

# A pin is {sha256, bytes} together; neither half alone is one. An entry that
# fails this is unarmed no matter what it is called.
malformed="$(jq -r '
  .timed_prompt_pool
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("timed_prompt_pool[" + (.key | tostring) + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${malformed}" ]] || fail "unarmed or malformed pool pin(s): ${malformed}"

hidden_sha="$(jq -r '.hidden_correctness_golden.sha256 // ""' "${CONTRACT}")"
hidden_bytes="$(jq -r '.hidden_correctness_golden.bytes // 0' "${CONTRACT}")"
printf '%s' "${hidden_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "hidden_correctness_golden.sha256 is not a 64-hex digest; the token-fidelity oracle is unarmed"
printf '%s' "${hidden_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "hidden_correctness_golden.bytes is not a positive integer"
ok "timed pool armed: 8 pinned tapes + a pinned hidden correctness golden"

# --- 4. the staged tapes match the pins -------------------------------------
GOLDEN_DIR="${MLXFAST_GEMMA4_GOLDEN_DIR:-}"
[[ -n "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_GEMMA4_GOLDEN_DIR is unset; the 8 timed-pool tapes are staged onto the box out of band and this job holds no credential to fetch them"
[[ -d "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_GEMMA4_GOLDEN_DIR does not exist or is not a directory: ${GOLDEN_DIR}"

verify_pin() {
  # verify_pin <path> <want_sha256> <want_bytes> <label>
  local path="$1" want_sha="$2" want_bytes="$3" label="$4" got_bytes got_sha
  [[ -f "${path}" ]] || fail "${label}: staged file is missing: ${path}"
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${want_bytes}" ]] \
    || fail "${label}: byte-count mismatch (staged ${got_bytes}, pinned ${want_bytes}): ${path}"
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${want_sha}" ]] \
    || fail "${label}: sha256 mismatch (staged ${got_sha}, pinned ${want_sha}): ${path}"
}

expected_list=""
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "timed-pool tape ${name}"
  expected_list="${expected_list}${name}
"
done <<EOF
$(jq -r '.timed_prompt_pool[] | [.r2_path, .sha256, (.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
ok "all 8 staged timed-pool tapes match their contract pins (bytes then sha256)"

# The staging directory must hold the cohort and NOTHING ELSE.
# tools/gemma4-measure-and-score.sh passes EVERY *.json in this directory as a
# --golden, so an extra file there is an extra cohort member -- it would change
# what is measured without failing anything downstream.
unexpected=""
for staged in "${GOLDEN_DIR}"/*.json; do
  [[ -e "${staged}" ]] || continue
  staged_name="${staged##*/}"
  if ! printf '%s' "${expected_list}" | grep -Fxq "${staged_name}"; then
    unexpected="${unexpected} ${staged_name}"
  fi
done
[[ -z "${unexpected}" ]] \
  || fail "MLXFAST_GEMMA4_GOLDEN_DIR holds *.json file(s) that are not in the pinned cohort:${unexpected} (every *.json there is passed as a --golden, so an extra file silently changes the measured cohort)"
ok "no unpinned *.json in the staging directory"

# The hidden correctness oracle is pinned by digest only -- the contract gives
# it no r2_path -- and benchctl resolves it itself. If the box names one
# through the existing MLXFAST_CORRECTNESS_GOLDEN_PATH convention
# (Sources/MLXFastCLI/main.swift), it must be the pinned bytes; if it names
# none, this asserts nothing about it rather than inventing a location.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  verify_pin "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" "${hidden_sha}" "${hidden_bytes}" "hidden correctness golden"
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH matches hidden_correctness_golden"
else
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH unset; benchctl resolves the oracle from the contract"
fi

# --- 5. the baseline workspace ----------------------------------------------
# The score is a PAIRED ratio: candidate against a pinned serial control
# measured in the same session. A baseline that resolves to the candidate
# workspace is not a control -- it would compare the submission to itself.
BASELINE="${MLXFAST_GEMMA4_BASELINE_WORKSPACE:-}"
[[ -n "${BASELINE}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE is unset; there is no serial control to pair against"
[[ -d "${BASELINE}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE does not exist or is not a directory: ${BASELINE}"
baseline_real="$(cd "${BASELINE}" && pwd -P)"
[[ "${baseline_real}" != "${REPO_ROOT}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE resolves to the candidate workspace (${REPO_ROOT}); a paired run needs a distinct pinned baseline"
ok "baseline workspace present and distinct from the candidate workspace"

echo "ranked-box-preflight: all checks passed"
