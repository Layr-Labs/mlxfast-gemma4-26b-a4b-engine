#!/usr/bin/env bash
#
# gemma4-null-control.sh -- the calibration NULL CONTROL for track
# gemma4-26b-a4b-mlx-v1 (David ruling 2026-08-27: re-add the expected-value
# anchor the qwen track carried, calibration-system scope only).
#
# WHAT IT PROVES. Every published score is a ratio of paired legs, and nothing
# in the correctness gates, the decode floor, or the serial band can see a
# SYSTEMATIC BIAS in that ratio: a harness where a do-nothing candidate scores
# 1.01 quietly turns near-no-op resubmissions into accepted "improvements"
# (acceptance is improve-or-reject, so bias corrupts the record ratchet
# directly), and one where it scores 0.99 quietly rejects honest small gains.
# The null control measures the bias: run the PINNED BASELINE as its own
# candidate through the full paired flow and require the sealed composite to
# land within a small band of the certified expectation (~1.0). The qwen track
# measured its anchor at 0.994 -- these biases are real at the
# fraction-of-a-percent level, which is exactly the level the record ratchet
# operates at.
#
# CALIBRATION-SYSTEM SCOPE, deliberately: this is an ORGANIZER procedure run at
# calibration/cutover time. It never touches the scoring formula, never runs on
# a submission, and its expected value is never a scoring input -- it is a
# health certificate for the measuring instrument, checked when the instrument
# is calibrated. The expectation lives in its own values-only fixture
# (fixtures/gemma4_26b_a4b_null_control.json), NOT in the track contract, so
# benchd's contract parsing is untouched and no ranked run ever reads it.
#
# TWO MODES:
#   verify (default)  run the null measurement, compare the sealed composite
#                     against the certified fixture, exit 0 in-band / 1 out.
#                     Refuses while the fixture is still "pending-measurement":
#                     an uncertified anchor certifies nothing.
#   --author          run the null measurement and REWRITE the fixture with the
#                     measured composite (status "measured"), for the organizer
#                     to review and commit. Authoring is explicitly a human-
#                     reviewed act: the script writes the file and prints what
#                     it wrote; git carries the decision.
#   --check <results.json>
#                     offline: skip the measurement and evaluate an existing
#                     sealed results file against the fixture (used by the
#                     verify path after its own run, and by the offline tests).
#
# Env (same conventions as tools/gemma4-measure-and-score.sh):
#   MLXFAST_GEMMA4_BASELINE_WORKSPACE  REQUIRED. The pinned baseline workspace;
#                                      it is used as BOTH legs of the pair.
#   MLXFAST_GEMMA4_GOLDEN_DIR          REQUIRED. The 8 hidden timed-pool goldens.
#   MLXFAST_CORRECTNESS_GOLDEN_PATH    REQUIRED. The staged correctness golden.
#   MLXFAST_WEIGHTS_PATH               transformed weights (default ./weights).
#   MLXFAST_GEMMA4_MEASURE_OUT_DIR     measure-job --out (default
#                                      ./null-control-results-local).
#   BENCHCTL                           benchctl override; default: the channel
#                                      binary tools/fetch-benchd.sh resolves.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

FIXTURE="${SCRIPT_DIR}/fixtures/gemma4_26b_a4b_null_control.json"
CONTRACT="${SCRIPT_DIR}/fixtures/gemma4_26b_a4b_track.json"

MODE="verify"
CHECK_RESULTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --author) MODE="author"; shift ;;
    --check) MODE="check"; CHECK_RESULTS="${2:?--check needs a results.json path}"; shift 2 ;;
    *)
      echo "gemma4-null-control.sh: unrecognized argument: $1" >&2
      exit 2
      ;;
  esac
done

command -v jq >/dev/null 2>&1 || { echo "gemma4-null-control.sh: jq is required" >&2; exit 1; }

read_composite() {
  # The sealed composite of the one scored cohort. null/absent -> empty string.
  jq -r '.per_cohort[0].composite.composite_score // empty' "$1" 2>/dev/null
}

evaluate() {
  # evaluate <results.json> -- compare against the fixture. Exit codes:
  # 0 in-band; 1 out-of-band or unusable results; 3 fixture not certified.
  local results="$1" composite status expected band
  [[ -s "${results}" ]] || { echo "gemma4-null-control.sh: results file missing or empty: ${results}" >&2; return 1; }
  composite="$(read_composite "${results}")"
  if [[ -z "${composite}" ]]; then
    echo "gemma4-null-control.sh: the null run sealed no composite (per_cohort[0].composite absent)." >&2
    echo "  reason: $(jq -r '.per_cohort[0].composite_absent_reason // "unknown"' "${results}" 2>/dev/null)" >&2
    echo "  a null control that cannot score is a FAILED control, not a pass." >&2
    return 1
  fi
  status="$(jq -r '.status // empty' "${FIXTURE}")"
  expected="$(jq -r '.expected_composite // empty' "${FIXTURE}")"
  band="$(jq -r '.band_pct // empty' "${FIXTURE}")"
  if [[ "${status}" != "measured" ]]; then
    echo "gemma4-null-control.sh: the fixture is '${status}', not 'measured' -- the anchor is not certified." >&2
    echo "  measured composite this run: ${composite}" >&2
    echo "  author it deliberately: ./tools/gemma4-null-control.sh --author   (then review + commit)" >&2
    return 3
  fi
  # In-band iff |composite/expected - 1| * 100 <= band_pct. jq does the float
  # arithmetic so the shell stays portable; the 1e-9 slack only absorbs float
  # rounding at the exact band edge, it is not a wider band.
  if [[ "$(jq -n --argjson c "${composite}" --argjson e "${expected}" --argjson b "${band}" \
        'if $e <= 0 then false else ((($c / $e) - 1) | if . < 0 then -. else . end) * 100 <= $b + 1e-9 end')" == "true" ]]; then
    echo "gemma4-null-control.sh: NULL CONTROL PASS -- composite ${composite} within ${band}% of expected ${expected}." >&2
    return 0
  fi
  echo "gemma4-null-control.sh: NULL CONTROL FAIL -- composite ${composite} is outside ${band}% of expected ${expected}." >&2
  echo "  The paired flow is showing a systematic bias (or the box/baseline changed)." >&2
  echo "  Do NOT arm or open a benchmark on this calibration; find the drift first." >&2
  return 1
}

author() {
  # author <results.json> -- rewrite the fixture with the measured composite.
  local results="$1" composite
  composite="$(read_composite "${results}")"
  if [[ -z "${composite}" ]]; then
    echo "gemma4-null-control.sh: refusing to author -- the null run sealed no composite." >&2
    return 1
  fi
  jq --argjson c "${composite}" '.status = "measured" | .expected_composite = $c' "${FIXTURE}" > "${FIXTURE}.tmp"
  mv "${FIXTURE}.tmp" "${FIXTURE}"
  echo "gemma4-null-control.sh: authored expected_composite ${composite} (status measured) into ${FIXTURE}" >&2
  echo "  review the value, then commit the fixture -- authoring is not certifying." >&2
}

if [[ "${MODE}" == "check" ]]; then
  evaluate "${CHECK_RESULTS}"
  exit $?
fi

# -- run the null measurement -------------------------------------------------
if [[ -z "${BENCHCTL:-}" ]]; then
  BENCHCTL="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
[[ -x "${BENCHCTL}" ]] || { echo "gemma4-null-control.sh: benchctl not found at ${BENCHCTL}" >&2; exit 1; }

BASELINE="${MLXFAST_GEMMA4_BASELINE_WORKSPACE:?MLXFAST_GEMMA4_BASELINE_WORKSPACE is required (the pinned baseline workspace; it runs as BOTH legs)}"
GOLDEN_DIR="${MLXFAST_GEMMA4_GOLDEN_DIR:?MLXFAST_GEMMA4_GOLDEN_DIR is required (the hidden timed-pool goldens)}"
CORRECTNESS="${MLXFAST_CORRECTNESS_GOLDEN_PATH:?MLXFAST_CORRECTNESS_GOLDEN_PATH is required}"
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${SCRIPT_DIR}/weights}"
OUT_DIR="${MLXFAST_GEMMA4_MEASURE_OUT_DIR:-${SCRIPT_DIR}/null-control-results-local}"
mkdir -p "${OUT_DIR}"

golden_args=()
for f in "${GOLDEN_DIR}"/*.json; do
  [[ -e "${f}" ]] || continue
  golden_args+=(--golden "${f}")
done
[[ "${#golden_args[@]}" -gt 0 ]] || { echo "gemma4-null-control.sh: no goldens in ${GOLDEN_DIR}" >&2; exit 1; }

echo "gemma4-null-control.sh: running the NULL measurement (baseline as both legs) ..." >&2
"${BENCHCTL}" measure-job \
  --contract "${CONTRACT}" \
  --candidate "${BASELINE}" \
  --baseline "${BASELINE}" \
  --weights "${WEIGHTS_PATH}" \
  --correctness-golden "${CORRECTNESS}" \
  "${golden_args[@]}" \
  --min-pairs 4 --target-pairs 4 \
  --tag "gemma4-null-control" \
  --out "${OUT_DIR}"

RESULTS_JSON="${OUT_DIR}/results.json"
if [[ "${MODE}" == "author" ]]; then
  author "${RESULTS_JSON}"
else
  evaluate "${RESULTS_JSON}"
fi
