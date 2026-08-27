#!/usr/bin/env bash
#
# emit-gemma4-score.sh -- convert a benchd `measure-job` results.json into the
# {score, metrics} shape src/benchmark/score.ts requires at scorePath
# (Yukon's ScoreFileSchema: `score` a finite number, `metrics` optional JSON).
#
# Track gemma4-26b-a4b-mlx-v1 scores on a COMPOSITE, not the qwen family's
# per-prompt median: composite = prefill_gain^0.25 * decode_gain^0.75, read
# from `per_cohort[0].composite` (see docs/participant-contract.md section 5.2).
# The PINNED benchd EMITS that field, computed from its own parent-clocked
# prefill and decode windows.
#
# `composite` IS AN OBJECT, NOT A NUMBER. This script read it as a bare number
# until 2026-08-26, and that is what killed a fully successful ranked round: the
# measurement finished, benchd sealed composite_score 0.9383 with the floor met,
# and this script refused it as "not a plain finite number" because jq handed it
# the whole object. The real shape, from the pinned benchd (./benchd.pin sha256
# e044e1f4..., `CompositeCohortScore` in crates/benchctl/src/measure_job.rs:6020)
# and confirmed against sealed results.json files off the box:
#
#   "composite": {
#     "composite_score": 0.945956243476148,
#     "composite_speedup_floor": 0.9,
#     "composite_speedup_floor_met": true,
#     "decode_gain": 0.9193183158707673,
#     "prefill_gain": 1.0305912562853754
#   }
#
# The SCORE is `composite.composite_score`. The absent case is unchanged and
# still real: `"composite": null` alongside a `composite_absent_reason` (a cohort
# that accepted no pair seals exactly that), so a null here remains a refusal to
# score rather than a missing feature.
#
# THE FLOOR IS ENFORCED HERE. benchd documents `composite_speedup_floor_met` as
# SEAL-ONLY -- it is deliberately wired to no exit code on the benchd side -- so
# without this seam a below-floor run would publish a score as if it had passed.
# Reporting a floor is not enforcing one: this script REFUSES on
# `composite_speedup_floor_met: false`, with a message distinct from the
# schema-mismatch refusal, because an honest below-floor measurement and a
# malformed seal are different events and must not read the same in a log.
#
# The floor VALUE is cross-checked against benchmark.json's
# scoring.decodeSpeedupFloor rather than hardcoded here. benchmark.json is
# trusted-side (not an editablePath) and tools/lint-benchmark-manifest.py already
# holds it equal to the track fixture's scoring_semantics, so this is a drift
# tripwire over two documents that are supposed to agree, not a third opinion.
#
# What has NOT changed is the refusal discipline: this script never fabricates a
# number from the `raw_ratio_of_means` diagnostic that is also present. That
# diagnostic is the shared-cohort-window ratio, not the ruled per-stream
# composite; substituting one for the other would silently score a different
# formula than the one David ruled.
#
# Usage: emit-gemma4-score.sh <results.json> <score-path>
# Exit:  0 on a real composite emitted to <score-path>
#        1 on a missing/malformed results.json, or a still-absent composite
#          (the expected, honest outcome today)
#        2 usage error
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: emit-gemma4-score.sh <results.json> <score-path>" >&2
  exit 2
fi

RESULTS_JSON="$1"
SCORE_PATH="$2"

if ! command -v jq >/dev/null 2>&1; then
  echo "emit-gemma4-score.sh: jq is required" >&2
  exit 1
fi

if [[ ! -s "${RESULTS_JSON}" ]]; then
  echo "emit-gemma4-score.sh: results file missing or empty: ${RESULTS_JSON}" >&2
  exit 1
fi

if ! jq -e . >/dev/null 2>&1 < "${RESULTS_JSON}"; then
  echo "emit-gemma4-score.sh: results file is not valid JSON: ${RESULTS_JSON}" >&2
  exit 1
fi

# `.per_cohort[0].composite` reads as `null` uniformly whether the key is fully
# absent from the schema or present-but-null (what a zero-accepted-pair cohort
# actually seals) -- jq does not distinguish, and this script does not need to:
# either way, no composite means refuse.
composite_obj="$(jq -c '.per_cohort[0].composite // empty' "${RESULTS_JSON}")"

if [[ -z "${composite_obj}" || "${composite_obj}" == "null" ]]; then
  reason="$(jq -r '.per_cohort[0].composite_absent_reason // "composite is null in results.json and no composite_absent_reason was found (composite cohort scoring has landed but stays gated pending per-stream instrumentation -- see docs/participant-contract.md section 5.2)"' "${RESULTS_JSON}")"
  echo "emit-gemma4-score.sh: refusing to emit a score -- composite is absent." >&2
  echo "  reason: ${reason}" >&2
  echo "  refuse, not degrade: this script will not substitute the raw_ratio_of_means" >&2
  echo "  shared-cohort-window diagnostic for the ruled per-stream composite formula." >&2
  exit 1
fi

# The SCORE is the object's composite_score. A composite object without one is a
# schema mismatch -- benchd changed shape under this script -- and is refused as
# such, never silently treated as "no composite".
# Guarded with a type check: a composite that is NOT an object (the bare number
# this script used to assume) must reach the schema refusal below, not abort jq.
composite="$(jq -r '.per_cohort[0].composite | if type == "object" then (.composite_score // empty) else empty end' "${RESULTS_JSON}")"

if [[ -z "${composite}" || "${composite}" == "null" ]]; then
  echo "emit-gemma4-score.sh: SCHEMA MISMATCH -- per_cohort[0].composite is present but carries no composite_score." >&2
  echo "  composite: ${composite_obj}" >&2
  echo "  expected the CompositeCohortScore object sealed by the pinned benchd" >&2
  echo "  ({composite_score, composite_speedup_floor, composite_speedup_floor_met, decode_gain, prefill_gain})." >&2
  echo "  Refusing: a shape this script does not understand is not a score." >&2
  exit 1
fi

# A composite that is present must still be finite (score.ts requires
# z.number().finite()); a non-finite value is a benchd bug, not a score.
case "${composite}" in
  *[!0-9.eE+-]*|"")
    echo "emit-gemma4-score.sh: composite_score is not a plain finite number: ${composite}" >&2
    exit 1
    ;;
esac

# THE FLOOR, ENFORCED. benchd reports this flag and wires it to nothing; the
# score seam is where a below-floor run has to stop, or the floor is decoration.
# NO `//` ON THESE TWO. jq's alternative operator treats `false` as absent, so
# `composite_speedup_floor_met // empty` would turn the one value this check
# exists to catch -- false -- into "field missing" and refuse with the wrong
# message. Read them straight and test for null explicitly.
floor_met="$(jq -r '.per_cohort[0].composite.composite_speedup_floor_met' "${RESULTS_JSON}")"
floor_value="$(jq -r '.per_cohort[0].composite.composite_speedup_floor' "${RESULTS_JSON}")"

if [[ "${floor_met}" != "true" && "${floor_met}" != "false" ]]; then
  echo "emit-gemma4-score.sh: SCHEMA MISMATCH -- composite carries no boolean composite_speedup_floor_met (got '${floor_met}')." >&2
  echo "  Refusing: the floor cannot be enforced against a flag that is not there." >&2
  exit 1
fi

if [[ "${floor_met}" == "false" ]]; then
  echo "emit-gemma4-score.sh: BELOW FLOOR -- refusing to emit a score." >&2
  echo "  composite_score ${composite} did not meet composite_speedup_floor ${floor_value}." >&2
  echo "  This is an honest measurement that did not clear the floor, NOT a malformed seal:" >&2
  echo "  the run measured cleanly and scored under the bar. benchd reports this flag" >&2
  echo "  seal-only, so this script is the seam that enforces it." >&2
  exit 1
fi

# Drift tripwire: the floor benchd applied must be the floor the trusted manifest
# publishes. Two documents that are supposed to agree, checked cheaply.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CONTRACT_FLOOR="$(jq -r '.scoring.decodeSpeedupFloor // empty' "${REPO_ROOT}/benchmark.json" 2>/dev/null || true)"

if [[ -z "${CONTRACT_FLOOR}" ]]; then
  echo "emit-gemma4-score.sh: cannot read scoring.decodeSpeedupFloor from ${REPO_ROOT}/benchmark.json" >&2
  echo "  Refusing: without the manifest's floor there is nothing to check benchd's floor against." >&2
  exit 1
fi

if [[ "$(jq -n --argjson a "${floor_value}" --argjson b "${CONTRACT_FLOOR}" '$a == $b')" != "true" ]]; then
  echo "emit-gemma4-score.sh: FLOOR DRIFT -- benchd applied composite_speedup_floor ${floor_value}," >&2
  echo "  but benchmark.json scoring.decodeSpeedupFloor is ${CONTRACT_FLOOR}." >&2
  echo "  Refusing: the run was scored against a different bar than the published one." >&2
  exit 1
fi

metrics="$(jq -c '{
  composite: .per_cohort[0].composite,
  scored_batch_size: .per_cohort[0].batch_size,
  cohort_sha256: .per_cohort[0].cohort_sha256,
  parity_ok: .per_cohort[0].parity_ok,
  accepted_pair_count: .per_cohort[0].accepted_pair_count,
  raw_ratio_of_means: .per_cohort[0].raw_ratio_of_means
}' "${RESULTS_JSON}")"

jq -n --argjson score "${composite}" --argjson metrics "${metrics}" \
  '{score: $score, metrics: $metrics}' > "${SCORE_PATH}"

echo "emit-gemma4-score.sh: wrote score=${composite} to ${SCORE_PATH}" >&2
