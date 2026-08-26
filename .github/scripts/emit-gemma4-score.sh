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
# prefill and decode windows. It is null only when the cohort accepted no pair
# or a window is degenerate, and every null one carries a
# `composite_absent_reason` -- so a null here is a real refusal to score, not a
# missing feature.
#
# This comment read "at the CURRENTLY PINNED benchd gitlink that field exists
# but is ALWAYS null" until 2026-08-26. Both halves had gone stale: the dist
# re-cut landed the per-stream score path, and benchd has been a pinned PREBUILT
# (./benchd.pin + tools/fetch-benchd.sh) with no submodule and no gitlink since
# before that. benchmark.yml already said the opposite in its own header, so the
# two documents contradicted each other.
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

# `.per_cohort[0].composite` reads as `null` uniformly whether the key is
# fully absent from the schema (today, at the pinned benchd gitlink) or
# present-but-null (the future PR #182 shape) -- jq does not distinguish, and
# this script does not need to: either way, no composite means refuse.
composite="$(jq -r '.per_cohort[0].composite // empty' "${RESULTS_JSON}")"

if [[ -z "${composite}" || "${composite}" == "null" ]]; then
  reason="$(jq -r '.per_cohort[0].composite_absent_reason // "composite is null in results.json and no composite_absent_reason was found (composite cohort scoring has landed but stays gated pending per-stream instrumentation -- see docs/participant-contract.md section 5.2)"' "${RESULTS_JSON}")"
  echo "emit-gemma4-score.sh: refusing to emit a score -- composite is absent." >&2
  echo "  reason: ${reason}" >&2
  echo "  refuse, not degrade: this script will not substitute the raw_ratio_of_means" >&2
  echo "  shared-cohort-window diagnostic for the ruled per-stream composite formula." >&2
  exit 1
fi

# A composite that is present must still be finite (score.ts requires
# z.number().finite()); a non-finite value is a benchd bug, not a score.
case "${composite}" in
  *[!0-9.eE+-]*|"")
    echo "emit-gemma4-score.sh: composite is not a plain finite number: ${composite}" >&2
    exit 1
    ;;
esac

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
