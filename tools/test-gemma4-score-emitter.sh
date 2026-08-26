#!/usr/bin/env bash
# Unit test for .github/scripts/emit-gemma4-score.sh. Offline, no GPU, no
# benchd invocation -- exercises the emitter against synthetic sealed-results
# samples shaped like measure-job's PerCohort record (mlxfast-bench crates/benchctl/
# src/measure_job.rs), both a real-composite-present case (the shape PR #182
# defined, not yet reachable end to end) and the current gated-absent-
# composite case (the pinned gitlink's actual shape, verified by direct grep
# -- see docs/participant-contract.md section 5.2).
#
# Usage: tools/test-gemma4-score-emitter.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMITTER="${SCRIPT_DIR}/.github/scripts/emit-gemma4-score.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $1" >&2
  failures=$((failures + 1))
}

# Case 1: composite present and finite -> emits {score, metrics} and exits 0.
cat > "${WORK}/results-with-composite.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "cohort_sha256": "deadbeef00000000000000000000000000000000000000000000000000ff",
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 4,
      "serial_seconds_per_token_mean": 0.012,
      "candidate_seconds_per_token_mean": 0.010,
      "raw_ratio_of_means": 1.2,
      "composite": 1.1875
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-with-composite.json" "${WORK}/score-1.json" >"${WORK}/case1.out" 2>&1; then
  score="$(jq -r '.score' "${WORK}/score-1.json" 2>/dev/null || echo "MISSING")"
  metrics_composite="$(jq -r '.metrics.composite' "${WORK}/score-1.json" 2>/dev/null || echo "MISSING")"
  if [[ "${score}" != "1.1875" ]]; then
    fail "case 1: expected score 1.1875, got '${score}'"
  fi
  if [[ "${metrics_composite}" != "1.1875" ]]; then
    fail "case 1: expected metrics.composite 1.1875, got '${metrics_composite}'"
  fi
  # score.ts's ScoreFileSchema requires `score` to parse as a finite number;
  # confirm the emitted file is valid JSON with a numeric (not string) score.
  if ! jq -e '.score | type == "number"' "${WORK}/score-1.json" >/dev/null 2>&1; then
    fail "case 1: score field is not a JSON number"
  fi
else
  fail "case 1: emitter exited nonzero on a valid composite; output: $(cat "${WORK}/case1.out")"
fi

# Case 2: composite absent (the CURRENT pinned-gitlink shape) -> refuses
# (nonzero exit), writes no score file, and does not silently fall back to
# raw_ratio_of_means.
cat > "${WORK}/results-no-composite.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "cohort_sha256": "deadbeef00000000000000000000000000000000000000000000000000ff",
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 4,
      "serial_seconds_per_token_mean": 0.012,
      "candidate_seconds_per_token_mean": 0.010,
      "raw_ratio_of_means": 1.2
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-no-composite.json" "${WORK}/score-2.json" >"${WORK}/case2.out" 2>&1; then
  fail "case 2: emitter exited 0 on an absent composite (should refuse)"
fi
if [[ -e "${WORK}/score-2.json" ]]; then
  fail "case 2: emitter wrote a score file despite refusing"
fi
if ! grep -q "refusing to emit a score" "${WORK}/case2.out"; then
  fail "case 2: refusal message missing expected text; got: $(cat "${WORK}/case2.out")"
fi

# Case 3: explicit composite_absent_reason is surfaced in the refusal.
cat > "${WORK}/results-absent-reason.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "composite": null,
      "composite_absent_reason": "per_stream_aggregate_source not yet implemented"
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-absent-reason.json" "${WORK}/score-3.json" >"${WORK}/case3.out" 2>&1; then
  fail "case 3: emitter exited 0 on a null composite (should refuse)"
fi
if ! grep -q "per_stream_aggregate_source not yet implemented" "${WORK}/case3.out"; then
  fail "case 3: expected composite_absent_reason to be echoed; got: $(cat "${WORK}/case3.out")"
fi

# Case 4: missing results.json -> refuses with a clear message, exit nonzero.
if "${EMITTER}" "${WORK}/does-not-exist.json" "${WORK}/score-4.json" >"${WORK}/case4.out" 2>&1; then
  fail "case 4: emitter exited 0 on a missing results file"
fi

# Case 5: malformed JSON -> refuses, exit nonzero.
echo "not json" > "${WORK}/malformed.json"
if "${EMITTER}" "${WORK}/malformed.json" "${WORK}/score-5.json" >"${WORK}/case5.out" 2>&1; then
  fail "case 5: emitter exited 0 on malformed JSON"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-gemma4-score-emitter.sh: all cases passed"
  exit 0
fi
echo "test-gemma4-score-emitter.sh: ${failures} case(s) failed" >&2
exit 1
