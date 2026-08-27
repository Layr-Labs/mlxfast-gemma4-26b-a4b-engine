#!/usr/bin/env bash
# Unit test for .github/scripts/emit-gemma4-score.sh. Offline, no GPU, no
# benchd invocation -- exercises the emitter against synthetic sealed-results
# samples shaped like measure-job's PerCohort record (mlxfast-bench crates/benchctl/
# src/measure_job.rs).
#
# THE FIXTURES ARE THE REAL SCHEMA NOW. Case 1 carried `"composite": 1.1875`, a
# bare number, until 2026-08-26 -- a shape the pinned benchd has never emitted.
# That invented fixture is why the emitter's object/number bug survived a green
# test suite and only surfaced by killing a finished ranked round. The composite
# blocks below are copied from sealed results.json files produced by the pinned
# benchd (./benchd.pin sha256 e044e1f4...) on the ranked box, values and all.
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
      "composite": {
        "composite_score": 0.945956243476148,
        "composite_speedup_floor": 0.9,
        "composite_speedup_floor_met": true,
        "decode_gain": 0.9193183158707673,
        "prefill_gain": 1.0305912562853754
      }
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-with-composite.json" "${WORK}/score-1.json" >"${WORK}/case1.out" 2>&1; then
  score="$(jq -r '.score' "${WORK}/score-1.json" 2>/dev/null || echo "MISSING")"
  metrics_composite="$(jq -r '.metrics.composite.composite_score' "${WORK}/score-1.json" 2>/dev/null || echo "MISSING")"
  if [[ "${score}" != "0.945956243476148" ]]; then
    fail "case 1: expected score 0.945956243476148, got '${score}'"
  fi
  if [[ "${metrics_composite}" != "0.945956243476148" ]]; then
    fail "case 1: expected metrics.composite.composite_score 0.945956243476148, got '${metrics_composite}'"
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

# Case 3b: THE REGRESSION THAT KILLED A FINISHED ROUND. A bare-number composite
# is the shape this script wrongly assumed; benchd does not emit it. It must now
# be refused as a SCHEMA MISMATCH -- and must never again be read as a score.
cat > "${WORK}/results-bare-number.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 2,
      "composite": 1.1875
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-bare-number.json" "${WORK}/score-3b.json" >"${WORK}/case3b.out" 2>&1; then
  fail "case 3b: emitter exited 0 on a bare-number composite (should refuse as a schema mismatch)"
fi
if ! grep -q "SCHEMA MISMATCH" "${WORK}/case3b.out"; then
  fail "case 3b: expected a SCHEMA MISMATCH refusal; got: $(cat "${WORK}/case3b.out")"
fi
if [[ -e "${WORK}/score-3b.json" ]]; then
  fail "case 3b: emitter wrote a score file despite refusing"
fi

# Case 3c: BELOW FLOOR. An honest, well-formed measurement that did not clear the
# floor must refuse -- benchd wires composite_speedup_floor_met to nothing, so
# this seam is the enforcement -- and must say so in its OWN words, not the
# schema-mismatch words.
cat > "${WORK}/results-below-floor.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 2,
      "composite": {
        "composite_score": 0.8421,
        "composite_speedup_floor": 0.9,
        "composite_speedup_floor_met": false,
        "decode_gain": 0.81,
        "prefill_gain": 0.95
      }
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-below-floor.json" "${WORK}/score-3c.json" >"${WORK}/case3c.out" 2>&1; then
  fail "case 3c: emitter exited 0 on a below-floor composite (should refuse)"
fi
if ! grep -q "BELOW FLOOR" "${WORK}/case3c.out"; then
  fail "case 3c: expected a BELOW FLOOR refusal; got: $(cat "${WORK}/case3c.out")"
fi
if grep -q "SCHEMA MISMATCH" "${WORK}/case3c.out"; then
  fail "case 3c: below-floor must not read as a schema mismatch; got: $(cat "${WORK}/case3c.out")"
fi
if [[ -e "${WORK}/score-3c.json" ]]; then
  fail "case 3c: emitter wrote a score file despite refusing"
fi

# Case 3d: FLOOR DRIFT. benchd applied a floor other than the one benchmark.json
# publishes -- the run was scored against a different bar, so refuse.
cat > "${WORK}/results-floor-drift.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 2,
      "composite": {
        "composite_score": 0.7,
        "composite_speedup_floor": 0.5,
        "composite_speedup_floor_met": true,
        "decode_gain": 0.7,
        "prefill_gain": 0.7
      }
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-floor-drift.json" "${WORK}/score-3d.json" >"${WORK}/case3d.out" 2>&1; then
  fail "case 3d: emitter exited 0 on a drifted floor (should refuse)"
fi
if ! grep -q "FLOOR DRIFT" "${WORK}/case3d.out"; then
  fail "case 3d: expected a FLOOR DRIFT refusal; got: $(cat "${WORK}/case3d.out")"
fi

# Case 3e: composite object missing the floor flag entirely -> schema mismatch,
# because a floor cannot be enforced against a flag that is not there.
cat > "${WORK}/results-no-floor-flag.json" <<'EOF'
{
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "parity_ok": true,
      "accepted_pair_count": 2,
      "composite": {
        "composite_score": 1.4,
        "decode_gain": 1.5,
        "prefill_gain": 1.1
      }
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-no-floor-flag.json" "${WORK}/score-3e.json" >"${WORK}/case3e.out" 2>&1; then
  fail "case 3e: emitter exited 0 on a composite with no floor flag (should refuse)"
fi
if ! grep -q "SCHEMA MISMATCH" "${WORK}/case3e.out"; then
  fail "case 3e: expected a SCHEMA MISMATCH refusal; got: $(cat "${WORK}/case3e.out")"
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
