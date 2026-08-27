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
  "timed_mode": "batched_free_run_v1_2_b8",
  "mtp_depth": 2,
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
      "effective_mean_draft_len": 1.008,
      "non_drafting_round_count": 119,
      "effective_spec": { "mode": "mtp", "mtp": { "depth": 2 } },
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
  "timed_mode": "batched_free_run_v1_2_b8",
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
  "timed_mode": "batched_free_run_v1_2_b8",
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
  "timed_mode": "batched_free_run_v1_2_b8",
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
  "timed_mode": "batched_free_run_v1_2_b8",
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
  "timed_mode": "batched_free_run_v1_2_b8",
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
  "timed_mode": "batched_free_run_v1_2_b8",
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

# ---------------------------------------------------------------------------
# THE SINGLE-STREAM SERIES (the DFlash arm) and the CROSS-SERIES REFUSALS.
#
# The load-bearing property is not "a single-stream record scores". It is that
# NEITHER series can be scored through the other's field. A one-line emitter
# that read whichever field happened to be present would pass case 6 and fail
# cases 8 and 9 -- which is why 8 and 9 exist.
# ---------------------------------------------------------------------------

# A well-formed single-stream record, shaped like benchd's free-run seal:
# `timed_mode` free_run_v1_1, an `aggregate`, and NO per_cohort at all
# (measure_job.rs seals `per_cohort: None` on this path).
cat > "${WORK}/results-single-stream.json" <<'EOF'
{
  "timed_mode": "free_run_v1_1",
  "mtp_depth": 1,
  "per_prompt": [
    { "effective_spec": { "mode": "dflash", "dflash": {} }, "effective_mean_draft_len": 1.0 }
  ],
  "aggregate": {
    "raw_decode_speedup_median": 1.42,
    "raw_ratios": [1.30, 1.38, 1.40, 1.44, 1.45, 1.47, 1.50, 1.55],
    "scoring_aggregation": "median-of-per-prompt",
    "median_rule": "even-n",
    "score_anchor": "serial-one",
    "decode_speedup_floor": 0.90,
    "decode_speedup_floor_met": true,
    "published_speedup_ceiling": 5.0,
    "effective_mean_draft_len_by_prompt": [1.00, 1.01, 1.00, 1.02, 1.00, 1.01, 1.00, 1.00],
    "non_drafting_round_count_total": 240
  }
}
EOF

# Case 6: single-stream record -> emits the published median, exit 0.
if "${EMITTER}" "${WORK}/results-single-stream.json" "${WORK}/score-6.json" >"${WORK}/case6.out" 2>&1; then
  score="$(jq -r '.score' "${WORK}/score-6.json" 2>/dev/null || echo "MISSING")"
  if [[ "${score}" != "1.42" ]]; then
    fail "case 6: expected score 1.42 from aggregate.raw_decode_speedup_median, got '${score}'"
  fi
  if ! jq -e '.score | type == "number"' "${WORK}/score-6.json" >/dev/null 2>&1; then
    fail "case 6: score field is not a JSON number"
  fi
  if [[ "$(jq -r '.metrics.series' "${WORK}/score-6.json")" != "free_run_v1_1" ]]; then
    fail "case 6: metrics.series does not name the single-stream series"
  fi
  if [[ "$(jq -r '.metrics.raw_decode_speedup_median' "${WORK}/score-6.json")" != "1.42" ]]; then
    fail "case 6: metrics.raw_decode_speedup_median missing or wrong"
  fi
  # The floor and the ceiling are REPORTED, not applied. Compared NUMERICALLY:
  # jq preserves the input's literal spelling ("0.90", "5.0"), so a string
  # compare would pin the fixture's formatting rather than the value.
  if ! jq -e '.metrics.decode_speedup_floor == 0.9' "${WORK}/score-6.json" >/dev/null 2>&1; then
    fail "case 6: metrics.decode_speedup_floor missing or wrong"
  fi
  if ! jq -e '.metrics.published_speedup_ceiling == 5.0' "${WORK}/score-6.json" >/dev/null 2>&1; then
    fail "case 6: metrics.published_speedup_ceiling missing or wrong"
  fi
  # The score is NOT clamped to the floor/ceiling here -- the overlay applies
  # them. 1.42 sits between the two, so this case cannot detect clamping; the
  # assertion that matters is that the emitted score equals the sealed field,
  # checked above.
else
  fail "case 6: emitter refused a valid single-stream record; output: $(cat "${WORK}/case6.out")"
fi

# Case 7: the single-stream metrics carry NO cohort field names. A DFlash
# record labelled with cohort fields is what would let the two series be pooled
# downstream, so the absence is asserted rather than assumed.
for cohort_field in composite scored_batch_size cohort_sha256 parity_ok \
  accepted_pair_count raw_ratio_of_means; do
  if jq -e --arg k "${cohort_field}" '.metrics | has($k)' "${WORK}/score-6.json" >/dev/null 2>&1; then
    fail "case 7: single-stream metrics carry the cohort field name '${cohort_field}'"
  fi
done

# Case 8: CROSS-SERIES, direction A -- a COHORT record must never be scored
# through the single-stream field. Here the cohort record's composite is absent
# but an `aggregate.raw_decode_speedup_median` IS present (benchd seals one on
# both paths); an emitter that fell back to it would emit a cohort run's
# per-pair median as if it were a single-stream score.
cat > "${WORK}/results-cohort-with-aggregate.json" <<'EOF'
{
  "timed_mode": "batched_free_run_v1_2_b8",
  "aggregate": {
    "raw_decode_speedup_median": 9.99
  },
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "composite": null,
      "composite_absent_reason": "the cohort accepted no pair"
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-cohort-with-aggregate.json" "${WORK}/score-8.json" >"${WORK}/case8.out" 2>&1; then
  fail "case 8: emitter scored a cohort record through the single-stream field"
fi
if [[ -e "${WORK}/score-8.json" ]]; then
  fail "case 8: emitter wrote a score file despite refusing"
fi
if grep -q "9.99" "${WORK}/case8.out"; then
  fail "case 8: the single-stream aggregate leaked into the cohort refusal path"
fi

# Case 9: CROSS-SERIES, direction B -- a SINGLE-STREAM record must never be
# scored through the cohort field, and a record that declares free_run_v1_1
# while carrying a per_cohort is internally inconsistent. Refuse; do not pick a
# half to believe.
cat > "${WORK}/results-single-stream-with-cohort.json" <<'EOF'
{
  "timed_mode": "free_run_v1_1",
  "aggregate": {
    "raw_decode_speedup_median": 1.42
  },
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "composite": 1.1875
    }
  ]
}
EOF
if "${EMITTER}" "${WORK}/results-single-stream-with-cohort.json" "${WORK}/score-9.json" >"${WORK}/case9.out" 2>&1; then
  fail "case 9: emitter scored a record whose two halves disagree about the series"
fi
if [[ -e "${WORK}/score-9.json" ]]; then
  fail "case 9: emitter wrote a score file despite refusing"
fi
if ! grep -q "per_cohort" "${WORK}/case9.out"; then
  fail "case 9: refusal does not name the contradicting field; got: $(cat "${WORK}/case9.out")"
fi

# Case 10: a single-stream record with NO published median refuses, and does
# not reach for any other field.
cat > "${WORK}/results-single-stream-no-median.json" <<'EOF'
{
  "timed_mode": "free_run_v1_1",
  "aggregate": {
    "mtp_decode_speedup": 1.9,
    "mtp_decode_speedup_median": 1.8
  }
}
EOF
if "${EMITTER}" "${WORK}/results-single-stream-no-median.json" "${WORK}/score-10.json" >"${WORK}/case10.out" 2>&1; then
  fail "case 10: emitter scored a single-stream record with no raw_decode_speedup_median"
fi
if grep -qE "1\.9|1\.8" "${WORK}/case10.out"; then
  fail "case 10: a diagnostic aggregate leaked into the refusal path"
fi

# Case 11: an UNKNOWN series refuses by name -- including the MIXED descriptor,
# which names a run whose legs crossed regimes and is never scored.
cat > "${WORK}/results-mixed.json" <<'EOF'
{
  "timed_mode": "mixed:teacher_forced_v1_serial+free_run_v1_1_candidate",
  "aggregate": { "raw_decode_speedup_median": 1.42 },
  "per_cohort": [ { "cohort_index": 0, "composite": 1.1875 } ]
}
EOF
if "${EMITTER}" "${WORK}/results-mixed.json" "${WORK}/score-11.json" >"${WORK}/case11.out" 2>&1; then
  fail "case 11: emitter scored a mixed-series record"
fi
if ! grep -q "mixed:" "${WORK}/case11.out"; then
  fail "case 11: refusal does not name the offending series; got: $(cat "${WORK}/case11.out")"
fi

# ===========================================================================
# DRAFTING-DEPTH OBSERVABILITY (David 2026-08-26). The website needs the
# DECLARED depth and the REALIZED draft length side by side -- tonight's
# baseline sealed declared 2 against a realized ~1.008, and that gap is the
# whole point. Every key is FORWARDED from a field the pinned benchd already
# seals; nothing here is computed or defaulted, so these cases also pin that
# the emitter never invents a number when benchd sealed none.
# ===========================================================================

# Case 13: the cohort series forwards all four canonical keys from the record.
if ! "${EMITTER}" "${WORK}/results-with-composite.json" "${WORK}/score-13.json" \
     >"${WORK}/case13.out" 2>&1; then
  fail "case 13: emitter refused a well-formed cohort record: $(cat "${WORK}/case13.out")"
fi
m13="$(jq -c '.metrics' "${WORK}/score-13.json")"
if [[ "$(jq -r '.mtp_depth' <<<"${m13}")" != "2" ]]; then
  fail "case 13: expected metrics.mtp_depth 2, got: ${m13}"
fi
if [[ "$(jq -r '.effective_mean_draft_len' <<<"${m13}")" != "1.008" ]]; then
  fail "case 13: expected metrics.effective_mean_draft_len 1.008, got: ${m13}"
fi
if [[ "$(jq -r '.non_drafting_round_count' <<<"${m13}")" != "119" ]]; then
  fail "case 13: expected metrics.non_drafting_round_count 119, got: ${m13}"
fi
if [[ "$(jq -c '.effective_spec' <<<"${m13}")" != '{"mode":"mtp","mtp":{"depth":2}}' ]]; then
  fail "case 13: expected the sealed effective_spec forwarded verbatim, got: ${m13}"
fi
# THE POINT OF THE FEATURE: declared and realized are both present and DIFFER.
# A regression that forwarded the declared depth as the realized length would
# make these equal and would otherwise look fine.
if [[ "$(jq -r '.mtp_depth' <<<"${m13}")" == "$(jq -r '.effective_mean_draft_len' <<<"${m13}")" ]]; then
  fail "case 13: declared depth and realized draft length must be distinct fields: ${m13}"
fi

# Case 14: the single-stream series forwards its aggregate-side equivalents
# under the same canonical keys, and OMITS the one it does not seal.
if ! "${EMITTER}" "${WORK}/results-single-stream.json" "${WORK}/score-14.json" \
     >"${WORK}/case14.out" 2>&1; then
  fail "case 14: emitter refused a well-formed single-stream record: $(cat "${WORK}/case14.out")"
fi
m14="$(jq -c '.metrics' "${WORK}/score-14.json")"
if [[ "$(jq -r '.mtp_depth' <<<"${m14}")" != "1" ]]; then
  fail "case 14: expected metrics.mtp_depth 1, got: ${m14}"
fi
# MAPPED from aggregate.non_drafting_round_count_total (this series seals the
# run total, not a first-pair sample) onto the canonical key.
if [[ "$(jq -r '.non_drafting_round_count' <<<"${m14}")" != "240" ]]; then
  fail "case 14: expected metrics.non_drafting_round_count 240 (mapped from the aggregate total), got: ${m14}"
fi
if [[ "$(jq -c '.effective_spec' <<<"${m14}")" != '{"mode":"dflash","dflash":{}}' ]]; then
  fail "case 14: expected the DFlash effective_spec forwarded verbatim, got: ${m14}"
fi
# OMISSION, NOT NULL. benchd seals no SCALAR realized draft length on this
# series -- only the per-prompt vector -- so the canonical scalar key must be
# ABSENT (the website renders a dash). `has()` is the check that matters: a
# null-valued key would read as present-but-empty and is exactly what the
# "never emit null-invented values" rule forbids.
if [[ "$(jq -r 'has("effective_mean_draft_len")' <<<"${m14}")" != "false" ]]; then
  fail "case 14: effective_mean_draft_len must be ABSENT on the single-stream series (benchd seals no scalar), got: ${m14}"
fi
# The vector IS forwarded, under its own sealed name, so nothing is lost.
# Compared NUMERICALLY: the merge forwards the sealed literals verbatim, so a
# string compare would pin jq's number formatting rather than the values.
if [[ "$(jq -n --argjson a "$(jq -c '.effective_mean_draft_len_by_prompt' <<<"${m14}")" \
             --argjson b '[1.00,1.01,1.00,1.02,1.00,1.01,1.00,1.00]' '$a == $b')" != "true" ]]; then
  fail "case 14: expected the sealed per-prompt draft-length vector forwarded, got: ${m14}"
fi
# NO COHORT FIELD NAMES leak in via the observability merge either.
for k in composite scored_batch_size cohort_sha256 accepted_pair_count; do
  if [[ "$(jq -r --arg k "${k}" 'has($k)' <<<"${m14}")" != "false" ]]; then
    fail "case 14: single-stream metrics must not carry cohort field '${k}': ${m14}"
  fi
done

# Case 12: an ABSENT timed_mode refuses. benchd always seals the field, so a
# record without one is malformed or foreign -- not an older shape to tolerate,
# because tolerating it is exactly how a series-blind fallback returns.
cat > "${WORK}/results-no-timed-mode.json" <<'EOF'
{
  "per_cohort": [ { "cohort_index": 0, "composite": 1.1875 } ]
}
EOF
if "${EMITTER}" "${WORK}/results-no-timed-mode.json" "${WORK}/score-12.json" >"${WORK}/case12.out" 2>&1; then
  fail "case 12: emitter scored a record that declares no timed_mode"
fi
if ! grep -q "timed_mode" "${WORK}/case12.out"; then
  fail "case 12: refusal does not name the missing field; got: $(cat "${WORK}/case12.out")"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-gemma4-score-emitter.sh: all cases passed"
  exit 0
fi
echo "test-gemma4-score-emitter.sh: ${failures} case(s) failed" >&2
exit 1
