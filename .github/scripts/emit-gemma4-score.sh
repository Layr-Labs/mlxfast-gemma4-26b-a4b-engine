#!/usr/bin/env bash
#
# emit-gemma4-score.sh -- convert a benchd `measure-job` results.json into the
# {score, metrics} shape src/benchmark/score.ts requires at scorePath
# (Yukon's ScoreFileSchema: `score` a finite number, `metrics` optional JSON).
#
# TWO SERIES, TWO FIELDS, ONE SWITCH (2026-08-26). This track has two scored
# series and they publish DIFFERENT quantities:
#
#   batched_free_run_v1_2_b8  ->  per_cohort[0].composite.composite_score
#                                 (serial / MTP: the BATCHED COHORT series,
#                                 composite = prefill_gain^0.25 * decode_gain^0.75,
#                                 docs/participant-contract.md section 5.2)
#   free_run_v1_1             ->  aggregate.raw_decode_speedup_median
#                                 (the DFlash arm: SINGLE-STREAM, the even-n
#                                 median of the per-prompt raw ratio-of-means,
#                                 docs/participant-contract.md section 5.1.1)
#   anything else / absent    ->  refuse
#
# The switch is `results.timed_mode`, the SEALED series descriptor, read as an
# EXHAUSTIVE match, never as a fallback chain.
#
# WHY THE SWITCH AND NOT "WHICHEVER FIELD IS PRESENT". BOTH shapes carry an
# `aggregate.raw_decode_speedup_median`: on the batched path it is the median of
# the per-PAIR cohort ratios, on the single-stream path the median of the
# per-PROMPT ratios. They are different physical quantities with the same field
# name, and benchd's own series fence exists to stop them being pooled. A
# presence test would happily emit the cohort run's aggregate as if it were a
# single-stream score. So each branch additionally REFUSES the other's shape:
# a single-stream record that carries `per_cohort` is a shape mismatch, and so
# is a batched record with no cohort. Neither is repaired; both exit 1.
#
# Until the DFlash arm existed, a single-stream run reached this script with no
# cohort record and was refused, so the arm could not produce a score at all.
#
# ---------------------------------------------------------------------------
# THE COHORT BRANCH: `composite` IS AN OBJECT, NOT A NUMBER.
# ---------------------------------------------------------------------------
# This script read it as a bare number until 2026-08-26, and that is what killed
# a fully successful ranked round: the measurement finished, benchd sealed
# composite_score 0.9383 with the floor met, and this script refused it as "not
# a plain finite number" because jq handed it the whole object. The real shape,
# from the pinned benchd (`CompositeCohortScore` in
# crates/benchctl/src/measure_job.rs) and confirmed against sealed results.json
# files off the box:
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
# THE STRUCTURED-COMPOSITE READ, THE FINITE CHECK, THE FLOOR REFUSAL AND THE
# DRIFT TRIPWIRE ALL LIVE INSIDE THE COHORT BRANCH -- they are properties of the
# cohort series' composite object, not of "whatever this script scored". The
# single-stream branch publishes a plain number and carries its own checks.
#
# What has NOT changed is the refusal discipline: this script never fabricates a
# number from the `raw_ratio_of_means` diagnostic that is also present. That
# diagnostic is the shared-cohort-window ratio, not the ruled per-stream
# composite; substituting one for the other would silently score a different
# formula than the one David ruled.
#
# Usage: emit-gemma4-score.sh <results.json> <score-path>
# Exit:  0 on a real score emitted to <score-path>
#        1 on a missing/malformed results.json, an unknown/absent series, a
#          cross-series shape mismatch, a schema mismatch, a below-floor run,
#          a floor drift, or an absent value for the series
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

# The SEALED series descriptor. benchd always writes it (`Results::timed_mode`
# is a non-optional field), so an absent one is a malformed or foreign record,
# not an older shape to be tolerated.
TIMED_MODE_COHORT="batched_free_run_v1_2_b8"
TIMED_MODE_SINGLE_STREAM="free_run_v1_1"

timed_mode="$(jq -r '.timed_mode // empty' "${RESULTS_JSON}")"

if [[ -z "${timed_mode}" ]]; then
  echo "emit-gemma4-score.sh: refusing to emit a score -- results.json declares no timed_mode." >&2
  echo "  The series decides WHICH field is the score; without it there is no honest choice to make." >&2
  exit 1
fi

# A run whose two legs crossed series seals the MIXED descriptor rather than
# either tag. It is caught by the exhaustive match below and named there.
case "${timed_mode}" in
  "${TIMED_MODE_COHORT}")
    # ------------------------------------------------------------------
    # The batched cohort series -- serial and MTP.
    # ------------------------------------------------------------------
    # Cross-series shape check FIRST: a cohort record with no cohort is not a
    # cohort record, and must not silently fall through to any other field.
    if ! jq -e 'has("per_cohort") and (.per_cohort | type == "array") and (.per_cohort | length > 0)' \
      >/dev/null 2>&1 < "${RESULTS_JSON}"; then
      echo "emit-gemma4-score.sh: refusing to emit a score -- results.json declares timed_mode" >&2
      echo "  '${timed_mode}' but carries no per_cohort record. That is a shape mismatch, not a" >&2
      echo "  missing feature: this script will not read a single-stream field off a record that" >&2
      echo "  says it measured the batched cohort." >&2
      exit 1
    fi

    # `.per_cohort[0].composite` reads as `null` uniformly whether the key is
    # fully absent from the schema or present-but-null (what a zero-accepted-pair
    # cohort actually seals) -- jq does not distinguish, and this script does not
    # need to: either way, no composite means refuse.
    composite_obj="$(jq -c '.per_cohort[0].composite // empty' "${RESULTS_JSON}")"

    if [[ -z "${composite_obj}" || "${composite_obj}" == "null" ]]; then
      reason="$(jq -r '.per_cohort[0].composite_absent_reason // "composite is null in results.json and no composite_absent_reason was found (composite cohort scoring has landed but stays gated pending per-stream instrumentation -- see docs/participant-contract.md section 5.2)"' "${RESULTS_JSON}")"
      echo "emit-gemma4-score.sh: refusing to emit a score -- composite is absent." >&2
      echo "  reason: ${reason}" >&2
      echo "  refuse, not degrade: this script will not substitute the raw_ratio_of_means" >&2
      echo "  shared-cohort-window diagnostic for the ruled per-stream composite formula." >&2
      exit 1
    fi

    # The SCORE is the object's composite_score. A composite object without one
    # is a schema mismatch -- benchd changed shape under this script -- and is
    # refused as such, never silently treated as "no composite".
    # Guarded with a type check: a composite that is NOT an object (the bare
    # number this script used to assume) must reach the schema refusal below,
    # not abort jq.
    score="$(jq -r '.per_cohort[0].composite | if type == "object" then (.composite_score // empty) else empty end' "${RESULTS_JSON}")"

    if [[ -z "${score}" || "${score}" == "null" ]]; then
      echo "emit-gemma4-score.sh: SCHEMA MISMATCH -- per_cohort[0].composite is present but carries no composite_score." >&2
      echo "  composite: ${composite_obj}" >&2
      echo "  expected the CompositeCohortScore object sealed by the pinned benchd" >&2
      echo "  ({composite_score, composite_speedup_floor, composite_speedup_floor_met, decode_gain, prefill_gain})." >&2
      echo "  Refusing: a shape this script does not understand is not a score." >&2
      exit 1
    fi

    # A composite that is present must still be finite (score.ts requires
    # z.number().finite()); a non-finite value is a benchd bug, not a score.
    case "${score}" in
      *[!0-9.eE+-]*|"")
        echo "emit-gemma4-score.sh: composite_score is not a plain finite number: ${score}" >&2
        exit 1
        ;;
    esac

    # THE FLOOR, ENFORCED. benchd reports this flag and wires it to nothing; the
    # score seam is where a below-floor run has to stop, or the floor is
    # decoration.
    # NO `//` ON THESE TWO. jq's alternative operator treats `false` as absent,
    # so `composite_speedup_floor_met // empty` would turn the one value this
    # check exists to catch -- false -- into "field missing" and refuse with the
    # wrong message. Read them straight and test for null explicitly.
    floor_met="$(jq -r '.per_cohort[0].composite.composite_speedup_floor_met' "${RESULTS_JSON}")"
    floor_value="$(jq -r '.per_cohort[0].composite.composite_speedup_floor' "${RESULTS_JSON}")"

    if [[ "${floor_met}" != "true" && "${floor_met}" != "false" ]]; then
      echo "emit-gemma4-score.sh: SCHEMA MISMATCH -- composite carries no boolean composite_speedup_floor_met (got '${floor_met}')." >&2
      echo "  Refusing: the floor cannot be enforced against a flag that is not there." >&2
      exit 1
    fi

    if [[ "${floor_met}" == "false" ]]; then
      echo "emit-gemma4-score.sh: BELOW FLOOR -- refusing to emit a score." >&2
      echo "  composite_score ${score} did not meet composite_speedup_floor ${floor_value}." >&2
      echo "  This is an honest measurement that did not clear the floor, NOT a malformed seal:" >&2
      echo "  the run measured cleanly and scored under the bar. benchd reports this flag" >&2
      echo "  seal-only, so this script is the seam that enforces it." >&2
      exit 1
    fi

    # Drift tripwire: the floor benchd applied must be the floor the trusted
    # manifest publishes. Two documents that are supposed to agree, checked
    # cheaply.
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

    score_field="per_cohort[0].composite.composite_score"
    metrics="$(jq -c '{
      series: .timed_mode,
      composite: .per_cohort[0].composite,
      scored_batch_size: .per_cohort[0].batch_size,
      cohort_sha256: .per_cohort[0].cohort_sha256,
      parity_ok: .per_cohort[0].parity_ok,
      accepted_pair_count: .per_cohort[0].accepted_pair_count,
      raw_ratio_of_means: .per_cohort[0].raw_ratio_of_means,
      serial_seconds_per_token_mean: .per_cohort[0].serial_seconds_per_token_mean,
      candidate_seconds_per_token_mean: .per_cohort[0].candidate_seconds_per_token_mean,
      prefill_token_total: .per_cohort[0].prefill_token_total,
      decode_token_total: .per_cohort[0].decode_token_total,
      serial_prefill_window_seconds_mean: .per_cohort[0].serial_prefill_window_seconds_mean,
      candidate_prefill_window_seconds_mean: .per_cohort[0].candidate_prefill_window_seconds_mean,
      serial_decode_window_seconds_mean: .per_cohort[0].serial_decode_window_seconds_mean,
      candidate_decode_window_seconds_mean: .per_cohort[0].candidate_decode_window_seconds_mean
    }' "${RESULTS_JSON}")"
    # DRAFTING-DEPTH OBSERVABILITY (David 2026-08-26) -- cohort series sources.
    # PerCohort seals these from the FIRST ACCEPTED PAIR and omits them on a
    # die-5 cohort; `.mtp_depth` is a non-optional Results field.
    observability="$(jq -c '{
      effective_spec: .per_cohort[0].effective_spec,
      mtp_depth: .mtp_depth,
      effective_mean_draft_len: .per_cohort[0].effective_mean_draft_len,
      non_drafting_round_count: .per_cohort[0].non_drafting_round_count
    } | with_entries(select(.value != null))' "${RESULTS_JSON}")"
    ;;

  "${TIMED_MODE_SINGLE_STREAM}")
    # ------------------------------------------------------------------
    # The single-stream v1.1 free-run series -- the DFlash arm.
    # ------------------------------------------------------------------
    # Cross-series shape check FIRST, the mirror of the one above: benchd seals
    # `per_cohort: None` on this path, so a single-stream record carrying a
    # cohort is not a single-stream record.
    if jq -e 'has("per_cohort") and (.per_cohort != null)' >/dev/null 2>&1 < "${RESULTS_JSON}"; then
      echo "emit-gemma4-score.sh: refusing to emit a score -- results.json declares timed_mode" >&2
      echo "  '${timed_mode}' but also carries a per_cohort record. benchd seals no cohort on the" >&2
      echo "  single-stream path, so this record's two halves disagree about which series ran." >&2
      exit 1
    fi

    score="$(jq -r '.aggregate.raw_decode_speedup_median // empty' "${RESULTS_JSON}")"

    if [[ -z "${score}" || "${score}" == "null" ]]; then
      echo "emit-gemma4-score.sh: refusing to emit a score -- aggregate.raw_decode_speedup_median" >&2
      echo "  is absent on a '${timed_mode}' record. That field IS the single-stream published" >&2
      echo "  value (the even-n median of the per-prompt raw ratio-of-means); there is no second" >&2
      echo "  field to fall back to, and the cohort composite is a different series' quantity." >&2
      exit 1
    fi

    # Finite, for the same score.ts reason as the cohort branch. The
    # single-stream published value is a plain number, so there is no object
    # unwrap to do first -- but it still must not reach score.ts non-finite.
    case "${score}" in
      *[!0-9.eE+-]*|"")
        echo "emit-gemma4-score.sh: aggregate.raw_decode_speedup_median is not a plain finite number: ${score}" >&2
        exit 1
        ;;
    esac

    score_field="aggregate.raw_decode_speedup_median"
    # NO COHORT FIELD NAMES. Everything here is a single-stream seal; the floor
    # (0.90 on this series) and the published ceiling (5.0) are carried as
    # benchd sealed them, and are reported, not applied -- clamping here would
    # make this script a second scoring authority.
    metrics="$(jq -c '{
      series: .timed_mode,
      raw_decode_speedup_median: .aggregate.raw_decode_speedup_median,
      scoring_aggregation: .aggregate.scoring_aggregation,
      median_rule: .aggregate.median_rule,
      score_anchor: .aggregate.score_anchor,
      prompt_count: (.aggregate.raw_ratios | length),
      decode_speedup_floor: .aggregate.decode_speedup_floor,
      decode_speedup_floor_met: .aggregate.decode_speedup_floor_met,
      published_speedup_ceiling: .aggregate.published_speedup_ceiling,
      per_prompt: [.per_prompt[] | {prompt_index, prompt_sha256, parity_ok, accepted_pair_count, serial_seconds_per_token_mean, mtp_seconds_per_token_mean, raw_ratio_of_means}]
    }' "${RESULTS_JSON}")"
    # DRAFTING-DEPTH OBSERVABILITY (David 2026-08-26) -- single-stream sources.
    # This series seals no cohort record, so the equivalents come from
    # `aggregate` and `per_prompt`:
    #
    #   effective_spec            <- per_prompt[0].effective_spec (the engine's
    #                                echoed spec; benchd itself seals the cohort
    #                                form from the first accepted pair, and this
    #                                mirrors that convention for prompt order)
    #   mtp_depth                 <- .mtp_depth (same non-optional Results field)
    #   non_drafting_round_count  <- aggregate.non_drafting_round_count_total
    #                                (MAPPED: this series seals the run TOTAL
    #                                summed across prompts, not a first-pair
    #                                sample)
    #
    # effective_mean_draft_len IS DELIBERATELY OMITTED HERE. benchd seals no
    # scalar realized draft length on this series -- only the per-prompt vector
    # `aggregate.effective_mean_draft_len_by_prompt`. Averaging it would be
    # COMPUTING a published number, and emitting one prompt's value under a
    # run-level key would misreport it. The vector is forwarded under its own
    # sealed name so nothing is lost, and the canonical scalar key is absent so
    # the website renders a dash rather than a number benchd never sealed.
    observability="$(jq -c '{
      effective_spec: .per_prompt[0].effective_spec,
      mtp_depth: .mtp_depth,
      non_drafting_round_count: .aggregate.non_drafting_round_count_total,
      effective_mean_draft_len_by_prompt: .aggregate.effective_mean_draft_len_by_prompt
    } | with_entries(select(.value != null))' "${RESULTS_JSON}")"
    ;;

  *)
    echo "emit-gemma4-score.sh: refusing to emit a score -- results.json declares timed_mode" >&2
    echo "  '${timed_mode}', which is not a series this track scores." >&2
    echo "  Scored series: '${TIMED_MODE_COHORT}' (serial/MTP, per_cohort[0].composite.composite_score) and" >&2
    echo "  '${TIMED_MODE_SINGLE_STREAM}' (DFlash, aggregate.raw_decode_speedup_median)." >&2
    echo "  A mixed-series descriptor names a run whose legs crossed regimes and is never scored." >&2
    exit 1
    ;;
esac

# PURE ADDITIVE: the observability keys are merged over the series metrics.
# Every value is FORWARDED from a field the pinned benchd already seals -- none
# is computed, defaulted or invented -- and null-valued keys were dropped above
# so an unsealed field is ABSENT rather than null.
metrics="$(jq -cn --argjson m "${metrics}" --argjson o "${observability}" '$m + $o')"

jq -n --argjson score "${score}" --argjson metrics "${metrics}" \
  '{score: $score, metrics: $metrics}' > "${SCORE_PATH}"

echo "emit-gemma4-score.sh: wrote score=${score} (${timed_mode}, ${score_field}) to ${SCORE_PATH}" >&2
