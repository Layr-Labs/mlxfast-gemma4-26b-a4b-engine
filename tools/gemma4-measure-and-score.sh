#!/usr/bin/env bash
#
# gemma4-measure-and-score.sh -- benchmark.json's benchmarkCommand /
# preSubmitCommand entry point for track gemma4-26b-a4b-mlx-v1.
#
# WHY THIS EXISTS INSTEAD OF THE QWEN SHAPE. The qwen manifest's
# benchmarkCommand targets the benchctl-iterate FACADE (vendored here as
# tools/benchmark.sh; upstream mlxfast-bench scripts/benchmark.sh).
# `benchctl iterate` (mlxfast-bench crates/benchctl/src/main.rs) is the
# single-run legacy path and never touches scored_batch_size / per_cohort /
# ScoredBatchPoint -- those live only in the measure_job module
# (mlxfast-bench crates/benchctl/src/measure_job.rs), reachable through the SEPARATE
# `benchctl measure-job` subcommand ("Option-A seam 2", per
# fixtures/gemma4_26b_a4b_track.json scoring_semantics). The facade has no
# mode that reaches it, so this track's benchmarkCommand cannot be the qwen
# facade invocation with the track name swapped -- it has to target
# measure-job directly, and then convert measure-job's results.json into the
# {score, metrics} shape src/benchmark/score.ts requires (Yukon does not read
# results.json itself).
#
# This script is TRUSTED-side tooling, like the repo-root benchmark.sh proxy
# in the qwen engine repo -- it is NOT in editablePaths (see
# docs/participant-contract.md section 2), so a submission cannot rewrite the
# measurement pipeline from inside its own archive.
#
# WHAT IS AND IS NOT REAL HERE.
#   REAL: every measure-job flag below is copied from
#     `benchctl measure-job --help` at the pinned benchd commit (./benchd.pin),
#     not invented. The score-conversion step is a real, testable jq/shell
#     transform (see tools/test-gemma4-score-emitter.sh).
#   NOT YET RUNNABLE END TO END, and this script says so rather than hiding
#   it: (1) fixtures/gemma4_26b_a4b_track.json timed_prompt_pool[] is
#   PENDING-ORGANIZER (box-only, unarmed), so there are no real --golden
#   files to pass; (2) that same fixture sets official_scoring_enabled to
#   false, and the pinned benchd refuses to seal an official scoring artifact
#   for an unarmed track (absent counts as unarmed) -- see
#   docs/participant-contract.md section 5.5. The `composite` field this track
#   scores on IS produced by the pinned commit, from benchd's own
#   parent-clocked windows, so the score path is not the blocker; the goldens
#   and the arm flag are. The score-conversion step below REFUSES (nonzero
#   exit) rather than fabricate a number while either is missing -- refuse,
#   not degrade, matching the kv_backend and byte-budget precedent elsewhere
#   in this repository.
#   --target-pairs below is 2, the RULED contest parameter (David, "do 2",
#   2026-08-24; benchmark.json scoring.pairsPerCohort and
#   docs/gemma4-port-notes.md section 9.1 are the authority on the ruling and
#   its citation chain). The PIN NOW AGREES: the benchd pin was advanced
#   to include 047e21833a66264310307e1cb86ae3a290b0fc27 (PR #184,
#   'pairs_per_cohort: 4 -> 2') in the same PR that flipped both
#   occurrences of `--target-pairs` below from 4 to 2, resolving the
#   pinned-vs-ruled discrepancy documented there (an official-shaped
#   invocation declaring --target-pairs 2 is no longer refused at the pin).
#
# Usage:
#   ./tools/gemma4-measure-and-score.sh                 # full measure + score
#   ./tools/gemma4-measure-and-score.sh --preflight-only # measure-job prereqs only, no GPU
#
# Env:
#   MLXFAST_SCORE_PATH               Where to write the {score, metrics} JSON.
#                                     Defaults to score.json (benchmark.json
#                                     always sets this explicitly).
#   MLXFAST_GEMMA4_BASELINE_WORKSPACE
#                                     Baseline (serial-control) workspace passed
#                                     to `measure-job --baseline`. REQUIRED for a
#                                     real (non-preflight) run; there is no
#                                     default because there is no verified
#                                     convention yet for where a pinned baseline
#                                     clone lives on this repository's boxes.
#   MLXFAST_GEMMA4_GOLDEN_DIR         Directory holding the 8 hidden timed-pool
#                                     golden files, one per
#                                     timed_prompt_pool[] entry. Unset/empty by
#                                     construction until the pool is armed
#                                     (box-only).
#   MLXFAST_WEIGHTS_PATH              Transformed weights directory, passed as
#                                     `measure-job --weights`. Default: ./weights
#                                     -- the SAME convention tools/benchmark.sh
#                                     already uses and the directory ./setup.sh
#                                     transforms into. benchd's own fallback is
#                                     its QMTP_TARGET_DIR env, which NOTHING in
#                                     this repository sets; passing the flag from
#                                     the repo's own convention is what makes the
#                                     documented invocation self-sufficient.
#   MLXFAST_CORRECTNESS_GOLDEN_PATH   The staged hidden correctness golden,
#                                     passed as `measure-job
#                                     --correctness-golden`. REQUIRED for a real
#                                     run: fixtures/gemma4_26b_a4b_track.json
#                                     pins `hidden_correctness_golden`, and
#                                     benchd is fail-closed BOTH ways -- a
#                                     fixture that pins the golden REQUIRES the
#                                     flag (die 8, pre-GPU). Box-only, and
#                                     tools/ranked-box-preflight.sh pin-verifies
#                                     it against the fixture before this runs.
#   MLXFAST_GEMMA4_MEASURE_OUT_DIR    measure-job --out directory.
#                                     Default: ./benchmark-results-local
#   BENCHCTL                         Path to the benchctl binary. Default: the
#                                     binary ./tools/fetch-benchd.sh resolves and
#                                     hash-verifies against ./benchd.pin
#                                     (benchd-bin/benchctl). No cargo build: the
#                                     ranked box has no Rust toolchain, so benchd
#                                     ships prebuilt and pinned by sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *)
      echo "gemma4-measure-and-score.sh: unrecognized argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# Resolve the PINNED benchctl. This used to be a gitlink check plus a
# cargo-built binary under benchd/target/release; benchd is now a pinned
# PREBUILT (./benchd.pin + ./tools/fetch-benchd.sh) and there is no submodule and
# no cargo step. fetch-benchd.sh accepts an already-present benchd-bin/benchctl
# whose sha256 and bytes match the pin -- the offline path on the ranked box,
# which has no Rust toolchain -- and otherwise downloads and verifies it. It
# never yields an unverified binary, so this refuses rather than measuring
# against unpinned scoring code.
#
# Stronger than the check it replaces: the gitlink comparison bound the
# checked-out COMMIT only, so a benchd at the pinned commit with locally edited
# sources still passed. The sha256 binds the bytes that actually run.
#
# BENCHCTL= from the caller is honoured and NOT hash-checked: that is a
# deliberate "use this other binary" for benchd development.
if [[ -z "${BENCHCTL:-}" ]]; then
  BENCHCTL="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
if [[ ! -x "${BENCHCTL}" ]]; then
  echo "gemma4-measure-and-score.sh: benchctl not found at ${BENCHCTL}." >&2
  echo "  fetch it: ./tools/fetch-benchd.sh   (resolves ./benchd.pin, verifies sha256)" >&2
  echo "  or set BENCHCTL to an existing binary." >&2
  exit 1
fi

CONTRACT="${SCRIPT_DIR}/fixtures/gemma4_26b_a4b_track.json"
OUT_DIR="${MLXFAST_GEMMA4_MEASURE_OUT_DIR:-${SCRIPT_DIR}/benchmark-results-local}"
mkdir -p "${OUT_DIR}"

# --weights. benchd resolves the transformed weights from `--weights` or, when
# the flag is absent, from its own QMTP_TARGET_DIR env -- and fails closed when
# NEITHER is set. This script passed neither and nothing in this repository sets
# QMTP_TARGET_DIR, so the documented invocation could only ever work if an
# operator exported that variable by hand. It now passes the flag, resolved from
# the convention this repository already has: MLXFAST_WEIGHTS_PATH, defaulting
# to ./weights (tools/benchmark.sh does exactly this, and ./setup.sh transforms
# into that directory). No new variable is coined.
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${SCRIPT_DIR}/weights}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # measure-job's own pre-GPU prereq/quiesce checks, no engine spawned. Still
  # requires --candidate/--baseline to resolve (they are validated as real
  # workspaces before preflight runs), so this is honest, not a rubber stamp.
  exec "${BENCHCTL}" measure-job \
    --contract "${CONTRACT}" \
    --candidate "${SCRIPT_DIR}" \
    --baseline "${MLXFAST_GEMMA4_BASELINE_WORKSPACE:-${SCRIPT_DIR}}" \
    --weights "${WEIGHTS_PATH}" \
    --min-pairs 2 --target-pairs 2 \
    --tag "gemma4-preflight" \
    --out "${OUT_DIR}" \
    --preflight-only
fi

if [[ -z "${MLXFAST_GEMMA4_BASELINE_WORKSPACE:-}" ]]; then
  echo "gemma4-measure-and-score.sh: MLXFAST_GEMMA4_BASELINE_WORKSPACE is required for a real run." >&2
  echo "  (there is no default: no verified convention yet for a pinned on-box baseline clone)" >&2
  exit 1
fi

if [[ -z "${MLXFAST_GEMMA4_GOLDEN_DIR:-}" || ! -d "${MLXFAST_GEMMA4_GOLDEN_DIR}" ]]; then
  cat >&2 <<'EOF'
gemma4-measure-and-score.sh: MLXFAST_GEMMA4_GOLDEN_DIR is unset or missing.
  This track's timed_prompt_pool is UNARMED (fixtures/gemma4_26b_a4b_track.json
  timed_prompt_pool[] entries are PENDING-ORGANIZER sentinels). The 8 hidden
  golden files this command needs do not exist publicly and are authored
  box-only (docs/gemma4-port-notes.md section 6.2-6.3). There is nothing this
  script can do here except refuse.
EOF
  exit 1
fi

# The hidden correctness golden. The fixture pins `hidden_correctness_golden`,
# and benchd is FAIL-CLOSED BOTH WAYS: a fixture that pins it REQUIRES
# --correctness-golden (die 8, pre-GPU), and passing the flag against a fixture
# that pins none is equally refused. This script passed it never, so the
# documented invocation died pre-GPU on an armed fixture unless an operator
# added the flag by hand. MLXFAST_CORRECTNESS_GOLDEN_PATH is the existing
# convention -- tools/ranked-box-preflight.sh already pin-verifies exactly that
# variable against the fixture's {sha256, bytes} -- so the flag is wired from it
# and REQUIRED here, mirroring benchd's own rule rather than deferring to it.
if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" || ! -f "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" ]]; then
  echo "gemma4-measure-and-score.sh: MLXFAST_CORRECTNESS_GOLDEN_PATH is unset or missing." >&2
  echo "  fixtures/gemma4_26b_a4b_track.json pins hidden_correctness_golden, so benchd REQUIRES" >&2
  echo "  --correctness-golden and refuses (die 8, pre-GPU) without it. The golden is box-only;" >&2
  echo "  tools/ranked-box-preflight.sh pin-verifies this path against the fixture." >&2
  exit 1
fi

if [[ ! -d "${WEIGHTS_PATH}" ]]; then
  echo "gemma4-measure-and-score.sh: the transformed weights directory is missing: ${WEIGHTS_PATH}" >&2
  echo "  Run ./setup.sh, or set MLXFAST_WEIGHTS_PATH to the transformed weights directory." >&2
  exit 1
fi

golden_args=()
for f in "${MLXFAST_GEMMA4_GOLDEN_DIR}"/*.json; do
  [[ -e "${f}" ]] || continue
  golden_args+=(--golden "${f}")
done
if [[ "${#golden_args[@]}" -eq 0 ]]; then
  echo "gemma4-measure-and-score.sh: no *.json golden files found in ${MLXFAST_GEMMA4_GOLDEN_DIR}" >&2
  exit 1
fi

RESULTS_JSON="${OUT_DIR}/results.json"

# Real, verified-against --help argv (mlxfast-bench crates/benchctl/src/main.rs
# MEASURE_JOB_USAGE): required flags first, then the pool of goldens.
"${BENCHCTL}" measure-job \
  --contract "${CONTRACT}" \
  --candidate "${SCRIPT_DIR}" \
  --baseline "${MLXFAST_GEMMA4_BASELINE_WORKSPACE}" \
  --weights "${WEIGHTS_PATH}" \
  --correctness-golden "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" \
  "${golden_args[@]}" \
  --min-pairs 2 --target-pairs 2 \
  --tag "gemma4-local-$(date -u +%Y%m%dT%H%M%SZ)" \
  --out "${OUT_DIR}"

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
"${SCRIPT_DIR}/.github/scripts/emit-gemma4-score.sh" "${RESULTS_JSON}" "${SCORE_PATH}"
