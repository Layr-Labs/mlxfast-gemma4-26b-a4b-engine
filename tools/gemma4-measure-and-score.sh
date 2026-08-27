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
#   THE THERMAL CONTRACT IS BENCHD'S, NOT THIS SCRIPT'S. Every timed phase this
#   invocation produces runs behind benchd's own cool-down gate: a fixed 40C
#   threshold (sealed as cool_gate_c with source "wrapper-constant-40" -- a
#   calibration fixture offering a different value is ignored), a 900s ceiling
#   per phase, prefill and decode gated separately on fresh workers. Telemetry
#   is point-sampled with `macmon pipe -s1` at gate boundaries, NOT streamed;
#   see docs/benchmark-window-freeze.md. A phase is rejected -- with one gated
#   retry -- on throttling under load, missing telemetry, or token mismatch.
#   Nothing here can relax any of that, and nothing here should try to: the one
#   thing the caller owes benchd is a usable temperature reader, which on the
#   ranked box is hard-verified by tools/ranked-box-preflight.sh before setup.
#
#   HOW MANY GATED PHASES THAT IS, at the ruled pair count below: 4 pairs x 2
#   legs (serial control + candidate) = 8 timed legs, and prefill and decode
#   gate SEPARATELY, so 16 gated phases per ranked run. That count is what
#   .github/workflows/benchmark.yml's timeout-minutes budget is derived from;
#   the two must move together, and they are cross-referenced in both
#   directions so neither can drift silently.
#
#   --target-pairs below is 4, the RULED contest parameter. David, 2026-08-26,
#   verbatim: "you run it using 4 pairs instead of 2 of 8 batches" -- 8 prompts
#   x 4 pairs is challenger-grade sample mass. SUPERSESSION CHAIN, each link
#   superseding the one above it: (1) batch-8 brief D2, default 4; (2) David
#   2026-08-24 "do 2", RULED 2, which landed in benchd as PR #184 at
#   047e21833a66264310307e1cb86ae3a290b0fc27; (3) David 2026-08-26, RULED 4,
#   this value. benchmark.json scoring.pairsPerCohort and
#   docs/gemma4-port-notes.md section 9.1 are the authority on the ruling and
#   its citation chain.
#
#   THE PIN DOES NOT YET AGREE, AND THIS SCRIPT SAYS SO RATHER THAN HIDING IT.
#   The benchd side of this ruling (PAIRS_PER_COHORT_TARGET 2 -> 4) must MERGE
#   AND PUBLISH before benchd.pin can name a commit that compiles 4. Until that
#   pin advance lands, the pinned benchd compiles
#   `PAIRS_PER_COHORT_TARGET: usize = 2` and an official-shaped invocation
#   declaring --target-pairs 4 IS REFUSED at the pin, by name, before any GPU
#   work happens. That is the ruled-ahead-of-pin state the 8/24 ruling also
#   passed through: the refusal is the conformance gate working, not a defect,
#   and it fails LOUD and pre-measurement rather than silently scoring over the
#   wrong sample count. The follow-up commit that advances benchd.pin to the
#   published 4-pair benchmarker closes it.
#
#   --min-pairs moves with it (4, not 2), and the FLOOR IS ENFORCED AT THE PIN
#   exactly like the target: benchd refuses an official batched cohort run whose
#   min_pairs != PAIRS_PER_COHORT_TARGET, by name, at the same pre-GPU seam
#   (--local-dev still explores other floors). Until that gate landed, benchd's
#   only floor rule was the parse-time `target_pairs >= min_pairs`, so a floor
#   left at 2 would have let a run accept only 2 of the 4 ruled pairs and still
#   publish -- a median over half the support the ruling bought, and exactly the
#   silent degradation the ruling was made to avoid.
#
#   The --min-pairs 4 written below is therefore a BELT-AND-SUSPENDERS
#   DECLARATION of the ruled floor, not the thing that enforces it. It is worth
#   keeping in that role: it states the ruled value at the call site where an
#   operator reads it, and this script lives under tools/ -- organizer-
#   controlled, outside editablePaths -- so a submission cannot rewrite it. But
#   the guarantee that a published median covers 4 pairs comes from benchd's
#   refusal, which no argv can talk its way past.
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

# ---------------------------------------------------------------------------
# THE SELECTION SEAM. The submission declares which arm it wants measured; this
# script is the ONLY thing that turns that declaration into a benchmarker flag.
#
# WHY IT LIVES HERE. benchd takes the candidate leg's mode from
# `--candidate-spec` (or the `--mtp-depth` convenience flag) and from nowhere
# else -- it reads no field of the candidate workspace to choose a mode. This
# script composes that argv and is NOT an editable path, so a submission can
# state an arm and can never state a spec: everything between the declaration
# and the flag is trusted code, and the flag's SHAPE is a literal in this file
# rather than anything copied out of the submission. The fixture's
# `allowed_modes` and the engine's advertised `spec_modes` remain FENCES on top
# of this -- they decide which modes are admitted, never which one runs.
#
# THE CHANNEL. `dflash-head.manifest.json`, key `arm`, values `dflash` | `mtp`.
# It is an editable AND optional editable path, so a submission can write it.
# ABSENT FILE, absent key, or `mtp` => this script passes NO spec flag and the
# argv below is byte-for-byte what it was before the seam existed; benchd builds
# its own `{"mode":"mtp","mtp":{"depth":2}}` and seals spec_source
# "mtp-depth-default", exactly as today. `dflash` => the one literal spec below.
#
# FAIL-CLOSED, never a silent default: a manifest that is unreadable, is not
# JSON, is not a JSON object, declares a value outside the vocabulary, or
# declares `arm` in the OTHER head manifest (where nothing reads it) is a
# refusal that names the file and the value. "Your declaration was broken so we
# quietly scored you on the other arm" is the failure this exists to prevent --
# the same posture Sources/MLXFastTrustedHarness/Gemma4MTPHeadDeclaration.swift
# takes on the rest of the declaration, and that file owns the VOCABULARY the
# case list below is drift-checked against
# (tools/test-gemma4-arm-selection.sh).
#
# WHAT THIS DOES NOT DO. It selects; it does not ADMIT. Admission is the pinned
# benchmarker's, against the fixture's `allowed_modes`, and it is checked
# pre-GPU (die 8) whatever this script passes. Verified live against the pinned
# binary (./benchd.pin -> 6dc978b7): a `dspark` spec refuses by name and cites
# the list `["serial", "mtp", "dflash"]` as having come from the --contract
# fixture, while the dflash spec below clears that gate and proceeds exactly as
# the mtp spec does. So a declaration this script does not recognize never
# reaches the box, and a mode the fixture does not admit never reaches a timed
# window -- two independent fences, neither of them this file.
#
# jq is required BEFORE any measurement now, not only at score-emission time:
# the arm is read from JSON. Refuse up front rather than half way through.
if ! command -v jq >/dev/null 2>&1; then
  echo "gemma4-measure-and-score.sh: jq is required (the declared arm and the emitted score are both JSON)." >&2
  exit 1
fi

DFLASH_MANIFEST="${SCRIPT_DIR}/dflash-head.manifest.json"
MTP_MANIFEST="${SCRIPT_DIR}/mtp-head.manifest.json"

# The literal candidate spec for the DFlash arm. benchd's SpecConfig envelope is
# CLOSED (`deny_unknown_fields`) and its module-coherence gate requires the ONE
# block matching the mode to be present, so the empty `dflash` block is required
# and is the whole of it: every knob the arm has is the engine module's, and
# RuntimeWorkerSpecConfig.swift resolves depth and drafter identity from the
# BOUND drafter, never from this object.
DFLASH_CANDIDATE_SPEC='{"mode":"dflash","dflash":{}}'

# Sets DECLARED_ARM, or exits. Deliberately NOT a command substitution: a
# refusal inside `$(...)` would only end the subshell, and the difference
# between "the script stopped" and "the script continued with an empty arm" is
# the whole point of a fail-closed reader.
DECLARED_ARM=""
read_declared_arm() {
  # An `arm` key in the head manifest that does NOT declare the arm is a
  # refusal, not a no-op: a participant who put it in the wrong file must be
  # told, or they are scored on an arm they did not ask for.
  if [[ -f "${MTP_MANIFEST}" ]]; then
    if ! jq -e 'type == "object"' >/dev/null 2>&1 < "${MTP_MANIFEST}"; then
      echo "gemma4-measure-and-score.sh: ${MTP_MANIFEST} is not a JSON object; refusing rather than assuming an arm." >&2
      exit 1
    fi
    if jq -e 'has("arm")' >/dev/null 2>&1 < "${MTP_MANIFEST}"; then
      echo "gemma4-measure-and-score.sh: mtp-head.manifest.json sets 'arm', but the arm is declared in" >&2
      echo "  dflash-head.manifest.json only. Move the key there rather than leaving it where nothing reads it." >&2
      exit 1
    fi
  fi

  # An ABSENT manifest is the organizer default -- the one and only silent path.
  if [[ ! -f "${DFLASH_MANIFEST}" ]]; then
    DECLARED_ARM="mtp"
    return 0
  fi
  if ! jq -e 'type == "object"' >/dev/null 2>&1 < "${DFLASH_MANIFEST}"; then
    echo "gemma4-measure-and-score.sh: ${DFLASH_MANIFEST} is not a JSON object (or is not valid JSON);" >&2
    echo "  refusing rather than assuming an arm." >&2
    exit 1
  fi
  if ! jq -e 'has("arm")' >/dev/null 2>&1 < "${DFLASH_MANIFEST}"; then
    DECLARED_ARM="mtp"
    return 0
  fi
  # A present key must be a STRING: `jq -r` would otherwise stringify a number
  # or a boolean into something that looks like a value.
  if ! jq -e '.arm | type == "string"' >/dev/null 2>&1 < "${DFLASH_MANIFEST}"; then
    echo "gemma4-measure-and-score.sh: dflash-head.manifest.json sets 'arm' to a non-string value;" >&2
    echo "  it must be one of: dflash, mtp." >&2
    exit 1
  fi
  DECLARED_ARM="$(jq -r '.arm' < "${DFLASH_MANIFEST}")"
}

read_declared_arm

# The vocabulary. Case-sensitive on purpose: benchd's mode strings are
# lowercase, so 'MTP' is a typo and is told so rather than normalized. Keep this
# list identical to DeclaredArm's cases in
# Sources/MLXFastTrustedHarness/Gemma4MTPHeadDeclaration.swift --
# tools/test-gemma4-arm-selection.sh fails if the two drift.
spec_args=()
case "${DECLARED_ARM}" in
  mtp)
    # NO FLAG. This is the no-perturbation branch: the argv below must stay
    # byte-for-byte identical to the pre-seam invocation.
    ;;
  dflash)
    spec_args=(--candidate-spec "${DFLASH_CANDIDATE_SPEC}")
    ;;
  *)
    echo "gemma4-measure-and-score.sh: dflash-head.manifest.json declares arm '${DECLARED_ARM}', which is not a mode this track admits." >&2
    echo "  Declare one of: dflash, mtp -- or omit the key to run mtp." >&2
    exit 1
    ;;
esac
# ---------------------------------------------------------------------------

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
  #
  # The declared arm is passed HERE TOO, and that is the point of preflight: a
  # submission that declares an arm the pinned benchmarker will not admit finds
  # out at preSubmitCommand time, pre-GPU, instead of on the ranked box. On the
  # `mtp`/absent branch `spec_args` is empty and this argv is unchanged.
  # `${spec_args[@]+...}` is the bash-3.2-safe empty-array expansion (`set -u`
  # treats a bare `${a[@]}` on an empty array as unbound there).
  exec "${BENCHCTL}" measure-job \
    --contract "${CONTRACT}" \
    --candidate "${SCRIPT_DIR}" \
    --baseline "${MLXFAST_GEMMA4_BASELINE_WORKSPACE:-${SCRIPT_DIR}}" \
    --weights "${WEIGHTS_PATH}" \
    ${spec_args[@]+"${spec_args[@]}"} \
    --min-pairs 4 --target-pairs 4 \
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
  ${spec_args[@]+"${spec_args[@]}"} \
  "${golden_args[@]}" \
  --min-pairs 4 --target-pairs 4 \
  --tag "gemma4-local-$(date -u +%Y%m%dT%H%M%SZ)" \
  --out "${OUT_DIR}"

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
"${SCRIPT_DIR}/.github/scripts/emit-gemma4-score.sh" "${RESULTS_JSON}" "${SCORE_PATH}"
