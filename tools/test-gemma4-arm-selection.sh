#!/usr/bin/env bash
# Unit test for the DFlash SELECTION SEAM in tools/gemma4-measure-and-score.sh.
# Offline, no GPU, no weights, no network, no real benchd: each case copies the
# real wrapper into a throwaway root, points BENCHCTL at a stub that records its
# argv, and reads the argv back.
#
# WHAT IS ACTUALLY UNDER TEST. Not "does dflash run" -- that needs a box. This
# suite pins the property a box cannot distinguish on its own:
#
#   A HARDCODED SELECTOR IS INDISTINGUISHABLE FROM A WORKING ONE unless the
#   negative controls run. A wrapper that always passed the dflash spec would
#   pass the declared-dflash case and nothing else here. A wrapper that never
#   passed it would pass the mtp/absent cases and nothing else. Both directions
#   are asserted, on the same wrapper, in the same run.
#
# THE NO-PERTURBATION CONTROL. Cases 1-3 compare the composed argv against a
# LITERAL GOLDEN, byte for byte. That golden is the argv the wrapper produced
# BEFORE the seam existed (engine main @ 5a59ca2b, the merged pair), with only
# the run-timestamped --tag value normalized. If any future edit perturbs the
# MTP arm's invocation -- adds a flag, reorders one, changes a default -- these
# cases go red, and that is the point: the seam is required to be inert on the
# arm every existing submission runs.
#
# Usage: tools/test-gemma4-arm-selection.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $1" >&2
  failures=$((failures + 1))
}

# --- the throwaway workspace ------------------------------------------------

# Build a candidate root holding only what the wrapper touches: itself, the
# emitter it calls, a contract file (passed by PATH, never parsed here), a
# weights directory, and whichever head manifests the case wants.
make_root() {
  local root="$1"
  mkdir -p "${root}/tools" "${root}/fixtures" "${root}/.github/scripts" \
    "${root}/weights" "${root}/goldens"
  cp "${REPO_ROOT}/tools/gemma4-measure-and-score.sh" "${root}/tools/"
  cp "${REPO_ROOT}/.github/scripts/emit-gemma4-score.sh" "${root}/.github/scripts/"
  chmod +x "${root}/tools/gemma4-measure-and-score.sh" \
    "${root}/.github/scripts/emit-gemma4-score.sh"
  echo '{"track_id":"gemma4-26b-a4b-mlx-v1"}' > "${root}/fixtures/gemma4_26b_a4b_track.json"
  echo '{"prompt_tokens":[]}' > "${root}/goldens/pool-a.json"
  echo '{"expected_tokens":[]}' > "${root}/correctness.json"
  # The emitter cross-checks benchd's applied floor against the manifest's
  # scoring.decodeSpeedupFloor (the floor-drift tripwire) and REFUSES when it
  # cannot read it. The synthetic root must therefore carry the real
  # benchmark.json, copied rather than invented -- an invented floor here would
  # let the tripwire drift out of agreement with the shipped manifest and this
  # suite would never notice.
  cp "${REPO_ROOT}/benchmark.json" "${root}/benchmark.json"
}

# A stub benchctl: records argv one entry per line, then seals a minimal but
# WELL-FORMED cohort results.json so the wrapper's emitter step succeeds and the
# run's exit code reports the wrapper, not the emitter.
write_stub_benchctl() {
  local path="$1"
  cat > "${path}" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
: > "${STUB_ARGV_FILE}"
for a in "$@"; do
  printf '%s\n' "${a}" >> "${STUB_ARGV_FILE}"
done
out=""
prev=""
for a in "$@"; do
  if [[ "${prev}" == "--out" ]]; then out="${a}"; fi
  prev="${a}"
done
if [[ -n "${out}" ]]; then
  mkdir -p "${out}"
  cat > "${out}/results.json" <<'RESULTS'
{
  "timed_mode": "batched_free_run_v1_2_b8",
  "per_cohort": [
    {
      "cohort_index": 0,
      "batch_size": 8,
      "composite": {
        "composite_score": 1.25,
        "composite_speedup_floor": 0.9,
        "composite_speedup_floor_met": true,
        "decode_gain": 1.25,
        "prefill_gain": 1.25
      }
    }
  ]
}
RESULTS
fi
STUB
  chmod +x "${path}"
}

# Run the wrapper in a fresh root and leave the recorded argv in
# "${WORK}/argv.txt". `--tag`'s value is run-timestamped, so it is normalized to
# a fixed placeholder; nothing else is touched.
#
# $1 case name, $2 mode ("preflight" | "full"), rest: files to plant, given as
# <relative-path>=<content> pairs.
run_case() {
  local name="$1" mode="$2"
  shift 2
  local root="${WORK}/${name}"
  make_root "${root}"
  local pair rel content
  for pair in "$@"; do
    rel="${pair%%=*}"
    content="${pair#*=}"
    printf '%s' "${content}" > "${root}/${rel}"
  done
  write_stub_benchctl "${WORK}/benchctl-stub"

  local rc=0
  if [[ "${mode}" == "preflight" ]]; then
    STUB_ARGV_FILE="${WORK}/argv.raw" \
    BENCHCTL="${WORK}/benchctl-stub" \
    MLXFAST_GEMMA4_MEASURE_OUT_DIR="${root}/out" \
      "${root}/tools/gemma4-measure-and-score.sh" --preflight-only \
        >"${WORK}/case.out" 2>&1 || rc=$?
  else
    STUB_ARGV_FILE="${WORK}/argv.raw" \
    BENCHCTL="${WORK}/benchctl-stub" \
    MLXFAST_GEMMA4_MEASURE_OUT_DIR="${root}/out" \
    MLXFAST_GEMMA4_BASELINE_WORKSPACE="${root}" \
    MLXFAST_GEMMA4_GOLDEN_DIR="${root}/goldens" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${root}/correctness.json" \
    MLXFAST_SCORE_PATH="${root}/score.json" \
      "${root}/tools/gemma4-measure-and-score.sh" \
        >"${WORK}/case.out" 2>&1 || rc=$?
  fi

  if [[ -f "${WORK}/argv.raw" ]]; then
    sed 's|^gemma4-local-[0-9TZ]*$|gemma4-local-<TIMESTAMP>|' \
      "${WORK}/argv.raw" > "${WORK}/argv.txt"
    # The workspace root varies per case; normalize it so the golden below is a
    # property of the ARGV SHAPE and not of mktemp.
    sed -i.bak "s|${root}|<ROOT>|g" "${WORK}/argv.txt"
    rm -f "${WORK}/argv.txt.bak"
  else
    : > "${WORK}/argv.txt"
  fi
  CASE_RC="${rc}"
}

reset_argv() { rm -f "${WORK}/argv.raw" "${WORK}/argv.txt"; }

# --- the goldens ------------------------------------------------------------
#
# Byte-for-byte the argv engine main composed, timestamp and root normalized.
# Derived by running THAT script through the same stub, not by transcribing the
# source. Do not edit either golden to make a change pass.
#
# REGENERATED at the post-#73 tip (was: main @ 5a59ca2b). The ONLY delta is
# --min-pairs/--target-pairs 2 -> 4, which is David's 2026-08-26 pairs ruling
# arriving in the wrapper, not a change this branch made. Regenerated through
# the stub above rather than hand-edited, per this block's own rule -- and the
# fact that the ruled pair count is the only difference is itself the evidence
# that the arm-selection seam perturbs nothing else in the invocation.

read -r -d '' GOLDEN_PREFLIGHT <<'ARGV' || true
measure-job
--contract
<ROOT>/fixtures/gemma4_26b_a4b_track.json
--candidate
<ROOT>
--baseline
<ROOT>
--weights
<ROOT>/weights
--min-pairs
4
--target-pairs
4
--tag
gemma4-preflight
--out
<ROOT>/out
--preflight-only
ARGV

read -r -d '' GOLDEN_FULL <<'ARGV' || true
measure-job
--contract
<ROOT>/fixtures/gemma4_26b_a4b_track.json
--candidate
<ROOT>
--baseline
<ROOT>
--weights
<ROOT>/weights
--correctness-golden
<ROOT>/correctness.json
--golden
<ROOT>/goldens/pool-a.json
--min-pairs
4
--target-pairs
4
--tag
gemma4-local-<TIMESTAMP>
--out
<ROOT>/out
ARGV

expect_argv() {
  local name="$1" expected="$2"
  if [[ "$(cat "${WORK}/argv.txt")" != "${expected}" ]]; then
    fail "${name}: composed argv differs from the golden"
    diff <(printf '%s\n' "${expected}") "${WORK}/argv.txt" >&2 || true
  fi
}

MANIFEST_PINNED='{"version":1,"source":"pinned","max_bytes":2147483648}'

# ===========================================================================
# NEGATIVE CONTROL 1 -- an ABSENT dflash manifest seals MTP, and the argv is
# byte-identical to the pre-seam invocation.
# ===========================================================================
reset_argv
run_case absent-preflight preflight
if [[ "${CASE_RC}" -ne 0 ]]; then
  fail "absent manifest (preflight): wrapper exited ${CASE_RC}; output: $(cat "${WORK}/case.out")"
fi
expect_argv "absent manifest (preflight)" "${GOLDEN_PREFLIGHT}"

reset_argv
run_case absent-full full
if [[ "${CASE_RC}" -ne 0 ]]; then
  fail "absent manifest (full run): wrapper exited ${CASE_RC}; output: $(cat "${WORK}/case.out")"
fi
expect_argv "absent manifest (full run)" "${GOLDEN_FULL}"

# ===========================================================================
# NEGATIVE CONTROL 2 -- a manifest with NO `arm` key seals MTP, same argv.
# ===========================================================================
reset_argv
run_case no-arm-key full "dflash-head.manifest.json=${MANIFEST_PINNED}"
if [[ "${CASE_RC}" -ne 0 ]]; then
  fail "manifest without an arm key: wrapper exited ${CASE_RC}; output: $(cat "${WORK}/case.out")"
fi
expect_argv "manifest without an arm key" "${GOLDEN_FULL}"

# ===========================================================================
# NEGATIVE CONTROL 3 -- an EXPLICIT "arm":"mtp" seals MTP, same argv. This is
# the case a hardcoded-dflash selector fails.
# ===========================================================================
reset_argv
run_case arm-mtp full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":"mtp","max_bytes":2147483648}'
if [[ "${CASE_RC}" -ne 0 ]]; then
  fail "arm mtp: wrapper exited ${CASE_RC}; output: $(cat "${WORK}/case.out")"
fi
expect_argv "arm mtp" "${GOLDEN_FULL}"
if grep -q -- "--candidate-spec" "${WORK}/argv.txt"; then
  fail "arm mtp: a spec flag was passed on the mtp arm"
fi

# ===========================================================================
# THE POSITIVE CONTROL -- "arm":"dflash" selects the dflash spec, and selects
# NOTHING ELSE. This is the case a never-select wrapper fails.
# ===========================================================================
reset_argv
run_case arm-dflash full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":"dflash","max_bytes":2147483648}'
if [[ "${CASE_RC}" -ne 0 ]]; then
  fail "arm dflash: wrapper exited ${CASE_RC}; output: $(cat "${WORK}/case.out")"
fi
if ! grep -q -- "--candidate-spec" "${WORK}/argv.txt"; then
  fail "arm dflash: no --candidate-spec was passed"
fi
# `|| true`: when the flag is absent the case above has already failed, and the
# suite must go on to report the rest rather than abort under `set -e`.
spec_value="$(grep -A1 -- "--candidate-spec" "${WORK}/argv.txt" | tail -n 1 || true)"
if [[ "${spec_value}" != '{"mode":"dflash","dflash":{}}' ]]; then
  fail "arm dflash: spec value is '${spec_value}', expected {\"mode\":\"dflash\",\"dflash\":{}}"
fi
# The declared arm must change the SPEC and nothing else: strip the two spec
# entries and the remainder must be the untouched golden.
if [[ "$(grep -v -e '^--candidate-spec$' -e '^{"mode":"dflash","dflash":{}}$' "${WORK}/argv.txt")" \
      != "${GOLDEN_FULL}" ]]; then
  fail "arm dflash: the declaration perturbed the invocation beyond the spec flag"
fi

# The dflash arm reaches the preflight invocation too -- a submission finds out
# pre-GPU, at preSubmitCommand time, not on the ranked box.
reset_argv
run_case arm-dflash-preflight preflight \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":"dflash","max_bytes":2147483648}'
if ! grep -q -- "--candidate-spec" "${WORK}/argv.txt"; then
  fail "arm dflash (preflight): no --candidate-spec was passed to the preflight invocation"
fi

# ===========================================================================
# THE REFUSALS -- every one must fire BEFORE benchctl is reached, so a bad
# declaration can never become a measured run of the other arm.
# ===========================================================================
expect_refusal() {
  local name="$1" needle="$2"
  if [[ "${CASE_RC}" -eq 0 ]]; then
    fail "${name}: wrapper exited 0; it must refuse"
  fi
  if [[ -s "${WORK}/argv.txt" ]]; then
    fail "${name}: benchctl was invoked despite the refusal"
  fi
  if ! grep -q -- "${needle}" "${WORK}/case.out"; then
    fail "${name}: refusal does not name '${needle}'; got: $(cat "${WORK}/case.out")"
  fi
}

reset_argv
run_case arm-unknown full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":"dspark"}'
expect_refusal "unknown arm 'dspark'" "dspark"

reset_argv
run_case arm-empty full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":""}'
expect_refusal "empty arm" "not a mode this track admits"

reset_argv
run_case arm-wrong-case full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":"MTP"}'
expect_refusal "wrong-case arm 'MTP'" "MTP"

reset_argv
run_case arm-non-string full \
  'dflash-head.manifest.json={"version":1,"source":"pinned","arm":1}'
expect_refusal "non-string arm" "non-string"

reset_argv
run_case manifest-malformed full \
  'dflash-head.manifest.json={"version":1,"source":'
expect_refusal "malformed manifest JSON" "not a JSON object"

reset_argv
run_case manifest-not-object full 'dflash-head.manifest.json=["dflash"]'
expect_refusal "manifest that is not a JSON object" "not a JSON object"

# The arm declared in the WRONG head manifest is a refusal, not a no-op: this
# is the tolerance that would otherwise score a participant on an arm they did
# not ask for.
reset_argv
run_case arm-in-mtp-manifest full \
  'mtp-head.manifest.json={"version":1,"source":"pinned","arm":"dflash"}'
expect_refusal "arm declared in mtp-head.manifest.json" "dflash-head.manifest.json only"

# ===========================================================================
# ANTI-DRIFT -- the wrapper's vocabulary is the Swift declaration parser's.
# Two readers of one declaration that disagree is the same silent-mis-score by
# another route, so the two lists are compared rather than kept in step by
# convention.
# ===========================================================================
DECL_SWIFT="${REPO_ROOT}/Sources/MLXFastTrustedHarness/Gemma4MTPHeadDeclaration.swift"
swift_arms="$(sed -n '/^public enum DeclaredArm/,/^}/p' "${DECL_SWIFT}" \
  | sed -n 's/^ *case \([a-z][a-zA-Z0-9]*\)$/\1/p' | sort | tr '\n' ' ')"
wrapper_arms="$(sed -n '/^case .*DECLARED_ARM.* in$/,/^esac$/p' \
  "${REPO_ROOT}/tools/gemma4-measure-and-score.sh" \
  | sed -n 's/^  \([a-z][a-zA-Z0-9]*\))$/\1/p' | sort | tr '\n' ' ')"
if [[ -z "${wrapper_arms// /}" ]]; then
  fail "anti-drift: found no case arms in the wrapper's selection switch (the extraction broke)"
fi
if [[ -z "${swift_arms// /}" ]]; then
  fail "anti-drift: found no DeclaredArm cases in ${DECL_SWIFT} (the extraction broke)"
fi
if [[ "${swift_arms}" != "${wrapper_arms}" ]]; then
  fail "anti-drift: DeclaredArm cases [${swift_arms}] != wrapper case arms [${wrapper_arms}]"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-gemma4-arm-selection.sh: all cases passed"
  exit 0
fi
echo "test-gemma4-arm-selection.sh: ${failures} case(s) failed" >&2
exit 1
