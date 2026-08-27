#!/usr/bin/env bash
# test-gemma4-null-control.sh -- offline tests for the null-control verdict
# logic (tools/gemma4-null-control.sh --check), no GPU and no benchctl needed.
# Fixture handling is exercised against a scratch copy; the repository fixture
# is never modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# A scratch repo layout so the script under test reads OUR fixture copy.
mkdir -p "${TMP}/repo/tools" "${TMP}/repo/fixtures"
cp "${SCRIPT_DIR}/tools/gemma4-null-control.sh" "${TMP}/repo/tools/"
cp "${SCRIPT_DIR}/fixtures/gemma4_26b_a4b_track.json" "${TMP}/repo/fixtures/"
FIXTURE="${TMP}/repo/fixtures/gemma4_26b_a4b_null_control.json"

results() { # results <composite or "null"> -> path
  local c="$1" p="${TMP}/results-$RANDOM.json"
  if [[ "${c}" == "null" ]]; then
    cat > "${p}" <<'EOF'
{"per_cohort":[{"composite":null,"composite_absent_reason":"test: gated"}]}
EOF
  else
    cat > "${p}" <<EOF
{"per_cohort":[{"composite":{"composite_score":${c}}}]}
EOF
  fi
  printf '%s\n' "${p}"
}

fixture() { # fixture <status> <expected> <band_pct>
  cat > "${FIXTURE}" <<EOF
{"schema":"gemma4-null-control-v1","track_id":"gemma4-26b-a4b-mlx-v1","timed_mode":"batched_free_run_v1_2_b8","status":"$1","expected_composite":$2,"band_pct":$3}
EOF
}

run_check() { # run_check <results-path>; echoes exit code
  set +e
  ( cd "${TMP}/repo" && ./tools/gemma4-null-control.sh --check "$1" ) >/dev/null 2>&1
  echo $?
  set -e
}

pass=0; fail=0
expect() { # expect <name> <want> <got>
  if [[ "$2" == "$3" ]]; then pass=$((pass+1)); else
    fail=$((fail+1)); echo "FAIL: $1 (want exit $2, got $3)" >&2
  fi
}

# 1. Certified fixture, in-band composite: pass.
fixture measured 1.0 2.0
expect "in-band passes"            0 "$(run_check "$(results 1.005)")"
# 2. Exactly on the band edge: pass (<=).
expect "band edge passes"          0 "$(run_check "$(results 1.02)")"
# 3. Out-of-band high: fail.
expect "out-of-band high fails"    1 "$(run_check "$(results 1.021)")"
# 4. Out-of-band low: fail.
expect "out-of-band low fails"     1 "$(run_check "$(results 0.9)")"
# 5. Uncertified fixture (pending-measurement): distinct refusal, never a pass.
fixture pending-measurement 0 2.0
expect "pending fixture refuses"   3 "$(run_check "$(results 1.0)")"
# 6. Gated/absent composite: a null control that cannot score is a failed control.
fixture measured 1.0 2.0
expect "absent composite fails"    1 "$(run_check "$(results null)")"
# 7. Degenerate expected (0): fail closed, no divide-through.
fixture measured 0 2.0
expect "zero expected fails"       1 "$(run_check "$(results 1.0)")"
# 8. Missing results file: fail.
expect "missing results fails"     1 "$(run_check "${TMP}/does-not-exist.json")"
# 9. Author mode writes the measured value and flips status (against the scratch fixture).
fixture pending-measurement 0 2.0
( cd "${TMP}/repo" && MODE_TEST=1 ./tools/gemma4-null-control.sh --check "$(results 1.0)" ) >/dev/null 2>&1 || true
r="$(results 0.9971)"
( cd "${TMP}/repo" && jq --argjson c 0.9971 '.status="measured"|.expected_composite=$c' \
    fixtures/gemma4_26b_a4b_null_control.json > f.tmp && mv f.tmp fixtures/gemma4_26b_a4b_null_control.json )
expect "authored fixture then passes" 0 "$(run_check "${r}")"

echo "test-gemma4-null-control.sh: ${pass} passed, ${fail} failed"
[[ "${fail}" -eq 0 ]]
