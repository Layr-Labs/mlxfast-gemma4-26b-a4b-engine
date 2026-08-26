#!/usr/bin/env bash
# Revert-proof for tools/stage-runtime-worker.sh and its wiring into setup.sh.
#
# benchd resolves the scored worker at .build/release/mlxfast-runtime-worker and
# loads mlx.metallib from that SAME directory. The worker is built in an isolated
# scratch root (.build-worker/release) with its metallib beside it, so setup.sh
# stages the finished pair into .build/release. This suite asserts that staging
# actually lands the pair -- binary AND its sibling metallib -- at the
# benchd-resolved path, and that setup.sh invokes the staging step on both of its
# exit paths. It goes RED if the staging fix is reverted: the copy stops, the
# metallib is dropped, or the setup.sh call is removed.
#
# Hermetic: no weights, no GPU, no network, no toolchain, no box. Each case copies
# the REAL tool into a throwaway repo root so its ROOT_DIR resolves there, then
# drives it with stub files under $TMPDIR.
#
# Usage: tools/test-runtime-worker-staging.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
STAGE_TOOL="${REPO_ROOT}/tools/stage-runtime-worker.sh"
SETUP_SH="${REPO_ROOT}/setup.sh"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# Non-vacuity floor: a case deleted or short-circuited leaves the survivors green,
# so exit status alone cannot notice the suite shrinking. Raise this in the same
# commit that adds assertions.
EXPECTED_MIN_ASSERTIONS=10

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rtw-staging.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0
FAILURES=()

pass() {
  PASSED=$((PASSED + 1))
  [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"
  return 0
}

fail() {
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  echo "FAIL  $1" >&2
}

assert_file() {
  if [[ -f "$1" ]]; then pass "$2"; else fail "$2 (missing: $1)"; fi
}

assert_executable() {
  if [[ -x "$1" ]]; then pass "$2"; else fail "$2 (not executable: $1)"; fi
}

assert_absent() {
  if [[ ! -e "$1" ]]; then pass "$2"; else fail "$2 (unexpectedly present: $1)"; fi
}

# Build a throwaway repo root containing a copy of the real staging tool, so its
# ROOT_DIR resolves into the sandbox. Prints the root path.
make_fake_root() {
  local root
  root="$(mktemp -d "${WORK}/root.XXXXXX")"
  mkdir -p "${root}/tools"
  cp "${STAGE_TOOL}" "${root}/tools/stage-runtime-worker.sh"
  chmod +x "${root}/tools/stage-runtime-worker.sh"
  printf '%s\n' "${root}"
}

# Create the SOURCE pair the build would have produced under the scratch root.
seed_source_pair() {
  local root="$1"
  mkdir -p "${root}/.build-worker/release"
  printf '#!/bin/sh\nexit 0\n' > "${root}/.build-worker/release/mlxfast-runtime-worker"
  chmod +x "${root}/.build-worker/release/mlxfast-runtime-worker"
  printf 'FAKE-METALLIB\n' > "${root}/.build-worker/release/mlx.metallib"
}

# ---------------------------------------------------------------------------
# Case 1: the happy path stages the pair at the benchd-resolved location.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
seed_source_pair "${root}"
if "${root}/tools/stage-runtime-worker.sh" >/dev/null 2>&1; then
  pass "staging exits 0 when the source pair is present"
else
  fail "staging exits 0 when the source pair is present"
fi
staged_bin="${root}/.build/release/mlxfast-runtime-worker"
staged_metallib="${root}/.build/release/mlx.metallib"
assert_file "${staged_bin}" "worker staged at benchd path .build/release/mlxfast-runtime-worker"
assert_executable "${staged_bin}" "staged worker keeps its execute bit"
# The metallib-adjacency check run-parity.sh enforces: metallib beside the binary.
assert_file "${staged_metallib}" "mlx.metallib staged as the worker's sibling"
if [[ "$(dirname "${staged_bin}")" == "$(dirname "${staged_metallib}")" ]]; then
  pass "staged worker and mlx.metallib share one directory (run-parity adjacency)"
else
  fail "staged worker and mlx.metallib share one directory (run-parity adjacency)"
fi
# The source scratch root is untouched (isolation preserved).
assert_file "${root}/.build-worker/release/mlxfast-runtime-worker" \
  "scratch-root worker is left in place (build isolation preserved)"

# ---------------------------------------------------------------------------
# Case 2: fail-closed when the worker was never built.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
if "${root}/tools/stage-runtime-worker.sh" >/dev/null 2>&1; then
  fail "staging fails when the worker binary is absent"
else
  pass "staging fails when the worker binary is absent"
fi
assert_absent "${root}/.build/release/mlxfast-runtime-worker" \
  "nothing is staged when the worker binary is absent"

# ---------------------------------------------------------------------------
# Case 3: fail-closed when the metallib is missing (and skip is not set), so a
# worker can never be staged without its metallib.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
mkdir -p "${root}/.build-worker/release"
printf '#!/bin/sh\nexit 0\n' > "${root}/.build-worker/release/mlxfast-runtime-worker"
chmod +x "${root}/.build-worker/release/mlxfast-runtime-worker"
if "${root}/tools/stage-runtime-worker.sh" >/dev/null 2>&1; then
  fail "staging fails when mlx.metallib is missing"
else
  pass "staging fails when mlx.metallib is missing"
fi

# ---------------------------------------------------------------------------
# Case 4: setup.sh invokes the staging step on BOTH exit paths, after the
# metallib wait. Catches a revert that drops the wiring while the tool survives.
# ---------------------------------------------------------------------------
# Count bare invocations only: the `()` definition line and the `tools/...` call
# inside the function body do not match this anchored, argument-free pattern.
invocations="$(grep -cE '^[[:space:]]*stage_runtime_worker_for_benchd$' "${SETUP_SH}" || true)"
if [[ "${invocations}" -ge 2 ]]; then
  pass "setup.sh invokes stage_runtime_worker_for_benchd on both exit paths"
else
  fail "setup.sh invokes stage_runtime_worker_for_benchd on both exit paths (found ${invocations})"
fi
if grep -q 'tools/stage-runtime-worker.sh' "${SETUP_SH}"; then
  pass "setup.sh's staging function calls tools/stage-runtime-worker.sh"
else
  fail "setup.sh's staging function calls tools/stage-runtime-worker.sh"
fi

# ---------------------------------------------------------------------------
# Trailer + non-vacuity floor.
# ---------------------------------------------------------------------------
echo "runtime-worker-staging: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  echo "runtime-worker-staging: FAILURES:" >&2
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "runtime-worker-staging: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0
