#!/usr/bin/env bash
# Hostile-archive E2E suite for the submission-restriction enforcers (issue #16).
#
# Every case synthesizes a malicious (or benign) submission and asserts that it
# dies at the RIGHT layer with the RIGHT diagnostic, or survives when it should.
# Nothing here needs weights, a GPU, a network, or a box.
#
# HARD GATE: this suite must be green before any real submission is dispatched.
# The `swift` job of .github/workflows/ci.yml runs it on every pull request and
# every push to main, so a regression turns the run red. That is CI coverage and
# not a mechanical interlock: NO STATUS CHECK IS REQUIRED ANYWHERE IN THIS
# REPOSITORY -- branch protection is unavailable on this plan (403), there is no
# CODEOWNERS, and a red run blocks neither a merge nor a dispatch. CI here is
# ADVISORY by ruling (David 2026-08-20; the two-session red-team is the de facto
# merge gate), and dispatch-time gating is deferred with the ranked runner
# (docs/submission-restriction-spec.md section 9), so today a red gate stops a
# dispatch only because a human reads it. Keep it hosted-runner-clean -- an
# assertion that needs a box, a credential or the network cannot go in this
# file.
#
# Layers under test:
#   Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift  (byte budget)
#   .github/scripts/overlay-editable-paths.sh                     (REPLACE overlay)
#   .github/scripts/submission-static-review-checks.sh            (static review)
#   .github/scripts/enforce-modifiable-surface.sh                 (surface gate)
#   benchmark.json                                                (single source)
#
# Usage: tools/test-submission-security.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SCRIPTS="${REPO_ROOT}/.github/scripts"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# Under Actions the caller's environment carries the job's real step-to-step
# channel files, and cases below run repo tools (tools/stage-ranked-heads.sh)
# that append to whatever GITHUB_ENV names. A test suite must never write to
# the job's real step-to-step state: on 2026-08-26 the ranked-heads cases
# leaked four QMTP_* fixture paths into every later CI step this way. All four
# writable channels are dropped -- the other three had no writer in-tree at
# the time, and this keeps one gaining a writer from reopening the class. The
# one case that tests the GITHUB_ENV path sets its own synthetic file
# explicitly.
unset GITHUB_ENV GITHUB_OUTPUT GITHUB_PATH GITHUB_STEP_SUMMARY

# A section that stops running -- deleted, renamed out of the flow, or skipped by
# an early `continue` -- leaves every REMAINING assertion green, so exit status
# alone cannot notice it and the gate silently shrinks. This floor is the single
# place the suite's expected size is recorded; raise it in the same commit that
# adds assertions. Checked in the trailer below, and deliberately a floor rather
# than an equality so adding a case does not fail the run that adds it.
#
# IT CANNOT BE THE ONLY FLOOR. This file runs `set -uo pipefail` without `-e`,
# and the check below is in this same file's trailer: truncate the file, or add
# an early `exit 0`, and the floor goes with the assertions it was guarding --
# the CI step then goes green on a gate that ran nothing. The workflow restates
# it from OUTSIDE, against this script's printed trailer, exactly as
# tools/ci-swift-warning-gate.sh guards the build from outside the build:
# .github/workflows/ci.yml, step "Submission-security suite did not shrink",
# whose CI_MIN_ASSERTIONS is cross-checked against the number below so the two
# cannot drift apart in silence. Raise BOTH in the same commit.
EXPECTED_MIN_ASSERTIONS=287

WORK="$(mktemp -d "${TMPDIR:-/tmp}/submission-security.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0
FAILURES=()

# --- assertion helpers -------------------------------------------------------

# assert_exit NAME WANT_STATUS WANT_SUBSTRING CWD CMD...
#   WANT_SUBSTRING "" skips the diagnostic check.
assert_exit() {
  local name="$1" want="$2" pattern="$3" cwd="$4"
  shift 4
  local out status
  out="$( (cd "${cwd}" && "$@") 2>&1 )"
  status=$?
  if (( VERBOSE )); then
    printf '\n--- %s (exit %d)\n%s\n' "${name}" "${status}" "${out}"
  fi
  if [[ "${status}" != "${want}" ]]; then
    FAILED=$((FAILED + 1))
    FAILURES+=("${name}: exit ${status}, wanted ${want}"$'\n'"${out}")
    printf 'FAIL  %s (exit %s, wanted %s)\n' "${name}" "${status}" "${want}"
    return
  fi
  if [[ -n "${pattern}" && "${out}" != *"${pattern}"* ]]; then
    FAILED=$((FAILED + 1))
    FAILURES+=("${name}: missing diagnostic '${pattern}'"$'\n'"${out}")
    printf 'FAIL  %s (diagnostic did not mention: %s)\n' "${name}" "${pattern}"
    return
  fi
  PASSED=$((PASSED + 1))
  printf 'ok    %s\n' "${name}"
}

assert_equal() {
  local name="$1" got="$2" want="$3"
  if [[ "${got}" == "${want}" ]]; then
    PASSED=$((PASSED + 1))
    printf 'ok    %s\n' "${name}"
  else
    FAILED=$((FAILED + 1))
    FAILURES+=("${name}: got '${got}', wanted '${want}'")
    printf 'FAIL  %s (got %s, wanted %s)\n' "${name}" "${got}" "${want}"
  fi
}

fill() { # fill PATH BYTES
  head -c "$2" /dev/zero | tr '\0' 'a' > "$1"
}

# --- build the byte-budget enforcer -----------------------------------------
#
# Compiled from the REAL enforcer source plus a thin driver, so this suite
# exercises the shipped implementation rather than a copy of its rules.
BUDGET_BIN="${WORK}/editable-surface-budget"
echo "building the byte-budget enforcer from Sources/MLXFastTrustedHarness ..."
if ! swiftc -O \
  "${REPO_ROOT}/Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift" \
  "${REPO_ROOT}/tools/editable-surface-budget-cli/main.swift" \
  -o "${BUDGET_BIN}" 2>"${WORK}/swiftc.log"; then
  cat "${WORK}/swiftc.log" >&2
  echo "FATAL: could not build the byte-budget enforcer" >&2
  exit 1
fi

budget() { "${BUDGET_BIN}" "$@"; }

# --- the enforcers this suite drives must be there to be driven --------------
#
# A large family of assertions below is of the form "the trusted checkout was
# NOT modified" -- the gitlink is intact, the sentinel above the root is
# untouched, the setuid bit never landed. Every one of them is trivially true
# when the enforcer never ran at all. Measured, with
# overlay-editable-paths.sh moved aside: FIFTEEN such assertions went green on a
# run where no overlay happened.
#
# Each of those does have a companion assert_exit on the same fixture that reds
# in that state, so the suite as a whole notices. But a neighbouring assertion
# failing is not the assertion in question binding -- the same distinction
# check 3 had to be taught about check 3c -- and stating it once here is far
# cheaper than fifteen individual guards. If this block passes, "untouched"
# below means the enforcer ran and left it alone, rather than never having run.
for driver in overlay-editable-paths.sh submission-static-review-checks.sh \
              enforce-modifiable-surface.sh; do
  assert_equal "harness/${driver} is present and executable" \
    "$([ -x "${SCRIPTS}/${driver}" ] && echo executable || echo MISSING)" "executable"
done
assert_equal "harness/the byte-budget enforcer compiled to an executable" \
  "$([ -x "${BUDGET_BIN}" ] && echo executable || echo MISSING)" "executable"

# --- synthetic fixture -------------------------------------------------------
#
# A miniature track: two source paths, one optional head declaration, one
# byte-budget-exempt weights directory, one pinned-submodule stand-in, and caps
# small enough that a few kilobytes trip them.
CONTRACT_JSON='{
  "editablePaths": ["src", "config.txt", "head.manifest.json", "head"],
  "optionalEditablePaths": ["head.manifest.json", "head"],
  "editableSurfaceByteBudget": {
    "exemptPaths": ["head"],
    "exemptPathMaxBytes": 8192,
    "exemptPathMaxFileBytes": 7000,
    "maxTotalBytes": 20000,
    "maxFileBytes": 4096,
    "maxGrowthBytes": 2048
  },
  "staticReviewTrackId": "test-track-v1"
}'

new_fixture() { # -> echoes a fresh DIR containing trusted/ and sub/
  local dir
  # mktemp, not a counter: this runs inside $( ), so a shell variable would
  # never increment in the parent and every fixture would be the same dirty
  # directory.
  dir="$(mktemp -d "${WORK}/fx.XXXXXX")"
  mkdir -p "${dir}/trusted/src" "${dir}/trusted/head" "${dir}/trusted/benchd" "${dir}/sub"
  printf '%s\n' "${CONTRACT_JSON}" > "${dir}/trusted/benchmark.json"
  printf 'trusted kernel\n' > "${dir}/trusted/src/kernel.txt"
  printf 'trusted helper\n' > "${dir}/trusted/src/helper.txt"
  printf 'trusted config\n' > "${dir}/trusted/config.txt"
  printf '{"source":"pinned"}\n' > "${dir}/trusted/head.manifest.json"
  printf 'trusted head weights\n' > "${dir}/trusted/head/weights.bin"
  # Non-editable trusted surface a hostile archive will try to reach.
  mkdir -p "${dir}/trusted/Sources/MLXFastCLI"
  printf 'trusted CLI\n' > "${dir}/trusted/Sources/MLXFastCLI/main.swift"
  printf '[submodule "benchd"]\n\tpath = benchd\n' > "${dir}/trusted/.gitmodules"
  printf 'pinned measurement daemon\n' > "${dir}/trusted/benchd/PIN"
  # Sentinel OUTSIDE the trusted checkout: a path-escaping archive must never
  # be able to write here.
  printf 'untouched\n' > "${dir}/sentinel.txt"
  echo "${dir}"
}

overlay() { # overlay TRUSTED SUB [extra env assignments...]
  local trusted="$1" sub="$2"
  shift 2
  (cd "${trusted}" && SUBMISSION_WORKTREE="${sub}" "$@" "${SCRIPTS}/overlay-editable-paths.sh")
}

echo
echo "=== A. byte budget (EditableSurfaceByteBudget.swift) ========================"

fx="$(new_fixture)"
assert_exit "budget/benign surface verifies" 0 "verified" "${fx}/trusted" \
  budget verify benchmark.json

fx="$(new_fixture)"
fill "${fx}/trusted/src/lookup.txt" 5000
assert_exit "budget/oversize single file rejected" 1 "above the per-file static review limit" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
for i in $(seq 1 8); do fill "${fx}/trusted/src/chunk${i}.txt" 4000; done
assert_exit "budget/oversize total surface rejected" 1 "above the static review limit" \
  "${fx}/trusted" budget verify benchmark.json

# A lookup table smuggled in as many small files still trips the TOTAL cap;
# smuggled as one big file it trips the PER-FILE cap above. Both routes closed.
fx="$(new_fixture)"
for i in $(seq 1 30); do fill "${fx}/trusted/src/tbl${i}.txt" 900; done
assert_exit "budget/lookup-table smuggle across many small files rejected" 1 \
  "above the static review limit" "${fx}/trusted" budget verify benchmark.json

# AGGREGATE overflow, kept clear of the per-file cap: three 3 KB shards are each
# under exemptPathMaxFileBytes (7000) and together clear exemptPathMaxBytes
# (8192), so this still exercises the aggregate and not the per-file bound.
fx="$(new_fixture)"
for i in 1 2 3; do fill "${fx}/trusted/head/shard${i}.bin" 3000; done
assert_exit "budget/exempt path over its own cap rejected" 1 "above the exempt-path limit" \
  "${fx}/trusted" budget verify benchmark.json

# PER-FILE exempt cap (David BYO-512 ruling 2026-08-26). One 9 KB blob is over
# exemptPathMaxFileBytes (7000) on its own. Before this cap an exempt file had
# no per-file bound at all -- maxFileBytes deliberately does not reach exempt
# paths -- so a head shipped as one monolithic shard passed the budget and then
# failed the promotion push against GitHub's 100 MB blob limit.
fx="$(new_fixture)"
fill "${fx}/trusted/head/big.bin" 9000
assert_exit "budget/exempt file over the per-file cap rejected" 1 \
  "above the exempt per-file limit" "${fx}/trusted" budget verify benchmark.json

# The refusal NAMES THE FILE, not the directory: an operator repacking a head
# into shards needs to know which blob was oversize.
assert_exit "budget/exempt per-file refusal names the offending file" 1 \
  "head/big.bin" "${fx}/trusted" budget verify benchmark.json

# NEGATIVE CONTROL: the per-file cap must not swallow the sharded case it exists
# to permit. Two 3 KB shards are under both bounds and still verify.
fx="$(new_fixture)"
fill "${fx}/trusted/head/shard1.bin" 3000
fill "${fx}/trusted/head/shard2.bin" 3000
assert_exit "budget/sharded exempt head under both exempt caps verifies" 0 "verified" \
  "${fx}/trusted" budget verify benchmark.json

# The exemption must NOT leak into the code budget: 6 KB of head weights is far
# over the 4 KB per-file code cap and still fine, while the code paths stay bound.
fx="$(new_fixture)"
fill "${fx}/trusted/head/weights.bin" 6000
assert_exit "budget/exempt path under its cap does not charge the code budget" 0 "verified" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c.pop("editablePaths")
json.dump(c, open(p, "w"))
PY
assert_exit "budget/contract without editablePaths fails closed" 1 "no usable editablePaths" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
assert_exit "budget/absent contract is skipped, not passed" 2 "no benchmark contract" \
  "${fx}/trusted" budget verify no-such-contract.json

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["maxFileBytes"] = "lots"
json.dump(c, open(p, "w"))
PY
assert_exit "budget/non-integer cap fails closed" 1 "not readable as a track manifest" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["maxFileBytes"] = 0
json.dump(c, open(p, "w"))
PY
assert_exit "budget/non-positive cap fails closed" 1 "must be a positive integer" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["maxFileBytes"] = 999999
json.dump(c, open(p, "w"))
PY
assert_exit "budget/per-file cap above total cap fails closed" 1 "can never bind" \
  "${fx}/trusted" budget verify benchmark.json

# Q2 (issue #20): the walk was symlink-blind. A symlink AT an editable path let a
# multi-megabyte target be sized through the link, or -- for a link the directory
# enumerator declines to follow -- counted as zero. Both routes now fail closed.
fx="$(new_fixture)"
fill "${fx}/big-payload.bin" 50000
rm -f "${fx}/trusted/config.txt"
ln -s "${fx}/big-payload.bin" "${fx}/trusted/config.txt"
assert_exit "budget/editable path that IS a symlink rejected" 1 "is a symlink" \
  "${fx}/trusted" budget verify benchmark.json

fx="$(new_fixture)"
fill "${fx}/big-payload.bin" 50000
ln -s "${fx}/big-payload.bin" "${fx}/trusted/src/leak.txt"
assert_exit "budget/symlink inside an editable path rejected" 1 "non-regular entry" \
  "${fx}/trusted" budget verify benchmark.json

# F1 (gate symmetry): a non-regular NON-symlink AT a root editable path (a FIFO
# here) used to fall through the root branch's regular-file guard to a silent
# skip and a verified exit 0, while the in-directory branch and the shell
# whole-surface gate both refuse a non-regular entry. The root branch now
# refuses too -- all three gates symmetric on a non-regular file at root.
fx="$(new_fixture)"
rm -f "${fx}/trusted/config.txt"
mkfifo "${fx}/trusted/config.txt"
assert_exit "budget/non-regular file AT a root editable path rejected" 1 \
  "is a non-regular file" "${fx}/trusted" budget verify benchmark.json

# Q3 (issue #20): a surface where EVERY editable path is absent walked to
# totalBytes=0 fileCount=0 and returned .verified -- absence read as a clean
# pass. It is a refusal.
fx="$(new_fixture)"
rm -rf "${fx}/trusted/src" "${fx}/trusted/config.txt" \
  "${fx}/trusted/head.manifest.json" "${fx}/trusted/head"
assert_exit "budget/every editable path absent fails closed" 1 "absence is a refusal" \
  "${fx}/trusted" budget verify benchmark.json

# Q4 (issue #20): the exempt-path cap is an AGGREGATE, so an overflow names the
# aggregate, not whichever exempt path happened to tip it over. Two exempt paths
# 6 KB each, cap 10 KB: each is individually under, together they overflow.
fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editablePaths"].append("head2")
c["editableSurfaceByteBudget"]["exemptPaths"] = ["head", "head2"]
c["editableSurfaceByteBudget"]["exemptPathMaxBytes"] = 10000
json.dump(c, open(p, "w"))
PY
mkdir -p "${fx}/trusted/head2"
fill "${fx}/trusted/head/weights.bin" 6000
fill "${fx}/trusted/head2/weights.bin" 6000
assert_exit "budget/exempt aggregate overflow names the aggregate, not one path" 1 \
  "exempt editable paths total" "${fx}/trusted" budget verify benchmark.json

echo
echo "=== B. overlay REPLACE semantics (overlay-editable-paths.sh) ==============="

# Benign: the archive's copy REPLACES the trusted copy, a file the archive does
# not carry is GONE (replace, not merge), and non-editable trusted files stay.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
assert_exit "overlay/benign archive overlays" 0 "trusted harness retained" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
assert_equal "overlay/editable file replaced" "$(cat "${fx}/trusted/src/kernel.txt")" "submitted kernel"
assert_equal "overlay/REPLACE deletes what the archive omits" \
  "$([[ -e "${fx}/trusted/src/helper.txt" ]] && echo present || echo absent)" "absent"
assert_equal "overlay/optional head declaration kept from the trusted copy" \
  "$(cat "${fx}/trusted/head.manifest.json")" '{"source":"pinned"}'
assert_equal "overlay/non-editable trusted file untouched" \
  "$(cat "${fx}/trusted/Sources/MLXFastCLI/main.swift")" "trusted CLI"

# A missing REQUIRED editable path is a stale clone, and it is a refusal.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
assert_exit "overlay/missing required editable path fails closed" 1 \
  "submitted editable path is missing" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

# Symlink smuggling: a submitted editable path that is, or contains, a symlink.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
ln -s /etc/passwd "${fx}/sub/src/leak.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
assert_exit "overlay/symlink inside an editable path rejected" 1 \
  "must not contain symlinks" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

fx="$(new_fixture)"
mkdir -p "${fx}/sub"
ln -s /etc "${fx}/sub/src"
printf 'submitted config\n' > "${fx}/sub/config.txt"
assert_exit "overlay/editable path that IS a symlink rejected" 1 \
  "must not contain symlinks" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

# Setuid smuggling. The asserted guarantee is that a setuid bit never LANDS in
# the trusted checkout, not which layer stops it: the unprivileged tar/cp the
# overlay uses already drops setuid/setgid on extraction, and
# validate_overlay_tree is the backstop for a copy that did preserve it (root,
# or a tar invoked with -p). See docs/submission-restriction-spec.md.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
chmod 4755 "${fx}/sub/src/kernel.txt"
( cd "${fx}/trusted" && SUBMISSION_WORKTREE="${fx}/sub" \
  "${SCRIPTS}/overlay-editable-paths.sh" >/dev/null 2>&1 ) || true
# `find` over a directory that does not exist prints nothing and `wc -l` says 0
# -- which is the answer the setuid assertion below wants. So prove the overlay
# actually produced the tree first, and that the file the setuid bit was set on
# is really there: without this, an overlay that silently wrote nothing reads as
# "the bit never landed".
assert_equal "overlay/setuid fixture: the overlaid file exists to be checked" \
  "$([ -f "${fx}/trusted/src/kernel.txt" ] && echo present || echo MISSING)" "present"
assert_equal "overlay/setuid positive control: find does see the bit on the source" \
  "$(find "${fx}/sub/src" \( -perm -4000 -o -perm -2000 \) -print -quit | wc -l | tr -d ' ')" "1"
assert_equal "overlay/setuid bit never lands in the trusted checkout" \
  "$(find "${fx}/trusted/src" \( -perm -4000 -o -perm -2000 \) -print -quit | wc -l | tr -d ' ')" "0"

# Hardlinked files inside an overlaid directory: tar recreates the link, so the
# post-copy tree check is what fires.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
ln "${fx}/sub/src/kernel.txt" "${fx}/sub/src/alias.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
assert_exit "overlay/hardlinked file rejected" 1 "must not contain hardlinked files" \
  "${fx}/trusted" env SUBMISSION_WORKTREE="${fx}/sub" \
  "${SCRIPTS}/overlay-editable-paths.sh"

# Path escape: the archive carries entries above its own root. The overlay only
# ever reads the TRUSTED contract's paths, so they are never even opened.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
printf 'pwned\n' > "${fx}/sub/../escape-attempt.txt"
mkdir -p "${fx}/sub/dotdot"
printf 'pwned\n' > "${fx}/sub/dotdot/payload.txt"
assert_exit "overlay/archive entries outside editablePaths are inert" 0 \
  "trusted harness retained" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
assert_equal "overlay/sentinel above the trusted root untouched" \
  "$(cat "${fx}/sentinel.txt")" "untouched"
assert_equal "overlay/non-editable archive payload not landed" \
  "$([[ -e "${fx}/trusted/dotdot" ]] && echo present || echo absent)" "absent"

# A contract whose editablePaths themselves try to escape or go absolute.
for evil in "../evil" "src/../../evil" "/etc" "./src" 'src\evil' ":pathspec"; do
  fx="$(new_fixture)"
  python3 - "${fx}/trusted/benchmark.json" "${evil}" <<'PY'
import json, sys
p, evil = sys.argv[1], sys.argv[2]
c = json.load(open(p))
c["editablePaths"] = [evil]
json.dump(c, open(p, "w"))
PY
  mkdir -p "${fx}/sub/src"
  printf 'x\n' > "${fx}/sub/src/kernel.txt"
  assert_exit "overlay/contract path '${evil}' rejected" 1 "invalid editable path" \
    "${fx}/trusted" env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
done

# Pinned measurement harness: an editable entry that reaches what decides which
# scorer runs is refused by construction. benchd used to be a SOURCE submodule
# (`benchd` + `.gitmodules`); it is now a channel PREBUILT (the retired `benchd.pin` spelling ->
# `benchd-bin/`). BOTH generations are exercised: the live pin paths because they
# carry the property today, and the retired submodule spellings because all three
# guard layers deliberately still carry them so a reintroduced gitlink is covered
# on arrival -- a case this suite must keep proving is non-vacuous.
for gitlink in "benchd" "benchd/crates" ".gitmodules" "benchd.pin" "benchd-bin" "benchd-bin/benchctl"; do
  fx="$(new_fixture)"
  python3 - "${fx}/trusted/benchmark.json" "${gitlink}" <<'PY'
import json, sys
p, gitlink = sys.argv[1], sys.argv[2]
c = json.load(open(p))
c["editablePaths"] = ["src", gitlink]
json.dump(c, open(p, "w"))
PY
  mkdir -p "${fx}/sub/src" "${fx}/sub/benchd/crates"
  printf 'x\n' > "${fx}/sub/src/kernel.txt"
  printf 'attacker daemon\n' > "${fx}/sub/benchd/PIN"
  printf 'attacker daemon\n' > "${fx}/sub/benchd/crates/PIN"
  printf '[submodule "benchd"]\n\tpath = elsewhere\n' > "${fx}/sub/.gitmodules"
  assert_exit "overlay/gitlink entry '${gitlink}' refused" 1 \
    "measurement-harness surface" "${fx}/trusted" \
    env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
done

# And with the real (gitlink-free) contract, an archive that merely CONTAINS a
# replacement benchd/.gitmodules cannot land it.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src" "${fx}/sub/benchd"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
printf 'attacker daemon\n' > "${fx}/sub/benchd/PIN"
printf '[submodule "benchd"]\n\tpath = attacker\n' > "${fx}/sub/.gitmodules"
mkdir -p "${fx}/sub/Sources/MLXFastCLI"
printf 'attacker CLI\n' > "${fx}/sub/Sources/MLXFastCLI/main.swift"
assert_exit "overlay/archive with a gitlink payload overlays only the surface" 0 \
  "trusted harness retained" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
assert_equal "overlay/gitlink content untouched" "$(cat "${fx}/trusted/benchd/PIN")" \
  "pinned measurement daemon"
assert_equal "overlay/.gitmodules untouched" \
  "$(grep -c 'path = benchd' "${fx}/trusted/.gitmodules")" "1"
assert_equal "overlay/non-editable path write not landed" \
  "$(cat "${fx}/trusted/Sources/MLXFastCLI/main.swift")" "trusted CLI"

# optionalEditablePaths abuse: the submission ships its own contract declaring a
# wider surface and more optional paths. The overlay reads the TRUSTED contract
# only, so the widened surface is inert and the missing REQUIRED path still
# fails closed.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
python3 - "${fx}/sub/benchmark.json" <<'PY'
import json, sys
json.dump({
    "editablePaths": ["src", "config.txt", "Sources", "benchd", ".gitmodules"],
    "optionalEditablePaths": ["config.txt", "Sources", "benchd", ".gitmodules"],
    "editableSurfaceByteBudget": {"exemptPaths": ["src"], "maxTotalBytes": 999999999},
    "staticReviewTrackId": "test-track-v1",
}, open(sys.argv[1], "w"))
PY
assert_exit "overlay/submission-supplied contract cannot make its gaps optional" 1 \
  "submitted editable path is missing" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

# CASE FOLDING (B3). APFS is case-insensitive by default, so an editable entry
# spelled BENCHD names the real submodule: the byte-comparison guard passed it
# and `rm -rf BENCHD` then deleted the pinned scorer before the submission's
# copy landed on top. Every case variant must be refused, and the pinned content
# must survive the attempt.
for gitlink_variant in "BENCHD" "Benchd" "BENCHD/crates" ".GITMODULES" ".GitModules"; do
  fx="$(new_fixture)"
  python3 - "${fx}/trusted/benchmark.json" "${gitlink_variant}" <<'PY'
import json, sys
p, gitlink = sys.argv[1], sys.argv[2]
c = json.load(open(p))
c["editablePaths"] = ["src", gitlink]
json.dump(c, open(p, "w"))
PY
  mkdir -p "${fx}/sub/src" "${fx}/sub/BENCHD/crates"
  printf 'x\n' > "${fx}/sub/src/kernel.txt"
  printf 'attacker daemon\n' > "${fx}/sub/BENCHD/PIN"
  printf 'attacker daemon\n' > "${fx}/sub/BENCHD/crates/PIN"
  printf '[submodule "benchd"]\n\tpath = elsewhere\n' > "${fx}/sub/.GITMODULES"
  assert_exit "overlay/case-folded gitlink entry '${gitlink_variant}' refused" 1 \
    "measurement-harness surface" "${fx}/trusted" \
    env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"
  assert_equal "overlay/case-folded gitlink '${gitlink_variant}' left benchd intact" \
    "$(cat "${fx}/trusted/benchd/PIN")" "pinned measurement daemon"
done

# Q8 (issue #20): the overlay must not print its success trailer after overlaying
# NOTHING. The allowlist is the TRUSTED contract's; an empty, absent, or
# unparseable one is a broken trusted checkout, and a "success" that moved
# nothing lets a pipeline measure the trusted tree and score it as the
# submission. All three fail closed before the trailer.
fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c["editablePaths"] = []
json.dump(c, open(p, "w"))
PY
assert_exit "overlay/empty editablePaths fails closed, not a no-op success" 1 \
  "lists no editablePaths" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c.pop("editablePaths")
json.dump(c, open(p, "w"))
PY
assert_exit "overlay/missing editablePaths key fails closed, not a no-op success" 1 \
  "lists no editablePaths" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

fx="$(new_fixture)"
mkdir -p "${fx}/sub/src"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'this is not json {' > "${fx}/trusted/benchmark.json"
assert_exit "overlay/unparseable contract fails closed, not a no-op success" 1 \
  "does not parse" "${fx}/trusted" \
  env SUBMISSION_WORKTREE="${fx}/sub" "${SCRIPTS}/overlay-editable-paths.sh"

echo
echo "=== C. static review deterministic gates ==================================="

fx="$(new_fixture)"
fill "${fx}/trusted/src/lookup.txt" 5000
assert_exit "static-review/oversize file rejected" 1 \
  "above the per-file static review limit" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
assert_exit "static-review/benign surface passes" 0 "deterministic checks passed" \
  "${fx}/trusted" env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
assert_exit "static-review/review base set-but-empty fails closed" 1 \
  "refusing to fall back to whole-surface review" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA= \
  "${SCRIPTS}/submission-static-review-checks.sh"

# Q2 (issue #20): whole-surface mode was symlink-BLIND. `find -type f` skips a
# symlink at or under an editable path, so a payload reachable only through a
# link counted zero bytes and the surface passed. Both a link INSIDE a reviewed
# path and a reviewed path that IS a link now fail closed. (Diff mode already
# rejected these; whole-surface did not.)
fx="$(new_fixture)"
fill "${fx}/big-payload.bin" 50000
ln -s "${fx}/big-payload.bin" "${fx}/trusted/src/leak.txt"
assert_exit "static-review/whole-surface symlink inside an editable path rejected" 1 \
  "symlink" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
fill "${fx}/big-payload.bin" 50000
rm -f "${fx}/trusted/config.txt"
ln -s "${fx}/big-payload.bin" "${fx}/trusted/config.txt"
assert_exit "static-review/whole-surface editable path that IS a symlink rejected" 1 \
  "symlink" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

# --- B1 residual: whole-surface mode is an EXPLICIT opt-in -------------------
#
# Whole-surface mode reads the contract from the WORK TREE. That is only safe
# in the trusted checkout after the overlay, and nothing in the script can
# verify it stands in one. While the mode was the silent fallback for "no base
# sha", the whole self-widening primitive was one missing environment variable
# away: an attacker checkout with self-widened caps and BASE_SHA unset reviewed
# itself against its own contract and exited 0.
attacker_contract_fixture() { # -> echoes a trusted/ dir whose contract widens itself
  local fx
  fx="$(new_fixture)"
  python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"].update(
    maxTotalBytes=999999999, maxFileBytes=999999999, maxGrowthBytes=999999999
)
c["staticReviewTrackId"] = "attacker-track"
json.dump(c, open(p, "w"))
PY
  fill "${fx}/trusted/src/lookup.txt" 5000
  echo "${fx}/trusted"
}

adir="$(attacker_contract_fixture)"
assert_exit "static-review/no mode selected fails closed on an attacker contract" 1 \
  "no review mode selected" "${adir}" \
  env CONTRACT_PATH=benchmark.json "${SCRIPTS}/submission-static-review-checks.sh"

# The same tree WITH the opt-in is the control: the mode still works, and the
# caps it honours are the work tree's -- which is exactly why entering it has
# to be a deliberate act.
adir="$(attacker_contract_fixture)"
assert_exit "static-review/whole-surface works only with the explicit opt-in" 0 \
  "for track attacker-track" "${adir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

for bad_mode in "diff" "whole_surface" "WHOLE-SURFACE" "yes" " whole-surface"; do
  fx="$(new_fixture)"
  assert_exit "static-review/unknown review mode '${bad_mode}' fails closed" 1 \
    "no review mode selected" "${fx}/trusted" \
    env CONTRACT_PATH=benchmark.json "MLXFAST_SUBMISSION_REVIEW_MODE=${bad_mode}" \
    "${SCRIPTS}/submission-static-review-checks.sh"
done

# A mode that contradicts an explicit base sha is ambiguous, not a preference.
# The refusal lands before any git resolution, so the sha need not resolve.
fx="$(new_fixture)"
assert_exit "static-review/whole-surface mode contradicting a base sha fails closed" 1 \
  "contradicts MLXFAST_SUBMISSION_REVIEW_BASE_SHA" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json \
  MLXFAST_SUBMISSION_REVIEW_BASE_SHA=0000000000000000000000000000000000000000 \
  MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
assert_exit "static-review/track id mismatch fails closed" 1 \
  "is not the track this contract declares" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface MLXFAST_SUBMISSION_TRACK_ID=some-other-track \
  "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
assert_exit "static-review/track id set-but-empty fails closed" 1 \
  "refusing to fall back to a default review policy" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface MLXFAST_SUBMISSION_TRACK_ID= \
  "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editablePaths"] = []
json.dump(c, open(p, "w"))
PY
assert_exit "static-review/empty editablePaths fails closed" 1 \
  "lists no editablePaths" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface "${SCRIPTS}/submission-static-review-checks.sh"

fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["exemptPaths"] = c["editablePaths"]
json.dump(c, open(p, "w"))
PY
assert_exit "static-review/all-exempt surface fails closed" 1 \
  "lists no reviewable editablePaths" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface "${SCRIPTS}/submission-static-review-checks.sh"

# Growth cap, in diff mode against a real git base.
git_fixture() { # git_fixture -> echoes a git repo with a base commit
  local fx dir
  fx="$(new_fixture)"
  dir="${fx}/trusted"
  # Git chatter goes to stderr so it cannot contaminate the echoed path.
  {
    git -C "${dir}" init -q -b main
    git -C "${dir}" add -A
    git -C "${dir}" -c user.name=t -c user.email=t@e commit -q -m base
  } >&2
  echo "${dir}"
}

gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
fill "${gdir}/src/table.txt" 3000
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m smuggle
assert_exit "static-review/lookup-table growth beyond the growth cap rejected" 1 \
  "above the growth limit" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
printf 'a slightly better kernel\n' > "${gdir}/src/kernel.txt"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m tweak
assert_exit "static-review/ordinary optimization diff passes" 0 \
  "deterministic checks passed" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# --- B1: in diff mode every rule comes from the review BASE ------------------
#
# Upstream runs this step in the SUBMISSION checkout, before the overlay
# (qwen-mtp-ranked-benchmark.yml@bfab0de:1198-1204), so the work-tree
# benchmark.json is attacker-controlled. Reading the caps, the exemptions or the
# track id from it is a self-widening primitive. Each case below mutates the
# contract in the HEAD commit and asserts the mutation is inert.

# Cap widening: the head raises every cap to something enormous and smuggles a
# table that the BASE contract's growth cap refuses.
mutate_head_contract() { # mutate_head_contract DIR PYTHON-BODY
  local dir="$1" body="$2"
  python3 -c "${body}" "${dir}/benchmark.json"
}

gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
mutate_head_contract "${gdir}" '
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"].update(
    maxTotalBytes=999999999, maxFileBytes=999999999, maxGrowthBytes=999999999
)
json.dump(c, open(p, "w"))
'
fill "${gdir}/src/table.txt" 3000
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m self-widened-caps
assert_exit "static-review/diff mode: head-widened growth cap does not widen the review" 1 \
  "above the growth limit" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# Same vector against the per-file cap. The base already carries a file just
# under the 4096-byte per-file cap, so the head can cross it while adding less
# than the 2048-byte growth cap -- otherwise the growth refusal fires first and
# this case would not be about the per-file cap at all.
fx="$(new_fixture)"
gdir="${fx}/trusted"
fill "${gdir}/src/big.txt" 4000
{
  git -C "${gdir}" init -q -b main
  git -C "${gdir}" add -A
  git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m base
} >&2
base="$(git -C "${gdir}" rev-parse HEAD)"
mutate_head_contract "${gdir}" '
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["maxFileBytes"] = 999999
c["editableSurfaceByteBudget"]["maxTotalBytes"] = 999999999
json.dump(c, open(p, "w"))
'
fill "${gdir}/src/big.txt" 5000
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m self-widened-file-cap
assert_exit "static-review/diff mode: head-widened per-file cap does not widen the review" 1 \
  "above the per-file static review limit" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# Self-exemption: the head declares its own source path byte-budget exempt, so
# the smuggled table would leave the reviewed payload entirely.
gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
mutate_head_contract "${gdir}" '
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"]["exemptPaths"] = ["head", "src"]
json.dump(c, open(p, "w"))
'
fill "${gdir}/src/table.txt" 3000
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m self-exempted
assert_exit "static-review/diff mode: head-declared exemptions do not hold paths out" 1 \
  "above the growth limit" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# Policy selection: the head renames the track it is reviewed under. The base's
# id is the one that binds -- this is what becomes a policy-selection primitive
# once the LLM judge lands.
gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
mutate_head_contract "${gdir}" '
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["staticReviewTrackId"] = "serial"
json.dump(c, open(p, "w"))
'
printf 'a slightly better kernel\n' > "${gdir}/src/kernel.txt"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m self-selected-track
assert_exit "static-review/diff mode: head-rewritten track id does not select the policy" 0 \
  "for track test-track-v1" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# ... and a head that DELETES the track id cannot make the review unnamed
# either: the base still names it.
gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
mutate_head_contract "${gdir}" '
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c.pop("staticReviewTrackId")
json.dump(c, open(p, "w"))
'
printf 'a slightly better kernel\n' > "${gdir}/src/kernel.txt"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m dropped-track
assert_exit "static-review/diff mode: head-dropped track id falls back to the base's" 0 \
  "for track test-track-v1" "${gdir}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_BASE_SHA="${base}" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# --- B2: malformed caps fail closed, and the two enforcers AGREE -------------
#
# Every row is applied to the SAME contract key and run through BOTH enforcers;
# the assertion is that they reach the same verdict. Previously they disagreed
# on 4/4 malformed inputs: jq's `// empty` read `false` and `null` as absent
# (silent fall to the constant) while `-r` stringified `"2000000"` into a value
# that passed the integer regex, i.e. a submission-supplied STRING widened the
# cap it was checked against.
malformed_cap_case() { # malformed_cap_case LABEL JSON-LITERAL WANT-EXIT
  local label="$1" literal="$2" want="$3" fx
  fx="$(new_fixture)"
  python3 - "${fx}/trusted/benchmark.json" "${literal}" <<'PY'
import json, sys
p, literal = sys.argv[1], sys.argv[2]
c = json.load(open(p))
# Write a placeholder and splice the RAW literal in, so `5e5`, `null` and
# `false` reach the enforcers exactly as an attacker would have written them
# rather than as whatever Python round-trips them to.
c["editableSurfaceByteBudget"]["maxTotalBytes"] = "@@LITERAL@@"
text = json.dumps(c).replace('"@@LITERAL@@"', literal)
open(p, "w").write(text)
PY
  assert_exit "malformed-cap/${label}: Swift enforcer" "${want}" "" \
    "${fx}/trusted" budget verify benchmark.json
  assert_exit "malformed-cap/${label}: shell enforcer" "${want}" "" \
    "${fx}/trusted" env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
    "${SCRIPTS}/submission-static-review-checks.sh"
}

malformed_cap_case 'string "2000000"' '"2000000"' 1
malformed_cap_case 'string "99999999"' '"99999999"' 1
malformed_cap_case 'boolean false'    'false'      1
malformed_cap_case 'boolean true'     'true'       1
malformed_cap_case 'explicit null'    'null'       1
malformed_cap_case 'fractional 3.5'   '3.5'        1
malformed_cap_case 'negative -1'      '-1'         1
malformed_cap_case 'array []'         '[]'         1
malformed_cap_case 'object {}'        '{}'         1
# Exponent notation IS a positive integer JSON number; both must ACCEPT it and
# resolve the same 500000.
malformed_cap_case 'exponent 5e5 (legal)' '5e5' 0

# A whole budget object that is not an object must be as fatal as one bad key.
fx="$(new_fixture)"
python3 - "${fx}/trusted/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editableSurfaceByteBudget"] = None
json.dump(c, open(p, "w"))
PY
assert_exit "malformed-cap/budget object is null: Swift enforcer" 1 "" \
  "${fx}/trusted" budget verify benchmark.json
assert_exit "malformed-cap/budget object is null: shell enforcer" 1 "" \
  "${fx}/trusted" env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  "${SCRIPTS}/submission-static-review-checks.sh"

echo
echo "=== D. modifiable-surface gate (enforce-modifiable-surface.sh) ============="

gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
printf 'attacker CLI\n' > "${gdir}/Sources/MLXFastCLI/main.swift"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m outside
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/write outside editablePaths rejected" 1 \
  "outside the modifiable surface" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
printf 'a better kernel\n' > "${gdir}/src/kernel.txt"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m inside
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/write inside editablePaths accepted" 0 "" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

# The allowlist comes from the BASE commit: a submission that widens its own
# contract in the HEAD commit is still judged by the base's surface.
gdir="$(git_fixture)"
base="$(git -C "${gdir}" rev-parse HEAD)"
python3 - "${gdir}/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editablePaths"] = ["src", "config.txt", "head.manifest.json", "head", "Sources"]
json.dump(c, open(p, "w"))
PY
printf 'attacker CLI\n' > "${gdir}/Sources/MLXFastCLI/main.swift"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m widen
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/self-widened contract does not grant access" 1 \
  "outside the modifiable surface" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

# B3: the gate carries the gitlink guard the other two layers already had, and
# it is case-folded. Both cases below drift the BASE (trusted) contract to list
# the gitlink, so the allowlist WOULD admit the write -- only the guard refuses
# it. The second spells the changed path in a different case, which on the
# case-insensitive ranked filesystem is the same file.
drifted_base_fixture() { # drifted_base_fixture ENTRY -> echoes a repo whose BASE contract lists ENTRY
  local entry="$1" fx dir
  fx="$(new_fixture)"
  dir="${fx}/trusted"
  python3 - "${dir}/benchmark.json" "${entry}" <<'PY'
import json, sys
p, entry = sys.argv[1], sys.argv[2]
c = json.load(open(p))
c["editablePaths"] = c["editablePaths"] + [entry]
json.dump(c, open(p, "w"))
PY
  {
    git -C "${dir}" init -q -b main
    git -C "${dir}" add -A
    git -C "${dir}" -c user.name=t -c user.email=t@e commit -q -m base
  } >&2
  echo "${dir}"
}

gdir="$(drifted_base_fixture benchd)"
base="$(git -C "${gdir}" rev-parse HEAD)"
printf 'attacker daemon\n' > "${gdir}/benchd/PIN"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m repoint-scorer
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/gitlink write refused even when the base contract lists it" 1 \
  "reaches the measurement-harness surface" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

# The LIVE pin paths (the retired-submodule cases above and below stay because
# all three guard layers still carry those spellings). Writing benchd.pin
# repoints which binary scores the submission; writing benchd-bin/benchctl swaps
# the bytes after fetch-benchd.sh verified them. Both must be refused even when
# the trusted contract has drifted to list them.
for live_pin in benchd.pin benchd-bin/benchctl; do
  gdir="$(drifted_base_fixture "${live_pin}")"
  base="$(git -C "${gdir}" rev-parse HEAD)"
  mkdir -p "$(dirname "${gdir}/${live_pin}")"
  printf 'attacker harness\n' > "${gdir}/${live_pin}"
  git -C "${gdir}" add -A
  git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m repoint-pinned-harness
  head_sha="$(git -C "${gdir}" rev-parse HEAD)"
  assert_exit "surface-gate/${live_pin} write refused even when the base contract lists it" 1 \
    "reaches the measurement-harness surface" "${gdir}" \
    env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
    "${SCRIPTS}/enforce-modifiable-surface.sh"
done

gdir="$(drifted_base_fixture BENCHD)"
base="$(git -C "${gdir}" rev-parse HEAD)"
# Record the path in the index directly: on a case-insensitive filesystem a
# worktree write to BENCHD/PIN would be stored under the existing benchd/ name,
# so the case variant has to be staged as a blob to exist at all.
case_blob="$(printf 'attacker daemon\n' | git -C "${gdir}" hash-object -w --stdin)"
git -C "${gdir}" update-index --add --cacheinfo "100644,${case_blob},BENCHD/PIN"
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m repoint-scorer-case-variant
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/case-folded gitlink write refused" 1 \
  "reaches the measurement-harness surface" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

gdir="$(drifted_base_fixture .gitmodules)"
base="$(git -C "${gdir}" rev-parse HEAD)"
printf '[submodule "benchd"]\n\tpath = attacker\n' > "${gdir}/.gitmodules"
git -C "${gdir}" add -A
git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m repoint-pointer
head_sha="$(git -C "${gdir}" rev-parse HEAD)"
assert_exit "surface-gate/.gitmodules write refused even when the base contract lists it" 1 \
  "reaches the measurement-harness surface" "${gdir}" \
  env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
  "${SCRIPTS}/enforce-modifiable-surface.sh"

# B3 (device:inode arm, the reason the guard is NOT ASCII-fold-only). A non-ASCII
# spelling of a CURRENT forbidden path IS constructible on the ranked APFS box:
# `.gitmoduleſ` (trailing U+017F LATIN SMALL LETTER LONG S) case-folds to
# `.gitmodules` and names the SAME inode, but `tr '[:upper:]' '[:lower:]'` leaves
# U+017F untouched, so the fold arm alone misses it. The drifted base contract
# lists the long-s spelling, so the allowlist admits the write; only the
# device:inode arm -- fed RAW bytes because the gate reads the diff with
# core.quotePath=false -- refuses it. Two independent reverts green this: drop the
# device arm (fold-only) and the raw long-s admits; drop quotePath=false and the
# arm stats a C-quoted literal (`".gitmodule\305\277"`) that names no file, so the
# gitlink refusal vanishes and the path is mis-rejected as merely out-of-surface.
# The parity claim: overlay-editable-paths.sh and lint-benchmark-manifest.py both
# refuse this same spelling (the overlay by the identical device:inode arm on its
# raw-JSON editablePaths, the linter by str.casefold(), which DOES fold U+017F).
#
# Guarded by an inode-collision probe: the vulnerability exists only where the
# filesystem folds U+017F (APFS). On a case/normalization-SENSITIVE filesystem
# (ext4, the hosted CI runner) `.gitmoduleſ` is a genuinely distinct file that
# does not touch the scorer, so there is nothing to refuse and the check is
# skipped rather than asserted.
fs_folds_longs() { # true when this filesystem folds `.gitmoduleſ` onto `.gitmodules`
  local d ref alt; d="$(mktemp -d)"; printf x > "${d}/.gitmodules"
  ref="$(stat -f '%d:%i' "${d}/.gitmodules" 2>/dev/null \
      || stat -c '%d:%i' "${d}/.gitmodules" 2>/dev/null)"
  alt="$(stat -f '%d:%i' "${d}/$(printf '.gitmodule\xc5\xbf')" 2>/dev/null \
      || stat -c '%d:%i' "${d}/$(printf '.gitmodule\xc5\xbf')" 2>/dev/null)"
  rm -rf "${d}"
  [[ -n "${ref}" && "${ref}" == "${alt}" ]]
}
if fs_folds_longs; then
  gdir="$(drifted_base_fixture "$(printf '.gitmodule\xc5\xbf')")"
  base="$(git -C "${gdir}" rev-parse HEAD)"
  # APFS folds a worktree write to `.gitmoduleſ` onto the existing `.gitmodules`,
  # so the long-s spelling can exist only as a staged blob, never as its own file.
  longs_blob="$(printf 'attacker daemon\n' | git -C "${gdir}" hash-object -w --stdin)"
  git -C "${gdir}" update-index --add \
    --cacheinfo "100644,${longs_blob},$(printf '.gitmodule\xc5\xbf')"
  git -C "${gdir}" -c user.name=t -c user.email=t@e commit -q -m repoint-pointer-longs
  head_sha="$(git -C "${gdir}" rev-parse HEAD)"
  assert_exit "surface-gate/non-ASCII long-s .gitmoduleſ write refused (device:inode, raw bytes)" 1 \
    "reaches the measurement-harness surface" "${gdir}" \
    env BASE_SHA="${base}" HEAD_SHA="${head_sha}" CONTRACT_PATH=benchmark.json \
    "${SCRIPTS}/enforce-modifiable-surface.sh"
else
  echo "skip  surface-gate/non-ASCII long-s .gitmoduleſ (filesystem does not fold U+017F; no inode collision here)"
fi

echo
echo "=== E. the real tree and manifest/enforcer drift ==========================="

assert_exit "real-tree/benchmark.json surface fits its own budget" 0 "verified" \
  "${REPO_ROOT}" budget verify benchmark.json

assert_exit "real-tree/deterministic static-review gates pass" 0 \
  "deterministic checks passed" "${REPO_ROOT}" \
  env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR="${WORK}/real-review" \
  "${SCRIPTS}/submission-static-review-checks.sh"

# Drift: the manifest, the Swift enforcer's resolution of it, and the shell
# gate's resolution of it must agree on every cap.
manifest_caps="$(python3 - "${REPO_ROOT}/benchmark.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))["editableSurfaceByteBudget"]
for k in (
    "maxTotalBytes",
    "maxFileBytes",
    "maxGrowthBytes",
    "exemptPathMaxBytes",
    "exemptPathMaxFileBytes",
):
    print(f"{k}={b[k]}")
PY
)"
# THE ANCHOR FOR EVERY DRIFT COMPARISON BELOW. All five compare one derived
# string against another, and assert_equal is a pure string compare: if BOTH
# sides come back empty -- the manifest read failing takes manifest_caps and,
# through it, expected_shell_caps with it -- every one of them passes on having
# compared nothing. Pinning the shape of this one value is what makes the other
# five mean something, so it is asserted before it is used, not assumed.
assert_equal "drift/manifest caps are readable and well-formed (anchors the five comparisons below)" \
  "$(printf '%s\n' "${manifest_caps}" | grep -cE '^(maxTotalBytes|maxFileBytes|maxGrowthBytes|exemptPathMaxBytes|exemptPathMaxFileBytes)=[1-9][0-9]*$')" \
  "5"

swift_caps="$( (cd "${REPO_ROOT}" && budget limits benchmark.json) )"
assert_equal "drift/Swift enforcer resolves the manifest caps" "${swift_caps}" "${manifest_caps}"

shell_caps="$( (cd "${REPO_ROOT}" && env CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR="${WORK}/drift-review" \
  "${SCRIPTS}/submission-static-review-checks.sh" 2>/dev/null \
  | sed -n 's/.*limits total=\([0-9]*\) file=\([0-9]*\) growth=\([0-9]*\).*/maxTotalBytes=\1\nmaxFileBytes=\2\nmaxGrowthBytes=\3/p') )"
# Both exempt caps are dropped: the shell static-review gate resolves the three
# CODE caps only, and exempt paths are bounded by the Swift walk (and benchd),
# never by it. `^exemptPathMax` catches exemptPathMaxFileBytes too -- a plain
# `grep -v exemptPathMaxBytes` does NOT, since that is not a substring of it.
expected_shell_caps="$(printf '%s\n' "${manifest_caps}" | grep -v '^exemptPathMax')"
assert_equal "drift/shell gate resolves the manifest caps" "${shell_caps}" "${expected_shell_caps}"

# D1 SINGLE-SOURCE, BOTH ENFORCERS. Issue #20 / finding B6 / R1.15 is RULED
# (David 2026-08-20): align shell to Swift, resolution is manifest > constant.
# The shell gate used to consult MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_* AHEAD of
# the manifest while the Swift enforcer had no environment path at all, so "the
# manifest is the single source" held for one enforcer and not the other.
#
# Run with all three overrides set to values that are legal, in range, and
# nothing like the contract's. If any of them still won, the caps below would
# come back as the override values and this assertion would name which. Note
# the caps must remain manifest-equal, NOT merely unchanged-from-each-other:
# an override that happened to match the manifest would prove nothing.
shell_caps_with_env="$( (cd "${REPO_ROOT}" && env CONTRACT_PATH=benchmark.json \
  MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
  MLXFAST_SUBMISSION_STATIC_REVIEW_OUT_DIR="${WORK}/env-override-review" \
  MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES=2000001 \
  MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_FILE_BYTES=131071 \
  MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_GROWTH_BYTES=65535 \
  "${SCRIPTS}/submission-static-review-checks.sh" 2>/dev/null \
  | sed -n 's/.*limits total=\([0-9]*\) file=\([0-9]*\) growth=\([0-9]*\).*/maxTotalBytes=\1\nmaxFileBytes=\2\nmaxGrowthBytes=\3/p') )"
assert_equal "drift/shell gate ignores the retired cap env overrides (manifest wins)" \
  "${shell_caps_with_env}" "${expected_shell_caps}"

# The two assertions below both COUNT OCCURRENCES IN A FILE, and a count over a
# file that cannot be read is zero -- which is the answer one of them wants. So
# the file is proven readable first, and proven to still contain the construct
# being counted. Without these, renaming or deleting the script turns the
# env-override assertion green on a script that is not there at all.
assert_equal "drift/static-review script is readable where this suite looks for it" \
  "$([ -r "${SCRIPTS}/submission-static-review-checks.sh" ] && echo readable || echo MISSING)" \
  "readable"
assert_equal "drift/static-review still resolves three caps through resolve_cap" \
  "$(grep -c '^resolve_cap ' "${SCRIPTS}/submission-static-review-checks.sh")" "3"

# And the variables are GONE from the CODE, not merely outranked: a script that
# still reads them has kept a second resolution path a later edit can re-promote.
# Comment lines are stripped first -- the ruling's own rationale names the three
# retired variables, and that prose is the record, not a resolution path.
assert_equal "drift/shell gate reads no cap env override at all" \
  "$(grep -vE '^[[:space:]]*#' "${SCRIPTS}/submission-static-review-checks.sh" \
      | grep -cE 'MLXFAST_SUBMISSION_STATIC_REVIEW_MAX(_FILE|_GROWTH)?_BYTES')" "0"

# And the compiled-in fallbacks (used only when a contract declares no caps)
# must still equal what the manifest declares, so a contract that loses a key
# does not silently change the budget.
fallback_caps="$(python3 - "${REPO_ROOT}/Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
names = {
    "defaultMaxTotalBytes": "maxTotalBytes",
    "defaultMaxFileBytes": "maxFileBytes",
    "defaultMaxGrowthBytes": "maxGrowthBytes",
    "defaultExemptPathMaxBytes": "exemptPathMaxBytes",
    "defaultExemptPathMaxFileBytes": "exemptPathMaxFileBytes",
}
for swift_name, key in names.items():
    m = re.search(rf"{swift_name}\s*=\s*([0-9_]+)", src)
    print(f"{key}={int(m.group(1).replace('_', ''))}")
PY
)"
assert_equal "drift/Swift fallback constants equal the manifest" "${fallback_caps}" "${manifest_caps}"

script_fallbacks="$(python3 - "${SCRIPTS}/submission-static-review-checks.sh" <<'PY'
import re, sys
# `resolve_cap VAR_NAME CONTRACT_KEY FALLBACK` -- three arguments since the cap
# environment overrides were retired (issue #20, ruled 2026-08-20). The ENV_VAR
# argument that used to sit between VAR_NAME and CONTRACT_KEY is gone. A hard
# failure here on a signature change is the point: the fallback constants are
# read out of the real call sites, so this cannot silently stop matching.
src = open(sys.argv[1]).read()
for key in ("maxTotalBytes", "maxFileBytes", "maxGrowthBytes"):
    m = re.search(rf"^resolve_cap \w+ {key} ([0-9]+)$", src, re.MULTILINE)
    if m is None:
        sys.exit(f"no resolve_cap call site found for {key}")
    print(f"{key}={m.group(1)}")
PY
)"
assert_equal "drift/shell fallback constants equal the manifest" "${script_fallbacks}" \
  "${expected_shell_caps}"

# --- linter injection harness (checks 3 and 3c) ------------------------------
#
# Both remaining lint groups work the same way: inject ONE entry into a copy of
# the REAL manifest -- so the linter resolves paths against the real tree -- and
# count the diagnostics that name it. Defined once, above both groups.

# inject_manifest DST BUCKET ENTRY -- copy the real manifest with ENTRY appended
# to BUCKET (dotted path into editableSurfaceByteBudget for exemptPaths).
inject_manifest() {
  python3 - "${REPO_ROOT}/benchmark.json" "$1" "$2" "$3" <<'PY'
import json, sys
src, dst, bucket, entry = sys.argv[1:5]
c = json.load(open(src))
if bucket == "editableSurfaceByteBudget.exemptPaths":
    # The real manifest declares NO exemptPaths since the requant-only ruling
    # (2026-08-26): nothing rides in a submission that needs holding out of the
    # code budget. This driver still has to be able to INJECT one, because the
    # property under test is that the linter's trusted-scope guard covers the
    # exemptPaths bucket -- a guard that stopped being exercised the day the
    # bucket went away would be a guard nobody notices losing. `.get(...)`
    # rather than `[...]`: an absent key must seed an empty list, not raise a
    # KeyError that aborts python and leaves the assertion reading
    # LINTER-DID-NOT-RUN (which is how this was caught).
    c["editableSurfaceByteBudget"]["exemptPaths"] = (
        c["editableSurfaceByteBudget"].get("exemptPaths", []) + [entry]
    )
else:
    c[bucket] = c[bucket] + [entry]
json.dump(c, open(dst, "w"))
PY
}

# lint_entry_hits PREFIX BUCKET ENTRY -- how many failures whose diagnostic
# starts PREFIX the injected entry produces. grep -cF, never -c: the entry is
# DATA. An unescaped BRE made '.' a wildcard, so 'Package.swift' would have
# counted a line about 'PackageXswift' -- the assertion was matching something
# looser than what it claimed to assert.
# PROOF OF LIFE. Every caller that expects ZERO hits -- the "not refused"
# near-misses -- is asking a grep to find nothing, and a grep over the output of
# a linter that never ran finds nothing too. Measured, with the linter replaced
# by a script that raises on import: EIGHT assertions went green, including all
# six near-miss cases, on a run where nothing was linted at all. The `|| true`
# that used to sit here is what made the two indistinguishable.
#
# So the linter must show it reached its own summary before any count is
# believed. When it did not, this prints a sentinel instead of a number: no
# caller expects it, so every assertion through this helper -- zero-expecting
# and one-expecting alike -- fails loudly with the sentinel in the diff.
# lint_needle_count OUTPUT NEEDLE -- fixed-string line count of NEEDLE in
# OUTPUT, or the sentinel if OUTPUT is not a COMPLETED linter run.
#
# The linter ends every completed run with one of these two summaries, on a
# clean run and a failing one alike, so matching either distinguishes "ran and
# found nothing" from "did not run" -- which a bare count cannot.
#
# Note the `|| true` on the counting grep: `grep -c` exits 1 when the count is
# zero, and zero is a legitimate answer here. Without it this helper's own exit
# status would depend on the value it printed.
lint_needle_count() {
  local out="$1" needle="$2"
  if ! printf '%s\n' "${out}" \
      | grep -qE '^(all [0-9]+ checks passed|[0-9]+ FAILURE\(S\), [0-9]+ check\(s\) passed)'; then
    printf 'LINTER-DID-NOT-RUN\n'
    return 0
  fi
  printf '%s\n' "${out}" | grep -cF "${needle}" || true
}

lint_entry_hits() {
  local prefix="$1" bucket="$2" entry="$3"
  local manifest="${WORK}/lint-scope-$(printf '%s' "${prefix}/${bucket}/${entry}" \
    | tr -c 'A-Za-z0-9' '_').json"
  inject_manifest "${manifest}" "${bucket}" "${entry}"
  local out
  out="$( (cd "${REPO_ROOT}" && python3 tools/lint-benchmark-manifest.py \
      --repo-root "${REPO_ROOT}" --manifest "${manifest}" 2>&1) )"
  lint_needle_count "${out}" "${prefix}: ${bucket} entry '${entry}'"
}

# trusted_scope_hits BUCKET ENTRY -- how many 3c failures the entry produces.
trusted_scope_hits() {
  lint_entry_hits "trusted scope" "$1" "$2"
}

# gitlink_hits BUCKET ENTRY -- how many check-3 failures the entry produces.
# Check 3 and check 3c walk the SAME buckets, so a shape refusal reported only
# by 3c is 3c covering for check 3, not check 3 binding. This counts check 3's
# own diagnostics and nothing else.
gitlink_hits() {
  lint_entry_hits "gitlink exclusion" "$1" "$2"
}

# trusted_scope_roster -- the linter's TRUSTED_SCOPE, one path per line, read
# from the linter itself rather than restated here.
trusted_scope_roster() {
  python3 - "${REPO_ROOT}/tools/lint-benchmark-manifest.py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_lint", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("\n".join(mod.TRUSTED_SCOPE))
PY
}

trusted_scope_roster_size() { trusted_scope_roster | grep -c .; }

# overlap_without_samefile ENTRY SCOPE -- _trusted_scope_overlap with the
# filesystem arm DISABLED, printing the relation word or "None".
#
# WHY THIS EXISTS, and why the injection-based cases below cannot replace it.
# This suite runs in the macos-26 job, on APFS, which is case-INSENSITIVE and
# resolves doubled separators. _prefixes() dropped empty segments even BEFORE
# _normalize() existed, so on this filesystem the samefile arm caught
# 'Sources//MLXFastCore' and 'sources//mlxfastcore' all along -- reverting
# _normalize() still leaves every one of those injection cases green here. They
# assert that SOMETHING refuses the entry; they cannot say which arm did, and
# the arm that matters is the other one.
#
# The lint job runs on ubuntu-latest, where ext4 is case-sensitive and neither
# spelling resolves, so the LEXICAL arm is the only binding arm there. Stubbing
# _same_file to False is that runner, reproduced deterministically on this one:
# it is the only form of this assertion that binds on both.
overlap_without_samefile() {
  python3 - "${REPO_ROOT}/tools/lint-benchmark-manifest.py" "${REPO_ROOT}" "$1" "$2" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_lint", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
lint = mod.Linter(sys.argv[2], "benchmark.json")
# Every path "fails to resolve": the case-sensitive-runner condition for any
# spelling that differs from the on-disk one by case or by empty segments.
lint._same_file = lambda a, b: False
print(lint._trusted_scope_overlap(sys.argv[3], sys.argv[4]))
PY
}

# trusted_scope_under PREFIX -- how many TRUSTED_SCOPE paths lie under PREFIX.
# The "contains" expectation below is DERIVED from this, never written as a
# literal: three of the nine are under Sources/, and a magic 3 would silently
# stop matching the moment a roster entry is added or moved -- as one just was.
# (Deriving it is why that assertion needed no edit when the roster went from
# seven to nine, and the literal count above did.)
trusted_scope_under() {
  trusted_scope_roster | grep -cE "^$(printf '%s' "$1" | sed 's|[][\.*^$/]|\\&|g')/"
}

# B3, manifest side: the linter's gitlink exclusion must fold case too, so a
# contract that spells the submodule BENCHD is refused at rest as well as by the
# overlay at run time.
for lint_variant in "BENCHD" ".GITMODULES"; do
  assert_equal "lint/case-folded gitlink entry '${lint_variant}' refused" \
    "$(gitlink_hits editablePaths "${lint_variant}")" "1"
done

# Trusted-scope overlap (linter check 3c, the interim rest-state form of
# upstream's verify_contract_does_not_expose_trusted_scope). Same technique as
# the gitlink cases above, asserting that the linter refuses the entry and names
# the bucket it came from. The whole point of this check is that the SEMANTICS
# are decided here rather than inherited, so exact / inside / containing /
# near-miss are each asserted in both directions.

# 1. Every protected path must be refused when named exactly. All SEVEN: the
#    last two, '.github' and 'tools', are this repository's deliberate
#    divergence from upstream's five-entry roster (ruled by David 2026-08-20 --
#    parity doctrine governs measurement surfaces, not security posture), and
#    they are asserted here exactly like the inherited five so the divergence is
#    a tested property rather than a comment.
for scope_path in "Sources/MLXFastTrustedHarness" "Sources/MLXFastCLI" \
                  "Sources/MLXFastCore" "Package.swift" "Package.resolved" \
                  ".github" "tools"; do
  assert_equal "lint/trusted-scope entry '${scope_path}' refused (equals)" \
    "$(trusted_scope_hits editablePaths "${scope_path}")" "1"
done

# 2. Prefix semantics, BOTH directions. An entry under a trusted path and an
#    entry that CONTAINS one are both overlap; neither direction is inherited
#    from whatever the byte-budget enforcers do with exemptPaths.
assert_equal "lint/trusted-scope entry inside a trusted dir refused" \
  "$(trusted_scope_hits editablePaths "Sources/MLXFastCore/Constants.swift")" "1"
assert_equal "lint/trusted-scope entry containing trusted dirs refused" \
  "$(trusted_scope_hits editablePaths "Sources")" "$(trusted_scope_under Sources)"

# 3. The separator is load-bearing: a sibling whose name merely STARTS with a
#    trusted path is not an overlap and must survive check 3c. (It still trips
#    check 2 for not existing; only 3c hits are counted.)
assert_equal "lint/trusted-scope near-miss 'Sources/MLXFastCoreExtras' not refused" \
  "$(trusted_scope_hits editablePaths "Sources/MLXFastCoreExtras")" "0"
assert_equal "lint/trusted-scope near-miss 'Package.swift.bak' not refused" \
  "$(trusted_scope_hits editablePaths "Package.swift.bak")" "0"

# 4. Case folding, for the same APFS reason as the gitlink guard.
assert_equal "lint/trusted-scope case-folded 'sources/mlxfastcore' refused" \
  "$(trusted_scope_hits editablePaths "sources/mlxfastcore")" "1"

# 5. The other two buckets are checked by the SAME relation. exemptPaths exempts
#    bytes from the code budget, not the path from the overlay, so an exempt
#    entry over trusted surface is refused exactly like an editable one.
assert_equal "lint/trusted-scope optionalEditablePaths entry refused" \
  "$(trusted_scope_hits optionalEditablePaths "Package.resolved")" "1"
assert_equal "lint/trusted-scope exemptPaths entry refused" \
  "$(trusted_scope_hits editableSurfaceByteBudget.exemptPaths "Sources/MLXFastCLI")" "1"

# 6. Entry spellings that DEFEAT the overlap arithmetic rather than failing it,
#    and so must be refused before it runs. A real absolute path to a trusted
#    directory is the sharp one: os.path.join(root, rel) discards root for an
#    absolute rel, so it compared as neither equal nor same-file. '' / '.' / './'
#    denote the repo root and render as an EMPTY prefix list, so nothing was
#    compared at all. All four are already refused at run time by
#    overlay-editable-paths.sh:87,91-96 -- this is defence in depth, not a live
#    hole -- and the linter now mirrors that script's validity rule exactly.
assert_equal "lint/trusted-scope absolute path to a trusted dir refused" \
  "$(trusted_scope_hits editablePaths "${REPO_ROOT}/Sources/MLXFastCore")" "1"
for degenerate in "" "." "./"; do
  assert_equal "lint/trusted-scope repo-root spelling '${degenerate}' refused" \
    "$(trusted_scope_hits editablePaths "${degenerate}")" "1"
done

# 7. EMPTY SEGMENTS. 'Sources//MLXFastCore' names the real trusted directory on
#    every POSIX filesystem, but the lexical arm used to compare the raw
#    strip('/') form while _prefixes() dropped the empty segment -- so the two
#    arms disagreed and only the samefile arm caught it. That arm cannot rescue
#    a WRONG-CASE spelling on the hosted lint job's case-SENSITIVE ext4, which
#    is the only place this check runs automatically, so 'sources//mlxfastcore'
#    bound nowhere at all (verified: _trusted_scope_overlap returned None under
#    case-sensitive samefile semantics). Both sides now go through the same
#    segment join. Asserted at the depths that matter: the trusted dir itself,
#    a file under it, and a doubled separator inside the path rather than at the
#    seam.
for empty_seg in "Sources//MLXFastCore" "sources//mlxfastcore" "SOURCES//MLXFASTCORE" \
                 "Sources///MLXFastCore" "Sources//MLXFastCore/Constants.swift" \
                 "Sources/MLXFastCore//Constants.swift"; do
  assert_equal "lint/trusted-scope empty-segment spelling '${empty_seg}' refused" \
    "$(trusted_scope_hits editablePaths "${empty_seg}")" "1"
done

# 8. Normalising must not WIDEN the relation: the separator is still load-bearing
#    after the empty segment is dropped, so a doubled-separator near-miss stays
#    legal for 3c exactly as its single-separator twin does.
for empty_seg_miss in "Sources//MLXFastCoreExtras" "Sources//MLXFastModel"; do
  assert_equal "lint/trusted-scope empty-segment near-miss '${empty_seg_miss}' not refused" \
    "$(trusted_scope_hits editablePaths "${empty_seg_miss}")" "0"
done

# 8b. THE SAME PROPERTY, WITH THE FILESYSTEM ARM DISABLED -- see
#     overlap_without_samefile() for why cases 7 and 8 above cannot substitute
#     for this. On APFS the samefile arm answers every one of them, so they stay
#     green even with _normalize() reverted; the case-sensitive lint runner has
#     no such arm and the LEXICAL comparison is the only thing standing between
#     'sources//mlxfastcore' and the trusted harness. These assert the relation
#     WORD, not merely that a hit occurred, so a future change that reaches the
#     right verdict by the wrong arm is still visible here.
for empty_seg_lex in "Sources//MLXFastCore" "sources//mlxfastcore" \
                     "SOURCES//MLXFASTCORE" "Sources///MLXFastCore" \
                     "Sources//MLXFastCore/" "/Sources//MLXFastCore"; do
  assert_equal "lint/trusted-scope lexical-only '${empty_seg_lex}' equals the trusted dir" \
    "$(overlap_without_samefile "${empty_seg_lex}" "Sources/MLXFastCore")" "equals"
done

assert_equal "lint/trusted-scope lexical-only file under the trusted dir is inside it" \
  "$(overlap_without_samefile "Sources//MLXFastCore/Constants.swift" "Sources/MLXFastCore")" \
  "is inside"
assert_equal "lint/trusted-scope lexical-only 'Sources' contains the trusted dir" \
  "$(overlap_without_samefile "Sources" "Sources/MLXFastCore")" "contains"

# And the near-misses must stay None with the filesystem arm gone too: the
# lexical arm has to be spelling-proof WITHOUT becoming greedy.
for empty_seg_lex_miss in "Sources//MLXFastCoreExtras" "Sources//MLXFastModel" \
                          "Sources//MLXFastCoreExtras/Thing.swift" "SourcesX//MLXFastCore"; do
  assert_equal "lint/trusted-scope lexical-only near-miss '${empty_seg_lex_miss}' does not overlap" \
    "$(overlap_without_samefile "${empty_seg_lex_miss}" "Sources/MLXFastCore")" "None"
done

# 9. CHECK 3 BINDS ON ITS OWN. The gitlink guard walks the same three buckets as
#    3c, so until it called the shared validity rule, an absolute or dot-spelled
#    entry naming the pinned measurement daemon produced a 3c failure and NO
#    check-3 failure -- the run went red, but not because the gitlink guard had
#    noticed. These count check 3's diagnostics only. The absolute case is the
#    sharp one: os.path.join(root, rel) discards root for an absolute rel and
#    strip('/') re-roots it, so a path naming the REAL benchd compared as
#    neither equal, inside, containing, nor same-file.
for shape in "${REPO_ROOT}/benchd" "/benchd" "./benchd" "benchd/../benchd" \
             ":benchd" "" "." "./"; do
  assert_equal "lint/gitlink shape refusal '${shape}' bound by check 3" \
    "$(gitlink_hits editablePaths "${shape}")" "1"
done

# 10. And the shape rule must not swallow the legitimate entries: every path the
#     REAL manifest declares is a legal spelling, so check 3 emits no FAILURE on
#     it. Matched on the "FAIL  " prefix, not on "gitlink exclusion:" alone --
#     the check's own success line contains that phrase too, and an assertion
#     that counts a passing line is an assertion that cannot fail.
#
#     A zero-count assertion needs TWO guards or it proves nothing. The linter's
#     EXIT STATUS is asserted separately rather than swallowed by `|| true`, so a
#     traceback (missing manifest, import error, a renamed flag) reds here
#     instead of producing no output and counting a satisfying zero; and the
#     positive control below proves the needle still matches SOMETHING, so a
#     diagnostic reworded out from under this grep cannot read as a pass.
real_lint_out="$( (cd "${REPO_ROOT}" && python3 tools/lint-benchmark-manifest.py \
    --repo-root "${REPO_ROOT}" --gitlink-targets report 2>&1) )"
real_lint_status=$?
assert_equal "lint/gitlink the real manifest lints cleanly (exit status asserted, not discarded)" \
  "${real_lint_status}" "0"
# Proof of life on THIS site too, not only via the two assertions either side of
# it. Those do fail when the linter dies -- but a neighbouring assertion failing
# is not this one binding, which is the same distinction check 3 itself had to
# be taught (case 9). Same sentinel as lint_entry_hits, for the same reason.
assert_equal "lint/gitlink no shape refusal on the real manifest" \
  "$(lint_needle_count "${real_lint_out}" "FAIL  gitlink exclusion:")" "0"

poscontrol_manifest="${WORK}/lint-gitlink-poscontrol.json"
inject_manifest "${poscontrol_manifest}" editablePaths "/etc/passwd"
assert_equal "lint/gitlink positive control: the needle matches when check 3 does fail" \
  "$( (cd "${REPO_ROOT}" && python3 tools/lint-benchmark-manifest.py \
        --repo-root "${REPO_ROOT}" --manifest "${poscontrol_manifest}" 2>&1 || true ) \
      | grep -cF "FAIL  gitlink exclusion:" )" "1"

# 11. Empty segments at the gitlink too, for the same reason as case 7: the two
#     guards share one reduction and must agree on what an entry spells.
for gitlink_seg in "benchd//scripts" "BENCHD//scripts" "benchd//"; do
  assert_equal "lint/gitlink empty-segment spelling '${gitlink_seg}' refused" \
    "$(gitlink_hits editablePaths "${gitlink_seg}")" "1"
done

# 12. THE TWO DIVERGENT ROSTER ENTRIES, at the same depths as the inherited
#     five. These are what the ruling actually bought: '.github' and 'tools'
#     hold every gate in this repository -- the overlay, the static review, the
#     surface gate, this linter, this suite, the CI tripwires -- so an entry
#     reaching either of them is a submission proposing to edit its own judge.
#
#     'contains' is not asserted for them because it is not reachable: both are
#     single-segment paths at the repository root, so the only entry that could
#     contain one is the root itself, which the shape rule refuses before the
#     overlap arithmetic runs (case 6). The other two relations bind.
for divergent_scope in ".github" "tools"; do
  assert_equal "lint/trusted-scope divergent entry '${divergent_scope}' refused (equals)" \
    "$(trusted_scope_hits editablePaths "${divergent_scope}")" "1"
done

# Inside: a real file under each, so the assertion is about a path that EXISTS
# and both the lexical and the samefile arm have something to bind to.
assert_equal "lint/trusted-scope entry inside '.github' refused" \
  "$(trusted_scope_hits editablePaths ".github/scripts/overlay-editable-paths.sh")" "1"
assert_equal "lint/trusted-scope entry inside 'tools' refused" \
  "$(trusted_scope_hits editablePaths "tools/lint-benchmark-manifest.py")" "1"

# Case folding and empty segments reach them too -- the divergent entries are
# guarded by the same relation as the inherited five, not by a weaker one.
for divergent_spelling in ".GITHUB" "TOOLS" "tools//lint-benchmark-manifest.py" \
                          ".github//workflows" "Tools"; do
  assert_equal "lint/trusted-scope divergent spelling '${divergent_spelling}' refused" \
    "$(trusted_scope_hits editablePaths "${divergent_spelling}")" "1"
done

# And the separator stays load-bearing for them: a sibling whose name merely
# STARTS with a divergent entry is not an overlap. (Both trip check 2 for not
# existing; only 3c hits are counted.)
for divergent_miss in "toolsmith" ".githubusercontent" "tools-extra"; do
  assert_equal "lint/trusted-scope divergent near-miss '${divergent_miss}' not refused" \
    "$(trusted_scope_hits editablePaths "${divergent_miss}")" "0"
done

# The roster is NINE. Asserted directly, so shrinking it back to upstream's
# five -- "reconciling the divergence" -- fails here rather than silently
# unguarding the gates. Counted from the linter's own roster, not restated.
#
# It read SEVEN until 2026-08-25, stale since `fixtures` and `benchmark.json`
# joined the roster on 2026-08-24 (folding a reviewer blocker into the
# gemma4-26b-a4b-mlx-v1 manifest PR). Note which way that staleness pointed:
# this assertion goes red when the roster GROWS as well as when it shrinks, so
# it cannot silently bless a widened trust boundary either -- re-count it
# deliberately, entry by entry, rather than matching it to whatever the roster
# currently holds. The nine, and why each is a path a submission must never be
# able to declare editable:
#
#   Package.swift, Package.resolved      the frozen dependency graph -- an
#                                        editable entry here reaims the build
#   Sources/MLXFastTrustedHarness        the byte-budget/gate enforcer
#   Sources/MLXFastCLI                   the trusted driver
#   Sources/MLXFastCore                  the pins and constants it drives from
#     (those five are upstream's own TRUSTED_SCOPE roster)
#   .github                              the overlay, the static review, the
#                                        surface gate, the CI tripwires
#   tools                                this linter, the hostile-archive
#                                        suite, the fetch/verify scripts
#     (those two are David's 2026-08-20 divergence: "parity doctrine governs
#      measurement surfaces, not security posture" -- must not be reconciled
#      back to five)
#   fixtures                             the track contract and the pinned
#                                        reference/head .sha256 manifests
#   benchmark.json                       the contract itself, including the
#                                        editablePaths list under discussion
#     (those two added 2026-08-24; each is a file whose editability would let a
#      submission rewrite the terms it is judged by)
assert_equal "lint/trusted-scope roster carries both divergent entries" \
  "$(trusted_scope_roster_size)" "9"
for required_entry in ".github" "tools"; do
  assert_equal "lint/trusted-scope roster still contains '${required_entry}'" \
    "$(trusted_scope_roster | grep -cFx "${required_entry}")" "1"
done

echo
echo "=== F. requant-only heads (David ruling 2026-08-26) ========================"
#
# THE RULING. The two speculative-decode heads are the organizer's PINNED
# weights. A participant may declare a re-quantization of them. A participant
# may NOT upload head weights of their own, and may not alter the organizer's.
#
# THE MECHANISM. `mtp-head/` and `dflash-head/` are no longer `editablePaths`
# entries; `mtp-head.manifest.json` and `spec-decoder-head.manifest.json` still are.
# So the rule is not a promise in prose, it is the allowlist -- and the layer
# that reads the allowlist against a submission's DIFF is
# .github/scripts/enforce-modifiable-surface.sh.
#
# WHY THESE CASES RUN AGAINST THE REAL benchmark.json AND NOT THE SYNTHETIC
# CONTRACT. Sections A-D prove the enforcers' MECHANICS on a miniature track,
# which is right for mechanics and wrong for this: the property under test here
# is a fact about THIS repository's declared surface. A synthetic contract would
# stay green if someone put `mtp-head` back in the real one, which is the exact
# regression these cases exist to catch.

real_contract_fixture() { # -> echoes a git repo carrying the REAL benchmark.json
  local dir
  dir="$(mktemp -d "${WORK}/rq.XXXXXX")"
  mkdir -p "${dir}/mtp-head" "${dir}/dflash-head" "${dir}/Sources/MLXFastModel"
  cp "${REPO_ROOT}/benchmark.json" "${dir}/benchmark.json"
  cp "${REPO_ROOT}/mtp-head.manifest.json" "${dir}/mtp-head.manifest.json"
  cp "${REPO_ROOT}/spec-decoder-head.manifest.json" "${dir}/spec-decoder-head.manifest.json"
  # Stand-ins for the organizer-staged head bytes. The real ones are gitignored
  # and fetched on box; what matters to this gate is only that a path under the
  # head directory shows up in the diff.
  cp "${REPO_ROOT}/mtp-head/README.md" "${dir}/mtp-head/README.md"
  cp "${REPO_ROOT}/dflash-head/README.md" "${dir}/dflash-head/README.md"
  printf 'let x = 1\n' > "${dir}/Sources/MLXFastModel/Placeholder.swift"
  {
    git -C "${dir}" init -q -b main
    git -C "${dir}" add -A
    git -C "${dir}" -c user.name=t -c user.email=t@e commit -q -m base
  } >&2
  echo "${dir}"
}

commit_all() { # commit_all DIR MESSAGE -> echoes the new HEAD sha
  {
    git -C "$1" add -A
    git -C "$1" -c user.name=t -c user.email=t@e commit -q -m "$2"
  } >&2
  git -C "$1" rev-parse HEAD
}

surface_gate() { # surface_gate DIR BASE HEAD
  env BASE_SHA="$2" HEAD_SHA="$3" CONTRACT_PATH=benchmark.json \
    "${SCRIPTS}/enforce-modifiable-surface.sh"
}

# --- F1. NEGATIVE CONTROL (a): a head weight file smuggled into a submission --
#
# The shape the ruling exists to stop. A participant adds their own drafter
# weights under the head directory and submits. Both head directories, because
# the DFlash arm got its own directory on 2026-08-25 and a rule that binds one
# and not the other is not a rule.
for smuggled in "mtp-head/model.safetensors" "dflash-head/model-00001-of-00002.safetensors" \
                "mtp-head/config.json" "dflash-head/nested/dir/shard.safetensors"; do
  rq="$(real_contract_fixture)"
  rq_base="$(git -C "${rq}" rev-parse HEAD)"
  mkdir -p "$(dirname "${rq}/${smuggled}")"
  printf 'attacker head weights\n' > "${rq}/${smuggled}"
  rq_head="$(commit_all "${rq}" smuggle-head)"
  assert_exit "requant-only/smuggled head weight '${smuggled}' refused by the surface gate" 1 \
    "outside the modifiable surface" "${rq}" \
    surface_gate "${rq}" "${rq_base}" "${rq_head}"
done

# --- F2. NEGATIVE CONTROL (b): the ORGANIZER's head bytes, modified -----------
#
# A different attack with the same remedy. Instead of adding a head, the
# submission EDITS one that is already there -- the shape a participant would
# reach for to "just re-quantize in place" and ship the result, and the shape a
# tamper would take. It is refused for the same reason: the path is not
# editable, so ANY diff touching it is outside the surface. Deletion counts too;
# a gate that admitted deletions would let a submission remove the organizer's
# head and let the loader fall through to something else.
rq="$(real_contract_fixture)"
rq_base="$(git -C "${rq}" rev-parse HEAD)"
printf 'tampered organizer bytes\n' >> "${rq}/mtp-head/README.md"
rq_head="$(commit_all "${rq}" tamper-head)"
assert_exit "requant-only/modified organizer head bytes refused by the surface gate" 1 \
  "outside the modifiable surface" "${rq}" \
  surface_gate "${rq}" "${rq_base}" "${rq_head}"

rq="$(real_contract_fixture)"
rq_base="$(git -C "${rq}" rev-parse HEAD)"
rm "${rq}/dflash-head/README.md"
rq_head="$(commit_all "${rq}" delete-head-content)"
assert_exit "requant-only/deleted organizer head content refused by the surface gate" 1 \
  "outside the modifiable surface" "${rq}" \
  surface_gate "${rq}" "${rq_base}" "${rq_head}"

# --- F3. NEGATIVE CONTROL (c): re-granting the path to yourself ---------------
#
# The obvious bypass, and the one worth pinning: the submission puts `mtp-head`
# back into its OWN copy of benchmark.json in the same commit that adds the
# weights. It does not work, because the gate reads editablePaths from the BASE
# commit. Note the diagnostic names benchmark.json itself first -- the contract
# is trusted scope, so editing it is already outside the surface -- and the
# weight file is named too.
rq="$(real_contract_fixture)"
rq_base="$(git -C "${rq}" rev-parse HEAD)"
python3 - "${rq}/benchmark.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["editablePaths"] = c["editablePaths"] + ["mtp-head", "dflash-head"]
c["optionalEditablePaths"] = c["optionalEditablePaths"] + ["mtp-head", "dflash-head"]
json.dump(c, open(p, "w"), indent=2)
PY
printf 'attacker head weights\n' > "${rq}/mtp-head/model.safetensors"
rq_head="$(commit_all "${rq}" self-widen-heads)"
rq_out="$( (cd "${rq}" && surface_gate "${rq}" "${rq_base}" "${rq_head}") 2>&1 )"
assert_equal "requant-only/self-widened contract does not re-grant the head directory" \
  "$(printf '%s\n' "${rq_out}" | grep -c 'mtp-head/model.safetensors is outside the modifiable surface')" "1"
assert_equal "requant-only/self-widening edit to benchmark.json is itself refused" \
  "$(printf '%s\n' "${rq_out}" | grep -c 'benchmark.json is outside the modifiable surface')" "1"

# --- F4. POSITIVE DISCRIMINATOR: a declaration-only requant submission passes -
#
# Without this the section proves only that the gate refuses things, not that it
# still admits the flow the ruling LEAVES open. A participant declaring a
# re-quantization edits the manifest and (optionally) their own runtime code.
# Nothing under the head directories moves, and the gate is silent.
rq="$(real_contract_fixture)"
rq_base="$(git -C "${rq}" rev-parse HEAD)"
python3 - "${rq}/mtp-head.manifest.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["bytes"] = 123456789
json.dump(c, open(p, "w"), indent=2)
PY
printf 'let x = 2\n' > "${rq}/Sources/MLXFastModel/Placeholder.swift"
rq_head="$(commit_all "${rq}" declare-requant)"
assert_exit "requant-only/declaration-only requant submission accepted" 0 "" "${rq}" \
  surface_gate "${rq}" "${rq_base}" "${rq_head}"

rq="$(real_contract_fixture)"
rq_base="$(git -C "${rq}" rev-parse HEAD)"
python3 - "${rq}/spec-decoder-head.manifest.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["bytes"] = 987654321
json.dump(c, open(p, "w"), indent=2)
PY
rq_head="$(commit_all "${rq}" declare-dflash-requant)"
assert_exit "requant-only/DFlash declaration-only requant submission accepted" 0 "" "${rq}" \
  surface_gate "${rq}" "${rq_base}" "${rq_head}"

# --- F5. DEFENCE IN DEPTH: the overlay never carries head bytes across --------
#
# The surface gate is the layer that REFUSES. The overlay is the layer that
# makes a refusal-bypass pointless: it copies editablePaths and only
# editablePaths from the submission worktree, so head bytes that reached a
# worktree some other way are simply never placed into the tree that gets built
# and measured. Proven positively (the trusted head content survives untouched)
# and negatively (the submission's file does not appear), because "the file is
# absent" alone would also be true of an overlay that deleted everything.
#
# Uses a synthetic contract shaped like the post-ruling real one -- head
# directory NOT editable, declaration file editable and optional -- because the
# overlay walks EVERY editablePaths entry and demands each non-optional one
# exist in the worktree, which no miniature fixture can satisfy for the real
# 92-entry surface.
REQUANT_CONTRACT_JSON='{
  "editablePaths": ["src", "config.txt", "head.manifest.json"],
  "optionalEditablePaths": ["head.manifest.json"],
  "editableSurfaceByteBudget": {
    "exemptPathMaxBytes": 8192,
    "exemptPathMaxFileBytes": 7000,
    "maxTotalBytes": 20000,
    "maxFileBytes": 4096,
    "maxGrowthBytes": 2048
  },
  "staticReviewTrackId": "test-track-v1"
}'
fx="$(new_fixture)"
printf '%s\n' "${REQUANT_CONTRACT_JSON}" > "${fx}/trusted/benchmark.json"
mkdir -p "${fx}/sub/src" "${fx}/sub/head"
printf 'submitted kernel\n' > "${fx}/sub/src/kernel.txt"
printf 'trusted helper\n' > "${fx}/sub/src/helper.txt"
printf 'submitted config\n' > "${fx}/sub/config.txt"
printf '{"source":"pinned","bytes":1}\n' > "${fx}/sub/head.manifest.json"
printf 'ATTACKER HEAD WEIGHTS\n' > "${fx}/sub/head/weights.bin"
printf 'ATTACKER HEAD SHARD\n' > "${fx}/sub/head/shard.safetensors"
assert_exit "requant-only/overlay succeeds with the head directory non-editable" 0 \
  "trusted harness retained" "${fx}/trusted" \
  env CONTRACT_PATH=benchmark.json SUBMISSION_WORKTREE="${fx}/sub" \
  "${SCRIPTS}/overlay-editable-paths.sh"
assert_equal "requant-only/overlay does not import submitted head weights" \
  "$(cat "${fx}/trusted/head/weights.bin")" "trusted head weights"
assert_equal "requant-only/overlay does not create submitted head shards" \
  "$([ -e "${fx}/trusted/head/shard.safetensors" ] && echo present || echo absent)" "absent"
assert_equal "requant-only/overlay still carries the head DECLARATION across" \
  "$(cat "${fx}/trusted/head.manifest.json")" '{"source":"pinned","bytes":1}'

# --- F6. THE MANIFEST ITSELF: no head directory anywhere in the surface -------
#
# Read straight off the real manifest, in all three buckets an overlay writes
# from. The Swift twin is `headWeightDirectoriesAreNotEditable` in
# Tests/MLXFastTests/Gemma4BenchmarkManifestTests.swift; this one binds in CI's
# shell job as well, and it is the assertion that goes red the moment someone
# re-adds a head directory for convenience.
head_dir_entries="$(python3 - "${REPO_ROOT}/benchmark.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
buckets = [
    c.get("editablePaths", []),
    c.get("optionalEditablePaths", []),
    c.get("editableSurfaceByteBudget", {}).get("exemptPaths", []),
]
hits = [
    e
    for b in buckets
    for e in b
    for d in ("mtp-head", "dflash-head")
    if e == d or e.startswith(d + "/")
]
print(len(hits))
PY
)"
assert_equal "requant-only/no head weights directory in any editable bucket" \
  "${head_dir_entries}" "0"

declaration_entries="$(python3 - "${REPO_ROOT}/benchmark.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
editable = set(c.get("editablePaths", []))
optional = set(c.get("optionalEditablePaths", []))
want = {"mtp-head.manifest.json", "spec-decoder-head.manifest.json"}
print(len(want & editable & optional))
PY
)"
assert_equal "requant-only/both head declarations stay editable and optional" \
  "${declaration_entries}" "2"

# exemptPaths is GONE, not emptied-and-forgotten: nothing rides in a submission
# that needs holding out of the code budget any more. The two exempt CAPS stay
# declared on purpose (see benchmark.json and the manifest test) -- they are what
# pins both enforcers' compiled-in exempt fallbacks to a reviewed number, and
# section E's drift assertions consume them.
assert_equal "requant-only/exemptPaths is absent from the real budget block" \
  "$(python3 -c 'import json,sys; print("exemptPaths" in json.load(open(sys.argv[1]))["editableSurfaceByteBudget"])' "${REPO_ROOT}/benchmark.json")" \
  "False"

# --- G. ranked head staging: both trees, before the gate ---------------------
#
# David's second 2026-08-26 ruling. benchd's write-divergence gate hashes every
# file under BOTH workspaces and refuses anything added, changed or deleted
# outside `editablePaths`. With `mtp-head/` and `dflash-head/` no longer on that
# list (section F), a head staged into one workspace and not the other is a
# divergence -- so the ranked flow has to stage them identically, before the
# gate runs. tools/stage-ranked-heads.sh is what does that, and this section is
# its test.
#
# WHY THE SCRIPT IS EXERCISED AGAINST A FAKE REPOSITORY ROOT. The real script
# calls the two organizer stagers, which download ~1.1 GB from Hugging Face and
# verify it against the checked-in manifests. That is exactly the right
# behaviour on a box and exactly the wrong thing for a unit suite to do. The
# script resolves its own root from `BASH_SOURCE`, so copying it into a
# temporary tree puts STUB stagers next to it and the download never happens.
# What is under test here is the script's OWN logic -- mirror, prune, verify,
# and every refusal -- not the stagers, which have their own pins.

ranked_stage_fixture() { # -> echoes a fake candidate root with stub stagers
  local dir
  dir="$(mktemp -d "${WORK}/rh.XXXXXX")"
  mkdir -p "${dir}/tools" "${dir}/mtp-head" "${dir}/dflash-head"
  cp "${REPO_ROOT}/tools/stage-ranked-heads.sh" "${dir}/tools/stage-ranked-heads.sh"
  chmod +x "${dir}/tools/stage-ranked-heads.sh"
  # The checked-in placeholders each workspace carries. Excluded from the head
  # tree digest by rule, and deliberately NOT mirrored by the script.
  printf 'candidate readme\n' > "${dir}/mtp-head/README.md"
  printf 'candidate readme\n' > "${dir}/dflash-head/README.md"
  # Stub stagers: same contract as the real ones (write pin-verified files into
  # the head directory, exit 0), without the download.
  cat > "${dir}/setup-gemma4-assistant.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'mtp weights\n' > "${root}/mtp-head/model.safetensors"
printf '{"quantization":{"bits":4,"group_size":64}}\n' > "${root}/mtp-head/config.json"
STUB
  cat > "${dir}/setup-gemma4-dflash.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${root}/dflash-head/nested"
printf 'dflash weights\n' > "${root}/dflash-head/model.safetensors"
printf 'dflash shard\n' > "${root}/dflash-head/nested/shard.safetensors"
printf '{"architectures":["DFlashDraftModel"]}\n' > "${root}/dflash-head/config.json"
STUB
  chmod +x "${dir}/setup-gemma4-assistant.sh" "${dir}/setup-gemma4-dflash.sh"
  echo "${dir}"
}

# A baseline workspace as the organizer stages it: the repository content
# (here, just the head READMEs) and no staged head weights at all. That absence
# is precisely the asymmetry the ruling exists to remove.
ranked_baseline_fixture() { # -> echoes a fake baseline workspace
  local dir
  dir="$(mktemp -d "${WORK}/rb.XXXXXX")"
  mkdir -p "${dir}/mtp-head" "${dir}/dflash-head"
  printf 'candidate readme\n' > "${dir}/mtp-head/README.md"
  printf 'candidate readme\n' > "${dir}/dflash-head/README.md"
  # The baseline is the ORGANIZER'S UNMODIFIED tree, so it is also where the
  # TRUSTED ORACLE worker binary lives. The stager names it as
  # MLXFAST_TRUSTED_ORACLE_WORKER_BIN and requires it to be executable.
  mkdir -p "${dir}/.build-worker/release"
  printf '#!/bin/sh\nexit 0\n' > "${dir}/.build-worker/release/mlxfast-runtime-worker"
  chmod +x "${dir}/.build-worker/release/mlxfast-runtime-worker"
  echo "${dir}"
}

head_tree_listing() { # head_tree_listing DIR HEAD -> "<sha>  <relpath>" lines
  (cd "$1/$2" && find . -type f | sed 's|^\./||' | LC_ALL=C sort \
    | while IFS= read -r rel; do
        printf '%s  %s\n' "$(shasum -a 256 "${rel}" | awk '{print $1}')" "${rel}"
      done)
}

rh_cand="$(ranked_stage_fixture)"
rh_base="$(ranked_baseline_fixture)"

assert_exit "ranked-heads/stages both trees and verifies them identical" 0 \
  "both organizer heads are staged identically" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" ./tools/stage-ranked-heads.sh

# THE PROPERTY THE GATE WILL CHECK, checked here directly on the bytes rather
# than trusting the script's own closing line.
assert_equal "ranked-heads/mtp-head is byte-identical in both workspaces" \
  "$(head_tree_listing "${rh_cand}" mtp-head)" \
  "$(head_tree_listing "${rh_base}" mtp-head)"
assert_equal "ranked-heads/dflash-head is byte-identical in both workspaces (nested shard included)" \
  "$(head_tree_listing "${rh_cand}" dflash-head)" \
  "$(head_tree_listing "${rh_base}" dflash-head)"
assert_equal "ranked-heads/the nested shard really did cross" \
  "$(cat "${rh_base}/dflash-head/nested/shard.safetensors")" "dflash shard"

# IDEMPOTENCE. A ranked box is warm on the second dispatch; a re-run must
# re-verify and copy nothing.
assert_exit "ranked-heads/a second run over an already-staged pair is clean" 0 \
  "both organizer heads are staged identically" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" ./tools/stage-ranked-heads.sh

# A STALE BASELINE SHARD IS A DIVERGENCE TOO -- pruned, not left behind. Without
# the prune the baseline would carry a file the candidate does not have, which
# reads to the gate as a DELETE outside the modifiable surface.
printf 'left over from an older stage\n' > "${rh_base}/dflash-head/stale-shard.safetensors"
assert_exit "ranked-heads/a stale baseline shard is pruned, not tolerated" 0 \
  "removed stale dflash-head/stale-shard.safetensors" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" ./tools/stage-ranked-heads.sh
assert_equal "ranked-heads/the stale shard is gone from the baseline" \
  "$([[ -e "${rh_base}/dflash-head/stale-shard.safetensors" ]] && echo present || echo absent)" \
  "absent"

# NEGATIVE CONTROL -- the verification is not a rubber stamp. A README that
# differs between the two trees is a real divergence (it is trusted-side
# content the gate judges like any other file), and the script must REFUSE it
# rather than mirror over it: mirroring would hide a genuine mismatch between
# the two checkouts.
printf 'baseline readme, different\n' > "${rh_base}/mtp-head/README.md"
assert_exit "ranked-heads/a README differing between the trees is refused, not overwritten" 1 \
  "differs between the two workspaces" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" ./tools/stage-ranked-heads.sh
assert_equal "ranked-heads/the refusal left the baseline README alone" \
  "$(cat "${rh_base}/mtp-head/README.md")" "baseline readme, different"
printf 'candidate readme\n' > "${rh_base}/mtp-head/README.md"

# The three workspace refusals. Each is a staging fault on the box, and each
# must be named rather than skipped past.
assert_exit "ranked-heads/an unset baseline workspace is refused" 1 \
  "MLXFAST_GEMMA4_BASELINE_WORKSPACE is unset" "${rh_cand}" \
  env -u MLXFAST_GEMMA4_BASELINE_WORKSPACE ./tools/stage-ranked-heads.sh
assert_exit "ranked-heads/a missing baseline workspace is refused" 1 \
  "does not exist or is not a directory" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}/nope" ./tools/stage-ranked-heads.sh
assert_exit "ranked-heads/a baseline equal to the candidate is refused" 1 \
  "resolves to the candidate workspace" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_cand}" ./tools/stage-ranked-heads.sh

# A read-only baseline is the one box-provisioning fact this script cannot fix,
# so it is named up front instead of failing halfway through a mirror. Skipped
# under a uid that ignores the mode bits (root), where the check cannot bind.
rh_ro="$(ranked_baseline_fixture)"
chmod a-w "${rh_ro}"
if [[ -w "${rh_ro}" ]]; then
  printf 'skip  ranked-heads/read-only baseline (this uid ignores the mode bit)\n'
else
  assert_exit "ranked-heads/a read-only baseline workspace is refused up front" 1 \
    "not writable by this job" "${rh_cand}" \
    env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_ro}" ./tools/stage-ranked-heads.sh
fi
chmod u+w "${rh_ro}"

# --- G2. the ranked workflow runs the stage BEFORE the measurement -----------
#
# The whole ruling is an ORDERING: both trees must carry the heads BEFORE
# benchd compares them. benchd's comparison happens inside the Benchmark step
# (`measure-job`), so the staging step has to sit above it in the job. A step
# that runs after is a step that does nothing.
# The needles are the `run:` lines, NOT bare script names: this file's header
# mentions tools/gemma4-measure-and-score.sh in prose long before the step that
# executes it, and comparing against a comment would compare the wrong number
# and read "out of order" on a workflow that is perfectly ordered.
RANKED_WF="${REPO_ROOT}/.github/workflows/benchmark.yml"
stage_line="$(grep -n '^ *run: \./tools/stage-ranked-heads\.sh$' "${RANKED_WF}" | head -1 | cut -d: -f1)"
bench_line="$(grep -n '^ *run: .*\./tools/gemma4-measure-and-score\.sh$' "${RANKED_WF}" | head -1 | cut -d: -f1)"
setup_line="$(grep -n '^ *run: \./setup\.sh$' "${RANKED_WF}" | head -1 | cut -d: -f1)"
# All three needles must have MATCHED before either comparison means anything:
# an unmatched needle is an empty line number, and an empty line number would
# make both ordering checks report "no" for a workflow that is fine.
assert_equal "ranked-heads/all three ranked run: steps are locatable" \
  "$([[ -n "${setup_line}" && -n "${stage_line}" && -n "${bench_line}" ]] && echo yes || echo no)" \
  "yes"
assert_equal "ranked-heads/the ranked workflow invokes the staging step" \
  "$([[ -n "${stage_line}" ]] && echo yes || echo no)" "yes"
assert_equal "ranked-heads/staging runs AFTER ./setup.sh (it needs the built tools)" \
  "$([[ -n "${setup_line}" && -n "${stage_line}" && "${stage_line}" -gt "${setup_line}" ]] && echo yes || echo no)" \
  "yes"
assert_equal "ranked-heads/staging runs BEFORE the measurement that compares the trees" \
  "$([[ -n "${bench_line}" && -n "${stage_line}" && "${stage_line}" -lt "${bench_line}" ]] && echo yes || echo no)" \
  "yes"

# --- G3. the per-leg MTP head directories benchd requires --------------------
#
# FINDING A. benchd resolves each leg's MTP head from the ENVIRONMENT, not from
# the workspaces: `resolve_head_dirs` reads QMTP_HEAD_DIR (serial leg) and
# QMTP_CANDIDATE_HEAD_DIR (candidate leg) at main.rs@dc7712ca:1400-1403, and
# once `--preflight-only` has returned an unset QMTP_HEAD_DIR is a die-8:
#
#   "QMTP_HEAD_DIR is unset: the pinned native-MTP head is required for a
#    measure run"                             (main.rs@dc7712ca:1617-1626)
#
# Nothing in this repository set it, so every real ranked measure run died
# there. The stager now names both, and this section is that wiring's test.
#
# PER-LEG, NOT SHARED. benchd DEFAULTS QMTP_CANDIDATE_HEAD_DIR to QMTP_HEAD_DIR
# when only the latter is set (measure_job.rs@dc7712ca:3151-3161), pointing BOTH
# legs at ONE directory. The stager therefore names both explicitly, each out of
# its own workspace -- byte-identical by section G, distinct by path.

# `mktemp -d` inherits TMPDIR verbatim, which commonly ends in a slash, so the
# fixture paths can carry a `//` the script's own `cd .. && pwd` normalises
# away. Compare against the normalised forms, not the raw fixture strings.
rh_cand_real="$(cd "${rh_cand}" && pwd)"
rh_base_real="$(cd "${rh_base}" && pwd)"
rh_env="$(mktemp "${WORK}/rh-github-env.XXXXXX")"
assert_exit "ranked-heads/the stager names both head directories" 0 \
  "written to GITHUB_ENV" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" GITHUB_ENV="${rh_env}" \
  ./tools/stage-ranked-heads.sh

# The SERIAL leg loads the BASELINE workspace's head; the CANDIDATE leg loads
# the candidate's. Asserted on the values, not on the fact that two lines were
# written: a stager that wrote both names with the same value would restore
# exactly the shared-directory shape this exists to avoid.
assert_equal "ranked-heads/QMTP_HEAD_DIR points at the BASELINE workspace head (serial leg)" \
  "$(grep '^QMTP_HEAD_DIR=' "${rh_env}" | tail -1)" \
  "QMTP_HEAD_DIR=${rh_base_real}/mtp-head"
assert_equal "ranked-heads/QMTP_CANDIDATE_HEAD_DIR points at the CANDIDATE workspace head" \
  "$(grep '^QMTP_CANDIDATE_HEAD_DIR=' "${rh_env}" | tail -1)" \
  "QMTP_CANDIDATE_HEAD_DIR=${rh_cand_real}/mtp-head"
# THE DFLASH DRAFTER'S OWN PER-LEG PAIR (David ruling 2026-08-26). DFlash is a
# scored mode now, and it is the family where a shared directory does the most
# damage: the engine's drafter loader resolved a bare relative `./dflash-head`
# against the WORKER's current directory, and the benchmarker spawns both legs
# with no `current_dir`, so without these two names both legs load ONE drafter
# -- the candidate's, resident on the scored DENOMINATOR leg, silently.
assert_equal "ranked-heads/QMTP_DFLASH_HEAD_DIR points at the BASELINE workspace drafter (serial leg)" \
  "$(grep '^QMTP_DFLASH_HEAD_DIR=' "${rh_env}" | tail -1)" \
  "QMTP_DFLASH_HEAD_DIR=${rh_base_real}/dflash-head"
assert_equal "ranked-heads/QMTP_CANDIDATE_DFLASH_HEAD_DIR points at the CANDIDATE workspace drafter" \
  "$(grep '^QMTP_CANDIDATE_DFLASH_HEAD_DIR=' "${rh_env}" | tail -1)" \
  "QMTP_CANDIDATE_DFLASH_HEAD_DIR=${rh_cand_real}/dflash-head"

# BOTH families, asserted on the VALUES: four names, and neither family's two
# legs collapsed onto one directory. A stager that wrote all four names with two
# shared values would restore exactly the shape this exists to avoid.
assert_equal "ranked-heads/the two legs get DIFFERENT head directories, not one shared one" \
  "$([[ "$(grep -c '^QMTP_' "${rh_env}")" == "4" ]] \
     && [[ "$(grep '^QMTP_HEAD_DIR=' "${rh_env}" | cut -d= -f2-)" \
        != "$(grep '^QMTP_CANDIDATE_HEAD_DIR=' "${rh_env}" | cut -d= -f2-)" ]] \
     && [[ "$(grep '^QMTP_DFLASH_HEAD_DIR=' "${rh_env}" | cut -d= -f2-)" \
        != "$(grep '^QMTP_CANDIDATE_DFLASH_HEAD_DIR=' "${rh_env}" | cut -d= -f2-)" ]] \
     && echo yes || echo no)" \
  "yes"

# THE TRUSTED ORACLE WORKER (2026-08-26 box evidence). benchd resolves it from
# MLXFAST_TRUSTED_ORACLE_WORKER_BIN alone, fails closed when unset, and has NO
# fallback -- a candidate-built oracle would judge a degraded model against
# itself. The stager names the BASELINE's binary, because the baseline IS the
# organizer's unmodified tree.
assert_equal "ranked-heads/MLXFAST_TRUSTED_ORACLE_WORKER_BIN names the BASELINE worker" \
  "$(grep '^MLXFAST_TRUSTED_ORACLE_WORKER_BIN=' "${rh_env}" | tail -1)" \
  "MLXFAST_TRUSTED_ORACLE_WORKER_BIN=${rh_base_real}/.build-worker/release/mlxfast-runtime-worker"

# NEGATIVE CONTROL: it must never be the CANDIDATE's binary. That substitution
# is the anti-gaming collapse benchd's own resolver refuses to allow, and the
# stager must not reintroduce it from the other side.
assert_equal "ranked-heads/the trusted oracle is NOT the candidate's worker" \
  "$(grep -c "^MLXFAST_TRUSTED_ORACLE_WORKER_BIN=${rh_cand_real}/" "${rh_env}")" "0"

# A value already in the runner environment that disagrees with the staged
# layout is a mis-staged box: it names a head this script did not stage and did
# not verify. Refused by name rather than silently overridden or obeyed.
assert_exit "ranked-heads/a disagreeing preset QMTP_HEAD_DIR is refused" 1 \
  "is not the staged head this script verified" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" QMTP_HEAD_DIR="/tmp/somewhere-else" \
  ./tools/stage-ranked-heads.sh
assert_exit "ranked-heads/a disagreeing preset QMTP_CANDIDATE_HEAD_DIR is refused" 1 \
  "is not the staged head this script verified" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" QMTP_CANDIDATE_HEAD_DIR="/tmp/somewhere-else" \
  ./tools/stage-ranked-heads.sh
# An AGREEING preset is not a fault -- a re-run on a warm box inherits its own
# previous values, and refusing those would make the stager non-idempotent.
assert_exit "ranked-heads/an agreeing preset is accepted (the stager stays idempotent)" 0 \
  "both organizer heads are staged identically" "${rh_cand}" \
  env MLXFAST_GEMMA4_BASELINE_WORKSPACE="${rh_base}" \
  QMTP_HEAD_DIR="${rh_base_real}/mtp-head" \
  QMTP_CANDIDATE_HEAD_DIR="${rh_cand_real}/mtp-head" \
  ./tools/stage-ranked-heads.sh

# ONE SOURCE. The stager is the only place the ranked workflow decides what the
# two head directories are; a second assignment elsewhere in the workflow would
# be a silent override of a verified path.
assert_equal "ranked-heads/the ranked workflow assigns QMTP_* nowhere but the stager" \
  "$(grep -c 'QMTP_HEAD_DIR\|QMTP_CANDIDATE_HEAD_DIR' "${RANKED_WF}")" "0"

# --- H. DFlash requant parity: CODE joins the surface, never WEIGHTS ---------
#
# David 2026-08-26: "dflash needs to be editable". The contract has promised a
# DFlash re-quantization since section 3.4 was written, and until now that
# promise was unreachable: `DFlashDraftModel` is in a non-editable vendored
# directory and its worker-side loader is in non-editable
# `Sources/MLXFastHarness`, so no participant code could decide the drafter's
# geometry. One MODEL FILE joins, mirroring the MTP head exactly -- editable
# model file, trusted loader.
#
# THE REVERSAL THIS SECTION EXISTS TO CATCH. What must NEVER come back is the
# head WEIGHTS directory. `dflash-head/` and `mtp-head/` left `editablePaths`
# in #61 (requant-only ruling) and re-granting either one would reopen the
# custom-head upload surface the whole ruling closed. Section F proves the
# surface gate refuses smuggled head bytes; this proves the manifest still
# declines to invite them.

dflash_model_entry="Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift"

assert_equal "dflash-editable/the DFlash drafter MODEL file is an editablePaths entry" \
  "$(python3 -c 'import json,sys; print(sys.argv[2] in json.load(open(sys.argv[1]))["editablePaths"])' \
      "${REPO_ROOT}/benchmark.json" "${dflash_model_entry}")" \
  "True"

# The entry must name a file that EXISTS and that actually carries the seam. An
# editablePaths entry naming a file with no quantize call would be a promise
# with nothing behind it.
assert_equal "dflash-editable/the entry names a real file" \
  "$([[ -f "${REPO_ROOT}/${dflash_model_entry}" ]] && echo yes || echo no)" "yes"
assert_equal "dflash-editable/that file carries the quantize-on-load seam" \
  "$(grep -c 'quantize(model: drafter)' "${REPO_ROOT}/${dflash_model_entry}")" "1"

# The MTP precedent, restated as an assertion so the two heads cannot drift:
# the MTP model file is editable for exactly the same reason.
assert_equal "dflash-editable/the MTP model file is editable for the same reason (parity)" \
  "$(python3 -c 'import json,sys; print(sys.argv[2] in json.load(open(sys.argv[1]))["editablePaths"])' \
      "${REPO_ROOT}/benchmark.json" \
      "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4MTP.swift")" \
  "True"

# NEITHER head WEIGHTS directory may be editable, in any spelling.
for hd_dir in dflash-head mtp-head; do
  assert_equal "dflash-editable/${hd_dir}/ is still NOT editable (no upload surface reopened)" \
    "$(python3 -c '
import json, sys
paths = json.load(open(sys.argv[1]))["editablePaths"]
want = sys.argv[2]
print(any(p == want or p == want + "/" or p.startswith(want + "/") for p in paths))
' "${REPO_ROOT}/benchmark.json" "${hd_dir}")" \
    "False"
done

# The count moves DELIBERATELY. 92 -> 93 is the one entry above and nothing
# else; a second entry riding along would pass every check above and this one
# is what refuses it.
assert_equal "dflash-editable/editablePaths holds exactly 93 entries" \
  "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["editablePaths"]))' \
      "${REPO_ROOT}/benchmark.json")" \
  "93"

# The count is stated in four places. A half-updated ripple is the defect class
# a reviewer catches, so the three prose sites are held to the manifest.
for count_site in README.md TASK.md docs/participant-contract.md; do
  assert_equal "dflash-editable/${count_site} states the same 93 entries" \
    "$(grep -c 'lists 93 entries' "${REPO_ROOT}/${count_site}")" "1"
done
assert_equal "dflash-editable/the Swift count pin states 93" \
  "$(grep -c 'editablePaths.count == 93' "${REPO_ROOT}/Tests/MLXFastTests/Gemma4BenchmarkManifestTests.swift")" \
  "1"
assert_equal "dflash-editable/no site still says 92 entries" \
  "$(grep -rl 'lists 92 entries\|count == 92' "${REPO_ROOT}/README.md" "${REPO_ROOT}/TASK.md" \
      "${REPO_ROOT}/docs" "${REPO_ROOT}/Tests" 2>/dev/null | wc -l | tr -d '[:space:]')" \
  "0"

# --- I. the target quantization bind is NON-CIRCULAR and COMPILED IN ---------
#
# David ruling 2026-08-26. The target's frozen format used to be enforced only
# against the on-disk `config.json`; the step from that declaration to the live
# quantized modules runs in `Sources/MLXFastModel`, which IS editable, so a
# candidate could re-quantize the target in memory and pass every disk-side
# gate. `validateLoadedTargetQuantization` reads the loaded modules instead.
#
# A GATE A CANDIDATE CAN EDIT IS NOT A GATE. Two halves have to hold, and both
# are structural rather than behavioural, which is why they are asserted here
# and not only in the Swift suite: the EXPECTED constants must live outside the
# editable surface, and so must the walk that reads the ACTUAL modules.
for bind_path in Sources/MLXFastCore Sources/MLXFastHarness; do
  assert_equal "target-bind/${bind_path} is NOT editable (the gate is out of the candidate's reach)" \
    "$(python3 -c '
import json, sys
paths = json.load(open(sys.argv[1]))["editablePaths"]
want = sys.argv[2]
print(any(p == want or p.startswith(want + "/") for p in paths))
' "${REPO_ROOT}/benchmark.json" "${bind_path}")" \
    "False"
done

TARGET_PINS="${REPO_ROOT}/Sources/MLXFastCore/Gemma4TargetQuantizationPins.swift"
TARGET_BIND="${REPO_ROOT}/Sources/MLXFastHarness/Gemma4TargetQuantizationBind.swift"
assert_equal "target-bind/the pinned geometry lives in trusted MLXFastCore" \
  "$([[ -f "${TARGET_PINS}" ]] && echo yes || echo no)" "yes"
assert_equal "target-bind/the walk lives in the non-editable worker target" \
  "$([[ -f "${TARGET_BIND}" ]] && echo yes || echo no)" "yes"

# The EXPECTED side must not be read from the editable config loader. A single
# reference to `Gemma4A4BConfig` from either file would reintroduce exactly the
# circularity this exists to remove.
for bind_file in "${TARGET_PINS}" "${TARGET_BIND}"; do
  assert_equal "target-bind/$(basename "${bind_file}") does not read the editable Gemma4A4BConfig" \
    "$(grep -c 'Gemma4A4BConfig\.' "${bind_file}" || true)" "0"
done

# COMPILED IN, not guarded out. The compiled-out trusted twin this takes its
# numbers from is the exact failure mode being fixed, so a `#if` anywhere in
# either file -- or around the call site -- would repeat it.
for bind_file in "${TARGET_PINS}" "${TARGET_BIND}"; do
  assert_equal "target-bind/$(basename "${bind_file}") carries no conditional compilation" \
    "$(grep -c '^ *#if' "${bind_file}" || true)" "0"
done

# WIRED ON THE SCORED PATH. benchd spawns exactly the `runtime-worker` verb,
# whose one call is `Gemma4Runtime.runWorker`; the bind must be called there.
assert_equal "target-bind/the scored worker calls the bind at startup" \
  "$(grep -c 'try validateLoadedTargetQuantization(' \
      "${REPO_ROOT}/Sources/MLXFastHarness/Gemma4RuntimeWorker.swift")" \
  "1"
# From MLXFastConstants, never from the candidate-loaded config: a candidate
# that reported zero layers would otherwise collapse the expected override set
# to empty and pass.
# FOUR call sites: the startup bind plus the three measured-window re-checks.
# Every one of them takes the layer count from the trusted constants and not
# from the candidate-loaded config -- a candidate reporting zero layers would
# otherwise collapse the expected override set to empty at whichever site read
# it, and one lax site is enough.
assert_equal "target-bind/every call site takes the layer count from the trusted constants" \
  "$(grep -c 'numHiddenLayers: MLXFastConstants.numHiddenLayers' \
      "${REPO_ROOT}/Sources/MLXFastHarness/Gemma4RuntimeWorker.swift")" \
  "4"
# The verified instance must be the one the worker goes on to run.
# `requireLibraryModel()` is editable code, so "it returns the same object every
# time" is a fact to CHECK, not to rely on.
assert_equal "target-bind/the verified target instance is checked for stability" \
  "$(grep -c 'requireLibraryModel() === verifiedTarget' \
      "${REPO_ROOT}/Sources/MLXFastHarness/Gemma4RuntimeWorker.swift")" \
  "1"
# The startup bind alone verifies a model that editable request-path code can
# still mutate in place. Every verb that OPENS a measured window must re-run the
# full check first, so the three are named individually rather than counted --
# a count would pass on three re-checks all guarding the same verb.
for mw_phase in prefill decode_begin free_decode_begin; do
  assert_equal "target-bind/the ${mw_phase} window re-checks the target before measuring" \
    "$(grep -c "phase: \"${mw_phase}\"" \
        "${REPO_ROOT}/Sources/MLXFastHarness/Gemma4RuntimeWorker.swift")" \
    "1"
done
# The re-check must re-run the FULL validation. An in-place `update(modules:)`
# leaves instance identity intact, so an identity-only guard would see nothing.
assert_equal "target-bind/the re-check re-runs the full validation, not just identity" \
  "$(grep -c 'try validateLoadedTargetQuantization(' \
      "${REPO_ROOT}/Sources/MLXFastHarness/Gemma4TargetQuantizationBind.swift")" \
  "1"

# --- J. the docs may not describe a DISK re-quantization -------------------
#
# A re-quantization happens ON LOAD, in memory (David clarification
# 2026-08-26). The mechanism was previously documented as a hook that REWROTE
# the staged head directory, and that framing survived in several places after
# the mechanism changed -- including inside the same file whose section 4.4
# states the on-load truth. A contradiction in the participant contract is worse
# than a gap: a participant who follows the stale half writes into a staged head
# and is refused by the sandbox and by the write-divergence gate.
#
# So the phrasing class is REFUSED here, not merely fixed once. Each needle
# below is a disk-artifact claim: something is produced, written, or uploaded.
# The prose is free to say a re-quantization happens on the benchmark machine --
# it does -- but not that a FILE is made there.
docs_disk_requant_hits() {
  # docs_disk_requant_hits <needle> -> matching "path:line: text" lines
  grep -rn -- "$1" \
    "${REPO_ROOT}/README.md" "${REPO_ROOT}/TASK.md" "${REPO_ROOT}/AGENTS.md" \
    "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/docs" \
    "${REPO_ROOT}/mtp-head/README.md" "${REPO_ROOT}/dflash-head/README.md" \
    2>/dev/null || true
}

for stale_phrase in \
  "re-quantized artifact" \
  "artifact is produced on the benchmark machine" \
  "head is produced on the benchmark machine" \
  "drafter is produced on the benchmark machine" \
  "StagedHeadRequant" \
  "rewrite the staged" \
  "staging directory to rewrite"; do
  assert_equal "docs-on-load/no participant doc claims a disk artifact: '${stale_phrase}'" \
    "$(docs_disk_requant_hits "${stale_phrase}" | wc -l | tr -d '[:space:]')" \
    "0"
done

# NEGATIVE CONTROL for the guard above. A grep that matches nothing because the
# needle is wrong, or because the file list is wrong, would report "clean" over
# a document full of the very phrasing it exists to refuse. This plants the
# phrase in a scratch copy of the contract and requires the SAME helper to see
# it -- so the helper's file list and matching are proven live, not assumed.
DOCS_GUARD_PROBE="${REPO_ROOT}/docs/.submission-security-docs-guard-probe.md"
printf 'The re-quantized artifact is produced on the benchmark machine.\n' \
  > "${DOCS_GUARD_PROBE}"
assert_equal "docs-on-load/the stale-phrasing guard actually matches (negative control)" \
  "$([[ "$(docs_disk_requant_hits 're-quantized artifact' | wc -l | tr -d '[:space:]')" -ge 1 ]] \
     && echo sees-it || echo blind)" \
  "sees-it"
rm -f "${DOCS_GUARD_PROBE}"

# "re-quantized checkpoint" is the third phrase in the class, and it is handled
# by MEANING rather than by presence: section 4.4 uses it correctly, in the
# NEGATION that defines the mechanism ("You do not make a re-quantized
# checkpoint"). Banning the string outright would delete the clearest sentence
# in the contract. So every line carrying it must be a denial.
ck_affirmative="$(docs_disk_requant_hits 're-quantized checkpoint' \
  | grep -v 'do not make\|does not make\|no re-quantized checkpoint' | wc -l | tr -d '[:space:]')"
assert_equal "docs-on-load/'re-quantized checkpoint' appears only as a denial" \
  "${ck_affirmative}" "0"

# And the POSITIVE half: the on-load mechanism is actually stated, per head, so
# the guard above cannot be satisfied by documenting nothing at all.
assert_equal "docs-on-load/section 4.4 states the on-load mechanism" \
  "$(grep -c 'A re-quantization happens ON LOAD, in memory' \
      "${REPO_ROOT}/docs/participant-contract.md")" \
  "1"
for head_readme in mtp-head dflash-head; do
  assert_equal "docs-on-load/${head_readme}/README.md names its own editable loader seam" \
    "$([[ "$(grep -c 'editable path' "${REPO_ROOT}/${head_readme}/README.md")" -ge 1 ]] \
       && echo yes || echo no)" \
    "yes"
  assert_equal "docs-on-load/${head_readme}/README.md says the requant is on load" \
    "$([[ "$(grep -c 'happens ON LOAD, in memory' "${REPO_ROOT}/${head_readme}/README.md")" -ge 1 ]] \
       && echo yes || echo no)" \
    "yes"
done

echo
echo "==========================================================================="
printf 'submission-security: %d passed, %d failed\n' "${PASSED}" "${FAILED}"
if (( FAILED > 0 )); then
  echo
  for failure in "${FAILURES[@]}"; do
    printf -- '--- %s\n' "${failure}"
  done
  exit 1
fi
if (( PASSED + FAILED < EXPECTED_MIN_ASSERTIONS )); then
  printf 'FAIL  suite size: %d assertions ran, expected at least %d -- a section stopped running\n' \
    "$(( PASSED + FAILED ))" "${EXPECTED_MIN_ASSERTIONS}"
  exit 1
fi
