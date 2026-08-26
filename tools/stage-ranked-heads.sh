#!/usr/bin/env bash
# Stage the organizer-pinned heads into BOTH ranked workspaces, identically.
#
# David's second 2026-08-26 ruling. The ranked flow must carry the same
# organizer head bytes in the candidate workspace AND in the baseline
# workspace BEFORE the benchmarker compares them, and this script is what puts
# them there.
#
# WHY THIS EXISTS AT ALL. benchd's write-divergence gate hashes EVERY file
# under both workspaces -- gitignored content included, since it excludes only
# `.git` and `.build` -- and refuses any path that was added, changed or
# deleted outside `benchmark.json`'s `editablePaths`. `mtp-head/` and
# `dflash-head/` left that list when custom head uploads were removed (David
# ruling 2026-08-26, PR #61). So a head present in one workspace and absent
# from the other is now a divergence outside the modifiable surface, and the
# gate refuses the run pre-GPU -- not because anything was smuggled, but
# because the two trees were staged differently. Staging them the same way is
# the fix; the alternative (teaching benchd to ignore the head directories)
# would need a benchd change and a new pin, and would blind the gate to a
# directory nobody is watching.
#
# THE PROPERTY THIS SCRIPT GUARANTEES. When it exits 0, the candidate's and the
# baseline's `mtp-head/` and `dflash-head/` trees are byte-identical, file for
# file. It does not assert that they are "probably the same": it computes both
# sides' digests and refuses on any mismatch, because a mismatch here is
# exactly the die-8 the caller is trying to avoid, and failing now names the
# file while failing later names only the path.
#
# WHERE THE BYTES COME FROM. The two organizer stagers, run against the
# CANDIDATE workspace, where they verify every downloaded file against the
# checked-in, trusted-side manifests (fixtures/gemma4_assistant.sha256 and
# fixtures/gemma4_dflash_drafter.sha256 -- neither is an editable path, so a
# submission cannot move the pins its own head is checked against). The
# baseline's copy is then MIRRORED from that verified tree rather than
# downloaded a second time: one download, and byte-identity by construction
# instead of by coincidence.
#
# WHAT IT DOES NOT TOUCH. `README.md` in either head directory. That file is
# checked-in repository content, one per workspace, excluded from the head tree
# digest by rule. Mirroring it would overwrite the baseline's own copy with the
# candidate's. The two READMEs still have to MATCH -- they are trusted-side
# files and the divergence gate judges them like any other -- so this script
# verifies them and refuses on a difference rather than papering over it by
# copying.
#
# IDEMPOTENT. Every step is skip-if-already-correct, so a re-run over a warm
# box copies nothing and re-verifies everything.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="stage-ranked-heads.sh"

fail() {
  echo "${LABEL}: REFUSED: $1" >&2
  exit 1
}

ok() {
  echo "${LABEL}: ok: $1"
}

usage() {
  cat <<EOF
Usage: ./tools/stage-ranked-heads.sh

Stages the organizer-pinned MTP and DFlash heads into both ranked workspaces,
verifies that the two trees are byte-identical, and NAMES the per-leg head
directories benchd requires for BOTH families: QMTP_HEAD_DIR /
QMTP_CANDIDATE_HEAD_DIR for the MTP assistant, and QMTP_DFLASH_HEAD_DIR /
QMTP_CANDIDATE_DFLASH_HEAD_DIR for the DFlash drafter.

Environment:
  MLXFAST_GEMMA4_BASELINE_WORKSPACE   REQUIRED. The pinned serial-control
                                      workspace benchd measures against
                                      (\`measure-job --baseline\`). Staged onto
                                      the box out of band by the organizer.
  GITHUB_ENV                          OPTIONAL. When set, the four QMTP_*
                                      assignments are appended to it, so the
                                      later Benchmark step inherits them. When
                                      unset, they are printed for a hand-driven
                                      run to eval.
EOF
}

if [[ "$#" -gt 0 ]]; then
  case "$1" in
    -h | --help | help)
      usage
      exit 0
      ;;
    *)
      echo "${LABEL}: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

# --- 0. the baseline workspace ----------------------------------------------
# Same three checks tools/ranked-box-preflight.sh makes, repeated here because
# this script is also runnable on its own and must not stage into a path it
# only assumed was a workspace.
BASELINE="${MLXFAST_GEMMA4_BASELINE_WORKSPACE:-}"
[[ -n "${BASELINE}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE is unset; there is no second workspace to stage into"
[[ -d "${BASELINE}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE does not exist or is not a directory: ${BASELINE}"
BASELINE="$(cd "${BASELINE}" && pwd)"
[[ "${BASELINE}" != "${ROOT_DIR}" ]] \
  || fail "MLXFAST_GEMMA4_BASELINE_WORKSPACE resolves to the candidate workspace (${ROOT_DIR}); a paired run needs a distinct pinned baseline"

# The organizer stages the baseline out of band. Whether the job's user may
# WRITE into it is a box-provisioning fact, not something this script can
# arrange, so check it up front and name it: a read-only baseline is a staging
# problem to fix on the box, and discovering it halfway through a mirror would
# leave the two trees in a worse state than it found them.
[[ -w "${BASELINE}" ]] \
  || fail "the baseline workspace is not writable by this job: ${BASELINE} (the ranked heads are staged into BOTH workspaces before the benchmarker compares them; fix the box provisioning rather than skipping the stage)"
ok "baseline workspace resolved and writable: ${BASELINE}"

# --- 1. stage into the candidate --------------------------------------------
# The stagers verify every file they write against the checked-in manifests and
# exit non-zero rather than leave an unverified byte on disk, so "staged" here
# means "pin-verified", and the mirror below inherits that.
#
# BOTH heads, always. The DFlash stager is opt-in for local work precisely so
# an MTP-only setup cannot be broken by it -- but on the RANKED path an
# asymmetric dflash-head/ is a die-8 like any other, so there is no such thing
# as skipping it here. If it cannot be staged, the run could not have gone
# ahead anyway.
echo "${LABEL}: staging the MTP assistant head into the candidate workspace"
"${ROOT_DIR}/setup-gemma4-assistant.sh" >&2 \
  || fail "the MTP assistant head could not be staged into the candidate workspace"
echo "${LABEL}: staging the DFlash drafter into the candidate workspace"
"${ROOT_DIR}/setup-gemma4-dflash.sh" >&2 \
  || fail "the DFlash drafter could not be staged into the candidate workspace"
ok "both heads staged and pin-verified in the candidate workspace"

# --- 2. mirror into the baseline, then verify -------------------------------

# List every regular file under a head directory, as paths relative to it, in
# LC_ALL=C order. `find | sort` rather than a glob so nested shards are seen
# and so the order is the same on both sides.
list_head_files() {
  # list_head_files <dir>
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  (cd "${dir}" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)
}

# Copy every file the candidate staged into the baseline, except README.md.
# Only writes a file whose bytes differ, and removes a baseline file the
# candidate does not have -- a leftover from an older stage is a divergence in
# its own right.
mirror_head() {
  # mirror_head <head-name>
  local head="$1"
  local src="${ROOT_DIR}/${head}"
  local dst="${BASELINE}/${head}"
  local rel src_file dst_file

  [[ -d "${src}" ]] || fail "${head}/ is missing from the candidate workspace after staging"
  mkdir -p "${dst}" || fail "could not create ${dst}"

  # Remove baseline-only files first, so the deletion of a stale shard cannot
  # be mistaken for a successful mirror of a smaller tree.
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    [[ "${rel}" != "README.md" ]] || continue
    if [[ ! -f "${src}/${rel}" ]]; then
      rm -f "${dst}/${rel}" || fail "could not remove the stale ${head}/${rel} from the baseline"
      echo "${LABEL}: removed stale ${head}/${rel} from the baseline"
    fi
  done <<EOF
$(list_head_files "${dst}")
EOF

  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    [[ "${rel}" != "README.md" ]] || continue
    src_file="${src}/${rel}"
    dst_file="${dst}/${rel}"
    if [[ -f "${dst_file}" ]] && cmp -s "${src_file}" "${dst_file}"; then
      continue
    fi
    mkdir -p "$(dirname "${dst_file}")" || fail "could not create the parent of ${head}/${rel} in the baseline"
    cp -f "${src_file}" "${dst_file}" \
      || fail "could not copy ${head}/${rel} into the baseline workspace"
    echo "${LABEL}: mirrored ${head}/${rel}"
  done <<EOF
$(list_head_files "${src}")
EOF
}

# The property the divergence gate will check, asserted here where a failure
# can name the file. Compares the FULL file list -- README.md included, even
# though the mirror leaves it alone -- because the gate does not exempt it
# either.
verify_head_identical() {
  # verify_head_identical <head-name>
  local head="$1"
  local src="${ROOT_DIR}/${head}"
  local dst="${BASELINE}/${head}"
  local src_list dst_list rel src_sha dst_sha

  src_list="$(list_head_files "${src}")"
  dst_list="$(list_head_files "${dst}")"
  if [[ "${src_list}" != "${dst_list}" ]]; then
    fail "${head}/ holds a different file set in the two workspaces; the write-divergence gate would refuse the run pre-GPU$(printf '\ncandidate:\n%s\nbaseline:\n%s' "${src_list}" "${dst_list}")"
  fi

  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    src_sha="$(shasum -a 256 "${src}/${rel}" | awk '{print $1}')"
    dst_sha="$(shasum -a 256 "${dst}/${rel}" | awk '{print $1}')"
    [[ "${src_sha}" == "${dst_sha}" ]] \
      || fail "${head}/${rel} differs between the two workspaces (candidate ${src_sha}, baseline ${dst_sha}); the write-divergence gate would refuse the run pre-GPU"
  done <<EOF
${src_list}
EOF

  ok "${head}/ is byte-identical in both workspaces"
}

for head in mtp-head dflash-head; do
  mirror_head "${head}"
  verify_head_identical "${head}"
done

ok "both organizer heads are staged identically in the candidate and the baseline"

# --- 3. name the per-leg head directories benchd requires --------------------
#
# BOTH HEAD FAMILIES (David ruling 2026-08-26). DFlash became a first-class
# SCORED mode, so the DFlash drafter needs the same per-leg naming the MTP head
# has always had: benchd resolves QMTP_DFLASH_HEAD_DIR / QMTP_CANDIDATE_DFLASH_HEAD_DIR
# through the SAME `resolve_head_dirs`, existence-checks both, and passes each to
# its own leg as `--dflash-head`. Section 2 above already staged and verified
# `dflash-head/` in both workspaces file by file; this section is what tells
# benchd where the two copies are.
#
# It matters more for DFlash than for MTP. The engine's DFlash loader used to
# resolve a BARE RELATIVE `./dflash-head` against the WORKER's current
# directory, and benchd spawns both legs with no `current_dir` -- so without
# these two variables both legs load ONE directory, whichever the benchmarker
# happened to run from, and the candidate's drafter ends up resident on the
# scored DENOMINATOR leg. benchd now REFUSES a dflash candidate with
# QMTP_DFLASH_HEAD_DIR unset (die 8, pre-GPU) rather than falling back to that,
# so an unstaged box fails loudly instead of measuring the wrong thing.
#
# WHY THIS IS HERE AND NOT IN THE WORKFLOW. benchd resolves the two legs'
# heads from the environment, not from the workspaces:
# `measure_job::resolve_head_dirs` (main.rs@dc7712ca:1400-1403) reads
# QMTP_HEAD_DIR for the SERIAL leg and QMTP_CANDIDATE_HEAD_DIR for the
# candidate leg, existence-checks both (die-8, :1406-1416), and then passes
# each to its own leg as `--mtp-head` (:1693-1698, via `leg_spawn_args`).
#
# Once `--preflight-only` has returned, an UNSET QMTP_HEAD_DIR is a hard
# refusal:
#
#   "QMTP_HEAD_DIR is unset: the pinned native-MTP head is required for a
#    measure run (the serial leg loads it; the candidate leg defaults to it)
#    -- die 8"                                (main.rs@dc7712ca:1617-1626)
#
# Nothing in this repository set it, so every real ranked measure run died
# there, after the checkpoint download and the build, before any measurement.
# This is the fix.
#
# PER-LEG, NOT SHARED. benchd DEFAULTS QMTP_CANDIDATE_HEAD_DIR to
# QMTP_HEAD_DIR when only the latter is set (`resolve_head_dirs`,
# measure_job.rs@dc7712ca:3151-3161), which points BOTH legs at ONE directory.
# That is not what a paired measurement means: each leg must load the head out
# of its OWN workspace, and the two are byte-identical because section 2 above
# just proved it file by file. So both variables are named explicitly and
# neither is left to default.
CANDIDATE_HEAD_DIR="${ROOT_DIR}/mtp-head"
BASELINE_HEAD_DIR="${BASELINE}/mtp-head"
[[ -d "${CANDIDATE_HEAD_DIR}" ]] || fail "the candidate MTP head directory is missing after staging: ${CANDIDATE_HEAD_DIR}"
[[ -d "${BASELINE_HEAD_DIR}" ]] || fail "the baseline MTP head directory is missing after staging: ${BASELINE_HEAD_DIR}"
CANDIDATE_DFLASH_HEAD_DIR="${ROOT_DIR}/dflash-head"
BASELINE_DFLASH_HEAD_DIR="${BASELINE}/dflash-head"
[[ -d "${CANDIDATE_DFLASH_HEAD_DIR}" ]] || fail "the candidate DFlash drafter directory is missing after staging: ${CANDIDATE_DFLASH_HEAD_DIR}"
[[ -d "${BASELINE_DFLASH_HEAD_DIR}" ]] || fail "the baseline DFlash drafter directory is missing after staging: ${BASELINE_DFLASH_HEAD_DIR}"

# A value already in the environment that DISAGREES with the staged layout is a
# mis-staged box, not a preference to honour: it would point a leg at a head
# this script did not stage and did not verify. Naming it here is the whole
# point -- silently overriding it would hide the box-provisioning fault, and
# silently obeying it would measure an unverified head.
check_preset() {
  # check_preset <var-name> <computed-value>
  local name="$1" want="$2" have
  eval "have=\${${name}:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  [[ -z "${have}" || "${have}" == "${want}" ]] \
    || fail "${name} is already set to '${have}' in this environment, which is not the staged head this script verified (${want}); fix the box provisioning rather than measuring an unverified head"
}
check_preset QMTP_HEAD_DIR "${BASELINE_HEAD_DIR}"
check_preset QMTP_CANDIDATE_HEAD_DIR "${CANDIDATE_HEAD_DIR}"
check_preset QMTP_DFLASH_HEAD_DIR "${BASELINE_DFLASH_HEAD_DIR}"
check_preset QMTP_CANDIDATE_DFLASH_HEAD_DIR "${CANDIDATE_DFLASH_HEAD_DIR}"

# Under Actions, GITHUB_ENV is how a step hands an environment variable to the
# steps after it; the Benchmark step is where benchd reads them. Outside
# Actions the same two lines go to stdout for a hand-driven run to eval, so
# there is exactly one place that decides what the two values are.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    printf 'QMTP_HEAD_DIR=%s\n' "${BASELINE_HEAD_DIR}"
    printf 'QMTP_CANDIDATE_HEAD_DIR=%s\n' "${CANDIDATE_HEAD_DIR}"
    printf 'QMTP_DFLASH_HEAD_DIR=%s\n' "${BASELINE_DFLASH_HEAD_DIR}"
    printf 'QMTP_CANDIDATE_DFLASH_HEAD_DIR=%s\n' "${CANDIDATE_DFLASH_HEAD_DIR}"
  } >> "${GITHUB_ENV}" \
    || fail "could not append the QMTP head directories to GITHUB_ENV (${GITHUB_ENV})"
  ok "the four QMTP head directories (MTP + DFlash, per leg) written to GITHUB_ENV for the measurement step"
else
  ok "GITHUB_ENV is unset; printing the four assignments for a hand-driven run"
fi
# --- 4. name the TRUSTED ORACLE worker binary --------------------------------
#
# FROM THE BOX (2026-08-26 evidence run). benchd's cohort-replay integrity check
# spawns a TRUSTED oracle to produce the reference argmax, and it resolves that
# binary from MLXFAST_TRUSTED_ORACLE_WORKER_BIN alone --
# `resolve_trusted_oracle_worker_bin`, which FAILS CLOSED when the variable is
# unset and has NO fallback, deliberately:
#
#   "a candidate-built oracle could poison the reference argmax (anti-gaming
#    collapse)"
#
# The variable was named NOWHERE in this repository, so the documented ranked
# path could not reach the oracle at all.
#
# WHY THE BASELINE'S BINARY IS THE RIGHT VALUE, and why this script computes it
# rather than a human exporting it: benchd requires "a build of the organizer's
# UNMODIFIED engine tree", and the BASELINE workspace is exactly that -- it is
# the pinned serial control the organizer stages out of band, the same tree this
# script has already resolved and required to be distinct from the candidate.
# Deriving it here means the trusted-oracle path and the baseline leg cannot
# disagree about which tree is the organizer's.
#
# The candidate's own worker is never a candidate for this value. If it were,
# the oracle would judge a degraded model against itself.
ORACLE_WORKER_BIN="${BASELINE}/.build-worker/release/mlxfast-runtime-worker"
[[ -x "${ORACLE_WORKER_BIN}" ]] \
  || fail "the baseline workspace has no built runtime worker at ${ORACLE_WORKER_BIN}; benchd resolves MLXFAST_TRUSTED_ORACLE_WORKER_BIN from the organizer's UNMODIFIED tree and fails closed without it -- build the baseline workspace (setup.sh) before measuring"
check_preset MLXFAST_TRUSTED_ORACLE_WORKER_BIN "${ORACLE_WORKER_BIN}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'MLXFAST_TRUSTED_ORACLE_WORKER_BIN=%s\n' "${ORACLE_WORKER_BIN}" >> "${GITHUB_ENV}" \
    || fail "could not append MLXFAST_TRUSTED_ORACLE_WORKER_BIN to GITHUB_ENV (${GITHUB_ENV})"
  ok "MLXFAST_TRUSTED_ORACLE_WORKER_BIN written to GITHUB_ENV for the measurement step"
fi
printf 'export MLXFAST_TRUSTED_ORACLE_WORKER_BIN=%q\n' "${ORACLE_WORKER_BIN}"

printf 'export QMTP_HEAD_DIR=%q\n' "${BASELINE_HEAD_DIR}"
printf 'export QMTP_CANDIDATE_HEAD_DIR=%q\n' "${CANDIDATE_HEAD_DIR}"
printf 'export QMTP_DFLASH_HEAD_DIR=%q\n' "${BASELINE_DFLASH_HEAD_DIR}"
printf 'export QMTP_CANDIDATE_DFLASH_HEAD_DIR=%q\n' "${CANDIDATE_DFLASH_HEAD_DIR}"
