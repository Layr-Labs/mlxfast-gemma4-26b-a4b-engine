#!/usr/bin/env python3
"""HumanEval pass@1 through the Gemma 4 `mlxfast-runtime-worker` wire protocol.

Why this exists
---------------
The challenge harness has no chat surface: the only way to get real
completions out of the resident 26B A4B target is the runtime worker's
JSON-lines protocol. This driver generates ONE greedy completion per HumanEval
problem over that protocol, on either

* ``--route serial``  -- the teacher-forced-shaped verbs ``decode_begin`` /
  ``decode_step``, a client-driven greedy loop that stops client-side on the
  weights tree's own stop-token set; or
* ``--route dflash``  -- ``free_decode_begin {"spec": {"mode": "dflash",
  "dflash": {"depth": D}}}`` followed by ``free_decode_run {"count": N}``,
  the engine's own speculative free-run loop,

writes EvalPlus-format samples, and (with ``--score``) scores them with the
EvalPlus 0.3.1 CLI in the evalplus checkout's own venv (CPU only, model gone).

Prompting is ``scripts/fable/humaneval_screen.py``'s convention verbatim
(EvalPlus 0.3.1's OpenAI-chat system+user pair, ``DEFAULT_MAX_TOKENS = 768``,
temperature 0 == the worker's only sampler, first-n-in-dataset-order
selection, the same samples/scoring/pass@1 definitions). The one thing that
script did not have to do is apply a chat template -- an MTPLX server did it --
so this driver applies the weights tree's own ``chat_template.jinja`` with
``add_generation_prompt=True`` and feeds the resulting ids as ``seed_tokens``.

PROTOCOL FACTS THIS DRIVER IS BUILT AROUND (verified in
``Sources/MLXFastHarness/``, main-control @ 8ae4c54c):

1. ``decode_begin`` opens a FRESH ``model.newCache`` every time
   (``plainSeedForward``), so the verb is repeatable on one resident worker --
   one process serves all 164 problems.
2. ``decode_begin`` returns ``seed_token`` (the first GENERATED token, argmax
   after the prompt); ``decode_step {"token": t}`` feeds ``t`` and returns the
   NEXT prediction. So a completion is ``[seed_token] + [each returned token]``
   and ``max_tokens`` bounds that whole list.
3. ``free_decode_run`` CONSUMES the session opened by ``free_decode_begin``
   (``runFreeDecode`` nils ``state.dflashFreeRunSession``), so one run per
   begin. A begin whose session is never run is safely abandoned by the next
   begin -- the single-stream form has no "session already open" refusal.
4. ``free_decode_run``'s N counts tokens AFTER the seed token, so
   ``count = max_tokens - 1`` yields ``max_tokens`` generated tokens.
5. **Any handler error poisons the session for its whole lifetime.**
   ``Gemma4RuntimeWorker.runWorker``'s catch sets ``state.poisoned = true`` and
   nothing ever clears it; guard 1 of ``validateGenericWorkerRequest`` then
   refuses every later request. There is no "re-open the session after the
   early-EOS failure" -- the process must be RESTARTED. That is what
   ``WorkerHandle.restart`` is for, and why ``--length-hint`` exists.
6. The early-EOS failure is
   ``RuntimeWorkerFreeRunError.stopTokenBeforeTarget``, surfaced as
   ``{"ok": false, "error": "free_decode_run dflash leg committed stop token
   <id> at committed position <p> of N=<n>; ..."}``. ``<p>`` counts the stop
   token, so a re-run with ``count = p`` commits exactly the same prefix and
   completes cleanly (greedy decoding is deterministic and the committed
   stream is the target's argmax chain regardless of block width).

The cost of fact 5: on a 164-problem HumanEval run nearly every dflash problem
stops on EOS well before 768 tokens, so the naive "ask for max_tokens, recover
from the failure" loop restarts a 21.6 GB worker ~164 times. ``--length-hint
<serial receipt.json>`` avoids that: it asks for the length the serial route
already measured, which -- if the two routes agree, which is the whole claim
under test -- lands exactly on the stop token and never fails. A wrong hint
costs at most one extra begin/run pair (too short) or falls back to the
restart path (too long); it can never change the recorded tokens.

Run it under the canonical GPU guard: ``MTPLX_GUARD_ATTEST_FD`` must be set or
the driver refuses to spawn the real worker (``--fake-worker`` is the unit-test
seam and is exempt).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# --------------------------------------------------------------------------
# Pinned paths
# --------------------------------------------------------------------------
#
# Mostly DEFAULTS: `--worker`, `--weights`, `--dflash-head`, `--out-dir` and
# `--gate-module` all override theirs on the command line. `EVALPLUS_ROOT` and
# the two paths derived from it are NOT overridable -- they locate the pinned
# EvalPlus 0.3.1 virtualenv the harness shells out to, and a different one
# would silently change what pass@1 means. The values below are what a run on
# the reference box resolves to when nothing is passed; they are written
# `REPO_ROOT`- and `Path.home()`-relative rather than as one operator's
# absolute home so the file is readable, and reproducible, from a checkout of
# this branch.
#
# REPO_ROOT is this file's own package root (tools/..), so a driver copied
# into a worktree defaults to that worktree. CHECKOUTS_ROOT is the directory
# holding sibling checkouts and worktrees -- for a git worktree under
# `.worktrees/` that is `.worktrees/`, which is where the control and
# depth3-tip trees live.

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKOUTS_ROOT = REPO_ROOT.parent
PROJECTS_ROOT = Path.home() / "projects"
OPENSOURCE_WTF = PROJECTS_ROOT / "OpenSourceWTF"

MAIN_CONTROL = CHECKOUTS_ROOT / "mlxfast-gemma4-main-control"
DEPTH3_TIP = CHECKOUTS_ROOT / "mlxfast-gemma4-mtp-depth3-tip"
DEFAULT_WORKER = MAIN_CONTROL / ".build-worker/release/mlxfast-runtime-worker"
DEFAULT_WEIGHTS = OPENSOURCE_WTF / "mlxfast-gemma4-26b-a4b" / "weights"
DEFAULT_DFLASH_HEAD = DEPTH3_TIP / "dflash-head"
DEFAULT_OUT_DIR = OPENSOURCE_WTF / ".benchmark-artifacts" / "gemma4-humaneval"
DEFAULT_GATE_MODULE = Path("/tmp/gemma_b1_exact_gate.py")

EVALPLUS_ROOT = PROJECTS_ROOT / "evalplus"
EVALPLUS_EVALUATE = EVALPLUS_ROOT / ".venv" / "bin" / "evalplus.evaluate"
EVALPLUS_SITE = EVALPLUS_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
EVALPLUS_VERSION = "0.3.1"
HUMANEVAL_TASK_COUNT = 164

# EvalPlus 0.3.1's OpenAI chat prompt, copied verbatim from
# scripts/fable/humaneval_screen.py so the two engines' pass@1 numbers are
# comparable. (That file's docstring calls this byte-identical to
# evalplus.provider.utility.make_raw_chat_prompt; it is not -- make_raw_chat_prompt
# is the direct-completion HF path that PREFILLS an assistant ```python fence.
# This is the openai-provider convention, which is the one the qwen38 numbers
# were produced with, so it is the one reused here.)
SYSTEM_MESSAGE = "You are a helpful assistant good at coding."
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
DEFAULT_MAX_TOKENS = 768

#: `free_decode_run.count` ceiling
#: (MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens).
FREE_RUN_COUNT_CEILING = 1_536

#: RuntimeWorkerFreeRunError.stopTokenBeforeTarget's rendered description.
STOP_BEFORE_TARGET_RE = re.compile(
    r"committed stop token (\d+) at committed position (\d+) of N=(\d+)"
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def token_digest(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# Stop tokens -- the EXACT set the worker itself halts on
# --------------------------------------------------------------------------


def resolve_stop_tokens(weights: Path) -> set[int]:
    """Mirror of `resolveRuntimeWorkerStopTokens` (RuntimeWorkerGenericDispatch.swift).

    Union of `eos_token_id` and `pad_token_id` -- each accepted as a scalar or
    a list -- over `config.json` and `generation_config.json`. A missing or
    unparseable file contributes nothing, exactly as the Swift does.

    This must be the worker's set and not the tokenizer's: the dflash leg
    halts on THIS set inside the engine, so a serial leg stopping on anything
    else would not be the same experiment. For the pinned Gemma 4 26B A4B tree
    it is {0, 1, 50, 106} == {<pad>, <eos>, <|tool_response>, <turn|>}.
    """

    ids: set[int] = set()
    for name in ("config.json", "generation_config.json"):
        path = weights / name
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(root, dict):
            continue
        for key in ("eos_token_id", "pad_token_id"):
            value = root.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, list):
                ids.update(int(v) for v in value if isinstance(v, int))
    return ids


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


def load_tokenizer(weights: Path) -> Any:
    """The weights tree's own tokenizer, loaded WITHOUT AutoTokenizer.

    Two deliberate choices:

    * ``PreTrainedTokenizerFast`` rather than ``AutoTokenizer``: the latter
      loads ``config.json`` through ``AutoConfig`` first, and this tree's
      ``final_logit_softcapping: 30`` (an int) fails transformers 5.8's strict
      dataclass validation. The tokenizer needs no model config.
    * NO ``fix_mistral_regex``: transformers warns that this ``tokenizer.json``
      carries the "incorrect" pre-tokenizer regex and offers to patch it.
      Patching CHANGES the ids -- and the repo's own pinned reference tapes
      (``correctness_prompts/public_longcopy_gate_english_1024_256.json``'s
      ``prompt_tokens``) reproduce byte-for-byte only with the UNPATCHED
      tokenizer. The shipped tokenizer.json is the ground truth here.
    """

    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(
        str(weights), local_files_only=True
    )


def chat_prompt_text(task_prompt: str) -> str:
    """The user turn's content -- humaneval_screen.build_chat_payload verbatim."""

    return f"{INSTRUCTION_PREFIX}\n```python\n{str(task_prompt).strip()}\n```"


def encode_prompt(tokenizer: Any, task_prompt: str) -> tuple[list[int], str]:
    """Apply the Gemma 4 chat template and tokenize to worker `seed_tokens`.

    Returns ``(ids, rendered_text)``. The template emits ``<bos>`` itself, so
    the ids are taken with ``add_special_tokens=False`` -- a second BOS would
    put the model off-distribution from token zero.
    """

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": chat_prompt_text(task_prompt)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(value) for value in ids], str(text)


def decode_completion(
    tokenizer: Any, tokens: Sequence[int], stop_ids: Iterable[int]
) -> tuple[str, str]:
    """``(text, raw_text)`` for one completion, stop tokens stripped from text."""

    stops = set(int(value) for value in stop_ids)
    body = [int(value) for value in tokens if int(value) not in stops]
    text = tokenizer.decode(body, skip_special_tokens=True)
    raw = tokenizer.decode(
        [int(value) for value in tokens], skip_special_tokens=False
    )
    return str(text), str(raw)


# --------------------------------------------------------------------------
# Worker session
# --------------------------------------------------------------------------


def parse_stop_before_target(error: str) -> dict[str, int] | None:
    """Pull ``(token, position, n)`` out of `stopTokenBeforeTarget`'s description.

    Returns ``None`` for any other error -- callers must NOT treat an
    unrecognised failure as an early EOS.
    """

    match = STOP_BEFORE_TARGET_RE.search(str(error))
    if match is None:
        return None
    return {
        "token": int(match.group(1)),
        "position": int(match.group(2)),
        "n": int(match.group(3)),
    }


class WorkerHandle:
    """One resident `mlxfast-runtime-worker`, restartable.

    Restart is not a convenience: a worker that answered ``ok: false`` even
    once is permanently poisoned (see the module docstring, fact 5), so the
    ONLY way to keep generating after an early-EOS failure is a fresh process.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        worktree: Path,
        run_dir: Path,
        gate: Any,
        required_mode: str,
    ):
        self.command = list(command)
        self.worktree = Path(worktree)
        self.run_dir = Path(run_dir)
        self.gate = gate
        self.required_mode = required_mode
        self.worker: Any = None
        self.spawns = 0
        self.restarts: list[dict[str, Any]] = []
        self.hello: dict[str, Any] | None = None
        self.exit_codes: list[int] = []

    def start(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        index = self.spawns
        self.spawns += 1
        self.worker = self.gate.Worker(
            self.command,
            self.worktree,
            self.run_dir / f"worker.{index}.stderr.log",
            self.run_dir / f"sidecars.{index}",
        )
        hello = self.worker.read()
        if not hello.get("ok"):
            raise RuntimeError(f"worker hello failed: {hello}")
        if hello.get("id") != 0 or not hello.get("nonce"):
            raise RuntimeError(f"invalid worker hello: {hello}")
        self.worker.nonce = hello["nonce"]
        modes = hello.get("spec_modes") or []
        if self.required_mode not in modes:
            raise RuntimeError(
                f"worker does not advertise '{self.required_mode}': spec_modes={modes}"
            )
        self.hello = {
            "spec_modes": hello.get("spec_modes"),
            "capabilities": hello.get("capabilities"),
            "protocol_version": hello.get("protocol_version"),
            "backend": hello.get("backend"),
            "device": hello.get("device"),
            "head_provenance": hello.get("head_provenance"),
        }
        return hello

    def request(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if self.worker is None:
            raise RuntimeError("worker is not running")
        return self.worker.request(dict(body))

    def request_ok(self, label: str, body: Mapping[str, Any]) -> dict[str, Any]:
        payload = self.request(body)
        if not payload.get("ok"):
            raise RuntimeError(f"{label} failed: {payload.get('error', payload)}")
        return payload

    def restart(self, *, reason: str, task_id: str | None = None) -> None:
        code = self.close()
        self.restarts.append(
            {
                "at_utc": utc_now(),
                "reason": reason,
                "task_id": task_id,
                "previous_exit_code": code,
            }
        )
        self.start()

    def close(self) -> int | None:
        if self.worker is None:
            return None
        code = self.worker.close()
        self.exit_codes.append(code)
        self.worker = None
        return code


def worker_command(
    *,
    worker: Path,
    weights: Path,
    route: str,
    dflash_head: Path | None,
    fake_worker: Path | None,
) -> list[str]:
    """The worker argv.

    ``--dflash-head`` rides only on the dflash route: a declared-but-unstageable
    head is the ONE case `loadGemma4DFlashHeadIfStaged` throws rather than
    failing soft, and a serial control leg has no business dying for a drafter
    it never uses.
    """

    if fake_worker is not None:
        command = [
            sys.executable,
            str(fake_worker),
            "runtime-worker",
            "--weights",
            str(weights),
        ]
        if dflash_head is not None:
            command += ["--dflash-head", str(dflash_head)]
        return command + ["--speculative-protocol", "v1.1"]
    command = [
        str(worker),
        "runtime-worker",
        "--weights",
        str(weights),
    ]
    if route == "dflash":
        if dflash_head is None:
            raise ValueError("the dflash route requires --dflash-head")
        command += ["--dflash-head", str(dflash_head)]
    command += ["--speculative-protocol", "v1.1"]
    return command


# --------------------------------------------------------------------------
# Generation -- one problem, one route
# --------------------------------------------------------------------------


def generate_serial(
    handle: WorkerHandle,
    *,
    seed_tokens: Sequence[int],
    max_tokens: int,
    stop_ids: set[int],
) -> dict[str, Any]:
    """`decode_begin` then a client-driven `decode_step` greedy loop.

    The worker never halts this route itself (the plain verbs know nothing
    about stop tokens), so the client stops on the worker's OWN stop-token set
    -- the same set the dflash engine halts on.
    """

    begin_start = time.perf_counter()
    begin = handle.request_ok(
        "decode_begin",
        {
            "kind": "decode_begin",
            "seed_tokens": list(seed_tokens),
            "spec": {"mode": "serial"},
        },
    )
    prefill_s = time.perf_counter() - begin_start

    tokens = [int(begin["seed_token"])]
    decode_start = time.perf_counter()
    stopped_on: int | None = tokens[0] if tokens[0] in stop_ids else None
    while stopped_on is None and len(tokens) < max_tokens:
        step = handle.request_ok(
            "decode_step", {"kind": "decode_step", "token": tokens[-1]}
        )
        token = int(step["token"])
        tokens.append(token)
        if token in stop_ids:
            stopped_on = token
    decode_s = time.perf_counter() - decode_start

    return {
        "tokens": tokens,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "stopped_on": stopped_on,
        "truncated": stopped_on is None,
        "rounds": len(tokens),
        "acceptance_lengths": None,
        "drafted_total": None,
        "accepted_total": None,
        "committed_total": None,
        "effective_spec": begin.get("effective_spec"),
        "attempts": 1,
        "rerun": False,
        "rerun_position": None,
        "respawned": False,
    }


def _dflash_attempt(
    handle: WorkerHandle,
    *,
    seed_tokens: Sequence[int],
    count: int,
    depth: int | None,
) -> dict[str, Any]:
    """One `free_decode_begin` + `free_decode_run(count)` pair.

    Returns a dict with ``ok``. On failure the worker is POISONED and the
    caller must restart before issuing anything else.
    """

    spec: dict[str, Any] = {"mode": "dflash"}
    if depth is not None:
        spec["dflash"] = {"depth": int(depth)}
    begin_start = time.perf_counter()
    begin = handle.request(
        {
            "kind": "free_decode_begin",
            "seed_tokens": list(seed_tokens),
            "spec": spec,
        }
    )
    prefill_s = time.perf_counter() - begin_start
    if not begin.get("ok"):
        return {"ok": False, "stage": "begin", "error": begin.get("error"), "response": begin}
    seed_token = int(begin["seed_token"])
    result: dict[str, Any] = {
        "ok": True,
        "seed_token": seed_token,
        "prefill_s": prefill_s,
        "effective_spec": begin.get("effective_spec"),
        "decode_s": 0.0,
        "tokens": [],
        "run": None,
    }
    if count <= 0:
        return result
    run_start = time.perf_counter()
    run = handle.request({"kind": "free_decode_run", "count": int(count)})
    result["decode_s"] = time.perf_counter() - run_start
    if not run.get("ok"):
        result.update(
            {"ok": False, "stage": "run", "error": run.get("error"), "response": run}
        )
        return result
    result["tokens"] = [int(value) for value in run.get("tokens", [])]
    result["run"] = run
    return result


def generate_dflash(
    handle: WorkerHandle,
    *,
    seed_tokens: Sequence[int],
    max_tokens: int,
    stop_ids: set[int],
    depth: int | None,
    task_id: str,
    hint_tokens: int | None = None,
) -> dict[str, Any]:
    """The dflash free-run route, including the early-EOS recovery.

    `free_decode_run`'s N counts tokens AFTER the seed, so the budget is
    ``max_tokens - 1``. Attempt order:

    1. If a hint is available, ask for exactly the hinted length first. A hint
       that lands on the stop token completes cleanly and costs no restart.
    2. Otherwise (or if the hinted run came back short of a stop token and
       under budget) ask for the full budget.
    3. An early-EOS failure names the committed position ``p``; the worker is
       restarted (it is poisoned) and re-run with ``count = p``, which commits
       the identical prefix and ends ON the stop token.
    """

    budget = max(int(max_tokens) - 1, 0)
    if budget > FREE_RUN_COUNT_CEILING:
        raise ValueError(
            f"free_decode_run count {budget} exceeds the engine ceiling "
            f"{FREE_RUN_COUNT_CEILING}"
        )
    attempts: list[dict[str, Any]] = []
    respawned = False
    rerun_position: int | None = None

    counts: list[int] = []
    if hint_tokens is not None:
        hinted = min(max(int(hint_tokens) - 1, 0), budget)
        if hinted > 0:
            counts.append(hinted)
    if not counts or counts[-1] != budget:
        counts.append(budget)

    attempt_index = 0
    while attempt_index < len(counts):
        count = counts[attempt_index]
        attempt_index += 1
        started = time.perf_counter()
        outcome = _dflash_attempt(
            handle, seed_tokens=seed_tokens, count=count, depth=depth
        )
        attempts.append(
            {
                "count": count,
                "ok": bool(outcome.get("ok")),
                "stage": outcome.get("stage"),
                "error": outcome.get("error"),
                "wall_s": time.perf_counter() - started,
            }
        )
        if outcome.get("ok"):
            tokens = [outcome["seed_token"], *outcome["tokens"]]
            stopped_on = next((t for t in tokens if t in stop_ids), None)
            if stopped_on is None and count < budget:
                # The hint was short: no stop token and budget left. Fall
                # through to the full-budget attempt. Nothing is poisoned, so
                # no restart is needed.
                continue
            run = outcome.get("run") or {}
            return {
                "tokens": tokens,
                "prefill_s": outcome["prefill_s"],
                "decode_s": outcome["decode_s"],
                "stopped_on": stopped_on,
                "truncated": stopped_on is None,
                "rounds": len(run.get("acceptance_lengths", []) or []),
                "acceptance_lengths": run.get("acceptance_lengths"),
                "drafted_total": run.get("drafted_total"),
                "accepted_total": run.get("accepted_total"),
                "committed_total": run.get("committed_total"),
                "physical_verifier_width": run.get("physical_verifier_width"),
                "effective_spec": outcome.get("effective_spec"),
                "attempts": attempts,
                "rerun": rerun_position is not None,
                "rerun_position": rerun_position,
                "respawned": respawned,
                "failed": False,
            }

        # Failure. The session is poisoned either way, so the process goes.
        early = parse_stop_before_target(str(outcome.get("error") or ""))
        handle.restart(
            reason=str(outcome.get("error") or "worker request failed"),
            task_id=task_id,
        )
        respawned = True
        if early is None or early["position"] <= 0 or rerun_position is not None:
            # Not an early EOS, an unusable position, or a re-run that failed
            # again: record a failed generation rather than looping.
            return {
                "tokens": [],
                "prefill_s": None,
                "decode_s": None,
                "stopped_on": None,
                "truncated": False,
                "rounds": 0,
                "acceptance_lengths": None,
                "drafted_total": None,
                "accepted_total": None,
                "committed_total": None,
                "effective_spec": None,
                "attempts": attempts,
                "rerun": rerun_position is not None,
                "rerun_position": rerun_position,
                "respawned": True,
                "failed": True,
                "failure": str(outcome.get("error") or "worker request failed"),
            }
        rerun_position = early["position"]
        counts = counts[:attempt_index] + [rerun_position]

    raise AssertionError("dflash attempt loop exhausted without a verdict")


# --------------------------------------------------------------------------
# Dataset / scoring (humaneval_screen conventions)
# --------------------------------------------------------------------------


def load_evalplus() -> tuple[Callable[[], dict[str, Any]], Callable[..., str]]:
    site = str(EVALPLUS_SITE)
    if site not in sys.path:
        sys.path.append(site)
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize

    return get_human_eval_plus, sanitize


def select_task_ids(task_ids: Sequence[str], n: int) -> list[str]:
    """The first ``n`` problems in dataset order (164 = verdict, 20 = smoke)."""

    ordered = list(task_ids)
    if len(ordered) != HUMANEVAL_TASK_COUNT:
        raise ValueError(
            f"expected {HUMANEVAL_TASK_COUNT} HumanEval problems, got {len(ordered)}"
        )
    if n not in (20, HUMANEVAL_TASK_COUNT):
        raise ValueError(f"--n must be 20 or {HUMANEVAL_TASK_COUNT}, got {n}")
    return ordered[:n]


def write_scoring_file(
    samples_path: Path, scoring_path: Path, all_task_ids: Iterable[str]
) -> dict[str, Any]:
    """Pad to all 164 problems so `evalplus.evaluate`'s coverage assert passes."""

    rows: dict[str, str] = {}
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["task_id"])] = str(row["solution"])
    padded: list[str] = []
    with scoring_path.open("w", encoding="utf-8") as handle:
        for task_id in all_task_ids:
            solution = rows.get(task_id)
            if solution is None:
                solution = ""
                padded.append(task_id)
            handle.write(
                json.dumps({"task_id": task_id, "solution": solution}) + "\n"
            )
    return {
        "path": str(scoring_path),
        "scored": len(rows),
        "padded": len(padded),
        "padded_task_ids": padded,
    }


def score_samples(scoring_path: Path, *, parallel: int, timeout: float) -> Path:
    result_path = Path(str(scoring_path).replace(".jsonl", "_eval_results.json"))
    if result_path.exists():
        result_path.unlink()
    command = [
        str(EVALPLUS_EVALUATE),
        "humaneval",
        "--samples",
        str(scoring_path),
        "--parallel",
        str(int(parallel)),
        "--i-just-wanna-run",
    ]
    print(f"[gemma4-humaneval] scoring: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(EVALPLUS_ROOT),
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evalplus.evaluate exited {completed.returncode} for {scoring_path}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"evalplus.evaluate wrote no results at {result_path}")
    return result_path


def summarize_scores(
    eval_results: Mapping[str, Any], scored_task_ids: Sequence[str]
) -> dict[str, Any]:
    """pass@1 over exactly the scored subset (humaneval_screen's definition).

    HumanEval+ pass@1 is ``base_status == plus_status == pass``, NOT
    ``plus_status == pass`` -- the plus tests are the EXTRA inputs only.
    """

    rows: list[dict[str, Any]] = []
    base_passed = 0
    plus_passed = 0
    for task_id in scored_task_ids:
        try:
            completions = eval_results["eval"][task_id]
        except KeyError as exc:
            raise KeyError(f"no eval result for {task_id}") from exc
        if len(completions) != 1:
            raise ValueError(
                f"{task_id}: expected exactly 1 completion, got {len(completions)}"
            )
        row = completions[0]
        base_ok = str(row.get("base_status")) == "pass"
        plus_ok = base_ok and str(row.get("plus_status")) == "pass"
        base_passed += int(base_ok)
        plus_passed += int(plus_ok)
        rows.append(
            {"task_id": task_id, "base_pass": base_ok, "plus_pass": plus_ok}
        )
    total = len(rows)
    return {
        "tasks": total,
        "humaneval": {
            "passed": base_passed,
            "pass_at_1": (base_passed / total) if total else 0.0,
        },
        "humaneval_plus": {
            "passed": plus_passed,
            "pass_at_1": (plus_passed / total) if total else 0.0,
        },
        "per_problem": rows,
        "base_failures": [r["task_id"] for r in rows if not r["base_pass"]],
        "plus_failures": [r["task_id"] for r in rows if not r["plus_pass"]],
    }


def read_length_hints(path: Path) -> dict[str, int]:
    """``{task_id: tokens_generated}`` from a previous receipt written by this driver."""

    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    hints: dict[str, int] = {}
    for row in receipt.get("rows", []):
        generated = row.get("tokens_generated")
        if isinstance(generated, int) and generated > 0:
            hints[str(row["task_id"])] = generated
    return hints


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HumanEval through the Gemma 4 runtime-worker wire protocol"
    )
    parser.add_argument("--route", choices=("serial", "dflash"), required=True)
    parser.add_argument(
        "--dflash-depth",
        type=int,
        default=None,
        help=(
            "k, tokens PROPOSED per round (block = 1 + k). Clamped worker-side "
            "to the drafter's ceiling; the echoed effective_spec is recorded. "
            "dflash route only."
        ),
    )
    parser.add_argument("--n", type=int, choices=(20, HUMANEVAL_TASK_COUNT),
                        default=HUMANEVAL_TASK_COUNT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dflash-head", type=Path, default=DEFAULT_DFLASH_HEAD)
    parser.add_argument("--worktree", type=Path, default=MAIN_CONTROL)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="worker stderr/sidecars (default: <out-dir>/<label>-run)")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-parallel", type=int, default=8)
    parser.add_argument("--score-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--length-hint",
        type=Path,
        default=None,
        help=(
            "a receipt.json from a previous run; its per-problem token counts "
            "are tried first on the dflash route so an early-EOS failure (and "
            "the mandatory worker restart it forces) is usually avoided."
        ),
    )
    parser.add_argument(
        "--fake-worker",
        type=Path,
        default=None,
        help="TEST SEAM: a python script speaking the worker protocol. Exempt "
             "from the guard-attestation requirement.",
    )
    parser.add_argument("--gate-module", type=Path, default=DEFAULT_GATE_MODULE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.route == "serial" and args.dflash_depth is not None:
        raise SystemExit("--dflash-depth is meaningful only on --route dflash")
    if args.fake_worker is None and not os.environ.get("MTPLX_GUARD_ATTEST_FD"):
        raise RuntimeError(
            "canonical guard attestation is required: MTPLX_GUARD_ATTEST_FD is "
            "unset. Run this through the GPU guard, or pass --fake-worker for "
            "the protocol unit tests."
        )
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")

    gate = load_module("_gemma4_eval_gate", args.gate_module)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else out_dir / f"{args.label}-run"
    samples_path = out_dir / f"{args.label}-samples.jsonl"
    scoring_path = out_dir / f"{args.label}-samples_scored.jsonl"
    raw_path = out_dir / f"{args.label}-samples.raw.jsonl"
    receipt_path = out_dir / f"{args.label}-receipt.json"

    get_humaneval, sanitize = load_evalplus()
    dataset = get_humaneval()
    all_task_ids = list(dataset)
    task_ids = select_task_ids(all_task_ids, args.n)

    stop_ids = resolve_stop_tokens(Path(args.weights))
    if not stop_ids:
        raise RuntimeError(
            f"no stop tokens resolved from {args.weights}; refusing to generate "
            "a stream nothing can halt"
        )
    tokenizer = load_tokenizer(Path(args.weights))
    hints = read_length_hints(args.length_hint) if args.length_hint else {}

    command = worker_command(
        worker=Path(args.worker),
        weights=Path(args.weights),
        route=args.route,
        dflash_head=Path(args.dflash_head) if args.dflash_head else None,
        fake_worker=Path(args.fake_worker) if args.fake_worker else None,
    )
    handle = WorkerHandle(
        command=command,
        worktree=Path(args.worktree),
        run_dir=run_dir,
        gate=gate,
        required_mode=args.route,
    )

    receipt: dict[str, Any] = {
        "schema": "gemma4-humaneval-worker-eval-v1",
        "status": "running",
        "label": args.label,
        "started_at_utc": utc_now(),
        "route": args.route,
        "dflash_depth_requested": args.dflash_depth,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "sampler": {
            "temperature": 0.0,
            "greedy": True,
            "n": 1,
            "prompt": "evalplus-0.3.1-openai-chat + gemma4 chat_template.jinja",
            "instruction_prefix": INSTRUCTION_PREFIX,
            "system_message": SYSTEM_MESSAGE,
            "add_generation_prompt": True,
        },
        "dataset": {
            "name": "humaneval",
            "evalplus_version": EVALPLUS_VERSION,
            "total_problems": HUMANEVAL_TASK_COUNT,
            "selection": "first n in dataset order",
        },
        "worker": {
            "path": str(args.worker),
            "sha256": (
                gate.sha256_file(Path(args.worker))
                if args.fake_worker is None and Path(args.worker).is_file()
                else None
            ),
            "command": command,
            "worktree": str(args.worktree),
            "fake_worker": str(args.fake_worker) if args.fake_worker else None,
        },
        "weights": {
            "path": str(args.weights),
            "config_sha256": gate.sha256_file(Path(args.weights) / "config.json"),
            "generation_config_sha256": gate.sha256_file(
                Path(args.weights) / "generation_config.json"
            ),
            "tokenizer_sha256": gate.sha256_file(
                Path(args.weights) / "tokenizer.json"
            ),
            "chat_template_sha256": gate.sha256_file(
                Path(args.weights) / "chat_template.jinja"
            ),
        },
        "dflash_head": (
            {
                "path": str(args.dflash_head),
                "files": gate.tree_files(Path(args.dflash_head)),
            }
            if args.route == "dflash" and args.fake_worker is None
            else {"path": str(args.dflash_head) if args.dflash_head else None}
        ),
        "stop_tokens": {
            "ids": sorted(stop_ids),
            "source": (
                "union of eos_token_id/pad_token_id over config.json and "
                "generation_config.json -- mirrors resolveRuntimeWorkerStopTokens"
            ),
            "tokens": {
                str(i): tokenizer.convert_ids_to_tokens(i) for i in sorted(stop_ids)
            },
        },
        "guard": {
            "attest_fd": os.environ.get("MTPLX_GUARD_ATTEST_FD"),
            "attest_nonce_present": bool(
                os.environ.get("MTPLX_GUARD_ATTEST_NONCE")
            ),
        },
        "length_hint": {
            "path": str(args.length_hint) if args.length_hint else None,
            "problems": len(hints),
        },
        "rows": [],
    }
    atomic_write_json(receipt_path, receipt)

    samples_path.write_text("", encoding="utf-8")
    raw_path.write_text("", encoding="utf-8")
    run_started = time.time()
    failure: BaseException | None = None
    try:
        handle.start()
        receipt["hello"] = handle.hello
        for index, task_id in enumerate(task_ids, start=1):
            task = dataset[task_id]
            seed_tokens, prompt_text = encode_prompt(tokenizer, str(task["prompt"]))
            if args.route == "serial":
                outcome = generate_serial(
                    handle,
                    seed_tokens=seed_tokens,
                    max_tokens=args.max_tokens,
                    stop_ids=stop_ids,
                )
                outcome.setdefault("failed", False)
            else:
                outcome = generate_dflash(
                    handle,
                    seed_tokens=seed_tokens,
                    max_tokens=args.max_tokens,
                    stop_ids=stop_ids,
                    depth=args.dflash_depth,
                    task_id=task_id,
                    hint_tokens=hints.get(task_id),
                )
            tokens = list(outcome["tokens"])
            text, raw_text = decode_completion(tokenizer, tokens, stop_ids)
            solution = sanitize(text, entrypoint=str(task["entry_point"]))
            with samples_path.open("a", encoding="utf-8") as handle_out:
                handle_out.write(
                    json.dumps({"task_id": task_id, "solution": solution}) + "\n"
                )
            with raw_path.open("a", encoding="utf-8") as handle_raw:
                handle_raw.write(
                    json.dumps({"task_id": task_id, "solution": raw_text}) + "\n"
                )
            row = {
                "task_id": task_id,
                "route": args.route,
                "dflash_depth_requested": args.dflash_depth,
                "effective_spec": outcome.get("effective_spec"),
                "prompt_tokens": len(seed_tokens),
                "prompt_sha256": sha256_text(prompt_text),
                "tokens_generated": len(tokens),
                "token_ids": tokens,
                "token_digest": token_digest(tokens),
                "stopped_on": outcome.get("stopped_on"),
                "truncated_at_max_tokens": bool(outcome.get("truncated")),
                "rounds": outcome.get("rounds"),
                "acceptance_lengths": outcome.get("acceptance_lengths"),
                "drafted_total": outcome.get("drafted_total"),
                "accepted_total": outcome.get("accepted_total"),
                "committed_total": outcome.get("committed_total"),
                "physical_verifier_width": outcome.get("physical_verifier_width"),
                "prefill_wall_s": outcome.get("prefill_s"),
                "decode_wall_s": outcome.get("decode_s"),
                "decode_tps": (
                    (len(tokens) - 1) / outcome["decode_s"]
                    if outcome.get("decode_s") and len(tokens) > 1
                    else None
                ),
                "attempts": outcome.get("attempts"),
                "rerun": bool(outcome.get("rerun")),
                "rerun_position": outcome.get("rerun_position"),
                "worker_respawned": bool(outcome.get("respawned")),
                "generation_failed": bool(outcome.get("failed")),
                "failure": outcome.get("failure"),
                "hint_tokens": hints.get(task_id),
                "solution_sha256": sha256_text(solution),
                "solution_chars": len(solution),
            }
            receipt["rows"].append(row)
            atomic_write_json(receipt_path, receipt)
            print(
                f"[gemma4-humaneval:{args.label}] {index}/{len(task_ids)} {task_id} "
                f"tokens={len(tokens)} stop={outcome.get('stopped_on')} "
                f"decode={row['decode_wall_s']} rerun={row['rerun']}",
                flush=True,
            )
    except BaseException as error:  # noqa: BLE001 -- receipt then re-raise
        failure = error
        receipt["status"] = "failed"
        receipt["failure"] = {
            "at_utc": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(receipt_path, receipt)
        raise
    finally:
        handle.close()
        receipt["worker_lifecycle"] = {
            "spawns": handle.spawns,
            "restarts": handle.restarts,
            "exit_codes": list(handle.exit_codes),
        }
        rows = receipt["rows"]
        receipt["totals"] = {
            "problems": len(rows),
            "tokens_generated": sum(int(r["tokens_generated"]) for r in rows),
            "generation_failures": sum(
                1 for r in rows if r["generation_failed"]
            ),
            "truncated_at_max_tokens": sum(
                1 for r in rows if r["truncated_at_max_tokens"]
            ),
            "reruns": sum(1 for r in rows if r["rerun"]),
            "worker_respawns": len(handle.restarts),
            "decode_wall_s": sum(
                float(r["decode_wall_s"] or 0.0) for r in rows
            ),
            "prefill_wall_s": sum(
                float(r["prefill_wall_s"] or 0.0) for r in rows
            ),
            "wall_s": time.time() - run_started,
        }
        if failure is None:
            receipt["status"] = "generated"
        receipt["artifacts"] = {
            "samples": str(samples_path),
            "samples_raw": str(raw_path),
            "receipt": str(receipt_path),
        }
        atomic_write_json(receipt_path, receipt)

    if args.score:
        scoring_file = write_scoring_file(samples_path, scoring_path, all_task_ids)
        result_path = score_samples(
            Path(scoring_file["path"]),
            parallel=args.score_parallel,
            timeout=args.score_timeout,
        )
        eval_results = json.loads(result_path.read_text(encoding="utf-8"))
        scores = summarize_scores(eval_results, task_ids)
        scores["dataset_hash"] = eval_results.get("hash")
        receipt["scoring_file"] = scoring_file
        receipt["evalplus_result"] = {
            "eval_results_path": str(result_path),
            "scores": {
                "tasks": scores["tasks"],
                "humaneval": scores["humaneval"],
                "humaneval_plus": scores["humaneval_plus"],
                "base_failures": scores["base_failures"],
                "plus_failures": scores["plus_failures"],
            },
            "per_problem": scores["per_problem"],
            "dataset_hash": scores["dataset_hash"],
        }
        receipt["status"] = "scored"
        atomic_write_json(receipt_path, receipt)
        base = scores["humaneval"]
        plus = scores["humaneval_plus"]
        print(
            f"[gemma4-humaneval:{args.label}] HumanEval "
            f"{base['passed']}/{scores['tasks']} = {base['pass_at_1']:.4f}; "
            f"HumanEval+ {plus['passed']}/{scores['tasks']} = "
            f"{plus['pass_at_1']:.4f}",
            flush=True,
        )

    print(f"wrote {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
