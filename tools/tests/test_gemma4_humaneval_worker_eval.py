#!/usr/bin/env python3
"""Unit tests for tools/gemma4_humaneval_worker_eval.py.

CPU only. No GPU, no Metal, no real `mlxfast-runtime-worker`, no flock. The
wire is driven against `fake_runtime_worker.py`, which reproduces the worker's
protocol semantics (including the permanent session poisoning that makes a
worker RESTART the only recovery from an early-EOS `free_decode_run`).

Run:

    nice -n 19 python3 -m pytest -p no:cacheprovider \\
        tools/tests/test_gemma4_humaneval_worker_eval.py -q

(on the reference box, with the interpreter that has pytest:
`~/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python`)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
DRIVER_PATH = TOOLS / "gemma4_humaneval_worker_eval.py"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_runtime_worker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = _load("gemma4_humaneval_worker_eval", DRIVER_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    return driver.load_tokenizer(driver.DEFAULT_WEIGHTS)


@pytest.fixture(scope="module")
def code_tokens(tokenizer):
    """Real ids for a short, sanitizer-friendly completion (no stop token)."""

    ids = tokenizer(
        "```python\ndef solve():\n    return 1\n```",
        add_special_tokens=False,
    )["input_ids"]
    return [int(value) for value in ids]


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_stop_tokens_match_the_workers_own_rule():
    """resolve_stop_tokens mirrors resolveRuntimeWorkerStopTokens exactly."""

    assert driver.resolve_stop_tokens(driver.DEFAULT_WEIGHTS) == {0, 1, 50, 106}


def test_stop_tokens_accept_scalar_and_list_and_tolerate_junk(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"eos_token_id": 7, "pad_token_id": 3})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 11, 12], "pad_token_id": None})
    )
    assert driver.resolve_stop_tokens(tmp_path) == {3, 7, 11, 12}
    assert driver.resolve_stop_tokens(tmp_path / "nope") == set()


def test_parse_stop_before_target_reads_the_swift_description():
    message = (
        "free_decode_run dflash leg committed stop token 106 at committed "
        "position 42 of N=767; the leg is invalid (a paired leg must reach N "
        "or both legs must stop)"
    )
    assert driver.parse_stop_before_target(message) == {
        "token": 106,
        "position": 42,
        "n": 767,
    }
    assert driver.parse_stop_before_target(
        'invalidInput("the generic decode session is poisoned after an '
        'earlier failure")'
    ) is None


def test_select_task_ids():
    ids = [f"HumanEval/{i}" for i in range(164)]
    assert driver.select_task_ids(ids, 20) == ids[:20]
    assert driver.select_task_ids(ids, 164) == ids
    with pytest.raises(ValueError):
        driver.select_task_ids(ids, 30)
    with pytest.raises(ValueError):
        driver.select_task_ids(ids[:10], 20)


def test_summarize_scores_requires_base_and_plus():
    results = {
        "eval": {
            "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
            "HumanEval/1": [{"base_status": "pass", "plus_status": "fail"}],
            "HumanEval/2": [{"base_status": "fail", "plus_status": "pass"}],
        }
    }
    scores = driver.summarize_scores(results, list(results["eval"]))
    assert scores["humaneval"]["passed"] == 2
    assert scores["humaneval_plus"]["passed"] == 1
    assert scores["base_failures"] == ["HumanEval/2"]


def test_worker_command_stages_dflash_head_only_on_the_dflash_route():
    serial = driver.worker_command(
        worker=Path("/w"),
        weights=Path("/W"),
        route="serial",
        dflash_head=Path("/D"),
        fake_worker=None,
    )
    assert "--dflash-head" not in serial
    assert serial[:2] == ["/w", "runtime-worker"]
    assert serial[-2:] == ["--speculative-protocol", "v1.1"]
    dflash = driver.worker_command(
        worker=Path("/w"),
        weights=Path("/W"),
        route="dflash",
        dflash_head=Path("/D"),
        fake_worker=None,
    )
    assert dflash[dflash.index("--dflash-head") + 1] == "/D"
    with pytest.raises(ValueError):
        driver.worker_command(
            worker=Path("/w"),
            weights=Path("/W"),
            route="dflash",
            dflash_head=None,
            fake_worker=None,
        )


def test_encode_prompt_applies_the_gemma4_chat_template(tokenizer):
    ids, text = driver.encode_prompt(tokenizer, "def f():\n    pass\n")
    assert text.startswith("<bos><|turn>system\n")
    assert driver.SYSTEM_MESSAGE in text
    assert driver.INSTRUCTION_PREFIX in text
    assert text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    # The template emits <bos> itself; add_special_tokens=False must not add
    # a second one.
    assert ids[0] == 2
    assert ids[1] != 2


def test_decode_completion_strips_stop_tokens(tokenizer, code_tokens):
    text, raw = driver.decode_completion(
        tokenizer, [*code_tokens, 106], {0, 1, 50, 106}
    )
    assert "<turn|>" not in text
    assert "def solve" in text
    assert "<turn|>" in raw


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


def test_real_worker_refuses_without_guard_attestation(tmp_path, monkeypatch):
    monkeypatch.delenv("MTPLX_GUARD_ATTEST_FD", raising=False)
    with pytest.raises(RuntimeError, match="MTPLX_GUARD_ATTEST_FD"):
        driver.main(
            [
                "--route", "serial",
                "--label", "guard",
                "--n", "20",
                "--out-dir", str(tmp_path),
            ]
        )
    assert not list(tmp_path.iterdir())


def test_fake_worker_is_exempt_from_the_guard(tmp_path, monkeypatch, code_tokens):
    monkeypatch.delenv("MTPLX_GUARD_ATTEST_FD", raising=False)
    receipt = _run(tmp_path, monkeypatch, route="serial",
                   tokens=[*code_tokens, 106], stop_ids=[0, 1, 50, 106])
    assert receipt["status"] == "generated"
    assert receipt["guard"]["attest_fd"] is None


# --------------------------------------------------------------------------
# End-to-end over the fake wire
# --------------------------------------------------------------------------


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    route: str,
    tokens: list[int],
    stop_ids: list[int],
    max_tokens: int = 768,
    extra: list[str] | None = None,
    label: str = "t",
) -> dict:
    config_path = tmp_path / "fake.json"
    config_path.write_text(
        json.dumps(
            {
                "serial": tokens,
                "dflash": tokens,
                "stop_ids": stop_ids,
                "filler": 99,
                "log": str(tmp_path / "requests.jsonl"),
                "spawn_log": str(tmp_path / "spawns.jsonl"),
            }
        )
    )
    monkeypatch.setenv("FAKE_RUNTIME_WORKER_CONFIG", str(config_path))
    argv = [
        "--route", route,
        "--label", label,
        "--n", "20",
        "--max-tokens", str(max_tokens),
        "--out-dir", str(tmp_path),
        "--fake-worker", str(FAKE_WORKER),
    ]
    if route == "dflash":
        argv += ["--dflash-depth", "15"]
    argv += extra or []
    assert driver.main(argv) == 0
    return json.loads((tmp_path / f"{label}-receipt.json").read_text())


def _requests(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)["request"]
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _spawns(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "spawns.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_serial_loop_stops_at_eos(tmp_path, monkeypatch, code_tokens):
    tokens = [*code_tokens, 106]
    receipt = _run(tmp_path, monkeypatch, route="serial", tokens=tokens,
                   stop_ids=[0, 1, 50, 106])

    assert receipt["route"] == "serial"
    assert receipt["stop_tokens"]["ids"] == [0, 1, 50, 106]
    assert len(receipt["rows"]) == 20
    row = receipt["rows"][0]
    assert row["token_ids"] == tokens
    assert row["tokens_generated"] == len(tokens)
    assert row["stopped_on"] == 106
    assert row["truncated_at_max_tokens"] is False
    assert row["generation_failed"] is False
    assert row["token_digest"] == driver.token_digest(tokens)

    # ONE resident worker for all 20 problems, and exactly len(tokens)-1
    # decode_steps per problem.
    assert len(_spawns(tmp_path)) == 1
    assert receipt["worker_lifecycle"]["spawns"] == 1
    assert receipt["worker_lifecycle"]["restarts"] == []
    requests = _requests(tmp_path)
    assert sum(1 for r in requests if r["kind"] == "decode_begin") == 20
    assert (
        sum(1 for r in requests if r["kind"] == "decode_step")
        == 20 * (len(tokens) - 1)
    )
    first_begin = next(r for r in requests if r["kind"] == "decode_begin")
    assert first_begin["spec"] == {"mode": "serial"}
    assert len(first_begin["seed_tokens"]) == row["prompt_tokens"]


def test_serial_truncates_at_max_tokens(tmp_path, monkeypatch, code_tokens):
    receipt = _run(tmp_path, monkeypatch, route="serial", tokens=code_tokens,
                   stop_ids=[0, 1, 50, 106], max_tokens=5)
    row = receipt["rows"][0]
    assert row["tokens_generated"] == 5
    assert row["stopped_on"] is None
    assert row["truncated_at_max_tokens"] is True
    assert receipt["totals"]["truncated_at_max_tokens"] == 20


def test_samples_and_receipt_files_are_written_in_evalplus_format(
    tmp_path, monkeypatch, code_tokens
):
    receipt = _run(tmp_path, monkeypatch, route="serial",
                   tokens=[*code_tokens, 106], stop_ids=[0, 1, 50, 106])
    samples = [
        json.loads(line)
        for line in (tmp_path / "t-samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(samples) == 20
    assert all(set(row) == {"task_id", "solution"} for row in samples)
    assert samples[0]["task_id"] == "HumanEval/0"
    assert "def solve" in samples[0]["solution"]
    assert (tmp_path / "t-samples.raw.jsonl").is_file()

    for key in (
        "schema", "label", "route", "n", "max_tokens", "sampler", "dataset",
        "worker", "weights", "stop_tokens", "guard", "rows", "totals",
        "worker_lifecycle", "artifacts", "hello",
    ):
        assert key in receipt, key
    assert receipt["schema"] == "gemma4-humaneval-worker-eval-v1"
    assert receipt["totals"]["problems"] == 20
    assert "evalplus_result" not in receipt

    # The padded scoring file is what evalplus.evaluate consumes for --n 20.
    padded = driver.write_scoring_file(
        tmp_path / "t-samples.jsonl",
        tmp_path / "scored.jsonl",
        [f"HumanEval/{i}" for i in range(164)],
    )
    assert padded["scored"] == 20 and padded["padded"] == 144


def test_dflash_clean_run_needs_no_rerun(tmp_path, monkeypatch, code_tokens):
    tokens = [*code_tokens, 106]
    receipt = _run(tmp_path, monkeypatch, route="dflash", tokens=tokens,
                   stop_ids=[0, 1, 50, 106], max_tokens=len(tokens))
    row = receipt["rows"][0]
    assert row["token_ids"] == tokens
    assert row["rerun"] is False
    assert row["worker_respawned"] is False
    assert receipt["worker_lifecycle"]["spawns"] == 1
    runs = [r for r in _requests(tmp_path) if r["kind"] == "free_decode_run"]
    # count is max_tokens - 1: N counts tokens AFTER the seed token.
    assert runs[0]["count"] == len(tokens) - 1
    begin = next(
        r for r in _requests(tmp_path) if r["kind"] == "free_decode_begin"
    )
    assert begin["spec"] == {"mode": "dflash", "dflash": {"depth": 15}}


def test_dflash_rerun_yields_the_prefix_through_the_stop(
    tmp_path, monkeypatch, code_tokens
):
    tokens = [*code_tokens, 106]
    receipt = _run(tmp_path, monkeypatch, route="dflash", tokens=tokens,
                   stop_ids=[0, 1, 50, 106], max_tokens=768)
    row = receipt["rows"][0]
    assert row["token_ids"] == tokens
    assert row["stopped_on"] == 106
    assert row["rerun"] is True
    assert row["rerun_position"] == len(tokens) - 1
    assert row["worker_respawned"] is True
    assert row["generation_failed"] is False

    counts = [
        r["count"] for r in _requests(tmp_path) if r["kind"] == "free_decode_run"
    ]
    assert counts[0] == 767
    assert counts[1] == len(tokens) - 1

    # The poisoned worker is REPLACED, not reused: one respawn per problem.
    assert receipt["totals"]["reruns"] == 20
    assert receipt["totals"]["worker_respawns"] == 20
    assert len(_spawns(tmp_path)) == 21
    assert receipt["worker_lifecycle"]["spawns"] == 21
    assert all(
        "committed stop token" in entry["reason"]
        for entry in receipt["worker_lifecycle"]["restarts"]
    )


def test_dflash_length_hint_avoids_the_respawn(
    tmp_path, monkeypatch, code_tokens
):
    tokens = [*code_tokens, 106]
    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    _run(serial_dir, monkeypatch, route="serial", tokens=tokens,
         stop_ids=[0, 1, 50, 106], label="ctl")

    dflash_dir = tmp_path / "dflash"
    dflash_dir.mkdir()
    receipt = _run(
        dflash_dir, monkeypatch, route="dflash", tokens=tokens,
        stop_ids=[0, 1, 50, 106], max_tokens=768,
        extra=["--length-hint", str(serial_dir / "ctl-receipt.json")],
    )
    assert receipt["length_hint"]["problems"] == 20
    row = receipt["rows"][0]
    assert row["hint_tokens"] == len(tokens)
    assert row["token_ids"] == tokens
    assert row["rerun"] is False
    assert row["worker_respawned"] is False
    assert receipt["totals"]["worker_respawns"] == 0
    assert len(_spawns(dflash_dir)) == 1
    counts = [
        r["count"]
        for r in _requests(dflash_dir)
        if r["kind"] == "free_decode_run"
    ]
    assert counts == [len(tokens) - 1] * 20


def test_dflash_short_hint_falls_through_to_the_full_budget(
    tmp_path, monkeypatch, code_tokens
):
    """A hint shorter than the real stop costs one extra pair, never a restart."""

    tokens = [*code_tokens, 106]
    hint_receipt = tmp_path / "hint.json"
    hint_receipt.write_text(
        json.dumps(
            {
                "rows": [
                    {"task_id": f"HumanEval/{i}", "tokens_generated": 3}
                    for i in range(20)
                ]
            }
        )
    )
    receipt = _run(
        tmp_path, monkeypatch, route="dflash", tokens=tokens,
        stop_ids=[0, 1, 50, 106], max_tokens=768,
        extra=["--length-hint", str(hint_receipt)],
    )
    row = receipt["rows"][0]
    assert row["token_ids"] == tokens
    assert row["rerun"] is True
    counts = [
        r["count"] for r in _requests(tmp_path) if r["kind"] == "free_decode_run"
    ]
    assert counts[:3] == [2, 767, len(tokens) - 1]


def test_dflash_truncates_when_no_stop_token_is_committed(
    tmp_path, monkeypatch, code_tokens
):
    receipt = _run(tmp_path, monkeypatch, route="dflash", tokens=code_tokens,
                   stop_ids=[0, 1, 50, 106], max_tokens=6)
    row = receipt["rows"][0]
    assert row["tokens_generated"] == 6
    assert row["truncated_at_max_tokens"] is True
    assert row["rerun"] is False
    assert receipt["totals"]["worker_respawns"] == 0


def test_dflash_records_a_failed_generation_on_an_unrecognised_error():
    """A non-early-EOS failure is recorded once, never retried in a loop."""

    class _Broken:
        def __init__(self):
            self.spawns = 1
            self.restarts = []
            self.worker = object()

        def request(self, body):
            return {"ok": False, "error": "invalidInput(\"boom\")"}

        def restart(self, *, reason, task_id=None):
            self.spawns += 1
            self.restarts.append({"reason": reason, "task_id": task_id})

    handle = _Broken()
    outcome = driver.generate_dflash(
        handle,
        seed_tokens=[1, 2, 3],
        max_tokens=768,
        stop_ids={106},
        depth=15,
        task_id="HumanEval/0",
    )
    assert outcome["failed"] is True
    assert outcome["tokens"] == []
    assert outcome["respawned"] is True
    assert handle.spawns == 2
    assert handle.restarts[0]["task_id"] == "HumanEval/0"


def test_free_run_count_ceiling_is_enforced_client_side():
    class _Unused:
        pass

    with pytest.raises(ValueError, match="engine ceiling"):
        driver.generate_dflash(
            _Unused(),
            seed_tokens=[1],
            max_tokens=driver.FREE_RUN_COUNT_CEILING + 2,
            stop_ids={106},
            depth=None,
            task_id="HumanEval/0",
        )
