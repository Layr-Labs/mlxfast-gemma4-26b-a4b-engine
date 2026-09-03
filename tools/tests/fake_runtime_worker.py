#!/usr/bin/env python3
"""A model-free stand-in for `mlxfast-runtime-worker`'s JSON-lines protocol.

It reproduces the SEMANTICS the driver depends on, taken from
`Sources/MLXFastHarness/` rather than invented:

* a `hello` on id 0 carrying a nonce, `spec_modes` and `capabilities`;
* every response echoes the session nonce and the request id;
* `decode_begin` returns `seed_token` (the FIRST generated token) and resets
  the step cursor -- it is repeatable, because the real one opens a fresh
  `model.newCache` every time;
* `decode_step {"token": t}` returns the NEXT token; the plain verbs know
  nothing about stop tokens, so the client is what halts;
* `free_decode_begin` returns `seed_token` and opens a session;
* `free_decode_run {"count": N}` CONSUMES that session and either returns
  exactly N tokens or fails with
  `RuntimeWorkerFreeRunError.stopTokenBeforeTarget`'s rendered description
  when a stop token is committed before N;
* ANY failure poisons the session permanently
  (`Gemma4Runtime.runWorker`'s catch sets `state.poisoned = true` and nothing
  clears it), so every later request is refused with guard 1's message.

Configured by `FAKE_RUNTIME_WORKER_CONFIG` (a JSON file):

    {
      "serial":     [ids...],   # seed token first
      "dflash":     [ids...],   # seed token first
      "filler":     99,         # returned once `serial` is exhausted
      "spec_modes": ["serial", "mtp", "dflash"],
      "log":        "/path/requests.jsonl",
      "spawn_log":  "/path/spawns.jsonl"
    }
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path


def append(path: str | None, row: dict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> int:
    config = json.loads(
        Path(os.environ["FAKE_RUNTIME_WORKER_CONFIG"]).read_text(encoding="utf-8")
    )
    serial = [int(v) for v in config.get("serial", [])]
    dflash = [int(v) for v in config.get("dflash", [])]
    filler = int(config.get("filler", 99))
    stop_ids = set(int(v) for v in config.get("stop_ids", []))
    log = config.get("log")
    nonce = uuid.uuid4().hex
    append(config.get("spawn_log"), {"argv": sys.argv[1:], "nonce": nonce})

    sys.stdout.write(
        json.dumps(
            {
                "id": 0,
                "nonce": nonce,
                "ok": True,
                "spec_modes": config.get(
                    "spec_modes", ["serial", "mtp", "dflash"]
                ),
                "capabilities": ["free_run_decode", "deferred_free_run_finalize"],
                "protocol_version": "v1.1",
                "backend": "fake",
                "device": "fake",
            }
        )
        + "\n"
    )
    sys.stdout.flush()

    poisoned = False
    step = 0
    free_session_open = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        append(log, {"nonce": nonce, "request": request})
        request_id = request.get("id", -1)
        kind = request.get("kind")

        def reply(payload: dict) -> None:
            body = {"id": request_id, "nonce": nonce, **payload}
            sys.stdout.write(json.dumps(body) + "\n")
            sys.stdout.flush()

        if poisoned:
            reply(
                {
                    "ok": False,
                    "error": "invalidInput(\"the generic decode session is "
                    "poisoned after an earlier failure\")",
                }
            )
            continue

        if kind == "decode_begin":
            step = 0
            reply(
                {
                    "ok": True,
                    "seed_token": serial[0],
                    "effective_spec": {"mode": "serial"},
                }
            )
        elif kind == "decode_step":
            step += 1
            token = serial[step] if step < len(serial) else filler
            reply({"ok": True, "token": token})
        elif kind == "free_decode_begin":
            free_session_open = True
            spec = request.get("spec") or {}
            depth = (spec.get("dflash") or {}).get("depth")
            reply(
                {
                    "ok": True,
                    "seed_token": dflash[0],
                    "effective_spec": {
                        "mode": "dflash",
                        "dflash": {"depth": depth, "draft": {"artifact": "dflash-head"}},
                    },
                }
            )
        elif kind == "free_decode_run":
            if not free_session_open:
                poisoned = True
                reply(
                    {
                        "ok": False,
                        "error": "invalidInput(\"runtime worker free_decode_run "
                        "before free_decode_begin\")",
                    }
                )
                continue
            free_session_open = False
            count = int(request.get("count", 0))
            body = dflash[1:]
            stop_index = next(
                (i for i, t in enumerate(body) if t in stop_ids), None
            )
            if stop_index is not None and stop_index + 1 < count:
                position = stop_index + 1
                poisoned = True
                reply(
                    {
                        "ok": False,
                        "error": (
                            f"free_decode_run dflash leg committed stop token "
                            f"{body[stop_index]} at committed position {position} "
                            f"of N={count}; the leg is invalid (a paired leg must "
                            f"reach N or both legs must stop)"
                        ),
                    }
                )
                continue
            emitted = [
                body[i] if i < len(body) else filler for i in range(count)
            ]
            reply(
                {
                    "ok": True,
                    "tokens": emitted,
                    "acceptance_lengths": [len(emitted)] if emitted else [],
                    "drafted_total": max(len(emitted) - 1, 0),
                    "accepted_total": max(len(emitted) - 1, 0),
                    "committed_total": len(emitted),
                }
            )
        else:
            poisoned = True
            reply(
                {
                    "ok": False,
                    "error": f"invalidInput(\"runtime worker received unknown "
                    f"request kind {kind}\")",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
