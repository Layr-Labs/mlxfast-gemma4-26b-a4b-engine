from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


MAX_RECEIPT_BYTES = 4 * 1024 * 1024
"""Receipts are tiny; 4 MiB leaves ample headroom while bounding JSON input."""


def _validated_tokens(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    for index, token in enumerate(value):
        if type(token) is not int:
            raise ValueError(f"{field}[{index}] must be an integer")
    return value


def _receipt_samples(receipt: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must be an object")
    if "samples" not in receipt:
        raise ValueError(f"{label} missing required field 'samples'")
    samples = receipt["samples"]
    if not isinstance(samples, list):
        raise ValueError(f"{label} samples must be a list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{label} sample {index} must be an object")
    return samples


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def parse_receipt(payload: bytes, label: str = "receipt") -> dict[str, Any]:
    try:
        receipt = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains invalid JSON: {error.msg}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must be an object")
    return receipt


def load_receipt(path: Path, label: str = "receipt") -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_RECEIPT_BYTES:
        raise ValueError(
            f"{label} exceeds {MAX_RECEIPT_BYTES}-byte size limit"
        )
    with path.open("rb") as stream:
        payload = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ValueError(
            f"{label} exceeds {MAX_RECEIPT_BYTES}-byte size limit"
        )
    return parse_receipt(payload, label)


def positional_mismatches(reference: list[int], candidate: list[int]) -> int:
    shared = sum(left != right for left, right in zip(reference, candidate))
    return shared + abs(len(reference) - len(candidate))


def validate_candidate(
    serial_tokens: list[int],
    samples: list[dict[str, Any]],
    max_mismatches: int,
    min_decode_tps: float,
    required_samples: int = 3,
) -> dict[str, Any]:
    if type(min_decode_tps) not in (int, float):
        raise ValueError("min_decode_tps must be a number")
    if not math.isfinite(min_decode_tps):
        raise ValueError(
            f"min_decode_tps {min_decode_tps} must be finite"
        )
    serial_tokens = _validated_tokens(serial_tokens, "serial tokens")
    if len(serial_tokens) != 128:
        raise ValueError(f"serial token count {len(serial_tokens)} != 128")
    if type(required_samples) is not int or required_samples < 1:
        raise ValueError("required_samples must be a positive integer")
    if not isinstance(samples, list):
        raise ValueError("candidate samples must be a list")
    if len(samples) != required_samples:
        if required_samples == 3:
            raise ValueError(
                f"expected three retained samples, found {len(samples)}"
            )
        raise ValueError(
            f"expected {required_samples} retained samples, found {len(samples)}"
        )
    mismatches: list[int] = []
    rates: list[int | float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        if "tokens" not in sample:
            raise ValueError(f"sample {index} missing required field 'tokens'")
        tokens = _validated_tokens(sample["tokens"], f"sample {index} tokens")
        if len(tokens) != 128:
            raise ValueError(f"sample {index} token count {len(tokens)} != 128")
        mismatch_count = positional_mismatches(serial_tokens, tokens)
        if mismatch_count > max_mismatches:
            raise ValueError(
                f"sample {index} has {mismatch_count} positional mismatches; "
                f"maximum is {max_mismatches}"
            )
        if "decode_tps" not in sample:
            raise ValueError(
                f"sample {index} missing required field 'decode_tps'"
            )
        rate = sample["decode_tps"]
        if type(rate) not in (int, float):
            raise ValueError(f"sample {index} decode_tps must be a number")
        if not math.isfinite(rate):
            raise ValueError(
                f"sample {index} decode_tps {rate} is not finite"
            )
        if rate < min_decode_tps:
            raise ValueError(
                f"sample {index} decode_tps {rate} < {min_decode_tps}"
            )
        mismatches.append(mismatch_count)
        rates.append(rate)
    mean_rate = statistics.fmean(rates)
    if not math.isfinite(mean_rate):
        raise ValueError(f"mean decode_tps {mean_rate} is not finite")
    if mean_rate < min_decode_tps:
        raise ValueError(f"mean decode_tps {mean_rate} < {min_decode_tps}")
    return {
        "passed": True,
        "positional_mismatches": mismatches,
        "decode_tps": rates,
        "mean_decode_tps": mean_rate,
    }


def first_tokens(receipt: dict[str, Any]) -> list[int]:
    samples = _receipt_samples(receipt, "serial receipt")
    if not samples:
        raise ValueError("serial receipt has no retained samples")
    sample = samples[0]
    if "tokens" not in sample:
        raise ValueError("serial sample 0 missing required field 'tokens'")
    return _validated_tokens(sample["tokens"], "serial sample 0 tokens").copy()


def validate_worker_sha(
    serial: dict[str, Any], candidate: dict[str, Any]
) -> str:
    if not isinstance(serial, dict):
        raise ValueError("serial receipt must be an object")
    if not isinstance(candidate, dict):
        raise ValueError("candidate receipt must be an object")
    serial_sha = serial.get("worker_sha256")
    candidate_sha = candidate.get("worker_sha256")
    if not isinstance(serial_sha, str) or not serial_sha.strip():
        raise ValueError("serial worker_sha256 must be a non-empty string")
    if not isinstance(candidate_sha, str) or not candidate_sha.strip():
        raise ValueError("candidate worker_sha256 must be a non-empty string")
    if serial_sha != candidate_sha:
        raise ValueError("serial and candidate worker SHA values differ")
    return serial_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-mismatches", type=int, default=12)
    parser.add_argument("--min-decode-tps", type=float, default=200.0)
    parser.add_argument("--required-samples", type=int, default=3)
    args = parser.parse_args()
    serial = load_receipt(args.serial, "serial receipt")
    candidate = load_receipt(args.candidate, "candidate receipt")
    validate_worker_sha(serial, candidate)
    result = validate_candidate(
        first_tokens(serial),
        _receipt_samples(candidate, "candidate receipt"),
        args.max_mismatches,
        args.min_decode_tps,
        args.required_samples,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
