from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "gemma4_dflash_fidelity_gate.py"
SPEC = importlib.util.spec_from_file_location("fidelity_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class FidelityGateTests(unittest.TestCase):
    def test_positional_mismatches_counts_values_and_length(self) -> None:
        self.assertEqual(gate.positional_mismatches([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(gate.positional_mismatches([1, 2], [1, 2, 3]), 1)

    def test_candidate_requires_token_lists(self) -> None:
        serial = list(range(128))
        samples = [
            {"tokens": serial, "decode_tps": 250.0},
            {"tokens": serial, "decode_tps": 250.0},
            {"tokens": serial, "decode_tps": 250.0},
        ]
        with self.assertRaisesRegex(ValueError, "serial tokens must be a list"):
            gate.validate_candidate("x" * 128, samples, 12, 200.0)

        string_tokens = "x" * 128
        string_samples = [
            {"tokens": string_tokens, "decode_tps": 250.0},
            {"tokens": string_tokens, "decode_tps": 250.0},
            {"tokens": string_tokens, "decode_tps": 250.0},
        ]
        with self.assertRaisesRegex(ValueError, "sample 0 tokens must be a list"):
            gate.validate_candidate(serial, string_samples, 12, 200.0)

        with self.assertRaisesRegex(
            ValueError, "serial sample 0 tokens must be a list"
        ):
            gate.first_tokens({"samples": [{"tokens": string_tokens}]})

    def test_candidate_rejects_non_finite_minimum_decode_tps(self) -> None:
        for threshold in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError, "min_decode_tps.*finite"
                ):
                    gate.validate_candidate(
                        serial_tokens=list(range(128)),
                        samples=[],
                        max_mismatches=12,
                        min_decode_tps=threshold,
                    )

    def test_candidate_rejects_non_integer_token_values(self) -> None:
        invalid_tokens = (1.5, "1", True)
        for invalid_token in invalid_tokens:
            valid_serial = list(range(128))
            valid_samples = [
                {"tokens": valid_serial, "decode_tps": 250.0},
                {"tokens": valid_serial, "decode_tps": 250.0},
                {"tokens": valid_serial, "decode_tps": 250.0},
            ]
            serial = list(range(128))
            serial[0] = invalid_token
            with self.subTest(path="serial", value=invalid_token):
                with self.assertRaisesRegex(
                    ValueError, r"serial tokens\[0\] must be an integer"
                ):
                    gate.validate_candidate(serial, valid_samples, 12, 200.0)

            candidate = valid_serial.copy()
            candidate[0] = invalid_token
            samples = [
                {"tokens": candidate, "decode_tps": 250.0},
                {"tokens": valid_serial, "decode_tps": 250.0},
                {"tokens": valid_serial, "decode_tps": 250.0},
            ]
            with self.subTest(path="candidate", value=invalid_token):
                with self.assertRaisesRegex(
                    ValueError, r"sample 0 tokens\[0\] must be an integer"
                ):
                    gate.validate_candidate(valid_serial, samples, 12, 200.0)

    def test_candidate_rejects_non_numeric_decode_tps(self) -> None:
        serial = list(range(128))
        for invalid_rate in ("250.0", True):
            samples = [
                {"tokens": serial, "decode_tps": invalid_rate},
                {"tokens": serial, "decode_tps": 250.0},
                {"tokens": serial, "decode_tps": 250.0},
            ]
            with self.subTest(rate=invalid_rate):
                with self.assertRaisesRegex(
                    ValueError, "sample 0 decode_tps must be a number"
                ):
                    gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_requires_all_three_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "three retained samples"):
            gate.validate_candidate(
                serial_tokens=list(range(128)),
                samples=[{"tokens": list(range(128)), "decode_tps": 250.0}],
                max_mismatches=12,
                min_decode_tps=200.0,
            )

    def test_candidate_rejects_thirteenth_mismatch(self) -> None:
        serial = list(range(128))
        candidate = serial.copy()
        candidate[:13] = [value + 1000 for value in candidate[:13]]
        samples = [
            {"tokens": candidate, "decode_tps": 250.0},
            {"tokens": candidate, "decode_tps": 251.0},
            {"tokens": candidate, "decode_tps": 252.0},
        ]
        with self.assertRaisesRegex(ValueError, "13 positional mismatches"):
            gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_rejects_one_slow_sample_even_when_mean_passes(self) -> None:
        serial = list(range(128))
        samples = [
            {"tokens": serial, "decode_tps": 199.0},
            {"tokens": serial, "decode_tps": 250.0},
            {"tokens": serial, "decode_tps": 250.0},
        ]
        with self.assertRaisesRegex(ValueError, "sample 0 decode_tps"):
            gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_rejects_non_finite_decode_rates(self) -> None:
        serial = list(range(128))
        samples = [
            {"tokens": serial, "decode_tps": float("nan")},
            {"tokens": serial, "decode_tps": float("nan")},
            {"tokens": serial, "decode_tps": float("nan")},
        ]
        with self.assertRaisesRegex(ValueError, "sample 0 decode_tps.*finite"):
            gate.validate_candidate(serial, samples, 12, 200.0)

    def test_candidate_returns_per_sample_mismatches_and_mean(self) -> None:
        serial = list(range(128))
        drifted = serial.copy()
        drifted[7] = 999
        result = gate.validate_candidate(
            serial,
            [
                {"tokens": drifted, "decode_tps": 201.0},
                {"tokens": serial, "decode_tps": 202.0},
                {"tokens": serial, "decode_tps": 203.0},
            ],
            12,
            200.0,
        )
        self.assertEqual(result["positional_mismatches"], [1, 0, 0])
        self.assertEqual(result["mean_decode_tps"], 202.0)

    def test_candidate_accepts_exactly_twelve_mismatches(self) -> None:
        serial = list(range(128))
        candidate = serial.copy()
        candidate[:12] = [value + 1000 for value in candidate[:12]]
        result = gate.validate_candidate(
            serial,
            [
                {"tokens": candidate, "decode_tps": 201.0},
                {"tokens": candidate, "decode_tps": 202.0},
                {"tokens": candidate, "decode_tps": 203.0},
            ],
            12,
            200.0,
        )
        self.assertEqual(result["positional_mismatches"], [12, 12, 12])

    def test_receipt_parser_rejects_duplicate_keys(self) -> None:
        duplicate_receipts = {
            "worker_sha256": b'{"worker_sha256":"a","worker_sha256":"b"}',
            "samples": b'{"samples":[],"samples":[]}',
            "tokens": b'{"samples":[{"tokens":[],"tokens":[]}]}',
            "decode_tps": b'{"samples":[{"decode_tps":1,"decode_tps":2}]}',
        }
        for key, payload in duplicate_receipts.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ValueError, f"duplicate JSON key '{key}'"
                ):
                    gate.parse_receipt(payload)

    def test_receipt_schema_errors_are_field_specific(self) -> None:
        with self.assertRaisesRegex(ValueError, "receipt must be an object"):
            gate.parse_receipt(b"[]")
        with self.assertRaisesRegex(ValueError, "candidate samples must be a list"):
            gate.validate_candidate(list(range(128)), {}, 12, 200.0)
        with self.assertRaisesRegex(ValueError, "sample 0 must be an object"):
            gate.validate_candidate(list(range(128)), [1, 2, 3], 12, 200.0)
        with self.assertRaisesRegex(
            ValueError, "sample 0 missing required field 'tokens'"
        ):
            gate.validate_candidate(
                list(range(128)),
                [{"decode_tps": 250.0}] * 3,
                12,
                200.0,
            )
        with self.assertRaisesRegex(
            ValueError, "sample 0 missing required field 'decode_tps'"
        ):
            gate.validate_candidate(
                list(range(128)),
                [{"tokens": list(range(128))}] * 3,
                12,
                200.0,
            )

    def test_receipt_loader_enforces_size_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            receipt_path.write_bytes(b"{}")
            with mock.patch.object(gate, "MAX_RECEIPT_BYTES", 1):
                with self.assertRaisesRegex(
                    ValueError, "receipt exceeds 1-byte size limit"
                ):
                    gate.load_receipt(receipt_path)

    def test_worker_sha_requires_present_non_empty_values(self) -> None:
        invalid_receipt_pairs = [
            ({}, {"worker_sha256": "sha"}),
            ({"worker_sha256": "sha"}, {}),
            ({"worker_sha256": ""}, {"worker_sha256": "sha"}),
            ({"worker_sha256": "sha"}, {"worker_sha256": ""}),
        ]
        for serial, candidate in invalid_receipt_pairs:
            with self.subTest(serial=serial, candidate=candidate):
                with self.assertRaisesRegex(
                    ValueError, "worker_sha256.*non-empty"
                ):
                    gate.validate_worker_sha(serial, candidate)

    def test_worker_sha_requires_identical_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker SHA values differ"):
            gate.validate_worker_sha(
                {"worker_sha256": "serial-sha"},
                {"worker_sha256": "candidate-sha"},
            )


if __name__ == "__main__":
    unittest.main()
