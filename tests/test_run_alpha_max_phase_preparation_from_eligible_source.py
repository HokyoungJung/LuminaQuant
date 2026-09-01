from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import socket
import tempfile
import time
import unittest
from unittest import mock
import polars as pl


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "research"
    / "run_alpha_max_phase_preparation_from_eligible_source.py"
)
ACQUIRER = SCRIPT.with_name("acquire_alpha_max_official_source.py")
SPEC = importlib.util.spec_from_file_location("eligible_phase_wrapper", SCRIPT)
assert SPEC and SPEC.loader
wrapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wrapper
SPEC.loader.exec_module(wrapper)


class EligiblePhasePreparationTests(unittest.TestCase):
    def test_frozen_acquirer_pin_matches_admitted_sibling(self) -> None:
        self.assertEqual(
            hashlib.sha256(ACQUIRER.read_bytes()).hexdigest(),
            wrapper.ACQUIRER_SHA256,
        )

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.acquirer = self._file("acquirer.py", b"acquirer")
        self.preparer = self._file("preparer.py", b"preparer")
        self.contract = self._file(
            "contract.json",
            wrapper.canonical_bytes(
                {
                    "schema_version": "alpha_max_contract_manifest.v2",
                    "exchange": "binance",
                    "records": [
                        {
                            "symbol": symbol,
                            "raw_availability_start_utc": "2024-01-01T00:00:00Z",
                            "raw_availability_end_utc": "2024-01-01T00:00:02Z",
                            "feature_availability_start_utc": "2024-01-01T00:00:00Z",
                            "feature_availability_end_utc": (
                                "2024-01-01T04:00:00Z"
                                if symbol == "TONUSDT"
                                else "2024-01-01T08:00:00Z"
                            ),
                        }
                        for symbol in (
                            "ADAUSDT",
                            "AVAXUSDT",
                            "BNBUSDT",
                            "BTCUSDT",
                            "DOGEUSDT",
                            "ETHUSDT",
                            "SOLUSDT",
                            "TONUSDT",
                            "TRXUSDT",
                            "XRPUSDT",
                        )
                    ],
                }
            ),
        )
        self.evidence = self._file("evidence.json", b"evidence")
        self.source = self.root / "source"
        (self.source / "market_ohlcv_1s").mkdir(parents=True)
        (self.source / "feature_points").mkdir()
        self.report = self.root / "report"
        self.report.mkdir()
        self._create_authentic_sources()
        (self.source / "snapshot-manifest.json").write_bytes(
            wrapper.canonical_bytes({"entries": self._snapshot_entries()})
        )
        self.source_artifact = self.source / "market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet"
        self.report_artifact = self._file("report/provenance/proof.json", b"proof")
        self._file("source/.alpha_max_owner.json", b"source-owner")
        self._file("report/.alpha_max_owner.json", b"report-owner")
        self._file("report/plan.json", b"plan")
        self._file("report/source_eligible_receipt.json", b"receipt")
        self._file("report/acquisition.journal.jsonl", b"journal")
        self._write_manifest()
        self.forbidden = self.root / "quarantined"
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _file(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def _replace_read_only_leaf(self, path: Path, data: bytes) -> None:
        parent = path.parent
        parent_mode = parent.stat().st_mode & 0o777
        os.chmod(parent, 0o700)
        try:
            replacement = path.with_name(f".{path.name}.replacement")
            replacement.write_bytes(data)
            os.chmod(replacement, 0o444)
            os.replace(replacement, path)
        finally:
            os.chmod(parent, parent_mode)

    def _remove_read_only_tree(self, path: Path) -> None:
        if not path.exists():
            return
        for directory in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_dir()),
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o700)
        os.chmod(path, 0o700)
        shutil.rmtree(path)

    @staticmethod
    def _ficlone(target_fd: int, request: int, source_fd: int) -> int:
        if request != wrapper.FICLONE:
            raise AssertionError("unexpected_ioctl_request")
        while block := os.read(source_fd, 1 << 20):
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise OSError("emulated_reflink_write_failed")
                view = view[written:]
        return 0

    def _create_authentic_sources(self) -> None:
        for index, symbol in enumerate(
            (
                "ADAUSDT",
                "AVAXUSDT",
                "BNBUSDT",
                "BTCUSDT",
                "DOGEUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "TONUSDT",
                "TRXUSDT",
                "XRPUSDT",
            )
        ):
            raw = pl.DataFrame(
                {
                    "datetime": pl.Series([1704067200000, 1704067201000], dtype=pl.Datetime("ms")),
                    "open": [float(index + 1), float(index + 1)],
                    "high": [float(index + 2), float(index + 2)],
                    "low": [float(index + 1), float(index + 1)],
                    "close": [float(index + 1.5), float(index + 1.5)],
                    "volume": [1.0, 2.0],
                    "exchange": ["binance", "binance"],
                    "symbol": [symbol, symbol],
                }
            )
            raw_path = self.source / f"market_ohlcv_1s/binance/{symbol}/2024-01.parquet"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.write_parquet(raw_path)
            feature = pl.DataFrame(
                {
                    "timestamp_ms": [1704067200000],
                    "funding_rate": [0.001],
                    "exchange": ["binance"],
                    "symbol": [symbol],
                }
            )
            feature_path = (
                self.source
                / f"feature_points/exchange=binance/symbol={symbol}/date=2024-01-01/funding.parquet"
            )
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature.write_parquet(feature_path)

    def _snapshot_entries(self) -> list[dict[str, object]]:
        entries = []
        for path in sorted(
            list((self.source / "market_ohlcv_1s").rglob("*.parquet"))
            + list((self.source / "feature_points").rglob("*.parquet"))
        ):
            relative = path.relative_to(self.source).as_posix()
            entries.append(
                {
                    "source_relative_path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "byte_count": path.stat().st_size,
                }
            )
        return entries

    def _write_manifest(self, artifacts: list[dict[str, str]] | None = None) -> None:
        if artifacts is None:
            artifacts = [
                {
                    "path": f"output/{entry['source_relative_path']}",
                    "sha256": entry["sha256"],
                }
                for entry in self._snapshot_entries()
            ]
            artifacts.append(
                {
                    "path": "report/provenance/proof.json",
                    "sha256": hashlib.sha256(self.report_artifact.read_bytes()).hexdigest(),
                }
            )
        self._file(
            "report/source_manifest.json",
            wrapper.canonical_bytes(
                {"schema": "alpha_max_official_source_manifest.v5", "artifacts": artifacts}
            ),
        )

    def _publish_output(self) -> bytes:
        self.output.mkdir()
        snapshot = {"entries": self._snapshot_entries()}
        contract = json.loads(self.contract.read_bytes())
        layout, availability = wrapper._expected_output_layout(contract, snapshot)
        files = []
        for expected in layout:
            source = self.source / expected["source_relative_path"]
            output = self.output / expected["output_relative_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            source_frame = pl.read_parquet(source)
            start = wrapper._parse_utc(expected["owned_start_utc"], "test")
            end = wrapper._parse_utc(expected["owned_end_utc"], "test")
            if expected["root_kind"] == "raw":
                frame = source_frame.filter(
                    (pl.col("datetime").dt.epoch("ms") >= int(start.timestamp() * 1000))
                    & (pl.col("datetime").dt.epoch("ms") < int(end.timestamp() * 1000))
                )
            else:
                interval = 14_400_000 if expected["symbol"] == "TONUSDT" else 28_800_000
                frame = (
                    source_frame.filter(
                        pl.col("funding_rate").is_not_null() & pl.col("funding_rate").is_finite()
                    )
                    .with_columns(
                        [
                            pl.col("timestamp_ms").cast(pl.Int64).alias("source_timestamp_ms"),
                            ((pl.col("timestamp_ms").cast(pl.Int64) // interval) * interval).alias(
                                "timestamp_ms"
                            ),
                            pl.lit("binance").alias("exchange"),
                            pl.lit(expected["symbol"]).alias("symbol"),
                            pl.col("funding_rate").cast(pl.Float64),
                        ]
                    )
                    .filter(
                        (pl.col("timestamp_ms") >= int(start.timestamp() * 1000))
                        & (pl.col("timestamp_ms") < int(end.timestamp() * 1000))
                    )
                    .select(
                        [
                            "timestamp_ms",
                            "source_timestamp_ms",
                            "exchange",
                            "symbol",
                            "funding_rate",
                        ]
                    )
                )
            frame.write_parquet(output)
            files.append(
                {
                    **expected,
                    "output_relative_path": expected["output_relative_path"],
                    "output_byte_count": output.stat().st_size,
                    "output_row_count": frame.height,
                    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )
        files.sort(key=lambda entry: entry["output_relative_path"])
        manifest = {
            "availability": availability,
            "availability_sha256_by_root_kind": {
                kind: hashlib.sha256(wrapper.canonical_bytes(availability[kind])).hexdigest()
                for kind in ("raw", "feature")
            },
            "contract_manifest_schema_version": contract["schema_version"],
            "contract_manifest_sha256": hashlib.sha256(self.contract.read_bytes()).hexdigest(),
            "exchange": "binance",
            "file_count": len(files),
            "files": files,
            "phase_intervals": [
                {"phase_id": phase_id, "start_utc": start, "end_utc": end}
                for phase_id, start, end in wrapper.PHASE_INTERVALS
            ],
            "schema_version": "alpha_max_phase_root_preparation_manifest.v1",
            "symbols": [record["symbol"] for record in contract["records"]],
        }
        manifest_path = self.output / "preparation_manifest.json"
        manifest_path.write_bytes(wrapper.canonical_bytes(manifest))
        return wrapper.canonical_bytes(
            {
                "file_count": len(files),
                "output_root": str(self.output),
                "preparation_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )

    def _refresh_output_manifest_entry(self, output: Path) -> None:
        manifest_path = self.output / "preparation_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        relative = output.relative_to(self.output).as_posix()
        entry = next(item for item in manifest["files"] if item["output_relative_path"] == relative)
        entry["output_byte_count"] = output.stat().st_size
        entry["output_row_count"] = pl.read_parquet(output).height
        entry["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest_path.write_bytes(wrapper.canonical_bytes(manifest))

    def _output_entry(self, kind: str, symbol: str = "ADAUSDT") -> Path:
        manifest = json.loads((self.output / "preparation_manifest.json").read_bytes())
        entry = next(
            item
            for item in manifest["files"]
            if item["root_kind"] == kind and item["symbol"] == symbol
        )
        return self.output / entry["output_relative_path"]

    def test_authenticator_accepts_complete_strict_inventory(self) -> None:
        self._publish_output()
        value = wrapper._authenticate_output(
            self.output, self.contract, {"entries": self._snapshot_entries()}, self.source
        )
        self.assertEqual(value["file_count"], 20)

    def test_authenticator_rejects_tampered_pinned_snapshot_leaf_before_parsing(self) -> None:
        self._publish_output()
        snapshot = {"entries": self._snapshot_entries()}
        self.source_artifact.write_bytes(b"adversarial replacement")
        with self.assertRaisesRegex(wrapper.PreparationError, "^snapshot_parquet_mismatch$"):
            wrapper._authenticate_output(self.output, self.contract, snapshot, self.source)

    def test_authenticator_uses_pinned_output_leaves_after_nested_directory_replacement(
        self,
    ) -> None:
        self._publish_output()
        target = self._output_entry("raw")
        replacement = self.root / "replacement-output-subtree"
        shutil.copytree(target.parent, replacement)
        original_hash = wrapper._pinned_file_sha256
        original_read = wrapper._read_pinned_parquet
        hash_descriptors: set[int] = set()
        read_descriptors: set[int] = set()
        swapped = False

        def hash_then_replace(descriptor, before, label):
            nonlocal swapped
            value = original_hash(descriptor, before, label)
            if label == "output_parquet":
                hash_descriptors.add(descriptor)
                if not swapped:
                    swapped = True
                    parked = self.root / "parked-output-subtree"
                    target.parent.rename(parked)
                    replacement.rename(target.parent)
            return value

        def record_pinned_parse(descriptor, before, label):
            if label == "output_parquet":
                read_descriptors.add(descriptor)
            return original_read(descriptor, before, label)

        with (
            mock.patch.object(wrapper, "_pinned_file_sha256", side_effect=hash_then_replace),
            mock.patch.object(wrapper, "_read_pinned_parquet", side_effect=record_pinned_parse),
        ):
            value = wrapper._authenticate_output(
                self.output, self.contract, {"entries": self._snapshot_entries()}, self.source
            )

        self.assertEqual(value["file_count"], 20)
        self.assertEqual(hash_descriptors, read_descriptors)

    def test_authenticator_uses_pinned_source_leaves_after_nested_directory_replacement(
        self,
    ) -> None:
        self._publish_output()
        target = self.source_artifact
        target_identity = (target.stat().st_dev, target.stat().st_ino)
        replacement = self.root / "replacement-source-subtree"
        shutil.copytree(target.parent, replacement)
        original_hash = wrapper._pinned_file_sha256
        original_read = wrapper._read_pinned_parquet
        source_descriptors: list[tuple[int, tuple[int, int]]] = []
        swapped = False

        def hash_then_replace(descriptor, before, label):
            nonlocal swapped
            value = original_hash(descriptor, before, label)
            if label == "output_parquet" and not swapped:
                swapped = True
                parked = self.root / "parked-source-subtree"
                target.parent.rename(parked)
                replacement.rename(target.parent)
            return value

        def record_pinned_parse(descriptor, before, label):
            if label == "snapshot_parquet":
                item = os.fstat(descriptor)
                source_descriptors.append((descriptor, (item.st_dev, item.st_ino)))
            return original_read(descriptor, before, label)

        with (
            mock.patch.object(wrapper, "_pinned_file_sha256", side_effect=hash_then_replace),
            mock.patch.object(wrapper, "_read_pinned_parquet", side_effect=record_pinned_parse),
        ):
            value = wrapper._authenticate_output(
                self.output, self.contract, {"entries": self._snapshot_entries()}, self.source
            )

        self.assertEqual(value["file_count"], 20)
        self.assertIn(
            target_identity,
            {identity for _, identity in source_descriptors},
        )

    def test_authenticator_uses_pinned_output_root_after_root_replacement_during_inventory(
        self,
    ) -> None:
        self._publish_output()
        replacement = self.root / "replacement"
        shutil.copytree(self.output, replacement)
        original_scandir = wrapper.os.scandir
        swapped = False

        def swap_after_scan(descriptor):
            nonlocal swapped
            children = original_scandir(descriptor)
            if not swapped:
                swapped = True
                parked = self.root / "parked-output"
                self.output.rename(parked)
                replacement.rename(self.output)
            return children

        with mock.patch.object(wrapper.os, "scandir", side_effect=swap_after_scan):
            value = wrapper._authenticate_output(
                self.output, self.contract, {"entries": self._snapshot_entries()}, self.source
            )

        self.assertEqual(value["file_count"], 20)

    def test_verified_parquet_preserves_stability_failure(self) -> None:
        source = self.source_artifact
        with (
            mock.patch.object(
                wrapper,
                "_assert_stable_file",
                side_effect=wrapper.PreparationError("snapshot_parquet_changed_during_read"),
            ),
            self.assertRaisesRegex(
                wrapper.PreparationError, "^snapshot_parquet_changed_during_read$"
            ),
        ):
            wrapper._read_verified_parquet(source, "snapshot_parquet")

    def test_authenticator_rejects_complete_inventory_and_semantic_mutations(self) -> None:
        def mutate_omitted() -> None:
            self._output_entry("raw").unlink()

        def mutate_extra() -> None:
            self._file("output/unexpected.parquet", b"unexpected")

        def mutate_availability() -> None:
            manifest_path = self.output / "preparation_manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["availability"]["raw"]["availability_end_by_symbol"]["ADAUSDT"] = (
                "2024-01-01T00:00:03Z"
            )
            manifest_path.write_bytes(wrapper.canonical_bytes(manifest))

        def mutate_raw_datetime() -> None:
            output = self._output_entry("raw")
            pl.read_parquet(output).with_columns(
                pl.lit(1704067202000).cast(pl.Datetime("ms")).alias("datetime")
            ).write_parquet(output)
            self._refresh_output_manifest_entry(output)

        def mutate_raw_schema() -> None:
            output = self._output_entry("raw")
            pl.read_parquet(output).drop("volume").write_parquet(output)
            self._refresh_output_manifest_entry(output)

        def mutate_raw_ohlcv() -> None:
            output = self._output_entry("raw")
            pl.read_parquet(output).with_columns(pl.lit(0.0).alias("open")).write_parquet(output)
            self._refresh_output_manifest_entry(output)

        def mutate_raw_source_slice() -> None:
            output = self._output_entry("raw")
            pl.read_parquet(output).with_columns(pl.lit(1.75).alias("close")).write_parquet(output)
            self._refresh_output_manifest_entry(output)

        def mutate_feature_timestamp_jitter() -> None:
            output = self._output_entry("feature")
            pl.read_parquet(output).with_columns(
                (pl.col("timestamp_ms") + 1).alias("timestamp_ms")
            ).write_parquet(output)
            self._refresh_output_manifest_entry(output)

        def mutate_feature_exchange() -> None:
            output = self._output_entry("feature")
            pl.read_parquet(output).with_columns(pl.lit("other").alias("exchange")).write_parquet(
                output
            )
            self._refresh_output_manifest_entry(output)

        def mutate_feature_symbol() -> None:
            output = self._output_entry("feature")
            pl.read_parquet(output).with_columns(pl.lit("OTHER").alias("symbol")).write_parquet(
                output
            )
            self._refresh_output_manifest_entry(output)

        def mutate_feature_funding_rate() -> None:
            output = self._output_entry("feature")
            pl.read_parquet(output).with_columns(pl.lit(0.5).alias("funding_rate")).write_parquet(
                output
            )
            self._refresh_output_manifest_entry(output)

        def mutate_feature_source_timestamp_transform() -> None:
            output = self._output_entry("feature")
            pl.read_parquet(output).with_columns(
                (pl.col("source_timestamp_ms") + 1).alias("source_timestamp_ms")
            ).write_parquet(output)
            self._refresh_output_manifest_entry(output)

        cases = (
            ("omitted_output", mutate_omitted, "preparation_manifest_mismatch"),
            ("extra_output", mutate_extra, "preparation_manifest_mismatch"),
            ("wrong_availability", mutate_availability, "semantic_mismatch"),
            ("raw_datetime", mutate_raw_datetime, "output_raw_grid_invalid"),
            ("raw_schema", mutate_raw_schema, "output_raw_schema_invalid"),
            ("raw_ohlcv", mutate_raw_ohlcv, "output_raw_values_invalid"),
            ("raw_source_slice", mutate_raw_source_slice, "output_parquet_content_mismatch"),
            (
                "feature_canonical_timestamp_jitter",
                mutate_feature_timestamp_jitter,
                "output_feature_grid_invalid",
            ),
            ("feature_exchange", mutate_feature_exchange, "output_feature_grid_invalid"),
            ("feature_symbol", mutate_feature_symbol, "output_feature_grid_invalid"),
            (
                "feature_funding_rate",
                mutate_feature_funding_rate,
                "output_parquet_content_mismatch",
            ),
            (
                "feature_source_timestamp_transform",
                mutate_feature_source_timestamp_transform,
                "output_parquet_content_mismatch",
            ),
        )
        for name, mutate, error in cases:
            with self.subTest(name=name):
                shutil.rmtree(self.output, ignore_errors=True)
                self._publish_output()
                mutate()
                with self.assertRaisesRegex(wrapper.PreparationError, error):
                    wrapper._authenticate_output(
                        self.output,
                        self.contract,
                        {"entries": self._snapshot_entries()},
                        self.source,
                    )

    def test_preseeded_output_through_main_never_runs_preparer(self) -> None:
        self._publish_output()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "output_root_without_invocation_descriptor"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_partial_preseeded_output_never_runs_preparer(self) -> None:
        self.output.mkdir()
        (self.output / "foreign.partial").write_bytes(b"foreign")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "output_root_without_invocation_descriptor"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_unsafe_invocation_lock_prevents_subprocesses(self) -> None:
        lock = wrapper._sidecars(self.output)["lock"]
        lock.symlink_to(self.acquirer)
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "invocation_lock_unsafe_file"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_hardlinked_invocation_lock_prevents_subprocesses(self) -> None:
        lock = wrapper._sidecars(self.output)["lock"]
        os.link(self.acquirer, lock)
        acquirer_mode = self.acquirer.stat().st_mode
        acquirer_bytes = self.acquirer.read_bytes()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "invocation_lock_unsafe_file"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()
        self.assertEqual(self.acquirer.read_bytes(), acquirer_bytes)
        self.assertEqual(self.acquirer.stat().st_mode, acquirer_mode)

    def test_replaced_invocation_lock_prevents_subprocesses(self) -> None:
        lock = wrapper._sidecars(self.output)["lock"]
        open_lock = wrapper._open_invocation_lock
        calls = 0

        def replace_before_recheck(path: Path):
            nonlocal calls
            calls += 1
            if calls == 2:
                lock.unlink()
                lock.write_bytes(b"replacement")
                os.chmod(lock, 0o600)
            return open_lock(path)

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper, "_open_invocation_lock", side_effect=replace_before_recheck),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "invocation_lock_replaced"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()
        self.assertEqual(lock.read_bytes(), b"replacement")
        self.assertFalse(self.output.exists())

    def test_forbidden_output_overlap_precedes_invocation_lock_creation(self) -> None:
        self.output = self.forbidden / "output"
        lock = wrapper._sidecars(self.output)["lock"]
        with (
            mock.patch.object(wrapper, "_open_invocation_lock") as open_lock,
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "forbidden_root_overlap:output_root"),
        ):
            wrapper.main(self._args())
        open_lock.assert_not_called()
        run.assert_not_called()
        self.assertFalse(lock.exists())
        self.assertFalse(self.output.exists())

    def test_descriptor_flock_serializes_processes_and_releases_after_exception(self) -> None:
        template = self.root / "output-template"
        original_output = self.output
        self.output = template
        self._publish_output()
        self.output = original_output
        counter = self.root / "preparer-count"
        entered = self.root / "preparer-entered"
        release = self.root / "preparer-release"
        lock_attempts = self.root / "invocation-lock-attempts"
        self.acquirer.write_text("raise SystemExit(0)\n")
        self.preparer.write_text(
            "\n".join(
                (
                    "import hashlib, json, os, shutil, sys, time",
                    f"template = {str(template)!r}",
                    f"counter = {str(counter)!r}",
                    "if os.environ.get('FAKE_PREPARER_MODE') == 'fail':",
                    "    raise SystemExit(1)",
                    "output = sys.argv[sys.argv.index('--output-root') + 1]",
                    "with open(counter, 'ab', buffering=0) as stream:",
                    "    stream.write(b'1\\n')",
                    "    os.fsync(stream.fileno())",
                    "entered = os.environ.get('FAKE_PREPARER_ENTERED')",
                    "release = os.environ.get('FAKE_PREPARER_RELEASE')",
                    "if entered:",
                    "    with open(entered, 'xb', buffering=0) as stream:",
                    "        stream.write(b'entered\\n')",
                    "        os.fsync(stream.fileno())",
                    "    while release and not os.path.exists(release):",
                    "        time.sleep(0.01)",
                    "shutil.copytree(template, output)",
                    "manifest = os.path.join(output, 'preparation_manifest.json')",
                    "print(json.dumps({'file_count': 20, 'output_root': output, "
                    "'preparation_manifest_sha256': hashlib.sha256(open(manifest, 'rb').read()).hexdigest()}, "
                    "sort_keys=True, separators=(',', ':')))",
                )
            )
            + "\n"
        )
        patched_wrapper = self.root / "fresh-wrapper.py"
        patched_source = SCRIPT.read_text()
        for name, path in (
            ("ACQUIRER_SHA256", self.acquirer),
            ("CONTRACT_SHA256", self.contract),
            ("EVIDENCE_SHA256", self.evidence),
            ("PREPARER_SHA256", self.preparer),
        ):
            patched_source = patched_source.replace(
                f'{name} = "{getattr(wrapper, name)}"',
                f'{name} = "{hashlib.sha256(path.read_bytes()).hexdigest()}"',
            )
        lock_probe = """        attempted = os.environ.get("FAKE_INVOCATION_LOCK_ATTEMPT")
        if attempted:
            with open(attempted, "ab", buffering=0) as stream:
                stream.write(b"1\\n")
                os.fsync(stream.fileno())
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current_fd, current_identity"""
        self.assertIn(
            "        fcntl.flock(lock_fd, fcntl.LOCK_EX)\n        current_fd, current_identity",
            patched_source,
        )
        patched_source = patched_source.replace(
            "        fcntl.flock(lock_fd, fcntl.LOCK_EX)\n        current_fd, current_identity",
            lock_probe,
        )
        patched_wrapper.write_text(patched_source)

        def command(output: Path) -> list[str]:
            return [
                sys.executable,
                str(patched_wrapper),
                "--acquirer",
                str(self.acquirer),
                "--source-root",
                str(self.source),
                "--source-report",
                str(self.report),
                "--forbidden-root",
                str(self.forbidden),
                "--contract-manifest",
                str(self.contract),
                "--availability-evidence",
                str(self.evidence),
                "--preparer",
                str(self.preparer),
                "--output-root",
                str(output),
            ]

        def wait_for(predicate, label: str) -> None:
            deadline = time.monotonic() + 10
            while not predicate():
                if time.monotonic() >= deadline:
                    self.fail(label)
                time.sleep(0.01)

        environment = {
            **os.environ,
            "FAKE_PREPARER_ENTERED": str(entered),
            "FAKE_PREPARER_RELEASE": str(release),
            "FAKE_INVOCATION_LOCK_ATTEMPT": str(lock_attempts),
        }
        first = subprocess.Popen(
            command(self.output), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment
        )
        second: subprocess.Popen[bytes] | None = None
        try:
            wait_for(entered.exists, "fresh_process_preparer_enter_timeout")
            second = subprocess.Popen(
                command(self.output),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            wait_for(
                lambda: lock_attempts.exists() and lock_attempts.read_bytes() == b"1\n1\n",
                "fresh_process_second_lock_attempt_timeout",
            )
            self.assertEqual(counter.read_bytes(), b"1\n")
            self.assertIsNone(second.poll())
            release.write_bytes(b"release\n")
            first_stdout, first_stderr = first.communicate(timeout=10)
            second_stdout, second_stderr = second.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            first.kill()
            if second is not None:
                second.kill()
            first.communicate()
            if second is not None:
                second.communicate()
            self.fail("fresh_process_descriptor_flock_timeout")
        finally:
            if first.poll() is None:
                first.kill()
                first.communicate()
            if second is not None and second.poll() is None:
                second.kill()
                second.communicate()
        self.assertIsNotNone(second)
        self.assertEqual(first.returncode, 0, first_stderr.decode())
        self.assertEqual(second.returncode, 0, second_stderr.decode())
        first_receipt = json.loads(first_stdout)
        second_receipt = json.loads(second_stdout)
        self.assertEqual(first_receipt["preparer_result"], second_receipt["preparer_result"])
        self.assertEqual(
            first_receipt["invocation_descriptor_sha256"],
            second_receipt["invocation_descriptor_sha256"],
        )
        self.assertEqual(counter.read_bytes(), b"1\n")

        retry_output = self.root / "retry-output"
        failed = subprocess.run(
            command(retry_output),
            capture_output=True,
            env={**os.environ, "FAKE_PREPARER_MODE": "fail"},
            check=False,
            timeout=10,
        )
        self.assertNotEqual(failed.returncode, 0)
        retried = subprocess.run(
            command(retry_output), capture_output=True, check=False, timeout=10
        )
        self.assertEqual(retried.returncode, 0, retried.stderr.decode())
        self.assertEqual(counter.read_bytes(), b"1\n1\n")

    def _args(self) -> list[str]:
        return [
            "--acquirer",
            str(self.acquirer),
            "--source-root",
            str(self.source),
            "--source-report",
            str(self.report),
            "--forbidden-root",
            str(self.forbidden),
            "--contract-manifest",
            str(self.contract),
            "--availability-evidence",
            str(self.evidence),
            "--preparer",
            str(self.preparer),
            "--output-root",
            str(self.output),
        ]

    @contextlib.contextmanager
    def _approved_hashes(self):
        with (
            mock.patch.object(wrapper, "ACQUIRER_SHA256", hashlib.sha256(b"acquirer").hexdigest()),
            mock.patch.object(
                wrapper, "CONTRACT_SHA256", hashlib.sha256(self.contract.read_bytes()).hexdigest()
            ),
            mock.patch.object(wrapper, "EVIDENCE_SHA256", hashlib.sha256(b"evidence").hexdigest()),
            mock.patch.object(wrapper, "PREPARER_SHA256", hashlib.sha256(b"preparer").hexdigest()),
            mock.patch.object(wrapper.fcntl, "ioctl", side_effect=self._ficlone),
        ):
            yield

    def _run(self, run_side_effect):
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=run_side_effect) as run,
            contextlib.redirect_stdout(stdout := io.StringIO()),
        ):
            code = wrapper.main(self._args())
        return code, stdout.getvalue(), run

    def test_materialized_inputs_are_real_subprocess_compatible_and_nofollow(self) -> None:
        pinned = {
            "acquirer": wrapper._open_pinned_file(self.acquirer, "acquirer"),
            "contract_manifest": wrapper._open_pinned_file(self.contract, "contract_manifest"),
            "availability_evidence": wrapper._open_pinned_file(
                self.evidence, "availability_evidence"
            ),
            "preparer": wrapper._open_pinned_file(self.preparer, "preparer"),
        }
        try:
            inputs = wrapper._materialize_invocation_inputs(
                wrapper._sidecars(self.output)["inputs"], pinned
            )
            child = self._file(
                "safe-child.py",
                (
                    b"import pathlib, sys\n"
                    b"for value in sys.argv[1:]:\n"
                    b"    if any(component.is_symlink() for component in pathlib.Path(value).parents):\n"
                    b"        raise SystemExit(17)\n"
                    + f"if pathlib.Path(sys.argv[1]).read_bytes() != {self.contract.read_bytes()!r}:\n".encode()
                    + b"    raise SystemExit(18)\n"
                    + f"if pathlib.Path(sys.argv[2]).read_bytes() != {self.evidence.read_bytes()!r}:\n".encode()
                    + b"    raise SystemExit(19)\n"
                ),
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(child),
                    str(inputs["contract_manifest"]),
                    str(inputs["availability_evidence"]),
                ],
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(inputs["acquirer"].read_bytes(), self.acquirer.read_bytes())
            self.assertEqual(inputs["preparer"].read_bytes(), self.preparer.read_bytes())
        finally:
            for descriptor, _, _ in pinned.values():
                os.close(descriptor)

    def test_materialized_input_link_prefix_recovers_exactly(self) -> None:
        pinned = {
            label: wrapper._open_pinned_file(path, label)
            for label, path in (
                ("acquirer", self.acquirer),
                ("contract_manifest", self.contract),
                ("availability_evidence", self.evidence),
                ("preparer", self.preparer),
            )
        }
        root = wrapper._sidecars(self.output)["inputs"]
        original_unlink = wrapper.os.unlink
        try:

            def crash_after_link(path, *args, **kwargs):
                if Path(path).name == ".acquirer.py.stage":
                    raise OSError("crash")
                return original_unlink(path, *args, **kwargs)

            with (
                self.assertRaisesRegex(wrapper.PreparationError, "publish_failed"),
                mock.patch.object(wrapper.os, "unlink", side_effect=crash_after_link),
            ):
                wrapper._materialize_invocation_inputs(root, pinned)
            stage = root / ".acquirer.py.stage"
            target = root / "acquirer.py"
            self.assertEqual(stage.stat().st_ino, target.stat().st_ino)
            self.assertEqual(stage.stat().st_nlink, 2)
            inputs = wrapper._materialize_invocation_inputs(root, pinned)
            self.assertEqual(inputs["acquirer"].read_bytes(), self.acquirer.read_bytes())
            self.assertFalse(stage.exists())
        finally:
            for descriptor, _, _ in pinned.values():
                os.close(descriptor)

    def test_midflight_replacements_use_pinned_scripts_and_materialized_inputs(self) -> None:
        original = {
            "acquirer": self.acquirer.read_bytes(),
            "preparer": self.preparer.read_bytes(),
            "contract": self.contract.read_bytes(),
            "evidence": self.evidence.read_bytes(),
        }
        markers = {
            "acquirer": b"REPLACED_acquirer",
            "preparer": b"REPLACED_preparer",
            "contract_manifest": b"REPLACED_contract",
            "availability_evidence": b"REPLACED_evidence",
        }

        def replace(path: Path, payload: bytes) -> None:
            replacement = path.with_name(f".{path.name}.replacement")
            replacement.write_bytes(payload)
            os.replace(replacement, path)

        paths = {
            "acquirer": self.acquirer,
            "preparer": self.preparer,
            "contract_manifest": self.contract,
            "availability_evidence": self.evidence,
        }
        originals = {
            "acquirer": original["acquirer"],
            "preparer": original["preparer"],
            "contract_manifest": original["contract"],
            "availability_evidence": original["evidence"],
        }
        for label in ("acquirer", "contract_manifest", "availability_evidence", "preparer"):
            with self.subTest(label=label):
                calls: list[list[str]] = []

                def fake_run(argv, _calls=calls, _label=label, **kwargs):
                    _calls.append(argv)
                    verifier = "--verify-eligible" in argv
                    if _label == "preparer" and verifier:
                        self.assertEqual(Path(argv[1]).read_bytes(), original["acquirer"])
                        self.assertEqual(Path(argv[3]).read_bytes(), original["contract"])
                        self.assertEqual(Path(argv[5]).read_bytes(), original["evidence"])
                        return subprocess.CompletedProcess(argv, 0, b"", b"")
                    if verifier:
                        self.assertEqual(Path(argv[1]).read_bytes(), original["acquirer"])
                        self.assertEqual(Path(argv[3]).read_bytes(), original["contract"])
                        self.assertEqual(Path(argv[5]).read_bytes(), original["evidence"])
                    else:
                        self.assertEqual(Path(argv[1]).read_bytes(), original["preparer"])
                        self.assertEqual(Path(argv[7]).read_bytes(), original["contract"])
                    replace(paths[_label], markers[_label])
                    return subprocess.CompletedProcess(argv, 0, b"", b"")

                try:
                    with self.assertRaisesRegex(
                        wrapper.PreparationError, f"{label}_changed_during_read"
                    ):
                        self._run(fake_run)
                    self.assertEqual(len(calls), 2 if label == "preparer" else 1)
                    self.assertEqual(paths[label].read_bytes(), markers[label])
                finally:
                    for name, path in paths.items():
                        path.write_bytes(originals[name])

    def test_verifier_precedes_preparer_with_exact_derived_argv_and_receipt_binding(self) -> None:
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(kwargs["capture_output"], True)
            self.assertEqual(kwargs["check"], False)
            self.assertEqual(len(kwargs["pass_fds"]), 4)
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        code, stdout, run = self._run(fake_run)
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        verifier, preparer = calls
        self.assertEqual(verifier[0], sys.executable)
        self.assertRegex(verifier[1], r"^/proc/self/fd/\d+$")
        self.assertEqual(verifier[2], "--contract-manifest")
        self.assertEqual(
            verifier[3],
            str(
                self.output.parent
                / ".output.alpha_max_phase_preparation.invocation-inputs"
                / "contract_manifest.json"
            ),
        )
        self.assertEqual(verifier[4], "--availability-evidence")
        self.assertEqual(
            verifier[5],
            str(
                self.output.parent
                / ".output.alpha_max_phase_preparation.invocation-inputs"
                / "availability_evidence.json"
            ),
        )
        self.assertEqual(
            verifier[6:-2],
            [
                "--output-root",
                str(self.source),
                "--report-dir",
                str(self.report),
                "--forbidden-root",
                str(self.forbidden),
                "--verify-eligible",
            ],
        )
        self.assertEqual(verifier[-2], "--verifier-code-fd")
        self.assertTrue(verifier[-1].isdigit())
        self.assertEqual(preparer[0], sys.executable)
        self.assertRegex(preparer[1], r"^/proc/self/fd/\d+$")
        self.assertEqual(
            preparer[2:7],
            [
                "--raw-root",
                str(
                    self.output.parent
                    / ".output.alpha_max_phase_preparation.source-snapshot"
                    / "market_ohlcv_1s"
                ),
                "--feature-root",
                str(
                    self.output.parent
                    / ".output.alpha_max_phase_preparation.source-snapshot"
                    / "feature_points"
                ),
                "--contract-manifest",
            ],
        )
        self.assertEqual(
            preparer[7],
            str(
                self.output.parent
                / ".output.alpha_max_phase_preparation.invocation-inputs"
                / "contract_manifest.json"
            ),
        )
        self.assertEqual(preparer[8:], ["--output-root", str(self.output)])
        receipt = json.loads(stdout)
        self.assertEqual(
            set(receipt),
            {
                "schema",
                "invocation_descriptor_sha256",
                "source_eligibility_snapshot",
                "verifier_argv_sha256",
                "preparer_argv_sha256",
                "preparer_result",
                "output_root_identity",
                "source_snapshot_manifest_sha256",
                "source_snapshot_identity",
                "output_manifest_sha256",
            },
        )
        self.assertEqual(
            receipt["schema"], "alpha_max_phase_preparation_eligible_source_receipt.v2"
        )
        self.assertEqual(
            receipt["preparer_result"],
            {
                "file_count": 20,
                "output_root": str(self.output),
                "preparation_manifest_sha256": hashlib.sha256(
                    (self.output / "preparation_manifest.json").read_bytes()
                ).hexdigest(),
            },
        )
        descriptor = json.loads(
            (self.output.parent / ".output.alpha_max_phase_preparation.invocation.json").read_text()
        )
        self.assertEqual(
            receipt["source_eligibility_snapshot"], descriptor["source_eligibility_snapshot"]
        )
        self.assertEqual(
            descriptor["frozen_sha256"]["wrapper"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            receipt["verifier_argv_sha256"], wrapper.argv_sha256(descriptor["verifier_argv"])
        )
        self.assertEqual(
            receipt["invocation_descriptor_sha256"],
            wrapper.sha256(
                (
                    self.output.parent / ".output.alpha_max_phase_preparation.invocation.json"
                ).read_bytes()
            ),
        )
        self.assertEqual(
            receipt["preparer_argv_sha256"], wrapper.argv_sha256(descriptor["preparer_argv"])
        )
        self.assertEqual(
            receipt["output_root_identity"],
            {"st_dev": self.output.stat().st_dev, "st_ino": self.output.stat().st_ino},
        )
        snapshot_root = wrapper._sidecars(self.output)["snapshot"]
        self.assertEqual(
            receipt["source_snapshot_identity"],
            {"st_dev": snapshot_root.stat().st_dev, "st_ino": snapshot_root.stat().st_ino},
        )
        self.assertEqual(
            receipt["source_snapshot_manifest_sha256"],
            hashlib.sha256((snapshot_root / "snapshot-manifest.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            receipt["output_manifest_sha256"],
            hashlib.sha256((self.output / "preparation_manifest.json").read_bytes()).hexdigest(),
        )

    def test_tampered_snapshot_leaf_prevents_receipt_publication(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            self._publish_output()
            snapshot_leaf = (
                wrapper._sidecars(self.output)["snapshot"]
                / "market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet"
            )
            self._replace_read_only_leaf(snapshot_leaf, b"adversarial replacement")
            return subprocess.CompletedProcess(argv, 0, b"{}", b"")

        stdout = io.StringIO()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(stdout),
            self.assertRaisesRegex(wrapper.PreparationError, "^snapshot_parquet_mismatch$"),
        ):
            wrapper.main(self._args())

        self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
        self.assertEqual(stdout.getvalue(), "")
        self._remove_read_only_tree(self.output)
        self._remove_read_only_tree(wrapper._sidecars(self.output)["snapshot"])

    def test_successful_preparer_root_authentication_failures_leave_no_receipt(self) -> None:
        original_open = wrapper.os.open
        original_fstat = wrapper.os.fstat
        original_close = wrapper.os.close

        def assert_root_failure(label: str, root: Path) -> None:
            with self.subTest(label=label):
                raced_descriptor: int | None = None
                closed: list[int] = []
                preparer_ran = False

                def fake_run(argv, **kwargs):
                    nonlocal preparer_ran
                    if "--verify-eligible" in argv:
                        return subprocess.CompletedProcess(argv, 0, b"verified", b"")
                    preparer_ran = True
                    if label != "missing_output":
                        return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")
                    return subprocess.CompletedProcess(argv, 0, b"{}", b"")

                def record_open(path, *args, **kwargs):
                    nonlocal raced_descriptor
                    descriptor = original_open(path, *args, **kwargs)
                    if preparer_ran and label.endswith("race") and Path(path) == root:
                        raced_descriptor = descriptor
                    return descriptor

                def fail_raced_fstat(descriptor):
                    if descriptor == raced_descriptor:
                        raise OSError("authentication root race")
                    return original_fstat(descriptor)

                def record_close(descriptor):
                    closed.append(descriptor)
                    return original_close(descriptor)

                stdout = io.StringIO()
                with (
                    self._approved_hashes(),
                    mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
                    mock.patch.object(wrapper.os, "open", side_effect=record_open),
                    mock.patch.object(wrapper.os, "fstat", side_effect=fail_raced_fstat),
                    mock.patch.object(wrapper.os, "close", side_effect=record_close),
                    contextlib.redirect_stdout(stdout),
                    self.assertRaisesRegex(
                        wrapper.PreparationError,
                        (
                            "^output_root_namespace_unavailable$"
                            if label != "snapshot_race"
                            else "^snapshot_root_namespace_unavailable$"
                        ),
                    ),
                ):
                    wrapper.main(self._args())

                self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
                self.assertEqual(stdout.getvalue(), "")
                if label.endswith("race"):
                    self.assertIsNotNone(raced_descriptor)
                    self.assertIn(raced_descriptor, closed)
                self._remove_read_only_tree(self.output)
                self._remove_read_only_tree(wrapper._sidecars(self.output)["snapshot"])

        for label, root in (
            ("missing_output", self.output),
            ("output_race", self.output),
            ("snapshot_race", wrapper._sidecars(self.output)["snapshot"]),
        ):
            assert_root_failure(label, root)

    def test_output_root_replacement_before_receipt_publication_leaves_no_receipt(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        original_identity = wrapper._assert_directory_identity
        replaced = False

        def replace_before_receipt(path, expected, label):
            nonlocal replaced
            if label == "output_root" and not replaced:
                replaced = True
                parked = self.root / "parked-output"
                self.output.rename(parked)
                self.output.mkdir()
                (self.output / "adversarial.txt").write_bytes(b"replacement")
            return original_identity(path, expected, label)

        stdout = io.StringIO()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                wrapper, "_assert_directory_identity", side_effect=replace_before_receipt
            ),
            contextlib.redirect_stdout(stdout),
            self.assertRaisesRegex(wrapper.PreparationError, "^output_root_replaced$"),
        ):
            wrapper.main(self._args())

        self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
        self.assertEqual(stdout.getvalue(), "")
        self._remove_read_only_tree(self.root / "parked-output")
        self._remove_read_only_tree(wrapper._sidecars(self.output)["snapshot"])

    def test_public_generation_rewalk_rejects_same_root_replacements_without_publication(
        self,
    ) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        def replace_leaf(path: Path) -> None:
            self._replace_read_only_leaf(path, b"replacement")

        mutations = {
            "manifest": lambda: replace_leaf(self.output / "preparation_manifest.json"),
            "output": lambda: replace_leaf(
                self.output / "train/raw/market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet"
            ),
            "snapshot": lambda: replace_leaf(
                wrapper._sidecars(self.output)["snapshot"]
                / "market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet"
            ),
        }
        original_rewalk = wrapper._assert_public_generation_unchanged
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                try:

                    def replace_before_rewalk(*args, _mutate=mutate, **kwargs):
                        _mutate()
                        return original_rewalk(*args, **kwargs)

                    stdout = io.StringIO()
                    with (
                        self._approved_hashes(),
                        mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
                        mock.patch.object(
                            wrapper,
                            "_assert_public_generation_unchanged",
                            side_effect=replace_before_rewalk,
                        ),
                        contextlib.redirect_stdout(stdout),
                        self.assertRaisesRegex(
                            wrapper.PreparationError,
                            "^(output|snapshot)_generation_(changed_during_read|mismatch)$",
                        ),
                    ):
                        wrapper.main(self._args())
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
                finally:
                    self._remove_read_only_tree(self.output)
                    self._remove_read_only_tree(wrapper._sidecars(self.output)["snapshot"])

    def test_public_generation_rewalk_rejects_unlisted_output_entry_without_publication(
        self,
    ) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        original_inventory = wrapper._assert_generation_inventory
        inserted = False

        def insert_before_inventory(root_fd, files, label):
            nonlocal inserted
            if label == "output" and not inserted:
                inserted = True
                (self.output / "unlisted").mkdir()
                (self.output / "unlisted" / "entry").write_bytes(b"race")
            return original_inventory(root_fd, files, label)

        stdout = io.StringIO()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                wrapper, "_assert_generation_inventory", side_effect=insert_before_inventory
            ),
            contextlib.redirect_stdout(stdout),
            self.assertRaisesRegex(
                wrapper.PreparationError, "^output_generation_inventory_mismatch$"
            ),
        ):
            wrapper.main(self._args())

        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())

    def test_public_generation_rewalk_normalizes_root_fstat_failures(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        original_rewalk = wrapper._assert_public_generation_unchanged
        original_fstat = wrapper.os.fstat
        enabled = False

        def enable_rewalk(*args, **kwargs):
            nonlocal enabled
            enabled = True
            return original_rewalk(*args, **kwargs)

        def fail_final_fstat(descriptor):
            if enabled:
                raise OSError("root fstat race")
            return original_fstat(descriptor)

        stdout = io.StringIO()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                wrapper, "_assert_public_generation_unchanged", side_effect=enable_rewalk
            ),
            mock.patch.object(wrapper.os, "fstat", side_effect=fail_final_fstat),
            contextlib.redirect_stdout(stdout),
            self.assertRaisesRegex(wrapper.PreparationError, "^output_root_namespace_unavailable$"),
        ):
            wrapper.main(self._args())

        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())

    def test_public_generation_rewalk_normalizes_leaf_reopen_and_stability_failures(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"verified", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        original_rewalk = wrapper._assert_public_generation_unchanged
        original_open = wrapper._open_generation_file
        original_stable = wrapper._assert_stable_file
        for name, error in (
            ("reopen", "output_generation_namespace_unavailable"),
            ("stability", "output_generation_namespace_unavailable"),
        ):
            with self.subTest(name=name):

                def fail_reopen(root_fd, relative, digest, size, label):
                    if label == "output_generation":
                        raise OSError("leaf reopen race")
                    return original_open(root_fd, relative, digest, size, label)

                def fail_stability(descriptor, before, label):
                    if label == "output_generation":
                        raise OSError("leaf stability race")
                    return original_stable(descriptor, before, label)

                stdout = io.StringIO()
                with (
                    self._approved_hashes(),
                    mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
                    mock.patch.object(
                        wrapper,
                        "_assert_public_generation_unchanged",
                        side_effect=original_rewalk,
                    ),
                    (
                        mock.patch.object(wrapper, "_open_generation_file", side_effect=fail_reopen)
                        if name == "reopen"
                        else mock.patch.object(
                            wrapper, "_assert_stable_file", side_effect=fail_stability
                        )
                    ),
                    contextlib.redirect_stdout(stdout),
                    self.assertRaisesRegex(wrapper.PreparationError, f"^{error}$"),
                ):
                    wrapper.main(self._args())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
                self._remove_read_only_tree(self.output)
                self._remove_read_only_tree(wrapper._sidecars(self.output)["snapshot"])

    def test_pin_failure_prevents_all_subprocesses(self) -> None:
        with (
            self._approved_hashes(),
            mock.patch.object(
                wrapper, "CONTRACT_SHA256", hashlib.sha256(b"wrong-contract").hexdigest()
            ),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "contract_sha256_not_approved"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_pin_validation_failure_closes_all_pinned_descriptors(self) -> None:
        original_open = wrapper._open_pinned_file
        pinned: list[int] = []

        def track_open(path: Path, label: str):
            value = original_open(path, label)
            pinned.append(value[0])
            return value

        with (
            self._approved_hashes(),
            mock.patch.object(
                wrapper, "CONTRACT_SHA256", hashlib.sha256(b"wrong-contract").hexdigest()
            ),
            mock.patch.object(wrapper, "_open_pinned_file", side_effect=track_open),
            self.assertRaisesRegex(wrapper.PreparationError, "contract_sha256_not_approved"),
        ):
            wrapper.main(self._args())
        self.assertEqual(len(pinned), 4)
        for descriptor in pinned:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_tampered_acquirer_prevents_all_subprocesses(self) -> None:
        self.acquirer.write_bytes(b"tampered")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "acquirer_sha256_not_approved"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_tamper_after_verification_prevents_preparer(self) -> None:
        def fake_run(argv, **kwargs):
            self.report_artifact.write_bytes(b"tampered")
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "source_manifest_artifact_hash_mismatch"
            ),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 1)

    def test_source_artifact_tamper_prevents_verifier(self) -> None:
        self.source_artifact.write_bytes(b"tampered")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "source_manifest_artifact_hash_mismatch"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_report_artifact_tamper_prevents_verifier(self) -> None:
        self.report_artifact.write_bytes(b"tampered")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "source_manifest_artifact_hash_mismatch"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_manifest_traversal_and_duplicate_entries_prevent_verifier(self) -> None:
        valid = {
            "path": "output/market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet",
            "sha256": hashlib.sha256(self.source_artifact.read_bytes()).hexdigest(),
        }
        for name, artifacts, error in (
            (
                "traversal",
                [{"path": "output/../outside", "sha256": valid["sha256"]}],
                "source_manifest_artifact_path_invalid",
            ),
            ("duplicate", [valid, valid], "source_manifest_artifact_duplicate"),
        ):
            with self.subTest(name=name):
                self._write_manifest(artifacts)
                with (
                    self._approved_hashes(),
                    mock.patch.object(wrapper.subprocess, "run") as run,
                    self.assertRaisesRegex(wrapper.PreparationError, error),
                ):
                    wrapper.main(self._args())
                run.assert_not_called()
        self._write_manifest()

    def test_mutation_during_preparer_emits_no_receipt(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            self._publish_output()
            self.source_artifact.write_bytes(b"tampered")
            return subprocess.CompletedProcess(argv, 0, b'{"phase":"prepared"}', b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            contextlib.redirect_stdout(stdout := io.StringIO()),
            self.assertRaisesRegex(
                wrapper.PreparationError, "source_manifest_artifact_hash_mismatch"
            ),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 2)
        self.assertTrue(self.output.is_dir())
        self.assertFalse(wrapper._sidecars(self.output)["receipt"].exists())
        self.assertEqual(stdout.getvalue(), "")

    def test_missing_or_failed_verifier_prevents_preparer(self) -> None:
        (self.report / "source_eligible_receipt.json").unlink()
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "source_eligible_receipt"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

        self._file("report/source_eligible_receipt.json", b"receipt")
        with (
            self._approved_hashes(),
            mock.patch.object(
                wrapper.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 9, b"", b"failed"),
            ) as run,
            self.assertRaisesRegex(wrapper.PreparationError, "eligible_source_verification_failed"),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 1)

    def test_preparer_is_called_once_when_it_fails(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return subprocess.CompletedProcess(argv, 3, b"", b"failed")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            self.assertRaisesRegex(wrapper.PreparationError, "phase_preparer_failed"),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 2)

    def test_tampered_preparer_prevents_all_subprocesses_with_frozen_pins(self) -> None:
        self.preparer.write_bytes(b"tampered-preparer")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(wrapper.PreparationError, "preparer_sha256_not_approved"),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_tampered_availability_evidence_prevents_all_subprocesses_with_frozen_pins(
        self,
    ) -> None:
        self.evidence.write_bytes(b"tampered-evidence")
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "availability_evidence_sha256_not_approved"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_sidecar_input_overlap_prevents_all_subprocesses(self) -> None:
        self.output = self.root / "phase-output"
        self.acquirer = self._file(
            ".phase-output.alpha_max_phase_preparation.invocation.json", b"acquirer"
        )
        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "path_overlap:acquirer:sidecar_descriptor"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

    def test_unsafe_sidecar_parent_prevents_all_subprocesses(self) -> None:
        os.chmod(self.root, 0o755)
        try:
            with (
                self._approved_hashes(),
                mock.patch.object(wrapper.subprocess, "run") as run,
                self.assertRaisesRegex(wrapper.PreparationError, "invocation_lock_parent_unsafe"),
            ):
                wrapper.main(self._args())
        finally:
            os.chmod(self.root, 0o700)
        run.assert_not_called()

    def test_output_published_retry_recovers_exact_receipt_without_second_preparer(self) -> None:
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        original_replace = wrapper.os.replace

        def crash_before_receipt_final(stage, final):
            if final == self.output.parent / ".output.alpha_max_phase_preparation.handoff.json":
                raise OSError("crash")
            return original_replace(stage, final)

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            mock.patch.object(wrapper.os, "replace", side_effect=crash_before_receipt_final),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(wrapper.PreparationError, "handoff_receipt_publish_failed"),
        ):
            wrapper.main(self._args())

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            contextlib.redirect_stdout(stdout := io.StringIO()),
        ):
            self.assertEqual(wrapper.main(self._args()), 0)

        self.assertEqual(run.call_count, 1)
        self.assertIn("--verify-eligible", calls[-1])
        self.assertEqual(
            json.loads(stdout.getvalue()),
            json.loads(
                (
                    self.output.parent / ".output.alpha_max_phase_preparation.handoff.json"
                ).read_text()
            ),
        )

    def test_complete_receipt_retry_replays_without_preparer(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            contextlib.redirect_stdout(first := io.StringIO()),
        ):
            self.assertEqual(wrapper.main(self._args()), 0)
        self.assertEqual(run.call_count, 2)

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            contextlib.redirect_stdout(second := io.StringIO()),
        ):
            self.assertEqual(wrapper.main(self._args()), 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first.getvalue(), second.getvalue())

    def test_divergent_published_output_fails_without_preparer(self) -> None:
        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(wrapper.main(self._args()), 0)
        self._file("output/unlisted.json", b"foreign")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            self.assertRaisesRegex(wrapper.PreparationError, "preparation_manifest_mismatch"),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 1)

    def test_descriptor_stage_recovery_precedes_any_subprocess(self) -> None:
        original_replace = wrapper.os.replace

        def crash_descriptor_stage(source, destination):
            if Path(destination) == wrapper._sidecars(self.output)["descriptor"]:
                raise OSError("crash")
            return original_replace(source, destination)

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.os, "replace", side_effect=crash_descriptor_stage),
            mock.patch.object(wrapper.subprocess, "run") as run,
            self.assertRaisesRegex(
                wrapper.PreparationError, "invocation_descriptor_publish_failed"
            ),
        ):
            wrapper.main(self._args())
        run.assert_not_called()

        def fake_run(argv, **kwargs):
            if "--verify-eligible" in argv:
                return subprocess.CompletedProcess(argv, 0, b"", b"")
            return subprocess.CompletedProcess(argv, 0, self._publish_output(), b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.os, "replace", original_replace),
            mock.patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(wrapper.main(self._args()), 0)
        self.assertEqual(run.call_count, 2)

    def test_sidecar_prefix_resumes_only_exact_canonical_payload(self) -> None:
        stage = self.root / ".stage.json"
        final = self.root / ".final.json"
        payload = wrapper.canonical_bytes({"stage": "complete"})
        stage.write_bytes(payload[:7])
        wrapper._persist_immutable(stage, final, {"stage": "complete"}, "prefix")
        self.assertEqual(final.read_bytes(), payload)

    def test_output_rejects_swapped_phase_metadata(self) -> None:
        self._publish_output()
        manifest_path = self.output / "preparation_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["phase_intervals"][0], manifest["phase_intervals"][1] = (
            manifest["phase_intervals"][1],
            manifest["phase_intervals"][0],
        )
        manifest_path.write_bytes(wrapper.canonical_bytes(manifest))
        snapshot = {"entries": self._snapshot_entries()}
        with self.assertRaisesRegex(wrapper.PreparationError, "semantic_mismatch"):
            wrapper._authenticate_output(self.output, self.contract, snapshot, self.source)

    def test_output_rejects_out_of_phase_or_unavailable_owned_interval(self) -> None:
        self._publish_output()
        manifest_path = self.output / "preparation_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][0]["owned_end_utc"] = "2026-07-02T00:00:00Z"
        manifest_path.write_bytes(wrapper.canonical_bytes(manifest))
        snapshot = {"entries": self._snapshot_entries()}
        with self.assertRaisesRegex(wrapper.PreparationError, "semantic_mismatch"):
            wrapper._authenticate_output(self.output, self.contract, snapshot, self.source)

    def test_output_rejects_contract_availability_violation(self) -> None:
        self._publish_output()
        contract = json.loads(self.contract.read_bytes())
        contract["records"][3]["raw_availability_end_utc"] = "2024-01-01T00:00:00Z"
        self.contract.write_bytes(wrapper.canonical_bytes(contract))
        manifest_path = self.output / "preparation_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["contract_manifest_sha256"] = hashlib.sha256(
            self.contract.read_bytes()
        ).hexdigest()
        manifest_path.write_bytes(wrapper.canonical_bytes(manifest))
        snapshot = {"entries": self._snapshot_entries()}
        with self.assertRaisesRegex(wrapper.PreparationError, "contract_availability_invalid"):
            wrapper._authenticate_output(self.output, self.contract, snapshot, self.source)

    def test_metadata_only_six_phase_oracle_has_frozen_scale_and_ton_boundaries(self) -> None:
        self.assertEqual(
            wrapper.PHASE_INTERVALS,
            (
                ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
                ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
                ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
                ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
                ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
                (
                    "historical_exposed_evaluation",
                    "2025-09-07T00:00:00Z",
                    "2026-07-01T00:00:00Z",
                ),
            ),
        )
        symbols = (
            "ADAUSDT",
            "AVAXUSDT",
            "BNBUSDT",
            "BTCUSDT",
            "DOGEUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "TONUSDT",
            "TRXUSDT",
            "XRPUSDT",
        )
        records = [
            {
                "symbol": symbol,
                "raw_availability_start_utc": (
                    "2024-03-01T12:31:10Z" if symbol == "TONUSDT" else "2022-12-31T00:00:00Z"
                ),
                "raw_availability_end_utc": (
                    "2026-06-23T09:00:00Z" if symbol == "TONUSDT" else "2026-07-01T00:00:00Z"
                ),
                "feature_availability_start_utc": (
                    "2024-03-01T16:00:00Z" if symbol == "TONUSDT" else "2022-12-31T00:00:00Z"
                ),
                "feature_availability_end_utc": (
                    "2026-06-23T09:00:00Z" if symbol == "TONUSDT" else "2026-07-01T00:00:00Z"
                ),
            }
            for symbol in symbols
        ]
        source_paths: set[str] = set()
        for record in records:
            raw_start = wrapper._parse_utc(record["raw_availability_start_utc"], "oracle")
            raw_end = wrapper._parse_utc(record["raw_availability_end_utc"], "oracle")
            cursor = raw_start.replace(day=1)
            while cursor < raw_end:
                source_paths.add(
                    f"market_ohlcv_1s/binance/{record['symbol']}/{cursor:%Y-%m}.parquet"
                )
                cursor = (cursor.replace(day=28) + wrapper.timedelta(days=4)).replace(day=1)
            feature_start = wrapper._parse_utc(record["feature_availability_start_utc"], "oracle")
            feature_end = wrapper._parse_utc(record["feature_availability_end_utc"], "oracle")
            cursor = feature_start.replace(hour=0, minute=0, second=0, microsecond=0)
            while cursor < feature_end:
                source_paths.add(
                    f"feature_points/exchange=binance/symbol={record['symbol']}/"
                    f"date={cursor:%Y-%m-%d}/part-0.parquet"
                )
                cursor += wrapper.timedelta(days=1)
        snapshot = {
            "entries": [
                {"source_relative_path": path, "sha256": "0" * 64, "byte_count": 0}
                for path in sorted(source_paths)
            ]
        }
        layout, _ = wrapper._expected_output_layout(
            {
                "schema_version": "alpha_max_contract_manifest.v2",
                "exchange": "binance",
                "records": records,
            },
            snapshot,
        )
        self.assertEqual(len(source_paths), 12_762)
        self.assertEqual(len(layout), 12_792)
        self.assertEqual(
            {entry["phase_id"] for entry in layout},
            {
                "warmup",
                "train",
                "purge",
                "validation",
                "embargo",
                "historical_exposed_evaluation",
            },
        )
        self.assertEqual(
            {
                phase_id: sum(
                    entry["root_kind"] == "raw" and entry["phase_id"] == phase_id
                    for entry in layout
                )
                for phase_id, _, _ in (
                    ("warmup", "", ""),
                    ("train", "", ""),
                    ("purge", "", ""),
                    ("validation", "", ""),
                    ("embargo", "", ""),
                    ("historical_exposed_evaluation", "", ""),
                )
            },
            {
                "warmup": 117,
                "train": 168,
                "purge": 10,
                "validation": 30,
                "embargo": 20,
                "historical_exposed_evaluation": 100,
            },
        )
        self.assertEqual(
            {
                phase_id: sum(
                    entry["root_kind"] == "feature" and entry["phase_id"] == phase_id
                    for entry in layout
                )
                for phase_id, _, _ in (
                    ("warmup", "", ""),
                    ("train", "", ""),
                    ("purge", "", ""),
                    ("validation", "", ""),
                    ("embargo", "", ""),
                    ("historical_exposed_evaluation", "", ""),
                )
            },
            {
                "warmup": 3294,
                "train": 5110,
                "purge": 70,
                "validation": 840,
                "embargo": 70,
                "historical_exposed_evaluation": 2963,
            },
        )
        raw_paths = [
            entry["source_relative_path"] for entry in layout if entry["root_kind"] == "raw"
        ]
        self.assertEqual(len(raw_paths) - len(set(raw_paths)), 30)
        ton_features = [
            entry
            for entry in layout
            if entry["symbol"] == "TONUSDT" and entry["root_kind"] == "feature"
        ]
        self.assertEqual(
            min(entry["owned_start_utc"] for entry in ton_features), "2024-03-01T16:00:00Z"
        )
        self.assertEqual(
            max(entry["owned_end_utc"] for entry in ton_features), "2026-06-23T09:00:00Z"
        )
        self.assertTrue(
            all("/symbol=TONUSDT/date=" in entry["source_relative_path"] for entry in ton_features)
        )
        ton_raw = [
            entry
            for entry in layout
            if entry["symbol"] == "TONUSDT" and entry["root_kind"] == "raw"
        ]
        self.assertEqual(min(entry["owned_start_utc"] for entry in ton_raw), "2024-03-01T12:31:10Z")
        self.assertEqual(max(entry["owned_end_utc"] for entry in ton_raw), "2026-06-23T09:00:00Z")

    def test_snapshot_reflink_unsupported_uses_authenticated_copy_fallback(self) -> None:
        snapshot = self.root / "snapshot"
        with mock.patch.object(
            wrapper.fcntl, "ioctl", side_effect=OSError(wrapper.errno.EOPNOTSUPP, "unsupported")
        ):
            value, raw_root, _ = wrapper._build_snapshot(
                snapshot, self.source, self.report, "0" * 64
            )
        self.assertEqual(
            (raw_root / "binance/ADAUSDT/2024-01.parquet").read_bytes(),
            self.source_artifact.read_bytes(),
        )
        self.assertIn(
            hashlib.sha256(self.source_artifact.read_bytes()).hexdigest(),
            [entry["sha256"] for entry in value["entries"]],
        )

    def test_snapshot_reflink_unexpected_error_fails_closed(self) -> None:
        with (
            mock.patch.object(wrapper.fcntl, "ioctl", side_effect=OSError(wrapper.errno.EIO, "io")),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_reflink_failed"),
        ):
            wrapper._build_snapshot(self.root / "snapshot", self.source, self.report, "0" * 64)

    def _snapshot_entry(self) -> dict[str, object]:
        return {
            "source_relative_path": "market_ohlcv_1s/binance/ADAUSDT/2024-01.parquet",
            "sha256": hashlib.sha256(self.source_artifact.read_bytes()).hexdigest(),
            "byte_count": self.source_artifact.stat().st_size,
        }

    def test_snapshot_stage_open_failure_closes_source_descriptor(self) -> None:
        entry = self._snapshot_entry()
        target = self.root / "stage-open-failure" / entry["source_relative_path"]
        target.parent.mkdir(parents=True, mode=0o700)
        for directory in target.parents[:4]:
            os.chmod(directory, 0o700)
        stage = target.with_name(f".{target.name}.snapshot-stage")
        original_open_verified = wrapper._open_verified_file
        source_descriptors: list[int] = []
        closed: list[int] = []
        original_close = wrapper.os.close

        def record_open_verified(path, label):
            descriptor, item = original_open_verified(path, label)
            if path == self.source_artifact and label == "snapshot_source":
                source_descriptors.append(descriptor)
            return descriptor, item

        def fail_stage_open(path, flags, mode=0o600):
            if path == stage:
                raise OSError(wrapper.errno.EIO, "stage open failure")
            return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)

        def record_close(descriptor):
            closed.append(descriptor)
            return original_close(descriptor)

        with (
            mock.patch.object(wrapper, "_open_verified_file", side_effect=record_open_verified),
            mock.patch.object(wrapper, "_open_no_follow", side_effect=fail_stage_open),
            mock.patch.object(wrapper.os, "close", side_effect=record_close),
            self.assertRaisesRegex(wrapper.PreparationError, "^snapshot_reflink_failed$"),
        ):
            wrapper._snapshot_entry(self.source, target, entry)

        self.assertEqual(len(source_descriptors), 1)
        self.assertIn(source_descriptors[0], closed)
        self.assertFalse(stage.exists())
        self.assertFalse(target.exists())

    def test_snapshot_copy_resumes_every_exact_prefix(self) -> None:
        entry = self._snapshot_entry()
        payload = self.source_artifact.read_bytes()
        for prefix_size in sorted({0, 1, 2, len(payload) // 2, len(payload) - 1, len(payload)}):
            with self.subTest(prefix_size=prefix_size):
                snapshot = self.root / f"prefix-{prefix_size}"
                target = snapshot / entry["source_relative_path"]
                target.parent.mkdir(parents=True, mode=0o700)
                for directory in target.parents[:4]:
                    os.chmod(directory, 0o700)
                stage = target.with_name(f".{target.name}.snapshot-stage")
                stage.write_bytes(payload[:prefix_size])
                with mock.patch.object(
                    wrapper.fcntl,
                    "ioctl",
                    side_effect=OSError(wrapper.errno.EOPNOTSUPP, "unsupported"),
                ):
                    wrapper._snapshot_entry(self.source, target, entry)
                self.assertEqual(target.read_bytes(), payload)

    def test_snapshot_copy_resumes_large_authenticated_prefix_with_bounded_reads(self) -> None:
        prefix_size = (16 << 20) + 1
        artifact_size = prefix_size + (1 << 20) + 1
        source = self.source / "market_ohlcv_1s/binance/ADAUSDT/large-prefix.parquet"
        with source.open("wb") as handle:
            handle.truncate(artifact_size)
        entry = {
            "source_relative_path": source.relative_to(self.source).as_posix(),
            "sha256": wrapper.file_sha256(source, "large_prefix_source"),
            "byte_count": artifact_size,
        }
        self.assertGreater(prefix_size, 16 << 20)
        original_read = os.read

        snapshot = self.root / "large-prefix"
        target = snapshot / entry["source_relative_path"]
        target.parent.mkdir(parents=True, mode=0o700)
        for directory in target.parents[:4]:
            os.chmod(directory, 0o700)
        source_identity = (source.stat().st_dev, source.stat().st_ino)
        stage_identity: tuple[int, int] | None = None
        authenticated_stage_bytes_remaining = 0
        read_requests: list[int] = []
        read_bytes = {
            "reflink_prefix_source": 0,
            "resume_source": 0,
            "resume_prefix_stage": 0,
        }
        reading_reflink_prefix = False

        def tracked_read(descriptor, size):
            nonlocal authenticated_stage_bytes_remaining
            result = original_read(descriptor, size)
            read_requests.append(size)
            item = os.fstat(descriptor)
            if reading_reflink_prefix:
                read_bytes["reflink_prefix_source"] += len(result)
            elif (item.st_dev, item.st_ino) == source_identity:
                read_bytes["resume_source"] += len(result)
            elif (
                stage_identity is not None
                and (item.st_dev, item.st_ino) == stage_identity
                and authenticated_stage_bytes_remaining
            ):
                read_bytes["resume_prefix_stage"] += len(result)
                authenticated_stage_bytes_remaining -= len(result)
            return result

        def unsupported_after_exact_prefix(stage_fd, request, source_fd, *, corrupt=False):
            nonlocal authenticated_stage_bytes_remaining, reading_reflink_prefix, stage_identity
            self.assertEqual(request, wrapper.FICLONE)
            stage_item = os.fstat(stage_fd)
            stage_identity = (stage_item.st_dev, stage_item.st_ino)
            remaining = prefix_size
            reading_reflink_prefix = True
            try:
                while remaining:
                    block = wrapper.os.read(source_fd, min(1 << 20, remaining))
                    self.assertTrue(block)
                    os.write(stage_fd, block)
                    remaining -= len(block)
            finally:
                reading_reflink_prefix = False
            authenticated_stage_bytes_remaining = prefix_size
            if corrupt:
                os.pwrite(stage_fd, b"\x01", prefix_size - 1)
            raise OSError(wrapper.errno.EOPNOTSUPP, "unsupported")

        with (
            mock.patch.object(wrapper.fcntl, "ioctl", side_effect=unsupported_after_exact_prefix),
            mock.patch.object(wrapper.os, "read", side_effect=tracked_read),
        ):
            wrapper._snapshot_entry(self.source, target, entry)
        self.assertTrue(target.exists())
        self.assertEqual(wrapper.file_sha256(target, "large_prefix_target"), entry["sha256"])
        self.assertTrue(read_requests)
        self.assertLessEqual(max(read_requests), 1 << 20)
        self.assertEqual(read_bytes["reflink_prefix_source"], prefix_size)
        self.assertEqual(read_bytes["resume_prefix_stage"], prefix_size)
        self.assertEqual(read_bytes["resume_source"], artifact_size)

        corrupt_snapshot = self.root / "large-prefix-corrupt"
        corrupt_target = corrupt_snapshot / entry["source_relative_path"]
        corrupt_target.parent.mkdir(parents=True, mode=0o700)
        for directory in corrupt_target.parents[:4]:
            os.chmod(directory, 0o700)

        with (
            mock.patch.object(
                wrapper.fcntl,
                "ioctl",
                side_effect=lambda stage_fd, request, source_fd: unsupported_after_exact_prefix(
                    stage_fd, request, source_fd, corrupt=True
                ),
            ),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_stage_diverged"),
        ):
            wrapper._snapshot_entry(self.source, corrupt_target, entry)
        self.assertFalse(corrupt_target.exists())

    def test_snapshot_copy_crash_prefix_recovers_on_retry(self) -> None:
        entry = self._snapshot_entry()
        target = self.root / "crash-prefix" / entry["source_relative_path"]
        target.parent.mkdir(parents=True, mode=0o700)
        for directory in target.parents[:4]:
            os.chmod(directory, 0o700)
        original_write = wrapper.os.write
        writes = 0

        def short_then_crash(descriptor, data):
            nonlocal writes
            writes += 1
            if writes == 1:
                return original_write(descriptor, data[:2])
            raise OSError("crash")

        with (
            mock.patch.object(
                wrapper.fcntl,
                "ioctl",
                side_effect=OSError(wrapper.errno.EOPNOTSUPP, "unsupported"),
            ),
            mock.patch.object(wrapper.os, "write", side_effect=short_then_crash),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_copy_failed"),
        ):
            wrapper._snapshot_entry(self.source, target, entry)
        self.assertFalse(target.exists())
        with mock.patch.object(
            wrapper.fcntl,
            "ioctl",
            side_effect=OSError(wrapper.errno.EOPNOTSUPP, "unsupported"),
        ):
            wrapper._snapshot_entry(self.source, target, entry)
        self.assertEqual(target.read_bytes(), self.source_artifact.read_bytes())

    def test_snapshot_copy_rejects_corrupt_symlink_and_hardlink_stages(self) -> None:
        entry = self._snapshot_entry()
        for kind in ("corrupt", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                snapshot = self.root / f"unsafe-{kind}"
                target = snapshot / entry["source_relative_path"]
                target.parent.mkdir(parents=True, mode=0o700)
                for directory in target.parents[:4]:
                    os.chmod(directory, 0o700)
                stage = target.with_name(f".{target.name}.snapshot-stage")
                if kind == "corrupt":
                    stage.write_bytes(b"not-a-prefix")
                elif kind == "symlink":
                    stage.symlink_to(self.source_artifact)
                else:
                    os.link(self.source_artifact, stage)
                with self.assertRaisesRegex(
                    wrapper.PreparationError,
                    "snapshot_(stage_diverged|entry_stage_unsafe_file)",
                ):
                    wrapper._snapshot_entry(self.source, target, entry)
                self.assertFalse(target.exists())

    def test_snapshot_copy_rejects_source_mutation_during_streaming_copy(self) -> None:
        entry = self._snapshot_entry()
        target = self.root / "mutation" / entry["source_relative_path"]
        target.parent.mkdir(parents=True, mode=0o700)
        for directory in target.parents[:4]:
            os.chmod(directory, 0o700)
        original_write = wrapper.os.write
        mutated = False

        def mutate_then_write(descriptor, data):
            nonlocal mutated
            if not mutated:
                mutated = True
                self.source_artifact.write_bytes(b"tampered")
            return original_write(descriptor, data)

        with (
            mock.patch.object(
                wrapper.fcntl,
                "ioctl",
                side_effect=OSError(wrapper.errno.EOPNOTSUPP, "unsupported"),
            ),
            mock.patch.object(wrapper.os, "write", side_effect=mutate_then_write),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_source_changed_during_read"),
        ):
            wrapper._snapshot_entry(self.source, target, entry)
        self.assertFalse(target.exists())

    def test_snapshot_failure_never_invokes_preparer(self) -> None:
        def verifier_only(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        with (
            self._approved_hashes(),
            mock.patch.object(wrapper.fcntl, "ioctl", side_effect=OSError(wrapper.errno.EIO, "io")),
            mock.patch.object(wrapper.subprocess, "run", side_effect=verifier_only) as run,
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_reflink_failed"),
        ):
            wrapper.main(self._args())
        self.assertEqual(run.call_count, 1)

    def test_completed_snapshot_target_replays_without_ioctl(self) -> None:
        entry = self._snapshot_entry()
        target = self.root / "replay" / entry["source_relative_path"]
        target.parent.mkdir(parents=True, mode=0o700)
        for directory in target.parents[:4]:
            os.chmod(directory, 0o700)
        target.write_bytes(self.source_artifact.read_bytes())
        os.chmod(target, 0o444)
        with mock.patch.object(wrapper.fcntl, "ioctl") as ioctl:
            wrapper._snapshot_entry(self.source, target, entry)
        ioctl.assert_not_called()

    def test_snapshot_rejects_leaf_and_parent_symlink_before_copy(self) -> None:
        for kind in ("leaf", "parent"):
            with self.subTest(kind=kind):
                if kind == "leaf":
                    self.source_artifact.unlink()
                    self.source_artifact.symlink_to(self.report_artifact)
                else:
                    raw_root = self.source / "market_ohlcv_1s"
                    raw_root.rename(self.source / "raw-real")
                    raw_root.symlink_to(self.source / "raw-real", target_is_directory=True)
                with self.assertRaisesRegex(
                    wrapper.PreparationError, "source_manifest_artifact|snapshot_source"
                ):
                    wrapper._build_snapshot(
                        self.root / f"symlink-{kind}", self.source, self.report, "0" * 64
                    )
                if kind == "leaf":
                    self.source_artifact.unlink()
                    self.source_artifact.write_bytes(b"parquet")
                else:
                    raw_root.unlink()
                    (self.source / "raw-real").rename(raw_root)

    def test_snapshot_preseeded_symlink_never_touches_sentinel_or_preparer(self) -> None:
        sentinel = self.root / "sentinel"
        sentinel.mkdir(mode=0o700)
        payload = sentinel / "keep"
        payload.write_bytes(b"unchanged")
        sentinel_mode = sentinel.stat().st_mode
        for nested in (False, True):
            with self.subTest(nested=nested):
                snapshot = self.root / f"preseeded-{nested}"
                if nested:
                    snapshot.mkdir(mode=0o700)
                    (snapshot / "market_ohlcv_1s").symlink_to(sentinel, target_is_directory=True)
                else:
                    snapshot.symlink_to(sentinel, target_is_directory=True)
                with (
                    mock.patch.object(wrapper.subprocess, "run") as run,
                    self.assertRaisesRegex(
                        wrapper.PreparationError, "snapshot_(root|target_parent)"
                    ),
                ):
                    wrapper._build_snapshot(snapshot, self.source, self.report, "0" * 64)
                run.assert_not_called()
                self.assertEqual(payload.read_bytes(), b"unchanged")
                self.assertEqual(sentinel.stat().st_mode, sentinel_mode)

    def test_stable_identity_excludes_atime_but_rejects_content_change(self) -> None:
        before = self.source_artifact.stat()

        def stat_like(**changes: int | float) -> object:
            fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_atime",
            )
            return type(
                "StatLike",
                (),
                {field: changes.get(field, getattr(before, field)) for field in fields},
            )()

        atime_changed = stat_like(st_atime=before.st_atime + 1)
        self.assertEqual(
            wrapper._stable_file_identity(before), wrapper._stable_file_identity(atime_changed)
        )
        changed = stat_like(st_size=before.st_size + 1)
        self.assertNotEqual(
            wrapper._stable_file_identity(before), wrapper._stable_file_identity(changed)
        )

    def test_file_sha256_streams_without_regular_bytes(self) -> None:
        expected = hashlib.sha256(self.source_artifact.read_bytes()).hexdigest()
        with mock.patch.object(wrapper, "_verified_file_bytes", side_effect=AssertionError):
            self.assertEqual(wrapper.file_sha256(self.source_artifact, "source"), expected)

    def test_snapshot_copy_streams_large_prefix_without_metadata_reader(self) -> None:
        payload = b"p" * ((16 << 20) + 17)
        source = self._file("large-source.parquet", payload)
        stage = self.root / "large.snapshot-stage"
        stage.write_bytes(payload[: (16 << 20) + 3])
        entry = {
            "source_relative_path": "large-source.parquet",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
        with mock.patch.object(wrapper, "_verified_file_bytes", side_effect=AssertionError):
            wrapper._copy_snapshot_entry(source, stage, entry)
        self.assertEqual(stage.read_bytes(), payload)

    def test_snapshot_publication_never_overwrites_existing_target(self) -> None:
        payload = b"safe"
        stage = self.root / "stage"
        target = self.root / "target"
        stage.write_bytes(payload)
        target.write_bytes(b"foreign")
        entry = {
            "source_relative_path": "unused",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
        with self.assertRaisesRegex(wrapper.PreparationError, "target_already_exists"):
            wrapper._publish_snapshot_stage(stage, target, entry)
        self.assertEqual(target.read_bytes(), b"foreign")

    def test_materialized_input_namespace_replacement_is_rejected(self) -> None:
        pinned = {
            "acquirer": wrapper._open_pinned_file(self.acquirer, "acquirer"),
            "contract_manifest": wrapper._open_pinned_file(self.contract, "contract_manifest"),
            "availability_evidence": wrapper._open_pinned_file(
                self.evidence, "availability_evidence"
            ),
            "preparer": wrapper._open_pinned_file(self.preparer, "preparer"),
        }
        materialized = {}
        try:
            inputs = wrapper._materialize_invocation_inputs(self.root / "inputs-replace", pinned)
            materialized = wrapper._open_materialized_inputs(inputs, pinned)
            inputs["acquirer"].unlink()
            inputs["acquirer"].write_bytes(b"foreign")
            os.chmod(inputs["acquirer"], 0o444)
            with self.assertRaisesRegex(wrapper.PreparationError, "acquirer_changed_during_read"):
                wrapper._assert_materialized_inputs_unchanged(inputs, materialized)
        finally:
            for descriptor, _, _ in materialized.values():
                os.close(descriptor)
            for descriptor, _, _ in pinned.values():
                os.close(descriptor)

    def test_materialized_input_open_failure_closes_prior_descriptors(self) -> None:
        files = {
            "acquirer": self.acquirer,
            "contract_manifest": self.contract,
        }
        expected = {
            label: (
                -1,
                path.stat(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for label, path in files.items()
        }
        original_open = wrapper._open_pinned_file
        opened: list[int] = []

        def fail_second(path: Path, label: str):
            if label == "materialized_contract_manifest":
                raise wrapper.PreparationError("materialized_contract_manifest_missing")
            value = original_open(path, label)
            opened.append(value[0])
            return value

        with (
            mock.patch.object(wrapper, "_open_pinned_file", side_effect=fail_second),
            self.assertRaisesRegex(
                wrapper.PreparationError, "materialized_contract_manifest_missing"
            ),
        ):
            wrapper._open_materialized_inputs(files, expected)
        for descriptor in opened:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_snapshot_inventory_closes_scandir_duplicate_descriptors(self) -> None:
        snapshot = self.root / "snapshot-fd-ownership"
        nested = snapshot / "market_ohlcv_1s" / "binance"
        nested.mkdir(parents=True)
        files = {
            "snapshot-manifest.json": b"{}",
            ".complete.json": b"{}",
            "market_ohlcv_1s/binance/BTCUSDT.parquet": b"data",
        }
        for relative, payload in files.items():
            path = snapshot / relative
            path.write_bytes(payload)
            os.chmod(path, 0o444)
        for directory in (snapshot, snapshot / "market_ohlcv_1s", nested):
            os.chmod(directory, 0o700)
        entries = [
            {
                "source_relative_path": "market_ohlcv_1s/binance/BTCUSDT.parquet",
                "sha256": hashlib.sha256(b"data").hexdigest(),
                "byte_count": 4,
            }
        ]
        real_dup, real_close = os.dup, os.close
        duplicate_descriptors: set[int] = set()

        def tracked_dup(descriptor: int) -> int:
            duplicate = real_dup(descriptor)
            duplicate_descriptors.add(duplicate)
            return duplicate

        def tracked_close(descriptor: int) -> None:
            duplicate_descriptors.discard(descriptor)
            real_close(descriptor)

        with (
            mock.patch.object(wrapper.os, "dup", side_effect=tracked_dup),
            mock.patch.object(wrapper.os, "close", side_effect=tracked_close),
        ):
            wrapper._snapshot_inventory(snapshot, entries, finalize_modes=False)
        self.assertEqual(duplicate_descriptors, set())

        (snapshot / "foreign").write_bytes(b"foreign")
        os.chmod(snapshot / "foreign", 0o444)
        duplicate_descriptors.clear()
        with (
            mock.patch.object(wrapper.os, "dup", side_effect=tracked_dup),
            mock.patch.object(wrapper.os, "close", side_effect=tracked_close),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_"),
        ):
            wrapper._snapshot_inventory(snapshot, entries, finalize_modes=False)
        self.assertEqual(duplicate_descriptors, set())

    def test_snapshot_inventory_rejects_fifo_and_socket_without_opening_them(self) -> None:
        for kind in ("fifo", "socket"):
            with self.subTest(kind=kind):
                snapshot = self.root / f"snapshot-special-{kind}"
                nested = snapshot / "market_ohlcv_1s" / "binance"
                nested.mkdir(parents=True)
                payload = b"data"
                source = nested / "BTCUSDT.parquet"
                source.write_bytes(payload)
                for path in (
                    snapshot / "snapshot-manifest.json",
                    snapshot / ".complete.json",
                    source,
                ):
                    if path != source:
                        path.write_bytes(b"{}")
                    os.chmod(path, 0o444)
                for directory in (snapshot, snapshot / "market_ohlcv_1s", nested):
                    os.chmod(directory, 0o700)
                special = snapshot / kind
                listener = None
                if kind == "fifo":
                    os.mkfifo(special, 0o600)
                else:
                    listener = socket.socket(socket.AF_UNIX)
                    listener.bind(os.fspath(special))
                actual_open = os.open
                opened: list[str] = []

                def guarded_open(
                    path,
                    *args,
                    _kind=kind,
                    _opened=opened,
                    _actual_open=actual_open,
                    **kwargs,
                ):
                    if path == _kind:
                        self.fail(f"attempted to open {_kind}")
                    _opened.append(os.fspath(path))
                    return _actual_open(path, *args, **kwargs)

                try:
                    with (
                        mock.patch.object(wrapper.os, "open", side_effect=guarded_open),
                        self.assertRaisesRegex(
                            wrapper.PreparationError, "snapshot_inventory_diverged"
                        ),
                    ):
                        wrapper._snapshot_inventory(
                            snapshot,
                            [
                                {
                                    "source_relative_path": "market_ohlcv_1s/binance/BTCUSDT.parquet",
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                    "byte_count": len(payload),
                                }
                            ],
                            finalize_modes=False,
                        )
                finally:
                    if listener is not None:
                        listener.close()
                self.assertNotIn(kind, opened)

    def test_snapshot_inventory_never_opens_mocked_device_leaf(self) -> None:
        snapshot = self.root / "snapshot-mocked-special"
        snapshot.mkdir()
        os.chmod(snapshot, 0o700)
        observed = type(
            "Observed",
            (),
            {
                "st_uid": os.geteuid(),
                "st_mode": wrapper.stat.S_IFCHR | 0o600,
                "st_nlink": 1,
            },
        )()

        class Entry:
            name = "sentinel"

            def stat(self, *, follow_symlinks: bool):
                if follow_symlinks:
                    raise AssertionError("special leaf stat followed symlink")
                return observed

        class Scan:
            def __enter__(self):
                return iter((Entry(),))

            def __exit__(self, *_args):
                return False

        actual_open = os.open

        def guarded_open(path, *args, **kwargs):
            if path == "sentinel":
                self.fail("attempted to open mocked device")
            return actual_open(path, *args, **kwargs)

        with (
            mock.patch.object(wrapper.os, "scandir", return_value=Scan()),
            mock.patch.object(wrapper.os, "open", side_effect=guarded_open),
            self.assertRaisesRegex(wrapper.PreparationError, "snapshot_inventory_diverged"),
        ):
            wrapper._snapshot_inventory(snapshot, [], finalize_modes=False)


if __name__ == "__main__":
    unittest.main()
