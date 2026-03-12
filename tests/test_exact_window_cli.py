from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.cli import exact_window as exact_window_cli
from lumina_quant.cli import main as cli_main


class _FakeGuard:
    def __init__(self, *, log_path, soft_limit_bytes=None, hard_limit_bytes=None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples: list[tuple[str, dict[str, object]]] = []

    def sample(self, *, event: str, context=None):
        self.samples.append((event, dict(context or {})))
        self.log_path.write_text("sample\n", encoding="utf-8")
        return {"event": event}

    def checkpoint(self, event: str, context=None):
        self.samples.append((event, dict(context or {})))

    def finalize(self, *, status: str, error: str | None = None):
        return {
            "status": status,
            "error": error,
            "peak_rss_mib": 123.0,
            "budget_mib": 512.0,
            "soft_limit_mib": 307.2,
            "hard_limit_mib": 409.6,
            "rss_log_path": str(self.log_path),
        }


def test_exact_window_cli_generates_fail_analysis_and_memory_payloads(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_suite(**kwargs):
        captured.update(kwargs)
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "generated_at": "2026-03-09T00:00:00Z",
            "windows": {"actual_max_timestamp": "2026-03-07T10:00:00+00:00"},
            "execution_profile": {"requested_timeframes": ["1m"]},
            "eligible_symbols": ["BTC/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "evaluated_count": 1,
            "portfolio": {"weights": []},
        }
        (out_dir / exact_window_cli.SUMMARY_LATEST).write_text(json.dumps(summary), encoding="utf-8")
        (out_dir / exact_window_cli.DETAILS_LATEST).write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "c1",
                        "name": "stub-1m",
                        "family": "stub",
                        "strategy_class": "StubStrategy",
                        "strategy_timeframe": "1m",
                        "oos": {"trade_count": 1, "mdd": 0.1, "sharpe": -0.5},
                        "hurdle_fields": {
                            "train": {"pass": True},
                            "val": {"pass": True},
                            "oos": {"pass": False},
                        },
                        "hard_reject_reasons": {"oos_sharpe": -0.5},
                        "metadata": {},
                    }
                ]
            ),
            encoding="utf-8",
        )
        return {
            **summary,
        }

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _stub_suite)

    rc = exact_window_cli.main(["--output-dir", str(tmp_path), "--timeframes", "1m"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["fail_analysis_latest"].endswith("exact_window_fail_analysis_latest.json")
    assert payload["memory_evidence_latest"].endswith("exact_window_memory_evidence_latest.json")
    assert payload["summary_latest"] == str((tmp_path / exact_window_cli.SUMMARY_LATEST).resolve())
    assert payload["details_latest"] == str((tmp_path / exact_window_cli.DETAILS_LATEST).resolve())
    assert payload["eligible_symbols"] == ["BTC/USDT"]
    assert captured["allow_metals"] is False
    latest_pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest_pointer["summary_path"] == str((tmp_path / exact_window_cli.SUMMARY_LATEST).resolve())


def test_exact_window_cli_passes_allow_metals_flag(tmp_path: Path, monkeypatch, capsys):
    captured: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_suite(**kwargs):
        captured.update(kwargs)
        return {
            "eligible_symbols": ["XPT/USDT", "XPD/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "portfolio": {"weights": []},
        }

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _stub_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "write_fail_analysis_bundle",
        lambda **kwargs: {
            "json_latest": tmp_path / "exact_window_fail_analysis_latest.json",
        },
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_memory_evidence_bundle",
        lambda **kwargs: {
            "json_latest": tmp_path / "exact_window_memory_evidence_latest.json",
        },
    )

    rc = exact_window_cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--timeframes",
            "5m",
            "--symbols",
            "XPT/USDT",
            "XPD/USDT",
            "--allow-metals",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allow_metals"] is True
    assert captured["allow_metals"] is True
    assert captured["symbols"] == ["XPT/USDT", "XPD/USDT"]


def test_exact_window_cli_passes_custom_window_overrides(tmp_path: Path, monkeypatch, capsys):
    captured: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_suite(**kwargs):
        captured.update(kwargs)
        return {
            "eligible_symbols": ["BTC/USDT", "XAU/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "portfolio": {"weights": []},
        }

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _stub_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "write_fail_analysis_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_fail_analysis_latest.json"},
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_memory_evidence_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_memory_evidence_latest.json"},
    )

    rc = exact_window_cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--timeframes",
            "4h",
            "--train-start",
            "2026-01-07",
            "--val-start",
            "2026-02-01",
            "--oos-start",
            "2026-02-20",
            "--requested-oos-end",
            "2026-03-09",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert captured["train_start"] == "2026-01-07"
    assert captured["val_start"] == "2026-02-01"
    assert captured["oos_start"] == "2026-02-20"
    assert captured["requested_oos_end_exclusive"] == "2026-03-09"


def test_exact_window_cli_derives_adaptive_windows_for_metal_profile(tmp_path: Path, monkeypatch, capsys):
    captured: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_suite(**kwargs):
        captured.update(kwargs)
        return {
            "eligible_symbols": ["BTC/USDT", "XAU/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "portfolio": {"weights": []},
        }

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _stub_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "resolve_coverage_adaptive_windows",
        lambda **kwargs: {
            "profile": "metals",
            "train_start": "2026-01-01T00:00:00+00:00",
            "val_start": "2026-02-01T00:00:00+00:00",
            "oos_start": "2026-02-20T00:00:00+00:00",
            "requested_oos_end_exclusive": "2026-03-09T00:00:00+00:00",
            "common_start": "2026-01-01T00:00:00+00:00",
            "common_end": "2026-03-08T23:59:00+00:00",
            "total_days": 67,
            "allocation_days": {"train": 31, "val": 19, "oos": 17},
        },
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_fail_analysis_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_fail_analysis_latest.json"},
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_memory_evidence_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_memory_evidence_latest.json"},
    )

    rc = exact_window_cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--timeframes",
            "4h",
            "1d",
            "--symbols",
            "BTC/USDT",
            "XAU/USDT",
            "--allow-metals",
            "--window-profile",
            "metals",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert captured["train_start"] == "2026-01-01T00:00:00+00:00"
    assert captured["val_start"] == "2026-02-01T00:00:00+00:00"
    assert captured["oos_start"] == "2026-02-20T00:00:00+00:00"
    assert captured["requested_oos_end_exclusive"] == "2026-03-09T00:00:00+00:00"
    manifest = json.loads((tmp_path / next(tmp_path.glob("exact_window_*/manifest.json")).relative_to(tmp_path)).read_text(encoding="utf-8"))
    assert manifest["window_profile"] == "metals"
    assert manifest["adaptive_windows"]["profile"] == "metals"


def test_exact_window_cli_accepts_timeframe_specific_adaptive_profile(tmp_path: Path, monkeypatch, capsys):
    captured: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_suite(**kwargs):
        captured.update(kwargs)
        return {
            "eligible_symbols": ["XPT/USDT", "XPD/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "portfolio": {"weights": []},
        }

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _stub_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "resolve_coverage_adaptive_windows",
        lambda **kwargs: {
            "profile": "metals_4h",
            "train_start": "2026-01-30T00:00:00+00:00",
            "val_start": "2026-02-18T00:00:00+00:00",
            "oos_start": "2026-02-26T00:00:00+00:00",
            "requested_oos_end_exclusive": "2026-03-09T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_fail_analysis_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_fail_analysis_latest.json"},
    )
    monkeypatch.setattr(
        exact_window_cli,
        "write_memory_evidence_bundle",
        lambda **kwargs: {"json_latest": tmp_path / "exact_window_memory_evidence_latest.json"},
    )

    rc = exact_window_cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--timeframes",
            "4h",
            "--symbols",
            "XPT/USDT",
            "XPD/USDT",
            "--allow-metals",
            "--window-profile",
            "metals_4h",
        ]
    )
    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert captured["train_start"] == "2026-01-30T00:00:00+00:00"
    assert captured["val_start"] == "2026-02-18T00:00:00+00:00"
    assert captured["oos_start"] == "2026-02-26T00:00:00+00:00"
    assert captured["requested_oos_end_exclusive"] == "2026-03-09T00:00:00+00:00"


def test_exact_window_cli_supports_baseline_probe(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    called = {"suite": False}

    def _unexpected_suite(**kwargs):
        called["suite"] = True
        raise AssertionError("suite should not run during baseline probe")

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _unexpected_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "write_memory_evidence_bundle",
        lambda **kwargs: {
            "json_latest": tmp_path / "exact_window_memory_evidence_latest.json",
        },
    )

    rc = exact_window_cli.main(["--output-dir", str(tmp_path), "--emit-memory-baseline"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "baseline_probe"
    assert called["suite"] is False


def test_exact_window_cli_blocks_when_another_heavy_run_is_active(tmp_path: Path, monkeypatch, capsys):
    class _BusyLock:
        @staticmethod
        def acquire(*, lock_path, label="exact_window", metadata=None):
            _ = label, metadata
            raise exact_window_cli.HeavyRunActiveError(
                lock_path=lock_path,
                active_payload={"pid": 4321, "run_id": "busy-run", "batch_id": "1h"},
            )

    monkeypatch.setattr(exact_window_cli, "HeavyRunLock", _BusyLock)

    rc = exact_window_cli.main(["--output-dir", str(tmp_path), "--timeframes", "1h"])
    assert rc == 3

    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "blocked_active_run"
    assert payload["active_run"]["run_id"] == "busy-run"


def test_cli_main_dispatches_exact_window(monkeypatch):
    captured: dict[str, object] = {}

    def _stub(argv=None):
        captured["argv"] = list(argv or [])
        return 9

    monkeypatch.setattr(cli_main.exact_window, "main", _stub)
    rc = cli_main.main(["exact-window", "--emit-memory-baseline"])
    assert rc == 9
    assert captured["argv"] == ["--emit-memory-baseline"]


def test_exact_window_cli_skips_duplicate_signature_when_not_forced(tmp_path: Path, monkeypatch, capsys):
    score_config = tmp_path / "score_config.json"
    score_config.write_text("{}", encoding="utf-8")

    parser = exact_window_cli._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "--score-config", str(score_config), "--timeframes", "1m", "--symbols", "BTC/USDT"])
    resolved_windows, _adaptive = exact_window_cli._build_resolved_windows(args=args, symbols=["BTC/USDT"] )
    signature = exact_window_cli._candidate_run_signature(
        candidate_library_hash=exact_window_cli._candidate_library_hash(),
        batch_timeframes=exact_window_cli._resolve_batch_timeframes(args.timeframes),
        symbols=["BTC/USDT"],
        requested_timeframes=["1m"],
        resolved_windows=resolved_windows,
        score_config_path=str(score_config),
        chunk_days=14,
        window_profile="default",
        allow_metals=False,
    )

    registry = exact_window_cli._registry_path(tmp_path)
    previous_run = tmp_path / "exact_window_prev"
    previous_batch = previous_run / "1m"
    previous_batch.mkdir(parents=True, exist_ok=True)
    summary = previous_batch / exact_window_cli.SUMMARY_LATEST
    details = previous_batch / exact_window_cli.DETAILS_LATEST
    summary.write_text("{}", encoding="utf-8")
    details.write_text("[]", encoding="utf-8")

    exact_window_cli._append_signature_entry(
        registry,
        signature=signature,
        run_id="cached",
        status="completed",
        batch_id="1m",
        run_root=str(previous_run),
        batch_dir=str(previous_batch),
        manifest_path=str(previous_run / "manifest.json"),
        summary_path=str(summary),
        details_path=str(details),
        fail_analysis_path=None,
        memory_evidence_path=None,
    )

    called = {"suite": False}

    def _unexpected_suite(**kwargs):
        called["suite"] = True
        raise AssertionError("suite should be skipped")

    monkeypatch.setattr(exact_window_cli, "run_exact_window_suite", _unexpected_suite)
    monkeypatch.setattr(
        exact_window_cli,
        "RSSGuard",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("guard should not run")),
    )

    rc = exact_window_cli.main([
        "--output-dir",
        str(tmp_path),
        "--score-config",
        str(score_config),
        "--timeframes",
        "1m",
        "--symbols",
        "BTC/USDT",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped_duplicate"
    assert payload["summary_latest"] == str(summary.resolve())
    assert called["suite"] is False


def test_exact_window_cli_force_rerun_ignores_signature_cache(tmp_path: Path, monkeypatch, capsys):
    score_config = tmp_path / "score_config.json"
    score_config.write_text("{}", encoding="utf-8")

    parser = exact_window_cli._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "--score-config", str(score_config), "--timeframes", "1m", "--symbols", "BTC/USDT"])
    resolved_windows, _adaptive = exact_window_cli._build_resolved_windows(args=args, symbols=["BTC/USDT"] )
    signature = exact_window_cli._candidate_run_signature(
        candidate_library_hash=exact_window_cli._candidate_library_hash(),
        batch_timeframes=exact_window_cli._resolve_batch_timeframes(args.timeframes),
        symbols=["BTC/USDT"],
        requested_timeframes=["1m"],
        resolved_windows=resolved_windows,
        score_config_path=str(score_config),
        chunk_days=14,
        window_profile="default",
        allow_metals=False,
    )

    registry = exact_window_cli._registry_path(tmp_path)
    previous_run = tmp_path / "exact_window_prev"
    previous_batch = previous_run / "1m"
    previous_batch.mkdir(parents=True, exist_ok=True)
    summary = previous_batch / exact_window_cli.SUMMARY_LATEST
    summary.write_text("{}", encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")

    details = previous_batch / exact_window_cli.DETAILS_LATEST
    details.write_text("[]", encoding="utf-8")

    exact_window_cli._append_signature_entry(
        registry,
        signature=signature,
        run_id="cached",
        status="completed",
        batch_id="1m",
        run_root=str(previous_run),
        batch_dir=str(previous_batch),
        manifest_path=str(previous_run / "manifest.json"),
        summary_path=str(summary),
        details_path=str(details),
        fail_analysis_path=None,
        memory_evidence_path=None,
    )

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)
    monkeypatch.setattr(
        exact_window_cli,
        "run_exact_window_suite",
        lambda **kwargs: {"eligible_symbols": ["BTC/USDT"], "best_per_strategy": [], "promoted_count": 0, "portfolio": {"weights": []}},
    )
    monkeypatch.setattr(exact_window_cli, "write_fail_analysis_bundle", lambda **kwargs: {"json_latest": tmp_path / "exact_window_fail_analysis_latest.json"})
    monkeypatch.setattr(exact_window_cli, "write_memory_evidence_bundle", lambda **kwargs: {"json_latest": tmp_path / "exact_window_memory_evidence_latest.json"})

    rc = exact_window_cli.main([
        "--output-dir",
        str(tmp_path),
        "--score-config",
        str(score_config),
        "--timeframes",
        "1m",
        "--symbols",
        "BTC/USDT",
        "--force-rerun",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] != "skipped_duplicate"
    assert payload["run_signature"] == signature
