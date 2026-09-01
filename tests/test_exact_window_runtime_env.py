from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from lumina_quant.cli import exact_window as exact_window_cli
from lumina_quant.eval import exact_window_suite as exact_window_suite_module


class _FakeGuard:
    def __init__(self, *, log_path, soft_limit_bytes=None, hard_limit_bytes=None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _ = soft_limit_bytes, hard_limit_bytes

    def sample(self, *, event: str, context=None):
        self.log_path.write_text("sample\n", encoding="utf-8")
        return {"event": event, "context": dict(context or {})}

    def checkpoint(self, event: str, context=None):
        _ = event, context

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


def _write_runtime_config(tmp_path: Path) -> str:
    cfg = textwrap.dedent(
        """
        trading:
          symbols: ["ETH/USDT", "SOL/USDT"]
          timeframe: "15m"
        storage:
          backend: "local"
          market_data_parquet_path: "var/data/runtime_exact_window"
          market_data_exchange: "kraken"
        live:
          mode: "paper"
          exchange:
            driver: "binance_futures"
            name: "binance"
            market_type: "future"
            position_mode: "HEDGE"
            margin_mode: "isolated"
            leverage: 2
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def test_exact_window_cli_uses_runtime_defaults_for_adaptive_windows_when_symbols_omitted(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    cfg_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured_adaptive: dict[str, object] = {}
    captured_suite: dict[str, object] = {}

    monkeypatch.setattr(exact_window_cli, "RSSGuard", _FakeGuard)

    def _stub_adaptive(**kwargs):
        captured_adaptive.update(kwargs)
        return {
            "train_start": "2026-01-01T00:00:00+00:00",
            "val_start": "2026-02-01T00:00:00+00:00",
            "oos_start": "2026-02-20T00:00:00+00:00",
            "requested_oos_end_exclusive": "2026-03-09T00:00:00+00:00",
        }

    def _stub_suite(**kwargs):
        captured_suite.update(kwargs)
        return {
            "eligible_symbols": ["ETH/USDT", "SOL/USDT"],
            "best_per_strategy": [{"candidate_id": "c1", "promoted": False}],
            "promoted_count": 0,
            "portfolio": {"weights": []},
        }

    monkeypatch.setattr(exact_window_cli, "resolve_coverage_adaptive_windows", _stub_adaptive)
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
            "--window-profile",
            "coverage_adaptive",
            "--requested-oos-end",
            "2026-03-09",
        ]
    )

    assert rc == 0
    _ = json.loads(capsys.readouterr().out)
    assert captured_adaptive["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert captured_adaptive["root_path"] == "var/data/runtime_exact_window"
    assert captured_adaptive["exchange"] == "kraken"
    assert captured_suite["symbols"] == ["ETH/USDT", "SOL/USDT"]


def test_run_exact_window_suite_uses_runtime_defaults_when_symbols_omitted(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_discover_symbol_coverage(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured defaults")

    monkeypatch.setattr(
        exact_window_suite_module,
        "discover_symbol_coverage",
        _stub_discover_symbol_coverage,
    )

    with pytest.raises(RuntimeError, match="captured defaults"):
        exact_window_suite_module.run_exact_window_suite(
            train_start="2026-01-01",
            val_start="2026-02-01",
            oos_start="2026-02-20",
            requested_oos_end_exclusive="2026-03-09",
        )

    assert captured["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert captured["root_path"] == "var/data/runtime_exact_window"
    assert captured["exchange"] == "kraken"
