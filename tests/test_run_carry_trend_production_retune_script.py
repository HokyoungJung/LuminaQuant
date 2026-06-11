from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "run_carry_trend_production_retune.py"
)
SPEC = importlib.util.spec_from_file_location("run_carry_trend_production_retune", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load run_carry_trend_production_retune module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_command_contains_exact_split_and_manifest(tmp_path: Path) -> None:
    command = MODULE.build_command(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        score_config=tmp_path / "score.json",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h", "4h"],
        base_timeframe="1m",
        train_start="2025-01-01",
        train_end="2025-12-31",
        validation_start="2026-01-01",
        validation_end="2026-02-28",
        oos_start="2026-03-01",
        oos_end="2026-04-14",
    )

    assert command[0] == sys.executable
    assert "run_research_candidates.py" in command[1]
    assert "--manifest" in command
    assert str(tmp_path / "manifest.json") in command
    assert "--base-timeframe" in command
    assert "1m" in command
    assert "--validation-start" in command
    assert "2026-04-14" in command
    assert "--skip-coverage-rebuild" not in command


def test_build_command_allows_explicit_coverage_rebuild_skip_for_debug(tmp_path: Path) -> None:
    command = MODULE.build_command(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        score_config=tmp_path / "score.json",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h", "4h"],
        base_timeframe="1m",
        train_start="2025-01-01",
        train_end="2025-12-31",
        validation_start="2026-01-01",
        validation_end="2026-02-28",
        oos_start="2026-03-01",
        oos_end="2026-04-14",
        skip_coverage_rebuild=True,
    )

    assert "--skip-coverage-rebuild" in command


def test_low_memory_env_contains_thread_caps() -> None:
    assert MODULE.LOW_MEMORY_ENV["POLARS_MAX_THREADS"] == "1"
    assert MODULE.LOW_MEMORY_ENV["LQ_BACKTEST_LOW_MEMORY"] == "1"
    assert MODULE.LOW_MEMORY_ENV["LQ_AUTO_COLLECT_DB"] == "0"


def test_manifest_symbols_and_timeframes_are_derived_from_candidates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "candidates": [
            {"name":"a","strategy_timeframe":"1h","symbols":["BTC/USDT","ETH/USDT"]},
            {"name":"b","strategy_timeframe":"4h","symbols":["ETH/USDT","BNB/USDT"]}
          ]
        }
        """,
        encoding="utf-8",
    )

    symbols, timeframes = MODULE._manifest_symbols_and_timeframes(manifest)

    assert symbols == ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    assert timeframes == ["1h", "4h"]
