from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_research_candidates.py"
    spec = importlib.util.spec_from_file_location("run_research_candidates_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_research_candidates module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_run_research_candidates_script_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_research_candidates.py"
    env = dict(os.environ)
    env["LQ_GPU_MODE"] = "cpu"
    env["LQ_CONFIG_PATH"] = str(root / "config.yaml")
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(tmp_path),
        "--max-candidates",
        "4",
        "--timeframes",
        "1m",
        "5m",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "--enable-synthetic-fallback",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Saved:" in result.stdout
    assert (tmp_path / "candidate_research_latest.json").exists()
    assert (tmp_path / "strategy_factory_report_latest.json").exists()


def test_run_research_candidates_script_dry_run_skips_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_research_candidates.py"
    env = dict(os.environ)
    env["LQ_GPU_MODE"] = "cpu"
    env["LQ_CONFIG_PATH"] = str(root / "config.yaml")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--max-candidates",
            "4",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[RESEARCH] dry-run mode: candidate_count=" in result.stdout
    assert not (tmp_path / "candidate_research_latest.json").exists()
    assert not (tmp_path / "strategy_factory_report_latest.json").exists()


def test_run_research_candidates_script_smoke_with_score_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_research_candidates.py"
    env = dict(os.environ)
    env["LQ_GPU_MODE"] = "cpu"
    env["LQ_CONFIG_PATH"] = str(root / "config.yaml")
    score_cfg_path = tmp_path / "score_config.json"
    score_cfg_path.write_text(
        json.dumps(
            {
                "candidate_rank_score_weights": {
                    "sharpe_weight": 0.2,
                    "return_weight": 45.0,
                },
                "keep_ratio_bounds": {"min": 0.1, "max": 0.8},
                "shortlist_selection": {
                    "max_per_family": 3,
                    "max_per_timeframe": 3,
                    "single_min_score": -0.1,
                    "single_min_return": -0.1,
                    "single_min_sharpe": -0.1,
                    "single_min_trades": 1,
                    "allow_multi_asset": True,
                    "include_weights": True,
                    "weight_temperature": 0.2,
                    "max_weight": 0.5,
                    "robust_score_params": {
                        "failed_candidate_scale": 0.2,
                        "mdd_risk_penalty_coeff": 3.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(tmp_path),
        "--max-candidates",
        "4",
        "--timeframes",
        "1m",
        "5m",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "--score-config",
        str(score_cfg_path),
        "--enable-synthetic-fallback",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    latest = tmp_path / "candidate_research_latest.json"
    assert latest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert (
        payload.get("scoring_config", {})
        .get("candidate_rank_score_weights", {})
        .get("return_weight")
        == 45.0
    )

    team_report = json.loads(
        (tmp_path / "strategy_factory_report_latest.json").read_text(encoding="utf-8")
    )
    shortlist_config = dict(team_report.get("shortlist_config") or {})
    assert int(shortlist_config.get("max_per_family", 0)) == 3
    assert int(shortlist_config.get("max_per_timeframe", 0)) == 3
    assert float(shortlist_config.get("weight_temperature", 0.0)) == 0.2
    assert float(shortlist_config.get("max_weight", 0.0)) == 0.5
    robust_params = dict(shortlist_config.get("robust_score_params") or {})
    assert float(robust_params.get("failed_candidate_scale", 0.0)) == 0.2
    assert float(robust_params.get("mdd_risk_penalty_coeff", 0.0)) == 3.0


def test_shortlist_selection_config_nested_scope_and_cli_precedence():
    payload = {
        "candidate_research": {
            "shortlist_selection": {
                "max_per_family": 7,
                "max_per_timeframe": 5,
                "max_per_lineage": 2,
                "weight_temperature": 0.2,
                "max_weight": 0.5,
            }
        }
    }
    scope = MODULE._score_config_scope(payload)
    resolved = MODULE._resolve_shortlist_selection_config(scope, top_k=11)
    assert resolved["max_total"] == 11
    assert resolved["max_per_family"] == 7
    assert resolved["max_per_timeframe"] == 5
    assert resolved["max_per_lineage"] == 2
    assert float(resolved["weight_temperature"]) == 0.2
    assert float(resolved["max_weight"]) == 0.5
    assert resolved["allow_multi_asset"] is False


def test_run_research_candidates_forwards_max_per_lineage_and_persists_it(
    monkeypatch, tmp_path: Path
):
    root = Path(__file__).resolve().parents[1]
    score_cfg_path = tmp_path / "score_config.json"
    score_cfg_path.write_text(
        json.dumps(
            {
                "candidate_research": {
                    "shortlist_selection": {
                        "max_per_lineage": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    candidates = [
        {
            "candidate_id": "cand-a",
            "name": "carry_trend_factor_rotation_1h_guarded",
            "strategy_class": "CarryTrendFactorRotationStrategy",
            "family": "cross_sectional",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 5.0}},
            "oos": {"return": 0.08, "sharpe": 1.6, "mdd": 0.05, "trades": 24},
            "return_streams": {"train": [], "val": [], "oos": []},
            "metadata": {},
        }
    ]

    captured: dict[str, Any] = {}

    def _stub_build_default_candidate_rows(*, symbols, timeframes, max_candidates):
        _ = (symbols, timeframes, max_candidates)
        return list(candidates)

    def _stub_run_candidate_research_with_optional_split(
        *,
        candidates,
        base_timeframe,
        strategy_timeframes,
        symbol_universe,
        stage1_keep_ratio,
        max_candidates,
        score_config,
        exact_split,
        min_bundle_bars,
        allow_csv_fallback,
        allow_synthetic_fallback,
        progress_callback,
    ):
        _ = (
            candidates,
            base_timeframe,
            strategy_timeframes,
            symbol_universe,
            stage1_keep_ratio,
            max_candidates,
            score_config,
            exact_split,
            min_bundle_bars,
            allow_csv_fallback,
            allow_synthetic_fallback,
            progress_callback,
        )
        return {
            "schema_version": "v2",
            "base_timeframe": "1s",
            "strategy_timeframes": ["1h"],
            "split": {},
            "candidates": list(candidates),
            "stage1": {},
            "scoring_config": {},
            "data_sources": {},
        }

    def _stub_select_diversified_shortlist(rows, **kwargs):
        captured.update(kwargs)
        return list(rows)

    monkeypatch.setattr(MODULE, "build_default_candidate_rows", _stub_build_default_candidate_rows)
    monkeypatch.setattr(
        MODULE,
        "_run_candidate_research_with_optional_split",
        _stub_run_candidate_research_with_optional_split,
    )
    monkeypatch.setattr(MODULE, "select_diversified_shortlist", _stub_select_diversified_shortlist)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(root / "scripts" / "run_research_candidates.py"),
            "--output-dir",
            str(tmp_path),
            "--symbols",
            "BTC/USDT",
            "--timeframes",
            "1h",
            "--max-candidates",
            "1",
            "--top-k",
            "1",
            "--score-config",
            str(score_cfg_path),
        ],
    )

    assert MODULE.main() == 0
    assert int(captured["max_per_lineage"]) == 2

    team_report = json.loads(
        (tmp_path / "strategy_factory_report_latest.json").read_text(encoding="utf-8")
    )
    shortlist_config = dict(team_report.get("shortlist_config") or {})
    assert int(shortlist_config.get("max_per_lineage", 0)) == 2


def test_shortlist_robust_score_params_cross_corr_penalty_and_override_precedence():
    payload = {
        "candidate_research": {
            "candidate_rank_score_weights": {
                "turnover_penalty": 2.2,
                "cross_corr_penalty": 0.9,
            },
            "reject_thresholds": {"max_turnover": 1.7},
            "shortlist_selection": {
                "robust_score_params": {
                    "cross_corr_penalty": 0.4,
                }
            },
        }
    }
    scope = MODULE._score_config_scope(payload)
    params = MODULE._shortlist_robust_score_params(scope)
    assert params is not None
    assert float(params.get("turnover_penalty", 0.0)) == 2.2
    assert float(params.get("turnover_threshold", 0.0)) == 1.7
    assert float(params.get("cross_corr_penalty", 0.0)) == 0.4


def test_exact_split_builder_and_passthrough_support():
    args = MODULE._build_parser().parse_args(
        [
            "--train-start",
            "2025-01-01",
            "--train-end",
            "2025-12-31",
            "--validation-start",
            "2026-01-01",
            "--validation-end",
            "2026-01-31",
            "--oos-start",
            "2026-02-01",
            "--oos-end",
            "2026-03-07T23:59:59Z",
        ]
    )
    exact_split = MODULE._build_exact_split(args)

    captured: dict[str, object] = {}

    def _stub_run_candidate_research(
        *,
        candidates,
        base_timeframe,
        strategy_timeframes,
        symbol_universe,
        stage1_keep_ratio,
        max_candidates,
        score_config,
        split,
    ):
        captured.update(
            {
                "candidates": candidates,
                "base_timeframe": base_timeframe,
                "strategy_timeframes": strategy_timeframes,
                "symbol_universe": symbol_universe,
                "stage1_keep_ratio": stage1_keep_ratio,
                "max_candidates": max_candidates,
                "score_config": score_config,
                "split": split,
            }
        )
        return {"schema_version": "v2", "split": split, "candidates": []}

    original = MODULE.run_candidate_research
    try:
        MODULE.run_candidate_research = _stub_run_candidate_research
        payload = MODULE._run_candidate_research_with_optional_split(
            candidates=[{"candidate_id": "demo"}],
            base_timeframe="1s",
            strategy_timeframes=["1m"],
            symbol_universe=["BTC/USDT"],
            stage1_keep_ratio=0.35,
            max_candidates=16,
            score_config={"candidate_rank_score_weights": {"return_weight": 25.0}},
            exact_split=exact_split,
        )
    finally:
        MODULE.run_candidate_research = original

    assert payload["split"] == exact_split
    assert captured["split"] == exact_split
    assert captured["base_timeframe"] == "1s"
    assert captured["strategy_timeframes"] == ["1m"]


def test_synthetic_fallback_is_disabled_by_default_and_requires_explicit_opt_in():
    default_args = MODULE._build_parser().parse_args([])
    enabled_args = MODULE._build_parser().parse_args(["--enable-synthetic-fallback"])

    assert bool(default_args.enable_synthetic_fallback) is False
    assert bool(default_args.disable_synthetic_fallback) is False
    assert bool(enabled_args.enable_synthetic_fallback) is True


def test_exact_split_coverage_rebuild_clamps_oos_end_and_filters_candidates(monkeypatch):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    oos_cap = datetime(2026, 3, 7, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                start.replace(tzinfo=None),
                oos_cap.replace(tzinfo=None),
                interval="1d",
                eager=True,
            ),
            "open": pl.Series([1.0] * 431, dtype=pl.Float64),
            "high": pl.Series([1.0] * 431, dtype=pl.Float64),
            "low": pl.Series([1.0] * 431, dtype=pl.Float64),
            "close": pl.Series([1.0] * 431, dtype=pl.Float64),
            "volume": pl.Series([1.0] * 431, dtype=pl.Float64),
        }
    )

    def _mock_load_data_dict_from_parquet(
        root_path,
        *,
        exchange,
        symbol_list,
        timeframe,
        start_date=None,
        end_date=None,
        chunk_days=7,
        warmup_bars=0,
        data_mode="legacy",
        staleness_threshold_seconds=None,
    ):
        _ = (
            root_path,
            exchange,
            timeframe,
            start_date,
            end_date,
            chunk_days,
            warmup_bars,
            data_mode,
            staleness_threshold_seconds,
        )
        return {symbol: frame for symbol in symbol_list if symbol == "BTC/USDT"}

    monkeypatch.setattr(MODULE, "load_data_dict_from_parquet", _mock_load_data_dict_from_parquet)

    candidates = [
        {
            "candidate_id": "keep",
            "name": "keep",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1d",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
        {
            "candidate_id": "drop",
            "name": "drop",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1d",
            "symbols": ["ETH/USDT"],
            "params": {},
        },
    ]

    rebuilt, split, summary = MODULE._rebuild_candidates_after_coverage(
        candidates=candidates,
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1d"],
        split={
            "train_start": "2025-01-01T00:00:00Z",
            "train_end": "2025-12-31T23:59:59.999000Z",
            "val_start": "2026-01-01T00:00:00Z",
            "val_end": "2026-01-31T23:59:59.999000Z",
            "oos_start": "2026-02-01T00:00:00Z",
            "oos_end": "2026-03-08T23:59:59.999000Z",
            "strategy_timeframe": "1d",
            "mode": "exact_dates",
        },
    )

    assert [row["candidate_id"] for row in rebuilt] == ["keep"]
    assert split is not None
    assert split["requested_oos_end"] == "2026-03-08T23:59:59.999000Z"
    assert split["oos_end"] == "2026-03-07T00:00:00Z"
    assert split["actual_max_timestamp"] == "2026-03-07T00:00:00Z"
    assert summary["used_candidate_count"] == 1


def test_exact_split_coverage_rebuild_uses_late_asset_data_window(monkeypatch):
    late_start = datetime(2026, 5, 15, tzinfo=UTC)
    end = datetime(2026, 6, 6, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                late_start.replace(tzinfo=None),
                end.replace(tzinfo=None),
                interval="1h",
                eager=True,
            ),
            "open": pl.Series([1.0] * 529, dtype=pl.Float64),
            "high": pl.Series([1.0] * 529, dtype=pl.Float64),
            "low": pl.Series([1.0] * 529, dtype=pl.Float64),
            "close": pl.Series([1.0] * 529, dtype=pl.Float64),
            "volume": pl.Series([1.0] * 529, dtype=pl.Float64),
        }
    )

    def _mock_load_data_dict_from_parquet(root_path, *, symbol_list, **kwargs):
        _ = root_path, kwargs
        return {symbol: frame for symbol in symbol_list if symbol == "ORCL/USDT"}

    monkeypatch.setattr(MODULE, "load_data_dict_from_parquet", _mock_load_data_dict_from_parquet)

    rebuilt, split, summary = MODULE._rebuild_candidates_after_coverage(
        candidates=[
            {
                "candidate_id": "late-tradfi",
                "name": "late-tradfi",
                "strategy_class": "CompositeTrendStrategy",
                "strategy_timeframe": "1h",
                "symbols": ["ORCL/USDT"],
                "params": {},
            }
        ],
        symbols=["ORCL/USDT"],
        timeframes=["1h"],
        split={
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-04-01T00:00:00Z",
            "val_start": "2026-04-01T00:00:00Z",
            "val_end": "2026-05-01T00:00:00Z",
            "oos_start": "2026-05-01T00:00:00Z",
            "oos_end": "2026-06-06T23:59:59.999000Z",
            "strategy_timeframe": "1h",
            "mode": "exact_dates",
        },
    )

    assert split is not None
    assert summary["candidate_window_adjusted_count"] == 1
    assert [row["candidate_id"] for row in rebuilt] == ["late-tradfi"]
    effective_split = rebuilt[0]["effective_split"]
    assert effective_split["mode"] == "candidate_data_window"
    assert effective_split["train_start"] == "2026-05-15T00:00:00Z"
    assert rebuilt[0]["metadata"]["data_window"]["first"] == "2026-05-15T00:00:00Z"
    assert MODULE._symbols_from_candidates(rebuilt, fallback=["BTC/USDT"]) == ["ORCL/USDT"]
    assert MODULE._timeframes_from_candidates(rebuilt, fallback=["30m"]) == ["1h"]


def test_exact_split_coverage_rebuild_does_not_fallback_when_all_candidates_lack_data(
    monkeypatch,
):
    frame = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                datetime(2026, 5, 15),
                datetime(2026, 5, 15, 10),
                interval="1h",
                eager=True,
            ),
            "open": pl.Series([1.0] * 11, dtype=pl.Float64),
            "high": pl.Series([1.0] * 11, dtype=pl.Float64),
            "low": pl.Series([1.0] * 11, dtype=pl.Float64),
            "close": pl.Series([1.0] * 11, dtype=pl.Float64),
            "volume": pl.Series([1.0] * 11, dtype=pl.Float64),
        }
    )

    monkeypatch.setattr(
        MODULE,
        "load_data_dict_from_parquet",
        lambda *args, **kwargs: {"ORCL/USDT": frame},
    )

    rebuilt, _split, summary = MODULE._rebuild_candidates_after_coverage(
        candidates=[
            {
                "candidate_id": "drop",
                "strategy_timeframe": "1h",
                "symbols": ["ORCL/USDT"],
            }
        ],
        symbols=["ORCL/USDT"],
        timeframes=["1h"],
        split={
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-02-01T00:00:00Z",
            "val_start": "2026-02-01T00:00:00Z",
            "val_end": "2026-03-01T00:00:00Z",
            "oos_start": "2026-03-01T00:00:00Z",
            "oos_end": "2026-06-06T23:59:59.999000Z",
            "strategy_timeframe": "1h",
            "mode": "exact_dates",
        },
    )

    assert rebuilt == []
    assert summary["used_candidate_count"] == 0
    assert summary["dropped_candidate_count"] == 1
    assert summary["fallback_to_precoverage"] is False


def test_manifest_candidates_can_be_restricted_to_screened_symbol_subset():
    candidates = [
        {
            "candidate_id": "keep",
            "name": "keep",
            "strategy_class": "CarryTrendFactorRotationStrategy",
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "metadata": {"retune_profile": "screen"},
        },
        {
            "candidate_id": "drop",
            "name": "drop",
            "strategy_class": "CarryTrendFactorRotationStrategy",
            "symbols": ["XRP/USDT"],
        },
    ]

    restricted = MODULE._restrict_candidates_to_symbol_universe(
        candidates,
        ["BTC/USDT", "ETH/USDT"],
    )

    assert [row["candidate_id"] for row in restricted] == ["keep"]
    assert restricted[0]["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert restricted[0]["metadata"]["screened_symbol_subset"] == ["BTC/USDT", "ETH/USDT"]
    assert restricted[0]["metadata"]["screened_symbol_count"] == 2


def test_run_research_candidates_writes_stage_progress_artifacts(monkeypatch, tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "demo-candidate",
                        "name": "demo-candidate",
                        "strategy_class": "CarryTrendFactorRotationStrategy",
                        "family": "cross_sectional",
                        "strategy_timeframe": "1h",
                        "symbols": ["BTC/USDT"],
                        "params": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidate_row = {
        "candidate_id": "demo-candidate",
        "name": "demo-candidate",
        "strategy_class": "CarryTrendFactorRotationStrategy",
        "family": "cross_sectional",
        "strategy_timeframe": "1h",
        "timeframe": "1h",
        "selection_score": 1.25,
        "pass": True,
        "hard_reject": False,
        "train": {
            "total_return": 0.10,
            "return": 0.10,
            "sharpe": 1.0,
            "deflated_sharpe": 0.9,
            "pbo": 0.1,
            "turnover": 0.2,
            "mdd": 0.05,
            "max_drawdown": 0.05,
            "trade_count": 12,
        },
        "val": {
            "total_return": 0.08,
            "return": 0.08,
            "sharpe": 0.9,
            "deflated_sharpe": 0.8,
            "pbo": 0.1,
            "turnover": 0.2,
            "mdd": 0.04,
            "max_drawdown": 0.04,
            "trade_count": 8,
        },
        "oos": {
            "total_return": 0.12,
            "return": 0.12,
            "sharpe": 1.1,
            "deflated_sharpe": 1.0,
            "pbo": 0.1,
            "turnover": 0.2,
            "mdd": 0.03,
            "max_drawdown": 0.03,
            "trade_count": 6,
            "cross_candidate_corr": 0.0,
        },
    }

    def _stub_run_candidate_research(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        assert callable(progress_callback)
        progress_callback(
            "resource_bundle_load_started",
            {
                "symbol_count": 1,
                "timeframe_count": 1,
                "total_count": 1,
                "symbol_universe": ["BTC/USDT"],
                "normalized_timeframes": ["1h"],
            },
        )
        progress_callback(
            "resource_bundle_timeframe_started",
            {
                "timeframe": "1h",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "symbol_count": 1,
                "loaded_count": 0,
                "total_count": 1,
            },
        )
        progress_callback(
            "resource_bundle_timeframe_completed",
            {
                "timeframe": "1h",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "symbol_count": 1,
                "parquet_symbol_count": 1,
                "missing_symbol_count": 0,
                "loaded_count": 0,
                "total_count": 1,
                "elapsed_seconds": 0.2,
            },
        )
        progress_callback(
            "resource_bundle_symbol_fetch_started",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "timeframe": "1h",
                "data_mode": "legacy",
            },
        )
        progress_callback(
            "resource_bundle_symbol_window_loaded",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "timeframe": "1h",
                "unit_kind": "chunk",
                "unit_index": 1,
                "unit_count": 2,
                "row_count": 128,
                "elapsed_seconds": 0.15,
            },
        )
        progress_callback(
            "resource_bundle_symbol_fetch_completed",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "timeframe": "1h",
                "data_mode": "legacy",
                "row_count": 512,
                "was_missing": False,
                "elapsed_seconds": 0.4,
            },
        )
        progress_callback(
            "resource_bundle_item_loaded",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "timeframe": "1h",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "loaded_count": 1,
                "total_count": 1,
                "source": "parquet",
                "bar_count": 512,
                "elapsed_seconds": 1.25,
            },
        )
        progress_callback(
            "resource_bundle_load_completed",
            {
                "bundle_count": 1,
                "total_count": 1,
                "elapsed_seconds": 1.25,
                "source_counts": {"parquet": 1},
            },
        )
        progress_callback(
            "resource_feature_load_started",
            {
                "symbol_count": 1,
                "feature_symbols": ["BTC/USDT"],
            },
        )
        progress_callback(
            "resource_feature_symbol_started",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "loaded_count": 0,
            },
        )
        progress_callback(
            "resource_feature_partition_scan_completed",
            {
                "symbol": "BTC/USDT",
                "partition_count": 3,
                "parquet_file_count": 3,
                "elapsed_seconds": 0.1,
            },
        )
        progress_callback(
            "resource_feature_collect_started",
            {
                "symbol": "BTC/USDT",
                "partition_count": 3,
                "parquet_file_count": 3,
            },
        )
        progress_callback(
            "resource_feature_collect_completed",
            {
                "symbol": "BTC/USDT",
                "partition_count": 3,
                "parquet_file_count": 3,
                "row_count": 256,
                "elapsed_seconds": 0.25,
            },
        )
        progress_callback(
            "resource_feature_symbol_loaded",
            {
                "symbol": "BTC/USDT",
                "symbol_index": 1,
                "symbol_count": 1,
                "loaded_count": 1,
                "row_count": 256,
                "elapsed_seconds": 0.5,
            },
        )
        progress_callback(
            "resource_feature_load_completed",
            {
                "symbol_count": 1,
                "feature_frame_count": 1,
                "nonempty_symbol_count": 1,
                "total_rows": 256,
                "elapsed_seconds": 0.5,
            },
        )
        progress_callback(
            "resource_benchmark_build_started",
            {
                "timeframe_count": 1,
                "normalized_timeframes": ["1h"],
            },
        )
        progress_callback(
            "resource_benchmark_timeframe_started",
            {
                "timeframe": "1h",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "built_count": 0,
            },
        )
        progress_callback(
            "resource_benchmark_timeframe_built",
            {
                "timeframe": "1h",
                "timeframe_index": 1,
                "timeframe_count": 1,
                "built_count": 1,
                "return_count": 511,
                "elapsed_seconds": 0.2,
            },
        )
        progress_callback(
            "resource_benchmark_build_completed",
            {
                "benchmark_count": 1,
                "timeframe_count": 1,
                "nonempty_timeframe_count": 1,
                "elapsed_seconds": 0.2,
            },
        )
        progress_callback(
            "resources_loaded",
            {
                "candidate_count": 1,
                "normalized_timeframes": ["1h"],
                "symbol_universe": ["BTC/USDT"],
                "bundle_count": 1,
                "feature_frame_count": 1,
                "benchmark_count": 1,
            },
        )
        progress_callback(
            "candidate_evaluated",
            {
                "candidate_index": 1,
                "candidate_count": 1,
                "candidate_id": "demo-candidate",
                "name": "demo-candidate",
                "strategy_timeframe": "1h",
                "stage1_prefilter_score": 0.75,
                "train": candidate_row["train"],
                "val": candidate_row["val"],
                "oos": candidate_row["oos"],
            },
        )
        progress_callback(
            "stage1_ranked",
            {
                "candidate_count": 1,
                "keep_count": 1,
                "keep_ratio_applied": 1.0,
                "top_stage1_candidates": [
                    {
                        "candidate_id": "demo-candidate",
                        "name": "demo-candidate",
                        "strategy_timeframe": "1h",
                        "stage1_prefilter_score": 0.75,
                        "train": candidate_row["train"],
                        "val": candidate_row["val"],
                        "oos": candidate_row["oos"],
                    }
                ],
            },
        )
        progress_callback(
            "stage2_selected",
            {
                "selected_count": 1,
                "selected_candidates": [
                    {
                        "candidate_id": "demo-candidate",
                        "name": "demo-candidate",
                        "strategy_timeframe": "1h",
                        "stage1_prefilter_score": 0.75,
                        "train": candidate_row["train"],
                        "val": candidate_row["val"],
                        "oos": candidate_row["oos"],
                    }
                ],
            },
        )
        progress_callback(
            "report_ready",
            {
                "reported_candidate_count": 1,
                "top_report_candidates": [
                    {
                        "candidate_id": "demo-candidate",
                        "name": "demo-candidate",
                        "selection_score": 1.25,
                        "oos_total_return": 0.12,
                        "oos_sharpe": 1.1,
                    }
                ],
            },
        )
        return {
            "schema_version": "v2",
            "base_timeframe": "1m",
            "strategy_timeframes": ["1h"],
            "split": {},
            "candidates": [candidate_row],
            "stage1": {"input_count": 1, "selected_count": 1},
            "scoring_config": {},
            "data_sources": {},
        }

    monkeypatch.setattr(MODULE, "run_candidate_research", _stub_run_candidate_research)
    monkeypatch.setattr(
        MODULE, "select_diversified_shortlist", lambda *args, **kwargs: [candidate_row]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_candidates.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path),
            "--symbols",
            "BTC/USDT",
            "--timeframes",
            "1h",
            "--base-timeframe",
            "1m",
        ],
    )
    exit_code = MODULE.main()

    assert exit_code == 0
    progress_json = json.loads(
        (tmp_path / "candidate_research_progress_latest.json").read_text(encoding="utf-8")
    )
    progress_md = (tmp_path / "candidate_research_progress_latest.md").read_text(encoding="utf-8")
    progress_log = (tmp_path / "candidate_research_progress_latest.log").read_text(encoding="utf-8")

    assert progress_json["status"] == "completed"
    assert progress_json["progress"]["evaluated_count"] == 1
    assert progress_json["progress"]["selected_count"] == 1
    assert progress_json["resource_load"]["bundle"]["loaded_count"] == 1
    assert progress_json["resource_load"]["bundle"]["elapsed_seconds"] == 1.25
    assert progress_json["resource_load"]["bundle"]["slowest_items"][0]["elapsed_seconds"] == 1.25
    assert progress_json["resource_load"]["overall"]["completed_units"] == 3
    assert progress_json["resource_load"]["overall"]["total_units"] == 3
    assert progress_json["resource_load"]["overall"]["completion_ratio"] == 1.0
    assert progress_json["resource_load"]["bundle"]["recent_timeframes"][0]["timeframe"] == "1h"
    assert progress_json["resource_load"]["bundle"]["latest_symbol_fetch"]["row_count"] == 512
    assert progress_json["resource_load"]["bundle"]["latest_window"]["unit_kind"] == "chunk"
    assert progress_json["resource_load"]["feature"]["total_rows"] == 256
    assert progress_json["resource_load"]["feature"]["elapsed_seconds"] == 0.5
    assert (
        progress_json["resource_load"]["feature"]["latest_partition_scan"]["partition_count"] == 3
    )
    assert progress_json["resource_load"]["feature"]["latest_collect"]["status"] == "completed"
    assert progress_json["resource_load"]["benchmark"]["nonempty_timeframe_count"] == 1
    assert progress_json["resource_load"]["benchmark"]["elapsed_seconds"] == 0.2
    assert progress_json["latest_candidate"]["candidate_id"] == "demo-candidate"
    assert "candidate_report" in progress_json["final_artifacts"]
    assert "Resource load progress" in progress_md
    assert "Overall resource progress" in progress_md
    assert "Active bundle symbol fetch" not in progress_md
    assert "Recent bundle timeframe scans" in progress_md
    assert "Latest bundle symbol fetch" in progress_md
    assert "Latest bundle window" in progress_md
    assert "Recent bundle windows" in progress_md
    assert "Latest feature partition scan" in progress_md
    assert "Latest feature collect" in progress_md
    assert "Latest bundle item" in progress_md
    assert "Slowest bundle items" in progress_md
    assert "Slowest feature symbols" in progress_md
    assert "Slowest benchmark timeframes" in progress_md
    assert "Top stage-1 candidates" in progress_md
    assert "resource_bundle_timeframe_started" in progress_log
    assert "resource_bundle_symbol_window_loaded" in progress_log
    assert "resource_feature_partition_scan_completed" in progress_log
    assert "resource_bundle_item_loaded" in progress_log
    assert "candidate_evaluated" in progress_log
