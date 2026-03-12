from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

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
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(tmp_path),
        "--max-candidates",
        "48",
        "--timeframes",
        "1m",
        "5m",
        "1h",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XAU/USDT",
        "XAG/USDT",
    ]

    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Saved:" in result.stdout
    assert (tmp_path / "candidate_research_latest.json").exists()
    assert (tmp_path / "strategy_factory_report_latest.json").exists()


def test_run_research_candidates_script_smoke_with_score_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_research_candidates.py"
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
        "24",
        "--timeframes",
        "1m",
        "5m",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "--score-config",
        str(score_cfg_path),
    ]

    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    latest = tmp_path / "candidate_research_latest.json"
    assert latest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload.get("scoring_config", {}).get("candidate_rank_score_weights", {}).get("return_weight") == 45.0

    team_report = json.loads((tmp_path / "strategy_factory_report_latest.json").read_text(encoding="utf-8"))
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
    assert float(resolved["weight_temperature"]) == 0.2
    assert float(resolved["max_weight"]) == 0.5


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
