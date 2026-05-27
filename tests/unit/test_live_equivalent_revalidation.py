from datetime import date
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "revalidate_live_equivalent_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("revalidate_live_equivalent_candidates", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reval
SPEC.loader.exec_module(reval)


def _fake_pair_definition(mode: str):
    component = SimpleNamespace(
        component_id="pair_fixture",
        label="pair_fixture",
        strategy_class="PairSpreadZScoreStrategy",
        symbols=("BNB/USDT", "TRX/USDT"),
        weight=1.0,
    )
    return SimpleNamespace(
        portfolio_mode=mode,
        components=(component,),
        symbols=["BNB/USDT", "TRX/USDT"],
        cash_weight=0.0,
    )


def _fake_risk_off_definition(mode: str):
    return SimpleNamespace(
        portfolio_mode=mode,
        components=(),
        symbols=[],
        cash_weight=1.0,
    )


def test_mode_preflight_marks_pair_modes_ready_on_complete_stubbed_days(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        reval,
        "_has_committed_materialized_day",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(reval, "_legacy_monthly_files", lambda **_kwargs: [])
    monkeypatch.setattr(reval, "resolve_portfolio_mode_definition", _fake_pair_definition)
    splits = [
        reval.SplitWindow("train", date(2025, 1, 1), date(2025, 1, 2), "sanity_filter"),
        reval.SplitWindow("val", date(2026, 1, 1), date(2026, 1, 2), "primary_selection"),
        reval.SplitWindow("oos", date(2026, 3, 1), date(2026, 3, 2), "report_only"),
    ]

    preflight = reval._mode_preflight(
        mode="state_vwap_pair",
        market_root=tmp_path,
        exchange="binance",
        timeframe="1s",
        splits=splits,
    )

    assert preflight.status == "ready_for_live_equivalent_backtest"
    assert preflight.blocking_reasons == []


def test_mode_preflight_blocks_missing_train_val_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        reval,
        "_has_committed_materialized_day",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(reval, "_legacy_monthly_files", lambda **_kwargs: [])
    monkeypatch.setattr(reval, "resolve_portfolio_mode_definition", _fake_pair_definition)
    splits = [
        reval.SplitWindow("train", date(2025, 1, 1), date(2025, 1, 1), "sanity_filter"),
        reval.SplitWindow("val", date(2026, 1, 1), date(2026, 1, 1), "primary_selection"),
        reval.SplitWindow("oos", date(2026, 3, 1), date(2026, 3, 1), "report_only"),
    ]

    preflight = reval._mode_preflight(
        mode="state_vwap_pair",
        market_root=tmp_path,
        exchange="binance",
        timeframe="1s",
        splits=splits,
    )

    assert preflight.status == "blocked_missing_raw_first_market_data"
    assert "BNB/USDT:train_raw_first_incomplete" in preflight.blocking_reasons


def test_revalidation_defaults_to_preflight_not_full_backtest(monkeypatch, tmp_path: Path) -> None:
    calls = {"backtests": 0}

    monkeypatch.setattr(reval, "supported_portfolio_modes", lambda: {"risk_off_mode"})
    monkeypatch.setattr(reval, "resolve_portfolio_mode_definition", _fake_risk_off_definition)
    monkeypatch.setattr(reval, "_load_full_universe", lambda _path: {"ranked_clean_candidates": []})

    def _unexpected_backtest(**_kwargs):
        calls["backtests"] += 1
        raise AssertionError("full backtests must be opt-in")

    monkeypatch.setattr(reval, "_run_live_equivalent_split", _unexpected_backtest)
    result = reval.build_live_equivalent_revalidation(
        output_dir=tmp_path,
        full_universe_path=tmp_path / "missing.json",
        market_root=tmp_path / "market",
        exchange="binance",
        update_live_decision=False,
    )

    assert calls["backtests"] == 0
    assert (
        result["payload"]["final_recommendations"]["best_full_universe_live_equivalent_candidate"]
        is None
    )


def test_split_windows_default_uses_latest_complete_utc_day(monkeypatch) -> None:
    monkeypatch.setattr(reval, "_latest_complete_utc_day", lambda: date(2026, 5, 2))

    splits = reval._split_windows()

    assert splits[-1].name == "oos"
    assert splits[-1].end == date(2026, 5, 2)


def test_metrics_cap_negative_equity_as_breach_not_investment_drawdown() -> None:
    metrics = reval._metrics_from_equity_totals([10_000.0, 12_000.0, -1_000.0], periods=252)

    assert metrics["equity_breach_observed"] is True
    assert metrics["total_return"] == -1.0
    assert metrics["max_drawdown"] == 1.0
    assert metrics["raw_total_return"] < -1.0
    assert metrics["raw_max_drawdown"] > 1.0
    assert metrics["min_equity"] < 0.0


def test_revalidation_runtime_config_allows_research_only_defined_modes(monkeypatch) -> None:
    monkeypatch.setattr(reval, "supported_portfolio_modes", lambda: {"shadow_mode"})
    monkeypatch.setattr(reval, "resolve_portfolio_mode_definition", _fake_pair_definition)
    monkeypatch.setattr(
        reval,
        "resolve_portfolio_mode_runtime_config",
        lambda _mode: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    config = reval._resolve_revalidation_runtime_config("shadow_mode")

    assert config["research_only_live_equivalent"] is True
    assert config["symbols"] == ["BNB/USDT", "TRX/USDT"]
    assert config["strategy_params"] == {"portfolio_mode": "shadow_mode"}


def test_fail_fast_alpha_gate_skips_val_after_train_floor_failure(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    splits = [
        reval.SplitWindow("train", date(2025, 1, 1), date(2025, 1, 1), "sanity_filter"),
        reval.SplitWindow("val", date(2026, 1, 1), date(2026, 1, 1), "primary_selection"),
        reval.SplitWindow("oos", date(2026, 3, 1), date(2026, 3, 1), "report_only"),
    ]
    preflight = reval.ModePreflight(
        mode="derivatives_flow_squeeze_mode",
        symbols=["BTC/USDT"],
        cash_weight=0.0,
        component_count=1,
        component_summary=[],
        coverage={"BTC/USDT": {"oos": {"complete_raw_first": True}}},
        status="ready_for_live_equivalent_backtest",
        blocking_reasons=[],
    )

    monkeypatch.setattr(reval, "_mode_equivalence_key", lambda _mode: "same-graph")

    def _fake_split(**kwargs):
        split = kwargs["split"].name
        calls.append(split)
        if split != "train":
            raise AssertionError("fail-fast should not run validation/OOS splits")
        return {
            "split": "train",
            "equivalence_key": "same-graph",
            "metrics": {
                "total_return": -0.20,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.05,
                "volatility": 0.0,
            },
            "trade_count": 30,
            "final_equity": 8000.0,
            "liquidation_count": 0,
        }

    monkeypatch.setattr(reval, "_run_live_equivalent_split", _fake_split)

    result = reval._run_mode_backtests(
        preflight=preflight,
        market_root=tmp_path,
        exchange="binance",
        timeframe="1s",
        splits=splits,
        chunk_days=7,
        backtest_poll_seconds=1,
        backtest_window_seconds=1,
        fail_fast_alpha_gate=True,
    )

    assert calls == ["train"]
    assert result["status"] == "failed_train_alpha_gate"
    assert [run["status"] for run in result["split_runs"]] == [
        "completed",
        "skipped_train_alpha_gate_failed",
        "skipped_train_alpha_gate_failed",
    ]
