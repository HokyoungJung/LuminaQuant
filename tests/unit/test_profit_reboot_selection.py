from pathlib import Path
import importlib.util
import sys

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


def _preflight(mode: str, *, cash_weight: float = 0.0):
    return reval.ModePreflight(
        mode=mode,
        symbols=["BTC/USDT", "ETH/USDT"],
        cash_weight=cash_weight,
        component_count=1 if cash_weight < 1.0 else 0,
        component_summary=[],
        coverage={},
        status="ready_for_live_equivalent_backtest",
        blocking_reasons=[],
    )


def _result(
    mode: str,
    *,
    train_return: float,
    val_return: float,
    oos_return: float = 0.009,
    train_mdd: float = 0.05,
    val_mdd: float = 0.02,
    oos_mdd: float = 0.001,
    train_trades: int = 40,
    val_trades: int = 6,
    oos_trades: int = 6,
    val_sharpe: float = 1.2,
    val_sortino: float = 2.0,
    oos_sharpe: float = 1.2,
    status: str = "live_equivalent_validated",
):
    metrics = {
        "train": {
            "total_return": train_return,
            "sharpe": 0.5,
            "sortino": 1.0,
            "calmar": 1.0,
            "max_drawdown": train_mdd,
            "volatility": 0.1,
            "cagr": train_return,
        },
        "val": {
            "total_return": val_return,
            "sharpe": val_sharpe,
            "sortino": val_sortino,
            "calmar": 1.0,
            "max_drawdown": val_mdd,
            "volatility": 0.1,
            "cagr": val_return,
        },
        "oos": {
            "total_return": oos_return,
            "sharpe": oos_sharpe,
            "sortino": 2.0,
            "calmar": 1.0,
            "max_drawdown": oos_mdd,
            "volatility": 0.1,
            "cagr": oos_return,
        },
    }
    return {
        "mode": mode,
        "status": status,
        "metrics": metrics,
        "scores": reval._score_from_split_metrics(metrics),
        "split_runs": [
            {
                "split": "train",
                "status": "completed",
                "trade_count": train_trades,
                "final_equity": 100000.0 * (1.0 + train_return),
                "liquidation_count": 0,
            },
            {
                "split": "val",
                "status": "completed",
                "trade_count": val_trades,
                "final_equity": 100000.0 * (1.0 + val_return),
                "liquidation_count": 0,
            },
            {
                "split": "oos",
                "status": "completed",
                "trade_count": oos_trades,
                "final_equity": 100000.0 * (1.0 + oos_return),
                "liquidation_count": 0,
            },
        ],
    }


def test_no_trade_flat_result_is_not_alpha_selection_eligible() -> None:
    preflights = {"flat": _preflight("flat")}
    rows = reval._mode_candidate_rows(
        [
            _result(
                "flat",
                train_return=0.0,
                val_return=0.0,
                train_trades=0,
                val_trades=0,
                oos_trades=0,
                val_sharpe=0.0,
                val_sortino=0.0,
                oos_sharpe=0.0,
            )
        ],
        preflights,
    )

    assert rows[0]["selection_eligible"] is False
    assert rows[0]["selection_role"] == "alpha"
    assert "train_trade_count_below_min" in rows[0]["alpha_blocking_reasons"]
    assert "val_total_return_not_positive" in rows[0]["alpha_blocking_reasons"]


def test_positive_active_result_passes_profit_alpha_gate() -> None:
    preflights = {"winner": _preflight("winner")}
    rows = reval._mode_candidate_rows(
        [_result("winner", train_return=0.04, val_return=0.03)],
        preflights,
    )

    assert rows[0]["selection_eligible"] is True
    assert rows[0]["selection_role"] == "alpha"
    assert rows[0]["alpha_blocking_reasons"] == ""


def test_cash_fallback_is_fallback_only_not_alpha() -> None:
    preflight = _preflight("risk_off_mode", cash_weight=1.0)
    preflight = reval.ModePreflight(
        mode=preflight.mode,
        status="eligible_conservative_cash_fallback",
        symbols=preflight.symbols,
        cash_weight=preflight.cash_weight,
        component_count=preflight.component_count,
        component_summary=preflight.component_summary,
        coverage=preflight.coverage,
        blocking_reasons=preflight.blocking_reasons,
    )
    result = {
        "mode": "risk_off_mode",
        "status": "eligible_conservative_cash_fallback",
        "metrics": {split: reval._empty_metrics() for split in ("train", "val", "oos")},
        "scores": reval._score_from_split_metrics(
            {split: reval._empty_metrics() for split in ("train", "val", "oos")}
        ),
        "split_runs": [],
    }

    rows = reval._mode_candidate_rows([result], {"risk_off_mode": preflight})

    assert rows[0]["selection_eligible"] is False
    assert rows[0]["fallback_eligible"] is True
    assert rows[0]["selection_role"] == "fallback"
