from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "build_committee_portfolio_followup.py"
SPEC = importlib.util.spec_from_file_location("build_committee_portfolio_followup", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load build_committee_portfolio_followup module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_committee_portfolio_followup = MODULE.build_committee_portfolio_followup
write_committee_portfolio_followup = MODULE.write_committee_portfolio_followup


def _row(name: str, split_return: float) -> dict[str, object]:
    return {
        "candidate_id": name,
        "name": name,
        "strategy_class": "CompositeTrendStrategy",
        "strategy_timeframe": "1h",
        "symbols": ["BTC/USDT"],
        "train": {
            "return": split_return,
            "sharpe": 1.0,
            "sortino": 1.1,
            "calmar": 1.2,
            "max_drawdown": 0.05,
            "volatility": 0.10,
            "trade_count": 10.0,
            "turnover": 0.10,
            "win_rate": 0.55,
            "avg_trade": 0.001,
            "exposure": 0.40,
            "deflated_sharpe": 0.20,
            "pbo": 0.30,
            "spa_pvalue": 0.20,
            "benchmark_corr": -0.10,
            "rolling_sharpe_min": -2.0,
            "stability": -0.5,
            "worst_month": -0.03,
        },
        "val": {
            "return": split_return,
            "sharpe": 1.1,
            "sortino": 1.2,
            "calmar": 1.3,
            "max_drawdown": 0.04,
            "volatility": 0.11,
            "trade_count": 9.0,
            "turnover": 0.09,
            "win_rate": 0.56,
            "avg_trade": 0.0011,
            "exposure": 0.41,
            "deflated_sharpe": 0.21,
            "pbo": 0.31,
            "spa_pvalue": 0.21,
            "benchmark_corr": -0.11,
            "rolling_sharpe_min": -2.1,
            "stability": -0.6,
            "worst_month": -0.02,
        },
        "oos": {
            "return": split_return,
            "sharpe": 1.2,
            "sortino": 1.3,
            "calmar": 1.4,
            "max_drawdown": 0.03,
            "volatility": 0.12,
            "trade_count": 8.0,
            "turnover": 0.08,
            "win_rate": 0.57,
            "avg_trade": 0.0012,
            "exposure": 0.42,
            "deflated_sharpe": 0.22,
            "pbo": 0.32,
            "spa_pvalue": 0.22,
            "benchmark_corr": -0.12,
            "rolling_sharpe_min": -2.2,
            "stability": -0.7,
            "worst_month": -0.01,
        },
        "return_streams": {
            "train": [{"t": 1_735_689_600_000.0, "v": split_return}],
            "val": [{"t": 1_767_225_600_000.0, "v": split_return}],
            "oos": [{"t": 1_769_904_000_000.0, "v": split_return}],
        },
        "metadata": {},
    }


def test_build_committee_portfolio_followup_uses_provided_rows():
    payload = build_committee_portfolio_followup(
        component_rows=[_row("a", 0.01), _row("b", 0.02)],
        pair_payload={
            "survives": False,
            "coverage_guard": {"pass": False, "observed_total_days": 36, "min_total_days": 60},
        },
    )

    assert payload["component_count"] == 2
    assert payload["pair_survives"] is False
    assert len(payload["selection"]) == 2
    assert "Coverage guard blocked pair inclusion" in " ".join(payload["notes"])


def test_write_committee_portfolio_followup_writes_files(tmp_path: Path):
    result = write_committee_portfolio_followup(
        report_root=tmp_path,
        component_rows=[_row("a", 0.01), _row("b", 0.02)],
        pair_payload={"survives": False},
    )

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
