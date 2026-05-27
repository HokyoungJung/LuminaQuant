from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "run_four_sleeve_regime_portfolio.py"
)
_SPEC = importlib.util.spec_from_file_location("run_four_sleeve_regime_portfolio", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("failed to load run_four_sleeve_regime_portfolio.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SleeveAllocatorParams = _MODULE.SleeveAllocatorParams
_load_named_candidate_row = _MODULE._load_named_candidate_row
run_four_sleeve_allocator = _MODULE.run_four_sleeve_allocator


def _stream(start_day: int, values: list[float]) -> list[dict[str, object]]:
    return [
        {
            "datetime": f"2026-01-{start_day + idx:02d}T00:00:00Z",
            "return": value,
        }
        for idx, value in enumerate(values)
    ]


def test_load_named_candidate_row_aliases_candidate(tmp_path: Path) -> None:
    payload = [
        {
            "candidate_id": "orig-1",
            "name": "alpha",
            "strategy_class": "PairSpreadZScoreStrategy",
            "return_streams": {"train": [], "val": [], "oos": []},
        }
    ]
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    row = _load_named_candidate_row(source, name="alpha", alias="alias-alpha")

    assert row["candidate_id"] == "alias-alpha"
    assert row["source_candidate_id"] == "orig-1"
    assert row["source_path"] == str(source)


def test_run_four_sleeve_allocator_rebalances_and_tracks_cash() -> None:
    rows = [
        {
            "candidate_id": "btc_trx_pair_balanced",
            "name": "btc pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "family": "market_neutral",
            "strategy_timeframe": "1d",
            "symbols": ["BTC/USDT", "TRX/USDT"],
            "metadata": {},
            "return_streams": {"train": _stream(1, [0.02] * 8), "val": [], "oos": []},
        },
        {
            "candidate_id": "bnb_trx_pair_tightstop",
            "name": "bnb pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "family": "market_neutral",
            "strategy_timeframe": "1h",
            "symbols": ["BNB/USDT", "TRX/USDT"],
            "metadata": {},
            "return_streams": {"train": _stream(1, [0.01] * 8), "val": [], "oos": []},
        },
        {
            "candidate_id": "composite_trend_core",
            "name": "trend",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "30m",
            "symbols": ["BTC/USDT"],
            "metadata": {},
            "return_streams": {"train": _stream(1, [0.005] * 8), "val": [], "oos": []},
        },
        {
            "candidate_id": "topcap_balanced",
            "name": "topcap",
            "strategy_class": "TopCapTimeSeriesMomentumStrategy",
            "family": "cross_sectional",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT"],
            "metadata": {},
            "return_streams": {"train": _stream(1, [0.0] * 8), "val": [], "oos": []},
        },
    ]
    params = SleeveAllocatorParams(
        lookback_days=3,
        rebalance_days=7,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.2,
        max_weight=0.55,
        regime_strength=1.0,
        hysteresis_bonus=0.2,
        turnover_cost_bps=8.0,
        kelly_shrinkage=1.0,
    )
    regime_features = {
        f"2026-01-{day:02d}": {
            "btc_above_ma192": True,
            "breadth_ma96_ge_60": True,
            "basket_vol_ratio_moderate": True,
        }
        for day in range(1, 9)
    }

    result = run_four_sleeve_allocator(
        rows,
        params,
        regime_features=regime_features,
        continuity_payload={"status": "completed"},
    )

    allocations = list(result["allocations"])
    assert allocations
    last = allocations[-1]
    assert sum(last["weights"].values()) <= 1.0 + 1e-9
    assert 0.0 <= last["cash_weight"] <= 1.0
    assert last["target_total_exposure"] <= 1.0
    assert "sleeve_turnover" in last
    assert "train" in result["split_metrics"]
    assert any(item["target_total_exposure"] >= 0.0 for item in allocations)
