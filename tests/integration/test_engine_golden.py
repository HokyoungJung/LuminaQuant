"""V1 integration test: event-driven engine goldens are enforced in the suite/CI.

Background
----------
Before this test, ``compare_to_golden`` was consumed only by
``test_walk_forward_golden.py`` (which loads the numpy-path walk-forward oracle
``walk_forward_results_warmup.json``).  The *event-driven engine* goldens
captured by ``scripts/capture_golden_baseline.py`` —
``ma_cross_stats.json`` / ``ma_cross_trades.json`` /
``ma_cross_positions_sample.json`` / ``buyholdstrategy_stats.json`` — were
compared by **nothing** in pytest, so the "golden green" attestations for the
engine/execution_sim/portfolio chain (where the funding + reduce_only critical
bugs lived) were manual-only and could silently drift.

This module reproduces the exact runs from
``scripts.capture_golden_baseline._run_ma_cross_backtest`` and
``_run_buyholdbacktest`` on the shipped OHLCV fixtures and asserts the outputs
match the committed goldens:

* Numeric fields of ``trades`` and the first-100 ``positions`` are compared with
  :func:`compare_to_golden` at the golden ``rtol`` (``validation.golden_rtol``,
  ``1e-8``) — the same contract the walk-forward golden uses.
* ``stats`` are formatted strings (``"-3.61%"``, ``"-0.9177"``), which the
  numeric walk cannot see, so they are pinned with exact equality against the
  golden JSON.  This also guards the ``Funding (Net)`` and ``Liquidations``
  rows exercised by the portfolio funding / liquidation path.

Invariants (must match ``scripts/capture_golden_baseline.py``)
-------------------------------------------------------------
* Same fixtures: ``tests/fixtures/ohlcv/{BTCUSDT,ETHUSDT}_seed42_1000d.parquet``.
* Same strategy params, same ``Backtest`` wiring, same record flags.
* Actual artifacts are normalised through the same ``json.dumps(..., default=str)``
  round-trip the capture script used, so numpy scalars / datetimes are coerced
  identically before comparison (``compare_to_golden`` only walks native
  ``float``/``int`` leaves).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.golden_comparator import compare_to_golden
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.strategies.bitcoin_buy_hold import BitcoinBuyHoldStrategy
from lumina_quant.strategies.moving_average import MovingAverageCrossStrategy

# ── Paths / constants (must match scripts/capture_golden_baseline.py) ─────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OHLCV_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "ohlcv"
_GOLDEN_DIR = _REPO_ROOT / "baseline" / "golden"

_NUMPY_SEED = 42
_OHLCV_DAYS = 1000
_OHLCV_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

_BTC_FIXTURE = _OHLCV_FIXTURE_DIR / f"BTCUSDT_seed{_NUMPY_SEED}_{_OHLCV_DAYS}d.parquet"
_ETH_FIXTURE = _OHLCV_FIXTURE_DIR / f"ETHUSDT_seed{_NUMPY_SEED}_{_OHLCV_DAYS}d.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise(obj: object) -> object:
    """Coerce numpy scalars / datetimes exactly like the capture script.

    ``scripts/capture_golden_baseline.py`` wrote every golden with
    ``json.dumps(obj, indent=2, default=str)``.  Round-tripping the live
    artifacts through the same serialisation makes numpy ``float64`` become
    native ``float`` (so ``compare_to_golden``'s ``isinstance(x, float)`` walk
    actually sees them) and datetimes become strings (ignored, non-numeric).
    """
    return json.loads(json.dumps(obj, default=str))


def _load_golden(name: str) -> object:
    return json.loads((_GOLDEN_DIR / name).read_text(encoding="utf-8"))


def _require_fixtures() -> dict[str, pl.DataFrame]:
    for path in (_BTC_FIXTURE, _ETH_FIXTURE):
        if not path.exists():
            pytest.skip(f"OHLCV fixture not found: {path}")
    return {
        sym: pl.read_parquet(_OHLCV_FIXTURE_DIR / f"{sym}_seed{_NUMPY_SEED}_{_OHLCV_DAYS}d.parquet")
        for sym in _OHLCV_SYMBOLS
    }


def _run_ma_cross(data_dict: dict[str, pl.DataFrame]) -> dict:
    """Mirror ``capture_golden_baseline._run_ma_cross_backtest`` exactly."""
    bt = Backtest(
        csv_dir="",
        symbol_list=list(data_dict.keys()),
        start_date=None,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=MovingAverageCrossStrategy,
        strategy_params={"short_window": 10, "long_window": 30, "allow_short": True},
        data_dict=data_dict,
        record_history=True,
        track_metrics=True,
        record_trades=True,
    )
    stats = bt.simulate_trading(output=True, persist_output=False, verbose=False)
    return {
        "stats": stats,
        "trades": getattr(bt.portfolio, "trades", []),
        "positions": getattr(bt.portfolio, "all_positions", []),
    }


def _run_buy_hold(data_dict: dict[str, pl.DataFrame]) -> dict:
    """Mirror ``capture_golden_baseline._run_buyholdbacktest`` exactly."""
    bt = Backtest(
        csv_dir="",
        symbol_list=["BTCUSDT"],
        start_date=None,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=BitcoinBuyHoldStrategy,
        strategy_params={"symbol": "BTCUSDT", "strength": 1.0},
        data_dict={"BTCUSDT": data_dict["BTCUSDT"]},
        record_history=True,
        track_metrics=True,
        record_trades=True,
    )
    stats = bt.simulate_trading(output=True, persist_output=False, verbose=False)
    return {"stats": stats, "trades": getattr(bt.portfolio, "trades", [])}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ma_cross_result() -> dict:
    return _run_ma_cross(_require_fixtures())


@pytest.fixture(scope="module")
def buy_hold_result() -> dict:
    return _run_buy_hold(_require_fixtures())


# ── MA-cross engine goldens ───────────────────────────────────────────────────


def test_ma_cross_stats_match_golden(ma_cross_result: dict) -> None:
    """Formatted MA-cross stats (incl. Funding/Liquidations rows) pin exactly."""
    golden = _load_golden("ma_cross_stats.json")
    actual = _normalise(ma_cross_result["stats"] or [])
    assert actual == golden, (
        "MA-cross engine stats diverged from baseline/golden/ma_cross_stats.json. "
        "If this is an intentional numeric improvement, write "
        "docs/divergences/<artifact>.md and re-capture the golden."
    )


def test_ma_cross_trades_match_golden(ma_cross_result: dict) -> None:
    """Every numeric trade field matches the golden within rtol 1e-8."""
    golden = _load_golden("ma_cross_trades.json")
    actual = _normalise(ma_cross_result["trades"])
    # Guard against silent divergence in trade *count*: compare_to_golden only
    # checks golden keys exist in actual, so extra/fewer trades must be pinned.
    assert isinstance(actual, list)
    assert len(actual) == len(golden), (
        f"MA-cross trade count changed: actual={len(actual)} golden={len(golden)}"
    )
    compare_to_golden(actual, _GOLDEN_DIR / "ma_cross_trades.json", label="ma_cross_trades")


def test_ma_cross_positions_match_golden(ma_cross_result: dict) -> None:
    """First-100 position rows (net/market value floats) match within rtol 1e-8."""
    golden = _load_golden("ma_cross_positions_sample.json")
    # Capture wrote only the first 100 rows.
    actual = _normalise((ma_cross_result["positions"] or [])[:100])
    assert isinstance(actual, list)
    assert len(actual) == len(golden), (
        f"MA-cross positions-sample length changed: actual={len(actual)} golden={len(golden)}"
    )
    compare_to_golden(
        actual, _GOLDEN_DIR / "ma_cross_positions_sample.json", label="ma_cross_positions"
    )


# ── BuyHold sanity golden ─────────────────────────────────────────────────────


def test_buy_hold_stats_match_golden(buy_hold_result: dict) -> None:
    """BuyHold sanity stats (incl. the Liquidations=1 catastrophic row) pin exactly."""
    golden = _load_golden("buyholdstrategy_stats.json")
    actual = _normalise(buy_hold_result["stats"] or [])
    assert actual == golden, (
        "BuyHold engine stats diverged from baseline/golden/buyholdstrategy_stats.json. "
        "If this is an intentional numeric improvement, write "
        "docs/divergences/<artifact>.md and re-capture the golden."
    )
