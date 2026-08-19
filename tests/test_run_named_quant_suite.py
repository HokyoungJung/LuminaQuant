from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research.build_quality_gated_allocation import build_manifest_from_input

SCRIPT = Path(__file__).parents[1] / "scripts/research/run_named_quant_suite.py"
SPEC = importlib.util.spec_from_file_location("run_named_quant_suite", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Frame:
    def __init__(self, empty: bool = False) -> None:
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class _Repository:
    def __init__(self, _root: str) -> None:
        self.db_path = "fake-data-root"

    def load_ohlcv(self, **kwargs):
        return _Frame(empty=kwargs["symbol"] == "MISSING/USDT")


class _Backtest:
    last_config = None

    def __init__(self, *args, **kwargs) -> None:
        assert kwargs["strategy_params"] == {"lookback": 20}
        assert kwargs["strategy_timeframe"] == "1d"
        assert set(kwargs["data_dict"]) == {"BTC/USDT"}
        assert kwargs["record_history"] is True
        assert kwargs["data_handler_kwargs"] == {
            "feature_db_path": "fake-data-root",
            "feature_exchange": "binance",
        }
        type(self).last_config = kwargs["config"]
        self.portfolio = SimpleNamespace(
            initial_capital=100.0,
            all_holdings=[
                (datetime(2024, 1, 1, tzinfo=UTC), 0, 0, 0, 100.0),
                (datetime(2024, 1, 1, 23, tzinfo=UTC), 0, 0, 0, 101.0),
                (datetime(2024, 1, 2, 12, tzinfo=UTC), 0, 0, 0, 98.0),
                (datetime(2024, 1, 2, 23, tzinfo=UTC), 0, 0, 0, 99.0),
                (datetime(2024, 1, 3, 23, tzinfo=UTC), 0, 0, 0, 102.0),
            ],
            trades=[{"fill_cost": 10.0}, {"fill_cost": 20.0}],
            trade_count=2,
        )

    def simulate_trading(self, *, output: bool) -> None:
        assert output is False


def test_runs_each_candidate_and_writes_allocator_inputs_and_failures(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "suite_id": "named",
                "method": "hrp",
                "min_sleeves": 1,
                "sleeves": {
                    "works": {"family": "trend", "returns": None, "turnover": None},
                    "missing-data": {
                        "family": "reversion",
                        "returns": None,
                        "turnover": None,
                    },
                },
                "candidates": [
                    {
                        "candidate_id": "works",
                        "family": "trend",
                        "strategy_class": "Strategy",
                        "symbols": ["BTC/USDT"],
                        "params": {"lookback": 20},
                        "timeframe": "1d",
                        "strategy_timeframe": "1d",
                    },
                    {
                        "candidate_id": "missing-data",
                        "family": "reversion",
                        "strategy_class": "Strategy",
                        "symbols": ["MISSING/USDT"],
                        "params": {},
                        "timeframe": "1d",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(MODULE, "MarketDataRepository", _Repository)
    monkeypatch.setattr(MODULE, "Backtest", _Backtest)
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, strict: object)
    monkeypatch.setattr(
        MODULE,
        "get_default_runtime_config",
        lambda: SimpleNamespace(
            trading=SimpleNamespace(timeframe="1m"),
            backtest=SimpleNamespace(persist_output=True),
            live=SimpleNamespace(symbol_limits={"BTC/USDT": {"price_tick_size": 0.25}}),
        ),
    )

    output = tmp_path / "out"
    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        output,
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )

    assert (result["pass_count"], result["fail_count"]) == (1, 1)
    assert _Backtest.last_config.live.symbol_limits == {"BTC/USDT": {"price_tick_size": 0.25}}
    passed = json.loads((output / "000_works.json").read_text())
    assert passed["return_timestamps"] == ["2024-01-02", "2024-01-03"]
    assert passed["returns"] == [99.0 / 101.0 - 1.0, 102.0 / 99.0 - 1.0]
    assert passed["turnover"] == 0.15
    failed = json.loads((output / "001_missing-data.json").read_text())
    assert failed["status"] == "fail"
    assert failed["error"] == "no local OHLCV for: MISSING/USDT"
    assert failed["returns"] == []
    assert json.loads((output / "suite_results.json").read_text())["fail_count"] == 1
    allocation_input = json.loads((output / "allocation_input.json").read_text())
    assert allocation_input["sleeves"]["works"]["returns"] == passed["returns"]
    assert allocation_input["sleeves"]["works"]["returns_are_net"] is True
    assert allocation_input["sleeves"]["works"]["returns_source"] == {
        "artifact": "named_quant_data_pc_walkforward",
        "candidate_id": "works",
        "selection_inputs": ["train", "validation"],
        "stream": "daily UTC train/validation net returns",
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_sizing": False,
    }
    assert allocation_input["sleeves"]["missing-data"]["run_status"] == "fail"
    assert allocation_input["sleeves"]["missing-data"]["returns"] is None
    assert allocation_input["source_artifacts"][0]["ready"] is False
    assert len(allocation_input["source_artifacts"][0]["sha256"]) == 64
    build_manifest_from_input(allocation_input)


def test_materialized_exchange_filters_are_injected_into_runtime_config(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "works",
                        "family": "trend",
                        "strategy_class": "Strategy",
                        "symbols": ["BTC/USDT"],
                        "params": {"lookback": 20},
                        "timeframe": "1d",
                    }
                ],
                "universe_materialization_receipt": {
                    "binance_filters": {
                        "BTC/USDT": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "stepSize": "0.001",
                            },
                            {
                                "filterType": "MARKET_LOT_SIZE",
                                "minQty": "0.01",
                                "stepSize": "0.005",
                            },
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                        "INVALID/USDT": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0"},
                            {"filterType": "LOT_SIZE", "minQty": "bad"},
                        ],
                    }
                },
            }
        )
    )
    monkeypatch.setattr(MODULE, "MarketDataRepository", _Repository)
    monkeypatch.setattr(MODULE, "Backtest", _Backtest)
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, strict: object)
    monkeypatch.setattr(
        MODULE,
        "get_default_runtime_config",
        lambda: SimpleNamespace(
            trading=SimpleNamespace(timeframe="1m"),
            backtest=SimpleNamespace(persist_output=True),
            live=SimpleNamespace(
                symbol_limits={
                    "BTC/USDT": {"price_tick_size": 0.25},
                    "KEEP/USDT": {"min_notional": 7.0},
                }
            ),
        ),
    )

    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        tmp_path / "out",
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )

    assert result["fail_count"] == 0
    assert _Backtest.last_config.live.symbol_limits == {
        "BTC/USDT": {
            "price_tick_size": 0.1,
            "min_qty": 0.01,
            "qty_step": 0.005,
            "min_notional": 5.0,
        },
        "KEEP/USDT": {"min_notional": 7.0},
    }


def test_duplicate_candidate_ids_fail_closed(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"candidates": [{"candidate_id": "same"}] * 2}))
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        MODULE.run_suite(
            manifest,
            tmp_path / "data",
            tmp_path / "out",
            exchange="binance",
            start=datetime(2024, 1, 1),
            end=datetime(2025, 1, 1),
        )
