from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import polars as pl

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

    def __getitem__(self, _key):
        return self

    def __lt__(self, _other):
        return self

    def filter(self, _predicate):
        return self

    @property
    def height(self) -> int:
        return 1000


class _Repository:
    calls = []

    def __init__(self, _root: str) -> None:
        self.db_path = "fake-data-root"

    def load_ohlcv(self, **kwargs):
        type(self).calls.append(kwargs)
        return _Frame(empty=kwargs["symbol"] == "MISSING/USDT")

    def load_futures_feature_points(self, **_kwargs):
        return pl.DataFrame()


class _Backtest:
    last_config = None

    def __init__(self, *args, **kwargs) -> None:
        assert kwargs["strategy_params"] == {"lookback": 20}
        assert kwargs["strategy_timeframe"] == "1d"
        assert set(kwargs["data_dict"]) == {"BTC/USDT"}
        assert kwargs["record_history"] is True
        assert kwargs["warmup_bars"] == 400
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


def _runtime_config(*, symbol_limits=None):
    return SimpleNamespace(
        trading=SimpleNamespace(timeframe="1m"),
        backtest=SimpleNamespace(persist_output=True),
        risk=SimpleNamespace(attach_default_protective_stop=True),
        execution=SimpleNamespace(
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0004,
            spread_rate=0.0002,
            slippage_rate=0.0005,
            slippage_impact_model="sqrt_impact",
            slippage_impact_coefficient=0.10,
            maintenance_margin_rate=0.005,
            liquidation_buffer_rate=0.0005,
            require_funding_coverage=True,
            funding_on_utc_boundary=True,
        ),
        live=SimpleNamespace(symbol_limits=symbol_limits or {}),
    )


def _receipt() -> dict:
    return {"as_of": "2023-12-31T00:00:00Z", "selected_symbols": {}}


@pytest.fixture(autouse=True)
def _realistic_cost_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "LQ_CONFIG_PATH",
        str(Path(__file__).parents[1] / "configs/profiles/backtest_cost_realistic.yaml"),
    )
    (tmp_path / "data").mkdir()


def test_runs_each_candidate_and_writes_allocator_inputs_and_failures(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "suite.json"
    _Repository.calls.clear()
    manifest.write_text(
        json.dumps(
            {
                "suite_id": "named",
                "universe_materialization_receipt": _receipt(),
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
        lambda: _runtime_config(symbol_limits={"BTC/USDT": {"price_tick_size": 0.25}}),
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
    execution = _Backtest.last_config.execution
    assert execution.slippage_impact_model == "sqrt_impact"
    assert execution.slippage_impact_coefficient > 0
    assert execution.require_funding_coverage is True
    assert execution.funding_on_utc_boundary is True
    assert execution.enforce_reduce_only is True
    assert execution.apply_liquidity_cap_to_conditional_fills is True
    assert _Backtest.last_config.risk.attach_default_protective_stop is False
    passed = json.loads((output / "000_works.json").read_text())
    assert passed["return_timestamps"] == ["2024-01-02", "2024-01-03"]
    assert passed["returns"] == [99.0 / 101.0 - 1.0, 102.0 / 99.0 - 1.0]
    assert passed["turnover"] == 0.15
    assert passed["returns_are_net"] is True
    assert passed["warmup_bars"] == 400
    assert passed["commission_paid"] == 0.0
    assert passed["net_funding_paid"] == 0.0
    assert passed["liquidation_count"] == 0
    assert passed["liquidation_model"].startswith("trade-price OHLC isolated")
    assert passed["research_execution_config"]["slippage_impact_model"] == "sqrt_impact"
    assert next(call for call in _Repository.calls if call["symbol"] == "BTC/USDT")[
        "start_date"
    ] == datetime(2022, 11, 27)
    failed = json.loads((output / "001_missing-data.json").read_text())
    assert failed["status"] == "fail"
    assert failed["error"] == "no local OHLCV for: MISSING/USDT"
    assert failed["returns"] == []
    assert json.loads((output / "suite_results.json").read_text())["fail_count"] == 1
    allocation_input = json.loads((output / "allocation_input.json").read_text())
    assert allocation_input["sleeves"]["works"]["returns"] == passed["returns"]
    assert (
        allocation_input["sleeves"]["works"]["source_artifact_id"]
        == allocation_input["source_artifacts"][0]["id"]
    )
    assert allocation_input["sleeves"]["works"]["returns_are_net"] is True
    assert allocation_input["sleeves"]["works"]["returns_source"] == {
        "artifact": "named_quant_data_pc_walkforward",
        "candidate_id": "works",
        "selection_inputs": ["train", "validation"],
        "stream": "daily UTC net returns over the caller-supplied selection window",
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_sizing": False,
    }
    assert allocation_input["sleeves"]["missing-data"]["run_status"] == "fail"
    assert allocation_input["sleeves"]["missing-data"]["returns"] is None
    assert allocation_input["source_artifacts"][0]["ready"] is False
    assert allocation_input["source_artifacts"][0]["lineage"] == result["lineage"]
    assert allocation_input["source_artifacts"][0]["frozen_at"] == result["period"]["end"]
    assert len(allocation_input["source_artifacts"][0]["sha256"]) == 64
    with pytest.raises(ValueError, match="not portfolio-ready"):
        build_manifest_from_input(allocation_input)


def test_materialized_exchange_filters_are_injected_into_runtime_config(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": {
                    **_receipt(),
                    "selected_symbols": {"crypto_top10": ["BTC/USDT"]},
                    "binance_filters": {
                        "BTC/USDT": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
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
                    }
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
        lambda: _runtime_config(
            symbol_limits={
                "BTC/USDT": {"price_tick_size": 0.25},
                "KEEP/USDT": {"min_notional": 7.0},
            }
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


def test_selected_symbol_with_incomplete_pit_limits_fails_closed(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": {
                    **_receipt(),
                    "selected_symbols": {"crypto_top10": ["BTC/USDT"]},
                    "binance_filters": {
                        "BTC/USDT": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "stepSize": "0.001",
                            },
                        ]
                    },
                },
                "candidates": [],
            }
        )
    )

    with pytest.raises(ValueError, match="incomplete PIT exchange limits"):
        MODULE.run_suite(
            manifest,
            tmp_path / "data",
            tmp_path / "out",
            exchange="binance",
            start=datetime(2024, 1, 1),
            end=datetime(2025, 1, 1),
        )


@pytest.mark.parametrize("has_funding", [False, True])
def test_required_feature_coverage_is_preflighted_for_candidate(
    tmp_path, monkeypatch, has_funding: bool
) -> None:
    class RequiredFeatureStrategy:
        required_features = ("funding_rate",)

    class FeatureRepository(_Repository):
        def load_futures_feature_points(self, **_kwargs):
            if not has_funding:
                return pl.DataFrame()
            first = datetime(2022, 11, 26, 16, tzinfo=UTC)
            end = datetime(2025, 1, 1, tzinfo=UTC)
            rows = []
            current = first
            while current <= end:
                rows.append(
                    {
                        "timestamp_ms": int(current.timestamp() * 1000),
                        "funding_rate": 0.0001,
                    }
                )
                current += MODULE.timedelta(hours=8)
            return pl.DataFrame(rows)

    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": _receipt(),
                "candidates": [
                    {
                        "candidate_id": "funding",
                        "family": "carry",
                        "strategy_class": "FundingStrategy",
                        "symbols": ["BTC/USDT"],
                        "params": {"lookback": 20},
                        "timeframe": "1d",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(MODULE, "MarketDataRepository", FeatureRepository)
    monkeypatch.setattr(MODULE, "Backtest", _Backtest)
    monkeypatch.setattr(
        MODULE, "resolve_strategy_class", lambda name, strict: RequiredFeatureStrategy
    )
    monkeypatch.setattr(MODULE, "get_default_runtime_config", _runtime_config)

    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        tmp_path / "out",
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )

    assert result["results"][0]["status"] == ("pass" if has_funding else "fail")
    if not has_funding:
        assert result["results"][0]["error"] == (
            "missing or stale required features: BTC/USDT:funding_rate"
        )


def test_duplicate_candidate_ids_fail_closed(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": _receipt(),
                "candidates": [{"candidate_id": "same"}] * 2,
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        MODULE.run_suite(
            manifest,
            tmp_path / "data",
            tmp_path / "out",
            exchange="binance",
            start=datetime(2024, 1, 1),
            end=datetime(2025, 1, 1),
        )


def test_blank_family_candidate_cannot_make_suite_ready(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "min_sleeves": 1,
                "min_families": 1,
                "universe_materialization_receipt": _receipt(),
                "candidates": [
                    {
                        "candidate_id": "blank-family",
                        "family": "  ",
                        "strategy_class": "Strategy",
                        "symbols": ["BTC/USDT"],
                        "params": {},
                        "timeframe": "1d",
                    }
                ],
            }
        )
    )
    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        tmp_path / "out",
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )
    assert result["results"][0]["error"] == "missing family"
    assert result["readiness"] == {
        "portfolio_ready": False,
        "min_sleeves": 1,
        "passing_sleeves": 0,
        "min_families": 1,
        "passing_families": 0,
    }


def test_locked_oos_run_never_emits_allocator_input(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": _receipt(),
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
            }
        )
    )
    monkeypatch.setattr(MODULE, "MarketDataRepository", _Repository)
    monkeypatch.setattr(MODULE, "Backtest", _Backtest)
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, strict: object)
    monkeypatch.setattr(
        MODULE,
        "get_default_runtime_config",
        _runtime_config,
    )
    output = tmp_path / "locked"
    selection = tmp_path / "selection.json"
    selection_manifest = json.loads(manifest.read_text())
    selection_manifest_path = tmp_path / "selection_manifest.json"
    selection_manifest_path.write_text(json.dumps(selection_manifest))
    cost_profile, runtime_defaults = MODULE._cost_profile()
    selection.write_text(
        json.dumps(
            {
                "purpose": "selection",
                "period": {"start": "2024-01-01T00:00:00", "end": "2025-01-01T00:00:00"},
                "lineage": MODULE._lineage(
                    selection_manifest,
                    selection_manifest_path,
                    exchange="binance",
                    warmup_bars=400,
                    cost_profile=cost_profile,
                    runtime_defaults=runtime_defaults,
                    data_inventory=MODULE._data_inventory(tmp_path / "data"),
                ),
            }
        )
    )

    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        output,
        exchange="binance",
        start=datetime(2025, 1, 2),
        end=datetime(2026, 1, 1),
        purpose="locked_oos",
        selection_artifact=selection,
    )

    assert result["purpose"] == "locked_oos"
    assert result["selection_artifact"]["sha256"]
    assert not (output / "allocation_input.json").exists()


def test_disabled_candidate_is_reported_as_skip_without_loading_data(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": _receipt(),
                "candidates": [
                    {
                        "candidate_id": "outside",
                        "enabled": False,
                        "disabled_reason": "outside point-in-time universe: AVAX/USDT",
                    }
                ],
            }
        )
    )
    _Repository.calls.clear()
    monkeypatch.setattr(MODULE, "MarketDataRepository", _Repository)

    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        tmp_path / "out",
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )

    assert (result["pass_count"], result["fail_count"], result["skip_count"]) == (0, 0, 1)
    assert _Repository.calls == []
    allocation = json.loads((tmp_path / "out" / "allocation_input.json").read_text())
    assert allocation["source_artifacts"][0]["portfolio_ready"] is False
    assert result["disallowed_skip_ids"] == ["outside"]


def test_materializer_authenticated_exclusion_can_still_be_ready(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "suite.json"
    receipt = _receipt()
    receipt["disabled_candidates"] = {"outside": ["AVAX/USDT"]}
    manifest.write_text(
        json.dumps(
            {
                "suite_id": "named",
                "min_sleeves": 1,
                "universe_materialization_receipt": receipt,
                "candidates": [
                    {
                        "candidate_id": "works",
                        "family": "trend",
                        "strategy_class": "Strategy",
                        "symbols": ["BTC/USDT"],
                        "params": {"lookback": 20},
                        "timeframe": "1d",
                    },
                    {
                        "candidate_id": "outside",
                        "enabled": False,
                        "disabled_reason": "outside point-in-time universe: AVAX/USDT",
                    },
                ],
                "sleeves": {
                    "works": {
                        "family": "trend",
                        "strategy_class": "Strategy",
                        "symbols": ["BTC/USDT"],
                        "params": {"lookback": 20},
                        "source_artifact_id": "legacy_source",
                    },
                    "outside": {
                        "family": "trend",
                        "strategy_class": "Strategy",
                        "symbols": ["AVAX/USDT"],
                        "params": {},
                        "source_artifact_id": "legacy_source",
                    },
                },
            }
        )
    )
    monkeypatch.setattr(MODULE, "MarketDataRepository", _Repository)
    monkeypatch.setattr(MODULE, "Backtest", _Backtest)
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, strict: object)
    monkeypatch.setattr(MODULE, "get_default_runtime_config", _runtime_config)
    result = MODULE.run_suite(
        manifest,
        tmp_path / "data",
        tmp_path / "out",
        exchange="binance",
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
    )
    assert result["readiness"]["portfolio_ready"] is True
    assert result["allowed_exclusions"] == [
        {
            "candidate_id": "outside",
            "reason": "outside point-in-time universe: AVAX/USDT",
        }
    ]
    assert result["exclusion_contract"] == {
        "receipt_disabled_count": 1,
        "allowed_exclusion_count": 1,
        "complete": True,
    }
    allocation = json.loads((tmp_path / "out" / "allocation_input.json").read_text())
    source_id = allocation["source_artifacts"][0]["id"]
    assert {row["source_artifact_id"] for row in allocation["sleeves"].values()} == {source_id}
    assert build_manifest_from_input(allocation)["children"]


def test_warmup_counts_actual_weekday_daily_buckets() -> None:
    class Weekdays:
        calls = 0

        def load_ohlcv(self, **kwargs):
            type(self).calls += 1
            dates = []
            current = kwargs["start_date"]
            while current <= kwargs["end_date"]:
                if current.weekday() < 5:
                    dates.append(current)
                current += MODULE.timedelta(days=1)
            return pl.DataFrame({"datetime": dates})

    start = datetime(2024, 1, 8)
    frame = MODULE._load_with_warmup(
        Weekdays(),
        exchange="binance",
        symbol="SPY/USDT",
        timeframe="1d",
        start=start,
        end=datetime(2024, 1, 10),
        warmup_bars=5,
    )
    assert frame.filter(frame["datetime"] < start).height >= 5
    assert Weekdays.calls == 2


def test_future_universe_receipt_fails_before_execution(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "universe_materialization_receipt": {"as_of": "2024-01-02T00:00:00Z"},
                "candidates": [],
            }
        )
    )
    with pytest.raises(ValueError, match="as_of"):
        MODULE.run_suite(
            manifest,
            tmp_path / "data",
            tmp_path / "out",
            exchange="binance",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )


def test_runner_rejects_stale_output_directory_before_execution(tmp_path) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps({"universe_materialization_receipt": _receipt(), "candidates": []})
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="output target already exists"):
        MODULE.run_suite(
            manifest,
            tmp_path / "data",
            output,
            exchange="binance",
            start=datetime(2024, 1, 1),
            end=datetime(2025, 1, 1),
        )


def test_cli_exits_nonzero_when_readiness_is_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        MODULE,
        "run_suite",
        lambda *args, **kwargs: {"readiness": {"portfolio_ready": False}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--manifest",
            str(tmp_path / "suite.json"),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 1
