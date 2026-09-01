from __future__ import annotations

import json
import multiprocessing
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from scripts.research import backtest_dacapogo_daily_tuner as tuner


def _run_locked_acquisition(cache_dir: str, entered, release) -> None:
    def transaction(*args, **kwargs):
        entered.set()
        if release is not None:
            assert release.wait(30)
        return pl.DataFrame()

    tuner._acquire_execution_candidates = transaction
    tuner.acquire_execution_candidates(
        pl.DataFrame(), cache_dir=Path(cache_dir), data_root=Path(cache_dir)
    )


def _panel(days: int = 10, symbols: int = 20) -> pl.DataFrame:
    start = date(2025, 1, 1)
    return pl.DataFrame(
        [
            {
                "market": f"S{symbol:02}USDT",
                "date": start + timedelta(days=day),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 101.0,
                "value": float(10_000 - symbol),
            }
            for symbol in range(symbols)
            for day in range(days)
        ]
    )


def test_current_high_low_cannot_change_preopen_order_and_ties_use_market() -> None:
    features = tuner.build_daily_features(_panel()).filter(pl.col("date") == date(2025, 1, 10))
    predictions = np.ones(features.height)
    before = tuner.preopen_select(
        features, predictions, universe_topk=15, position_cap=5, breakout=0.04
    )
    mutated = features.with_columns((pl.col("high") * 100).alias("high"))
    after = tuner.preopen_select(
        mutated, predictions, universe_topk=15, position_cap=5, breakout=0.04
    )
    assert before["market"].to_list() == after["market"].to_list()
    assert before["market"].to_list() == sorted(before["market"].to_list())


def test_position_cap_is_applied_before_trigger() -> None:
    rows = tuner.build_daily_features(_panel()).filter(pl.col("date") == date(2025, 1, 10))
    rows = rows.with_columns(
        pl.when(pl.col("market") < "S05USDT").then(100.0).otherwise(110.0).alias("high")
    )
    selected = tuner.preopen_select(
        rows, np.arange(rows.height, 0, -1), universe_topk=15, position_cap=5, breakout=0.04
    )
    assert selected.height == 5
    assert selected.filter(pl.col("filled")).is_empty()


def test_model_presets_are_deterministic_and_single_threaded() -> None:
    presets = tuner.model_presets()
    assert {item.family for item in presets} == {
        "turnover",
        "ridge",
        "tree",
        "extra_trees",
        "hist_gbdt",
        "lightgbm",
    }
    for preset in presets:
        if preset.family in {"tree", "extra_trees", "hist_gbdt", "lightgbm"}:
            assert preset.params["random_state"] == 0
        if "n_jobs" in preset.params:
            assert preset.params["n_jobs"] == 1
    hist = next(item for item in presets if item.family == "hist_gbdt")
    assert hist.params["early_stopping"] is False
    assert hist.params["max_bins"] == 63


def test_latest_noncausal_upstream_ranker_is_provenance_only() -> None:
    assert tuner.LATEST_UPSTREAM_HEAD == "633ba5d6bc0c84a20696af6b2bf807cf55d21248"
    assert tuner.MULTIFREQ_FILE_SHA256 == (
        "dc675e925a0a0ac7e0c8e49e8c943aa432514ac937ef352b30027a009897115c"
    )
    assert tuner.NESTED_CV_FILE_SHA256 == (
        "0c6cbd411e046724c0595f118b7fd6939bdf4f2a80841745ad1ed2f4e379543f"
    )
    assert tuner.RANKER_FILE_SHA256 == (
        "e338f296b75fbf0eb9a3c2e1181fd14d813c898993ddc9f6c285a621466c37a7"
    )
    assert tuner.RANKER_USED_FOR_SIGNAL is False


def test_ridge_prediction_is_feature_scale_invariant() -> None:
    preset = next(
        item for item in tuner.model_presets(include_lightgbm=False) if item.name == "ridge_1"
    )
    train_x = np.array([[1.0, 1e9], [2.0, 2e9], [3.0, 3e9], [4.0, 4e9]])
    train_y = np.array([1.0, 2.0, 3.0, 4.0])
    predict_x = np.array([[5.0, 5e9]])
    prediction = tuner._predict(
        preset, train_x, train_y, predict_x, turnover=np.zeros(len(predict_x))
    )
    scaled = tuner._predict(
        preset,
        train_x / np.array([1.0, 1e9]),
        train_y,
        predict_x / np.array([1.0, 1e9]),
        turnover=np.zeros(len(predict_x)),
    )
    np.testing.assert_allclose(prediction, scaled)


def test_winner_and_trade_gate_have_no_oos_input() -> None:
    rows = [
        {"candidate_id": "a", "validation_score": 0.1, "gate_passed": True},
        {"candidate_id": "b", "validation_score": 0.2, "gate_passed": False},
    ]
    winner = tuner.choose_validation_winner(rows)
    assert winner is not None and winner["candidate_id"] == "a"
    assert "oos" not in tuner.trade_count_gate.__annotations__


def test_oos_outcome_mutation_cannot_change_validation_selection(monkeypatch) -> None:
    monkeypatch.setattr(tuner, "BREAKOUTS", (0.04,))
    monkeypatch.setattr(tuner, "UNIVERSE_TOPKS", (10,))
    monkeypatch.setattr(tuner, "POSITION_CAPS", (5,))
    monkeypatch.setattr(tuner, "EXIT_PROFILES", {"source": (0.005, 0.008)})
    monkeypatch.setattr(tuner, "LEVERAGES", (1,))
    monkeypatch.setattr(
        tuner,
        "SCENARIOS",
        (
            ("close_exit_stop_first", "close", "stop_first"),
            ("tp_sl_stop_first", "tp_sl", "stop_first"),
        ),
    )
    panel = _panel(days=242, symbols=31)
    union = tuner.expanded_union(panel)
    execution = pl.DataFrame(
        [
            {
                "market": row["market"],
                "date": row["date"],
                "breakout": 0.04,
                "exit_profile": "source",
                "scenario": scenario,
                "leverage": 1,
                "slot_return": 0.01,
            }
            for row in union.iter_rows(named=True)
            for scenario in tuner.ADVERSE_SCENARIOS
        ]
    )
    fold = tuner.make_folds(date(2025, 1, 1), date(2025, 8, 30))[:1]
    source_trades = pl.DataFrame(schema={"market": pl.String, "date": pl.Date})
    preset = (tuner.ModelPreset("turnover", "turnover", {}),)
    before_grid, before_selections, before_daily = tuner.evaluate_walk_forward(
        panel, execution, source_trades, fold, presets=preset
    )
    oos_start = fold[0].oos_start
    mutated_execution = execution.with_columns(
        pl.when(pl.col("date") >= oos_start)
        .then(-0.9)
        .otherwise(pl.col("slot_return"))
        .alias("slot_return")
    )
    after_grid, after_selections, _ = tuner.evaluate_walk_forward(
        panel, mutated_execution, source_trades, fold, presets=preset
    )
    assert before_grid == after_grid
    assert {row["candidate_id"] for row in before_selections} == {
        row["candidate_id"] for row in after_selections
    }
    assert not any(row["promotion_eligible"] for row in before_selections)
    assert {row["deploy_action"] for row in before_selections} == {"cash"}
    assert not any(row["promotion_eligible"] for row in before_daily)
    assert {row["deploy_action"] for row in before_daily} == {"cash"}
    challenger = [row for row in before_selections if row["strategy"] == "trade_count_challenger"]
    assert {row["selection_status"] for row in challenger} == {"diagnostic_not_promoted"}
    assert not any(row["validation_gate_passed"] for row in challenger)
    assert not any(row["promotion_eligible"] for row in challenger)
    assert {row["deploy_action"] for row in challenger} == {"cash"}
    challenger_daily = [row for row in before_daily if row["strategy"] == "trade_count_challenger"]
    assert not any(row["promotion_eligible"] for row in challenger_daily)
    assert {row["deploy_action"] for row in challenger_daily} == {"cash"}


def test_locked_cash_has_zero_actual_fills() -> None:
    selections = pl.DataFrame(
        {
            "strategy": ["locked", "research_best_ungated"],
            "research_replay_action": ["cash", "trade"],
            "filled": [True, True],
        }
    )
    assert tuner._actual_strategy_rows(selections, "locked").filter(pl.col("filled")).height == 0
    assert (
        tuner._actual_strategy_rows(selections, "research_best_ungated")
        .filter(pl.col("filled"))
        .height
        == 1
    )


def test_acquisition_cache_resumes_and_expands_candidate_count(tmp_path: Path, monkeypatch) -> None:
    panel = _panel(days=2, symbols=1)
    calls = {"minute": 0, "funding": 0}
    start = datetime(2025, 1, 2)
    bars = [
        {
            "datetime": start + timedelta(minutes=i),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 101.0,
        }
        for i in range(1_440)
    ]

    def funding(*args):
        calls["funding"] += 1
        return []

    def minute(*args):
        calls["minute"] += 1
        return bars, bars, "test"

    def simulate(*args, **kwargs):
        return {
            "entry_time": start,
            "exit_time": start,
            "entry_price": kwargs["entry_trigger"],
            "exit_price": 101.0,
            "reason": "daily_close",
            "raw_return": 0.01,
            "funding_rate": 0.0,
            "funding_return": 0.0,
            "funding_events": 0,
            "funding_margin_shift": 0.0,
            "slot_return": 0.01,
            "liquidated": False,
            "mark_liquidation_breach": False,
            "ambiguous_minute": False,
        }

    assumptions = {"maintenance_margin_rate": 0.005}
    source = SimpleNamespace(
        _fetch_funding_rates=funding,
        _load_minute_day=minute,
        _simulate_trade=simulate,
        _execution_assumptions=lambda: assumptions.copy(),
    )
    first = tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert first.height == 4 * 5 * 2 * 3
    assert first.height > 4 * 5
    read_parquet = pl.read_parquet
    read_paths: list[Path] = []

    def tracked_read(path, *args, **kwargs):
        read_paths.append(Path(path))
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(tuner.pl, "read_parquet", tracked_read)
    second = tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert second.height == first.height
    assert calls == {"minute": 1, "funding": 1}
    assert read_paths == [tmp_path / "execution_candidates.parquet"]
    manifest = json.loads((tmp_path / "parts" / "S00USDT.manifest.json").read_text())
    assert manifest["context"]["exchange"] == "binance"
    assert manifest["context"]["data_root"] == str(tmp_path.resolve())
    assert manifest["context"]["execution_model"] == assumptions
    assert manifest["acquisition"]["funding"]["snapshot"]["sha256"]
    assert manifest["acquisition"]["days"][0]["trade_bars"]["sha256"]
    assert manifest["acquisition"]["days"][0]["mark_bars"]["sha256"]
    aggregate = tmp_path / "execution_candidates.parquet"
    aggregate_mtime = aggregate.stat().st_mtime_ns
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert aggregate.stat().st_mtime_ns == aggregate_mtime
    assert calls == {"minute": 1, "funding": 1}

    acquisition_manifest = tmp_path / "execution_acquisition_manifest.json"
    acquisition_manifest.write_text("not json")
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert calls == {"minute": 1, "funding": 1}
    assert json.loads(acquisition_manifest.read_text())["aggregate"]

    assumptions["maintenance_margin_rate"] = 0.006
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert calls == {"minute": 2, "funding": 2}
    manifest = json.loads((tmp_path / "parts" / "S00USDT.manifest.json").read_text())
    assert manifest["context"]["execution_model"] == assumptions

    tuner.acquire_execution_candidates(
        panel,
        cache_dir=tmp_path,
        data_root=tmp_path,
        source_module=source,
        refresh=True,
    )
    assert calls == {"minute": 3, "funding": 3}

    (tmp_path / "parts" / "S00USDT.manifest.json").unlink()
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path, data_root=tmp_path, source_module=source
    )
    assert calls == {"minute": 4, "funding": 4}


def test_execution_acquisition_rejects_non_binance_exchange(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only exchange='binance'"):
        tuner.acquire_execution_candidates(
            _panel(days=2, symbols=1),
            cache_dir=tmp_path,
            data_root=tmp_path,
            exchange="kraken",
            source_module=SimpleNamespace(),
        )


def test_acquisition_cache_transaction_is_process_exclusive(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    second_entered = context.Event()
    first = context.Process(
        target=_run_locked_acquisition,
        args=(str(tmp_path), first_entered, release_first),
    )
    second = context.Process(
        target=_run_locked_acquisition,
        args=(str(tmp_path), second_entered, None),
    )
    first.start()
    assert first_entered.wait(15)
    second.start()
    assert not second_entered.wait(0.2)
    release_first.set()
    assert second_entered.wait(15)
    first.join(15)
    second.join(15)
    assert first.exitcode == second.exitcode == 0


def test_symbol_acquisition_is_parallel_and_cache_resume_is_deterministic(
    tmp_path: Path,
) -> None:
    panel = _panel(days=2, symbols=4)
    state = {"active": 0, "maximum": 0, "funding": 0, "minute": 0}
    lock = threading.Lock()
    start = datetime(2025, 1, 2)
    bars = [
        {
            "datetime": start + timedelta(minutes=i),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 101.0,
        }
        for i in range(1_440)
    ]

    def funding(*args):
        with lock:
            state["active"] += 1
            state["funding"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return []

    def minute(*args):
        with lock:
            state["minute"] += 1
        return bars, bars, "test"

    source = SimpleNamespace(
        _fetch_funding_rates=funding,
        _load_minute_day=minute,
        _simulate_trade=lambda *args, **kwargs: {
            "slot_return": 0.01,
            "liquidated": False,
            "mark_liquidation_breach": False,
            "ambiguous_minute": False,
        },
    )
    first = tuner.acquire_execution_candidates(
        panel,
        cache_dir=tmp_path,
        data_root=tmp_path,
        source_module=source,
        workers=3,
    )
    aggregate = (tmp_path / "execution_candidates.parquet").read_bytes()
    second = tuner.acquire_execution_candidates(
        panel,
        cache_dir=tmp_path,
        data_root=tmp_path,
        source_module=source,
        workers=3,
    )
    assert state["active"] == 0
    assert 1 < state["maximum"] <= 3
    assert state["funding"] == state["minute"] == 4
    assert first.equals(second)
    assert (tmp_path / "execution_candidates.parquet").read_bytes() == aggregate
    assert first["market"].to_list() == sorted(first["market"].to_list())


def test_acquisition_worker_parser_bounds() -> None:
    parser = tuner.build_arg_parser()
    assert parser.parse_args(["--acquisition-workers", "8"]).acquisition_workers == 8
    with pytest.raises(SystemExit):
        parser.parse_args(["--acquisition-workers", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--acquisition-workers", "9"])


def test_local_input_identity_invalidates_execution_cache(tmp_path: Path) -> None:
    panel = _panel(days=2, symbols=1)
    calls = {"minute": 0, "funding": 0}
    start = datetime(2025, 1, 2)
    bars = [
        {
            "datetime": start + timedelta(minutes=i),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 101.0,
        }
        for i in range(1_440)
    ]

    def funding(*args):
        calls["funding"] += 1
        return []

    def minute(*args):
        calls["minute"] += 1
        return bars, bars, "test"

    source = SimpleNamespace(
        _fetch_funding_rates=funding,
        _load_minute_day=minute,
        _simulate_trade=lambda *args, **kwargs: {
            "slot_return": 0.01,
            "liquidated": False,
            "mark_liquidation_breach": False,
            "ambiguous_minute": False,
        },
    )
    local = (
        tmp_path
        / "exchange=binance"
        / "symbol=S00USDT"
        / "timeframe=1m"
        / "date=2025-01-02"
        / "part.parquet"
    )
    local.parent.mkdir(parents=True)
    local.write_bytes(b"first")
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path / "cache", data_root=tmp_path, source_module=source
    )
    local.write_bytes(b"second")
    tuner.acquire_execution_candidates(
        panel, cache_dir=tmp_path / "cache", data_root=tmp_path, source_module=source
    )
    assert calls == {"minute": 2, "funding": 2}


def test_batched_adverse_returns_match_legacy_per_scenario() -> None:
    days = [date(2025, 1, 1), date(2025, 1, 2)]
    selected = pl.DataFrame(
        {
            "market": ["AUSDT", "BUSDT", "AUSDT", "BUSDT"],
            "date": [days[0], days[0], days[1], days[1]],
            "filled": [True, False, True, True],
        }
    )
    lookups = {
        scenario: {
            ("AUSDT", days[0]): 0.01 + index,
            ("AUSDT", days[1]): -0.02 - index,
            ("BUSDT", days[1]): 0.03 + index,
        }
        for index, scenario in enumerate(tuner.ADVERSE_SCENARIOS)
    }
    batched = tuner._selected_adverse_returns(selected, lookups, position_cap=2)
    for scenario in tuner.ADVERSE_SCENARIOS:
        assert batched[scenario]["base"] == tuner._selected_lookup_returns(
            selected, lookups[scenario], position_cap=2, doubled_cost=False
        )
        assert batched[scenario]["doubled_cost"] == tuner._selected_lookup_returns(
            selected, lookups[scenario], position_cap=2, doubled_cost=True
        )
