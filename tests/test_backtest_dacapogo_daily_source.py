from __future__ import annotations

import importlib.util
import io
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research" / "backtest_dacapogo_daily_source.py"
SPEC = importlib.util.spec_from_file_location("backtest_dacapogo_daily_source", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load backtest_dacapogo_daily_source module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_binance_request_timeout_is_retried(monkeypatch) -> None:
    calls = 0

    def urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        if calls == 1:
            raise urllib.error.HTTPError("https://example.test", 408, "timeout", Message(), None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    assert MODULE._fetch_json("https://example.test") == {"ok": True}
    assert calls == 2


def test_binance_connection_reset_is_retried(monkeypatch) -> None:
    calls = 0

    def urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("reset")
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    assert MODULE._fetch_json("https://example.test") == {"ok": True}
    assert calls == 2


def test_fapi_daily_rejects_an_unclosed_candle(monkeypatch) -> None:
    day = datetime.now(UTC).date() + timedelta(days=1)
    open_ms = int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    monkeypatch.setattr(
        MODULE,
        "_fetch_json",
        lambda _url: [
            [open_ms, "100", "101", "99", "100", "1", open_ms + MODULE.DAY_MS - 1, "1000"]
        ],
    )

    with pytest.raises(ValueError, match="fully closed UTC day"):
        MODULE._fetch_fapi_daily("BTCUSDT", load_start=day, end=day)


def test_fapi_daily_rejects_a_malformed_close_timestamp(monkeypatch) -> None:
    day = datetime(2026, 1, 1, tzinfo=UTC).date()
    open_ms = int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    monkeypatch.setattr(
        MODULE,
        "_fetch_json",
        lambda _url: [
            [open_ms, "100", "101", "99", "100", "1", open_ms + MODULE.DAY_MS - 2, "1000"]
        ],
    )

    with pytest.raises(ValueError, match="daily coverage failed"):
        MODULE._fetch_fapi_daily("BTCUSDT", load_start=day, end=day)


@pytest.mark.parametrize(
    ("open_", "high", "low", "close", "value"),
    [
        (100, 101, 99, 102, 1_000),
        (100, 101, 99, 98, 1_000),
        (100, 101, 0, 100, 1_000),
        (100, float("inf"), 99, 100, 1_000),
    ],
)
def test_fapi_daily_rejects_invalid_ohlcv(
    monkeypatch, open_: float, high: float, low: float, close: float, value: float
) -> None:
    day = datetime(2026, 1, 1, tzinfo=UTC).date()
    open_ms = int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    monkeypatch.setattr(
        MODULE,
        "_fetch_json",
        lambda _url: [
            [
                open_ms,
                str(open_),
                str(high),
                str(low),
                str(close),
                "1",
                open_ms + MODULE.DAY_MS - 1,
                str(value),
            ]
        ],
    )

    with pytest.raises(ValueError, match="daily coverage failed"):
        MODULE._fetch_fapi_daily("BTCUSDT", load_start=day, end=day)


def _bar(minute: int, *, open_: float, high: float, low: float, close: float) -> dict:
    return {
        "datetime": datetime(2026, 1, 1) + timedelta(minutes=minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def test_parse_leverages_sorts_and_deduplicates_valid_values() -> None:
    assert MODULE._parse_leverages("10, 1,10,125") == (1, 10, 125)
    assert MODULE._parse_symbols("BTCUSDT, BTC/USDT,ETH-USDT") == ("BTCUSDT", "ETHUSDT")


@pytest.mark.parametrize("value", ["", "0", "126", "1,126"])
def test_parse_leverages_rejects_values_outside_exchange_range(value: str) -> None:
    with pytest.raises(ValueError, match="integers from 1 to 125"):
        MODULE._parse_leverages(value)


def test_simulate_trade_enters_when_minute_high_exactly_reaches_four_percent_trigger() -> None:
    bars = [
        _bar(0, open_=100.0, high=103.99, low=99.0, close=103.0),
        _bar(1, open_=103.0, high=104.0, low=103.0, close=104.0),
    ]

    result = MODULE._simulate_trade(
        bars,
        bars,
        entry_trigger=104.0,
        leverage=1,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )

    assert result["entry_time"] == bars[1]["datetime"]
    assert result["entry_price"] == 104.0


@pytest.mark.parametrize(
    ("priority", "expected_reason", "expected_price"),
    [
        ("stop_first", "stop_loss", 100.0 * (1.0 - MODULE.SL)),
        ("tp_first", "take_profit", 100.0 * (1.0 + MODULE.TP)),
    ],
)
def test_simulate_trade_resolves_same_minute_stop_and_take_profit_by_priority(
    priority: str,
    expected_reason: str,
    expected_price: float,
) -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=101.0, low=99.0, close=100.0),
    ]

    result = MODULE._simulate_trade(
        bars,
        bars,
        entry_trigger=100.0,
        leverage=2,
        variant="tp_sl",
        same_bar_priority=priority,
        funding_rates=[],
    )

    assert result["reason"] == expected_reason
    assert result["exit_price"] == pytest.approx(expected_price)
    assert result["ambiguous_minute"] is True


def test_simulate_trade_applies_cost_and_actual_funding_before_leverage() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=102.0, low=100.0, close=102.0),
    ]

    funding_time = int(
        (bars[0]["datetime"] + timedelta(seconds=30)).replace(tzinfo=UTC).timestamp() * 1000
    )
    result = MODULE._simulate_trade(
        bars,
        bars,
        entry_trigger=100.0,
        leverage=3,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[(funding_time, 0.001, 100.0)],
    )

    assert result["raw_return"] == pytest.approx(0.02)
    assert result["funding_rate"] == pytest.approx(0.001)
    assert result["funding_return"] == pytest.approx(0.001)
    assert result["slot_return"] == pytest.approx(3 * (0.02 - MODULE.COST - 0.001))


def test_simulate_trade_accepts_tuned_tp_sl_and_cost_without_changing_defaults() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=101.0, low=99.25, close=100.0),
    ]
    kwargs = {
        "entry_trigger": 100.0,
        "leverage": 2,
        "same_bar_priority": "tp_first",
        "funding_rates": [],
    }

    default = MODULE._simulate_trade(bars, bars, **kwargs, variant="tp_sl")
    explicit_default = MODULE._simulate_trade(
        bars,
        bars,
        **kwargs,
        variant="tp_sl",
        stop_pct=MODULE.SL,
        take_profit_pct=MODULE.TP,
        round_trip_cost=MODULE.COST,
    )
    tuned_tp = MODULE._simulate_trade(bars, bars, **kwargs, variant="tp_sl", take_profit_pct=0.005)
    tuned_sl = MODULE._simulate_trade(bars, bars, **kwargs, variant="close", stop_pct=0.005)
    tuned_cost = MODULE._simulate_trade(
        bars, bars, **kwargs, variant="tp_sl", round_trip_cost=0.005
    )

    assert explicit_default == default
    assert tuned_tp["exit_price"] == pytest.approx(100.5)
    assert tuned_sl["exit_price"] == pytest.approx(99.5)
    assert tuned_cost["slot_return"] == pytest.approx(2 * (tuned_cost["raw_return"] - 0.005))


def test_positive_funding_can_trigger_isolated_slot_liquidation() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=100.0, low=100.0, close=100.0),
    ]
    mark_bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=99.9, high=99.9, low=99.9, close=99.9),
    ]
    funding_time = int(bars[1]["datetime"].replace(tzinfo=UTC).timestamp() * 1000)

    result = MODULE._simulate_trade(
        bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[(funding_time, 0.003, 99.9)],
    )

    assert result["reason"] == "liquidation_funding"
    assert result["funding_events"] == 1
    assert result["funding_return"] == pytest.approx(0.003 * 99.9 / 100.0)
    assert result["funding_margin_shift"] > 0.3
    assert result["mark_liquidation_breach"] is True
    assert result["slot_return"] == -1.0


def test_negative_funding_cannot_rescue_an_existing_mark_open_liquidation() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=100.0, low=100.0, close=100.0),
    ]
    mark_bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=99.7, high=99.7, low=99.7, close=99.7),
    ]
    funding_time = int(bars[1]["datetime"].replace(tzinfo=UTC).timestamp() * 1000)

    result = MODULE._simulate_trade(
        bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[(funding_time, -0.01, 99.7)],
    )

    assert result["reason"] == "liquidation_gap"
    assert result["funding_events"] == 0
    assert result["slot_return"] == -1.0


def test_entry_at_open_liquidates_before_later_negative_funding() -> None:
    bar = _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0)
    funding_time = int(
        (bar["datetime"] + timedelta(seconds=30)).replace(tzinfo=UTC).timestamp() * 1000
    )

    result = MODULE._simulate_trade(
        [bar],
        [_bar(0, open_=99.7, high=100.0, low=99.7, close=100.0)],
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[(funding_time, -0.01, 99.7)],
    )

    assert result["reason"] == "liquidation_gap"
    assert result["funding_events"] == 0
    assert result["mark_liquidation_breach"] is True


def test_negative_funding_credit_lowers_later_liquidation_boundary() -> None:
    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(2, open_=100.0, high=100.0, low=100.0, close=100.0),
    ]
    mark_bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(2, open_=99.5, high=99.5, low=99.5, close=99.5),
    ]
    funding_time = int(bars[1]["datetime"].replace(tzinfo=UTC).timestamp() * 1000)
    without_credit = MODULE._simulate_trade(
        bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )
    with_credit = MODULE._simulate_trade(
        bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[(funding_time, -0.003, 100.0)],
    )

    assert without_credit["reason"] == "liquidation_gap"
    assert with_credit["reason"] == "daily_close"
    assert with_credit["funding_events"] == 1
    assert with_credit["funding_margin_shift"] < -0.3
    assert with_credit["mark_liquidation_breach"] is False


def test_simulate_trade_reports_entry_minute_path_order_scenarios() -> None:
    bars = [_bar(0, open_=100.0, high=104.5, low=99.0, close=104.0)]
    adverse = MODULE._simulate_trade(
        bars,
        bars,
        entry_trigger=104.0,
        leverage=1,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )
    favorable = MODULE._simulate_trade(
        bars,
        bars,
        entry_trigger=104.0,
        leverage=1,
        variant="close",
        same_bar_priority="tp_first",
        funding_rates=[],
    )

    assert adverse["reason"] == "stop_loss"
    assert favorable["reason"] == "daily_close"
    assert adverse["ambiguous_minute"] is favorable["ambiguous_minute"] is True
    assert isinstance(adverse["funding_rate"], float)
    assert isinstance(adverse["funding_return"], float)


def test_simulate_trade_rejects_missing_or_misaligned_mark_bars() -> None:
    bars = [_bar(0, open_=100.0, high=100.0, low=100.0, close=100.0)]
    with pytest.raises(ValueError, match="timestamp-aligned"):
        MODULE._simulate_trade(
            bars,
            [],
            entry_trigger=100.0,
            leverage=1,
            variant="close",
            same_bar_priority="stop_first",
            funding_rates=[],
        )


def test_trade_low_does_not_liquidate_when_mark_price_stays_above_boundary() -> None:
    trade_bars = [_bar(0, open_=100.0, high=100.0, low=90.0, close=100.0)]
    mark_bars = [_bar(0, open_=100.0, high=100.0, low=100.0, close=100.0)]

    result = MODULE._simulate_trade(
        trade_bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )

    assert result["reason"] == "stop_loss"
    assert result["liquidated"] is False
    assert result["mark_liquidation_breach"] is False


def test_mark_low_liquidates_at_modeled_boundary_when_trade_price_stays_above() -> None:
    model = MODULE._execution_model(20)
    liquidation_price = model.liquidation_price(qty=1.0, entry_price=100.0)
    assert liquidation_price is not None
    result = MODULE._simulate_trade(
        [_bar(0, open_=100.0, high=100.0, low=100.0, close=100.0)],
        [_bar(0, open_=100.0, high=100.0, low=90.0, close=100.0)],
        entry_trigger=100.0,
        leverage=20,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )

    assert result["reason"] == "liquidation"
    assert result["exit_price"] == pytest.approx(liquidation_price)
    assert result["slot_return"] == -1.0


@pytest.mark.parametrize(
    ("priority", "expected_reason"),
    [("stop_first", "liquidation"), ("tp_first", "stop_loss")],
)
def test_same_minute_trade_stop_and_mark_liquidation_use_path_order_scenario(
    priority: str,
    expected_reason: str,
) -> None:
    trade_bars = [_bar(0, open_=100.0, high=100.0, low=99.0, close=100.0)]
    mark_bars = [_bar(0, open_=100.0, high=100.0, low=90.0, close=100.0)]

    result = MODULE._simulate_trade(
        trade_bars,
        mark_bars,
        entry_trigger=100.0,
        leverage=20,
        variant="close",
        same_bar_priority=priority,
        funding_rates=[],
    )

    assert result["reason"] == expected_reason
    assert result["ambiguous_minute"] is True
    assert result["mark_liquidation_breach"] is True


def test_entry_minute_mark_close_below_liquidation_cannot_be_ignored() -> None:
    result = MODULE._simulate_trade(
        [_bar(0, open_=100.0, high=104.0, low=99.0, close=104.0)],
        [_bar(0, open_=100.0, high=100.0, low=90.0, close=90.0)],
        entry_trigger=104.0,
        leverage=20,
        variant="close",
        same_bar_priority="tp_first",
        funding_rates=[],
    )

    assert result["reason"] == "stop_loss"
    assert result["mark_liquidation_breach"] is True


def test_funding_error_payload_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_fetch_json", lambda _url: {"code": -1, "msg": "failed"})
    with pytest.raises(ValueError, match="funding response"):
        MODULE._fetch_funding_rates(
            "BTCUSDT", datetime(2026, 1, 1).date(), datetime(2026, 1, 2).date()
        )


def test_empty_or_nonpositive_funding_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "_fetch_json", lambda _url: [])
    with pytest.raises(ValueError, match="empty Binance funding"):
        MODULE._fetch_funding_rates(
            "BTCUSDT", datetime(2026, 1, 1).date(), datetime(2026, 1, 2).date()
        )

    bars = [
        _bar(0, open_=100.0, high=100.0, low=100.0, close=100.0),
        _bar(1, open_=100.0, high=100.0, low=100.0, close=100.0),
    ]
    funding_time = int(
        (bars[0]["datetime"] + timedelta(seconds=30)).replace(tzinfo=UTC).timestamp() * 1000
    )
    with pytest.raises(ValueError, match="nonpositive funding mark"):
        MODULE._simulate_trade(
            bars,
            bars,
            entry_trigger=100.0,
            leverage=1,
            variant="close",
            same_bar_priority="stop_first",
            funding_rates=[(funding_time, 0.001, 0.0)],
        )


def test_mark_price_gap_liquidates_while_fill_uses_trade_price() -> None:
    entry_price = 100.0
    model = MODULE._execution_model(125)
    liquidation_price = model.liquidation_price(qty=1.0, entry_price=entry_price)
    assert liquidation_price is not None
    bars = [
        _bar(0, open_=entry_price, high=entry_price, low=entry_price, close=entry_price),
        _bar(1, open_=entry_price, high=entry_price, low=entry_price, close=entry_price),
    ]
    mark_bars = [
        _bar(0, open_=entry_price, high=entry_price, low=entry_price, close=entry_price),
        _bar(
            1,
            open_=liquidation_price - 0.01,
            high=liquidation_price - 0.01,
            low=liquidation_price - 0.01,
            close=liquidation_price - 0.01,
        ),
    ]

    result = MODULE._simulate_trade(
        bars,
        mark_bars,
        entry_trigger=entry_price,
        leverage=125,
        variant="close",
        same_bar_priority="stop_first",
        funding_rates=[],
    )

    assert result["reason"] == "liquidation_gap"
    assert result["exit_price"] == pytest.approx(entry_price)
    assert result["liquidated"] is True
    assert result["slot_return"] == -1.0
