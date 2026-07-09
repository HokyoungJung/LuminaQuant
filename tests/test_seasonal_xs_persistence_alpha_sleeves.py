"""Deterministic build-gate + hygiene tests for the seasonal-persistence lane.

Direct class import only (no ``@register`` on this lane).  The build gate proves
the same-week-of-quarter XS-relative object is a DIFFERENT statistical object
from every shipped seasonal incumbent, and that it is neutral to both absolute
calendar effects (the graveyard-#1 dodge) and TSMOM drift:

- leg 1: candidate LONGs SEAS_UP / SHORTs SEAS_DN (divergent action, both books).
- leg 2: ``CalendarSeasonalityOverlayStrategy`` no-ops on a crypto-only universe
  (``index_symbol == ''``) with ``_is_turn_of_month`` False on the decision date;
  ``IntradaySeasonalMomentumRiderStrategy`` pools every midnight bar into slot 0;
  ``OvernightSessionReturnRiderStrategy`` pools every bar into one session --
  each blind to the week-of-quarter structure (the WHY, pinned at the mapping).
- leg 2b: incumbent-LIVE positive control -- the calendar overlay WITH its index
  present on a turn-of-month date emits its tilt (harness wiring proof).
- leg 3: TIME-DEMEANING invariance -- a constant +drift on every week of SEAS_UP
  leaves its seasonal score and the emitted book unchanged (no TSMOM leakage).
- leg 4: ABSOLUTE-CALENDAR neutrality -- a common week-3 return added to ALL
  symbols leaves the XS ranking identical and the book net-neutral.

All randomness is a small seeded LCG (no ``random`` module); the piecewise-
constant-within-week fixture is fully deterministic.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.calendar_overlay_alpha_sleeves import (
    CalendarSeasonalityOverlayStrategy,
    _is_turn_of_month,
)
from lumina_quant.strategies.intraday_overnight_alpha_sleeves import (
    IntradaySeasonalMomentumRiderStrategy,
    OvernightSessionReturnRiderStrategy,
)
from lumina_quant.strategies.seasonal_xs_persistence_alpha_sleeves import (
    CrossSectionalSeasonalPersistenceStrategy,
    _SEASONAL_XS_PERSISTENCE_SLICE,
    _quarter_id,
    _week_of_quarter,
)
from lumina_quant.tuning import HyperParam

# 2024-01-01 .. 2025-04-22 (first bar of week-of-quarter 3 of the live quarter).
_START = date(2024, 1, 1)
_DECISION = date(2025, 4, 22)


def _dates() -> list[date]:
    out: list[date] = []
    cursor = _START
    while cursor <= _DECISION:
        out.append(cursor)
        cursor += timedelta(days=1)
    return out


_DATES = _dates()


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _woq(dt: date) -> int | None:
    return _week_of_quarter(datetime(dt.year, dt.month, dt.day, tzinfo=UTC))


def _qid(dt: date) -> int:
    return _quarter_id(datetime(dt.year, dt.month, dt.day, tzinfo=UTC))


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _event(symbol: str, dt: date, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"{dt.isoformat()}T00:00:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


# --------------------------------------------------------------------------- #
# piecewise-constant-within-week fixture generators
# --------------------------------------------------------------------------- #


def _seasonal_series(
    bump: float, noise_seed: int, *, extra_drift: float = 0.0, common_week3: float = 0.0
) -> list[float]:
    """Constant within each quarter-aligned week; a +``bump`` weekly step in bucket 3."""
    gen = _lcg_stream(noise_seed)
    closes: list[float] = []
    price = 100.0
    prev_wk: tuple[int, int] | None = None
    for dt in _DATES:
        week = (_qid(dt), _woq(dt))
        if prev_wk is None:
            prev_wk = week
        elif week != prev_wk:
            step = (bump + common_week3) if week[1] == 3 else 0.0
            step += (next(gen) - 0.5) * 0.001 + extra_drift
            price *= math.exp(step)
            prev_wk = week
        closes.append(price)
    return closes


def _filler_series(noise_seed: int, *, common_week3: float = 0.0) -> list[float]:
    gen = _lcg_stream(noise_seed)
    closes: list[float] = []
    price = 100.0
    prev_wk: tuple[int, int] | None = None
    for dt in _DATES:
        week = (_qid(dt), _woq(dt))
        if prev_wk is None:
            prev_wk = week
        elif week != prev_wk:
            step = (common_week3 if week[1] == 3 else 0.0) + (next(gen) - 0.5) * 0.01
            price *= math.exp(step)
            prev_wk = week
        closes.append(price)
    return closes


def _base_universe(*, common_week3: float = 0.0) -> tuple[list[str], dict[str, list[float]]]:
    symbols = ["SEAS_UP/USDT", "SEAS_DN/USDT", *[f"N{i}/USDT" for i in range(6)]]
    series = {
        "SEAS_UP/USDT": _seasonal_series(0.02, 1, common_week3=common_week3),
        "SEAS_DN/USDT": _seasonal_series(-0.02, 2, common_week3=common_week3),
    }
    for i in range(6):
        series[f"N{i}/USDT"] = _filler_series(100 + i, common_week3=common_week3)
    return symbols, series


def _strategy(symbols: list[str], **overrides: Any) -> CrossSectionalSeasonalPersistenceStrategy:
    params: dict[str, Any] = dict(
        seasonal_lookback_quarters=6,
        min_history_quarters=2,
        vol_window=20,
        quantile_pct=0.25,
        min_hold_weeks=0,
        min_symbols=5,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    params.update(overrides)
    return CrossSectionalSeasonalPersistenceStrategy(_Bars(symbols), _Queue(), **params)


def _run(symbols: list[str], series: dict[str, list[float]], **overrides: Any):
    strategy = _strategy(symbols, **overrides)
    for idx, dt in enumerate(_DATES):
        for symbol in symbols:
            strategy.calculate_signals(_event(symbol, dt, series[symbol][idx]))
    return strategy


# --------------------------------------------------------------------------- #
# leg 1 -- candidate divergent action
# --------------------------------------------------------------------------- #


def test_leg1_candidate_longs_seasonal_up_shorts_seasonal_down() -> None:
    symbols, series = _base_universe()
    strategy = _run(symbols, series)
    side = _final_side(strategy.events.items)
    assert side.get("SEAS_UP/USDT") == "LONG", side
    assert side.get("SEAS_DN/USDT") == "SHORT", side
    # both book sides non-empty
    assert any(v == "LONG" for v in side.values())
    assert any(v == "SHORT" for v in side.values())


# --------------------------------------------------------------------------- #
# leg 2 -- named seasonal incumbents are blind to the week-of-quarter object
# --------------------------------------------------------------------------- #


def test_leg2_calendar_overlay_noops_on_crypto_universe() -> None:
    symbols, series = _base_universe()
    overlay = CalendarSeasonalityOverlayStrategy(_Bars(symbols), _Queue())
    # No index perp present -> the overlay resolves no symbol and is a total no-op.
    assert overlay.index_symbol == ""
    for idx, dt in enumerate(_DATES):
        for symbol in symbols:
            overlay.calculate_signals(_event(symbol, dt, series[symbol][idx]))
    assert overlay.events.items == []
    # The decision date is NOT a turn-of-month day (the day-of-week / month
    # tilts the overlay could otherwise fire on are absent here).
    decision_dt = datetime(_DECISION.year, _DECISION.month, _DECISION.day, tzinfo=UTC)
    assert _is_turn_of_month(decision_dt, pre_days=3, post_days=3) is False


def test_leg2_intraday_and_overnight_riders_pool_the_seasonal_structure() -> None:
    up = _seasonal_series(0.02, 1)
    intraday = IntradaySeasonalMomentumRiderStrategy(_Bars(["SEAS_UP/USDT"]), _Queue())
    overnight = OvernightSessionReturnRiderStrategy(_Bars(["SEAS_UP/USDT"]), _Queue())
    for idx, dt in enumerate(_DATES):
        intraday.calculate_signals(_event("SEAS_UP/USDT", dt, up[idx]))
        overnight.calculate_signals(_event("SEAS_UP/USDT", dt, up[idx]))
    # Every 00:00-UTC bar maps to intraday slot 0 -> the whole week-of-quarter
    # structure collapses into a single slot; the rider cannot see it.
    assert set(intraday._slot_stats["SEAS_UP/USDT"].keys()) <= {0}
    # Every bar lands in the single "overnight" session (default 0..8h) -> the
    # session-return rider likewise cannot resolve week-of-quarter structure.
    assert set(overnight._session_stats["SEAS_UP/USDT"].keys()) <= {"overnight"}


def test_leg2b_calendar_overlay_incumbent_live_positive_control() -> None:
    overlay = CalendarSeasonalityOverlayStrategy(_Bars(["SPYUSDT", "SEAS_UP/USDT"]), _Queue())
    assert overlay.index_symbol == "SPYUSDT"
    overlay.calculate_signals(_event("SPYUSDT", date(2025, 3, 28), 100.0))
    overlay.calculate_signals(_event("SPYUSDT", date(2025, 4, 1), 101.0))  # turn-of-month
    kinds = [str(sig.signal_type).upper() for sig in overlay.events.items]
    assert "LONG" in kinds, kinds


# --------------------------------------------------------------------------- #
# leg 3 -- time-demeaning invariance (no TSMOM leakage)
# --------------------------------------------------------------------------- #


def test_leg3_time_demeaning_invariance() -> None:
    symbols, base = _base_universe()
    baseline = _run(symbols, base)
    base_score = baseline._symbol_score(baseline._state["SEAS_UP/USDT"], 3)

    drifted = dict(base)
    drifted["SEAS_UP/USDT"] = _seasonal_series(0.02, 1, extra_drift=0.003)
    shifted = _run(symbols, drifted)
    shifted_score = shifted._symbol_score(shifted._state["SEAS_UP/USDT"], 3)

    assert base_score is not None and shifted_score is not None
    # Adding a constant +drift to every week strips out under the time-demeaning.
    assert abs(base_score - shifted_score) < 1e-9
    assert _final_side(baseline.events.items) == _final_side(shifted.events.items)


# --------------------------------------------------------------------------- #
# leg 4 -- absolute-calendar neutrality (graveyard-#1 dodge)
# --------------------------------------------------------------------------- #


def test_leg4_absolute_calendar_neutrality() -> None:
    symbols, base = _base_universe()
    baseline = _run(symbols, base)
    base_side = _final_side(baseline.events.items)

    symbols_c, common = _base_universe(common_week3=0.02)
    shifted = _run(symbols_c, common)
    shifted_side = _final_side(shifted.events.items)

    # A market-wide week-3 return added to EVERY symbol leaves the XS ranking
    # (and thus the emitted book) identical: XS-demeaning strips the common term.
    assert base_side == shifted_side
    longs = sum(1 for v in shifted_side.values() if v == "LONG")
    shorts = sum(1 for v in shifted_side.values() if v == "SHORT")
    assert longs == shorts and longs >= 1  # net-neutral book


# --------------------------------------------------------------------------- #
# leg 5 -- hygiene
# --------------------------------------------------------------------------- #


def test_leg5_determinism_two_runs_identical_signals() -> None:
    symbols, series = _base_universe()

    def _stream() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _run(symbols, series)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _stream()
    second = _stream()
    assert first == second
    assert first, "expected at least one signal"


def test_leg5_state_roundtrip_lossless() -> None:
    symbols, series = _base_universe()
    strategy = _run(symbols, series)
    snapshot = strategy.get_state()
    restored = _strategy(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_leg5_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = _strategy(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("nope")  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"last_decision_week": "bad", "symbol_state": {"A/USDT": "nope"}})
    strategy.set_state(
        {
            "last_decision_week": [1],
            "tick": "x",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), 12.5, None],
                    "cur_week_key": "bad",
                    "cur_week_woq": "bad",
                    "pending_sum": "bad",
                    "pending_count": {"bad": "bad"},
                    "buckets": {"3": ["x", 1.0, None]},
                    "mode": 999,
                    "weeks_held": "oops",
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_leg5_degenerate_input_never_raises() -> None:
    strategy = _strategy(["Z/USDT"], min_symbols=5)
    strategy.calculate_signals(_event("Z/USDT", date(2025, 1, 1), 0.0))
    strategy.calculate_signals(_event("Z/USDT", date(2025, 1, 2), float("nan")))
    strategy.calculate_signals(_event("Z/USDT", date(2025, 1, 3), float("inf")))
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", symbol="Z/USDT", bars_1s={}, time="t0")
    )
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert [s for s in strategy.events.items if str(s.signal_type).upper() != "EXIT"] == []


def test_leg5_min_history_floor_skips_young_admits_two_quarter_alt() -> None:
    # YOUNG: real prices only from Q3-start (2 past-quarter week-3 obs -> admitted).
    # VERYYOUNG: real prices only from mid-Q4 (< 2 week-3 obs -> skipped).
    symbols = [
        "SEAS_UP/USDT",
        "SEAS_DN/USDT",
        "N0/USDT",
        "N1/USDT",
        "N2/USDT",
        "YOUNG/USDT",
        "VERYYOUNG/USDT",
    ]
    series = {
        "SEAS_UP/USDT": _seasonal_series(0.02, 1),
        "SEAS_DN/USDT": _seasonal_series(-0.02, 2),
        "N0/USDT": _filler_series(100),
        "N1/USDT": _filler_series(101),
        "N2/USDT": _filler_series(102),
        "YOUNG/USDT": _seasonal_series(0.02, 3),
        "VERYYOUNG/USDT": _seasonal_series(0.02, 4),
    }
    young_start = date(2024, 10, 1)
    veryyoung_start = date(2025, 2, 1)
    for idx, dt in enumerate(_DATES):
        if dt < young_start:
            series["YOUNG/USDT"][idx] = 0.0  # <= min_price -> not recorded (young)
        if dt < veryyoung_start:
            series["VERYYOUNG/USDT"][idx] = 0.0
    strategy = _run(symbols, series)
    young_score = strategy._symbol_score(strategy._state["YOUNG/USDT"], 3)
    veryyoung_score = strategy._symbol_score(strategy._state["VERYYOUNG/USDT"], 3)
    assert young_score is not None  # >= 2 quarter observations -> admitted
    assert veryyoung_score is None  # < min_history_quarters -> skipped


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalSeasonalPersistenceStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in ("seasonal_lookback_quarters", "min_history_quarters", "min_hold_weeks"):
        assert required in schema


def test_slice_multi_timeframe_scaling() -> None:
    """1h/4h cells mirror 1d; only the bar-denominated vol window scales (x24/x6)."""
    slice_map = _SEASONAL_XS_PERSISTENCE_SLICE
    assert set(slice_map.keys()) == {"1h", "4h", "1d"}
    base = slice_map["1d"]
    names = tuple(spec["variant"] for spec in base)
    assert names == ("k6", "k4")
    for timeframe, variants in slice_map.items():
        assert len(variants) == len(base), timeframe
        assert tuple(spec["variant"] for spec in variants) == names, timeframe

    # The quarter-aligned weekly decision clock is TIMESTAMP-based, so the
    # quarter/week horizons stay timeframe-invariant; the ONLY bar-denominated
    # param is the realized-vol window, which scales x6 (4h) / x24 (1h).
    unchanged_keys = (
        "seasonal_lookback_quarters",
        "min_history_quarters",
        "quantile_pct",
        "min_hold_weeks",
        "min_symbols",
        "allow_short",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
    )
    for factor, timeframe in ((6, "4h"), (24, "1h")):
        for base_spec, tf_spec in zip(base, slice_map[timeframe], strict=True):
            assert tf_spec["vol_window"] == base_spec["vol_window"] * factor, timeframe
            assert tf_spec["vol_window"] <= 9000, timeframe
            for key in unchanged_keys:
                assert tf_spec[key] == base_spec[key], (timeframe, key)
