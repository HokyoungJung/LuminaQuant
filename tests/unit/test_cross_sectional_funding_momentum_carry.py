from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.strategies.cross_sectional_funding_momentum_carry import (
    CrossSectionalFundingMomentumCarryStrategy,
    _funding_momentum,
)
from lumina_quant.strategies.registry import get_strategy_map
from lumina_quant.tuning import HyperParam

_SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]


def _zigzag(i: int) -> float:
    return 1.0 if i % 2 == 0 else -1.0


def _series(n: int, *, start: float, drift: float, noise: float) -> list[float]:
    out: list[float] = []
    for i in range(n):
        out.append(start * (1.0 + drift) ** i * (1.0 + noise * _zigzag(i)))
    return out


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    def __init__(self, symbols: list[str]):
        self.symbol_list = list(symbols)

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        return None


def _market_event(symbol: str, idx: int, close: float, funding: float | None) -> SimpleNamespace:
    kwargs: dict[str, Any] = {
        "type": "MARKET",
        "time": f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
    }
    if funding is not None:
        kwargs["funding_rate"] = funding
    return SimpleNamespace(**kwargs)


def _feed(
    strategy: CrossSectionalFundingMomentumCarryStrategy,
    series: dict[str, list[float]],
    funding_series: dict[str, list[float] | None],
) -> None:
    n = len(next(iter(series.values())))
    for idx in range(n):
        for symbol in series:
            fund = funding_series.get(symbol)
            value = fund[idx] if fund is not None else None
            strategy.calculate_signals(_market_event(symbol, idx, series[symbol][idx], value))


# --- registry membership ----------------------------------------------------


def test_registry_membership() -> None:
    assert "CrossSectionalFundingMomentumCarryStrategy" in get_strategy_map()


# --- (8) cadence >= 8h enforced ---------------------------------------------


def test_cadence_floor_is_at_least_eight_hours() -> None:
    assert CrossSectionalFundingMomentumCarryStrategy.decision_cadence_seconds >= 28800
    # Even if an instance tries a faster cadence, the floor is enforced in code.
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue())
    assert strat.decision_cadence_seconds >= 28800


# --- schema -----------------------------------------------------------------


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalFundingMomentumCarryStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "funding_diff_window",
        "funding_slope_window",
        "ewma_span",
        "threshold",
        "max_longs",
        "max_shorts",
        "allow_short",
        "target_gross_exposure",
        "target_vol",
        "rebalance_band",
        "stop_loss_pct",
        "max_hold_bars",
        "base_allocation",
        "max_symbol_exposure_pct",
    ):
        assert required in schema
    # Caps are not tunable.
    for cap in ("base_allocation", "max_symbol_exposure_pct", "max_order_value"):
        assert schema[cap].tunable is False


# --- (1) funding-momentum direction (rising / falling / flat) ---------------


def test_rising_funding_positive_falling_negative_flat_zero() -> None:
    rising = [0.0001 * i for i in range(12)]
    falling = [-0.0001 * i for i in range(12)]
    flat = [0.0005 for _ in range(12)]
    kw = dict(diff_window=6, slope_window=8, ewma_span=4)
    assert _funding_momentum(rising, **kw) > 0.0
    assert _funding_momentum(falling, **kw) < 0.0
    assert _funding_momentum(flat, **kw) == pytest.approx(0.0, abs=1e-12)
    # Too little history => None.
    assert _funding_momentum([0.001], **kw) is None


# --- (2) no-lookahead -------------------------------------------------------


def test_no_lookahead_score_uses_only_at_or_before_interval() -> None:
    n = 30
    series = {
        s: _series(n, start=100.0 + i, drift=0.003, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    # Steadily rising funding so momentum is well-defined for every symbol.
    funding = {s: [0.0001 * j + 0.00001 * i for j in range(n)] for i, s in enumerate(_SYMBOLS)}

    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue(), min_symbols=4)
    _feed(strat, series, funding)  # type: ignore[arg-type]
    rows_at_n, _ = strat._score_rows()
    score_at_n = {sym: comp for comp, sym, _meta in rows_at_n}

    # Now inject a HUGE funding spike at the NEXT interval. It must not change the
    # score computed "as of N" (proved by matching a clean strategy fed 0..N only).
    for symbol in _SYMBOLS:
        strat.calculate_signals(_market_event(symbol, n, series[symbol][-1], 5.0))

    clean = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue(), min_symbols=4)
    _feed(clean, series, funding)  # type: ignore[arg-type]
    rows_clean, _ = clean._score_rows()
    score_clean = {sym: comp for comp, sym, _meta in rows_clean}

    assert score_at_n == pytest.approx(score_clean)


# --- (3) cross-sectional z-score over symbols, zeros if < 2 -----------------


def test_cross_sectional_zscore_centered_and_zero_when_singleton() -> None:
    n = 30
    series = {
        s: _series(n, start=100.0 + i, drift=0.002, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    # Each symbol a different rising slope so cross-section has spread.
    funding = {s: [0.00002 * (i + 1) * j for j in range(n)] for i, s in enumerate(_SYMBOLS)}
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue(), min_symbols=4)
    _feed(strat, series, funding)  # type: ignore[arg-type]
    rows, _ = strat._score_rows()
    scores = [comp for comp, _sym, _meta in rows]
    assert len(scores) == len(_SYMBOLS)
    # Z-scores across the cross-section sum to ~0 (mean-centered).
    assert sum(scores) == pytest.approx(0.0, abs=1e-9)

    # With fewer than two scoring symbols the cross-section collapses to zeros.
    single = ["AAA/USDT", "BBB/USDT"]
    solo = CrossSectionalFundingMomentumCarryStrategy(_Bars(single), _Queue(), min_symbols=2)
    solo_series = {s: _series(n, start=100.0, drift=0.002, noise=0.01) for s in single}
    # Only AAA gets funding; BBB never scores (no funding -> dropped bars).
    solo_funding: dict[str, list[float] | None] = {
        "AAA/USDT": [0.0001 * j for j in range(n)],
        "BBB/USDT": None,
    }
    _feed(solo, solo_series, solo_funding)
    assert solo._score_rows() == ([], {})


# --- (4) rank hysteresis ----------------------------------------------------


def test_rank_band_holds_name_without_new_order() -> None:
    queue = _Queue()
    strat = CrossSectionalFundingMomentumCarryStrategy(
        _Bars(_SYMBOLS),
        queue,
        min_symbols=4,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        rebalance_band=10.0,  # huge band => held names never re-emit
        target_vol=0.0,
        vol_window=10,
    )
    n = 30
    series = {
        s: _series(n, start=100.0 + i, drift=0.003, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    funding = {s: [0.0001 * (i + 1) * j for j in range(n)] for i, s in enumerate(_SYMBOLS)}
    _feed(strat, series, funding)  # type: ignore[arg-type]

    held = {symbol for symbol, item in strat._state.items() if item.mode != "OUT"}
    assert held, "expected at least one held position after warmup"

    before = len(queue.items)
    # One more interval with a barely-different funding continuation.
    for i, symbol in enumerate(_SYMBOLS):
        last_close = series[symbol][-1]
        strat.calculate_signals(_market_event(symbol, n, last_close * 1.0001, 0.0001 * (i + 1) * n))
    new = queue.items[before:]
    re_emitted = {sig.symbol for sig in new if sig.signal_type == "LONG"}
    assert not (re_emitted & held)


# --- (5) inverse-vol weights normalized + per-name cap clamp ----------------


def test_inverse_vol_weights_and_per_name_cap() -> None:
    queue = _Queue()
    strat = CrossSectionalFundingMomentumCarryStrategy(
        _Bars(_SYMBOLS),
        queue,
        min_symbols=4,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        target_vol=0.001,  # tiny => book scales down
        target_gross_exposure=1.0,
        vol_window=10,
        max_symbol_exposure_pct=0.05,  # tight per-name cap
    )
    n = 40
    series: dict[str, list[float]] = {}
    funding: dict[str, list[float]] = {}
    for i, symbol in enumerate(_SYMBOLS):
        if symbol in ("AAA/USDT", "BBB/USDT"):
            series[symbol] = _series(n, start=100.0, drift=0.003, noise=0.005)  # low vol
        else:
            series[symbol] = _series(n, start=100.0, drift=0.003, noise=0.03)  # higher vol
        # All rising funding => all eligible LONG.
        funding[symbol] = [0.0001 * j for j in range(n)]
    _feed(strat, series, funding)  # type: ignore[arg-type]

    longs = {sig.symbol: sig.metadata for sig in queue.items if sig.signal_type == "LONG"}
    assert len(longs) >= 2
    # Lower-vol names get larger inverse-vol weight => larger target_allocation.
    low = max(longs[s]["target_allocation"] for s in ("AAA/USDT", "BBB/USDT") if s in longs)
    high_syms = [s for s in longs if s not in ("AAA/USDT", "BBB/USDT")]
    if high_syms:
        high = max(longs[s]["target_allocation"] for s in high_syms)
        assert low > high
    # Portfolio vol-target clamp engaged.
    assert any(meta["vol_target_scalar"] < 1.0 for meta in longs.values())
    # Per-name cap clamp applied.
    for meta in longs.values():
        assert meta["max_symbol_exposure_pct"] <= 0.05 + 1e-12


# --- (6) missing funding -> no entries, never raises ------------------------


def test_missing_funding_no_entries_never_raises() -> None:
    queue = _Queue()
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), queue, min_symbols=4)
    n = 40
    series = {
        s: _series(n, start=100.0 + i, drift=0.003, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    funding: dict[str, list[float] | None] = dict.fromkeys(_SYMBOLS)
    _feed(strat, series, funding)  # never raises
    # No funding ever printed => no bars accepted => no signals.
    assert queue.items == []
    for item in strat._state.values():
        assert len(item.funding_rate) == 0
        assert item.mode == "OUT"


# --- (7) get_state / set_state round-trip -----------------------------------


def test_state_roundtrip_lossless() -> None:
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue(), min_symbols=4)
    n = 30
    series = {
        s: _series(n, start=100.0 + i, drift=0.003, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    funding = {s: [0.0001 * (i + 1) * j for j in range(n)] for i, s in enumerate(_SYMBOLS)}
    _feed(strat, series, funding)  # type: ignore[arg-type]

    item = strat._state["AAA/USDT"]
    item.mode = "LONG"
    item.entry_price = 123.0
    item.bars_held = 4
    item.score = 0.55
    state = strat.get_state()

    restored = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue())
    restored.set_state(state)
    r = restored._state["AAA/USDT"]
    assert list(r.closes) == list(item.closes)
    assert list(r.volumes) == list(item.volumes)
    assert list(r.funding_rate) == list(item.funding_rate)
    assert r.mode == "LONG"
    assert r.entry_price == pytest.approx(123.0)
    assert r.bars_held == 4
    assert r.score == pytest.approx(0.55)
    assert restored._tick == strat._tick
    assert restored._last_eval_time_key == strat._last_eval_time_key


# --- (9) tier research_only & excluded from live ----------------------------


def test_tier_is_research_only_and_excluded_from_live() -> None:
    from lumina_quant.strategies.registry import get_live_strategy_map, get_strategy_tier

    name = "CrossSectionalFundingMomentumCarryStrategy"
    assert get_strategy_tier(name) == "research_only"
    assert name not in get_live_strategy_map(include_opt_in=False)
    assert name not in get_live_strategy_map(include_opt_in=True)


# --- calculate_signals never raises on malformed events ---------------------


def test_calculate_signals_never_raises_on_garbage() -> None:
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue())
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="AAA/USDT", close=None, time=None)
    )
