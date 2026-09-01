"""Deterministic tests for the CUSUM change-point + variance-ratio sleeves.

Covers registry presence; the variance-ratio helper (>1 for a persistent path,
<1 for a mean-reverting one); that ``CusumChangePointTrendRiderStrategy`` ENTERS
LONG after an upward drift regime shift and SHORT after a downward one; that
``VarianceRatioTrendRiderStrategy`` ENTERS on a persistent (VR>1) uptrend and is
SUPPRESSED on a mean-reverting path that still nets upward; candidate wiring
(both single-asset, allowed timeframes, cadence >= 1800 from metadata);
None/empty-window safety; and get_state/set_state roundtrip.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.cusum_varratio_alpha_sleeves import (
    CusumChangePointTrendRiderStrategy,
    VarianceRatioTrendRiderStrategy,
    _variance_ratio,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "CusumChangePointTrendRiderStrategy",
    "VarianceRatioTrendRiderStrategy",
)

_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}

_CRYPTO_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "ADA/USDT",
    "XRP/USDT",
]

_START = datetime(2026, 1, 1, tzinfo=UTC)


class _BarsNoFeatures:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    type = "MARKET"

    def __init__(
        self,
        time: str,
        symbol: str,
        close: float,
        *,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        self.time = time
        self.open = close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = 1_000_000.0


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


def _feed(strategy: Any, symbol: str, prices: list[float]) -> None:
    for i, p in enumerate(prices):
        rng = p * 0.0005
        strategy.calculate_signals(_Market(_ts(i), symbol, p, high=p + rng, low=p - rng))


# --------------------------------------------------------------------------- #
# Registry presence + variance-ratio helper
# --------------------------------------------------------------------------- #
def test_new_sleeves_registered() -> None:
    names = set(get_strategy_names())
    for cls in NEW_STRATEGIES:
        assert cls in names, cls
        assert get_strategy_param_schema(cls), cls
        assert get_default_strategy_params(cls), cls


def test_variance_ratio_helper() -> None:
    # Mean-reverting: alternating returns -> 2-period sums cancel -> VR < 1.
    alternating = [0.01 if i % 2 == 0 else -0.01 for i in range(64)]
    vr_mr = _variance_ratio(alternating, 2)
    assert vr_mr is not None and vr_mr < 1.0, vr_mr
    # Persistent: runs of same-sign returns -> 2-period sums amplified -> VR > 1.
    block = [0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01]
    persistent = block * 8
    vr_pers = _variance_ratio(persistent, 2)
    assert vr_pers is not None and vr_pers > 1.0, vr_pers
    assert _variance_ratio([0.01, 0.01], 4) is None  # too few points


# --------------------------------------------------------------------------- #
# CusumChangePointTrendRider
# --------------------------------------------------------------------------- #
def _cusum_strategy(bars: Any, events: _Events) -> CusumChangePointTrendRiderStrategy:
    return CusumChangePointTrendRiderStrategy(
        bars,
        events,
        cusum_vol_window=20,
        cusum_k=0.5,
        cusum_h=5.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _regime_shift_prices(direction: int, base: float = 100.0) -> list[float]:
    """A low-vol flat regime then a strong drift in ``direction`` (+1 up / -1 down)."""
    prices = [base]
    p = base
    for i in range(40):  # flat: tiny alternating noise -> small but non-zero std
        p *= 1.0 + (0.0008 if i % 2 == 0 else -0.0008)
        prices.append(p)
    for _ in range(15):  # regime shift: strong sustained drift
        p *= 1.0 + direction * 0.012
        prices.append(p)
    return prices


def test_cusum_enters_long_after_upward_shift() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _cusum_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _regime_shift_prices(+1))
    entries = _entries(events)
    assert entries, "expected a LONG entry after the upward CUSUM change-point"
    assert entries[0].signal_type == "LONG", entries[0].signal_type


def test_cusum_enters_short_after_downward_shift() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _cusum_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _regime_shift_prices(-1))
    entries = _entries(events)
    assert entries, "expected a SHORT entry after the downward CUSUM change-point"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type


# --------------------------------------------------------------------------- #
# VarianceRatioTrendRider
# --------------------------------------------------------------------------- #
def _vr_strategy(bars: Any, events: _Events) -> VarianceRatioTrendRiderStrategy:
    return VarianceRatioTrendRiderStrategy(
        bars,
        events,
        vr_window=64,
        vr_k=2,
        vr_threshold=0.10,
        trend_lookback=10,
        trend_ma_window=10,
        min_trend_roc=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def test_vr_enters_on_persistent_uptrend() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _vr_strategy(_BarsNoFeatures([symbol]), events)
    # Positively-autocorrelated up-runs (fast-up runs then slow-up runs, all
    # positive): same-sign return deviations cluster -> VR>1, and the path trends
    # up -> LONG.
    block = [0.03] * 5 + [0.001] * 5
    p = 100.0
    prices = [p]
    for i in range(120):
        p *= 1.0 + block[i % len(block)]
        prices.append(p)
    _feed(strategy, symbol, prices)
    entries = _entries(events)
    assert entries, "expected a LONG entry on a persistent (VR>1) uptrend"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    vr = (entries[0].metadata or {}).get("variance_ratio")
    assert vr is not None and vr >= 1.10, entries[0].metadata


def test_vr_suppressed_on_mean_reverting_path() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _vr_strategy(_BarsNoFeatures([symbol]), events)
    # Alternating large moves with a small net upward drift: mean-reverting returns
    # -> VR<1 -> no entry, even though the path nets up (a plain rider would buy).
    p = 100.0
    prices = [p]
    for i in range(120):
        p *= 1.02 if i % 2 == 0 else (1.0 / 1.018)
        prices.append(p)
    _feed(strategy, symbol, prices)
    assert _entries(events) == [], _entries(events)


# --------------------------------------------------------------------------- #
# None / empty-window safety + state roundtrip
# --------------------------------------------------------------------------- #
def test_sleeves_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    for factory in (_cusum_strategy, _vr_strategy):
        events = _Events()
        strategy = factory(_BarsNoFeatures([symbol]), events)

        class _EmptyWindow:
            type = "MARKET_WINDOW"
            bars_1s: list[Any] = []

        empty = _EmptyWindow()
        empty.symbol = symbol
        strategy.calculate_signals(empty)  # must not raise
        strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close, no raise
        assert _entries(events) == []


def test_cusum_state_roundtrip() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _cusum_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _regime_shift_prices(+1))
    snapshot = strategy.get_state()
    restored = _cusum_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["cusum_state"][symbol]["returns"] == snapshot["cusum_state"][symbol]["returns"]
    assert again["cusum_state"][symbol]["s_hi"] == snapshot["cusum_state"][symbol]["s_hi"]


def test_vr_state_roundtrip() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _vr_strategy(_BarsNoFeatures([symbol]), events)
    p = 100.0
    prices = [p]
    for _ in range(80):
        p *= 1.003
        prices.append(p)
    _feed(strategy, symbol, prices)
    snapshot = strategy.get_state()
    restored = _vr_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["vr_returns"][symbol] == snapshot["vr_returns"][symbol]


# --------------------------------------------------------------------------- #
# Candidate-admission wiring
# --------------------------------------------------------------------------- #
def _candidates_for(strategy_class: str) -> list[Any]:
    timeframes = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=_CRYPTO_UNIVERSE)
    return [c for c in candidates if c.strategy_class == strategy_class]


def test_cusum_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("CusumChangePointTrendRiderStrategy")
    assert rows, "expected cusum candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata


def test_vr_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("VarianceRatioTrendRiderStrategy")
    assert rows, "expected variance-ratio candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata
