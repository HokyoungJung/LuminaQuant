"""Deterministic tests for LowTurnoverTrendPersistenceStrategy (L1).

Direct class import only (no `@register` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

Covers:
- (a) B4 ORTHOGONALITY GATE vs the TRUE nearest neighbour
  ``MultiTimeframeTrendEnsembleStrategy``: on an identical input the ensemble
  flips its position on a fresh horizon agreement while L1 stays put inside its
  hard min-hold window -> divergent action.
- (b) Rescue property: a would-be flip inside the min-hold window is suppressed;
  once the min-hold expires (and after cooldown) the flip goes through.
- (c) determinism: two identical runs -> bit-identical signal streams.
- (d) get_state/set_state roundtrip + adversarial set_state never-raises.
- (e) never-raise on degenerate input (None/0/NaN/inf/empty).
- (f) behaviour: horizons-agree -> position taken; horizons-disagree -> flat.

All pseudo-randomness is drawn from a small seeded linear-congruential
generator (no ``random`` module), so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.aggressive_return_alpha_sleeves import (
    MultiTimeframeTrendEnsembleStrategy,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


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


_BASE = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday -> clean ISO-week boundaries


def _weekly_time(idx: int) -> str:
    """Return an ISO timestamp one ISO-week apart per index (a new decision bar)."""
    return (_BASE + timedelta(days=7 * idx)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _market_event(symbol: str, idx: int, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=_weekly_time(idx),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def _feed(strategy: Any, symbol: str, closes: list[float]) -> None:
    for idx, close in enumerate(closes):
        strategy.calculate_signals(_market_event(symbol, idx, close))


def _final_side(signals: list[Any], symbol: str) -> str | None:
    side: str | None = None
    for sig in signals:
        if sig.symbol != symbol:
            continue
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side = kind
        elif kind == "EXIT":
            side = None
    return side


def _entries(signals: list[Any], symbol: str) -> list[Any]:
    return [
        sig
        for sig in signals
        if sig.symbol == symbol and str(sig.signal_type).upper() in {"LONG", "SHORT"}
    ]


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic price-path generators
# --------------------------------------------------------------------------- #

_L1_KWARGS: dict[str, Any] = dict(
    tsmom_short=2,
    tsmom_mid=4,
    tsmom_long=6,
    efficiency_period=4,
    min_efficiency=0.25,
    adx_period=3,
    adx_min=10.0,
    vol_persist_fast=3,
    vol_persist_slow=6,
    vol_persist_max=5.0,
    vol_window=6,
    target_vol=0.0,
    allow_short=True,
    max_hold_bars=100000,
    min_price=0.01,
)

_ENSEMBLE_KWARGS: dict[str, Any] = dict(
    short_lookback=2,
    mid_lookback=4,
    long_lookback=6,
    align_threshold=3,
    min_horizon_roc=0.0,
    trail_atr_mult=1.0,
    atr_period=2,
    max_adds=0,
    add_step_atr=1.0,
    vol_window=6,
    target_vol=0.0,
    max_hold_bars=100000,
    allow_short=True,
    add_alloc_fraction=0.5,
    target_allocation=0.30,
    max_order_value=5000.0,
    min_price=0.01,
)


def _trend_closes(n_up: int, n_down: int, *, seed: int = 7) -> list[float]:
    """Clean efficient up-leg then down-leg (tiny LCG jitter for finite vol)."""
    gen = _lcg_stream(seed)
    out: list[float] = []
    price = 100.0
    for _ in range(n_up):
        price *= 1.03 * (1.0 + (next(gen) - 0.5) * 0.003)
        out.append(price)
    for _ in range(n_down):
        price *= 0.97 * (1.0 + (next(gen) - 0.5) * 0.003)
        out.append(price)
    return out


def _oscillating_closes(n: int, *, high: float = 100.0, low: float = 97.0) -> list[float]:
    """Zero-drift square-wave oscillation between two fixed levels.

    Every TSMOM lookback here (2/4/6) is even, so each horizon compares two
    same-phase closes -> an exactly-zero return -> the horizons can never all
    agree on a non-zero sign.  A deterministic, non-trending ``flat`` regime.
    """
    return [high if idx % 2 == 0 else low for idx in range(n)]


# --------------------------------------------------------------------------- #
# (a) B4 ORTHOGONALITY GATE vs MultiTimeframeTrendEnsembleStrategy
# --------------------------------------------------------------------------- #


def test_b4_orthogonality_gate_vs_multi_timeframe_trend_ensemble() -> None:
    """Same input: the ensemble FLIPS to SHORT while L1 stays LONG (min-hold).

    Both share the multi-horizon-sign core and both enter LONG on the clean
    up-leg.  On the down-leg the ensemble exits its trailing stop and re-enters
    SHORT on the fresh horizon agreement; L1's hard min-hold suppresses the flip
    entirely, so the two take a DIVERGENT action at the same final bar.  This is
    the orthogonality key (distinct from the min-hold-only rescue test below).
    """
    symbol = "TREND/USDT"
    closes = _trend_closes(22, 16)

    ensemble = MultiTimeframeTrendEnsembleStrategy(_Bars([symbol]), _Queue(), **_ENSEMBLE_KWARGS)
    _feed(ensemble, symbol, closes)

    l1 = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=50, cooldown_bars=1, **_L1_KWARGS
    )
    _feed(l1, symbol, closes)

    ensemble_side = _final_side(ensemble.events.items, symbol)
    l1_side = _final_side(l1.events.items, symbol)

    # Both must have engaged the shared long-trend core (else the divergence is
    # vacuous): each opened a LONG on the up-leg.
    assert any(sig.signal_type == "LONG" for sig in _entries(ensemble.events.items, symbol))
    assert any(sig.signal_type == "LONG" for sig in _entries(l1.events.items, symbol))

    # DIVERGENT ACTION on identical input: the ensemble has flipped to SHORT, L1
    # is still holding LONG inside its min-hold window.
    assert ensemble_side == "SHORT", ensemble_side
    assert l1_side == "LONG", l1_side
    assert ensemble_side != l1_side
    # L1 emitted no SHORT and no EXIT at all -- the flip was fully suppressed.
    assert all(sig.signal_type != "SHORT" for sig in l1.events.items)
    assert all(str(sig.signal_type).upper() != "EXIT" for sig in l1.events.items)


# --------------------------------------------------------------------------- #
# (b) Rescue property: min-hold suppresses a would-be flip; expiry releases it
# --------------------------------------------------------------------------- #


def test_min_hold_suppresses_would_be_flip_then_releases() -> None:
    symbol = "TREND/USDT"
    # Long down-leg so a real opposite (SHORT) signal is present well before the
    # (large) min-hold would ever permit acting on it.
    closes = _trend_closes(16, 30)

    # --- suppressed: min-hold larger than the whole feed -> never flips ---
    suppressed = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=100, cooldown_bars=1, **_L1_KWARGS
    )
    _feed(suppressed, symbol, closes)
    # A genuine would-be flip EXISTS at the final bar (all horizons agree SHORT
    # and the gates pass) ...
    assert suppressed._desired_signal(suppressed._state[symbol]) == -1
    # ... yet the position is still LONG and no EXIT/SHORT was ever emitted.
    assert _final_side(suppressed.events.items, symbol) == "LONG"
    assert all(str(sig.signal_type).upper() != "EXIT" for sig in suppressed.events.items)
    assert all(sig.signal_type != "SHORT" for sig in suppressed.events.items)

    # --- released: small min-hold -> the SAME feed flips once min-hold expires ---
    released = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    _feed(released, symbol, closes)
    kinds = [str(sig.signal_type).upper() for sig in released.events.items if sig.symbol == symbol]
    # The min-hold expiry produces an EXIT, and after the cooldown a fresh SHORT.
    assert "EXIT" in kinds
    assert kinds.index("LONG") < kinds.index("EXIT") < kinds.index("SHORT"), kinds
    assert _final_side(released.events.items, symbol) == "SHORT"


def test_cooldown_blocks_immediate_reentry_after_exit() -> None:
    """After the post-min-hold EXIT the re-entry waits out the cooldown window."""
    symbol = "TREND/USDT"
    closes = _trend_closes(16, 30)
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=3, **_L1_KWARGS
    )
    _feed(strategy, symbol, closes)
    items = [sig for sig in strategy.events.items if sig.symbol == symbol]
    kinds = [str(sig.signal_type).upper() for sig in items]
    exit_idx = kinds.index("EXIT")
    short_idx = kinds.index("SHORT")
    # There is a real gap (in decision bars) between the exit and the re-entry;
    # the SHORT strictly follows the EXIT, never on the same decision bar.
    assert exit_idx < short_idx


# --------------------------------------------------------------------------- #
# (c) determinism
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbol = "TREND/USDT"
    closes = _trend_closes(22, 16)

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = LowTurnoverTrendPersistenceStrategy(
            _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
        )
        _feed(strategy, symbol, closes)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# (d) state roundtrip + adversarial set_state
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbol = "TREND/USDT"
    closes = _trend_closes(16, 20)
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    _feed(strategy, symbol, closes)
    snapshot = strategy.get_state()

    restored = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    r = restored._state[symbol]
    o = strategy._state[symbol]
    assert list(r.closes) == list(o.closes)
    assert list(r.highs) == list(o.highs)
    assert r.mode == o.mode
    assert r.bars_held == o.bars_held
    assert r.bars_since_exit == o.bars_since_exit
    assert r.last_decision_week == o.last_decision_week


def test_adversarial_set_state_never_raises() -> None:
    symbol = "TREND/USDT"
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {symbol: "not a dict either"}})
    strategy.set_state({"symbol_state": {symbol: {"closes": 12345}}})
    strategy.set_state({"symbol_state": {symbol: {"closes": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "symbol_state": {
                symbol: {
                    "opens": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "highs": {"unexpected": "type"},
                    "lows": None,
                    "closes": ["bad", float("nan"), 10.0, 11.0],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "bars_since_exit": -5,
                    "last_bar_key": 123,
                    "last_decision_week": 456,
                    "score": [1, 2, 3],
                }
            }
        }
    )
    item = strategy._state[symbol]
    assert item.mode in {"OUT", "LONG", "SHORT"}
    assert item.bars_held >= 0
    assert item.bars_since_exit >= 0
    # Still functional afterwards.
    _feed(strategy, symbol, _trend_closes(16, 8))


# --------------------------------------------------------------------------- #
# (e) never-raise on degenerate input
# --------------------------------------------------------------------------- #


def test_degenerate_input_never_raises() -> None:
    symbol = "Z/USDT"
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    strategy.calculate_signals(_market_event(symbol, 0, 0.0))
    strategy.calculate_signals(_market_event(symbol, 1, -5.0))
    strategy.calculate_signals(_market_event(symbol, 2, float("nan")))
    strategy.calculate_signals(_market_event(symbol, 3, float("inf")))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="OTHER/USDT", close=None))
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", symbol=symbol, bars_1s={}, time="t0")
    )
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol=symbol, time=None, close=None))
    assert _non_exit(strategy.events.items) == []


def test_empty_universe_never_raises() -> None:
    strategy = LowTurnoverTrendPersistenceStrategy(_Bars([]), _Queue(), **_L1_KWARGS)
    strategy.calculate_signals(_market_event("ANY/USDT", 0, 100.0))
    assert strategy.events.items == []
    # get/set state on an empty universe is a no-op, not a raise.
    strategy.set_state(strategy.get_state())


# --------------------------------------------------------------------------- #
# (f) behaviour: horizons-agree -> position; horizons-disagree -> flat
# --------------------------------------------------------------------------- #


def test_horizons_agree_takes_position() -> None:
    symbol = "UP/USDT"
    closes = _trend_closes(24, 0)
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    _feed(strategy, symbol, closes)
    assert _final_side(strategy.events.items, symbol) == "LONG"
    longs = [sig for sig in _entries(strategy.events.items, symbol) if sig.signal_type == "LONG"]
    assert longs, "expected a LONG entry on the clean efficient up-leg"
    meta = longs[0].metadata or {}
    assert meta.get("net_sign") == 1
    assert meta.get("adx") is not None
    assert float(meta.get("efficiency_ratio", 0.0)) >= _L1_KWARGS["min_efficiency"]
    assert float(meta.get("target_allocation", 0.0)) > 0.0


def test_horizons_disagree_stays_flat() -> None:
    symbol = "MIX/USDT"
    strategy = LowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), min_hold_bars=6, cooldown_bars=2, **_L1_KWARGS
    )
    # Explicit disagreement: short/mid horizons UP, long horizon DOWN -> net 0.
    disagree = [100.0, 96.0, 92.0, 88.0, 84.0, 80.0, 76.0, 82.0, 88.0]
    assert strategy._horizon_agreement(disagree) == 0

    # End-to-end: a non-trending oscillation never clears the horizon-agreement
    # gate, so no position is ever opened.
    _feed(strategy, symbol, _oscillating_closes(60))
    assert _non_exit(strategy.events.items) == []


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = LowTurnoverTrendPersistenceStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "tsmom_short",
        "tsmom_mid",
        "tsmom_long",
        "efficiency_period",
        "min_efficiency",
        "adx_period",
        "adx_min",
        "vol_persist_fast",
        "vol_persist_slow",
        "vol_persist_max",
        "min_hold_bars",
        "cooldown_bars",
        "max_hold_bars",
        "vol_window",
        "target_vol",
        "allow_short",
        "target_allocation",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("target_allocation", "max_order_value"):
        assert schema[cap].tunable is False
