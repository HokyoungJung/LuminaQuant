"""Deterministic tests for CrossSectionalFlowShareRotationStrategy.

Direct class import only (no `@register` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).
Covers: rising-share + positive-return LONG entry; collapsing-share SHORT
entry; a distinct blow-off (extreme share + negative return) SHORT entry that
must NOT be confused with a LONG; self-skip below `min_symbols` (both a too-
small universe and a too-short history case); never-raise on missing volume,
degenerate closes, and an empty window; get_state/set_state roundtrip
equality; adversarial set_state safety; and run-to-run determinism.

Any pseudo-randomness used to shape the "neutral filler" symbols is drawn from
a small seeded linear-congruential generator (no ``random`` module), so every
run is bit-for-bit reproducible.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.flow_share_rotation_alpha_sleeves import (
    CrossSectionalFlowShareRotationStrategy,
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


def _market_event(symbol: str, idx: int, close: float, volume: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _feed(
    strategy: CrossSectionalFlowShareRotationStrategy,
    symbols: list[str],
    prices: dict[str, list[float]],
    volumes: dict[str, list[float] | None],
) -> None:
    n = len(prices[symbols[0]])
    for idx in range(n):
        for symbol in symbols:
            vol_series = volumes.get(symbol)
            vol = vol_series[idx] if vol_series is not None else None
            strategy.calculate_signals(_market_event(symbol, idx, prices[symbol][idx], vol))


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if sig.signal_type in {"LONG", "SHORT"}]


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


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
# deterministic price/volume path generators
# --------------------------------------------------------------------------- #

_COMMON_KWARGS: dict[str, Any] = dict(
    share_z_window=20,
    confirm_window=3,
    share_z_entry=1.0,
    blowoff_extremeness=0.85,
    top_n=8,
    vol_window=8,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)


def _control_series(index: int, n: int) -> tuple[list[float], list[float]]:
    """Deterministic, mildly-jittered flat-ish filler path (LCG-seeded)."""
    gen = _lcg_stream(seed=1000 + index)
    prices: list[float] = []
    volumes: list[float] = []
    price = 100.0
    for _ in range(n):
        price *= 1.0 + (next(gen) - 0.5) * 0.004
        volumes.append(1000.0 * (1.0 + (next(gen) - 0.5) * 0.10))
        prices.append(price)
    return prices, volumes


def _riser_series(n: int) -> tuple[list[float], list[float]]:
    """Steadily rising price AND dollar-volume share (both close and volume ramp).

    A tiny LCG-seeded jitter is folded into the price path so realized
    volatility is non-degenerate (a perfectly smooth compounding path has
    exactly zero variance in its log-returns, which would fail the vol
    eligibility gate outright).
    """
    gen = _lcg_stream(seed=42)
    prices: list[float] = []
    volumes: list[float] = []
    price = 100.0
    volume = 500.0
    for _ in range(n):
        price *= 1.02 * (1.0 + (next(gen) - 0.5) * 0.006)
        volume *= 1.10
        prices.append(price)
        volumes.append(volume)
    return prices, volumes


def _fader_series(n: int) -> tuple[list[float], list[float]]:
    """Starts with a large share of turnover, then decays (collapsing share)."""
    gen = _lcg_stream(seed=77)
    prices: list[float] = []
    volumes: list[float] = []
    price = 100.0
    volume = 5000.0
    for _ in range(n):
        price *= 1.0005 * (1.0 + (next(gen) - 0.5) * 0.006)
        volume *= 0.90
        prices.append(price)
        volumes.append(volume)
    return prices, volumes


def _blowoff_series(n: int, *, spike_at: int) -> tuple[list[float], list[float]]:
    """Flat-ish share pre-spike, then a late volume explosion + price reversal."""
    gen = _lcg_stream(seed=99)
    prices: list[float] = []
    volumes: list[float] = []
    price = 100.0
    volume = 900.0
    for i in range(n):
        if i < spike_at:
            price *= 1.006 * (1.0 + (next(gen) - 0.5) * 0.006)
        else:
            price *= 0.94 * (1.0 + (next(gen) - 0.5) * 0.006)
            volume *= 12.0
        prices.append(price)
        volumes.append(volume)
    return prices, volumes


def _build_universe(
    special_symbol: str,
    special_series: tuple[list[float], list[float]],
    n: int,
    num_controls: int = 5,
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float] | None]]:
    controls = [f"C{i}/USDT" for i in range(num_controls)]
    symbols = [special_symbol, *controls]
    prices: dict[str, list[float]] = {special_symbol: special_series[0]}
    volumes: dict[str, list[float] | None] = {special_symbol: special_series[1]}
    for i, symbol in enumerate(controls):
        p, v = _control_series(i, n)
        prices[symbol] = p
        volumes[symbol] = v
    return symbols, prices, volumes


# --------------------------------------------------------------------------- #
# (1) rising share + positive return -> LONG
# --------------------------------------------------------------------------- #


def test_rising_share_and_positive_return_enters_long() -> None:
    n = 40
    symbols, prices, volumes = _build_universe("RISER/USDT", _riser_series(n), n)
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)
    signals = strategy.events.items
    side = _final_side(signals)
    assert side.get("RISER/USDT") == "LONG", side
    riser_entries = [
        sig for sig in _entries(signals) if sig.symbol == "RISER/USDT" and sig.signal_type == "LONG"
    ]
    assert riser_entries, "expected at least one LONG entry for the rising-share symbol"
    meta = riser_entries[-1].metadata or {}
    assert meta.get("share_z") is not None and meta["share_z"] >= _COMMON_KWARGS["share_z_entry"]
    assert meta.get("confirm_return") is not None and meta["confirm_return"] > 0.0
    assert float(meta.get("target_allocation", 0.0)) > 0.0


# --------------------------------------------------------------------------- #
# (2) collapsing share -> SHORT
# --------------------------------------------------------------------------- #


def test_collapsing_share_enters_short() -> None:
    n = 40
    symbols, prices, volumes = _build_universe("FADER/USDT", _fader_series(n), n)
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)
    signals = strategy.events.items
    side = _final_side(signals)
    assert side.get("FADER/USDT") == "SHORT", side
    fader_entries = [
        sig
        for sig in _entries(signals)
        if sig.symbol == "FADER/USDT" and sig.signal_type == "SHORT"
    ]
    assert fader_entries, "expected at least one SHORT entry for the collapsing-share symbol"
    meta = fader_entries[-1].metadata or {}
    assert meta.get("share_z") is not None and meta["share_z"] <= -_COMMON_KWARGS["share_z_entry"]


# --------------------------------------------------------------------------- #
# (3) blow-off (extreme share + negative return) -> SHORT, not LONG
# --------------------------------------------------------------------------- #


def test_blowoff_extreme_share_and_negative_return_shorts_not_longs() -> None:
    n = 40
    spike_at = n - 3
    symbols, prices, volumes = _build_universe(
        "BLOWOFF/USDT", _blowoff_series(n, spike_at=spike_at), n
    )
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)
    signals = strategy.events.items
    side = _final_side(signals)
    assert side.get("BLOWOFF/USDT") == "SHORT", side
    blowoff_entries = [
        sig
        for sig in _entries(signals)
        if sig.symbol == "BLOWOFF/USDT" and sig.signal_type == "SHORT"
    ]
    assert blowoff_entries, "expected a SHORT entry on the blow-off reversal"
    meta = blowoff_entries[-1].metadata or {}
    # The defining feature of blow-off (vs. plain collapse): share_z is strongly
    # POSITIVE (a spike, not a collapse) with an extreme reading, paired with a
    # negative short-horizon return.
    assert meta.get("share_z") is not None and meta["share_z"] > 0.0, meta
    assert (
        meta.get("extremeness") is not None
        and meta["extremeness"] >= _COMMON_KWARGS["blowoff_extremeness"]
    ), meta
    assert meta.get("confirm_return") is not None and meta["confirm_return"] < 0.0, meta
    # The blow-off decision itself resolves to SHORT, not LONG, even though
    # share_z is strongly POSITIVE at that instant (a naive "positive z means
    # long" reading would get this wrong): the last directional signal for
    # this symbol is the SHORT above, not a LONG.
    assert blowoff_entries[-1].signal_type == "SHORT"


# --------------------------------------------------------------------------- #
# (4) self-skip below min_symbols
# --------------------------------------------------------------------------- #


def test_self_skip_when_universe_smaller_than_min_symbols() -> None:
    n = 40
    symbols = ["A/USDT", "B/USDT", "C/USDT"]
    prices = {s: _riser_series(n)[0] for s in symbols}
    volumes = {s: _riser_series(n)[1] for s in symbols}
    kwargs = dict(_COMMON_KWARGS, min_symbols=5)
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **kwargs)
    _feed(strategy, symbols, prices, volumes)
    assert _non_exit(strategy.events.items) == []


def test_self_skip_when_history_too_short() -> None:
    n = 5  # well under share_history_min / needed_closes for these window sizes
    symbols, prices, volumes = _build_universe("RISER/USDT", _riser_series(n), n)
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)
    assert _non_exit(strategy.events.items) == []


# --------------------------------------------------------------------------- #
# (5) never-raise: missing volume / degenerate closes / empty window
# --------------------------------------------------------------------------- #


def test_missing_volume_never_raises() -> None:
    n = 30
    symbols, prices, volumes = _build_universe("RISER/USDT", _riser_series(n), n)
    volumes["RISER/USDT"] = None  # every RISER bar arrives with volume=None
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)  # must not raise
    assert strategy._state["RISER/USDT"].volumes and all(
        v == 0.0 for v in strategy._state["RISER/USDT"].volumes
    )


def test_degenerate_closes_never_raise() -> None:
    strategy = CrossSectionalFlowShareRotationStrategy(
        _Bars(["Z/USDT"]), _Queue(), **_COMMON_KWARGS
    )
    strategy.calculate_signals(_market_event("Z/USDT", 0, 0.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 1, -5.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 2, float("nan"), 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 3, float("inf"), 1000.0))
    assert _non_exit(strategy.events.items) == []


def test_empty_window_never_raises() -> None:
    strategy = CrossSectionalFlowShareRotationStrategy(
        _Bars(["Z/USDT"]), _Queue(), **_COMMON_KWARGS
    )
    empty = SimpleNamespace(type="MARKET_WINDOW", symbol="Z/USDT", bars_1s={}, time="t0")
    strategy.calculate_signals(empty)
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# (6) get_state / set_state roundtrip
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    n = 30
    symbols, prices, volumes = _build_universe("RISER/USDT", _riser_series(n), n)
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    _feed(strategy, symbols, prices, volumes)
    snapshot = strategy.get_state()

    restored = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in symbols:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.volumes) == list(o.volumes)
        assert list(r.shares) == list(o.shares)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
        assert r.last_time_key == o.last_time_key
    assert restored._tick == strategy._tick
    assert restored._last_eval_time_key == strategy._last_eval_time_key


# --------------------------------------------------------------------------- #
# (7) adversarial set_state safety
# --------------------------------------------------------------------------- #


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = CrossSectionalFlowShareRotationStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "volumes": {"unexpected": "type"},
                    "shares": [-5.0, 2.0, "bad", float("nan"), 0.4],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert all(0.0 <= share <= 1.0 for share in item.shares)

    # Strategy must still function normally afterward, using the SAME symbols
    # it was constructed with.
    n = 20
    riser_prices, riser_volumes = _riser_series(n)
    prices = {symbols[0]: riser_prices}
    volumes: dict[str, list[float] | None] = {symbols[0]: riser_volumes}
    for i, symbol in enumerate(symbols[1:]):
        p, v = _control_series(i, n)
        prices[symbol] = p
        volumes[symbol] = v
    _feed(strategy, symbols, prices, volumes)  # must not raise


# --------------------------------------------------------------------------- #
# (8) determinism: two identical runs -> identical signal streams
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    n = 40
    symbols, prices, volumes = _build_universe("RISER/USDT", _riser_series(n), n)

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalFlowShareRotationStrategy(
            _Bars(symbols), _Queue(), **_COMMON_KWARGS
        )
        _feed(strategy, symbols, prices, volumes)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalFlowShareRotationStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "share_z_window",
        "confirm_window",
        "share_z_entry",
        "blowoff_extremeness",
        "top_n",
        "vol_window",
        "allow_short",
        "min_symbols",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
        "max_hold_bars",
        "base_allocation",
        "max_symbol_exposure_pct",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("base_allocation", "max_symbol_exposure_pct", "max_order_value"):
        assert schema[cap].tunable is False
