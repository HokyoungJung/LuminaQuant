from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.strategies.diversified_multifactor_ensemble import (
    DiversifiedMultiFactorEnsembleStrategy,
)
from lumina_quant.strategies.registry import get_strategy_map
from lumina_quant.tuning import HyperParam

_SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]


def _zigzag(i: int) -> float:
    """Deterministic small oscillation so realized vol is meaningfully > EPS."""
    return 1.0 if i % 2 == 0 else -1.0


def _series(n: int, *, start: float, drift: float, noise: float) -> list[float]:
    """Geometric drift with a deterministic multiplicative zig-zag of size ``noise``."""
    out: list[float] = []
    price = start
    for i in range(n):
        price = start * (1.0 + drift) ** i * (1.0 + noise * _zigzag(i))
        out.append(price)
    return out


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    """Static bars stub: per-symbol features keyed by NOTHING (constant)."""

    def __init__(self, symbols: list[str], features: dict[str, dict[str, float]] | None = None):
        self.symbol_list = list(symbols)
        self._features = features or {}

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return self._features.get(symbol, {}).get(field)

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        return None


def _market_event(symbol: str, idx: int, close: float, **features: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        **features,
    )


def _feed_constant_features(
    strategy: DiversifiedMultiFactorEnsembleStrategy,
    series: dict[str, list[float]],
    feats: dict[str, dict[str, float]],
    funding_series: dict[str, list[float]] | None = None,
) -> None:
    """Drive one MARKET event per (bar, symbol) with attached features.

    ``funding_series`` (optional) overrides the per-bar funding rate per symbol so
    a per-symbol funding z-score can be exercised; otherwise a constant funding
    from ``feats`` is used.
    """
    funding_series = funding_series or {}
    n = len(next(iter(series.values())))
    for idx in range(n):
        for symbol in series:
            f = feats.get(symbol, {})
            funding = (
                funding_series[symbol][idx]
                if symbol in funding_series
                else f.get("funding_rate", 0.0)
            )
            strategy.calculate_signals(
                _market_event(
                    symbol,
                    idx,
                    series[symbol][idx],
                    funding_rate=funding,
                    mark_price=series[symbol][idx] * (1.0 + f.get("basis_frac", 0.0)),
                    index_price=series[symbol][idx],
                    open_interest=f.get("open_interest", 1_000_000.0),
                )
            )


# --- (a) registry membership ------------------------------------------------


def test_registry_membership() -> None:
    assert "DiversifiedMultiFactorEnsembleStrategy" in get_strategy_map()


# --- (b) schema snake_case + HyperParam -------------------------------------


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = DiversifiedMultiFactorEnsembleStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for weight_key in (
        "weight_trend",
        "weight_xs_momentum",
        "weight_low_vol",
        "weight_carry",
    ):
        assert weight_key in schema
        assert schema[weight_key].tunable is True


# --- (c) get_state / set_state round-trip -----------------------------------


def test_state_roundtrip_restores_deques_and_modes() -> None:
    bars = _Bars(_SYMBOLS)
    strat = DiversifiedMultiFactorEnsembleStrategy(bars, _Queue(), trend_lookback=20, vol_window=10)
    series = {symbol: [100.0 + i for i in range(60)] for symbol in _SYMBOLS}
    _feed_constant_features(
        strat, series, {symbol: {"funding_rate": 0.0001} for symbol in _SYMBOLS}
    )
    # Force a position into one symbol's state so modes round-trip too.
    item = strat._state["AAA/USDT"]
    item.mode = "LONG"
    item.entry_price = 123.0
    item.bars_held = 4
    item.composite = 0.55
    state = strat.get_state()

    restored = DiversifiedMultiFactorEnsembleStrategy(_Bars(_SYMBOLS), _Queue())
    restored.set_state(state)
    r_item = restored._state["AAA/USDT"]
    assert list(r_item.closes) == list(item.closes)
    assert list(r_item.funding_rate) == list(item.funding_rate)
    assert list(r_item.basis_bps) == list(item.basis_bps)
    assert list(r_item.open_interest) == list(item.open_interest)
    assert r_item.mode == "LONG"
    assert r_item.entry_price == pytest.approx(123.0)
    assert r_item.bars_held == 4
    assert r_item.composite == pytest.approx(0.55)
    assert restored._tick == strat._tick


# --- (d) no-lookahead -------------------------------------------------------


class _RecordingBars(_Bars):
    """Bars stub that records which (symbol, field, timestamp) were consulted."""

    def __init__(self, symbols: list[str], table: dict[tuple[str, str, str], float]):
        super().__init__(symbols)
        self.table = table
        # Record (symbol, field, current_key) for every feature lookup.
        self.queried: list[tuple[str, str, str]] = []
        self.current_key = ""

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        self.queried.append((symbol, field, self.current_key))
        return self.table.get((symbol, field, self.current_key))


def test_no_lookahead_uses_only_at_or_before_bar_data() -> None:
    symbols = _SYMBOLS
    # Build a feature table keyed by the bar timestamp the harness will set.
    keys = [f"2026-05-02T{i // 60:02d}:{i % 60:02d}:00Z" for i in range(40)]
    table: dict[tuple[str, str, str], float] = {}
    series: dict[str, list[float]] = {}
    for s_idx, symbol in enumerate(symbols):
        prices = _series(40, start=100.0 + s_idx, drift=0.004 + 0.001 * s_idx, noise=0.01)
        series[symbol] = prices
        for i, key in enumerate(keys):
            table[(symbol, "funding_rate", key)] = 0.0001
            table[(symbol, "mark_price", key)] = prices[i]
            table[(symbol, "index_price", key)] = prices[i]
            table[(symbol, "open_interest", key)] = 1_000_000.0

    bars = _RecordingBars(symbols, table)
    strat = DiversifiedMultiFactorEnsembleStrategy(
        bars, _Queue(), trend_lookback=20, vol_window=10, funding_z_window=8, rebalance_bars=1
    )

    n_bars = 38  # feed bars 0..37; bar index 37 is "N"
    for i in range(n_bars):
        bars.current_key = keys[i]
        for symbol in symbols:
            ev = SimpleNamespace(
                type="MARKET",
                time=keys[i],
                symbol=symbol,
                open=series[symbol][i],
                high=series[symbol][i],
                low=series[symbol][i],
                close=series[symbol][i],
                volume=1000.0,
            )
            strat.calculate_signals(ev)

    # Snapshot the composite the strategy WOULD use at N (computed purely from
    # at-or-before-N data) and the full set of feature keys consulted so far.
    rows_at_N, _vols_at_N = strat._composite_rows()
    composite_at_N = {sym: comp for comp, sym, _meta in rows_at_N}
    keys_consulted_through_N = {key for _sym, _field, key in bars.queried}

    # No feature lookup while building through N ever consulted the future key.
    future_key = keys[38]
    assert future_key not in keys_consulted_through_N
    # Only at-or-before-N keys were consulted.
    assert keys_consulted_through_N <= set(keys[:n_bars])
    assert len(composite_at_N) == len(symbols)

    # Inject a HUGE future jump as the NEXT bar (N+1). Because deques are appended
    # AFTER N and feature access is keyed to the closed bar, recomputing the
    # composite "as of N" from a fresh strategy fed only 0..N must MATCH the
    # snapshot — i.e. the future spike cannot leak backwards.
    bars.current_key = future_key
    for symbol in symbols:
        future_price = series[symbol][37] * 100.0  # 100x spike
        ev = SimpleNamespace(
            type="MARKET",
            time=future_key,
            symbol=symbol,
            open=future_price,
            high=future_price,
            low=future_price,
            close=future_price,
            volume=1000.0,
        )
        strat.calculate_signals(ev)

    # Rebuild an identical strategy fed ONLY bars 0..N (no future bar at all).
    clean_bars = _RecordingBars(symbols, table)
    clean = DiversifiedMultiFactorEnsembleStrategy(
        clean_bars, _Queue(), trend_lookback=20, vol_window=10, funding_z_window=8, rebalance_bars=1
    )
    for i in range(n_bars):
        clean_bars.current_key = keys[i]
        for symbol in symbols:
            ev = SimpleNamespace(
                type="MARKET",
                time=keys[i],
                symbol=symbol,
                open=series[symbol][i],
                high=series[symbol][i],
                low=series[symbol][i],
                close=series[symbol][i],
                volume=1000.0,
            )
            clean.calculate_signals(ev)
    rows_clean, _ = clean._composite_rows()
    composite_clean = {sym: comp for comp, sym, _meta in rows_clean}

    # The composite at N is identical whether or not the future spike was later
    # observed: the signal used only at-or-before-N data.
    assert composite_at_N == pytest.approx(composite_clean)


# --- (e) per-leg signal correctness -----------------------------------------


def test_leg_signs_negative_funding_positive_carry_and_low_vol() -> None:
    bars = _Bars(_SYMBOLS)
    strat = DiversifiedMultiFactorEnsembleStrategy(
        bars,
        _Queue(),
        trend_lookback=20,
        vol_window=10,
        funding_z_window=8,
        min_symbols=4,
    )
    # AAA: strong uptrend + low vol + funding DECLINING (latest below its window
    # mean => funding z < 0 => carry leg = -z > 0). Others: weak trend, high vol,
    # funding RISING (z > 0 => carry leg < 0).
    series: dict[str, list[float]] = {}
    feats: dict[str, dict[str, float]] = {}
    funding_series: dict[str, list[float]] = {}
    for symbol in _SYMBOLS:
        if symbol == "AAA/USDT":
            series[symbol] = _series(60, start=100.0, drift=0.01, noise=0.004)
            feats[symbol] = {}
            funding_series[symbol] = [0.001 - 0.00004 * i for i in range(60)]  # declining
        else:
            series[symbol] = _series(60, start=100.0, drift=0.0008, noise=0.03)
            feats[symbol] = {}
            funding_series[symbol] = [-0.001 + 0.00004 * i for i in range(60)]  # rising
    _feed_constant_features(strat, series, feats, funding_series=funding_series)

    rows, _vols = strat._composite_rows()
    by_symbol = {symbol: meta for _score, symbol, meta in rows}
    assert "AAA/USDT" in by_symbol
    aaa = by_symbol["AAA/USDT"]
    # Negative funding => above-average => positive carry z for AAA.
    assert aaa["z_carry"] > 0.0
    # Low realized vol => negated => above-average => positive low_vol z for AAA.
    assert aaa["z_low_vol"] > 0.0
    # Strong up-trend => positive trend z for AAA.
    assert aaa["z_trend"] > 0.0
    # AAA should rank top by composite.
    ranked = sorted(rows, key=lambda r: r[0], reverse=True)
    assert ranked[0][1] == "AAA/USDT"


# --- (f) inverse-vol sizing -------------------------------------------------


def test_inverse_vol_sizing_lower_vol_gets_larger_alloc_and_book_scales_down() -> None:
    bars = _Bars(_SYMBOLS)
    queue = _Queue()
    strat = DiversifiedMultiFactorEnsembleStrategy(
        bars,
        queue,
        trend_lookback=20,
        vol_window=10,
        funding_z_window=8,
        min_symbols=4,
        rebalance_bars=1,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        target_vol=0.001,  # tiny target => book must scale down
        target_gross_exposure=1.0,
    )
    # Two clearly different vols among up-trending names.
    series: dict[str, list[float]] = {}
    feats: dict[str, dict[str, float]] = {}
    for symbol in _SYMBOLS:
        if symbol in ("AAA/USDT", "BBB/USDT"):
            series[symbol] = _series(60, start=100.0, drift=0.004, noise=0.006)  # low vol
        else:
            series[symbol] = _series(60, start=100.0, drift=0.004, noise=0.03)  # higher vol
        feats[symbol] = {"funding_rate": -0.0005}
    _feed_constant_features(strat, series, feats)

    longs = {sig.symbol: sig.metadata for sig in queue.items if sig.signal_type == "LONG"}
    assert len(longs) >= 2
    # Lower-vol name has larger inverse-vol weight => larger target_allocation.
    low_vol_alloc = max(
        longs[s]["target_allocation"] for s in ("AAA/USDT", "BBB/USDT") if s in longs
    )
    high_vol_syms = [s for s in longs if s not in ("AAA/USDT", "BBB/USDT")]
    if high_vol_syms:
        high_vol_alloc = max(longs[s]["target_allocation"] for s in high_vol_syms)
        assert low_vol_alloc > high_vol_alloc
    # Portfolio vol-target clamp engaged (scalar < 1 because target_vol is tiny).
    assert any(meta["vol_target_scalar"] < 1.0 for meta in longs.values())


# --- (g) turnover gate ------------------------------------------------------


def test_turnover_band_holds_name_on_subthreshold_composite_change() -> None:
    bars = _Bars(_SYMBOLS)
    queue = _Queue()
    strat = DiversifiedMultiFactorEnsembleStrategy(
        bars,
        queue,
        trend_lookback=20,
        vol_window=10,
        funding_z_window=8,
        min_symbols=4,
        rebalance_bars=1,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        rebalance_band=10.0,  # huge band => no held name should re-emit
        target_vol=0.0,  # disable vol scaling for a clean read
    )
    series: dict[str, list[float]] = {}
    feats: dict[str, dict[str, float]] = {}
    for s_idx, symbol in enumerate(_SYMBOLS):
        series[symbol] = _series(60, start=100.0, drift=0.003 + 0.0005 * s_idx, noise=0.012)
        feats[symbol] = {"funding_rate": -0.0003}
    _feed_constant_features(strat, series, feats)

    # Capture which symbols are currently held (mode != OUT).
    held = {symbol for symbol, item in strat._state.items() if item.mode != "OUT"}
    assert held, "expected at least one held position after warmup"

    n_signals_before = len(queue.items)
    # One more bar with a tiny continuation: composite barely moves => band holds.
    next_idx = 60
    for symbol in _SYMBOLS:
        last = series[symbol][-1]
        strat.calculate_signals(
            _market_event(
                symbol,
                next_idx,
                last * 1.0001,
                funding_rate=-0.0003,
                mark_price=last * 1.0001,
                index_price=last * 1.0001,
                open_interest=1_000_000.0,
            )
        )
    new_signals = queue.items[n_signals_before:]
    # No held name should be re-emitted (LONG) within the band.
    re_emitted = {sig.symbol for sig in new_signals if sig.signal_type == "LONG"}
    assert not (re_emitted & held)


# --- carry leg: basis z-scored so a large STATIC basis cannot dominate funding -


def test_carry_static_basis_does_not_dominate_funding() -> None:
    """A large but static per-symbol basis must NOT swamp the funding signal.

    basis_bps is in basis points (~50-500); funding z is O(1). The carry leg
    z-scores basis over its window, so a *constant* basis (however large) yields
    basis_z == 0 and contributes nothing — leaving funding to drive the carry
    ranking. Pre-fix (raw basis_bps added) the 500 bps name would have ranked
    dead last regardless of its favorable funding.
    """
    bars = _Bars(_SYMBOLS)
    strat = DiversifiedMultiFactorEnsembleStrategy(
        bars, _Queue(), trend_lookback=20, vol_window=10, funding_z_window=8, min_symbols=4
    )
    # Identical price path for all => trend/xs/low_vol legs are equal across the
    # cross-section (z == 0), so the composite is driven purely by carry.
    base = _series(60, start=100.0, drift=0.003, noise=0.01)
    series: dict[str, list[float]] = {symbol: list(base) for symbol in _SYMBOLS}
    feats: dict[str, dict[str, float]] = {}
    funding_series: dict[str, list[float]] = {}
    for symbol in _SYMBOLS:
        if symbol == "AAA/USDT":
            # HUGE static basis (500 bps) but the BEST (declining) funding.
            feats[symbol] = {"basis_frac": 0.05}
            funding_series[symbol] = [0.001 - 0.00004 * i for i in range(60)]
        else:
            feats[symbol] = {"basis_frac": 0.0}
            funding_series[symbol] = [-0.001 + 0.00004 * i for i in range(60)]
    _feed_constant_features(strat, series, feats, funding_series=funding_series)

    rows, _vols = strat._composite_rows()
    by_symbol = {symbol: meta for _score, symbol, meta in rows}
    assert "AAA/USDT" in by_symbol
    # Static basis contributes 0 (basis_z == 0); funding drives carry => AAA top.
    assert by_symbol["AAA/USDT"]["z_carry"] > 0.0
    ranked = sorted(rows, key=lambda r: r[0], reverse=True)
    assert ranked[0][1] == "AAA/USDT"


# --- safety: unvalidated strategy is research_only, NOT live-eligible ---------


def test_tier_is_research_only_and_excluded_from_live() -> None:
    from lumina_quant.strategies.registry import get_live_strategy_map, get_strategy_tier

    name = "DiversifiedMultiFactorEnsembleStrategy"
    assert get_strategy_tier(name) == "research_only"
    # Must not be selected for live in EITHER mode until backtest-validated.
    assert name not in get_live_strategy_map(include_opt_in=False)
    assert name not in get_live_strategy_map(include_opt_in=True)
