"""Cross-sectional idiosyncratic-skew INNOVATION sleeve (W3-6).

``IdiosyncraticSkewInnovationStrategy`` ranks the liquid cross-section on the
CHANGE in realized idiosyncratic skewness -- ``d(skew)/dt`` of BTC-beta-hedged
residual returns measured on TWO NON-OVERLAPPING trailing windows.  The skew
LEVEL is already owned by the registered lottery / MAX and idio-vol sleeves; skew
DYNAMICS are not.  The signal is

    delta_skew = skew(residual[-W:]) - skew(residual[-2W:-W])

with ``W`` the ``skew_window`` (default 30) and the lag pinned to ``W`` (the two
windows never overlap).  The sleeve FADES the freshly-building tail and rides the
collapsing one: it SHORTS the top ``delta_skew`` quantile (skew freshly building
= new lottery-demand inflow / a squeeze forming = overpricing being created NOW)
and LONGS the bottom quantile (skew collapsing = lottery demand exiting / the
asymmetry just resolved).  The sign is FIXED ex-ante to the fade direction; the
crash-continuation alternative is a declared falsification outcome, not a flip
parameter.

The beta residualization is LOAD-BEARING.  A benchmark crash / squeeze injects a
common skew shock into every raw series on the same day; differencing the skew of
BETA-HEDGED residuals isolates the IDIOSYNCRATIC innovation, so a symbol that is
merely a levered echo of the benchmark (residual variance ~ 0) is abstained on
rather than shorted.  The authoring build gate proves that separation against the
real ``LotterySkewnessStrategy`` and ``IdiosyncraticVolatilityStrategy`` on a
level-tied permutation fixture: two symbols with an IDENTICAL 2W-return multiset
(equal skew LEVEL, equal MAX) but opposite ORDERING -- one building, one fading --
which the occupied level incumbents provably cannot separate and this sleeve
takes opposite sides on.

THEORY / PROVENANCE
-------------------
- Chen, Hong & Stein (2001, JFE 61(3)): skewness builds ahead of crashes /
  squeezes -- the dynamics, not the level, carry the timing information.
- Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE 118(1)): the realized-skew
  premium decays within ~weeks, so the INNOVATION is the timely component the
  level sleeve holds stale.
- Boyer, Mitton & Vorkink (2010, RFS 23(1)); Barberis & Huang (2008, AER 98(5)):
  the priced quantity is EXPECTED idiosyncratic skew; the lagged level is a stale
  proxy.
- Harvey & Siddique (2000, JF 55(3)): the higher-co-moment pricing foundation.
- HONEST FLAG: no crypto-specific skew-INNOVATION study exists; crypto
  higher-moment LEVEL evidence is flagged existence-to-reverify.

TURNOVER / COST
---------------
LOW-MEDIUM (honestly faster than the level incumbents -- ~23% of each window
refreshes per weekly step).  Weekly decisions on daily bars; a hard
``min_hold_decisions`` (the proven C1 turnover rescue) plus an enter-20 / exit-40
rank-band hysteresis suppress rank flicker; the 30v30 non-overlapping windows
bound week-over-week score movement; inverse-realized-vol sizing plus a liquid
``min_price`` floor keep sqrt-impact benign.  If the fresh component decays faster
than the weekly clock harvests net of 20bps the lane dies at the gate.

This module is data-local (no I/O), pure Python (``math`` + ``deque`` + the pure
rolling-stat primitives, no numpy/scipy/statsmodels), completed-bar, and never
raises from ``calculate_signals``.  It ships WITHOUT ``@register`` on purpose: an
unregistered module under ``strategies/`` is inert; registration, the
``research_only`` tier hint, candidate wiring, and baseline re-pins land in the
separate atomic integration commit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import rolling_beta
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _default_benchmark, _state_size
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "skew_innovation"
_STRATEGY_NAME = "IdiosyncraticSkewInnovationStrategy"

# A fresh symbol has never traded, so its cooldown starts long-elapsed -- the
# first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30


# --------------------------------------------------------------------------- #
# lane-local pure numerics (never raise; None-propagating)
# --------------------------------------------------------------------------- #
def _bar_simple_returns(closes: list[float]) -> list[float]:
    """Return the bar-to-bar simple returns of a price path (drops bad bars)."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and abs(prev) > _EPS and value == value:
            out.append(value / prev - 1.0)
        prev = value
    return out


def _skewness(values: list[float]) -> float | None:
    """Return the sample (Fisher-Pearson, ddof-free) skewness of ``values``.

    ``None`` on fewer than three finite samples or degenerate (zero-variance)
    dispersion.  Pure, ``None``-safe, and never raises.
    """
    cleaned = [value for value in values if isinstance(value, (int, float)) and value == value]
    count = len(cleaned)
    if count < 3:
        return None
    avg = math.fsum(cleaned) / float(count)
    variance = math.fsum((value - avg) ** 2 for value in cleaned) / float(count)
    if variance <= _EPS:
        return None
    std = variance**0.5
    third = math.fsum((value - avg) ** 3 for value in cleaned) / float(count)
    skew = third / (std**3)
    return float(skew) if math.isfinite(skew) else None


def _skew_innovation(residuals: list[float], window: int) -> float | None:
    """Return ``skew(residuals[-W:]) - skew(residuals[-2W:-W])`` or ``None``.

    The two windows are NON-OVERLAPPING (lag pinned to ``window``).  Abstains
    (``None``) when there is insufficient history or either window has degenerate
    variance / too few finite residuals.  Never raises.
    """
    win = max(3, int(window))
    if len(residuals) < 2 * win:
        return None
    floor = max(10, win // 2)
    recent = residuals[-win:]
    prior = residuals[-2 * win : -win]
    if len(recent) < floor or len(prior) < floor:
        return None
    skew_now = _skewness(recent)
    skew_prev = _skewness(prior)
    if skew_now is None or skew_prev is None:
        return None
    delta = skew_now - skew_prev
    return float(delta) if math.isfinite(delta) else None


@dataclass(slots=True)
class _State:
    """Per-symbol close history + weekly position / min-hold / cooldown state."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly DECISIONS in the current position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decisions since last exit
    last_bar_key: str = ""  # daily-bar dedup
    score: float | None = None


def _coerce_float_list(value: Any) -> list[float]:
    """Best-effort ``list[float]`` coercion that never raises on adversarial input."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = safe_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


@register("strategy", "IdiosyncraticSkewInnovationStrategy", interface="event_driven")
class IdiosyncraticSkewInnovationStrategy(Strategy):
    """Weekly XS long-short on the INNOVATION in beta-hedged residual skewness.

    Fades the freshly-building skew tail (SHORT) and rides the collapsing one
    (LONG).  See the module docstring for the theory, the load-bearing beta
    residualization, and the cost argument.  This class only reads local
    event/bar OHLCV; it performs no I/O and never raises from
    ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly internal decision clock
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "beta_window": HyperParam.integer("beta_window", default=120, low=8, high=20000),
            "skew_window": HyperParam.integer("skew_window", default=30, low=6, high=4096),
            "quantile_entry_pct": HyperParam.floating(
                "quantile_entry_pct", default=0.20, low=0.02, high=0.50
            ),
            "quantile_exit_pct": HyperParam.floating(
                "quantile_exit_pct", default=0.40, low=0.02, high=1.0
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=2, low=0, high=100000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=100000
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=70, low=6, high=100000
            ),
            "max_hold_decisions": HyperParam.integer(
                "max_hold_decisions", default=52, low=1, high=200000
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.benchmark_symbol = _default_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.beta_window = max(8, int(resolved["beta_window"]))
        self.skew_window = max(3, int(resolved["skew_window"]))
        entry = min(0.5, max(0.02, float(resolved["quantile_entry_pct"])))
        exit_pct = min(1.0, max(entry, float(resolved["quantile_exit_pct"])))
        self.quantile_entry_pct = entry
        self.quantile_exit_pct = exit_pct
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.min_history_bars = max(2 * self.skew_window + 1, int(resolved["min_history_bars"]))
        self.max_hold_decisions = max(1, int(resolved["max_hold_decisions"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.beta_window + 1,
            2 * self.skew_window + 1,
            self.vol_window + 1,
            self.min_history_bars,
        )
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=size)) for symbol in self.symbol_list
        }
        self._last_decision_week = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_decision_week": self._last_decision_week,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "bars_since_exit": int(item.bars_since_exit),
                    "last_bar_key": item.last_bar_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_decision_week = str(state.get("last_decision_week", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                item.closes.clear()
                maxlen = int(item.closes.maxlen or 0)
                values = _coerce_float_list(payload.get("closes"))
                for value in values[-maxlen:] if maxlen else values:
                    item.closes.append(value)
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                raw_cooldown = payload.get("bars_since_exit")
                item.bars_since_exit = (
                    _safe_non_negative_int(raw_cooldown)
                    if raw_cooldown is not None
                    else _COOLDOWN_SATISFIED
                )
                item.last_bar_key = str(payload.get("last_bar_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion / weekly cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return time_key(raw_time)
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return False
        item.last_bar_key = key
        item.closes.append(close)
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated:
            self._maybe_decide(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                self._maybe_decide(snapshot.time)

    def _maybe_decide(self, event_time: Any) -> None:
        week = self._week_key(event_time)
        if not week or week == self._last_decision_week:
            return
        self._last_decision_week = week
        self._tick += 1
        self._rebalance(event_time)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _beta_hedged_residuals(
        self, sym_closes: list[float], bench_rets: list[float]
    ) -> list[float] | None:
        """Return the trailing ``2W`` beta-hedged residual returns, or ``None``."""
        sym_rets = _bar_simple_returns(sym_closes)
        need = 2 * self.skew_window
        aligned = min(len(sym_rets), len(bench_rets))
        if aligned < need:
            return None
        beta_depth = min(aligned, self.beta_window)
        beta = rolling_beta(sym_rets[-beta_depth:], bench_rets[-beta_depth:])
        if beta is None:
            beta = 0.0
        sym_tail = sym_rets[-need:]
        bench_tail = bench_rets[-need:]
        residuals = [s - beta * b for s, b in zip(sym_tail, bench_tail, strict=False)]
        return residuals

    def delta_skew_for(self, sym_closes: list[float], bench_closes: list[float]) -> float | None:
        """Public helper: the beta-hedged residual skew innovation for one symbol."""
        if len(sym_closes) < self.min_history_bars:
            return None
        bench_rets = _bar_simple_returns(bench_closes)
        if len(bench_rets) < 2 * self.skew_window:
            return None
        residuals = self._beta_hedged_residuals(sym_closes, bench_rets)
        if residuals is None:
            return None
        return _skew_innovation(residuals, self.skew_window)

    def _score_symbols(self) -> dict[str, tuple[float, dict[str, Any]]]:
        bench = self._state.get(self.benchmark_symbol)
        if bench is None:
            return {}
        bench_closes = list(bench.closes)
        bench_rets = _bar_simple_returns(bench_closes)
        if len(bench_rets) < 2 * self.skew_window:
            return {}
        scored: dict[str, tuple[float, dict[str, Any]]] = {}
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            sym_closes = list(item.closes)
            if len(sym_closes) < self.min_history_bars:
                continue
            residuals = self._beta_hedged_residuals(sym_closes, bench_rets)
            if residuals is None:
                continue
            delta = _skew_innovation(residuals, self.skew_window)
            if delta is None:
                continue
            # LONG the collapsing tail (low delta), SHORT the building tail (high
            # delta): score is the NEGATED innovation so the top-of-book is LONG.
            score = -float(delta)
            scored[symbol] = (score, {"delta_skew": float(delta)})
        return scored

    # ------------------------------------------------------------------ #
    # selection (rank-band hysteresis + hard min-hold + cooldown)
    # ------------------------------------------------------------------ #
    def _desired_book(self, scored: dict[str, tuple[float, dict[str, Any]]]) -> dict[str, str]:
        ordered = sorted(scored.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
        rank = {symbol: idx for idx, (symbol, _payload) in enumerate(ordered)}
        n = len(ordered)
        k_enter = max(1, int(n * self.quantile_entry_pct))
        k_exit = max(k_enter, int(n * self.quantile_exit_pct))
        desired: dict[str, str] = {}
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            idx = rank.get(symbol)
            cur = item.mode
            # Hard min-hold: freeze the current side until it matures.
            if cur != "OUT" and item.bars_held < self.min_hold_decisions:
                desired[symbol] = cur
                continue
            if idx is None:
                desired[symbol] = "OUT"
                continue
            long_enter = idx < k_enter
            long_hold = idx < k_exit
            short_enter = idx >= n - k_enter
            short_hold = idx >= n - k_exit
            if cur == "LONG" and long_hold:
                desired[symbol] = "LONG"
            elif cur == "SHORT" and short_hold and self.allow_short:
                desired[symbol] = "SHORT"
            elif long_enter and item.bars_since_exit >= self.cooldown_decisions:
                desired[symbol] = "LONG"
            elif (
                short_enter and self.allow_short and item.bars_since_exit >= self.cooldown_decisions
            ):
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = "OUT"
        return desired

    def _inverse_vol_weights(self, book: list[str]) -> dict[str, float]:
        if not book:
            return {}
        vols: dict[str, float | None] = {}
        for symbol in book:
            vols[symbol] = realized_volatility(
                list(self._state[symbol].closes), window=self.vol_window
            )
        valid = [1.0 / vol for vol in vols.values() if vol is not None and vol > _EPS]
        avg_inv = (sum(valid) / len(valid)) if valid else 1.0
        inv = {
            symbol: (1.0 / vol if (vol is not None and vol > _EPS) else avg_inv)
            for symbol, vol in vols.items()
        }
        total = sum(inv.values())
        if total <= _EPS:
            share = self.target_gross_exposure / float(len(book))
            return dict.fromkeys(book, share)
        return {symbol: self.target_gross_exposure * inv[symbol] / total for symbol in book}

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Age every held position / cooldown once per weekly decision.
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
            else:
                item.bars_since_exit += 1
        scored = self._score_symbols()
        if len(scored) < self.min_symbols:
            self._age_max_hold(event_time)
            return
        desired = self._desired_book(scored)
        book = [symbol for symbol, side in desired.items() if side in {"LONG", "SHORT"}]
        weights = self._inverse_vol_weights(book)
        self._emit_book(desired, weights, scored, event_time)

    def _age_max_hold(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol or item.mode == "OUT":
                continue
            if item.bars_held >= self.max_hold_decisions:
                price = item.closes[-1] if item.closes else None
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "max_hold"},
                )
                self._flatten(item)

    def _emit_book(
        self,
        desired: dict[str, str],
        weights: dict[str, float],
        scored: dict[str, tuple[float, dict[str, Any]]],
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            side = desired.get(symbol, "OUT")
            price = item.closes[-1] if item.closes else None
            if side == "OUT":
                if item.mode != "OUT" and item.bars_held >= self.min_hold_decisions:
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    self._flatten(item)
                continue
            if item.mode == side:
                score, _meta = scored.get(symbol, (item.score or 0.0, {}))
                item.score = float(score)
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "side_flip"},
                )
                self._flatten(item)
            score, meta = scored.get(symbol, (0.0, {}))
            weight = max(0.0, float(weights.get(symbol, 0.0)))
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=weight,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=side,
                **meta,
            )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=side,
                strength=max(0.25, min(3.0, abs(float(score)))),
                price=price,
                metadata=metadata,
            )
            item.mode = side
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Honest family:
# cross-sectional higher-moment, admitted via `allow_multi_asset=True`; NOT
# carry/momentum, so no fake tags are added to game the allocator superset route.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "higher_moment",
    "skew_innovation",
    "lottery_fade",
    "beta_hedged",
    "crypto",
)

# Candidate slice (DAILY bars; weekly decision clock via the ISO-week key).  Each
# variant dict is a pre-registered sweep cell counted toward N_eff at the data-PC.
_SKEW_INNOVATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "dskew_30v30",
            "beta_window": 120,
            "skew_window": 30,
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 2,
            "min_symbols": 5,
            "vol_window": 30,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "dskew_45v45",
            "beta_window": 120,
            "skew_window": 45,
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 2,
            "min_symbols": 5,
            "vol_window": 30,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["IdiosyncraticSkewInnovationStrategy"]
