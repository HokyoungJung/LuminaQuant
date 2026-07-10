"""Cross-sectional price-delay PREMIUM sleeve (Hou-Moskowitz D1, unconditional).

``CrossSectionalPriceDelayPremiumStrategy`` ranks the liquid cross-section by
each symbol's Hou & Moskowitz (2005, RFS) ``D1`` price-delay share -- the
fraction of its market-model return variance explained by LAGGED benchmark
returns rather than the contemporaneous benchmark return (a Dimson-beta
nonsynchronicity statistic) -- and takes a weekly long-short book: LONG the
highest-delay names (slow-recognition names carrying the Merton shadow-cost
premium), SHORT the lowest-delay names (instantly-priced majors).

The delay share is a sign-free, scale-free, months-persistent ``[0, 1]``
CHARACTERISTIC.  It is harvested UNCONDITIONALLY -- the sleeve collects the
equilibrium rent of slow diffusion instead of TIMING it, which is the
load-bearing distinction from every lead-lag incumbent (whose book flips with
the recent leader sign -- the conditional timing harvest that died as graveyard
#6 at 20-30bps).

THEORY / PROVENANCE
-------------------
- Hou & Moskowitz (2005, RFS 18(3) 981-1020): the ``D1`` delay measure predicts
  the equity cross-section; the premium survives size/liquidity controls -- the
  friction is investor RECOGNITION (Merton 1987), not trading cost.
- Koechling, Mueller & Posch (2019, Economics Letters 174 39-41): the HM delay
  computed on 75 cryptos is real, declining, and correlated with size/liquidity
  -- the crypto-specific characteristic verification.
- JEDC 2024 (S0165188924000551): slow diffusion via limited attention with a
  cost-surviving long-short.  Dimson (1979); Scholes & Williams (1977);
  Merton (1987); Hong & Stein (1999).
- HONEST FLAG: the unconditional delay PREMIUM is verified in equities only;
  crypto verification covers the mechanism and the characteristic -- the premium
  transplant IS the hypothesis (a falsification probe, not a promise of edge).

SIGNAL SPEC
-----------
Per completed WEEKLY ISO decision bar (internal ``_week_key`` clock -- decides at
most once per week even when fed daily/higher-frequency bars), per symbol with
>= ``delay_window`` trailing log returns:

1. ``D = price_delay_share(asset_log_returns, benchmark_log_returns,
   lags=delay_lags, ...)`` -- two nested OLS market-model regressions (restricted
   contemporaneous vs full lagged) solved by the normal equations; ``None`` when
   the full-model ``R^2`` is below the systematic-loading floor (the
   pure-idiosyncratic exclusion) or the history is too short.  The benchmark is
   BTC (``_resolve_benchmark``), never itself ranked.  ``score_mode="d1"`` uses
   the ``R^2`` ratio; ``"lag_weighted"`` uses the lag-weighted coefficient share.
2. Rank the liquid book by ``D``; LONG the top ``quantile_entry_pct`` fraction,
   SHORT the bottom fraction, with a WIDER ``quantile_exit_pct`` retention band,
   a hard ``min_hold_decisions`` minimum hold, a post-exit ``cooldown_decisions``
   window, and a ``max_hold_decisions`` backstop -- a rank flip inside the hold
   window is suppressed (the proven low-turnover rescue, weekly decision bars).
3. Inverse-realized-vol risk-parity sizing normalized to
   ``target_gross_exposure``.  The book self-skips (emits nothing) when fewer
   than ``min_symbols`` names carry a valid delay + vol history.

This module is data-local (no I/O, no hidden configuration bus), pure Python in
the strategy layer (the numpy normal-equations delay numeric lives in
``indicators/price_delay.py``), completed-bar, and never raises from
``calculate_signals``.  It ships WITHOUT ``@register`` (inert): registration,
tier hint, and candidate wiring land atomically in the integration wave.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.price_delay import price_delay_share
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

_STRATEGY_ID = "price_delay_premium"
_STRATEGY_NAME = "CrossSectionalPriceDelayPremiumStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed --
# the first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30

_SCORE_MODES = ("d1", "lag_weighted")
_BENCHMARK_PREFERENCES = ("BTC/USDT", "BTCUSDT", "ETH/USDT", "ETHUSDT")


@dataclass(slots=True)
class _State:
    """Per-symbol trailing closes + position / min-hold / cooldown bookkeeping."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly decision bars in the CURRENT position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decision bars since last exit
    last_bar_key: str = ""  # dedup identical ingested bars
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


def _resolve_benchmark(symbols: list[str], preferred: str) -> str:
    """Resolve the benchmark symbol: the configured one if present, else BTC."""
    if preferred in symbols:
        return preferred
    for candidate in _BENCHMARK_PREFERENCES:
        if candidate in symbols:
            return candidate
    return symbols[0] if symbols else preferred


def _log_returns(closes: list[float], count: int) -> list[float]:
    """Return the trailing ``count`` one-step log returns (skips non-positive)."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and prev > 0.0 and value > 0.0:
            out.append(math.log(value / prev))
        prev = value
    return out[-count:] if count and len(out) > count else out


@register("strategy", "CrossSectionalPriceDelayPremiumStrategy", interface="event_driven")
class CrossSectionalPriceDelayPremiumStrategy(Strategy):
    """Weekly XS long-short on the Hou-Moskowitz D1 price-delay characteristic.

    See the module docstring for the full theory, signal spec, and the
    load-bearing distinction from the conditional lead-lag axis.  This class only
    reads local event/bar closes; it performs no I/O and never raises from
    ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly effective via _week_key
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
            "delay_window": HyperParam.integer("delay_window", default=180, low=16, high=20000),
            "delay_lags": HyperParam.integer("delay_lags", default=5, low=1, high=64),
            "min_delay_obs": HyperParam.integer("min_delay_obs", default=30, low=8, high=20000),
            "min_r2": HyperParam.floating("min_r2", default=0.10, low=0.0, high=1.0, tunable=False),
            "score_mode": HyperParam.categorical(
                "score_mode", default="d1", choices=list(_SCORE_MODES)
            ),
            "quantile_entry_pct": HyperParam.floating(
                "quantile_entry_pct", default=0.20, low=0.05, high=0.50
            ),
            "quantile_exit_pct": HyperParam.floating(
                "quantile_exit_pct", default=0.40, low=0.05, high=0.90
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=1, high=200000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=200000
            ),
            "max_hold_decisions": HyperParam.integer(
                "max_hold_decisions", default=52, low=1, high=1000000
            ),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=0, low=0, high=100000
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=3, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
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
        self.benchmark_symbol = _resolve_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.delay_window = max(16, int(resolved["delay_window"]))
        self.delay_lags = max(1, int(resolved["delay_lags"]))
        self.min_delay_obs = max(8, int(resolved["min_delay_obs"]))
        self.min_r2 = max(0.0, min(1.0, float(resolved["min_r2"])))
        mode = str(resolved["score_mode"]).lower()
        self.score_mode = mode if mode in _SCORE_MODES else "d1"
        self.quantile_entry = max(0.01, min(0.50, float(resolved["quantile_entry_pct"])))
        self.quantile_exit = max(
            self.quantile_entry, min(0.90, float(resolved["quantile_exit_pct"]))
        )
        self.min_hold_decisions = max(1, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.max_hold_decisions = max(self.min_hold_decisions, int(resolved["max_hold_decisions"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(3, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Closes needed to compute ``delay_window`` returns (plus the lag design)
        # and a valid realized-vol reading; an explicit floor may raise it.
        self._min_history = max(
            int(resolved["min_history_bars"]),
            self.delay_window + 1,
            self.vol_window + 1,
        )
        # Trailing return sample handed to the delay numeric (window + lag design).
        self._return_count = self.delay_window + self.delay_lags
        history = max(self.delay_window + self.delay_lags, self.vol_window) + 8
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=history)) for symbol in self.symbol_list
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
                target = item.closes
                target.clear()
                maxlen = int(target.maxlen or 0)
                values = _coerce_float_list(payload.get("closes"))
                for value in values[-maxlen:] if maxlen else values:
                    target.append(value)
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
    # ingestion / cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return time_key(raw_time)
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return False
        item.last_bar_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item.closes.append(float(close))
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        week = self._week_key(getattr(event, "time", None))
        if updated and week and week != self._last_decision_week:
            self._last_decision_week = week
            self._tick += 1
            self._evaluate(getattr(event, "time", None))

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
                week = self._week_key(snapshot.time)
                if week and week != self._last_decision_week:
                    self._last_decision_week = week
                    self._tick += 1
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _delay_scores(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(delay_by_symbol, vol_by_symbol)`` over the eligible book."""
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None or len(benchmark.closes) < self._min_history:
            return {}, {}
        bench_returns = _log_returns(list(benchmark.closes), self._return_count)
        if len(bench_returns) < self.delay_lags + self.min_delay_obs:
            return {}, {}
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            closes = list(item.closes)
            if len(closes) < self._min_history:
                continue
            asset_returns = _log_returns(closes, self._return_count)
            delay = price_delay_share(
                asset_returns,
                bench_returns,
                lags=self.delay_lags,
                min_obs=self.min_delay_obs,
                min_r2=self.min_r2,
                score_mode=self.score_mode,
            )
            if delay is None:
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            scores[symbol] = float(delay)
            vols[symbol] = float(vol)
        return scores, vols

    @staticmethod
    def _zscore(scores: dict[str, float]) -> dict[str, float]:
        """Cross-sectional z-score of the delay values (diagnostic metadata)."""
        values = list(scores.values())
        count = len(values)
        if count < 2:
            return dict.fromkeys(scores, 0.0)
        mean_value = sum(values) / float(count)
        variance = sum((value - mean_value) ** 2 for value in values) / float(count - 1)
        sigma = variance**0.5
        if sigma <= _EPS:
            return dict.fromkeys(scores, 0.0)
        return {symbol: (value - mean_value) / sigma for symbol, value in scores.items()}

    def _select_book(self, scores: dict[str, float]) -> dict[str, str | None]:
        """Resolve the desired per-symbol book with hysteresis / hold / cooldown."""
        ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
        count = len(ranked)
        n_enter = max(1, int(count * self.quantile_entry))
        n_hold = max(1, int(count * self.quantile_exit))
        long_entry = set(ranked[:n_enter])
        long_hold = set(ranked[:n_hold])
        short_entry: set[str] = set()
        short_hold: set[str] = set()
        if self.allow_short:
            short_entry = set(ranked[count - n_enter :])
            short_hold = set(ranked[count - n_hold :])

        desired: dict[str, str | None] = {}
        for symbol, item in self._state.items():
            cur = item.mode
            if cur in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                desired[symbol] = None
                continue
            if cur in {"LONG", "SHORT"} and item.bars_held < self.min_hold_decisions:
                # Hard min-hold: a would-be flip inside the window is suppressed.
                desired[symbol] = cur
                continue
            if cur == "LONG":
                desired[symbol] = "LONG" if symbol in long_hold else None
            elif cur == "SHORT":
                desired[symbol] = "SHORT" if symbol in short_hold else None
            elif item.bars_since_exit < self.cooldown_decisions:
                desired[symbol] = None
            elif symbol in long_entry:
                desired[symbol] = "LONG"
            elif symbol in short_entry:
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = None
        return desired

    # ------------------------------------------------------------------ #
    # decision (weekly cadence)
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Age every held / flat symbol once per weekly decision BEFORE selection.
        for item in self._state.values():
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
            else:
                item.bars_since_exit += 1

        scores, vols = self._delay_scores()
        if len(scores) < self.min_symbols:
            self._enforce_max_hold(event_time)
            return
        zscores = self._zscore(scores)
        desired = self._select_book(scores)
        self._emit_targets(desired, scores, zscores, vols, event_time)

    def _enforce_max_hold(self, event_time: Any) -> None:
        """Exit only positions past ``max_hold_decisions`` when the book self-skips."""
        for symbol, item in self._state.items():
            if item.mode in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                self._emit_exit(symbol, item, event_time, reason="max_hold")

    def _inverse_vol_weights(
        self, book: dict[str, str], vols: dict[str, float]
    ) -> dict[str, float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in book
            if vols.get(symbol, 0.0) > _EPS
        }
        total = sum(inv.values())
        if total <= _EPS:
            return {}
        return {symbol: (inv[symbol] / total) * self.target_gross_exposure for symbol in inv}

    def _emit_targets(
        self,
        desired: dict[str, str | None],
        scores: dict[str, float],
        zscores: dict[str, float],
        vols: dict[str, float],
        event_time: Any,
    ) -> None:
        book = {symbol: mode for symbol, mode in desired.items() if mode in {"LONG", "SHORT"}}
        weights = self._inverse_vol_weights(book, vols)
        for symbol, item in self._state.items():
            target_mode = desired.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target_mode not in {"LONG", "SHORT"}:
                if item.mode != "OUT":
                    self._emit_exit(symbol, item, event_time, reason="rank_lapsed")
                continue
            if item.mode == target_mode:
                if symbol in scores:
                    item.score = float(scores[symbol])
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
            weight = float(weights.get(symbol, 0.0))
            delay = float(scores.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if alloc <= 0.0:
                # Zero-alloc entries omit ``target_allocation`` from metadata and
                # the engine resizes them to its DEFAULT allocation -- an unsized,
                # un-vol-gated position. Skip the entry; if a side-flip EXIT was
                # just emitted, drop to OUT (mirroring ``_emit_exit``) so state
                # matches it.
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.bars_since_exit = 0
                    item.score = None
                continue
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=delay,
                target_mode=target_mode,
                price_delay=delay,
                delay_z=float(zscores.get(symbol, 0.0)),
                inverse_vol_weight=weight,
            )
            if self.max_symbol_exposure_pct > 0.0:
                metadata["max_symbol_exposure_pct"] = min(
                    float(metadata.get("max_symbol_exposure_pct", self.max_symbol_exposure_pct)),
                    self.max_symbol_exposure_pct,
                )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(float(zscores.get(symbol, delay))))),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = delay

    def _emit_exit(self, symbol: str, item: _State, event_time: Any, *, reason: str) -> None:
        price = item.closes[-1] if item.closes else None
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": _STRATEGY_NAME, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Cross-sectional
# long-short BOOK: admission route is ``allow_multi_asset=True`` -- ranked by an
# unconditional delay CHARACTERISTIC, NOT carry, so it is honestly EXCLUDED from
# any carry tag-superset allowlist (no fake carry tag).  The data-PC gates it on
# RPT >= 10bps per split and incremental orthogonal ``factor_ic`` vs the lead-lag
# incumbents PLUS a binding illiquidity-alias (dollar-volume/Amihud) control.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "price_delay",
    "dimson_beta",
    "investor_recognition",
    "information_diffusion",
    "low_turnover",
    "crypto",
)

# Candidate slice.  The decision clock is the internal ISO ``_week_key``
# (timestamp-based, fires once per calendar week at ANY feed frequency), so the
# WEEKLY-decision params (min_hold_decisions and the exit band) are timeframe
# INVARIANT and stay fixed at 4h/1h.  The regression, however, is DAY-scaled.
#
# FAITHFUL-SCALING CHOICE (task option 2, "scale delay_window so the regression
# spans the same wall-clock"): ``delay_window`` (the count of trailing log returns
# feeding the Dimson market-model) scales x6 / x24 to hold the ~180d/270d
# estimation SPAN, and ``vol_window`` scales likewise.  ``delay_lags`` is the
# Dimson lag ORDER (number of lagged-benchmark regressors), NOT a wall-clock
# horizon -- it is held fixed: scaling it to a day-equivalent count (5 -> 30 -> 120)
# would blow the schema cap (64) and load the normal equations with dozens of
# near-collinear autocorrelated lag columns, degrading the R^2 share.  NOTE /
# HONEST FLAG for the PM: with a fixed small lag order at sub-daily bars the
# statistic measures the BAR-NATIVE (intraday, ~5-bar) lead-lag rather than the
# multi-day information-diffusion the daily cell captures -- a documented change in
# economic horizon.  This lane is an explicit falsification probe, so the shift is
# recorded rather than papered over.  The variant NAMES keep their daily-window
# labels (d1_180d_l5, d1_270d_l2) as stable slice identifiers across timeframes.
# ``delay_window`` at 1h (4320/6480) sits under the ~9000-bar cap.
_PRICE_DELAY_PREMIUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "d1_180d_l5",
            "delay_window": 1080,
            "delay_lags": 5,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 180,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "d1_270d_l2",
            "delay_window": 1620,
            "delay_lags": 2,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 180,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1h": (
        {
            "variant": "d1_180d_l5",
            "delay_window": 4320,
            "delay_lags": 5,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 720,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "d1_270d_l2",
            "delay_window": 6480,
            "delay_lags": 2,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 720,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1d": (
        {
            "variant": "d1_180d_l5",
            "delay_window": 180,
            "delay_lags": 5,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 30,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "d1_270d_l2",
            "delay_window": 270,
            "delay_lags": 2,
            "score_mode": "d1",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "vol_window": 30,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalPriceDelayPremiumStrategy"]
