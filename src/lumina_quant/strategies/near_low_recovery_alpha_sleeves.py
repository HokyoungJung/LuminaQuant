"""Cross-sectional 52-week-LOW capitulation-recovery sleeve (residualized anchor).

``CrossSectionalNearLowRecoveryStrategy`` is the untested mirror of the shipped
``CrossSectionalNearHighAnchoringStrategy``: instead of ranking on nearness to
the trailing HIGH (a level statistic), it ranks the cross-section on the
DYNAMICS around the trailing LOW -- how far a name has REBOUNDED off its
window-minimum and how long AGO that capitulation low was printed -- then
residualizes that composite on the incumbent's exact nearness-to-high statistic
so the sleeve trades ONLY the low-side information the high anchor cannot see.

The trailing low is a salient reference point that carries two pieces of
information the ``max(highs)`` scorer structurally cannot access: the rebound
MAGNITUDE off the capitulation price, and the RECENCY (argmin timing) of that
low.  Long aged, confirmed recoveries; short fresh-low grinders / failed
recoveries.

THEORY / PROVENANCE
--------------------
- George & Hwang (2004), *Journal of Finance* 59(5) -- the 52-week ratio anchors
  the cross-section; this is its low-side counterpart.
- Bhootra & Hur (2013), *Journal of Banking & Finance* 37(10) -- the recency
  ratio: time since the salient extreme predicts returns (transplanted here to
  the LOW).
- De Bondt & Thaler (1985), *JF* 40(3); Odean (1998), *JF* 53(5) +
  Barberis & Xiong (2009), *JF* 64(2) -- disposition/realization utility: holders
  capitulate INTO the first bounce, stretching post-capitulation under-reaction.
- Jia, Simkins, Yan, Zhang & Zhao (2025), *JBF* (S0378426625002122 / SSRN
  5386180) -- Nearness-52 predicts the CRYPTO cross-section; the LOW is the
  untested mirror.  George-Hwang's own evidence makes the HIGH the DOMINANT
  anchor, which is carried honestly into the EXPECTED NULL.

SIGNAL SPEC
-----------
Per completed decision bar (OHLC), per symbol:

1. Append ``close`` and ``low`` (``low`` CAPPED at ``close`` so the rebound stays
   ``>= 0``; when ``low`` is missing the ``close`` is used), plus ``high``
   (floored at ``close`` so nearness stays in ``(0, 1]``) for the residualizer.
2. ``eff_lookback = min(low_lookback_bars, bars_available)`` (young-symbol
   ``max_available`` admission above the ``min_history_bars`` floor).
3. ``trailing_low = min(low over eff_lookback)``;
   ``REBOUND = ln(close / max(trailing_low, eps)) >= 0``.
4. ``LOW_RECENCY = bars_since_last_touch(window_min) / eff_lookback in [0, 1)`` --
   a single reverse scan, tie-break MOST RECENT touch (time since the market
   last traded at the capitulation price).
5. ``composite = z_xs(REBOUND) + z_xs(LOW_RECENCY)`` -- fixed equal weights.
6. LOAD-BEARING DECOUPLER: XS-residualize ``composite`` on ``nearness_z`` where
   ``nearness = close / max(high over the SAME eff_lookback)`` -- the near-high
   incumbent's EXACT statistic -- via the shared single-regressor Gram-Schmidt
   ``cross_sectional_residualize`` primitive (var<=eps guard -> degenerate
   regressor dropped; residual collapse -> abstain).  The sleeve trades ONLY the
   low-side component unexplained by the high anchor.
7. Rank the residual: LONG the top ``quantile_pct`` (confirmed recovery -- a
   large, aged rebound beyond what distance-from-high explains), SHORT the bottom
   ``quantile_pct`` (fresh-low grinders / failed recoveries).

Sizing is inverse-realized-vol risk parity normalised to
``target_gross_exposure`` and clamped by ``target_vol`` (sibling convention).
Cadence is a slow ``rebalance_bars`` clock plus a hard ``min_hold_bars`` floor;
stops / ``max_hold_bars`` age every bar.  The book self-skips below
``min_symbols``.  The SIGN is FIXED ex-ante (long recovered/aged-low, short
fresh-low); the opposite outcome is a declared falsification result, never a
flip parameter.

DISTINCT-FROM
-------------
``CrossSectionalNearHighAnchoringStrategy`` scores ``close / max(highs)`` -- a
level statistic with no access to the low order statistic or its timing.  Two
paths with IDENTICAL nearness-to-high (a V-shaped recovery vs a monotone grind
into a fresh low) are TIED there but take OPPOSITE sides here, and the
residualizer cannot erase that split because both carry the same regressor
value.  The build-gate test pins this against the REAL near-high incumbent and
against ``LongRunOverreactionReversalStrategy`` (path-blind formation return).

This module is data-local (no I/O, no hidden configuration bus), pure
Python/numpy-free (``math`` only), and never raises from ``calculate_signals``.
It ships WITHOUT ``@register`` (inert until the integration wave wires it).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _annualize_per_bar_vol,
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

_STRATEGY_ID = "near_low_recovery"
_STRATEGY_NAME = "CrossSectionalNearLowRecoveryStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    lows: deque[float]
    highs: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
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


def _cross_z(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score of ``values`` (0.0 for every symbol if degenerate)."""
    count = len(values)
    if count == 0:
        return {}
    mean_value = sum(values.values()) / float(count)
    variance = sum((value - mean_value) ** 2 for value in values.values()) / float(
        max(1, count - 1)
    )
    sigma = variance**0.5
    if sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: (value - mean_value) / sigma for symbol, value in values.items()}


def _rebound_log(close: float, trailing_low: float) -> float | None:
    """Log rebound off the trailing low ``ln(close / max(trailing_low, eps)) >= 0``."""
    denom = max(trailing_low, _EPS)
    if close <= 0.0 or denom <= 0.0:
        return None
    value = math.log(close / denom)
    if not math.isfinite(value):
        return None
    # trailing_low is the window minimum (<= the current bar low <= close), so a
    # tiny negative from float noise is clamped to the 0 floor.
    return float(max(0.0, value))


def _low_recency(window_lows: list[float], eff_lookback: int) -> float | None:
    """Fraction of the window since the MOST-RECENT touch of the window minimum.

    ``0`` when the current bar prints the window low (a fresh capitulation), up
    toward ``1`` as the low ages.  Deterministic single reverse scan with a
    most-recent tie-break: the newest bar equal to the window minimum wins.
    """
    if eff_lookback <= 0 or not window_lows:
        return None
    window_min = min(window_lows)
    last = len(window_lows) - 1
    touch = 0
    for offset in range(last, -1, -1):
        if window_lows[offset] <= window_min:
            touch = offset
            break
    bars_since = last - touch
    value = bars_since / float(eff_lookback)
    if not math.isfinite(value):
        return None
    return float(value)


@register("strategy", "CrossSectionalNearLowRecoveryStrategy", interface="event_driven")
class CrossSectionalNearLowRecoveryStrategy(Strategy):
    """Long-short XS 52-week-LOW rebound/recency, residualized on nearness-to-high.

    See the module docstring for the full theory, signal spec, and the
    distinct-from rationale versus the near-high anchoring incumbent (level
    statistic) and the long-run overreaction reversal book (path-blind formation
    return).  Reads only local event/bar OHLC; performs no I/O and never raises
    from ``calculate_signals``.
    """

    # Daily-bar weekly-cadence cross-sectional book; >= 30-minute live floor
    # trivially cleared.
    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # 52-week-equivalent trailing-low window (bars).  The data-PC sweeps
            # 10/20/30/52wk of 1d bars.
            "low_lookback_bars": HyperParam.integer(
                "low_lookback_bars", default=364, low=20, high=200000
            ),
            # Per-symbol history floor: below this a symbol is skipped; between
            # this and ``low_lookback_bars`` it is admitted via max_available.
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=60, low=5, high=100000
            ),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=2000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=7, low=1, high=100000),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=14, low=0, high=100000),
            # When True, the composite is residualized on nearness-to-high before
            # ranking (the load-bearing decoupler); False is the ablation cell.
            "residualize": HyperParam.boolean("residualize", default=True, grid=[True, False]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=6, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
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
        self.low_lookback_bars = max(3, int(resolved["low_lookback_bars"]))
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.residualize = bool(resolved["residualize"])
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.low_lookback_bars, self.vol_window + 1, self.max_hold_bars)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                lows=deque(maxlen=size),
                highs=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target scalar annualizes the per-bar portfolio vol
        # via sqrt(bars_per_year) derived from the median gap here.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "lows": list(item.lows),
                    "highs": list(item.highs),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                for attr in ("closes", "lows", "highs"):
                    target = getattr(item, attr)
                    target.clear()
                    maxlen = int(target.maxlen or 0)
                    values = _coerce_float_list(payload.get(attr))
                    for value in values[-maxlen:] if maxlen else values:
                        target.append(value)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        low = safe_float(snapshot.low)
        if low is None or low > close:
            low = close
        high = safe_float(snapshot.high)
        if high is None or high < close:
            high = close
        item.closes.append(close)
        item.lows.append(low)
        item.highs.append(high)
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
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
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _raw_features(
        self,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, dict[str, Any]],
    ]:
        """Return ``(rebound, low_recency, nearness, vols, metas)`` per eligible symbol."""
        rebound: dict[str, float] = {}
        low_recency: dict[str, float] = {}
        nearness: dict[str, float] = {}
        vols: dict[str, float] = {}
        metas: dict[str, dict[str, Any]] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            lows = list(item.lows)
            highs = list(item.highs)
            if (
                len(closes) < self.min_history_bars
                or len(lows) < self.min_history_bars
                or len(highs) < self.min_history_bars
            ):
                continue
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            eff_lookback = min(self.low_lookback_bars, len(lows))
            window_lows = lows[-eff_lookback:]
            trailing_low = min(window_lows)
            if trailing_low <= _EPS:
                continue
            reb = _rebound_log(close, trailing_low)
            rec = _low_recency(window_lows, eff_lookback)
            if reb is None or rec is None:
                continue
            trailing_high = max(highs[-eff_lookback:])
            if trailing_high <= _EPS:
                continue
            near = close / trailing_high
            if not math.isfinite(near):
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            rebound[symbol] = float(reb)
            low_recency[symbol] = float(rec)
            nearness[symbol] = float(near)
            vols[symbol] = float(vol)
            metas[symbol] = {
                "rebound_log": float(reb),
                "low_recency": float(rec),
                "nearness": float(near),
                "trailing_low": float(trailing_low),
                "trailing_high": float(trailing_high),
                "lookback_used": int(eff_lookback),
                "full_lookback": bool(eff_lookback >= self.low_lookback_bars),
            }
        return rebound, low_recency, nearness, vols, metas

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        rebound, low_recency, nearness, vols, metas = self._raw_features()
        if len(rebound) < self.min_symbols:
            return {}, {}

        z_rebound = _cross_z(rebound)
        z_recency = _cross_z(low_recency)
        nearness_z = _cross_z(nearness)
        composite = {symbol: z_rebound[symbol] + z_recency[symbol] for symbol in rebound}

        ordered = sorted(composite)  # deterministic symbol order for the residualizer
        if self.residualize:
            residual_vec = cross_sectional_residualize(
                [composite[symbol] for symbol in ordered],
                [[nearness_z[symbol] for symbol in ordered]],
            )
            if residual_vec is None or len(residual_vec) != len(ordered):
                return {}, {}
            residual = {symbol: float(residual_vec[idx]) for idx, symbol in enumerate(ordered)}
        else:
            residual = {symbol: float(composite[symbol]) for symbol in ordered}

        # Residual collapse (composite collinear with nearness, or no orthogonal
        # dispersion) -> nothing to trade -> abstain.
        resid_values = list(residual.values())
        resid_mean = sum(resid_values) / float(len(resid_values))
        resid_var = sum((value - resid_mean) ** 2 for value in resid_values) / float(
            max(1, len(resid_values) - 1)
        )
        if resid_var**0.5 <= _EPS:
            return {}, {}

        for symbol in rebound:
            metas[symbol]["rebound_z"] = float(z_rebound[symbol])
            metas[symbol]["low_recency_z"] = float(z_recency[symbol])
            metas[symbol]["nearness_z"] = float(nearness_z[symbol])
            metas[symbol]["composite"] = float(composite[symbol])
            metas[symbol]["residual"] = float(residual[symbol])

        # Ascending by residual (symbol tiebreak): top quantile LONG (confirmed
        # recovery), bottom quantile SHORT (fresh-low grinders).
        rank_order = sorted(residual, key=lambda symbol: (residual[symbol], symbol))
        count = len(rank_order)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        short_syms = rank_order[:n_side]
        long_syms = rank_order[-n_side:]

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in long_syms:
            targets[symbol] = ("LONG", float(residual[symbol]), metas[symbol])
        if self.allow_short:
            for symbol in short_syms:
                if symbol in targets:
                    continue
                targets[symbol] = ("SHORT", float(residual[symbol]), metas[symbol])
        return targets, vols

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        vols: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in targets
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        # ``portfolio_vol`` is the inverse-vol-weighted PER-BAR vol; the
        # ``inv / total_inv`` normalization above is scale-invariant (annualizing
        # every vol_i cancels), so the risk-parity weights are horizon-free.
        # The vol-target SCALAR, however, compares ``portfolio_vol`` against an
        # annual-scale ``target_vol`` (0.20): annualize the per-bar estimate via
        # sqrt(bars_per_year) inferred from observed bar spacing first, otherwise
        # the Moreira-Muir clamp is INERT. When spacing is unavailable we pass
        # through (scalar=1.0) rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            portfolio_vol_ann = _annualize_per_bar_vol(portfolio_vol, self._recent_times)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    # ------------------------------------------------------------------ #
    # aging / emission
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        max_hold = self.max_hold_bars if self.max_hold_bars > 0 else (1 << 62)
        _age_cross_positions(
            self.events,
            self._state,  # type: ignore[arg-type]
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=max_hold,
        )

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Record the decision-bar epoch so the vol-target scalar can infer bar
        # spacing (this runs once per new bar, before the rebalance gate).
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Stops / max-hold age EVERY bar so a held name is always protected,
        # independent of the slow rebalance clock.
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        targets, vols = self._score_and_select()
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    # Min-hold floor: a would-be exit inside the hold window is
                    # suppressed (turnover discipline).
                    if item.bars_held < self.min_hold_bars:
                        continue
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                continue
            # Min-hold floor: a would-be side-flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.bars_held < self.min_hold_bars:
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
            alloc = max(0.0, self.base_allocation * weight)
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                **meta,
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
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a pure cross-sectional long-short (NOT carry, NOT momentum), so it
# is honestly EXCLUDED from any carry/momentum tag-superset allowlist -- no fake
# carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "anchoring",
    "fifty_two_week_low",
    "capitulation_recovery",
    "recency",
    "low_turnover",
    "crypto",
)

# Candidate slice (weekly rebalance via ``rebalance_bars``).  The published effect
# is a 52-week window; the data-PC owns the 10/20/30/52wk horizon factor_ic sweep,
# so we seed two anchoring lookbacks (~20wk and ~52wk) with the residualized
# decoupler on.  This lane is a pure wall-clock cross-section: EVERY bar param
# (low/vol windows, the min-history floor, and the ``rebalance_bars`` /
# ``min_hold_bars`` decision clock) scales uniformly x6 at 4h and x24 at 1h so the
# same calendar horizons are preserved; the fractions/counts (quantile_pct,
# min_symbols, residualize, allow_short, target_gross_exposure) are unit-free and
# stay fixed.  ``low_lookback_bars`` at 1h (8736) sits just under the ~9000-bar cap.
_NEAR_LOW_RECOVERY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "low52_recovery_resid",
            "low_lookback_bars": 2184,
            "min_history_bars": 360,
            "vol_window": 120,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_bars": 84,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "low20wk_recovery_resid",
            "low_lookback_bars": 840,
            "min_history_bars": 360,
            "vol_window": 120,
            "quantile_pct": 0.25,
            "rebalance_bars": 42,
            "min_hold_bars": 84,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1h": (
        {
            "variant": "low52_recovery_resid",
            "low_lookback_bars": 8736,
            "min_history_bars": 1440,
            "vol_window": 480,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_bars": 336,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "low20wk_recovery_resid",
            "low_lookback_bars": 3360,
            "min_history_bars": 1440,
            "vol_window": 480,
            "quantile_pct": 0.25,
            "rebalance_bars": 168,
            "min_hold_bars": 336,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1d": (
        {
            "variant": "low52_recovery_resid",
            "low_lookback_bars": 364,
            "min_history_bars": 60,
            "vol_window": 20,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_bars": 14,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "low20wk_recovery_resid",
            "low_lookback_bars": 140,
            "min_history_bars": 60,
            "vol_window": 20,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "min_hold_bars": 14,
            "min_symbols": 6,
            "residualize": True,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalNearLowRecoveryStrategy"]
