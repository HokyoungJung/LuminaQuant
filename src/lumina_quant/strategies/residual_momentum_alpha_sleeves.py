"""BTC-purged, ADF-anti-gated cross-sectional residual MOMENTUM (W2-1).

``TrendGatedResidualMomentumStrategy`` is a weekly cross-sectional long-short on
the Blitz-Huij-Martens (2011) residual-momentum score computed on BTC-PURGED
returns, with one load-bearing differentiator versus every registered
residual/momentum incumbent: a residual NON-stationarity anti-gate.  A symbol is
tradeable ONLY when its beta-hedged residual-LEVEL series
``log close_i - beta * log close_btc`` FAILS the augmented Dickey-Fuller
stationarity screen (``adf_t_statistic``) -- the exact COMPLEMENT of
``StationarityGatedResidualReversionStrategy``'s gate.  Momentum in a hedged
spread is only coherent where the spread is NOT mean-reverting: stationary
residuals belong to the reversion sleeve (fade), trending residuals belong here
(ride).  The two sleeves partition residual space by the ADF statistic with
opposite signs.

THEORY / PROVENANCE
-------------------
- Blitz, Huij & Martens (2011, J. Empirical Finance 18(3)): residual momentum
  (momentum on the idiosyncratic return after purging factor exposure) carries
  roughly half the volatility and roughly double the information ratio of raw
  momentum -- the residual-vol scaling is what delivers that IR-doubling, so it
  is applied here.
- Blitz, Hanauer & Vidojevic (2020, IREF 69); Liu, Tsyvinski & Wu (2022, JF
  77(2)) establish that a crypto cross-sectional momentum premium exists (with a
  documented decay caveat, Rosen-Wang 2024) -- the honest expectation here is a
  modest, possibly-null edge, measured standalone at the data-PC.
- Daniel & Moskowitz (2016, JFE 122(2)): raw-momentum crashes are driven by the
  crowded market-timing (beta) component; residualizing removes exactly that
  component, which is why the residual ranks churn structurally less than raw
  ranks (a turnover argument, not a return promise).
- Hong & Stein (1999, JF 54(6)): gradual information diffusion produces the
  under-reaction the residual isolates (the counterparty is the beta-anchored
  holder who sells idiosyncratic winners back toward the beta-implied path).
- The ADF screen is the pure-Python ``adf_t_statistic``
  (``indicators/stationarity.py``, no numpy/scipy/statsmodels) evaluated on the
  TRAILING residual-level window, so there is no full-sample lookahead.

MECHANISM (per completed daily bar; weekly decision clock)
----------------------------------------------------------
Per symbol, on an ISO-week internal decision clock (decide at most once per
week even when fed daily bars):

1. ``beta`` = ``rolling_beta`` of the asset's daily log-returns on the
   benchmark's over ``beta_window_bars`` samples (the hedge ratio).
2. Residual return series ``r_i,t - beta * r_btc,t`` over the ``formation_weeks``
   formation window, EXCLUDING the most recent ``skip_weeks`` (the BHM skip that
   removes 1-week mechanical reversal); ``score`` = cumulated residual return /
   residual-return volatility (the BHM residual-vol scaling).
3. NON-STATIONARITY ANTI-GATE (the sole differentiator): the trailing
   residual-LEVEL series ``log close_i - beta * log close_btc`` over
   ``adf_window_bars`` must FAIL the ADF stationarity screen, i.e.
   ``adf_t_statistic > adf_nonstationarity_min_t`` (a residual whose level is
   mean-reverting -- ADF strongly negative -- is ABSTAINED and left to the
   reversion sleeve).
4. Cross-sectionally rank the surviving scores; LONG the top ``quantile_pct``
   (positive residual momentum only), SHORT the bottom (negative only),
   inverse-residual-vol sized, normalized to ``max_gross``.

Cadence discipline: a hard ``min_hold_decisions`` (>= 4 weekly decisions -- the
repo's PROVEN C1 min-hold turnover rescue) plus a ``cooldown_decisions`` window
suppress rank flicker; a per-symbol ``min_history_bars`` + ``min_price`` liquid
screen gates entry.  An optional offsetting benchmark hedge leg
(``hedge_benchmark``, default OFF) nets the book's beta exposure to ~0.

DISTINCT-FROM (the incumbents this sleeve was built to diverge from)
--------------------------------------------------------------------
- ``ResidualMomentumRotationStrategy`` (adaptive_crypto_alpha_sleeves.py) scores
  a POINT residual (a single ``log_return`` over one lookback) with no skip week
  and no min-hold: it reacts to a spike inside the skip window that this sleeve
  deliberately excludes.
- ``ResidualEquityMomentumStrategy`` (equity_xs_factor_alpha_sleeves.py) is the
  true BHM construction (settable benchmark, residual-vol scaling) but is
  UNCONDITIONAL: it ranks a stationary/oscillating residual this sleeve's ADF
  anti-gate abstains on.  The ADF anti-gate is the SOLE mechanism differentiator
  versus this nearest neighbour.
- ``TopCapTimeSeriesMomentumStrategy`` (topcap_tsmom.py) trades RAW momentum: a
  hidden idiosyncratic winner whose raw formation return is negative (dragged
  down by a falling benchmark) is short/flat there but long here.
- ``StationarityGatedResidualReversionStrategy`` (residual_reversion_alpha_sleeves.py)
  fades a STATIONARY residual; this sleeve rides a NON-stationary one -- the two
  gates are exact complements.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` + ``deque`` + the pure indicator primitives, no numpy), completed-bar,
and never raises from ``calculate_signals``.  It ships WITHOUT ``@register``
(inert until the integration wave wires it as ``research_only``).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import rolling_beta, sample_std
from lumina_quant.indicators.stationarity import adf_t_statistic
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

_STRATEGY_ID = "trend_gated_residual_momentum"
_STRATEGY_NAME = "TrendGatedResidualMomentumStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed --
# the first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30


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
    beta_exposure: float = 0.0  # signed beta contribution while live (for hedge)


@dataclass(slots=True)
class _AssetScore:
    """Scored residual-momentum result for one symbol at a decision bar."""

    score: float
    beta: float
    residual_cumret: float
    residual_vol: float
    sizing_vol: float
    adf_t: float


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


def _log_return_series(values: list[float]) -> list[float]:
    """Return the log-return series of a strictly-positive level series."""
    out: list[float] = []
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        cur = values[idx]
        if prev > 0.0 and cur > 0.0:
            out.append(math.log(cur / prev))
    return out


def _log_levels(values: list[float]) -> list[float] | None:
    """Return log of each strictly-positive close, or ``None`` if any is invalid."""
    out: list[float] = []
    for value in values:
        parsed = safe_float(value)
        if parsed is None or parsed <= 0.0:
            return None
        out.append(math.log(parsed))
    return out


@register("strategy", "TrendGatedResidualMomentumStrategy", interface="event_driven")
class TrendGatedResidualMomentumStrategy(Strategy):
    """Weekly XS long-short BHM residual momentum, gated on residual NON-stationarity.

    See the module docstring for the full theory, mechanism, and the
    single-load-bearing-differentiator (ADF non-stationarity anti-gate)
    rationale versus the residual/momentum incumbents.  This class only reads
    local event/bar OHLCV; it performs no I/O and never raises from
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
            "beta_window_bars": HyperParam.integer(
                "beta_window_bars", default=90, low=8, high=20000
            ),
            "formation_weeks": HyperParam.integer("formation_weeks", default=8, low=2, high=104),
            "skip_weeks": HyperParam.integer("skip_weeks", default=1, low=0, high=52),
            "bars_per_week": HyperParam.integer("bars_per_week", default=7, low=1, high=168),
            "adf_window_bars": HyperParam.integer("adf_window_bars", default=90, low=9, high=20000),
            "adf_lags": HyperParam.integer("adf_lags", default=0, low=0, high=32),
            "adf_nonstationarity_min_t": HyperParam.floating(
                "adf_nonstationarity_min_t", default=-1.5, low=-10.0, high=5.0
            ),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "residual_vol_window_bars": HyperParam.integer(
                "residual_vol_window_bars", default=56, low=3, high=8192
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=0, high=100000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=100000
            ),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=120, low=5, high=100000
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "hedge_benchmark": HyperParam.boolean(
                "hedge_benchmark", default=False, grid=[True, False]
            ),
            "max_gross": HyperParam.floating("max_gross", default=1.0, low=0.0, high=3.0),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.0, low=0.0, high=0.50),
            "min_price": HyperParam.floating("min_price", default=0.01, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.benchmark_symbol = _default_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.beta_window_bars = max(8, int(resolved["beta_window_bars"]))
        self.formation_weeks = max(2, int(resolved["formation_weeks"]))
        self.skip_weeks = max(0, int(resolved["skip_weeks"]))
        self.bars_per_week = max(1, int(resolved["bars_per_week"]))
        self.adf_window_bars = max(9, int(resolved["adf_window_bars"]))
        self.adf_lags = max(0, int(resolved["adf_lags"]))
        self.adf_nonstationarity_min_t = float(resolved["adf_nonstationarity_min_t"])
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.residual_vol_window_bars = max(3, int(resolved["residual_vol_window_bars"]))
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.min_history_bars = max(5, int(resolved["min_history_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.hedge_benchmark = bool(resolved["hedge_benchmark"])
        self.max_gross = max(0.0, float(resolved["max_gross"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Derived bar spans of the formation / skip windows.
        self.formation_bars = self.formation_weeks * self.bars_per_week
        self.skip_bars = self.skip_weeks * self.bars_per_week
        size = _state_size(
            self.beta_window_bars,
            self.formation_bars + self.skip_bars,
            self.adf_window_bars,
            self.residual_vol_window_bars,
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
                    "beta_exposure": float(item.beta_exposure),
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
                exposure = safe_float(payload.get("beta_exposure"))
                item.beta_exposure = float(exposure) if exposure is not None else 0.0
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
    def _residual_returns(
        self, asset_rets: list[float], bench_rets: list[float], beta: float
    ) -> list[float] | None:
        """Return the formation-window residual returns excluding the skip window."""
        need = self.formation_bars + self.skip_bars
        aligned = min(len(asset_rets), len(bench_rets))
        if aligned < need:
            return None
        asset_tail = asset_rets[-need:]
        bench_tail = bench_rets[-need:]
        if self.skip_bars > 0:
            asset_seg = asset_tail[: -self.skip_bars]
            bench_seg = bench_tail[: -self.skip_bars]
        else:
            asset_seg = asset_tail
            bench_seg = bench_tail
        residuals = [a - beta * b for a, b in zip(asset_seg, bench_seg, strict=False)]
        return residuals if len(residuals) >= 2 else None

    def _sizing_vol(
        self, asset_rets: list[float], bench_rets: list[float], beta: float
    ) -> float | None:
        """Trailing residual-return volatility over ``residual_vol_window_bars``."""
        aligned = min(len(asset_rets), len(bench_rets))
        window = min(aligned, self.residual_vol_window_bars)
        if window < 3:
            return None
        residuals = [
            a - beta * b for a, b in zip(asset_rets[-window:], bench_rets[-window:], strict=False)
        ]
        return sample_std(residuals)

    def _asset_score(self, closes: list[float], bench_closes: list[float]) -> _AssetScore | None:
        """Return the BHM residual-momentum score gated on residual NON-stationarity."""
        if len(closes) < self.min_history_bars or len(bench_closes) < self.min_history_bars:
            return None
        asset_rets = _log_return_series(closes)
        bench_rets = _log_return_series(bench_closes)
        aligned = min(len(asset_rets), len(bench_rets))
        if aligned < 3:
            return None
        beta_depth = min(aligned, self.beta_window_bars)
        if beta_depth < 3:
            return None
        beta = rolling_beta(asset_rets[-beta_depth:], bench_rets[-beta_depth:])
        if beta is None:
            return None

        residuals = self._residual_returns(asset_rets, bench_rets, beta)
        if residuals is None:
            return None
        cumret = math.fsum(residuals)
        res_vol = sample_std(residuals)
        if res_vol is None or res_vol <= _EPS:
            return None
        score = cumret / res_vol
        if not math.isfinite(score):
            return None

        sizing_vol = self._sizing_vol(asset_rets, bench_rets, beta)
        if sizing_vol is None or sizing_vol <= _EPS:
            return None

        # NON-STATIONARITY ANTI-GATE: the residual LEVEL series must FAIL the ADF
        # stationarity screen (adf_t above the threshold) before the trend is
        # ridden -- the sole mechanism differentiator versus the reversion sleeve.
        window = min(len(closes), len(bench_closes), self.adf_window_bars)
        if window < 9:
            return None
        asset_levels = _log_levels(closes[-window:])
        bench_levels = _log_levels(bench_closes[-window:])
        if asset_levels is None or bench_levels is None:
            return None
        residual_levels = [asset_levels[idx] - beta * bench_levels[idx] for idx in range(window)]
        adf_t = adf_t_statistic(residual_levels, lags=self.adf_lags)
        if adf_t is None or adf_t <= self.adf_nonstationarity_min_t:
            # Stationary (mean-reverting) or undefined residual -> abstain: this
            # is the reversion sleeve's territory, not momentum's.
            return None
        return _AssetScore(
            score=float(score),
            beta=float(beta),
            residual_cumret=float(cumret),
            residual_vol=float(res_vol),
            sizing_vol=float(sizing_vol),
            adf_t=float(adf_t),
        )

    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, _AssetScore]]:
        bench = self._state.get(self.benchmark_symbol)
        if bench is None:
            return {}, {}
        bench_closes = list(bench.closes)
        scores: dict[str, _AssetScore] = {}
        metas: dict[str, dict[str, Any]] = {}
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            result = self._asset_score(list(item.closes), bench_closes)
            if result is None:
                continue
            scores[symbol] = result
            metas[symbol] = {
                "beta": result.beta,
                "residual_cumret": result.residual_cumret,
                "residual_vol": result.residual_vol,
                "sizing_vol": result.sizing_vol,
                "adf_t_statistic": result.adf_t,
                "adf_nonstationarity_min_t": self.adf_nonstationarity_min_t,
            }

        if len(scores) < self.min_symbols:
            return {}, scores

        # Deterministic ascending order by score (symbol tiebreak): the top
        # quantile with POSITIVE residual momentum is long, the bottom quantile
        # with NEGATIVE residual momentum is short.
        ordered = sorted(scores, key=lambda symbol: (scores[symbol].score, symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, scores
        short_candidates = ordered[:n_side]
        long_candidates = ordered[-n_side:]

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in long_candidates:
            if scores[symbol].score > 0.0:
                targets[symbol] = ("LONG", float(scores[symbol].score), metas[symbol])
        if self.allow_short:
            for symbol in short_candidates:
                if symbol in targets:
                    continue
                if scores[symbol].score < 0.0:
                    targets[symbol] = ("SHORT", float(scores[symbol].score), metas[symbol])
        return targets, scores

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        scores: dict[str, _AssetScore],
    ) -> dict[str, float]:
        """Inverse-residual-vol risk-parity weights normalized to ``max_gross``."""
        inv = {
            symbol: 1.0 / max(scores[symbol].sizing_vol, _EPS)
            for symbol in targets
            if symbol in scores and scores[symbol].sizing_vol > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}
        return {symbol: (inv[symbol] / total_inv) * self.max_gross for symbol in inv}

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Age every held position / cooldown once per weekly decision bar.
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
            else:
                item.bars_since_exit += 1
        targets, scores = self._score_and_select()
        weights = self._inverse_vol_weights(targets, scores)
        self._emit_targets(targets, weights, event_time)
        self._emit_benchmark_hedge(event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol:
                continue
            price = item.closes[-1] if item.closes else None
            target = targets.get(symbol)
            if target is None:
                # Would-be exit: honour the hard min-hold before flattening.
                if item.mode != "OUT":
                    if item.bars_held < self.min_hold_decisions:
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
                    self._flatten(item)
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                item.beta_exposure = self._beta_exposure(
                    target_mode, weights.get(symbol, 0.0), meta
                )
                continue
            # Would-be side flip: suppressed inside the hard min-hold window.
            if item.mode != "OUT" and item.bars_held < self.min_hold_decisions:
                continue
            # Cooldown gate for a fresh entry from flat.
            if item.mode == "OUT" and item.bars_since_exit < self.cooldown_decisions:
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
            item.beta_exposure = self._beta_exposure(target_mode, weight, meta)

    @staticmethod
    def _beta_exposure(mode: str, weight: float, meta: dict[str, Any]) -> float:
        signed = 1.0 if mode == "LONG" else -1.0
        beta = float(meta.get("beta", 0.0) or 0.0)
        return signed * float(weight) * beta

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None
        item.beta_exposure = 0.0

    def _emit_benchmark_hedge(self, event_time: Any) -> None:
        """Offset the book's net beta with a benchmark leg (market-neutral)."""
        bench = self._state.get(self.benchmark_symbol)
        if bench is None:
            return
        price = bench.closes[-1] if bench.closes else None
        net = math.fsum(
            item.beta_exposure
            for symbol, item in self._state.items()
            if symbol != self.benchmark_symbol and item.mode in {"LONG", "SHORT"}
        )
        if not self.hedge_benchmark or abs(net) <= _EPS:
            if bench.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=self.benchmark_symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "hedge_flat"},
                )
                bench.mode = "OUT"
                bench.entry_price = None
            return
        hedge_mode = "SHORT" if net > 0.0 else "LONG"
        if bench.mode != "OUT" and bench.mode != hedge_mode:
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=self.benchmark_symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={"strategy": _STRATEGY_NAME, "reason": "hedge_flip"},
            )
            bench.mode = "OUT"
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=abs(net),
            max_order_value=self.max_order_value,
            target_mode=hedge_mode,
            beta=1.0,
            reason="benchmark_hedge",
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=self.benchmark_symbol,
            event_time=event_time,
            signal_type=hedge_mode,
            strength=1.0,
            price=price,
            metadata=metadata,
        )
        bench.mode = hedge_mode
        bench.entry_price = price


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a cross-sectional market-neutral residual-momentum long-short --
# NOT carry -- so it is honestly EXCLUDED from any carry-tag allowlist route; no
# fake carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "residual_momentum",
    "market_neutral",
    "stationarity_gated",
    "momentum",
    "crypto",
)

# Candidate slice.  The BHM construction (8/12wk formation, 1wk skip, residual-vol
# scaling) is anchored to a WEEKLY decision clock (the internal ISO-week key), so
# it stays honest at every timeframe once ``bars_per_week`` is scaled to the bar
# cadence: 7 bars/week (1d) -> 42 (4h) -> 168 (1h).  The bar-denominated estimation
# windows (beta / adf / residual-vol / min-history) scale x6 (4h) / x24 (1h) to
# hold their wall-clock span; the week-denominated formation / skip, the weekly
# ``min_hold_decisions`` / ``cooldown_decisions`` clock (the proven C1 min-hold
# rescue that lifts RPT above the 10bps floor), the quantile, and the ADF
# z-thresholds are timeframe-invariant and stay fixed.
_RESIDUAL_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "bhm_8wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 540,
            "formation_weeks": 8,
            "skip_weeks": 1,
            "bars_per_week": 42,
            "adf_window_bars": 540,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 336,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 720,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
        {
            "variant": "bhm_12wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 540,
            "formation_weeks": 12,
            "skip_weeks": 1,
            "bars_per_week": 42,
            "adf_window_bars": 540,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 336,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 900,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
    ),
    "1h": (
        {
            "variant": "bhm_8wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 2160,
            "formation_weeks": 8,
            "skip_weeks": 1,
            "bars_per_week": 168,
            "adf_window_bars": 2160,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 1344,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 2880,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
        {
            "variant": "bhm_12wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 2160,
            "formation_weeks": 12,
            "skip_weeks": 1,
            "bars_per_week": 168,
            "adf_window_bars": 2160,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 1344,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 3600,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
    ),
    "1d": (
        {
            "variant": "bhm_8wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 90,
            "formation_weeks": 8,
            "skip_weeks": 1,
            "bars_per_week": 7,
            "adf_window_bars": 90,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 56,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 120,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
        {
            "variant": "bhm_12wk_skip1_gated",
            "benchmark_symbol": "BTC/USDT",
            "beta_window_bars": 90,
            "formation_weeks": 12,
            "skip_weeks": 1,
            "bars_per_week": 7,
            "adf_window_bars": 90,
            "adf_nonstationarity_min_t": -1.5,
            "quantile_pct": 0.25,
            "min_symbols": 5,
            "residual_vol_window_bars": 56,
            "min_hold_decisions": 4,
            "cooldown_decisions": 1,
            "min_history_bars": 150,
            "allow_short": True,
            "hedge_benchmark": False,
            "max_gross": 1.0,
        },
    ),
}

__all__ = ["TrendGatedResidualMomentumStrategy"]
