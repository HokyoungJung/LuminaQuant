"""Stationarity-gated, market-neutral idiosyncratic residual reversion (N4).

``StationarityGatedResidualReversionStrategy`` mean-reverts the IDIOSYNCRATIC
residual of each asset versus a benchmark -- it fades the beta-hedged
(market-neutral) component of a move, NOT the raw price level.  Its single
load-bearing differentiator from the registered incumbent
``EquityBenchmarkResidualReversalStrategy``
(``cross_asset_tradfi_alpha_sleeves.py``) is an augmented Dickey-Fuller (ADF)
STATIONARITY GATE: it only fades a residual whose trailing residual-level series
clears an ADF reject threshold (statistical evidence that the hedged spread
actually mean-reverts).  A residual that looks dislocated but whose spread is a
random walk / non-stationary is left ALONE (abstain) -- exactly the reading the
incumbent, which has no such gate, would trade.

THEORY / PROVENANCE
-------------------
- Beta-hedged residual (idiosyncratic) reversal is the repo's next-direction
  residual-reversion note (sd:76); N4 is its deferred build (rn:96).
- Nagel (2012, RFS) frames short-horizon reversal as compensated
  liquidity-provision -- but on liquid crypto majors naive price-level reversal
  is wrong-signed (external verdict 9), so N4 fades only the beta-NEUTRAL
  residual and requires stationarity before doing so.
- The ADF screen is the pure-Python ``adf_t_statistic``
  (``indicators/stationarity.py``; no numpy/scipy/statsmodels), so the sleeve
  can demand quantitative mean-reversion evidence on the TRAILING window with no
  full-sample lookahead.

MECHANISM (per completed decision bar; OHLCV only)
--------------------------------------------------
Per asset, on a slow (weekly-ish) ``rebalance_bars`` clock:

1. ``beta = rolling_beta(asset_returns, benchmark_returns)`` over
   ``beta_window`` log-return samples (``indicators/rolling_stats.py``) -- the
   hedge ratio.
2. Residual reversion score (identical core to the incumbent):
   ``residual = raw_ret - beta * benchmark_ret`` over ``lookback_bars`` and
   ``score = -residual / realized_vol`` -- a large positive score means the
   name has under-performed its beta-implied path and is a mean-reversion LONG;
   a large negative score is a SHORT.
3. STATIONARITY GATE (the sole differentiator): build the trailing
   residual-LEVEL series ``log(close_asset) - beta * log(close_bench)`` over
   ``adf_window`` bars and require
   ``adf_t_statistic(...) <= adf_reject_threshold`` (more negative = stronger
   stationarity).  A residual whose level series is non-stationary (ADF fails to
   reject) is ABSTAINED -- no trade -- even when the dislocation is large.
4. Liquid-book restriction: a name whose trailing median dollar volume is below
   ``min_dollar_volume`` is skipped (reversion churns, so it stays in the liquid
   book where cost survivability is plausible).

Surviving names are ranked long/short via the shared ``_ranked_targets`` (top
``max_longs`` / ``max_shorts`` past ``entry_score``), sized equal-gross, and
carry a ``min_hold_bars`` lock (a rank-driven flip/exit inside the min-hold
window is suppressed; hard stop-loss / ``max_hold_bars`` still fire).  Because
the book is idiosyncratic-residual, an offsetting BENCHMARK hedge leg is emitted
so the net beta exposure of the book is ~0 by construction (market-neutral).

DISTINCT-FROM
-------------
Versus ``EquityBenchmarkResidualReversalStrategy`` the ONLY mechanism change is
the ADF stationarity gate (plus the explicit market-neutral hedge leg,
min-hold, and liquid-book filter which do not alter the trade/abstain decision
on a stationary, liquid residual): on a stationary residual BOTH fade it; on a
non-stationary residual the incumbent trades and N4 abstains.  This is NOT
``DispersionConditionedReversion`` (a dispersion-regime gate, not stationarity)
and NOT price-level reversal (the residual is beta-hedged).

This module is data-local (no I/O, no hidden config bus), pure Python
(``math`` + ``deque`` + the pure indicator primitives, no numpy), never raises
from ``calculate_signals``, and ships WITHOUT ``@register`` (inert until the W3
integration wave wires it as ``research_only``).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import log_return, realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import rolling_beta
from lumina_quant.indicators.stationarity import adf_t_statistic
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _ranked_targets,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "stationarity_gated_residual_reversion"
_STRATEGY_NAME = "StationarityGatedResidualReversionStrategy"


def _parse_symbol_list(raw: str, universe: list[str]) -> list[str]:
    """Return the configured symbols kept in universe order (empty -> ``[]``)."""
    wanted = {token.strip() for token in str(raw).split(",") if token.strip()}
    if not wanted:
        return []
    return [symbol for symbol in universe if symbol in wanted]


def _pack_cross(item: _CrossSectionalState) -> dict[str, Any]:
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "mode": item.mode,
        "entry_price": item.entry_price,
        "bars_held": int(item.bars_held),
        "last_time_key": item.last_time_key,
    }


def _restore_cross(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    item.mode = _mode(payload.get("mode"))
    item.entry_price = safe_float(payload.get("entry_price"))
    item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    item.last_time_key = str(payload.get("last_time_key", ""))


def _median(values: list[float]) -> float | None:
    """Return the median of a list of finite floats, or ``None`` if empty."""
    cleaned = [float(value) for value in values if math.isfinite(value)]
    if not cleaned:
        return None
    ordered = sorted(cleaned)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _log_levels(values: list[float]) -> list[float] | None:
    """Return log of each strictly-positive close, or ``None`` if any is invalid."""
    out: list[float] = []
    for value in values:
        parsed = safe_float(value)
        if parsed is None or parsed <= 0.0:
            return None
        out.append(math.log(parsed))
    return out


def _return_series(values: list[float]) -> list[float]:
    """Return the log-return series of a strictly-positive level series."""
    out: list[float] = []
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        cur = values[idx]
        if prev > 0.0 and cur > 0.0:
            out.append(math.log(cur / prev))
    return out


@dataclass(slots=True)
class _AssetScore:
    score: float
    beta: float
    residual: float
    adf_t: float
    realized_vol: float


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class StationarityGatedResidualReversionStrategy(Strategy):
    """ADF-gated, market-neutral idiosyncratic residual reversion sleeve.

    See the module docstring for the full theory, mechanism, and the
    single-differentiator (ADF stationarity gate) rationale versus
    ``EquityBenchmarkResidualReversalStrategy``.  Reads only local event/bar
    OHLCV; performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "equity_symbols": HyperParam.string("equity_symbols", default="", tunable=False),
            "benchmark_symbol": HyperParam.string("benchmark_symbol", default="", tunable=False),
            "lookback_bars": HyperParam.integer("lookback_bars", default=5, low=1, high=4096),
            "beta_window": HyperParam.integer("beta_window", default=60, low=4, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=4096),
            "adf_window": HyperParam.integer("adf_window", default=120, low=9, high=20000),
            "adf_lags": HyperParam.integer("adf_lags", default=0, low=0, high=32),
            "adf_reject_threshold": HyperParam.floating(
                "adf_reject_threshold", default=-2.86, low=-10.0, high=0.0
            ),
            "entry_score": HyperParam.floating("entry_score", default=0.45, low=0.0, high=20.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=12, low=1, high=10080),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=12, low=0, high=100000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=100000),
            "max_longs": HyperParam.integer("max_longs", default=4, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=4, low=0, high=128),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "hedge_benchmark": HyperParam.boolean(
                "hedge_benchmark", default=True, grid=[True, False]
            ),
            "min_dollar_volume": HyperParam.floating(
                "min_dollar_volume", default=0.0, low=0.0, high=1e15
            ),
            "dollar_volume_window": HyperParam.integer(
                "dollar_volume_window", default=20, low=1, high=4096
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.05, low=0.0, high=0.50),
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
        benchmark = str(resolved["benchmark_symbol"] or "")
        if benchmark not in self.symbol_list:
            benchmark = self.symbol_list[0] if self.symbol_list else ""
        self.benchmark_symbol = benchmark
        self.equity_symbols = _parse_symbol_list(
            str(resolved["equity_symbols"]), self.symbol_list
        ) or [symbol for symbol in self.symbol_list if symbol != self.benchmark_symbol]
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.beta_window = max(4, int(resolved["beta_window"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.adf_window = max(9, int(resolved["adf_window"]))
        self.adf_lags = max(0, int(resolved["adf_lags"]))
        self.adf_reject_threshold = float(resolved["adf_reject_threshold"])
        self.entry_score = max(0.0, float(resolved["entry_score"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.hedge_benchmark = bool(resolved["hedge_benchmark"])
        self.min_dollar_volume = max(0.0, float(resolved["min_dollar_volume"]))
        self.dollar_volume_window = max(1, int(resolved["dollar_volume_window"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.lookback_bars,
            self.beta_window,
            self.vol_window,
            self.adf_window,
            self.dollar_volume_window,
            self.max_hold_bars,
        )
        self._state: dict[str, _CrossSectionalState] = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        # Signed beta exposure of each LIVE asset leg (used to size the
        # offsetting benchmark hedge so the book's net beta stays ~0).
        self._exposure: dict[str, float] = {}
        self._tick = 0
        self._last_eval_time_key = ""

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "exposure": {symbol: float(value) for symbol, value in self._exposure.items()},
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    try:
                        _restore_cross(self._state[symbol], payload)
                    except Exception:
                        continue
        self._exposure = {}
        exposure = state.get("exposure")
        if isinstance(exposure, dict):
            for symbol, value in exposure.items():
                parsed = safe_float(value)
                if symbol in self._state and parsed is not None:
                    self._exposure[symbol] = float(parsed)
        # Only keep exposure for names actually held (defensive vs adversarial state).
        self._exposure = {
            symbol: value
            for symbol, value in self._exposure.items()
            if self._state[symbol].mode != "OUT"
        }

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _update(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _liquid(self, item: _CrossSectionalState) -> bool:
        if self.min_dollar_volume <= 0.0:
            return True
        closes = list(item.closes)[-self.dollar_volume_window :]
        volumes = list(item.volumes)[-self.dollar_volume_window :]
        if not closes or not volumes:
            return False
        dollar_volumes = [
            close * volume for close, volume in zip(closes, volumes, strict=False) if close > 0.0
        ]
        median_dv = _median(dollar_volumes)
        return median_dv is not None and median_dv >= self.min_dollar_volume

    def _asset_score(
        self, item: _CrossSectionalState, bench: _CrossSectionalState, bench_ret: float
    ) -> _AssetScore | None:
        """Return the ADF-gated residual reversion score, or ``None`` (abstain)."""
        closes = list(item.closes)
        bench_closes = list(bench.closes)
        raw_ret = log_return(closes, lookback=self.lookback_bars)
        vol = realized_volatility(closes, window=self.vol_window)
        if raw_ret is None or vol is None or vol <= _EPS:
            return None

        # Hedge ratio from aligned log-return samples (pure rolling_beta).
        depth = min(len(closes), len(bench_closes), self.beta_window + 1)
        if depth < 4:
            return None
        asset_returns = _return_series(closes[-depth:])
        bench_returns = _return_series(bench_closes[-depth:])
        aligned = min(len(asset_returns), len(bench_returns))
        if aligned < 2:
            return None
        beta = rolling_beta(asset_returns[-aligned:], bench_returns[-aligned:])
        if beta is None:
            return None

        residual = raw_ret - beta * bench_ret
        score = -residual / max(vol, _EPS)
        if not math.isfinite(score):
            return None

        # STATIONARITY GATE: the beta-hedged residual LEVEL series must clear the
        # ADF reject threshold before the reversion is faded (the sole
        # mechanism differentiator versus the incumbent).
        window = min(len(closes), len(bench_closes), self.adf_window)
        if window < 9:
            return None
        asset_levels = _log_levels(closes[-window:])
        bench_levels = _log_levels(bench_closes[-window:])
        if asset_levels is None or bench_levels is None:
            return None
        residual_levels = [asset_levels[idx] - beta * bench_levels[idx] for idx in range(window)]
        adf_t = adf_t_statistic(residual_levels, lags=self.adf_lags)
        if adf_t is None or adf_t > self.adf_reject_threshold:
            # Non-stationary residual -> abstain (the incumbent would trade here).
            return None
        return _AssetScore(
            score=float(score),
            beta=float(beta),
            residual=float(residual),
            adf_t=float(adf_t),
            realized_vol=float(vol),
        )

    def _score_rows(
        self,
    ) -> tuple[list[tuple[float, str, dict[str, Any]]], dict[str, _AssetScore]]:
        bench = self._state.get(self.benchmark_symbol)
        if bench is None:
            return [], {}
        bench_ret = log_return(list(bench.closes), lookback=self.lookback_bars)
        if bench_ret is None:
            return [], {}
        rows: list[tuple[float, str, dict[str, Any]]] = []
        scores: dict[str, _AssetScore] = {}
        for symbol in self.equity_symbols:
            item = self._state.get(symbol)
            if item is None or not self._liquid(item):
                continue
            result = self._asset_score(item, bench, bench_ret)
            if result is None:
                continue
            scores[symbol] = result
            rows.append(
                (
                    result.score,
                    symbol,
                    {
                        "residual_return": result.residual,
                        "benchmark_return": float(bench_ret),
                        "beta": result.beta,
                        "adf_t_statistic": result.adf_t,
                        "adf_reject_threshold": self.adf_reject_threshold,
                        "realized_vol": result.realized_vol,
                    },
                )
            )
        return rows, scores

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        _age_cross_positions(
            self.events,
            self._state,
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
        )
        # Aging may have closed positions (stop / max-hold): drop their exposure.
        self._exposure = {
            symbol: value
            for symbol, value in self._exposure.items()
            if self._state[symbol].mode != "OUT"
        }

    def _rebalance(self, event_time: Any) -> None:
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        rows, scores = self._score_rows()
        targets = _ranked_targets(
            rows,
            threshold=self.entry_score,
            max_longs=self.max_longs,
            max_shorts=self.max_shorts,
            allow_short=self.allow_short,
        )
        targets = self._apply_min_hold(targets)
        self._emit_asset_targets(targets, scores, event_time)
        self._emit_benchmark_hedge(event_time)

    def _apply_min_hold(
        self, targets: dict[str, tuple[str, float, dict[str, Any]]]
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:
        """Suppress rank-driven flips/exits of a position younger than min-hold."""
        if self.min_hold_bars <= 0:
            return targets
        for symbol in self.equity_symbols:
            item = self._state.get(symbol)
            if item is None or item.mode == "OUT":
                continue
            if item.bars_held >= self.min_hold_bars:
                continue
            target = targets.get(symbol)
            if target is None or target[0] != item.mode:
                # Lock the existing side: treat as a "continue" this rebalance.
                targets[symbol] = (item.mode, 0.0, {"reason": "min_hold_lock"})
        return targets

    def _emit_asset_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        scores: dict[str, _AssetScore],
        event_time: Any,
    ) -> None:
        selected = sum(1 for value in targets.values() if value[0] in {"LONG", "SHORT"})
        base_alloc = self.target_gross_exposure / float(selected) if selected else 0.0
        for symbol in self.equity_symbols:
            item = self._state.get(symbol)
            if item is None:
                continue
            price = item.closes[-1] if item.closes else None
            target = targets.get(symbol)
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rebalance_removed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    self._exposure.pop(symbol, None)
                continue
            target_mode, score, _meta = target
            if item.mode == target_mode:
                # Held or min-hold-locked: keep the position and its exposure.
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
                self._exposure.pop(symbol, None)
            detail = scores.get(symbol)
            beta = detail.beta if detail is not None else 0.0
            signed = 1.0 if target_mode == "LONG" else -1.0
            beta_exposure = signed * base_alloc * float(beta)
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=max(0.0, base_alloc),
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                beta=float(beta),
                beta_exposure=float(beta_exposure),
                residual_return=detail.residual if detail is not None else None,
                adf_t_statistic=detail.adf_t if detail is not None else None,
            )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score) / max(self.entry_score, _EPS))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            self._exposure[symbol] = float(beta_exposure)

    def _emit_benchmark_hedge(self, event_time: Any) -> None:
        """Re-emit the benchmark leg so the book's net beta exposure is ~0."""
        bench = self._state.get(self.benchmark_symbol)
        if bench is None:
            return
        price = bench.closes[-1] if bench.closes else None
        net = math.fsum(self._exposure.values())
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
                bench.bars_held = 0
            return
        # Offset a net-positive book beta by SHORTING the benchmark (and vice versa).
        hedge_mode = "SHORT" if net > 0.0 else "LONG"
        hedge_notional = abs(net)
        hedge_exposure = -net  # benchmark self-beta is 1 by definition.
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
            target_allocation=max(0.0, hedge_notional),
            max_order_value=self.max_order_value,
            target_mode=hedge_mode,
            beta=1.0,
            beta_exposure=float(hedge_exposure),
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
        bench.bars_held = 0


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a cross-sectional market-neutral residual sleeve but NOT carry,
# so it is honestly EXCLUDED from any carry-tag allowlist route -- no fake carry
# tag is added.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "market_neutral",
    "residual_reversion",
    "stationarity_gated",
    "mean_reversion",
    "crypto",
)

_RESIDUAL_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "strict_adf",
            "lookback_bars": 6,
            "beta_window": 90,
            "vol_window": 24,
            "adf_window": 120,
            "adf_reject_threshold": -2.86,
            "entry_score": 0.50,
            "rebalance_bars": 42,
            "min_hold_bars": 42,
            "max_hold_bars": 252,
            "max_longs": 3,
            "max_shorts": 3,
            "allow_short": True,
        },
    ),
    "1d": (
        {
            "variant": "strict_adf",
            "lookback_bars": 5,
            "beta_window": 60,
            "vol_window": 20,
            "adf_window": 120,
            "adf_reject_threshold": -2.86,
            "entry_score": 0.45,
            "rebalance_bars": 7,
            "min_hold_bars": 7,
            "max_hold_bars": 42,
            "max_longs": 4,
            "max_shorts": 4,
            "allow_short": True,
        },
        {
            "variant": "lenient_adf",
            "lookback_bars": 5,
            "beta_window": 60,
            "vol_window": 20,
            "adf_window": 90,
            "adf_reject_threshold": -2.57,
            "entry_score": 0.45,
            "rebalance_bars": 7,
            "min_hold_bars": 7,
            "max_hold_bars": 42,
            "max_longs": 4,
            "max_shorts": 4,
            "allow_short": True,
        },
    ),
    # 1h mirrors the hand-tuned 4h cell: the ADF/beta/vol ESTIMATOR windows are
    # sample-count driven (a unit-root test / beta regression needs the same
    # number of observations regardless of bar size), so they carry the 4h
    # values verbatim; only the bar-denominated decision cadence and hold
    # horizons scale x24 from the 1d cell (rebalance/min_hold 7->168,
    # max_hold 42->1008) so the weekly-ish rebalance and multi-week hold stay
    # fixed in wall-clock terms.
    "1h": (
        {
            "variant": "strict_adf",
            "lookback_bars": 6,
            "beta_window": 90,
            "vol_window": 24,
            "adf_window": 120,
            "adf_reject_threshold": -2.86,
            "entry_score": 0.50,
            "rebalance_bars": 168,
            "min_hold_bars": 168,
            "max_hold_bars": 1008,
            "max_longs": 3,
            "max_shorts": 3,
            "allow_short": True,
        },
    ),
}


__all__ = ["StationarityGatedResidualReversionStrategy"]
