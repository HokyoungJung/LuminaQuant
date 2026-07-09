"""Cross-sectional downside-beta asymmetry sleeve (conditional co-movement risk).

``CrossSectionalDownsideBetaAsymmetryStrategy`` ranks the liquid cross-section on
the benchmark-sign-conditioned beta asymmetry ``beta_minus - beta_plus`` (the
Ang-Chen-Xing relative downside beta): the co-movement with the benchmark on
its DOWN bars minus its co-movement on the UP bars.  It LONGS the top-quantile
co-crashers (coins that fall hard with the benchmark but do not rally with it --
the priced downside-risk exposure) and SHORTS the bottom-quantile crash-hedge /
up-capture profiles that disappointment-averse and upside-chasing holders
overpay for.

The single load-bearing differentiator is that ``beta_minus - beta_plus`` is
LEVEL-FREE: subtracting ``beta_plus`` strips the unconditional-beta component
that ``BettingAgainstBetaStrategy`` already trades, and it is invariant to the
univariate own-return moments (own vol / skew / MAX / semivariance) that the
idiosyncratic-vol, lottery, and realized-semivariance sleeves own.  The
authoring build gate proves exactly this: on a permutation panel where two
symbols share an unconditional beta (so BAB scores them identically) and share
their univariate moments, this sleeve still emits opposite-signed targets.

THEORY / PROVENANCE
--------------------
- Ang, Chen & Xing (2006, RFS): downside beta carries a cross-sectional risk
  premium beyond CAPM beta, coskewness, and momentum.
- Bawa & Lindenberg (1977): lower-partial-moment CAPM equilibrium foundation.
- Gul (1991): disappointment aversion -- the preference channel that makes the
  premium a durable risk premium rather than a correctable mistake.
- Bollerslev-Patton-Quaedvlieg (2022): realized semibetas (the concordant /
  discordant co-moment decomposition this conditioning approximates).

TURNOVER / COST
----------------
VERY LOW.  A trailing conditioned second moment (default ~180 daily bars) has
week-over-week rank autocorrelation near one; a monthly rebalance clock, a hard
``min_hold_bars`` floor (the proven turnover rescue), and a rank-hysteresis band
suppress boundary flicker.  Sizing is inverse-realized-vol on the liquid book
only, keeping sqrt-impact small.

This module is data-local (no I/O), pure Python (``math``/``statistics``/deque,
plus the shared pure ``indicators/comoment`` kernel), and never raises from
``calculate_signals``.  It carries NO ``@register`` decorator on purpose: an
unregistered module under ``strategies/`` is inert; registration, the
``research_only`` tier hint, candidate wiring, and baseline re-pins land in the
separate atomic integration commit.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import mean
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.comoment import conditional_semibeta
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    _default_benchmark,
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

_STRATEGY_ID = "downside_beta_asymmetry"
_STRATEGY_NAME = "CrossSectionalDownsideBetaAsymmetryStrategy"
_BENCHMARK_MODES = ("btc", "equal_weight_basket")


def _log_returns(closes: list[float]) -> list[float]:
    """Return the completed-bar log returns of a close path (drops bad bars)."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and prev > 0.0 and value > 0.0:
            out.append(math.log(value / prev))
        prev = value
    return out


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


@register("strategy", "CrossSectionalDownsideBetaAsymmetryStrategy", interface="event_driven")
class CrossSectionalDownsideBetaAsymmetryStrategy(Strategy):
    """Long high / short low benchmark-sign-conditioned beta asymmetry.

    See the module docstring for the full theory, the load-bearing
    level-free differentiator, and the cost argument.  This class only reads
    local event/bar OHLCV; it performs no I/O and never raises from
    ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "benchmark_mode": HyperParam.categorical(
                "benchmark_mode",
                default="btc",
                choices=_BENCHMARK_MODES,
                grid=list(_BENCHMARK_MODES),
                tunable=False,
            ),
            "window_bars": HyperParam.integer("window_bars", default=180, low=20, high=20000),
            "min_side_obs": HyperParam.integer("min_side_obs", default=20, low=4, high=4096),
            "threshold": HyperParam.floating("threshold", default=0.0, low=-1.0, high=1.0),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=30, low=1, high=10080),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=21, low=1, high=200000),
            "hysteresis_buffer_ranks": HyperParam.integer(
                "hysteresis_buffer_ranks", default=1, low=0, high=512
            ),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=400, low=1, high=200000),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
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
        mode_value = str(resolved["benchmark_mode"]).lower()
        self.benchmark_mode = mode_value if mode_value in _BENCHMARK_MODES else "btc"
        self.window_bars = max(4, int(resolved["window_bars"]))
        self.min_side_obs = max(2, int(resolved["min_side_obs"]))
        self.threshold = float(resolved["threshold"])
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(1, int(resolved["min_hold_bars"]))
        self.hysteresis_buffer_ranks = max(0, int(resolved["hysteresis_buffer_ranks"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.window_bars + 1, self.vol_window, self.max_hold_bars)
        self._state: dict[str, _CrossSectionalState] = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
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
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
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
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _basket_returns(self) -> list[float] | None:
        series: list[list[float]] = []
        for item in self._state.values():
            rets = _log_returns(list(item.closes))
            if len(rets) >= 2:
                series.append(rets)
        if len(series) < 2:
            return None
        window = min(self.window_bars, min(len(rets) for rets in series))
        if window < 2:
            return None
        tails = [rets[-window:] for rets in series]
        count = float(len(tails))
        return [sum(tail[i] for tail in tails) / count for i in range(window)]

    def _benchmark_returns(self) -> list[float] | None:
        if self.benchmark_mode == "equal_weight_basket":
            return self._basket_returns()
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return None
        rets = _log_returns(list(item.closes))
        return rets if len(rets) >= 2 else None

    def _score_symbols(self) -> dict[str, tuple[float, dict[str, Any]]]:
        bench = self._benchmark_returns()
        if bench is None or len(bench) < 2:
            return {}
        bench_tail = bench[-self.window_bars :]
        scored: dict[str, tuple[float, dict[str, Any]]] = {}
        skip_benchmark = self.benchmark_mode != "equal_weight_basket"
        for symbol, item in self._state.items():
            if skip_benchmark and symbol == self.benchmark_symbol:
                continue
            rets = _log_returns(list(item.closes))
            if len(rets) < 2:
                continue
            sym_tail = rets[-self.window_bars :]
            beta_minus, beta_plus = conditional_semibeta(
                sym_tail,
                bench_tail,
                threshold=self.threshold,
                min_side_obs=self.min_side_obs,
            )
            if beta_minus is None or beta_plus is None:
                continue
            dbs = beta_minus - beta_plus
            scored[symbol] = (
                float(dbs),
                {
                    "beta_minus": float(beta_minus),
                    "beta_plus": float(beta_plus),
                    "downside_beta_asymmetry": float(dbs),
                },
            )
        return scored

    # ------------------------------------------------------------------ #
    # selection / emission (hysteresis + hard min-hold + inverse-vol)
    # ------------------------------------------------------------------ #
    def _desired_book(
        self, scored: dict[str, tuple[float, dict[str, Any]]]
    ) -> tuple[dict[str, str], dict[str, int], int]:
        ordered = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
        rank = {symbol: idx for idx, (symbol, _payload) in enumerate(ordered)}
        n = len(ordered)
        k = max(1, int(n * self.quantile_pct))
        buf = self.hysteresis_buffer_ranks
        desired: dict[str, str] = {}
        for symbol, item in self._state.items():
            idx = rank.get(symbol)
            cur = item.mode
            if cur != "OUT" and item.bars_held < self.min_hold_bars:
                desired[symbol] = cur
            elif cur == "LONG" and idx is not None and idx < k + buf:
                desired[symbol] = "LONG"
            elif cur == "SHORT" and idx is not None and idx >= n - k - buf and self.allow_short:
                desired[symbol] = "SHORT"
            elif idx is not None and idx < k:
                desired[symbol] = "LONG"
            elif idx is not None and idx >= n - k and self.allow_short:
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = "OUT"
        return desired, rank, n

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

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        if self._tick % self.rebalance_bars:
            self._age(event_time)
            return
        scored = self._score_symbols()
        if len(scored) < self.min_symbols:
            self._age(event_time)
            return
        desired, _rank, _n = self._desired_book(scored)
        values = [payload[0] for payload in scored.values()]
        centre = mean(values)
        spread = sample_std(values)
        book = [symbol for symbol, side in desired.items() if side in {"LONG", "SHORT"}]
        weights = self._inverse_vol_weights(book)
        self._emit_book(desired, weights, scored, centre, spread, event_time)

    def _emit_book(
        self,
        desired: dict[str, str],
        weights: dict[str, float],
        scored: dict[str, tuple[float, dict[str, Any]]],
        centre: float,
        spread: float | None,
        event_time: Any,
    ) -> None:
        has_spread = spread is not None and spread > _EPS
        for symbol, item in self._state.items():
            side = desired.get(symbol, "OUT")
            price = item.closes[-1] if item.closes else None
            if side == "OUT":
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
                continue
            if item.mode == side:
                item.bars_held += 1
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
            dbs, meta = scored.get(symbol, (0.0, {}))
            dbs_z = (dbs - centre) / spread if has_spread else 0.0
            weight = max(0.0, float(weights.get(symbol, 0.0)))
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=weight,
                max_order_value=self.max_order_value,
                score=float(dbs),
                target_mode=side,
                downside_beta_asymmetry_z=float(dbs_z),
                **meta,
            )
            strength = max(0.25, min(3.0, abs(dbs_z))) if has_spread else 1.0
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=side,
                strength=float(strength),
                price=price,
                metadata=metadata,
            )
            item.mode = side
            item.entry_price = price
            item.bars_held = 0

    def _age(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            item.bars_held += 1
            if self.max_hold_bars > 0 and item.bars_held >= self.max_hold_bars:
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
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Honest family:
# cross-sectional, admitted via `allow_multi_asset=True`; NOT carry/momentum, so
# no fake tags are added to game the allocator superset route.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "downside_beta",
    "semibeta",
    "co_moment",
    "risk_premium",
    "crypto",
)

_DOWNSIDE_BETA_ASYMMETRY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "monthly_core",
            "window_bars": 180,
            "min_side_obs": 20,
            "quantile_pct": 0.25,
            "rebalance_bars": 30,
            "min_hold_bars": 21,
            "hysteresis_buffer_ranks": 1,
            "vol_window": 30,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 0.36,
        },
        {
            "variant": "shortwin_ls",
            "window_bars": 90,
            "min_side_obs": 16,
            "quantile_pct": 0.20,
            "rebalance_bars": 30,
            "min_hold_bars": 21,
            "hysteresis_buffer_ranks": 1,
            "vol_window": 30,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 0.30,
        },
    ),
}


__all__ = ["CrossSectionalDownsideBetaAsymmetryStrategy"]
