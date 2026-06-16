"""Selection-aware cross-sectional alpha sleeves.

These sleeves *compose* the multi-factor target-pool selector
(:class:`lumina_quant.strategy_factory.universe_selection.TargetPoolSelector`)
with a directional cross-sectional signal.  On each banded rebalance the sleeve
screens its full symbol universe into a tradeable target pool
(price-position / volume / volatility / VWAP composite) and then trades *only*
within that screened pool.  Two decorrelated sleeves are provided:

- :class:`SelectionGatedMomentumStrategy`: vol-scaled cross-sectional momentum
  on the screened pool, optionally tilted by the selection composite.  The
  thesis is that momentum measured on a liquidity / volatility / position
  pre-screened universe is cleaner and decorrelated from raw cross-sectional
  momentum that includes illiquid or range-extreme names.
- :class:`SelectionGatedReversionStrategy`: short-horizon mean-reversion (fade
  the most recent return) on the same screened mid-range liquid pool.  It is
  intentionally directionally opposite to the momentum sleeve so the pair is
  decorrelated.

Both sleeves are data-local: they ingest only the local event/bar contract,
maintain bounded per-symbol deques, emit through the shared
``_target_metadata`` / ``_emit`` helpers, and never raise on degenerate input
(short history, ``None`` prices, or a too-small screened pool all result in a
graceful skip).

CRITICAL import-cycle note: the strategy-factory ``universe_selection`` module
is imported *lazily* inside the rebalance method, never at module top level,
because ``candidate_library`` top-level-imports a strategies module and a
top-level ``strategy_factory`` import here would risk an import cycle during
plugin-registry discovery.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _ranked_targets,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

if TYPE_CHECKING:
    from lumina_quant.strategy_factory.universe_selection import TargetPoolResult


def _state_size(*values: int) -> int:
    """Return a bounded deque length covering the largest configured window."""
    return max(8, max(int(value) for value in values) + 8)


def _pack_cross(item: _SelectionAwareState) -> dict[str, Any]:
    """Serialize a :class:`_SelectionAwareState` into a JSON-friendly dict."""
    cross = item.cross
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "highs": list(item.highs),
        "lows": list(item.lows),
        "mode": cross.mode,
        "entry_price": cross.entry_price,
        "bars_held": int(cross.bars_held),
        "last_time_key": cross.last_time_key,
    }


def _restore_cross(item: _SelectionAwareState, payload: dict[str, Any]) -> None:
    """Restore a :class:`_SelectionAwareState` from a serialized payload."""
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    _restore_deque(item.highs, payload.get("highs"))
    _restore_deque(item.lows, payload.get("lows"))
    cross = item.cross
    cross.mode = _mode(payload.get("mode"))
    cross.entry_price = safe_float(payload.get("entry_price"))
    cross.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    cross.last_time_key = str(payload.get("last_time_key", ""))


class _SelectionAwareState:
    """Per-symbol OHLCV deques plus a cross-sectional position state.

    The momentum / reversion math only needs ``closes`` and ``volumes`` (which
    :class:`_CrossSectionalState` carries), but the selector consumes full OHLCV
    series, so this state also keeps bounded ``highs`` / ``lows`` deques.  The
    position bookkeeping (``mode`` / ``entry_price`` / ``bars_held``) is reused
    from :class:`_CrossSectionalState` so the shared emit/age helpers work
    unchanged.
    """

    __slots__ = ("closes", "cross", "highs", "lows", "volumes")

    def __init__(self, size: int) -> None:
        """Initialize bounded deques of length ``size`` and an OUT position."""
        self.closes: deque[float] = deque(maxlen=size)
        self.volumes: deque[float] = deque(maxlen=size)
        self.highs: deque[float] = deque(maxlen=size)
        self.lows: deque[float] = deque(maxlen=size)
        # The cross-sectional state shares the *same* close/volume deques so the
        # emit/age helpers read the latest ingested price without duplication.
        self.cross = _CrossSectionalState(self.closes, self.volumes)


class _SelectionAwareMixin:
    """Shared OHLCV ingestion, dispatch, and selector composition.

    Subclasses define ``_score_pool`` to produce ranked rows from the screened
    pool and supply their own ``strategy_id`` / ``strategy_name`` constants.
    The mixin owns the deque updates, the event dispatch, the cross-sectional
    state serialization, and the lazy selector call.
    """

    # Subclasses set these at class scope.
    strategy_id: str
    strategy_name: str
    vwap_sign: float = -1.0

    symbol_list: list[str]
    min_price: float
    rebalance_bars: int
    min_symbols: int
    pool_size: int
    history_window: int
    weight_price_position: float
    weight_volume: float
    weight_volatility: float
    weight_vwap: float
    selection_tilt: float
    stop_loss_pct: float
    max_hold_bars: int
    target_gross_exposure: float
    max_order_value: float
    events: Any

    def _build_states(self, size: int) -> None:
        self._size = int(size)
        self._symbol_states: dict[str, _SelectionAwareState] = {
            symbol: _SelectionAwareState(size) for symbol in self.symbol_list
        }
        # Cross-sectional state view used by the shared emit/age helpers.
        self._state: dict[str, _CrossSectionalState] = {
            symbol: item.cross for symbol, item in self._symbol_states.items()
        }
        self._last_eval_time_key = ""
        self._tick = 0

    # -- state serialization -------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the sleeve state."""
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: _pack_cross(item) for symbol, item in self._symbol_states.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the sleeve state from a serialized snapshot."""
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                item = self._symbol_states.get(symbol)
                if item is not None and isinstance(payload, dict):
                    _restore_cross(item, payload)

    # -- ingestion / dispatch ------------------------------------------------
    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._symbol_states[symbol]
        key = time_key(snapshot.time)
        if key and key == item.cross.last_time_key:
            return False
        item.cross.last_time_key = key
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        item.highs.append(float(high) if high is not None else close)
        item.lows.append(float(low) if low is not None else close)
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        """Ingest a MARKET_WINDOW batch and rebalance on a new evaluation bar."""
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
        """Dispatch MARKET / MARKET_WINDOW events to the per-symbol ingest path."""
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._symbol_states:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    # -- selection -----------------------------------------------------------
    def _select_pool(self) -> TargetPoolResult | None:
        """Screen the universe into a target pool via the lazy-imported selector.

        Returns ``None`` when no symbol carries enough history to build any
        usable series, so the caller can skip without raising.
        """
        # LAZY import: keep ``strategy_factory`` out of module-import scope to
        # avoid an import cycle during plugin-registry discovery.
        from lumina_quant.strategy_factory.universe_selection import (
            TargetPoolSelector,
            UniverseSelectionConfig,
        )

        series: dict[str, dict[str, list[float]]] = {}
        for symbol, item in self._symbol_states.items():
            if len(item.closes) < 2:
                continue
            series[symbol] = {
                "high": list(item.highs),
                "low": list(item.lows),
                "close": list(item.closes),
                "volume": list(item.volumes),
            }
        if not series:
            return None
        config = UniverseSelectionConfig(
            weights={
                "price_position": self.weight_price_position,
                "volume": self.weight_volume,
                "volatility": self.weight_volatility,
                "vwap": self.weight_vwap,
            },
            hard_filters={
                "min_dollar_volume": 0.0,
                # The synthetic / warmup window is short; require only a couple
                # of bars so the gate is data-driven, not history-starved.
                "min_history_bars": 2.0,
                "vol_band_low": 0.0,
                "vol_band_high": 1.0,
            },
            windows={
                "price_position": min(self.history_window, 60),
                "volume": min(self.history_window, 64),
                "volatility": 14,
                "vwap": min(self.history_window, 64),
            },
            pool_size=self.pool_size,
            vwap_sign=self.vwap_sign,
        )
        try:
            return TargetPoolSelector(config).select(series)
        except Exception:
            return None

    @staticmethod
    def _composite_by_symbol(result: TargetPoolResult) -> dict[str, float]:
        """Return the selection composite per symbol from the result breakdowns."""
        return {bd.symbol: float(bd.composite) for bd in result.breakdowns}

    def _score_pool(
        self,
        pool: tuple[str, ...],
        composites: dict[str, float],
    ) -> list[tuple[float, str, dict[str, Any]]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        result = self._select_pool()
        if result is None:
            return
        pool = tuple(result.pool)
        if len(pool) < self.min_symbols:
            # Selected pool too small to trade -> graceful skip (age only).
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        composites = self._composite_by_symbol(result)
        rows = self._score_pool(pool, composites)
        if len(rows) < self.min_symbols:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        targets = self._build_targets(rows)
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, _EPS),
        )

    def _build_targets(
        self,
        rows: list[tuple[float, str, dict[str, Any]]],
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:  # pragma: no cover - overridden
        raise NotImplementedError


@register("strategy", "SelectionGatedMomentumStrategy", interface="event_driven")
class SelectionGatedMomentumStrategy(_SelectionAwareMixin, Strategy):
    """Vol-scaled cross-sectional momentum traded only within the screened pool.

    Each banded rebalance screens the universe into the multi-factor target pool
    and, within that pool, ranks names by ``simple_return`` over
    ``momentum_lookback_bars`` divided by realized volatility, optionally tilting
    the score by the selection composite.  Long the top ``top_fraction``; short
    the symmetric bottom fraction when ``allow_short`` is set.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_id = "selection_gated_momentum"
    strategy_name = "SelectionGatedMomentumStrategy"
    # +1 rewards distance from VWAP (momentum-friendly screening).
    vwap_sign = 1.0

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=48, low=2, high=10080
            ),
            "vol_window": HyperParam.integer("vol_window", default=24, low=2, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "history_window": HyperParam.integer("history_window", default=96, low=8, high=20000),
            "pool_size": HyperParam.integer("pool_size", default=12, low=2, high=512),
            "top_fraction": HyperParam.floating("top_fraction", default=0.34, low=0.05, high=0.50),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.0, low=0.0, high=20.0
            ),
            "selection_tilt": HyperParam.floating(
                "selection_tilt", default=0.25, low=0.0, high=5.0
            ),
            "weight_price_position": HyperParam.floating(
                "weight_price_position", default=1.0, low=0.0, high=10.0, tunable=False
            ),
            "weight_volume": HyperParam.floating(
                "weight_volume", default=1.5, low=0.0, high=10.0, tunable=False
            ),
            "weight_volatility": HyperParam.floating(
                "weight_volatility", default=1.0, low=0.0, high=10.0, tunable=False
            ),
            "weight_vwap": HyperParam.floating(
                "weight_vwap", default=0.8, low=0.0, high=10.0, tunable=False
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=False, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.060, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.history_window = max(8, int(resolved["history_window"]))
        self.pool_size = max(2, int(resolved["pool_size"]))
        self.top_fraction = max(0.01, min(0.50, float(resolved["top_fraction"])))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.selection_tilt = max(0.0, float(resolved["selection_tilt"]))
        self.weight_price_position = max(0.0, float(resolved["weight_price_position"]))
        self.weight_volume = max(0.0, float(resolved["weight_volume"]))
        self.weight_volatility = max(0.0, float(resolved["weight_volatility"]))
        self.weight_vwap = max(0.0, float(resolved["weight_vwap"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.momentum_lookback_bars,
            self.vol_window,
            self.history_window,
            self.max_hold_bars,
        )
        self._build_states(size)

    def _score_pool(
        self,
        pool: tuple[str, ...],
        composites: dict[str, float],
    ) -> list[tuple[float, str, dict[str, Any]]]:
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in pool:
            item = self._symbol_states.get(symbol)
            if item is None:
                continue
            mom = simple_return(item.closes, lookback=self.momentum_lookback_bars)
            vol = realized_volatility(item.closes, window=self.vol_window)
            if mom is None or vol is None or vol <= _EPS:
                continue
            base_score = mom / max(vol, _EPS)
            composite = composites.get(symbol, 0.0)
            score = base_score * (1.0 + self.selection_tilt * composite)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "momentum": mom,
                        "realized_vol": vol,
                        "selection_composite": composite,
                        "screened": True,
                    },
                )
            )
        return rows

    def _build_targets(
        self,
        rows: list[tuple[float, str, dict[str, Any]]],
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:
        max_side = max(1, int(len(rows) * self.top_fraction))
        return _ranked_targets(
            rows,
            threshold=self.signal_threshold,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )


@register("strategy", "SelectionGatedReversionStrategy", interface="event_driven")
class SelectionGatedReversionStrategy(_SelectionAwareMixin, Strategy):
    """Short-horizon mean-reversion traded only within the screened pool.

    On each banded rebalance the universe is screened into the multi-factor
    target pool, then within that pool names are ranked to *fade* their recent
    ``reversion_lookback_bars`` return: long the biggest losers, short the
    biggest winners (when ``allow_short`` is set).  Because it acts only on
    mid-range liquid screened names and trades opposite to recent returns, it is
    decorrelated from :class:`SelectionGatedMomentumStrategy`.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_id = "selection_gated_reversion"
    strategy_name = "SelectionGatedReversionStrategy"
    # -1 rewards proximity to VWAP (mean-reversion-friendly screening).
    vwap_sign = -1.0

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "reversion_lookback_bars": HyperParam.integer(
                "reversion_lookback_bars", default=6, low=1, high=4096
            ),
            "vol_window": HyperParam.integer("vol_window", default=24, low=2, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=12, low=1, high=10080),
            "history_window": HyperParam.integer("history_window", default=96, low=8, high=20000),
            "pool_size": HyperParam.integer("pool_size", default=12, low=2, high=512),
            "top_fraction": HyperParam.floating("top_fraction", default=0.34, low=0.05, high=0.50),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.0, low=0.0, high=20.0
            ),
            "selection_tilt": HyperParam.floating(
                "selection_tilt", default=0.25, low=0.0, high=5.0
            ),
            "weight_price_position": HyperParam.floating(
                "weight_price_position", default=1.0, low=0.0, high=10.0, tunable=False
            ),
            "weight_volume": HyperParam.floating(
                "weight_volume", default=1.5, low=0.0, high=10.0, tunable=False
            ),
            "weight_volatility": HyperParam.floating(
                "weight_volatility", default=1.0, low=0.0, high=10.0, tunable=False
            ),
            "weight_vwap": HyperParam.floating(
                "weight_vwap", default=0.8, low=0.0, high=10.0, tunable=False
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=False, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.040, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.reversion_lookback_bars = max(1, int(resolved["reversion_lookback_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.history_window = max(8, int(resolved["history_window"]))
        self.pool_size = max(2, int(resolved["pool_size"]))
        self.top_fraction = max(0.01, min(0.50, float(resolved["top_fraction"])))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.selection_tilt = max(0.0, float(resolved["selection_tilt"]))
        self.weight_price_position = max(0.0, float(resolved["weight_price_position"]))
        self.weight_volume = max(0.0, float(resolved["weight_volume"]))
        self.weight_volatility = max(0.0, float(resolved["weight_volatility"]))
        self.weight_vwap = max(0.0, float(resolved["weight_vwap"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.reversion_lookback_bars,
            self.vol_window,
            self.history_window,
            self.max_hold_bars,
        )
        self._build_states(size)

    def _score_pool(
        self,
        pool: tuple[str, ...],
        composites: dict[str, float],
    ) -> list[tuple[float, str, dict[str, Any]]]:
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in pool:
            item = self._symbol_states.get(symbol)
            if item is None:
                continue
            recent = simple_return(item.closes, lookback=self.reversion_lookback_bars)
            vol = realized_volatility(item.closes, window=self.vol_window)
            if recent is None or vol is None or vol <= _EPS:
                continue
            # Fade the recent move: a positive recent return -> negative score
            # (short the winner); a negative recent return -> positive score
            # (long the loser).
            base_score = -recent / max(vol, _EPS)
            composite = composites.get(symbol, 0.0)
            score = base_score * (1.0 + self.selection_tilt * composite)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "recent_return": recent,
                        "realized_vol": vol,
                        "selection_composite": composite,
                        "screened": True,
                    },
                )
            )
        return rows

    def _build_targets(
        self,
        rows: list[tuple[float, str, dict[str, Any]]],
    ) -> dict[str, tuple[str, float, dict[str, Any]]]:
        max_side = max(1, int(len(rows) * self.top_fraction))
        return _ranked_targets(
            rows,
            threshold=self.signal_threshold,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )


__all__ = [
    "SelectionGatedMomentumStrategy",
    "SelectionGatedReversionStrategy",
]
