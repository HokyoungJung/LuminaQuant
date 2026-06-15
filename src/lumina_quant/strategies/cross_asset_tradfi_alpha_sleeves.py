"""Cross-asset TradFi alpha sleeves for mixed crypto/metals/equity universes.

These event-driven strategies deliberately use only local bars/features.  They
are intended for forward testing when the configured universe includes precious
metals (gold/silver futures, spot, or ETFs) and equity/ETF/index instruments.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    drawdown_from_peak,
    log_return,
    realized_volatility,
    rolling_zscore,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
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
    _rolling_beta,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


_GOLD_HINTS = ("XAU", "GOLD", "GC", "GLD")
_SILVER_HINTS = ("XAG", "SILVER", "SI", "SLV")
_PLATINUM_HINTS = ("XPT", "PLATINUM", "PL", "PPLT")
_PALLADIUM_HINTS = ("XPD", "PALLADIUM", "PA", "PALL")
_EQUITY_HINTS = ("SPY", "QQQ", "IWM", "DIA", "ES", "NQ", "YM", "RTY", "STOCK", "EQUITY")
_METAL_HINT_GROUPS = (_GOLD_HINTS, _SILVER_HINTS, _PLATINUM_HINTS, _PALLADIUM_HINTS)


@dataclass(slots=True)
class _RatioPairState:
    gold: str
    silver: str
    gold_closes: deque[float]
    silver_closes: deque[float]
    ratios: deque[float]
    mode: str = "OUT"
    entry_ratio: float | None = None
    bars_held: int = 0
    last_eval_key: str = ""


def _symbol_upper(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")


def _find_symbol(symbols: list[str], explicit: str, hints: tuple[str, ...]) -> str:
    if explicit and explicit in symbols:
        return explicit
    for symbol in symbols:
        upper = _symbol_upper(symbol)
        if any(hint in upper for hint in hints):
            return symbol
    return ""


def _parse_symbols(raw: str, symbols: list[str]) -> list[str]:
    out: list[str] = []
    for token in str(raw or "").replace(";", ",").split(","):
        symbol = token.strip()
        if symbol and symbol in symbols and symbol not in out:
            out.append(symbol)
    return out


def _auto_metals(symbols: list[str], gold_symbol: str = "", silver_symbol: str = "") -> list[str]:
    out: list[str] = []
    gold = _find_symbol(symbols, gold_symbol, _GOLD_HINTS)
    silver = _find_symbol(symbols, silver_symbol, _SILVER_HINTS)
    for symbol in (gold, silver):
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _auto_equities(symbols: list[str], exclude: set[str]) -> list[str]:
    equities = []
    for symbol in symbols:
        if symbol in exclude:
            continue
        upper = _symbol_upper(symbol)
        if any(hint in upper for hint in _EQUITY_HINTS):
            equities.append(symbol)
    return equities or [symbol for symbol in symbols if symbol not in exclude]


def _ratio(gold_price: float | None, silver_price: float | None) -> float | None:
    if gold_price is None or silver_price is None or gold_price <= 0.0 or silver_price <= 0.0:
        return None
    return float(gold_price / silver_price)


def _latest_z(values: deque[float], window: int) -> tuple[float | None, float | None, float | None]:
    win = max(3, int(window))
    if len(values) < win:
        return None, None, None
    tail = list(values)[-win:]
    latest = tail[-1]
    mean_value = sum(tail) / float(len(tail))
    variance = sum((value - mean_value) ** 2 for value in tail) / float(len(tail) - 1)
    sigma = math.sqrt(max(0.0, variance))
    if sigma <= _EPS:
        return None, mean_value, sigma
    return float((latest - mean_value) / sigma), mean_value, sigma


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


def _emit_pair_leg(
    events: Any,
    *,
    strategy_id: str,
    strategy_name: str,
    symbol: str,
    event_time: Any,
    signal_type: str,
    price: float | None,
    ratio_value: float | None,
    reason: str,
    target_allocation: float,
    max_order_value: float,
    **metadata_extra: Any,
) -> None:
    metadata = _target_metadata(
        strategy=strategy_name,
        target_allocation=target_allocation,
        max_order_value=max_order_value,
        ratio=ratio_value,
        reason=reason,
        **metadata_extra,
    )
    _emit(
        events,
        strategy_id=strategy_id,
        symbol=symbol,
        event_time=event_time,
        signal_type=signal_type,
        price=price,
        metadata=metadata,
    )


@register("strategy", "GoldSilverRatioMeanReversionStrategy", interface="event_driven")
class GoldSilverRatioMeanReversionStrategy(Strategy):
    """Relative-value gold/silver ratio mean reversion sleeve."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "gold_symbol": HyperParam.string("gold_symbol", default="", tunable=False),
            "silver_symbol": HyperParam.string("silver_symbol", default="", tunable=False),
            "ratio_window": HyperParam.integer("ratio_window", default=240, low=20, high=20000),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.25, high=10.0),
            "exit_z": HyperParam.floating("exit_z", default=0.30, low=0.0, high=5.0),
            "stop_z": HyperParam.floating("stop_z", default=4.0, low=0.5, high=20.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        gold = _find_symbol(self.symbol_list, str(resolved["gold_symbol"]), _GOLD_HINTS)
        silver = _find_symbol(self.symbol_list, str(resolved["silver_symbol"]), _SILVER_HINTS)
        self.ratio_window = max(3, int(resolved["ratio_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.stop_z = max(self.entry_z, float(resolved["stop_z"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = _state_size(self.ratio_window, self.max_hold_bars)
        self._pair = (
            _RatioPairState(
                gold, silver, deque(maxlen=size), deque(maxlen=size), deque(maxlen=size)
            )
            if gold and silver and gold != silver
            else None
        )
        self._last_seen: dict[str, str] = {}
        self._latest: dict[str, float] = {}

    def _pair_strategy_id(self) -> str:
        return "gold_silver_ratio_mean_reversion"

    def _pair_strategy_name(self) -> str:
        return "GoldSilverRatioMeanReversionStrategy"

    def get_state(self) -> dict[str, Any]:
        pair = self._pair
        return {
            "last_seen": dict(self._last_seen),
            "latest": dict(self._latest),
            "pair": None
            if pair is None
            else {
                "gold_closes": list(pair.gold_closes),
                "silver_closes": list(pair.silver_closes),
                "ratios": list(pair.ratios),
                "mode": pair.mode,
                "entry_ratio": pair.entry_ratio,
                "bars_held": int(pair.bars_held),
                "last_eval_key": pair.last_eval_key,
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or self._pair is None:
            return
        self._last_seen = {str(k): str(v) for k, v in dict(state.get("last_seen") or {}).items()}
        self._latest = {
            str(k): float(v)
            for k, v in dict(state.get("latest") or {}).items()
            if safe_float(v) is not None
        }
        payload = state.get("pair")
        if not isinstance(payload, dict):
            return
        _restore_deque(self._pair.gold_closes, payload.get("gold_closes"))
        _restore_deque(self._pair.silver_closes, payload.get("silver_closes"))
        _restore_deque(self._pair.ratios, payload.get("ratios"))
        self._pair.mode = _mode(payload.get("mode"))
        self._pair.entry_ratio = safe_float(payload.get("entry_ratio"))
        self._pair.bars_held = _safe_non_negative_int(payload.get("bars_held"))
        self._pair.last_eval_key = str(payload.get("last_eval_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._update(symbol, snapshot)
        self._evaluate(getattr(event, "time", None), time_key(getattr(event, "time", None)))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self.symbol_list:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._update(str(symbol), snapshot)
                self._evaluate(snapshot.time, time_key(snapshot.time))

    def _update(self, symbol: str, snapshot: _Snapshot) -> None:
        if self._pair is None or symbol not in {self._pair.gold, self._pair.silver}:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= 0.0:
            return
        key = time_key(snapshot.time)
        if key and key == self._last_seen.get(symbol):
            return
        self._last_seen[symbol] = key
        self._latest[symbol] = close
        if symbol == self._pair.gold:
            self._pair.gold_closes.append(close)
        if symbol == self._pair.silver:
            self._pair.silver_closes.append(close)

    def _evaluate(self, event_time: Any, event_key: str = "") -> None:
        pair = self._pair
        if pair is None:
            return
        if event_key:
            if pair.last_eval_key == event_key:
                return
            if (
                self._last_seen.get(pair.gold) != event_key
                or self._last_seen.get(pair.silver) != event_key
            ):
                return
            pair.last_eval_key = event_key
        ratio_value = _ratio(self._latest.get(pair.gold), self._latest.get(pair.silver))
        if ratio_value is None:
            return
        pair.ratios.append(ratio_value)
        zscore, ratio_mean, ratio_sigma = _latest_z(pair.ratios, self.ratio_window)
        if zscore is None:
            return
        if pair.mode != "OUT":
            pair.bars_held += 1
            if abs(zscore) <= self.exit_z:
                self._exit_pair(event_time, ratio_value, zscore, "ratio_mean_reverted")
            elif abs(zscore) >= self.stop_z:
                self._exit_pair(event_time, ratio_value, zscore, "ratio_stop")
            elif pair.bars_held >= self.max_hold_bars:
                self._exit_pair(event_time, ratio_value, zscore, "max_hold")
            return
        if zscore >= self.entry_z:
            self._enter_pair(
                event_time, ratio_value, zscore, "SHORT_RATIO", ratio_mean, ratio_sigma
            )
        elif zscore <= -self.entry_z:
            self._enter_pair(event_time, ratio_value, zscore, "LONG_RATIO", ratio_mean, ratio_sigma)

    def _enter_pair(
        self,
        event_time: Any,
        ratio_value: float,
        zscore: float,
        mode: str,
        ratio_mean: float | None,
        ratio_sigma: float | None,
    ) -> None:
        pair = self._pair
        if pair is None:
            return
        if mode == "SHORT_RATIO":
            gold_signal, silver_signal = "SHORT", "LONG"
        else:
            gold_signal, silver_signal = "LONG", "SHORT"
        extra = {
            "zscore": zscore,
            "ratio_mean": ratio_mean,
            "ratio_sigma": ratio_sigma,
            "mode": mode,
        }
        strategy_id = self._pair_strategy_id()
        strategy_name = self._pair_strategy_name()
        _emit_pair_leg(
            self.events,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=pair.gold,
            event_time=event_time,
            signal_type=gold_signal,
            price=self._latest.get(pair.gold),
            ratio_value=ratio_value,
            reason="ratio_entry",
            target_allocation=self.target_allocation / 2.0,
            max_order_value=self.max_order_value,
            leg="gold",
            **extra,
        )
        _emit_pair_leg(
            self.events,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=pair.silver,
            event_time=event_time,
            signal_type=silver_signal,
            price=self._latest.get(pair.silver),
            ratio_value=ratio_value,
            reason="ratio_entry",
            target_allocation=self.target_allocation / 2.0,
            max_order_value=self.max_order_value,
            leg="silver",
            **extra,
        )
        pair.mode = mode
        pair.entry_ratio = ratio_value
        pair.bars_held = 0

    def _exit_pair(self, event_time: Any, ratio_value: float, zscore: float, reason: str) -> None:
        pair = self._pair
        if pair is None:
            return
        strategy_id = self._pair_strategy_id()
        strategy_name = self._pair_strategy_name()
        _emit_pair_leg(
            self.events,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=pair.gold,
            event_time=event_time,
            signal_type="EXIT",
            price=self._latest.get(pair.gold),
            ratio_value=ratio_value,
            reason=reason,
            target_allocation=self.target_allocation / 2.0,
            max_order_value=self.max_order_value,
            leg="gold",
            zscore=zscore,
        )
        _emit_pair_leg(
            self.events,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=pair.silver,
            event_time=event_time,
            signal_type="EXIT",
            price=self._latest.get(pair.silver),
            ratio_value=ratio_value,
            reason=reason,
            target_allocation=self.target_allocation / 2.0,
            max_order_value=self.max_order_value,
            leg="silver",
            zscore=zscore,
        )
        pair.mode = "OUT"
        pair.entry_ratio = None
        pair.bars_held = 0


@register("strategy", "GoldSilverRatioTrendStrategy", interface="event_driven")
class GoldSilverRatioTrendStrategy(GoldSilverRatioMeanReversionStrategy):
    """Trend-follow the gold/silver ratio after high/low breakouts."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = dict(super().get_param_schema())
        schema.update(
            {
                "breakout_lookback": HyperParam.integer(
                    "breakout_lookback", default=240, low=20, high=20000
                ),
                "exit_lookback": HyperParam.integer("exit_lookback", default=80, low=5, high=10000),
                "ratio_break_buffer": HyperParam.floating(
                    "ratio_break_buffer", default=0.001, low=0.0, high=0.20
                ),
            }
        )
        return schema

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.breakout_lookback = max(3, int(resolved["breakout_lookback"]))
        self.exit_lookback = max(2, int(resolved["exit_lookback"]))
        self.ratio_break_buffer = max(0.0, float(resolved["ratio_break_buffer"]))

    def _pair_strategy_id(self) -> str:
        return "gold_silver_ratio_trend"

    def _pair_strategy_name(self) -> str:
        return "GoldSilverRatioTrendStrategy"

    def _evaluate(self, event_time: Any, event_key: str = "") -> None:
        pair = self._pair
        if pair is None:
            return
        if event_key:
            if pair.last_eval_key == event_key:
                return
            if (
                self._last_seen.get(pair.gold) != event_key
                or self._last_seen.get(pair.silver) != event_key
            ):
                return
            pair.last_eval_key = event_key
        ratio_value = _ratio(self._latest.get(pair.gold), self._latest.get(pair.silver))
        if ratio_value is None:
            return
        prior_ratios = list(pair.ratios)
        enough_breakout = len(prior_ratios) >= self.breakout_lookback
        enough_exit = len(prior_ratios) >= self.exit_lookback
        prior_high = max(prior_ratios[-self.breakout_lookback :]) if enough_breakout else None
        prior_low = min(prior_ratios[-self.breakout_lookback :]) if enough_breakout else None
        exit_mid = None
        if enough_exit:
            exit_mid = (
                max(prior_ratios[-self.exit_lookback :]) + min(prior_ratios[-self.exit_lookback :])
            ) / 2.0
        pair.ratios.append(ratio_value)
        if pair.mode != "OUT":
            pair.bars_held += 1
            if (pair.mode == "LONG_RATIO" and exit_mid is not None and ratio_value < exit_mid) or (
                pair.mode == "SHORT_RATIO" and exit_mid is not None and ratio_value > exit_mid
            ):
                self._exit_pair(event_time, ratio_value, 0.0, "ratio_mid_exit")
            elif pair.bars_held >= self.max_hold_bars:
                self._exit_pair(event_time, ratio_value, 0.0, "max_hold")
            return
        if prior_high is not None and ratio_value > prior_high * (1.0 + self.ratio_break_buffer):
            self._enter_pair(event_time, ratio_value, 0.0, "LONG_RATIO", prior_high, None)
        elif prior_low is not None and ratio_value < prior_low * (1.0 - self.ratio_break_buffer):
            self._enter_pair(event_time, ratio_value, 0.0, "SHORT_RATIO", prior_low, None)


@register("strategy", "EquityMetalRiskRegimeRotationStrategy", interface="event_driven")
class EquityMetalRiskRegimeRotationStrategy(Strategy):
    """Rotate between equity risk-on sleeves and metal defensive sleeves."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "equity_symbols": HyperParam.string("equity_symbols", default="", tunable=False),
            "metal_symbols": HyperParam.string("metal_symbols", default="", tunable=False),
            "gold_symbol": HyperParam.string("gold_symbol", default="", tunable=False),
            "silver_symbol": HyperParam.string("silver_symbol", default="", tunable=False),
            "equity_benchmark_symbol": HyperParam.string(
                "equity_benchmark_symbol", default="", tunable=False
            ),
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=120, low=4, high=10080
            ),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "risk_off_return": HyperParam.floating(
                "risk_off_return", default=-0.025, low=-1.0, high=0.0
            ),
            "risk_off_drawdown": HyperParam.floating(
                "risk_off_drawdown", default=-0.060, low=-1.0, high=0.0
            ),
            "risk_on_return": HyperParam.floating(
                "risk_on_return", default=0.006, low=-1.0, high=1.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=4, low=0, high=128),
            "allow_equity_short_risk_off": HyperParam.boolean(
                "allow_equity_short_risk_off", default=True, grid=[True, False]
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.045, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.45, low=0.0, high=5.0, tunable=False
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
        explicit_metals = _parse_symbols(str(resolved["metal_symbols"]), self.symbol_list)
        auto_metals = _auto_metals(
            self.symbol_list, str(resolved["gold_symbol"]), str(resolved["silver_symbol"])
        )
        self.metal_symbols = explicit_metals or auto_metals
        self.equity_symbols = _parse_symbols(
            str(resolved["equity_symbols"]), self.symbol_list
        ) or _auto_equities(self.symbol_list, set(self.metal_symbols))
        preferred_benchmark = str(resolved["equity_benchmark_symbol"] or "")
        self.equity_benchmark_symbol = (
            preferred_benchmark
            if preferred_benchmark in self.symbol_list
            else (self.equity_symbols[0] if self.equity_symbols else "")
        )
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.risk_off_return = min(0.0, float(resolved["risk_off_return"]))
        self.risk_off_drawdown = min(0.0, float(resolved["risk_off_drawdown"]))
        self.risk_on_return = float(resolved["risk_on_return"])
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.allow_equity_short_risk_off = bool(resolved["allow_equity_short_risk_off"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.momentum_lookback_bars, self.vol_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._tick = 0
        self._last_eval_time_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": self._tick,
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
                    _restore_cross(self._state[symbol], payload)

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

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="equity_metal_risk_regime_rotation",
                strategy_name="EquityMetalRiskRegimeRotationStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        benchmark = self._state.get(self.equity_benchmark_symbol)
        if benchmark is None:
            return
        bench_ret = log_return(benchmark.closes, lookback=self.momentum_lookback_bars)
        bench_dd = drawdown_from_peak(benchmark.closes, window=self.momentum_lookback_bars)
        risk_off = (bench_ret is not None and bench_ret <= self.risk_off_return) or (
            bench_dd is not None and bench_dd <= self.risk_off_drawdown
        )
        risk_on = bench_ret is not None and bench_ret >= self.risk_on_return and not risk_off
        rows: list[tuple[float, str, dict[str, Any]]] = []
        if risk_off:
            for symbol in self.metal_symbols:
                item = self._state.get(symbol)
                if item is None:
                    continue
                mom = log_return(item.closes, lookback=self.momentum_lookback_bars)
                vol = realized_volatility(item.closes, window=self.vol_window)
                if mom is None or vol is None or vol <= _EPS:
                    continue
                rows.append(
                    (
                        mom / vol,
                        symbol,
                        {
                            "regime": "risk_off",
                            "momentum": mom,
                            "realized_vol": vol,
                            "benchmark_return": bench_ret,
                            "benchmark_drawdown": bench_dd,
                        },
                    )
                )
            if self.allow_equity_short_risk_off:
                for symbol in self.equity_symbols[: self.max_longs]:
                    if symbol == self.equity_benchmark_symbol or symbol not in self._state:
                        continue
                    rows.append(
                        (
                            -1.0,
                            symbol,
                            {
                                "regime": "risk_off_equity_hedge",
                                "benchmark_return": bench_ret,
                                "benchmark_drawdown": bench_dd,
                            },
                        )
                    )
        elif risk_on:
            for symbol in self.equity_symbols:
                item = self._state.get(symbol)
                if item is None:
                    continue
                mom = log_return(item.closes, lookback=self.momentum_lookback_bars)
                vol = realized_volatility(item.closes, window=self.vol_window)
                if mom is None or vol is None or vol <= _EPS or mom <= 0.0:
                    continue
                rows.append(
                    (
                        mom / vol,
                        symbol,
                        {
                            "regime": "risk_on",
                            "momentum": mom,
                            "realized_vol": vol,
                            "benchmark_return": bench_ret,
                            "benchmark_drawdown": bench_dd,
                        },
                    )
                )
        targets = _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=self.max_longs,
            max_shorts=self.max_longs if risk_off and self.allow_equity_short_risk_off else 0,
            allow_short=True,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="equity_metal_risk_regime_rotation",
            strategy_name="EquityMetalRiskRegimeRotationStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=1.0,
        )


@register("strategy", "EquityBenchmarkResidualReversalStrategy", interface="event_driven")
class EquityBenchmarkResidualReversalStrategy(Strategy):
    """Mean-revert stock/ETF residual moves versus an equity benchmark."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "equity_symbols": HyperParam.string("equity_symbols", default="", tunable=False),
            "benchmark_symbol": HyperParam.string("benchmark_symbol", default="", tunable=False),
            "lookback_bars": HyperParam.integer("lookback_bars", default=24, low=2, high=4096),
            "beta_window": HyperParam.integer("beta_window", default=240, low=20, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=12, low=1, high=10080),
            "entry_score": HyperParam.floating("entry_score", default=0.45, low=0.0, high=20.0),
            "max_longs": HyperParam.integer("max_longs", default=4, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=4, low=0, high=128),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.035, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=100000),
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
        self.benchmark_symbol = str(resolved["benchmark_symbol"] or "")
        if self.benchmark_symbol not in self.symbol_list:
            equity_guess = _auto_equities(self.symbol_list, set())
            self.benchmark_symbol = (
                equity_guess[0]
                if equity_guess
                else (self.symbol_list[0] if self.symbol_list else "")
            )
        self.equity_symbols = _parse_symbols(str(resolved["equity_symbols"]), self.symbol_list) or [
            symbol
            for symbol in _auto_equities(self.symbol_list, set())
            if symbol != self.benchmark_symbol
        ]
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.beta_window = max(3, int(resolved["beta_window"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.entry_score = max(0.0, float(resolved["entry_score"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.lookback_bars, self.beta_window, self.vol_window, self.max_hold_bars
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._tick = 0
        self._last_eval_time_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": self._tick,
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
                    _restore_cross(self._state[symbol], payload)

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

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="equity_benchmark_residual_reversal",
                strategy_name="EquityBenchmarkResidualReversalStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None:
            return
        bench_ret = log_return(benchmark.closes, lookback=self.lookback_bars)
        if bench_ret is None:
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in self.equity_symbols:
            item = self._state.get(symbol)
            if item is None:
                continue
            raw_ret = log_return(item.closes, lookback=self.lookback_bars)
            vol = realized_volatility(item.closes, window=self.vol_window)
            beta = _rolling_beta(item.closes, benchmark.closes, self.beta_window, 0.0, 5.0)
            if raw_ret is None or vol is None or vol <= _EPS or beta is None:
                continue
            residual = raw_ret - beta * bench_ret
            score = -residual / max(vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "raw_return": raw_ret,
                        "benchmark_return": bench_ret,
                        "beta": beta,
                        "residual_return": residual,
                        "realized_vol": vol,
                    },
                )
            )
        targets = _ranked_targets(
            rows,
            threshold=self.entry_score,
            max_longs=self.max_longs,
            max_shorts=self.max_shorts,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="equity_benchmark_residual_reversal",
            strategy_name="EquityBenchmarkResidualReversalStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=self.entry_score,
        )


@register("strategy", "MetalEquityDivergenceReversalStrategy", interface="event_driven")
class MetalEquityDivergenceReversalStrategy(Strategy):
    """Fade extreme short-horizon divergence between metal basket and equity basket."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "equity_symbols": HyperParam.string("equity_symbols", default="", tunable=False),
            "metal_symbols": HyperParam.string("metal_symbols", default="", tunable=False),
            "gold_symbol": HyperParam.string("gold_symbol", default="", tunable=False),
            "silver_symbol": HyperParam.string("silver_symbol", default="", tunable=False),
            "lookback_bars": HyperParam.integer("lookback_bars", default=48, low=2, high=4096),
            "divergence_threshold": HyperParam.floating(
                "divergence_threshold", default=0.035, low=0.0, high=1.0
            ),
            "exit_threshold": HyperParam.floating(
                "exit_threshold", default=0.008, low=0.0, high=1.0
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.030, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=100000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.012, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=350.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.metal_symbols = _parse_symbols(
            str(resolved["metal_symbols"]), self.symbol_list
        ) or _auto_metals(
            self.symbol_list, str(resolved["gold_symbol"]), str(resolved["silver_symbol"])
        )
        self.equity_symbols = _parse_symbols(
            str(resolved["equity_symbols"]), self.symbol_list
        ) or _auto_equities(self.symbol_list, set(self.metal_symbols))
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.divergence_threshold = max(0.0, float(resolved["divergence_threshold"]))
        self.exit_threshold = max(0.0, float(resolved["exit_threshold"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.lookback_bars, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._mode = "OUT"
        self._bars_held = 0
        self._last_eval_time_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "bars_held": self._bars_held,
            "last_eval_time_key": self._last_eval_time_key,
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        mode = str(state.get("mode", "OUT")).upper()
        self._mode = (
            mode if mode in {"OUT", "SHORT_METAL_LONG_EQUITY", "LONG_METAL_SHORT_EQUITY"} else "OUT"
        )
        self._bars_held = _safe_non_negative_int(state.get("bars_held"))
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    _restore_cross(self._state[symbol], payload)

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
            if snapshot is not None and self._update(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._evaluate(snapshot.time)

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

    def _basket_return(self, symbols: list[str]) -> float | None:
        returns: list[float] = []
        for symbol in symbols:
            item = self._state.get(symbol)
            if item is None:
                continue
            ret = log_return(item.closes, lookback=self.lookback_bars)
            if ret is not None:
                returns.append(ret)
        if not returns:
            return None
        return float(sum(returns) / len(returns))

    def _emit_group(
        self, event_time: Any, symbols: list[str], signal_type: str, reason: str, divergence: float
    ) -> None:
        per_symbol_alloc = self.target_allocation / max(1, len(symbols))
        for symbol in symbols:
            item = self._state.get(symbol)
            price = item.closes[-1] if item and item.closes else None
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0 and signal_type in {"LONG", "SHORT"}:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if signal_type == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy="MetalEquityDivergenceReversalStrategy",
                target_allocation=per_symbol_alloc,
                max_order_value=self.max_order_value,
                reason=reason,
                divergence=divergence,
            )
            _emit(
                self.events,
                strategy_id="metal_equity_divergence_reversal",
                symbol=symbol,
                event_time=event_time,
                signal_type=signal_type,
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )

    def _evaluate(self, event_time: Any) -> None:
        metal_ret = self._basket_return(self.metal_symbols)
        equity_ret = self._basket_return(self.equity_symbols)
        if metal_ret is None or equity_ret is None:
            return
        divergence = metal_ret - equity_ret
        if self._mode != "OUT":
            self._bars_held += 1
            if abs(divergence) <= self.exit_threshold or self._bars_held >= self.max_hold_bars:
                self._emit_group(
                    event_time,
                    self.metal_symbols + self.equity_symbols,
                    "EXIT",
                    "divergence_closed" if abs(divergence) <= self.exit_threshold else "max_hold",
                    divergence,
                )
                self._mode = "OUT"
                self._bars_held = 0
            return
        if divergence >= self.divergence_threshold:
            self._emit_group(
                event_time, self.metal_symbols, "SHORT", "metal_outperformance_fade", divergence
            )
            self._emit_group(
                event_time, self.equity_symbols, "LONG", "equity_underperformance_fade", divergence
            )
            self._mode = "SHORT_METAL_LONG_EQUITY"
            self._bars_held = 0
        elif divergence <= -self.divergence_threshold:
            self._emit_group(
                event_time, self.metal_symbols, "LONG", "metal_underperformance_fade", divergence
            )
            self._emit_group(
                event_time, self.equity_symbols, "SHORT", "equity_outperformance_fade", divergence
            )
            self._mode = "LONG_METAL_SHORT_EQUITY"
            self._bars_held = 0


def _resolve_metal_basket(
    symbols: list[str],
    raw: str,
    *,
    gold_symbol: str = "",
    silver_symbol: str = "",
    platinum_symbol: str = "",
    palladium_symbol: str = "",
) -> list[str]:
    """Resolve the ordered XAU/XAG/XPT/XPD basket from config or symbol hints."""
    explicit = _parse_symbols(raw, symbols)
    if len(explicit) >= 2:
        out: list[str] = []
        for symbol in explicit:
            if symbol not in out:
                out.append(symbol)
        return out
    out = []
    preferred = (gold_symbol, silver_symbol, platinum_symbol, palladium_symbol)
    for explicit_symbol, hints in zip(preferred, _METAL_HINT_GROUPS, strict=True):
        found = _find_symbol(symbols, str(explicit_symbol or ""), hints)
        if found and found not in out:
            out.append(found)
    return out


@dataclass(slots=True)
class _MetalLegState:
    numerator: str
    denominator: str
    ratios: deque[float]
    mode: str = "OUT"
    bars_held: int = 0
    entry_ratio: float | None = None


@register("strategy", "MetalsRelativeValueBasketStrategy", interface="event_driven")
class MetalsRelativeValueBasketStrategy(Strategy):
    """Market-neutral relative-value basket across precious-metal perps.

    The sleeve forms a chain of metal-pair ratio legs across the configured
    XAU/XAG/XPT/XPD basket (reusing the ``_RatioPairState`` ratio-leg pattern)
    and z-scores each ratio against its own trailing window.  When a leg's
    z-score crosses ``entry_z`` it fades the dislocation by going long the cheap
    metal and short the rich metal (each leg is dollar-balanced, so the book is
    market-neutral by construction); the leg unwinds when the z-score reverts
    inside ``exit_z`` or ``max_hold_bars`` elapses.  Rebalance is event-banded
    (only on a z-cross) so turnover stays well under the 2.5 hard cap.
    """

    decision_cadence_seconds = 14400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "metal_symbols": HyperParam.string("metal_symbols", default="", tunable=False),
            "gold_symbol": HyperParam.string("gold_symbol", default="", tunable=False),
            "silver_symbol": HyperParam.string("silver_symbol", default="", tunable=False),
            "platinum_symbol": HyperParam.string("platinum_symbol", default="", tunable=False),
            "palladium_symbol": HyperParam.string("palladium_symbol", default="", tunable=False),
            "zscore_window": HyperParam.integer("zscore_window", default=120, low=10, high=20000),
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.25, high=10.0),
            "exit_z": HyperParam.floating("exit_z", default=0.40, low=0.0, high=5.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "min_price": HyperParam.floating("min_price", default=0.0, low=0.0, high=1_000_000.0),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.020, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.metal_symbols = _resolve_metal_basket(
            self.symbol_list,
            str(resolved["metal_symbols"]),
            gold_symbol=str(resolved["gold_symbol"]),
            silver_symbol=str(resolved["silver_symbol"]),
            platinum_symbol=str(resolved["platinum_symbol"]),
            palladium_symbol=str(resolved["palladium_symbol"]),
        )
        self.zscore_window = max(3, int(resolved["zscore_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = _state_size(self.zscore_window, self.max_hold_bars)
        # Chain consecutive metals into ratio legs (XAU/XAG, XAG/XPT, XPT/XPD, ...).
        self._legs: list[_MetalLegState] = []
        for numerator, denominator in zip(
            self.metal_symbols, self.metal_symbols[1:], strict=False
        ):
            self._legs.append(
                _MetalLegState(numerator, denominator, deque(maxlen=size))
            )
        self._latest: dict[str, float] = {}
        self._last_seen: dict[str, str] = {}
        self._last_eval_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "latest": dict(self._latest),
            "last_seen": dict(self._last_seen),
            "last_eval_key": self._last_eval_key,
            "legs": [
                {
                    "ratios": list(leg.ratios),
                    "mode": leg.mode,
                    "bars_held": int(leg.bars_held),
                    "entry_ratio": leg.entry_ratio,
                }
                for leg in self._legs
            ],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._latest = {
            str(k): float(v)
            for k, v in dict(state.get("latest") or {}).items()
            if safe_float(v) is not None
        }
        self._last_seen = {
            str(k): str(v) for k, v in dict(state.get("last_seen") or {}).items()
        }
        self._last_eval_key = str(state.get("last_eval_key", ""))
        payload = state.get("legs")
        if isinstance(payload, list):
            for leg, leg_payload in zip(self._legs, payload, strict=False):
                if not isinstance(leg_payload, dict):
                    continue
                _restore_deque(leg.ratios, leg_payload.get("ratios"))
                leg.mode = _mode(leg_payload.get("mode"))
                leg.bars_held = _safe_non_negative_int(leg_payload.get("bars_held"))
                leg.entry_ratio = safe_float(leg_payload.get("entry_ratio"))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.metal_symbols):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._update(symbol, snapshot)
        self._evaluate(getattr(event, "time", None), time_key(getattr(event, "time", None)))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self.metal_symbols:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._update(str(symbol), snapshot)
                self._evaluate(snapshot.time, time_key(snapshot.time))

    def _update(self, symbol: str, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price or close <= 0.0:
            return
        key = time_key(snapshot.time)
        if key and key == self._last_seen.get(symbol):
            return
        self._last_seen[symbol] = key
        self._latest[symbol] = close

    def _evaluate(self, event_time: Any, event_key: str = "") -> None:
        # Graceful degradation: need at least one valid ratio leg (>=2 metals).
        if not self._legs:
            return
        if event_key:
            if self._last_eval_key == event_key:
                return
            self._last_eval_key = event_key
        for leg in self._legs:
            self._evaluate_leg(leg, event_time)

    def _evaluate_leg(self, leg: _MetalLegState, event_time: Any) -> None:
        ratio_value = _ratio(self._latest.get(leg.numerator), self._latest.get(leg.denominator))
        if ratio_value is None:
            return
        leg.ratios.append(ratio_value)
        zscore = rolling_zscore(leg.ratios, window=self.zscore_window)
        if zscore is None:
            return
        if leg.mode != "OUT":
            leg.bars_held += 1
            if abs(zscore) <= self.exit_z:
                self._exit_leg(leg, event_time, ratio_value, zscore, "ratio_mean_reverted")
            elif leg.bars_held >= self.max_hold_bars:
                self._exit_leg(leg, event_time, ratio_value, zscore, "max_hold")
            return
        if zscore >= self.entry_z:
            # Rich numerator vs basket -> short numerator, long denominator.
            self._enter_leg(leg, event_time, ratio_value, zscore, "SHORT_RATIO")
        elif zscore <= -self.entry_z:
            self._enter_leg(leg, event_time, ratio_value, zscore, "LONG_RATIO")

    def _leg_allocation(self) -> float:
        # Dollar-balance each leg across the whole basket so gross stays bounded.
        legs = max(1, len(self._legs))
        return self.target_allocation / float(2 * legs)

    def _emit_leg(
        self,
        *,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float | None,
        ratio_value: float | None,
        zscore: float,
        reason: str,
        leg_role: str,
    ) -> None:
        if signal_type == "EXIT":
            # EXIT closes regardless of allocation; match the bare-metadata EXIT
            # convention used by the sibling sleeves.
            metadata = {
                "strategy": "MetalsRelativeValueBasketStrategy",
                "reason": reason,
                "leg_role": leg_role,
                "ratio": ratio_value,
                "zscore": zscore,
            }
        else:
            metadata = _target_metadata(
                strategy="MetalsRelativeValueBasketStrategy",
                target_allocation=self._leg_allocation(),
                max_order_value=self.max_order_value,
                ratio=ratio_value,
                zscore=zscore,
                reason=reason,
                leg_role=leg_role,
            )
        _emit(
            self.events,
            strategy_id="metals_relative_value_basket",
            symbol=symbol,
            event_time=event_time,
            signal_type=signal_type,
            price=price,
            metadata=metadata,
        )

    def _enter_leg(
        self, leg: _MetalLegState, event_time: Any, ratio_value: float, zscore: float, mode: str
    ) -> None:
        if mode == "SHORT_RATIO":
            numerator_signal, denominator_signal = "SHORT", "LONG"
        else:
            numerator_signal, denominator_signal = "LONG", "SHORT"
        self._emit_leg(
            symbol=leg.numerator,
            event_time=event_time,
            signal_type=numerator_signal,
            price=self._latest.get(leg.numerator),
            ratio_value=ratio_value,
            zscore=zscore,
            reason="ratio_entry",
            leg_role="numerator",
        )
        self._emit_leg(
            symbol=leg.denominator,
            event_time=event_time,
            signal_type=denominator_signal,
            price=self._latest.get(leg.denominator),
            ratio_value=ratio_value,
            zscore=zscore,
            reason="ratio_entry",
            leg_role="denominator",
        )
        leg.mode = mode
        leg.entry_ratio = ratio_value
        leg.bars_held = 0

    def _exit_leg(
        self, leg: _MetalLegState, event_time: Any, ratio_value: float, zscore: float, reason: str
    ) -> None:
        for symbol, leg_role in ((leg.numerator, "numerator"), (leg.denominator, "denominator")):
            self._emit_leg(
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=self._latest.get(symbol),
                ratio_value=ratio_value,
                zscore=zscore,
                reason=reason,
                leg_role=leg_role,
            )
        leg.mode = "OUT"
        leg.entry_ratio = None
        leg.bars_held = 0


__all__ = [
    "EquityBenchmarkResidualReversalStrategy",
    "EquityMetalRiskRegimeRotationStrategy",
    "GoldSilverRatioMeanReversionStrategy",
    "GoldSilverRatioTrendStrategy",
    "MetalEquityDivergenceReversalStrategy",
    "MetalsRelativeValueBasketStrategy",
]
