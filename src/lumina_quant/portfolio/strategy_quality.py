"""Common portfolio-level quality overlays for existing strategies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from itertools import pairwise
from typing import Any

from lumina_quant.core.events import SignalEvent

_EPS = 1e-12
_EDGE_KEYS = (
    "expected_edge_bps",
    "edge_bps",
    "forecast_edge_bps",
    "pred_edge_bps",
    "return_bps",
    "pred_return_bps",
)
_RETURN_KEYS = (
    "expected_return",
    "forecast_return",
    "pred_return",
    "predicted_return",
    "mean_pred_return",
)
_TREND_TOKENS = (
    "trend",
    "momentum",
    "breakout",
    "continuation",
    "compression",
    "topcap",
    "moving_average",
)
_REVERSION_TOKENS = (
    "reversion",
    "rsi",
    "vwap",
    "mean",
    "panic",
    "shock",
    "exhaustion",
)
_PAIR_TOKENS = ("pair", "zscore", "leadlag", "lag_convergence", "spread")
_CARRY_TOKENS = ("carry", "funding", "crowding", "derivatives")


@dataclass(slots=True)
class StrategyQualityDecision:
    signal: SignalEvent | None
    blocked_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _StrategyHealth:
    returns: deque[float]
    cooldown_until_bar: int = 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _event_day(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    token = str(value or "")
    return token[:10] if len(token) >= 10 else token


def _returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in pairwise(values):
        if prev > 0.0 and cur > 0.0:
            out.append(cur / prev - 1.0)
    return out


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / float(len(values))
    var = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(0.0, var))


def _strategy_key(signal: SignalEvent, metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("strategy")
        or metadata.get("strategy_id")
        or getattr(signal, "strategy_id", "")
        or "unknown"
    )


def _family(strategy_key: str) -> str:
    token = strategy_key.lower()
    if any(item in token for item in _PAIR_TOKENS):
        return "pair"
    if any(item in token for item in _CARRY_TOKENS):
        return "carry"
    if any(item in token for item in _REVERSION_TOKENS):
        return "reversion"
    if any(item in token for item in _TREND_TOKENS):
        return "trend"
    return "general"


def _direction(signal_type: str) -> int:
    token = str(signal_type or "").upper()
    if token == "LONG":
        return 1
    if token == "SHORT":
        return -1
    return 0


# Pre-registered protective exit_reason whitelist for the L-C min-hold gate.
# Descriptive labels on ordinary exits (e.g. "rebalance") are deliberately NOT
# in this set — any truthy exit_reason must not neutralize the gate.
_PROTECTIVE_EXIT_REASONS: frozenset[str] = frozenset(
    {
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "trailing_exit",
        "liquidation",
        "kill_switch",
        "risk_off",
        "drawdown_flatten",
        "protective_stop",
    }
)


class StrategyQualityOverlay:
    """Applies edge, regime, sizing, turnover, exit and health overlays."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "STRATEGY_QUALITY_ENABLED", False))
        # L-C min-hold gate: active independently of ``enabled`` so the cost
        # lever can be A/B-measured without the full overlay. 0 = OFF
        # (byte-identical default path). The gate keeps its own ledger keyed
        # by "<component_id or __net__>|<symbol>" — separate from ``_entry``
        # (the health tracker) so the enabled-overlay state stays untouched —
        # and DEFERS blocked bare EXITs instead of dropping them: the blocked
        # exit is released as a synthetic EXIT signal at hold maturity via
        # ``pop_matured_pending_exits`` (one-shot transition-emit strategies
        # never re-emit, so a dropped exit would leak the position forever).
        self.min_hold_bars = max(0, int(getattr(config, "STRATEGY_QUALITY_MIN_HOLD_BARS", 0) or 0))
        self.bar_index = 0
        self._turnover_day = ""
        self._daily_turnover = 0.0
        self._health: dict[str, _StrategyHealth] = {}
        self._entry: dict[str, dict[str, Any]] = {}
        self._min_hold_entries: dict[str, dict[str, Any]] = {}
        self._profit_moonshot_intent: dict[str, tuple[int, int]] = {}

    def get_state(self) -> dict[str, Any]:
        return {
            "bar_index": int(self.bar_index),
            "turnover_day": self._turnover_day,
            "daily_turnover": float(self._daily_turnover),
            "health": {
                key: {
                    "returns": list(value.returns),
                    "cooldown_until_bar": int(value.cooldown_until_bar),
                }
                for key, value in self._health.items()
            },
            "entry": {key: dict(value) for key, value in self._entry.items()},
            "profit_moonshot_intent": {
                key: [int(value[0]), int(value[1])]
                for key, value in self._profit_moonshot_intent.items()
            },
            # Emitted only when the min-hold ledger holds records so the state
            # shape stays byte-identical for every pre-existing configuration.
            **(
                {
                    "min_hold_entries": {
                        key: dict(value) for key, value in self._min_hold_entries.items()
                    }
                }
                if self._min_hold_entries
                else {}
            ),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.bar_index = int(state.get("bar_index") or 0)
        self._turnover_day = str(state.get("turnover_day") or "")
        self._daily_turnover = float(state.get("daily_turnover") or 0.0)
        window = max(1, int(getattr(self.config, "STRATEGY_QUALITY_HEALTH_WINDOW_TRADES", 20)))
        health: dict[str, _StrategyHealth] = {}
        for key, raw in dict(state.get("health") or {}).items():
            if not isinstance(raw, dict):
                continue
            values = deque(maxlen=window)
            for item in list(raw.get("returns") or [])[-window:]:
                parsed = _safe_float(item)
                if parsed is not None:
                    values.append(parsed)
            health[str(key)] = _StrategyHealth(
                returns=values,
                cooldown_until_bar=int(raw.get("cooldown_until_bar") or 0),
            )
        self._health = health
        self._entry = {
            str(key): dict(value)
            for key, value in dict(state.get("entry") or {}).items()
            if isinstance(value, dict)
        }
        self._profit_moonshot_intent = {}
        for key, value in dict(state.get("profit_moonshot_intent") or {}).items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                self._profit_moonshot_intent[str(key)] = (int(value[0]), int(value[1]))
        raw_min_hold = state.get("min_hold_entries")
        self._min_hold_entries = {}
        if isinstance(raw_min_hold, dict):
            for key, value in raw_min_hold.items():
                if isinstance(value, dict):
                    self._min_hold_entries[str(key)] = dict(value)

    def next_bar(self, event_time: Any) -> None:
        self.bar_index += 1
        day = _event_day(event_time)
        if day and day != self._turnover_day:
            self._turnover_day = day
            self._daily_turnover = 0.0

    def _min_hold_key(self, symbol: str, component_id: Any) -> str:
        token = str(component_id).strip() if component_id else ""
        return f"{token or '__net__'}|{symbol}"

    def _note_min_hold_fill(
        self,
        symbol: str,
        old_qty: float,
        new_qty: float,
        *,
        component_id: Any = None,
    ) -> None:
        """Maintain the min-hold ledger from the book the fill actually moved."""
        key = self._min_hold_key(symbol, component_id)
        if abs(new_qty) < 1e-12:
            self._min_hold_entries.pop(key, None)
            return
        if old_qty == 0.0 or (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
            self._min_hold_entries[key] = {
                "qty": float(new_qty),
                "entry_bar": int(self.bar_index),
            }
            return
        record = self._min_hold_entries.get(key)
        if record is not None:
            record["qty"] = float(new_qty)

    def note_fill(
        self,
        fill: Any,
        old_qty: float,
        new_qty: float,
        fill_price: float,
        *,
        component_id: Any = None,
        component_old_qty: float | None = None,
        component_new_qty: float | None = None,
    ) -> None:
        symbol = str(getattr(fill, "symbol", ""))
        if self.min_hold_bars > 0 and symbol:
            if component_id and component_old_qty is not None and component_new_qty is not None:
                self._note_min_hold_fill(
                    symbol,
                    float(component_old_qty),
                    float(component_new_qty),
                    component_id=component_id,
                )
            else:
                self._note_min_hold_fill(symbol, float(old_qty), float(new_qty))
        if fill_price <= 0.0:
            return
        if not self.enabled:
            return
        notional = abs(float(getattr(fill, "quantity", 0.0) or 0.0)) * float(fill_price)
        self._daily_turnover += notional
        metadata = dict(getattr(fill, "metadata", {}) or {})
        signal_metadata = dict(metadata.get("signal_metadata") or {})
        strategy_key = str(signal_metadata.get("strategy_quality_strategy") or "unknown")
        if not symbol:
            return

        if old_qty == 0.0 and new_qty != 0.0:
            self._entry[symbol] = {
                "strategy": strategy_key,
                "qty": float(new_qty),
                "price": fill_price,
            }
            return

        entry = self._entry.get(symbol)
        if old_qty != 0.0 and (
            new_qty == 0.0 or (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty)
        ):
            if entry is not None:
                entry_price = _safe_float(entry.get("price")) or fill_price
                direction = 1.0 if old_qty > 0 else -1.0
                realized = direction * (fill_price / max(entry_price, _EPS) - 1.0)
                self._record_health(str(entry.get("strategy") or strategy_key), realized)
            if new_qty == 0.0:
                self._entry.pop(symbol, None)
            else:
                self._entry[symbol] = {
                    "strategy": strategy_key,
                    "qty": float(new_qty),
                    "price": fill_price,
                }

    def _record_health(self, strategy_key: str, realized_return: float) -> None:
        window = max(1, int(getattr(self.config, "STRATEGY_QUALITY_HEALTH_WINDOW_TRADES", 20)))
        health = self._health.get(strategy_key)
        if health is None:
            health = _StrategyHealth(returns=deque(maxlen=window))
            self._health[strategy_key] = health
        health.returns.append(float(realized_return))
        if realized_return < 0.0:
            health.cooldown_until_bar = max(
                health.cooldown_until_bar,
                self.bar_index
                + int(getattr(self.config, "STRATEGY_QUALITY_LOSS_COOLDOWN_BARS", 0)),
            )

    def _min_hold_block(self, signal: SignalEvent) -> str:
        """L-C min-hold gate reason, or "" when the signal may pass.

        While a tracked position is younger than ``min_hold_bars``:

        - a BARE strategy EXIT is DEFERRED, not dropped: the record is marked
          ``exit_pending`` (with the signal's identity preserved) and released
          as a synthetic EXIT at hold maturity via
          ``pop_matured_pending_exits`` — one-shot transition-emit strategies
          never re-emit, so dropping would leak the position;
        - exits carrying a protective marker always pass: truthy ``risk_exit``
          or ``overlay_reason`` metadata, or ``exit_reason`` in the
          pre-registered protective whitelist (a descriptive label like
          ``rebalance`` is NOT protective and stays gated); engine-level
          stop / take-profit / liquidation fills never route through signals
          and are never delayed;
        - opposite-direction reversals are blocked, and while an exit is
          pending SAME-direction re-entries are blocked too (the strategy
          believes it is flat — a fresh entry would stack exposure).

        The ledger is keyed by ``<component_id or __net__>|<symbol>`` so one
        component's fresh entry never gates another component's exits.
        """
        symbol = str(getattr(signal, "symbol", "") or "")
        metadata = dict(getattr(signal, "metadata", {}) or {})
        component_id = str(metadata.get("component_id") or "").strip()
        key = self._min_hold_key(symbol, component_id or None)
        record = self._min_hold_entries.get(key)
        if record is None:
            return ""
        entry_bar = record.get("entry_bar")
        if entry_bar is None:
            return ""
        matured = self.bar_index - int(entry_bar) >= self.min_hold_bars
        signal_type = str(getattr(signal, "signal_type", "")).upper()
        if signal_type == "EXIT":
            if matured:
                return ""
            if metadata.get("risk_exit") or metadata.get("overlay_reason"):
                return ""
            if str(metadata.get("exit_reason") or "").strip().lower() in _PROTECTIVE_EXIT_REASONS:
                return ""
            record["exit_pending"] = True
            record["exit_state"] = "PENDING"
            record["pending_strategy_id"] = str(getattr(signal, "strategy_id", "") or "overlay")
            record["pending_metadata"] = {
                str(k): v
                for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            return "min_hold_active"
        direction = _direction(signal_type)
        entry_qty = _safe_float(record.get("qty")) or 0.0
        if direction != 0 and record.get("exit_pending"):
            return "min_hold_exit_pending"
        if matured:
            return ""
        if direction != 0 and entry_qty != 0.0 and (direction > 0) != (entry_qty > 0):
            return "min_hold_reversal_block"
        return ""

    def mark_pending_exit_state(self, key: str, state: str) -> None:
        """Persist the latest dispatch outcome without clearing a live exit intent."""
        record = self._min_hold_entries.get(str(key))
        if record is not None and record.get("exit_pending"):
            record["exit_state"] = str(state).strip().upper() or "PENDING"

    def reconcile_min_hold_positions(
        self,
        positions: dict[str, Any],
        component_positions: dict[str, Any],
    ) -> None:
        """Drop intents only when the authoritative book is flat."""
        for key in tuple(self._min_hold_entries):
            component_token, _, symbol = key.partition("|")
            if component_token == "__net__":
                qty = _safe_float(positions.get(symbol))
            else:
                qty = _safe_float(dict(component_positions.get(component_token) or {}).get(symbol))
            if qty is None or abs(qty) < 1e-12:
                self._min_hold_entries.pop(key, None)

    def pop_matured_pending_exits(self) -> list[dict[str, Any]]:
        """Release deferred bare EXITs whose min-hold has matured.

        Returns one dict per matured pending exit: ``symbol``,
        ``component_id`` ("" for the net book), ``strategy_id``, and the
        preserved signal ``metadata``.  The caller (Portfolio.update_timeindex)
        synthesizes EXIT SignalEvents from these; the ledger record itself
        stays until the authoritative closing fill removes it via ``note_fill``.
        """
        if self.min_hold_bars <= 0 or not self._min_hold_entries:
            return []
        released: list[dict[str, Any]] = []
        for key, record in self._min_hold_entries.items():
            if not record.get("exit_pending"):
                continue
            entry_bar = record.get("entry_bar")
            if entry_bar is None:
                continue
            if self.bar_index - int(entry_bar) < self.min_hold_bars:
                continue
            component_token, _, symbol = key.partition("|")
            metadata = dict(record.get("pending_metadata") or {})
            strategy_id = str(record.get("pending_strategy_id", "") or "overlay")
            client_order_id = str(
                record.setdefault("pending_client_order_id", f"min-hold-exit:{key}")
            )
            released.append(
                {
                    "key": key,
                    "symbol": symbol,
                    "component_id": "" if component_token == "__net__" else component_token,
                    "strategy_id": strategy_id,
                    "metadata": metadata,
                    "client_order_id": client_order_id,
                }
            )
        return released

    def apply(
        self,
        signal: SignalEvent,
        *,
        bars: Any,
        current_price: float,
        current_equity: float,
    ) -> StrategyQualityDecision:
        if self.min_hold_bars > 0:
            block_reason = self._min_hold_block(signal)
            if block_reason:
                return StrategyQualityDecision(
                    None,
                    block_reason,
                    {"min_hold_bars": int(self.min_hold_bars)},
                )
        if not self.enabled:
            return StrategyQualityDecision(signal=signal)
        metadata = dict(getattr(signal, "metadata", {}) or {})
        strategy_key = _strategy_key(signal, metadata)
        metadata["strategy_quality_strategy"] = strategy_key
        if str(getattr(signal, "signal_type", "")).upper() == "EXIT":
            return StrategyQualityDecision(signal=replace(signal, metadata=metadata))
        family = _family(strategy_key)
        diagnostics: dict[str, Any] = {"family": family}

        health_scale = self._health_scale(strategy_key)
        if health_scale <= 0.0:
            return StrategyQualityDecision(None, "strategy_health_cooldown", diagnostics)

        edge_bps = self._expected_edge_bps(signal, metadata)
        cost_floor = self._cost_floor_bps()
        diagnostics["expected_edge_bps"] = edge_bps
        diagnostics["cost_floor_bps"] = cost_floor
        if self._edge_blocks(edge_bps, cost_floor):
            return StrategyQualityDecision(None, "edge_below_cost_floor", diagnostics)

        regime = self._regime(bars, signal.symbol)
        diagnostics["regime"] = regime
        regime_scale = self._regime_scale(family, regime, signal, metadata)
        if regime_scale <= 0.0:
            return StrategyQualityDecision(None, "regime_router_block", diagnostics)

        conflict_reason = self._profit_moonshot_conflict(strategy_key, signal)
        if conflict_reason:
            return StrategyQualityDecision(None, conflict_reason, diagnostics)

        pair_reason = self._pair_quality_block(family, metadata)
        if pair_reason:
            return StrategyQualityDecision(None, pair_reason, diagnostics)

        vol_scale = self._vol_scale(bars, signal.symbol)
        turnover_scale = self._turnover_scale(current_equity)
        if turnover_scale <= 0.0:
            return StrategyQualityDecision(None, "daily_turnover_budget_exhausted", diagnostics)
        scale = max(
            0.0,
            min(1.0, health_scale * regime_scale * vol_scale * turnover_scale),
        )
        diagnostics.update(
            {
                "health_scale": health_scale,
                "regime_scale": regime_scale,
                "vol_scale": vol_scale,
                "turnover_scale": turnover_scale,
                "strategy_quality_scale": scale,
            }
        )
        if scale <= 0.0:
            return StrategyQualityDecision(None, "strategy_quality_scale_zero", diagnostics)

        adjusted = self._scaled_signal(signal, metadata, scale, diagnostics)
        adjusted = self._attach_exit_overlay(adjusted, bars=bars, current_price=current_price)
        return StrategyQualityDecision(adjusted, diagnostics=diagnostics)

    def _expected_edge_bps(self, signal: SignalEvent, metadata: dict[str, Any]) -> float | None:
        for key in _EDGE_KEYS:
            parsed = _safe_float(metadata.get(key))
            if parsed is not None:
                return abs(parsed)
        for key in _RETURN_KEYS:
            parsed = _safe_float(metadata.get(key))
            if parsed is not None:
                return abs(parsed) * 10_000.0
        strength = _safe_float(getattr(signal, "strength", None))
        if strength is not None and 0.0 < abs(strength) < 1.0:
            return abs(strength) * 100.0
        return None

    def _cost_floor_bps(self) -> float:
        fee = max(
            _safe_float(getattr(self.config, "TAKER_FEE_RATE", 0.0)) or 0.0,
            _safe_float(getattr(self.config, "MAKER_FEE_RATE", 0.0)) or 0.0,
        )
        slippage = _safe_float(getattr(self.config, "SLIPPAGE_RATE", 0.0)) or 0.0
        spread = _safe_float(getattr(self.config, "SPREAD_RATE", 0.0)) or 0.0
        funding = abs(_safe_float(getattr(self.config, "FUNDING_RATE_PER_8H", 0.0)) or 0.0) / 8.0
        buffer = (
            _safe_float(getattr(self.config, "STRATEGY_QUALITY_EDGE_COST_BUFFER_BPS", 3.0)) or 0.0
        )
        minimum = (
            _safe_float(getattr(self.config, "STRATEGY_QUALITY_MIN_EXPECTED_EDGE_BPS", 8.0)) or 0.0
        )
        return max(minimum, (2.0 * fee + slippage + spread + funding) * 10_000.0 + buffer)

    def _edge_blocks(self, edge_bps: float | None, cost_floor_bps: float) -> bool:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_EDGE_GATE_ENABLED", True)):
            return False
        if edge_bps is None:
            return not bool(getattr(self.config, "STRATEGY_QUALITY_ALLOW_UNKNOWN_EDGE", True))
        return abs(edge_bps) < cost_floor_bps

    def _closes(self, bars: Any, symbol: str, n: int) -> list[float]:
        getter = getattr(bars, "get_latest_bars_values", None)
        if not callable(getter):
            return []
        try:
            values = getter(symbol, "close", max(1, int(n)))
        except Exception:
            return []
        out: list[float] = []
        for value in values or []:
            parsed = _safe_float(value)
            if parsed is not None and parsed > 0.0:
                out.append(parsed)
        return out

    def _volumes(self, bars: Any, symbol: str, n: int) -> list[float]:
        getter = getattr(bars, "get_latest_bars_values", None)
        if not callable(getter):
            return []
        try:
            values = getter(symbol, "volume", max(1, int(n)))
        except Exception:
            return []
        return [
            float(value)
            for value in (_safe_float(item) for item in values or [])
            if value is not None
        ]

    def _regime(self, bars: Any, symbol: str) -> str:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_REGIME_ROUTER_ENABLED", True)):
            return "unknown"
        lookback = int(getattr(self.config, "STRATEGY_QUALITY_REGIME_LOOKBACK_BARS", 60))
        closes = self._closes(bars, symbol, lookback)
        if len(closes) < max(5, min(lookback, 10)):
            return "unknown"
        total_return_bps = abs(closes[-1] / max(closes[0], _EPS) - 1.0) * 10_000.0
        last_return_bps = abs(closes[-1] / max(closes[-2], _EPS) - 1.0) * 10_000.0
        trend_threshold = float(getattr(self.config, "STRATEGY_QUALITY_TREND_RETURN_BPS", 35.0))
        panic_threshold = float(getattr(self.config, "STRATEGY_QUALITY_PANIC_RETURN_BPS", 120.0))
        if last_return_bps >= panic_threshold:
            return "panic"
        volumes = self._volumes(bars, symbol, lookback)
        if len(volumes) >= 5:
            current_volume = volumes[-1]
            avg_volume = sum(volumes[:-1]) / float(max(1, len(volumes) - 1))
            low_ratio = float(
                getattr(self.config, "STRATEGY_QUALITY_LOW_LIQUIDITY_VOLUME_RATIO", 0.2)
            )
            if avg_volume > 0.0 and current_volume / avg_volume <= low_ratio:
                return "low_liquidity"
        if total_return_bps >= trend_threshold:
            return "trend"
        return "chop"

    def _regime_scale(
        self,
        family: str,
        regime: str,
        signal: SignalEvent,
        metadata: dict[str, Any],
    ) -> float:
        if regime in {"unknown", "chop"}:
            return 1.0
        signal_direction = _direction(signal.signal_type)
        mean_pred = _safe_float(metadata.get("mean_pred_return"))
        if regime == "trend" and family == "reversion" and mean_pred is None:
            return 0.0
        if regime == "panic" and family in {"trend", "pair", "carry"}:
            return 0.0
        if regime == "low_liquidity" and family != "carry":
            return 0.5
        if mean_pred is not None and signal_direction and mean_pred * signal_direction < 0.0:
            return 0.0
        return 1.0

    def _profit_moonshot_conflict(self, strategy_key: str, signal: SignalEvent) -> str:
        if "profit_moonshot" not in strategy_key.lower():
            return ""
        cooldown = int(
            getattr(self.config, "STRATEGY_QUALITY_PROFIT_MOONSHOT_CONFLICT_COOLDOWN_BARS", 0)
        )
        if cooldown <= 0:
            return ""
        direction = _direction(signal.signal_type)
        if not direction:
            return ""
        key = str(signal.symbol)
        previous = self._profit_moonshot_intent.get(key)
        self._profit_moonshot_intent[key] = (direction, self.bar_index)
        if previous is None:
            return ""
        prev_direction, prev_bar = previous
        if prev_direction != direction and self.bar_index - int(prev_bar) <= cooldown:
            return "profit_moonshot_conflicting_sleeves"
        return ""

    def _pair_quality_block(self, family: str, metadata: dict[str, Any]) -> str:
        if family != "pair":
            return ""
        corr = _safe_float(metadata.get("correlation"))
        if corr is None:
            return ""
        threshold = float(getattr(self.config, "STRATEGY_QUALITY_PAIR_MIN_CORRELATION", 0.35))
        return "pair_correlation_breakdown" if abs(corr) < threshold else ""

    def _vol_scale(self, bars: Any, symbol: str) -> float:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_POSITION_SIZING_ENABLED", True)):
            return 1.0
        lookback = int(getattr(self.config, "STRATEGY_QUALITY_REGIME_LOOKBACK_BARS", 60))
        realized = _std(_returns(self._closes(bars, symbol, lookback)))
        if realized <= 0.0:
            return 1.0
        target = float(getattr(self.config, "STRATEGY_QUALITY_TARGET_VOL_PER_BAR", 0.006))
        raw = target / max(realized, _EPS) if target > 0.0 else 1.0
        min_scale = float(getattr(self.config, "STRATEGY_QUALITY_MIN_POSITION_SCALE", 0.2))
        max_scale = float(getattr(self.config, "STRATEGY_QUALITY_MAX_POSITION_SCALE", 1.0))
        return max(0.0, min(max_scale, max(min_scale, raw)))

    def _health_scale(self, strategy_key: str) -> float:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_STRATEGY_HEALTH_ENABLED", True)):
            return 1.0
        health = self._health.get(strategy_key)
        if health is None:
            return 1.0
        if self.bar_index < int(health.cooldown_until_bar):
            return 0.0
        if not health.returns:
            return 1.0
        avg_return = sum(health.returns) / float(len(health.returns))
        if avg_return >= 0.0:
            return 1.0
        min_scale = float(getattr(self.config, "STRATEGY_QUALITY_MIN_HEALTH_SCALE", 0.25))
        return max(min_scale, 1.0 + avg_return * 100.0)

    def _turnover_scale(self, current_equity: float) -> float:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_TURNOVER_BUDGET_ENABLED", True)):
            return 1.0
        budget_pct = float(getattr(self.config, "STRATEGY_QUALITY_MAX_DAILY_TURNOVER_PCT", 0.0))
        if budget_pct <= 0.0 or current_equity <= 0.0:
            return 1.0
        budget = current_equity * budget_pct
        remaining = budget - self._daily_turnover
        if remaining <= 0.0:
            return 0.0
        return max(0.0, min(1.0, remaining / max(budget, _EPS)))

    def _scaled_signal(
        self,
        signal: SignalEvent,
        metadata: dict[str, Any],
        scale: float,
        diagnostics: dict[str, Any],
    ) -> SignalEvent:
        for key in ("target_allocation", "max_order_value"):
            value = _safe_float(metadata.get(key))
            if value is not None and value > 0.0:
                metadata[key] = float(value) * scale
        metadata["strategy_quality"] = dict(diagnostics)
        return replace(
            signal,
            strength=float(getattr(signal, "strength", 1.0) or 1.0) * scale,
            metadata=metadata,
        )

    def _atr_pct(self, bars: Any, symbol: str) -> float | None:
        window = int(getattr(self.config, "STRATEGY_QUALITY_ATR_WINDOW_BARS", 20))
        getter = getattr(bars, "get_latest_bars", None)
        if not callable(getter):
            return None
        try:
            rows = list(getter(symbol, window + 1) or [])
        except Exception:
            return None
        if len(rows) < 2:
            return None
        true_ranges: list[float] = []
        prev_close = _safe_float(rows[0][4] if len(rows[0]) > 4 else None)
        for row in rows[1:]:
            high = _safe_float(row[2] if len(row) > 2 else None)
            low = _safe_float(row[3] if len(row) > 3 else None)
            close = _safe_float(row[4] if len(row) > 4 else None)
            if high is None or low is None or close is None:
                continue
            candidates = [high - low]
            if prev_close is not None:
                candidates.extend([abs(high - prev_close), abs(low - prev_close)])
            true_ranges.append(max(candidates))
            prev_close = close
        if not true_ranges or prev_close is None or prev_close <= 0.0:
            return None
        return (sum(true_ranges) / float(len(true_ranges))) / prev_close

    def _attach_exit_overlay(
        self,
        signal: SignalEvent,
        *,
        bars: Any,
        current_price: float,
    ) -> SignalEvent:
        if not bool(getattr(self.config, "STRATEGY_QUALITY_EXIT_OVERLAY_ENABLED", True)):
            return signal
        direction = _direction(signal.signal_type)
        if direction == 0 or current_price <= 0.0:
            return signal
        atr_pct = self._atr_pct(bars, signal.symbol)
        if atr_pct is None or atr_pct <= 0.0:
            return signal
        min_stop = float(getattr(self.config, "STRATEGY_QUALITY_MIN_STOP_BPS", 20.0)) / 10_000.0
        max_stop = float(getattr(self.config, "STRATEGY_QUALITY_MAX_STOP_BPS", 500.0)) / 10_000.0
        stop_pct = min(
            max_stop,
            max(
                min_stop,
                atr_pct * float(getattr(self.config, "STRATEGY_QUALITY_ATR_STOP_MULT", 2.0)),
            ),
        )
        take_pct = min(
            max_stop * 2.0,
            max(
                min_stop,
                atr_pct * float(getattr(self.config, "STRATEGY_QUALITY_ATR_TAKE_PROFIT_MULT", 3.0)),
            ),
        )
        trailing_pct = min(
            max_stop,
            max(
                min_stop,
                atr_pct * float(getattr(self.config, "STRATEGY_QUALITY_TRAILING_ATR_MULT", 1.5)),
            ),
        )
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        if direction > 0:
            stop_loss = stop_loss if stop_loss is not None else current_price * (1.0 - stop_pct)
            take_profit = (
                take_profit if take_profit is not None else current_price * (1.0 + take_pct)
            )
        else:
            stop_loss = stop_loss if stop_loss is not None else current_price * (1.0 + stop_pct)
            take_profit = (
                take_profit if take_profit is not None else current_price * (1.0 - take_pct)
            )
        trailing = signal.trailing_percent
        if trailing is None and trailing_pct > 0.0:
            trailing = trailing_pct
        metadata = dict(signal.metadata or {})
        metadata.setdefault("strategy_quality_exit_overlay", {})
        metadata["strategy_quality_exit_overlay"] = {
            "atr_pct": float(atr_pct),
            "stop_pct": float(stop_pct),
            "take_profit_pct": float(take_pct),
            "trailing_percent": float(trailing or 0.0),
        }
        return replace(
            signal,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_percent=trailing,
            metadata=metadata,
        )


__all__ = ["StrategyQualityDecision", "StrategyQualityOverlay"]
