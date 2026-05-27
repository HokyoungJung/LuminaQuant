"""Session-filtered pair-carry strategy for profit-reboot research."""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import stdev
from typing import Any

from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy
from lumina_quant.tuning import HyperParam


class SessionFilteredPairCarryStrategy(PairTradingZScoreStrategy):
    """Pair z-score strategy with UTC session and expected-move entry gates.

    The parent pair strategy owns spread/hedge state, exits, stops, and pair
    signal emission.  This wrapper only rejects fresh entries when the current
    UTC hour is outside a configured session set or when the estimated spread
    reversion edge is too small to cover fees/slippage.  Existing positions can
    still exit on every bar.
    """

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = dict(super().get_param_schema())
        schema.update(
            {
                "allowed_session_utc_hours": HyperParam.string(
                    "allowed_session_utc_hours",
                    default="0,1,8,9,13,14,15,20,21",
                    tunable=False,
                    description="Comma-separated UTC hours allowed for new pair entries.",
                ),
                "min_expected_move_pct": HyperParam.floating(
                    "min_expected_move_pct",
                    default=0.0015,
                    low=0.0,
                    high=0.20,
                    tunable=False,
                ),
            }
        )
        return schema

    def __init__(
        self,
        bars: Any,
        events: Any,
        *args: Any,
        allowed_session_utc_hours: str = "0,1,8,9,13,14,15,20,21",
        min_expected_move_pct: float = 0.0015,
        **kwargs: Any,
    ) -> None:
        super().__init__(bars, events, *args, **kwargs)
        self.allowed_session_utc_hours = str(allowed_session_utc_hours or "").strip()
        self._allowed_hours = self._parse_hours(self.allowed_session_utc_hours)
        self.min_expected_move_pct = max(0.0, float(min_expected_move_pct))
        self._entry_gate_blocked = False
        self._last_session_hour: int | None = None
        self._last_expected_move_pct: float | None = None

    @staticmethod
    def _parse_hours(raw: str) -> frozenset[int] | None:
        token = str(raw or "").strip().lower()
        if not token or token in {"*", "all", "any"}:
            return None
        hours: set[int] = set()
        for part in token.replace(";", ",").split(","):
            if not part.strip():
                continue
            try:
                hour = int(part.strip())
            except ValueError:
                continue
            if 0 <= hour <= 23:
                hours.add(hour)
        return frozenset(hours) if hours else None

    @staticmethod
    def _event_hour_utc(event_time: Any) -> int | None:
        if isinstance(event_time, datetime):
            value = event_time
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return int(value.astimezone(UTC).hour)
        hour = getattr(event_time, "hour", None)
        if hour is not None:
            try:
                return int(hour)
            except Exception:
                return None
        return None

    def _session_allowed(self, event_time: Any) -> bool:
        hour = self._event_hour_utc(event_time)
        self._last_session_hour = hour
        if hour is None or self._allowed_hours is None:
            return True
        return int(hour) in self._allowed_hours

    def _expected_move_pct(self, zscore: float, beta: float) -> float | None:
        close_x, close_y = self._resolve_pair_closes()
        if close_x is None or close_y is None:
            return None
        spread_values = list(self._spread_history)
        if len(spread_values) < 4:
            return None
        spread_std = stdev(spread_values)
        if spread_std <= 1e-12:
            return None
        expected_spread_move = max(0.0, abs(float(zscore)) - float(self.exit_z)) * spread_std
        pair_notional = (abs(float(close_x)) + abs(float(beta)) * abs(float(close_y))) / 2.0
        if pair_notional <= 1e-12:
            return None
        return expected_spread_move / pair_notional

    def _entry_gate_passed(self, event_time: Any, zscore: float, beta: float) -> bool:
        if not self._session_allowed(event_time):
            self._last_expected_move_pct = None
            return False
        expected = self._expected_move_pct(zscore, beta)
        self._last_expected_move_pct = expected
        if expected is None:
            return self.min_expected_move_pct <= 0.0
        return expected >= self.min_expected_move_pct

    def _reset_rejected_entry(self) -> None:
        self._mode = "OUT"
        self._bars_in_position = 0
        self._entry_x_price = None
        self._entry_y_price = None
        self._entry_beta = None
        self._extreme_side = "NONE"

    def _emit_signal(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        metadata: dict[str, Any],
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        enriched = dict(metadata)
        enriched.update(
            {
                "strategy": "SessionFilteredPairCarryStrategy",
                "allowed_session_utc_hours": self.allowed_session_utc_hours,
                "session_hour_utc": self._last_session_hour,
                "min_expected_move_pct": float(self.min_expected_move_pct),
                "expected_move_pct": self._last_expected_move_pct,
            }
        )
        super()._emit_signal(
            symbol,
            event_time,
            signal_type,
            enriched,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _emit_pair_entry(
        self, event_time: Any, mode: str, zscore: float, beta: float, corr: float | None
    ) -> None:
        if not self._entry_gate_passed(event_time, zscore, beta):
            self._entry_gate_blocked = True
            return
        super()._emit_pair_entry(event_time, mode, zscore, beta, corr)

    def calculate_signals(self, event: Any) -> None:
        self._entry_gate_blocked = False
        super().calculate_signals(event)
        if self._entry_gate_blocked:
            self._reset_rejected_entry()


__all__ = ["SessionFilteredPairCarryStrategy"]
