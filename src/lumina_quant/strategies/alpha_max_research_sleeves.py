"""Research-only native-timeframe adapters for the alpha-max current nodes.

The classes in this module are deliberately thin adapters over the existing
strategy implementations.  They own native-bar causality (completed 1d/4h bars,
forming-bucket exclusion, explicit boundary finalization, near-high atomic
cross-section barriers, and indicator-only research capsules) while delegating
all indicator, score, position, sizing, and emission formulas to the inherited
original classes.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar
from collections.abc import Mapping

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.market_data import timeframe_to_milliseconds
from lumina_quant.strategies.aggressive_return_alpha_sleeves import FundingHarvestCarryStrategy
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.tuning import HyperParam

CANDIDATE_SYMBOLS: tuple[str, ...] = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "TONUSDT",
    "TRXUSDT",
    "XRPUSDT",
)

CANONICAL_ALPHA_MAX_COMPONENT_NODES: dict[str, dict[str, Any]] = {
    "ResearchOnlyFourHourFundingHarvestCarryStrategy": {
        "row_id": "component_carry_1x",
        "timeframe": "4h",
        "candidate_symbols": CANDIDATE_SYMBOLS,
        "params": {
            "add_alloc_fraction": 0.5,
            "add_step_atr": 1.0,
            "allow_short": True,
            "atr_period": 14,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "funding_window": 6,
            "max_adds": 2,
            "max_hold_bars": 180,
            "max_order_value": 5000.0,
            "min_price": 0.1,
            "no_fight_roc": 0.06,
            "no_fight_roc_period": 4,
            "target_allocation": 0.3,
            "target_vol": 0.03,
            "trail_atr_mult": 4.0,
            "vol_window": 36,
        },
    },
    "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": {
        "row_id": "component_near_high_1x",
        "timeframe": "1d",
        "candidate_symbols": CANDIDATE_SYMBOLS,
        "params": {
            "allow_short": True,
            "base_allocation": 0.2,
            "high_lookback_bars": 364,
            "max_hold_bars": 0,
            "max_order_value": 400.0,
            "max_symbol_exposure_pct": 0.4,
            "min_history_bars": 60,
            "min_hold_bars": 7,
            "min_price": 0.1,
            "min_symbols": 5,
            "quantile_pct": 0.25,
            "rebalance_bars": 7,
            "stop_loss_pct": 0.1,
            "target_gross_exposure": 1.0,
            "target_vol": 0.2,
            "vol_window": 20,
        },
    },
    "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": {
        "row_id": "component_trend_1x",
        "timeframe": "1d",
        "candidate_symbols": CANDIDATE_SYMBOLS,
        "params": {
            "adx_min": 20.0,
            "adx_period": 14,
            "allow_short": True,
            "cooldown_bars": 4,
            "efficiency_period": 20,
            "max_hold_bars": 2000,
            "max_order_value": 400.0,
            "min_efficiency": 0.3,
            "min_hold_bars": 36,
            "min_price": 0.1,
            "target_allocation": 0.2,
            "target_vol": 0.2,
            "tsmom_long": 84,
            "tsmom_mid": 56,
            "tsmom_short": 28,
            "vol_persist_fast": 16,
            "vol_persist_max": 1.5,
            "vol_persist_slow": 64,
            "vol_window": 56,
        },
    },
}

_ONE_DAY_MS = int(timeframe_to_milliseconds("1d"))
_FOUR_HOUR_MS = int(timeframe_to_milliseconds("4h"))
_FUNDING_MAX_AGE_MS = 8 * 60 * 60 * 1000
_COOLDOWN_SATISFIED = 1 << 30
_NATIVE_FINALIZATION_ROLLBACK_KIND = "alpha_max.native_finalization_rollback.v1"


def _canonical_params(class_name: str) -> dict[str, Any]:
    return copy.deepcopy(CANONICAL_ALPHA_MAX_COMPONENT_NODES[class_name]["params"])


def _schema_with_defaults(
    schema: Mapping[str, HyperParam], defaults: Mapping[str, Any]
) -> dict[str, HyperParam]:
    out: dict[str, HyperParam] = {}
    for name, spec in schema.items():
        if name in defaults:
            out[name] = replace(spec, default=copy.deepcopy(defaults[name]))
        else:
            out[name] = spec
    return out


def _timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    if isinstance(value, (int, float)):
        raw = int(float(value))
        return raw if abs(raw) >= 100_000_000_000 else raw * 1000
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    dt = dt.astimezone(UTC) if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _utc_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).replace(tzinfo=None)


def _bar_time(bar: Any) -> Any:
    if isinstance(bar, Mapping):
        return bar.get("time") or bar.get("datetime")
    if isinstance(bar, (tuple, list)) and bar:
        return bar[0]
    return getattr(bar, "time", None)


def _bar_tuple(bar: Any) -> tuple[Any, float, float, float, float, float] | None:
    if isinstance(bar, Mapping):
        t = bar.get("time") or bar.get("datetime")
        close = safe_float(bar.get("close"))
        if t is None or close is None:
            return None
        open_ = safe_float(bar.get("open"))
        high = safe_float(bar.get("high"))
        low = safe_float(bar.get("low"))
        vol = safe_float(bar.get("volume"))
        return (
            t,
            open_ if open_ is not None else close,
            high if high is not None else close,
            low if low is not None else close,
            close,
            vol if vol is not None else 0.0,
        )
    if isinstance(bar, (tuple, list)) and len(bar) >= 5:
        close = safe_float(bar[4])
        if close is None:
            return None
        open_ = safe_float(bar[1]) if len(bar) > 1 else None
        high = safe_float(bar[2]) if len(bar) > 2 else None
        low = safe_float(bar[3]) if len(bar) > 3 else None
        vol = safe_float(bar[5]) if len(bar) > 5 else None
        return (
            bar[0],
            open_ if open_ is not None else close,
            high if high is not None else close,
            low if low is not None else close,
            close,
            vol if vol is not None else 0.0,
        )
    close = safe_float(getattr(bar, "close", None))
    t = getattr(bar, "time", None)
    if t is None or close is None:
        return None
    open_ = safe_float(getattr(bar, "open", None))
    high = safe_float(getattr(bar, "high", None))
    low = safe_float(getattr(bar, "low", None))
    vol = safe_float(getattr(bar, "volume", None))
    return (
        t,
        open_ if open_ is not None else close,
        high if high is not None else close,
        low if low is not None else close,
        close,
        vol if vol is not None else 0.0,
    )


def _bar_signature(bar: Any) -> tuple[Any, float, float, float, float, float]:
    parsed = _bar_tuple(bar)
    if parsed is None:
        raise ValueError("invalid_native_bar")
    return parsed


def _bucket_key(bar: Any) -> str:
    return time_key(_bar_time(bar))


def _event_watermark_ms(event: Any) -> int | None:
    for name in ("event_time_watermark_ms", "watermark_ms", "raw_watermark_ms"):
        value = getattr(event, name, None)
        if value is not None:
            return _timestamp_ms(value)
    return _timestamp_ms(getattr(event, "time", None))


def _get_completed_bars(
    aggregator: Any, symbol: str, timeframe: str, *, include_final: bool
) -> list[Any]:
    bars = list(aggregator.get_bars(symbol=symbol, timeframe=timeframe, n=100_000) or [])
    if include_final:
        return bars
    return bars[:-1] if len(bars) >= 2 else []


def _window_event_for_bars(time_value: Any, bars_by_symbol: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET_WINDOW",
        time=time_value,
        bars_1s={symbol: (_bar_signature(bar),) for symbol, bar in bars_by_symbol.items()},
    )


def _point_value_and_source_ms(point: Any) -> tuple[float | None, int | None]:
    if point is None:
        return None, None
    if isinstance(point, tuple):
        value = point[0] if point else None
        source = point[1] if len(point) > 1 else None
        return safe_float(value), _timestamp_ms(source)
    if isinstance(point, Mapping):
        value = point.get("value", point.get("v"))
        source = point.get("source_timestamp_ms", point.get("timestamp_ms", point.get("time")))
        return safe_float(value), _timestamp_ms(source)
    value = getattr(point, "value", point)
    source = getattr(point, "source_timestamp_ms", getattr(point, "timestamp_ms", None))
    return safe_float(value), _timestamp_ms(source)


class _NativeCompletedAdapterMixin:
    """Completed-native-bar gate shared by trend/carry research adapters."""

    native_timeframe: ClassVar[str]
    native_timeframe_ms: ClassVar[int]
    minimum_completed_bars: ClassVar[int]
    uses_timeframe_aggregator: ClassVar[bool] = True
    preferred_contract: ClassVar[str] = "market_window"
    research_only: ClassVar[bool] = True

    def _init_native_adapter_state(self) -> None:
        self.required_native_timeframes = (self.native_timeframe,)
        self.required_timeframes = (self.native_timeframe,)
        self._alpha_max_last_completed_native_key_by_symbol: dict[str, str] = {}
        self._alpha_max_completed_native_keys: set[tuple[str, str]] = set()
        self._alpha_max_completed_native_count_by_symbol: dict[str, int] = {}
        self._alpha_max_bound_aggregator: Any = None
        self._alpha_max_partial_bucket_error: str | None = None
        self._alpha_max_last_completed_cutoff: int | None = None

    @classmethod
    def canonical_component_params(cls) -> dict[str, Any]:
        return _canonical_params(cls.__name__)

    @classmethod
    def canonical_component_node(cls) -> dict[str, Any]:
        return copy.deepcopy(CANONICAL_ALPHA_MAX_COMPONENT_NODES[cls.__name__])

    def _completed_enough(self, bar: Any, watermark_ms: int | None) -> bool:
        start_ms = _timestamp_ms(_bar_time(bar))
        return (
            start_ms is not None
            and watermark_ms is not None
            and start_ms + self.native_timeframe_ms <= watermark_ms
        )

    def _ingest_completed_native_bar(self, symbol: str, bar: Any) -> bool:
        key = _bucket_key(bar)
        if not key or (symbol, key) in self._alpha_max_completed_native_keys:
            return False
        event = _window_event_for_bars(_bar_time(bar), {symbol: bar})
        super().calculate_signals_window(event, None)  # type: ignore[misc]
        self._alpha_max_completed_native_keys.add((symbol, key))
        self._alpha_max_completed_native_count_by_symbol[symbol] = (
            self._alpha_max_completed_native_count_by_symbol.get(symbol, 0) + 1
        )
        self._alpha_max_last_completed_native_key_by_symbol[symbol] = key
        return True

    def _process_from_aggregator(
        self, aggregator: Any, watermark_ms: int | None, *, include_final: bool
    ) -> int:
        self._alpha_max_bound_aggregator = aggregator
        cutoff = None if watermark_ms is None else int(watermark_ms) // self.native_timeframe_ms
        if (
            not include_final
            and cutoff is not None
            and self._alpha_max_last_completed_cutoff is not None
            and cutoff <= self._alpha_max_last_completed_cutoff
        ):
            return 0
        processed = 0
        for symbol in list(getattr(self, "symbol_list", []) or []):
            for bar in _get_completed_bars(
                aggregator, symbol, self.native_timeframe, include_final=include_final
            ):
                if self._completed_enough(bar, watermark_ms) and self._ingest_completed_native_bar(
                    symbol, bar
                ):
                    processed += 1
        if not include_final and cutoff is not None:
            self._alpha_max_last_completed_cutoff = cutoff
        return processed

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        if aggregator is not None:
            self._process_from_aggregator(
                aggregator, _event_watermark_ms(event), include_final=False
            )
            return
        # Direct native-bar test/harness path requires an explicit completed-native
        # contract or a watermark that proves the native bucket is complete; ordinary
        # raw/forming market-window rows must never be silently promoted.
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        explicit_completed = bool(getattr(event, "completed_native_bars", False))
        watermark_ms = _event_watermark_ms(event)
        for symbol in list(getattr(self, "symbol_list", []) or []):
            rows = list(bars_1s.get(symbol) or [])
            if rows and (explicit_completed or self._completed_enough(rows[-1], watermark_ms)):
                self._ingest_completed_native_bar(symbol, rows[-1])

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)

    def finalize_completed_native_buckets(self, watermark: Any) -> int:
        watermark_ms = _timestamp_ms(watermark)
        if watermark_ms is None:
            raise ValueError("invalid_native_watermark")
        aggregator = self._alpha_max_bound_aggregator
        if aggregator is None:
            return 0
        processed = self._process_from_aggregator(aggregator, watermark_ms, include_final=True)
        # If a visible final bucket is not yet temporally complete, remember the
        # fail-closed state so callers cannot accidentally serialize through it.
        for symbol in list(getattr(self, "symbol_list", []) or []):
            bars = _get_completed_bars(
                aggregator, symbol, self.native_timeframe, include_final=True
            )
            if bars and not self._completed_enough(bars[-1], watermark_ms):
                self._alpha_max_partial_bucket_error = "partial_native_bucket"
        return processed

    def validate_research_warmup_ready(self) -> None:
        missing = [
            symbol
            for symbol in list(getattr(self, "symbol_list", []) or [])
            if self._alpha_max_completed_native_count_by_symbol.get(symbol, 0)
            < self.minimum_completed_bars
        ]
        if missing:
            raise ValueError(
                f"insufficient_research_warmup_history:{self.native_timeframe}:"
                + ",".join(sorted(missing))
            )

    def _capsule_prefix(self) -> dict[str, Any]:
        if self._alpha_max_partial_bucket_error:
            raise ValueError(self._alpha_max_partial_bucket_error)
        return {
            "adapter_class": type(self).__name__,
            "native_timeframe": self.native_timeframe,
            "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
            "completed_native_count_by_symbol": dict(
                self._alpha_max_completed_native_count_by_symbol
            ),
            "last_completed_native_key_by_symbol": dict(
                self._alpha_max_last_completed_native_key_by_symbol
            ),
        }

    def _restore_native_capsule_prefix(self, capsule: Mapping[str, Any]) -> None:
        if str(capsule.get("adapter_class")) != type(self).__name__:
            raise ValueError("research_indicator_capsule_class_mismatch")
        raw_keys = capsule.get("completed_native_keys", [])
        self._alpha_max_completed_native_keys = {
            (str(item[0]), str(item[1]))
            for item in raw_keys
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        raw_counts = capsule.get("completed_native_count_by_symbol", {})
        self._alpha_max_completed_native_count_by_symbol = (
            {str(k): int(v) for k, v in raw_counts.items()}
            if isinstance(raw_counts, Mapping)
            else {}
        )
        raw_last = capsule.get("last_completed_native_key_by_symbol", {})
        self._alpha_max_last_completed_native_key_by_symbol = (
            {str(k): str(v) for k, v in raw_last.items()} if isinstance(raw_last, Mapping) else {}
        )

    def _native_finalization_auxiliary_state(self) -> dict[str, Any]:
        return {}

    def get_native_finalization_evidence(self) -> dict[str, Any]:
        """Expose the exact completed-native coverage sealed at a score boundary."""
        if self._alpha_max_partial_bucket_error:
            raise ValueError(self._alpha_max_partial_bucket_error)
        return copy.deepcopy(
            {
                "adapter_class": type(self).__name__,
                "native_timeframe": self.native_timeframe,
                "barrier_mode": "none",
                "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
                "completed_native_count_by_symbol": dict(
                    sorted(self._alpha_max_completed_native_count_by_symbol.items())
                ),
                "last_completed_native_key_by_symbol": dict(
                    sorted(self._alpha_max_last_completed_native_key_by_symbol.items())
                ),
                "barrier_pending_keys": [],
                "barrier_closed_keys": [],
                "barrier_symbol_coverage": {},
                "failed_native_keys": {},
                "partial_bucket_error": None,
            }
        )

    def _restore_native_finalization_auxiliary_state(self, state: Mapping[str, Any]) -> None:
        if dict(state):
            raise ValueError("invalid_native_finalization_rollback_state")

    def get_native_finalization_rollback_state(self) -> dict[str, Any]:
        """Return a deep, deterministic snapshot of every finalizer-owned mutation."""
        return copy.deepcopy(
            {
                "kind": _NATIVE_FINALIZATION_ROLLBACK_KIND,
                "adapter_class": type(self).__name__,
                "native_timeframe": self.native_timeframe,
                "base_state": self.get_state(),
                "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
                "completed_native_count_by_symbol": dict(
                    sorted(self._alpha_max_completed_native_count_by_symbol.items())
                ),
                "last_completed_native_key_by_symbol": dict(
                    sorted(self._alpha_max_last_completed_native_key_by_symbol.items())
                ),
                "partial_bucket_error": self._alpha_max_partial_bucket_error,
                "auxiliary_state": self._native_finalization_auxiliary_state(),
            }
        )

    def set_native_finalization_rollback_state(self, snapshot: Mapping[str, Any]) -> None:
        required_keys = {
            "kind",
            "adapter_class",
            "native_timeframe",
            "base_state",
            "completed_native_keys",
            "completed_native_count_by_symbol",
            "last_completed_native_key_by_symbol",
            "partial_bucket_error",
            "auxiliary_state",
        }
        if not isinstance(snapshot, Mapping) or set(snapshot) != required_keys:
            raise ValueError("invalid_native_finalization_rollback_state")
        if (
            snapshot.get("kind") != _NATIVE_FINALIZATION_ROLLBACK_KIND
            or snapshot.get("adapter_class") != type(self).__name__
            or snapshot.get("native_timeframe") != self.native_timeframe
        ):
            raise ValueError("invalid_native_finalization_rollback_state")
        base_state = snapshot.get("base_state")
        raw_keys = snapshot.get("completed_native_keys")
        raw_counts = snapshot.get("completed_native_count_by_symbol")
        raw_last = snapshot.get("last_completed_native_key_by_symbol")
        partial_error = snapshot.get("partial_bucket_error")
        auxiliary_state = snapshot.get("auxiliary_state")
        if (
            not isinstance(base_state, Mapping)
            or not isinstance(raw_keys, (list, tuple))
            or not isinstance(raw_counts, Mapping)
            or not isinstance(raw_last, Mapping)
            or (
                partial_error is not None
                and (not isinstance(partial_error, str) or not partial_error)
            )
            or not isinstance(auxiliary_state, Mapping)
        ):
            raise ValueError("invalid_native_finalization_rollback_state")
        admitted_symbols = {str(symbol) for symbol in (getattr(self, "symbol_list", []) or [])}
        completed: set[tuple[str, str]] = set()
        for item in raw_keys:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or item[0] not in admitted_symbols
                or not isinstance(item[1], str)
                or not item[1]
            ):
                raise ValueError("invalid_native_finalization_rollback_state")
            completed.add((item[0], item[1]))
        if len(completed) != len(raw_keys):
            raise ValueError("invalid_native_finalization_rollback_state")
        counts: dict[str, int] = {}
        for symbol, count in raw_counts.items():
            if (
                not isinstance(symbol, str)
                or symbol not in admitted_symbols
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError("invalid_native_finalization_rollback_state")
            counts[symbol] = count
        expected_counts = {
            symbol: sum(1 for completed_symbol, _key in completed if completed_symbol == symbol)
            for symbol in admitted_symbols
        }
        expected_counts = {symbol: count for symbol, count in expected_counts.items() if count}
        if counts != expected_counts:
            raise ValueError("invalid_native_finalization_rollback_state")
        last: dict[str, str] = {}
        for symbol, key in raw_last.items():
            if (
                not isinstance(symbol, str)
                or symbol not in counts
                or not isinstance(key, str)
                or (symbol, key) not in completed
            ):
                raise ValueError("invalid_native_finalization_rollback_state")
            last[symbol] = key
        if set(last) != set(counts):
            raise ValueError("invalid_native_finalization_rollback_state")

        self.set_state(copy.deepcopy(dict(base_state)))
        self._alpha_max_completed_native_keys = completed
        self._alpha_max_completed_native_count_by_symbol = counts
        self._alpha_max_last_completed_native_key_by_symbol = last
        self._alpha_max_partial_bucket_error = partial_error
        self._restore_native_finalization_auxiliary_state(copy.deepcopy(dict(auxiliary_state)))


@register(
    "strategy",
    "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
    interface="event_driven",
)
class ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(
    _NativeCompletedAdapterMixin, LowTurnoverTrendPersistenceStrategy
):
    """Daily completed-bar research adapter for low-turnover trend persistence."""

    strategy_name = "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy"
    strategy_id = "research_only_daily_low_turnover_trend_persistence"
    native_timeframe = "1d"
    native_timeframe_ms = _ONE_DAY_MS
    minimum_completed_bars = 366
    decision_cadence_seconds = 86400

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return _schema_with_defaults(
            LowTurnoverTrendPersistenceStrategy.get_param_schema(), cls.canonical_component_params()
        )

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        merged = {**self.canonical_component_params(), **params}
        super().__init__(bars, events, **merged)
        self._init_native_adapter_state()

    def get_research_indicator_state(self) -> dict[str, Any]:
        state = self.get_state()
        symbol_state: dict[str, Any] = {}
        for symbol, payload in dict(state.get("symbol_state") or {}).items():
            item = dict(payload)
            item.update(
                {
                    "mode": "OUT",
                    "entry_price": None,
                    "bars_held": 0,
                    "bars_since_exit": _COOLDOWN_SATISFIED,
                    "score": None,
                }
            )
            symbol_state[str(symbol)] = item
        return {
            **self._capsule_prefix(),
            "recent_times": list(state.get("recent_times") or []),
            "symbol_state": symbol_state,
        }

    def set_research_indicator_state(self, capsule: Mapping[str, Any]) -> None:
        self._restore_native_capsule_prefix(capsule)
        self.set_state(
            {
                "recent_times": list(capsule.get("recent_times") or []),
                "symbol_state": dict(capsule.get("symbol_state") or {}),
            }
        )


class _NearHighBarrierError(ValueError):
    pass


@register(
    "strategy",
    "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
    interface="event_driven",
)
class ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
    CrossSectionalNearHighAnchoringStrategy
):
    """Daily atomic-cross-section research adapter for near-high anchoring."""

    strategy_name = "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"
    strategy_id = "research_only_daily_cross_sectional_near_high_anchoring"
    native_timeframe = "1d"
    native_timeframe_ms = _ONE_DAY_MS
    minimum_completed_bars = 366
    uses_timeframe_aggregator = True
    preferred_contract = "market_window"
    research_only = True
    _chunk_state_key = "_alpha_max_chunk_state"

    @classmethod
    def canonical_component_params(cls) -> dict[str, Any]:
        return _canonical_params(cls.__name__)

    @classmethod
    def canonical_component_node(cls) -> dict[str, Any]:
        return copy.deepcopy(CANONICAL_ALPHA_MAX_COMPONENT_NODES[cls.__name__])

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return _schema_with_defaults(
            CrossSectionalNearHighAnchoringStrategy.get_param_schema(),
            cls.canonical_component_params(),
        )

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        admitted = tuple(
            sorted(
                str(s)
                for s in (
                    params.pop("admitted_symbols", None) or getattr(bars, "symbol_list", []) or []
                )
            )
        )
        if admitted:
            try:
                bars.symbol_list = list(admitted)
            except Exception:
                pass
        merged = {**self.canonical_component_params(), **params}
        super().__init__(bars, events, **merged)
        if admitted:
            self.symbol_list = list(admitted)
            self._state = {
                symbol: self._state[symbol] for symbol in admitted if symbol in self._state
            }
        self.required_native_timeframes = (self.native_timeframe,)
        self.required_timeframes = (self.native_timeframe,)
        self._alpha_max_admitted_symbols = tuple(self.symbol_list)
        self._alpha_max_barrier_pending: dict[
            str, dict[str, tuple[Any, float, float, float, float, float]]
        ] = {}
        self._alpha_max_barrier_closed: set[str] = set()
        self._alpha_max_failed_native_keys: dict[str, str] = {}
        self._alpha_max_bound_aggregator: Any = None
        self._alpha_max_partial_bucket_error: str | None = None
        self._alpha_max_completed_native_keys: set[tuple[str, str]] = set()
        self._alpha_max_completed_native_count_by_symbol: dict[str, int] = {}
        self._alpha_max_last_completed_native_key_by_symbol: dict[str, str] = {}

    def get_state(self) -> dict[str, Any]:
        """Capture base strategy and barrier state without binding the aggregator."""
        state = super().get_state()
        state[self._chunk_state_key] = {
            "version": 1,
            "adapter_class": type(self).__name__,
            "native_timeframe": self.native_timeframe,
            "admitted_symbols": list(self._alpha_max_admitted_symbols),
            "barrier_pending": {
                key: {
                    symbol: tuple(copy.deepcopy(signature))
                    for symbol, signature in sorted(pending.items())
                }
                for key, pending in sorted(self._alpha_max_barrier_pending.items())
            },
            "barrier_closed": sorted(self._alpha_max_barrier_closed),
            "failed_native_keys": dict(sorted(self._alpha_max_failed_native_keys.items())),
            "partial_bucket_error": self._alpha_max_partial_bucket_error,
            "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
            "completed_native_count_by_symbol": dict(
                sorted(self._alpha_max_completed_native_count_by_symbol.items())
            ),
            "last_completed_native_key_by_symbol": dict(
                sorted(self._alpha_max_last_completed_native_key_by_symbol.items())
            ),
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a full chunk snapshot while leaving the new aggregator unbound."""
        if not isinstance(state, dict):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_chunk = state.get(self._chunk_state_key)
        if raw_chunk is None:
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        required_keys = {
            "version",
            "adapter_class",
            "native_timeframe",
            "admitted_symbols",
            "barrier_pending",
            "barrier_closed",
            "failed_native_keys",
            "partial_bucket_error",
            "completed_native_keys",
            "completed_native_count_by_symbol",
            "last_completed_native_key_by_symbol",
        }
        if not isinstance(raw_chunk, Mapping) or set(raw_chunk) != required_keys:
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        if (
            raw_chunk.get("version") != 1
            or raw_chunk.get("adapter_class") != type(self).__name__
            or raw_chunk.get("native_timeframe") != self.native_timeframe
            or tuple(raw_chunk.get("admitted_symbols") or ()) != self._alpha_max_admitted_symbols
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_pending = raw_chunk.get("barrier_pending")
        if not isinstance(raw_pending, Mapping):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        pending_state: dict[str, dict[str, tuple[Any, float, float, float, float, float]]] = {}
        for raw_key, raw_symbols in sorted(raw_pending.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, str) or not raw_key or not isinstance(raw_symbols, Mapping):
                raise ValueError("invalid_alpha_max_near_high_chunk_state")
            symbol_state: dict[str, tuple[Any, float, float, float, float, float]] = {}
            for raw_symbol, raw_signature in sorted(
                raw_symbols.items(), key=lambda item: str(item[0])
            ):
                if (
                    not isinstance(raw_symbol, str)
                    or raw_symbol not in self._alpha_max_admitted_symbols
                    or not isinstance(raw_signature, (list, tuple))
                    or len(raw_signature) != 6
                ):
                    raise ValueError("invalid_alpha_max_near_high_chunk_state")
                try:
                    signature = _bar_signature(raw_signature)
                except ValueError:
                    raise ValueError("invalid_alpha_max_near_high_chunk_state") from None
                if _bucket_key(signature) != raw_key:
                    raise ValueError("invalid_alpha_max_near_high_chunk_state")
                symbol_state[raw_symbol] = signature
            if not symbol_state:
                raise ValueError("invalid_alpha_max_near_high_chunk_state")
            pending_state[raw_key] = symbol_state

        raw_closed = raw_chunk.get("barrier_closed")
        if not isinstance(raw_closed, (list, tuple)) or any(
            not isinstance(key, str) or not key for key in raw_closed
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        closed_state = set(raw_closed)
        if len(closed_state) != len(raw_closed) or any(
            key not in pending_state
            or set(pending_state[key]) != set(self._alpha_max_admitted_symbols)
            for key in closed_state
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_failed = raw_chunk.get("failed_native_keys")
        if not isinstance(raw_failed, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(reason, str) or not reason
            for key, reason in raw_failed.items()
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        failed_state = {str(key): str(reason) for key, reason in sorted(raw_failed.items())}

        partial_error = raw_chunk.get("partial_bucket_error")
        if partial_error is not None and (not isinstance(partial_error, str) or not partial_error):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_completed = raw_chunk.get("completed_native_keys")
        if not isinstance(raw_completed, (list, tuple)):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        completed_state: set[tuple[str, str]] = set()
        for item in raw_completed:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or item[0] not in self._alpha_max_admitted_symbols
                or not isinstance(item[1], str)
                or not item[1]
            ):
                raise ValueError("invalid_alpha_max_near_high_chunk_state")
            completed_state.add((item[0], item[1]))
        if (
            len(completed_state) != len(raw_completed)
            or not {
                (symbol, key) for key in closed_state for symbol in self._alpha_max_admitted_symbols
            }
            <= completed_state
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_counts = raw_chunk.get("completed_native_count_by_symbol")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        count_state: dict[str, int] = {}
        for symbol, count in raw_counts.items():
            if (
                not isinstance(symbol, str)
                or symbol not in self._alpha_max_admitted_symbols
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("invalid_alpha_max_near_high_chunk_state")
            count_state[symbol] = count
        expected_counts = {
            symbol: sum(1 for completed_symbol, _ in completed_state if completed_symbol == symbol)
            for symbol in self._alpha_max_admitted_symbols
        }
        expected_counts = {symbol: count for symbol, count in expected_counts.items() if count}
        if count_state != expected_counts:
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        raw_last = raw_chunk.get("last_completed_native_key_by_symbol")
        if not isinstance(raw_last, Mapping) or any(
            not isinstance(symbol, str)
            or symbol not in self._alpha_max_admitted_symbols
            or not isinstance(key, str)
            or (symbol, key) not in completed_state
            for symbol, key in raw_last.items()
        ):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")
        last_state = {str(symbol): str(key) for symbol, key in sorted(raw_last.items())}
        if set(last_state) != set(expected_counts):
            raise ValueError("invalid_alpha_max_near_high_chunk_state")

        super().set_state(state)
        self._alpha_max_barrier_pending = pending_state
        self._alpha_max_barrier_closed = closed_state
        self._alpha_max_failed_native_keys = failed_state
        self._alpha_max_partial_bucket_error = partial_error
        self._alpha_max_completed_native_keys = completed_state
        self._alpha_max_completed_native_count_by_symbol = count_state
        self._alpha_max_last_completed_native_key_by_symbol = last_state
        self._alpha_max_bound_aggregator = None

    def get_native_finalization_rollback_state(self) -> dict[str, Any]:
        """Capture base, barrier, failure, partial, and completed-native state."""
        return {
            "kind": _NATIVE_FINALIZATION_ROLLBACK_KIND,
            "adapter_class": type(self).__name__,
            "native_timeframe": self.native_timeframe,
            "state": copy.deepcopy(self.get_state()),
        }

    def get_native_finalization_evidence(self) -> dict[str, Any]:
        """Expose exact atomic-barrier and completed-native score-boundary coverage."""
        if self._alpha_max_partial_bucket_error:
            raise ValueError(self._alpha_max_partial_bucket_error)
        pending_keys = sorted(self._alpha_max_barrier_pending)
        return copy.deepcopy(
            {
                "adapter_class": type(self).__name__,
                "native_timeframe": self.native_timeframe,
                "barrier_mode": "atomic_cross_section",
                "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
                "completed_native_count_by_symbol": dict(
                    sorted(self._alpha_max_completed_native_count_by_symbol.items())
                ),
                "last_completed_native_key_by_symbol": dict(
                    sorted(self._alpha_max_last_completed_native_key_by_symbol.items())
                ),
                "barrier_pending_keys": pending_keys,
                "barrier_closed_keys": sorted(self._alpha_max_barrier_closed),
                "barrier_symbol_coverage": {
                    key: sorted(self._alpha_max_barrier_pending[key]) for key in pending_keys
                },
                "failed_native_keys": dict(sorted(self._alpha_max_failed_native_keys.items())),
                "partial_bucket_error": None,
            }
        )

    def set_native_finalization_rollback_state(self, snapshot: Mapping[str, Any]) -> None:
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != {"kind", "adapter_class", "native_timeframe", "state"}
            or snapshot.get("kind") != _NATIVE_FINALIZATION_ROLLBACK_KIND
            or snapshot.get("adapter_class") != type(self).__name__
            or snapshot.get("native_timeframe") != self.native_timeframe
            or not isinstance(snapshot.get("state"), Mapping)
        ):
            raise ValueError("invalid_native_finalization_rollback_state")
        bound_aggregator = self._alpha_max_bound_aggregator
        self.set_state(copy.deepcopy(dict(snapshot["state"])))
        self._alpha_max_bound_aggregator = bound_aggregator

    def _completed_enough(self, bar: Any, watermark_ms: int | None) -> bool:
        start_ms = _timestamp_ms(_bar_time(bar))
        return (
            start_ms is not None
            and watermark_ms is not None
            and start_ms + self.native_timeframe_ms <= watermark_ms
        )

    def _barrier_accept(self, symbol: str, bar: Any) -> bool:
        key = _bucket_key(bar)
        if not key:
            raise _NearHighBarrierError("invalid_near_high_completed_key")
        if key in self._alpha_max_failed_native_keys:
            raise _NearHighBarrierError(self._alpha_max_failed_native_keys[key])
        sig = _bar_signature(bar)
        pending = self._alpha_max_barrier_pending.setdefault(key, {})
        existing = pending.get(symbol)
        if existing is not None:
            if existing == sig:
                return False
            reason = f"conflicting_near_high_duplicate:{key}:{symbol}"
            self._alpha_max_failed_native_keys[key] = reason
            raise _NearHighBarrierError(reason)
        if key in self._alpha_max_barrier_closed:
            reason = f"conflicting_near_high_duplicate:{key}:{symbol}"
            self._alpha_max_failed_native_keys[key] = reason
            raise _NearHighBarrierError(reason)
        pending[symbol] = sig
        return True

    def _barrier_flush_if_complete(self, key: str) -> bool:
        if key in self._alpha_max_barrier_closed:
            return False
        pending = self._alpha_max_barrier_pending.get(key, {})
        missing = [s for s in self._alpha_max_admitted_symbols if s not in pending]
        if missing:
            return False
        ordered = {symbol: pending[symbol] for symbol in self._alpha_max_admitted_symbols}
        event = _window_event_for_bars(next(iter(ordered.values()))[0], ordered)
        super().calculate_signals_window(event, None)
        self._alpha_max_barrier_closed.add(key)
        for symbol in self._alpha_max_admitted_symbols:
            self._alpha_max_completed_native_keys.add((symbol, key))
            self._alpha_max_completed_native_count_by_symbol[symbol] = (
                self._alpha_max_completed_native_count_by_symbol.get(symbol, 0) + 1
            )
            self._alpha_max_last_completed_native_key_by_symbol[symbol] = key
        return True

    def _fail_missing_if_past(self, key: str) -> None:
        pending = self._alpha_max_barrier_pending.get(key, {})
        missing = [s for s in self._alpha_max_admitted_symbols if s not in pending]
        if missing:
            reason = "incomplete_near_high_cross_section:" + key + ":" + ",".join(missing)
            self._alpha_max_failed_native_keys[key] = reason
            raise _NearHighBarrierError(reason)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        watermark_ms = _event_watermark_ms(event)
        if aggregator is not None:
            self._alpha_max_bound_aggregator = aggregator
            accepted_keys: set[str] = set()
            for symbol in self._alpha_max_admitted_symbols:
                for bar in _get_completed_bars(
                    aggregator, symbol, self.native_timeframe, include_final=False
                ):
                    if self._completed_enough(bar, watermark_ms) and self._barrier_accept(
                        symbol, bar
                    ):
                        accepted_keys.add(_bucket_key(bar))
            for key in sorted(accepted_keys):
                self._barrier_flush_if_complete(key)
            for key, pending in sorted(self._alpha_max_barrier_pending.items()):
                if key in self._alpha_max_barrier_closed:
                    continue
                first = next(iter(pending.values()), None)
                start_ms = _timestamp_ms(first[0]) if first else _timestamp_ms(key)
                if start_ms is not None and start_ms + self.native_timeframe_ms <= watermark_ms:
                    self._fail_missing_if_past(key)
            return
        explicit_completed = bool(getattr(event, "completed_native_bars", False))
        accepted_keys = set()
        for symbol, rows in dict(getattr(event, "bars_1s", {}) or {}).items():
            if symbol not in self._alpha_max_admitted_symbols:
                continue
            row_list = list(rows or [])
            if row_list and (
                explicit_completed or self._completed_enough(row_list[-1], watermark_ms)
            ):
                key = _bucket_key(row_list[-1])
                if self._barrier_accept(str(symbol), row_list[-1]):
                    accepted_keys.add(key)
        for key in sorted(accepted_keys):
            self._barrier_flush_if_complete(key)
        for key, pending in sorted(self._alpha_max_barrier_pending.items()):
            if key in self._alpha_max_barrier_closed:
                continue
            first = next(iter(pending.values()), None)
            start_ms = _timestamp_ms(first[0]) if first else _timestamp_ms(key)
            if (
                watermark_ms is not None
                and start_ms is not None
                and start_ms + self.native_timeframe_ms <= watermark_ms
            ):
                self._fail_missing_if_past(key)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)

    def finalize_completed_native_buckets(self, watermark: Any) -> int:
        watermark_ms = _timestamp_ms(watermark)
        if watermark_ms is None:
            raise ValueError("invalid_native_watermark")
        processed = 0
        agg = self._alpha_max_bound_aggregator
        if agg is not None:
            for symbol in self._alpha_max_admitted_symbols:
                for bar in _get_completed_bars(
                    agg, symbol, self.native_timeframe, include_final=True
                ):
                    if self._completed_enough(bar, watermark_ms) and self._barrier_accept(
                        symbol, bar
                    ):
                        processed += int(self._barrier_flush_if_complete(_bucket_key(bar)))
                    elif not self._completed_enough(bar, watermark_ms):
                        self._alpha_max_partial_bucket_error = "partial_native_bucket"
        for key, pending in list(self._alpha_max_barrier_pending.items()):
            if key in self._alpha_max_barrier_closed:
                continue
            first = next(iter(pending.values()), None)
            start_ms = _timestamp_ms(first[0]) if first else _timestamp_ms(key)
            if start_ms is not None and start_ms + self.native_timeframe_ms <= watermark_ms:
                self._fail_missing_if_past(key)
        return processed

    def validate_research_warmup_ready(self) -> None:
        missing = [
            symbol
            for symbol in self._alpha_max_admitted_symbols
            if self._alpha_max_completed_native_count_by_symbol.get(symbol, 0)
            < self.minimum_completed_bars
        ]
        if missing:
            raise ValueError("insufficient_research_warmup_history:1d:" + ",".join(missing))

    def get_research_indicator_state(self) -> dict[str, Any]:
        if self._alpha_max_partial_bucket_error:
            raise ValueError(self._alpha_max_partial_bucket_error)
        state = self.get_state()
        symbol_state: dict[str, Any] = {}
        for symbol, payload in dict(state.get("symbol_state") or {}).items():
            item = dict(payload)
            item.update({"mode": "OUT", "entry_price": None, "bars_held": 0, "score": None})
            symbol_state[str(symbol)] = item
        return {
            "adapter_class": type(self).__name__,
            "native_timeframe": self.native_timeframe,
            "completed_native_keys": sorted(self._alpha_max_completed_native_keys),
            "completed_native_count_by_symbol": dict(
                self._alpha_max_completed_native_count_by_symbol
            ),
            "last_completed_native_key_by_symbol": dict(
                self._alpha_max_last_completed_native_key_by_symbol
            ),
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(state.get("recent_times") or []),
            "symbol_state": symbol_state,
        }

    def set_research_indicator_state(self, capsule: Mapping[str, Any]) -> None:
        if str(capsule.get("adapter_class")) != type(self).__name__:
            raise ValueError("research_indicator_capsule_class_mismatch")
        raw_keys = capsule.get("completed_native_keys", [])
        self._alpha_max_completed_native_keys = {
            (str(item[0]), str(item[1]))
            for item in raw_keys
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        raw_counts = capsule.get("completed_native_count_by_symbol", {})
        self._alpha_max_completed_native_count_by_symbol = (
            {str(k): int(v) for k, v in raw_counts.items()}
            if isinstance(raw_counts, Mapping)
            else {}
        )
        raw_last = capsule.get("last_completed_native_key_by_symbol", {})
        self._alpha_max_last_completed_native_key_by_symbol = (
            {str(k): str(v) for k, v in raw_last.items()} if isinstance(raw_last, Mapping) else {}
        )
        super().set_state(
            {
                "last_eval_time_key": capsule.get("last_eval_time_key", ""),
                "tick": capsule.get("tick", 0),
                "recent_times": list(capsule.get("recent_times") or []),
                "symbol_state": dict(capsule.get("symbol_state") or {}),
            }
        )


@register("strategy", "ResearchOnlyFourHourFundingHarvestCarryStrategy", interface="event_driven")
class ResearchOnlyFourHourFundingHarvestCarryStrategy(
    _NativeCompletedAdapterMixin, FundingHarvestCarryStrategy
):
    """Four-hour completed-bar carry adapter with causal as-of funding lookup."""

    strategy_name = "ResearchOnlyFourHourFundingHarvestCarryStrategy"
    strategy_id = "research_only_four_hour_funding_harvest_carry"
    native_timeframe = "4h"
    native_timeframe_ms = _FOUR_HOUR_MS
    minimum_completed_bars = 64
    decision_cadence_seconds = 14400
    required_features = ("funding_rate",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return _schema_with_defaults(
            FundingHarvestCarryStrategy.get_param_schema(), cls.canonical_component_params()
        )

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        merged = {**self.canonical_component_params(), **params}
        super().__init__(bars, events, **merged)
        self._init_native_adapter_state()
        self._alpha_max_active_funding_lookup: Any = None
        self._alpha_max_current_bar_close_ms: int | None = None
        self._alpha_max_last_funding_error: str | None = None

    def _ingest_completed_native_bar(self, symbol: str, bar: Any) -> bool:
        start_ms = _timestamp_ms(_bar_time(bar))
        self._alpha_max_current_bar_close_ms = (
            None if start_ms is None else start_ms + self.native_timeframe_ms
        )
        try:
            return super()._ingest_completed_native_bar(symbol, bar)
        finally:
            self._alpha_max_current_bar_close_ms = None

    def _funding_rate_asof(self, symbol: str, timestamp_ms: int) -> float | None:
        lookup = self._alpha_max_active_funding_lookup
        if lookup is None:
            self._alpha_max_last_funding_error = "ambient_feature_lookup_forbidden"
            return None
        getter = getattr(lookup, "get_latest_point", None)
        if not callable(getter):
            self._alpha_max_last_funding_error = "invalid_funding_lookup"
            return None
        try:
            point = getter(symbol, "funding_rate", timestamp_ms=int(timestamp_ms))
        except Exception as exc:
            self._alpha_max_last_funding_error = f"funding_lookup_error:{exc}"
            return None
        value, source_ms = _point_value_and_source_ms(point)
        if value is None or source_ms is None:
            self._alpha_max_last_funding_error = f"funding_coverage_missing:{symbol}:{timestamp_ms}"
            return None
        if source_ms > timestamp_ms:
            self._alpha_max_last_funding_error = (
                f"future_funding_point:{symbol}:{source_ms}>{timestamp_ms}"
            )
            return None
        if timestamp_ms - source_ms > _FUNDING_MAX_AGE_MS:
            self._alpha_max_last_funding_error = (
                f"stale_funding_point:{symbol}:{source_ms}:{timestamp_ms}"
            )
            return None
        return float(value)

    def _extract_feature(self, symbol: str, field: str) -> float | None:
        if field != "funding_rate" or self._alpha_max_current_bar_close_ms is None:
            return None
        return self._funding_rate_asof(symbol, self._alpha_max_current_bar_close_ms)

    def calculate_signals_context(self, context: Any) -> None:
        lookup = (
            context.get("feature_lookup")
            if isinstance(context, Mapping)
            else getattr(context, "feature_lookup", None)
        )
        aggregator = (
            context.get("aggregator")
            if isinstance(context, Mapping)
            else getattr(context, "aggregator", None)
        )
        event = (
            context.get("event")
            if isinstance(context, Mapping)
            else getattr(context, "event", None)
        )
        watermark = (
            context.get("watermark")
            if isinstance(context, Mapping)
            else getattr(context, "watermark", None)
        )
        if event is None:
            event = SimpleNamespace(type="MARKET_WINDOW", time=watermark, bars_1s={})
        self._alpha_max_active_funding_lookup = lookup
        self._alpha_max_last_funding_error = None
        try:
            self.calculate_signals_window(event, aggregator)
            if self._alpha_max_last_funding_error:
                raise ValueError(self._alpha_max_last_funding_error)
        finally:
            self._alpha_max_active_funding_lookup = None

    def _native_finalization_auxiliary_state(self) -> dict[str, Any]:
        if (
            self._alpha_max_active_funding_lookup is not None
            or self._alpha_max_current_bar_close_ms is not None
        ):
            raise ValueError("native_finalization_rollback_snapshot_not_quiescent")
        return {"last_funding_error": self._alpha_max_last_funding_error}

    def _restore_native_finalization_auxiliary_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"last_funding_error"}:
            raise ValueError("invalid_native_finalization_rollback_state")
        last_error = state.get("last_funding_error")
        if last_error is not None and (not isinstance(last_error, str) or not last_error):
            raise ValueError("invalid_native_finalization_rollback_state")
        self._alpha_max_active_funding_lookup = None
        self._alpha_max_current_bar_close_ms = None
        self._alpha_max_last_funding_error = last_error

    def get_research_indicator_state(self) -> dict[str, Any]:
        state = self.get_state()
        symbol_state: dict[str, Any] = {}
        for symbol, payload in dict(state.get("symbol_state") or {}).items():
            item = dict(payload)
            item.update(
                {
                    "mode": "OUT",
                    "entry_price": None,
                    "stop_price": None,
                    "high_watermark": None,
                    "low_watermark": None,
                    "adds": 0,
                    "last_add_price": None,
                    "bars_held": 0,
                }
            )
            symbol_state[str(symbol)] = item
        return {
            **self._capsule_prefix(),
            "symbol_state": symbol_state,
            "funding": state.get("funding", {}),
        }

    def set_research_indicator_state(self, capsule: Mapping[str, Any]) -> None:
        self._restore_native_capsule_prefix(capsule)
        self.set_state(
            {
                "symbol_state": dict(capsule.get("symbol_state") or {}),
                "funding": dict(capsule.get("funding") or {}),
            }
        )


__all__ = [
    "CANDIDATE_SYMBOLS",
    "CANONICAL_ALPHA_MAX_COMPONENT_NODES",
    "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy",
    "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy",
    "ResearchOnlyFourHourFundingHarvestCarryStrategy",
]
