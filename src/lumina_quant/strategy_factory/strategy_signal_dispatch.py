"""Strategy-signal dispatch helpers for research runner orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds

StrategySignalHandler = Callable[
    [dict[str, Any], dict[str, np.ndarray], Sequence[str], int, np.ndarray, dict[str, Any]],
    None,
]


class StrategySignalDispatchError(RuntimeError):
    """Actual-engine strategy evaluation cannot safely produce a signal."""

    def __init__(self, strategy_class: str, detail: str) -> None:
        self.strategy_class = strategy_class or "<missing>"
        self.detail = detail
        super().__init__(f"strategy signal dispatch failed for {self.strategy_class}: {detail}")


# Optional route for candidates whose ``strategy_class`` has no bespoke handler:
# ``(strategy_class, params, aligned, symbols) -> exposures | None``.  In non-strict
# dispatch, returning ``None`` (or raising) falls back to the generic proxy, which is
# always labelled in ``meta`` so proxy rows cannot be silently attributed to a lane.
UnmappedStrategyRouter = Callable[
    [str, dict[str, Any], dict[str, np.ndarray], Sequence[str]],
    "np.ndarray | None",
]


def _returns_from_close(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.zeros(closes.shape, dtype=float)
    return np.diff(closes, prepend=closes[0]) / np.clip(
        np.r_[closes[0], closes[:-1]], 1e-12, np.inf
    )


@dataclass(frozen=True, slots=True)
class StrategySignalDispatcher:
    """Route strategy candidates to concrete exposure builders."""

    handlers: Mapping[str, StrategySignalHandler]
    minimum_symbol_counts: Mapping[str, int] = field(default_factory=dict)

    def dispatch(
        self,
        candidate: dict[str, Any],
        *,
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
        unmapped_router: UnmappedStrategyRouter | None = None,
        require_actual_engine: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if require_actual_engine:
            strategy_class = ""
            try:
                strategy_class = str(
                    candidate.get("strategy_class") or candidate.get("strategy") or ""
                )
                params = dict(candidate.get("params") or {})
            except Exception as exc:
                raise StrategySignalDispatchError(
                    strategy_class, "invalid candidate or params"
                ) from exc
        else:
            strategy_class = str(candidate.get("strategy_class") or candidate.get("strategy") or "")
            params = dict(candidate.get("params") or {})
        if require_actual_engine:
            n = self._validate_actual_engine_input(
                strategy_class=strategy_class,
                candidate=candidate,
                aligned=aligned,
                symbols=symbols,
            )
        else:
            n = len(next(iter(aligned.values()))) if aligned else 0
        if n <= 0:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty, {}
        if not symbols:
            empty = np.zeros(n, dtype=float)
            return empty, empty, empty, {}

        exposures = np.zeros((len(symbols), n), dtype=float)
        returns = np.zeros((len(symbols), n), dtype=float)

        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            if require_actual_engine:
                close = np.asarray(close, dtype=float)
            returns[s_idx] = _returns_from_close(close)

        meta: dict[str, Any] = {}
        if require_actual_engine:
            meta["generic_fallback_proxy_count"] = 0
        handler = self.handlers.get(strategy_class)
        required_symbols = int(self.minimum_symbol_counts.get(strategy_class, 1))
        if require_actual_engine:
            if handler is None and unmapped_router is None:
                raise StrategySignalDispatchError(
                    strategy_class, "no handler or registry simulator is available"
                )
            if handler is not None and len(symbols) < required_symbols:
                raise StrategySignalDispatchError(
                    strategy_class,
                    f"requires at least {required_symbols} symbols, received {len(symbols)}",
                )
            if handler is None:
                try:
                    routed = unmapped_router(strategy_class, params, aligned, symbols)
                except Exception as exc:
                    raise StrategySignalDispatchError(
                        strategy_class, "registry simulator raised an exception"
                    ) from exc
                if routed is None:
                    raise StrategySignalDispatchError(
                        strategy_class, "registry simulator returned no exposures"
                    )
                self._set_actual_engine_exposures(
                    strategy_class=strategy_class,
                    source="registry simulator",
                    target=exposures,
                    routed=routed,
                )
                meta["evaluation_mode"] = "registry_simulator"
            else:
                meta["_strict_actual_engine"] = True
                try:
                    handler(params, aligned, symbols, n, exposures, meta)
                except Exception as exc:
                    raise StrategySignalDispatchError(
                        strategy_class, "handler raised an exception"
                    ) from exc
                finally:
                    meta.pop("_strict_actual_engine", None)
                if meta.get("missing_support_data") or meta.get("missing_support_symbols"):
                    raise StrategySignalDispatchError(
                        strategy_class, "handler reported missing required support data"
                    )
                self._validate_actual_engine_exposures(
                    strategy_class=strategy_class,
                    source="handler",
                    exposures=exposures,
                    expected_shape=(len(symbols), n),
                )
                meta.setdefault("evaluation_mode", "handler")
                if meta["evaluation_mode"] != "handler":
                    raise StrategySignalDispatchError(
                        strategy_class,
                        "handler must use evaluation mode 'handler'",
                    )
                if (
                    any(
                        key == "generic_fallback_proxy"
                        or (
                            key.startswith("generic_fallback_proxy_")
                            and key != "generic_fallback_proxy_count"
                        )
                        for key in meta
                    )
                    or meta.get("generic_fallback_proxy_count") != 0
                    or type(meta.get("generic_fallback_proxy_count")) is not int
                ):
                    raise StrategySignalDispatchError(
                        strategy_class,
                        "handler reported generic fallback activity",
                    )
        elif handler is None and unmapped_router is not None:
            routed: np.ndarray | None = None
            try:
                routed = unmapped_router(strategy_class, params, aligned, symbols)
            except Exception:
                routed = None
            if routed is not None and np.asarray(routed).shape == exposures.shape:
                exposures[:] = np.asarray(routed, dtype=float)
                meta["evaluation_mode"] = "registry_simulator"
                meta["event_driven_proxy"] = True
            else:
                self._apply_generic_fallback(aligned=aligned, symbols=symbols, exposures=exposures)
                meta["evaluation_mode"] = "generic_fallback_proxy"
        elif handler is None or len(symbols) < required_symbols:
            self._apply_generic_fallback(aligned=aligned, symbols=symbols, exposures=exposures)
            meta["evaluation_mode"] = "generic_fallback_proxy"
        else:
            handler(params, aligned, symbols, n, exposures, meta)
            meta.setdefault("evaluation_mode", "handler")

        # Previous-bar exposure with a ZERO first column: there is no position
        # before the first bar.  (The old ``np.roll`` wrapped the LAST bar's
        # exposure into bar 0, leaking end-of-sample state into the window head
        # and charging a spurious first-bar trade.)
        prev = np.zeros_like(exposures)
        prev[:, 1:] = exposures[:, :-1]
        if require_actual_engine:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    exposure = np.nanmean(exposures, axis=0)
                    portfolio_ret = np.nanmean(prev * returns, axis=0)
                    turnover = np.nanmean(np.abs(exposures - prev), axis=0)
            except FloatingPointError as exc:
                raise StrategySignalDispatchError(
                    strategy_class, "derived portfolio outputs overflowed or became invalid"
                ) from exc
            self._validate_actual_engine_outputs(
                strategy_class=strategy_class,
                portfolio_ret=portfolio_ret,
                turnover=turnover,
                exposure=exposure,
                expected_shape=(n,),
            )
        else:
            exposure = np.nanmean(exposures, axis=0)
            portfolio_ret = np.nanmean(prev * returns, axis=0)
            turnover = np.nanmean(np.abs(exposures - prev), axis=0)
        return portfolio_ret, turnover, exposure, meta

    @staticmethod
    def _validate_actual_engine_input(
        *,
        strategy_class: str,
        candidate: Mapping[str, Any],
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
    ) -> int:
        if not strategy_class:
            raise StrategySignalDispatchError(strategy_class, "strategy class is missing")
        if not aligned:
            raise StrategySignalDispatchError(strategy_class, "aligned input is empty")
        if not symbols:
            raise StrategySignalDispatchError(strategy_class, "no symbols were supplied")
        if any(
            not isinstance(symbol, str) or not symbol or symbol != symbol.strip()
            for symbol in symbols
        ):
            raise StrategySignalDispatchError(
                strategy_class, "symbols must be non-empty exact strings"
            )
        if len(symbols) != len(set(symbols)):
            raise StrategySignalDispatchError(strategy_class, "symbols must be unique")

        required_bar_keys = {
            f"{symbol}:{field}"
            for symbol in symbols
            for field in ("open", "high", "low", "close", "volume")
        }
        required_keys = {"datetime", *required_bar_keys}
        for key in required_keys:
            if key not in aligned:
                if key == "datetime":
                    raise StrategySignalDispatchError(strategy_class, "missing datetime array")
                raise StrategySignalDispatchError(
                    strategy_class, f"missing required bar array for {key}"
                )
        first_close_key = f"{symbols[0]}:close"
        try:
            n = len(aligned[first_close_key])
        except TypeError as exc:
            raise StrategySignalDispatchError(
                strategy_class, f"close array for {symbols[0]} has no length"
            ) from exc
        if n <= 0:
            raise StrategySignalDispatchError(strategy_class, "aligned input is empty")
        for key, value in aligned.items():
            try:
                values = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise StrategySignalDispatchError(
                    strategy_class, f"invalid aligned array for {key}"
                ) from exc
            if values.ndim != 1 or values.shape[0] != n:
                raise StrategySignalDispatchError(strategy_class, f"misaligned array for {key}")
            if key == "datetime":
                try:
                    if np.issubdtype(values.dtype, np.datetime64):
                        if np.isnat(values).any():
                            raise ValueError("NaT")
                        timestamps = values.astype("datetime64[ns]").astype(np.int64)
                    else:
                        normalized: list[float] = []
                        for value in values:
                            if not isinstance(value, datetime):
                                raise ValueError("non-datetime value")
                            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                                raise ValueError("non-UTC datetime")
                            normalized.append(value.timestamp())
                        timestamps = np.asarray(normalized, dtype=float)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise StrategySignalDispatchError(
                        strategy_class, f"invalid datetime array for {key}"
                    ) from exc
                if not np.isfinite(timestamps).all() or (
                    timestamps.size > 1 and np.any(timestamps[1:] <= timestamps[:-1])
                ):
                    raise StrategySignalDispatchError(
                        strategy_class, f"nonmonotone or nonfinite datetime array for {key}"
                    )
                StrategySignalDispatcher._validate_actual_engine_datetime_grid(
                    strategy_class=strategy_class,
                    candidate=candidate,
                    values=values,
                )
                continue
            try:
                numeric = values.astype(float, copy=False)
            except (TypeError, ValueError) as exc:
                raise StrategySignalDispatchError(
                    strategy_class, f"non-numeric aligned array for {key}"
                ) from exc
            if key in required_bar_keys and not np.isfinite(numeric).all():
                raise StrategySignalDispatchError(
                    strategy_class, f"nonfinite aligned array for {key}"
                )

        for symbol in symbols:
            close_key = f"{symbol}:close"
            if close_key not in aligned:
                raise StrategySignalDispatchError(
                    strategy_class, f"missing close array for {symbol}"
                )
            try:
                close = np.asarray(aligned[close_key], dtype=float)
            except (TypeError, ValueError) as exc:
                raise StrategySignalDispatchError(
                    strategy_class, f"non-numeric close array for {symbol}"
                ) from exc
            if close.ndim != 1 or close.shape[0] != n:
                raise StrategySignalDispatchError(
                    strategy_class, f"misaligned close array for {symbol}"
                )
            if not np.isfinite(close).all() or np.any(close <= 0.0):
                raise StrategySignalDispatchError(
                    strategy_class, f"nonpositive or nonfinite close array for {symbol}"
                )
        return n

    @staticmethod
    def _validate_actual_engine_datetime_grid(
        *,
        strategy_class: str,
        candidate: Mapping[str, Any],
        values: np.ndarray,
    ) -> None:
        if values.size < 2:
            raise StrategySignalDispatchError(
                strategy_class, "datetime grid requires at least two values"
            )
        try:
            if np.issubdtype(values.dtype, np.datetime64):
                normalized = values.astype("datetime64[ns]")
                if np.isnat(normalized).any():
                    raise ValueError("NaT")
                if not np.array_equal(normalized.astype(values.dtype), values):
                    raise ValueError("datetime conversion overflow")
                intervals = np.diff(normalized.astype(np.int64))
                units_per_second = 1_000_000_000
                if (
                    np.any(intervals <= 0)
                    or np.any(intervals != intervals[0])
                    or intervals[0] % units_per_second
                ):
                    raise ValueError("irregular cadence")
                cadence_ms = int(intervals[0] // 1_000_000)
            else:
                datetimes = list(values)
                if any(
                    not isinstance(value, datetime)
                    or value.tzinfo is None
                    or value.utcoffset() != UTC.utcoffset(value)
                    for value in datetimes
                ):
                    raise ValueError("non-UTC datetime")
                intervals = [
                    datetimes[index] - datetimes[index - 1] for index in range(1, len(datetimes))
                ]
                if (
                    any(interval <= timedelta(0) for interval in intervals)
                    or any(interval != intervals[0] for interval in intervals)
                    or intervals[0].microseconds
                ):
                    raise ValueError("irregular cadence")
                cadence_ms = (
                    intervals[0].days * 86_400_000
                    + intervals[0].seconds * 1_000
                    + intervals[0].microseconds // 1_000
                )
        except (TypeError, ValueError, OverflowError) as exc:
            raise StrategySignalDispatchError(
                strategy_class, "datetime grid is not positive, regular whole seconds"
            ) from exc

        declared_timeframes: list[int] = []
        try:
            for field in ("strategy_timeframe", "timeframe"):
                if field not in candidate:
                    continue
                raw_timeframe = candidate[field]
                if not isinstance(raw_timeframe, str) or not raw_timeframe.strip():
                    raise ValueError("invalid declared timeframe")
                token = raw_timeframe
                declared_timeframes.append(
                    int(timeframe_to_milliseconds(normalize_timeframe_token(token)))
                )
        except Exception as exc:
            raise StrategySignalDispatchError(
                strategy_class, "declared strategy timeframe is invalid"
            ) from exc
        if not declared_timeframes:
            return
        if len(set(declared_timeframes)) != 1:
            raise StrategySignalDispatchError(
                strategy_class, "declared strategy timeframes disagree"
            )
        declared_ms = declared_timeframes[0]
        if cadence_ms != declared_ms:
            raise StrategySignalDispatchError(
                strategy_class, "datetime grid does not match declared strategy timeframe"
            )

    @staticmethod
    def _set_actual_engine_exposures(
        *,
        strategy_class: str,
        source: str,
        target: np.ndarray,
        routed: np.ndarray,
    ) -> None:
        try:
            values = np.asarray(routed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise StrategySignalDispatchError(
                strategy_class, f"{source} returned non-numeric exposures"
            ) from exc
        StrategySignalDispatcher._validate_actual_engine_exposures(
            strategy_class=strategy_class,
            source=source,
            exposures=values,
            expected_shape=target.shape,
        )
        target[:] = values

    @staticmethod
    def _validate_actual_engine_exposures(
        *,
        strategy_class: str,
        source: str,
        exposures: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> None:
        if exposures.shape != expected_shape:
            raise StrategySignalDispatchError(
                strategy_class,
                f"{source} returned exposures with shape {exposures.shape}, expected {expected_shape}",
            )
        if not np.isfinite(exposures).all():
            raise StrategySignalDispatchError(
                strategy_class, f"{source} returned nonfinite exposures"
            )

    @staticmethod
    def _validate_actual_engine_outputs(
        *,
        strategy_class: str,
        portfolio_ret: np.ndarray,
        turnover: np.ndarray,
        exposure: np.ndarray,
        expected_shape: tuple[int],
    ) -> None:
        for name, values in (
            ("portfolio return", portfolio_ret),
            ("turnover", turnover),
            ("exposure", exposure),
        ):
            if values.shape != expected_shape:
                raise StrategySignalDispatchError(
                    strategy_class,
                    f"{name} has shape {values.shape}, expected {expected_shape}",
                )
            if not np.isfinite(values).all():
                raise StrategySignalDispatchError(strategy_class, f"{name} is nonfinite")

    @staticmethod
    def _apply_generic_fallback(
        *,
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
        exposures: np.ndarray,
    ) -> None:
        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            ret = _returns_from_close(close)
            mom = np.nan_to_num(_rolling_z(ret, 64), nan=0.0)
            exposures[s_idx] = np.where(mom >= 0.4, 1.0, np.where(mom <= -0.4, -1.0, 0.0))


# Local helper retained to keep the generic fallback self-contained.
def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if window <= 1:
        return np.zeros(arr.shape, dtype=float)

    out = np.zeros(arr.shape, dtype=float)
    for idx in range(window - 1, arr.size):
        segment = arr[idx - window + 1 : idx + 1]
        mean = float(np.nanmean(segment))
        std = float(np.nanstd(segment))
        if not np.isfinite(std) or std <= 0.0:
            out[idx] = 0.0
            continue
        out[idx] = (float(arr[idx]) - mean) / std
    return out
