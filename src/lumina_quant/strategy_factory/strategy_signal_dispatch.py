"""Strategy-signal dispatch helpers for research runner orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

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
# ``(strategy_class, params, aligned, symbols) -> exposures | None``.  Returning
# ``None`` (or raising) falls back to the generic proxy, which is now always
# labelled in ``meta`` so proxy rows can never be silently attributed to a lane.
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
        strategy_class = str(candidate.get("strategy_class") or candidate.get("strategy") or "")
        params = dict(candidate.get("params") or {})

        if require_actual_engine:
            n = self._validate_actual_engine_input(
                strategy_class=strategy_class,
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
                try:
                    handler(params, aligned, symbols, n, exposures, meta)
                except Exception as exc:
                    raise StrategySignalDispatchError(
                        strategy_class, "handler raised an exception"
                    ) from exc
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
        exposure = np.nanmean(exposures, axis=0)
        portfolio_ret = np.nanmean(prev * returns, axis=0)
        turnover = np.nanmean(np.abs(exposures - prev), axis=0)
        return portfolio_ret, turnover, exposure, meta

    @staticmethod
    def _validate_actual_engine_input(
        *,
        strategy_class: str,
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

        first_close_key = f"{symbols[0]}:close"
        if first_close_key not in aligned:
            raise StrategySignalDispatchError(
                strategy_class, f"missing close array for {symbols[0]}"
            )
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
            if key == "datetime" or np.issubdtype(values.dtype, np.datetime64):
                try:
                    if np.issubdtype(values.dtype, np.datetime64):
                        if np.isnat(values).any():
                            raise ValueError("NaT")
                        timestamps = values.astype("datetime64[ns]").astype(np.int64)
                    elif np.issubdtype(values.dtype, np.number):
                        timestamps = values.astype(float, copy=False)
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
                continue
            try:
                numeric = values.astype(float, copy=False)
            except (TypeError, ValueError) as exc:
                raise StrategySignalDispatchError(
                    strategy_class, f"non-numeric aligned array for {key}"
                ) from exc
            if not np.isfinite(numeric).all():
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
        if values.shape != target.shape:
            raise StrategySignalDispatchError(
                strategy_class,
                f"{source} returned exposures with shape {values.shape}, expected {target.shape}",
            )
        target[:] = values
        StrategySignalDispatcher._validate_actual_engine_exposures(
            strategy_class=strategy_class,
            source=source,
            exposures=target,
            expected_shape=target.shape,
        )

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
