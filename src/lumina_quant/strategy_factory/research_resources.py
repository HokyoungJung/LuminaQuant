"""Helpers for loading candidate-research resources with optional compatibility fallbacks."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    /,
    **kwargs: Any,
) -> Any:
    """Call ``func`` with the subset of keyword arguments it can accept.

    Candidate-research tests and downstream plugin hooks commonly monkeypatch
    resource loaders with older, narrower signatures.  The production loaders
    accept newer split/data-mode/progress kwargs, so keep the compatibility
    shim local to the resource boundary instead of forcing every caller to
    implement a catch-and-retry matrix.
    """
    try:
        params = signature(func).parameters
    except TypeError, ValueError:
        return func(**kwargs)

    if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
        return func(**kwargs)

    supported = {
        name: value
        for name, value in kwargs.items()
        if name in params
        and params[name].kind
        in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
    }
    return func(**supported)


@dataclass(frozen=True)
class ResearchResourceLoader:
    """Load bundle, feature, and benchmark resources for candidate research."""

    split_window_bounds: Callable[[Mapping[str, Any]], tuple[Any, Any]]
    datetime_to_iso_z: Callable[[Any], str | None]
    load_bundle_cache: Callable[..., tuple[dict[tuple[str, str], Any], dict[str, list[str]]]]
    load_feature_cache: Callable[..., dict[str, pl.DataFrame]]
    benchmark_cache: Callable[
        [Mapping[tuple[str, str], Any], Sequence[str]],
        dict[str, dict[str, np.ndarray] | np.ndarray],
    ]
    canonicalize_symbol_list: Callable[[Any], list[str]]

    def load(
        self,
        *,
        adapted: Sequence[dict[str, Any]],
        normalized_timeframes: Sequence[str],
        universe: Sequence[str],
        resolved_split: Mapping[str, Any],
        data_mode: str,
        allow_csv_fallback: bool,
        allow_synthetic_fallback: bool,
        min_bundle_bars: int,
        market_data_settings: Mapping[str, Any] | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        feature_support_strategy_classes: Sequence[str] = (
            "CarryTrendFactorRotationStrategy",
            "PerpCrowdingCarryStrategy",
            "CompositeTrendStrategy",
        ),
    ) -> tuple[
        dict[tuple[str, str], Any],
        dict[str, list[str]],
        dict[str, pl.DataFrame],
        dict[str, dict[str, np.ndarray] | np.ndarray],
    ]:
        load_start, load_end = self.split_window_bounds(resolved_split)
        load_bundle_kwargs = {
            "symbols": universe,
            "timeframes": normalized_timeframes,
            "start_date": self.datetime_to_iso_z(load_start),
            "end_date": self.datetime_to_iso_z(load_end),
            "data_mode": str(data_mode or "legacy"),
            "allow_csv_fallback": bool(allow_csv_fallback),
            "allow_synthetic_fallback": bool(allow_synthetic_fallback),
            "min_bars": max(1, int(min_bundle_bars)),
            "market_data_settings": market_data_settings,
        }
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_load_started",
                {
                    "symbol_count": len(universe),
                    "timeframe_count": len(normalized_timeframes),
                    "total_count": len(universe) * len(normalized_timeframes),
                    "symbol_universe": list(universe),
                    "normalized_timeframes": list(normalized_timeframes),
                },
            )
        bundle_started_at = perf_counter()
        cache, data_sources = _call_with_supported_kwargs(
            self.load_bundle_cache,
            **load_bundle_kwargs,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_load_completed",
                {
                    "bundle_count": len(cache),
                    "total_count": len(universe) * len(normalized_timeframes),
                    "elapsed_seconds": round(max(0.0, perf_counter() - bundle_started_at), 6),
                    "source_counts": {
                        str(source): len(list(items or []))
                        for source, items in dict(data_sources or {}).items()
                    },
                },
            )

        feature_support_class_set = set(feature_support_strategy_classes)
        support_feature_symbols = self.canonicalize_symbol_list(
            itertools.chain.from_iterable(
                list(row.get("symbols") or [])
                for row in adapted
                if str(row.get("strategy_class") or row.get("strategy") or "")
                in feature_support_class_set
            )
        )
        if progress_callback is not None:
            progress_callback(
                "resource_feature_load_started",
                {
                    "symbol_count": len(support_feature_symbols),
                    "feature_symbols": list(support_feature_symbols),
                },
            )
        feature_started_at = perf_counter()
        feature_cache = _call_with_supported_kwargs(
            self.load_feature_cache,
            symbols=support_feature_symbols,
            start_date=self.datetime_to_iso_z(load_start),
            end_date=self.datetime_to_iso_z(load_end),
            market_data_settings=market_data_settings,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(
                "resource_feature_load_completed",
                {
                    "symbol_count": len(support_feature_symbols),
                    "feature_frame_count": len(feature_cache),
                    "nonempty_symbol_count": sum(
                        1
                        for frame in feature_cache.values()
                        if hasattr(frame, "is_empty") and not frame.is_empty()
                    ),
                    "total_rows": sum(
                        int(getattr(frame, "height", 0) or 0) for frame in feature_cache.values()
                    ),
                    "elapsed_seconds": round(max(0.0, perf_counter() - feature_started_at), 6),
                },
            )
            progress_callback(
                "resource_benchmark_build_started",
                {
                    "timeframe_count": len(normalized_timeframes),
                    "normalized_timeframes": list(normalized_timeframes),
                },
            )
        benchmark_started_at = perf_counter()
        try:
            benchmark = self.benchmark_cache(
                cache,
                normalized_timeframes,
                progress_callback=progress_callback,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            benchmark = self.benchmark_cache(cache, normalized_timeframes)
        if progress_callback is not None:

            def _benchmark_return_count(payload: Any) -> int:
                if isinstance(payload, Mapping):
                    values = payload.get("returns", [])
                else:
                    values = payload
                if values is None:
                    values = []
                return len(np.asarray(values, dtype=float))

            progress_callback(
                "resource_benchmark_build_completed",
                {
                    "benchmark_count": len(benchmark),
                    "timeframe_count": len(normalized_timeframes),
                    "elapsed_seconds": round(max(0.0, perf_counter() - benchmark_started_at), 6),
                    "nonempty_timeframe_count": sum(
                        1 for payload in benchmark.values() if _benchmark_return_count(payload) > 0
                    ),
                },
            )
        return cache, data_sources, feature_cache, benchmark
