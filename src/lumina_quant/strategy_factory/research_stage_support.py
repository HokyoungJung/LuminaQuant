"""Shared resource/stage helpers for candidate-research orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.strategy_factory.research_resources import ResearchResourceLoader
from lumina_quant.strategy_factory.research_stage_selection import ResearchStageSelector


def build_research_resource_loader(
    *,
    split_window_bounds: Any,
    datetime_to_iso_z: Any,
    load_bundle_cache: Any,
    load_feature_cache: Any,
    benchmark_cache: Any,
    canonicalize_symbol_list: Any,
) -> ResearchResourceLoader:
    return ResearchResourceLoader(
        split_window_bounds=split_window_bounds,
        datetime_to_iso_z=datetime_to_iso_z,
        load_bundle_cache=load_bundle_cache,
        load_feature_cache=load_feature_cache,
        benchmark_cache=benchmark_cache,
        canonicalize_symbol_list=canonicalize_symbol_list,
    )


def load_research_run_resources(
    *,
    loader: ResearchResourceLoader,
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
) -> tuple[
    dict[tuple[str, str], Any],
    dict[str, list[str]],
    dict[str, pl.DataFrame],
    dict[str, dict[str, np.ndarray]],
]:
    return loader.load(
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
    )


def evaluate_candidate_with_optional_split(
    *,
    evaluate_candidate: Any,
    stage1_prefilter_score: Any,
    candidate: dict[str, Any],
    cache: Mapping[tuple[str, str], Any],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    return ResearchStageSelector(
        evaluate_candidate=evaluate_candidate,
        stage1_prefilter_score=stage1_prefilter_score,
    ).evaluate_candidate_with_optional_split(
        candidate,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        benchmark_cache=benchmark_cache,
        candidate_count=candidate_count,
        scoring_config=scoring_config,
        split=split,
    )


def select_stage2_results(
    *,
    evaluate_candidate: Any,
    stage1_prefilter_score: Any,
    adapted: Sequence[dict[str, Any]],
    cache: Mapping[tuple[str, str], Any],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None,
    benchmark: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    scoring: Any,
    resolved_split: Mapping[str, Any],
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    return ResearchStageSelector(
        evaluate_candidate=evaluate_candidate,
        stage1_prefilter_score=stage1_prefilter_score,
    ).select_stage2_results(
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
        progress_callback=progress_callback,
    )


__all__ = [
    "build_research_resource_loader",
    "evaluate_candidate_with_optional_split",
    "load_research_run_resources",
    "select_stage2_results",
]
