"""Candidate-research entrypoints extracted from the monolithic runner module.

This module intentionally keeps the public entrypoint contract stable while
moving the orchestration-facing surface out of ``research_runner.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES
from lumina_quant.strategy_factory.runtime_settings import (
    current_research_market_data_settings as _current_research_market_data_settings_impl,
)

from . import research_run_support as _research_run_support


def _runner_module():
    """Resolve the legacy runner lazily to reduce import-time coupling.

    Deliberately NOT ``@lru_cache``-d: ``import_module`` on an already-imported
    module is just a ``sys.modules`` dict lookup (negligible cost), whereas a
    cached reference can go stale if anything elsewhere in the process ever
    replaces ``sys.modules["lumina_quant.strategy_factory.research_runner"]``
    with a new module object (observed in the wider test suite, where some
    script-loading test harnesses reload modules under the same dotted name) --
    a cached stale reference would silently keep resolving to the OLD module
    object, so callers' ``monkeypatch.setattr(research_runner, ...)`` on the
    CURRENT module would have no effect on code reached through this accessor.
    """
    return import_module("lumina_quant.strategy_factory.research_runner")


def _report_candidates_from_stage2_results(
    *,
    stage2_results: Sequence[dict[str, Any]],
    candidate_count: int,
    resolved_split: Mapping[str, Any],
    scoring: Any,
) -> list[dict[str, Any]]:
    runner = _runner_module()
    return runner._research_report_builder().report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=candidate_count,
        resolved_split=resolved_split,
        scoring=scoring,
    )


def _attach_cross_candidate_correlations(report_candidates: Sequence[dict[str, Any]]) -> None:
    runner = _runner_module()
    runner._research_report_builder().attach_cross_candidate_correlations(report_candidates)


def _sorted_report_candidates(
    report_candidates: Sequence[dict[str, Any]],
    *,
    scoring: Any,
) -> list[dict[str, Any]]:
    rows = list(report_candidates)
    runner = _runner_module()
    runner._attach_cross_candidate_correlations(rows)
    rows.sort(
        key=lambda item: float(item.get("selection_score", scoring.sort_missing_selection_score)),
        reverse=True,
    )
    return rows


def _candidate_research_report_payload(
    *,
    base_tf: str,
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    adapted: Sequence[dict[str, Any]],
    stage2_results: Sequence[dict[str, Any]],
    stage1_keep_ratio: float,
    scoring: Any,
    data_sources: dict[str, list[str]],
    report_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_timeframe": base_tf,
        "strategy_timeframes": normalized_timeframes,
        "symbol_universe": universe,
        "split": resolved_split,
        "candidates": report_candidates,
        "stage1": {
            "input_count": len(adapted),
            "selected_count": len(stage2_results),
            "keep_ratio": float(stage1_keep_ratio),
            "keep_ratio_applied": float(scoring.keep_ratio_applied),
        },
        "scoring_config": scoring.resolved_scoring_config,
        "data_sources": data_sources,
    }


def _run_candidate_research_with_adapted_candidates(
    *,
    base_tf: str,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: Any,
    split: Mapping[str, Any] | None,
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any],
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    split_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runner = _runner_module()
    normalized_timeframes, universe = (
        _research_run_support._resolve_research_run_timeframes_and_universe(
            adapted=adapted,
            strategy_timeframes=strategy_timeframes,
            symbol_universe=symbol_universe,
        )
    )
    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    resolved_split = _research_run_support.apply_split_overrides(
        runner._resolve_split_config(split, strategy_timeframe=split_timeframe),
        split_overrides,
    )

    if progress_callback is not None:
        progress_callback(
            "resource_load_started",
            {
                "candidate_count": len(adapted),
                "normalized_timeframes": list(normalized_timeframes),
                "symbol_universe": list(universe),
            },
        )
    cache, data_sources, feature_cache, benchmark = runner._load_research_run_resources(
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=str(data_mode or "legacy"),
        allow_csv_fallback=bool(allow_csv_fallback),
        allow_synthetic_fallback=bool(allow_synthetic_fallback),
        min_bundle_bars=max(1, int(min_bundle_bars)),
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            "resources_loaded",
            {
                "candidate_count": len(adapted),
                "normalized_timeframes": list(normalized_timeframes),
                "symbol_universe": list(universe),
                "bundle_count": len(cache),
                "feature_frame_count": len(feature_cache or {}),
                "benchmark_count": len(benchmark),
            },
        )
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    stage2_results = runner._select_stage2_results(
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
        progress_callback=progress_callback,
    )
    report_candidates = runner._report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=len(adapted),
        resolved_split=resolved_split,
        scoring=scoring,
    )
    sorted_candidates = runner._sorted_report_candidates(report_candidates, scoring=scoring)
    if progress_callback is not None:
        progress_callback(
            "report_ready",
            {
                "reported_candidate_count": len(sorted_candidates),
                "top_report_candidates": [
                    {
                        "candidate_id": str(row.get("candidate_id") or row.get("name") or ""),
                        "name": str(row.get("name") or ""),
                        "selection_score": float(row.get("selection_score", 0.0) or 0.0),
                        "oos_total_return": float(
                            dict(row.get("oos") or {}).get(
                                "total_return",
                                dict(row.get("oos") or {}).get("return", 0.0),
                            )
                            or 0.0
                        ),
                        "oos_sharpe": float(dict(row.get("oos") or {}).get("sharpe", 0.0) or 0.0),
                    }
                    for row in sorted_candidates[: min(5, len(sorted_candidates))]
                ],
            },
        )
    return runner._candidate_research_report_payload(
        base_tf=base_tf,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        data_sources=data_sources,
        report_candidates=sorted_candidates,
    )


def _merge_score_config_with_research_overrides(
    score_config: Mapping[str, Any] | None,
    *,
    score_config_research_overrides: Mapping[str, Any],
    deflation_kwargs: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Fold ``research_config``-derived flags into ``score_config['research']``.

    Caller-provided keys in the caller's own ``score_config['research']`` section
    always win; the config only fills gaps. ``deflation_kwargs`` is merged in
    too because ``hac_inference`` is read back out of this same
    ``score_config['research']`` section by ``research_runner``'s per-candidate
    DSR computation (see ``_evaluate_candidate_metric_payload``); the other
    deflation keys (``single_correlation_discount`` / ``cscv_pbo``) have no
    consumer in this module today and simply ride along inertly for audit
    visibility in the report's ``scoring_config``. Returns ``score_config``
    untouched when there is nothing to inject.
    """
    section_overrides: dict[str, Any] = {**score_config_research_overrides, **deflation_kwargs}
    if not section_overrides:
        return score_config
    merged: dict[str, Any] = dict(score_config) if isinstance(score_config, Mapping) else {}
    caller_research = merged.get("research")
    merged_research: dict[str, Any] = dict(section_overrides)
    if isinstance(caller_research, Mapping):
        merged_research.update(caller_research)
    merged["research"] = merged_research
    return merged


def run_candidate_research(
    *,
    candidates: Iterable[dict[str, Any]],
    base_timeframe: str = "1s",
    strategy_timeframes: Sequence[str] | None = None,
    symbol_universe: Sequence[str] | None = None,
    stage1_keep_ratio: float = 0.35,
    max_candidates: int = 512,
    score_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = False,
    min_bundle_bars: int = 360,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    research_config: Any = None,
) -> dict[str, Any]:
    """Evaluate candidate manifest into train/val/OOS report contract (v2).

    ``research_config`` (default ``None``) is the strict-research-profile
    activation seam: pass a ``ResearchConfig`` (or a full ``RuntimeConfig``,
    whose ``.research`` is used) to fold ``use_lockbox_split`` /
    ``purge_embargo_bars`` into ``split``, ``strict_selection_gate`` /
    ``hac_inference`` (plus ``dsr_gate_floor`` / ``spa_gate_ceiling`` when
    present) into ``score_config['research']``, so the strict-selection gate,
    lockbox split, purge/embargo bars, and HAC-corrected DSR studentization
    actually activate end-to-end instead of sitting inert on the config object.
    Any key the caller already set explicitly on ``split`` / ``score_config``
    wins over the config-derived value. The default ``None`` is a strict no-op:
    ``split`` / ``score_config`` are forwarded unchanged, so behavior is
    byte-identical to calling this function before ``research_config`` existed.
    """
    runner = _runner_module()
    base_tf = runner._normalize_candidate_research_base_timeframe(base_timeframe)
    market_data_settings = _current_research_market_data_settings_impl()
    overrides = _research_run_support.research_config_to_overrides(research_config)
    split_overrides = overrides["split"]
    score_config = _merge_score_config_with_research_overrides(
        score_config,
        score_config_research_overrides=overrides["score_config_research"],
        deflation_kwargs=overrides["deflation_kwargs"],
    )
    scoring = runner._resolve_research_run_scoring_config(
        score_config=score_config,
        stage1_keep_ratio=stage1_keep_ratio,
    )
    adapted = runner._adapt_candidate_inputs(candidates, max_candidates=max_candidates)

    if not adapted:
        return runner._empty_candidate_research_report(
            base_timeframe=base_tf,
            strategy_timeframes=strategy_timeframes,
            symbol_universe=symbol_universe,
            stage1_keep_ratio=stage1_keep_ratio,
            scoring=scoring,
            split=split,
            split_overrides=split_overrides,
        )

    return _run_candidate_research_with_adapted_candidates(
        base_tf=base_tf,
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        split=split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
        split_overrides=split_overrides,
    )


def build_default_candidate_rows(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    max_candidates: int = 512,
) -> list[dict[str, Any]]:
    """Build candidate rows from strategy-factory candidate library."""
    from lumina_quant.strategy_factory.candidate_library import build_binance_futures_candidates

    rows = build_binance_futures_candidates(
        symbols=symbols or _current_research_market_data_settings_impl()["symbols"],
        timeframes=timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
    )
    out = [_research_run_support.adapt_legacy_candidate(item.to_dict()) for item in rows]
    if int(max_candidates) > 0:
        out = out[: int(max_candidates)]
    return out


__all__ = [
    "build_default_candidate_rows",
    "run_candidate_research",
]
