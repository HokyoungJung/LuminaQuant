"""Helpers for split-aware candidate evaluation and stage selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any


@dataclass(frozen=True)
class ResearchStageSelector:
    """Evaluate candidates and rank the stage-2 shortlist."""

    evaluate_candidate: Callable[..., dict[str, Any]]
    stage1_prefilter_score: Callable[..., float]

    def evaluate_candidate_with_optional_split(
        self,
        candidate: dict[str, Any],
        *,
        cache: Mapping[tuple[str, str], Any],
        feature_cache: Mapping[str, Any] | None,
        aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None,
        benchmark_cache: Mapping[str, Mapping[str, Any] | Any],
        candidate_count: int,
        scoring_config: Mapping[str, Any] | None,
        split: Mapping[str, Any],
    ) -> dict[str, Any]:
        evaluate_kwargs = {
            "cache": cache,
            "feature_cache": feature_cache,
            "aligned_cache": aligned_cache,
            "benchmark_cache": benchmark_cache,
            "candidate_count": max(1, candidate_count),
            "scoring_config": scoring_config,
        }
        evaluate_kwargs_with_split = {**evaluate_kwargs, "split": split}
        try:
            return self.evaluate_candidate(candidate, **evaluate_kwargs_with_split)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return self.evaluate_candidate(
                candidate,
                **_supported_kwargs(self.evaluate_candidate, evaluate_kwargs_with_split),
            )

    def select_stage2_results(
        self,
        *,
        adapted: Sequence[dict[str, Any]],
        cache: Mapping[tuple[str, str], Any],
        feature_cache: Mapping[str, Any] | None,
        aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None,
        benchmark: Mapping[str, Mapping[str, Any] | Any],
        scoring: Any,
        resolved_split: Mapping[str, Any],
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        scored_stage1: list[tuple[float, dict[str, Any]]] = []
        candidate_count = len(adapted)
        for index, row in enumerate(adapted, start=1):
            raw_result = self.evaluate_candidate_with_optional_split(
                row,
                cache=cache,
                feature_cache=feature_cache,
                aligned_cache=aligned_cache,
                benchmark_cache=benchmark,
                candidate_count=candidate_count,
                scoring_config=scoring.resolved_scoring_config,
                split=resolved_split,
            )
            result = _stage_result_with_candidate_identity(raw_result, candidate=row)
            score = self.stage1_prefilter_score(
                result,
                stage1_weights=scoring.stage1_weights,
                stage1_error_score=scoring.stage1_error_score,
            )
            scored_stage1.append((float(score), result))
            if progress_callback is not None:
                progress_callback(
                    "candidate_evaluated",
                    {
                        "candidate_index": index,
                        "candidate_count": candidate_count,
                        **_candidate_progress_snapshot(result, stage1_prefilter_score=float(score)),
                    },
                )

        ranked = sorted(scored_stage1, key=lambda item: item[0], reverse=True)
        keep_count = max(1, round(len(ranked) * scoring.keep_ratio_applied))
        selected = [item[1] for item in ranked[:keep_count]]
        if progress_callback is not None:
            progress_callback(
                "stage1_ranked",
                {
                    "candidate_count": candidate_count,
                    "keep_count": keep_count,
                    "keep_ratio_applied": float(scoring.keep_ratio_applied),
                    "top_stage1_candidates": [
                        _candidate_progress_snapshot(result, stage1_prefilter_score=float(score))
                        for score, result in ranked[: min(5, len(ranked))]
                    ],
                },
            )
            progress_callback(
                "stage2_selected",
                {
                    "selected_count": len(selected),
                    "selected_candidates": [
                        _candidate_progress_snapshot(result, stage1_prefilter_score=float(score))
                        for score, result in ranked[:keep_count]
                    ],
                },
            )
        return selected


def _supported_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        params = signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)

    return {
        name: value
        for name, value in kwargs.items()
        if name in params
        and params[name].kind
        in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
    }


def _candidate_progress_snapshot(
    result: Mapping[str, Any],
    *,
    stage1_prefilter_score: float,
) -> dict[str, Any]:
    def _split_summary(name: str) -> dict[str, Any]:
        split_block = dict(result.get(name) or {})
        return {
            "total_return": float(
                split_block.get("total_return", split_block.get("return", 0.0)) or 0.0
            ),
            "sharpe": float(split_block.get("sharpe", 0.0) or 0.0),
            "max_drawdown": float(
                split_block.get("max_drawdown", split_block.get("mdd", 0.0)) or 0.0
            ),
            "trade_count": float(
                split_block.get("trade_count", split_block.get("trades", 0.0)) or 0.0
            ),
        }

    return {
        "candidate_id": str(result.get("candidate_id") or result.get("name") or ""),
        "name": str(result.get("name") or ""),
        "strategy_class": str(result.get("strategy_class") or ""),
        "family": str(result.get("family") or ""),
        "strategy_timeframe": str(
            result.get("strategy_timeframe") or result.get("timeframe") or ""
        ),
        "stage1_prefilter_score": float(stage1_prefilter_score),
        "pass": bool(result.get("pass", False)),
        "hard_reject": bool(result.get("hard_reject", False)),
        "error": str(result.get("error") or ""),
        "train": _split_summary("train"),
        "val": _split_summary("val"),
        "oos": _split_summary("oos"),
    }


def _stage_result_with_candidate_identity(
    result: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = dict(result)
    identity_defaults = {
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("name") or ""),
        "name": str(candidate.get("name") or candidate.get("candidate_id") or ""),
        "strategy_class": str(candidate.get("strategy_class") or candidate.get("strategy") or ""),
        "family": str(candidate.get("family") or ""),
        "strategy_timeframe": str(
            candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""
        ),
        "timeframe": str(candidate.get("timeframe") or candidate.get("strategy_timeframe") or ""),
    }
    for key, value in identity_defaults.items():
        if not str(enriched.get(key) or "").strip():
            enriched[key] = value
    return enriched
