"""Helpers for split-aware candidate report construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResearchReportBuilder:
    """Build candidate report payloads and attach cross-candidate diagnostics."""

    split_masks_from_datetimes: Callable[..., dict[str, np.ndarray]]
    split_lengths: Callable[[int], tuple[slice, slice, slice]]
    compute_metrics: Callable[..., dict[str, Any]]
    hurdle_fields: Callable[..., tuple[dict[str, Any], bool, Any]]
    family_from_strategy: Callable[[str], str]
    canonicalize_symbol_list: Callable[[Any], list[str]]
    series_to_stream: Callable[..., list[dict[str, float | int]]]
    candidate_rank_score: Callable[..., float]
    correlation: Callable[[np.ndarray, np.ndarray], float]
    periods_per_year: Mapping[str, int]
    metric_config: Any

    def result_timestamps_and_split_masks(
        self,
        result: Mapping[str, Any],
        *,
        resolved_split: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
        returns = np.asarray(result.get("returns"), dtype=float)
        raw_timestamps = result.get("timestamps")
        timestamps = (
            np.asarray(raw_timestamps, dtype="datetime64[ms]")
            if raw_timestamps is not None
            else np.asarray([], dtype="datetime64[ms]")
        )
        has_aligned_timestamps = timestamps.size == returns.size
        if has_aligned_timestamps:
            split_masks = self.split_masks_from_datetimes(timestamps, split=resolved_split)
        else:
            train_slice, val_slice, oos_slice = self.split_lengths(returns.size)
            split_masks = {
                "train": np.zeros(returns.size, dtype=bool),
                "val": np.zeros(returns.size, dtype=bool),
                "oos": np.zeros(returns.size, dtype=bool),
            }
            split_masks["train"][train_slice] = True
            split_masks["val"][val_slice] = True
            split_masks["oos"][oos_slice] = True
        return timestamps, split_masks, has_aligned_timestamps

    def error_candidate_report_payload(
        self,
        *,
        result: Mapping[str, Any],
        resolved_scoring_config: Mapping[str, Any],
        failed_candidate_selection_score: float,
        candidate_count: int,
    ) -> dict[str, Any]:
        row = dict(result.get("candidate") or {})
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "1m")
        empty_metrics = self.compute_metrics(
            np.asarray([], dtype=float),
            turnover=np.asarray([], dtype=float),
            exposure=np.asarray([], dtype=float),
            benchmark_returns=np.asarray([], dtype=float),
            periods_per_year=int(self.periods_per_year.get(timeframe, 365)),
            num_trials=max(1, candidate_count),
            metric_config=self.metric_config,
            timestamps=np.asarray([], dtype="datetime64[ms]"),
        )
        hurdles, passed, _ = self.hurdle_fields(
            empty_metrics,
            empty_metrics,
            empty_metrics,
            scoring_config=resolved_scoring_config,
        )
        return {
            "candidate_id": str(row.get("candidate_id")),
            "name": str(row.get("name")),
            "strategy_class": str(row.get("strategy_class")),
            "strategy": str(row.get("strategy") or row.get("strategy_class") or ""),
            "family": str(
                row.get("family") or self.family_from_strategy(str(row.get("strategy_class")))
            ),
            "strategy_timeframe": timeframe,
            "timeframe": timeframe,
            "symbols": self.canonicalize_symbol_list(list(row.get("symbols") or [])),
            "params": dict(row.get("params") or {}),
            "train": empty_metrics,
            "val": empty_metrics,
            "oos": empty_metrics,
            "hurdle_fields": hurdles,
            "return_streams": {"train": [], "val": [], "oos": []},
            "cost_metrics": {"turnover": 0.0, "fee_cost": 0.0, "slippage_cost": 0.0},
            "oos_cost_stress": {
                "x2": {"sharpe": 0.0, "return": 0.0},
                "x3": {"sharpe": 0.0, "return": 0.0},
            },
            "hard_reject": True,
            "hard_reject_reasons": {"insufficient_data": True},
            "selection_score": failed_candidate_selection_score,
            "pass": bool(passed),
            "metadata": dict(result.get("metadata") or {}),
        }

    def candidate_return_streams(
        self,
        *,
        returns: np.ndarray,
        timestamps: np.ndarray,
        split_masks: Mapping[str, np.ndarray],
        has_aligned_timestamps: bool,
    ) -> dict[str, list[dict[str, float | int]]]:
        return {
            "train": self.series_to_stream(
                returns[split_masks["train"]],
                timestamps=timestamps[split_masks["train"]] if has_aligned_timestamps else None,
            ),
            "val": self.series_to_stream(
                returns[split_masks["val"]],
                timestamps=timestamps[split_masks["val"]] if has_aligned_timestamps else None,
            ),
            "oos": self.series_to_stream(
                returns[split_masks["oos"]],
                timestamps=timestamps[split_masks["oos"]] if has_aligned_timestamps else None,
            ),
        }

    def successful_candidate_report_payload(
        self,
        *,
        result: Mapping[str, Any],
        resolved_split: Mapping[str, Any],
        resolved_scoring_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        row = dict(result.get("candidate") or {})
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "1m")
        returns = np.asarray(result.get("returns"), dtype=float)
        timestamps, split_masks, has_aligned_timestamps = self.result_timestamps_and_split_masks(
            result,
            resolved_split=resolved_split,
        )
        train = dict(result.get("train") or {})
        val = dict(result.get("val") or {})
        oos = dict(result.get("oos") or {})
        hard_reject = dict(result.get("hard_reject_reasons") or {})
        hard_reject_flag = bool(hard_reject)

        candidate_payload = {
            "candidate_id": str(row.get("candidate_id")),
            "name": str(row.get("name")),
            "strategy_class": str(row.get("strategy_class")),
            "strategy": str(row.get("strategy") or row.get("strategy_class") or ""),
            "family": str(
                row.get("family") or self.family_from_strategy(str(row.get("strategy_class")))
            ),
            "strategy_timeframe": timeframe,
            "timeframe": timeframe,
            "symbols": self.canonicalize_symbol_list(list(row.get("symbols") or [])),
            "params": dict(row.get("params") or {}),
            "notes": str(row.get("notes") or result.get("notes") or ""),
            "tags": [
                str(tag) for tag in list(row.get("tags") or result.get("tags") or []) if str(tag)
            ],
            "train": train,
            "val": val,
            "oos": oos,
            "hurdle_fields": dict(result.get("hurdle_fields") or {}),
            "return_streams": self.candidate_return_streams(
                returns=returns,
                timestamps=timestamps,
                split_masks=split_masks,
                has_aligned_timestamps=has_aligned_timestamps,
            ),
            "cost_metrics": {
                "turnover": float(oos.get("turnover", 0.0)),
                "fee_cost": float(result.get("metadata", {}).get("cost_rate", 0.0)),
                "slippage_cost": float(result.get("metadata", {}).get("cost_rate", 0.0) * 0.7),
            },
            "oos_cost_stress": dict(result.get("oos_cost_stress") or {}),
            "hard_reject": hard_reject_flag,
            "hard_reject_reasons": hard_reject,
            "pass": bool(result.get("pass", False)) and not hard_reject_flag,
            "metadata": {
                **dict(row.get("metadata") or {}),
                **dict(result.get("metadata") or {}),
            },
        }
        candidate_payload["selection_score"] = self.candidate_rank_score(
            candidate_payload,
            scoring_config=resolved_scoring_config,
        )
        return candidate_payload

    def report_candidates_from_stage2_results(
        self,
        *,
        stage2_results: Sequence[dict[str, Any]],
        candidate_count: int,
        resolved_split: Mapping[str, Any],
        scoring: Any,
    ) -> list[dict[str, Any]]:
        report_candidates: list[dict[str, Any]] = []
        for result in stage2_results:
            if result.get("error"):
                report_candidates.append(
                    self.error_candidate_report_payload(
                        result=result,
                        resolved_scoring_config=scoring.resolved_scoring_config,
                        failed_candidate_selection_score=scoring.failed_candidate_selection_score,
                        candidate_count=candidate_count,
                    )
                )
                continue
            report_candidates.append(
                self.successful_candidate_report_payload(
                    result=result,
                    resolved_split=resolved_split,
                    resolved_scoring_config=scoring.resolved_scoring_config,
                )
            )
        return report_candidates

    def attach_cross_candidate_correlations(
        self,
        report_candidates: Sequence[dict[str, Any]],
    ) -> None:
        oos_series = {
            row["candidate_id"]: np.asarray(
                [point["v"] for point in row["return_streams"]["oos"]],
                dtype=float,
            )
            for row in report_candidates
            if row.get("return_streams", {}).get("oos")
        }
        for row in report_candidates:
            cid = str(row.get("candidate_id"))
            base = oos_series.get(cid)
            if base is None or base.size < 8:
                row.setdefault("oos", {})["cross_candidate_corr"] = 0.0
                continue
            corr_values: list[float] = []
            for other_id, other in oos_series.items():
                if other_id == cid:
                    continue
                corr_values.append(self.correlation(base, other))
            row.setdefault("oos", {})["cross_candidate_corr"] = (
                float(np.mean(corr_values)) if corr_values else 0.0
            )
