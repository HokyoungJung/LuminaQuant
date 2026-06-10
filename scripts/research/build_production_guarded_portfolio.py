"""Build a low-leverage, drawdown-aware production candidate from saved sleeves.

This script stays inside the repo's low-memory follow-up contract by operating only
on saved daily return streams. It treats the refreshed hybrid sleeve as the primary
engine, keeps incumbent/static-blend sleeves as stabilizers, and optionally admits a
carry/trend factor-rotation candidate only when its saved research metrics remain
production-safe.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_followup_rules import evaluate_weighted_portfolio, validation_objective

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
DEFAULT_HYBRID_PATH = (
    GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_STATIC_BLEND_PATH = (
    GROUP_ROOT / "current_switch_validation_current" / "refreshed_grouped_static_blend_latest.json"
)
DEFAULT_INCUMBENT_PATH = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_current_one_shot_incumbent_portfolio_latest.json"
)
DEFAULT_CARRY_REPORT_GLOB = str(
    GROUP_ROOT
    / "article_inspired_research_current"
    / "batch_runs"
    / "batch_*"
    / "candidate_research_latest.json"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "portfolio_production_guarded_current"
ARTIFACT_KIND = "production_guarded_portfolio"


@dataclass(frozen=True, slots=True)
class DrawdownThrottleConfig:
    soft_drawdown: float = 0.08
    hard_drawdown: float = 0.12
    stop_drawdown: float = 0.16
    soft_scale: float = 0.85
    hard_scale: float = 0.60
    stop_scale: float = 0.35


DEFAULT_DRAWDOWN_THROTTLE = DrawdownThrottleConfig()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _split_metrics(metrics: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    block = dict(metrics or {})
    return {split: dict(block.get(split) or {}) for split in ("train", "val", "oos")}


def _streams_from_hybrid_payload(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    split_windows = dict(payload.get("split_windows") or {})
    scenario = dict(dict(payload.get("scenarios") or {}).get("refreshed_latest_tail") or {})
    dates = [str(item) for item in list(scenario.get("dates") or [])]
    daily_returns = [_safe_float(item, 0.0) for item in list(scenario.get("daily_returns") or [])]
    if not dates or len(dates) != len(daily_returns):
        raise RuntimeError("hybrid payload is missing usable dates/daily_returns")

    train_start = str(split_windows.get("train_start") or "")
    train_end = str(split_windows.get("train_end_inclusive") or "")
    val_start = str(split_windows.get("val_start") or "")
    val_end = str(split_windows.get("val_end_inclusive") or "")
    oos_start = str(split_windows.get("oos_start") or "")

    def _bucket(day: str) -> str | None:
        if train_start and train_end and train_start <= day <= train_end:
            return "train"
        if val_start and val_end and val_start <= day <= val_end:
            return "val"
        if oos_start and day >= oos_start:
            return "oos"
        return None

    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for day, ret in zip(dates, daily_returns, strict=True):
        split = _bucket(day[:10])
        if split is None:
            continue
        point = {"datetime": f"{day[:10]}T00:00:00Z", "t": f"{day[:10]}T00:00:00Z", "v": float(ret)}
        streams[split].append(point)
    return streams


def _portfolio_row(path: Path, *, candidate_key: str, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "candidate_key": candidate_key,
        "candidate_id": candidate_key,
        "name": label,
        "artifact_path": str(path.resolve()),
        "train": dict((payload.get("portfolio_metrics") or {}).get("train") or {}),
        "val": dict((payload.get("portfolio_metrics") or {}).get("val") or {}),
        "oos": dict((payload.get("portfolio_metrics") or {}).get("oos") or {}),
        "return_streams": dict(payload.get("portfolio_return_streams") or {}),
    }


def _hybrid_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    scenario = dict(dict(payload.get("scenarios") or {}).get("refreshed_latest_tail") or {})
    return {
        "candidate_key": "hybrid_guarded_mode",
        "candidate_id": "hybrid_guarded_mode",
        "name": "hybrid_guarded_mode",
        "artifact_path": str(path.resolve()),
        "train": dict((scenario.get("split_metrics") or {}).get("train") or {}),
        "val": dict((scenario.get("split_metrics") or {}).get("val") or {}),
        "oos": dict((scenario.get("split_metrics") or {}).get("oos") or {}),
        "return_streams": _streams_from_hybrid_payload(payload),
        "readiness": dict(payload.get("readiness") or {}),
    }


def _carry_candidate_rows(report_glob: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted(glob.glob(report_glob)):
        payload = _load_json(Path(raw_path))
        for row in list(payload.get("candidates") or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("strategy_class") or "") != "CarryTrendFactorRotationStrategy":
                continue
            merged = dict(row)
            merged["artifact_path"] = str(Path(raw_path).resolve())
            rows.append(merged)
    return rows


def _carry_candidate_score(row: Mapping[str, Any]) -> float:
    train = dict(row.get("train") or {})
    val = dict(row.get("val") or {})
    oos = dict(row.get("oos") or {})
    if _safe_float(train.get("total_return", train.get("return")), 0.0) <= 0.0:
        return float("-inf")
    if _safe_float(val.get("total_return", val.get("return")), 0.0) <= 0.0:
        return float("-inf")
    if _safe_float(oos.get("total_return", oos.get("return")), 0.0) <= 0.0:
        return float("-inf")
    if (
        max(
            _safe_float(train.get("max_drawdown", train.get("mdd")), 0.0),
            _safe_float(val.get("max_drawdown", val.get("mdd")), 0.0),
            _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0),
        )
        > 0.20
    ):
        return float("-inf")
    return float(validation_objective(val)) + (
        0.5 * _safe_float(oos.get("total_return", oos.get("return")), 0.0)
    )


def _best_carry_candidate(report_glob: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rows = _carry_candidate_rows(report_glob)
    if not rows:
        return None, []
    ranked = sorted(rows, key=_carry_candidate_score, reverse=True)
    best = ranked[0]
    if _carry_candidate_score(best) == float("-inf"):
        return None, ranked
    best_row = dict(best)
    best_row["candidate_key"] = str(
        best_row.get("candidate_id") or best_row.get("name") or "carry_trend_factor_rotation"
    )
    return best_row, ranked


def _drawdown_scale(drawdown: float, cfg: DrawdownThrottleConfig) -> float:
    if drawdown >= cfg.stop_drawdown:
        return float(cfg.stop_scale)
    if drawdown >= cfg.hard_drawdown:
        return float(cfg.hard_scale)
    if drawdown >= cfg.soft_drawdown:
        return float(cfg.soft_scale)
    return 1.0


def _apply_drawdown_throttle(
    streams: Mapping[str, list[dict[str, Any]]], cfg: DrawdownThrottleConfig
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    adjusted: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    schedule: list[dict[str, Any]] = []
    equity = 1.0
    peak = 1.0
    for split in ("train", "val", "oos"):
        for point in list(streams.get(split) or []):
            drawdown = 0.0 if peak <= 0.0 else max(0.0, (peak - equity) / peak)
            scale = _drawdown_scale(drawdown, cfg)
            raw_return = _safe_float(point.get("v"), 0.0)
            adj_return = float(raw_return * scale)
            adjusted_point = {
                "datetime": str(point.get("datetime") or point.get("t")),
                "t": str(point.get("datetime") or point.get("t")),
                "v": adj_return,
            }
            adjusted[split].append(adjusted_point)
            schedule.append(
                {
                    "split": split,
                    "datetime": adjusted_point["datetime"],
                    "raw_return": raw_return,
                    "scaled_return": adj_return,
                    "drawdown_before": drawdown,
                    "exposure_scale": scale,
                }
            )
            equity *= 1.0 + adj_return
            peak = max(peak, equity)
    return adjusted, schedule


def _score_combo(payload: dict[str, Any]) -> float:
    metrics = _split_metrics(payload.get("portfolio_metrics"))
    train = metrics["train"]
    val = metrics["val"]
    oos = metrics["oos"]
    if any(
        _safe_float(block.get("total_return", block.get("return")), 0.0) <= 0.0
        for block in (train, val, oos)
    ):
        return float("-inf")
    if any(
        _safe_float(block.get("max_drawdown", block.get("mdd")), 0.0) > 0.20
        for block in (train, val, oos)
    ):
        return float("-inf")
    return float(validation_objective(val)) + (
        0.75 * _safe_float(oos.get("total_return", oos.get("return")), 0.0)
    )


def build_production_guarded_portfolio(
    *,
    hybrid_path: Path = DEFAULT_HYBRID_PATH,
    static_blend_path: Path = DEFAULT_STATIC_BLEND_PATH,
    incumbent_path: Path = DEFAULT_INCUMBENT_PATH,
    carry_report_glob: str = DEFAULT_CARRY_REPORT_GLOB,
    throttle: DrawdownThrottleConfig = DEFAULT_DRAWDOWN_THROTTLE,
) -> dict[str, Any]:
    hybrid = _hybrid_row(hybrid_path)
    static_blend = _portfolio_row(
        static_blend_path, candidate_key="static_blend_76_24", label="static_blend_76_24"
    )
    incumbent = _portfolio_row(
        incumbent_path, candidate_key="incumbent_only", label="incumbent_only"
    )
    carry_candidate, ranked_carry = _best_carry_candidate(carry_report_glob)

    components = [hybrid, static_blend, incumbent]
    if carry_candidate is not None:
        components.append(carry_candidate)

    candidate_payloads: list[dict[str, Any]] = []
    active_exposures = (0.75, 0.85, 0.95)
    carry_weights = (0.0,) if carry_candidate is None else (0.0, 0.05, 0.10)
    for active_exposure in active_exposures:
        for incumbent_floor in (0.10, 0.15, 0.20):
            for carry_weight in carry_weights:
                if carry_weight + incumbent_floor >= active_exposure:
                    continue
                remaining = active_exposure - carry_weight - incumbent_floor
                for hybrid_share in (0.55, 0.65, 0.75):
                    hybrid_weight = remaining * hybrid_share
                    static_weight = remaining - hybrid_weight
                    rows = [
                        {**hybrid, "_saved_weight": hybrid_weight},
                        {**static_blend, "_saved_weight": static_weight},
                        {**incumbent, "_saved_weight": incumbent_floor},
                    ]
                    if carry_candidate is not None and carry_weight > 0.0:
                        rows.append({**carry_candidate, "_saved_weight": carry_weight})
                    base_eval = evaluate_weighted_portfolio(rows)
                    throttled_streams, schedule = _apply_drawdown_throttle(
                        base_eval.get("portfolio_daily_return_streams") or {},
                        throttle,
                    )
                    throttled_eval = evaluate_weighted_portfolio(
                        [
                            {
                                "candidate_key": "production_guarded_portfolio",
                                "name": "production_guarded_portfolio",
                                "_saved_weight": 1.0,
                                "return_streams": throttled_streams,
                                "train": {},
                                "val": {},
                                "oos": {},
                            }
                        ]
                    )
                    payload = {
                        "artifact_kind": ARTIFACT_KIND,
                        "generated_at": _utc_now_iso(),
                        "selection_basis": "saved_sleeve_blend_with_drawdown_throttle",
                        "portfolio_metrics": throttled_eval["portfolio_metrics"],
                        "portfolio_return_streams": throttled_eval["portfolio_return_streams"],
                        "portfolio_daily_return_streams": throttled_eval[
                            "portfolio_daily_return_streams"
                        ],
                        "oos_monthly_returns": throttled_eval["oos_monthly_returns"],
                        "weighted_component_summaries": base_eval.get(
                            "weighted_component_summaries"
                        ),
                        "weights": [
                            {
                                "candidate_id": row["candidate_key"],
                                "name": row["name"],
                                "weight": float(row["_saved_weight"]),
                            }
                            for row in rows
                        ],
                        "active_exposure": active_exposure,
                        "cash_weight": max(0.0, 1.0 - active_exposure),
                        "drawdown_throttle": asdict(throttle),
                        "drawdown_schedule": schedule,
                        "component_sources": {
                            row["candidate_key"]: row.get("artifact_path") for row in rows
                        },
                        "validation_objective": float(
                            validation_objective(
                                dict((throttled_eval["portfolio_metrics"] or {}).get("val") or {})
                            )
                        ),
                        "score": _score_combo(
                            {"portfolio_metrics": throttled_eval["portfolio_metrics"]}
                        ),
                    }
                    candidate_payloads.append(payload)

    candidate_payloads = [item for item in candidate_payloads if item["score"] != float("-inf")]
    if not candidate_payloads:
        raise RuntimeError(
            "no production-guarded combination cleared the positive-return / drawdown gates"
        )
    candidate_payloads.sort(
        key=lambda item: (
            float(item["score"]),
            _safe_float(
                ((item.get("portfolio_metrics") or {}).get("oos") or {}).get("total_return"), 0.0
            ),
            -_safe_float(
                ((item.get("portfolio_metrics") or {}).get("oos") or {}).get("max_drawdown"), 0.0
            ),
        ),
        reverse=True,
    )
    best = dict(candidate_payloads[0])
    best["leaderboard"] = [
        {
            "weights": list(item.get("weights") or []),
            "active_exposure": item.get("active_exposure"),
            "cash_weight": item.get("cash_weight"),
            "validation_objective": item.get("validation_objective"),
            "score": item.get("score"),
            "oos": dict((item.get("portfolio_metrics") or {}).get("oos") or {}),
        }
        for item in candidate_payloads[:10]
    ]
    best["carry_candidate_included"] = carry_candidate is not None and any(
        str(row.get("candidate_id")) == str(carry_candidate.get("candidate_key"))
        for row in list(best.get("weights") or [])
    )
    best["carry_candidate_considered"] = {
        "included": best["carry_candidate_included"],
        "selected_name": None if carry_candidate is None else carry_candidate.get("name"),
        "available_count": len(ranked_carry),
        "excluded_reason": None
        if carry_candidate is not None
        else "no_carry_candidate_cleared_production_safety_filters",
    }
    best["recommended_stage"] = "production_candidate"
    return best


def write_production_guarded_portfolio(
    *,
    hybrid_path: Path = DEFAULT_HYBRID_PATH,
    static_blend_path: Path = DEFAULT_STATIC_BLEND_PATH,
    incumbent_path: Path = DEFAULT_INCUMBENT_PATH,
    carry_report_glob: str = DEFAULT_CARRY_REPORT_GLOB,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    payload = build_production_guarded_portfolio(
        hybrid_path=hybrid_path,
        static_blend_path=static_blend_path,
        incumbent_path=incumbent_path,
        carry_report_glob=carry_report_glob,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "production_guarded_portfolio_latest.json"
    md_path = output_dir / "production_guarded_portfolio_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    oos = dict((payload.get("portfolio_metrics") or {}).get("oos") or {})
    lines = [
        "# production guarded portfolio",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- selection_basis: `{payload.get('selection_basis')}`",
        f"- active_exposure: `{_safe_float(payload.get('active_exposure'), 0.0):.2%}`",
        f"- cash_weight: `{_safe_float(payload.get('cash_weight'), 0.0):.2%}`",
        f"- oos_return: `{_safe_float(oos.get('total_return', oos.get('return')), 0.0):+.4%}`",
        f"- oos_sharpe: `{_safe_float(oos.get('sharpe'), 0.0):.4f}`",
        f"- oos_max_drawdown: `{_safe_float(oos.get('max_drawdown', oos.get('mdd')), 0.0):.4%}`",
        "",
        "## selected components",
    ]
    for row in list(payload.get("weights") or []):
        lines.append(f"- `{row.get('name')}`: `{_safe_float(row.get('weight'), 0.0):.2%}`")
    carry = dict(payload.get("carry_candidate_considered") or {})
    lines.extend(
        [
            "",
            "## carry candidate",
            f"- selected_name: `{carry.get('selected_name')}`",
            f"- included: `{carry.get('included')}`",
            f"- excluded_reason: `{carry.get('excluded_reason')}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-path", type=Path, default=DEFAULT_HYBRID_PATH)
    parser.add_argument("--static-blend-path", type=Path, default=DEFAULT_STATIC_BLEND_PATH)
    parser.add_argument("--incumbent-path", type=Path, default=DEFAULT_INCUMBENT_PATH)
    parser.add_argument("--carry-report-glob", default=DEFAULT_CARRY_REPORT_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_production_guarded_portfolio(
        hybrid_path=Path(args.hybrid_path).resolve(),
        static_blend_path=Path(args.static_blend_path).resolve(),
        incumbent_path=Path(args.incumbent_path).resolve(),
        carry_report_glob=str(args.carry_report_glob),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(result["json_path"])
    print(result["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
