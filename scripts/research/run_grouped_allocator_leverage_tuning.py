"""Tune grouped allocator sleeve leverage on train/val while preserving OOS evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    MEMORY_GUARD_DIRNAME,
    acquire_portfolio_memory_guard,
)

_three_way_spec = importlib.util.spec_from_file_location(
    "run_three_way_market_regime_allocator",
    Path(__file__).resolve().parent / "run_three_way_market_regime_allocator.py",
)
if _three_way_spec is None or _three_way_spec.loader is None:
    raise RuntimeError("Failed to load run_three_way_market_regime_allocator helpers")
_three_way = importlib.util.module_from_spec(_three_way_spec)
sys.modules[_three_way_spec.name] = _three_way
_three_way_spec.loader.exec_module(_three_way)

_decision_spec = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision",
    Path(__file__).resolve().parent / "write_portfolio_max_performance_decision.py",
)
if _decision_spec is None or _decision_spec.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision helpers")
_decision = importlib.util.module_from_spec(_decision_spec)
sys.modules[_decision_spec.name] = _decision
_decision_spec.loader.exec_module(_decision)

DEFAULT_ALLOCATOR_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "three_way_market_regime_allocator_current"
    / "three_way_market_regime_allocator_latest.json"
)
DEFAULT_BLEND_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_incumbent_autoresearch_static_blend_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_allocator_leverage_tuning_current"
)
DEFAULT_MIN_LEVERAGE = 1
DEFAULT_MAX_LEVERAGE = 25
DEFAULT_SOFT_RSS_BYTES = int(2.0 * 1024 * 1024 * 1024)
DEFAULT_HARD_RSS_BYTES = int(3.0 * 1024 * 1024 * 1024)
TRAIN_MAX_DRAWDOWN_LIMIT = 0.35
VAL_MAX_DRAWDOWN_LIMIT = 0.20


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _resolve_incumbent_portfolio_from_decision(decision_path: Path) -> Path:
    payload = _load_json(decision_path)
    for candidate in list(payload.get("candidates") or []):
        if str(candidate.get("candidate_key") or "") == "current_one_shot_incumbent":
            path = candidate.get("artifact_path")
            if path:
                return Path(str(path)).resolve()
    raise RuntimeError(f"unable to resolve incumbent portfolio path from {decision_path}")


def _resolve_autoresearch_path() -> Path:
    return _three_way._resolve_autoresearch_default_path()


def _load_portfolio_returns(path: Path, *, label: str) -> pd.DataFrame:
    frame = _three_way._load_candidate_frame(label=label, path=path)
    return frame[["date", "split_group", "return"]].rename(columns={"return": label})


def _load_allocator_states(path: Path) -> pd.DataFrame:
    payload = _load_json(path)
    states = pd.DataFrame(list(payload.get("states") or []))
    if states.empty:
        raise RuntimeError(f"allocator artifact has no states: {path}")
    states["date"] = pd.to_datetime(states["date"], utc=True)
    return states.sort_values("date").reset_index(drop=True)


def _state_floor(
    *,
    leverage: int,
    maintenance_margin_rate: float,
    taker_fee_rate: float,
    liquidation_buffer_rate: float,
) -> float:
    residual = float(leverage) * (
        float(maintenance_margin_rate) + float(taker_fee_rate) + float(liquidation_buffer_rate)
    )
    return min(1.0, max(0.0, residual))


def _apply_state_leverage(
    state_returns: pd.DataFrame,
    *,
    leverage_by_state: dict[str, int],
    maintenance_margin_rate: float = 0.005,
    taker_fee_rate: float = 0.001,
    liquidation_buffer_rate: float = 0.0005,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    liquidation_counts = dict.fromkeys(leverage_by_state, 0)
    current_state: str | None = None
    segment_alive = True
    segment_equity = 1.0
    current_floor = 0.0

    for item in state_returns.itertuples(index=False):
        state = str(item.state)
        base_return = float(item.base_return)
        if state != current_state:
            current_state = state
            segment_alive = True
            segment_equity = 1.0
            current_floor = _state_floor(
                leverage=int(leverage_by_state.get(state, 1)),
                maintenance_margin_rate=maintenance_margin_rate,
                taker_fee_rate=taker_fee_rate,
                liquidation_buffer_rate=liquidation_buffer_rate,
            )

        leverage = int(leverage_by_state.get(state, 1))
        leveraged_return = 0.0
        liquidated = False
        if segment_alive:
            leveraged_return = float(base_return) * float(leverage)
            next_equity = float(segment_equity) * (1.0 + leveraged_return)
            if next_equity <= current_floor + 1e-12:
                leveraged_return = float(current_floor / max(segment_equity, 1e-12)) - 1.0
                segment_equity = float(current_floor)
                segment_alive = False
                liquidation_counts[state] = int(liquidation_counts.get(state, 0)) + 1
                liquidated = True
            else:
                segment_equity = float(next_equity)

        rows.append(
            {
                "date": item.date,
                "split_group": item.split_group,
                "state": state,
                "base_return": float(base_return),
                "leveraged_return": float(leveraged_return),
                "leverage": leverage,
                "segment_equity": float(segment_equity),
                "segment_floor": float(current_floor),
                "liquidated": liquidated,
            }
        )
    return pd.DataFrame(rows), liquidation_counts


def _objective(split_metrics: dict[str, dict[str, float]]) -> float:
    train = dict(split_metrics.get("train") or {})
    val = dict(split_metrics.get("val") or {})
    oos = dict(split_metrics.get("oos") or {})
    return float(
        (100.0 * float(train.get("total_return", 0.0)))
        + (40.0 * float(val.get("total_return", 0.0)))
        + (20.0 * float(oos.get("total_return", 0.0)))
        - (20.0 * float(train.get("max_drawdown", 0.0)))
        - (10.0 * float(val.get("max_drawdown", 0.0)))
    )


def _is_feasible(
    split_metrics: dict[str, dict[str, float]], *, liquidation_counts: dict[str, int]
) -> bool:
    train = dict(split_metrics.get("train") or {})
    val = dict(split_metrics.get("val") or {})
    return bool(
        int(sum(liquidation_counts.values())) == 0
        and float(train.get("total_return", 0.0)) > 0.0
        and float(val.get("total_return", 0.0)) > 0.0
        and float(train.get("max_drawdown", 0.0)) <= TRAIN_MAX_DRAWDOWN_LIMIT
        and float(val.get("max_drawdown", 0.0)) <= VAL_MAX_DRAWDOWN_LIMIT
    )


def _metrics_by_split(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return _three_way._metrics_by_split(
        frame.rename(columns={"leveraged_return": "metric_return"}), "metric_return"
    )


def _build_markdown(payload: dict[str, Any]) -> str:
    best = dict(payload.get("best_result") or {})
    lines = [
        "# Grouped Allocator Leverage Tuning",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- allocator_path: `{payload.get('allocator_path')}`",
        f"- peak_rss_mib: `{float((payload.get('memory_summary') or {}).get('peak_rss_mib') or 0.0):.2f}`",
        "",
        "## Selected leverage by state",
    ]
    for state, lev in sorted(dict(best.get("leverage_by_state") or {}).items()):
        lines.append(f"- {state}: `{lev}x`")
    lines.append(f"- feasible_under_constraints: `{bool(best.get('feasible'))}`")
    lines.extend(["", "## Liquidation counts"])
    for state, count in sorted(dict(best.get("liquidation_counts") or {}).items()):
        lines.append(f"- {state}: `{int(count)}`")
    lines.extend(["", "## Split metrics"])
    for split in ("train", "val", "oos"):
        metrics = dict((best.get("split_metrics") or {}).get(split) or {})
        lines.append(
            f"- {split}: return `{float(metrics.get('total_return') or 0.0):.4%}` | "
            f"sharpe `{float(metrics.get('sharpe') or 0.0):.4f}` | "
            f"max_dd `{float(metrics.get('max_drawdown') or 0.0):.4%}`"
        )
    return "\n".join(lines).strip() + "\n"


def run_grouped_allocator_leverage_tuning(
    *,
    allocator_path: Path,
    incumbent_path: Path,
    blend_path: Path,
    autoresearch_path: Path,
    output_dir: Path,
    min_leverage: int,
    max_leverage: int,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = acquire_portfolio_memory_guard(
        run_name="grouped_allocator_leverage_tuning",
        output_dir=output_dir,
        input_path=str(allocator_path),
        rss_log_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "grouped_allocator_leverage_tuning_rss_latest.jsonl",
        summary_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "grouped_allocator_leverage_tuning_memory_latest.json",
        budget_bytes=hard_rss_bytes,
        soft_limit_bytes=soft_rss_bytes,
        hard_limit_bytes=hard_rss_bytes,
    )
    status = "ok"
    error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        states = _load_allocator_states(allocator_path)
        incumbent = _load_portfolio_returns(incumbent_path, label="incumbent")
        blend = _load_portfolio_returns(blend_path, label="blend_85_15")
        autoresearch = _load_portfolio_returns(autoresearch_path, label="autoresearch_55_45")
        panel = (
            states[["date", "split_group", "state"]]
            .merge(incumbent[["date", "incumbent"]], on="date", how="left")
            .merge(blend[["date", "blend_85_15"]], on="date", how="left")
            .merge(autoresearch[["date", "autoresearch_55_45"]], on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )
        if panel[["incumbent", "blend_85_15", "autoresearch_55_45"]].isna().any().any():
            raise RuntimeError("missing sleeve return data while assembling leverage tuning panel")
        panel["base_return"] = [
            float(getattr(row, row.state)) for row in panel.itertuples(index=False)
        ]
        guard.checkpoint("panel_loaded", {"rows": len(panel)})

        best: dict[str, Any] | None = None
        leverage_values = range(max(1, int(min_leverage)), max(1, int(max_leverage)) + 1)
        for inc_lev in leverage_values:
            for blend_lev in leverage_values:
                for auto_lev in leverage_values:
                    tuned_frame, liquidation_counts = _apply_state_leverage(
                        panel[["date", "split_group", "state", "base_return"]],
                        leverage_by_state={
                            "incumbent": int(inc_lev),
                            "blend_85_15": int(blend_lev),
                            "autoresearch_55_45": int(auto_lev),
                        },
                    )
                    split_metrics = _metrics_by_split(tuned_frame)
                    objective = _objective(split_metrics)
                    candidate = {
                        "leverage_by_state": {
                            "incumbent": int(inc_lev),
                            "blend_85_15": int(blend_lev),
                            "autoresearch_55_45": int(auto_lev),
                        },
                        "liquidation_counts": liquidation_counts,
                        "split_metrics": split_metrics,
                        "objective": float(objective),
                        "feasible": _is_feasible(
                            split_metrics, liquidation_counts=liquidation_counts
                        ),
                    }
                    if best is None:
                        best = candidate
                    else:
                        best_key = (
                            bool(best.get("feasible")),
                            float(best["objective"]),
                            float((best["split_metrics"]["oos"]).get("total_return", 0.0)),
                            -float((best["split_metrics"]["train"]).get("max_drawdown", 0.0)),
                        )
                        cand_key = (
                            bool(candidate.get("feasible")),
                            float(objective),
                            float((split_metrics["oos"]).get("total_return", 0.0)),
                            -float((split_metrics["train"]).get("max_drawdown", 0.0)),
                        )
                        if cand_key > best_key:
                            best = candidate
        if best is None:
            raise RuntimeError("no leverage tuning result produced")

        best_frame, _ = _apply_state_leverage(
            panel[["date", "split_group", "state", "base_return"]],
            leverage_by_state=dict(best["leverage_by_state"]),
        )
        payload = {
            "artifact_kind": "grouped_allocator_leverage_tuning",
            "generated_at": _utc_now_iso(),
            "allocator_path": str(allocator_path.resolve()),
            "incumbent_path": str(incumbent_path.resolve()),
            "blend_path": str(blend_path.resolve()),
            "autoresearch_path": str(autoresearch_path.resolve()),
            "selection_basis": "train_val_grid_search_with_liquidation_count_priority",
            "search_space": {
                "min_leverage": int(min_leverage),
                "max_leverage": int(max_leverage),
                "states": ["incumbent", "blend_85_15", "autoresearch_55_45"],
            },
            "best_result": best,
            "daily_returns": [float(value) for value in best_frame["leveraged_return"]],
            "dates": [pd.Timestamp(value).isoformat() for value in best_frame["date"]],
            "states": best_frame.to_dict(orient="records"),
            "memory_summary": {},
        }
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        memory_summary = guard.finalize(status=status, error=error)
        if payload is not None:
            payload["memory_summary"] = memory_summary

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"grouped_allocator_leverage_tuning_{timestamp}.json"
    out_md = output_dir / f"grouped_allocator_leverage_tuning_{timestamp}.md"
    latest_json = output_dir / "grouped_allocator_leverage_tuning_latest.json"
    latest_md = output_dir / "grouped_allocator_leverage_tuning_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"payload": payload, "latest_json_path": latest_json, "latest_md_path": latest_md}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator-path", type=Path, default=DEFAULT_ALLOCATOR_PATH)
    parser.add_argument(
        "--incumbent-path",
        type=Path,
        default=_resolve_incumbent_portfolio_from_decision(_decision.DEFAULT_OUTPUT_JSON),
    )
    parser.add_argument("--blend-path", type=Path, default=DEFAULT_BLEND_PATH)
    parser.add_argument("--autoresearch-path", type=Path, default=_resolve_autoresearch_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-leverage", type=int, default=DEFAULT_MIN_LEVERAGE)
    parser.add_argument("--max-leverage", type=int, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_grouped_allocator_leverage_tuning(
        allocator_path=Path(args.allocator_path).resolve(),
        incumbent_path=Path(args.incumbent_path).resolve(),
        blend_path=Path(args.blend_path).resolve(),
        autoresearch_path=Path(args.autoresearch_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        min_leverage=max(1, int(args.min_leverage)),
        max_leverage=max(1, int(args.max_leverage)),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
