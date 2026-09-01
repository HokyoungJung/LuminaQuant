"""Build monthly attribution tables for grouped incumbent/55-45 regime research."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT, resolve_current_optimization_path

_hard_spec = importlib.util.spec_from_file_location(
    "run_three_way_market_regime_allocator",
    Path(__file__).resolve().parent / "run_three_way_market_regime_allocator.py",
)
if _hard_spec is None or _hard_spec.loader is None:
    raise RuntimeError("Failed to load run_three_way_market_regime_allocator helpers")
_hard = importlib.util.module_from_spec(_hard_spec)
sys.modules[_hard_spec.name] = _hard
_hard_spec.loader.exec_module(_hard)

DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped" / "monthly_attribution_current"
)
DEFAULT_BLEND_PATH = _hard.DEFAULT_BLEND_PATH
DEFAULT_HARD_ALLOCATOR_PATH = (
    _hard.DEFAULT_OUTPUT_DIR / "three_way_market_regime_allocator_latest.json"
)
DEFAULT_SOFT_ALLOCATOR_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "soft_three_way_market_regime_allocator_current"
    / "soft_three_way_market_regime_allocator_latest.json"
)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _load_allocator_returns(path: Path, *, label: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(payload.get("dates") or [], utc=True).floor("D"),
            label: [float(value) for value in list(payload.get("daily_returns") or [])],
        }
    )
    if frame.empty:
        raise ValueError(f"{label}: missing daily returns in {path}")
    frame["split_group"] = frame["date"].map(
        lambda day: _hard.split_for_date(pd.Timestamp(day).date())
    )
    return frame


def _monthly_return_table(panel: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    frame = panel.copy()
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (split_group, month), sample in frame.groupby(["split_group", "month"], sort=True):
        row: dict[str, Any] = {"split_group": split_group, "month": month}
        best_label = None
        best_value = None
        for column in value_columns:
            monthly_return = float((1.0 + sample[column].astype(float)).prod() - 1.0)
            row[column] = monthly_return
            if best_value is None or monthly_return > best_value:
                best_label = column
                best_value = monthly_return
        row["winner"] = best_label
        row["winner_return"] = float(best_value or 0.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_group", "month"]).reset_index(drop=True)


def _monthly_state_summary(path: Path, *, label: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload.get("states") or [])
    if rows.empty:
        return pd.DataFrame(columns=["month", f"{label}_dominant_state", f"{label}_avg_turnover"])
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    rows["month"] = rows["date"].dt.strftime("%Y-%m")
    summary: list[dict[str, Any]] = []
    for month, sample in rows.groupby("month", sort=True):
        dominant = sample["state"].value_counts().idxmax()
        summary.append(
            {
                "month": month,
                f"{label}_dominant_state": str(dominant),
                f"{label}_avg_turnover": float(sample["turnover"].astype(float).mean()),
            }
        )
    return pd.DataFrame(summary)


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Grouped Regime Monthly Attribution",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "",
        "## Latest comparison summary",
    ]
    for item in payload["summary_rows"]:
        lines.append(
            f"- {item['label']}: train `{item['train_return']:.4%}` | val `{item['val_return']:.4%}` | oos `{item['oos_return']:.4%}` | oos_sharpe `{item['oos_sharpe']:.4f}`"
        )
    lines.extend(["", "## Monthly table"])
    lines.append(
        "| split | month | incumbent | blend_85_15 | autoresearch_55_45 | hard_allocator | soft_allocator | winner |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in payload["monthly_rows"]:
        lines.append(
            f"| {row['split_group']} | {row['month']} | {row['incumbent']:.4%} | {row['blend_85_15']:.4%} | {row['autoresearch_55_45']:.4%} | {row['hard_allocator']:.4%} | {row['soft_allocator']:.4%} | {row['winner']} |"
        )
    lines.extend(["", "## Monthly allocator state summaries"])
    for row in payload["state_rows"]:
        lines.append(
            f"- {row['month']}: hard `{row.get('hard_dominant_state', 'n/a')}` turnover `{row.get('hard_avg_turnover', 0.0):.4f}` | soft `{row.get('soft_dominant_state', 'n/a')}` turnover `{row.get('soft_avg_turnover', 0.0):.4f}`"
        )
    return "\n".join(lines).strip() + "\n"


def run_grouped_regime_monthly_attribution(
    *,
    incumbent_path: Path,
    blend_path: Path,
    autoresearch_path: Path,
    hard_allocator_path: Path,
    soft_allocator_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    incumbent = _hard._load_candidate_frame(label="incumbent", path=incumbent_path).rename(
        columns={"return": "incumbent"}
    )
    blend = _hard._load_candidate_frame(label="blend_85_15", path=blend_path).rename(
        columns={"return": "blend_85_15"}
    )
    autoresearch = _hard._load_candidate_frame(
        label="autoresearch_55_45", path=autoresearch_path
    ).rename(columns={"return": "autoresearch_55_45"})
    hard_alloc = _load_allocator_returns(hard_allocator_path, label="hard_allocator")
    soft_alloc = _load_allocator_returns(soft_allocator_path, label="soft_allocator")
    panel = (
        incumbent[["date", "split_group", "incumbent"]]
        .merge(blend[["date", "blend_85_15"]], on="date", how="inner")
        .merge(autoresearch[["date", "autoresearch_55_45"]], on="date", how="inner")
        .merge(hard_alloc[["date", "hard_allocator"]], on="date", how="inner")
        .merge(soft_alloc[["date", "soft_allocator"]], on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    value_columns = [
        "incumbent",
        "blend_85_15",
        "autoresearch_55_45",
        "hard_allocator",
        "soft_allocator",
    ]
    monthly = _monthly_return_table(panel, value_columns=value_columns)
    hard_state = _monthly_state_summary(hard_allocator_path, label="hard")
    soft_state = _monthly_state_summary(soft_allocator_path, label="soft")
    state_rows = (
        monthly[["month"]]
        .drop_duplicates()
        .merge(hard_state, on="month", how="left")
        .merge(soft_state, on="month", how="left")
        .sort_values("month")
        .to_dict(orient="records")
    )

    summary_rows = []
    for label in value_columns:
        metrics = _hard._metrics_by_split(
            panel.rename(columns={label: "metric_return"}), "metric_return"
        )
        summary_rows.append(
            {
                "label": label,
                "train_return": float(metrics["train"]["total_return"]),
                "val_return": float(metrics["val"]["total_return"]),
                "oos_return": float(metrics["oos"]["total_return"]),
                "oos_sharpe": float(metrics["oos"]["sharpe"]),
            }
        )

    payload = {
        "artifact_kind": "grouped_regime_monthly_attribution",
        "generated_at": _utc_now_iso(),
        "input_paths": {
            "incumbent": str(incumbent_path.resolve()),
            "blend_85_15": str(blend_path.resolve()),
            "autoresearch_55_45": str(autoresearch_path.resolve()),
            "hard_allocator": str(hard_allocator_path.resolve()),
            "soft_allocator": str(soft_allocator_path.resolve()),
        },
        "summary_rows": summary_rows,
        "monthly_rows": monthly.to_dict(orient="records"),
        "state_rows": state_rows,
    }

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"grouped_regime_monthly_attribution_{timestamp}.json"
    out_md = output_dir / f"grouped_regime_monthly_attribution_{timestamp}.md"
    latest_json = output_dir / "grouped_regime_monthly_attribution_latest.json"
    latest_md = output_dir / "grouped_regime_monthly_attribution_latest.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"payload": payload, "latest_json_path": latest_json, "latest_md_path": latest_md}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-path", type=Path, default=resolve_current_optimization_path())
    parser.add_argument("--blend-path", type=Path, default=DEFAULT_BLEND_PATH)
    parser.add_argument(
        "--autoresearch-path", type=Path, default=_hard._resolve_autoresearch_default_path()
    )
    parser.add_argument("--hard-allocator-path", type=Path, default=DEFAULT_HARD_ALLOCATOR_PATH)
    parser.add_argument("--soft-allocator-path", type=Path, default=DEFAULT_SOFT_ALLOCATOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_grouped_regime_monthly_attribution(
        incumbent_path=Path(args.incumbent_path).resolve(),
        blend_path=Path(args.blend_path).resolve(),
        autoresearch_path=Path(args.autoresearch_path).resolve(),
        hard_allocator_path=Path(args.hard_allocator_path).resolve(),
        soft_allocator_path=Path(args.soft_allocator_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
