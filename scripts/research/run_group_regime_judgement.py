"""Learn a low-memory regime judgement between the incumbent and 55/45 group."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    MEMORY_GUARD_DIRNAME,
    ROOT,
    acquire_portfolio_memory_guard,
    resolve_current_optimization_path,
    resolve_followup_artifact_path,
    split_for_date,
)

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped" / "regime_judgement_current"
)
DEFAULT_HORIZON_DAYS = 5
DEFAULT_SOFT_RSS_BYTES = int(7.5 * 1024 * 1024 * 1024)
DEFAULT_HARD_RSS_BYTES = int(8.0 * 1024 * 1024 * 1024)
MIN_TRAIN_COUNT = 20
MIN_VAL_COUNT = 6
MIN_WIN_RATE = 0.52
MIXED_MARGIN_THRESHOLD = 0.20

BOOLEAN_FEATURE_LABELS = {
    "autoresearch_leading_5d": "55/45 trailing 5-day return above incumbent",
    "autoresearch_leading_20d": "55/45 trailing 20-day return above incumbent",
    "autoresearch_vol_higher_20d": "55/45 trailing 20-day vol above incumbent",
    "autoresearch_drawdown_worse_20d": "55/45 trailing 20-day drawdown worse than incumbent",
    "relative_hit_rate_ge_55": "55/45 won at least 55% of trailing 20 days",
}

CONTINUOUS_FEATURE_LABELS = {
    "incumbent_ret_5d": "incumbent trailing 5-day return",
    "incumbent_ret_20d": "incumbent trailing 20-day return",
    "autoresearch_ret_5d": "55/45 trailing 5-day return",
    "autoresearch_ret_20d": "55/45 trailing 20-day return",
    "rel_ret_5d": "55/45 minus incumbent trailing 5-day return",
    "rel_ret_20d": "55/45 minus incumbent trailing 20-day return",
    "incumbent_vol_20d": "incumbent trailing 20-day vol",
    "autoresearch_vol_20d": "55/45 trailing 20-day vol",
    "rel_vol_ratio_20d": "55/45 / incumbent trailing 20-day vol ratio",
    "incumbent_drawdown_20d": "incumbent trailing 20-day max drawdown",
    "autoresearch_drawdown_20d": "55/45 trailing 20-day max drawdown",
    "rel_drawdown_20d": "55/45 minus incumbent trailing 20-day max drawdown",
    "relative_hit_rate_20d": "55/45 trailing 20-day daily win rate",
}


@dataclass(slots=True)
class GroupPortfolio:
    """Saved portfolio artifact plus normalized daily returns."""

    label: str
    path: Path
    weights: list[dict[str, Any]]
    returns: pd.DataFrame
    symbols: list[str]


@dataclass(slots=True)
class RuleCandidate:
    """Serializable candidate condition for regime selection."""

    rule_id: str
    label: str
    family: str
    feature_names: tuple[str, ...]
    threshold: float | None
    comparator: str | None
    polarity: str


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_autoresearch_default_path() -> Path:
    candidate = resolve_followup_artifact_path(
        "var/reports/exact_window_backtests/followup_status/"
        "autoresearch_candidate_portfolio_opt/portfolio_optimization_latest.json"
    )
    if not candidate.exists():
        raise FileNotFoundError("unable to locate autoresearch 55/45 portfolio artifact")
    return candidate.resolve()


def _load_group_portfolio(*, label: str, path: Path) -> GroupPortfolio:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = list(payload.get("weights") or [])
    if not weights:
        raise ValueError(f"{label}: missing weights in {path}")
    frames: list[pd.DataFrame] = []
    for split, rows in dict(payload.get("portfolio_return_streams") or {}).items():
        frame = pd.DataFrame(list(rows))
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["t"], utc=True).dt.floor("D")
        frame["split"] = str(split)
        frame["return"] = frame["v"].astype(float)
        frames.append(frame[["date", "split", "return"]])
    if not frames:
        raise ValueError(f"{label}: missing portfolio_return_streams in {path}")
    returns = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    symbols = sorted({str(symbol) for row in weights for symbol in list(row.get("symbols") or [])})
    return GroupPortfolio(
        label=label, path=path.resolve(), weights=weights, returns=returns, symbols=symbols
    )


def _forward_compound(series: pd.Series, horizon_days: int) -> pd.Series:
    values = np.asarray(series, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    gross = 1.0 + values
    if horizon_days <= 0:
        return pd.Series(out, index=series.index, dtype=float)
    for index in range(max(0, values.size - horizon_days)):
        out[index] = float(np.prod(gross[index + 1 : index + 1 + horizon_days]) - 1.0)
    return pd.Series(out, index=series.index, dtype=float)


def _split_group(day_value: pd.Timestamp) -> str:
    return split_for_date(day_value.date())


def _backward_compound(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series.astype(float)).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0


def _rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    def _window_drawdown(values: np.ndarray) -> float:
        equity = np.cumprod(1.0 + values.astype(float))
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / np.where(peaks == 0.0, 1.0, peaks) - 1.0
        return float(drawdown.min())

    return (
        series.astype(float).rolling(window, min_periods=window).apply(_window_drawdown, raw=True)
    )


def _build_performance_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for label in ("incumbent", "autoresearch"):
        out[f"{label}_ret_5d"] = _backward_compound(out[label], 5)
        out[f"{label}_ret_20d"] = _backward_compound(out[label], 20)
        out[f"{label}_vol_20d"] = out[label].astype(float).rolling(20, min_periods=20).std(
            ddof=0
        ) * math.sqrt(20.0)
        out[f"{label}_drawdown_20d"] = _rolling_max_drawdown(out[label], 20)

    out["rel_ret_5d"] = out["autoresearch_ret_5d"] - out["incumbent_ret_5d"]
    out["rel_ret_20d"] = out["autoresearch_ret_20d"] - out["incumbent_ret_20d"]
    out["rel_vol_ratio_20d"] = np.where(
        out["incumbent_vol_20d"].astype(float) > 0.0,
        out["autoresearch_vol_20d"].astype(float) / out["incumbent_vol_20d"].astype(float),
        np.nan,
    )
    out["rel_drawdown_20d"] = out["autoresearch_drawdown_20d"].astype(float) - out[
        "incumbent_drawdown_20d"
    ].astype(float)
    relative_daily_edge = (
        out["autoresearch"].astype(float) > out["incumbent"].astype(float)
    ).astype(float)
    out["relative_hit_rate_20d"] = relative_daily_edge.rolling(20, min_periods=20).mean()

    out["autoresearch_leading_5d"] = out["rel_ret_5d"].astype(float) > 0.0
    out["autoresearch_leading_20d"] = out["rel_ret_20d"].astype(float) > 0.0
    out["autoresearch_vol_higher_20d"] = out["rel_vol_ratio_20d"].astype(float) > 1.0
    out["autoresearch_drawdown_worse_20d"] = out["rel_drawdown_20d"].astype(float) < 0.0
    out["relative_hit_rate_ge_55"] = out["relative_hit_rate_20d"].astype(float) >= 0.55
    return out


def _build_rule_mask(frame: pd.DataFrame, candidate: RuleCandidate) -> pd.Series:
    if candidate.family == "bool":
        base = frame[candidate.feature_names[0]].fillna(False).astype(bool)
        return ~base if candidate.polarity == "negated" else base
    if candidate.family == "combo":
        values = [frame[name].fillna(False).astype(bool) for name in candidate.feature_names]
        base = values[0]
        for value in values[1:]:
            base = base & value
        return ~base if candidate.polarity == "negated" else base
    feature = candidate.feature_names[0]
    threshold = float(candidate.threshold or 0.0)
    series = frame[feature].astype(float)
    if candidate.comparator == "ge":
        return series >= threshold
    return series <= threshold


def _candidate_rules(train_val: pd.DataFrame) -> list[RuleCandidate]:
    rules: list[RuleCandidate] = []
    for feature_name, label in BOOLEAN_FEATURE_LABELS.items():
        rules.append(
            RuleCandidate(
                rule_id=feature_name,
                label=label,
                family="bool",
                feature_names=(feature_name,),
                threshold=None,
                comparator=None,
                polarity="normal",
            )
        )
        rules.append(
            RuleCandidate(
                rule_id=f"not_{feature_name}",
                label=f"NOT {label}",
                family="bool",
                feature_names=(feature_name,),
                threshold=None,
                comparator=None,
                polarity="negated",
            )
        )

    combo_rules = (
        ("autoresearch_leading_5d", "autoresearch_leading_20d"),
        ("autoresearch_leading_5d", "autoresearch_vol_higher_20d"),
        ("autoresearch_leading_20d", "relative_hit_rate_ge_55"),
        ("autoresearch_leading_20d", "autoresearch_drawdown_worse_20d"),
        ("autoresearch_leading_5d", "relative_hit_rate_ge_55"),
    )
    for feature_names in combo_rules:
        label = " + ".join(BOOLEAN_FEATURE_LABELS[name] for name in feature_names)
        rules.append(
            RuleCandidate(
                rule_id="__".join(feature_names),
                label=label,
                family="combo",
                feature_names=tuple(feature_names),
                threshold=None,
                comparator=None,
                polarity="normal",
            )
        )

    quantiles = {}
    for feature_name in CONTINUOUS_FEATURE_LABELS:
        series = train_val[feature_name].dropna().astype(float)
        if series.empty:
            continue
        lo = float(series.quantile(0.35))
        hi = float(series.quantile(0.65))
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        quantiles[feature_name] = {"lo": lo, "hi": hi}

    for feature_name, label in CONTINUOUS_FEATURE_LABELS.items():
        thresholds = quantiles.get(feature_name)
        if thresholds is None:
            continue
        rules.append(
            RuleCandidate(
                rule_id=f"{feature_name}_ge_q65",
                label=f"{label} >= train+val q65 ({thresholds['hi']:.4f})",
                family="continuous",
                feature_names=(feature_name,),
                threshold=thresholds["hi"],
                comparator="ge",
                polarity="normal",
            )
        )
        rules.append(
            RuleCandidate(
                rule_id=f"{feature_name}_le_q35",
                label=f"{label} <= train+val q35 ({thresholds['lo']:.4f})",
                family="continuous",
                feature_names=(feature_name,),
                threshold=thresholds["lo"],
                comparator="le",
                polarity="normal",
            )
        )
    return rules


def _rule_split_stats(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    forward_column: str,
    incumbent_column: str,
    autoresearch_column: str,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "oos"):
        sample = frame.loc[
            mask
            & frame["split_group"].eq(split)
            & frame[forward_column].notna()
            & frame[incumbent_column].notna()
            & frame[autoresearch_column].notna()
        ].copy()
        if sample.empty:
            stats[split] = {
                "count": 0,
                "mean_rel_delta": 0.0,
                "autoresearch_forward_mean": 0.0,
                "incumbent_forward_mean": 0.0,
                "autoresearch_win_rate": 0.0,
                "incumbent_win_rate": 0.0,
            }
            continue
        rel_delta = sample[forward_column].astype(float)
        stats[split] = {
            "count": len(sample),
            "mean_rel_delta": float(rel_delta.mean()),
            "autoresearch_forward_mean": float(sample[autoresearch_column].astype(float).mean()),
            "incumbent_forward_mean": float(sample[incumbent_column].astype(float).mean()),
            "autoresearch_win_rate": float((rel_delta > 0.0).mean()),
            "incumbent_win_rate": float((rel_delta < 0.0).mean()),
        }
    return stats


def _select_rules(
    frame: pd.DataFrame, *, horizon_days: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_val = frame.loc[frame["split_group"].isin(("train", "val"))].copy()
    candidates = _candidate_rules(train_val)
    forward_column = f"forward_{horizon_days}d_rel"
    incumbent_column = f"forward_{horizon_days}d_incumbent"
    autoresearch_column = f"forward_{horizon_days}d_autoresearch"

    selected_by_side: dict[str, list[dict[str, Any]]] = {"incumbent": [], "autoresearch": []}
    diagnostics: list[dict[str, Any]] = []

    for candidate in candidates:
        mask = _build_rule_mask(frame, candidate)
        split_stats = _rule_split_stats(
            frame,
            mask=mask,
            forward_column=forward_column,
            incumbent_column=incumbent_column,
            autoresearch_column=autoresearch_column,
        )
        train_stats = split_stats["train"]
        val_stats = split_stats["val"]
        if int(train_stats["count"]) < MIN_TRAIN_COUNT or int(val_stats["count"]) < MIN_VAL_COUNT:
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "insufficient_support",
                    "split_stats": split_stats,
                }
            )
            continue

        train_mean = float(train_stats["mean_rel_delta"])
        val_mean = float(val_stats["mean_rel_delta"])
        if (
            train_mean == 0.0
            or val_mean == 0.0
            or math.copysign(1.0, train_mean) != math.copysign(1.0, val_mean)
        ):
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "direction_mismatch",
                    "split_stats": split_stats,
                }
            )
            continue

        favored_group = "autoresearch" if train_mean > 0.0 else "incumbent"
        train_win = float(train_stats[f"{favored_group}_win_rate"])
        val_win = float(val_stats[f"{favored_group}_win_rate"])
        if train_win < MIN_WIN_RATE or val_win < MIN_WIN_RATE:
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "low_win_rate",
                    "split_stats": split_stats,
                    "favored_group": favored_group,
                }
            )
            continue

        score = ((0.65 * abs(train_mean)) + (0.35 * abs(val_mean))) * math.sqrt(
            float(min(int(train_stats["count"]), int(val_stats["count"])))
        )
        qualified = {
            **asdict(candidate),
            "qualified": True,
            "reason": "selected_candidate",
            "favored_group": favored_group,
            "score": float(score),
            "split_stats": split_stats,
        }
        diagnostics.append(qualified)
        selected_by_side[favored_group].append(qualified)

    for side in selected_by_side:
        selected_by_side[side].sort(key=lambda item: float(item["score"]), reverse=True)
        selected_by_side[side] = selected_by_side[side][:3]

    return selected_by_side["incumbent"] + selected_by_side["autoresearch"], {
        "selected_by_side": selected_by_side,
        "diagnostics": sorted(
            diagnostics,
            key=lambda item: (bool(item.get("qualified")), float(item.get("score") or 0.0)),
            reverse=True,
        ),
    }


def _feature_value_snapshot(row: pd.Series) -> dict[str, Any]:
    keys = (
        "date",
        "incumbent_ret_5d",
        "incumbent_ret_20d",
        "autoresearch_ret_5d",
        "autoresearch_ret_20d",
        "rel_ret_5d",
        "rel_ret_20d",
        "incumbent_vol_20d",
        "autoresearch_vol_20d",
        "rel_vol_ratio_20d",
        "incumbent_drawdown_20d",
        "autoresearch_drawdown_20d",
        "rel_drawdown_20d",
        "relative_hit_rate_20d",
        "autoresearch_leading_5d",
        "autoresearch_leading_20d",
        "autoresearch_vol_higher_20d",
        "autoresearch_drawdown_worse_20d",
        "relative_hit_rate_ge_55",
    )
    return {key: row.get(key) for key in keys}


def _current_judgement(
    *,
    latest_row: pd.Series,
    selected_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    current_frame = pd.DataFrame([latest_row])
    incumbent_score = 0.0
    autoresearch_score = 0.0
    active_rules: list[dict[str, Any]] = []

    for rule in selected_rules:
        candidate = RuleCandidate(
            rule_id=str(rule["rule_id"]),
            label=str(rule["label"]),
            family=str(rule["family"]),
            feature_names=tuple(rule["feature_names"]),
            threshold=rule.get("threshold"),
            comparator=rule.get("comparator"),
            polarity=str(rule["polarity"]),
        )
        active = bool(_build_rule_mask(current_frame, candidate).iloc[0])
        if not active:
            continue
        side = str(rule["favored_group"])
        score = float(rule["score"])
        if side == "incumbent":
            incumbent_score += score
        else:
            autoresearch_score += score
        active_rules.append(
            {
                "rule_id": rule["rule_id"],
                "label": rule["label"],
                "favored_group": side,
                "score": score,
            }
        )

    total_score = incumbent_score + autoresearch_score
    if total_score <= 0.0:
        favored_group = "mixed"
        confidence = 0.0
    else:
        margin = abs(incumbent_score - autoresearch_score) / total_score
        if margin < MIXED_MARGIN_THRESHOLD:
            favored_group = "mixed"
        else:
            favored_group = "incumbent" if incumbent_score > autoresearch_score else "autoresearch"
        confidence = margin
    return {
        "date": latest_row["date"],
        "feature_snapshot": _feature_value_snapshot(latest_row),
        "incumbent_score": float(incumbent_score),
        "autoresearch_score": float(autoresearch_score),
        "favored_group": favored_group,
        "confidence": float(confidence),
        "active_rules": active_rules,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    current = dict(payload.get("current_judgement") or {})
    selected_by_side = dict((payload.get("rule_selection") or {}).get("selected_by_side") or {})
    memory = dict(payload.get("memory_summary") or {})

    def _rule_lines(side: str, title: str) -> list[str]:
        lines = [f"## {title}"]
        rules = list(selected_by_side.get(side) or [])
        if not rules:
            lines.append("- 학습 구간에서 일관되게 분리된 규칙을 찾지 못함")
            return lines
        for rule in rules:
            train_stats = dict((rule.get("split_stats") or {}).get("train") or {})
            val_stats = dict((rule.get("split_stats") or {}).get("val") or {})
            oos_stats = dict((rule.get("split_stats") or {}).get("oos") or {})
            lines.extend(
                [
                    f"- `{rule['label']}`",
                    f"  - score: `{float(rule.get('score') or 0.0):.6f}`",
                    f"  - train mean Δ(auto-inc): `{float(train_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(train_stats.get('count') or 0)}`",
                    f"  - val mean Δ(auto-inc): `{float(val_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(val_stats.get('count') or 0)}`",
                    f"  - oos mean Δ(auto-inc): `{float(oos_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(oos_stats.get('count') or 0)}`",
                ]
            )
        return lines

    lines = [
        "# Group Regime Judgement",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- horizon_days: `{payload.get('horizon_days')}`",
        f"- symbol_universe: `{', '.join(payload.get('symbol_universe') or [])}`",
        f"- peak_rss_mib: `{float(memory.get('peak_rss_mib') or 0.0):.2f}`",
        f"- memory_log: `{memory.get('rss_log_path', '')}`",
        "",
        "## Current Regime Call",
        f"- as_of: `{current.get('date')}`",
        f"- favored_group: `{current.get('favored_group')}`",
        f"- confidence: `{float(current.get('confidence') or 0.0):.4f}`",
        f"- incumbent_score: `{float(current.get('incumbent_score') or 0.0):.6f}`",
        f"- autoresearch_score: `{float(current.get('autoresearch_score') or 0.0):.6f}`",
    ]

    snapshot = dict(current.get("feature_snapshot") or {})
    if snapshot:
        lines.extend(
            [
                "",
                "## Current Performance Snapshot",
                f"- incumbent_ret_5d: `{float(snapshot.get('incumbent_ret_5d') or 0.0):.6f}`",
                f"- autoresearch_ret_5d: `{float(snapshot.get('autoresearch_ret_5d') or 0.0):.6f}`",
                f"- rel_ret_5d: `{float(snapshot.get('rel_ret_5d') or 0.0):.6f}`",
                f"- rel_ret_20d: `{float(snapshot.get('rel_ret_20d') or 0.0):.6f}`",
                f"- incumbent_vol_20d: `{float(snapshot.get('incumbent_vol_20d') or 0.0):.6f}`",
                f"- autoresearch_vol_20d: `{float(snapshot.get('autoresearch_vol_20d') or 0.0):.6f}`",
                f"- rel_vol_ratio_20d: `{float(snapshot.get('rel_vol_ratio_20d') or 0.0):.6f}`",
                f"- incumbent_drawdown_20d: `{float(snapshot.get('incumbent_drawdown_20d') or 0.0):.6f}`",
                f"- autoresearch_drawdown_20d: `{float(snapshot.get('autoresearch_drawdown_20d') or 0.0):.6f}`",
                f"- rel_drawdown_20d: `{float(snapshot.get('rel_drawdown_20d') or 0.0):.6f}`",
                f"- relative_hit_rate_20d: `{float(snapshot.get('relative_hit_rate_20d') or 0.0):.6f}`",
                f"- autoresearch_leading_5d: `{snapshot.get('autoresearch_leading_5d')}`",
                f"- autoresearch_leading_20d: `{snapshot.get('autoresearch_leading_20d')}`",
                f"- autoresearch_vol_higher_20d: `{snapshot.get('autoresearch_vol_higher_20d')}`",
                f"- autoresearch_drawdown_worse_20d: `{snapshot.get('autoresearch_drawdown_worse_20d')}`",
                f"- relative_hit_rate_ge_55: `{snapshot.get('relative_hit_rate_ge_55')}`",
            ]
        )

    active_rules = list(current.get("active_rules") or [])
    lines.extend(["", "## Active Rules Now"])
    if active_rules:
        for rule in active_rules:
            lines.append(
                f"- `{rule['label']}` -> `{rule['favored_group']}` (`score={float(rule.get('score') or 0.0):.6f}`)"
            )
    else:
        lines.append("- 현재 활성화된 선택 규칙 없음")

    lines.extend(
        [
            "",
            *_rule_lines("incumbent", "Incumbent 성적 좋을 때"),
            "",
            *_rule_lines("autoresearch", "55/45 성적 좋을 때"),
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- 규칙은 외부 가격 레짐이 아니라 `두 그룹의 최근 성과 상태`만으로 학습됨",
            "- `Incumbent 성적 좋을 때` 규칙이 활성화되면 incumbent 쪽이 55/45 대비 상대 성과 우위였던 구간을 뜻함",
            "- `55/45 성적 좋을 때` 규칙이 활성화되면 pair-only 55/45가 incumbent 대비 상대 성과 우위였던 구간을 뜻함",
            "- 반대편 규칙은 해당 그룹의 `성적 안 좋을 때` 기준으로 읽으면 됨",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_group_regime_judgement(
    *,
    incumbent_path: Path,
    autoresearch_path: Path,
    output_dir: Path,
    horizon_days: int,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="group_regime_judgement",
        output_dir=output_dir,
        input_path=incumbent_path,
        rss_log_path=output_dir / MEMORY_GUARD_DIRNAME / "group_regime_judgement_rss_latest.jsonl",
        summary_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "group_regime_judgement_memory_latest.json",
        budget_bytes=hard_rss_bytes,
        soft_limit_bytes=soft_rss_bytes,
        hard_limit_bytes=hard_rss_bytes,
    )
    status = "ok"
    error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        memory_guard.sample(
            event="group_regime_judgement_start",
            context={
                "incumbent_path": str(incumbent_path),
                "autoresearch_path": str(autoresearch_path),
                "horizon_days": horizon_days,
            },
        )

        incumbent = _load_group_portfolio(label="incumbent", path=incumbent_path)
        autoresearch = _load_group_portfolio(label="autoresearch", path=autoresearch_path)
        merged = (
            incumbent.returns.rename(columns={"return": "incumbent"})
            .merge(
                autoresearch.returns[["date", "return"]].rename(columns={"return": "autoresearch"}),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged["split_group"] = merged["date"].map(_split_group)
        merged[f"forward_{horizon_days}d_incumbent"] = _forward_compound(
            merged["incumbent"], horizon_days
        )
        merged[f"forward_{horizon_days}d_autoresearch"] = _forward_compound(
            merged["autoresearch"], horizon_days
        )
        merged[f"forward_{horizon_days}d_rel"] = (
            merged[f"forward_{horizon_days}d_autoresearch"]
            - merged[f"forward_{horizon_days}d_incumbent"]
        )
        memory_guard.checkpoint("group_regime_returns_loaded", {"rows": len(merged)})

        symbols = sorted(set(incumbent.symbols) | set(autoresearch.symbols))
        feature_frame = _build_performance_feature_frame(merged)
        memory_guard.checkpoint(
            "group_regime_features_loaded",
            {"feature_rows": len(feature_frame), "symbol_count": len(symbols)},
        )

        combined = feature_frame
        selected_rules, rule_selection = _select_rules(combined, horizon_days=horizon_days)
        latest_row = combined.sort_values("date").iloc[-1]
        current = _current_judgement(latest_row=latest_row, selected_rules=selected_rules)

        payload = {
            "artifact_kind": "portfolio_group_regime_judgement",
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "selection_basis": "train_val_performance_state_rule_learning",
            "groups_move_as_set": True,
            "horizon_days": int(horizon_days),
            "input_paths": {
                "incumbent": str(incumbent.path),
                "autoresearch": str(autoresearch.path),
            },
            "symbol_universe": symbols,
            "group_summary": {
                "incumbent": {
                    "path": str(incumbent.path),
                    "weights": incumbent.weights,
                },
                "autoresearch": {
                    "path": str(autoresearch.path),
                    "weights": autoresearch.weights,
                },
            },
            "current_judgement": current,
            "rule_selection": rule_selection,
            "selected_rules": selected_rules,
            "memory_summary": {},
        }
    except Exception as exc:  # pragma: no cover - surfaced in payload/logs
        status = "error"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="group_regime_judgement_finish",
            context={"status": status, "error": error},
        )
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={
                "incumbent_path": str(incumbent_path),
                "autoresearch_path": str(autoresearch_path),
                "horizon_days": horizon_days,
            },
        )
        memory_guard.release()
        if payload is not None:
            payload["memory_summary"] = memory_summary

    if payload is None:
        raise RuntimeError("group regime judgement did not produce a payload")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"group_regime_judgement_{timestamp}.json"
    out_md = output_dir / f"group_regime_judgement_{timestamp}.md"
    latest_json = output_dir / "group_regime_judgement_latest.json"
    latest_md = output_dir / "group_regime_judgement_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {
        "payload": payload,
        "json_path": out_json,
        "md_path": out_md,
        "latest_json_path": latest_json,
        "latest_md_path": latest_md,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent-path",
        type=Path,
        default=resolve_current_optimization_path(),
    )
    parser.add_argument(
        "--autoresearch-path",
        type=Path,
        default=_resolve_autoresearch_default_path(),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_group_regime_judgement(
        incumbent_path=Path(args.incumbent_path).resolve(),
        autoresearch_path=Path(args.autoresearch_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        horizon_days=max(1, int(args.horizon_days)),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
