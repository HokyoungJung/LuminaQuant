"""Post-backtest alpha scoreboard: multi-metric leaderboards + composite rank.

The repo already had two selection lenses — the weighted
``robust_score_from_metrics`` scalar (strategy_factory.selection) and the
diversified shortlist admission — but no surface that (a) ranks candidates
PER METRIC (return, CAGR, Sharpe, Sortino, Calmar, MDD, turnover,
liquidations, trade frequency, ...), (b) aggregates those ranks into a
scale-free COMPOSITE standing, and (c) applies explicit eligibility gates
with recorded exclusion reasons. This module is that surface, built for the
data-bearing PC to run over walk-forward/backtest result rows.

Design rules:
- Deterministic: identical input -> identical payload (stable tie-breaks by
  candidate id; no wall-clock anywhere in the body).
- Consumes the repo's canonical FLAT metric dict keys ("return", "cagr",
  "sharpe", "sortino", "calmar", "mdd", "turnover", "trade_count",
  "win_rate", ...) when precomputed metrics are supplied, and derives the
  battery from a raw returns series (optimizer_core.metrics) when not.
  Precomputed values always take precedence over derived ones.
- Composite = weighted mean of direction-adjusted dense-rank percentiles, so
  metrics on wildly different scales (Sharpe vs MDD vs liquidation count)
  contribute comparably and outliers cannot dominate through magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import metrics as _core_metrics

# Metric direction: +1 higher-is-better, -1 lower-is-better.
METRIC_DIRECTIONS: dict[str, int] = {
    "return": 1,
    "cagr": 1,
    "sharpe": 1,
    "sortino": 1,
    "calmar": 1,
    "win_rate": 1,
    "mdd": -1,
    "volatility": -1,
    "turnover": -1,
    "liquidation_count": -1,
    "trade_count": 1,
    "trade_frequency": 1,
}

# Metrics that enter the composite by default (equal weight). trade_count /
# trade_frequency / volatility stay INFORMATIONAL by default: more trades or
# more vol is not intrinsically better or worse, so they are ranked in the
# per-metric tables but excluded from the composite unless explicitly weighted.
DEFAULT_COMPOSITE_METRICS: tuple[str, ...] = (
    "return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "mdd",
    "turnover",
    "liquidation_count",
)

# Opinionated default eligibility gates (each overridable / removable with
# None): a candidate that never trades is unrankable, and a single simulated
# liquidation is a catastrophic-path signal, not a ranking nuance.
DEFAULT_GATES: dict[str, float | int | None] = {
    "min_trades": 1,
    "max_liquidations": 0,
    "max_mdd": None,
    "min_sharpe": None,
    "min_bars": None,
}


@dataclass(slots=True)
class ScoreboardRow:
    """Normalized per-candidate inputs after metric resolution."""

    candidate_id: str
    label: str
    family: str
    strategy_class: str
    metrics: dict[str, float]
    bars: int
    gate_failures: list[str]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _derive_from_returns(returns: Any) -> dict[str, float]:
    """Derive the metric battery from a raw per-period returns series."""
    series = np.asarray(list(returns or []), dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return {}
    core = _core_metrics(series)
    total_return = float(np.expm1(np.log1p(np.clip(series, -0.999999, None)).sum()))
    win_rate = float(np.mean(series > 0.0)) if series.size else 0.0
    derived = {
        "return": total_return,
        "cagr": float(core.get("cagr", 0.0)),
        "sharpe": float(core.get("sharpe", 0.0)),
        "sortino": float(core.get("sortino", 0.0)),
        "calmar": float(core.get("calmar", 0.0)),
        "mdd": float(core.get("max_drawdown", 0.0)),
        "volatility": float(core.get("volatility", core.get("vol", 0.0)) or 0.0),
        "win_rate": win_rate,
    }
    return {key: value for key, value in derived.items() if math.isfinite(value)}


def resolve_row(raw: dict[str, Any]) -> ScoreboardRow:
    """Normalize one candidate result row into the scoreboard schema.

    Accepted fields: ``id`` (required), ``label``, ``family``,
    ``strategy_class``, ``metrics`` (canonical flat dict, takes precedence),
    ``returns`` (raw series, used to derive missing metrics), ``turnover``,
    ``trade_count``/``trades``, ``liquidation_count``, ``bars``.
    """
    candidate_id = str(raw.get("id") or raw.get("candidate_id") or raw.get("name") or "")
    metrics: dict[str, float] = {}
    metrics.update(_derive_from_returns(raw.get("returns")))
    supplied = raw.get("metrics")
    if isinstance(supplied, dict):
        for key, value in supplied.items():
            parsed = _safe_float(value)
            if parsed is not None:
                metrics[str(key)] = parsed
    for field, key in (
        ("turnover", "turnover"),
        ("trade_count", "trade_count"),
        ("trades", "trade_count"),
        ("liquidation_count", "liquidation_count"),
    ):
        parsed = _safe_float(raw.get(field))
        if parsed is not None and key not in metrics:
            metrics[key] = parsed
    bars = int(_safe_float(raw.get("bars"), 0.0) or 0.0)
    if "trade_frequency" not in metrics and "trade_count" in metrics and bars > 0:
        metrics["trade_frequency"] = metrics["trade_count"] / float(bars)
    metrics.setdefault("liquidation_count", 0.0)
    return ScoreboardRow(
        candidate_id=candidate_id,
        label=str(raw.get("label") or candidate_id),
        family=str(raw.get("family") or ""),
        strategy_class=str(raw.get("strategy_class") or raw.get("strategy") or ""),
        metrics=metrics,
        bars=bars,
        gate_failures=[],
    )


def apply_gates(
    rows: list[ScoreboardRow], gates: dict[str, float | int | None]
) -> tuple[list[ScoreboardRow], list[ScoreboardRow]]:
    """Split rows into (eligible, excluded); failures recorded on each row."""
    eligible: list[ScoreboardRow] = []
    excluded: list[ScoreboardRow] = []
    for row in rows:
        failures: list[str] = []
        min_trades = gates.get("min_trades")
        if min_trades is not None and row.metrics.get("trade_count", 0.0) < float(min_trades):
            failures.append(
                f"trade_count {row.metrics.get('trade_count', 0.0):g} < min_trades {min_trades:g}"
            )
        max_liq = gates.get("max_liquidations")
        if max_liq is not None and row.metrics.get("liquidation_count", 0.0) > float(max_liq):
            failures.append(
                f"liquidation_count {row.metrics.get('liquidation_count', 0.0):g} > "
                f"max_liquidations {max_liq:g}"
            )
        max_mdd = gates.get("max_mdd")
        if max_mdd is not None and row.metrics.get("mdd", 0.0) > float(max_mdd):
            failures.append(f"mdd {row.metrics.get('mdd', 0.0):g} > max_mdd {max_mdd:g}")
        min_sharpe = gates.get("min_sharpe")
        if min_sharpe is not None and row.metrics.get("sharpe", 0.0) < float(min_sharpe):
            failures.append(
                f"sharpe {row.metrics.get('sharpe', 0.0):g} < min_sharpe {min_sharpe:g}"
            )
        min_bars = gates.get("min_bars")
        if min_bars is not None and row.bars < int(min_bars):
            failures.append(f"bars {row.bars} < min_bars {int(min_bars)}")
        row.gate_failures = failures
        (excluded if failures else eligible).append(row)
    return eligible, excluded


def _dense_ranks(values: list[float], direction: int) -> list[int]:
    """Dense 1-based ranks; rank 1 = best given the metric direction."""
    ordered = sorted(set(values), reverse=(direction > 0))
    position = {value: index + 1 for index, value in enumerate(ordered)}
    return [position[value] for value in values]


def rank_by_metric(rows: list[ScoreboardRow], metric: str) -> list[dict[str, Any]]:
    """Full ranking table for one metric (missing values rank last)."""
    direction = METRIC_DIRECTIONS.get(metric, 1)
    scored = [row for row in rows if metric in row.metrics]
    unscored = [row for row in rows if metric not in row.metrics]
    values = [row.metrics[metric] for row in scored]
    ranks = _dense_ranks(values, direction) if scored else []
    table = [
        {
            "id": row.candidate_id,
            "value": row.metrics[metric],
            "rank": rank,
        }
        for row, rank in zip(scored, ranks, strict=False)
    ]
    table.sort(key=lambda item: (item["rank"], item["id"]))
    worst = (table[-1]["rank"] + 1) if table else 1
    table.extend(
        {"id": row.candidate_id, "value": None, "rank": worst}
        for row in sorted(unscored, key=lambda r: r.candidate_id)
    )
    return table


def _rank_percentile(rank: int, worst_rank: int) -> float:
    """Map a dense rank to [0, 1] where 1 is best. Single-rank fields score 1."""
    if worst_rank <= 1:
        return 1.0
    return 1.0 - (rank - 1) / float(worst_rank - 1)


def composite_ranking(
    rows: list[ScoreboardRow],
    *,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Weighted mean of direction-adjusted rank percentiles, ranked."""
    if weights is None:
        weights = dict.fromkeys(DEFAULT_COMPOSITE_METRICS, 1.0)
    weights = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
    total_weight = sum(weights.values())
    if not rows or total_weight <= 0.0:
        return []

    per_metric_rank: dict[str, dict[str, int]] = {}
    per_metric_worst: dict[str, int] = {}
    for metric in weights:
        table = rank_by_metric(rows, metric)
        per_metric_rank[metric] = {item["id"]: item["rank"] for item in table}
        per_metric_worst[metric] = max(item["rank"] for item in table)

    out: list[dict[str, Any]] = []
    for row in rows:
        score = 0.0
        detail: dict[str, int] = {}
        for metric, weight in weights.items():
            rank = per_metric_rank[metric][row.candidate_id]
            detail[metric] = rank
            score += weight * _rank_percentile(rank, per_metric_worst[metric])
        out.append(
            {
                "id": row.candidate_id,
                "label": row.label,
                "family": row.family,
                "strategy_class": row.strategy_class,
                "composite_score": score / total_weight,
                "per_metric_ranks": detail,
            }
        )
    out.sort(key=lambda item: (-item["composite_score"], item["id"]))
    for index, item in enumerate(out):
        item["rank"] = index + 1
    return out


def build_scoreboard(
    raw_rows: list[dict[str, Any]],
    *,
    gates: dict[str, float | int | None] | None = None,
    weights: dict[str, float] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build the full scoreboard payload from candidate result rows."""
    effective_gates = dict(DEFAULT_GATES)
    if gates:
        effective_gates.update(gates)
    rows = [resolve_row(raw) for raw in raw_rows]
    rows = [row for row in rows if row.candidate_id]
    rows.sort(key=lambda row: row.candidate_id)
    eligible, excluded = apply_gates(rows, effective_gates)

    ranked_metrics = sorted(
        {metric for row in eligible for metric in row.metrics if metric in METRIC_DIRECTIONS}
    )
    per_metric_top = {
        metric: rank_by_metric(eligible, metric)[: max(1, int(top_n))] for metric in ranked_metrics
    }
    composite = composite_ranking(eligible, weights=weights)

    return {
        "input_count": len(rows),
        "eligible_count": len(eligible),
        "gates": {k: v for k, v in effective_gates.items() if v is not None},
        "excluded": [
            {"id": row.candidate_id, "reasons": list(row.gate_failures)} for row in excluded
        ],
        "metrics_table": [
            {
                "id": row.candidate_id,
                "label": row.label,
                "family": row.family,
                "strategy_class": row.strategy_class,
                "bars": row.bars,
                "metrics": dict(sorted(row.metrics.items())),
            }
            for row in eligible
        ],
        "per_metric_top": per_metric_top,
        "composite": composite,
        "composite_weights": (
            {k: float(v) for k, v in weights.items()}
            if weights
            else dict.fromkeys(DEFAULT_COMPOSITE_METRICS, 1.0)
        ),
    }


def render_scoreboard_markdown(payload: dict[str, Any], *, title: str = "Alpha Scoreboard") -> str:
    """Deterministic markdown report for the scoreboard payload."""
    lines = [f"# {title}", ""]
    lines.append(
        f"Candidates: {payload['input_count']} in / {payload['eligible_count']} eligible. "
        f"Gates: {payload['gates'] or 'none'}."
    )
    if payload["excluded"]:
        lines.append("")
        lines.append("## Excluded")
        for item in payload["excluded"]:
            lines.append(f"- `{item['id']}`: {'; '.join(item['reasons'])}")
    lines.append("")
    lines.append("## Composite ranking")
    lines.append("")
    lines.append("| rank | candidate | family | composite | ")
    lines.append("|---|---|---|---|")
    for item in payload["composite"]:
        lines.append(
            f"| {item['rank']} | `{item['id']}` | {item['family'] or '-'} | "
            f"{item['composite_score']:.4f} |"
        )
    for metric, table in payload["per_metric_top"].items():
        direction = "high" if METRIC_DIRECTIONS.get(metric, 1) > 0 else "low"
        lines.append("")
        lines.append(f"## Top by {metric} (better = {direction})")
        lines.append("")
        lines.append("| rank | candidate | value |")
        lines.append("|---|---|---|")
        for item in table:
            value = "n/a" if item["value"] is None else f"{item['value']:.6g}"
            lines.append(f"| {item['rank']} | `{item['id']}` | {value} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_COMPOSITE_METRICS",
    "DEFAULT_GATES",
    "METRIC_DIRECTIONS",
    "ScoreboardRow",
    "apply_gates",
    "build_scoreboard",
    "composite_ranking",
    "rank_by_metric",
    "render_scoreboard_markdown",
    "resolve_row",
]
