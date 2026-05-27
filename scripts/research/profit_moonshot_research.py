"""Bounded profit-moonshot research report harness.

The harness is intentionally report-only: it scans previously produced JSON
artifacts under ``var/reports/profit_moonshot_20260501``, normalizes common
live-equivalent and vector/research metric shapes, ranks candidates, and writes
small latest JSON/Markdown summaries.  It does not run backtests or import the
strategy registry, which keeps worker/report integration memory-safe and avoids
shared strategy-file ownership.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = REPO_ROOT / "var" / "reports" / "profit_moonshot_20260501"
SUMMARY_JSON_NAME = "profit_moonshot_summary_latest.json"
SUMMARY_MD_NAME = "profit_moonshot_summary_latest.md"
DEFAULT_MAX_FILES = 128
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TOP_N = 25

_METRIC_KEYS = (
    "total_return",
    "max_drawdown",
    "sharpe",
    "sortino",
    "trades",
    "liquidations",
    "final_equity",
)

_LIST_KEYS = (
    "mode_candidate_rows",
    "mode_rows",
    "ranked_candidates",
    "candidates",
    "candidate_rows",
    "results",
    "rows",
    "best_per_strategy",
    "shortlist",
    "selected_candidates",
)

_ID_KEYS = (
    "candidate_id",
    "id",
    "uid",
    "mode",
    "name",
    "strategy_id",
)

_BLOCKER_KEYS = (
    "alpha_blocking_reasons",
    "blocking_reasons",
    "blockers",
    "hard_reject_reasons",
    "reject_reasons",
    "reasons",
)


@dataclass(frozen=True, slots=True)
class ScanIssue:
    """Non-fatal scan issue kept in the summary artifact."""

    path: str
    reason: str
    bytes: int = 0

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path, "reason": self.reason}
        if self.bytes:
            payload["bytes"] = self.bytes
        return payload


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    token = str(value).strip().replace("%", "")
    if not token:
        return default
    try:
        parsed = float(token)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    # Treat percent-looking strings as display percentages only when the caller
    # gave an explicit percent sign ("5%" -> 0.05). Numeric JSON stays as-is.
    if isinstance(value, str) and "%" in value:
        parsed /= 100.0
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    return round(_safe_float(value, float(default)))


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "y", "pass", "passed", "eligible", "promoted"}


def _first_present(source: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _compact_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _artifact_kind(payload: Any, path: Path) -> str:
    if isinstance(payload, Mapping):
        raw = str(payload.get("artifact_kind") or payload.get("kind") or "").strip()
        if raw:
            return raw
    return path.stem


def _source_kind(*, artifact_kind: str, path: Path, row: Mapping[str, Any]) -> str:
    haystack = " ".join(
        (
            artifact_kind,
            str(path),
            str(row.get("source_kind") or ""),
            str(row.get("source") or ""),
            str(row.get("selection_role") or ""),
            str(row.get("status") or ""),
        )
    ).lower()
    if "live_equivalent" in haystack or "live-equivalent" in haystack:
        return "live_equivalent"
    if "exact_window" in haystack or "vector" in haystack or "research" in haystack:
        return "vector"
    return "unknown"


def _candidate_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    candidate_id = str(_first_present(row, _ID_KEYS) or "").strip()
    name = str(row.get("name") or row.get("mode") or candidate_id or "unknown_candidate").strip()
    if not candidate_id:
        candidate_id = name
    return candidate_id, name


def _metric_from_sources(
    row: Mapping[str, Any],
    split_metrics: Mapping[str, Any],
    *,
    flat_keys: Sequence[str],
    split_keys: Sequence[str],
    default: float = 0.0,
) -> float:
    flat = _first_present(row, flat_keys)
    if flat is not None:
        return _safe_float(flat, default)
    split_value = _first_present(_as_dict(split_metrics), split_keys)
    if split_value is not None:
        return _safe_float(split_value, default)
    return default


def _extract_split_metrics(row: Mapping[str, Any], split: str) -> dict[str, float]:
    nested_metrics = _as_dict(row.get("metrics"))
    split_metrics = _as_dict(nested_metrics.get(split))
    prefix = f"{split}_"
    total_return = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}total_return", f"{prefix}return", "total_return", "return"),
        split_keys=("total_return", "return"),
    )
    max_drawdown = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}max_drawdown", f"{prefix}mdd", "max_drawdown", "mdd"),
        split_keys=("max_drawdown", "mdd"),
    )
    sharpe = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}sharpe", "sharpe"),
        split_keys=("sharpe",),
    )
    sortino = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}sortino", "sortino"),
        split_keys=("sortino",),
    )
    trades = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}trade_count", f"{prefix}trades", "trade_count", "trades"),
        split_keys=("trade_count", "trades"),
    )
    liquidations = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(
            f"{prefix}liquidation_count",
            f"{prefix}liquidations",
            "liquidation_count",
            "liquidations",
        ),
        split_keys=("liquidation_count", "liquidations"),
    )
    final_equity = _metric_from_sources(
        row,
        split_metrics,
        flat_keys=(f"{prefix}final_equity", "final_equity"),
        split_keys=("final_equity",),
    )
    return {
        "total_return": total_return,
        "max_drawdown": abs(max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": trades,
        "liquidations": liquidations,
        "final_equity": final_equity,
    }


def _primary_metrics(
    split_metrics: Mapping[str, Mapping[str, float]],
) -> tuple[str, dict[str, float]]:
    for split in ("val", "oos", "train"):
        metrics = dict(split_metrics.get(split) or {})
        if any(abs(float(metrics.get(key, 0.0))) > 1e-12 for key in _METRIC_KEYS):
            return split, metrics
    return "val", dict(split_metrics.get("val") or {})


def _blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in _BLOCKER_KEYS:
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            blockers.extend(
                [item.strip() for item in value.replace("|", ";").split(";") if item.strip()]
            )
        elif isinstance(value, Mapping):
            blockers.extend(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, Iterable):
            blockers.extend(str(item).strip() for item in value if str(item).strip())
        else:
            blockers.append(str(value))
    status = str(row.get("status") or "").strip()
    if status and status.lower() not in {
        "promoted",
        "passed",
        "pass",
        "live_equivalent_validated",
        "completed",
        "selection_eligible",
    }:
        blockers.append(f"status={status}")
    return list(dict.fromkeys(blockers))


def _explicit_selection_eligible(row: Mapping[str, Any]) -> bool | None:
    for key in (
        "selection_eligible",
        "promotion_eligible",
        "eligible_for_promotion",
        "promoted",
        "pass",
    ):
        if key in row:
            return _is_truthy(row.get(key))
    return None


def _infer_live_promotion_eligible(
    *,
    source_kind: str,
    primary: Mapping[str, float],
    blockers: Sequence[str],
    row: Mapping[str, Any],
) -> bool:
    if source_kind != "live_equivalent":
        return False
    explicit = _explicit_selection_eligible(row)
    if explicit is not None:
        return explicit
    status = str(row.get("status") or "").strip().lower()
    if status not in {"live_equivalent_validated", "promoted", "passed", "completed"}:
        return False
    return (
        _safe_float(primary.get("total_return")) > 0.0
        and _safe_float(primary.get("sharpe")) > 0.0
        and _safe_float(primary.get("sortino")) > 0.0
        and _safe_float(primary.get("trades")) > 0.0
        and _safe_float(primary.get("liquidations")) <= 0.0
        and not blockers
    )


def _ranking_score(
    *,
    source_kind: str,
    promotion_eligible: bool,
    metrics: Mapping[str, float],
    blockers: Sequence[str],
) -> float:
    total_return = _safe_float(metrics.get("total_return"))
    sharpe = _safe_float(metrics.get("sharpe"))
    sortino = _safe_float(metrics.get("sortino"))
    max_drawdown = abs(_safe_float(metrics.get("max_drawdown")))
    trades = max(0.0, _safe_float(metrics.get("trades")))
    liquidations = max(0.0, _safe_float(metrics.get("liquidations")))
    source_bonus = 5.0 if source_kind == "live_equivalent" else 0.0
    eligible_bonus = 100.0 if promotion_eligible else 0.0
    blocker_penalty = 10.0 * float(len(blockers))
    return (
        eligible_bonus
        + source_bonus
        + total_return * 100.0
        + sharpe * 2.0
        + sortino
        + min(trades, 500.0) / 100.0
        - max_drawdown * 50.0
        - liquidations * 100.0
        - blocker_penalty
    )


def _normalize_row(row: Mapping[str, Any], *, artifact_kind: str, path: Path) -> dict[str, Any]:
    candidate_id, name = _candidate_identity(row)
    split_metrics = {split: _extract_split_metrics(row, split) for split in ("train", "val", "oos")}
    primary_split, primary = _primary_metrics(split_metrics)
    source_kind = _source_kind(artifact_kind=artifact_kind, path=path, row=row)
    blockers = _blockers(row)
    promotion_eligible = _infer_live_promotion_eligible(
        source_kind=source_kind,
        primary=primary,
        blockers=blockers,
        row=row,
    )
    score = _ranking_score(
        source_kind=source_kind,
        promotion_eligible=promotion_eligible,
        metrics=primary,
        blockers=blockers,
    )
    return {
        "candidate_id": candidate_id,
        "name": name,
        "mode": str(row.get("mode") or ""),
        "strategy_class": str(row.get("strategy_class") or row.get("strategy") or ""),
        "family": str(row.get("family") or row.get("category") or ""),
        "source_kind": source_kind,
        "source_artifact": _compact_path(path),
        "artifact_kind": artifact_kind,
        "status": str(row.get("status") or ""),
        "selection_role": str(row.get("selection_role") or ""),
        "promotion_eligible": bool(promotion_eligible),
        "primary_split": primary_split,
        "total_return": _safe_float(primary.get("total_return")),
        "max_drawdown": abs(_safe_float(primary.get("max_drawdown"))),
        "sharpe": _safe_float(primary.get("sharpe")),
        "sortino": _safe_float(primary.get("sortino")),
        "trades": _safe_int(primary.get("trades")),
        "liquidations": _safe_int(primary.get("liquidations")),
        "final_equity": _safe_float(primary.get("final_equity")),
        "split_metrics": split_metrics,
        "blockers": blockers,
        "ranking_score": score,
    }


def _looks_like_candidate(row: Mapping[str, Any]) -> bool:
    return any(key in row for key in _ID_KEYS) or any(
        key in row
        for key in ("metrics", "val_total_return", "total_return", "sharpe", "max_drawdown")
    )


def _extract_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [
            dict(item)
            for item in payload
            if isinstance(item, Mapping) and _looks_like_candidate(item)
        ]
    if not isinstance(payload, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for key in _LIST_KEYS:
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        rows.extend(
            dict(item)
            for item in value
            if isinstance(item, Mapping) and _looks_like_candidate(item)
        )
    if rows:
        return rows
    if _looks_like_candidate(payload):
        return [dict(payload)]
    return []


def _iter_json_artifacts(root: Path, *, max_files: int) -> list[Path]:
    if not root.exists():
        return []
    paths = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {SUMMARY_JSON_NAME} or path.name.startswith("profit_moonshot_summary_"):
            continue
        paths.append(path)
        if len(paths) >= max_files:
            break
    return paths


def _load_json_bounded(path: Path, *, max_bytes: int) -> tuple[Any | None, ScanIssue | None]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, ScanIssue(_compact_path(path), f"stat_failed:{exc.__class__.__name__}")
    if size > max_bytes:
        return None, ScanIssue(_compact_path(path), "skipped_max_bytes", bytes=int(size))
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, ScanIssue(
            _compact_path(path), f"json_parse_failed:{exc.__class__.__name__}", bytes=int(size)
        )


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        _safe_float(row.get("ranking_score")),
        _safe_float(row.get("total_return")),
        _safe_float(row.get("sharpe")),
        -abs(_safe_float(row.get("max_drawdown"))),
        str(row.get("candidate_id") or ""),
    )


def _summarize_blockers(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        for blocker in _as_list(row.get("blockers")):
            token = str(blocker)
            counts[token] = counts.get(token, 0) + 1
    return [
        {"blocker": blocker, "count": count}
        for blocker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _source_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        token = str(row.get("source_kind") or "unknown")
        counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))


def _round_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    rounded = dict(row)
    for key in (
        "ranking_score",
        "total_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "final_equity",
    ):
        if key in rounded:
            rounded[key] = round(_safe_float(rounded[key]), 10)
    return rounded


def build_summary(
    *,
    input_dir: Path = DEFAULT_REPORT_ROOT,
    output_dir: Path | None = None,
    generated_at: str | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Scan bounded JSON artifacts and return a normalized summary payload."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    issues: list[ScanIssue] = []
    artifacts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    paths = _iter_json_artifacts(input_dir, max_files=max(1, int(max_files)))
    for path in paths:
        payload, issue = _load_json_bounded(path, max_bytes=max(1, int(max_bytes)))
        if issue is not None:
            issues.append(issue)
            continue
        artifact_kind = _artifact_kind(payload, path)
        raw_rows = _extract_candidate_rows(payload)
        artifacts.append(
            {
                "path": _compact_path(path),
                "artifact_kind": artifact_kind,
                "candidate_rows": len(raw_rows),
            }
        )
        for raw_row in raw_rows:
            candidates.append(_normalize_row(raw_row, artifact_kind=artifact_kind, path=path))

    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)
    promoted = next((row for row in ranked if bool(row.get("promotion_eligible"))), None)
    best_return = max(
        (row for row in ranked if bool(row.get("promotion_eligible"))),
        key=lambda row: _safe_float(row.get("total_return"), 0.0),
        default=None,
    )
    best_report_only = ranked[0] if ranked else None
    decision = "promoted_candidate_found" if promoted else "no_live_equivalent_promotion_candidate"
    summary: dict[str, Any] = {
        "artifact_kind": "profit_moonshot_research_summary",
        "generated_at": generated_at or _utc_now_iso(),
        "input_dir": _compact_path(input_dir),
        "output_dir": _compact_path(output_dir),
        "scan_limits": {
            "max_files": int(max_files),
            "max_bytes": int(max_bytes),
            "top_n": int(top_n),
        },
        "decision": decision,
        "candidate_count": len(candidates),
        "promotion_eligible_count": sum(
            1 for row in candidates if bool(row.get("promotion_eligible"))
        ),
        "source_counts": _source_counts(candidates),
        "scanned_artifacts": artifacts,
        "skipped_artifacts": [issue.as_payload() for issue in issues],
        "blocker_summary": _summarize_blockers(candidates),
        "promoted_candidate": _round_candidate(promoted) if promoted else None,
        "best_return_candidate": _round_candidate(best_return) if best_return else None,
        "best_report_only_candidate": _round_candidate(best_report_only)
        if best_report_only
        else None,
        "ranked_candidates": [_round_candidate(row) for row in ranked[: max(1, int(top_n))]],
    }
    return summary


def _format_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def _format_float(value: Any) -> str:
    return f"{_safe_float(value):.3f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown report."""
    promoted = _as_dict(summary.get("promoted_candidate"))
    best_return = _as_dict(summary.get("best_return_candidate"))
    best = _as_dict(summary.get("best_report_only_candidate"))
    lines = [
        "# Profit Moonshot Research Summary",
        "",
        f"Generated: `{summary.get('generated_at', '')}`",
        f"Decision: `{summary.get('decision', '')}`",
        f"Candidates scanned: `{summary.get('candidate_count', 0)}`",
        f"Promotion-eligible candidates: `{summary.get('promotion_eligible_count', 0)}`",
        "",
    ]
    if promoted:
        lines.extend(
            [
                "## Promoted Candidate",
                "",
                f"- Candidate: `{promoted.get('candidate_id')}`",
                f"- Source: `{promoted.get('source_kind')}` from `{promoted.get('source_artifact')}`",
                f"- {promoted.get('primary_split', 'val')} return: `{_format_pct(promoted.get('total_return'))}`",
                f"- max_drawdown: `{_format_pct(promoted.get('max_drawdown'))}`",
                f"- Sharpe / Sortino: `{_format_float(promoted.get('sharpe'))}` / `{_format_float(promoted.get('sortino'))}`",
                f"- trades / liquidations: `{promoted.get('trades', 0)}` / `{promoted.get('liquidations', 0)}`",
                f"- final_equity: `{_format_float(promoted.get('final_equity'))}`",
                "",
            ]
        )
        if best_return and best_return.get("candidate_id") != promoted.get("candidate_id"):
            lines.extend(
                [
                    "## Best Validation Return Candidate",
                    "",
                    f"- Candidate: `{best_return.get('candidate_id')}`",
                    f"- Source: `{best_return.get('source_kind')}` from `{best_return.get('source_artifact')}`",
                    f"- {best_return.get('primary_split', 'val')} return: `{_format_pct(best_return.get('total_return'))}`",
                    f"- max_drawdown: `{_format_pct(best_return.get('max_drawdown'))}`",
                    f"- Sharpe / Sortino: `{_format_float(best_return.get('sharpe'))}` / `{_format_float(best_return.get('sortino'))}`",
                    f"- trades / liquidations: `{best_return.get('trades', 0)}` / `{best_return.get('liquidations', 0)}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Promoted Candidate",
                "",
                "- `NONE` — no live-equivalent candidate passed promotion gates.",
                "",
            ]
        )
        if best:
            lines.extend(
                [
                    "## Best Report-Only Candidate",
                    "",
                    f"- Candidate: `{best.get('candidate_id')}`",
                    f"- Source: `{best.get('source_kind')}` from `{best.get('source_artifact')}`",
                    f"- {best.get('primary_split', 'val')} return: `{_format_pct(best.get('total_return'))}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Top Ranked Candidates",
            "",
            "| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(_as_list(summary.get("ranked_candidates")), start=1):
        item = _as_dict(row)
        blockers = ", ".join(str(x) for x in _as_list(item.get("blockers"))) or "-"
        lines.append(
            "| {rank} | `{candidate}` | `{source}` | `{split}` | {ret} | {mdd} | {sharpe} | {sortino} | {trades} | {liq} | {equity} | {blockers} |".format(
                rank=idx,
                candidate=item.get("candidate_id", ""),
                source=item.get("source_kind", ""),
                split=item.get("primary_split", ""),
                ret=_format_pct(item.get("total_return")),
                mdd=_format_pct(item.get("max_drawdown")),
                sharpe=_format_float(item.get("sharpe")),
                sortino=_format_float(item.get("sortino")),
                trades=item.get("trades", 0),
                liq=item.get("liquidations", 0),
                equity=_format_float(item.get("final_equity")),
                blockers=blockers.replace("|", "/"),
            )
        )

    blocker_summary = _as_list(summary.get("blocker_summary"))
    if blocker_summary:
        lines.extend(["", "## Blocker Summary", ""])
        for item in blocker_summary[:20]:
            row = _as_dict(item)
            lines.append(f"- `{row.get('blocker')}`: {row.get('count')}")

    skipped = _as_list(summary.get("skipped_artifacts"))
    if skipped:
        lines.extend(["", "## Skipped Artifacts", ""])
        for item in skipped:
            row = _as_dict(item)
            lines.append(f"- `{row.get('path')}` — {row.get('reason')}")

    lines.append("")
    return "\n".join(lines)


def write_summary(summary: Mapping[str, Any], *, output_dir: Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SUMMARY_JSON_NAME
    md_path = output_dir / SUMMARY_MD_NAME
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize bounded profit moonshot research artifacts."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--generated-at", default="", help="Deterministic timestamp override for tests/replays."
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the summary payload after writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir or args.input_dir)
    summary = build_summary(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        generated_at=str(args.generated_at or "") or None,
        max_files=int(args.max_files),
        max_bytes=int(args.max_bytes),
        top_n=int(args.top_n),
    )
    paths = write_summary(summary, output_dir=output_dir)
    if args.print_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(paths["json"])
        print(paths["markdown"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
