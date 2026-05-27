"""Autonomous portfolio research loop artifact index, audit, and milestone helpers."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.eval.exact_window_log_archive import (
    CANONICAL_REGISTRY_LATEST,
    write_exact_window_canonical_registry,
)
from lumina_quant.portfolio_split_contract import (
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH,
    PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH,
)
from lumina_quant.strategy_factory.candidate_library import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    build_candidate_manifest,
)
from lumina_quant.workflows.alpha_research_pipeline import (
    DEFAULT_FAMILIES,
    write_alpha_research_pipeline_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = REPO_ROOT / "var" / "reports" / "exact_window_backtests"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_ROOT / "followup_status" / "autonomous_research_loop"
DEFAULT_FOLLOWUP_ROOT = DEFAULT_REPORT_ROOT / "followup_status"
DEFAULT_ARTICLE_URL = "https://www.linkedin.com/pulse/ko-managing-real-life-portfolio-based-multi-agent-llms-yeachan-heo-fcmac"
DEFAULT_IMAGE_PATHS = [
    str(REPO_ROOT.parent / "KakaoTalk_20260309_210834478.jpg"),
    str(REPO_ROOT.parent / "KakaoTalk_20260309_210834478_01.jpg"),
]
DEFAULT_BACKLOG_TIMEFRAMES = ("5m", "15m", "30m", "1h", "4h", "1d")

STACK_AUDIT_LATEST = "stack_audit_latest.md"
IDEAS_BACKLOG_LATEST = "ideas_backlog_latest.md"
EXPERIMENT_LEDGER_LATEST = "experiments.tsv"
RESEARCH_STATE_LATEST = "research_state_latest.json"

LEDGER_FIELDNAMES: tuple[str, ...] = (
    "timestamp",
    "experiment_id",
    "hypothesis",
    "changed_files",
    "artifact_inputs",
    "method_category",
    "status",
    "train_total_return",
    "train_sharpe",
    "val_total_return",
    "val_sharpe",
    "oos_total_return",
    "oos_sharpe",
    "memory_evidence_path",
    "notes",
    "source_type",
    "source_run_id",
    "source_artifact_path",
    "crash_kind",
)

_PRIORITY_CANDIDATE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("perp_crowding_carry", "PerpCrowdingCarryStrategy"),
    ("leadlag_spillover", "LeadLagSpilloverStrategy"),
    ("alpha101_formula", "Alpha101FormulaStrategy"),
    ("volcomp_vwap_rev_guarded", "VolCompressionVWAPReversionStrategy"),
    ("lag_convergence", "LagConvergenceStrategy"),
    ("rolling_breakout", "RollingBreakoutStrategy"),
    ("regime_breakout", "RegimeBreakoutCandidateStrategy"),
    ("topcap_tsmom", "TopCapTimeSeriesMomentumStrategy"),
)


@dataclass(slots=True)
class ExperimentLedgerRow:
    timestamp: str
    experiment_id: str
    hypothesis: str
    changed_files: list[str]
    artifact_inputs: list[str]
    method_category: str
    status: str
    train_total_return: float | None = None
    train_sharpe: float | None = None
    val_total_return: float | None = None
    val_sharpe: float | None = None
    oos_total_return: float | None = None
    oos_sharpe: float | None = None
    memory_evidence_path: str | None = None
    notes: str = ""
    source_type: str = ""
    source_run_id: str = ""
    source_artifact_path: str | None = None
    crash_kind: str = ""

    def to_row(self) -> dict[str, str]:
        payload = asdict(self)
        payload["changed_files"] = json.dumps(
            sorted({_path_text(item) for item in self.changed_files if _path_text(item)}),
            sort_keys=True,
        )
        payload["artifact_inputs"] = json.dumps(
            sorted({_path_text(item) for item in self.artifact_inputs if _path_text(item)}),
            sort_keys=True,
        )
        out: dict[str, str] = {}
        for field in LEDGER_FIELDNAMES:
            value = payload.get(field)
            if value is None:
                out[field] = ""
            elif isinstance(value, float):
                out[field] = f"{value:.12g}"
            else:
                out[field] = str(value)
        return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_load(path: str | Path | None) -> Any:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def _path_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    token = str(path).strip()
    return token


def _metric(metrics: dict[str, Any] | None, key: str) -> float | None:
    return _safe_float((metrics or {}).get(key))


def _status_counts(rows: list[ExperimentLedgerRow]) -> dict[str, int]:
    counts = Counter(str(row.status) for row in rows)
    return {key: int(counts[key]) for key in ("keep", "discard", "crash")}


def classify_experiment_error(error: str | BaseException | None) -> str:
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error or "")
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("rss hard limit", "out of memory", "memory", "heavy exact-window run")
    ):
        return "resource_limit"
    if any(
        token in lowered
        for token in (
            "no evaluated candidates",
            "insufficient overlapping history",
            "unsupported",
            "no candidates",
            "coverage",
            "empty timeframe token",
        )
    ):
        return "invalid_idea"
    if any(
        token in lowered
        for token in (
            "typeerror",
            "valueerror",
            "keyerror",
            "indexerror",
            "attributeerror",
            "importerror",
            "modulenotfounderror",
            "syntaxerror",
            "assertionerror",
        )
    ):
        return "implementation_bug"
    return "unclassified"


def read_experiment_ledger(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def write_experiment_ledger(*, path: str | Path, rows: list[ExperimentLedgerRow]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (str(row.timestamp), str(row.experiment_id)),
    )
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_FIELDNAMES), delimiter="\t")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row.to_row())
    return {
        "path": str(resolved),
        "row_count": len(ordered),
        "counts_by_status": _status_counts(ordered),
    }


def _default_paths(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    resolved_report_root = Path(report_root).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    return {
        "report_root": resolved_report_root,
        "followup_root": resolved_report_root / "followup_status",
        "output_dir": resolved_output_dir,
        "canonical_registry": resolved_report_root / CANONICAL_REGISTRY_LATEST,
        "ledger": resolved_output_dir / EXPERIMENT_LEDGER_LATEST,
        "stack_audit": resolved_output_dir / STACK_AUDIT_LATEST,
        "ideas_backlog": resolved_output_dir / IDEAS_BACKLOG_LATEST,
        "state": resolved_output_dir / RESEARCH_STATE_LATEST,
        "portfolio_decision": resolved_report_root
        / "followup_status"
        / "portfolio_max_performance_decision_latest.json",
        "incumbent_portfolio": resolved_report_root
        / "followup_status"
        / "portfolio_one_shot_current_opt"
        / "portfolio_optimization_latest.json",
        "incumbent_bundle": resolved_report_root
        / "followup_status"
        / "portfolio_one_shot_incumbent_bundle_latest.json",
        "exact_window_decision": resolved_report_root / "exact_window_decision_latest.json",
        "log_archive": resolved_report_root
        / "followup_status"
        / "backtest_log_archive_latest.json",
        "pipeline_root": resolved_report_root / "pipeline",
    }


def _best_exact_window_row(
    summary_path: str | Path | None, details_path: str | Path | None = None
) -> dict[str, Any]:
    summary = _json_load(summary_path)
    rows = []
    if isinstance(summary, dict):
        rows.extend(
            dict(row)
            for row in list(summary.get("best_per_strategy") or [])
            if isinstance(row, dict)
        )
    details = _json_load(details_path)
    if isinstance(details, list):
        rows.extend(dict(row) for row in details if isinstance(row, dict))
    if not rows:
        return {}

    def _score(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
        val = dict(row.get("val") or {})
        oos = dict(row.get("oos") or {})
        return (
            1.0 if bool(row.get("promoted")) else 0.0,
            float(_safe_float(val.get("sharpe")) or 0.0)
            + (20.0 * float(_safe_float(val.get("return")) or 0.0)),
            float(_safe_float(oos.get("sharpe")) or 0.0),
            float(_safe_float(oos.get("return")) or 0.0),
            str(row.get("candidate_id") or row.get("name") or ""),
        )

    return max(rows, key=_score)


def _memory_evidence_for_artifact(artifact_path: str | Path | None) -> str | None:
    token = _path_text(artifact_path)
    if not token:
        return None
    payload = _json_load(token)
    if isinstance(payload, dict):
        memory_summary = payload.get("memory_summary")
        if isinstance(memory_summary, dict):
            rss_log = _path_text(memory_summary.get("rss_log_path"))
            if rss_log:
                parent = Path(rss_log).resolve().parent
                matches = sorted(parent.glob("*_memory_latest.json"))
                if matches:
                    return str(matches[0].resolve())
        direct = payload.get("memory_evidence_path")
        if direct:
            return _path_text(direct) or None
    parent = Path(token).resolve().parent
    matches = sorted((parent / "_memory_guard").glob("*_memory_latest.json"))
    if matches:
        return str(matches[0].resolve())
    return None


def collect_exact_window_registry_records(
    *, report_root: str | Path = DEFAULT_REPORT_ROOT
) -> list[ExperimentLedgerRow]:
    resolved_report_root = Path(report_root).resolve()
    write_exact_window_canonical_registry(report_root=resolved_report_root)
    payload = _json_load(resolved_report_root / CANONICAL_REGISTRY_LATEST)
    entries = list((payload or {}).get("entries") or []) if isinstance(payload, dict) else []
    rows: list[ExperimentLedgerRow] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary_path = _path_text(entry.get("summary_path"))
        details_path = _path_text(entry.get("details_path"))
        best_row = _best_exact_window_row(summary_path, details_path)
        promoted_count = int(entry.get("promoted_count") or 0)
        status = "keep" if promoted_count > 0 else "discard"
        requested_timeframes = list(entry.get("requested_timeframes") or [])
        run_id = str(entry.get("run_id") or "")
        rows.append(
            ExperimentLedgerRow(
                timestamp=str(entry.get("updated_at_utc") or _now_iso()),
                experiment_id=f"exact-window::{run_id}",
                hypothesis=(
                    f"Exact-window batch {run_id or '<unknown>'} on "
                    f"{', '.join(str(token) for token in requested_timeframes) or 'unspecified timeframes'}"
                ),
                changed_files=[],
                artifact_inputs=[
                    path
                    for path in (entry.get("manifest_path"), summary_path, details_path)
                    if _path_text(path)
                ],
                method_category="validation",
                status=status,
                train_total_return=_metric(best_row.get("train"), "total_return"),
                train_sharpe=_metric(best_row.get("train"), "sharpe"),
                val_total_return=_metric(best_row.get("val"), "total_return"),
                val_sharpe=_metric(best_row.get("val"), "sharpe"),
                oos_total_return=_metric(best_row.get("oos"), "total_return"),
                oos_sharpe=_metric(best_row.get("oos"), "sharpe"),
                memory_evidence_path=_path_text(entry.get("memory_evidence_path")) or None,
                notes=(
                    f"window_profile={entry.get('window_profile') or 'n/a'}; "
                    f"evaluated_count={int(entry.get('evaluated_count') or 0)}; "
                    f"promoted_count={promoted_count}"
                ),
                source_type="exact_window_registry",
                source_run_id=run_id,
                source_artifact_path=summary_path or None,
            )
        )
    return rows


def collect_archive_crash_records(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    max_records: int = 8,
) -> list[ExperimentLedgerRow]:
    resolved_report_root = Path(report_root).resolve()
    payload = _json_load(
        resolved_report_root / "followup_status" / "backtest_log_archive_latest.json"
    )
    entries = list((payload or {}).get("entries") or []) if isinstance(payload, dict) else []
    rows: list[ExperimentLedgerRow] = []
    for entry in sorted(
        (dict(item) for item in entries if isinstance(item, dict)),
        key=lambda item: (str(item.get("updated_at_utc") or ""), str(item.get("run_id") or "")),
        reverse=True,
    ):
        status = str(entry.get("status") or "")
        error = str(entry.get("error") or "").strip()
        if status == "completed" and not error:
            continue
        run_id = str(entry.get("run_id") or "")
        crash_kind = classify_experiment_error(error or status)
        rows.append(
            ExperimentLedgerRow(
                timestamp=str(entry.get("updated_at_utc") or _now_iso()),
                experiment_id=f"exact-window-crash::{run_id}",
                hypothesis=(
                    f"Recovered exact-window crash {run_id or '<unknown>'} on "
                    f"{', '.join(str(token) for token in list(entry.get('requested_timeframes') or [])) or 'unspecified timeframes'}"
                ),
                changed_files=[],
                artifact_inputs=[
                    path
                    for path in (
                        entry.get("manifest_path"),
                        entry.get("summary_path"),
                        entry.get("details_path"),
                        entry.get("log_path"),
                    )
                    if _path_text(path)
                ],
                method_category="backtest",
                status="crash",
                memory_evidence_path=_path_text(entry.get("memory_evidence_path")) or None,
                notes=error or status or "archived failure",
                source_type="exact_window_log_archive",
                source_run_id=run_id,
                source_artifact_path=_path_text(entry.get("log_path")) or None,
                crash_kind=crash_kind,
            )
        )
        if len(rows) >= max(0, int(max_records)):
            break
    return rows


def collect_portfolio_decision_records(
    *, followup_root: str | Path = DEFAULT_FOLLOWUP_ROOT
) -> list[ExperimentLedgerRow]:
    resolved_followup_root = Path(followup_root).resolve()
    payload = _json_load(resolved_followup_root / "portfolio_max_performance_decision_latest.json")
    if not isinstance(payload, dict):
        return []
    winner_key = str(((payload.get("winner") or {}).get("candidate_key")) or "")
    generated_at = str(payload.get("generated_at") or _now_iso())
    rows: list[ExperimentLedgerRow] = []
    for candidate in list(payload.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        candidate_key = str(candidate.get("candidate_key") or "")
        artifact_path = _path_text(candidate.get("artifact_path")) or None
        rows.append(
            ExperimentLedgerRow(
                timestamp=generated_at,
                experiment_id=f"portfolio::{candidate_key or candidate.get('label')}",
                hypothesis=str(
                    candidate.get("decision_reason")
                    or candidate.get("label")
                    or "portfolio comparison"
                ),
                changed_files=[],
                artifact_inputs=[
                    path
                    for path in (
                        artifact_path,
                        resolved_followup_root / "portfolio_max_performance_decision_latest.json",
                    )
                    if _path_text(path)
                ],
                method_category="portfolio",
                status="keep" if candidate_key and candidate_key == winner_key else "discard",
                train_total_return=_metric(candidate.get("train"), "total_return"),
                train_sharpe=_metric(candidate.get("train"), "sharpe"),
                val_total_return=_metric(candidate.get("val"), "total_return"),
                val_sharpe=_metric(candidate.get("val"), "sharpe"),
                oos_total_return=_metric(candidate.get("oos"), "total_return"),
                oos_sharpe=_metric(candidate.get("oos"), "sharpe"),
                memory_evidence_path=_memory_evidence_for_artifact(artifact_path),
                notes=" | ".join(
                    token
                    for token in (
                        str(candidate.get("label") or "").strip(),
                        str(candidate.get("decision_reason") or "").strip(),
                        str(payload.get("recommended_action") or "").strip(),
                    )
                    if token
                ),
                source_type="portfolio_decision",
                source_run_id=candidate_key,
                source_artifact_path=artifact_path,
            )
        )
    return rows


def build_stack_audit(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = _default_paths(
        report_root=report_root,
        output_dir=DEFAULT_OUTPUT_DIR
        if output_path is None
        else Path(output_path).resolve().parent,
    )
    incumbent_portfolio = _json_load(paths["incumbent_portfolio"]) or {}
    incumbent_bundle = _json_load(paths["incumbent_bundle"]) or {}
    portfolio_decision = _json_load(paths["portfolio_decision"]) or {}
    exact_window_decision = _json_load(paths["exact_window_decision"]) or {}

    bundle_rows = [
        dict(row) for row in list(incumbent_bundle.get("candidates") or []) if isinstance(row, dict)
    ]
    contributors: list[dict[str, Any]] = []
    for row in bundle_rows:
        weight = float(_safe_float(row.get("portfolio_weight")) or 0.0)
        train = dict(row.get("train") or {})
        val = dict(row.get("val") or {})
        oos = dict(row.get("oos") or {})
        contributors.append(
            {
                "name": str(row.get("name") or ""),
                "strategy_class": str(row.get("strategy_class") or ""),
                "timeframe": str(row.get("strategy_timeframe") or row.get("timeframe") or ""),
                "weight": weight,
                "weighted_train_return": weight
                * float(_safe_float(train.get("total_return")) or 0.0),
                "train_total_return": float(_safe_float(train.get("total_return")) or 0.0),
                "train_sharpe": float(_safe_float(train.get("sharpe")) or 0.0),
                "train_stability": float(_safe_float(train.get("stability")) or 0.0),
                "train_rolling_sharpe_min": float(
                    _safe_float(train.get("rolling_sharpe_min")) or 0.0
                ),
                "val_total_return": float(_safe_float(val.get("total_return")) or 0.0),
                "oos_total_return": float(_safe_float(oos.get("total_return")) or 0.0),
            }
        )
    contributors.sort(key=lambda item: item["weighted_train_return"])

    portfolio_metrics = dict(incumbent_portfolio.get("portfolio_metrics") or {})
    train_metrics = dict(portfolio_metrics.get("train") or {})
    val_metrics = dict(portfolio_metrics.get("val") or {})
    oos_metrics = dict(portfolio_metrics.get("oos") or {})
    winner = dict(portfolio_decision.get("winner") or {})
    winner_status = str(winner.get("status") or "")
    next_action = str(exact_window_decision.get("next_action") or "")
    promoted_total = int(exact_window_decision.get("promoted_total") or 0)

    high_impact_issues = [
        {
            "title": "Negative train sleeve contributions dominate the incumbent blend.",
            "evidence": [
                f"{row['strategy_class']} {row['timeframe']} weighted_train_return={row['weighted_train_return']:.2%}"
                for row in contributors[:3]
            ],
        },
        {
            "title": (
                "Validation-only selection keeps the incumbent despite train fragility."
                if winner_status in {"", "retained_incumbent"}
                else "A challenger now wins locked OOS while the current incumbent artifacts still describe the older baseline."
            ),
            "evidence": [
                f"train_total_return={float(_safe_float(train_metrics.get('total_return')) or 0.0):.2%}",
                f"val_total_return={float(_safe_float(val_metrics.get('total_return')) or 0.0):.2%}",
                f"oos_total_return={float(_safe_float(oos_metrics.get('total_return')) or 0.0):.2%}",
                f"winner_status={winner_status or 'unknown'}",
            ],
        },
        {
            "title": "Robustness guardrails already exist and should be reused rather than replaced.",
            "evidence": [
                "src/lumina_quant/strategy_factory/research_runner.py",
                "src/lumina_quant/eval/exact_window_decision.py",
                "src/lumina_quant/eval/exact_window_runtime.py",
                "src/lumina_quant/portfolio_split_contract.py",
            ],
        },
    ]

    lines = [
        "# Autonomous Research Stack Audit",
        "",
        f"- Generated at: `{_now_iso()}`",
        f"- Incumbent train total return: {float(_safe_float(train_metrics.get('total_return')) or 0.0):.2%}",
        f"- Incumbent validation total return: {float(_safe_float(val_metrics.get('total_return')) or 0.0):.2%}",
        f"- Incumbent locked-OOS total return: {float(_safe_float(oos_metrics.get('total_return')) or 0.0):.2%}",
        f"- Current promotion winner: `{winner.get('label') or 'unknown'}` ({winner.get('status') or 'unknown'})",
        f"- Exact-window promoted_total: `{promoted_total}` | next_action=`{next_action}`",
        f"- Heavy lock path: `{PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH.resolve()}`",
        (
            "- Explicit heavy-run memory budget: "
            f"`{PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES}` bytes "
            f"({PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES / (1024**3):.2f} GiB)"
        ),
        "",
        "## Highest-impact instability drivers",
        "",
    ]
    for issue in high_impact_issues:
        lines.append(f"### {issue['title']}")
        lines.extend(f"- {item}" for item in issue["evidence"])
        lines.append("")

    lines.extend(
        [
            "## Incumbent sleeve contributors",
            "",
            "| Strategy | TF | Weight | Train Return | Weighted Train Return | Train Sharpe | Train Stability | Min Rolling Sharpe | Val Return | OOS Return |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contributors:
        lines.append(
            f"| {row['strategy_class']} | {row['timeframe']} | {row['weight']:.2%} | "
            f"{row['train_total_return']:.2%} | {row['weighted_train_return']:.2%} | "
            f"{row['train_sharpe']:.3f} | {row['train_stability']:.3f} | {row['train_rolling_sharpe_min']:.3f} | "
            f"{row['val_total_return']:.2%} | {row['oos_total_return']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## File-backed stack map",
            "",
            "- `src/lumina_quant/strategy_factory/research_runner.py` — candidate evaluation, instability penalties, train/val/OOS metrics, and robust ranking hooks.",
            "- `src/lumina_quant/strategy_factory/candidate_library.py` — unused strategy/alpha inventory that should feed the autonomous backlog before adding new architecture.",
            "- `scripts/run_portfolio_optimization.py` — validation-fit / locked-OOS-report portfolio constructor used by incumbent and challengers.",
            "- `src/lumina_quant/eval/exact_window_decision.py` — promotion and candidate-pool decision logic for exact-window sweeps.",
            "- `src/lumina_quant/eval/exact_window_runtime.py` + `src/lumina_quant/portfolio_split_contract.py` — single-heavy-lane lock and explicit memory guard surfaces.",
            "",
            "## Audit conclusion",
            "",
        ]
    )
    if winner_status in {"", "retained_incumbent"}:
        lines.extend(
            [
                "- The incumbent is still the locked-OOS winner, but train instability is real and concentrated in the cross-sectional + regime-breakout sleeves.",
                "- The safest next step is not a new scheduler; it is a deterministic artifact index plus focused follow-up experiments that reuse the existing exact-window registry and heavy-lock contract.",
            ]
        )
    else:
        lines.extend(
            [
                f"- The locked-OOS promotion flow now favors `{winner.get('label') or 'a challenger'}`, while the saved incumbent artifacts still reflect the prior baseline.",
                "- The safest next step is to formalize the promoted challenger as the incumbent baseline, then continue the same deterministic artifact-index workflow under the existing exact-window registry and heavy-lock contract.",
            ]
        )

    target = Path(output_path).resolve() if output_path is not None else paths["stack_audit"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "incumbent_train_total_return": float(
            _safe_float(train_metrics.get("total_return")) or 0.0
        ),
        "incumbent_val_total_return": float(_safe_float(val_metrics.get("total_return")) or 0.0),
        "incumbent_oos_total_return": float(_safe_float(oos_metrics.get("total_return")) or 0.0),
        "top_negative_contributors": contributors[:3],
    }


def build_ideas_backlog(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    output_path: str | Path | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> dict[str, Any]:
    resolved_symbols = list(symbols or DEFAULT_BINANCE_TOP10_PLUS_METALS)
    resolved_timeframes = list(timeframes or DEFAULT_BACKLOG_TIMEFRAMES)
    manifest = build_candidate_manifest(timeframes=resolved_timeframes, symbols=resolved_symbols)
    paths = _default_paths(
        report_root=report_root,
        output_dir=DEFAULT_OUTPUT_DIR
        if output_path is None
        else Path(output_path).resolve().parent,
    )
    incumbent_bundle = _json_load(paths["incumbent_bundle"]) or {}
    current_strategy_classes = {
        str(row.get("strategy_class") or "")
        for row in list(incumbent_bundle.get("candidates") or [])
        if isinstance(row, dict)
    }
    priority_rows: list[dict[str, Any]] = []
    for keyword, strategy_class in _PRIORITY_CANDIDATE_KEYWORDS:
        matches = [
            dict(row)
            for row in list(manifest.get("candidates") or [])
            if isinstance(row, dict)
            and (
                keyword in str(row.get("name") or "")
                or strategy_class == str(row.get("strategy_class") or "")
            )
        ]
        if not matches:
            continue
        exemplar = matches[0]
        priority_rows.append(
            {
                "keyword": keyword,
                "strategy_class": strategy_class,
                "candidate_count": len(matches),
                "already_in_incumbent": strategy_class in current_strategy_classes,
                "example": exemplar,
            }
        )

    lines = [
        "# Autonomous Research Ideas Backlog",
        "",
        f"- Generated at: `{_now_iso()}`",
        f"- Candidate universe size: `{int(manifest.get('candidate_count') or 0)}`",
        f"- Backlog timeframes: `{', '.join(resolved_timeframes)}`",
        f"- Incumbent strategy classes: `{', '.join(sorted(current_strategy_classes))}`",
        "",
        "## Highest-priority in-repo experiments",
        "",
    ]
    for item in priority_rows:
        example = dict(item["example"])
        lines.append(
            "- "
            f"`{item['strategy_class']}` via `{example.get('name')}` | "
            f"candidates={item['candidate_count']} | "
            f"already_in_incumbent={item['already_in_incumbent']} | "
            f"tf={example.get('strategy_timeframe') or example.get('timeframe')}"
        )

    lines.extend(
        [
            "",
            "## Research-backed methodology upgrades",
            "",
            "- Add stronger train-instability penalties or minimum-train gates before portfolio promotion.",
            "- Reuse exact-window validation artifacts as the canonical duplicate/history source instead of adding a parallel scheduler.",
            "- Keep HRP / risk-parity / volatility-managed portfolio variants behind the existing locked-OOS promotion rule.",
            (
                "- Keep dynamic and overlay allocators under the explicit "
                f"{PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES / (1024**3):.2f} GiB "
                "heavy-run contract and single-heavy-lane discipline."
            ),
            "",
            "## Pipeline thesis map",
            "",
        ]
    )
    for family in DEFAULT_FAMILIES:
        lines.append(
            f"- `{family.family_id}` | exec={family.execution_style} | tf={', '.join(family.target_timeframes)} | rationale={family.rationale}"
        )

    lines.extend(
        [
            "",
            "## Anti-repeat rules",
            "",
            "- Reuse `exact_window_run_registry.jsonl` and the canonical registry snapshot before launching a heavy rerun.",
            "- Keep discarded challengers in the ledger; do not silently carry them forward as production candidates.",
            "- Treat recovered log archives as crash context only, not as the canonical duplicate-signature index.",
        ]
    )

    target = Path(output_path).resolve() if output_path is not None else paths["ideas_backlog"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "candidate_count": int(manifest.get("candidate_count") or 0),
        "priority_candidates": priority_rows,
        "timeframes": resolved_timeframes,
        "symbols": resolved_symbols,
    }


def build_private_git_milestone_gate(
    *,
    repo_root: str | Path = REPO_ROOT,
    followup_root: str | Path = DEFAULT_FOLLOWUP_ROOT,
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).resolve()
    decision_payload = _json_load(
        Path(followup_root).resolve() / "portfolio_max_performance_decision_latest.json"
    )
    winner = dict((decision_payload or {}).get("winner") or {})
    branch = ""
    clean = False
    status_output = ""
    try:
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=resolved_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        status_output = status_result.stdout.strip()
        clean = status_result.returncode == 0 and not status_output
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=resolved_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if branch_result.returncode == 0:
            branch = branch_result.stdout.strip()
    except OSError:
        clean = False

    winner_status = str(winner.get("status") or "")
    materially_better = winner_status not in {"", "retained_incumbent"}
    ready = bool(materially_better and clean)
    if ready:
        reason = "winner improved under the locked-OOS gate and the worktree is clean"
    elif not materially_better:
        reason = "latest portfolio milestone still retains the incumbent"
    elif not clean:
        reason = "worktree is not clean enough for private-git gating"
    else:
        reason = "milestone is not ready for private git"
    return {
        "generated_at": _now_iso(),
        "branch": branch,
        "winner_label": str(winner.get("label") or ""),
        "winner_status": winner_status,
        "working_tree_clean": bool(clean),
        "ready": ready,
        "reason": reason,
        "git_status_short": status_output,
        "suggested_sync_command": "./sync_private.sh" if ready else "",
    }


def build_autonomous_experiment_ledger(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_archive_crashes: int = 8,
) -> dict[str, Any]:
    paths = _default_paths(report_root=report_root, output_dir=output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    collected: dict[str, ExperimentLedgerRow] = {}
    existing_rows = read_experiment_ledger(paths["ledger"])
    for raw in existing_rows:
        if not isinstance(raw, dict):
            continue
        experiment_id = str(raw.get("experiment_id") or "").strip()
        if not experiment_id:
            continue
        changed_files = json.loads(str(raw.get("changed_files") or "[]"))
        artifact_inputs = json.loads(str(raw.get("artifact_inputs") or "[]"))
        collected[experiment_id] = ExperimentLedgerRow(
            timestamp=str(raw.get("timestamp") or _now_iso()),
            experiment_id=experiment_id,
            hypothesis=str(raw.get("hypothesis") or ""),
            changed_files=[str(item) for item in list(changed_files or [])],
            artifact_inputs=[str(item) for item in list(artifact_inputs or [])],
            method_category=str(raw.get("method_category") or ""),
            status=str(raw.get("status") or ""),
            train_total_return=_safe_float(raw.get("train_total_return")),
            train_sharpe=_safe_float(raw.get("train_sharpe")),
            val_total_return=_safe_float(raw.get("val_total_return")),
            val_sharpe=_safe_float(raw.get("val_sharpe")),
            oos_total_return=_safe_float(raw.get("oos_total_return")),
            oos_sharpe=_safe_float(raw.get("oos_sharpe")),
            memory_evidence_path=str(raw.get("memory_evidence_path") or "") or None,
            notes=str(raw.get("notes") or ""),
            source_type=str(raw.get("source_type") or ""),
            source_run_id=str(raw.get("source_run_id") or ""),
            source_artifact_path=str(raw.get("source_artifact_path") or "") or None,
            crash_kind=str(raw.get("crash_kind") or ""),
        )
    for row in (
        collect_exact_window_registry_records(report_root=paths["report_root"])
        + collect_archive_crash_records(
            report_root=paths["report_root"], max_records=max_archive_crashes
        )
        + collect_portfolio_decision_records(followup_root=paths["followup_root"])
    ):
        collected[row.experiment_id] = row
    summary = write_experiment_ledger(path=paths["ledger"], rows=list(collected.values()))
    summary["records"] = sorted(collected)
    return summary


def run_autonomous_portfolio_research_loop(
    *,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    max_archive_crashes: int = 8,
) -> dict[str, Any]:
    paths = _default_paths(report_root=report_root, output_dir=output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    pipeline_manifest = write_alpha_research_pipeline_manifest(
        report_root=paths["report_root"],
        article_url=DEFAULT_ARTICLE_URL,
        image_paths=list(DEFAULT_IMAGE_PATHS),
    )
    ledger = build_autonomous_experiment_ledger(
        report_root=paths["report_root"],
        output_dir=paths["output_dir"],
        max_archive_crashes=max_archive_crashes,
    )
    audit = build_stack_audit(report_root=paths["report_root"], output_path=paths["stack_audit"])
    backlog = build_ideas_backlog(
        report_root=paths["report_root"],
        output_path=paths["ideas_backlog"],
        symbols=symbols,
        timeframes=timeframes,
    )
    milestone_gate = build_private_git_milestone_gate(
        repo_root=REPO_ROOT,
        followup_root=paths["followup_root"],
    )

    state = {
        "generated_at": _now_iso(),
        "schema_version": "1.0",
        "artifact_kind": "autonomous_portfolio_research_state",
        "report_root": str(paths["report_root"]),
        "output_dir": str(paths["output_dir"]),
        "canonical_registry_path": str(paths["canonical_registry"]),
        "experiment_ledger": ledger,
        "stack_audit": audit,
        "ideas_backlog": backlog,
        "pipeline_manifest": pipeline_manifest,
        "milestone_gate": milestone_gate,
        "memory_contract": {
            "total_budget_bytes": PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
            "heavy_lock_path": str(PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH.resolve()),
            "session_memory_lease_path": str(
                PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH.resolve()
            ),
            "single_heavy_lane": True,
            "registry_only_duplicate_guard": str(
                (paths["report_root"] / "exact_window_run_registry.jsonl").resolve()
            ),
            "explicit_budget_injection": {
                "portfolio_optimizer": PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
                "causal_dynamic_portfolio": PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
                "causal_overlay_portfolio": PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
            },
        },
        "next_actions": [
            "Review stack_audit_latest.md before promoting any train-fragile challenger.",
            "Use experiments.tsv as the artifact index; do not introduce a second heavy-run scheduler.",
            "Prioritize unused in-repo candidates from ideas_backlog_latest.md before adding new complexity.",
            "Only push privately when milestone_gate.ready becomes true under the locked-OOS decision artifact.",
        ],
    }
    paths["state"].write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "state_path": str(paths["state"]),
        "stack_audit_path": str(paths["stack_audit"]),
        "ideas_backlog_path": str(paths["ideas_backlog"]),
        "ledger_path": str(paths["ledger"]),
        "canonical_registry_path": str(paths["canonical_registry"]),
        "pipeline_manifest_json": str(
            paths["pipeline_root"] / "alpha_research_pipeline_latest.json"
        ),
        "pipeline_manifest_md": str(paths["pipeline_root"] / "alpha_research_pipeline_latest.md"),
        "counts_by_status": dict(ledger.get("counts_by_status") or {}),
        "milestone_gate_ready": bool(milestone_gate.get("ready")),
    }


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_ROOT",
    "EXPERIMENT_LEDGER_LATEST",
    "IDEAS_BACKLOG_LATEST",
    "LEDGER_FIELDNAMES",
    "RESEARCH_STATE_LATEST",
    "STACK_AUDIT_LATEST",
    "ExperimentLedgerRow",
    "build_autonomous_experiment_ledger",
    "build_ideas_backlog",
    "build_private_git_milestone_gate",
    "build_stack_audit",
    "classify_experiment_error",
    "collect_archive_crash_records",
    "collect_exact_window_registry_records",
    "collect_portfolio_decision_records",
    "read_experiment_ledger",
    "run_autonomous_portfolio_research_loop",
    "write_experiment_ledger",
]
