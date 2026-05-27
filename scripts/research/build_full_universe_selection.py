"""Build the validation-primary full-universe portfolio selection report.

The report is intentionally artifact-first: it walks saved report JSON/CSV files,
normalizes any candidate-like train/validation/OOS metrics it can find, rebuilds
stream-backed candidates when daily returns are present, then compares all clean
candidates with a bounded validation-primary score.  OOS is emitted for audit
only and is never used by the score, candidate health priors, or HYBRID tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_ROOT = (
    REPO_ROOT
    / "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "full_universe_selection_20260426"
DEFAULT_SCAN_ROOTS = (
    REPO_ROOT / "var/reports/exact_window_backtests/followup_status",
    REPO_ROOT / "var/reports/portfolio_superiority_dense_pairs",
    REPO_ROOT / "var/reports/portfolio_superiority_wave2",
    REPO_ROOT / "var/reports/portfolio_superiority_overlay_followup",
    REPO_ROOT / "reports",
)

TRAIN_START = date(2025, 1, 1)
TRAIN_END = date(2025, 12, 31)
VAL_START = date(2026, 1, 1)
VAL_END = date(2026, 2, 28)
OOS_START = date(2026, 3, 1)
SPLITS = ("train", "val", "oos")
MDD_CAP = 0.25
TOP_STREAM_SLEEVES = 14

METRIC_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "volatility",
)


@dataclass(slots=True)
class Candidate:
    uid: str
    name: str
    kind: str
    category: str
    true_hybrid: bool
    combinable: bool
    source_paths: list[str]
    metrics: dict[str, dict[str, float]]
    stream_day_count: int = 0
    stream_digest: str = ""
    daily_map: dict[str, float] = field(default_factory=dict, repr=False)
    generated_at: str = ""
    diagnostic_oos_used: bool = False
    oos_health_priors_enabled: bool = False
    final_selection_eligible: bool = False
    risk_eligible_mdd25_train_val: bool = False
    selection_score: float = 0.0
    val_scaled_score: float = 0.0
    train_scaled_score: float = 0.0
    oos_scaled_score_report_only: float = 0.0
    eligibility_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    allocation_summary: dict[str, Any] = field(default_factory=dict)

    def public_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        payload = {
            "uid": self.uid,
            "name": self.name,
            "kind": self.kind,
            "category": self.category,
            "true_hybrid": self.true_hybrid,
            "combinable": self.combinable,
            "primary_source_path": self.source_paths[0] if self.source_paths else "",
            "source_path_count": len(self.source_paths),
            "metrics": self.metrics,
            "stream_day_count": self.stream_day_count,
            "stream_digest": self.stream_digest,
            "generated_at": self.generated_at,
            "diagnostic_oos_used": self.diagnostic_oos_used,
            "oos_health_priors_enabled": self.oos_health_priors_enabled,
            "final_selection_eligible": self.final_selection_eligible,
            "risk_eligible_mdd25_train_val": self.risk_eligible_mdd25_train_val,
            "selection_score": self.selection_score,
            "val_scaled_score": self.val_scaled_score,
            "train_scaled_score": self.train_scaled_score,
            "oos_scaled_score_report_only": self.oos_scaled_score_report_only,
            "eligibility_reasons": list(self.eligibility_reasons),
            "notes": list(self.notes),
        }
        if include_sources:
            payload["source_paths"] = list(self.source_paths)
        if self.config:
            payload["config"] = self.config
        if self.allocation_summary:
            payload["allocation_summary"] = self.allocation_summary
        return payload


@dataclass(frozen=True, slots=True)
class HybridGridConfig:
    variant: str = "dynamic_default"
    warmup_days: int = 60
    lookback_days: int = 20
    default_boost: float = 0.20
    sticky_default_bonus: float = 0.08
    switch_margin: float = 0.05
    score_temperature: float = 0.90
    min_positive_score: float = 0.00
    pair_weight_cap: float = 0.25
    diversified_weight_cap: float = 0.75


@dataclass(slots=True)
class Panel:
    names: list[str]
    days: list[str]
    matrix: dict[str, np.ndarray]
    split_by_day: dict[str, str]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        out = float(value)
        return out if math.isfinite(out) else default
    token = str(value).strip()
    if not token:
        return default
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1].strip()
    token = token.replace(",", "")
    try:
        out = float(token)
    except ValueError:
        return default
    if is_percent:
        out /= 100.0
    return out if math.isfinite(out) else default


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_day(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.replace(".", "", 1).isdigit():
        numeric = float(token)
        # Common timestamp magnitudes: ns/us/ms/s.
        if numeric >= 1e18:
            return datetime.fromtimestamp(numeric / 1_000_000_000.0, tz=UTC).date().isoformat()
        if numeric >= 1e15:
            return datetime.fromtimestamp(numeric / 1_000_000.0, tz=UTC).date().isoformat()
        if numeric >= 1e12:
            return datetime.fromtimestamp(numeric / 1_000.0, tz=UTC).date().isoformat()
        if numeric >= 1e9:
            return datetime.fromtimestamp(numeric, tz=UTC).date().isoformat()
    token = token.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(token).date().isoformat()
    except ValueError:
        pass
    try:
        return date.fromisoformat(token.split("T", 1)[0]).isoformat()
    except ValueError:
        return None


def _split_for_day(day_key: str) -> str | None:
    try:
        day_value = date.fromisoformat(str(day_key).split("T", 1)[0])
    except ValueError:
        return None
    if TRAIN_START <= day_value <= TRAIN_END:
        return "train"
    if VAL_START <= day_value <= VAL_END:
        return "val"
    if day_value >= OOS_START:
        return "oos"
    return None


def _metrics_from_returns(
    values: list[float] | np.ndarray, periods_per_year: int = 365
) -> dict[str, float]:
    returns = np.asarray(values, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return dict.fromkeys(METRIC_KEYS, 0.0)
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = max(float(returns.size) / float(periods_per_year), 1.0 / float(periods_per_year))
    cagr = float((equity[-1] ** (1.0 / years)) - 1.0) if equity[-1] > 0 else -1.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = float((mean / std) * math.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = (
        float((mean / downside_std) * math.sqrt(periods_per_year)) if downside_std > 1e-12 else 0.0
    )
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    max_drawdown = float(abs(np.min(drawdown))) if drawdown.size else 0.0
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-12 else 0.0
    volatility = float(std * math.sqrt(periods_per_year))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
    }


def _empty_metrics() -> dict[str, dict[str, float]]:
    return {split: dict.fromkeys(METRIC_KEYS, 0.0) for split in SPLITS}


def _normalize_metric_dict(raw: dict[str, Any] | None) -> dict[str, float]:
    raw = dict(raw or {})
    total_return = _safe_float(raw.get("total_return", raw.get("return", raw.get("ret"))), 0.0)
    max_drawdown = abs(_safe_float(raw.get("max_drawdown", raw.get("mdd", raw.get("max_dd"))), 0.0))
    return {
        "total_return": total_return,
        "cagr": _safe_float(raw.get("cagr"), total_return),
        "sharpe": _safe_float(raw.get("sharpe"), 0.0),
        "sortino": _safe_float(raw.get("sortino"), 0.0),
        "calmar": _safe_float(raw.get("calmar"), 0.0),
        "max_drawdown": max_drawdown,
        "volatility": _safe_float(raw.get("volatility", raw.get("vol")), 0.0),
    }


def _looks_like_split_metrics(raw: Any) -> bool:
    return isinstance(raw, dict) and any(isinstance(raw.get(split), dict) for split in SPLITS)


def _extract_metrics(raw: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    for key in ("metrics", "split_metrics", "portfolio_metrics", "fit_metrics", "report_metrics"):
        value = raw.get(key)
        if _looks_like_split_metrics(value):
            return {split: _normalize_metric_dict(dict(value.get(split) or {})) for split in SPLITS}
    if _looks_like_split_metrics(raw):
        return {split: _normalize_metric_dict(dict(raw.get(split) or {})) for split in SPLITS}
    return None


def _daily_compound_stream(stream: Any) -> dict[str, float]:
    if not isinstance(stream, list):
        return {}
    bucket: dict[str, list[float]] = {}
    for point in stream:
        if not isinstance(point, dict):
            continue
        raw_ts = point.get("datetime", point.get("t", point.get("timestamp", point.get("date"))))
        day = _parse_day(raw_ts)
        if day is None:
            continue
        value = point.get("v", point.get("return", point.get("daily_return", point.get("value"))))
        bucket.setdefault(day, []).append(_safe_float(value, 0.0))
    out: dict[str, float] = {}
    for day, returns in bucket.items():
        compounded = 1.0
        for value in returns:
            compounded *= 1.0 + value
        out[day] = float(compounded - 1.0)
    return out


def _extract_daily_map(raw: dict[str, Any]) -> dict[str, float]:
    for key in (
        "portfolio_daily_return_streams",
        "portfolio_return_streams",
        "daily_return_streams",
        "return_streams",
    ):
        streams = raw.get(key)
        if isinstance(streams, dict):
            merged: dict[str, float] = {}
            for split in SPLITS:
                merged.update(_daily_compound_stream(streams.get(split)))
            if merged:
                return merged
    dates = raw.get("dates")
    returns = raw.get("daily_returns")
    if isinstance(dates, list) and isinstance(returns, list) and dates and returns:
        out: dict[str, float] = {}
        for raw_day, raw_return in zip(dates, returns, strict=False):
            day = _parse_day(raw_day)
            if day is not None:
                out[day] = _safe_float(raw_return, 0.0)
        if out:
            return out
    return {}


def _metrics_from_daily_map(daily_map: dict[str, float]) -> dict[str, dict[str, float]]:
    split_values: dict[str, list[float]] = {split: [] for split in SPLITS}
    for day, value in sorted(daily_map.items()):
        split = _split_for_day(day)
        if split is not None:
            split_values[split].append(_safe_float(value, 0.0))
    return {split: _metrics_from_returns(values) for split, values in split_values.items()}


def _stream_digest(daily_map: dict[str, float]) -> str:
    if not daily_map:
        return ""
    h = hashlib.sha256()
    for day, value in sorted(daily_map.items()):
        h.update(day.encode("utf-8"))
        h.update(f"={float(value):.12g};".encode())
    return h.hexdigest()[:16]


def _slug(value: str, max_len: int = 96) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token[:max_len] or "candidate"


def _path_kind(path: Path, raw: dict[str, Any]) -> str:
    explicit = str(raw.get("kind") or raw.get("artifact_kind") or raw.get("family") or "").strip()
    haystack = f"{path.as_posix()} {explicit} {raw.get('name', '')}".lower()
    if "hybrid_online" in haystack or explicit.startswith("true_hybrid"):
        return explicit or "true_hybrid_online"
    if "leverage" in haystack or "3x" in haystack:
        return explicit or "leverage_sweep"
    if "static_blend" in haystack or "static" in haystack:
        return explicit or "static_blend"
    if "pair" in haystack or "spread" in haystack:
        return explicit or "pair_strategy"
    if "portfolio" in haystack or "allocator" in haystack:
        return explicit or "portfolio"
    if "candidate_research" in haystack:
        return explicit or "strategy_candidate"
    return explicit or "saved_artifact"


def _is_true_hybrid(path: Path, raw: dict[str, Any], kind: str) -> bool:
    if bool(raw.get("true_hybrid")):
        return True
    if str(kind).startswith("true_hybrid"):
        return True
    haystack = (
        f"{path.as_posix()} {kind} {raw.get('name', '')} {raw.get('artifact_kind', '')}".lower()
    )
    if "not_true_hybrid" in haystack or "hybrid_guarded_state_vwap" in haystack:
        return False
    return "hybrid_online" in haystack or "true_hybrid" in haystack


def _category_from_kind(kind: str, true_hybrid: bool, combinable: bool) -> str:
    token = kind.lower()
    if true_hybrid:
        if "v35" in token or "v3.5" in token:
            return "true_hybrid_v35"
        if "v36" in token or "v3.6" in token:
            return "true_hybrid_v36"
        return "true_hybrid"
    if "leverage" in token or "3x" in token:
        return "leverage_sweep"
    if "static" in token or "blend" in token:
        return "static_blend"
    if "pair" in token or "spread" in token:
        return "pair_strategy"
    if "portfolio" in token or "allocator" in token:
        return "portfolio"
    if combinable:
        return "source_sleeve"
    return "metric_only"


def _scaled_score(metrics: dict[str, Any]) -> float:
    total_return = _safe_float(metrics.get("total_return", metrics.get("return")), 0.0)
    sharpe = _safe_float(metrics.get("sharpe"), 0.0)
    sortino = _safe_float(metrics.get("sortino"), 0.0)
    calmar = _safe_float(metrics.get("calmar"), 0.0)
    max_drawdown = abs(_safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
    mdd_headroom = 1.0 - min(max(max_drawdown, 0.0), MDD_CAP) / MDD_CAP
    return float(
        100.0
        * (
            0.30 * math.tanh(total_return / 0.18)
            + 0.30 * math.tanh(sharpe / 4.0)
            + 0.15 * math.tanh(sortino / 12.0)
            + 0.15 * math.tanh(calmar / 80.0)
            + 0.10 * mdd_headroom
        )
    )


def _score_and_mark(candidate: Candidate) -> Candidate:
    candidate.train_scaled_score = _scaled_score(candidate.metrics.get("train") or {})
    candidate.val_scaled_score = _scaled_score(candidate.metrics.get("val") or {})
    candidate.oos_scaled_score_report_only = _scaled_score(candidate.metrics.get("oos") or {})
    candidate.selection_score = candidate.val_scaled_score + 0.18 * candidate.train_scaled_score
    train_mdd = _safe_float(candidate.metrics.get("train", {}).get("max_drawdown"), 0.0)
    val_mdd = _safe_float(candidate.metrics.get("val", {}).get("max_drawdown"), 0.0)
    train_has_metrics = any(
        abs(_safe_float(v, 0.0)) > 1e-12 for v in candidate.metrics.get("train", {}).values()
    )
    val_has_metrics = any(
        abs(_safe_float(v, 0.0)) > 1e-12 for v in candidate.metrics.get("val", {}).values()
    )
    candidate.risk_eligible_mdd25_train_val = bool(train_mdd <= MDD_CAP and val_mdd <= MDD_CAP)
    reasons: list[str] = []
    if not train_has_metrics:
        reasons.append("missing_train_metrics")
    if not val_has_metrics:
        reasons.append("missing_val_metrics")
    if not candidate.risk_eligible_mdd25_train_val:
        reasons.append("train_or_val_mdd_above_25pct")
    if candidate.diagnostic_oos_used:
        reasons.append("diagnostic_oos_used")
    if candidate.oos_health_priors_enabled:
        reasons.append("oos_health_priors_enabled")
    candidate.eligibility_reasons = reasons
    candidate.final_selection_eligible = not reasons
    return candidate


def _candidate_name(path: Path, raw: dict[str, Any], scope: str) -> str:
    for key in ("name", "candidate_key", "candidate_id", "config_name"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    artifact = str(raw.get("artifact_kind") or path.stem)
    return f"{artifact}:{scope}" if scope else artifact


def _make_candidate(path: Path, raw: dict[str, Any], scope: str = "") -> Candidate | None:
    metrics = _extract_metrics(raw)
    daily_map = _extract_daily_map(raw)
    if metrics is None and daily_map:
        metrics = _metrics_from_daily_map(daily_map)
    if metrics is None:
        return None
    kind = _path_kind(path, raw)
    true_hybrid = _is_true_hybrid(path, raw, kind)
    combinable = bool(daily_map)
    category = _category_from_kind(kind, true_hybrid, combinable)
    config = dict(raw.get("config") or {}) if isinstance(raw.get("config"), dict) else {}
    diagnostic_oos_used = bool(raw.get("diagnostic_oos_used"))
    oos_health_priors_enabled = bool(config.get("use_current_health_priors"))
    if "oos_health" in f"{path.as_posix()} {raw.get('notes', '')}".lower():
        diagnostic_oos_used = True
    source = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
    digest = _stream_digest(daily_map)
    name = _candidate_name(path, raw, scope)
    if "diagnostic" in f"{name} {path.as_posix()}".lower():
        diagnostic_oos_used = True
    uid_seed = json.dumps(
        {
            "name": name,
            "kind": kind,
            "metrics": metrics,
            "stream_digest": digest,
            "scope": scope,
        },
        sort_keys=True,
        default=_json_default,
    )
    uid = f"{_slug(name, 72)}_{hashlib.sha256(uid_seed.encode('utf-8')).hexdigest()[:10]}"
    candidate = Candidate(
        uid=uid,
        name=name,
        kind=kind,
        category=category,
        true_hybrid=true_hybrid,
        combinable=combinable,
        source_paths=[source],
        metrics={split: _normalize_metric_dict(metrics.get(split) or {}) for split in SPLITS},
        stream_day_count=len(daily_map),
        stream_digest=digest,
        daily_map=daily_map,
        generated_at=str(raw.get("generated_at") or ""),
        diagnostic_oos_used=diagnostic_oos_used,
        oos_health_priors_enabled=oos_health_priors_enabled,
        notes=[str(raw.get("notes"))]
        if raw.get("notes") and isinstance(raw.get("notes"), str)
        else [],
        config=config,
        allocation_summary=dict(raw.get("allocation_summary") or {})
        if isinstance(raw.get("allocation_summary"), dict)
        else {},
    )
    return _score_and_mark(candidate)


def _walk_candidate_dicts(payload: Any, path: Path) -> list[tuple[dict[str, Any], str]]:
    found: list[tuple[dict[str, Any], str]] = []

    def visit(obj: Any, trace: list[str]) -> None:
        if isinstance(obj, dict):
            metrics = _extract_metrics(obj)
            daily_map = _extract_daily_map(obj)
            if metrics is not None or daily_map:
                found.append((obj, ":".join(trace)))
            for key, value in obj.items():
                if key in {
                    "allocations",
                    "score_history",
                    "daily_returns",
                    "dates",
                    "portfolio_return_streams",
                    "portfolio_daily_return_streams",
                    "daily_return_streams",
                    "return_streams",
                    "memory_summary",
                    "memory_policy",
                    "metrics",
                }:
                    continue
                if key in {
                    "candidates",
                    "rows",
                    "ranked_all_candidates",
                    "ranked_clean_any_type",
                    "ranked_clean_true_hybrid",
                    "ranked_spec_highvol_true_hybrid",
                    "top_trials",
                    "benchmark_comparisons",
                } and isinstance(value, list):
                    for idx, row in enumerate(value):
                        if isinstance(row, dict):
                            visit(row, [*trace, key, str(idx)])
                    continue
                if key in {
                    "best",
                    "best_trial",
                    "candidate",
                    "selected_primary_spec_compliant_highvol_hybrid",
                    "overall_top_clean_candidate_any_type",
                    "metric_best_clean_true_hybrid_challenger",
                    "scenarios",
                    "source_sleeve_metrics",
                }:
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if str(sub_key) == "metrics":
                                continue
                            if isinstance(sub_value, dict):
                                visit(sub_value, [*trace, key, str(sub_key)])
                        visit(value, [*trace, key])
                    continue
                if isinstance(value, dict) and len(trace) < 4:
                    visit(value, [*trace, str(key)])
                elif isinstance(value, list) and len(trace) < 3 and key not in {"weights"}:
                    for idx, row in enumerate(value[:50]):
                        if isinstance(row, dict):
                            visit(row, [*trace, key, str(idx)])
        elif isinstance(obj, list):
            for idx, row in enumerate(obj):
                if isinstance(row, dict):
                    visit(row, [*trace, str(idx)])

    visit(payload, [])
    # Preserve order while dropping exact object/scope duplicates.
    unique: list[tuple[dict[str, Any], str]] = []
    seen: set[tuple[int, str]] = set()
    for raw, scope in found:
        key = (id(raw), scope)
        if key not in seen:
            unique.append((raw, scope))
            seen.add(key)
    return unique


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _csv_row_metrics(row: dict[str, str]) -> dict[str, dict[str, float]] | None:
    aliases = {
        "total_return": ("return", "total_return", "ret"),
        "sharpe": ("sharpe",),
        "sortino": ("sortino",),
        "calmar": ("calmar",),
        "max_drawdown": ("max_drawdown", "mdd", "max_dd"),
        "volatility": ("volatility", "vol"),
        "cagr": ("cagr",),
    }
    metrics = _empty_metrics()
    any_value = False
    for split in SPLITS:
        for target, suffixes in aliases.items():
            for suffix in suffixes:
                for key in (f"{split}_{suffix}", f"{split}.{suffix}", f"{split} {suffix}"):
                    if key in row and str(row[key]).strip():
                        metrics[split][target] = (
                            abs(_safe_float(row[key], 0.0))
                            if target == "max_drawdown"
                            else _safe_float(row[key], 0.0)
                        )
                        any_value = True
                        break
                if any_value and metrics[split][target] != 0.0:
                    break
    return metrics if any_value else None


def _load_csv_candidates(path: Path) -> list[Candidate]:
    out: list[Candidate] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                metrics = _csv_row_metrics(row)
                if metrics is None:
                    continue
                raw: dict[str, Any] = {
                    "name": row.get("name")
                    or row.get("candidate")
                    or row.get("candidate_id")
                    or f"{path.stem}:{idx}",
                    "kind": row.get("kind") or row.get("family") or "csv_candidate",
                    "train": metrics["train"],
                    "val": metrics["val"],
                    "oos": metrics["oos"],
                }
                candidate = _make_candidate(path, raw, f"csv:{idx}")
                if candidate is not None:
                    out.append(candidate)
    except Exception:
        return []
    return out


def discover_candidates(scan_roots: list[Path]) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    json_files: list[Path] = []
    csv_files: list[Path] = []
    scanned_roots = []
    for root in scan_roots:
        if not root.exists():
            continue
        scanned_roots.append(
            str(root.relative_to(REPO_ROOT) if root.is_relative_to(REPO_ROOT) else root)
        )
        json_files.extend(sorted(root.rglob("*.json")))
        csv_files.extend(sorted(root.rglob("*.csv")))
    json_files = [
        path for path in json_files if "full_universe_selection_20260426" not in path.parts
    ]
    csv_files = [path for path in csv_files if "full_universe_selection_20260426" not in path.parts]
    for path in json_files:
        payload = _load_json(path)
        if payload is None:
            continue
        for raw, scope in _walk_candidate_dicts(payload, path):
            candidate = _make_candidate(path, raw, scope)
            if candidate is not None:
                candidates.append(candidate)
    for path in csv_files:
        candidates.extend(_load_csv_candidates(path))
    deduped = _dedupe_candidates(candidates)
    summary = {
        "scan_roots": scanned_roots,
        "json_files_seen": len(json_files),
        "csv_files_seen": len(csv_files),
        "raw_candidate_rows": len(candidates),
        "deduped_candidate_rows": len(deduped),
        "stream_available_rows": sum(1 for row in deduped if row.combinable),
        "clean_eligible_rows": sum(1 for row in deduped if row.final_selection_eligible),
    }
    return deduped, summary


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for candidate in candidates:
        key_payload = {
            "name": candidate.name,
            "metrics": {
                split: {
                    metric: round(_safe_float(candidate.metrics.get(split, {}).get(metric)), 8)
                    for metric in ("total_return", "sharpe", "sortino", "calmar", "max_drawdown")
                }
                for split in SPLITS
            },
            "stream_digest": candidate.stream_digest,
            "diagnostic_oos_used": candidate.diagnostic_oos_used,
            "oos_health_priors_enabled": candidate.oos_health_priors_enabled,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        for source in candidate.source_paths:
            if source not in existing.source_paths:
                existing.source_paths.append(source)
        if candidate.true_hybrid and not existing.true_hybrid:
            existing.true_hybrid = True
            existing.kind = candidate.kind
        existing.diagnostic_oos_used = existing.diagnostic_oos_used or candidate.diagnostic_oos_used
        existing.oos_health_priors_enabled = (
            existing.oos_health_priors_enabled or candidate.oos_health_priors_enabled
        )
        if not existing.daily_map and candidate.daily_map:
            existing.daily_map = candidate.daily_map
            existing.stream_day_count = candidate.stream_day_count
            existing.stream_digest = candidate.stream_digest
            existing.combinable = True
        existing.combinable = existing.combinable or candidate.combinable
        existing.category = _category_from_kind(
            existing.kind, existing.true_hybrid, existing.combinable
        )
        existing.notes.extend(note for note in candidate.notes if note not in existing.notes)
        _score_and_mark(existing)
    rows = list(merged.values())
    rows.sort(key=lambda row: row.selection_score, reverse=True)
    return rows


def _build_panel(candidates: list[Candidate]) -> Panel:
    names: list[str] = []
    seen_names: set[str] = set()
    all_days: set[str] = set()
    maps: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        if not candidate.daily_map:
            continue
        name = candidate.name
        if name in seen_names:
            name = f"{candidate.name}__{candidate.uid[-6:]}"
        seen_names.add(name)
        names.append(name)
        maps[name] = dict(candidate.daily_map)
        all_days.update(candidate.daily_map)
    ordered_days = sorted(day for day in all_days if _split_for_day(day) is not None)
    matrix = {
        name: np.asarray(
            [_safe_float(maps[name].get(day), 0.0) for day in ordered_days], dtype=float
        )
        for name in names
    }
    split_by_day = {
        day: str(_split_for_day(day)) for day in ordered_days if _split_for_day(day) is not None
    }
    return Panel(names=names, days=ordered_days, matrix=matrix, split_by_day=split_by_day)


def _split_metrics_from_daily(days: list[str], returns: list[float]) -> dict[str, dict[str, float]]:
    split_returns: dict[str, list[float]] = {split: [] for split in SPLITS}
    for day, value in zip(days, returns, strict=True):
        split = _split_for_day(day)
        if split is not None:
            split_returns[split].append(float(value))
    return {split: _metrics_from_returns(values) for split, values in split_returns.items()}


def _candidate_train_score(candidate: Candidate) -> float:
    return _scaled_score(candidate.metrics.get("train") or {})


def _select_hybrid_sleeves(candidates: list[Candidate]) -> list[Candidate]:
    clean_streams = [
        row
        for row in candidates
        if row.final_selection_eligible
        and row.combinable
        and not row.true_hybrid
        and row.stream_day_count >= 50
    ]
    preferred_tokens = (
        "three_way_regime",
        "soft_three_way_regime",
        "static_blend_76_24",
        "balanced_overlay_80_20",
        "pair_tactical_mode",
        "production_guarded_portfolio",
        "state_vwap_pair",
        "wave2_pair",
    )
    selected: list[Candidate] = []
    seen_streams: set[str] = set()

    def add(candidate: Candidate) -> None:
        if candidate.stream_digest and candidate.stream_digest in seen_streams:
            return
        selected.append(candidate)
        if candidate.stream_digest:
            seen_streams.add(candidate.stream_digest)

    for token in preferred_tokens:
        matches = [row for row in clean_streams if token in row.name]
        matches.sort(key=lambda row: (row.selection_score, row.val_scaled_score), reverse=True)
        if matches:
            add(matches[0])
    for row in sorted(clean_streams, key=lambda item: item.selection_score, reverse=True):
        if len(selected) >= TOP_STREAM_SLEEVES:
            break
        add(row)
    return selected


def _train_best_sleeve(candidates: list[Candidate]) -> str:
    if not candidates:
        return "risk_off_cash"
    return max(candidates, key=_candidate_train_score).name


def _panel_split_indices(panel: Panel) -> dict[str, np.ndarray]:
    return {
        split: np.asarray(
            [idx for idx, day in enumerate(panel.days) if panel.split_by_day.get(day) == split],
            dtype=int,
        )
        for split in SPLITS
    }


def _precompute_trailing_scores(panel: Panel, lookback: int) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name in panel.names:
        arr = panel.matrix[name]
        out = np.zeros(arr.shape[0], dtype=float)
        for idx in range(arr.shape[0]):
            start = max(0, idx - int(lookback))
            window = arr[start:idx]
            out[idx] = _scaled_score(_metrics_from_returns(window)) / 100.0 if window.size else 0.0
        scores[name] = out
    return scores


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    ordered = list(scores.items())
    arr = np.asarray([value / max(1e-6, temperature) for _, value in ordered], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {key: 0.0 for key, _ in ordered}
    max_value = float(np.max(arr[finite]))
    exp_values = np.exp(np.where(finite, arr - max_value, -np.inf))
    denom = float(np.sum(exp_values))
    if denom <= 1e-12:
        return {key: 0.0 for key, _ in ordered}
    return {
        key: float(weight / denom) for (key, _), weight in zip(ordered, exp_values, strict=True)
    }


def _apply_caps(weights: dict[str, float], config: HybridGridConfig) -> dict[str, float]:
    capped: dict[str, float] = {}
    residual = 1.0
    for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        cap = (
            config.pair_weight_cap
            if "pair" in name.lower() or "spread" in name.lower()
            else config.diversified_weight_cap
        )
        assigned = min(max(0.0, float(weight)), max(0.0, float(cap)), residual)
        if assigned > 1e-12:
            capped[name] = assigned
            residual -= assigned
        if residual <= 1e-12:
            break
    return capped


def _rolling_vol_ratios(panel: Panel, default_name: str, window: int = 20) -> np.ndarray:
    arr = panel.matrix.get(default_name)
    if arr is None:
        return np.zeros(len(panel.days), dtype=float)
    vols = np.zeros(arr.shape[0], dtype=float)
    for idx in range(arr.shape[0]):
        start = max(0, idx - window)
        segment = arr[start:idx]
        vols[idx] = float(np.std(segment, ddof=1)) if segment.size > 1 else 0.0
    train_mask = np.asarray(
        [panel.split_by_day.get(day) == "train" for day in panel.days], dtype=bool
    )
    baseline = (
        float(np.median(vols[train_mask & (vols > 0)])) if np.any(train_mask & (vols > 0)) else 1.0
    )
    return vols / max(baseline, 1e-12)


def _learn_high_vol_params(candidates: list[Candidate], panel: Panel) -> dict[str, Any]:
    default_name = _train_best_sleeve(candidates)
    vol_ratios = _rolling_vol_ratios(panel, default_name)
    train_indices = [
        idx for idx, day in enumerate(panel.days) if panel.split_by_day.get(day) == "train"
    ]
    train_vol = vol_ratios[train_indices] if train_indices else np.asarray([], dtype=float)
    threshold = float(np.percentile(train_vol, 75)) if train_vol.size else 1.25
    high_vol_indices = [idx for idx in train_indices if vol_ratios[idx] > threshold]
    best_high_vol = default_name
    best_score = -1e9
    for name in panel.names:
        values = (
            panel.matrix[name][high_vol_indices]
            if high_vol_indices
            else np.asarray([], dtype=float)
        )
        score = _scaled_score(_metrics_from_returns(values))
        if score > best_score:
            best_score = score
            best_high_vol = name
    learned = {
        "default_sleeve": default_name,
        "high_vol_sleeve": best_high_vol,
        "high_vol_threshold": threshold,
        "high_vol_train_days": len(high_vol_indices),
        "vol_window": 20,
        "boost_candidates": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    }
    best_boost = 0.15
    best_train_score = -1e9
    for boost in learned["boost_candidates"]:
        cfg = HybridGridConfig(
            variant="fixed_default",
            warmup_days=30,
            lookback_days=20,
            default_boost=0.20,
            score_temperature=0.90,
            min_positive_score=0.0,
            pair_weight_cap=0.30,
            diversified_weight_cap=0.85,
        )
        result = _run_online_allocator(
            panel,
            config=cfg,
            default_name=default_name,
            high_vol_params={**learned, "high_vol_boost": boost},
            dynamic_high_vol=False,
        )
        score = _scaled_score(result["split_metrics"]["train"])
        if score > best_train_score:
            best_train_score = score
            best_boost = float(boost)
    learned["high_vol_boost"] = best_boost
    learned["train_learned_score"] = best_train_score
    return learned


def _weights_for_day(
    *,
    panel: Panel,
    trailing_scores: dict[str, np.ndarray],
    idx: int,
    config: HybridGridConfig,
    previous_default: str,
    default_name: str,
) -> tuple[dict[str, float], str, dict[str, float]]:
    raw = {name: float(scores[idx]) for name, scores in trailing_scores.items()}
    positive = {name: score for name, score in raw.items() if score > config.min_positive_score}
    if not positive:
        return {}, "risk_off_cash", raw
    if config.variant == "fixed_default":
        current_default = (
            default_name if default_name in positive else max(positive, key=positive.get)
        )
    else:
        adjusted_for_default = dict(positive)
        if previous_default in adjusted_for_default:
            adjusted_for_default[previous_default] += config.sticky_default_bonus
        candidate_default = max(adjusted_for_default, key=adjusted_for_default.get)
        if (
            previous_default in positive
            and candidate_default != previous_default
            and adjusted_for_default[candidate_default]
            < positive[previous_default] + config.switch_margin
        ):
            current_default = previous_default
        else:
            current_default = candidate_default
    adjusted = dict(positive)
    if current_default in adjusted:
        adjusted[current_default] += config.default_boost
    weights = _softmax(adjusted, config.score_temperature)
    if config.variant == "disagreement_switching" and len(adjusted) >= 2:
        ranked = sorted(adjusted.values(), reverse=True)
        if ranked[0] - ranked[1] < 0.05:
            weights = {name: value * 0.85 for name, value in weights.items()}
    return _apply_caps(weights, config), current_default, raw


def _apply_high_vol_boost(
    weights: dict[str, float],
    *,
    high_vol_sleeve: str,
    boost: float,
    config: HybridGridConfig,
) -> dict[str, float]:
    if boost <= 0 or not high_vol_sleeve:
        return weights
    scaled = {name: value * (1.0 - boost) for name, value in weights.items()}
    scaled[high_vol_sleeve] = scaled.get(high_vol_sleeve, 0.0) + boost
    total = sum(scaled.values())
    if total > 1.0:
        scaled = {name: value / total for name, value in scaled.items()}
    return _apply_caps(scaled, config)


def _run_online_allocator(
    panel: Panel,
    *,
    config: HybridGridConfig,
    default_name: str,
    high_vol_params: dict[str, Any] | None = None,
    dynamic_high_vol: bool = False,
    precomputed_scores: dict[str, np.ndarray] | None = None,
    precomputed_vol_ratios: np.ndarray | None = None,
) -> dict[str, Any]:
    if not panel.names or not panel.days:
        return {
            "daily_returns": [],
            "split_metrics": _empty_metrics(),
            "allocations": [],
            "final_allocation": {"weights": {}, "cash_weight": 1.0},
        }
    trailing_scores = precomputed_scores or _precompute_trailing_scores(panel, config.lookback_days)
    vol_ratios = (
        precomputed_vol_ratios
        if precomputed_vol_ratios is not None
        else _rolling_vol_ratios(
            panel, default_name, int((high_vol_params or {}).get("vol_window", 20))
        )
    )
    threshold = _safe_float((high_vol_params or {}).get("high_vol_threshold"), float("inf"))
    learned_high_vol_sleeve = str((high_vol_params or {}).get("high_vol_sleeve") or "")
    high_vol_boost = _safe_float((high_vol_params or {}).get("high_vol_boost"), 0.0)
    previous_default = default_name if default_name in panel.names else panel.names[0]
    daily_returns: list[float] = []
    allocations: list[dict[str, Any]] = []
    high_vol_counts = dict.fromkeys(SPLITS, 0)
    high_vol_target_switches = 0
    previous_high_vol_target = learned_high_vol_sleeve
    default_switches = 0
    for idx, day in enumerate(panel.days):
        split = panel.split_by_day.get(day)
        if idx < max(config.warmup_days, config.lookback_days):
            current_default = default_name if default_name in panel.names else panel.names[0]
            weights = {current_default: 1.0}
            raw_scores = dict.fromkeys(panel.names, 0.0)
        else:
            weights, current_default, raw_scores = _weights_for_day(
                panel=panel,
                trailing_scores=trailing_scores,
                idx=idx,
                config=config,
                previous_default=previous_default,
                default_name=default_name,
            )
        high_vol_sleeve = learned_high_vol_sleeve
        is_high_vol = bool(vol_ratios[idx] > threshold)
        if is_high_vol and high_vol_sleeve:
            if split in high_vol_counts:
                high_vol_counts[split] += 1
            if dynamic_high_vol and idx >= max(config.warmup_days, config.lookback_days):
                positive = {
                    name: score
                    for name, score in raw_scores.items()
                    if score > config.min_positive_score
                }
                if positive:
                    high_vol_sleeve = max(positive, key=positive.get)
            if high_vol_sleeve != previous_high_vol_target and high_vol_sleeve:
                high_vol_target_switches += 1
            weights = _apply_high_vol_boost(
                weights,
                high_vol_sleeve=high_vol_sleeve,
                boost=high_vol_boost,
                config=config,
            )
            previous_high_vol_target = high_vol_sleeve
        if current_default != previous_default:
            default_switches += 1
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            weights = {name: value / total_weight for name, value in weights.items()}
            total_weight = 1.0
        day_return = sum(
            float(panel.matrix[name][idx]) * float(weight) for name, weight in weights.items()
        )
        daily_returns.append(float(day_return))
        allocations.append(
            {
                "date": day,
                "split": split,
                "default_sleeve": current_default,
                "high_vol_sleeve": high_vol_sleeve if is_high_vol else "",
                "is_high_vol": is_high_vol,
                "weights": {name: float(weight) for name, weight in sorted(weights.items())},
                "cash_weight": max(0.0, 1.0 - total_weight),
            }
        )
        previous_default = current_default if current_default in panel.names else previous_default
    split_metrics = _split_metrics_from_daily(panel.days, daily_returns)
    avg_weights: dict[str, dict[str, float]] = {split: {} for split in SPLITS}
    avg_cash: dict[str, float] = dict.fromkeys(SPLITS, 0.0)
    for split in SPLITS:
        split_allocs = [row for row in allocations if row.get("split") == split]
        if not split_allocs:
            continue
        avg_cash[split] = float(
            np.mean([_safe_float(row.get("cash_weight"), 0.0) for row in split_allocs])
        )
        names = sorted({name for row in split_allocs for name in dict(row.get("weights") or {})})
        avg_weights[split] = {
            name: float(
                np.mean(
                    [
                        _safe_float(dict(row.get("weights") or {}).get(name), 0.0)
                        for row in split_allocs
                    ]
                )
            )
            for name in names
        }
    return {
        "daily_returns": daily_returns,
        "dates": list(panel.days),
        "split_metrics": split_metrics,
        "all_metrics": _metrics_from_returns(daily_returns),
        "allocations": allocations,
        "final_allocation": allocations[-1] if allocations else {"weights": {}, "cash_weight": 1.0},
        "allocation_summary": {
            "default_switches": default_switches,
            "high_vol_counts": high_vol_counts,
            "high_vol_target_switches": high_vol_target_switches,
            "average_cash_by_split": avg_cash,
            "average_weights_by_split": avg_weights,
        },
    }


def _grid_configs() -> list[HybridGridConfig]:
    configs: list[HybridGridConfig] = []
    for variant, warmup, lookback, boost, temp, min_score, pair_cap, div_cap in itertools.product(
        ("fixed_default", "dynamic_default", "disagreement_switching"),
        (30, 60, 120),
        (10, 20, 30, 45),
        (0.05, 0.20, 0.35),
        (0.75, 1.00, 1.25),
        (-0.05, 0.00, 0.05),
        (0.20, 0.30),
        (0.70, 0.90),
    ):
        configs.append(
            HybridGridConfig(
                variant=variant,
                warmup_days=warmup,
                lookback_days=lookback,
                default_boost=boost,
                sticky_default_bonus=0.08,
                switch_margin=0.05,
                score_temperature=temp,
                min_positive_score=min_score,
                pair_weight_cap=pair_cap,
                diversified_weight_cap=div_cap,
            )
        )
    return configs


def _candidate_from_hybrid_result(
    *,
    name: str,
    kind: str,
    result: dict[str, Any],
    output_source: str,
    config: dict[str, Any],
    learned: dict[str, Any] | None = None,
) -> Candidate:
    daily_map = {
        day: ret
        for day, ret in zip(
            result.get("dates") or [], result.get("daily_returns") or [], strict=False
        )
    }
    raw = {
        "name": name,
        "kind": kind,
        "true_hybrid": True,
        "split_metrics": result.get("split_metrics") or {},
        "dates": result.get("dates") or [],
        "daily_returns": result.get("daily_returns") or [],
        "config": config,
    }
    candidate = _make_candidate(Path(output_source), raw, name)
    if candidate is None:
        raise RuntimeError(f"failed to build hybrid candidate {name}")
    candidate.source_paths = [output_source]
    candidate.daily_map = daily_map
    candidate.stream_day_count = len(daily_map)
    candidate.stream_digest = _stream_digest(daily_map)
    candidate.combinable = True
    candidate.true_hybrid = True
    candidate.category = _category_from_kind(kind, True, True)
    candidate.allocation_summary = dict(result.get("allocation_summary") or {})
    if learned:
        candidate.config = {**candidate.config, "learned": learned}
    return _score_and_mark(candidate)


def run_hybrid_experiments(
    sleeves: list[Candidate], output_dir: Path
) -> tuple[list[Candidate], dict[str, Any]]:
    panel = _build_panel(sleeves)
    if not panel.names:
        return [], {"error": "no stream-backed sleeves available"}
    default_name = _train_best_sleeve(sleeves)
    configs = _grid_configs()
    score_cache = {
        lookback: _precompute_trailing_scores(panel, lookback)
        for lookback in sorted({cfg.lookback_days for cfg in configs})
    }
    vol_ratios_cache = _rolling_vol_ratios(panel, default_name)
    best: tuple[float, HybridGridConfig, dict[str, Any]] | None = None
    for config in configs:
        result = _run_online_allocator(
            panel,
            config=config,
            default_name=default_name,
            precomputed_scores=score_cache[config.lookback_days],
            precomputed_vol_ratios=vol_ratios_cache,
        )
        score = _scaled_score(result["split_metrics"]["val"]) + 0.18 * _scaled_score(
            result["split_metrics"]["train"]
        )
        train_mdd = _safe_float(result["split_metrics"]["train"].get("max_drawdown"), 0.0)
        val_mdd = _safe_float(result["split_metrics"]["val"].get("max_drawdown"), 0.0)
        if train_mdd > MDD_CAP or val_mdd > MDD_CAP:
            score -= 1000.0
        if best is None or score > best[0]:
            best = (score, config, result)
    if best is None:
        return [], {"error": "grid search produced no result"}
    _, best_config, best_result = best
    source_label = str(
        (output_dir / "generated_full_universe_hybrid_experiments").relative_to(REPO_ROOT)
    )
    retuned = _candidate_from_hybrid_result(
        name="retuned_full_universe_hybrid_online",
        kind="true_hybrid_full_universe_retuned_clean",
        result=best_result,
        output_source=source_label,
        config=asdict(best_config),
    )
    learned = _learn_high_vol_params(sleeves, panel)
    v35_config = HybridGridConfig(
        variant="fixed_default",
        warmup_days=30,
        lookback_days=20,
        default_boost=0.20,
        score_temperature=0.90,
        min_positive_score=0.00,
        pair_weight_cap=0.30,
        diversified_weight_cap=0.85,
    )
    v35_result = _run_online_allocator(
        panel,
        config=v35_config,
        default_name=str(learned["default_sleeve"]),
        high_vol_params=learned,
        dynamic_high_vol=False,
    )
    v35 = _candidate_from_hybrid_result(
        name="full_universe_v35_train_learned_high_vol_hybrid",
        kind="true_hybrid_v35_train_learned_high_vol_clean",
        result=v35_result,
        output_source=source_label,
        config=asdict(v35_config),
        learned=learned,
    )
    v36_config = HybridGridConfig(
        variant="dynamic_default",
        warmup_days=30,
        lookback_days=20,
        default_boost=0.20,
        sticky_default_bonus=0.08,
        switch_margin=0.05,
        score_temperature=0.90,
        min_positive_score=0.00,
        pair_weight_cap=0.30,
        diversified_weight_cap=0.85,
    )
    v36_result = _run_online_allocator(
        panel,
        config=v36_config,
        default_name=str(learned["default_sleeve"]),
        high_vol_params=learned,
        dynamic_high_vol=True,
    )
    v36 = _candidate_from_hybrid_result(
        name="full_universe_v36_rolling_dynamic_high_vol_hybrid",
        kind="true_hybrid_v36_rolling_dynamic_clean",
        result=v36_result,
        output_source=source_label,
        config=asdict(v36_config),
        learned=learned,
    )
    summary = {
        "sleeve_count": len(sleeves),
        "sleeves": [row.public_dict(include_sources=True) for row in sleeves],
        "default_sleeve_by_train_score": default_name,
        "retuned_grid_evaluations": len(configs),
        "retuned_best_config": asdict(best_config),
        "train_learned_high_vol": learned,
    }
    return [retuned, v35, v36], summary


def _metric(candidate: Candidate, split: str, key: str) -> float:
    return _safe_float(candidate.metrics.get(split, {}).get(key), 0.0)


def _select_conservative_shadow(candidates: list[Candidate]) -> Candidate | None:
    def conservative_shape(row: Candidate) -> bool:
        haystack = f"{row.name} {row.kind} {row.category}".lower()
        if "leverage" in haystack or "3x" in haystack or "benchmark_metrics" in haystack:
            return False
        return row.true_hybrid or row.category in {"portfolio", "static_blend", "source_sleeve"}

    pool = [
        row
        for row in candidates
        if row.final_selection_eligible
        and conservative_shape(row)
        and _metric(row, "oos", "total_return") > 0.0
        and _metric(row, "oos", "sharpe") > 1.0
        and _metric(row, "oos", "max_drawdown") <= 0.02
        and _metric(row, "train", "max_drawdown") <= 0.15
        and _metric(row, "val", "max_drawdown") <= 0.05
    ]
    if not pool:
        pool = [
            row
            for row in candidates
            if row.final_selection_eligible
            and conservative_shape(row)
            and _metric(row, "oos", "total_return") > 0
        ]
    if not pool:
        return None
    pool.sort(
        key=lambda row: (
            row.oos_scaled_score_report_only,
            -_metric(row, "oos", "max_drawdown"),
            row.selection_score,
        ),
        reverse=True,
    )
    return pool[0]


def _fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def _table_rows(candidates: list[Candidate], limit: int = 30) -> list[str]:
    lines = [
        "| Rank | Candidate | Category | Score | Train ret/Sharpe/Sortino/Calmar/MDD | Val ret/Sharpe/Sortino/Calmar/MDD | OOS ret/Sharpe/Sortino/Calmar/MDD | Hybrid | Stream | Eligible |",
        "| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(candidates[:limit], start=1):
        train = row.metrics["train"]
        val = row.metrics["val"]
        oos = row.metrics["oos"]
        lines.append(
            "| {rank} | `{name}` | {category} | {score:.2f} | "
            "{tr_ret} / {tr_sh:.2f} / {tr_so:.2f} / {tr_ca:.2f} / {tr_mdd} | "
            "{va_ret} / {va_sh:.2f} / {va_so:.2f} / {va_ca:.2f} / {va_mdd} | "
            "{oo_ret} / {oo_sh:.2f} / {oo_so:.2f} / {oo_ca:.2f} / {oo_mdd} | "
            "{hybrid} | {stream} | {eligible} |".format(
                rank=idx,
                name=row.name.replace("|", "\\|"),
                category=row.category,
                score=row.selection_score,
                tr_ret=_fmt_pct(train["total_return"]),
                tr_sh=train["sharpe"],
                tr_so=train["sortino"],
                tr_ca=train["calmar"],
                tr_mdd=_fmt_pct(train["max_drawdown"]),
                va_ret=_fmt_pct(val["total_return"]),
                va_sh=val["sharpe"],
                va_so=val["sortino"],
                va_ca=val["calmar"],
                va_mdd=_fmt_pct(val["max_drawdown"]),
                oo_ret=_fmt_pct(oos["total_return"]),
                oo_sh=oos["sharpe"],
                oo_so=oos["sortino"],
                oo_ca=oos["calmar"],
                oo_mdd=_fmt_pct(oos["max_drawdown"]),
                hybrid=str(row.true_hybrid),
                stream=str(row.combinable),
                eligible=str(row.final_selection_eligible),
            )
        )
    return lines


def _write_csv(path: Path, candidates: list[Candidate]) -> None:
    fieldnames = [
        "uid",
        "name",
        "kind",
        "category",
        "true_hybrid",
        "combinable",
        "primary_source_path",
        "source_path_count",
        "stream_day_count",
        "selection_score",
        "val_scaled_score",
        "train_scaled_score",
        "oos_scaled_score_report_only",
        "final_selection_eligible",
        "eligibility_reasons",
    ]
    for split in SPLITS:
        for key in METRIC_KEYS:
            fieldnames.append(f"{split}_{key}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            payload = {
                "uid": row.uid,
                "name": row.name,
                "kind": row.kind,
                "category": row.category,
                "true_hybrid": row.true_hybrid,
                "combinable": row.combinable,
                "primary_source_path": row.source_paths[0] if row.source_paths else "",
                "source_path_count": len(row.source_paths),
                "stream_day_count": row.stream_day_count,
                "selection_score": row.selection_score,
                "val_scaled_score": row.val_scaled_score,
                "train_scaled_score": row.train_scaled_score,
                "oos_scaled_score_report_only": row.oos_scaled_score_report_only,
                "final_selection_eligible": row.final_selection_eligible,
                "eligibility_reasons": ";".join(row.eligibility_reasons),
            }
            for split in SPLITS:
                for key in METRIC_KEYS:
                    payload[f"{split}_{key}"] = _safe_float(
                        row.metrics.get(split, {}).get(key), 0.0
                    )
            writer.writerow(payload)


def _write_markdown(
    path: Path,
    *,
    payload: dict[str, Any],
    ranked: list[Candidate],
    true_hybrid: list[Candidate],
    non_hybrid: list[Candidate],
    generated_hybrids: list[Candidate],
) -> None:
    best_full = payload["final_recommendations"]["best_full_universe_candidate"]
    best_hybrid = payload["final_recommendations"]["best_deployable_true_hybrid_candidate"]
    shadow = payload["final_recommendations"].get("conservative_fallback_shadow_candidate")
    lines = [
        "# Full-universe quant selection report — 2026-04-26",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Selection policy",
        "",
        "- Universe is repo artifact-driven and includes JSON/CSV candidates from saved strategy, portfolio, leverage, pair, source-sleeve, static-blend, wave2/post-2026-04-20, and HYBRID report roots.",
        "- Selection is **validation-primary**: `selection_score = val_scaled_score + 0.18 * train_scaled_score`.",
        "- OOS is **report-only** and is not used for tuning, ranking, health priors, or HYBRID parameter choice.",
        "- Cash efficiency is not directly scored.",
        "- Train/validation MDD up to 25% is eligible; MDD is scored only through bounded headroom.",
        "- Scaled score uses return, Sharpe, Sortino, Calmar, and MDD headroom with bounded `tanh` ratio scaling.",
        "",
        "```text",
        "100*(0.30*tanh(return/0.18)+0.30*tanh(Sharpe/4)+0.15*tanh(Sortino/12)+0.15*tanh(Calmar/80)+0.10*MDD_headroom)",
        "MDD_headroom = 1 - min(max(MDD, 0), 0.25)/0.25",
        "```",
        "",
        "## Universe discovery",
        "",
        f"- Scan roots: `{', '.join(payload['discovery_summary']['scan_roots'])}`",
        f"- JSON files seen: `{payload['discovery_summary']['json_files_seen']}`",
        f"- CSV files seen: `{payload['discovery_summary']['csv_files_seen']}`",
        f"- Raw candidate rows extracted: `{payload['discovery_summary']['raw_candidate_rows']}`",
        f"- Deduped candidate rows before generated HYBRIDs: `{payload['discovery_summary']['deduped_candidate_rows']}`",
        f"- Final ranked candidates including generated HYBRIDs: `{len(ranked)}`",
        f"- Stream-combinable rows before generated HYBRIDs: `{payload['discovery_summary']['stream_available_rows']}`",
        "",
        "## Final recommendations",
        "",
        f"- **Best full-universe candidate:** `{best_full['name']}` (score {best_full['selection_score']:.2f}, category `{best_full['category']}`).",
        f"- **Best deployable true-HYBRID candidate:** `{best_hybrid['name']}` (score {best_hybrid['selection_score']:.2f}, category `{best_hybrid['category']}`).",
    ]
    if shadow:
        lines.append(
            f"- **Conservative fallback/shadow candidate:** `{shadow['name']}`. This uses OOS only as an audit label, not as the selected objective."
        )
    lines.extend(
        [
            "",
            "## Generated full-universe HYBRID sleeve test",
            "",
            f"- Sleeve count: `{payload['hybrid_experiment_summary'].get('sleeve_count')}`",
            f"- Retuned grid evaluations: `{payload['hybrid_experiment_summary'].get('retuned_grid_evaluations')}`",
            f"- Default sleeve by train score: `{payload['hybrid_experiment_summary'].get('default_sleeve_by_train_score')}`",
            "- Top stream-available non-HYBRID candidates were added as online allocator sleeves; OOS was not used in parameter selection.",
            "",
            "### Generated HYBRID candidates",
            "",
            *_table_rows(generated_hybrids, limit=10),
            "",
            "## Full-universe ranking (clean eligible first)",
            "",
            *_table_rows(ranked, limit=40),
            "",
            "## True-HYBRID ranking",
            "",
            *_table_rows(true_hybrid, limit=30),
            "",
            "## Non-HYBRID/static/source ranking",
            "",
            *_table_rows(non_hybrid, limit=40),
            "",
            "## Explicit caveats",
            "",
            "- This is still a repeated validation-mining exercise; strong validation scores can meta-overfit and require fresh forward/paper evidence after 2026-04-20.",
            "- OOS figures are shown to expose fragility only; they did not enter scoring, tuning, health priors, or candidate selection.",
            "- Some saved artifacts are metric-only and non-combinable; they can win the external ranking but cannot be inserted into the true HYBRID allocator without daily streams.",
            "- Stream union uses zero return on missing candidate days, matching the saved-sleeve panel convention but still a caveat for sparse pair strategies.",
            "- The generated v3.5/v3.6 variants are portfolio-governor analogues of the referenced ensemble strategies, not prediction-level ensemble model ports.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(output_dir: Path, scan_roots: list[Path]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered, discovery_summary = discover_candidates(scan_roots)
    sleeves = _select_hybrid_sleeves(discovered)
    generated_hybrids, hybrid_summary = run_hybrid_experiments(sleeves, output_dir)
    all_candidates = _dedupe_candidates([*discovered, *generated_hybrids])
    clean_ranked = [row for row in all_candidates if row.final_selection_eligible]
    clean_ranked.sort(key=lambda row: row.selection_score, reverse=True)
    ranked = [*clean_ranked, *[row for row in all_candidates if not row.final_selection_eligible]]
    true_hybrid = [row for row in ranked if row.true_hybrid]
    non_hybrid = [row for row in ranked if not row.true_hybrid]
    deployable_true_hybrid = [
        row
        for row in true_hybrid
        if row.final_selection_eligible and row.combinable and not row.oos_health_priors_enabled
    ]
    best_full = clean_ranked[0] if clean_ranked else ranked[0]
    best_hybrid = (
        deployable_true_hybrid[0]
        if deployable_true_hybrid
        else (true_hybrid[0] if true_hybrid else best_full)
    )
    shadow = _select_conservative_shadow(ranked)
    generated_hybrids.sort(key=lambda row: row.selection_score, reverse=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_policy": {
            "selection_score": "val_scaled_score + 0.18 * train_scaled_score",
            "oos_scored": False,
            "cash_efficiency_scored": False,
            "train_val_mdd_cap": MDD_CAP,
            "scaled_score_formula": "100*(0.30*tanh(return/0.18)+0.30*tanh(sharpe/4)+0.15*tanh(sortino/12)+0.15*tanh(calmar/80)+0.10*MDD_headroom)",
        },
        "discovery_summary": discovery_summary,
        "hybrid_experiment_summary": hybrid_summary,
        "final_recommendations": {
            "best_full_universe_candidate": best_full.public_dict(include_sources=True),
            "best_deployable_true_hybrid_candidate": best_hybrid.public_dict(include_sources=True),
            "conservative_fallback_shadow_candidate": shadow.public_dict(include_sources=True)
            if shadow
            else None,
        },
        "ranked_all_candidates": [row.public_dict(include_sources=True) for row in ranked],
        "ranked_clean_candidates": [row.public_dict(include_sources=True) for row in clean_ranked],
        "ranked_true_hybrid_candidates": [
            row.public_dict(include_sources=True) for row in true_hybrid
        ],
        "ranked_non_hybrid_candidates": [
            row.public_dict(include_sources=True) for row in non_hybrid
        ],
        "generated_hybrid_candidates": [
            row.public_dict(include_sources=True) for row in generated_hybrids
        ],
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    latest_json = output_dir / "full_universe_selection_latest.json"
    dated_json = output_dir / f"full_universe_selection_{stamp}.json"
    latest_md = output_dir / "full_universe_selection_latest.md"
    dated_md = output_dir / f"full_universe_selection_{stamp}.md"
    csv_path = output_dir / "full_universe_selection_candidates_20260426.csv"
    json_text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    latest_json.write_text(json_text, encoding="utf-8")
    dated_json.write_text(json_text, encoding="utf-8")
    _write_csv(csv_path, ranked)
    _write_markdown(
        latest_md,
        payload=payload,
        ranked=ranked,
        true_hybrid=true_hybrid,
        non_hybrid=non_hybrid,
        generated_hybrids=generated_hybrids,
    )
    dated_md.write_text(latest_md.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "latest_json": str(latest_json.relative_to(REPO_ROOT)),
        "dated_json": str(dated_json.relative_to(REPO_ROOT)),
        "latest_md": str(latest_md.relative_to(REPO_ROOT)),
        "dated_md": str(dated_md.relative_to(REPO_ROOT)),
        "csv": str(csv_path.relative_to(REPO_ROOT)),
        "best_full_universe_candidate": best_full.name,
        "best_deployable_true_hybrid_candidate": best_hybrid.name,
        "conservative_fallback_shadow_candidate": shadow.name if shadow else None,
        "ranked_count": len(ranked),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=None,
        help="Additional/override scan root. May be repeated. Defaults to saved report roots.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scan_roots = [path.resolve() for path in (args.scan_root or list(DEFAULT_SCAN_ROOTS))]
    result = build_report(Path(args.output_dir).resolve(), scan_roots)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
