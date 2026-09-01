#!/usr/bin/env python3
"""Ingest candidate-grid research output and classify it for second-gen design.

The data-PC round emits per-cost-cell candidate reports (see
``lumina_quant.strategy_factory.research_reporting`` for the per-candidate row
shape and ``research_entrypoints._candidate_research_report_payload`` for the
``{"candidates": [...]}`` report envelope) plus combined summaries that
aggregate pass / hard-reject counts.  This analyzer is liberal in what it
accepts -- a single report JSON, a directory of per-cost report JSONs, or a
combined summary -- and strict in what it emits: a deterministic two-tier gate
classification, per-dimension aggregates, cross-cost robustness flags, and a
ranked second-generation shortlist plus a family kill-list.

Pool-admission gate (per row, evaluated on locked OOS):

* ``PASS``        -- net edge > 0 AND deflated Sharpe > 0 AND (where a factor IC
  is present) factor IC > 0.
* ``NEAR-MISS``   -- fails exactly one clause (records the blocking clause and
  its margin below the threshold).
* ``DEAD``        -- fails two or more clauses, or trips a hard-reject gate.
* ``INSUFFICIENT``-- flagged ``insufficient_data`` (records the missing-symbol
  histogram).

Pure stdlib + numpy, never raises on partial / missing fields, deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

# --- classification vocabulary (fixed order for deterministic emit) ----------
PASS = "PASS"
NEAR_MISS = "NEAR-MISS"
DEAD = "DEAD"
INSUFFICIENT = "INSUFFICIENT"
CLASS_ORDER: tuple[str, ...] = (PASS, NEAR_MISS, DEAD, INSUFFICIENT)

# Report envelopes and combined summaries nest candidate rows under these keys.
_CONTAINER_KEYS: tuple[str, ...] = (
    "candidates",
    "rows",
    "reports",
    "results",
    "cells",
    "per_cost",
    "cost_cells",
)
# A dict that carries any of these keys is treated as an individual candidate row.
_ROW_SIGNAL_KEYS: frozenset[str] = frozenset(
    {
        "strategy_class",
        "candidate_id",
        "family",
        "oos",
        "train",
        "val",
        "hard_reject",
        "hard_reject_reasons",
        "params",
        "timeframe",
        "strategy_timeframe",
    }
)
_MAX_INGEST_DEPTH = 8
_COST_BPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*bps", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Liberal ingest.                                                             #
# --------------------------------------------------------------------------- #
def _looks_like_row(obj: Mapping[str, Any]) -> bool:
    """A mapping is a candidate row when it signals row fields but no container."""
    if any(isinstance(obj.get(key), (list, dict)) for key in _CONTAINER_KEYS):
        return False
    return bool(_ROW_SIGNAL_KEYS.intersection(obj.keys()))


def iter_rows(obj: Any, *, depth: int = 0) -> Iterable[Mapping[str, Any]]:
    """Yield candidate rows from any accepted envelope, liberally and safely."""
    if depth > _MAX_INGEST_DEPTH:
        return
    if isinstance(obj, list):
        for item in obj:
            yield from iter_rows(item, depth=depth + 1)
        return
    if not isinstance(obj, Mapping):
        return
    if _looks_like_row(obj):
        yield obj
        return
    matched_container = False
    for key in _CONTAINER_KEYS:
        value = obj.get(key)
        if isinstance(value, (list, dict)):
            matched_container = True
            yield from iter_rows(value, depth=depth + 1)
    if not matched_container:
        # Unknown envelope: descend into every nested collection rather than raise.
        for value in obj.values():
            if isinstance(value, (list, dict)):
                yield from iter_rows(value, depth=depth + 1)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except OSError, ValueError:
        return None


def _cost_from_filename(name: str) -> str:
    match = _COST_BPS_RE.search(name)
    if match is None:
        return "unknown"
    return _normalize_cost_label(match.group(1))


def _is_analyzer_output(payload: Any) -> bool:
    """This analyzer's own JSON output is not a candidate grid; never re-ingest it.

    Guards the common footgun of writing ``--json out.json`` into the same
    directory that is later scanned, whose classification detail rows would
    otherwise be mistaken for candidate rows.
    """
    return isinstance(payload, Mapping) and "classification_counts" in payload


def load_grid(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load rows from a file or directory; tag directory rows with file cost cells."""
    prepared: list[dict[str, Any]] = []
    files = sorted(p for p in path.glob("*.json") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        payload = _load_json(file)
        if payload is None or _is_analyzer_output(payload):
            continue
        default_cost = _cost_from_filename(file.name)
        for row in iter_rows(payload):
            prepared.append(_prepare_row(row, default_cost))
    return prepared, str(path)


def _prepare_row(row: Mapping[str, Any], default_cost: str) -> dict[str, Any]:
    """Shallow-copy a row and stamp a fallback cost cell when it lacks one."""
    prepared = dict(row)
    if default_cost != "unknown" and _raw_cost_cell(prepared) is None:
        prepared["_analyzer_cost_cell"] = default_cost
    return prepared


# --------------------------------------------------------------------------- #
# Metric extraction (never raises).                                           #
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    """Coerce to float, rejecting booleans, NaN, and unparseable values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _first_num(*values: Any) -> float | None:
    for value in values:
        parsed = _num(value)
        if parsed is not None:
            return parsed
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_cost_label(value: Any) -> str:
    parsed = _num(value)
    if parsed is None:
        text = str(value).strip()
        return text or "unknown"
    if parsed == int(parsed):
        return f"{int(parsed)}bps"
    return f"{parsed:g}bps"


def _raw_cost_cell(row: Mapping[str, Any]) -> str | None:
    meta = _mapping(row.get("metadata"))
    direct = _first_num(
        row.get("cost_bps"),
        row.get("round_trip_cost_bps"),
        meta.get("cost_bps"),
        meta.get("round_trip_cost_bps"),
    )
    if direct is not None:
        return _normalize_cost_label(direct)
    cost_rate = _num(meta.get("cost_rate"))
    if cost_rate is not None:
        return _normalize_cost_label(cost_rate * 10_000.0)
    for candidate in (row.get("cost_cell"), meta.get("cost_cell")):
        if candidate not in (None, ""):
            return _normalize_cost_label(candidate)
    return None


def cost_cell_of(row: Mapping[str, Any]) -> str:
    resolved = _raw_cost_cell(row)
    if resolved is not None:
        return resolved
    injected = row.get("_analyzer_cost_cell")
    if injected not in (None, ""):
        return str(injected)
    return "unknown"


def extract_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the gate metrics from wherever the producer placed them."""
    oos = _mapping(row.get("oos"))
    meta = _mapping(row.get("metadata"))

    oos_sharpe = _first_num(oos.get("sharpe"), row.get("oos_sharpe"))
    dsr = _first_num(
        oos.get("deflated_sharpe"),
        oos.get("dsr"),
        row.get("deflated_sharpe"),
        row.get("dsr"),
    )
    factor_ic = _first_num(
        row.get("factor_ic"),
        oos.get("factor_ic"),
        meta.get("factor_ic"),
    )

    net_edge = _first_num(
        row.get("net_edge"),
        row.get("net_edge_bps"),
        oos.get("net_edge"),
        oos.get("net_edge_bps"),
        meta.get("net_edge"),
    )
    net_edge_source = "explicit"
    if net_edge is None:
        # No explicit net-edge field: fall back to the cost-adjusted OOS return,
        # which the backtest already books net of fees/slippage.
        net_edge = _first_num(oos.get("total_return"), oos.get("return"))
        net_edge_source = "oos_return_proxy" if net_edge is not None else "missing"

    return {
        "net_edge": net_edge,
        "net_edge_source": net_edge_source,
        "dsr": dsr,
        "factor_ic": factor_ic,
        "oos_sharpe": oos_sharpe,
    }


def build_identity_key(row: Mapping[str, Any]) -> str:
    """Cost-independent identity so a strategy can be tracked across cost cells."""
    strategy_class = str(row.get("strategy_class") or row.get("strategy") or "")
    timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
    params = row.get("params")
    params_key = json.dumps(params, sort_keys=True, default=str) if params else "{}"
    symbols = row.get("symbols") or []
    if isinstance(symbols, (list, tuple)):
        symbols_key = ",".join(sorted(str(s) for s in symbols))
    else:
        symbols_key = str(symbols)
    return "|".join((strategy_class, timeframe, params_key, symbols_key))


# --------------------------------------------------------------------------- #
# Two-tier gate classification.                                               #
# --------------------------------------------------------------------------- #
def _clause_status(value: float | None) -> str:
    if value is None:
        return "missing"
    return "pass" if value > 0.0 else "fail"


def classify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one row into PASS / NEAR-MISS / DEAD / INSUFFICIENT.

    Never raises: a row missing a metric is classified on the evidence that
    exists and flagged ``incomplete``.
    """
    metrics = extract_metrics(row)
    reasons = _mapping(row.get("hard_reject_reasons"))
    meta = _mapping(row.get("metadata"))
    identity = build_identity_key(row)

    detail: dict[str, Any] = {
        "candidate_id": str(row.get("candidate_id") or ""),
        "name": str(row.get("name") or ""),
        "strategy_class": str(row.get("strategy_class") or row.get("strategy") or "unknown"),
        "family": str(row.get("family") or "unknown"),
        "timeframe": str(row.get("strategy_timeframe") or row.get("timeframe") or "unknown"),
        "cost_cell": cost_cell_of(row),
        "identity_key": identity,
        "net_edge": metrics["net_edge"],
        "net_edge_source": metrics["net_edge_source"],
        "dsr": metrics["dsr"],
        "factor_ic": metrics["factor_ic"],
        "oos_sharpe": metrics["oos_sharpe"],
        "incomplete": False,
        "failing_clauses": [],
        "missing_clauses": [],
        "near_miss_blocker": None,
        "hard_reject_reasons": {},
        "missing_symbols": [],
    }

    # INSUFFICIENT takes precedence: no usable evidence to gate on.
    if reasons.get("insufficient_data"):
        detail["classification"] = INSUFFICIENT
        detail["incomplete"] = True
        detail["missing_symbols"] = _missing_symbols(meta)
        return detail

    # Any hard-reject gate (below-floor OOS Sharpe, DSR, cost stress, ...) is DEAD.
    if bool(row.get("hard_reject")) and reasons:
        detail["classification"] = DEAD
        detail["hard_reject_reasons"] = {str(k): reasons[k] for k in sorted(reasons)}
        return detail

    net_status = _clause_status(metrics["net_edge"])
    dsr_status = _clause_status(metrics["dsr"])
    ic_present = metrics["factor_ic"] is not None
    ic_status = _clause_status(metrics["factor_ic"]) if ic_present else "na"

    clause_values = {
        "net_edge": metrics["net_edge"],
        "dsr": metrics["dsr"],
        "factor_ic": metrics["factor_ic"],
    }
    clause_status = {"net_edge": net_status, "dsr": dsr_status, "factor_ic": ic_status}

    failing = [c for c in ("net_edge", "dsr", "factor_ic") if clause_status[c] == "fail"]
    # A missing mandatory metric cannot clear the gate; it blocks like a failure.
    missing_mandatory = [c for c in ("net_edge", "dsr") if clause_status[c] == "missing"]

    detail["failing_clauses"] = failing
    detail["missing_clauses"] = missing_mandatory
    detail["incomplete"] = bool(missing_mandatory) or metrics["oos_sharpe"] is None

    blockers = failing + missing_mandatory
    if not blockers:
        detail["classification"] = PASS
    elif len(blockers) == 1:
        detail["classification"] = NEAR_MISS
        clause = blockers[0]
        if clause in failing:
            value = clause_values[clause]
            detail["near_miss_blocker"] = {
                "clause": clause,
                "kind": "fail",
                "value": value,
                "margin": value,  # signed distance below the > 0 threshold
            }
        else:
            detail["near_miss_blocker"] = {
                "clause": clause,
                "kind": "missing",
                "value": None,
                "margin": None,
            }
    else:
        detail["classification"] = DEAD

    return detail


def _missing_symbols(meta: Mapping[str, Any]) -> list[str]:
    for key in ("missing_symbols", "missing_support_symbols"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)):
            return [str(s) for s in value]
    return []


# --------------------------------------------------------------------------- #
# Aggregation.                                                                #
# --------------------------------------------------------------------------- #
def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(CLASS_ORDER, 0)


def _row_ref(detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": detail["candidate_id"],
        "name": detail["name"],
        "strategy_class": detail["strategy_class"],
        "family": detail["family"],
        "timeframe": detail["timeframe"],
        "cost_cell": detail["cost_cell"],
        "oos_sharpe": detail["oos_sharpe"],
        "dsr": detail["dsr"],
        "classification": detail["classification"],
    }


def _group_summary(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = _empty_counts()
    for detail in details:
        counts[detail["classification"]] += 1
    scored = [d for d in details if d["oos_sharpe"] is not None]
    best = worst = None
    median_oos_sharpe = None
    if scored:
        # Rank on OOS Sharpe, break ties on DSR; identity keeps it deterministic.
        def _key(d: Mapping[str, Any]) -> tuple[float, float, str]:
            return (
                d["oos_sharpe"],
                d["dsr"] if d["dsr"] is not None else float("-inf"),
                d["identity_key"],
            )

        best = _row_ref(max(scored, key=_key))
        worst = _row_ref(min(scored, key=_key))
        median_oos_sharpe = float(np.median(np.asarray([d["oos_sharpe"] for d in scored])))
    return {
        "total": len(details),
        "counts": counts,
        "best": best,
        "worst": worst,
        "median_oos_sharpe": median_oos_sharpe,
    }


def aggregate(details: Sequence[Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for detail in details:
        groups[str(detail.get(dimension, "unknown"))].append(detail)
    return {key: _group_summary(groups[key]) for key in sorted(groups)}


# --------------------------------------------------------------------------- #
# Cross-cost robustness.                                                       #
# --------------------------------------------------------------------------- #
def _cost_sort_key(label: str) -> tuple[float, str]:
    match = _COST_BPS_RE.search(label)
    if match is not None:
        return (float(match.group(1)), label)
    return (float("inf"), label)


def cross_cost_robustness(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flag sign-stability of OOS Sharpe for identities seen at multiple costs."""
    by_identity: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for detail in details:
        identity = detail["identity_key"]
        cost = detail["cost_cell"]
        existing = by_identity[identity].get(cost)
        # Deterministic pick when an identity repeats at one cost: keep best Sharpe.
        if existing is None or _better_for_cost(detail, existing):
            by_identity[identity][cost] = detail

    reports: list[dict[str, Any]] = []
    for identity in sorted(by_identity):
        per_cost = by_identity[identity]
        if len(per_cost) < 2:
            continue
        ordered = sorted(per_cost.items(), key=lambda kv: _cost_sort_key(kv[0]))
        sharpes = [(cost, d["oos_sharpe"]) for cost, d in ordered]
        signs = [np.sign(v) for _, v in sharpes if v is not None and v != 0.0]
        if len(signs) < 2:
            sign_stable: bool | None = None
        else:
            sign_stable = all(s == signs[0] for s in signs)
        exemplar = ordered[0][1]
        reports.append(
            {
                "identity_key": identity,
                "strategy_class": exemplar["strategy_class"],
                "family": exemplar["family"],
                "timeframe": exemplar["timeframe"],
                "cost_cells": [cost for cost, _ in sharpes],
                "oos_sharpe_by_cost": {cost: value for cost, value in sharpes},
                "sign_stable": sign_stable,
                "monotone_decay": _is_monotone_decay([v for _, v in sharpes]),
            }
        )
    return reports


def _better_for_cost(candidate: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
    cand = candidate["oos_sharpe"]
    prev = existing["oos_sharpe"]
    if cand is None:
        return False
    if prev is None:
        return True
    if cand != prev:
        return cand > prev
    return candidate["candidate_id"] < existing["candidate_id"]


def _is_monotone_decay(values: Sequence[float | None]) -> bool:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return False
    non_increasing = all(later <= earlier for earlier, later in pairwise(present))
    return non_increasing and present[-1] < present[0]


def _stability_score(report: Mapping[str, Any]) -> int:
    stable = report.get("sign_stable")
    if stable is True:
        return 1
    if stable is False:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Second-generation shortlist + kill-list.                                     #
# --------------------------------------------------------------------------- #
def build_shortlist(
    details: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    stability_by_identity = {r["identity_key"]: _stability_score(r) for r in robustness}
    eligible = [d for d in details if d["classification"] != INSUFFICIENT]

    def _sort_key(d: Mapping[str, Any]) -> tuple[float, float, int, str, str, str]:
        dsr = d["dsr"] if d["dsr"] is not None else float("-inf")
        oos = d["oos_sharpe"] if d["oos_sharpe"] is not None else float("-inf")
        stability = stability_by_identity.get(d["identity_key"], 0)
        return (-dsr, -oos, -stability, d["identity_key"], d["candidate_id"], d["cost_cell"])

    ranked = sorted(eligible, key=_sort_key)
    shortlist: list[dict[str, Any]] = []
    for rank, detail in enumerate(ranked[: max(0, top_n)], start=1):
        entry = _row_ref(detail)
        entry["rank"] = rank
        entry["identity_key"] = detail["identity_key"]
        entry["classification"] = detail["classification"]
        entry["incomplete"] = detail["incomplete"]
        entry["stability_score"] = stability_by_identity.get(detail["identity_key"], 0)
        shortlist.append(entry)
    return shortlist


def build_kill_list(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Families whose every evaluated (non-INSUFFICIENT) row is DEAD."""
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for detail in details:
        if detail["classification"] == INSUFFICIENT:
            continue
        by_family[detail["family"]].append(detail)
    kill: list[dict[str, Any]] = []
    for family in sorted(by_family):
        rows = by_family[family]
        if rows and all(d["classification"] == DEAD for d in rows):
            kill.append({"family": family, "dead_row_count": len(rows)})
    return kill


# --------------------------------------------------------------------------- #
# Top-level analysis.                                                          #
# --------------------------------------------------------------------------- #
def _detail_sort_key(detail: Mapping[str, Any]) -> tuple[str, str, str, tuple[float, str], str]:
    return (
        detail["strategy_class"],
        detail["family"],
        detail["timeframe"],
        _cost_sort_key(detail["cost_cell"]),
        detail["candidate_id"] or detail["identity_key"],
    )


def build_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str = "in-memory",
    min_bars: int | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    details = sorted((classify_row(row) for row in rows), key=_detail_sort_key)
    counts = _empty_counts()
    for detail in details:
        counts[detail["classification"]] += 1
    robustness = cross_cost_robustness(details)
    return {
        "source": source,
        "generated_at": datetime.now(UTC).isoformat(),
        "min_bars_context": min_bars,
        "row_count": len(details),
        "classification_counts": counts,
        "rows": details,
        "aggregates": {
            "by_strategy_class": aggregate(details, "strategy_class"),
            "by_family": aggregate(details, "family"),
            "by_timeframe": aggregate(details, "timeframe"),
            "by_cost_cell": aggregate(details, "cost_cell"),
        },
        "cross_cost_robustness": robustness,
        "shortlist": build_shortlist(details, robustness, top_n=top_n),
        "kill_list": build_kill_list(details),
    }


# --------------------------------------------------------------------------- #
# Rendering.                                                                   #
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(payload: Mapping[str, Any]) -> str:
    counts = payload["classification_counts"]
    lines: list[str] = []
    lines.append("# Candidate Grid Analysis")
    lines.append("")
    lines.append(f"- Source: `{payload['source']}`")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Rows analyzed: {payload['row_count']}")
    if payload.get("min_bars_context") is not None:
        lines.append(f"- Min-bars context: {payload['min_bars_context']}")
    lines.append("")
    lines.append("## Classification summary")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("| --- | ---: |")
    for label in CLASS_ORDER:
        lines.append(f"| {label} | {counts[label]} |")
    lines.append("")

    for title, key in (
        ("By strategy class", "by_strategy_class"),
        ("By family", "by_family"),
        ("By timeframe", "by_timeframe"),
        ("By cost cell", "by_cost_cell"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Group | Total | PASS | NEAR-MISS | DEAD | INSUFFICIENT | Median OOS Sharpe | Best |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for group, summary in payload["aggregates"][key].items():
            counts_g = summary["counts"]
            best = summary["best"]
            best_label = (
                f"{best['candidate_id'] or best['name'] or best['identity_key']}"
                f" ({_fmt(best['oos_sharpe'])})"
                if best
                else "n/a"
            )
            lines.append(
                f"| {group} | {summary['total']} | {counts_g[PASS]} | {counts_g[NEAR_MISS]} "
                f"| {counts_g[DEAD]} | {counts_g[INSUFFICIENT]} "
                f"| {_fmt(summary['median_oos_sharpe'])} | {best_label} |"
            )
        lines.append("")

    lines.append("## Cross-cost robustness")
    lines.append("")
    robustness = payload["cross_cost_robustness"]
    if not robustness:
        lines.append("_No identity evaluated at multiple cost cells._")
    else:
        lines.append("| Strategy class | Timeframe | Cost cells | Sign stable | Monotone decay |")
        lines.append("| --- | --- | --- | --- | --- |")
        for report in robustness:
            cells = ", ".join(report["cost_cells"])
            lines.append(
                f"| {report['strategy_class']} | {report['timeframe']} | {cells} "
                f"| {_fmt(report['sign_stable'])} | {_fmt(report['monotone_decay'])} |"
            )
    lines.append("")

    lines.append("## Second-generation shortlist")
    lines.append("")
    shortlist = payload["shortlist"]
    if not shortlist:
        lines.append("_No non-INSUFFICIENT rows to rank._")
    else:
        lines.append("| Rank | Candidate | Class | DSR | OOS Sharpe | Stability | Cost |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
        for entry in shortlist:
            label = entry["candidate_id"] or entry["name"] or entry["identity_key"]
            lines.append(
                f"| {entry['rank']} | {label} | {entry['classification']} "
                f"| {_fmt(entry['dsr'])} | {_fmt(entry['oos_sharpe'])} "
                f"| {entry['stability_score']} | {entry['cost_cell']} |"
            )
    lines.append("")

    lines.append("## Family kill-list")
    lines.append("")
    kill_list = payload["kill_list"]
    if not kill_list:
        lines.append("_No family is uniformly dead._")
    else:
        lines.append("| Family | Dead rows |")
        lines.append("| --- | ---: |")
        for entry in kill_list:
            lines.append(f"| {entry['family']} | {entry['dead_row_count']} |")
    lines.append("")
    return "\n".join(lines)


_TSV_COLUMNS: tuple[str, ...] = (
    "classification",
    "incomplete",
    "strategy_class",
    "family",
    "timeframe",
    "cost_cell",
    "candidate_id",
    "name",
    "net_edge",
    "net_edge_source",
    "dsr",
    "factor_ic",
    "oos_sharpe",
    "near_miss_blocker",
    "identity_key",
)


def render_tsv(payload: Mapping[str, Any]) -> str:
    lines = ["\t".join(_TSV_COLUMNS)]
    for detail in payload["rows"]:
        blocker = detail.get("near_miss_blocker")
        blocker_text = blocker["clause"] if blocker else ""
        values = [
            detail["classification"],
            "1" if detail["incomplete"] else "0",
            detail["strategy_class"],
            detail["family"],
            detail["timeframe"],
            detail["cost_cell"],
            detail["candidate_id"],
            detail["name"],
            _fmt(detail["net_edge"]),
            detail["net_edge_source"],
            _fmt(detail["dsr"]),
            _fmt(detail["factor_ic"]),
            _fmt(detail["oos_sharpe"]),
            blocker_text,
            detail["identity_key"],
        ]
        lines.append("\t".join(v.replace("\t", " ").replace("\n", " ") for v in values))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Report JSON, directory of per-cost report JSONs, or a combined summary.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Shortlist size (default 20).")
    parser.add_argument("--json", dest="json_path", default=None, help="Write full analysis JSON.")
    parser.add_argument(
        "--min-bars", type=int, default=None, help="Context passthrough recorded in the output."
    )
    parser.add_argument("--format", choices=("md", "tsv"), default="md", help="Stdout format.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    rows, source = load_grid(input_path)
    payload = build_analysis(rows, source=source, min_bars=args.min_bars, top_n=args.top_n)
    if args.json_path:
        json_path = Path(args.json_path).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
        )
    if args.format == "tsv":
        print(render_tsv(payload))
    else:
        print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
