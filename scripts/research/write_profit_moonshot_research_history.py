#!/usr/bin/env python3
"""Write the profit-moonshot dynamic-restart research history ledger.

The writer is artifact-only: it scans local profit-moonshot handoffs/plans/reports,
adds the known repeated external-reference clusters, and emits a compact Markdown
and JSON ledger that future restart sessions can read before searching again.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "profit_moonshot_research_history.v1"
DEFAULT_ROOTS = (
    Path("docs"),
    Path(".omx/plans"),
    Path("var/reports"),
)
DEFAULT_DOCS_PATH = Path("docs/profit_moonshot_research_history_20260510.md")
DEFAULT_REPORT_DIR = Path("var/reports/profit_moonshot_20260501/research_history")
DEFAULT_NOTEPAD_PATH = Path(".omx/notepad.md")
DATE_RE = re.compile(r"20(26)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")
DATE_TOKEN_RE = re.compile(r"(2026)[-_]?(0[3-5])[-_]?(0[1-9]|[12]\d|3[01])")
RESEARCH_PATH_RE = re.compile(
    r"profit[-_ ]moonshot|moonshot|portfolio|candidate|research|alpha|strategy|"
    r"backtest|validation|live[-_ ]equivalent|exact[-_ ]window|regime|allocator|"
    r"hybrid|leverage|carry|trend|pair|reboot|raw[-_ ]first|oos|walk[-_ ]forward",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".md", ".json", ".csv", ".log", ".txt", ".jsonl"}
START_DATE = "2026-03-01"
END_DATE = "2026-05-10"

GIT_RESEARCH_KEYWORDS = (
    "profit",
    "moonshot",
    "portfolio",
    "candidate",
    "research",
    "alpha",
    "strategy",
    "backtest",
    "validation",
    "validate",
    "live-equivalent",
    "live equivalent",
    "exact-window",
    "exact window",
    "regime",
    "allocator",
    "hybrid",
    "leverage",
    "kelly",
    "carry",
    "trend",
    "pair",
    "reboot",
    "raw-first",
    "raw first",
    "oos",
    "walk-forward",
    "walk forward",
)

GIT_NON_RESEARCH_PREFIXES = (
    "omx(team): auto-checkpoint",
    "omx(team): merge",
    "merge commit",
    "merge pull request",
)

KNOWN_EXTERNAL_CLUSTERS: tuple[dict[str, Any], ...] = (
    {
        "normalized_key": "external_reference:binance_funding_oi_taker_liquidation_docs",
        "query_or_title": "Binance USD-M funding/open-interest/taker-flow/liquidation docs",
        "source_type": "official_api_doc",
        "path_or_url": "cluster:binance_usdm_market_data_and_margin_docs",
        "match_tokens": ("binance", "funding", "open interest", "taker", "liquidation"),
        "content_summary": (
            "Binance USD-M market-data and margin references describe funding rates, open "
            "interest, taker buy/sell flow, liquidation/margin mechanics, fees, and exchange "
            "metadata needed for live-feasible replay."
        ),
        "what_was_used": (
            "Used as the live-feasibility checklist for funding/OI/taker-flow features, "
            "integer leverage, conservative liquidation replay, fees, slippage, and margin buffers."
        ),
        "families": ("funding_oi_carry", "liquidity_taker_flow", "liquidation_aware_replay"),
        "decision_impact": "Required funding/flow/liquidation candidates to prove data coverage and margin safety.",
        "staleness_policy": "Recheck before live deployment or whenever Binance endpoint/margin rules change.",
        "recheck_before_use": True,
        "do_not_repeat_note": (
            "Do not repeat broad Binance funding/OI/taker/liquidation searches unless endpoint "
            "coverage, date range, or venue rules changed; start from this normalized cluster."
        ),
    },
    {
        "normalized_key": "external_reference:hyperliquid_metadata_candles_funding_fees",
        "query_or_title": "Hyperliquid metadata/candles/funding/fees docs",
        "source_type": "official_api_doc",
        "path_or_url": "cluster:hyperliquid_info_and_exchange_docs",
        "match_tokens": ("hyperliquid", "metadata", "candles", "funding", "fees"),
        "content_summary": (
            "Hyperliquid references were explored for alternative venue metadata, candle, funding, "
            "and fee coverage during external-alpha expansion."
        ),
        "what_was_used": (
            "Kept as a coverage-expansion reference only; no live candidate may promote without "
            "the same train/validation, integer leverage, liquidation, and source-ledger gates."
        ),
        "families": ("external_venue_expansion", "funding_oi_carry"),
        "decision_impact": "External-venue ideas stayed support-lane only because coverage/live constraints dominate.",
        "staleness_policy": "Recheck before adding Hyperliquid data ingestion or live routing.",
        "recheck_before_use": True,
        "do_not_repeat_note": "Do not repeat generic Hyperliquid doc searches unless adding venue data or fees.",
    },
    {
        "normalized_key": "external_reference:tickmill_instruments_spreads_swaps",
        "query_or_title": "Tickmill instruments/spreads/swaps references",
        "source_type": "docs",
        "path_or_url": "cluster:tickmill_instruments_spreads_swaps",
        "match_tokens": ("tickmill", "spread", "swap", "instrument"),
        "content_summary": (
            "Tickmill instrument, spread, and swap references were considered for non-crypto "
            "external market expansion and cost realism."
        ),
        "what_was_used": (
            "Used only to frame external expansion risks; no Tickmill-derived live promotion was made."
        ),
        "families": ("external_venue_expansion", "cost_realism"),
        "decision_impact": "External market expansion remains quarantined until data/cost/live parity is proven.",
        "staleness_policy": "Recheck before any Tickmill/MT5 strategy replay or live feasibility claim.",
        "recheck_before_use": True,
        "do_not_repeat_note": "Do not repeat generic Tickmill searches unless the external-market scope changes.",
    },
    {
        "normalized_key": "external_reference:crypto_momentum_reversal_risk_factor_literature",
        "query_or_title": "Crypto momentum/reversal/common risk-factor literature",
        "source_type": "paper",
        "path_or_url": "cluster:crypto_momentum_reversal_risk_factor_papers",
        "match_tokens": (
            "momentum",
            "reversal",
            "risk-factor",
            "risk factor",
            "paper",
            "literature",
        ),
        "content_summary": (
            "Crypto momentum, reversal, and common risk-factor literature supports dynamic/state "
            "hypotheses but does not validate fixed month/asset calendar-primary rules."
        ),
        "what_was_used": (
            "Used to keep allowed primary signals focused on trend, residual/pair, cross-sectional, "
            "funding/carry, liquidity shock, and volatility state rather than calendar selection."
        ),
        "families": (
            "trend_momentum",
            "residual_pair_reversion",
            "cross_sectional_rank",
            "volatility_compression",
        ),
        "decision_impact": "Calendar-primary winners were rejected; dynamic families remain eligible if gates pass.",
        "staleness_policy": "Recheck only when adding a new academic/mechanistic thesis family.",
        "recheck_before_use": False,
        "do_not_repeat_note": "Do not repeat broad crypto momentum/reversal literature searches for this restart.",
    },
)

NOT_RECONSTRUCTABLE_ITEMS: tuple[dict[str, str], ...] = (
    {
        "inventory_id": "agent_logs:prior_web_search_prompts_20260301_20260510",
        "research_date": "2026-03-01..2026-05-10",
        "source_type": "agent_log",
        "query_or_title": "Prior transient web-search prompts and tmux pane transcripts",
        "normalized_key": "agent_log:prior_web_search_prompts_20260301_20260510",
        "path_or_url": "transient:unpersisted_agent_search_prompts",
        "not_reconstructable_reason": (
            "The durable repo contains handoff summaries and artifact paths, but not every transient "
            "agent prompt/search transcript from earlier sessions."
        ),
    },
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd().resolve(), Path("/home/hoky/Quants-agent/LuminaQuant").resolve()):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _read_text(path: Path, *, max_chars: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _date_from_token(token: str) -> str:
    digits = re.sub(r"\D", "", token)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _research_dates_from_text(value: str) -> list[str]:
    dates = sorted({_date_from_token(match.group(0)) for match in DATE_TOKEN_RE.finditer(value)})
    return [date for date in dates if START_DATE <= date <= END_DATE]


def _first_research_date(value: str) -> str:
    dates = _research_dates_from_text(value)
    return dates[0] if dates else START_DATE


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "unknown"


def normalize_source_key(*, source_type: str, path_or_url: str, query_or_title: str) -> str:
    path_token = str(path_or_url or "").strip()
    if path_token and path_token != "inline":
        normalized = path_token.replace("\\", "/").lower()
        normalized = re.sub(r"^\./", "", normalized)
        return f"{source_type}:{_slug(normalized)}"
    return f"{source_type}:{_slug(query_or_title)}"


def _source_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return "local_artifact"
    if suffix in {".json", ".jsonl", ".csv"}:
        return "local_artifact"
    if suffix == ".log":
        return "agent_log"
    return "local_artifact"


def _summary_from_text(text: str, *, max_len: int = 260) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not compact:
        return "No durable text summary was reconstructable from this artifact."
    return compact[:max_len].rstrip() + ("..." if len(compact) > max_len else "")


def _families_from_text(value: str) -> list[str]:
    token = value.lower()
    families: list[str] = []
    checks = (
        ("calendar", "calendar_primary_invalid"),
        ("liquidation", "liquidation_aware_replay"),
        ("integer", "integer_leverage"),
        ("funding", "funding_oi_carry"),
        ("open interest", "funding_oi_carry"),
        ("taker", "liquidity_taker_flow"),
        ("flow", "liquidity_taker_flow"),
        ("residual", "residual_pair_reversion"),
        ("pair", "residual_pair_reversion"),
        ("momentum", "trend_momentum"),
        ("trend", "trend_momentum"),
        ("cross", "cross_sectional_rank"),
        ("compression", "volatility_compression"),
        ("hybrid", "dynamic_hybrid_allocator"),
        ("portfolio", "dynamic_portfolio_allocator"),
        ("external", "external_venue_expansion"),
        ("live-equivalent", "live_equivalent_validation"),
        ("live equivalent", "live_equivalent_validation"),
        ("exact-window", "exact_window_validation"),
        ("exact window", "exact_window_validation"),
        ("timeframe", "timeframe_sweep"),
        ("raw-first", "raw_first_data_pipeline"),
        ("raw first", "raw_first_data_pipeline"),
        ("regime", "regime_switching_allocator"),
        ("kelly", "regime_switching_allocator"),
        ("walk-forward", "walk_forward_validation"),
        ("walk forward", "walk_forward_validation"),
        ("source", "source_history_ledger"),
        ("ledger", "source_history_ledger"),
    )
    for needle, family in checks:
        if needle in token and family not in families:
            families.append(family)
    return families or ["profit_moonshot_research_history"]


def _should_include_path(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    path_text = str(path).replace("\\", "/")
    if not DATE_TOKEN_RE.search(path_text):
        return False
    return bool(RESEARCH_PATH_RE.search(path_text))


def discover_source_history_inventory(roots: Iterable[Path | str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root_value in roots:
        root = Path(root_value)
        candidates: Iterable[Path]
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = (path for path in root.rglob("*") if path.is_file())
        else:
            continue
        for path in candidates:
            if not _should_include_path(path):
                continue
            display = _display_path(path)
            if display in seen:
                continue
            seen.add(display)
            text = _read_text(path)
            dates = _research_dates_from_text(display) or _research_dates_from_text(text)
            research_date = dates[0] if dates else START_DATE
            source_type = _source_type_for_path(path)
            title = path.stem.replace("_", " ").replace("-", " ")
            normalized_key = normalize_source_key(
                source_type=source_type,
                path_or_url=display,
                query_or_title=title,
            )
            families = _families_from_text(f"{display}\n{text}")
            items.append(
                {
                    "inventory_id": normalized_key,
                    "research_date": research_date,
                    "all_research_dates": dates or [research_date],
                    "source_type": source_type,
                    "query_or_title": title,
                    "normalized_key": normalized_key,
                    "path_or_url": display,
                    "content_summary": _summary_from_text(text),
                    "consulted_history_summary": _summary_from_text(text, max_len=180),
                    "what_was_used": _what_was_used_for_families(families),
                    "associated_strategy_families": families,
                    "decision_impact": _decision_impact_for_families(families),
                    "ledger_refs": [normalized_key],
                    "not_reconstructable": False,
                    "not_reconstructable_reason": "",
                }
            )
    return sorted(items, key=lambda item: (str(item["research_date"]), str(item["path_or_url"])))


def _what_was_used_for_families(families: Sequence[str]) -> str:
    if "calendar_primary_invalid" in families:
        return "Used to reject fixed calendar/month/asset rules as live-primary alpha."
    if "liquidation_aware_replay" in families:
        return (
            "Used to require conservative liquidation, margin-buffer, and account-wipeout evidence."
        )
    if "source_history_ledger" in families:
        return "Used to seed durable source/history inventory and duplicate-search guards."
    if "raw_first_data_pipeline" in families or "live_equivalent_validation" in families:
        return "Used to establish raw-first/live-equivalent data and execution parity before trusting research metrics."
    if "exact_window_validation" in families or "timeframe_sweep" in families:
        return "Used to control split-window/timeframe evidence and prevent accidental validation drift."
    if "regime_switching_allocator" in families or "dynamic_portfolio_allocator" in families:
        return "Used as allocator and portfolio-selection history for later moonshot/live-promotion gates."
    return (
        "Used as historical context for dynamic restart candidate families and rejection decisions."
    )


def _decision_impact_for_families(families: Sequence[str]) -> str:
    if "calendar_primary_invalid" in families:
        return (
            "Blocks calendar-primary live promotion unless a separate robustness program proves it."
        )
    if "integer_leverage" in families:
        return "Non-integer leverage rows are benchmark-only and cannot be live promoted."
    if "dynamic_hybrid_allocator" in families:
        return "Hybrids inherit source candidate validity, leverage, liquidation, and source-ledger gates."
    if "raw_first_data_pipeline" in families:
        return "Raw-first/live data assumptions must stay explicit before comparing strategy performance."
    if "exact_window_validation" in families or "timeframe_sweep" in families:
        return "Later candidates should inherit exact split/timeframe validation rather than rerun ambiguous windows."
    if "regime_switching_allocator" in families:
        return "Regime-switching and Kelly-style allocators remain historical context unless current live gates pass."
    return "Preserves prior research evidence for train/validation-only dynamic restart decisions."


def _is_research_commit(subject: str, body: str = "") -> bool:
    text = f"{subject}\n{body}".lower()
    stripped = subject.lower().strip()
    if any(stripped.startswith(prefix) for prefix in GIT_NON_RESEARCH_PREFIXES):
        return False
    return any(keyword in text for keyword in GIT_RESEARCH_KEYWORDS)


def parse_git_log_records(raw: str) -> list[dict[str, str]]:
    """Parse a NUL-free git log stream using unit separators.

    The format is `%H%x1f%ad%x1f%s%x1f%b%x1e`, which keeps commit subjects and
    bodies reconstructable without depending on line-oriented parsing.
    """
    records: list[dict[str, str]] = []
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        parts = chunk.split("\x1f", 3)
        if len(parts) != 4:
            continue
        commit_hash, commit_date, subject, body = (part.strip() for part in parts)
        if not commit_hash or not commit_date or not subject:
            continue
        records.append(
            {
                "commit": commit_hash,
                "short_commit": commit_hash[:7],
                "date": commit_date,
                "subject": subject,
                "body": body,
            }
        )
    return records


def discover_git_commit_history(
    *, start_date: str = START_DATE, end_date: str = END_DATE
) -> list[dict[str, Any]]:
    """Return research-relevant semantic commits as durable source inventory.

    Git history is intentionally a first-class source because many March/April
    research decisions predate the profit-moonshot-specific artifact directory.
    Runtime scaffolding commits are filtered out; semantic commits remain.
    """
    cmd = [
        "git",
        "log",
        f"--since={start_date} 00:00:00",
        f"--until={end_date} 23:59:59",
        "--date=short",
        "--pretty=format:%H%x1f%ad%x1f%s%x1f%b%x1e",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return []
    if result.returncode != 0:
        return []

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in parse_git_log_records(result.stdout):
        commit_hash = record["commit"]
        if commit_hash in seen:
            continue
        seen.add(commit_hash)
        subject = record["subject"]
        body = record["body"]
        if not _is_research_commit(subject, body):
            continue
        research_date = record["date"]
        if not (start_date <= research_date <= end_date):
            continue
        text = f"{subject}\n{body}"
        families = _families_from_text(text)
        normalized_key = normalize_source_key(
            source_type="git_commit",
            path_or_url=f"git:{commit_hash}",
            query_or_title=subject,
        )
        content_summary = _summary_from_text(
            f"{record['short_commit']} {research_date} {subject}. {body}",
            max_len=320,
        )
        items.append(
            {
                "inventory_id": normalized_key,
                "research_date": research_date,
                "all_research_dates": [research_date],
                "source_type": "git_commit",
                "query_or_title": subject,
                "normalized_key": normalized_key,
                "path_or_url": f"git:{commit_hash}",
                "commit_hash": commit_hash,
                "short_commit": record["short_commit"],
                "content_summary": content_summary,
                "consulted_history_summary": content_summary,
                "what_was_used": _what_was_used_for_families(families),
                "associated_strategy_families": families,
                "decision_impact": _decision_impact_for_families(families),
                "ledger_refs": [normalized_key],
                "not_reconstructable": False,
                "not_reconstructable_reason": "",
                "staleness_policy": "Use as historical decision context; re-run tests/backtests before promotion.",
                "recheck_before_use": False,
                "do_not_repeat_note": (
                    "Check this commit and linked artifacts before re-opening the same research lane."
                ),
            }
        )
    return sorted(items, key=lambda item: (str(item["research_date"]), str(item["path_or_url"])))


def _external_inventory_items(local_items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    joined_by_date: dict[str, str] = defaultdict(str)
    for item in local_items:
        for date in item.get("all_research_dates") or [item.get("research_date")]:
            joined_by_date[str(date)] += (
                " " + str(item.get("path_or_url")) + " " + str(item.get("content_summary"))
            )

    items: list[dict[str, Any]] = []
    for cluster in KNOWN_EXTERNAL_CLUSTERS:
        dates = []
        tokens = [str(token).lower() for token in cluster["match_tokens"]]
        for date, text in joined_by_date.items():
            lower = text.lower()
            if any(token in lower for token in tokens):
                dates.append(date)
        if not dates:
            dates = [START_DATE, END_DATE]
        normalized_key = str(cluster["normalized_key"])
        items.append(
            {
                "inventory_id": normalized_key,
                "research_date": ",".join(sorted(set(dates))),
                "all_research_dates": sorted(set(dates)),
                "source_type": cluster["source_type"],
                "query_or_title": cluster["query_or_title"],
                "normalized_key": normalized_key,
                "path_or_url": cluster["path_or_url"],
                "content_summary": cluster["content_summary"],
                "consulted_history_summary": cluster["content_summary"],
                "what_was_used": cluster["what_was_used"],
                "associated_strategy_families": list(cluster["families"]),
                "decision_impact": cluster["decision_impact"],
                "ledger_refs": [normalized_key],
                "not_reconstructable": False,
                "not_reconstructable_reason": "",
                "staleness_policy": cluster["staleness_policy"],
                "recheck_before_use": cluster["recheck_before_use"],
                "do_not_repeat_note": cluster["do_not_repeat_note"],
            }
        )
    return items


def _not_reconstructable_items() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in NOT_RECONSTRUCTABLE_ITEMS:
        out.append(
            {
                **item,
                "all_research_dates": [START_DATE, END_DATE],
                "content_summary": item["not_reconstructable_reason"],
                "consulted_history_summary": item["not_reconstructable_reason"],
                "what_was_used": "Used only as a gap marker; do not infer source evidence from missing logs.",
                "associated_strategy_families": ["source_history_ledger"],
                "decision_impact": "Requires future agents to rely on durable artifacts, not missing transient logs.",
                "ledger_refs": [],
                "not_reconstructable": True,
            }
        )
    return out


def build_source_search_ledger(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in inventory:
        if item.get("not_reconstructable"):
            continue
        key = str(item.get("normalized_key") or item.get("inventory_id") or "")
        if not key:
            continue
        entry = grouped.setdefault(
            key,
            {
                "research_dates": set(),
                "source_type": item.get("source_type") or "local_artifact",
                "query_or_title": item.get("query_or_title") or key,
                "normalized_key": key,
                "path_or_url": item.get("path_or_url") or "inline",
                "content_summaries": [],
                "what_was_used": [],
                "associated_strategy_families": set(),
                "decision_impacts": [],
                "inventory_refs": [],
                "staleness_policy": item.get("staleness_policy")
                or "Recheck when data coverage, venue docs, or strategy family changes.",
                "recheck_before_use": bool(item.get("recheck_before_use", False)),
                "do_not_repeat_note": item.get("do_not_repeat_note")
                or "Check this normalized key before repeating the same local/source search.",
            },
        )
        for date in item.get("all_research_dates") or [item.get("research_date")]:
            if date:
                entry["research_dates"].add(str(date))
        if item.get("content_summary"):
            entry["content_summaries"].append(str(item["content_summary"]))
        if item.get("what_was_used"):
            entry["what_was_used"].append(str(item["what_was_used"]))
        for family in item.get("associated_strategy_families") or []:
            entry["associated_strategy_families"].add(str(family))
        if item.get("decision_impact"):
            entry["decision_impacts"].append(str(item["decision_impact"]))
        entry["inventory_refs"].append(str(item.get("inventory_id") or key))
        if item.get("recheck_before_use"):
            entry["recheck_before_use"] = True

    ledger = []
    for key, entry in sorted(grouped.items()):
        ledger.append(
            {
                "research_date": ",".join(sorted(entry["research_dates"])) or START_DATE,
                "source_type": entry["source_type"],
                "query_or_title": entry["query_or_title"],
                "normalized_key": key,
                "path_or_url": entry["path_or_url"],
                "content_summary": _collapse_text(entry["content_summaries"]),
                "what_was_used": _collapse_text(entry["what_was_used"]),
                "associated_strategy_families": sorted(entry["associated_strategy_families"]),
                "decision_impact": _collapse_text(entry["decision_impacts"]),
                "staleness_policy": entry["staleness_policy"],
                "recheck_before_use": bool(entry["recheck_before_use"]),
                "do_not_repeat_note": entry["do_not_repeat_note"],
                "inventory_refs": sorted(set(entry["inventory_refs"])),
            }
        )
    return ledger


def _collapse_text(values: Sequence[str], *, max_len: int = 420) -> str:
    seen: list[str] = []
    for value in values:
        compact = " ".join(str(value).split())
        if compact and compact not in seen:
            seen.append(compact)
    joined = " | ".join(seen) if seen else "No durable summary available."
    return joined[:max_len].rstrip() + ("..." if len(joined) > max_len else "")


def _ledger_keys_with(*, ledger: Sequence[Mapping[str, Any]], families: Iterable[str]) -> list[str]:
    wanted = set(families)
    keys = [
        str(entry["normalized_key"])
        for entry in ledger
        if wanted.intersection(set(entry.get("associated_strategy_families") or []))
    ]
    return keys[:8]


def _artifact_paths_with(inventory: Sequence[Mapping[str, Any]], *tokens: str) -> list[str]:
    wanted = [token.lower() for token in tokens]
    paths = []
    for item in inventory:
        text = f"{item.get('path_or_url')} {item.get('content_summary')}".lower()
        if all(token in text for token in wanted):
            paths.append(str(item.get("path_or_url")))
    return paths[:12]


def _commit_refs_with(inventory: Sequence[Mapping[str, Any]], *tokens: str) -> list[str]:
    wanted = [token.lower() for token in tokens]
    refs = []
    for item in inventory:
        if item.get("source_type") != "git_commit":
            continue
        text = f"{item.get('query_or_title')} {item.get('content_summary')}".lower()
        if all(token in text for token in wanted):
            refs.append(str(item.get("path_or_url")))
    return refs[:12]


def build_strategy_chronology(
    *, inventory: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _chronology_entry(
            research_date="2026-03-02",
            strategy_family="raw_first_live_data_foundation",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260302")
                + _commit_refs_with(inventory, "raw-first")
                + _commit_refs_with(inventory, "live", "data")
            )[:12],
            hypothesis="Research metrics should be rebuilt on raw-first/live-data foundations before promotion decisions.",
            primary_signal_type="research_infrastructure",
            features=[
                "raw-first ingestion",
                "live data decoupling",
                "Binance/live migration",
                "memory-safe refresh",
            ],
            universe="live crypto universe and exact-window research inputs",
            timeframe="pre-profit-moonshot March data and refresh foundations",
            split_periods="foundation work for later exact-window train/validation/OOS splits",
            implementation_files=[
                "scripts/research/run_candidate_research.py",
                "src/lumina/live",
                "src/lumina/backtesting",
            ],
            train_metrics="not a strategy-result lane; established data/replay prerequisites",
            validation_metrics="not a strategy-result lane; established validation reproducibility",
            oos_metrics="not a ranking lane; later live-equivalent OOS gates depend on it",
            leverage_status="not a live leverage selection lane yet",
            liquidation_status="liquidation/crowding features existed as strategy inputs but not final replay gates",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=[
                    "raw_first_data_pipeline",
                    "exact_window_validation",
                    "live_equivalent_validation",
                ],
            ),
            advantages=["made later exact replay possible", "reduced data-path ambiguity"],
            disadvantages=[
                "does not itself prove alpha",
                "requires later strategy/live-equivalent gates",
            ],
            final_decision="preserve_as_foundation",
            reason="March data/validation foundations are required context for later moonshot research claims.",
        ),
        _chronology_entry(
            research_date="2026-03-08",
            strategy_family="exact_window_timeframe_and_strategy_expansion",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260308")
                + _artifact_paths_with(inventory, "timeframe")
                + _commit_refs_with(inventory, "exact-window")
                + _commit_refs_with(inventory, "strategy")
            )[:12],
            hypothesis="Memory-safe exact-window sweeps can identify stronger timeframe/strategy candidates without split drift.",
            primary_signal_type="dynamic_state_signal",
            features=[
                "timeframe sweep",
                "exact split validation",
                "strategy expansion",
                "candidate scoring",
            ],
            universe="candidate research universe before the May profit-moonshot naming",
            timeframe="multi-timeframe exact-window research windows",
            split_periods="exact train/validation/OOS windows where artifacts record them",
            implementation_files=[
                "scripts/research/run_candidate_research.py",
                "scripts/research/exact_window_suite.py",
            ],
            train_metrics="candidate-specific in historical artifacts/commits",
            validation_metrics="candidate-specific exact-window validation; no standalone live promotion",
            oos_metrics="locked-OOS style evidence later became gate/report-only",
            leverage_status="not uniformly live-integer constrained",
            liquidation_status="not uniformly liquidation-aware",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["exact_window_validation", "timeframe_sweep", "trend_momentum"],
            ),
            advantages=["broadened research search space", "kept split-window evidence explicit"],
            disadvantages=["pre-live-gate rows can be misleading if reused without current checks"],
            final_decision="carry_forward_as_search_context",
            reason="Useful predecessor search history, but every row still needs current live/source/liquidation gates.",
        ),
        _chronology_entry(
            research_date="2026-03-14",
            strategy_family="dynamic_causal_portfolio_and_walk_forward_research",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260314")
                + _artifact_paths_with(inventory, "walk")
                + _commit_refs_with(inventory, "portfolio")
                + _commit_refs_with(inventory, "walk")
            )[:12],
            hypothesis="Dynamic causal and walk-forward portfolio allocation may improve robustness versus single-candidate ranking.",
            primary_signal_type="dynamic_portfolio_allocator",
            features=[
                "dynamic causal allocation",
                "walk-forward tuning",
                "portfolio weights",
                "candidate ensembles",
            ],
            universe="candidate portfolio universe before later live-equivalent narrowing",
            timeframe="walk-forward and latest-tail research windows",
            split_periods="train/validation/OOS where historical artifacts recorded them",
            implementation_files=[
                "scripts/research/run_portfolio_followup.py",
                "scripts/research/tune_portfolio_weights.py",
            ],
            train_metrics="portfolio-specific historical metrics; not directly comparable after later gates",
            validation_metrics="walk-forward validation context only",
            oos_metrics="historical OOS evidence; not sufficient for May live promotion",
            leverage_status="not yet restricted to positive integer live leverage",
            liquidation_status="not yet conservative Binance-style liquidation replay",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["dynamic_portfolio_allocator", "walk_forward_validation"],
            ),
            advantages=[
                "introduced ensemble/portfolio thinking",
                "surfaced allocator robustness needs",
            ],
            disadvantages=[
                "pre-source-ledger and pre-liquidation gate",
                "can overfit if OOS is used for selection",
            ],
            final_decision="retain_as_allocator_predecessor",
            reason="Allocator ideas informed later candidate-hybrid rules but cannot bypass current train/validation-only gates.",
        ),
        _chronology_entry(
            research_date="2026-03-19",
            strategy_family="strict_portfolio_validation_and_latest_tail",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260319")
                + _artifact_paths_with(inventory, "strict")
                + _commit_refs_with(inventory, "validation")
                + _commit_refs_with(inventory, "oos")
            )[:12],
            hypothesis="Strict portfolio validation and latest-tail checks can prevent fragile candidate promotion.",
            primary_signal_type="validation_control_plane",
            features=[
                "strict validation",
                "latest-tail checks",
                "OOS materialization",
                "memory guard",
            ],
            universe="portfolio/candidate research universe with live-tail data concerns",
            timeframe="March latest-tail and exact-window validation slices",
            split_periods="strict train/validation/OOS split discipline",
            implementation_files=[
                "scripts/research/validate_portfolio_candidates.py",
                "scripts/research/materialize_oos.py",
            ],
            train_metrics="used to enforce prerequisite checks, not to promote a final strategy here",
            validation_metrics="strict validation evidence recorded by commits/artifacts",
            oos_metrics="OOS evidence preserved but not later selection authority",
            leverage_status="strict leverage tooling started emerging but was not final live policy",
            liquidation_status="not yet the final split liquidation-count/margin-buffer contract",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["live_equivalent_validation", "raw_first_data_pipeline"],
            ),
            advantages=["reduced false positives", "made memory/session limits explicit"],
            disadvantages=["historical metrics need recalculation under May gates"],
            final_decision="preserve_validation_precedent",
            reason="Strict validation discipline is a predecessor to the later fail-closed live-promotion contract.",
        ),
        _chronology_entry(
            research_date="2026-03-29",
            strategy_family="regime_switching_and_archived_pair_challengers",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260329")
                + _artifact_paths_with(inventory, "regime")
                + _commit_refs_with(inventory, "regime")
                + _commit_refs_with(inventory, "pair")
            )[:12],
            hypothesis="Regime-switching/Kelly-sized allocators and archived pair blends might improve risk-adjusted OOS.",
            primary_signal_type="regime_switching_allocator",
            features=[
                "regime switch",
                "weekly Kelly sizing",
                "archived pair blend",
                "sparse-symbol live tails",
            ],
            universe="saved allocators and archived pair challengers",
            timeframe="late-March validation and live-tail continuity windows",
            split_periods="strict validation slices; OOS evidence preserved for later comparison only",
            implementation_files=[
                "scripts/research/score_regime_switch_allocators.py",
                "scripts/research/run_portfolio_followup.py",
            ],
            train_metrics="historical allocator/candidate-specific metrics",
            validation_metrics="historical strict validation; not current promotion evidence",
            oos_metrics="risk-adjusted OOS profile improved for some archived challengers but remains historical",
            leverage_status="not final live-integer leverage policy",
            liquidation_status="not final conservative liquidation replay",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["regime_switching_allocator", "residual_pair_reversion"],
            ),
            advantages=[
                "added allocator diversity",
                "made sparse-symbol/live-tail failure modes visible",
            ],
            disadvantages=[
                "archived artifacts can be stale",
                "must not be promoted without current data refresh",
            ],
            final_decision="retain_but_revalidate_before_use",
            reason="Regime and pair ideas remain useful hypotheses only after current train/validation/live gates rerun.",
        ),
        _chronology_entry(
            research_date="2026-04-02",
            strategy_family="portfolio_superiority_8gb_moonshot_and_leverage",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260402")
                + _artifact_paths_with(inventory, "portfolio_superiority")
                + _commit_refs_with(inventory, "portfolio", "superiority")
                + _commit_refs_with(inventory, "leverage")
            )[:12],
            hypothesis="Under-8GiB meta-search and leverage validation can find a superior live portfolio challenger.",
            primary_signal_type="portfolio_meta_search",
            features=[
                "meta-search winners",
                "basis leakage guard",
                "strict leverage validation",
                "8GiB cap",
            ],
            universe="portfolio-superiority candidate set",
            timeframe="April portfolio-superiority and saved-stream research windows",
            split_periods="robust gate evidence and strict validation paths",
            implementation_files=[
                "scripts/research/run_portfolio_superiority.py",
                "scripts/research/sweep_portfolio_leverage.py",
            ],
            train_metrics="meta-search/portfolio-specific; see April artifacts and commits",
            validation_metrics="robust gate evidence; basis leakage explicitly guarded",
            oos_metrics="follow-on challenger OOS evidence preserved but later gates supersede it",
            leverage_status="strict leverage tooling existed; live positive-integer policy finalized later",
            liquidation_status="not yet final Binance-style liquidation/margin replay",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=[
                    "dynamic_portfolio_allocator",
                    "integer_leverage",
                    "live_equivalent_validation",
                ],
            ),
            advantages=["made 8GiB a hard research constraint", "surfaced leakage/basis risks"],
            disadvantages=[
                "portfolio-superiority artifacts are predecessors, not final live candidates"
            ],
            final_decision="preserve_as_moonshot_predecessor",
            reason="April portfolio-superiority work explains the moonshot lineage but must be rerun under current gates.",
        ),
        _chronology_entry(
            research_date="2026-04-16",
            strategy_family="hybrid_reboot_and_profit_first_switch_replay",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260416")
                + _artifact_paths_with(inventory, "hybrid")
                + _commit_refs_with(inventory, "hybrid")
                + _commit_refs_with(inventory, "switch")
            )[:12],
            hypothesis="Profit-first hybrid switch thresholds and replayable policies may improve default portfolio routing.",
            primary_signal_type="dynamic_hybrid_allocator",
            features=[
                "hybrid warm-up",
                "profit-first switch",
                "reboot split replay",
                "market-state coverage",
            ],
            universe="hybrid/reboot portfolio mode universe",
            timeframe="April reboot split and switch-threshold replay windows",
            split_periods="reboot split with replayed switch performance",
            implementation_files=[
                "scripts/research/replay_hybrid_switch.py",
                "scripts/research/tune_profit_first_switch.py",
            ],
            train_metrics="switch-threshold-specific historical metrics",
            validation_metrics="reboot validation context only",
            oos_metrics="replayed switch outcomes preserved; not current live-promotion evidence",
            leverage_status="hybrid source leverage not yet required to be integer for every active source",
            liquidation_status="not final split liquidation replay",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["dynamic_hybrid_allocator", "walk_forward_validation"],
            ),
            advantages=["made hybrid policy replayable", "separated warm-up/lookback mechanics"],
            disadvantages=[
                "hybrid rows can inherit invalid source sleeves",
                "coverage limits can cap replay trust",
            ],
            final_decision="retain_hybrid_design_with_source_gates",
            reason="Hybrid mechanics are useful, but later candidate-derived hybrids must inherit source validity gates.",
        ),
        _chronology_entry(
            research_date="2026-04-20",
            strategy_family="production_carry_trend_and_portfolio_superiority_followup",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260420")
                + _artifact_paths_with(inventory, "carry")
                + _commit_refs_with(inventory, "carry")
                + _commit_refs_with(inventory, "trend")
            )[:12],
            hypothesis="A guarded production carry/trend sleeve and portfolio-superiority follow-up could become live-ready.",
            primary_signal_type="production_candidate_sleeve",
            features=[
                "carry/trend retune",
                "production mode",
                "portfolio overlays",
                "candidate diversity",
            ],
            universe="production-safe carry/trend and portfolio-superiority candidate set",
            timeframe="April production retune and follow-up windows",
            split_periods="candidate research progress and portfolio optimization splits where recorded",
            implementation_files=[
                "scripts/research/run_production_carry_trend_retune.py",
                "scripts/research/portfolio_superiority_followup.py",
            ],
            train_metrics="historical production-lane metrics in April reports",
            validation_metrics="low-memory runner/retune validation; not current final selection",
            oos_metrics="candidate progress artifacts preserve OOS context but do not promote alone",
            leverage_status="live-readiness bridge existed before final integer-only policy",
            liquidation_status="not final Binance-style liquidation replay",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["funding_oi_carry", "trend_momentum", "dynamic_portfolio_allocator"],
            ),
            advantages=["kept production-lane constraints explicit", "improved observability"],
            disadvantages=[
                "still predecessor evidence",
                "requires current source/liquidation gate replay",
            ],
            final_decision="retain_as_production_predecessor",
            reason="Carry/trend production lane is historical context until refreshed under current live candidate rules.",
        ),
        _chronology_entry(
            research_date="2026-04-26",
            strategy_family="live_equivalent_hybrid_and_full_universe_selection",
            artifact_paths=(
                _artifact_paths_with(inventory, "20260426")
                + _artifact_paths_with(inventory, "live_equivalent")
                + _commit_refs_with(inventory, "live-equivalent")
                + _commit_refs_with(inventory, "hybrid")
            )[:12],
            hypothesis="Live-equivalent backtests and deployable-mode filters should separate research scores from live promotion.",
            primary_signal_type="live_equivalent_validation",
            features=[
                "live-equivalent revalidation",
                "deployable HYBRID modes",
                "full-universe selection",
                "candidate reset",
            ],
            universe="full live-equivalent candidate universe before May profit search",
            timeframe="late-April live-equivalent validation and artifact-reset windows",
            split_periods="live-equivalent backtests and deployable portfolio modes",
            implementation_files=[
                "scripts/research/revalidate_live_equivalent_candidates.py",
                "scripts/research/tune_hybrid_deployable_modes.py",
            ],
            train_metrics="candidate-specific live-equivalent backtest inputs",
            validation_metrics="deployable-mode filtering evidence",
            oos_metrics="not enough by itself; fail-closed until live-equivalent candidates pass",
            leverage_status="deployable mode filtering existed before final integer leverage hard gate",
            liquidation_status="not final liquidation/margin-buffer gate",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["live_equivalent_validation", "dynamic_hybrid_allocator"],
            ),
            advantages=[
                "separated research HYBRID scores from live-deployable modes",
                "forced fail-closed posture",
            ],
            disadvantages=["still needed refreshed May data and source-history metadata"],
            final_decision="bridge_to_may_profit_moonshot",
            reason="Late-April live-equivalent work is the direct predecessor to May profit-moonshot gates.",
        ),
        _chronology_entry(
            research_date="2026-05-01",
            strategy_family="profit_first_strategy_factory_restart",
            artifact_paths=_artifact_paths_with(inventory, "20260501")
            or _artifact_paths_with(inventory, "next_session_plan_20260501"),
            hypothesis="Initial profit-first restart should discover live-equivalent dynamic sleeves from refreshed artifacts.",
            primary_signal_type="mixed_dynamic_state",
            features=["trend", "reversion", "breakout", "portfolio state"],
            universe="BTC/ETH/SOL/BNB/TRX and tracked live-equivalent symbols where available",
            timeframe="1h to 1d compounded candidate windows",
            split_periods="train/validation/locked-OOS when recorded by source artifacts",
            implementation_files=[
                "scripts/research/profit_moonshot_research.py",
                "scripts/research/replay_profit_moonshot_fresh_start.py",
            ],
            train_metrics="varied by candidate; see source artifacts",
            validation_metrics="varied by candidate; see source artifacts",
            oos_metrics="report-only; early rows were not final live promotions",
            leverage_status="live leverage not yet strict for every row",
            liquidation_status="not consistently replayed in early artifacts",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger, families=["profit_moonshot_research_history"]
            ),
            advantages=["broad search surface", "preserved early candidate context"],
            disadvantages=["incomplete live gates", "risk of repeated source searches"],
            final_decision="continue_dynamic_restart",
            reason="Useful historical baseline, but not sufficient for live promotion without later gates.",
        ),
        _chronology_entry(
            research_date="2026-05-05",
            strategy_family="current_tail_dynamic_live_equivalent",
            artifact_paths=_artifact_paths_with(inventory, "20260505"),
            hypothesis="Latest-tail dynamic shock/reversion/funding/lead-lag variants may improve OOS return.",
            primary_signal_type="dynamic_state_signal",
            features=["shock reversion", "funding guard", "lead-lag", "taker flow"],
            universe="BTC/ETH/SOL/BNB/TRX where data coverage existed",
            timeframe="hourly and multi-hour holding windows",
            split_periods="train/validation/locked-OOS from current-tail reports",
            implementation_files=[
                "scripts/research/revalidate_live_equivalent_candidates.py",
                "scripts/research/screen_profit_moonshot_external_alpha.py",
            ],
            train_metrics="positive for selected dynamic rows when available",
            validation_metrics="mixed; some rows positive but weak risk-adjusted quality",
            oos_metrics="report-only; weak positive rows did not satisfy final live quality",
            leverage_status="integer leverage not universally enforced yet",
            liquidation_status="liquidation rows often zero/missing; no final promotion",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["funding_oi_carry", "liquidity_taker_flow", "liquidation_aware_replay"],
            ),
            advantages=["state-based families remained viable", "documented weak baselines"],
            disadvantages=["low Sharpe", "coverage gaps", "not fully liquidation-aware"],
            final_decision="reject_for_live_promote_as_baseline",
            reason="Useful dynamic evidence, but live gates and risk-adjusted OOS quality were insufficient.",
        ),
        _chronology_entry(
            research_date="2026-05-08",
            strategy_family="all_family_dynamic_expansion",
            artifact_paths=_artifact_paths_with(inventory, "20260508"),
            hypothesis="A broader family sweep may find dynamic candidates beyond the incumbent tail rows.",
            primary_signal_type="dynamic_state_signal",
            features=["momentum", "residual", "cross-sectional", "funding/OI", "compression"],
            universe="multi-asset crypto universe with available raw-first features",
            timeframe="bounded memory-safe replay windows",
            split_periods="train/validation for selection, locked-OOS for report/gate",
            implementation_files=[
                "scripts/research/replay_profit_moonshot_fresh_start.py",
                "scripts/research/tune_profit_moonshot_fresh_portfolio.py",
            ],
            train_metrics="candidate-specific; retained for train/validation ranking only",
            validation_metrics="candidate-specific; retained for train/validation ranking only",
            oos_metrics="locked-OOS report-only",
            leverage_status="integer leverage became a required live feasibility concern",
            liquidation_status="required later liquidation-aware replay before promotion",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["trend_momentum", "residual_pair_reversion", "cross_sectional_rank"],
            ),
            advantages=["first-principles dynamic scope", "train/validation-only selection shape"],
            disadvantages=["heavy compute", "candidate rows needed stricter metadata"],
            final_decision="continue_with_stricter_gates",
            reason="Dynamic search was allowed, but source/live evidence had to become fail-closed.",
        ),
        _chronology_entry(
            research_date="2026-05-09",
            strategy_family="calendar_conditioned_and_integer_leverage_audit",
            artifact_paths=_artifact_paths_with(inventory, "20260509"),
            hypothesis="High-return calendar-conditioned rows and leverage variants might pass live constraints.",
            primary_signal_type="calendar_primary_invalid",
            features=["fixed month/asset seasonality", "integer leverage retune"],
            universe="profit-moonshot live candidate universe",
            timeframe="monthly/calendar-conditioned and validation retune windows",
            split_periods="train/validation/locked-OOS when available",
            implementation_files=[
                "scripts/research/optuna_tune_profit_moonshot_calendar.py",
                "scripts/research/run_profit_moonshot_liquidation_aware_validation.py",
            ],
            train_metrics="attractive for some calendar rows",
            validation_metrics="attractive for some calendar rows",
            oos_metrics="not sufficient because thesis invalidity dominates",
            leverage_status="non-integer leverage classified benchmark-only",
            liquidation_status="liquidation replay required before any live claim",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=["calendar_primary_invalid", "integer_leverage"],
            ),
            advantages=["surfaced high-return failure mode", "hardened integer leverage policy"],
            disadvantages=[
                "calendar-primary edge is not live-defensible",
                "hybrids inherited defects",
            ],
            final_decision="reject_calendar_primary",
            reason="Calendar-primary month/asset rules are invalid for live promotion by default.",
        ),
        _chronology_entry(
            research_date="2026-05-10",
            strategy_family="liquidation_aware_final_selection_and_validity_audit",
            artifact_paths=_artifact_paths_with(inventory, "20260510"),
            hypothesis="A final candidate or hybrid can promote only if dynamic, traceable, integer, and liquidation safe.",
            primary_signal_type="fail_closed_dynamic_state_required",
            features=[
                "strategy validity",
                "source ledger",
                "integer leverage",
                "liquidation replay",
                "memory RSS",
            ],
            universe="candidate portfolio, direct candidate, and candidate-derived hybrid rows",
            timeframe="latest complete OOS cutoff with locked-OOS report-only",
            split_periods="train/validation selection; locked-OOS gate/report only",
            implementation_files=[
                "scripts/research/write_profit_moonshot_live_final_selection.py",
                "scripts/research/audit_profit_moonshot_strategy_validity.py",
                "scripts/research/run_profit_moonshot_candidate_hybrid.py",
            ],
            train_metrics="selection input only",
            validation_metrics="selection input only",
            oos_metrics="gate/report only; cannot rank or tune candidates",
            leverage_status="positive integer leverage required for live candidates and active hybrid sources",
            liquidation_status="split liquidation count, margin buffer, margin ratio, and no wipeout required",
            source_ledger_refs=_ledger_keys_with(
                ledger=ledger,
                families=[
                    "liquidation_aware_replay",
                    "source_history_ledger",
                    "dynamic_hybrid_allocator",
                ],
            ),
            advantages=["fail-closed live gate", "clear no-promotion terminal state"],
            disadvantages=["may reject all rows until metadata and replay are complete"],
            final_decision="dynamic_restart_required",
            reason="No live promotion without dynamic thesis, source metadata, split/memory/liquidation evidence.",
        ),
        _chronology_entry(
            research_date="2026-05-10",
            strategy_family="dynamic_restart_research_history_ledger",
            artifact_paths=[
                str(DEFAULT_DOCS_PATH),
                str(DEFAULT_REPORT_DIR / "profit_moonshot_research_history_latest.json"),
            ],
            hypothesis="Future sessions should read a durable research history before new searches or tuning.",
            primary_signal_type="research_control_plane",
            features=["source_history_inventory", "source_search_ledger", "duplicate_search_guard"],
            universe="all reconstructable research artifacts and semantic git commits from 2026-03-01 through 2026-05-10",
            timeframe="2026-03-01..2026-05-10 research history",
            split_periods="not applicable",
            implementation_files=["scripts/research/write_profit_moonshot_research_history.py"],
            train_metrics="not applicable",
            validation_metrics="not applicable",
            oos_metrics="not applicable",
            leverage_status="records integer-leverage policy as a gate",
            liquidation_status="records liquidation/margin evidence as a gate",
            source_ledger_refs=_ledger_keys_with(ledger=ledger, families=["source_history_ledger"]),
            advantages=["prevents repeated searches", "preserves rejection reasons"],
            disadvantages=["must be updated by future research passes"],
            final_decision="require_future_sessions_to_read_first",
            reason="Research memory is part of the live-promotion contract.",
        ),
    ]


def _chronology_entry(
    *,
    research_date: str,
    strategy_family: str,
    artifact_paths: Sequence[str],
    hypothesis: str,
    primary_signal_type: str,
    features: Sequence[str],
    universe: str,
    timeframe: str,
    split_periods: str,
    implementation_files: Sequence[str],
    train_metrics: str,
    validation_metrics: str,
    oos_metrics: str,
    leverage_status: str,
    liquidation_status: str,
    source_ledger_refs: Sequence[str],
    advantages: Sequence[str],
    disadvantages: Sequence[str],
    final_decision: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "research_date": research_date,
        "strategy_family": strategy_family,
        "artifact_paths": list(artifact_paths),
        "hypothesis": hypothesis,
        "primary_signal_type": primary_signal_type,
        "state_variables_or_features": list(features),
        "universe": universe,
        "timeframe": timeframe,
        "split_periods": split_periods,
        "implementation_files": list(implementation_files),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "oos_metrics": oos_metrics,
        "leverage_status": leverage_status,
        "liquidation_status": liquidation_status,
        "source_ledger_refs": list(source_ledger_refs),
        "advantages": list(advantages),
        "disadvantages_or_risks": list(disadvantages),
        "final_decision": final_decision,
        "rejection_or_promotion_reason": reason,
    }


def build_research_history_payload(
    *,
    roots: Iterable[Path | str] = DEFAULT_ROOTS,
    generated_at_utc: str | None = None,
    include_git_history: bool = True,
) -> dict[str, Any]:
    artifact_inventory = discover_source_history_inventory(roots)
    git_commit_history = discover_git_commit_history() if include_git_history else []
    local_inventory = artifact_inventory + git_commit_history
    inventory = (
        local_inventory + _external_inventory_items(local_inventory) + _not_reconstructable_items()
    )
    ledger = build_source_search_ledger(inventory)
    chronology = build_strategy_chronology(inventory=inventory, ledger=ledger)
    return {
        "strategy_chronology": chronology,
        "git_commit_history": git_commit_history,
        "source_history_inventory": inventory,
        "source_search_ledger": ledger,
        "decision_log": [
            {
                "decision": "calendar_primary_alpha_rejected_by_default",
                "reason": "Fixed month/window/asset rules are not a live-defensible primary signal.",
                "effective_date": "2026-05-10",
            },
            {
                "decision": "locked_oos_report_gate_only",
                "reason": "Train+validation may select; locked-OOS may veto/report only.",
                "effective_date": "2026-05-10",
            },
            {
                "decision": "missing_source_or_research_history_metadata_blocks_promotion",
                "reason": "Future live rows must link back to this ledger or a successor ledger.",
                "effective_date": "2026-05-10",
            },
        ],
        "invalidity_lessons": [
            "Calendar-primary month/asset rules are invalid live alpha unless separately justified and robustness-tested.",
            "Non-integer leverage is benchmark-only; live candidates require positive integer leverage.",
            "Candidate-derived hybrids inherit source validity, leverage, liquidation, memory, and source-ledger defects.",
            "Missing strategy-validity, source-ledger, split, liquidation, or memory evidence fails closed.",
        ],
        "future_session_instructions": [
            "Read docs/profit_moonshot_research_history_20260510.md before new profit-moonshot searches.",
            "Check source_search_ledger.normalized_key before repeating funding/OI/taker/liquidation/external-venue/literature searches.",
            "Use train+validation only for ranking/tuning; keep locked-OOS gate/report-only.",
            "Update this history after every material profit-moonshot research pass.",
        ],
        "generation_metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": generated_at_utc or _utc_now_iso(),
            "date_range": f"{START_DATE}..{END_DATE}",
            "roots": [str(root) for root in roots],
            "artifact_inventory_count": len(artifact_inventory),
            "git_commit_count": len(git_commit_history),
            "local_inventory_count": len(local_inventory),
            "inventory_count": len(inventory),
            "ledger_count": len(ledger),
        },
    }


def build_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Profit Moonshot Research History — March-to-May Dynamic Restart Ledger",
        "",
        "Future profit-moonshot sessions must read this file before repeating searches or promoting candidates.",
        "",
        "## Generation metadata",
    ]
    metadata = payload.get("generation_metadata") or {}
    for key in (
        "schema_version",
        "generated_at_utc",
        "date_range",
        "artifact_inventory_count",
        "git_commit_count",
        "local_inventory_count",
        "ledger_count",
    ):
        lines.append(f"- {key}: `{metadata.get(key)}`")

    lines.extend(["", "## Strategy chronology"])
    for entry in payload.get("strategy_chronology") or []:
        item = dict(entry)
        lines.extend(
            [
                "",
                f"### {item.get('research_date')} — {item.get('strategy_family')}",
                f"- Hypothesis: {item.get('hypothesis')}",
                f"- Primary signal type: `{item.get('primary_signal_type')}`",
                f"- State/features: {', '.join(item.get('state_variables_or_features') or [])}",
                f"- Universe/timeframe: {item.get('universe')} / {item.get('timeframe')}",
                f"- Splits: {item.get('split_periods')}",
                f"- Metrics: train={item.get('train_metrics')}; validation={item.get('validation_metrics')}; locked-OOS={item.get('oos_metrics')}",
                f"- Leverage/liquidation: {item.get('leverage_status')} / {item.get('liquidation_status')}",
                f"- Decision: `{item.get('final_decision')}` — {item.get('rejection_or_promotion_reason')}",
                f"- Source ledger refs: {', '.join(item.get('source_ledger_refs') or []) or 'none'}",
                f"- Artifacts: {', '.join(item.get('artifact_paths') or []) or 'none'}",
            ]
        )

    lines.extend(["", "## Git commit research ledger"])
    for commit in payload.get("git_commit_history") or []:
        item = dict(commit)
        lines.append(
            "- "
            f"{item.get('research_date')} `{item.get('short_commit')}` — "
            f"{item.get('query_or_title')} "
            f"({', '.join(item.get('associated_strategy_families') or [])})"
        )

    lines.extend(["", "## Source/history inventory"])
    for item in payload.get("source_history_inventory") or []:
        inv = dict(item)
        status = "not_reconstructable" if inv.get("not_reconstructable") else "mapped"
        lines.append(
            f"- `{inv.get('inventory_id')}` ({status}) — {inv.get('research_date')} — {inv.get('path_or_url')}"
        )

    lines.extend(["", "## Source/search ledger"])
    for entry in payload.get("source_search_ledger") or []:
        item = dict(entry)
        lines.extend(
            [
                "",
                f"### `{item.get('normalized_key')}`",
                f"- Research dates: {item.get('research_date')}",
                f"- Source type: `{item.get('source_type')}`",
                f"- Title/path: {item.get('query_or_title')} — {item.get('path_or_url')}",
                f"- Content summary: {item.get('content_summary')}",
                f"- What was used: {item.get('what_was_used')}",
                f"- Families: {', '.join(item.get('associated_strategy_families') or [])}",
                f"- Decision impact: {item.get('decision_impact')}",
                f"- Staleness/recheck: {item.get('staleness_policy')} / `{item.get('recheck_before_use')}`",
                f"- Do-not-repeat note: {item.get('do_not_repeat_note')}",
            ]
        )

    lines.extend(["", "## Invalidity lessons"])
    for lesson in payload.get("invalidity_lessons") or []:
        lines.append(f"- {lesson}")

    lines.extend(["", "## Future-use instructions"])
    for instruction in payload.get("future_session_instructions") or []:
        lines.append(f"- {instruction}")
    lines.append("")
    return "\n".join(lines)


def write_research_history_outputs(
    payload: Mapping[str, Any], *, docs_path: Path, report_dir: Path
) -> dict[str, Path]:
    markdown = build_markdown(payload)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / "profit_moonshot_research_history_latest.md"
    report_json = report_dir / "profit_moonshot_research_history_latest.json"
    docs_path.write_text(markdown, encoding="utf-8")
    report_md.write_text(markdown, encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"docs_markdown": docs_path, "report_markdown": report_md, "report_json": report_json}


def ensure_notepad_pointer(notepad_path: Path, docs_path: Path) -> bool:
    pointer = (
        "Profit moonshot dynamic restart: read "
        f"{docs_path} before new source searches, tuning, or live promotion decisions."
    )
    existing = (
        notepad_path.read_text(encoding="utf-8", errors="ignore") if notepad_path.exists() else ""
    )
    if pointer in existing:
        return False
    notepad_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n" if existing and not existing.endswith("\n") else ""
    notepad_path.write_text(f"{existing}{separator}\n- {pointer}\n", encoding="utf-8")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", action="append", default=[], help="Root/file to scan; repeatable."
    )
    parser.add_argument("--docs-path", default=str(DEFAULT_DOCS_PATH))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--notepad-path", default=str(DEFAULT_NOTEPAD_PATH))
    parser.add_argument("--no-notepad", action="store_true")
    parser.add_argument("--no-git-history", action="store_true")
    parser.add_argument("--generated-at-utc", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(item) for item in args.root] if args.root else list(DEFAULT_ROOTS)
    payload = build_research_history_payload(
        roots=roots,
        generated_at_utc=str(args.generated_at_utc or "") or None,
        include_git_history=not args.no_git_history,
    )
    paths = write_research_history_outputs(
        payload,
        docs_path=Path(args.docs_path),
        report_dir=Path(args.report_dir),
    )
    notepad_updated = False
    if not args.no_notepad:
        notepad_updated = ensure_notepad_pointer(Path(args.notepad_path), Path(args.docs_path))
    print(
        json.dumps(
            {
                "ok": True,
                "paths": {key: str(path) for key, path in paths.items()},
                "git_commit_count": len(payload["git_commit_history"]),
                "inventory_count": len(payload["source_history_inventory"]),
                "ledger_count": len(payload["source_search_ledger"]),
                "notepad_updated": notepad_updated,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
