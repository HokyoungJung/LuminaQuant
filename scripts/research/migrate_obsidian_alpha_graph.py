#!/usr/bin/env python3
"""Merge immutable generated strategy facts into canonical Obsidian notes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

BEGIN = "<!-- alpha-research:begin -->"
END = "<!-- alpha-research:end -->"
TODAY = "2026-08-23"
COMMON_EVAL = "[[LuminaQuant/Evaluations/2026-08-15 Common-Period Strategy Screen]]"
SMOKE_EVAL = "[[LuminaQuant/Run Preintegration Strategy Smoke]]"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_note(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"frontmatter_missing:{path}")
    raw, body = text[4:].split("\n---\n", 1)
    parsed = yaml.safe_load(raw)
    if type(parsed) is not dict:
        raise ValueError(f"frontmatter_invalid:{path}")
    return parsed, body.rstrip() + "\n"


def link(path: Path, vault: Path) -> str:
    relative = path.relative_to(vault).with_suffix("").as_posix()
    return f"[[{relative}]]"


def metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return f"{number * 100:.3f}%" if percent else f"{number:.6f}"


def state_for(frontmatter: dict[str, Any]) -> str:
    status = str(frontmatter.pop("status", "") or "")
    if not status and type(frontmatter.get("research_state")) is str:
        return str(frontmatter["research_state"])
    return {
        "live_default": "research_only",
        "live_opt_in": "research_only",
        "research-only": "research_only",
        "research_only": "research_only",
    }.get(status, "research_only")


def canonical_frontmatter(
    current: dict[str, Any], generated: dict[str, Any] | None, provenance: str | None
) -> dict[str, Any]:
    result = dict(current)
    result["type"] = "strategy"
    result["research_state"] = state_for(result)
    result["implementation_state"] = "implemented"
    tier = str((generated or {}).get("tier") or result.get("tier") or "research_only")
    result["tier"] = tier
    if generated is not None:
        family = str(generated["family"])
        result["family"] = [
            f"[[LuminaQuant/Strategy Research Generated/Families/{FAMILY_FILES[family]}]]"
        ]
        result["evaluated_by"] = [SMOKE_EVAL, COMMON_EVAL]
        result["provenance"] = [provenance]
    else:
        result.setdefault("family", [])
        result["evaluated_by"] = [SMOKE_EVAL]
        result.setdefault("provenance", [])
    for field in (
        "evidence_for",
        "evidence_against",
        "derived_from",
        "similar_to",
        "complements",
        "conflicts_with",
        "supersedes",
        "superseded_by",
        "rejected_by",
    ):
        result.setdefault(field, [])
    result["updated"] = TODAY
    result["tags"] = ["lq/type/strategy", f"lq/state/{result['research_state'].replace('_', '-')}"]
    return result


def rendered_section(name: str, generated: dict[str, Any] | None, provenance: str | None) -> str:
    if generated is None:
        return (
            f"{BEGIN}\n## Alpha-research graph\n\n"
            "- Generated common-period counterpart: unavailable.\n"
            "- Existing smoke evidence is execution coverage only.\n"
            "- Performance, family, and relationship fields remain unfilled rather than inferred.\n"
            f"{END}\n"
        )
    family = str(generated["family"])
    rows = (
        f"| Full | {metric(generated.get('full_return'), percent=True)} | "
        f"{metric(generated.get('full_sharpe'))} | {metric(generated.get('full_max_drawdown'), percent=True)} |\n"
        f"| Recent nested | {metric(generated.get('recent_return'), percent=True)} | "
        f"{metric(generated.get('recent_sharpe'))} | NA |"
    )
    return (
        f"{BEGIN}\n## Alpha-research graph\n\n"
        f"- family: [[LuminaQuant/Strategy Research Generated/Families/{FAMILY_FILES[family]}|{family}]]\n"
        f"- immutable generated provenance: {provenance}\n"
        f"- interface: `{generated.get('execution_interface')}`; runner: `{generated.get('runner_kind')}`\n"
        f"- scope: `{generated.get('scope_status')}`; cadence: `{generated.get('cadence_status')}`\n\n"
        f"### {COMMON_EVAL}\n\n"
        "This is an unsealed common-period diagnostic. The recent window is nested; neither row is independent OOS or promotion evidence.\n\n"
        "| Window | Return | Sharpe | MDD |\n|---|---:|---:|---:|\n"
        f"{rows}\n\n"
        "Smoke pass, quality survival, suite acceptance, and deployment are separate states.\n"
        f"{END}\n"
    )


def strip_managed_section(body: str) -> str:
    if BEGIN in body:
        prefix, rest = body.split(BEGIN, 1)
        if END not in rest:
            raise ValueError("managed_section_unterminated")
        _managed, suffix = rest.split(END, 1)
        return prefix.rstrip() + suffix.rstrip() + "\n"
    return body


def dump_note(frontmatter: dict[str, Any], body: str) -> str:
    yaml_text = yaml.safe_dump(
        frontmatter,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    return f"---\n{yaml_text}\n---\n{body.rstrip()}\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


FAMILY_FILES = {
    "benchmark": "benchmark--0e89820860c3",
    "breakout": "breakout--e2fd4410c00e",
    "cross_sectional": "cross_sectional--24e2ef8ccf6e",
    "derivatives_directional_crowding": "derivatives_directional_crowding--6c654600d000",
    "ensemble_regime_router": "ensemble_regime_router--fd638c741eea",
    "event_alpha": "event_alpha--a809782c63aa",
    "formulaic_alpha": "formulaic_alpha--2342128e878a",
    "intermarket": "intermarket--01d86feb2c13",
    "mean_reversion_relative_value": "mean_reversion_relative_value--e05e27323a91",
    "microstructure_intraday": "microstructure_intraday--da295918e94b",
    "rebalancing_diversification": "rebalancing_diversification--f9705e533db4",
    "seasonality": "seasonality--ce56cb28417d",
    "trend_momentum": "trend_momentum--fc59d54457fc",
    "volatility_risk_overlay": "volatility_risk_overlay--8af27b54a57b",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    vault = args.vault.resolve(strict=True)
    lq = vault / "LuminaQuant"
    manual_root = lq / "Strategies"
    generated_root = lq / "Strategy Research Generated" / "Strategies"
    generated_hashes = {p.name: sha256(p) for p in generated_root.glob("*.md")}
    generated: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(generated_root.glob("*.md")):
        frontmatter, _body = read_note(path)
        name = str(frontmatter.get("strategy") or "")
        if not name or name in generated:
            raise ValueError(f"generated_strategy_invalid:{path}")
        if frontmatter.get("family") not in FAMILY_FILES:
            raise ValueError(f"generated_family_invalid:{path}")
        generated[name] = (path, frontmatter)
    manual = {path.stem: path for path in manual_root.glob("*.md")}
    archived_probe = lq / "Archive" / "Operational Probes" / "E2eDropinProbeStrategy.md"
    if archived_probe.is_file():
        manual[archived_probe.stem] = archived_probe
    paired = sorted(set(manual) & set(generated))
    if len(paired) not in (141, 142):
        raise ValueError(f"paired_strategy_count_invalid:{len(paired)}")
    if sorted(set(manual) - set(generated)) != [
        "E2eDropinProbeStrategy",
        "MicroRangeExpansion1sStrategy",
    ]:
        raise ValueError("manual_only_strategy_set_invalid")
    if sorted(set(generated) - set(manual)) not in (
        ["DacapogoDailySourceStrategy", "VolCompressionVwapReversionStrategy"],
        ["VolCompressionVwapReversionStrategy"],
    ):
        raise ValueError("generated_only_strategy_set_invalid")
    changes: dict[Path, str] = {}
    for name in paired:
        current, body = read_note(manual[name])
        source_path, source = generated[name]
        source_link = link(source_path, vault)
        frontmatter = canonical_frontmatter(current, source, source_link)
        body = (
            strip_managed_section(body).rstrip()
            + "\n\n"
            + rendered_section(name, source, source_link)
        )
        changes[manual[name]] = dump_note(frontmatter, body)
    for name in ("E2eDropinProbeStrategy", "MicroRangeExpansion1sStrategy"):
        current, body = read_note(manual[name])
        frontmatter = canonical_frontmatter(current, None, None)
        if name == "E2eDropinProbeStrategy":
            frontmatter["research_state"] = "retired"
            frontmatter["tags"] = ["lq/type/operational-probe", "lq/state/retired"]
        else:
            frontmatter["research_state"] = "historical"
            frontmatter["tags"] = ["lq/type/strategy", "lq/state/historical", "lq/scope/subminute"]
        body = strip_managed_section(body).rstrip() + "\n\n" + rendered_section(name, None, None)
        changes[manual[name]] = dump_note(frontmatter, body)
    dacapogo_path, dacapogo = generated["DacapogoDailySourceStrategy"]
    canonical_dacapogo = manual_root / "DacapogoDailySourceStrategy.md"
    if not canonical_dacapogo.exists():
        frontmatter = canonical_frontmatter(
            {
                "id": "lq-strategy-dacapogodailysourcestrategy",
                "type": "strategy",
                "title": "DacapogoDailySourceStrategy",
                "project": "[[LuminaQuant]]",
                "created": TODAY,
            },
            dacapogo,
            link(dacapogo_path, vault),
        )
        frontmatter["research_state"] = "rejected"
        frontmatter["rejected_by"] = ["[[LuminaQuant/Run G003 Dacapogo 2026-08-16]]"]
        body = (
            "# DacapogoDailySourceStrategy\n\n"
            "Dedicated authenticated daily-source strategy. Locked validation completed in cash with 0/13 promoted folds; this is a completed failed gate, not an interruption.\n\n"
            + rendered_section("DacapogoDailySourceStrategy", dacapogo, link(dacapogo_path, vault))
        )
        changes[canonical_dacapogo] = dump_note(frontmatter, body)
    planned = {
        "paired_updated": len(paired),
        "manual_only_updated": 2,
        "created": int(not canonical_dacapogo.exists()),
        "generated_only_unpromoted": ["VolCompressionVwapReversionStrategy"],
        "generated_manifest_untouched": True,
    }
    if args.apply:
        for path, content in sorted(changes.items(), key=lambda item: str(item[0])):
            atomic_write(path, content)
        if generated_hashes != {p.name: sha256(p) for p in generated_root.glob("*.md")}:
            raise RuntimeError("generated_snapshot_changed")
    result = {
        "artifact_kind": "obsidian_alpha_strategy_migration.v1",
        "applied": args.apply,
        "vault": str(vault),
        "planned": planned,
        "changed_paths": [str(path.relative_to(vault)) for path in sorted(changes)],
        "generated_strategy_count": len(generated),
        "generated_strategy_sha256": generated_hashes,
    }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if args.receipt:
        atomic_write(args.receipt, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
